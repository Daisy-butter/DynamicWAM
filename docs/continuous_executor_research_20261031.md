# 连续仿真执行器调研与 V8 取舍（2026-10-31）

结论：有成熟的相似处理，但没有发现可直接替换 DOMINO 连续执行器、同时保证原训练时间语义和成功率的通用组件。要分清三层：推理/RTC 调度、带时间戳的参考轨迹、每个物理 tick 的伺服控制。连续仿真不要求线性插值；平滑也不能代替时间对齐。

## 一手资料与适用范围

| 工作/实现 | 实际处理 | 对本项目的启发和边界 |
|---|---|---|
| [RTC / Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339v2)；[Kinetix 开源实验](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | 执行旧 chunk 时生成新 chunk；冻结必然执行的前缀并补全其余部分，也讨论软掩码。Kinetix 使用力控制及模拟推理延迟。 | 与我们的异步承诺直接相关；不是关节位置轨迹的 C1/C2 执行器，也不是对 SAPIEN 渲染下 1× 墙钟实时性的验证。 |
| [Diffusion Policy 实时控制进程](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/real_world/rtde_interpolation_controller.py) | 独立进程高频发送伺服命令，减少 Python GIL 干扰；队列携带目标时刻，执行使用 monotonic 时间。 | 推理、接收、执行分离，比阻塞式 env.step 更接近我们的部署需求。它是实机控制，不能直接证明仿真引擎线程安全。 |
| [Diffusion Policy 轨迹插值器](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/pose_trajectory_interpolator.py) | 平移线性插值、旋转 Slerp；修剪旧轨迹、安排未来 waypoint，并按速度限制调整所需时间。 | 证明异步执行并不必然用高阶曲线；其限速可能延后到达时间，不能原样搬到固定 180 ms 标签。 |
| [ROS 2 joint_trajectory_controller](https://control.ros.org/kilted/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html) | 只给位置时线性；给位置/速度时三次；给位置/速度/加速度时五次。相应保证位置、速度、加速度连续；不一致的边界导数会造成过冲。 | C1/C2 必须在 chunk 替换边界也成立。不同 ROS 版本的轨迹替换支持不同，不能只照文档名字认定功能已经接入。 |
| [Ruckig，RSS 2021](https://www.roboticsproceedings.org/rss17/p015.pdf)；[官方实现](https://github.com/pantor/ruckig) | 从给定位置/速度/加速度生成受速度、加速度、jerk 约束的在线轨迹。 | 适合运动学可行性与平滑。官方明确 minimum_duration 不保证精确时长；Community 中间 waypoint 路径还涉及云服务，不能默认用于本地实时闭环。 |
| [MoveIt 时间参数化与 Ruckig 平滑](https://moveit.github.io/moveit_tutorials/doc/time_parameterization/time_parameterization_tutorial.html) | 路径规划后做时间参数化，再做 jerk 平滑。 | TOPP/TOTG 与在线接续是不同问题；改时间参数化必须把新的动作时间反馈到模型、预算与监督。 |
| [ManiSkill PDJointPosController 源码](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/controllers/pd_joint_pos.py) | 可在一个 policy action 内按物理子步线性更新 PD target；该实现从实测 qpos 取起点。 | 仿真里已有多频控制插值；但默认训练 step 接口不等于推理期间物理继续按墙钟运行。其起点语义也与我们的上一命令端点不同。 |
| [robosuite 控制器](https://robosuite.ai/docs/simulation/controller.html) | policy frequency 与底层控制分开，可使用 interpolator 在两次动作输入之间生成控制目标。 | 支持把策略、插值、伺服分别验收；不能据此宣称已有严格实时墙钟契约。 |
| [MuJoCo simulate 官方说明](https://mujoco.readthedocs.io/en/stable/programming/samples.html) | UI/渲染在主线程，仿真在同步的后台线程。 | 仿真器并非本质只能同步阻塞；引擎支持与上层 benchmark API 是两层。迁移仍需快照所有权、互斥和计时审计。 |
| [World Action Models in Real Time（2026，Motubrain）](https://arxiv.org/html/2608.01880v1) | 比较异步切换、输出加权、推理期 RTC、训练期前缀条件；强调时间错配会造成持续抖动，直接加权有精度代价。 | 与我们问题很接近，但实机在线每方法每任务仅 5 次试验，结论限于其平台，不能推导出我们必然超过 Full。 |

## 针对本项目的工程判断

当前 `.50` V8 仍是固定端点线性执行：物理步约 4 ms，每个动作 45 步；手臂从上一命令端点向下一端点移动，夹爪使用区间起点的原生绝对命令。它通常仅保证命令位置 C0 连续，节点速度可以跳变。实际机器人还受 PD、惯性和接触影响，因此“命令平滑”和“真实运动稳定准确”须分别测量。

首选下一步候选是 **固定时间网格、带边界状态的受约束轨迹接续**，而不是先改成变时长 TOPP，也不是把每个 waypoint 都变成零速度停点。这是本项目适配建议，不是上述文献已证明的最优解：

1. 保留物理 tick、180 ms 网格、任务动作预算、夹爪时刻、RTC 保护区间。执行器在新轨迹首次允许生效的绝对物理时刻接续。
2. 冻结已承诺的整段轨迹，而不仅是端点数值。接续时继承旧命令曲线的 q/v（C1）或 q/v/a（C2）；不能用收到消息那一刻的状态去替代未来接续边界。实测 q/qdot 用于反馈、误差记录和显式重新规划，不能隐式改写已承诺前缀。
3. 先测试限幅斜率的三次 Hermite C1，再测试有约束的五次 C2。邻接节点共享导数，不在每个动作/chunk 边界强制速度归零；限制过冲、关节范围、速度和加速度。C2 本身不保证 jerk 上界，也不保证轨迹可行。
4. 全局重算样条时，必须把受保护区间及其边界导数固定。否则即使旧端点没变，未来点也可能改变当前执行曲线，违反 RTC。
5. Ruckig 可作为另一候选及可行性检查工具；不能偷偷拉长 45 tick。若约束下无法按时到达，必须显式报告不可行，或定义新的投影/重定时协议并同步训练标签、A10 区间、Z 时间和预算。也不能调用完一个 180 ms 的新规划后才允许下一次更新。
6. 连续仿真的物理时间由真实 step 累积；墙钟用于 1× pace、发布可用性与截止时间。轨迹采样、动作参考、Z 都按物理时间匹配；后台线程不各自构造独立“仿真现在”。渲染慢不能通过重置墙钟或偷停世界抵消。

不建议本轮最终 ckpt 评测直接启用新平滑器。即使保留端点与时间，高阶曲线也会改变接触过程；当前 Z 来自既定物理轨迹，必须先做部署对照再决定是否重建数据/重训。限速造成的额外滞后尤其可能伤害动态物体抓取。

## 后续验收与对照

- 录制动作回放：线性/C1/C2 的 tick 数、所有端点时刻、夹爪时刻、保护前缀逐样本一致；改变渲染/推理负载不能改变已给定的控制序列。
- 接续与故障注入：迟到、乱序、换 chunk、反向运动、回合 reset、接触与队列不足；验证 q/v/a 边界、限幅与不可行分支，不隐藏超时。
- 闭环：相同 ckpt/700 场景/资源/500 ms backlog，分别记录命令与实际关节速度/加速度/jerk、跟踪误差、接触事件、观测到首次可变动作时延、SR/MS。
- 先短任务对照再全套；不能仅凭 MP4 更顺或 jerk 降低判定策略更准确。25.43% 是旧 250 ms Full 协议，不能与本轮 500 ms 直接等价比较。

本次只完成调研与独立评测入口，不修改正在训练的 V8 模型、数据、执行器或环境依赖。
