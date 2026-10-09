# 可复用世界上下文与连续物理时钟：实现及实验协议

## 这次实际改变什么

保留官方 full WAM 的完整采样：2 次 video 联合计算、10 次 action 去噪，生成完整 16 行动作。Stage1 和官方 video/action 参数均不重训。这次 stage2/3 训练的是快速反馈网络及新增世界上下文读取分支；不是把官方 stage2/3 原配方重新跑一遍。

慢查询同时发布一个不可变 bundle：`plan_id + source_sim_time + source_action_index + plan + source_state + world_context`。context 来自第二次联合去噪、最后一层真实 video V；按原生条件/未来时间片分别做 4×4 空间池化，保留时间顺序。它既不是 GT future，也不是纠偏后动作的重新仿真。视频内部缓存不被反馈网络修改。

快查询用当前三视角、上一份实际拍摄的 head 图像及真实 dt、当前机器人 drive target、剩余计划和 age 查询这份 context。读取结果通过零初始化投影加到已有视觉 tokens，不追加改变 softmax 分母的空 tokens。网络仍只输出 4 行有界 arm 残差；gripper 和 age0 保持不修正。剩余动作以原计划为基准，每次纠偏不递归叠加。

相关代码：`models/world_context_feedback.py`、`runtime/world_context_worker.py`、`training/data/world_context_feedback_dataset.py`。初始函数在旧头原先覆盖的 ages 0/4/8/12 保持；新增年龄插值不能作为性能保证。100ms 协议下模型实际请求年龄由时钟计算，不能用“查询了几次”代替。

## 动机与可检验的收益

旧反馈只有当前视觉和动作计划，没有直接读取产生该计划的世界表征。相同动作序列可能对应不同目标位置、遮挡或接触阶段；压缩动作序列不保证保存这些区别。新增 context 的目标是保留原 WAM 已付费算出的场景信息，让当前观测选择其中与纠偏有关的内容。

令 R 为当前教师相对于旧计划的有界残差，X 为旧反馈可见信息，C 为缓存世界特征。平方误差下，理想预测器加入 C 的风险差满足

`E[tr Cov(R|X)] - E[tr Cov(R|X,C)] = E[||E[R|X,C]-E[R|X]||²] >= 0`，仅对有效 arm 坐标求和。

这是关于同一数据分布、足够表达能力和理想优化的结论；不意味着有限数据网络必然提升。若 C 已被 X 完全概括、池化丢失关键细节、或旧预测过期，增益可以为零甚至出现估计误差。真正需要观察的是：真实控制分布中，fresh context 是否优于初始化、零修正、关闭 context，以及同任务不同计划的打乱 context。预注册开发门要求至少比关闭 context 改善 1%，不是靠低训练 loss 宣布成功。

保留完整 WAM 是为了避免再次用小模型重学整个动作生成问题。零初始化仅保护起点；有界残差和提前重规划仅限制改动范围，不构成闭环稳定性或成功率不下降的证明。

## 与 AHA-WAM / RTC 的关系

[AHA-WAM](https://arxiv.org/html/2606.09811v1) 把慢世界分支形成的上下文供快 action DiT 使用，并用当前观测进行路由及偏移训练；[官方代码](https://github.com/serene-sivy/AHA-WAM) 的 OVCR 有零初始化的上下文修正。这里借鉴的是“已计算世界信息可复用、读取由当前观测决定、训练要覆盖消费时的偏移”，不直接移植其双 DiT 训练配方。我们的快路径是受限反馈头，慢路径仍保留原生完整 WAM。

[RTC](https://arxiv.org/html/2506.07339v1) 强调推理与执行重叠以及已承诺动作的一致性。这里首先实现实际进程重叠、时间戳和过期动作处理，并保留当前已开始的控制段。没有实现其完整 diffusion inpainting/guidance，不将它称为原论文 RTC 的等价复现。

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) 提醒模仿学习的行为分布会受策略自身影响。因此 stage2 使用实际 graph-zero 执行时的观测；stage3 再采 stage2 策略实际执行后的观测，并与旧数据按 1:1 混合。这里教师是冻结 WAM，不是无误差专家，不能套用无条件性能保证。

## 新 benchmark 的时间契约

`.92` 只读参考其物理时间运动、独立推理进程与 RTC 控制思路；不修改该服务器。`.50` 新增独立 `runtime/domino_rt` 和 `scripts/eval_domino_continuous.py`，不替换旧同步评测入口。

物理进程以 4ms 步长推进，100ms 一个关节目标时间槽；推理进程可在 GPU 上进行慢/快计算，物理进程继续执行已有目标。没有“推理完成后补停表”“虚拟延迟结束再补执行”分支。所有开销仍计入墙钟，超过 100ms tick gap/backlog 标为基础设施无效，不偷偷改时钟。

真实场景预检发现原始三相机渲染一次约95ms，其中逐相机渲染约85ms；即使推理独立，直接在物理线程调用 get_obs 也会停顿。因此新入口增加独立渲染进程，保存相同 RT32spp/OIDN 画质。物理进程只发送不可变场景快照；渲染副本恢复全部实体、articulation root/qpos 和相机姿态，并核对几何/材质/光照身份及逐 link 正向运动学，不执行物理 step。图像保留快照时刻，不能把图像完成时刻伪装为采样时刻。队列只保留最新请求；计划切换前拍摄的过期分支帧丢弃。渲染仍占用 GPU，端到端计时包含其影响。

每个查询在真实物理时刻采样。动作第 i 行对应 `action_origin + (i+1)*100ms`。已过期行丢弃，新 tail 只在下一尚未开始的控制段原子发布；当前段保持连续。速度/加速度受限 Hermite 控制器按 tick 输出 drive target。无有效目标时保持命令，但世界继续推进；接触、边界、成功及任务 transient checks 每 tick 检查。

每 2 槽可请求一次快速反馈；计划 query age 达 12 时重新做完整 WAM，旧计划剩余 4 槽供慢推理期间执行。它不是固定每 16 槽刷新。若慢推理超过剩余覆盖，显式记录 hold/过期。单张 GPU 上慢/快查询顺序运行，不能声称快推理与慢推理并行；二者都与物理执行重叠。

这个控制器把 WAM 的动作行解释为 100ms knots，和原生每行 TOPP 可变时长不同。原 DOMINO 成功率不能直接移入新表。三主臂 `official_eager / graph_zero / world_context` 共用相同的连续物理、刷新、渲染、控制和超时协议。另在开发集合做 `context_off` 消融。

## 数据与训练

固定原自采 clean 分区映射，stage2 每任务前 4 个训练 episode，validation 每任务 1 个；stage3 使用后续 2 个训练 episode。不能用官方 700 个评测 seed 训练，也不能为准备失败换 seed。预先冻结选择清单、控制协议及身份。

在线只保存真实 query 的输入、plan/context/noise bundle 和实际行为身份，每 episode 最多 64 个非 fresh 样本。教师标签在轨迹结束后离线生成，避免额外教师推理改变行为延迟和控制轨迹。复用原 bundle 的 video noise、按 age 对齐公共 action noise suffix，减少无关采样差异；这不是教师方差估计。标签仅来自当前观测的完整 WAM，不混合未执行的专家未来。DOMINO `joint_action` 是命令目标，不是实测 qpos，不再把专家路径与未执行计划的差称为控制误差。

plan、text、context、noise 每 bundle 保存一次；raw/label/index 都有 SHA。离线标注不能改变记录输入。训练/验证按 episode 分离。stage3 每 microbatch 恰好半数旧行为、半数精确 stage2 权重生成的新行为。在线记录本身仍有复制/写盘开销：记录 `recording_wall_s`，物理时钟照常推进；不能把这部分时间扣掉后声称采集行为与无记录部署完全相同。

当前硬件实际 6 张 A6000：每卡 batch2、累积2，global batch24。stage2 4200 步即 100800 样本，lr2e-5；stage3 1200 步即 28800 样本，lr5e-6。预算按总曝光量明确记录，不谎称仍在用7卡。资源被占用时自动等待，不终止其他进程。物理采集和评测默认一次一个场景，避免未经验证的多实例 Vulkan/CPU 争用；训练仍用6卡，离线标注不推进物理世界。

## 自动执行与证据

服务器脚本 `run_world_context_rt_pipeline.py` 固化源码、配置、协议、数据选择、权重身份；顺序执行 CPU 检查、加载后真实模型 profile、真实连续物理开发采集、离线标注、多卡4步 smoke、固定预算机制探针及小闭环、正式数据/训练、stage3 新行为采集与训练、三臂评测、分析。关键门失败保存日志/权重/样本并停止，不能放宽门槛或自动挑 epoch/seed。

正式每臂请求同 35×20 场景；先保存一次 accepted expert motion、指令、初始化 RNG、任务标量状态，三臂复用同 SHA。修复 fresh env replay 遗漏 `arm_tag` 的问题。相同 accepted 输入不代表浮点物理轨迹逐位相同。准备失败属于缺口，不能算策略失败或用700虚假分母。

分析分别报告有效交集成功率、逐任务差异、配对不确定性、快/慢/冷启动耗时、真实消费动作加权成本、obs 到返回/安装/开始执行的时间、决策间隔、hold、丢弃和 wall/sim lag。Graph 加速与反馈的增量成本分开。模型调用耗时倒数不等于真实控制频率。

开发预检已在真实 DOMINO Python/Torch 环境验证9组原生 action Graph 输出与 eager 完全一致。第二轮合成图像 fixture 显示快路径 p95 8.834ms、Graph 完整规划均值211.991ms、eager317.096ms，按12槽内一次规划和5次快反馈摊销21.336ms/槽，相对eager26.425ms/槽降低19.26%；相对Graph零反馈仍增加21.21%计算。这只是算力可行性证据，不是正式时钟有效性或任务成功率。流水线对最终冻结源码及实际训练权重重新测量，不复用旧 profile SHA。

启动入口为 `bash scripts/launch_world_context_rt_20261008.sh`，固定新run路径且已有目录则拒绝重复启动。根盘空间曾两次耗尽；本次将自己的conda已下载安装包校验归档到SSD，并把uv缓存逐文件校验迁移、保留原路径软链接。归档与清单在 `/SSD_DISK/users/wuruihan/cache_archive_20261008`，没有删除训练数据、权重或其他用户文件。流水线临时文件与编译缓存放SSD。

旧会话 heartbeat 已暂停。后续由服务器自身流水线推进并写 `status.json`、逐阶段日志及分析文件，不依赖聊天轮询。

## 真实连续时钟预检与交付边界

`outputs/evaluation/world_context_rt_dev_20261008/bench/mock_shoe_replica_r6` 已通过真实 DOMINO 场景、独立渲染和 250ms 模拟推理测试：2.000s 物理对应 2.00221s 墙钟，500 ticks，最大 tick gap/lag 分别22.346/22.345ms。每个已完成的真实 worker 工作窗口内物理推进244–252ms；渲染最长117.06ms而相机工作期间仍推进341 ticks。10帧快照 FK/零物理步 oracle 通过。这个结果证明连续执行框架，不能替代跨进程 RGB 像素检查、真实 WAM 与渲染争用测试或学习反馈成功率。

所有新规划的随机噪声现在由明确的 task/episode/capture slot seed 驱动局部 generator；warmup 和此前查询数量不会改变相同查询的噪声。离线教师显式使用记录 SHA 派生 seed。CPU 回归验证两个噪声流、全局 RNG 隔离、离线 override 和非法 seed 拒绝。相同 seed 仍不能把不同策略的连续动态轨迹称为逐位相同。

正式流水线在长训练前核验同一 accepted 场景的 RGB 一致性和真实 WAM 物理时钟；在学习探针/各阶段闭环后要求至少有非零纠偏目标实际进入正时长控制段。若四行全部过期，不会仅凭快路径算力 profile 进入后续阶段。这个消费门仅证明控制器收到修正目标，速度/加速度约束后的真实机械响应仍由闭环结果衡量。

根盘持续满盘问题另通过逐文件 SHA 校验迁移自己的 Conda 包缓存、项目旧 `.venv` 和 pip 缓存到 SSD，保留原路径软链接；安装环境、数据、权重和其他用户文件均保留。新进程的临时文件、CUDA/Triton/GL/XDG 缓存全部放 SSD。服务器自动推进不调用任何 LLM；失败写明阶段与原因后停止，不静默重试、改 seed 或启动会话监督。
