# Route1 v3.3：信息保留审查与训练决策（2026-10-16）

本轮目标是补齐信息与训推链路，并做可归因的验证；结构合理不等于成功率已提升。新分支为 `feat/route1-retention-recipe-v33-20261016`，工作树位于 .50 的 `/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/joint-motionkv-20261015`。本文件的运行结果在实际测试后补齐。

## 历史比较能说明什么

旧 bounded-feedback 在 29 个任务上 242/580=41.72%，原 full 为 231/580=39.83%；已完成的旧 MotionKV 连续慢钟评测对应 29 任务为 162/580=27.93%，全 35 任务为 182/700=26%。二者只有 517/580 场景 seed 重合，旧同步 TOPP 与新连续固定 0.16s 的物理协议也不同。旧版每 4 个命令的物理时长中位数约 1.616s，新版为 0.64s。因此这是退化警报，不能把差值全部归因于网络信息损失。

旧版的优势是保留完整 16 步联合计划，只改有界的机械臂残差，抓手两通道保持原计划。新快路每次重新采样；即使 K/V 改变量有界，也不保证非线性的 Action DiT 输出有界，更不保证抓手事件一致。v3.2 已补回同噪声 2 次联合 + 8 次 Action 完整源计划、按物理时间对齐的软先验、逐去噪步的动作提案反馈及 source Action FM 保留。原生运动 token 始终保留。完整计划是软条件，不等于旧版的抓手硬保护；不能把功能 smoke 当作恢复抓取稳定性的证据。

## 本轮定位的具体缺口

1. 源观测已有 action_target_offsets，但未送入原生 Video/Action 联合调用。动作索引和扩散噪声 timestep 都无法表达第一目标距离观测还剩多少物理时间。补入零初始化的小型物理时间编码，作用于 16 个源动作 token；原生跨注意力将该信息传给 Video。2 次联合、8 次 Action 尾部和 source FM 使用相同端点时间，不改变 Video token 布局。快路已有自己的精确目标时间编码。
2. 训练以实际推理延迟推算 0–4 步 prefix，部署稳态固定承诺 H4。新增显式 fixed_h4_v1；实际 ready latency 与承诺 horizon 分离，当前稳态前缀固定 4，源侧 0 只代表 bootstrap 条件丢弃，稳态为 4。旧采样模式仍保留可复现实验。模拟异步训练仍不等于真实策略访问分布。
3. 真实 replay 的 ready_wall_s 是 episode 相对墙钟，曾误写为绝对 receive_monotonic；source publication/consumption 则是绝对 monotonic。补实际接收时钟并严格拒绝缺少该字段的旧记录，避免伪造因果时间。
4. 补只读的物理端点 plan-switch 和抓手事件诊断。比较仅覆盖相同可编辑端点，忽略已承诺前缀和未知尾部；不通过强行平滑改变既有策略。失败 replay 保存为诊断，不冒充专家 BC。

## Stage1 与 step12000 的准确含义

已核对官方源码、下载 full 权重的配置及实际张量，而非只根据文件名推断。原链路是 PCA 统计 → Wan 切片初始化的紧凑 Video 蒸馏（Video FM、hidden/motion KD）→ initialize_motion_checkpoint 合并 Video 与 wo_motion 已训练 Action 并新建运动 token → Stage2 冻结 Video 训练 Action/运动 → Stage3 联合。

本地 `external/checkpoints/DynamicWAM_full.pt` 的元数据是 Stage3 / step40000，初始化源记为发布者机器上的 Stage2 step80000。上游 Stage1/2 的完整训练日志未在本地核实。

`outputs/training/motionkv_rtc_adapt_8gpu_20261014/exports/multirate_step_12000.pt` 是我们从官方 full 做的 12k 步 MotionKV 分频适配。实际对比：Video 339/339 张量逐位相同；Action 131 个中 130 个改变；运动模块 10 个中 8 个改变，mean/scale 与 action normalizer 保留。它不是官方 Stage2 或 Stage3 的第 12000 步。

当前无需重跑 PCA / Stage1，也不要再次调用 initialize_motion_checkpoint 重置已学运动。新增源时间编码和快路接口不改变紧凑 Video 的宽度、深度、latent/text 布局；应从已训练 full/适配权重继续，用新物理时间数据做 Video FM 联合适配。若以后改变压缩结构/特征投影/教师契约，或测得明显视频预测遗忘，才有理由重建对应 PCA/蒸馏或加入保留蒸馏。Video 解冻本身不是重训 Stage1 的理由。[官方代码](https://github.com/Autumn1337/DynamicWAM)

## 下一轮 recipe

| 阶段 | 初始化与训练对象 | 数据/损失与验收 |
|---|---|---|
| D：数据准备 | 冻结 Wan VAE 编码 | 精确物理网格、当前/源图像与运动、动作真实时间；新未来 latent 与输入网格对应，完整覆盖/hash 后才放行正式训练 |
| P：初始化对照 | A=官方 backbone+同一已训练旧接口；B=step12000 backbone+同一旧接口；新模块同 seed | 固定数据/噪声先无更新诊断，再相同预算短 pilot；不能用 4 步 smoke 或初始 MSE 决定最终初始化 |
| W：接口与源计划适配 | 选定 P 的起点，Video 暂冻；训练 Action、原生运动、时间编码与更新器 | fast suffix FM + source Action FM(0.1) + motion forecast(0.1)；固定 H4、真实物理端点，保护完整源计划 |
| J：联合优化 | W 的完整权重；Video、Action、运动更新器共同训练 | fast FM + source Action FM(0.1) + source Video FM(0.1) + motion forecast(0.1)；旧 cache 的独立快查询与后台源快照语义一致 |
| R：真实分布细化 | J 权重 | 独立 development 场景采集真实延迟/cache age/旧队列/动作替换，成功自模仿或专家修正才作为 BC；失败留作诊断，hold/超时样本不能伪装成成功轨迹 |
| E：冻结验证 | 严格 full/fast reload、CUDA Graph 数值一致后部署 | 1x 连续物理钟、H4/R4/K10/refresh16，同场景/seed 对比；报告成功率、源/快路尾延迟、时限违约、抓手切换、MP4 |

现有候选预算 W=2k、J=18k、R=2k；8 卡每卡 batch1、累积2，有效 batch16。W 的 Action/interface LR 为 1e-5/1e-4；J 的 Video/Action/interface 为 2e-6/5e-6/3e-5；R 为 1e-6/2e-6/1e-5。这些是待 pilot 验证的起点，不是已证明最优 recipe。W+J 合计仅约 32 万次样本呈现，不能等同于完整遍历 141.6 万个物理采样点，更不能据此认定新增模块已收敛；实际训练长度要依据分桶验证曲线与闭环结果调整。R 只有在真实合格样本存在时才能运行。正式训练不能使用 4 步 smoke 权重初始化。

RTC 训练需要把部署的承诺前缀与旧动作分布纳入条件和损失；单纯推理时混合旧动作会带来新的分布偏差。保持前缀并不保证可编辑尾部没有跳变。Soft RTC 的先验去噪值得后续对照，但不能在当前 checkpoint 上直接加混合并声称训推一致。[训练时 RTC](https://arxiv.org/html/2512.05964v2)；[动作先验去噪](https://arxiv.org/html/2605.25537v1)

## 性能论证的范围

新框架能获得旧 bounded updater 缺少的“源计划之后的新运动、新图像和新 action proposal”，同时保留源计划、Video K/V 和原生运动交互，具备更优的条件信息与适应能力。但这些条件不是充分统计量，粗光流和低秩更新存在表示限制；数据、优化和闭环分布仍决定实际收益。

后台慢路消除快路等待，静态 CUDA Graph 降低采样开销；这支持吞吐/时限改善，不自动证明反应更快。当前 H4 固定 0.64s 承诺窗口；即使采样 60ms，执行器也不允许越过承诺边界立即改动作。缩短 H/R 应作为单独的训推同时修改实验。完整源计划仅覆盖 2.56s，cache 最长 4.96s；超出计划端点必须 mask，旧 K/V 可作为历史线索但不能被当作仍有效的未来预测。

必须用固定协议的消融分辨：完整源计划、源时间编码、运动创新、action proposal 修正、联合训练的各自作用；通过实际闭环的成功率/抓手事件与 tail latency 才能接受性能改进。

## 本轮执行证据（待实际测试后更新）

不启动正式训练，不建立定时巡检。完整 VAE 会使用独立冻结工作树启动；运行中不等于完成。GPU 全部通过临时 all-adapter FD guard 管理，不进行任何驱动/PCI恢复。
