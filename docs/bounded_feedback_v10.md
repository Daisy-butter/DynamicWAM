# Native TOPP 分频残差 V10：历史机制审计与可执行修订

本修订继续当前分频路线，不退回旧的 192 维、2 层反馈头。部署仍是 Full 冻结慢路、448/6/7 独立快路、fast1/slow4、原 DOMINO TOPP；后续结果只在这一原生协议内比较。本文是 11 月 2 日完整设计文档的实现修订，冲突处以本修订和代码契约为准。

## 1. 历史正向实验到底支持什么

原始配对审计为共同 29 任务、580 个相同 seed：official=231、zero=231、feedback=242。feedback 相对 zero 新增 32、丢失 21，配对 bootstrap 95% CI 为 [-0.345,+4.310] 个百分点。总成功相同也不是逐 episode 相同；zero 与 official 各有 25 次成败互换。旧的 native/TOPP 结果不能与连续仿真结果混分。

旧模型没有替换 Full 动作生成器。完整 2 joint + 8 action 生成 A16；每 4 命令读新 RGB、state、语言、完整计划和 state-plan 差，输出最多 0.1 normalized qpos 的四步修正。夹爪修正为零；age0 为零；原计划不可变。Full 每16命令刷新。主要训练目标是当前状态 Full 教师相对旧计划的有界变化，专家修正为辅助；当前教师和源教师对重叠绝对命令共享 action noise，video noise 也配对。它不是仅靠专家动作训练一个新策略。

这提示四个可检验瓶颈：①完整动作先验保留；②随机模态变化污染反馈目标；③新观测、运动和真实到达状态是否进入动作；④反馈样本是否覆盖非零年龄。没有证据能将 +11 成功唯一归因到某一模块，也不能保证更大网络一定提升 SR。

来源：历史 `plan_feedback_v1_result_analysis_20261006.md`、`models/plan_feedback.py`、`runtime/ciwam/wam/plan_feedback_policy.py`、`scripts/train_plan_feedback.py:177–238`。历史底座 SHA 为 7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7；反馈导出 SHA 为 be393d437a5c750b4642e8f685a86971a86cea07810b8e1a3f357014cb3bc17d。源快照无 .git，不能把某单文件匹配的 commit 冒充整个运行身份。

## 2. 当前架构与信息通路

- 慢路：保留官方 Full 的 Video/Action/native motion、原 2+8 采样。独立子进程只读冻结观测快照；不触碰仿真器。fast1/slow4 由命令序号决定；物理时钟只描述观测/历史年龄，不选择动作行。
- 快路：单次确定性 448维/6层/7头交互。A token 以精确命令对齐的 Full A16 为数值起点；12 个可学习 P token 对应 target/左夹爪/右夹爪 × +1/+2/+4/+8 事件。当前观测、历史世界、预测世界分别有 mask/attention，P 和 A 在同一主干交互。一个 action decoder 输出 arm residual；不存在第二套动作生成器或独立扩散采样。
- 动作：`A = A_full_aligned + valid_nonfresh * arm_mask * radius * tanh(head)`。arm 半径为 0.06 raw radians，再按 checkpoint std 转为网络单位；不累积写回 A_full。夹爪目前保留慢路事件，属于首轮可辨识设计，后续可独立研究事件修正。fresh0 精确为零，零 head 精确恢复 Full proposal；这不等于整个闭环原 baseline SR 保底，因为 refresh/execution cadence 已变化。
- CCT 启发的交互：最近8条已完成命令，读真实 articulation qpos/qvel，记录 command、before/after arm q、after velocity、command-achieved error、实际 duration 和 source age，形成64维记录，由两层4-query bottleneck转成4个当前上下文 token。原 Full 的 drive-target 输入语义不改。夹爪缓存值不冒充实际夹爪位置。没有真实记录的专家样本使用缺失 mask。
- OSWAM 启发的单次预测：动作不再从新 A/Z 噪声做10步求解，而从完整慢计划和确定性 P token 做一次预测。未来标签只监督 action-conditioned world decoder，绝不输入动作主干。此轮没有宣称复现 OSWAM 的 privileged-feature teacher/AIT/SIT；独立特权教师适配器不在本轮必需训练路径，避免将未经接线的模块写成已实现能力。
- 未来预测：action decoder 之后，一层 world decoder 读取实际指定动作序列和 P hidden。训练专家未来时显式给专家动作；真实回放没有未来标签则全部 mask。部署给最终快路动作，投影后未来预测标记失效。预测 Z 不反过来冒充已执行运动。

## 3. 训练契约

每个 query 使用旧 Full source 和当前状态 Full teacher；按绝对命令共享底噪，且 source/current teacher 的 video noise 配对。共同噪声降低模态方差，不保证同一策略模态。Full 参数冻结，慢路生成和教师均 no_grad。

主动作损失对未裁剪当前教师 endpoint 使用按物理 radius 标准化的 Huber；输出本身严格有界。越界目标仍提供有限的局部方向梯度，并记录 reachable_fraction，不把投影后的动作冒充到达教师终点。专家辅助只作用于非 replay 且与教师修正方向不冲突的行；夹爪事件冲突的监督另行屏蔽。保留项只惩罚不能监督的非零修正；轨迹项拟合教师差分，而不是直接压低动作速度。

跨查询：同绝对命令对齐旧 proposal，作为软条件；专家配对窗口重算本模型前一次预测，不把专家 previous action 作为本模型输出。实际回放记录当时的 proposal。continuation 按教师一致性加权，不能成为旧错误计划的硬承诺。

数据分三种：expert bootstrap、真实 native collection replay、二者混合。前者必须包含非零命令年龄；不能继续 V9 只保留 age0 的过滤。后者必须有原 source 身份、真实 exposure/publication/命令时钟、完整交互 mask；不能借用另条专家轨迹的未来。当前预约的初始 replay 来自零初始化反馈策略，属于 baseline-policy rollout，不是训练完成学生的 on-policy 覆盖；后续学生分布扩展需单独数据版本和训练验收，不能称本轮已完成 DAgger。

半量方案按 task/style 选完整 train episode，固定种子20261102；不动 validation/test，不重拟合原统计。缩减固定更新预算到500/1500/3000，有效 batch16（8×1×2）；不是声称删数据就自动缩短固定步数训练。正式全程从官方 Full + 新快路初始化；诊断权重禁止成为正式起点。

## 4. TOPP、异步发布和缓存边界

原生推理只消耗墙钟，只有 TOPP推进物理；不会加入连续 catch-up、固定45tick、RTC硬前缀或C2执行器。动作按整数命令，观测/运动按FP64真实曝光时间。slow snapshot 的已选首 waypoint 保持精确；新参考发布带 source 时间，不改成 publication 时间。旧 action draft 是否覆盖当前命令和 world TTL 是否有效是两项判断。

可执行 draft 缺失时在原生边界等待 fresh Full，禁止执行 masked suffix。新 query 的所有输入都重新准备；CUDA Graph 逐次复制 draft、correction mask、context K/V、world gates，不能借用旧缓存。episode reset 必须丢弃旧 plan、交互历史和proposal。原TOPP、原预算、碰撞、成功判定与 MP4 目录保持不变。

## 5. 必须先通过的 checklist

- CPU：零head/fresh/gripper精确还原；残差半径；非零head下 observation/M/P/真实interaction影响动作；masked suffix和缺失history无污染；future label不进入动作；配对命令噪声一致；旧模型回归。
- 数据：半量实际比例及每task/style覆盖、train-only、非零年龄/terminal窗口、原始motion/normalizer契约；实回放不能缺失mandatory字段或借用未来。
- GPU：现场空闲和归属、all-device FD先于遥测；真实Full eager/Graph（零/非零）、更换输入parity、独立慢路bootstrap精确比对。
- 真实DOMINO：完整episode、实际TOPP和发布/命令时钟、非零慢路重叠/新M消费、原生MP4完整解码、退出码与worker/guard关闭、内核日志不新增异常。
- 三阶段8rank短训、有限loss/grad、全部rank严格全状态恢复，烟测权重隔离。只有这些和数据/冻结身份同时通过，才允许formal。
- 预约只决定最早运行时刻，不替代任何门禁。02:00有他人进程则停止；若只能届时开始GPU验收，formal实际会晚于02:00。无自动重试、抢占或驱动恢复。

## 6. 审稿人需要的实验与结论边界

比较 original Full H16、同分频/同执行器零残差、完整当前模型；再做无真实interaction、无world/P交互、无paired-noise教师、无continuation、无未来辅助。共享任务、accepted scene、指令、seed和预算；记录模型服务延迟、摊销成本、观测到动作、实际TOPP时长、动作差分/命令翻转、接触阶段和SR。训练数据/步数/算力一致，不能把CUDA Graph收益算到学习模块上。最终35×20不是这次预约自动执行范围。

三轮审查和每轮修复/测试在 `docs/bounded_feedback_reviews.md`；当前 CPU 通过不代表GPU就绪、性能保底或已达到SOTA。所有缺口保留为明确 gate，禁止通过改写报告字段放行。
