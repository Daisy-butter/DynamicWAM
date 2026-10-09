# DynamicWAM 分频失败审查与保留完整规划的反馈方案

日期：2026-10-05。现有训练、评测和源码快照全部保留；本方案另建训练目录，使用官方 `DynamicWAM_full.pt`，不继承已退化的 K1 权重。

## 1. 当前结果能证明什么

原评测在 4 个任务的固定 seed 专家准备失败后结束，完整部分为 31 任务、620 episodes。26/700=3.71% 不是完整 benchmark 成绩；可核验的完整任务子集是 26/620=4.19%。同任务同 seed 的官方 H16 是 236/620=38.06%。再取与官方 H4 共同的 564 个 seed，官方 H4 为 23.40%，旧分频为 4.61%。因此不能用 H16→H4 单独解释退化。

| 指标（相同31任务） | 官方H16 | 旧native H4/K1 |
|---|---:|---:|
| 加权模型查询 ms/实际执行动作 | 21.450 | 16.291 |
| 总查询时间 | 3487.3秒 | 3283.9秒 |
| 实际执行动作总数 | 162577 | 201577 |

旧方案摊销下降24.05%，但失败增多、执行动作增加约24%，总查询时间只下降5.83%。快查询 p50/p95/p99 为9.275/9.713/10.101ms；已有观测进入adapter至动作返回为31.748/32.583/34.966ms。定期刷新分别为203.312/210.877/229.144ms和225.733/233.297/251.804ms。冷捕获查询p50约682ms。不能把9ms倒数当真实控制频率，也不能把inline刷新称作无阻塞RTC。

旧基线没有保存可重放的专家动态轨迹和指令；同seed仍不保证轨迹逐位相同。上述31任务也不是35任务的随机子样本，不外推完整最终成绩。

## 2. 实现排查

已排除的明显错误：

- 模型确有训练：正式Zero0优化器有152,323,182个FP32主权重，268组Adam状态为FP32，训练没有跳步，action expert、motion、conditioner都有更新。
- 训练forward通过包装模型反传；所有可训练参数进入optimizer。Stage3实际反传部署K1 sampler，不是只修改推理步数。
- checkpoint SHA与部署日志一致；恢复官方慢骨干后，快模型另从训练payload严格加载，未被覆盖回官方权重。
- 动作仍是14维绝对qpos；归一化与反归一化各一次。三相机输入、mask、state、motion、text、age、noise、KV在图重放前更新。
- 配对标签按原始视频帧偏移对齐。训练使用已经VAE编码的condition/future latent；旧native没有更新视频模型，也没有把GT未来作为缓存输入。新方案同样不需要重采或重做stage1。

发现的确定实现问题：

1. **第二步视频时间嵌入不等价。** 官方 `HeadFlowRunner` 预先把所有video timestep固定为WAN的BF16精度。WAN输出头却返回FP32，第一次更新后future会升为FP32；旧teacher/bridge按 `timestep.to(future)` 动态转换，第二步变成833.3333，而官方是BF16舍入的832。修复为branch独立的固定timestep精度。只修下面的scalar问题后，整条路径仍有约0.0128的qpos差；修正timestep后才通过实际runner的逐元素相等检查。
2. **共享Euler辅助函数的标量精度错误。** `flow_schedule.euler_update` 曾将CPU的FP32零维sigma差搬到CUDA，与低精度预测相乘时不再等价于官方CPU标量乘法。修复以FP32做低精度预测的乘积、再按官方结果dtype舍入，action FP32路径不变。CPU/CUDA共150个vendor step反例测试通过，旧实现53例不等价。但此处WAN实际预测为FP32，因此不能把这项单独称为上面完整输出差的原因。
3. 旧快路径部署还把action expert整体转成BF16，而官方保留FP32；旧eager/graph自比较不能排除相对官方的数值分布变化。新完整规划器固定保留官方的video BF16、action FP32及solver精度。

前两项已修复，并新增实际vendor scheduler和实际部署runner作为独立oracle。旧teacher与优化采样器互相对照会漏报它们共享的偏差。尚无单独闭环消融证明这些数值差异能解释全部退化；旧结果不被修复后的源码追溯替换。

主要设计风险：

1. 教师在当前观测上执行完整两步联合前缀和后八步动作生成；学生却以历史缓存加当前小视觉编码器，从全新噪声开始。条件和生成轨迹同时改变，并非同条件ODE步数压缩。
2. H4反复丢弃16步计划的尾部、重新采样，可损害轨迹连续性。官方H4自身也比H16弱，但不能解释剩余差距。
3. 96条训练pair抽样的age4/8/12实际时间中位数为0.600/1.316/1.896秒；在线为2.024/3.964/6.024秒，age12在线p95达12.068秒。动作计数相同不保证未来语义有效。TOPP耗时也受预测动作幅度影响，所以这是明确分布偏移及可能的反馈放大，尚非独立确证的首因。
4. 旧stage2仅验证随机扩散时间的速度场MSE；旧pilot只检查能运行、有限数值和seed配对。没有闭环质量门，低loss不能支撑“保留原WAM性能”。

## 3. AHA-WAM中应该参考的内容

其蒸馏代码用教师rollout产生的同一 `video_state` 与真实中间去噪状态监督学生，保持条件接口；先进行CUDA优化，再压缩步数。主Flash质量对应两步，不应把一步延迟与两步成功率混用。

它的注意力设计与当前DynamicWAM的联合未来缓存也不同，不能仅凭双DiT外形直接照搬。这里保留DynamicWAM真实的未来—动作联合计算，以输出的动作计划作为快反馈的明确参考，而不再把历史KV宣称为按物理时间准确切片的未来。

来源：[论文](https://arxiv.org/html/2606.09811v1)、[官方蒸馏实现](https://github.com/serene-sivy/AHA-WAM/blob/main/src/ahawam/models/wan22/distill/ahawam_ode.py)。

## 4. 新架构：完整WAM规划 + 有界视觉反馈

慢路保持官方完整10步action/2步video联合采样，每16条已执行命令重新生成16步计划P。不改原WAM权重、不缩减其未来输入、不截断联合前缀。只把最后8步动作计算捕获为CUDA Graph，以减少调度开销。

快路每4条已执行命令读取新三视角RGB、当前state、相对参考计划的跟踪误差、语言和完整参考计划。一个小网络输出下一窗口4条动作的残差：

`u = P[age:age+4] + 0.1 * tanh(feedback(...))`，其中age仅为0/4/8/12。

- age0强制残差0，刚生成的首4步直接执行。
- 抓手通道6/13强制残差0，保留原WAM的开合事件。代价是快路不能独立提前/延迟抓手事件，需在结果中披露。
- 0.1的单位是原mean/std归一化后的qpos，并非0.1弧度或10%关节行程。
- 残差不写回参考计划，避免逐次累积漂移；不超出当前16步计划，不外推其尾部。
- 输出层零初始化。零反馈必须在相同实际输入与随机噪声下，四窗口拼接逐元素等于官方完整16步输出。
- 快路不再运行动作扩散或读取大KV。世界与动作的耦合保留在完整原生规划中；快路只是局部反馈，不宣称更新了完整反事实未来。
- 快路不消费光流，因此只积累相机历史，在16步慢刷新时计算光流。记录模型查询与adapter预处理两个时间边界。

这把“改变采样分布”和“增加视觉反馈”分开。保留原模型及零修正等价性给出明确回退点，但非零反馈仍可能损伤闭环，必须通过评测验证。

## 5. 训练与验证

Stage1无需重训：数据、VAE、视频模型及官方完整采样能力均保留。新stage2/3的名称表示反馈学习/精调，不是重新训练原视频或action expert。

- stage2：官方完整模型冻结，只训练FP32反馈头；7卡，batch1、累积2，全局14，计划7000步。
- stage3：同一架构、小学习率继续2000步；不再突然压缩成K1。
- 数据沿用原自采DOMINO Level-1及按episode划分的holdout。源时刻由官方模型生成完整计划；当前时刻teacher提供短窗口目标，专家动作作小权重锚。source/current重叠动作噪声共用以降方差，但不宣称保证同一行为模态。
- 监督有界残差，只计算可修正的arm通道及age>0。单列teacher/expert目标被截断比例、符号冲突、zero与feedback误差，避免低loss掩盖“只能做很小修改”。
- video latent/noise使用BF16，action/state/solver为FP32；反馈参数及Adam状态FP32，前向BF16 autocast，训推一致。
- 训练前：CPU契约与梯度测试、实际vendor/runner逐元素对照、完整查询延迟实测、7卡smoke与导出加载检查。
- 训练后：零反馈及新反馈在预先选定3任务×2episodes做闭环canary；通过后两臂各35任务×20episodes，使用固定清单。seed准备失败明确保留incomplete，不能换seed或当策略失败。

速度预算是 `(完整优化WAM查询 + 3次快反馈)/16`。必须同时报告原eager H16和同样Graph优化的H16：相对前者的加速含工程优化；相对后者新增反馈会增加计算，换取更频繁观察和修正，不能宣称纯架构降低全部算力。

## 6. 实现入口

- `models/plan_feedback.py`：4步有界反馈与显式plan切片。
- `inference/native_plan_graph.py`：完整原生规划，后8步Graph。
- `runtime/ciwam/wam/plan_feedback_policy.py`、`adapters/domino/deploy_policy_plan_feedback.py`：保存完整参考计划、4步反馈及分项计时。
- `scripts/train_plan_feedback.py`：独立冻结教师训练、holdout、严格导出。
- `scripts/profile_plan_feedback.py`：真实runner零反馈等价性、完整查询预算。
- `scripts/run_plan_feedback_pipeline.py`：smoke、latency门、两阶段训练、canary与完整评测；失败停止并保留日志。

## 7. 已完成的预检与正式任务

2026-10-05，A6000单卡，同一合成packet、缓存文本，每项30个测量样本：

| 查询 | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| 官方完整eager | 308.98ms | 305.60ms | 321.21ms | 357.10ms |
| 新完整Graph规划 | 222.68ms | 220.19ms | 236.87ms | 245.50ms |
| 新视觉反馈 | 5.61ms | 5.58ms | 5.77ms | 5.85ms |

摊销为14.970 vs19.312ms/action，下降22.48%；相对同样Graph优化的纯H16则增加7.56%。首个冷规划查询970.69ms，其中graph capture约701.83ms；这只是一次冷启动样本，不能称为冷启动p95。计时包含VAE、RGB转化、完整10/2采样、反归一化和CPU动作返回，不含相机采集、外部Farneback、文本冷加载和环境执行。最终正式评测会另记adapter端到端边界，不能把这些数值直接当真实反应频率。零初始化反馈头仍执行完整反馈前向；此表不是训练后的精度或质量结果。

已通过：26项CPU契约/训练/pipeline/WAN包装层检查；150组vendor step CPU/CUDA精度对照；9类条件下与实际HeadFlowRunner逐元素一致；零反馈4×4窗口拼接与原生16步完整输出一致；bridge K/V一致。单卡真实数据4步预检已完成，65/65组参数参与梯度，2,236,974个反馈参数可训练，teacher可训练参数为0，32条holdout覆盖全部age，导出成功。这些检查证明实现可运行、基础路径等价，不证明学习后的闭环反馈有益。

最终预检原始记录：

- `/SSD_DISK_1/users/wuruihan/DynamicWAM/outputs/evaluation/native_repair_preflight_20261005/plan_graph_actual_oracle.json`
- 同目录 `scalar_precision.json`、`plan_feedback_latency_final.json`、`single_smoke/training/metrics.jsonl`。

已启动独立冻结源码流水线：

`/SSD_DISK_1/users/wuruihan/DynamicWAM/outputs/training/plan_feedback_v1_20261005/`

其 `status.json` 为状态真源，`source/` 为本轮冻结实现，`source_manifest.json` 与 `pipeline_source_manifest.json` 保存SHA；7卡smoke及加载后延迟门通过才进入正式stage2，再连续stage3、canary、两个完整评测臂。最初启动PID为331982。正式训练/评测完成前，不声称新成功率改善或已保住原WAM性能。

启动后确认：七卡smoke已完成4步、全局batch14，导出后加载的零反馈等价检查再次通过。加载后的预算为14.854ms/action、相对原eager比值0.7798，延迟门通过；协调进程已转入正式stage2（7000步），随后计划stage3（2000步）和上述评测。已有自动跟进已切换至本轮新状态文件，正常运行时不发送例行通知。
