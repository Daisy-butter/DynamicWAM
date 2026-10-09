# Plan-feedback v1：配对成功率与延迟审计（2026-10-06）

**核心判断：在三臂共同完成的 29 个任务、580 条相同 seed 上，feedback 为 242/580，zero 为 231/580，官方 eager H16 为 231/580。反馈的 +1.90 个百分点尚未得到统计显著支持。实际 query 计算成本从官方的 21.52 降到 17.07 ms/执行动作，但同样优化过的 zero 仅为 15.82；主要提速来自慢规划的执行优化，反馈头额外增加计算。**

本次分析只读远程日志、aggregate 和状态文件；未启动 GPU、未恢复失败任务、未修改远程源码或结果。报告以完整共同任务为主，不将不完整任务冒充 700 条正式结果。

## 1. 数据口径与配对边界

| arm | 完整任务 | 已完成任务内 successes / episodes | 主分析共同范围 |
| --- | --- | --- | --- |
| official eager H16 | 35 | 257 / 700 | 231 / 580 |
| zero：官方 16 步计划 + CUDA graph，4 步返回剩余片段 | 31 | 235 / 620 | 231 / 580 |
| feedback：相同慢规划 + 已训练的有界反馈头 | 30 | 243 / 600 | 242 / 580 |

上表的 31-task 与 30-task 子集不同，因此不能直接将 235/620 与 243/600 当配对实验，也不能复用其他历史 31-task 子集的官方数字。三臂交集逐任务核验：每个任务正好 20 个唯一 seed，三臂 seed 列表逐项一致；所有 success 为 bool，逐 episode 求和与 task aggregate 一致。zero 与 feedback 的 580 条 instruction 文本也逐条一致。

**重要：当前 suite aggregate 的 `success_rate` 使用 requested 700 作为分母。** 因任务未完成，其 235/700、243/700 是 suite 的保守占位口径，并非完整样本的策略成功率；`episode_seeds_match=false` 同样反映整个 35-task suite 尚未完整。本报告使用重新验证过的完成任务和明确分母。

固定 seed 和相同指令仍不等于接受后的 expert motion 记录完整重放。本次没有证据证明旧官方臂与两新臂的准备过程、物体初始运动轨迹逐帧相同；旧官方臂亦没有可核验的逐 episode 指令日志。因而这是严格 seed 配对、部分环境条件已核验的比较，而非消除了全部模拟器非确定性的因果实验。

两新臂 manifest SHA256：`821c09a4b62235620ee4fdcd56d806bbd74e90933215bf5df1b7ce22ecf18c26`。它们使用同一 stage3 export，raw timing 中 checkpoint 为 `stage3/exports/feedback_step_2000.pt`，base checkpoint SHA256 均为 `7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`；zero 的 `feedback_enabled` 全为 false，feedback 全为 true。

## 2. 成功率：有正向点估计，没有稳定提升的结论

| 比较（右减左） | 差值 pp | 配对分层 bootstrap 95% CI，pp | 新增成功 / 丢失成功 | 精确 McNemar p |
| --- | --- | --- | --- | --- |
| feedback − zero | +1.897 | [-0.345, +4.310] | 32 / 21 | 0.1690 |
| feedback − official | +1.897 | [-0.517, +4.310] | 33 / 22 | 0.1770 |
| zero − official | +0.000 | [-2.241, +2.241] | 25 / 25 | 1.0000 |

zero/feedback 配对表：都失败 317，都成功 210，仅 feedback 成功 32，仅 zero 成功 21。两臂有 53 条结果变化。官方与 zero 虽然总成功数相同，仍有 25 条新增、25 条丢失，不是逐 episode 行为完全等价。

主 CI 对固定的 29 个任务，在每个任务内部对 20 条三臂 episode tuple 联合有放回抽样；20,000 次、seed=20261006，使用 percentile CI。它保留配对与固定任务权重，不把三臂独立抽样。任务簇 bootstrap 的敏感性分析改为对 29 个 task 差值抽样，feedback−zero 的 CI 为 [−1.03,+5.34] pp；它针对不同的任务总体估计量，也跨零。

这些 CI 不覆盖“未评完任务是否更难”的选择偏差，也未预设非劣性界值，因此可以说“未观察到明显整体退化、点估计略升”，不能说“证明性能不降”或“已显著提分”。

独立重算较大两两完成子集，仅作补充：official/zero 的 31 个共同任务为 236/620 对 235/620；official/feedback 的 30 个共同任务为 232/600 对 243/600。它们不是三臂同一评估范围。

## 3. 逐任务结果（每行均为 20 episodes）

| 任务 | official | zero | feedback | feedback−zero | gain / loss |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | 3 | 5 | 5 | +0 | 0 / 0 |
| beat_block_hammer | 16 | 17 | 18 | +1 | 2 / 1 |
| click_alarmclock | 3 | 3 | 2 | -1 | 0 / 1 |
| click_bell | 0 | 0 | 0 | +0 | 0 / 0 |
| dump_bin_bigbin | 6 | 6 | 6 | +0 | 1 / 1 |
| grab_roller | 11 | 11 | 11 | +0 | 1 / 1 |
| handover_mic | 5 | 5 | 7 | +2 | 2 / 0 |
| hanging_mug | 5 | 1 | 7 | +6 | 6 / 0 |
| move_can_pot | 3 | 2 | 3 | +1 | 1 / 0 |
| move_pillbottle_pad | 16 | 15 | 15 | +0 | 1 / 1 |
| move_playingcard_away | 0 | 0 | 0 | +0 | 0 / 0 |
| move_stapler_pad | 7 | 7 | 12 | +5 | 5 / 0 |
| place_a2b_left | 2 | 1 | 1 | +0 | 0 / 0 |
| place_a2b_right | 1 | 0 | 0 | +0 | 0 / 0 |
| place_bread_skillet | 4 | 4 | 4 | +0 | 1 / 1 |
| place_container_plate | 15 | 20 | 18 | -2 | 0 / 2 |
| place_empty_cup | 20 | 20 | 20 | +0 | 0 / 0 |
| place_fan | 9 | 11 | 12 | +1 | 3 / 2 |
| place_mouse_pad | 10 | 10 | 9 | -1 | 3 / 4 |
| place_object_scale | 8 | 7 | 7 | +0 | 1 / 1 |
| place_object_stand | 10 | 11 | 11 | +0 | 1 / 1 |
| place_phone_stand | 16 | 16 | 15 | -1 | 0 / 1 |
| place_shoe | 19 | 19 | 18 | -1 | 0 / 1 |
| press_stapler | 11 | 12 | 11 | -1 | 0 / 1 |
| put_bottles_dustbin | 5 | 5 | 4 | -1 | 0 / 1 |
| scan_object | 0 | 1 | 1 | +0 | 0 / 0 |
| shake_bottle | 6 | 5 | 6 | +1 | 1 / 0 |
| shake_bottle_horizontally | 6 | 8 | 7 | -1 | 0 / 1 |
| stamp_seal | 14 | 9 | 12 | +3 | 3 / 0 |

29 个任务中 8 个提高、8 个降低、13 个持平。增益主要集中在 hanging_mug（+6）、move_stapler_pad（+5）、stamp_seal（+3）。hanging_mug 的 zero 低于旧官方（1 对 5），所以其增益有一部分可能是恢复原基线表现；20 条的小样本不适合做逐任务显著性或机制归因。

## 4. 延迟：使用实际执行动作加权，避免查询频率混淆

下表主指标严格为 `sum(query_ms) / sum(executed_steps)`，包含冷启动；不是各 episode ratio 的简单平均，也不是每次查询均值。query 定义为 policy.query 调用起止；adapter 端到端定义为收到 observation 后进入 adapter 到动作返回，包含前处理但不含上游图像获取/渲染、动作执行或 RTC 通信。

| arm | query 数 | 实际动作数 | query 平均 ms | query ms/动作 | adapter-inclusive ms/动作 |
| --- | --- | --- | --- | --- | --- |
| official | 9,028 | 140,868 | 335.86 | 21.52 | 旧日志未记录 |
| zero | 35,139 | 139,982 | 63.02 | 15.82 | 17.56 |
| feedback | 34,789 | 138,549 | 67.99 | 17.07 | 18.81 |

反馈相对原 eager H16 的摊销 query 成本下降 **20.69%**；zero 相对原 eager 下降 **26.51%**；反馈相对 zero 增加 **7.92%**，即约 **1.25 ms/动作**。快反馈 query 平均 7.44 ms，而 zero 的同位置片段返回平均 0.90 ms。当前成功率差异也会改变 episode 长度和动作总数，故 ms/动作不直接等同于任务完成时间。

旧官方基线与新臂来自不同运行时段；日志未控制所有硬件负载因素。因此 20.69% 是本次测得的服务成本差异，不能全部归因为学习架构。两个新臂慢刷新约 235 ms 相同，zero 已经得到大部分提速，支持“CUDA graph 等慢路径执行优化是主要来源，反馈头为频繁视觉修正支付额外成本”的判断。

## 5. 快、慢、冷启动与 adapter 端到端分布

单位 ms；每格为 p50 / p95 / p99。`steady refresh` 排除每 episode 首次查询和发生 graph capture 的查询；`episode first` 与 `graph cold` 有交集，不可相加。

| arm | 分组 | n | policy query | adapter 端到端 |
| --- | --- | --- | --- | --- |
| zero | 快路径 | 26163 | 0.88 / 1.05 / 1.19 | 2.91 / 3.16 / 3.68 |
| zero | 稳态慢刷新 | 8396 | 233.25 / 239.99 / 262.01 | 255.72 / 262.72 / 284.96 |
| zero | 每 episode 首次 | 580 | 341.76 / 413.97 / 1123.23 | 345.69 / 418.18 / 1127.81 |
| zero | graph 冷查询 | 29 | 1106.78 / 1212.61 / 1791.68 | 1110.48 / 1216.31 / 1795.44 |
| feedback | 快路径 | 25910 | 7.29 / 8.05 / 8.99 | 9.32 / 10.24 / 11.56 |
| feedback | 稳态慢刷新 | 8299 | 233.71 / 239.17 / 261.67 | 256.20 / 261.99 / 284.61 |
| feedback | 每 episode 首次 | 580 | 341.42 / 412.29 / 1144.29 | 345.29 / 416.45 / 1148.10 |
| feedback | graph 冷查询 | 29 | 1095.92 / 1612.64 / 1862.08 | 1099.61 / 1616.38 / 1866.32 |

feedback 的全部查询混合分布：query p50/p95/p99=7.41/236.56/340.81 ms；adapter=9.45/259.23/344.64 ms。不要只展示快路径 p95=8.05 ms 而遗漏约四分之一查询需要慢规划。旧官方只保存 episode-level query summary；无法从这些分位数重新构造真实的全体 query p50/p95/p99，本报告不伪造旧官方尾延迟。

冷查询定义：同一个 worker slot.log 中 `planner_graph.captures` 比前一 query 增大；每个日志起始计数为 0。两新臂各有 29 次捕获，恰好每个共同任务 worker 一次，**不是每个 episode 重新捕获**。feedback 冷查询占全部查询 0.0834%，占慢刷新 0.3266%；zero 分别为 0.0825%、0.3231%。所有 query 的 graph 状态均为 enabled=true、eager_calls=0。

graph 冷查询耗时包括该次完整 policy 冷路径，不能当成纯 capture 时间。feedback 的纯 `last_capture_ms` p50/p95/p99=744.94/1251.57/1513.28 ms；每 episode 首次共 580 次，除 29 次首次 capture 外，还包含其他初始化/条件准备成本。冷组只有 29 条，p99 的不确定性大。

feedback 快路径 adapter 前处理均值 2.05 ms，稳态慢刷新前处理均值 22.39 ms。优化快头之外，慢刷新前处理仍有明确预算；稳态慢 query 的 p99 约 262 ms，但罕见最大值超过 1.5 秒，主 JSON 保留 max，不能用 p99 隐藏极端值。

## 6. 真实决策间隔和过期计划的物理年龄

相邻决策间隔以两次进入 adapter、需要查询策略的时刻为端点，排除每 episode 首次。wall 包含上次推理和随后执行；sim 来自场景时钟。它们不是一个 query 的服务时间，也不应使用 1/快头延迟推导实际控制 Hz。

| arm | 间隔数 | wall p50/p95/p99 ms | sim p50/p95/p99 ms |
| --- | --- | --- | --- |
| zero | 34559 | 864.72 / 1462.41 / 2451.76 | 1632.00 / 3428.00 / 4400.00 |
| feedback | 34209 | 856.75 / 1430.29 / 2423.24 | 1616.00 / 3320.00 / 4355.68 |

DOMINO 当前仍是 native 同步执行；4 条 qpos command 经 TOPP 后的仿真执行时长并不固定。feedback 的相邻查询 wall 中位数约 0.857 秒，sim 中位数约 1.616 秒。快路径 7–9 ms 证明的是服务时间，不能声明已经实现 100Hz 连续物理闭环或 RTC。

进一步将慢刷新后的决策间隔累计，恢复每个 age_actions 相对最近慢观测的真实年龄。单位秒，以下均为 feedback：

| age_actions | n | plan age sim p50/p95/p99 s | plan age wall p50/p95/p99 s |
| --- | --- | --- | --- |
| 0 | 8879 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| 4 | 8757 | 1.72 / 3.29 / 4.35 | 1.08 / 1.76 / 2.65 |
| 8 | 8631 | 3.29 / 7.24 / 7.71 | 1.86 / 3.01 / 4.70 |
| 12 | 8522 | 4.94 / 9.37 / 10.93 | 2.68 / 4.38 / 5.82 |

该表是训练数据匹配必须使用的量：同样的 age=4、8、12 不能当成固定的 0.6、1.2、1.8 秒。新训练/探针应同时报告 physical dt 分层与 action age 分层，并明确是否覆盖部署的长 dt；加入时间特征本身并不自动解决训练域外的长间隔。

反馈确实执行了非零修正：快路径 `residual_abs_max` 的 p50/p95/p99 为 0.00799/0.03291/0.06305，最大 0.08744（normalized qpos）；慢刷新 residual 为零。没有普遍撞到 0.1 上界的证据。这不能单独证明修正方向正确，也不能换算成物理关节误差。

## 7. 不完整任务与仅作敏感性分析的 partial outcomes

| arm | 未完整任务 | 已核验 raw outcome | 完整 timing 条数 | 停止原因 |
| --- | --- | --- | --- | --- |
| zero | handover_block | 0 / 8 | 7 | 100041 exhausted 10 attempts |
| zero | place_bread_basket | 11 / 12 | 11 | 100125 exhausted 10 attempts |
| zero | place_can_basket | 1 / 11 | 10 | 100020 exhausted 10 attempts |
| zero | rotate_qrcode | 1 / 16 | 15 | 100033 exhausted 10 attempts |
| feedback | handover_block | 1 / 8 | 7 | 100041 exhausted 10 attempts |
| feedback | place_bread_basket | 10 / 12 | 11 | 100125 exhausted 10 attempts |
| feedback | place_can_basket | 0 / 11 | 10 | 100020 exhausted 10 attempts |
| feedback | place_object_basket | 3 / 15 | 14 | 100059 exhausted 10 attempts |
| feedback | put_object_cabinet | 0 / 12 | 11 | 100295 exhausted 10 attempts |

这些失败均记录固定 seed 专家准备连续 10 次未接受，随后拒绝换 seed；它们不是自动填为策略失败的完整 episode。报告不建议为凑齐 700 静默换 seed，也没有启动重试。每个不完整 task 最后一个已报告 outcome 尚无对应 flush 后 timing，故 partial outcome 与 timing 数差 1；不能把它们直接混进主延迟表。

逐条核验 partial outcome 的 seed 是官方 manifest 的确切前缀、episode_index 连续、success 为 bool 后，三臂共同可核验 outcome 扩至 654 条：official 248、zero 247、feedback 257。feedback−zero 为 +10/654（+1.53 pp），gain/loss=34/24，McNemar p=0.237；这只是任务覆盖不均的描述性敏感性分析，不是完整 35×20 benchmark。

## 8. 对下一轮设计的约束

1. 保留三个对照：旧 eager full、同执行优化但零反馈、学习反馈。否则把 graph 的收益算到反馈学习上，会误判设计的贡献。
2. 当前快反馈成本可接受，但成功率提升证据偏弱；用相同数据与固定 teacher targets 做 temporal/static、fresh/shuffled-history 的廉价 heldout 探针，比立刻以低 loss 宣称闭环提升更合适。
3. 必须显式检查 physical age 的训练覆盖。部署 age12 中位数约 4.94 秒、p95 约 9.37 秒，长于“12 条动作”直觉对应的时间；短历史快头可能仍面对严重的过期计划。
4. 慢路径依然是约 235 ms query +22 ms 前处理的周期性阻塞；快头优化不能消除这个瓶颈。下一版速度目标要分别约束 fast p95、refresh p95 和实际动作加权成本。
5. 成功率不下降和控制平滑性尚不能由本次 mean success、query 延迟直接推出。完整相同 accepted-motion replay、接触阶段与失败视频分析，以及 joint/jerk 指标，属于后续机制验证。

## 9. 复现与证据文件

本地分析目录：`work/plan-feedback-v2-20261006/analysis/`。

- `read_core.sh`：只读提取各臂 aggregate、task episode outcomes，核验 exact seed。
- `read_timing.sh`：只读扫描 `PLAN-FEEDBACK-QUERY`、`PLAN-FEEDBACK-TIMING`、旧官方 `SYNC-QUERY-TIMING`；保存原日志路径/SHA256；恢复累计 physical plan age。
- `read_pairing_detail.sh`：核验两新臂逐 episode 指令，并读取 pipeline 状态。
- `core_raw.json`、`timing_raw.json`、`pairing_detail_raw.json`：提取原始证据，不依赖报告中的手写数字。
- `analyze.py`：stdlib 配对统计和 bootstrap；`paired_analysis.json`：完整逐任务、配对、时延和 provenance。
- `render_report.py`：从上述 JSON 生成此报告，并复制到待同步源码树。

本地复现：`python analysis/analyze.py`，随后 `python analysis/render_report.py`；统计不依赖 GPU。远程数据路径如下：

- zero: `/SSD_DISK_1/users/wuruihan/DynamicWAM/outputs/training/plan_feedback_v1_20261005/eval/zero_plan_control`
- feedback: `/SSD_DISK_1/users/wuruihan/DynamicWAM/outputs/training/plan_feedback_v1_20261005/continuation_20261006/eval/feedback`
- official: `/SSD_DISK_1/users/wuruihan/DynamicWAM/outputs/evaluation/native_graph_20261004/eval/official_sync_h16`

主 JSON 含提取输入 SHA256 与读取时间；`timing.arms.*.log_manifest` 记录每个原始 slot.log 的路径、SHA256、episode timing 数与 capture 数，可检查本报告是否对应同一份未修改结果。

