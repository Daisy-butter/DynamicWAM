# Route1 信息保留修复与 GPU 验证（2026-10-16）

本轮基于异步 joint v3 继续修改，保留新框架；旧 41.72% 版本用于定位丢失的信息。真实 8 卡 warmup/joint 和实际 CUDA Graph＋后台慢路并发均已通过。未启动正式训练。

## 旧评测与历史对照

旧 step12000 的 0.25 倍物理钟连续评测已经完整结束：35×20，182/700 成功，SR **26.00%**，MS **41.8594**；700 报告和 700 原生目录 MP4 均完成并核验。所有 worker/guard 已退出，驱动证据保持 307。它是单独的缩放连续协议，`realtime_valid=false`，不是 1× 实时成绩，也不是新 v3 的成绩。

同旧版 29 任务比较：旧 official full 231/580=39.83%，完整 WAM＋有界反馈 242/580=41.72%，当前旧 checkpoint 162/580=27.93%。仅 517/580 seed 重合，旧版 native 同步/TOPP，新版连续 RTC/固定 0.16 秒动作网格；因此掉分不只来自多评六个任务，但现有分数不能把掉分全部归因于架构。旧有界反馈相对官方仅增加 11 次成功，配对区间也不能证明确定收益。

完整原始审计见 `route1_scaled_eval_audit_20261016.md` 和 `route1_bounded_vs_motionkv_20261016.md`。后者的 warmup 段记录修复前状态；本文件及 v3.2 设计记录其后的 source Action FM 修复。

## 确认的信息变化与本轮修复

1. **完整动作轨迹被丢弃。** 旧版沿同一噪声轨迹执行 2 次 Video/Action 联合去噪＋8 次 Action 去噪，得到完整 16 步计划；旧 MotionKV 只留第二次 joint 的 KV，快路另起噪声。现在补回同噪声的最终 `source_completed_plan`，同时保留真正生成 KV 时的早期 clean estimate；两者语义不混淆。增加 8 次 Action forward，零额外 Video forward。
2. **源计划、旧队列与新计划需要直接对齐。** 新 proposal feedback 在每次 K10 去噪中使用当前 noisy proposal、源计划/完整计划/旧队列差异、已承诺前缀及精确物理目标时间。未知 suffix 采用 `1-sigma` 置信权重，训练不输入 clean GT suffix。小模块修正快路读取的有效 KV，不改写异步共享的源 KV。
3. **抓手阶段与计划模态保护变弱。** 旧反馈头不改抓手，只在原计划上作小残差。现在完整计划是软先验，H4 只硬保护已承诺段，后缀仍能重规划；因此补回信息不等于恢复原抓手事件稳定性。需要开发集记录抓手翻转、计划切换、接触后预算耗尽，不能未经训练直接冻结抓手或平滑输出。
4. **原动力学通路仍在，源 actor 训练约束曾不足。** native state/action/motion/register tokens 一直保留。新增 warmup source Action FM 0.1，使用已有 detached source KV、原生无快路 adapter 的 Action 路径，补充共享 actor 的源规划约束；joint 保留 Video/Action FM 与运动预测损失。Video 在 warmup 冻结，在 joint 真正更新。

完整计划是当前模型的 stop-gradient 自条件预测，不是独立 teacher 或执行标签；source KV/原 source FM 保持联合训练梯度。源计划覆盖 2.56 秒，缓存最大年龄 4.96 秒；过期目标正确 mask，但高龄缓存的完整计划先验会耗尽。这仍需 age 分桶和刷新延迟验证。

## 当前 recipe 与初始化

第一阶段从 `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/motionkv_rtc_adapt_8gpu_20261014/exports/multirate_step_12000.pt` 初始化；旧 backbone/interface 保留，只迁移新增接口参数。不是从零重新训练原 Stage1。

| 阶段 | 步数 | 更新模块 / LR | 主要损失 |
|---|---:|---|---|
| Warmup | 2,000 | Action/native motion 1e-5；接口 1e-4；Video 冻结 | fast suffix FM 1；source Action FM 0.1；motion forecast 0.1 |
| Joint | 18,000 | Video 2e-6；Action 5e-6；接口 3e-5 | fast FM 1；source Video/Action FM 各 0.1；forecast 0.1 |
| 真实 replay refinement | 2,000 | Video 1e-6；Action 2e-6；接口 1e-5 | 保留上述损失，使用同轨迹真实策略前缀与合法监督 |

正式 profile 有效 batch=1/GPU×8×accumulation2=16；K10/H4/R4、0.16 秒物理网格、异步年龄 0..31；VAE/T5 冻结。后阶段加载前阶段完整 export。这是新的 warmup→joint→replay recipe，不能简单称原 Stage2＋Stage3；没有开启自动正式训练链。

初始化仍有实质性待验证项：旧 step12000 的慢路来自 official，而新共享慢路采用 adapted step12000 actor。恢复同一个 2＋8 solver 不会自动恢复 official 先验。正式长训前应做 official-full 与 step12000 初始化的短开发对照，结合源 FM、完整计划/抓手事件、闭环开发场景选型，不能只看总 loss。

## 已完成验证与边界

- CPU：22 个原集成程序通过；新增 7 个 native source completion 检查、5 个合同检查及真实 step12000 schema 迁移通过。包含同噪声原生求解器一致性、前缀钳制、源 KV 不变、proposal/完整计划有效性、梯度和新旧 checkpoint 严格加载。
- 数据：10,500 集形成 1,416,299 个合法物理网格样本。实际 8 GPU 仅完成独立标记的 16 集/256 样本 VAE smoke 子集，原时间/动作/flow/训练验证 split 均保留。完整新 VAE cache、成功 development replay 尚未生成。
- 真实训练：8 GPU warmup 4 步＋joint 4 步均通过；完整原生 Video/Action，loss/grad 有限，full/fast checkpoint 严格重载通过。warmup Video 339 个 tensor 全部不变；joint Video 279/339、Action 106/131、native motion 8/10 及新增接口权重发生更新。权重变化证明 optimizer/export 路径，不代替逐参数梯度或任务收益证明。
- 训练源码：`8599d5d8acf2de5236d30a7f15af4d7a7e5adaf9`，分支 `feat/route1-action-feedback-v31-20261016`。Graph 检查的标量 BF16 指纹修复：`b0570698ac5d9610bc119dab0631c1a217f0a9dc`。
- 首次 Graph 检查在 capture 前因检查脚本不能 hash 标量 BF16 失败；原失败保存。修复 flatten 后 CPU 回归通过，随后显式启动第二次检查。两次都使用真实训练 checkpoint 与真实数据 packet，不涉及驱动恢复。

SSD 证据根：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/joint_motionkv_v31_20261016`。所有 GPU 阶段均使用临时 all-adapter FD guard，worker 结束后关闭 guard；不使用永久 keepalive，不创建巡检。

### 部署 NaN 修复与真实 Graph 验收

Graph 第二次检查发现全空注意力导致 NaN。真实部署 Torch 2.4.1/cu121 BF16 SDPA 最小复现产生 1,536 个非有限值；训练 Torch 2.7.1 与 CPU 检查不能覆盖这个后端差别。修复公共 attention：仅全空 batch 行在 SDPA 前置零 Q/K/V，开放一个零 sentinel，返回显式零；有有效输入行保持原值、mask、长度与梯度。四项 CPU 空行/梯度回归、七项 proposal 和七项 source-completion 回归通过；历史失败完整保留。修复提交 `1dc8cf01201dfa3818c4d9d74b6c67f3b30c66b6`。

`graph_03/graph_parity.json`：实际训练后 checkpoint、真实记录 observation、原生完整模型；20 项值更新/前缀/过期 source plan 对照全部通过，最大绝对误差 0，只有一次 capture，输入与权重未改写。采样器 eager p50 223.77 ms，Graph p50 54.75 ms（各五次，含公共 sampler 校验与输入 copy，不含相机/物理执行）。完整后台 slow 刷新约 390.74 ms；五次 fast 均在 slow 发布前开始，耗时 136.93/73.01/66.17/63.06/60.84 ms，旧 cache 未变，新完整 cache 随后可消费且不触发 recapture。不同 CUDA stream 与软件时间区间证明消费路径未等待发布，不单独证明硬件 kernel 同时执行，也不代表闭环 p95。

Graph worker、后台 worker 和所有 guard FD 均已关闭，exit0、driver clean 307。

### 1× 连续闭环开发 smoke

独立 `adjust_bottle` seed1610016，已经与正式 700 场景的 seed/manifest/motion 排除清单核验。完整原生 episode、未截短 action budget、未放宽 H4 或 backlog 门限，`simulation_rate=1`，实际协议 `continuous_world_actual_latency_joint_motionkv_async_v3`，`realtime_valid=true`。

正常落盘并 exit0：原生指标 63 步（62 个已完成 target）、17 次 inference，physics10.5200秒/wall10.5221秒，比例0.99980，最大 backlog0.1164秒，最大 inference clock error0.00228秒。执行与推理实际重叠；尾段 source_action_index48、当前60，age12，证明后台刷新后的 cache 实际进入后续控制。首尾均为实际物理时间，未排除在线启动延迟。该集任务失败，原因 out_of_bounds；4步 smoke 权重不用于性能结论。单场景10.5秒不能证明全任务/长程实时稳定性。

原生 MP4 已完整 ffmpeg 解码：
`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/joint_motionkv_v31_20261016/development_1x/episode/eval_result/adjust_bottle/motionkv_rtc/demo_clean_dynamic/multirate_step_4/development-1x-functional-20261016/episode0.mp4`

`development_1x/acceptance.json` 与 guard 最终收据确认全部 owned worker 和 FD 退出、driver clean307。没有采集/伪造真实训练 replay，没有开启后续任务或巡检。完整新 VAE cache、有效 development replay、初始化短对照和正式长训仍待完成；完整异步训练分布覆盖、性能提升、SOTA 仍未证明。smoke 子集 checkpoint 禁止作为正式初始化或续训起点。主仓库仍冻结在 `448782a`，Route2 未改动。
