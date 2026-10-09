# DynamicWAM 分频实现、训练与评测审计

> 本文保留本轮第一次代码快照及已有实验的审计依据。用户随后明确“保留 WAM 原生预测能力”的目标后，已进一步实现冻结原生教师、时间分组 compact 学生和可选噪声复用，见 [最新实现说明](multirate_temporal_student.md)。工作期间也合并了新的预取、prefix loss 与无桥验证改动；本批历史评测没有使用这些新增路径。

日期：2026-09-29。对象：`/home/wuruihan/DynamicWAM`，`original-baseline`，基准提交 `70a0c89a82f562a6f9565f846c9737da58b326a7` 上的当前未提交实现。本文以代码、实际 checkpoint、训练状态和评测日志为依据。下文把已验证事实、代码推导、尚待实验的假设分开。

## 1. 判断：真正分频已经成立，但尚未实现目标中的整体提速

当前 V1 已经实现**跨控制查询复用慢分支表示**：慢分支用历史观测生成视频 K/V，后续快查询使用这份 K/V，同时读取最新三视角、运动和本体状态。它具有新的输入条件、新的训练分布和新增参数，不是只把视频去噪步数设成 2、动作设成 10。

但是，已有 episodes 用的是 `controlled_ratio`：刷新查询先建桥再采动作，同一查询内串行。它们没有验证 `independent_worker` 的并发收益。即使启用 worker，当前 DOMINO adapter 也仍然是 NativeStepper；不能据此称为执行器与动作推理并行的 RTC。

已测快查询约 231–255 ms，确实短于同步整次查询约 374–459 ms；但每执行 4 步就重新采样，而同步执行 16 步。因此单位已执行动作分摊的推理时间约 70–81 ms，对照约 23–30 ms，当前约慢三倍。**最迫切的结构问题是快核心仍然太重，而不是有没有线程。**

已训练的权重属于 `reference` V1；代码里的 `compact` V2 是另一条可选架构，当前结果不能用于证明 V2 的速度或成功率。本文没有把未经训练的 compact 模型切换为部署默认。

## 2. 代码实际做了什么

### 2.1 慢桥与快策略的调用关系

调用链：

```text
deploy_policy_multirate_flow.eval
  ├─ 记录当前 head motion / 三视角 / state / 精确 sim_time
  └─ MultiratePolicy.query（每 H=4 个已完成动作查询一次）
       ├─ 需要慢刷新：VAE -> CausalWorldBridge
       │    source 观测 + 随机未来 latent + 随机 source action
       │    joint step 0 -> Euler 更新 -> joint step 1 -> 各层视频 K/V
       └─ 每次快查询：最新 RGB/motion/state + bridge age + 历史 K/V
            FastConditioner.prepare_memory（一次）
            新动作噪声 -> 10 次 action flow 更新 -> 16 条动作
            controller 只执行前 4 条，然后重新观测/查询
```

桥保留的是第二次联合前向各层 attention **之前**的 K/V，不是解码后的干净未来视频。深层 K/V 可包含此前层的 source 动作信息，不能称为完全不考虑动作；但它没有根据当前 fast 候选动作重新滚动世界。

快分支从独立噪声开始完整 10 步。即使桥年龄为 0，也不等于原同步 sampler 的后 8 步。因此只加载原动作专家、不给新增视觉/age 路径训练，不能期待得到原策略的等价行为。

### 2.2 调度与模型分离

| 项目 | 当前行为 | 结论 |
|---|---|---|
| controlled_ratio，H=4、R=4 | 查询年龄正常为 0、4、8、12；每 4 次查询 inline 建桥 | 真跨查询分频，刷新时仍阻塞 |
| independent_worker | 独立线程/producer CUDA stream；到年龄边界预取，启动或过期时等待 | 可以重叠部分慢/快工作；不是无条件零等待 |
| isolated_fast_runtime | 在线动作前向用第二份独立 fast 模型，没有 WAN/VAE/T5 参数 | 不再只停留在“导出了 fast 权重”的层面 |
| 设备放置 | 当前在线 adapter 的 slow 与 fast 用同一设备 | 模块独立不等于 GPU 资源独立 |
| DOMINO 执行 | NativeStepper 调用 take_action；主线程 query 时不推进物理 | 当前成功率不能验证真实墙钟反应延迟收益 |

worker 的 producer 可以在主线程执行动作期间构建下一份桥；但主线程的 fast query 与物理执行依然同步。这两个边界必须分别报告。

### 2.3 V1 为什么快不起来

V1 每个 denoising step 仍执行 768 维、12 层动作专家，每层读取完整世界 K/V，再执行新视觉条件 residual。世界注意力使用 WAN 的 16×128 头空间，不能只按动作隐藏维 768 估算计算量。每次 query 的视觉编码/conditioner K/V 已经只算一次，这部分无需再次“缓存优化”。

本次进一步发现：原当前版本没有从 dense token producer 传递 `all_video_tokens_valid`，导致无 padding 数据仍重复执行 gather；世界 K/V 还会先转成 action Q 的 FP32、拼接/重排后再转回 attention 的 BF16。已修复这些可兼容权重的搬运冗余，见第 7 节。**这属于工程优化，不能事先承诺把 240 ms 降到 40 ms。**

更大的成本下降需要真正缩小快核心。现有 compact V2 的方向合理：当前观测与选定世界层先融合一次，压成 32 个 memory token；10 步动作采样只用 384 维、4 层学生网络。它减少了去噪循环反复读取大世界表示的成本，但能力是否够必须训练和评测。

## 3. 现有实验说明什么、不说明什么

证据目录：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/multirate_joint_probe/`，核对了 `probe_report.json`、两组 task config、aggregate 和 slot 原始日志。

| 任务 | 同步成功 | V1 分频成功 | 同步查询 ms | 分频复用查询 ms | 分频含慢桥查询 ms | 同步 ms/已执行动作 | 分频 ms/已执行动作 |
|---|---:|---:|---:|---:|---:|---:|---:|
| place_shoe | 2/2 | 1/2 | 401–459 | 252–255 | 474–535 | 27–30 | 79–81 |
| place_fan | 0/2 | 0/2 | 375–405 | 245–249 | 462–512 | 24–26 | 75–79 |
| adjust_bottle | 0/2 | 0/2 | 374–399 | 231–243 | 422–452 | 23–25 | 70–74 |

这里的区间是两个 episode 各自均值的范围，不是单次查询的 min/max 或置信区间。对应种子分别为 shoe 的 100000/100006、fan 的 100002/100005、bottle 的 100000/100001，两组匹配。

限制：

1. 同步 H=16，分频 H=4，改变了闭环频率、执行前缀、查询次数。当前结果不能单独归因于架构。
2. 两组分别在 GPU 1/2/3 与 4/5/6 并行测试，并非同卡、相同其他负载的配对测速。
3. `blocking_refreshes=0` 不表示没有慢刷新阻塞。旧字段只统计特殊启动/过期等待，controlled 的计划刷新没有加这个计数。含桥查询的 422–535 ms 已经体现 inline 成本。本次补上实际 query mode。
4. query timer 不包含 Farneback `flow.observation()`、相机/render、take_action 等；这不是完整 observation-to-actuation latency。
5. 仅 6 对 episodes，不能给出可靠成功率差异，更不能据此宣布某个架构提升/退化。当前配置没有记录评测视频，因此本批日志不能支持逐帧失败机制归因。
6. 历史 probe 的汇总是 episode 均值的宏平均，长短 episode 等权。本次新增明确的 macro 字段，并让主汇总按实际 query 数/执行动作数加权；历史文件没有改写。

### 3.1 与实验指标对应的成本推导

定义：B 为一次 VAE+慢桥成本，F 为一次当前观测条件下的完整快采样成本，S 为原同步一次完整查询成本。忽略起止边界，H=4、R=4 时：

\[
C_{\mathrm{multi/action}}=\frac{B/R+F}{H}=\frac{B+4F}{16},\qquad
C_{\mathrm{sync16/action}}=\frac{S}{16}.
\]

因此与原长 chunk 比总计算时间，需要 `B+4F<S`；与同样 H=4 的同步比，则比较 `B+4F` 与 `4S`。这是两个不同问题。由现有 episode 均值近似估计 `B ≈ (含桥查询均值 − 复用查询均值)`，要胜过 H=16 对照，F 需约小于 **35–48 ms**。这是基于均值差的预算估算，不是 B 的独立 profiling 测量。

即使理想地完全隐藏 B，也需 `4F<S`，即 F 约低于 94–115 ms；当前 F 约 231–255 ms，单靠异步化慢桥仍不满足条件。实际并发还会有资源竞争和等待，不能把理论上界当实测。

已有 `outputs/profile_pretrain_latency.json` 和 `outputs/profile_two_gpu_latency.json` 是合成帧、未训练 fast 分支的 profile，不能替代这些真实 episode。双卡文件记录约 44.24 MB bridge、3.83 ms 传输，但并发 pair 约 300.74 ms，并不等于单独测得 B/F 的 max（约 226 ms）。移动到第二张卡不是免费获得理想重叠。

## 4. 数据应该如何进入训练

### 4.1 原数据可以复用；改的是配对与目标，不需要为 2D 分频重采

实际 packed 路径：`/SSD_DISK/users/wuruihan/DynamicWAM/data/packed/domino_absolute_motion_v2`，source 路径通过 `data/converted/domino_exact_time` 读取视频和时间 sidecar。

已确认：1,420,472 条 packed 样本，10,500 个 episodes；downsample rate=3、action chunk=16、三视角 composite 384×320、num_video_frames=8。parent/warmup/joint 的 dataset identity 一致：

```text
ae752bcde6f5a25c84e44a96d0798322ae84d256cb2a4ced59c7206174171247
```

令 s 为建桥历史源帧，t 为当前快查询帧。一次 paired sample 必须来自同一个 episode、同一任务、同一数据 split，顺序满足 t≥s。

| 输入/监督 | 应取的时间 | 当前实现 |
|---|---|---|
| 慢分支 condition latent、state、motion、text | s | source sample；因果 |
| 慢分支未来 latent/action 起点 | 随机噪声 | 不混入真实未来或专家动作 |
| fast RGB、motion、state | t | 视频精确定位到 fast_frame，三视角分别预处理 |
| bridge age | t−s | 已完成动作数 + sidecar 精确模拟时间差 |
| state_delta | state(t)−state(s) | 同一归一化坐标系 |
| fast action target | t 之后的 16 条动作 | 取 fast_sample_id 的 action_sequence，不能取 s 的标签 |
| 对齐的 action/video 正则 | t | 当前帧和其后真实标签；这张图不发布 bridge |

下采样率 3 下，名义 age 0/4/8/12 对应帧偏移 0/12/24/36；动作标签是 t+3、t+6、…、t+48 的 14 维目标 qpos。不要把偏移 12 帧当成 12 个已执行动作。

对实际 `adjust_bottle/0` 做了 safetensors + MP4 + clock 的完整 `__getitem__` 抽查：

| s | t | age_actions | 精确 Δsim_seconds |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 12 | 4 | 0.664000 |
| 0 | 24 | 8 | 1.384000 |
| 0 | 36 | 12 | 2.092000 |
| 0 | 48 | 16，stress | 2.696000 |

这些秒数不等于帧号除以 30。`clock_mode=required` 读真实 sidecar；`action_only` 则显式将秒数置零。不能用固定 FPS 伪造模拟时间。

真实张量形状：source condition `[48,2,24,20]`、未来噪声 shape `[48,2,12,10]`、fast RGB `[3,3,160,160]`、state `[14]`、motion `[4,12]`、action `[16,14]`。这里只完成了抽样验证，未宣称所有 MP4/边界样本均已逐个审计。

### 4.2 配对分布与部署仍需对齐

episode 先划分，再做 pair，这是正确的，避免相邻帧跨 train/val。但是 parent stage3 已用过这些 demonstrations，所以这里是**分频适配阶段的 holdout**，不能称为基座从未见过的独立测试集。

`schedule_mixture` 将年龄权重设为 1:2:2:2，是组合 controlled/worker 年龄序列的启发式；它不是实际部署延迟测得的分布，也不是严格的 50/50 查询混合。controlled R=4 的正常查询分布为 1:1:1:1；若只研究该协议，可使用已有 `pairing.age_weighting: uniform`。worker 则应依据日志中实际 age_actions、age_sim_seconds、等待和状态变化另做配对分布匹配。

默认 stress offset 48 对应 age16，但 stress_probability=0；当前导出的稳态支持上限仍为 age12。不能因为 manifest 含有 age16 就允许推理无限复用。当前 worker 的过期等待是保护边界，不应为减少等待计数而删除。

真实在线执行的 4 条动作未必具有演示 12 帧的相同物理持续时间/轨迹。输入 Δsim 和 Δstate 是必要补偿，但不构成分布完全匹配的证明。还需比较这些条件的训练/部署分布，并做离轨、接触后和多模态未来的分析。

### 4.3 损失与梯度

\[
L=L_{\mathrm{fast}}(a_t\mid o_t,\mathcal B_s,\Delta t,\Delta a,\Delta q)
+\lambda_a L_{\mathrm{aligned\ action}}(a_t\mid o_t)
+\lambda_v L_{\mathrm{aligned\ video}}(v_{>t}\mid o_t).
\]

三个监督都以 fast 时间 t 为准；桥只来自 s 的因果输入。warmup 建桥 `no_grad`、冻结视频，训练动作/motion/新 fast 路径；joint 保留两步桥的梯度，视频也能收到 fast loss 的反馈，并加入视频正则。当前 joint λa=1、λv=0.001。

所以“桥和快策略完全没耦合”不准确：joint 确实能通过 K/V 反传。但这只是训练图上的耦合，**不保证模型在行为上利用桥**。需对 current RGB、历史桥、age 分别做控制消融。特别是随机 source 动作诱导的世界分支，与当前独立 fast 动作未必一致；这是待验证的表示充分性问题，不宜未经实验直接改成更复杂模块。

## 5. 已有训练到底训练了多少

检查了实际 checkpoint state、`multirate_progress.json`、scheduler state 和 TensorBoard events，不能只看配置文件名判断。

| 阶段 | 实际 world_size | batch×accum | optimizer steps | 配对抽样次数 | 观察 |
|---|---:|---:|---:|---:|---|
| fast_warmup | 1 | 16×1 | 1000 | 16,000 | 动作/motion 改变；抽查视频 q 权重不变 |
| cross_time_joint | 1 | 4×4 | 1000 | 16,000 | 视频 q、动作、motion、新视觉卷积均改变 |

两个运行实际是单进程训练，不能按曾经的“8 卡计划”计算曝光量。fast 第 0 层零初始化输出投影 RMS 从 warmup 的 0.000674 到 joint 的 0.000947，证明新路径不是始终保持零输出。state_dict 中新增 fast 分支约 32.54M 个元素，包含 buffers，不将其冒充精确 trainable parameter 数。

训练有真实更新，但存在两项设计问题：

1. 两阶段都从 data_epoch=0、相同 sampling seed 开始，且任务配对/权重一致。根据确定性采样代码，前 16,000 条 pair 索引序列重复；噪声和权重不同，所以不是重复同一优化。每个 quota epoch 为 9,450×10≈94,500 次抽样，两阶段各只完成约 16.9% 的一轮。5,275,417 条候选 pair 不等于已经训练过 527 万条，也不等于同样多独立样本。
2. 1000-step cosine 在终点将 action/fast LR 降到约 5e−10、video 到约 1e−10。直接沿用这个终点优化器“再跑一会儿”几乎没有学习率；改变 max_steps 也不是 exact resume。

已核对这些运行的 scheduler step_count=1000，因此不能把早先已修过的重复 scheduler stepping 或双重 sharding 问题归咎于这批权重。

旧 val/loss 为 warmup 0.028254、joint 0.022922，但这是不同阶段的加权 flow loss，并非抓取成功率；batch size 改变也改变了验证噪声分配。验证样本数量还受到第 7 节的尾部统计 bug 影响，不应把这两个数作为严格因果消融结论。

## 6. 是否需要重训，训练流程如何变化

| 改动/目标 | 现有权重能否用 | 需要的训练 |
|---|---|---|
| 本次缓存搬运优化、统计修复、同步对照入口 | 可以 | 不需要因此丢弃或从头重训；先重测 |
| 在相同 V1 架构上提高适配质量 | 可以作为初始化 | 建议新 run 的进一步适配，重设优化器/LR/采样 seed |
| 更改年龄分布、H 或超龄处理协议 | 需检查训练支持 | 不应把新协议当作原 checkpoint 已验证能力；必要时适配 |
| 换 compact V2 快策略 | 不能把 V1 当成训练过的 compact | 必须训练新学生/当前视觉路径，再决定是否 joint |
| 重做视频基座预训练/stage1 | 当前无必要 | 可复用已有 backbone；质量需后续验证 |

**建议先完成成本与利用率消融，再决定把训练预算投给哪个快核心。** 继续把 V1 训长也许能改善成功率，但不会改变它每查询 10×12 层的基本计算成本。若核心目标是相对 H16 同步也获得明显总时延下降，compact 快核心更符合成本预算，需要把它作为新训练实验。

V1 继续适配时，复制配置到新的文件和 run name，至少明确：

```yaml
# 以下是字段位置示例，不是可直接覆盖整个配置的片段。
paths:
  init_checkpoint: /SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/world_action_multirate_joint/exports/multirate_step_1000.pt
pairing:
  partition_seed: 20260928  # 固定原 episode holdout
  sampler_seed: 20260930    # 新增的独立采样随机种子
training:
  name: world_action_multirate_joint_continue
  resume_from: ""           # 权重初始化；不是终点 optimizer 的 exact resume
```

另外重新制定 max_steps、warmup、各组 LR、min_lr_ratio 和保存/验证频率，以有效样本曝光和 holdout/闭环验证决定停止，而非默认再固定 1000 步。这里没有未经比较替用户选定“最佳 LR”或自动启动训练。分频仍从 `scripts/train_multirate.py --config ...` 进入，不通过原 stage1–3 的 `scripts/train.py`。

compact V2 当前严格加载器可由 absolute-motion v2 parent 初始化 backbone；**不能直接把 V1 full multirate checkpoint 传给 compact model 的 strict loader**，两者 fast key 不同。应显式指定已验证的 absolute-motion stage3 parent，而不要假定旧模板中的 `external/checkpoints/DynamicWAM_full.pt` 别名就是相同格式。新增 temporal 配置已明确指定该 parent。若要沿用新 joint backbone，必须另外显式实现、验证 backbone-only 迁移，不能用 `strict=False` 静默吞掉错误。本文新增的 backbone 提取函数用于同步对照，未偷偷改变训练加载合同。

原 V2 共享 teacher 的 aligned regularizer 只监督 teacher，不自动构成 student distillation。后续新增的 temporal 配置把原生策略蒸馏作为单独、需与 expert-only 学生对照的设计，并冻结完整 teacher，详见最新说明；不将这项未训练的新设计计入上面的 V1 结果。

## 7. 本次直接修复的代码

### 7.1 不改变权重形状的缓存优化

- `training/models/wan_model.py` 与 `inference/models/wan_model.py`：dense token 构造器在创建等长 seq_lens 时显式记录无 padding。依据是张量形状和构造路径，不读取 CUDA tensor 的布尔结果。
- `models/small_wam.py`、`models/causal_world_bridge.py`：把该 host 元数据传递至缓存与桥。未知/有 padding 输入仍默认走安全打包路径；混合 null bridge 的逐行有效长度仍保留。
- `models/action_core_ops.py`：拼接和 gather 前直接使用 attention dtype，避免 FP32 世界缓存中间副本。没有减少 denoising 步数、丢弃动作尾 token 或更改 checkpoint 参数。

这对数学上送入 attention 的 K/V 保持等价；具体 GPU kernel、数值微差和实测加速仍需要后续真实 GPU 回归。

### 7.2 验证样本被错误截尾

现有 `ProcessGroupAlignedSampler` 已主动截去不完整 process round，且 Accelerate 的 `even_batches=False` 没有补齐重复样本。但 `gather_for_metrics` 在最后一批依照 **dataset 长度**的 remainder 再截一次；dataset 是百万级候选 pair，而 sampler 一轮仅按 episode quota 取样，两者长度不一致。

修复：验证改用 `accelerator.gather`，保留全部真实行。在 CPU 上用真实 Accelerate loader 复现了 dataset=13、sampler=8、batch=4 时原方法误丢行的问题。新增 `val/fast`、`val/action`、`val/video`、`val/fast_executed_prefix` 和各 age 的样本数/fast/prefix 指标。原训练 checkpoint 的权重不受这个验证统计 bug 影响。

### 7.3 训练采样与恢复合同

- `config/multirate.py`：新增可选 `pairing.sampler_seed`；合并后的默认行为是 warmup 使用 partition_seed、joint 增加固定盐值；显式 sampler_seed 优先。恢复旧运行应明确写原采样种子，且训练目标必须一致。
- `training/train/multirate_trainer.py`：训练 sampler 独立取种子，validation/episode holdout 仍固定；resume 合同保存该值，老合同按旧语义推断，改变采样序列不能伪称 exact resume。
- 后续 full checkpoint 保存 `training_recipe`，便于追溯曝光量、优化器时钟和采样设置。

### 7.4 查询统计与可比较对照

- `MultiratePolicy.last_query_trace` 记录实际模式：`inline_startup / inline_scheduled / inline_overage / wait_startup / wait_overage / reuse`，以及 bridge source、action age、模拟时间 age、worker wait、H/R。
- DOMINO adapter 按实际执行路径统计刷新/等待，不再靠 controlled R 时钟给 worker 查询贴标签。保留旧计数便于历史日志兼容。
- `execution.query_timing_summary` 增加 executed_steps 和 p95。
- 同步 adapter 新增可选 `execute_horizon`，默认仍 16；新增 `multirate_backbone_checkpoint`，严格加载同一份分频 checkpoint 的 backbone，以分离“微调过权重”与“使用分频 sampler”的效果。
- probe 脚本新增 `--sync-horizon`、`--scheduling`、`--sync-backbone-checkpoint`、`--serial` 与 GPU lists；要求新输出目录，不覆盖历史结果，检查配对 seeds。默认 probe H=4，明确区别于原 H16 协议。主汇总改为按查询/执行数加权，保留明确标注的 episode macro 指标。

代码位于 `/home/wuruihan/DynamicWAM/src/dynamicwam/` 与 `/home/wuruihan/DynamicWAM/scripts/`。同步前核对了目标文件 hash，保存被改文件到 `.codex_backups/`；没有重置用户已有改动或改写 checkpoint、历史实验文件。

## 8. 仍未被解决、应优先验证的问题

| 优先级 | 问题 | 当前证据强度 | 应如何判定 |
|---|---|---|---|
| P1 | V1 快核心成本超预算 | 真实 episode + 成本不等式 | 修复后再 profile；达不到预算转 compact 训练 |
| P1 | H16/H4 与不同权重混在同一对照 | config 已确认 | 相同 H、相同 backbone 的控制组 |
| P1 | 慢桥是否提供有效增益 | 未测；有梯度不能证明行为利用 | 正常/无桥/错配桥，按 age 和接触阶段分析 |
| P1 | 新视觉与 age 路径适配不足 | 短训练/重复序列已确认；不足程度未知 | 多 seed、较长曝光、分年龄验证和 closed-loop |
| P2 | 世界未来与实际执行动作不一致 | 结构性风险，尚非失败归因 | off-policy/接触前后误差、桥新鲜度与成功关系 |
| P2 | train/runtime 图像与时间条件分布 | 一部分抽查通过 | MP4 编解码、wrist resize、Δsim/Δstate 的一致性检查 |
| P2 | 独立 worker 的资源竞争及超龄率 | 本批 episodes 未覆盖 | actual source age、wait p95、sameGPU 并发 profile |
| P2 | V1 两层 scratch 视觉和 BN 的泛化 | 代码可见，因果效果未测 | 与 compact/预训练视觉作参数预算匹配实验 |

这里没有发现足以判定现有全部训练权重必须报废的标签泄漏或监督时间错位。最致命的是**成本目标与对照指标错配**：把“更频繁决策且单查询更快”写成“总推理时间降低”，会得到错误结论。

## 9. 最小有判别力的后续实验

先固定一批 episode manifests，并在同一张空闲 GPU 上顺序评测，记录 checkpoint、H/R、调度、actual bridge age、query p50/p95、等待与总执行数。协议不同的结果分别命名，不混进原 DOMINO 表格。

| 对照 | 权重 | H | 桥/调度 | 解决的问题 |
|---|---|---:|---|---|
| A | parent | 16 | 原同步 | 原协议锚点 |
| B | parent | 4 | 原同步 | 只增加闭环频率的效果 |
| C | joint backbone | 4 | 原同步 | 适配训练改变 backbone 的效果 |
| D | joint V1 | 4 | controlled R=1 | 新 fast sampler、当前视觉/age 路径的效果 |
| E | joint V1 | 4 | controlled R=4 | 相对 D 的跨查询复用代价/收益 |
| F | 同一 joint V1 | 4 | independent_worker | 相对 E 的调度收益，报告实际 age 分布 |
| G | 经训练 compact | 4 | 对齐 E/F 的协议 | 快核心压缩是否真正改善成本—成功率关系 |

D 的每次新桥 age0 处于训练支持内，可通过 adapter 的 bridge_refresh_ratio=1 配置；probe 默认 R4，不能把默认脚本误当作完成了整张表。

本次提供的可执行 matched-H probe 示例（**未运行**；需自行选择空闲 GPU 与新目录）。合并后的脚本默认使用相同 joint backbone；示例加 `--official-sync` 保留上表 B 对 parent 的比较：

```bash
cd /home/wuruihan/DynamicWAM
.venv/bin/python scripts/eval_multirate_probe.py \
  --run-root /SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/multirate_matched_h4_new \
  --episodes 20 --sync-horizon 4 --scheduling controlled_ratio --official-sync \
  --sync-gpus 0,0,0 --multirate-gpus 0,0,0 --serial
```

要让同步组用相同 joint backbone，去掉 `--official-sync` 即可；或显式指定：

```text
--sync-backbone-checkpoint /SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/world_action_multirate_joint/exports/multirate_step_1000.pt
```

这个示例只覆盖三个 smoke tasks，不是全 benchmark。正常/无桥/打乱桥、当前视觉遮挡、按接触阶段统计、完整 observation-to-actuation 计时尚需专门实验入口；本次没有声称这些消融已经完成。无桥/错配桥实验也必须保留 age/state 条件一致，避免同时改变多个变量。

## 10. 验证与交付边界

本次仅运行 CPU 检查和数据/checkpoint 只读抽查，没有启动训练、GPU profile 或新的 DOMINO episodes，没有停止或修改现有作业。

CPU 验证包括：bf16 十步动作采样、padding/null bridge 正确性与梯度、无 CUDA-to-host bool 的关键路径、worker 过期等待、跨 episode 噪声、真实 MP4 解码、Accelerate 分片/验证截尾、采样种子、H4/H16 执行前缀、checkpoint strict mapping、训练步/优化器与调度恢复、compact 模块合同。最终 8 项新 temporal 测试、33 项修复回归、14 项 compact 回归，以及 training/checkpoint/runtime/schedule 四个专项脚本通过；probe CLI 与 `git diff --check` 通过。日志：`/tmp/dynamicwam-multirate-cpu-wqCCV0/`。

这些检查证明所覆盖的执行合同和局部数值行为，不能替代 GPU 真实权重 rollout。**当前可交付的是经修复的实现、可信的实验解释和可比较的下一轮入口；新的提速幅度与成功率仍未测。**
