# 跨控制周期分频：当前实现说明

> 最新架构扩展：[冻结原生 WAM 教师与时间分组快学生](multirate_temporal_student.md)。独立配置 `configs/world_action_multirate_temporal.yaml` 已实现，需训练新学生；已有 V1 评测不代表该新结构的性能。

> 2026-09-29 后续审计：实际训练权重、6 对 episodes、成本推导、数据对齐、重训判断及本次代码修复见 [multirate_audit_20260929.md](multirate_audit_20260929.md)。本文主要保留实现脉络；涉及已实测结果与新统计入口时，以该审计记录为准。同步 adapter 现在可显式选择 H=4 或加载 joint backbone 做对照，默认仍为原 H=16 同步策略。

这份说明按仓库里现在的代码写，日期是 2026-09-29。它不是 `video_action_frequency_split_90828c09.plan.md` 或 `implementation_sections.md` 里的完整合同。那些文件仍然是设计目标。这里只记录代码实际在做什么、改了哪些地方，以及还不能当成已经完成的部分。

同步基线默认执行含义不变。`configs/absolute_motion_v2.yaml`、`scripts/train.py` 的 stage1–stage3，以及 `HeadFlowRunner.sample_chunk` / `sample_chunk_slow` / `sample_chunk_fast` 保持原有接口。同步 adapter 新增可选 H 与 backbone 对照参数。分频使用单独的配置、训练脚本和 DOMINO adapter。`inference/model_loader.py` 对旧 v2 checkpoint 的严格检查没有放宽。

## 1. 改动要解决什么

现有 DynamicWAM 的 Video DiT 和 Action DiT 在每一层做双向联合注意力。推理时的 video K/V cache 只在同一次查询内部复用：10 步去噪里，视频在 step 0 和 step 1 刷新，step 2–9 接着用这次前向写下的 cache。下一次控制周期重新规划时，原来的路径会把慢路径再跑一遍。

这次要的是跨控制周期复用。慢分支继续做视频—动作联合预测，把 step 1 联合前向、进入该层联合注意力之前的视频 K/V 发布成桥。快分支不再重跑 Video DiT，读这张桥，同时读当前三视角、当前绝对运动和当前本体状态。

联合注意力保留在慢路径上。速度来自后续控制周期不再重跑 Video DiT。

桥必须是因果的。训练器原来的流匹配混合不能拿来建桥：未来 latent 是 `(1-σ)·真实未来 + σ·噪声`，动作是 `(1-τ)·专家动作 + τ·噪声`。`detach` 只切断梯度，去不掉已经混进去的未来信息。所以建桥和当前时刻的视频/动作正则是两张图：

- 建桥只用 source 时刻已经看得到的条件 latent、状态、运动和文本。未来 latent 和动作都是随机噪声，噪声水平是推理日程的 step 0 和 step 1。
- 正则在 fast 时刻用真实未来视频和专家动作做原来的流匹配，并且 `build_video_cache=False`。这张图的 cache 丢掉，不发布成桥。

## 2. 现在实际成立的行为

三件事里，前两件在当前代码里成立。

1. 快查询读的是当前观测。`MultiratePolicy._sample_fast` 每次用当前三视角、当前 motion、当前状态，年龄是已执行动作差 `q_t - q_s`。十步去噪复用同一次视觉编码和同一份桥。这是一条新的动作轨迹，不是把慢 sampler 从第 2 步继续积完。`MultirateWAM.sample_fast_actions` 的文档写明了这一点，年龄为 0 时也一样。
2. 桥不用未来标签。`CausalWorldBridge` 先用随机未来 latent 和随机动作做 step 0，在设备上做 Euler 更新，再做 step 1，发布 step 1 里 joint attention 之前的 K/V。后面八步动作去噪不进入建桥。`reject_leaky_source_fields` 拒绝源对象带未来视频或专家动作。warmup 整段 `no_grad`；joint 两步都保留图，包括 step 0 的 Euler。
3. 慢刷新不挡住快查询，只在独立 worker、并且桥还在寿命内时成立。部署默认仍是 `controlled_ratio`，刷新点会阻塞。同频模式是另开的：`bridge_lead_actions > 0` 时决策不再当场建桥，慢桥在决策前若干步交给同一个 worker。见第 4 节。还没有新的计时，不能写成已经比同步查询更快。

训练把历史桥和当前标签拆开。桥来自源时刻。fast loss 和 regularizer 用快时刻的专家动作。默认稳态年龄是 0、4、8、12 个动作，对应帧偏移 0、12、24、36。`stress_offsets: [48]` 是压力偏移，`stress_probability: 0.0`，默认训练不抽它。

共享 Action Expert 和 motion token 是合同第 11 节的 V1 决定。快损失看历史桥，正则看当前帧，两边都更新同一套动作专家。这不是把专家意外注册了两次。

## 3. 具体改了什么代码

### 3.1 旧模型上的开关

`src/dynamicwam/models/small_wam.py` 是主要改过的旧模型文件。默认参数保持原来的 `detach_cache=True`、无 residual、`null_bridge=False`，所以旧的同步调用不改变含义。

- `_video_attention_kv_for_cache(..., detach_cache=True)`。联合训练传 `False` 时，K/V 留在图里。
- `forward_action_with_video_cache(..., fast_residual=None, null_bridge=False)`。世界注意力之后、FFN 之前可以插入 fast residual。
- cache 仍写在每一层联合注意力之前。第 0 层的 K/V 还没看见这次前向的动作 token，更深的层已经看见了上面几层的动作信息。

世界注意力的可复用部分抽到 `src/dynamicwam/models/action_core_ops.py`。`ActionAttentionSpec` 记下精度和每层 window，快路径不再为了这些元数据去碰 WAN block 对象。`StandaloneActionCore` 在 `src/dynamicwam/inference/fast_runtime.py` 里把动作专家、运动 token 和这份 spec 装成一个不含 WAN/VAE/T5 的模块。

`null_bridge` 现在可以是整批布尔，也可以是形状 `[B]` 的掩码：

- 全无桥：只做动作自注意力。
- 全有桥：视频 K/V 在前，和原来一样。固定长度、没有 padding 时直接拼接，不再 gather。
- 混合且视频行本身没有 padding：同样直接拼接，无桥行改走动作自注意力，再按行混合。有 padding 且 window 是 `(-1, -1)` 时，仍把动作放到有效视频前缀之后。
- 混合且 window 不是全局：分别做一次有桥和一次无桥，再用 `torch.where` 按行混合。有 padding 时这一路仍 gather。
- 一次快轨迹在进 10 步循环前把视频 K/V 转到注意力精度。这一步去掉的是重复搬运，10 步 × 12 层的整份视频注意力还在。

### 3.2 日程、桥和快条件

`src/dynamicwam/models/flow_schedule.py` 从现有 flow-match 配置生成设备端 sigma。Euler 是 `sample + velocity * (sigma_next - sigma)`，留在样本所在设备上，不走 `FlowMatchScheduler.step` 里的 `.cpu()` 和 `argmin`。视频日程只有两个 sigma，step 1 不能调用 `video_pair(1)`。step 1 只取 `video_timesteps[1]`，不再更新视频 latent。动作最后一步可以积到 0。数值对照只保证测试容差，不声称 bf16 逐位相同。

`src/dynamicwam/models/causal_world_bridge.py` 编排上面的两步。`detach_cache` 同时传给 step 0 和 step 1。step 0 不建 cache，所以这个开关本身切不断 Euler 图；joint 能回传到 step 0，是因为整段桥不在 `no_grad` 里。`source_id` 和 `sim_time` 如果已经是张量就原样保留，不再压成一个 Python 标量。

`src/dynamicwam/models/fast_observation_encoder.py` 是 V1 视觉。结构是从头训练的两层卷积，3 个视角各 4×4 token，加上 view embedding 和空间坐标。`forward(rgb, mask)` 返回 `(tokens, token_valid)`。mask 先乘到输入上，再插值到特征图，按有效质量做 4×4 池化。BatchNorm 跟着 `train()` / `eval()` 走：训练时更新 running stats，部署在 `finalize_inference` 之后处于 eval，不再更新。这和合同第 12.4 节写的“冻结预训练 ResNet BN”不是同一件事。

`src/dynamicwam/models/fast_conditioner.py` 是 768 维、12 层、8×96 的独立残差，不进入 WAN 的 16×128 注意力，也不改动作 token 的位置编码。残差插在世界注意力和 FFN 之间。`W_O` 零初始化，没有第二道零标量门。输入会先转到权重 dtype，加回残差前再转回 token dtype。key mask 保留在设备上并传给 SDPA，避免在每层去噪中将 Tensor 转成 Python bool。世界注意力仍走 flash attention，CPU 测试里把那个函数换成 SDPA。

年龄特征是归一化的 Δ 秒、Δ 动作数、桥是否有效，再拼上归一化状态差 `q_t - q_s`。状态差用已经归一化的状态相减，部署时也只归一化一次。

`src/dynamicwam/models/compact_fast_policy.py` 是可选的 V2 学生：宽 384、6 头、4 层，世界 K/V 是输入，自己不持有 WAN/VAE/T5。无桥行用逐行系数把世界修正乘成 0，图保持连通。它的 `train()` 仍把内部 BN 固定成 eval。默认 yaml 没有启用它。`MultirateWAM` 要求 `FastConditioner` 和 `CompactFastPolicy` 恰好有一个。

`src/dynamicwam/models/multirate_wam.py` 是训练唯一入口：`forward(batch, mode="multirate_train", phase=...)`。`fast_warmup` / `warmup` 是同一阶段，`cross_time_joint` / `joint` 是同一阶段。Accelerate 看到的是这一个模块。损失用 `accelerator.backward`。没有打开 `find_unused_parameters`。慢路径和正则不调用 fast residual。

### 3.3 数据

`src/dynamicwam/training/data/paired_index.py` 检查动作窗口是否放得下，并给出稳态帧偏移。

`src/dynamicwam/training/data/paired_world_action_dataset.py` 读 packed v2 shard，配出 `(s, t)`。首版 `M=1`。语言钉在该语言组的第 0 条。`clock_mode: required` 时，缺时钟文件直接失败，不用 `seconds_per_frame` 去编模拟时间。`action_only` 才把模拟时间写成 0。样本带上 `source_id` 和 `source_sim_time`。RGB 在 VAE letterbox 之前从原始 composite 裁切：头 `[0:240, 0:320]`，左腕 `[240:360, 0:160]`，右腕 `[240:360, 160:320]`，再 letterbox 到 160×160 并做 ImageNet 归一化。视频句柄是按 episode 路径的 LRU，上限是 `max_open_clocks`，读失败会重开一次。

`src/dynamicwam/training/data/multirate_sampling.py` 的 `BalancedPairSampler` 每个 episode 正好抽 `samples_per_episode` 条，再打乱一次，按 `rank::world_size` 切。它不是 `DistributedSampler`。`stress_probability: 0` 时不抽 stress。单个 episode 的分层全部留在训练集。验证集按 `subset="validation"` 另建，允许为空。

`src/dynamicwam/observations/multiview.py` 负责裁切、letterbox 和有效 mask。

### 3.4 训练和权重

`configs/world_action_multirate_v1.yaml` 的 schema 是 3，baseline 是 `world_action_multirate_v1`。`scripts/train.py` 不读它。默认阶段是 `fast_warmup`，`video_weight` 必须为 0。`H_exec=4`，`R=4`，`null_bridge_probability=0.05`。DeepSpeed 是 ZeRO stage 0、bf16。

启动：

```bash
.venv/bin/python scripts/train_multirate.py --config configs/world_action_multirate_v1.yaml
```

联合阶段改三处：`phase: cross_time_joint`，`loss.video_weight` 设成正数，`paths.init_checkpoint` 指向 warmup 导出的 multirate checkpoint。联合阶段是新进程、新优化器，不要在同一次运行里解冻视频专家后继续用原来的优化器。不写 `sampler_seed` 时，warmup 用 `partition_seed`，joint 再加上 `JOINT_SAMPLER_SEED_SALT`，两条阶段不再从同一条配对 shuffle 开头。显式写下的 `sampler_seed` 覆盖这个规则。episode 划分仍只看 `partition_seed`。

快损失只对 `actions[:execute_horizon]` 反传。chunk 其余位置仍算出 `loss/fast_chunk`，不进入总损失。同一时刻的动作正则仍监督完整 chunk。验证在同一批目标上再跑一次 `null_bridge=True`，记下 `val/fast_null` 和 `val/bridge_prefix_gain`（无桥前缀损失减有桥前缀损失）。

`src/dynamicwam/training/train/multirate_trainer.py` 里，`prepare_multirate_batch` 按样本抽无桥掩码，桥噪声和 fast / regularizer 的流匹配噪声分开。`source_id`、`source_sim_time` 按样本堆叠。优化器分组是 `action_core`、`fast_conditioner`（或 `fast_policy`），joint 再加 `compact_wan.video_model.wan_model`。存 checkpoint 前跑验证 loader；`validation_batches` 为非负整数时才截断。验证统计用 `accelerator.gather`，不用 `gather_for_metrics`。

权重格式在 `src/dynamicwam/inference/multirate_model_loader.py`：

| 格式 | 用途 |
|---|---|
| `dynamicwam_absolute_motion_checkpoint` v2 | 只作初始化。旧键加 `backbone.` 前缀。多出来的键只能是 `fast_conditioner.*` 或 `fast_policy.*`，然后 `strict=True` |
| `dynamicwam_multirate_checkpoint` version 2 | 训练导出和断点续训。键集合必须和模型一致。部署只接受这种格式，并核对 `runtime_contract`、日程和 action normalizer |
| `dynamicwam_multirate_fast_only` version 2 | 参考架构抽出动作专家、运动 token 和 fast conditioner。若状态里有 `fast_policy.`，则只导出学生，不带动作专家 |

fast-only 文件不允许出现 `compact_wan`、VAE、T5。动作专家里的 `wan_action_qkv` / `wan_action_o` 是投到 WAN 注意力空间的动作侧投影，不是 Video DiT 权重。在线部署用 `isolated_fast_runtime` 从完整 checkpoint 抽出这份权重，单独做成快采样器。慢桥仍走完整 `HeadFlowRunner`。

### 3.5 部署

新 adapter 是 `src/dynamicwam/runtime/ciwam/adapters/domino/deploy_policy_multirate_flow.py`。它不改同步 adapter，也不改 RTC sidecar。需要 `usr_args["multirate_checkpoint"]` 和 `TASK_ENV._scene_step_clock`。

`ControlledMultirateController` 在队列空了、或自上次查询以来已执行动作达到 `H_exec` 时重规划。队列只保留 `actions[:H_exec]`。`take_action` 不增加年龄，环境步完成之后才 `mark_executed`。`bridge_lead_actions == 0` 时，`query_id % R == 0` 是控制器自己的建桥时钟，只被受控模式使用。`bridge_lead_actions > 0` 时决策点的 `rebuild_bridge()` 为假，`needs_bridge_snapshot()` 在队列里还剩这么多条动作时为真，每个 chunk 只记一次。查询用 `flow.observation()` 的当前运动，不用冻结运动。快复合图的手腕用 `INTER_AREA`；慢路径 VAE 复合图仍走原来的 `composite_robotwin_frame`。

同频部署的入口参数是 `execute_horizon: 16` 和 `bridge_lead_actions: 4`。`multirate_scheduling` 仍写 `controlled_ratio` 时会改走 `independent_worker`，因为中途快照必须在后台建桥。`bridge_wait_budget_ms` 默认 150，只花在「当前桥还在训练年龄里、另有一份更新的预取」或者「当前桥已经超龄、先等这 150ms」；超龄并且 150ms 内没好，就继续等到一份年龄不超过 `max_age_actions` 的桥，不拿超龄桥去采样。启动时还没有桥，这次等待单独记成 `wait_startup`。`pace_action_interval: true` 时，没有快采样的环境步会补足到 `action_interval`；含快采样的那一步不补，避免把决策等待再垫长。不打开这个开关的 DOMINO 只适合看成功率。官方同步 10 步采样没有这个开关。

`MultiratePolicy.setup` 自己建完整 `HeadFlowRunner`，不调用同步策略的 `setup()`，所以启动时不会先跑一遍同步 `sample_chunk`。然后 `validate_runtime_contract`，装上 `MultirateWAM`，`finalize_inference` 只作用在快分支上，整个 wrapper 再 `eval()`。

`src/dynamicwam/runtime/ciwam/wam/multirate_control.py` 的 `forward_budget()` 记下默认工作量。每 16 个已执行动作：分频是 2 次联合前向加 40 次快动作前向；原来一次 `H=16` 长 chunk 是 2 次联合加 8 次动作前向；原来也每 4 步重规划是 8 次联合加 32 次动作前向。这是次数，不是毫秒。

## 4. 独立 worker 现在怎么调度

`scheduling` 默认是 `controlled_ratio`。设成 `independent_worker` 时，`LatestWorldWorker` 用一个后台线程建桥。待处理请求只有一个，后来的覆盖先前的。完成屏障在生产者线程上：side stream 等待默认流，event 在生产者上 `synchronize`。快线程看不到写了一半的 K/V。交给后台线程的观测包会复制 ndarray、tensor，以及 tensor 列表（文本嵌入）。

`plan_worker_bridge` 和控制器的 `query_id % R` 不是同一套时钟。默认 `H_exec=4`、`max_age_actions=12` 时：

- 没有桥时同步建一次。这是唯一允许等待的启动。
- 年龄 0、4、8 复用已有桥。
- 下一次查询会让年龄超过 12 时，当前这次查询先提交建桥，和随后 4 步执行重叠。这次查询自己仍用旧桥。
- 下一次查询如果新桥已好，而且新桥年龄不超过 12，就装上。装上后的年龄是 4，之后是 8、12，再提前建。成功时年龄停在 4、8、12，不再回到 0。
- 新桥没好、旧桥年龄已经超过 12：这次查询会阻塞等待一次可用的刷新。如果已经提交的源本身也超过训练年龄，就改成用当前观测重新提交，再 `wait_for`。装上之后的年龄落在训练支持的范围内。`over_age_queries` 和 `blocking_refreshes` 都会加一。慢刷新因此仍可能挡住控制回路。

受控模式没有独立 prefetch 路径。它在 `rebuild_bridge()` 或年龄超过上限时，当场等 VAE 和两步联合前向做完。

同频模式不走这套 `query_id % R` 时钟，也不在决策时调用 `plan_worker_bridge`。`eval()` 每步都有当前帧。队列还剩 `bridge_lead_actions` 条时，`submit_bridge_snapshot()` 把这份观测交给 worker，不等待，也不 `commit` 动作。下一次 `query()` 只跑快采样。已经装上的桥若来自提前 4 步的快照，年龄约是 4，落在训练过的 0、4、8、12 里。checkpoint 里的 `execute_horizon` 仍是训练时的 4；部署写成 16 只在 `bridge_lead_actions` 处于 `1..max_age_actions` 时放行，因为当时的快损失监督的是整段 chunk，而桥龄仍被上限挡住。没有提前量、或提前量超过训练年龄时，部署 horizon 必须和契约一致。

H=4、R=4 仍是每 4 步重规划的闭环实验。它每 16 个已执行动作有 4 次快采样。同频模式每 16 步只有 1 次快采样和 1 次中途慢桥快照，用来和官方每 16 步一次的同步查询比延迟。控制器上的 `slow_builds` 计的是这种快照，启动那次同步建桥算在 `actual_slow_builds` 和 `blocking_refreshes` 里。

回合摘要在 `policy.reset()` 之前打印，包含 `actual_slow_builds`、`blocking_refreshes`、`over_age_queries`。过期查询计入 `over_age_queries`。

`src/dynamicwam/runtime/ciwam/wam/bridge_store.py` 有槽状态机。`publish` 会先清掉 `ready_id` 再回收，过期 generation 会被拒绝。DOMINO adapter 没有使用这个 store，在线桥走的是 `LatestWorldWorker`。

## 5. 可能存在的问题

### 5.1 还不能说控制延迟已经下降

默认部署是串行的 `controlled_ratio`，`H_exec=4`、`R=4`。这条路径每 16 个已执行动作做 4 次 10 步快采样，其中刷新点还把建桥算进同一次等待。它的工作量高于官方一次 H=16 同步查询。独立模式里，慢计算只和随后几步执行重叠；一旦年龄越过训练支持，快查询会停下来等新桥。无桥策略仍只在训练里按概率采样，线上过期路径不再走无桥。

同频模式把两条通路的次数都放回每 16 步一次，决策等待里不再包含一次完整建桥，前提是快照到决策之间的执行墙钟盖得住建桥。DOMINO 一步往往快于 100ms，不打开 `pace_action_interval` 时这 4 步盖不住，决策会把等待预算吃掉，墙钟不能当成加速。调速之后也要重新计时才能比较决策等待和同权重的同步查询。任务成功率要单独看：桥比同步采样更旧，快通路也不是同一次查询里的联合去噪。掉了应记成桥龄和快函数的差距。

默认配置下，动作前向比原来的长 chunk 更多。世界前向只比“每 4 步都重跑完整模型”更少。同卡耗时和 DOMINO 成功率都还没有测到。工作量变化不能写成时延已经下降。

### 5.2 慢、快仍是同一个进程里的同一套完整模型

`MultiratePolicy.setup` 仍构建完整 `HeadFlowRunner`，因为建桥要 VAE 和 Video DiT。快动作采样走 `isolated_fast_runtime`：从同一份 checkpoint 抽出动作专家、运动 token 和 fast conditioner（或紧凑学生），单独 `eval`，参数里没有 WAN/VAE/T5。两套权重是只读副本，快查询不再和建桥共用视频模块的缓冲。文本仍由慢侧 runner 编码后交给快副本。`BridgeStore` 没有接到 DOMINO adapter。

### 5.3 训练图和验收还缺合同里的闭环

- 两次联合前向加 regularizer 没有按第 14.3 节做无副作用的激活重算。checkpoint 函数必须自己返回张量，不能在重算块里 append cache。这件事还没做。
- 没有和真实 `HeadFlowRunner` sampler 的 CUDA 逐层 cache 对照。因果桥的 CPU 测试用的是假联合模型。
- 训练建桥读 packed 好的 `condition_latent`。部署建桥在 `inference_mode` 里重新 VAE 编码。两条路径的条件 latent 来源不同，数值还没对齐。
- 没有 DeepSpeed 多卡 save/resume 脚本，也没有 DOMINO 上固定 `H_exec` / `R` 的成功率。
- 验证默认跑完整验证 loader；配置里的 `validation_batches` 设成非负整数时才会截断。
- 每个 rank 仍会各自构建完整配对索引。样本不会重复训练，但时钟和 pairing 扫描会在每张卡上各做一遍。
- V1 视觉是从头训练的卷积。合同第 12.4 节写的是冻结预训练 ResNet，并核对权重哈希。默认 yaml 没有启用紧凑学生。
- 零初始化的 `W_O` 使 fast 编码器在第一步没有有效梯度。早期 fast 损失不动时，先看 `out_proj` 是否还接近 0。

### 5.4 数据、时钟和采样还要单独看

帧偏移、在线动作数、模拟秒和墙上时间不是同一个量。`global_downsample_rate=3` 只表示演示里动作标签的采样间隔。`seconds_per_frame` 留在 yaml 里，不再用来编模拟时间。

视频用 OpenCV 随机 seek，读出来按 BGR 转 RGB。如果某条 mp4 不是这个封装，快视觉会偏色。手腕在快复合图里用 `INTER_AREA`，同步复合图仍是原来的默认插值。慢路径 VAE 和快路径 RGB 不是同一张下采样结果。

多卡训练现在由全局采样器加 `ProcessGroupAlignedSampler` 丢掉不完整的 process round，再交给 Accelerate 切一次。`even_batches` 对 multirate 入口关闭。默认 `age_weighting: schedule_mixture` 把年龄 0 抽一次、被复用的 4/8/12 各抽两次，对应受控刷新和独立 worker 成功循环合在一起的读取次数。压力偏移仍由 `stress_probability` 控制，默认是 0，线上过期路径会等待，不会把超过 12 的年龄送进快策略。验证集和训练集共用同一份 packed reader 和 episode 划分，各自只扫自己子集的时钟。两个阶段的采样种子按上面的盐值分开，划分种子不分开。

### 5.5 这些缺口不影响什么

同步基线的配置和采样路径没有改。现在这套代码可以做受控比例下的表示复用训练，也可以在独立 worker 里让未过期的快查询走不含世界权重的只读副本。过期时会等待一次可用刷新。快副本和慢桥仍在同一进程里，慢桥仍要跑 VAE 和 Video DiT。还没有测到控制延迟下降。

### 5.6 官方锚定的紧凑 temporal 路径（2026-09-30）

诊断见 `multirate_r2_diagnosis_20260930.md`。新的 `configs/official_temporal_stage{2,3}_7gpu.yaml` 从 `DynamicWAM_full.pt` 初始化紧凑学生：动作归一化与自采数据一致，**运动特征统计保留官方原值**，并写入 `initialization_provenance`；后续 stage 延续该记录，避免换权重时悄悄改输入尺度。训练监督完整 16 步快动作，并加固定原生教师；部署合同含 `max_age_seconds=0.72`，超时强制刷新；评测可开 `multirate_reuse_action_noise`。秒数 TTL 会增加刷新，延迟要以新测量为准。自动化入口：`scripts/run_official_temporal_7gpu_pipeline.sh`。
