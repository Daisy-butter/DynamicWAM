# DynamicWAM 分频 R2：当前降分意味着什么

审计日期：2026-09-30。依据为 10.12.208.50 当前代码、实际 checkpoint、TensorBoard scalar、逐 episode 结果和逐 query 日志；仅读取服务器，没有改动服务器代码，也没有启动新的训练或评测。本文中的机制解释与待验证假设，区别于已经观测到的结果。

**判断：当前版本尚未证明能以可接受的任务性能代价实现快速控制，值得暂停扩大训练；但这批实验不能证明“分频必然损害 WAM”。真正需要检验的是，改变采样过程、时间对应关系和执行长度后，原 WAM 的策略行为是否仍被保留。已有 stage3 权重只提供初始化，不提供这个保证。**

## 1. 先确定测的是哪一版

实际配置为 `/home/wuruihan/DynamicWAM/configs/world_action_multirate_v1_joint_r2_7gpu.yaml`。

- checkpoint 的 `modules.architecture=reference`，`compact_fast=None`。
- 12 层原 action expert 仍被快慢两条训练路径共享；新加入当前 RGB、语言和 age 的 residual conditioner。
- 本次没有训练 compact action student、temporal world memory 或 native-teacher distillation。当前配置没有 `native_teacher_weight`，验证中的 native teacher loss 为零。
- 权重继承链为：本机 `absolute_motion_stage3_joint/exports/stage3_step_40000.pt` → `world_action_multirate_fast_warmup_r2/exports/multirate_step_1000.pt` → `world_action_multirate_joint_r2_7gpu/exports/multirate_step_1000.pt`。
- warmup 为单卡、batch 16、1,000 步，约 16,000 次 pair draw；joint 为 7 卡、单卡 batch 1、累积 2、1,000 步，约 14,000 次 pair draw。增加卡数并没有变成 7 倍训练样本。joint 只经过配额采样 epoch 的约 0.148；这不是“不够收敛”的证明，但也不能凭末步 loss 证明充分训练。
- 两阶段 sampler seed 已不同；不能继续沿用上一批“两个阶段重复同一采样顺序”的批评。

这些事实来自 checkpoint 内部的 `training_recipe`、`modules`，以及 `step_1000/multirate_progress.json`，不只是当前 YAML。

## 2. 现有结果的因果边界

| 比较 | 实际差异 | 目前可得结论 |
|---|---|---|
| 微调后的 backbone、原同步采样 H=4 vs 同 checkpoint 分频 H=4 | 推理路径/conditioner/bridge 调度改变 | 三任务各两次均合计 0/6；查询摊销下降约 25%，没有观察到这一匹配比较上的成功率差异 |
| 官方 sync H=16 vs 微调分频 H=4 | 权重来源、微调、采样路径、执行长度、调度均改变 | 3/6 vs 1/6 是整套系统的观察，不能单独归因于分频 |
| matched 与 paced 两次分频 rollout | GPU/调度时序等改变，权重不变 | 成功率与部分 manipulation score 存在波动，不能当作独立新增的场景样本合并统计 |

**官方 baseline 不是此次训练的原始父 checkpoint。** `--official-sync` 读取 `configs/absolute_motion_v2.yaml`，其权重为 `external/checkpoints/DynamicWAM_full.pt`。本机 stage3 与它的第 0 个 video attention Q 权重已不相同（抽查 RMS 差约 0.00611）。所以最关键的“本机父 checkpoint 的原生 H=16 成绩”尚缺。

逐 episode 的 raw seed 对应成立：shoe 为 100000/100006，fan 为 100002/100005，bottle 为 100000/100001。但每任务两次远不足以估计可靠成功率。匹配组双双失败也不证明两者等价，存在任务选择和指标下限问题。

## 3. 最明确的实现风险：动作计数没有对应上物理时间

我读取了三个任务 clean/randomized 共 900 条自采 episode 的真实仿真时钟，按训练使用的 `offset=age_actions×3` 映射统计可容纳完整后续 chunk 的窗口，并与 matched H=4 的 query 日志比较：

| 名义 bridge age | 数据窗口的时间中位数 | matched rollout 的时间中位数（三任务范围） |
|---|---:|---:|
| 4 actions | 0.66 秒左右 | 1.64–1.76 秒 |
| 8 actions | 1.29–1.31 秒 | 3.30–4.02 秒 |
| 12 actions | 1.90–1.91 秒 | 4.99–5.70 秒 |

统计的数据窗口中 age=4 最大约 0.72 秒、age=12 最大约 2.16 秒；matched rollout 的 age=12 最大达 10.60 秒。

这是仿真物理时间，不是 GPU 推理 wall time。`NativeStepper.execute()` 调用 DOMINO `take_action(qpos)`；DOMINO 对每个目标 qpos 进行 TOPP 轨迹规划，再推进若干物理步。一次调用不是固定 0.1 秒。专家数据中隔三个采样帧的一条 action label，和 rollout 中一次完整 qpos 执行，物理持续时间并不相同。

当前实现已经输入真实 `delta_sim_seconds`，因此不能说模型“完全不知道时间”。问题是：

1. 训练配对按动作/帧偏移构造，相关任务的 seconds 分布很窄；在线却明显超出这个分布。
2. 缓存有效性主要检查 `age_actions<=max_age_actions`，没有检查实际 seconds 支持范围和未来预测的剩余时间。
3. 输入一个超出训练分布的 age scalar，不会自动把历史预测变成当前预测，也不会自动补出已经过期的未来。

例如，训练映射中的 16 个 action slots 覆盖约 2.6 秒，而在线取到 age=12 的缓存时，可能已经过去 5.7 秒。即使动作计数还满足 `12+H4<=16`，也不能说下一段动作仍受“未来”表征覆盖。这里的 2.6 秒是数据映射的名义覆盖，不是把混合 K/V 精确解释成物理时间格点。

**这是已观测到的支持范围错位；它导致多少失败还需要干预实验。** 不能据此声称所有失败都来自过期缓存。sync H=4 也失败，说明还有原 backbone、执行长度或更一般闭环分布偏移的问题。

这也修正了之前 temporal-memory 方案的边界：只按 action age 对未来 token 分组和过期屏蔽，仍然没有解决这次发现的物理时间错位。必须同时核对 source 时间、预测目标时间和实际执行时间；不能直接把 nominal action bin 称为可靠的未来时间区间。

## 4. 为什么 stage3 初始化不保证原性能

完整策略可以写为：

\[
\pi=\operatorname{Execute}_{H}\circ\operatorname{Sampler}_{S}
       (f_\theta,\operatorname{Condition}(o,\text{history})).
\]

继承参数 \(\theta\) 并不保持采样过程 \(S\)、条件分布和执行算子 \(H\)。当前 V1 同时改变了这些部分。

原生采样中，一份动作噪声经历前两次 video/action 联合更新，得到视频缓存和与之配套的动作去噪中间态；后八次 action 更新继续同一条轨迹。V1 的 bridge builder 构造缓存后没有把这份动作中间态交给 fast sampler；fast sampler 从另一份高斯噪声重新走完整十步。

因此，即使 age=0、快路径 residual 初始化为零，也不保证输出等于原生 WAM。视频 K/V 是联合去噪中的内部状态，尤其较深层已经受 action tokens 影响；把它当作通用外部条件，再重启一个动作采样器，需要学习新的条件分布。这是可训练的设计选择，但不是等价加速。

缓存变老后，还多了一层差异：它来自历史观察及当时的潜在动作方案，现实已经由实际执行动作改变。`current RGB + state_delta + age` 可能学会纠正部分误差，但没有结构性保证它会维护预测与实际动作的一致性。

此外，warmup 仍更新共享 action expert 和 motion tokens；joint 又更新 video expert。aligned action/video loss 保留的是专家数据上的去噪任务，不是原策略完整采样后的行为。没有固定教师行为约束，就可能在学习新接口时改变原生策略。它是不是已经发生，必须用“新 backbone + 原生 sampler + H16”与本机父模型直接比较。

## 5. 前缀 loss 与完整动作去噪存在不一致

当前训练给全部 16 个动作加噪：

\[
x_\sigma=(1-\sigma)a^*+\sigma\epsilon,
\]

但快路径只监督前 4 个速度分量。与此同时，`ActionCoreOps` 仍让前 4 个动作读到全部动作 token，推理也仍对全部 16 个分量做 Euler 更新。

后 12 个分量没有快路径的直接速度监督，仍然会进入下一步采样、再影响前 4 个分量。训练中尾部是专家动作的正确加噪样本，推理中则是快路径自身积分得到的尾部；这一差异可能沿十次去噪放大。前缀损失会对部分共享参数产生间接约束，因此不能说尾部“完全没有任何训练”；问题是其完整采样向量场没有直接得到同条件约束。

joint 验证记录为：

- fast executed-prefix MSE：0.00508。
- fast full-chunk MSE：0.08296。
- null-bridge prefix MSE：0.04387。
- 验证样本数：224。

后两项说明：bridge 在专家分布上有用，但完整快路径动作场没有与很低的前缀误差同步变好。**“后面的动作不执行”不等于“后面的动作不会影响执行前缀”。** aligned 原生 action loss 在不同条件路径计算，不能直接当作快路径尾部已经受监督的证明。

应比较两种清楚的目标：保留 16 个耦合动作并给完整向量场监督（可提高前缀权重，或对尾部加入固定教师约束）；或者真正隔离/移除不受监督的尾部，使其不能反馈到执行前缀。不能只把 loss 切短而保持原来的完整耦合 sampler。

以上是代码揭示的风险和可检验机制；224 个单步 flow-matching 验证样本还不能证明其为在线失败的主因。还需测完整十步采样误差及 chunk 边界行为。

## 6. 对延迟和连续性的解释也需要收紧

`ms_per_executed_action` 是 query wall time 之和除以执行动作数，不包含全部渲染、光流和动作执行，也不是稳定控制周期。H4 下约 25% 的查询摊销收益是有价值的工程结果，但它不是控制频率提高到约 14 Hz 的证据。

同样，`reuse` 表示消费者这次直接使用可用缓存，不表示后台没有更新。实际日志中 `source_action_index` 多次推进；`actual_slow_builds` 才是构桥计数。独立 worker 的构桥主要被 simulator 执行时间掩盖，并不是被消除。启动耗时也包含在截图的 query mean 中，所以不能把该均值当成纯稳态 action DiT 用时。

目前每次 query 仍从新噪声采样完整 chunk，配置明确 `multirate_reuse_action_noise=false`。把 H16 改成 H4 会使重规划边界增多；并没有自动加入已承诺动作约束、轨迹衔接或跨 query 的行为一致性。新观察更频繁不必然带来更平滑的动作，可能更频繁地改变接近路径或夹爪时机。是否发生抖动需要边界 qpos/velocity/jerk 与接触事件记录，不能只凭成功率猜测。

原生 DOMINO 主 query 时不推进物理；因此降低这段 wall time 本身不会让物体在仿真里少移动一段距离。只有执行语义/观察频率改变后，闭环轨迹才变化。`pace` 的 sleep 也不把它变成严格实时评测。

## 7. 下一步优先做分解实验，而非继续扩大同一训练

固定同一组有效 episode/instruction、原生执行协议、动作归一化及同一 GPU 测时。先三任务各 20–50 次排查，作方法结论再扩展 35 任务。不同组还要固定模型采样种子；异步组记录实际 source 序列，并重复运行检查时序敏感性。

| 顺序 | 权重 | 采样与执行 | 要回答的问题 |
|---|---|---|---|
| A | 本机原 stage3 | 原生 WAM，H16 | 真正的训练父模型是否本来就低于官方版？ |
| B | 同 A | 原生 WAM，H4 | 单独缩短执行长度会损失多少？ |
| C | 当前 joint 的 backbone | 原生 WAM，H16 | 微调是否损伤原生策略？ |
| D | 同 C | 原生 WAM，H4 | H 与微调的交互；现有双 0/6 扩样 |
| E | 当前完整分频 checkpoint | 每 query 新建 bridge、H4 | 排除跨 query 陈旧性后，新接口/采样过程是否仍损失？ |
| F | 同 E | 现有 independent worker、H4 | E→F 的差距才更直接指向缓存复用与时序 |

E 必须显式配置当前 bridge，不能只减少 wait budget 冒充 age=0。E→F 仍包含 worker 调度差异，需要固定 source trace 的受控缓存回放进一步定位。

评测要同时保存成功率、manipulation score、首次接触/稳定抓持/掉落、每次 query 的 source 与 current 时间、实际 source future coverage、动作边界跳变、完整采样误差和端到端延迟。不要把单步 flow MSE 当成闭环成功率替代指标。

无需先重训就能做 A–F。若 A→B 已显著下降，先解决短 chunk 的行为稳定性；若 A→C 下降，先解决原能力保留；若 D→E 下降，先修复新条件/采样接口；若 E→F 下降，再集中处理未来过期。

## 8. 我建议保留的架构原则

第一，先要求一个可检验的原生恢复点：相同权重、观察、随机种子与执行长度，关闭新分支且 bridge age=0 时，能恢复原来“2 次 joint + 8 次 action”的动作轨迹。当前“2 次建桥 + 新噪声 10 次 action”没有这个性质。这个恢复点不能证明最终分频性能，但能避免把接口改变与参数学习混成一件事。

第二，冻结一份原 WAM 作为明确教师，使用当前因果观察产生完整动作轨迹；训练目标同时考虑专家动作、教师行为与跨 query 衔接。若进入联合微调，教师必须独立固定，不能把同步变化的 backbone 当作固定锚点。教师约束仍然只是约束，不能承诺继承教师成功率。

第三，把 future validity 定义在真实时间和预测覆盖上。短期可做 seconds TTL/当前观察重建的受控对照，但刷新增多可能抵消延迟收益；长期需让数据配对、表示中的预测时间与执行时间一致。若必须改执行器为固定周期，应作为新的评测协议单列，不与原生 DOMINO 分数混算。

第四，压缩模块大小、减少重启次数与保留联合预测，是不同设计轴。紧凑 student 需要训练证明；不是因为更小就天然保留 world-model 能力。应优先保留可修正的预测/动作状态，或验证蒸馏后的条件是否仍携带有用未来信息。

DuoCore 的慢路径主要提供任务语义和粗动作指导，其跨时间采样是为适应条件陈旧；不能据此假定 WAM 的动作相关未来 K/V 同样可以任意复用。RVQ-VAE 的 tokenizer 结果同时涉及输出长度、顺序预测错误和速度，也不是“token 越少成功率越高”的一般定理。[DuoCore 原文](https://arxiv.org/html/2512.20188)

一个相关的替代思路是 Rolling-WAM：跨重规划保留部分去噪的 video/action 未来，在新观察下继续更新，并在训练时匹配各 chunk 的噪声层级。这针对“每次从零重启”而非仅缓存旧条件。但它改变了时间调制和 attention mask，视频不读 action tokens，也没有提供 DynamicWAM/DOMINO 上的等价性结论，不能直接套用其加速倍数。[Rolling-WAM 原文](https://arxiv.org/html/2609.30247v1)

DynamicWAM 官方本身以 joint world–action attention 耦合未来和动作，并区分原生同步 DOMINO 与真机 RTC；分析时需要保持这个边界。[官方实现说明](https://github.com/Autumn1337/DynamicWAM)

## 证据索引

服务器根目录 `/home/wuruihan/DynamicWAM` 下：

- `configs/world_action_multirate_v1_warmup_r2.yaml`：本机父权重、warmup recipe。
- `configs/world_action_multirate_v1_joint_r2_7gpu.yaml`：实际七卡 joint recipe。
- `configs/absolute_motion_v2.yaml`：官方 sync 实际权重来源。
- `src/dynamicwam/models/causal_world_bridge.py`：两次 joint 构桥，返回 K/V，不返回动作中间态。
- `src/dynamicwam/models/multirate_wam.py`：独立噪声完整快采样；训练的 aligned regularizer。
- `src/dynamicwam/models/action_core_ops.py`：全部动作 token 参与注意力。
- `src/dynamicwam/training/train/multirate_objective.py`：executed-prefix 快损失。
- `src/dynamicwam/training/data/paired_index.py`：训练帧偏移到名义 action age 的映射。
- `src/dynamicwam/runtime/ciwam/wam/multirate_policy.py`：主要按动作数限制 bridge；秒数作为输入；独立噪声和 producer 更新。
- `src/dynamicwam/runtime/ciwam/execution.py` 与 `external/DOMINO/envs/_base_task.py:4119`：qpos/TOPP 变长执行。

结果根目录 `/SSD_DISK/users/wuruihan/DynamicWAM/outputs` 下：

- `evaluation/multirate_joint_r2_7gpu_pipeline/pipeline_summary.json`。
- 同目录 `probe_*/{sync,multirate}/<task>/config.yml`、`run/aggregate.json`、`run/logs/slot*.log`。
- `training/world_action_multirate_joint_r2_7gpu/step_1000/multirate_progress.json`、`logs/dynamicwam/events.*`。
- `training/world_action_multirate_fast_warmup_r2/step_1000/multirate_progress.json`、`logs/dynamicwam/events.*`。

本地审计复现材料保存于 `work/multirate-r2-analysis/`：`artifacts.json`、`cpu_audit.txt`、`audit_cpu.sh`、`summary.txt` 和当前代码只读副本。没有用单次成败判定具体接触失败类型，也没有运行新的因果消融。
