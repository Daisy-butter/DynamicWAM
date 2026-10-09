---
name: DynamicWAM Cross-Cycle Fast-Slow Policy
overview: 跨控制周期分频。慢分支只跑现有联合采样的前两步并写出 K/V，未来 latent 和源动作都从随机噪声开始；快分支复用这份 bridge，同时读当前三视角、运动和本体。先用固定 H_exec 和 R 验证复用，再做独立 worker。单次 query 的 2/10 去噪步和 RTC 不是本方案的分频。
todos:
  - id: causal-bridge-contract
    content: 明确推理一致的 bridge 来源、时间戳、噪声阶段和只读生命周期，禁止从含真实未来的训练前向发布 bridge
    status: pending
  - id: paired-observation-data
    content: 新增同 episode 的早慢晚快配对数据，补齐当前三视角 RGB、真实时间和动作索引，保持已有数据只读
    status: pending
  - id: fast-current-observation
    content: 新增轻量当前视觉通道和 age conditioning，快路径使用当前运动与状态，不再冻结到旧 world snapshot
    status: pending
  - id: cross-timescale-training
    content: 实现因果慢采样、跨时间动作损失和独立原始 WAM 正则，先适配快分支再联合优化
    status: pending
  - id: controlled-multirate-eval
    content: 固定执行长度和评测 episode，对比同频、裸缓存、当前观测快分支及跨时间训练，验证准确率—计算量权衡
    status: pending
  - id: independent-workers
    content: 实现独立慢 producer 与快 consumer、不可变 bridge 发布和并发资源隔离，验证慢更新不阻塞快推理
    status: pending
  - id: causal-and-runtime-tests
    content: 检查未来信息泄漏、跨分支梯度、bridge 实际贡献、缓存年龄、并发正确性及端到端尾延迟
    status: pending
isProject: false
---

# DynamicWAM 跨控制周期分频方案

本方案基于实际代码审阅，代码锚点为 DynamicWAM `70a0c89` 及 2026-09-28 的分频工作区改动。下文路径相对于 `/home/wuruihan/DynamicWAM`。§1–17 是完整实现合同，不表示全部落地；§18 和 [当前实现审查](multirate_review_20260928.md) 明确实际完成程度、确定错误和重构建议。本次仅更新文档及运行小型 CPU 诊断，没有修改模型代码、启动训练或 GPU 评测。

**最新判断：当前已实现串行的跨控制周期分频，未实现独立并行。默认 H=4/R=4 相比原版 H=16 并未减少每16个执行动作的世界刷新次数；要实现明显总提速，必须按 §7 的预算约束压缩快路径。§11–17 的逐层 residual 是 V1 参照，不应被当成必须保留的最终快网络；V2 重构决策见 §18。**

## 1. 核心判断：把世界预测的刷新与当前观测下的动作决策分开

建议实现的结构是：**低频运行完整世界—动作联合预测，生成可跨时间使用的表示；高频读取最新真实观测，在该表示的引导下重新生成动作。**慢分支还没完成下一次更新时，快分支应能继续工作。其关键不是少算某一步 Video DiT，而是允许多个具有不同真实观测的动作 query 共用一个慢分支结果。

这项改造需要同时成立三个条件：

1. **信息条件**：快分支直接读取当前视觉和运动，能够纠正旧预测；仅更新机械臂关节角无法观察外部物体的新位置。
2. **学习条件**：训练就让快分支在旧 bridge 与新观测的组合下学习，且旧 bridge 必须以部署时可获得的信息生成。
3. **运行条件**：慢分支刷新与快分支推理没有逐轮等待依赖。固定每 R 次 query 刷新一次可用来验证模型，但还不能冒充最终的独立并行运行。

DynamicWAM 的 Video DiT 预测的是会快速变化的场景，而 DuoCore 的慢表示更偏任务语义。因此，**不能预设世界表示可以像任务语义一样长时间保持有效**。本研究应验证：在保留当前视觉闭环的前提下，未来预测的昂贵计算是否存在可利用的时间冗余，以及这个冗余的有效范围。

### 1.1 四类容易混淆的时间尺度

| 量 | 当前含义 | 本方案是否修改 |
|---|---|---|
| Video 2 次 / Action 10 次去噪 | 一次 query 内的求解器调用次数 | 首轮固定，避免与新机制混淆 |
| `method.video.video_action_frequency_ratio: 2` | 数据中未来视频帧与动作索引的采样对应关系 | 保持；不是运行时慢快更新比 |
| 生成长度 `K=16`、执行长度 `H_exec` | 一次生成多少动作、执行多少后再观察和重规划 | K 保持；H_exec 是独立实验控制量 |
| 慢刷新周期 / 快 query 周期 | 多次新观测之间，世界表示与动作分别多久更新 | 本方案的真正研究变量 |

RTC 处理动作执行与推理的重叠及动作衔接；本方案处理模型内部两条计算通路的更新频率。即使禁用 RTC，也应能实现、训练和测量这里的结构。

## 2. 从论文迁移什么，以及哪些内容不能直接照搬

[DuoCore，§3.1、§3.3、§3.4，Fig. 1–2](https://arxiv.org/html/2512.20188v1) 的可迁移原则是：慢表示经 buffer 被多轮复用，快网络仍读取当前视觉和本体状态；联合训练时，慢输入取较早时刻、快输入取较晚时刻。论文快分支不是只靠旧缓存反复输出动作。其 VLM、fusion queries 和全身动作 tokenizer 属于具体实现；我们不需要为此替换 WAM 或引入离散动作 tokenizer。论文也不能直接证明这种设计在 DOMINO 动态抓取上有效。

[DynamicWAM，§3.2、附录 B.6](https://arxiv.org/html/2608.00793v1) 已有运动条件、联合 world–action attention 和 query 内 K/V 复用。它们提供初始化与计算基础，但不自动提供跨真实时间的有效缓存。旧计划只训练另一个噪声水平读取同一份 K/V，主要解决的是同一 query 内的近似；将跨时间训练写成“可选”，不足以支撑本次目标。

以下提出的是基于实际代码的迁移设计，不是 DuoCore 原文已有的 WAM 方法，也不沿用论文的速度数字作为本项目性能承诺。

## 3. 原有代码基础及当前扩展边界

| 原有实现及代码入口 | 核实结果 | 对设计的影响 |
|---|---|---|
| `models/small_wam.py::_joint_attention` | Video 与 Action 双向联合 attention | 保留慢刷新内部的耦合，不先重写成 video-only 模型 |
| `runtime/ciwam/wam/flow_runner.py::sample_chunk_slow` | 第 0、1 个动作去噪步运行完整 video，之后只更新 action，返回最后一次 video cache | 可作为因果 bridge 构造的部署参照 |
| `small_wam.py::_append_video_cache_layer` | 每层在本层 joint attention **之前**记录 video K/V | 第一层 cache 尚未读本次 action；后续层已受前层联合交互影响。不能把所有层统一说成“已融合最终动作” |
| `small_wam.py::_video_attention_kv_for_cache` | 旧版无条件 detach；当前新增 detach_cache 开关 | 方向正确，真实 joint 梯度尚需验收 |
| `flow_runner.py::sample_chunk_fast`、`wam/policy.py::sample_fast` | 快路径没有当前 RGB 输入 | 必须增加当前视觉入口，不能仅扩展调度器 |
| `adapters/domino/deploy_policy_async_flow.py::eval` | 快路径以 `packet.update(frozen)` 用旧运动覆盖当前运动 | 新模型训练后应使用当前运动；旧模型上直接替换仍存在分布偏移 |
| `wam/world_buffer.py::WorldBridge` | 记录 cache、旧 motion、模拟时间及动作步龄；年龄从发布重置 | 真并行后必须按**源观测时刻**计算年龄，包含慢计算期间的过时量 |
| `wam/async_control.py` 和 `configs/absolute_motion_v2_async.yml` | `fast_horizon=4` 且 `max_world_age_steps=4`；每执行 4 步重新规划时 cache 已 stale | 默认设置实际上每次重规划都走慢路径；不能仅看分支名称就认定已有分频 |
| 同一 `eval` 函数 | 同步调用 slow 或 fast，再执行动作 | 当前是串行路径选择，不是慢快 worker 并行 |
| `training/train/video_action_trainer.py::_prepare_batch` | future latent 和 action 都由真实目标加噪构造 | 不能把这一训练前向的 cache 当作部署可用的历史 bridge |
| `training/data/packed_dataset.py` 与 packer | v2 存单时刻条件、未来 latent、动作和索引；运行时不返回当前 RGB 和配对时间 | 需要明确的成对数据契约，不是简单把 batch 的 state 换成下一行 |

另外，`adapters/domino/composite.py` 明确显示：**WAM 当前外观输入是 head + 两个 wrist 的合成图；只有光流取 head camera。**新增快视觉不应因误读而悄悄退化为只看 head。

## 4. 建议架构：因果 world bridge + 当前观测快策略

### 4.1 信息流与条件分解

用 s 表示慢分支的源观测时刻，t 表示快分支的当前观测时刻，s ≤ t：

\[
B_s=G_\phi(O_{\le s},q_s,\ell;\xi_s),
\qquad
A_t=\pi_\theta(I_t,m_t,q_t,\ell,B_s,\delta_{s,t};\epsilon_t).
\]

- `O_{≤s}`：源时刻三视角外观、只来自过去的 head-flow 历史、运动统计。
- `G`：保留联合 attention 的 WAM 慢采样过程；未来视频和源动作从随机噪声开始。
- `B_s`：该过程导出的逐层 K/V 和来源元数据。
- `I_t, m_t, q_t`：当前三视角 RGB、截至 t 的数值运动历史、本体状态。
- `δ`：当前观测与源观测的模拟时间差及动作索引差；不把网络去噪时间当成场景时间。
- `ξ_s, ε_t`：相互独立、可复现实验的慢/快采样噪声。

结构如下。慢分支生成的是候选未来相关的表示，不是已被证明充分的物理信念状态。

```text
源观测 s：三视角 + head-flow 历史 + 源状态/运动 + 指令
                         │
               慢 WAM 联合采样（低频）
               Video DiT ↔ Action DiT
                         │
                完整、只读的 bridge B_s
                         │  多次读取
                         ▼
当前观测 t ── 轻量视觉编码 ──► 快 Action DiT ──► 16 条候选动作
当前运动 m_t / 状态 q_t ─────►       ▲             执行前 H_exec 条
指令 / cache age ──────────────────┘

慢分支独立计算 B_s' 时，快分支继续用 B_s；完成后原子发布 B_s'。
```

### 4.2 慢分支：先保留 joint attention，明确 cache 的含义

慢分支保留现有联合 sampler 的前缀，而不是把 10 步 action 去噪都算进 bridge。`video_refresh_steps: [0, 1]` 时，部署写出的 cache 是第 2 次联合前向里、每层 joint attention 之前的 K/V；第 2–9 步只读这份 cache，不再改它。因此 bridge 构造在第 2 次联合前向结束时停止：包含 step 0 的联合前向、video/action scheduler 更新，以及 step 1 的联合前向。源动作只是这两步里从随机噪声走出的局部 proposal，不写入执行队列。稳态动作只由快分支提交。

这样可以继承原 checkpoint，也避免一开始将问题扩大为重建视频模型。但需要明确：

1. 最后一次 cache 来自某个中间噪声阶段，并不是解码后的干净未来，也不是在第二次 video 更新完成后再做一次前向提取的表示。
2. 深层 K/V 与源时刻的动作 proposal 有关。快策略修正动作、抓取接触发生或机械臂改变视角后，旧预测可能失配。
3. 保留刷新内部的联合 attention，只保留了原来的一个计算子图；整个跨时间条件策略已经改变，不能声称“不改变数学模型所以必然不掉点”。

第一版直接复用原始 K/V，使效果归因集中在跨时间训练与当前观测。只有 profile 显示其注意力带宽或缓存量成为主要瓶颈时，再研究 learned-query 压缩 bridge。压缩应作为独立架构消融，不能在首轮把分频、压缩和改 attention 同时混入。

主实现只运行两步联合前缀；完整十步慢 sampler 仅作为 cache 数值对照，不进入新桥生产路径。这是依赖图裁剪，不能代替跨控制周期分频贡献。当前代码已实现该前缀，尚未完成真实 CUDA cache 对照。

### 4.3 快分支：每次 query 都有当前外部观测

首版建议使用一个共享权重的轻量 2D RGB encoder，处理当前 head、left wrist、right wrist；保留每个视角的空间网格 token 和 view embedding。可从预训练 ResNet-18 级别骨干开始，建议输入约 160 像素尺度，每视角池化到 4×4，共 48 个视觉 token。具体缩放保持宽高比；小目标可见性和编码耗时决定是否提高分辨率。不能只全局池化成一个向量后假定定位信息充分。

实现上采用独立的 **current-observation cross-attention residual**，插入现有 action block，在原来的 cached-world attention 外补充当前视觉。视觉编码及其逐层 K/V 投影每次 query 只计算一次，在十步动作去噪中复用；不要把整个视觉 encoder 重算十次。输出投影 W_O 零初始化，不再叠加零标量门。首次反传允许 encoder 梯度为零，更新 W_O 后再检查视觉梯度。

快分支的其它输入：

- 当前 `absolute_motion_features` 和有效性 mask，保留原来的四个运动 token；不再用 bridge 中旧 motion 覆盖它们。
- 当前 `q_t`、源 `q_s` 的归一化差值，以及 bridge age，经小投影加入条件。旧 `q_s` 是已观测量，不是未来标签。
- 指令使用原有 T5 表示，经独立小投影/池化形成静态条件，按指令缓存，避免每个快 query 重跑 T5。
- 当前 token 与历史 bridge 在类型和时间上区分。新视觉 residual 使用自己的空间/view position，不把新图像硬拼进旧 WAN 时空 RoPE 网格。

这是一个实现起点，不是关于 ResNet 或 48 token 最优性的结论。当前数值运动仍是全局摘要，未解决多物体运动对应关系。首轮不同时引入检测、分割和跟踪；若在多物体条件下失败，再以同预算的当前 head-flow 空间编码作针对性消融。

### 4.4 Bridge 数据契约

建议定义不可变 `WorldBridgeSnapshot`，至少包含：

```text
episode_id, instruction_id, model_version, bridge_id
source_observation_id, source_action_index, source_sim_time
source_capture_monotonic, ready_monotonic
source_state, source_motion + masks
video_k[layer], video_v[layer], seq_lens, grid_sizes
cache_stage_id, source_video_sigma, source_action_sigma
forecast_index_offsets / forecast_time_offsets + validity
device, dtype, ready_event
```

其中 source sigma 描述 cache 提取位置，并非整条采样轨迹唯一噪声。采样 schedule 也要纳入模型/运行配置指纹。模拟时间与 wall clock 分开记录，不相减。

快 query 开始时取得一次 snapshot，十步去噪始终使用同一 `bridge_id`；新版本只在下一次 query 生效。发布后不允许原地覆盖 K/V。训练阶段 bridge 可保留 autograd；部署发布版本 detach。

**时间处理的边界**：age embedding 只能帮助网络解释旧信息，不能把旧未来自动变成以 t 为起点的新未来。不对已经做过 RoPE 的 key 简单平移以冒充物理传播。旧预测时间窗结束、episode/instruction 改变或 age 超出训练支持时，应将 bridge 标为不可用，并进入明确的 fallback。

建议首轮采用固定 TTL 与强制身份失效，不先训练不确定度刷新器。对于 age 超界，使用训练过的 bridge-missing 快策略临时运行并请求刷新；记录发生率。若该退化策略尚未验证，就显式降级到同步刷新，并记录阻塞，不能掩盖为异步成功。

## 5. 训练的关键：因果 bridge 与跨时间 supervision

### 5.1 先排除最危险的信息泄漏

现有训练输入包含：

\[
z^v_\sigma=(1-\sigma)z^{v,*}_{future}+\sigma\epsilon_v,
\qquad
x^a_\tau=(1-\tau)A^*+\tau\epsilon_a.
\]

这些是正常的 flow-matching 训练变量。但如果从这张图导出 `B_s`，再让 t>s 的快策略用它预测动作，cache 可能通过 future latent 或专家动作获得部署时不存在的信息。**只做 detach 并不能删除这种信息。**提高噪声、随机跨时间、再增加一个动作 loss，也不能自动解决。

规定两张用途不同的计算图：

| 图 | 输入是否允许真实未来加噪 | 输出用途 |
|---|---|---|
| 因果 bridge 构造图 | 不允许；只用 s 时刻及过去观测、独立随机初始噪声 | 为 t 时刻快动作训练提供条件 |
| 原始 WAM 正则图 | 允许，保持标准 video/action flow matching | 维持原有联合预测能力；其 cache 不进入快分支 |

源专家未来 action 与源真实未来 video 可以作独立监督目标，但不能作为因果构造图的输入。t 时刻的专家动作加噪则属于快分支本身的正常 flow-matching 训练，不与“从 s 偷看未来”混为一谈。

### 5.2 配对数据：同 episode 的早慢、晚快

训练样本由 `(episode, s, t, Δ)` 定义：

- 慢侧：s 的条件 latent、状态、运动历史、指令；这些必须仅由截至 s 的观测得到。
- 快侧：t 的三视角 RGB、状态、截至 t 的运动历史、从 t 对齐的专家动作 chunk。
- 元数据：source/fast condition frame、action indices、真实 simulator timestamps、history endpoint indices、有效性 mask。
- 以 episode 划分 train/validation 后再构造配对，禁止通过窗口重叠跨集合泄漏。

v2 packer 已存 `condition_frame_indices`、`video_indices`、`action_indices`，但 Dataset 不完整暴露；样本 id 是存储编号，不能当时间。当前 dataset 的首个动作索引是 `condition_frame_idx + global_downsample_rate`，新配对必须延续这个约定，不能悄悄改成以当前 frame 的动作作为标签。

新增独立的 paired dataset reader 和小型 manifest，以指纹关联已有 v2；沿用现有预处理、归一化、语言和光流质量契约。所需 RGB 优先从现有 converted 三视角合成视频及对应 frame 索引读取，按固定布局恢复视角；若原始独立视角可用，则保证与部署预处理一致。缺少所需时间戳或图像时明确失败，不用未来 latent 解码充当当前真实图像。

不要批量落盘逐层 K/V，也不需要 RGB-D、点云或重采深度。原始 RGB/时间信息完整时可复用现有采集；是否需要补采仅由数据审计决定。可选的小尺寸 uint8/JPEG 缓存按帧去重，先测容量再生成。v2 shards 保持不变。

`Δ` 是必需训练变量：覆盖 age≈0、常用 age、接近 TTL 的尾部，并按实际运行日志校准抽样比例。不要照搬其他平台的“25 帧”，也不能把 DOMINO 的动作计数直接换算成固定毫秒。第一版以 fast 每执行 4 个动作重规划、slow 每 2 或 4 个 fast query 更新的轨迹为采样参照，最终用真实时间和实测并发发布延迟修正。

### 5.3 目标函数与梯度路径

对同一源 s，可抽取 M 个较晚时刻，第一版 M=1 或 2 控制显存：

\[
\mathcal L_{fast}
=\frac{1}{M}\sum_{j=1}^{M}
\mathbb E_{\tau,\epsilon}
\left\|v_\theta(x_\tau^{t_j},\tau;I_{t_j},m_{t_j},q_{t_j},\ell,B_s,\delta_j)
-(\epsilon-A^*_{t_j})\right\|^2.
\]

\[
\mathcal L
=\mathcal L_{fast}
+\lambda_a\mathcal L_{original\ action}
+\lambda_v\mathcal L_{original\ video}.
\]

沿用现有 scheduler 的 sigma、shift 与向量场符号，不在新模块自行发明另一套噪声时间。M 项取平均，避免多抽几个快样本就隐式增大其损失权重。`λ_a, λ_v` 由验证集和梯度规模选定；视频正则可沿用现有 stage3 权重退火逻辑，不直接复制 DuoCore 的权重。

联合阶段要求这条梯度路径实际存在：

```text
晚时刻动作损失 → 快 Action DiT → 历史 bridge K/V
              → 源时刻 Video DiT / 联合交互 → 慢分支参数
```

为此把无条件 detach 改为显式 `detach_cache`，并将训练使用的慢采样实现放在可微模型入口，不能调用带 `@torch.inference_mode()` 的部署 runner。冻结参数与禁止反向是两回事：即使 video 权重冻结，若要沿联合图更新源 action 参数，也不能不加区分地把所有 video 运算包进 `no_grad()`。

建议分两阶段：

1. **快通道适配**：从已有完整 DynamicWAM checkpoint 初始化。因果 bridge 构造先 stop-gradient，video 固定；训练当前视觉、时间条件和快 action 路径。由明确的阶段契约决定源 action 与快 action 的权重共享；即使共享，buffer 输入仍每轮以当前权重重新生成，不能把离线旧权重 cache 当常驻训练集。
2. **跨时间联合微调**：开放 Video DiT 与 action 的梯度，保留独立原 WAM 正则。梯度应通过产生 cache 的两次联合前向及其间的 solver update；缓存产生后不再影响 cache 的 action-only 步不必为此保留反向图。这是依赖图裁剪，不是用专家 latent 替代采样。

优先用 activation checkpointing、小 microbatch 与梯度累积处理可微采样成本。若只能使用截断反向，应明确标为近似方案并与完整梯度的小规模实验比较，不能声称已验证完全端到端耦合。

新增模块必须进入 optimizer、trainable-parameter 校验、checkpoint 保存/恢复及 strict load 契约。旧 checkpoint 仅通过显式初始化迁移加载，列出允许缺失的新参数；不靠全局 `strict=False` 吞掉错误。无需因这次控制结构修改重新做视频蒸馏 stage1。

### 5.4 训练覆盖不等于闭环覆盖

跨时间示范训练可以覆盖“旧条件 + 新观测”，但源 proposal 生成的假想未来与专家真实后续并不必然一致；部署快策略又可能偏离专家。因此，该方法学的是可纠正的预测先验，不是严格的动作条件动力学求解器。

首轮用短 TTL、当前视觉和原始 WAM 正则限制风险；必须补做闭环动作偏离、接触附近和不同 cache age 的评估。若常用 age 范围仍有系统性失配，应缩短 TTL 或停止该结构路线，再考虑显式执行动作历史条件，不能靠更长缓存强行制造计算收益。

## 6. 推理与调度：两个阶段，最终必须是真并行

### 6.1 阶段 I：可复现的跨周期分频

目的在于先回答“表征能否复用”，暂时使用确定性串行调度：

```text
第一次：用 O_s 构造 B_s，再由 fast(O_s, B_s) 生成动作
之后：每执行 H_exec 条动作，读取 O_t，调用 fast(O_t, 最新 B)
      每 R 次 fast query 刷新 B；R=1 / 2 / 4
```

建议主对比 `K=16, H_exec=4`，所有方法保持相同 H_exec；原始 H_exec=16 作为单独历史参考。R=1 是同结构每轮刷新控制组，R=2/4 才引入跨周期复用。这里的 slow 周期为 8/16 个已执行动作，不能再把 TTL 默认设成 4 并称为分频。

这阶段允许慢刷新阻塞，因此只能叫“受控跨周期分频”，不能叫完成版 DuoCore 式独立异步。默认关闭 motion-drift 自适应刷新以隔离变量；以后启用时报告实际 slow/fast 计数，防止阈值触发把 R=4 退化为 R≈1。

### 6.2 阶段 II：独立 producer / consumer

```text
observation publisher:
    发布当前不可变观测包及递增 observation_id

slow worker:
    初始或刷新到期时，取最新观测快照 s（不排队处理过时请求）
    B_new = causal_slow_sample(snapshot_s)
    完成设备同步/传输后，核验 episode/instruction/model/source_id
    原子发布完整 B_new；丢弃过时结果

fast worker:
    在自己的决策时刻，取当前观测 O_t
    pin 最近一个已 ready 的 B_s
    十步动作去噪全部使用该 B_s 和 O_t
    输出 actions + observation_id + bridge_id
```

慢 worker 的完成时间不触发快 worker 等待；慢 worker 的周期由更新预算/可用性控制。首轮可设置目标周期，但实际比例随计算完成时间变化，并逐次记录。启动时允许一次同步建立 bridge，作为 warm start 单独计时。

并发工程需显式处理：

- 环境由单一 owner 读取/step，worker 只收观测包，避免在不同线程操作仿真器。
- 快慢 worker 各自拥有 scheduler、采样 RNG、临时 latent、指令缓存及 CUDA stream；不能共享一个含可变状态的 runner。
- 权重在 eval 下只读。优先独立快模块实例/轻量权重副本，避免连完整 WAN 一起复制到快路径。复用权重引用也必须排除 hooks、临时 buffer 等写操作。
- bridge 采用引用计数/事件保护的有限缓冲槽；旧版本被快 query 持有时不能覆盖。槽不足时慢 producer 等待或丢弃待发布结果，快 consumer 不因此等待。CUDA 完成事件就绪后才能发布。
- 初步可用两张独立设备检验真正计算重叠，但要计入 K/V 跨设备传输及设备资源差异；最终必须单独给单卡成本与延迟。两个线程或两个 CUDA stream 并不保证大 Video kernel 与 fast kernel 并行，更不保证 fast p95 延迟下降。

本方案先不接入 RTC guidance、硬前缀或动作执行重叠。慢快网络能否并行与仿真时间是否在推理期间流逝，是两项不同的验证。

### 6.3 DOMINO Level-1 中能证明什么

原生同步评测可比较相同执行长度下的成功率、操控分数、更新次数和每个 episode 的累计模型计算，也能检验旧表示条件下的策略。但推理耗时不自动转化为目标运动时间，因此不能仅凭原生 SR 宣称解决了真实延迟下的动态拦截。

对于并发版本，额外以固定录制观测流的 wall-clock 回放测试慢快重叠、发布年龄和反应延迟；该测试验证运行系统，不当成闭环 SR。若以后需要推理延迟影响运动的闭环比较，必须另设时间推进明确的 DOMINO 协议并与原生分开报告，不在这次分频模型设计中混入 RTC 贡献。

## 7. 代价模型与可证伪假设

区分原完整 query 成本 `S`、两步联合前缀建桥成本 `B` 和新增当前视觉后的快 query 成本 `F`。相同 H_exec 时，每个快决策的平均工作量约为：

\[
C_{split}\approx F+B/R.
\]

节约条件是 `F+B/R<S`，即 `F<S` 且 `R>B/(S-F)`。不能再用完整 S 代替已裁剪的 B。

更重要的是，相比原版 H=16，默认 H=4/R=4 每16个执行动作要付出 `B+4F`，原版只需 S。设一次联合前向为 J、一次原 action-only 步为 A、源预处理为 P、新增快条件开销为 E，则 `S≈P+2J+8A`、`B≈P+2J`、`F≈10A+E`，新版多约 `32A+4E`。世界刷新次数没有减少；不能期待默认设置自然超过 H16 原版吞吐。

主架构比较应使用原版 H=4 的 `4S` 对新模型的 `B+4F`，另单列 H16 原版以回答实际使用成本。若两者都要显著提速，需要进一步压缩 F，不能只依靠增大 R 或增加 worker。以上是算子工作量近似，不是实测毫秒。

此式描述总工作量，不是并行延迟公式。独立资源充分时快 query 的响应可接近 F；单卡争用时响应可能比 F 大很多。必须报告快分支在慢 worker 开/关下的 p50/p95/p99，而不是只测 isolated action forward。

逐层原始 K/V 内存约为 `2 × L × N_video × D_video × bytes_per_element`，再乘同时存活的 cache 版本数。要实测 `N_video`、峰值显存和传输带宽，不把 cache 当免费的小向量。

需要证实的三个假设：

1. **可复用性**：在相同 H_exec 下，R>1 能以更少 world 前向维持接近同频版本的性能。
2. **当前观测的纠偏价值**：保留同一旧 bridge 时，当前 RGB/运动显著优于冻结观测；收益不是只来自缩短执行 chunk。
3. **世界预测的额外价值**：在同一快视觉/action 骨干下，加入真实 bridge 优于训练好的无 bridge 基线。若没有差异，应承认快策略绕开了 WAM，而不是靠注意力图解释成“有效世界建模”。

第三条还不足以把收益归因于“未来预测”：bridge 也包含昂贵慢骨干对历史视觉和任务的编码。若论文主张预测带来价值，还需要训练一个仅保留历史/当前条件的慢表示对照，移除未来 token 及其交互，再与完整 bridge 比较。该对照需重新适配训练并报告计算差异，不能只在测试时删 token 后把下降解释成预测能力的因果证据。

暂不声称两时间尺度稳定性定理。当前神经策略、接触切换和部分可观测动力学不满足现成稳定性保证；准确的研究表述是“任务相关预测计算能否在短时间窗内摊销”的可验证假设。

## 8. 实验矩阵：先定位收益来源，再讨论加速

所有主实验先只用 DOMINO Level-1，固定 episode manifest、任务集合、seed、观测接口、动作归一化和最大执行预算。新模型使用相同数据划分和可比较训练预算；筛选后对关键结果用多训练 seed 复验，并报告按任务分层的配对置信区间。

| 组 | 每轮 world 刷新 | 当前快 RGB/运动 | 跨时间训练 | 回答的问题 |
|---|---|---|---|---|
| A | 原始 DynamicWAM 每 query | 原始输入 | 原始训练 | 主基线；另保留其 H_exec=16 历史参考 |
| B | R=2/4 | 旧 cache + 旧 motion，仅新 state | 无 | 裸缓存会退化多少 |
| C | R=1 | 有 | 有 | 新快通道本身与原架构的差异 |
| D | R=2/4 | 有 | 仅 Δ=0 | 单靠新视觉是否足够，跨时训练是否必要 |
| E | R=2/4 | 有 | 有，因果 bridge | 完整模型 |
| F | 不使用 bridge | 与 E 相同 | 对应无 bridge 训练 | 世界预测是否仍然有价值 |
| G | R=2/4，仅历史/当前条件的慢表示 | 与 E 相同 | 有，单独适配训练 | 收益来自预测交互，还是仅来自更强的慢视觉编码 |

在 E 上做少量机制消融：去 age、冻结/更新 motion、bridge stop-gradient/联合训练、强制旧 RGB/当前 RGB。先保证主矩阵与关键消融足够样本，不把每个维度全排列。

额外需要两类验证：

- **干预实验**：固定当前观测和快噪声，改变 bridge 年龄/来源；固定 bridge，改变当前 RGB 或 motion。打乱 bridge 属于分布外诊断，需结合训练好的 F 组，不能单凭破坏输入后下降证明语义因果性。
- **事件分组**：接近目标、首次接触附近、抓持后、目标改变运动方向、多物体干扰。Level-1 不主动改成其它等级；依据已有任务事件或离线分析标签分组，不把仿真特权状态喂给策略。

指标分开列：

1. 策略：SR、MS、执行动作数/模拟时间、接触前后失败类别、chunk 边界动作变化。
2. 计算：slow/fast 次数及实际比例、每 episode GPU 时间、视觉/flow/VAE/Video/Action 各部分耗时。
3. 时效：source-to-fast age、ready-to-fast age、快输出所用观测年龄、TTL 超界率、fallback 率、更新丢弃率。
4. 并发：slow 活跃期间 fast 的 p95/p99、计算重叠 trace、峰值显存、传输成本。

建议在实验前预注册质量容忍阈值，例如“不低于同执行长度基线 2 个百分点”的非劣目标；2 点只是初始实验标准，不能将置信区间覆盖明显退化的结果判为通过。若 E 不优于 F，或性能维持必须 R≈1，应停止将分频 WAM 作为主贡献。

## 9. 文件级实施顺序与验收门槛

| 顺序 | 文件/新增模块建议 | 修改范围 | 必须通过的检查 |
|---|---|---|---|
| P0 | 配置与 benchmark 小脚本 | 分清 denoise、数据采样、执行 horizon、slow 周期；记录原基线时延分解 | 相同观测的原 sampler 输出可复现；实际 slow/fast 次数可核对 |
| P1 | `training/data/paired_world_action_dataset.py`（新增），v2 reader/packer 索引接口 | 配对 manifest、RGB、时间和 mask；不覆盖 v2 shards | 同 episode、s≤t、所有输入历史≤对应时刻、标签首动作偏移正确 |
| P2 | `models/fast_observation_encoder.py`（新增），`models/small_wam.py` | 当前视觉 residual、age/状态差条件、cache detach 开关、统一训练 forward mode | 旧 aligned 模式保持可复现；新参数全部被 checkpoint/optimizer 管理 |
| P3 | `models/causal_world_bridge.py`（新增），`training/train/video_action_trainer.py` 和 objective | 因果源采样、M 个跨时间快 loss、独立 WAM 正则 | 更换真实未来标签不改变固定输入/噪声下的 B；快 loss 可回传 video |
| P4 | `runtime/ciwam/wam/{world_buffer,flow_runner,policy}.py`，新分频配置/adapter | 当前 packet、只读 bridge、R=1/2/4 受控部署 | 单 query 内 bridge 固定；跨 query 确实复用；motion 按当前时间更新 |
| P5 | 新 `fast_slow_workers.py`，DOMINO adapter | 独立 sampler 状态、快慢 worker、事件发布/版本校验 | 人为延长 slow 不挂起正常 TTL 内 fast；旧 episode 结果不能污染新 episode |
| P6 | 验证脚本与评测汇总 | 主矩阵、配对统计、trace、失败分类 | 满足质量/计算/尾延迟标准后才宣布完成 |

入口设计应让 paired training 通过一次受 Accelerate/DDP 管理的 `model.forward(mode=...)` 编排多分支；避免绕过 wrapper 随意调用底层子方法，导致梯度同步或 unused parameters 行为失控。新增可训练模块要加入现有 `_build_optimizer_and_scheduler`、`_verify_trainable_setup` 的显式校验。

配置建议独立成 `world_action_multirate_v1`，schema 显式区分：

```yaml
denoising: {video_steps: 2, action_steps: 10}
policy: {chunk_size: 16, execute_horizon: 4}
slow_update:
  mode: controlled_ratio   # 完成 P5 后增加 independent_worker
  every_fast_queries: 2    # 对照组 1；复用组 2/4
bridge:
  representation: layerwise_video_kv
  source: causal_sampler
  train_detach: false      # warmup 阶段显式覆盖为 true
fast_observation:
  cameras: [head, left_wrist, right_wrist]
  use_current_motion: true
  use_cache_age: true
```

此片段是待实现 schema 示意，不是当前仓库可直接运行的配置；TTL 秒数、age 分布、快 encoder 分辨率需由 P0/P1 的实际时间和资源测量确定，不给一个伪精确默认值。

## 10. 开始大规模训练前必须排除的失败模式

- **泄漏检查**：固定源观测与慢噪声，替换源未来 video/action 标签，B 必须不变；训练程序不能在构造 B 前读取这些字段作为输入。
- **梯度检查**：关闭原 WAM 正则，仅保留 fast loss，联合阶段 Video 参数应有梯度；warmup 则按契约无梯度。另检查视觉门控没有让新 encoder 永久收不到梯度。
- **配对检查**：检查 s/t 的图像、flow 历史端点、动作索引及 simulator time 可视化抽样，尤其 episode 开头缺历史、结尾不足 chunk 的样本。
- **年龄检查**：延长慢计算后，新发布 bridge 的 source age 应增加，不能因 write 时重置年龄就显示“全新”。
- **缓存检查**：层数、device/dtype、episode、instruction、model version、cache stage 均匹配；fast 十步中禁止混入另一个版本。
- **动作条件检查**：评估快动作偏离源 proposal 后的性能；不把旧 K/V 当对任意候选动作都有效的未来预测。
- **并发检查**：用确定性 toy producer/consumer 测试延迟、丢帧、reset 和超时，再上真实模型；不要以 Python 线程存在作为并行成功证据。
- **归因检查**：所有关键组固定 H_exec 与去噪步数；报告并行版本的硬件数量，不能用双卡对单卡速度比解释成纯架构收益。

最终完成标准是：**当前观测驱动的快策略可以在多个控制周期复用因果生成的世界表示；跨时间训练让该表示对后续动作确有价值；慢更新不要求快推理停下来；在公平控制条件下同时验证质量与计算收益。**仅减少 video 前向次数、仅增加 cache、仅加入 RTC，均不满足这个标准。

## 11. 实现范围与必须固定的架构决定

这里以完整的可训练、可导出、可并行部署的系统为交付单位。只给现有方法多加几个参数、只改 `choose_path()`、或只跑通一个 forward，均不算实现完成。可以分里程碑验证，但模型、数据、训练、配置、checkpoint、运行时和评测必须形成闭环。

本节及后续内容细化 V1 参照系统的完整实现合同；已有代码只完成其中一部分。最终快网络可以按 §18 重构为 V2，不强制保留全部12层 current residual 或源/快 Action Expert 共享。V1 的以下决定用于提供可比较、可验证的起点：

| 问题 | 主实现决定 | 理由/边界 |
|---|---|---|
| 慢路径到底算几步 | 精确运行原 sampler 的 step 0 联合前向、一次 video/action 更新、step 1 联合前向并导出 cache | 后续八个 action 步不影响 cache；完整 sampler 仅用于等价性对照 |
| 源 action 与快 action 是否共享 | **训练时共享同一套原 Action Expert 和 motion-token 参数** | 保持联合 WAM 初始化；快观测模块只在 fast 路径启用，源路径不偷读 t 的信息 |
| 当前视觉怎样接入 | 新的 `FastConditioner`，12 层独立 residual；原 token 序列和解码位置不变 | 不把新视觉挤入旧 WAN 时空 RoPE，也不破坏原 action position embedding |
| 训练入口 | 一个顶层 `model(batch, mode="multirate_train")`，内部编排所有子图 | 让 Accelerate/DeepSpeed 包装器看到完整计算图 |
| 推理入口 | 区分 `predict_fast_velocity` 与 `sample_fast_actions` | 前者只返回一次向量场，后者才完成十步去噪和动作反归一化 |
| cache 类型 | 训练使用带图的 `BridgeTensorBundle`；部署才封装只读 `WorldBridgeSnapshot` | 模型不依赖线程锁、CUDA event、模拟器对象 |
| 旧格式如何兼容 | 旧 v2 只作初始化来源；新增 multirate profile、checkpoint 格式和加载分派 | 当前训练、部署都严格检查键和值；不能伪装成原始 v2 export |
| 无效 bridge 如何运行 | 训练 null-bridge 快策略，明确可评测的 fallback；启动允许一次同步建桥 | 不把未训练的“全零 cache”当作可靠退化模式 |
| 独立 worker 是否可选 | 完整交付必须包含，固定 R 调度仅是一个验收模式 | 模型质量不足可以否定研究假设，不能把串行实现重新命名为异步 |

新增运行模式均保留原同步 adapter 可调用，不改变已有基线配置的含义。主实验之外的压缩 bridge、RTC、分割跟踪仍单列研究，不作为补齐上述工程闭环的替代。

## 12. 模型拆分、张量接口与数值语义

### 12.1 模块边界与文件归属

建议新增 `models/multirate_wam.py`，实现/完善 `MultirateWAM`，组合现有共享计算函数。按当前 wrapper 使用 `backbone.compact_wan.*`、`backbone.action_expert.*`、`backbone.absolute_motion_tokens.*`；新参数归到 `fast_conditioner.*`。旧 v2 迁移显式添加 backbone 前缀，不能把有前缀和无前缀的格式混在同一版本中。不要把同一个 Action Expert 同时注册到两个父路径，制造重复 state_dict 键或 optimizer 重复参数。

| 文件 | 负责什么 | 明确不负责什么 |
|---|---|---|
| `models/multirate_types.py`（新增） | 源输入、当前输入、训练 bridge 与 fast memory 的张量契约 | 不放环境或线程对象 |
| `models/causal_world_bridge.py`（新增） | 两次联合前向及中间 solver update 的编排 | 不编码 RGB，不读动作/未来视频标签，不建 worker |
| `models/fast_observation_encoder.py`（新增） | 共享三视角 encoder、位置/view 信息、静态文本池化 | 不运行 Video DiT、VAE 或 T5 |
| `models/fast_conditioner.py`（新增） | 当前条件逐层 K/V、action query residual、age/状态差条件 | 不修改旧 WAN RoPE 或 action core 的位置 |
| `models/action_core_ops.py`（新增） | 提取可共用的 cached-world attention、FFN、动作解码计算 | 不持有完整 WAN 模型作为运行依赖 |
| `models/flow_schedule.py`（新增） | 从现有 solver 日程生成设备端只读系数、Euler 更新 | 不改变第三方 `vendor/wan/utils/fm.py` |
| `models/small_wam.py`（修改） | 显式 cache detach 开关、可复用的纯 block 接口、保持 aligned 模式行为 | 不直接加入 DOMINO 调度分支 |

当前 `_action_attention_with_video_cache` 仍通过 `compact_wan.video_model.wan_model.blocks[layer_idx]` 取得 attention metadata。需要拆掉这个对象依赖：把 head 数、head dim、precision、window size 收敛为 `ActionAttentionSpec`，真正的快部署模型只持有 action、motion、fast conditioner 权重。否则名义上分了快路径，构建 runtime 时仍会加载完整 WAN/VAE/T5。

训练与部署存在两套 `CompactWANModel` 包装器，分别在 `training/models/compact_wan.py`、`inference/models/compact_wan.py`；前者没有 `encode_video`。因果采样必须只依赖两边都有的 token/head/rope 接口，源 `condition_latent` 由数据或部署输入层提供。

### 12.2 明确输入和输出，不用一个大 batch 字典到处传

`B` 为 batch size，`M` 为同源快时刻数，首版固定 M=1，支持 M=2 后再扩展。

```python
# 示意签名；均为待实现接口
build_causal_bridge(source: SourceObservation,
                    noise: SourceNoise,
                    schedule: BridgeSchedule,
                    *, detach_cache: bool) -> BridgeTensorBundle

prepare_fast_memory(current: CurrentObservation,
                    bridge_meta: BridgeAge) -> FastConditionMemory

predict_fast_velocity(noisy_actions, action_t,
                      state, motion, motion_masks,
                      bridge: BridgeTensorBundle,
                      memory: FastConditionMemory) -> Tensor  # [B,16,14]

sample_fast_actions(current, snapshot, schedule,
                    *, generator) -> ActionChunk             # 10 steps

forward(paired_batch, *, mode="multirate_train",
        phase="warmup") -> dict[str, Tensor]
```

关键字段及契约：

| 对象 | 张量/字段 |
|---|---|
| `SourceObservation` | `condition_latent` 的原 shard 形状；`state [B,14]`；`motion [B,4,12]`；两个 `bool [B,4]` mask；T5 embeddings +有效长度；源身份与观测时间 |
| `SourceNoise` | 随机未来 latent `[B,C,2,12,10]`（当前 future_size=192×160、8 帧）；随机动作 `[B,16,14]`。C 从当前 latent/模型契约校验，不凭经验硬写 |
| `CurrentObservation` | `rgb [B,3,3,160,160]`，轴依次为 batch/view/channel/height/width；像素有效 mask；当前 state/motion/masks；与源相同指令；当前身份、时间 |
| `BridgeTensorBundle` | tuple of 12 个 `K_l,V_l [B,N_v,16,128]`；int seq_lens；layout；源噪声日程标识。训练时不 detach、不复制到 CPU |
| `FastConditionMemory` | 一次当前视觉/文本/age 编码后，12 层当前条件 K/V 及其 mask；十步去噪均复用 |
| `ActionChunk` | 已反归一化的 `[16,14]` qpos，附 `query_id, observation_id, bridge_id, episode_generation` |

训练中的 state/action 已经由现有 packer 归一化；部署的 qpos 只在输入层归一化一次。`q_t-q_s` 用同一套 action stats 下的归一化状态差，不能对已经归一化的状态再次调用 normalizer。motion 仍把原始 12-D 数值送进 `AbsoluteMotionTokenModule`，由它使用原 mean/scale 与 mask 处理。

源与快输入用不同类型/显式构造函数隔离。不得通过 `{**batch}` 把 fast 图像或未来标签传入源构造器。禁止模型对象属性保存“上一轮训练 cache”，避免跨 batch 图残留和图串台。

### 12.3 因果两步前缀：复现现有 solver，而不是重新定义二步模型

首先一次性从原 `FlowMatchScheduler.set_timesteps` 取得日程，保持 `shift=5, sigma_min=0, extra_one_step=True`。用 action 的**十步日程前两项**，绝不能给源 action 新建二步日程。按当前配置，理论上：

```text
video sigma:  [1, 5/6]                 # 两步 video 日程
action sigma: [1, 45/46, ...]          # 十步 action 日程的开头
step 0: joint(z0, a0, video_t0, action_t0)
z1 = z0 + video_pred0  * (video_sigma1  - video_sigma0)
a1 = a0 + action_pred0 * (action_sigma1 - action_sigma0)
step 1: joint(z1, a1, video_t1, action_t1) -> cache1
return cache1                         # 不需要 step 1 之后的更新
```

实际系数从代码生成并记录，分数只作人工核查。模型输入 `timestep` 是 sigma×1000，并保持旧路径的 dtype；sigma 不是直接当 model timestep。固定噪声检查两条路径的逐层 K/V，而不是只比较最终动作。

现有 `step()` 对 timestep 做 `.cpu()`/argmin，会产生不必要的 CPU 路径；但这**不意味着 sample/model_output 的梯度自动被切断，也不能断言 CPU 零维 sigma 一定导致 device error**。替换的目的应表述为显式按步索引、减少同步及统一训练/部署数值。新 helper 保留原系数精度与乘加 promotion；不要为了省事把所有 delta_sigma 截成 bf16 后又宣称完全等价。

step 0 不需要导出并保存 cache；step 1 才需要返回 cache。初版 step 1 可仍计算完整联合前向，避免改变 checkpointing 结构；确认依赖后可裁剪未使用的最终预测 head。对 cache 本身，第二次 forward 最后一层记录点后的一些运算没有依赖，不要求 fast loss 对这些无关参数都有非零梯度。

`detach_cache` 从公共接口一路传到 `_append_video_cache_layer`、`_video_attention_kv_for_cache`。warmup 对完整源生成使用 `torch.no_grad()` 并 detach；joint 全过程保留图。训练禁止用 `inference_mode` 生成要送入可反向 fast attention 的张量，二者不能互换。[PyTorch 2.7 inference_mode](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html)

### 12.4 当前条件 residual：给出可实现的算子

原 action token 布局为 `state 1 + actions 16 + motion 4 + registers 4 = 25`，原解码只取前 17 个再去掉 state。保持该布局、`core_length=17` 和原 state/action/register position 不变。

新 `FastConditioner` 把当前 RGB 的 48 个 token、4 个文本池化 token、1 个 age/state-delta token 合成 53 个 memory token。T5 输入通道从已加载 embedding 的契约获得，4 个文本 token 由 learned pooling queries 得到，并正确屏蔽 padding。空间位置与 view embedding 必须同时存在；仅 view embedding 不能区分同一相机内的空间位置。

每个 action block 的 fast 模式计算：

\[
h'_l=\operatorname{WorldAttention}_l(h_l,B_s),\qquad
h''_l=h'_l+W^O_l\operatorname{Attention}
\bigl(W^Q_l\operatorname{LN}^{cur}_l(h'_l),K^{cur}_l,V^{cur}_l\bigr),
\]
\[
h_{l+1}=\operatorname{OriginalAdaLNFFN}_l(h''_l).
\]

当前 attention 使用独立的 768-D 投影，例如 8 heads×96；不要误用世界 attention 的 16×128=2048 作为 768 维的直接 reshape。`Kcur/Vcur` 每 query 算一次，Q 随 noisy action 每个去噪步重算。慢联合路径和原 WAM 正则路径完全不调用这一 residual。

age/state-delta 也放入此 residual memory，避免随机初始化的 age MLP 直接加到旧 action token 后破坏初始化对照。输入包含 `delta_sim_seconds`、已完成动作数差、bridge-valid 标记和归一化 q 差；采用训练集/配置固定的 scale，越界要标记，不能静默 clip 后把旧 cache 伪装成年轻 cache。

**初始化选择固定为：Q/K/V 正常初始化，输出投影 W_O 为零，无第二个零门控。**第一步 W_O 获得梯度，encoder/QKV 的梯度可为零；做一次有效 optimizer 更新后再验收其梯度。不能同时要求精确零残差和第一步 encoder 梯度非零。初始化等价性只在相同 cache、state、motion、noise 下成立，不代表完整新策略与旧慢 sampler 的动作必然相同。

若选择预训练 ResNet-18，BatchNorm 需专门处理（当前两层随机 stem 并不是该方案）：首版固定预训练 running statistics，保持 encoder 中 BN 为 eval（父模型 `.train()` 后也要恢复），允许其它权重学习；否则 microbatch=1、不同相机混 batch、激活重算会引入额外统计偏移。将这一策略和预训练权重 hash 写入 checkpoint。

### 12.5 Null bridge 与 M 个快样本

训练加入少量 whole-bridge dropout，初始候选率 5%，由验证集调整。null 路径仍保留 action self-attention、当前视觉、当前 motion/state 和文本；不把旧 bridge 留着但仅把 V 置零。用 `N_v=0` 的明确分支或严格 attention mask 实现。

首版每个 batch 使用一致的 bridge-present 模式，避免把不同有效 N_v 的 padded world tokens 与 action tokens拼接后，错误地用 `k_lens=N_valid+25` 把 action 截掉。若扩展为 per-sample dropout，必须实现正确 packed/varlen 排列或完整掩膜，单靠修改 seq_lens 不够。

M>1 时循环消费同一带图 B_s，累计平均 fast loss；不要对 B_s 做 M 份 clone/repeat 来制造显存开销，也不要每个 t backward 后再复用已释放计算图。整组 forward 完成后统一 backward，batch 结束不保留图。

## 13. 数据与输入处理：把时间、语言、像素对齐做到可复现

### 13.1 配对索引与采样分布

新增 `paired_world_action_dataset.py`、`build_paired_manifest.py`。已有 `episodes.jsonl` 有 split/task/episode/range/lang_group_id，`samples.jsonl` 有 sample_id 与 condition_frame_idx，可以构建紧凑索引；不要在每个 DataLoader worker 各自扫描全部 raw 数据或建立巨大的 Python tuple 字典。

以 `(parent_dataset_fingerprint, episode_id, condition_frame_idx)` 为键，检查 manifest 与 shard 的 sample/episode/frame 索引一致。原始 v2 packer 为所有合法 condition frame 打包，`samples_per_episode=10` 是 epoch sampler 的抽样契约，不是“每 episode 只存 10 帧”；配对可覆盖完整合法索引范围。

先固定 train/validation episode 清单，再产生 source 与 fast 配对。保留 episode-balanced 的 source 采样，固定每个有效 source 的 M，防止长 episode 或较小 Δ 获得隐式更高权重。对不足以容纳最大 Δ 的 source 明确指定预过滤策略，并对所有组使用相同有效 episode/source 范围，报告丢弃量。不能无限 rejection 或将 t 夹到末尾。

`global_downsample_rate=3` 是**演示 raw frame 与 policy action 标签**的映射，不是在线一次 action 永远推进三个相机帧。在线用成功完成的 action counter 和实际观测时间，离线偏移按演示映射模拟该计数。

在固定 H_exec=4 的受控调度下，稳态旧 bridge 实际被读取的 age 为：

| R | 在线动作龄 | 对应演示 raw frame offset |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 0、4 | 0、12 |
| 4 | 0、4、8、12 | 0、12、24、36 |

第 8/16 个动作处分别已经刷新，所以 +24/+48 是 R=2/4 的刷新边界，并非普通稳态 cache 的必然使用年龄。独立 worker 的生成/发布延迟才会增加尾部年龄。训练可保留 +48 等压力样本，但必须标明来源；不得把 query 周期、raw frame 差和 simulator seconds 混成一个 Δ。

### 13.2 语言与时间信息的泄漏边界

当前 `_load_language_embedding(lang_group_id, sample_id)` 通过 sample_id 选择提示词。同 episode 的 s/t 可能因此抽到不同表述。配对数据必须明确只选一次 `lang_id`，让源、快与对应正则共用同一句文本；部署 `scene_prefix` 的处理也必须一致。

`interception/*.pt` 中仅允许读取当前/过去观测的 `frame_index, sim_step_index, sim_time_seconds` 等时钟字段。目标位置、规划拦截时刻、接触标签等特权字段不能作为模型输入。使用 allowlist loader，不能把整个 interception dict 放入 batch。

同样不能把 source 之后的**真实未来 simulator timestamps**作为 bridge 预测时间表：这些时间在线未必可知，且可能依赖后续执行。bridge 保存已知的 forecast action offsets `[2,4,...,16]` 及 source clock；确需 nominal seconds 时按明确模型假设生成并标为 nominal。TTL 使用观测年龄与执行计数，真实未来时间仅作离线评估目标。

各个 token 已经过多层全局交互，不能因为某个预测 action offset 过期，就假定删除对应 latent token 后其余 cache 完全不受影响。首版整份 bridge 到期失效；不实现未经训练的 RoPE 平移或 token 切片“续命”。

### 13.3 训练/部署共用三视角预处理

新增 `observations/multiview.py`，共享相机规范化、裁切布局、缩放/补边、RGB channel 顺序和归一化。训练原始合成 mp4 在 head=240×320 时为 360×320，裁切区为 head `[0:240,0:320]`、left `[240:360,0:160]`、right `[240:360,160:320]`。实际以声明布局和 decoded shape 校验，不对未知分辨率静默使用常量。

重要纠正：`image.load_video_frames(..., target_size=...)` 调用的是 `resize_with_padding`，并非无条件拉伸。360×320 到 384×320 的当前情形是上下补边，坐标原点也会变化。快视觉应从**未经 VAE canvas 处理**的图像裁切，而不是猜偏移后裁 VAE 输入。

部署可直接读取三个 camera，但先将腕图规范到训练存储的半分辨率，再执行快 encoder 的公共 resize。converter 的腕图缩放使用 `INTER_AREA`，当前部署 composite 的默认插值不同；新 fast 通道必须显式统一插值，不能仅凭 shape 相同就认为 train/deploy 等价。mp4 压缩与在线无损帧的差别属于残余输入差异，可用离线成对图像检查量化。

图像进入 GPU 前尽量保留 uint8；三视角统一 letterbox 到160×160，保留 padding mask。确认 head/左右腕不交换、BGR 在解码端转 RGB、像素只除255一次。encoder 的 ImageNet mean/std 与 WAN 的 [-1,1] 预处理分别维护，不混用。

避免 DataLoader I/O 抵消全部训练收益：每个 worker 持有有界 `VideoReader`/帧 LRU，一次 `get_batch` 取同 episode 的所需帧，限制 decoder 线程数。CPU worker 不初始化 CUDA，不缓存计算图；图片增强保持每视角几何一致，不随意做会改变动作含义的水平翻转。

### 13.4 分布式复现与数据身份

paired manifest 保存父 v2 指纹、episode split、索引/时钟来源摘要、preprocess version、采样分布、seed、M 和 schema 版本；checkpoint 同时记录 parent_dataset_identity 与 paired_dataset_identity，不能假装新数据身份等于旧 v2。

配对抽样的随机性由 `(epoch, source_id, draw_index, seed)` 确定，不依赖 worker 恰好抢到哪条任务。训练 sampler 在 Accelerate 分片前只生成一套逻辑序列，防止手工 DistributedSampler 再被 Accelerate 二次分片。验证集 pairs 固定；恢复训练保存 epoch/draw cursor 和独立噪声 generator 状态。

## 14. 训练器、反向传播、优化器与模型格式

### 14.1 单一 wrapped forward，保留现有后端语义

当前 `training/train/common.py::build_accelerator` 显式创建 `DeepSpeedPlugin`，默认 profile 为 ZeRO stage 0。真正的主后端是 Accelerate/DeepSpeed，DDP 是可选对照，不应只写一个“DDP 不报错”的实现。

新增 `MultirateTrainer` 和 `MultirateLossObjective`，复用/提取当前日志、分布式循环、归一化和 export 公共设施；不要把新阶段硬塞成旧 `stage2/stage3` 而绕过其严格校验。建议新入口为 `scripts/train_multirate.py`，阶段命名 `fast_warmup`、`cross_time_joint`。

顶层模型一次 forward 内执行：因果 bridge → M 个 fast velocity → 独立 aligned WAM regularization，返回 fast/regularization 预测或可微 loss 张量。Dataset 提供干净标签，trainer 明确分开生成 source noise、fast training noise/time 和 regularization noise/time；检查随机流独立。

```python
# 包装前：构建全部模块、迁移权重、设定本阶段 requires_grad、建 optimizer
model, optimizer, loader, lr_scheduler = accelerator.prepare(...)
for batch in loader:
    with accelerator.accumulate(model):
        prepared = prepare_multirate_training_batch(batch, phase, rngs)
        outputs = model(prepared, mode="multirate_train", phase=phase)
        loss = objective(outputs, prepared)  # mean over B, M and valid actions
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_parameters, grad_clip)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

这是 wrapped optimizer/scheduler 的调用约定，不代表每个 microbatch 都做一次实际参数更新。使用与现有代码相同的 Accelerate 后端契约，不再手动多除一次 accumulation steps，也不在 microbatch 开头清掉之前的累积梯度。[Accelerate 梯度累积说明](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)

不得在训练器里使用 `model.module.build_causal_bridge` 或 `unwrap_model(model).forward_fast` 计算 loss。内部子函数当然可以调用；关键是这些调用位于顶层 wrapped forward 的计算图内，而不是绕过分布式入口。

### 14.2 warmup 与 joint 是两个明确的参数契约

| 模块 | fast_warmup | cross_time_joint |
|---|---|---|
| Video DiT | 固定；源构造 no_grad；无 video loss | 可训练；两次源联合前向保留图；有独立 video 正则 |
| 原 Action Expert / motion encoder | 可训练；源构造 stop-gradient，fast 与 aligned action loss 更新 | 可训练；源、快、正则共同贡献梯度 |
| FastConditioner | 可训练 | 可训练 |
| VAE、T5 | 固定；训练用已编码条件/文本 | 固定 |
| bridge | 在线按当前共享权重生成、detach | 在线生成、不 detach |

`SmallWAMActionConfig.wan_frozen` 当前还影响普通 forward 是否输出 video_pred。warmup 虽冻结 video，源 step 0 仍需要 video_pred 做更新，必须通过显式 `include_video_prediction=True` 请求，不能让冻结标志误删这个计算。

阶段切换建议用 warmup export 初始化新的 joint run，重建模型冻结集合、optimizer 和分布式 wrapper；不在已经 prepare 的模型上临时解冻参数并期望旧 optimizer/ZeRO 自动接管。`init_from` 与同阶段完整 `resume` 分开，后者恢复 optimizer、scheduler、sampler 和 RNG。

optimizer 分组至少 `action_core`、`fast_conditioner`、`video_expert`。`_configure_trainable_parameters` 只控制训练开关，真正的 param groups 在 `_build_optimizer_and_scheduler` 创建。以参数 id 做三组无交集和全集覆盖检查：每个 requires_grad 参数出现一次，固定参数不出现。零初始化参数可能首步梯度为零，不能把数值零与 `grad is None` 混为一谈。

### 14.3 激活重算必须是纯函数

当前 `_append_video_cache_layer` 对列表 append。**不要直接 checkpoint 一个带 append 副作用的 block**：反向重算可能再次写列表。block 应返回 `(video_next, action_next, k, v)` 张量，由 checkpoint 调用之外一次性组装 tuple；源 step 0 不收集 cache，step 1 收集。

同一权重被两次源前向和一次正则反复使用，默认使用 `checkpoint(..., use_reentrant=False)`，对照无 checkpoint 的 loss/梯度。参数注册和计算路径在包装前固定，不用 `find_unused_parameters=True` 掩盖漏调模块；但诊断时也不能声称所有“unused”都由 fast encoder 引起，应区分没有损失依赖的 video head/末端 block。[PyTorch 2.7 DDP 与 checkpoint 限制](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)

先实测 microbatch=1 的峰值、重算耗时与梯度，再调累积步数；不承诺“显存固定是原来的两倍”。原图、缓存存活、重算和 attention 实现都会改变峰值。不要把 step 0 的更新放到 no_grad 来代替正确重算。

### 14.4 配置及 checkpoint 必须贯穿全部入口

建议新增 profile schema 3 / `baseline: world_action_multirate_v1`，旧 v2 验证器仍严格保持原语义。`config/loader.py::load_profile` 按 schema 分派；新 profile 提供 training、inference、benchmark 的完整 resolved config，并同样写入 config snapshot/hash。

新 schema 的必需组：base WAM architecture、`fast_conditioner`、`bridge_sampling`、`paired_data`、`training.phases`、`multirate_runtime`、normalization、checkpoint 与资源路径。全局拒绝未知键，明确 H_exec≤K、R≥1、layout/dim/head 相容、TTL 不超过训练支持、阶段和 loss 开关一致。不能只新建一个 sidecar 文件而绕过训练/部署加载器。

部署新 loader 不能继续接受“只含旧 model keys”的校验。修改点包括 `_load_checkpoint_payload`、`_validate_model_config`、runtime builder，以及 trainer 的 `_validate_stage_contract`、initial checkpoint 校验、dataset identity 校验和 export。应通过新格式分派或独立 `multirate_model_loader.py` 实现，不放宽旧 loader。

建议新导出格式 `dynamicwam_multirate_checkpoint`，version 1，至少包含：完整共享模型 state、resolved model/preprocess/schedule config、父权重 SHA、父/配对数据身份、归一化和 motion 契约、训练阶段/step、受训 age 支持和是否训练 null bridge。设备 event、缓存内容、worker 状态不属于模型 checkpoint。

旧权重迁移流程：校验原 v2 → 校验全部旧键名/shape → 构建新模型 → 以初始化值补齐**明确列出的** `fast_conditioner.*` 新键 → 组合完整 state → `load_state_dict(..., strict=True)`。其它缺失/多余键直接失败。新 checkpoint 的 resume/deploy 不走迁移特例，必须严格完整。

除了主完整 export，再生成可核验的 fast-only 部署视图，仅含 action、motion、conditioner 与 attention metadata；使用同一 model_version/parent SHA，随机输入验证与完整模型 fast 输出一致。不能让快 worker 构建完整 WAN 再宣称已经完成模型资源解耦。

## 15. 运行时：受控调度、独立 worker 与缓存状态机

### 15.1 受控阶段 I 的队列语义

新增 `deploy_policy_multirate_flow.py`、`multirate_policy.py`，不要在旧 RTC sidecar 中堆新参数。每次 callback 仍由环境 owner 执行至多一条 action，并发布当前观测。快 query 按 `executed_since_query >= H_exec` 触发，**不是等完整 16 条 pending 都执行完**。

```text
query_id 从 0 开始
if 首次 or executed_since_query == H_exec:
    O = freeze_current_observation()
    if query_id % R == 0:
        B = build_causal_bridge(O)       # 受控模式允许阻塞
    actions = sample_fast_actions(O, B)  # 完整十步，不是一次 velocity
    pending = actions[:H_exec]          # 未执行尾部明确丢弃
    executed_since_query = 0
    query_id += 1
execute_one(pending.pop_front())
成功执行后更新 action_index 与 executed_since_query
```

保持 K=16 生成契约，H_exec=4 决策间隔；必要时可保留16条作诊断，但不能让队列非空阻止第4步重规划。计数以环境已完成的动作定义，不能沿用 `pop_action()` 时就 tick 的行为来表示“执行已完成”。每次 task reset 清空 pending、flow history、query id 与桥身份。

当前 `HeadFlowBuffer.push` 应在每次环境观测更新时执行，flow history 的采样步幅仍维持原契约。不要只在 slow refresh 时喂图，也不要把 fast query 次数当作 history 的 policy stride。当前 motion 在 packet 中保持 t 时刻版本，删除新 adapter 对 `frozen_motion()` 的依赖。

受控模式关闭事件刷新和延迟注入；N 次 fast query（含首轮）应产生 `ceil(N/R)` 次 bridge 构造。缓存失效/fallback 另记，触发时不再把实际过程标为严格固定 R。

### 15.2 独立 worker 的资源与消息接口

完整部署分为 `ObservationPublisher`、`SlowBridgeWorker`、`FastPolicyWorker`、`BridgeStore` 和环境 owner。建议第一种实现是同进程两 worker、独立 eval 模型对象与 stream；再提供双设备模式隔离计算干扰。跨进程/CUDA IPC 不作为第一版默认，以免把数据传输问题混入架构验证。

训练时共享 action 权重，部署时从同一冻结 checkpoint 加载只读快副本与慢完整模型，参数一致但临时状态独立。快模型不加载 WAN/VAE/T5；T5 静态结果在 episode 开始时生成并提供给两边。慢侧只在拿到新的 source snapshot 时编码 VAE。每个 worker 独立 RNG/日程/临时 memory，不能共用 `HeadFlowRunner.reset()`、`current_instruction`、`_rope_freq_grid_cache` 等可变状态。

消息至少包括：

```text
ObservationPacket: generation, observation_id, action_index,
                   sim_time, capture_wall_time, immutable RGB/state/history
RefreshRequest:    generation, request_id, source_observation_id
BridgeReady:       generation, bridge_id, source_id, model_version, slot_id
ActionResult:      generation, query_id, observation_id, bridge_id, actions
```

队列容量有界：source/refresh 请求采用 latest-wins，只保留最新未启动请求；已经运行的 GPU 工作不能假设可以安全强制取消，完成后检查 generation 丢弃过时结果。TTL 内 slow 忙碌不阻止 fast 用当前 ready bridge 工作。模型加载、warmup、episode reset 与异常传播有明确 ready/error 信号；没有健康 worker 时不能悄悄重复提交旧动作。

### 15.3 slot 回收必须同时满足 CPU 引用和 GPU 完成

缓存槽状态为 `FREE → WRITING → PENDING_GPU → READY → RETIRED → FREE`。`current_ready_slot` 的发布指针与引用计数受同一短锁保护；锁只保护元数据，不覆盖慢前向或任何同步等待。

协议：

1. slow 只取 FREE 槽写。**当前已发布槽即使 CPU refcount=0，也不能被改写**，因为下一次 fast acquire 仍可能选它。
2. slow 完成写入/跨设备拷贝后记录 producer event，先进入 PENDING_GPU。确认 event 已完成、身份仍匹配、source_id 不倒退，才原子置为 READY。
3. fast 在 query 开始时 acquire 一次 READY snapshot；这一 query 的十步都用它。memory 的 `.to(device)`/布局转换在 acquire 处完成并计时，不在每层每步隐式复制。
4. 新槽发布后，旧槽成为 RETIRED；已有消费者仍可使用。CPU refcount 归零还不够，所有 fast consumer 的 done event 也完成后，旧槽才能 FREE。
5. 槽耗尽时慢方跳过/等待空槽并累计背压指标，不能让 fast 等新 bridge，也不能无上限分配 K/V。

`record_stream` 负责 allocator 生命周期跟踪，并不表示数据可读；生产事件和消费者完成事件负责先后依赖。不能在 Python 把推理 kernel 入队后立即减引用并重用存储。对于读旧 READY bridge 的 fast 路径，不调用 `torch.cuda.synchronize()` 去等待整张卡慢前向。[PyTorch 2.7 CUDA Stream](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.Stream.html)

正常同设备流程使用 2–3 个有界槽即可验证，具体数量由实测存活版本数和显存确定。训练 autograd bridge 不进入这个池；重用训练 storage 会破坏反向所需版本。

### 15.4 Age、TTL 与动作结果接收

source age 始终取 `current_sim_time - source_sim_time` 和已完成 action counter 差；另记 source 到 ready 的 wall duration。不得把“已执行动作数”与“新增观测数”再相加重复计龄，重复观测也不能让 action age 增长。

warm start 用一次同步 bridge。之后只接受同 generation、同指令/模型且不倒退的 ready 结果；query 输出同样带 generation 和 observation_id，reset 后到达的旧 action 不能提交。独立 worker 模式中计时器和 TTL 需明确采用 sim clock 还是 wall clock，原生 DOMINO 不把墙钟延迟擅自换成物体运动时间。

未到新 query 时不能把完成的新 action chunk强行写进当前执行队列。当前范围不包含动作执行并行/RTC：环境 owner 在需要新 query 时可等待 fast 结果，但不能等待 slow 刷新；如果以后允许执行期间持续 fast 推理，再另行设计 action timestamp 对齐与提交截止点。

## 16. 逐文件实施清单与可执行验收

### 16.1 必须覆盖的文件面

| 改动包 | 新增/修改位置 | 完成标志 |
|---|---|---|
| 公共输入契约 | `observations/multiview.py`、`models/multirate_types.py` | train/deploy 像素和时间含义一致，source 类型无标签字段 |
| 模型重构 | `models/small_wam.py`、`action_core_ops.py`、`multirate_wam.py`、`fast_conditioner.py`、`fast_observation_encoder.py`、`causal_world_bridge.py`、`flow_schedule.py` | 旧 aligned 回归通过；纯 fast 模型无 WAN 依赖；两步桥可微 |
| 数据 | `training/data/paired_world_action_dataset.py`、独立 manifest builder；v2 reader 的只读索引接口 | 固定 episode split、语言一致、paired sampler 可恢复 |
| 训练 | `training/train/multirate_trainer.py`、`multirate_objective.py`；提取 `video_action_trainer.py`/`common.py` 的公共循环设施 | warmup/joint 两阶段均通过实际 Accelerate/DeepSpeed 路径 |
| 模型迁移 | 新 `training/migrate_multirate_checkpoint.py`、export 逻辑、checkpoint verifier | 原键严格验证，新键明确初始化，新格式可完整恢复 |
| 配置 | `config/schema.py`、`loader.py`、新 profile/schema、`scripts/train_multirate.py` | snapshot/hash 包含所有训练和部署超参，旧 profile 保持可加载 |
| 部署加载 | 新 `inference/multirate_model_loader.py`、runtime 类型；提取共用 attention metadata | 完整模型和 fast-only 视图输出对照通过 |
| 运行时 | `runtime/ciwam/wam/{multirate_policy,fast_slow_workers,bridge_store}.py`、新 DOMINO adapter | controlled_ratio 与 independent_worker 都能按定义运行 |
| 注册和评测 | adapter 注册/export 路径、benchmark resolved config、验证脚本 | 实际入口能选择新 policy，不只存在孤立类 |

新增文件名是实施建议，允许组织成等价模块；不能省掉对应责任。尤其原 `scripts/train.py`、`config/loader.py`、`inference/model_loader.py` 的严格边界必须有明确分派方案。

### 16.2 验收脚本与通过条件

建议新增以下脚本，先以 synthetic/tiny tensors 验证控制和梯度，再按显存预算读取实际权重。命令行应提供 `--device cpu|cuda` 的适用范围；不是所有 WAN/FlashAttention 检查都能在 CPU 跑。

| 测试 | 关键断言 |
|---|---|
| `verify_multirate_schedule.py` | source action 用十步日程前两项；设备端 Euler 与原 solver 逐步数值/梯度一致；cache 取点无偏移 |
| `verify_causal_bridge.py` | 固定 source/noise，变更 source 未来标签、t 的观测/动作和正则样本均不改变 B；和旧完整 sampler 返回的逐层 cache 对齐 |
| `verify_fast_conditioner.py` | 同条件零 residual 时 velocity 与旧 cached forward 对齐；首次仅 W_O 梯度可非零，更新后 encoder 获梯度；25 token/17 core 不变 |
| `verify_paired_data.py` | 直接抽查 decoded RGB、s/t 状态、首动作索引、motion mask、共同 lang_id；split、时间、越界和图片布局失败均能报错 |
| `verify_multirate_training.py` | 单卡微批与等效累积的更新接近；joint fast-only loss 到源 video/action 的预期依赖成立；不要求无依赖末端参数非零 |
| `verify_multirate_distributed.py` | 同一小数据用多 rank 实际 DeepSpeed 后端跑多次累积、save/resume；参数组不重叠，rank 间权重一致，梯度不被多除 |
| `verify_multirate_checkpoint.py` | 迁移缺失键 allowlist、strict reload、训练/部署包装器一致；fast-only 不含 WAN/VAE/T5，输出一致 |
| `verify_multirate_runtime.py` | 受控 R 计数、H_exec=4 而非16、slow 延迟/乱序/reset/池耗尽、CPU release 早于 GPU done、异常传播均按协议工作 |
| `profile_multirate.py` | 分离预处理/flow/VAE/slow/fast/传输耗时，记录当前视觉只编码一次，slow 开关下 fast p50/p95/p99 和 GPU trace |

对 bf16 比较使用预先约定的绝对/相对误差并报告最大误差；不能用一个过宽阈值掩盖不同 schedule，也不能要求所有重构逐 bit 相同。缓存等价性测试固定 eval mode、随机噪声、精度和 attention backend。

多卡 smoke 是实现后的测试任务，本次文档修改不会启动。先 1 卡/少量 rank 发现错误，最后验证预期多卡配置；没有通过 tiny run 的代码不进入长训练。

### 16.3 实施里程碑及停止条件

1. **M0：契约和基线**。记录代码/权重/配置指纹，跑可复现的旧 sampler，建立时延分解与输入图像样本。
2. **M1：共同模型底座**。纯 block 重构、source 两步日程、cache 等价性、fast-only attention；先不训练。
3. **M2：当前条件和配对数据**。RGB/文本/age residual、null bridge、paired manifest 与完整 dataloader；完成单次图/梯度测试。
4. **M3：训练与格式闭环**。新 trainer/profile/初始化迁移/export/load/resume，一次 wrapped forward；先短 warmup 再 joint。
5. **M4：控制实验入口**。真实 DOMINO Level-1 adapter、固定 H_exec 和 R、配对评测与全部身份日志。
6. **M5：独立运行系统**。worker、有限 buffer、正确事件回收和 fast-only runtime；synthetic 延迟测试与实际 profile 两者都通过。
7. **M6：机制与性能评估**。运行 §8 主矩阵；至少 A(H=4)/C/E/F 的同协议对照，预测归因包含 G；达不到指标则如实否定对应假设。

M4/M5 是同一个完整方案的两种模式，不能在 M4 结束后把 M5 标为“以后可选”。质量门槛控制大规模实验投入，不豁免并行实现的正确性验证。成功率提升、计算节约、并行尾延迟改善是三项独立结果，分别报告。

## 17. 实施时必须逐项处理的问题

| 风险/错误 | 必须采取的措施 |
|---|---|
| 把两步 source action 当新的二步日程 | 使用原十步 action 日程的前缀；独立记录 step index、sigma 与模型 timestep |
| 把 detach 当防泄漏 | source 输入类型排除未来标签；做标签/当前快观测替换不变性测试 |
| 把 `forward_fast` 当采样 | 预测与 sampler 分开命名；部署必须十步更新、反归一化后才提交 |
| checkpoint 重算时重复 append cache | block 为纯张量返回；在重算函数外组装 cache tuple |
| 零投影与零门控同时存在 | 只零初始化输出投影；分首步/更新后两阶段测梯度 |
| 只加 view embedding 没有空间坐标 | 新视觉采用二维位置与 view 信息，旧 WAN RoPE 不修改 |
| age/state 小模块没进 optimizer/export | 新参数统一 namespace，全集/交集检查与严格 reload |
| frozen video 导致 step 0 无 video_pred | 源采样显式要求视频输出；冻结与是否计算预测头分开 |
| 源和快共享模块但可变 runtime 也共享 | 训练共享权重；部署隔离副本、scheduler、RNG、缓存和 text state |
| v2 同 episode 的语言表述不同 | pair 只选一个 lang_id，源/快/正则共用 |
| R=4 的稳态 age 写成 0/16 | 明确实际消费龄0/4/8/12；延迟尾部另外采样与计量 |
| 数据帧差直接当线上动作数/秒 | frame/action/sim/wall 四种量分字段，禁止隐式换算 |
| 在线直接用高分辨率腕图，训练用半分辨率 | 共享 canonical view 处理和插值；原 VAE padding 画布不参与快裁切 |
| padded K/V 加 k_lens 后丢掉 action token | 首版固定 world 长度或整批 null 分支；变长时实现正确 packed 序列 |
| 训练 no_grad 与 inference_mode 混用 | warmup 用 no_grad；joint 可微；部署 inference_mode 在线程内显式启用 |
| DeepSpeed 路径被普通 `.backward()` 绕过 | 单一 wrapped forward + accelerator.backward；按真实后端测累积/save/resume |
| refcount=0 就覆写已发布槽 | current READY 槽不可写；旧槽 retired、引用释放且 GPU done 后才回收 |
| `record_stream` 被当同步 | producer ready event + consumer done event；不做全卡同步阻塞 fast |
| 所有 slow 结果照单全收 | generation/model/instruction/source_id 检查；丢弃过时结果和旧 action |
| query 创建16条、只等 pending 为空 | 第 H_exec 条完成即重规划，尾部处理明确 |
| null-bridge 没训练就上 fallback | 纳入训练/dropout与闭环评估，并单列 fallback 发生率 |
| 预计缓存/显存成本凭口头估算 | 实测 N_v、存活版本、设备传输、activation 峰值与 kernel 争用 |
| A 组只保留 H_exec=16 | A(H=4) 是主公平基线；A(H=16) 仅历史参考 |
| 改 source prefix 后仍用旧成本公式 | 分开 C_base、C_bridge、C_fast；测真实摊销和吞吐，不把工作量当并行时延 |
| 推理每轮仍构建完整模型 | fast-only export/runtime 通过模块与显存检查，否则资源分离未完成 |

完整实现的主要风险不是“改动是否足够小”，而是训练所见条件、在线时钟、模型导出与并行读取是否真的描述同一个策略。优先消除这些跨模块不一致，之后再做局部 kernel 或缓存压缩优化。

## 18. 对 2026-09-28 初步实现的审查与后续重构决定

完整证据、CPU 复现、成本推导和实验矩阵见 [multirate_review_20260928.md](multirate_review_20260928.md)。这份审查优先于将早期计划误读为“当前已完成”的说法。

### 18.1 已经实现及必须修复的内容

当前完成了固定 H/R 的跨周期复用、源随机噪声因果桥、当前 RGB/motion/state、单一 wrapped training forward 和单独 checkpoint 格式。尚未完成独立 worker、真正可执行的 fast-only 模型、真实 sampler cache 等价性、真实分布式梯度及闭环质量/速度评测。

| 优先级 | 当前问题 | 下一步 |
|---|---|---|
| P0 | 新 FastConditioner 部署未 eval；CPU 已复现 BN 推理更新 | 真实部署工厂统一 eval/冻结，验证持久 buffer 不变 |
| P0 | residual 的 FP32 activation / bf16 parameter 边界未定义 | 明确逐算子 dtype，真实 CUDA/DeepSpeed tiny 验证；目前只有 CPU 边界复现 |
| P0 | source/fast schedule 默认构建，未贯穿 config；部署未完整核验行为身份 | checkpoint 保存并严格构建 schedule、normalizer、preprocess、age support |
| P1 | 默认 H4/R4 对原 H16 不减少世界调用，快调用变四次 | 先测 S/B/F；分别比较 H4 与 H16，不用 R 直接声称加速 |
| P1 | 随机两层视觉 stem、12层×10步 residual 有显著额外成本 | 将其作为 V1 参照，按预算构造 V2 |
| P1 | clean/randomized 不是 held-out episode 划分；stress 无独立权重；无时钟假定30Hz | 显式 manifest、分层采样、真实时钟或一致的 action-age-only 模式 |
| P1 | fast-only 只有 ModuleDict/权重，没有独立 sampler | 抽出 ActionAttentionSpec/action_core_ops 并运行真实前向 |
| P1 | BridgeStore 旧槽回收顺序、跨 generation 发布错误，均已 CPU 复现 | 修复后再接线程/event；当前串行 adapter 未使用它 |
| P1 | toy cache 与 video 参数无依赖，不能证明 fast loss 到 video | 真实骨干仅 fast loss 的联合梯度检查 |

### 18.2 最终快网络不受“尽量小改”限制

保留因果慢桥与明确跨时数据合同。V2 的候选结构为：当前紧凑视觉编码 → 一次/query 的观测—世界融合 → 固定小 memory/调制条件 → 独立紧凑 Action Model。小 memory 会减少原先每层按 noisy action 动态读世界的能力，是需实测的取舍，不是等价重构。

先抽取原 action 算子做数值参照，再根据 profile 缩减 decoder 深度/宽度；学生不必与源 Action Expert 共享全部参数。专家 FM 是主监督，可用可部署 WAM teacher 的向量场蒸馏帮助迁移。bridge 压缩、执行历史条件和少步蒸馏均需有针对性的证据再加入；主分频对照保持 video2/action10，少步提速单列。

当前观察不是对任意候选动作的世界重预测。源 proposal 与实际执行偏离、接触切换和方向变化都可能使旧桥失配；age/q差只是线索，不能当作误差保证。R1 能力对齐后才检验 R4，current-only 必须单独训练，不能只靠测试时删桥证明世界预测价值。

### 18.3 实施顺序

1. 修 V1 的确定错误和数据/身份合同，冻结一个可复现参照。
2. 验证真实 cache、因果性、仅 fast loss 的梯度；再做 S/B/F profile。
3. 对齐 age0 的闭环能力，确定跨龄复用的质量曲线。
4. 按真实快预算实现 V2，比较相同 H/数据/精度/步数下的速度—质量。
5. 修缓存生命周期并接独立 worker，报告单卡争用和尾延迟；双卡结果标明资源。

这一顺序不等于允许把并行永久省略。它避免在快网络本身既昂贵又质量未对齐时，把大量工作投入线程调度并误以为解决了架构问题。达不到世界表示增益或速度预算时，应允许简化或放弃该路线。
