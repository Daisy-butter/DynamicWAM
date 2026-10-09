## 11. 实现范围与必须固定的架构决定

这里以完整的可训练、可导出、可并行部署的系统为交付单位。只给现有方法多加几个参数、只改 `choose_path()`、或只跑通一个 forward，均不算实现完成。可以分里程碑验证，但模型、数据、训练、配置、checkpoint、运行时和评测必须形成闭环。

本节及后续内容细化前面的设计，以下决定应在编码前落实：

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

建议新增 `models/multirate_wam.py`，实现 `MultirateWAMActionModel`，继承/组合现有共享计算函数，沿用旧参数名 `compact_wan.*`、`action_expert.*`、`absolute_motion_tokens.*`；新参数统一归到 `fast_conditioner.*`。不要把同一个 Action Expert 同时注册到两个父路径，制造重复 state_dict 键或 optimizer 重复参数。

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

ResNet-18 的 BatchNorm 需专门处理：首版固定预训练 running statistics，保持 encoder 中 BN 为 eval（父模型 `.train()` 后也要恢复），允许其它权重学习；否则 microbatch=1、不同相机混 batch、激活重算会引入额外统计偏移。将这一策略和预训练权重 hash 写入 checkpoint。

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

