# DynamicWAM 分频实现审查与重构建议

审查日期：2026-09-28。对象：`/home/wuruihan/DynamicWAM` 中的工作区实现及 `multirate_implementation.md`，基于 `original-baseline` / `70a0c89a82f562a6f9565f846c9737da58b326a7` 上尚未全部提交的改动。以下判断来自实际调用链，不仅来自实现说明。路径均相对仓库根目录。

本次只检查代码、补充设计文档并运行小型 CPU 诊断，没有启动 GPU 推理、训练或 DOMINO 评测。数值诊断使用实际模块的小维度实例；它们不代表完整模型的速度或成功率。

## 1. 结论：有真实跨周期分频，但尚不是达到目标的快慢系统

**已经实现了固定比例、串行调度的跨控制周期分频。** 这比同一次 query 内的 video 两步/action 十步前进了一步：默认 `H_exec=4,R=4` 时，四次读取新观测、重新生成动作，共享一次历史世界 K/V。每个快 query 都读取当前三视角、当前运动和本体状态。

但是，**当前没有实现独立并行运行，也没有证据证明相对原版 H=16 的策略会提速或保住性能。** 默认配置甚至没有降低“每执行16个动作”的世界刷新次数，却大幅增加动作推理。现在最应先解决的是快分支计算预算与训练/部署正确性，而不是马上加线程或者大规模训练。

| 判定对象 | 当前状态 | 证据 |
|---|---|---|
| 多轮真实观测间复用世界表示 | 已实现 | `multirate_control.py` 的固定 R；`multirate_policy.py::query` 仅刷新轮建桥 |
| 快分支根据当前视觉纠偏 | 结构已接入，效果未验证 | `FastConditioner.prepare_memory` + 当前 RGB/motion/state |
| 不使用专家未来构造历史桥 | 已按因果方式实现 | `CausalWorldBridge` 使用源条件和随机噪声；真实未来另用于 FM 正则 |
| 快损失端到端训练慢表示 | 代码意图成立，真实骨干梯度尚待验证 | joint 设置 `detach_cache=False`；现有 toy 测试不能证明该梯度路径 |
| 慢计算不阻塞快计算 | 未实现 | 同一 `query` 中先 `_publish_bridge`，再 `_sample_fast` |
| 不加载 Video DiT 的独立快推理 | 未实现 | fast-only 仅导出/构造模块，真正前向仍依赖完整 SmallWAM 对象 |
| 明显加速且不掉点 | 未验证，默认比较口径尤其危险 | 没有真实 GPU profile / 配对闭环结果 |

不应把“没有并行”误说成“完全没有分频”；也不应把“确实复用了世界计算”直接等同于“整体一定更快”。

## 2. 它实际上如何运行

主要调用链：

```text
DOMINO deploy_policy_multirate_flow.eval
  → 当前观测、HeadFlowBuffer.push、ControlledMultirateController
  → 到 H_exec 后 MultiratePolicy.query
      → 若 query_id % R == 0：
          VAE 当前条件 + 随机未来/源动作
          → 联合前向 step 0 → Euler 更新
          → 联合前向 step 1 → 发布各层 pre-attention video K/V
      → 当前 RGB/motion/q + bridge age
          → FastConditioner.prepare_memory，一次/query
          → sample_fast_actions，十次 action velocity forward
          → 反归一化，提交前 H_exec 条
  → 当前回调执行一条动作，完成后计数
```

默认每16个已执行动作对应：

| 快查询 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 当前执行位置 | 0 | 4 | 8 | 12 |
| 是否刷新世界 | 是 | 否 | 否 | 否 |
| 使用桥的动作步龄 | 0 | 4 | 8 | 12 |
| 快动作去噪步数 | 10 | 10 | 10 | 10 |

CPU 控制器诊断得到了上述实际计数。第16个动作之后开启下一轮慢刷新。

这里有两点做得正确，重构时应保留：

1. bridge 来自原 sampler 的两步联合前缀，后八个源 action-only 步不再修改它，因而可以不运行。它不是重新发明的“只训练两步视频模型”。
2. 因果桥与有真实未来标签的正则图分开。源 `future_latent` 只用于确定随机张量形状，实际数值被 `randn` 替代；不能把这一点误判为已经发生数值标签泄漏。

## 3. 最重要的速度问题：默认世界刷新并未比原版更少

### 3.1 必须用每个已执行动作的成本比较

设：

- `J`：一次联合 video/action 前向；
- `A`：一次原 action-only 去噪前向；
- `P`：一次源条件 VAE 等处理；
- `E`：一整个快 query 新增的当前视觉、条件投影与十轮 residual 开销。

保留同样两步 video、十步 action 的条件下，原版完整 query 近似为：

\[
S\approx P+2J+8A.
\]

新桥构造与快 query 分别近似为：

\[
B\approx P+2J,\qquad F\approx10A+E.
\]

默认 `H=4,R=4`，每16个执行动作：

\[
C_{new,16}\approx B+4F=P+2J+40A+4E.
\]

而原版 `H=16`：

\[
C_{old,16}\approx S=P+2J+8A.
\]

**近似多出了 `32A+4E`，世界计算仍是两次联合前向。** 若把联合前向内的 action 也计数，原版总共10次 action 层堆栈，新版是2次 source + 4×10次 fast，即42次。

这些是同类算子的工作量分解，不是实测毫秒；共享预处理、内核与数据搬运需单独测量。但它足以否定“R=4，所以相比原版总体应该大幅提速”的直接推论。

### 3.2 相同 H=4 时，分频才有清楚的计算节约空间

若为了更及时观察动态物体，原版也每4个动作执行一次完整 query，则每16个动作成本为 `4S`。新版为 `B+4F`。这是正确的主要架构对照。

对相同执行长度，平均每次快决策成本为 `F+B/R`，节约条件为：

\[
F+B/R<S.
\]

相对 H=16 原版，在默认 H=4,R=4 下真正降低总模型工作量，则需：

\[
B+4F<S.
\]

用上述近似展开，就是 `F<2A`。当前 F 本身含十次 A，显然不能靠移除重复世界计算实现这个目标。这里的 `2A` 只用于暴露预算矛盾，最终门槛应由实测 S/B/F 决定。

因此应明确目标：

- **目标一：以低于“每4步完整WAM”的成本，获得每4步重新观察的策略。** 当前路线有合理空间。
- **目标二：比“每16步完整WAM”还明显更快，同时保持更高反应频率。** 需要显著压缩快推理成本，或者把世界刷新间隔延长到超过16个动作；后者增加过时风险，不能单独依靠它。

### 3.3 周期性阻塞也会影响尾延迟

非刷新轮时延为 F，刷新轮为 B+F；四轮中一轮刷新，正常统计条件下 p95 仍会覆盖刷新轮。相对原版单次 S，刷新轮并不天然更快。

并行可以改变关键路径，但不能删除总工作量。单卡两条 stream 会争用计算、带宽和显存；双卡降低响应时延必须同时报告 GPU 数量及总资源成本。原生同步 DOMINO 中，wall-clock 加速也不自动转化为仿真目标少移动了一段时间。

## 4. 已确定的问题与尚待验证的风险

### P0-1：部署忘记将新增条件模块切到 eval，已复现

代码：`runtime/ciwam/wam/multirate_policy.py::setup`；`models/fast_observation_encoder.py::{__init__,train}`。

旧 runner 加载的 backbone 已经 eval，但新 `FastConditioner` 和新 wrapper 仍处于默认 training 状态。setup 加载权重后没有调用 `wrapper.eval()`。encoder 的 BN 固定逻辑只在显式调用 `.train()`/`.eval()` 时生效，构造函数并不会触发它。

训练器每步调用 `self.model.train()`，因此训练时 BN 被固定为 eval；部署恰恰没有这个调用。`torch.inference_mode()` 不会替代 `.eval()`；这是 PyTorch 明确区分的两种机制。[官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.inference_mode.html)

实际模块小维度 CPU 诊断：

- inference_mode 内 running mean 仍发生变化；
- 同一随机输入，默认状态与显式 eval 的视觉特征最大绝对差约1.351；
- 显式 eval 后 BN 状态正常。

此数值只证明特征路径有差异，不能换算成 SR 损失。

**处理：**在最终加载完成的部署工厂统一 `eval().requires_grad_(False)`；检查整棵模块树的训练状态；多次 inference 前后检查所有持久 buffer 不变。测试必须调用真实 setup 工厂，而不是手工创建一个已经 eval 的替代模型。

### P0-2：新 residual 的精度边界没有完整约束

代码：`small_wam.py::_action_attention_with_video_cache`、`forward_action_with_video_cache`；`fast_conditioner.py::FastResidualBlock.forward`。

原世界注意力在 bf16 路径中计算，但其调制残差显式涉及 float32；旧 FFN 在接收前主动转为权重 dtype。新增 residual 直接对 tokens 做 LayerNorm/Linear，没有等效的 dtype 契约。

部署新 conditioner 默认是 FP32，`.to(runner.device)` 不会把它变为 bf16；当前新 residual 也没有自己的混合精度上下文。训练的 DeepSpeed bf16 转换则可能把参数变为 bf16，暴露 activation/weight 不匹配。

CPU 上把真实 residual 的权重转成 bf16、输入保持 FP32，复现了 mixed dtype 错误。**这还不是对真实 CUDA/DeepSpeed 一定崩溃的证明**，外层 autocast 的实际行为仍需检查，但已经说明边界不能靠默认状态成立。

**处理：**规定 master parameter、投影、attention、归一化和残差累加各自 dtype；建立 fp32 与 bf16 真实路径测试。不能用全模块粗暴 `.float()` 解决后继续宣称低精度快推理。

### P0-3：配置、checkpoint 和部署并未形成同一个模型身份

代码：`MultirateWAM.__init__`；`MultirateTrainer._build_model/_checkpoint_config`；`MultiratePolicy.setup`。

- wrapper 未传 schedule 时固定使用 `BridgeSchedule.from_flow_match()` 默认值；训练随机 FM scheduler 却读取配置。当前默认 shift=5 恰好一致，修改配置可能使训练日程和 bridge/fast 采样日程分叉。
- 导出 config 未完整保存 source/fast schedule、配对划分、训练年龄支持、图像预处理和归一化身份。
- 训练初始化检查 dataset/motion 身份；部署先按旧 deploy config 建 runner，再加载新 checkpoint，只额外显式比较 age 两个 scale，不能保证所用 state/action normalizer、采样和预处理匹配。

strict state_dict 只能排除键/形状错误，不能排除这些行为错误。应从 checkpoint 构建 runtime spec，并对外部配置冲突明确拒绝；不要依靠旧配置刚好相同。

### P1-1：当前“轻量视觉”不是预训练骨干，而且有不小的重复成本

代码：`fast_observation_encoder.py`、`fast_conditioner.py`。

实际 encoder 是随机初始化的两层卷积 `3→64→768`，不是计划中的预训练 ResNet-18。160×160 三视角经两次 stride2 后产生40×40×768特征，再池化到4×4。训练中 BN running stats 又固定在初始值，不能把它描述成“固定预训练 BN 统计”。这不是必然错误，但它需要从机器人数据中重新学习外观特征，泛化与小目标定位均未被验证。

按默认尺寸、batch=1 的参数/乘加计数：

| 项目 | 计数 |
|---|---:|
| FastConditioner 总参数 | 32,542,016 |
| 其中视觉 encoder 参数 | 458,048 |
| 三视角两层卷积 | 约2.304 G MAC/query |
| 12层、25个 action-side token、十步 residual Q/O 投影 | 约3.539 G MAC/query |
| 每 query 的逐层当前条件 K/V 投影 | 约0.750 G MAC/query |

后面三项合计约6.593 G MAC；没有计入 attention、文本投影、norm 等，也没有包含原 Action Expert。按一乘一加计2 FLOP约13.19 GFLOP。**它们只是静态算量，不能替代 GPU 时延。** 参数少不意味着在高分辨率特征图上计算便宜。

文本虽然没有每次重跑 T5，但新 text pooling 仍每 query 重做，可在部署指令不变、模型固定时缓存。训练更新权重后不能复用旧的池化张量。

训练和部署计算出的 letterbox mask 均丢弃；池化 token 可能混入 padding。应以相同预处理和有效区域处理验证定位能力，而不只检查张量形状。

### P1-2：bridge 与实际快动作并不保持同一条预测轨迹

代码：`causal_world_bridge.py`、`MultirateWAM.sample_fast_actions`、`MultiratePolicy._sample_fast`。

源桥受源动作 proposal 影响；快动作却重新从独立噪声开始十步生成。它们不是同一条联合采样轨迹的自然延续。**即使 R=1、age=0，新结构也不等价于原 sampler。** 因而 R=1 的损失不能归罪于缓存过时。

当 t>s，接触、转向、夹爪动作和策略偏离会进一步使旧世界表示失配。现在只提供 age 和 `q_t-q_s`，没有显式已执行动作序列，也没有对旧预测与新观测做经过训练的状态更新。相同本体位移不能唯一确定中途接触及物体变化。

独立重采样还可能造成预测模式切换和 chunk 边界动作跳变。需要观察实际结果，不能断言必然发生；但较高重规划频率会增加检查其连续性的必要性。可用固定噪声/共用初始噪声作诊断，不能把更换噪声策略后产生的差异全部归给分频。

### P1-3：配对数据没有落实完整实验划分和年龄采样合同

代码：`PairedWorldActionDataset`、`build_pair_records`、`MultirateTrainer.__init__`、`configs/world_action_multirate_v1.yaml`。

- 当前 `splits=[clean,randomized]` 是数据域选择，不是 episode train/validation 划分。函数注释说“在 episode split 后构建”，但 reader 中没有实际落实这个划分。不能由此断言已经发生验证泄漏；应明确当前尚无这项防护及有效 held-out 验证。
- 把所有 source×offset 配对列出来再全局 shuffle，长 episode 和保留更多窗口的任务会有更大权重；这改变了原采样分布。
- `stress_offsets=[48]` 只写了标签，仍混入同一个训练集合；`offset_label` 未用于区别损失/采样。其权重与真实部署长尾无明确关系。
- 缺少 simulator clock 时静默使用 `offset/30` 秒，而在线使用真实 sim time。视频帧率不自动等于每条机器人动作消耗的仿真时间，尤其动作执行耗时变化时。
- 每个 rank 扫完整索引、构造 Python 配对列表，逐样本打开/seek/关闭 mp4；episode clock 的缓存缺少明确上限。这里只能认定存在启动、内存和 I/O 扩展风险，不能未经实测就称为必然 OOM。

**处理：**先固定 episode manifest，再生成可复用的配对索引；按任务/episode/年龄分层采样。缺时钟要么失败，要么明确训练/部署一致的“只有动作龄”模式。压力样本独立报告或显式限定占比，不伪装成在线稳态频率。

### P1-4：fast-only 文件不是 fast-only 可执行模型

代码：`inference/multirate_model_loader.py::build_fast_only_modules` 返回 ModuleDict；`small_wam.py::_action_attention_with_video_cache` 仍访问完整 WAN block 的 attention metadata。

文件确实不包含 Video DiT 权重，但目前不能靠它独立执行十步 sampler。应抽出 `ActionAttentionSpec` 和共享 `action_core_ops`，由独立 `FastActionModel` 实现真正 forward/sample。加载后检查没有 WAN/VAE/T5 模块，并实际生成动作。

这首先是依赖与显存隔离问题；**加载了完整 WAN 不等于非刷新轮还执行了 WAN 前向**，不能混淆这两类成本。

### P1-5：预留 BridgeStore 有两个已复现错误，但尚未影响当前串行策略

代码：`runtime/ciwam/wam/bridge_store.py::publish/_reclaim`。adapter 目前未使用此对象。

1. 发布新桥时先 reclaim 旧桥、后更新 ready_id。reclaim 因旧桥仍等于 ready_id 而拒绝回收；若旧消费者早已完成，没有后续事件再触发回收。capacity=2、无消费者时，连续两次发布后状态为 `[RETIRED, READY]`，第三次 claim 即失败。
2. 只有同 generation 才检查 source_id 顺序；上一 generation 的迟到结果会被接受并覆盖当前 READY。CPU 诊断中随后 acquire 当前 generation 报错。

并行实现需要 episode generation 屏障、锁保护的原子发布顺序及 producer/consumer CUDA event。仅维护布尔“完成”标志不能替代实际 GPU 完成检查。[CUDA Stream 官方语义](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.Stream.html)

### P1-6：现有测试不足以证明端到端耦合和可训练性

`verify_multirate_training.py` 使用假的 `_Accel`，且 `_Backbone` 输出的 cache 是不依赖 video 参数的零张量。joint 中 video 参数变化可以来自视频正则。因此该测试不能证明 `fast loss → bridge → video` 存在，更不能证明真实 Accelerate/DeepSpeed 的同步与 bf16 路径正常。

真实验收应关闭两个原 WAM 正则，只保留 fast loss；非 null bridge 样本下逐层检查源 video、源 action、中间 latent、桥 K/V 梯度，并做 detach 对照。warmup 无源梯度是其设计，不能用它替代 joint 验证。

activation checkpoint 尚未实现；joint 保留两次联合前向加快图和正则图，不能直接沿用旧训练 batch 假定显存足够。优先验证真实 microbatch=1，再确定资源预算；本次没有运行这些 GPU 检查。

## 5. 为什么有可能掉点：应拆成三个问题验证

### 5.1 新快路径能否在 age=0 复现原来的策略能力

新增视觉尚未成熟、重启的动作噪声、不同 sampler 条件、额外 residual 和 dtype 都会影响 age=0。先比较原模型 H=4 与新结构 R=1；否则直接比较 R=4 会把结构退化、训练问题和时间过时混在一起。

### 5.2 新观测能否纠正历史桥，而不是被历史桥牵着走

零初始化输出投影便于迁移，但开始时当前视觉不起作用；旧世界 attention 先运行，再叠加 current residual。后续网络可能学会纠偏，也可能过分依赖旧桥或完全忽略桥。低训练 loss 不能区分这三种情况。

必须同时看固定旧桥时新观测的作用，以及训练好的 current-only 策略。只在测试时把桥清零造成掉点，是分布外破坏，不能证明世界预测有用。

### 5.3 可复用的是哪种信息，而非“世界整体变化较慢”

任务意图、场景布局和阶段信息可能慢变；目标瞬时运动、接触和遮挡变化可以很快。当前 K/V 混合这些内容，分频其实假设“新观测通路能够修正旧预测中快速失效的部分”。这才是需要检验的研究假设。

从局部动力学看，旧预测与真实状态的差异至少取决于初始估计误差、实际动作与源 proposal 的偏差以及外部扰动；时间越长，这些差异有越多机会积累。即使在局部 Lipschitz 动力学下，误差界也会包含动作偏差和扰动的时间积分，而不是只由 age 决定。接触切换附近还不能直接使用同一平滑模型的界。

因此 age embedding 是条件变量，不是误差证书；固定 TTL 是实验控制，不是稳定性保证。应按转向/接触/抓持等事件和真实龄分桶测质量，而不是只报告平均 SR。

## 6. 建议：保留 V1 作参照，重构真正昂贵且欠约束的部分

不建议把现有工作全部丢掉；因果建桥、当前 motion、单一 wrapped forward、显式 checkpoint 格式值得保留。但若“明显加速”是主要目标，**不建议只修几个 bug 后继续堆逐层 residual，再以加 worker 收尾。** 应将当前实现定义为研究参照 V1，独立构造面向预算的 V2 快网络。

### 6.1 先修正确性，再冻结一份能比较的 V1

修复 eval、dtype、配置/normalizer 身份、时间与划分；真实骨干验证 cache 和梯度。此时不改变模型宽度、步数和 H/R，保留公平的结构参照。否则无法判断重构收益来自哪里。

### 6.2 V2：每次观测融合一次，动作去噪阶段尽量小

候选接口：

```python
current = current_encoder(rgb_t, motion_t, state_t, valid_masks)
memory = observation_world_fuser(
    current, historical_bridge, bridge_age, executed_action_history
)  # 一次/query；允许显式无桥模式
actions = compact_action_sampler(noise, memory, state_t, schedule)
```

具体决定：

1. **当前视觉使用有明确初始化来源的紧凑 encoder。** 对预训练小骨干和现有两层 stem 做相同输入、相同时延预算的比较；保持空间信息与三视角身份。避免过早扩到768通道后才把40×40池化到4×4。归一化可用明确冻结的预训练 BN 或适合小 batch 的其他设计，不能沿用当前隐含状态。
2. **昂贵的当前观测/历史世界融合尽量每 query 一次。** 首个候选为固定数量融合 token 和小型融合网络，再把 K/V 或调制参数交给动作采样。可以从32/64个 token、256/384维的预算网格选型，数字只是待 profile 的候选，不是已经证实最优配置。
3. **实现独立 `FastActionModel`。** 先抽取原 action 算子做数值等价参照，再以较浅/较窄 decoder 作为真正提速候选。源 WAM 可以保留原 Action Expert，学生不必为了参数兼容强制与它共享全部权重；二者的 checkpoint、优化器和部署身份必须明确。
4. **若 profile 显示逐层旧 K/V 读带宽或拼接占主要成本，再加入桥压缩。** 压缩由训练图中的明确模块产生；不先默认压缩一定必要。它会改变条件分布，要单独比较未压缩版本。
5. **执行历史作为可检验补充。** 若接触或转向分组显示 `q_t-q_s` 不足，再提供截至 t 的已执行动作及有效长度；这些是在线可得信息。暂不引入一套没有监督闭环的“信念网络”来增加复杂度。

关键取舍：原动作 query 随去噪状态动态读取世界；一次融合的小 memory 会减少这种交互能力。因此必须用速度—质量曲线选择，并保留 V1 作能力上界参照，不能把一次融合当作数学等价变换。

### 6.3 用已有 WAM 帮助迁移，不让学生只靠新数据重学

学生训练保留专家 action FM 监督，并可加入原 WAM teacher 的动作向量场蒸馏；在相同 noisy action、timestep 下比较，teacher 使用 t 时刻可获得的观测及因果采样，不能把真实未来作为 student 条件。

若 teacher 目标来自有真实未来加噪的训练前向，应明确它含额外训练信息，只能作为特权教师实验，不能把它和可部署教师混称同一对照。对动态任务优先使用可部署教师定义。

学生训练必须覆盖历史桥与当前观测的跨时差异；teacher 的在线策略本身也可能不够好，因此专家监督是主目标，蒸馏权重由 held-out 结果确定。后续若验证全梯度联合训练有收益，再开放慢 WAM；不要把所有新模块一次解冻后凭训练 loss 判断。

少步 action 蒸馏可以作为压缩 F 的额外手段，但这是与跨周期分频独立的优化。主分频对照先固定 video2/action10；少步结果另列，不能再把“2/10”或降低去噪步数当作分频本身。

### 6.4 并行化在预算和表示可用后接入

修正 BridgeStore 后实现独立的 slow producer / fast consumer；快模型不依赖完整 Video DiT。每次快查询固定一个 bridge snapshot，直到十步采样结束都不更换。源观测时间、GPU ready 时间和实际消费时间分别记录。

先做同卡 worker 开/关的受控 replay profile；如果相互争用使 fast p95 明显恶化，再比较分卡部署。报告相同硬件预算结果，不用双卡对单卡的时延差代替架构效率证据。本阶段仍不引入 RTC 动作前缀或执行—推理重叠语义。

## 7. 最小但能作决定的验证顺序

### 7.1 正确性门槛

| 检查 | 必须证明 |
|---|---|
| 部署工厂 | 全部 eval、没有推理期间 BN 状态更新、dtype 合同一致 |
| bridge 等价 | 固定真实条件/噪声/backend，比较真实原 sampler step1 的逐层 K/V 与新桥；检查 packed/在线条件 latent 一致性 |
| 因果性 | 更换源真实未来标签不改变桥；更换源已观测输入能够改变桥 |
| joint 梯度 | 只保留 fast loss 时真实 video 参数有梯度；detach 对照切断；未被零门永久屏蔽 |
| 数据 | episode 划分不交叉；当前RGB/flow/动作/时钟对齐；采样比例可复现 |
| runtime | 真实 fast-only 生成动作；reset/乱序/缓存耗尽与 GPU 完成生命周期正确 |

### 7.2 先测四个成本，决定是否值得继续

同 GPU、相同输入形状/精度、固定 warmup 次数和测量区间，测 S、B、F、F 在 slow 活跃时的分布。分解 VAE、video、原 action、新视觉/融合、文本、拷贝和 CPU 光流/预处理。CUDA event 用于设备分段，wall-clock 用于完整请求；不同用途不要混为同一指标。

同时测每16个**已执行**动作的模型累计耗时、每 episode 总 wall-clock 和峰值显存。隔离模型前向与环境物理步开销。若 `B+4F` 已明显大于 S，则不能声称超过 H16 原版吞吐；若 `B+4F<4S`，可以准确宣称在同 H4 下减少计算。

### 7.3 逐步闭环比较

固定 Level-1 episode manifest、每任务数量和随机 seed；先用少量任务筛查确定错误，再对关键结论做完整配对评估和多个训练 seed。

| 实验 | H / R | 用途 |
|---|---|---|
| 原版 WAM | H16 | 原始吞吐/质量参照，回答是否比现在使用的策略更快 |
| 原版 WAM | H4 | 相同重规划频率的公平架构基线 |
| 修正后 V1 | H4/R1 | 新快结构本身是否损伤能力 |
| 修正后 V1 | H4/R4 | 在同模型下，复用造成的质量/计算变化 |
| V2 | H4/R1、R4 | 小快网络的压缩代价及分频收益，先筛选再做完整评估 |
| 训练好的 current-only | H4 | 历史世界信息是否值得其计算成本 |

额外机制实验只优先做：跨时间训练 vs 仅 age0 训练；当前观测 vs 冻结观测；关键接触事件及 age 分桶。若要主张“预测未来”而非“更强历史编码”有效，再增加独立训练的无未来 token 慢表示对照。

不能因成功率点估计差1–2个百分点就宣称保住性能。先设非劣容忍界，再用配对置信区间判断；边界内证据不充分时继续采样，不把“不显著”当作“等价”。

## 8. 实施决定与停止条件

建议顺序是：**修正 V1 正确性 → 验证真实因果/梯度路径 → profile 明确快预算 → age0 能力对齐 → R>1 的复用质量 → 有针对性的 V2 快模型压缩 → 并行运行验收。**

以下任一结果都应改变路线：

- R1 已显著低于原版 H4：先修结构/迁移，不能用更大的 R 掩盖问题。
- R1 正常，R4 在接触/转向时持续退化：限制有效龄、增强当前纠偏；不要为速度继续延长缓存。
- 训练好的 current-only 与 WAM bridge 同样好：承认慢世界分支尚无额外价值，优先更简单策略，不能以缓存 attention 图充当收益证据。
- 当前 F 超出目标总预算：压缩动作与观测融合成本；单加 worker 无法证明计算效率提升。
- 同卡并行后 fast 尾延迟恶化：改变资源调度/模型规模或单列双卡实验，不声称线程存在就完成加速。

当前最准确的定位是：**一个方向合理、因果建桥较认真、已经跑通跨周期调度逻辑的研究原型；尚不是可以开始宣称显著加速且性能不降的完成架构。**
