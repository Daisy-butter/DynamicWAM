# Causal spatial WAM：训练配方与验收边界，2026-10-21

本说明对应独立候选 `causal_spatial_v4`，配置以
`configs/causal_spatial/{interface_adapt,policy_pretrain,joint}.yaml` 为准。
训练入口是 `scripts/train_causal_spatial.py`，目标与数据准备在
`src/dynamicwam/training/train/causal_spatial_trainer.py`。这是实现说明；运行是否通过，
应读取对应阶段的独立 smoke、导出审计和清洁退出收据。

## 1. 初始化与结构

学生从已审计的 `absolute_motion_init.pt` 初始化：Stage1 Video、wo-motion Action、
新初始化的 native motion 模块。其 SHA256 固定为：

```
601936c61503d173c38d7c9ae2ce956dfa22825e4b5c4e1096424f9586df540d
```

路径为 `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/checkpoint_init/absolute_motion_init.pt`。
每次正式初始化都验证该哈希，并对原生 backbone 严格加载。旧 multirate 权重和
四步 diagnostic 权重不是正式初始化来源。

冻结教师独立加载
`/SSD_DISK/users/wuruihan/DynamicWAM/external/checkpoints/DynamicWAM_full.pt`，
保持 `eval()`、`requires_grad=False`。教师的训练时内容身份保存为
`initialization.teacher_sha256`；导出审计要求字段存在，并重新计算实际文件哈希。
教师和学生动作 mean/std 必须一致，动作归一化在 FP32 中完成。

模型接口：

- 慢 memory `M`：原生 2048 维 Video backbone 只读取已观测 condition latent 和文本，
  不读取未执行候选动作或未来 latent。每层输出完整观察 K/V，后台只发布已完成的结果。
- 当前 belief `b`：三个视角的空间 RGB、测得 flow/dt、有效性和有限历史 capture commands，
  经 192 维空间状态预测、观测校正，形成每视角 8×8、合计 192 个空间 token。
  一个请求只构造一次 b 和各层投影 K/V；它们在 K10 内固定。
- 未来 `Z`：768 维 residual、FFN ratio 2、每个原生 Action 层对应一个轻量未来 block。
  原生 Action residual 也是 768 维，原有投影把两路接到共同的 16×128=2048 注意力接口。
  Query 为 `[Z,A]`，Key/Value 为 `[M,b,Z,A]`，每层都有 Z↔A 交互。
  每个新控制请求使用新的 Action/Z 噪声，并共同积分十步；不跨请求冻结旧的
  action-conditioned future K/V。
- 短重叠候选计划仅在 rows 2:4 进入 Action/Z 的 gated residual；不进入 M、b 或 b K/V。
  初始 gate 为零，无效计划与范围外行严格无效，不额外加入改变 softmax 分母的空 token。

本轮实际 full 模型构造统计约 **1.108B** 参数：原生 Video 约 849M、Action 约 139M、
belief 497,226、未来模块约 104M，其余为运动模块、投影与编码器等。
这是模型规模统计，不是 FLOPs、峰值显存或加速比。fast checkpoint 去掉原生 Video 参数，
仍需外部提供已经完成的 M；full checkpoint 保留原生联合支路供刷新和训练约束使用。

## 2. 三阶段预算

三个阶段均使用 **8 GPU × microbatch 4 × accumulation 4 = global batch 128**，
DeepSpeed stage 0、BF16、AdamW weight decay 0.001、全局梯度裁剪 1.0，
开启新模型 gradient checkpointing。所有新模块使用表中的 `new` 学习率；motion 与原生
Action 共用 `action` 学习率。

| 阶段 | 更新步数 | 原生 Video LR | 原生 Action/motion | 新模块 LR | 慢 memory 数据 |
|---|---:|---:|---|---:|---|
| interface_adapt | 2,000 | 1e-6 | 参数冻结，LR 0 | 1e-4 | 强制当前观测，age=0 |
| policy_pretrain | 30,000 | 1e-6 | 解冻，LR 1e-5 | 3e-5 | 已配置的历史 source/age 采样 |
| joint | 20,000 | 2e-6 | 解冻，LR 5e-6 | 3e-5 | 同上 |

总预算为 52,000 次 optimizer update。每阶段 step 从 0 开始；30k 和 20k 是各自预算，
不是累计终点。第二阶段加载第一阶段 step 2000，第三阶段加载第二阶段 step 30000。
阶段切换使用新 optimizer 和新 scheduler。

每阶段 cosine scheduler 的 warmup 为 200 步，最低学习率为各参数组初始 LR 的 0.01。
训练器在每个 microbatch 的 backward 前设置当前 optimizer update 对应的 LR，兼容
DeepSpeed 在 backward 内完成更新。首步使用 `base_lr/200`；scheduler 的 step_count
仍在成功更新后递增。四步 smoke 的 warmup 为零，因此直接验证 cosine 配方。
该顺序沿用本地 `ClampedCosineScheduler`，若要调整必须作为独立配方变更重新验证。

正式配置每 10 步记录指标，每 500 步验证 4 个 batch，每 2000 步保存；终点额外执行验证、
最终导出及完整状态保存。对应三个 `_smoke.yaml` 均为实际八 rank、相同 microbatch 和
累积、四次更新、warmup 0、每步日志、终点 1 个验证 batch；均从审计初始化独立起步。
smoke 验证执行路径和资源，不足以证明收敛。

## 3. 损失、梯度与可解释范围

所有阶段当前权重一致：

| 项 | 权重 | 目标及作用 |
|---|---:|---|
| Action FM | 1 | 专家动作后缀速度；H2 clean prefix 不计误差 |
| Future FM | 0.1 | 实际缓存未来 VAE latent 的速度，约束轻量 Z 的预测语义 |
| Belief Gaussian NLL | 0.05 | 先预测、再看当前图像；监督真实 RGB/flow 分量，不使用模型自身预测作 target |
| Native Video FM anchor | 0.05 | 同一学生原生联合支路上的未来速度损失 |
| Native teacher action velocity | 0.05 | 相同 noisy A/Z 与 timestep 下，冻结教师的动作速度，仍只约束后缀 |
| Short overlap continuity | 0.01 | 对有效短重叠、且与专家相符的候选计划，约束预测 clean endpoint |

训练采样 `x_sigma=(1-sigma)x_clean+sigma*noise`，速度目标 `noise-x_clean`。
Action 与未来 latent 使用相同采样 sigma/timestep，flow schedule shift=5、1000 个训练时刻。
部署从噪声向 clean 方向共同 Euler 积分 K10。continuity 使用
`noisy_action - sigma * predicted_velocity` 恢复 clean endpoint；只有计划与专家归一化动作
的逐行平均平方误差小于 0.04、且非 committed prefix 时生效，并按离 row 2 的距离衰减。
此专家一致性只控制训练 loss 权重，不作为部署输入。

belief 输出五个观测分量（RGB3、flow velocity2）的预测与 log variance。
损失中的 log variance 截断在 [-4,4]；同时报告普通 MSE，避免把放大方差造成的 NLL
下降误读为预测改善。RGB 与 flow 目前按分量同权，其数值尺度不同，外观与运动贡献应
分别检查。belief 是空间预测/校正表示，不能据此宣称显式 3D 状态、物体身份跟踪或遮挡恢复。

新 M 路径和 native anchor 共用学生 Video 参数；后者也可能通过原生 joint attention 向
已解冻 Action/motion 传递梯度。冻结教师始终不更新。native Video anchor 与单次动作
速度蒸馏约束训练函数，但**不证明保留原生 2+8 采样的闭环行为或成功率**；这需要独立
原生对照和完整任务评测。

## 4. 时间、观察与候选计划

固定接口为 action dt=0.16 s、H2/R2、chunk16、K10、simulation_rate=1。
两步承诺提供 0.32 s 的名义窗口，完整预测覆盖 2.56 s；相机、预处理、排队、推理和安装
都消耗真实窗口。截止时间不随相机返回或结果消费时间后移。

空间窗口 T=4，按当前观测往前取 0、0.32、0.64、0.96 s 处最后可用的真实 capture。
启动历史不足时重复最早真实图像，但零 dt 的 flow 与命令转移无效。flow 是选中图像对的
Farneback displacement，带独立 forward/backward 一致性和 padding 有效性；RGB 有效
不代表 flow 有效。模型再除真实 dt 和图像宽、高，得到归一化速度。

命令 row i 是前一选中 capture 的 native drive target，与到当前 row i 的 dt 对齐。
模型不再额外 shift。它不是实测 qpos，也不是区间内完整执行控制轨迹。部署端必须在曝光
时采样该 target，不用相机完成后的状态替代。

condition latent 为 `[48,2,24,20]`，未来监督 latent 为 `[48,2,12,10]`。
Z 的时间输入使用请求时已知的目标网格 endpoints 8、16（标准相位对应 1.28、2.56 s）。
实际未来 capture support end 只用于标签审计，允许的落后不超过 0.16 s，不泄漏为推理输入。
后两阶段 source age 支持 0..31 个动作间隔，上界 4.96 s；runtime 每 16 个动作尝试刷新 M，
消费最新完成版本，并在安装时再次核对 source age。慢刷新完成不覆盖当前 b。

训练的 H2 prefix 来自专家示范。soft plan 从冻结原生教师在上一快查询的已观测条件上，
以独立噪声通过原生 2 joint + 8 cached-action 采样产生；该 source 为当前时刻减 0.32 s
之前最后合格 capture，不复用可能已经过期的旧慢 memory source。计划按绝对目标时间
对齐，手臂线性插值、夹爪 previous-value hold，超出覆盖范围不外推。

`teacher_plan_interval=2`：训练仅在偶数 optimizer step 对应的累积微批生成计划，再对每
个样本以 20% 概率移除；因此稳态约 40% 训练样本有候选上下文，实际还受有效覆盖限制。
验证每个 batch 都尝试生成且不做这项随机移除。仅 rows 2:4 可保留，计划冻结在一次 K10
轨迹内。

部署短重叠来自已安装策略队列；训练来自教师在离线专家状态上的候选。这是明确存在的
分布差异，`on_policy_ready=false`。没有把失败策略动作作为成功 BC target，也没有证明
反事实环境响应或真实在线 replay 的覆盖。该差异需要后续 replay/闭环实验量化。

## 5. 数据身份、保存与阶段切换

数据根目录：

- packed：`/SSD_DISK/users/wuruihan/DynamicWAM/data/packed/domino_absolute_motion_v2`
- 原始转换：`/SSD_DISK/users/wuruihan/DynamicWAM/data/converted/domino_exact_time`
- physical grid：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/data/joint_motionkv_physical_grid_20261015`
- future cache：上述目录的 `future_latents/`

使用 clean/randomized、固定 episode partition seed 20260928、validation fraction 0.1、
exact physical grid 和完整原生 VAE cache；不允许旧 future latent fallback。
动作与运动尺度沿用已审计初始化/packed 数据的固定尺度，不声称重新拟合了无污染的
train-only 统计。数据、物理网格、划分、空间 pairing 和统计身份进入 checkpoint，阶段间须一致。

每次 checkpoint 包含 full 与 fast 导出、recipe、native/model config、runtime contract、
data identity、初始权重与教师 SHA。`latest_export.json` 绑定最终路径、step、哈希。
另外 `state/step_N/` 保存 DeepSpeed 模型/optimizer、scheduler、八 rank RNG、
`trainer_state.json` 和逐文件哈希 manifest。

`continue_from` 是阶段间 weights-only 初始化，重置 optimizer/scheduler，且禁止 diagnostic
checkpoint。完整状态保留是为恢复准备；当前训练入口没有已验收的 `--resume` 恢复流程，
不能把保存成功描述成已证明精确续训或数据游标恢复。

`scripts/check_causal_spatial_export.py --config … --output …` 在 CPU 上调用实际 full/fast
loader，逐项检查 shape/dtype/value/finite，验证 fast 恰好去掉 Video；绑定初始化和教师
SHA、数据与归一化身份；要求八个 rank 都完成预期更新，所有 loss/grad 检查覆盖完整、
zero OOM、有效梯度证据和有限终点验证；校验全部保存状态哈希、八 rank RNG、scheduler
内容及 DeepSpeed 模型与导出逐值一致。该审计不执行 GPU forward 或 optimizer 恢复，
worker/guard FD 清洁退出由独立 supervisor gate 负责。

## 6. 三轮审查的关键修正

1. **真实接口与计算预算**：保留原生 2048 注意力/768 Action、state/motion/register；
   未来支路用 768 residual，未把旧混合 Video K/V 宣称为可切分观察 memory。
   full 与 fast 均用真实结构严格加载；BF16 原生观察归一化改为显式 FP32 affine 运算。
2. **因果与训练语义**：统一 T4/0.32 s 真实 capture 选择，修正 interval-left 命令二次 shift，
   分离 RGB/flow mask；未来实际 capture 时间只作标签 metadata；teacher proposal 改用
   上一快查询 source；移除 proposal→b 路径，只在 Action/Z 条件化。
3. **部署与验收**：独立 H2/R2 控制器显式 reserve 2，M-only bootstrap 后保持 native drive，
   曝光/动作网格/迟到拒绝与报告互相校验。GPU Graph 首轮定位到 flow 归一化内部
   `new_tensor([w,h])` 的捕获期 H2D，已在本地改成设备上分通道除宽/高；下一份实际
   Graph 收据才决定该修复是否通过。

CPU 小模型、参数统计和单独采样器通过分别只支持其测试范围。正式阶段需真实八 rank
四步 smoke、严格导出和清洁退出通过；宣称实际 1x、加速或任务收益仍需完整相机、IPC、
后台刷新、连续物理执行和原生视频/报告的独立验收。
