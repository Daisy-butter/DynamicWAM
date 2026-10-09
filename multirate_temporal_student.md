# 保留 WAM 未来预测能力的高速动作路径

2026-09-29。此文记录本次最新实现，接续 [实现与实验审计](multirate_audit_20260929.md)。代码已加入当前工作目录；没有启动新训练或 GPU 评测。

## 1. 对 DuoCore 实验的判断

DuoCore 表 5 在 popcorn-scooping 的慢路径实验中报告：FAST 平均 81.12、最多 205 个 token，0.95 Hz、成功率 0%；RVQ-VAE 固定 36 个 token，3.27 Hz、成功率 83.3%。作者还指出分部位顺序预测带来的错误累积。因此该表不单独证明“缩短 token 序列必然提高成功率”。[原文 §4 / Table 5](https://arxiv.org/html/2512.20188v1#S4)

**应迁移的是在控制时间预算内保留行动所需信息的原则。** DynamicWAM 不是逐 token 自回归生成视频；它在每个 diffusion step 并行更新视频 latent。其 token 数影响每轮注意力、FFN、缓存带宽，作用机制不同。直接加 RVQ，或把所有未来时刻平均压成一个小向量，既不保证提速，也可能损伤运动与接触信息。

## 2. 旧未来为什么会过期

历史源 s 的世界预测覆盖名义动作区间 `(s,s+16]`。当前已执行到 t=s+12 时，桥里最多只剩 4 步名义未来；把它仍解释成“当前之后的完整 16 步未来”是错的。此外，机器人实际执行动作可能偏离产生该预测的 source 动作分支。前者是时间坐标问题，后者是 action-conditioned dynamics 的分布偏移；不能只用一个 age embedding 宣称两者都解决了。

现有 K/V 已经受 VAE 时域感受野、RoPE 和跨 token 联合注意力影响，不是带精确物理时间戳的独立物体状态。下面保留的是**预测时间段的结构与有效范围**，不是假装从 K/V 中恢复了精确几何轨迹。

## 3. 已实现：冻结原生 WAM + 时间分组 memory + 紧凑快策略

本次新增 `configs/world_action_multirate_temporal.yaml`，延用已有 compact 模型加载框架，显式启用 `temporal_world_memory: true`。旧 V1/V2 checkpoint 的默认结构不改变。

```text
历史观测 s -> 完整原生 WAM 两步 joint -> 原始视频 K/V
                                              |
当前三视角、motion、state -> 当前条件融合 ------+
                                              v
                       16 个条件槽 + 16 个未来段1槽 + 16 个未来段2槽
                       按当前年龄重定位；已过期段不再提供未来校正
                                              |
                       384 维 / 4 层快动作网络，完整 10 步采样
                                              |
                       预测16条，只执行训练约定的前4条，再观测
```

具体改动：

- `models/temporal_world_memory.py`：依据原生 `condition_seq_len / future_seq_len / future_grid_shape` 分组。当前数据的未来 latent 有两个时间平面；8 张未来视频帧、每视频帧相隔2个名义动作，分别形成约 `(0,8]`、`(8,16]` 两段。每组只读取本组世界 K/V；条件/历史槽与未来槽不混成一个无时间标签的池。
- `models/compact_fast_policy.py`：每次 query 先用最新观测构造 memory，再加入相对当前时刻的区间起点、终点、剩余覆盖比例和条件/未来类型。age12 时第一未来段过期，第二段剩4步。过期段的世界校正被置零，槽仍可承载当前观测信息；不从旧未来外推一个没有证据的新未来。
- 世界融合只在每次观测时做一次。后续10步动作网络读48个紧凑 memory token，不反复扫描12层完整世界 K/V。选择3个世界层仍是明确的信息压缩假设，需要消融确认。
- 推理约束 `age_actions + execute_horizon <= source forecast horizon`，并拒绝给新 temporal 学生放行未经训练的 H16。按动作索引检查名义覆盖，真实 Δsim/Δstate 仍作为输入；没有用固定 FPS 伪造未来的物理时间。

**这部分降低的是快路径成本，并没有缩短慢 Video DiT 的序列或更换 VAE。** 保留慢系统原始计算，避免尚未验证的世界模型压缩同时引入第二个性能变量。如果后续仍被慢路径限制，应单独研究空间分辨率、时间平面数、对象/接触信息压缩与重建/控制误差的关系，不能把“压缩桥”说成“慢视频已经变快”。

## 4. 性能保留不是口号：加入原生策略的因果蒸馏

只让 compact 学专家动作，可能学成普通 reactive policy 并忽略预测；原先 aligned loss 更新大模型，也不等于学生保留了原 WAM 的行为。本次新增：

1. `models/native_wam_teacher.py` 保留原推理轨迹：同一份动作噪声经过两次 video/action joint 更新，再经过八次 action-only 更新。没有把第2步重新初始化为新动作噪声。
2. 教师读取**当前 t 的因果观测**，未来 latent 与动作均从噪声出发。真实未来视频只用于原训练正则，不进入这个教师采样入口；因此没有给学生提供推理时不存在的“真实未来教师输入”。
3. temporal warmup 冻结整个 WAM backbone，包括 video、原 action expert、motion 模块，只训练新 compact 学生。优化器排除全部教师参数，运行中检查冻结合同，不能误将该配置切 joint。
4. 教师给出当前观测下完整16步的归一化动作轨迹，独立采样新的 flow corruption noise，训练学生拟合教师动作分布；同时保留专家动作执行前缀监督。

\[
L=L_{\mathrm{expert},1:H}
+\lambda_T\,\mathbb E_{\tau,\epsilon}
\|v_\theta((1-\sigma_\tau)A_T+\sigma_\tau\epsilon,\tau,o_t,\mathcal B_s)
-(\epsilon-A_T)\|^2,
\quad A_T\sim\pi_{\mathrm{native\ WAM}}(o_t).
\]

新配置 λT=0.5 是可执行起点，不是调参结论。教师采样噪声与蒸馏 corruption noise 分离，避免相关噪声破坏标准 flow-matching 目标。完整 teacher 动作还约束了未立即执行的尾部，而专家监督优先看实际执行前缀。

冻结能保证教师权重不退化，**不能保证学生成功率等于教师**。蒸馏有容量/表示误差，teacher 本身也会失败；必须比较无蒸馏、无时间分组、无桥和原生教师。后续若需要 joint，需新实验保留固定教师锚点；当前不让正在变化的模型偷偷充当“不变教师”。

训练成本会增加：当前 batch 还需要原生教师完整采样。它换取因果、可复现的行为目标，不增加部署快路径前向。必要时可缓存教师动作，但必须记录 checkpoint、episode/frame、采样seed，且不能预先把全部 bridge K/V 落盘造成新的存储问题。

## 5. 连贯性：减少重采样跳变，不能等同于平滑保证

新增可选 `multirate_reuse_action_noise`，probe 中对应 `--reuse-action-noise`。新查询对相同绝对动作位置复用上次采样的初始噪声，只为新进入预测窗口的尾部采新噪声。每次仍由最新观测重新生成全部动作；没有强制沿用旧动作、延迟修正或对输出做低通滤波。episode reset 清空缓存。

这减少一个随机模式切换来源，边际噪声分布保持不变，不需为该开关重新训练。但它不保证动力学平滑，不代替 RTC 对已承诺前缀的处理。需要单独报告 chunk 边界 qpos/velocity/acceleration 跳变、跟踪误差、接触成功率和观测到动作延迟。默认关闭，避免把该变化与架构提升混在一起。

## 6. 数据、训练和现有结果如何使用

继续使用已核对的自采 DOMINO absolute-motion v2 paired 数据。source s 提供历史桥；当前 t 提供三视角、状态、motion 和 t 之后动作标签；新增教师也只使用 t 可获得的输入。不要求重采 3D 或生成新的点云。

新架构**需要训练学生**，已有 V1 joint 权重不是训练好的 temporal compact checkpoint。新配置明确从现有 `absolute_motion_stage3_joint/exports/stage3_step_40000.pt` 初始化完整教师，不从随机视频模型重新预训练，也不隐式放宽 strict loader。原 V1 权重仍可用于此前的对照和缓存工程优化测速。

入口（已准备，未执行）：

```bash
cd /home/wuruihan/DynamicWAM
.venv/bin/python scripts/train_multirate.py --config configs/world_action_multirate_temporal.yaml
```

配置中的12000步是 effective batch8 时约一轮 episode quota 的试验预算；不同 world_size 应按总曝光重新设定，不是“训练到这里就收敛”。先进行真实 GPU 显存和短训练检查，再使用正式训练预算；本次没有启动这些作业。

## 7. 与并行改动的合并边界

工作中发现并合并了另一批新增的 same-cadence 预取、query pacing、phase sampler salt、执行前缀 loss、无桥验证与一次性 K/V dtype 物化代码，没有覆盖它们。

两项必要约束随合并补上：

- 新 checkpoint 记录 `supervised_action_horizon`，只监督前4步的 fast checkpoint 不能因“输出张量有16条”就放行执行16条。temporal 学生另外要求训练执行长度一致及剩余预测覆盖。
- 验证样本仍由 partition seed 固定，不随训练 sampler salt 改变。验证 total 与训练的前缀 loss 定义对齐，同时记录 full chunk、teacher 和 null bridge 指标。训练目标变化写入 exact-resume 合同；旧优化器状态不能不加区分地继续称为完全恢复。

最新 probe 默认同步组加载同一个 multirate checkpoint 的 backbone；要保留 parent 同步组需加 `--official-sync`。`--same-cadence` 或 `--pace-action-interval` 都改变了控制/计时协议，不作为原始 H4 高频实验的替代，也不拿新增 sleep 当算法提速。

## 8. 验证边界

新增8项 CPU 测试通过，覆盖：未来区间重定位、过期槽不读取旧未来、有效未来仍影响输出和梯度、十步学生导出重载一致、原生教师2+8轨迹、学生更新但教师参数不变、噪声重叠位置正确和新配置加载。原有33项修复回归、14项 compact 回归及 training/checkpoint/runtime/schedule 四个专项脚本也通过；probe CLI 和 diff 格式检查通过。日志：`/tmp/dynamicwam-multirate-cpu-wqCCV0/`。

目前能确认的是实现机制和 CPU 合同，不能确认新增架构的 GPU 延迟、闭环成功率或平滑性增益。保留 WAM 能力是本设计中显式受到监督、可以证伪的目标，尚不是已得到的实验结论。
