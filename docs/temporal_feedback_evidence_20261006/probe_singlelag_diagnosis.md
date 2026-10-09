# 单 lag temporal probe 失败诊断（2026-10-06）

## 结论与证据范围

保留 `feasibility_1` 的失败判断。现有证据首先指向“从零训练的两个纠偏头都没有学到可靠纠偏”，而不是已经证明“历史图像无效”。Temporal 相对 static 差 0.47%，这个微小差异的不确定性很大；但两者相对不做纠偏的 zero 都显著更差。训练样本少可能影响学习，不能用它抹掉已经观察到的伤害。更多历史输入本身并不能解决该问题。

本报告只对已冻结的单 lag v1 报告与缓存做 CPU 数值分析，未重训、未重新推理、未使用 GPU、未修改远端源码或结果。后续多 lag v2 与保函数初始化 v3 是开发过程中新的诊断，不能把连续修改后通过的探针当作独立确认性证据。本报告不把后续结果混入 v1 成绩。

源报告 SHA256：`2c806294c53a56ab2c87dc96fa22e82e2f1da1c68fab2780e03da90f65f42e2d`。固定目标缓存 SHA256：`be2107b0445e30954bd2d9dc76b118b34de1a5c6d932a71b45ccc4cbae92e70b`。完整路径、CPU 脚本和输入哈希见同名 JSON。

## 1. 失败有两个层次

目标为 `r*=(clip(teacher-base,±0.1)+0.25 clip(expert-base,±0.1))/1.25`，只统计 age>0、16 步计划内仍有效的未来槽和 12 个机械臂维度，排除夹爪 6/13。MSE 单位为归一化 qpos 的平方；该可达目标不包含训练时 L2 正则。

|预测|heldout target MSE|相对 zero|
|---|---:|---:|
|zero（无纠偏）|0.0007734663|基准|
|static|0.0009722266|增加 25.70%|
|temporal fresh|0.0009768164|增加 26.29%|
|temporal shuffled|0.0009749912|增加 26.06%|

有 244 个 active 样本、223 个 episode、10,704 个有效坐标。对同一 episode 内的样本合并后进行 20,000 次 paired cluster bootstrap，避免把坐标当作独立样本：

- temporal−static 的 MSE 差为 4.5898e−6，95% CI [−1.3896e−5, 2.1727e−5]；相对收益 CI [−2.30%, +1.41%]。
- fresh−shuffled 为 1.8252e−6，95% CI [−2.7824e−6, 6.8277e−6]。
- temporal−zero 为 2.0335e−4，95% CI [1.3928e−4, 2.7056e−4]；相对误差增加 CI [16.77%, 38.22%]。

因此，无法确证很小的历史增益，也不能证明时序信息没有价值；但“两个新头在固定 heldout 目标上比不纠偏更差”的证据明确。原门槛要求 fresh 比 static 至少改善 1%，且不差于 shuffle，两个条件都未通过。

## 2. 样本与时间分布：不足是真实限制，不是全部解释

512 个训练 pair 来自 502 个 episode，256 个 heldout pair 来自 232 个 episode，episode 隔离。各任务 heldout 只有 3–15 个样本，35×15 个 active task-age 单元有 330 个空单元。不能从此数据得出逐任务排名或完整 DOMINO 成功率判断。

训练有效 history 511/512，所有有效 dt 均为约 0.18 秒；heldout 的 dt 最小 0.004、p50/p95/max 均约 0.18 秒。valid dt 未覆盖旧线上每四条命令约 1.616 秒的典型物理间隔。多 lag v2 能改善输入时间范围，却不保证学到有效纠偏。

13/15 个 active age 组的 fresh MSE 高于 zero，只有 age7/12 略低。尾部 age13–15 共 41 行、39 episodes、960 坐标：zero 0.00138362，static 0.00166020，fresh 0.00170402。fresh 比 zero 增加约 23.16%，增幅 CI [12.10%, 37.69%]。分组小样本只用于定位问题，不用于事后选择有利 age。

训练 objective 从 epoch1 到20下降：static 0.001163→0.000709，temporal 0.001163→0.000731。最终 heldout objective 分别约 0.001293/0.001299。训练 trace 是移动 mini-batch 均值，heldout 是最终模型坐标汇总，不能把差值当精确泛化差距，但它不支持“损失下降已证明可用”。

## 3. Teacher/expert 冲突不是主要能量来源

约 24.84% 坐标的 teacher/expert delta 符号相反，初看很高。然而两个 delta 幅值都超过 0.01 的仅 2.31%，都超过 0.05 的仅 0.252%；冲突坐标只承载 3.99% 的可达目标能量。不能把大量微小符号差直接称作严重监督冲突。

任一 delta 超过 0.1 的坐标仅 5.61%，却承载 56.62% 的可达目标能量。clipped teacher/expert disagreement MSE 为 0.000377979。两目标权重 1:0.25 时不可消除的最小二次冲突项是 `0.2 E[(Tc−Ec)^2]≈0.000075596`。这仍不足以解释新增头相对 zero 的全部伤害。

当前每个输入只有一次固定 teacher 随机样本，无法估计条件噪声、可预测的 Bayes risk 或 teacher 多模态性。必须固定输入和 source plan，再只改变 current teacher 噪声，才能区分不可预测监督与学习失败；本次未做该实验。

## 4. 极端旧计划失配与有限纠偏能力

最大两个样本承载 raw zero-teacher SSE 的 93.89%。以下物理差值由原 normalization std 还原，std 并不异常小：

|样本|age|最大差所在维度|归一化差|物理差|本头最大修正|
|---|---:|---:|---:|---:|---:|
|put_bottles_dustbin randomized/72，frame2→38|12|0|7.991|约2.431 rad|0.03042 rad|
|place_shoe randomized/51，frame95→119|8|12|8.892|约5.824 rad|0.06550 rad|

当前 teacher 与专家相对接近，而 source plan 与二者都远离。第一个样本 46/48 teacher 有效坐标、48/48 expert 坐标超过纠偏上界。这不是把 residual 多训练几轮就一定能修复的范围。具体差值是否含关节周期分支、限位或生成失败，需对应 URDF 和执行语义验证；本报告不把 5.824 rad 直接定性为角度 wrap 或控制灾难。

由于极端值主导原始 MSE，即使 oracle 选择每坐标最佳 ±0.1 residual，heldout raw teacher MSE 也只能从 0.0807857 降到 0.0762576，下降约 5.61%。这说明固定小残差不能承诺修复大幅计划错误；可达目标上的改善也不等于原始动作或任务成功率改善。

## 5. 一个无需学习的对照揭示可预测信息与 off-policy 混淆

事后固定诊断规则：

`r = clip(current_state − source_plan[age−1], −0.1, 0.1)`

复制到四个未来槽，仍 mask age>0、有效 tail 和机械臂维度。这个规则只用模型本来已有的输入，无 teacher/expert 标签、无拟合系数。heldout MSE 为 0.000304860，比 zero 低 60.59%；训练缓存上也低约 56.14%。它说明至少相当一部分目标有简单可预测结构，不能把失败全部归于目标随机性。

但这不是建议直接部署该规则。离线 current_state 来自专家轨迹，source plan 并未实际执行；因此规则可能主要在纠正“生成计划和另一个控制器实际走出的专家状态”之间的差。真实在线由 source plan 产生的 current_state 可能更接近计划，输入和标签的条件分布会改变。该结果是监督/状态分布审计证据，不是 on-policy 控制效果，也不是预注册得分。

## 6. 旧正式评测不能恢复 on-policy 训练状态

只读检查旧 `zero_plan_control` 和 `continuation_20261006/eval/feedback` 两个正式输出树：只存在 json/jsonl/yml/txt/log/yaml，未发现 MP4/PT/NPZ/HDF5/PKL；所有任务快照均 `eval_video_log: false`。结构化 JSON 没有 RGB、joint state、source plan 或动作 payload。

`PLAN-FEEDBACK-QUERY` 只有时延、age、mode、residual 最大值和 planner graph 等10类字段。`NativeStepRecord` 只保存 index/wall_ms/overrun_ms，`NativeStepper.execute(action)` 未保存传入 action。episode 汇总仅 steps/kinds/overrun_steps。不能用任务 seed、成功标记和耗时精确重建已经发生的观测、动作、计划或随机噪声。这个结论限定于已检查输出树及其冻结写入代码，不声称服务器其他私人目录绝对没有数据。

最小真实采集位置是 `ciwam/adapters/domino/deploy_policy_plan_feedback.py::eval` 的 `not model.pending` 分支，在 `policy.query(packet)` 前后记录真实 packet、计划、source_state、该次 age、执行动作、sim/wall timestamp。`PlanFeedbackPolicy.query` 内会在末尾 `self.age +=4`，不能把调用后的 age 当作本次输入 age。初次/refresh 的新计划在调用后取得；reuse 时使用未被改写的参考计划。

Shadow teacher 应对同一个真实当前 observation 独立产生全 WAM 计划，并隔离随机状态；不覆盖行为策略的参考计划，不执行 shadow 动作。需要完整当前 composite 和真实 flow history，而不是只有快头缩略图。保存小尺寸 RGB、state/plan/noise/teacher action 和实际执行前后时钟足够开展轻量实验，无需点云。专家离线轨迹不能冒充这类策略状态，旧 head-only 视频即使存在也不能补齐精确腕部视图、state 和 plan。

## 7. 有区分力的下一步

1. 保留单 lag 失败和新多 lag 的独立结果；只把多 lag 看作物理时间覆盖修正，不预设它能解决目标或学习问题。
2. 用严格迁移的已训练 v1 头做保函数初始化，固定原预算比较 static/fresh/shuffle/zero/initial，区分“小数据从零学不动”和“新增历史无条件收益”。所有开发轮次公开，不把换初始化后的通过当独立验证。
3. 在少量固定 episode 上采真实 on-policy 决策状态，比较 `current_state−plan[age−1]`、teacher delta 和 clipping 能量在 offline/online 的分布。若差异大，应优先纠正监督来源，而不是继续增加历史分支容量。
4. 对固定的一小组源计划和真实观测重复 current teacher 噪声，拆开 source-plan 失败、teacher 条件方差和可学纠偏。保持源噪声与执行记录固定，不把不同计划模式的均值当可靠控制命令。
5. 未通过的结果保留。不要在这个 heldout 上搜索 residual 缩放、选择有利 age/任务或反复换 seed，来制造门槛通过。

## 可复现文件

- `analyze_singlelag_probe.py`：原报告 paired episode bootstrap、age/dt/task 分组。
- `read_singlelag_targets.sh`：CPU 只读固定 tensor 缓存，clipping/conflict/物理尺度/无参数 tracking rule。
- `read_onpolicy_artifacts.sh`：旧正式输出及写入函数审计。
- `probe_singlelag_diagnosis.json`：汇总指标、完整路径、身份、top outlier 和输入哈希。
- `singlelag_probe_audit.json`、`singlelag_targets_raw.json`、`onpolicy_artifacts_raw.json`：保留原始审计证据。

## 附录：独立的预训练头对照（45 行，非主门槛）

另一审计脚本使用 CPU 将已训练 v1 stage3 头作用于同一缓存中旧 age4/8/12 的45行：zero MSE 0.00106428，v1 预训练头 0.00102677（较zero低3.52%），scratch static 0.00121231，scratch temporal 0.00123707。它支持检验“保函数迁移是否优于小预算从零训练”，但仅45行，不能替代原 heldout gate，也不能证明真实控制收益。此项含 CPU 模型前向，与上文只做 tensor 算术的主体审计分开标记；未使用 GPU。

证据及可复现脚本在 `../tree/docs/temporal_feedback_evidence_20261006/pretrained_v1_probe_diagnostic.json` 与同目录 `read_pretrained_probe_diagnostic.sh`，JSON SHA256 `ce099dab42873ced2967fb11f6388dfb2a34493f0acce6c58fb20bee82f7ec30`。
