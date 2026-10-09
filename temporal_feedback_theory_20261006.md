# Plan Feedback 下一版：从可观测性、监督耦合和闭环分布出发

日期：2026-10-06。范围：DOMINO level-1；当前冻结官方 DynamicWAM、每16条刷新完整计划、每4条执行一次有界视觉反馈。本文只做调研和设计，没有修改项目源码或启动GPU任务。

## 1. 先明确当前证据支持什么

主任务刚核验的共同29任务、每任务20条结果为：官方与zero-plan-control均231/580，feedback为242/580；其中32条由失败变成功、21条由成功变失败。这个结果来自主任务的评测核验，本调研没有重新读取远端计分文件；完整35任务与配对区间仍应以最终汇总为准。当前合理问题是“为什么只得到小幅改善，以及怎样增加可靠的纠偏收益”，不是再次认定分频必然造成严重退化。

**最值得先试的下一版：保留完整官方计划和现有反馈头，在快路径加入短、因果、带真实时间间隔的视觉历史；使用固定教师标签做单帧/双帧/伪历史对照。** 这比继续加大动作网络或重新做一步蒸馏更直接，因为当前监督接口本身就丢失了教师所使用的运动信息。

源码依据是本地 `work/multirate-repair-20261005/tree` 的实际文件：

| 位置 | 已核实行为 | 对设计的约束 |
| --- | --- | --- |
| `src/dynamicwam/models/plan_feedback.py:54`，`PlanFeedbackHead.forward` | 输入当前RGB、当前/源state、旧16条plan和离散age；没有近期图像历史或真实秒数 | 旧plan可以携带过去的运动判断，却不能表达慢周期内新发生的转向 |
| `scripts/train_plan_feedback.py:176`，`prepare_targets` | current teacher输入当前motion与运动有效掩码；student inputs没有这些motion字段 | 教师和学生的信息集合不同；目标噪声不全来自模型能力不足 |
| `scripts/train_plan_feedback.py:210`，`feedback_objective` | 教师残差和专家残差分别截断，权重1与0.25，残差L2权重0.01 | 学到的是这个混合、截断目标的条件均值，不是最优控制修正 |
| `src/dynamicwam/training/data/paired_world_action_dataset.py:489` | 读取同episode的source/current数据，真实秒数由clock查表 | 可添加历史帧与真实Δt，不需要改视频模型或重采3D数据 |
| `src/dynamicwam/training/data/paired_index.py:1` | 明确区分“3 raw frames/动作标签”的数据索引与在线物理时间 | 不能把1 row解释成固定0.1秒 |
| `src/dynamicwam/runtime/ciwam/adapters/domino/deploy_policy_plan_feedback.py:48` | 每条动作入口已有observation与sim clock，只有query时计算fast输入；flow历史每次入口更新，光流只在慢刷新计算 | 可在每次入口缓存小RGB历史，仍只在query时运行额外编码器 |

## 2. 八篇最相关的一手来源

以下区分论文已经做的机制和对本项目的推论；没有把他人论文的成功率当成DOMINO证据。

| 工作与实际阅读位置 | 可借鉴的具体机制 | 在当前WAM中不能直接迁移的部分 |
| --- | --- | --- |
| **AHA-WAM**，§3.1–3.4、Appendix A/D。[正文](https://arxiv.org/html/2606.09811v1)，[官方代码](https://github.com/serene-sivy/AHA-WAM) | 训练planner/executor相位偏移，当前观测路由旧KV；本文§3.1明确将未来视频作为训练监督，并阻止action读取future tokens，从而部署时移除未来预测路径。 | 其接口不是DynamicWAM原生“含预测未来的联合去噪缓存”；不能把KV复用等同于保留原未来预测。offset覆盖相位不等于覆盖所有实际秒龄，也不等于覆盖策略偏离专家后的状态。 |
| **DuoCore-FS**，§3.1–3.3、Table 5。[正文](https://arxiv.org/html/2512.20188v1) | 慢latent与较新的fast observation做跨时刻训练；RVQ的36固定token对比FAST平均81.12、最大205，实验同时存在长度与表示/顺序预测差异。 | token长度主要限制自回归慢路径，不能推出压缩WAM video tokens就会同样提高成功率。该消融也不是“仅改变token数”的纯因果实验。本文公开页面说明实现经商业平台提供，本次未声称审计其私有代码。 |
| **Reactive Diffusion Policy (RDP)**，§IV-B、Appendix F。[正文](https://arxiv.org/html/2503.02881v3)，[官方实现](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | 慢latent保留动作序列，快GRU decoder在执行过程中接收新触觉；非对称tokenizer使反馈只在decoder进入。 | 收益依赖额外且高频的触觉/力信息。DOMINO目前没有等价传感输入；可迁移的是“快支路必须消费新的可纠偏信息”，而不是照搬其latent tokenizer。 |
| **RTC**，§3–4。[正文](https://arxiv.org/html/2506.07339v1) | 对已经承诺执行的prefix施加强约束，对其余重叠部分施加衰减约束，通过inpainting兼顾连续性与新观测。 | RTC假设控制器在推理时继续消费动作。当前DOMINO同步执行不能据此宣称真实异步收益；其逐去噪梯度引导也不适合直接塞入毫秒级快头。可借鉴计划一致性的监督目标。 |
| **ACT**，§IV-A/B。[正文](https://arxiv.org/html/2304.13705v1) | 预测动作chunk降低有效决策跨度；temporal ensemble平均的是同一目标时刻的多次预测，而非相邻时刻。 | 同时刻对齐非常关键；对不同动作行直接平滑会产生时间偏差。多模态路径的均值未必是可行路径，不宜把ACT平均当作动态任务的通用解法。 |
| **Consistency Policy**，§III-B/C。[正文](https://arxiv.org/html/2405.07503v2)，[官方代码](https://github.com/Aaditya-Prasad/consistency-policy) | 在同一ODE轨迹的不同噪声时刻做一致性蒸馏，辅以DSM；保留观测条件，暖启动并使用特定chaining/noise设置。 | 这不等价于用MSE把随机噪声直接回归到另一个条件下的教师动作。其EDM参数化、采样方差经验不能原封不动套到WAM flow solver，更不能解决历史信息缺失。 |
| **Residual RL for Robot Control**，§III-A/B。[正文](https://arxiv.org/html/1812.03201v2) | 固定已有controller，加一个可学习残差，并用真实环境回报学习其影响。 | 论文的基础控制器具有特定动力学结构；当前WAM计划不是已证明稳定的控制律。残差幅度小与零初始化都不自动保证闭环稳定或成功率不降；BC残差也不等于RL残差。 |
| **DAgger**，§2–4、Algorithm 3.1、Theorem 3.1/3.2。[原始论文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | 在学习者实际到达的状态上查询专家并聚合数据；风险目标依赖学习者自身状态分布。 | 定理依赖损失、在线学习与专家等条件，不能把神经网络训练直接代入其保证。当前官方WAM只有约四成任务成功率，是可查询参考而非正确纠错oracle；聚合错误标签可能更糟。 |

代码阅读补充：RDP官方 [`latent_diffusion_unet_image_policy.py`](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/main/reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py) 的 `predict_action` 先采latent，再调用带 `temporal_cond` 的decoder；`predict_from_latent_action` 则直接复用latent并消费新的extended observation。该文件还显式用 `n_obs_steps * dataset_obs_temporal_downsample_ratio` 对齐动作切片。其[官方数据说明](https://github.com/xiaoxiaoxh/reactive_diffusion_policy#-training)要求控制频率与数据下采样一致。这里最可迁移的是接口对齐，而非“存在快慢两个网络”这一外形。

AHA有两个应审慎处理的报告细节：正文说从10步蒸馏到2步，而Table 5列teacher 16步，复现应以具体checkpoint/实现为准；正文推理延迟包括action path但将异步video prefill另列，不能直接与串行整次WAM查询相除得到我们系统的速度。其延迟表也明确包含CUDA/TensorRT/runtime工程优化，架构和工程收益需要分别报告。[AHA Appendix A/D](https://arxiv.org/html/2606.09811v1#A1)

## 3. 当前最扎实的理论问题：学生无法观测教师依赖的运动状态

令现有快头输入为

\[
X_t=(I_t,q_t,q_s,P_s,k,l),\qquad k\in\{0,4,8,12\}.
\]

其中旧计划 \(P_s\) 生成于过去观测。令 \(H_t\) 为近期因果图像历史。两次执行可以拥有相同的当前物体位置、机器人状态和旧计划，但物体近期向相反方向运动，或刚发生不同方向的转弯。只看 \(X_t\) 的确定性反馈不能区分二者。旧计划可能减少这种歧义，但不能保证覆盖之后的随机改变。

这个问题在当前代码中进一步表现为监督信息差：current teacher使用 `current.motion`，快头不使用。对同一固定标签 \(Y\)、同一分布，在平方损失和无限表达能力下，Bayes最优风险满足

\[
R^*(X)=\mathbb E\|Y-\mathbb E[Y\mid X]\|^2,
\]

\[
R^*(X)-R^*(X,H)
=\mathbb E\|\mathbb E[Y\mid X,H]-\mathbb E[Y\mid X]\|^2\geq0.
\]

这是条件期望的正交分解，说明额外历史**有可能降低不可约回归误差**；不是“多输入必然提升”的保证。有限数据、优化不充分、错误教师、时序分布变化，都可能让实际模型退化。因此固定标签与伪历史对照，比加模块后只看总loss更重要。

双帧不等于完整运动状态：遮挡、相机运动、深度歧义、时间间隔过大、接触造成的非连续变化仍可能不可辨。它也不等价于教师四帧Farneback历史。这里的目的，是检验最小的新增信息能否带来可重复的纠偏收益。

## 4. 历史该取前1 row还是前4 rows

首选**上一个执行命令入口的观测**，即1 row历史。上一个query通常是4 rows前，宜保留作消融而不是默认。每次adapter入口已获得observation，只需保存head小图和sim timestamp；下一次query时才计算历史特征，无需每条动作做GPU编码。

用一维运动的局部分析说明取舍。假设短时间内位置测量为 \(y(t)=p(t)+\epsilon_t\)，噪声独立、方差 \(\sigma^2\)，后向差分为

\[
\widehat v(t)=\frac{y(t)-y(t-h)}h
=v(t)-\frac12a(t)h+O(h^2)+\frac{\epsilon_t-\epsilon_{t-h}}h.
\]

因此局部偏差随 \(h\) 增大，噪声方差约为 \(2\sigma^2/h^2\)。这只是解释选择的简化模型，不把RGB特征声称为已标定物理位置。更近的帧适合转弯/接触后的变化；更远的帧可能具有更高运动信噪比，却更易混入遮挡、非匀速和阶段切换。不能脱离真实间隔说“1 row永远优于4 rows”。

建议数据契约：

1. 训练默认历史索引 `previous_frame = fast_frame - global_downsample_rate`；当前rate为3。这表示前1个动作标签间隔，**不表示固定0.1秒**。
2. `history_dt = clock_time_seconds(fast_frame) - clock_time_seconds(previous_frame)`。由现有episode clock读取；找不到对应帧、Δt≤0或跨episode时，history mask为0，历史图可复制当前图，不能伪造有效速度。
3. 在线保存每次adapter入口的head crop与 `_scene_step_clock` 时间；使用前一个入口，不依赖墙钟，也不把网络推理耗时算作模拟物体运动时间。
4. 显式给head输入有序图像对、`log1p(history_dt / reference_dt)`、history有效掩码，以及可选的源计划真实秒龄。数据索引age和真实时间必须分别编码。
5. 先测训练与部署1-row/4-row的Δt直方图。若在线gap超过训练支持范围，可从**真实更早的帧**采样历史，覆盖在线Δt分位区间；不能只修改dt数值、保留原图像对。

当前采样只能在命令入口拿到相机观测。如果一次TOPP命令已消耗很久，前1 row本身仍可能很旧；双帧网络无法凭空恢复中间运动。要获得固定物理频率图像，需要另改观测/仿真协议，应作为另一实验，而不是暗中混入这一版。

## 5. 有界残差目标到底在学什么

令 \(b=P_s[k:k+4]\)，\(\Delta_T=A_T^{current}[:4]-b\)，\(\Delta_E=A_E[:4]-b\)。忽略mask和逐维求和，当前目标为

\[
\mathcal L(r)=\|r-C_c(\Delta_T)\|^2
+0.25\|r-C_c(\Delta_E)\|^2+0.01\|r\|^2,
\quad c=0.1.
\]

逐样本无函数约束时最优值是

\[
r^*=\frac{C_c(\Delta_T)+0.25C_c(\Delta_E)}{1.26}.
\]

确定性网络实际近似其给定输入的条件均值。若教师重采样存在不同抓取模式，或教师与专家相互矛盾，均值会被拉小或位于两种有效动作之间。`shifted_action_noise`共享重叠噪声有利于降低方差，但条件观测与未来noise的时间语义发生变化，不能保证采样模态一致。

此外，“分别clip后混合”不等价于“先优化原始误差再投影到残差盒”：后者是 \(C_c((\Delta_T+0.25\Delta_E)/1.26)\)。前者可解释为更鲁棒的标签截断，本身不是实现bug；但需要明确它改变了学习目标。不要在加入历史的同一次对照里再改clip顺序，否则无法确定收益来源。

应先做一个教师标签噪声审计：固定256个source/current pair及source plan，current teacher对每pair取4个可复现noise；其中一个复用现有shift方案，其余只作离线诊断。统计active arm维度的重复标签方差、方向一致率、teacher/expert符号冲突率和clip率，分age、任务、运动强度报告。重复采样方差不是环境不确定性的校准置信度，但可以判断残差幅度是否主要追逐生成噪声。

质量验证必须排除age0和gripper的恒等区域，否则零修正部分会稀释误差；同时报告未clip真实teacher/expert误差，防止只学会贴着clip边界而掩盖实际偏差。

## 6. 半马尔可夫执行与MPC类比的边界

DOMINO的一条动作通过TOPP执行，其时长依赖当前机器人状态和目标，而不是固定采样周期。更合理的转移是

\[
(s_{n+1},\tau_n)\sim p(\cdot\mid s_n,a_n),\qquad
t_{n+1}=t_n+\tau_n.
\]

因此四条动作的闭环间隔是 \(\sum_{i=n}^{n+3}\tau_i\)，而非4×100ms。高控制查询频率必须同时给出action-row、sim-time和wall-time口径；按row训练的旧plan也不能自动被解释为对真实秒数的有效未来预测。

当前方案与receding-horizon control相似之处，仅是执行计划局部后再接收反馈；它没有在线约束优化、终端集合或可验证的模型误差界，不能借用MPC稳定性结论。若假设局部系统对状态/动作Lipschitz，可写出

\[
\|e_{n+1}\|\le L_s(\tau_n)\|e_n\|+L_a(\tau_n)\|r_n\|+\|w_n\|.
\]

即使 \(\|r_n\|\) 有界，长 \(\tau_n\)、接触切换或不稳定局部动力学仍可能放大误差。这里的意义是指导按Δt、phase与残差幅度分层评测，而不是宣称小残差确保安全或不降低准确率。

速度预算也应直接写清：设相同工程优化下完整WAM查询成本为 \(C_W\)，快头为 \(C_F\)，则

\[
\bar C_{H16}=C_W/16,\qquad
\bar C_{feedback}=(C_W+3C_F)/16.
\]

所以反馈架构不会比**同样优化的**H16零反馈在总计算量上更低；它购买的是每4条动作一次新反馈。与每4条都重跑WAM相比则可明显减少计算。下一版增加历史应争取“更高反馈收益/很小新增代价”，不能归因于新的总延迟下降。若未来要进一步压低慢查询开销，可另研究经同条件蒸馏的少步slow sampler，但那是独立质量风险，应单独消融。

## 7. 推荐下一版A：固定目标的短历史反馈

**目标：验证新增运动信息是否改善快头，而不是通过增加训练预算、换teacher或改变执行协议取得表面收益。**

结构保持现有三视角当前appearance、完整16条plan和H4输出，额外加入head相机64或96像素有序图像对的小支路。可以输入当前/过去图像及差分，输出少量motion tokens；差分和Δt作为特征，不硬把像素差解释成真实速度。新增投影零初始化，加载已有反馈权重时应逐元素恢复原输出。保留age0零修正、gripper零修正、FP32主权重和0.1幅度约束。

最小试验过程：

1. 从既有episode划分中取分任务/age平衡的小集，先512pair测缓存成本，再扩到约2000训练、512验证pair。训练/验证必须按episode隔离，既有测试700条种子不能回流训练。
2. 缓存一次官方source plan、current teacher目标和专家目标，保存noise种子、模型/代码SHA、数据ID；所有对照使用同一份固定目标，避免teacher重采样方差淹没历史收益。
3. 对照：现单帧；双帧真实history+Δt；同结构但previous=current且mask置无效；在验证时打乱历史或交换顺序的诊断。4-row历史仅作为附加消融。伪历史分支控制额外参数，乱序诊断检验是否利用了时序信息；二者作用不同。
4. 先用同一个旧head初始化、同数据顺序、同优化预算做500–1000step小规模精调；至少两个训练seed复验方向。原teacher/expert权重与clip不变；不跑新的stage1，也不做K1蒸馏。
5. 看active区未clip teacher/expert误差、按Δt/运动/age分层的差值、residual符号与幅值，以及95%分位耗时。只有在真实history优于duplicate且破坏时序使收益下降时，才有支持该机制的证据。
6. 做实际input与缓存/缺帧情况的完整pipeline profile：新快头p95、每16条总查询耗时、额外CPU图像拷贝，不能只测新增CNN。预先设一个小开销门槛，例如总查询预算增加不超过5%，具体以当前实测分解决定。
7. 通过后训练正式新版，并用旧checkpoint/zero/new三臂同一manifest评测；报告paired胜负、任务分层区间、teacher收益和expert收益是否一致。不从6条canary推断质量等价。

代码落点：dataset只增加因果历史帧与clock字段；trainer把字段传入head；runtime在每条动作入口更新原始小图缓存并在query时消费；export保存history取样/尺度/时间编码契约。任何字段缺失必须显式mask或拒绝加载，不可默默把4-row缓存当1-row。

## 8. 候选下一版B：先审计，再做有限的状态分布修复

离线训练时current RGB/state来自专家轨迹；在线则来自旧plan加反馈的轨迹。即使输入还包含source plan，网络也未必见过“这个plan实际把机器人带到的状态”。这是DAgger可以启发的问题，但当前官方教师不是强纠错oracle。

建议把它列为第二优先级：先单独收集与评测种子不重叠的少量on-policy训练rollout，保存query时刻的RGB/history、q、source q、source plan、age和实际Δt。只在这些真实状态上调用官方current teacher，比较其目标是否集中落在现有0.1可达修正范围内，以及同一状态多次teacher采样是否一致。大残差或不同抓取模态不应被强行标成局部修正。

如果需要训练，先用小比例（例如25%）这种数据混合原专家pair；其标签只能称“参考策略重标注”，不能称最优动作。对这些状态没有真实专家action时，不能把原演示的同索引action拼过来。保留原expert监督与回退到原plan的幅度约束，以held-out闭环表现判断是否值得扩大。

更强的标签需要在同一完整仿真状态分支执行zero与候选修正，获得可比的短期/最终结果；这要求完整恢复物理状态、动态物体进程和planner状态，工程与计算成本明显高于A。只按相同seed重启不等价于同一中间状态分支。因此本轮不建议把这项变成默认大规模DAgger训练，更不建议直接加入离线critic来包装未经验证的teacher质量。

## 9. 本轮决策顺序

先完成当前固定manifest评测并保留全部结果；随后优先完成A的固定目标信息消融。若历史无收益且目标采样方差很高，应先清理监督耦合；若历史改善离线误差却不改善闭环，应优先检查真实Δt和到达状态分布。只有在现有轻量反馈的质量与开销边界明确后，才考虑更大动作模型、慢模型蒸馏或改变刷新周期。

本文的理论价值应体现在可证伪的实验上：**输入历史是否减少可观测性缺口、标签噪声是否限制残差学习、纠偏是否在真实状态与时间间隔下有效。** 这些结论都不要求先声称一个新的稳定性定理，也不依赖把已有WAM强行改成VLA式的语义缓存。

## 10. 2026-10-06 实施决策：全部计划年龄、固定四行输出与 H2 主线

前文第7节描述了研究候选。落实时选择了更便于归因的版本：新建时序头而不强行映射旧头权重，当前三视角和上一命令 head 图像统一为160像素，共享视觉编码器；时序与静态探针从完全相同的新初始化开始。所有监督目标先固定缓存，避免额外教师采样噪声变成组间变量。正式训练只训练新头，冻结原WAM；不启动stage1，也不改变slow sampler。

**H1比H4增加的是决策机会，不能直接推出物理闭环速度提高四倍。** 若观测变化发生在某条命令中，H1能在这条命令完成后的第一个边界反应；H4可能还需等待最多三条额外TOPP命令。单条命令自身不可中断，其持续时间仍可能很长。除了每动作查询数，应报告实际决策间隔秒数、每条TOPP时长分位数，以及成功/失败的任务差异。H2介于二者之间，仍能减少H4内部一个命令边界的等待。

同一16行slow plan下，H1/H2/H4各需15/7/3次额外feedback，age0直接执行官方plan而不浪费反馈调用。总服务预算为

\[
B(H)=C_W+(16/H-1)C_F.
\]

主任务的真实模型profile给出相对官方eager H16总预算比值：H1约0.897、H2约0.791、H4约0.738；预注册预算门槛为不超过0.85，因而当前**选H2作为正式主线**，H4用于同权重频率消融，H1保留为未通过本轮预算门槛的候选。这些数值是主任务提供的本轮实测结果，原始profile与源码SHA应随正式流水线保存；不能把它们归因于时序信息本身，因为Graph/slow路径等价工程优化也在其中。相对同样优化的zero-plan基线，feedback仍有正开销。

**为什么固定输出4，而不是仅输出1：** 输出1最贴合H1，尾部没有越界问题，但不能以同一权重公平比较H2/H4；届时输出维度、监督密度和控制频率同时变化。固定输出4允许只改变执行前缀长度。训练覆盖age0..15，定义

\[
m_{k,j,d}=\mathbf1[k>0]\mathbf1[k+j<16]\mathbf1[d\notin\{6,13\}],
\quad j\in\{0,1,2,3\}.
\]

同一个mask必须用于loss、误差/裁剪指标和梯度，网络输出也强制末尾越界行为零。安全gather可以重复第15行以避免非法索引，但绝不能把这些重复行当监督目标。全局验证用有效arm坐标加权，age0/夹爪/填充尾部不参与分母。否则age13..15的无效行会人为降低误差并产生虚假的提升。

重叠四行监督会让同一绝对动作在多个source/current上下文中出现；这是针对不同可观测状态的条件监督，不等于独立样本增加四倍。数据划分仍按episode，任何训练和验证目标缓存都不可跨越这个边界。测试seed不回流训练。训练遍历所有age有助于同权重控制频率消融，但正式H2只执行偶数age、H4只执行4的倍数；指标需分别报告各自部署支持上的误差，不能以all-age均值替代目标部署结果。

“前一命令history”在所有H下都保持同样定义：运行时每条命令入口更新CPU图像和simulator时钟，query时读取上一条命令入口；训练在演示里读取当前条件帧之前3个raw frame。这是动作索引语义的对应，不保证专家演示和TOPP在线时长分布相等。显式输入真实history_dt与plan_age_seconds仅让模型有机会识别分布差异，不是自动完成时间尺度外推。正式前的probe/数据审计应给出真实Δt范围，在线测试按超出训练分位范围与否分层报告。

代码采用20 bytes/pair的int32索引和按episode/age连续范围抽样，避免把全部16个age展开为大量Python对象；验证序列前16个全局样本明确覆盖0..15，之后各rank切片，保证短smoke也检验尾部年龄。该工程调整不增加训练预算，仍按实际global batch和优化步数计算曝光样本数。

## 11. 真实数据审计后的修正：多时间间隔历史，计划秒龄只做遥测

CPU真实读取覆盖35任务、训练/验证各256 pair，所有16个age；9450/1050个episode的划分无交集，动作normalizer与官方full匹配。两个子集共478个已抽样episode的完整clock扫描证明，前3个raw frame间隔的最大值均为0.180000009秒；它不只是512次随机抽样恰好未命中长尾。训练age4/8/12在抽样中的计划秒龄最大值分别为0.72/1.44/2.16秒。

相比之下，既有共同29任务的反馈评测有34209个四命令间隔，中位数1.616秒、p95=3.320秒，96.04%大于4×0.18秒。每个这样的间隔必有至少一条命令超出单lag训练范围；不能把四条之和的分位数除4，伪称恢复了单命令分位数。旧adapter每四命令才写query时钟，NativeStepper也不保存单命令simulator时长，因此精确单命令分布不能从旧日志识别。在线age4/8/12计划秒龄分别有97.49%/97.73%/99.55%超过训练抽样最大值。

因此原先“保留前3帧并增加dt输入”不足以支持正式训练，改为**真实历史图像间隔增强**：训练和验证从RAW frame lag `{1,2,3,6,9,12,18,24,36}`中选合法过去帧，使用那个帧的真实时间戳；由pair身份和partition seed的固定SHA256选择，使同pair各epoch、cache对照和验证均可复现。理论上按已观测clock最大步幅可延伸到约2.16秒，但最终分布仍以新版审计为准。前3帧只是其中一个取值，不再承担所有部署时长。clock范围内缺少必需时间戳直接报错，episode开头没有合法历史才mask；不通过伪造dt或悄悄切换一个更容易的lag隐藏数据缺失。

首次多lag真实审计还发现相邻raw图像可能具有相同simulator时间戳。这样的图像没有正时间间隔，不构成运动历史；候选筛选明确排除dt=0，全部候选都无正间隔时用invalid mask。审计记录每个pair的零间隔候选及被选lag，单独测试多帧同tick与全同tick；缺失/负值/非有限时间仍失败，不以极小epsilon伪造速度信息。这是时钟支持规则，而不是根据模型效果挑选容易的历史。

运行时仍读取上一条实际命令入口的观测，真实history_dt直接记录。多lag训练并未宣称每个lag就是一条机器人命令；它提供不同真实时间跨度的运动线索，以覆盖可变命令时长的视觉输入。新评测的history_dt日志才能检验实际覆盖比例，特别是超过训练最大跨度的尾部。

计划秒龄的情况不同：仅更换历史帧不会改变source plan与当前观测的对应，也不能让同一动作age对应的训练物理时间覆盖在线分布。为避免网络把几乎由动作age确定的训练秒龄当作可外推变量，**v2从模型time MLP删除plan_age_seconds，仅保留其遥测与输入合同检查**；真正进入网络的时间量是history_dt和history_valid。动作age、当前状态、source状态、原完整plan及当前视觉仍保留。这个改动去除了一个明显不受训练数据约束的外推通道，不能宣称修复了策略执行后状态分布偏移。若新版闭环仍受此问题限制，需要另行产生真实执行状态的数据，而不能任意拉伸source/current索引并冒充对应专家轨迹。

原单lag版本冻结为feasibility1诊断。正式门槛必须使用多lag、v2模型/合同、全新的固定target cache与源码SHA，不覆盖原结果，也不把旧probe通过当新版质量证据。

修正后的真实CPU审计已用相同512个pair复验通过，九种lag全部实际选中。训练history_dt中位数0.360秒、p95=1.877秒、最大2.160秒；验证为0.420/1.869/2.160秒。训练9个lag1候选、验证10个lag1和1个lag2候选属于同sim tick，按显式规则排除。样本、源码SHA和逐pair真实索引保存在 `docs/temporal_feedback_evidence_20261006/dataset_audit_multilag_v2.json`。这些结果证明数据实现和更宽时间支撑，不等价于已证明在线history覆盖充分或控制质量提高；后两者仍需新运行时日志和闭环结果。

## 12. 两次探针失败后的诊断：先区分优化起点与监督的状态捷径

第10节记录了当时选择从零初始化的过程；该决定在小样本探针中没有获得支持，v3改回第7节建议的已有head保函数初始化。单lag探针固定512训练pair、256验证pair，训练20 epoch、batch32共320次更新。静态与时序头的视觉编码器均为 `CompactVision(compact_gn)`，四层卷积和GroupNorm从零训练；时序头总计约243万参数，视觉body约35.6万参数。训练涉及35任务、16个age，512个pair来自502个episode，这不是足以单独排除表征学习困难的大样本信息实验。固定样本重复20次也不同于旧正式训练约126000次新pair/教师目标曝光。

真实结果比“历史无收益”更具体：单lagheld-out有效坐标上，zero reachable-target MSE约0.00077347，scratch static为0.00097223、temporal为0.00097682；时序相对静态差约0.47%，两者比zero差约26%。新多lagv2探针也未过预设门槛。这足以否定当前从零小样本recipe，尚不能证明历史中没有可学信息；更不能以此为借口直接扩大长训。已有低维状态特征也没被scratch头学好，问题不应全部归咎于视觉CNN。

只读CPU追加检查把正式训练过的旧v1头放到同一缓存中，仅取其原支持age4/8/12的45个held-out pair、2160个有效arm坐标。旧v1 MSE为0.00102677，zero为0.00106428，scratch static/temporal在同45个pair上为0.00121231/0.00123707。旧v1相对zero约好3.52%，但age4单独退化，样本很小。这是“已有表征提供更合理起点”的证据，不是新历史架构成功或闭环准确率提升的证据。检查已验证官方权重SHA、normalizer、packed数据身份与原episode划分一致，原始逐pair结果和复现实验在 `docs/temporal_feedback_evidence_20261006/pretrained_v1_probe_diagnostic.json`。

迁移必须保留attention语义。把新历史token值设为零，但仍让它们参与softmax，分母仍会变大；8个真实输入的CPU FP32检查中，这种做法相对旧head最大输出差约0.0146，而屏蔽新token且零化time增量时约1.86e-8。因此v3保留旧69-token memory布局，将新增时序MLP零初始化输出加到既有head-view token上；time MLP的新增输出也为零。共享权重全部严格复制。旧age embedding在0/4/8/12原样保留，其他age线性插值或在13..15保持末项；只有旧支持age能声称初始化函数保持，其他age必须实测。两臂使用同一旧checkpoint、同一初始化、同一缓存/样本顺序，低LR2e-5、320次更新，目标与裁剪方式均不改变。

更迫切的限制是**离线current state并非执行source plan到达的状态**。令旧计划为 \(P_s\)，age为 \(k>0\)。不看图像、不拟合任何参数的规则

\[
r_{k,j,d}=C_{0.1}\left(q_k[d]-P_s[k-1,d]\right),
\quad j=0,1,2,3
\]

再施加有效age/tail/arm mask，在held-out reachable-target上MSE约0.00030486，比zero低60.59%。其训练集MSE也明显降低。当前head本来已显式接收这个state差值，故这不是新架构特征。它说明离线目标很大部分能被“专家轨迹当前位置与未执行生成计划的偏差”解释；不能直接解释成动态物体运动的观测收益。

离线的 \(q_k\) 来自专家轨迹；source plan由独立生成器产生且未实际执行，所以这个差值可能表示两条行为模态的分离。后续源码核实纠正了“跟踪误差”解释：DOMINO的joint_action取自drive target，不是实测qpos。在线TOPP执行source plan后，此命令状态接近 \(P_s[k-1]\) 加上既往修正；真实跟踪误差不在这个字段中。固定规则甚至可能把“上一次自身修正”重新传播到未来。这是可检验的distribution shift，不是可以由小残差、迁移学习或更宽history_dt自动消除的问题。

原始teacher目标里确实有大偏差，但不能把它们当作全部原因：前5个样本占未clip teacher误差约97%，clip后仅占目标能量约15.7%；teacher/expert的符号冲突率约24.8%，但两侧幅度都大于0.01时仅约2.31%，冲突占目标能量约3.99%。数字更支持先检查状态捷径，而不是立即设计复杂的多模态混合头。某些关节差接近2π是否属于等价角度表示，尚需具体关节范围/拓扑验证，不能仅凭差值自动做wrap修正。

正式长训前增加一个独立的**真实zero-plan执行shadow审计**：执行原完整官方source plan，保存真实到达状态及history；current官方teacher只生成影子标签、不参与控制，标签用独立随机数流，不能改变策略后续采样。在相同真实状态和标签上比较zero、上述固定state-error规则、原v1及v3。重点报告state误差分布、可达目标/未clip误差、裁剪率、teacher重采样方差、按age/dt/任务分层的差别；若可收集多次teacher标签，需与控制RNG隔离。参考teacher自身成功率有限，这些标签不能当作最优动作，也不能把缺少专家的on-policy状态拼上原演示同索引action。

v3 warmstart只处理优化与表征起点，离线probe过门仍是开发证据。若真实shadow显示旧离线优势依赖专家轨迹偏差，下一步应先修监督分布与目标定义，不能直接启动7000/2000步长训；如果shadow至少没有暴露这种失配，仍需有界canary与固定manifest闭环对照，才能判断控制收益。当前不增加DAgger、critic或新的大模型，优先让这个最小证据链成立。
