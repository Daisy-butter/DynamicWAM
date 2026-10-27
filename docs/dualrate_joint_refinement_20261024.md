# DualRate 架构、时序与数据修订建议

日期：2026-10-24。状态：设计提案，尚未替换已保存实现，未启动训练或评测。

依据：用户 `DualRate_DynamicWAM_Design.md`、harness v1.18、.92 候选 `3d5a9a8c4342acb896e30bfe2baebf4e09b5c26a` 的已核实源码快照，以及下列官方论文/代码。可写远端仅 .92；本轮只写本地说明。

## 1. 决策

保留完整原生 DynamicWAM 慢路与其 2 次联合、8 次动作采样；保留快路 384 维、6 层、6 头的 A/Z 联合 flow-matching 骨架。重点修订当前视觉编码、结构化慢路读出、过期计划拒绝，以及真实物理时间数据生产。

快路预测的是动作和紧凑实体未来，不承担像素视频解码。慢路负责较丰富的视觉预测。模型大小、采样次数、更新频率是三个不同的量。

当前实现：快路约 15.09M；16×14 动作与16×134未来联合去噪；观测编码器是从零训练的三层无padding CNN，总stride32；慢路读出是VAE latent逐slice池化到4×4后投影，并未提取DiT隐藏状态。当前99项CPU检查不等于真实标签、完整GPU前后向、连续DOMINO或正式训练验收。

## 2. 外部实现如何使用

| 来源 | 核实的内容 | 本方案的用法 |
|---|---|---|
| [PocketWAM](https://github.com/penpenzh/pocketwam)，模型 [wam.py](https://github.com/penpenzh/pocketwam/blob/b5082594b958d466a04d443d4746f769d18bc889/pwam/model/wam.py) | 小型空间CNN和联合世界/动作建模；约3M，二维64×64任务；还有额外逆动力学训练路径 | 借鉴保留空间结构和简洁联合建模；不移植其任务动作空间、RGB目标或独立IDM训练路径 |
| [SmolWAM](https://arxiv.org/abs/2610.13740)，[model.py](https://github.com/iitimii/SmolWAM/blob/dde1e5bfe35c439499343c6d70e9cd32c96d0b36/smolwam/model.py) | 从Cosmos预训练主干裁剪、蒸馏；动作和未来latent读出；约0.88B | 参考时步调制与query设计；不将其Cosmos布局、容量和蒸馏后一步推理直接套到新快路 |
| [Faster-WAM](https://github.com/hustvl/FasterWAM) | 生成未来表征并在动作去噪中复用 | 支持保留推理时未来条件；其一次采样内复用不能当成跨控制周期缓存可靠性的证据 |
| [Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) | 官方ImageNet预训练视觉模型 | 截断至layer3，作为轻量当前感知的首选初始化；实际时延待测 |

## 3. 快路结构

### 3.1 当前观测 E

首选：三相机共享 `ResNet18(stem, layer1, layer2, layer3)`，stride16、输出256维；截断主干约2.78M参数，随后1×1投影到384维。去掉layer4、全局池化和分类头。小batch冻结BN运行统计，卷积参数可训练。

图像等比resize并padding，不使用分类模型的中心裁剪。保留padding有效掩码、变换后的像素坐标和相机标记。输出空间grid按输入实际尺寸确定，不先全局平均；若需减少token，必须按相机分别处理并评估小物体信息损失。当前240高图像被无padding stride32丢掉底部16像素的问题同时修复。

加入已有光流的空间信息：复用当前运动历史中最近有效光流对，按head视图4×4网格提取位移/速度与有效比例，形成16个运动token，附起止曝光时间。保留原4×12全局motion token，以免同时删除已有运动接口。无有效历史时明确mask，不能用零流冒充静止。

其他条件：本体状态token、缓存语言token、每个观测的真实曝光/状态时间。保留原DynamicWAM的动作/状态定义，不把drive target和实测qpos混用。语言编码器不在每次快查询重跑。任务实体角色可由已知任务定义提供，未来可见性和GT分割只用于监督。

### 3.2 A/Z joint transformer

```text
A_sigma: [B,16,14] -> Linear -> 16 tokens
Z_sigma: [B,16,134] -> Linear -> 16 tokens
                         ↓
            concatenate: [B,32,384]
                         ↓ ×6
      joint self-attention(A ↔ Z)
      cross-attention to current E
      gated cross-attention to slow M
      FFN (1536 hidden)
                         ↓
       action velocity / Z velocity / contact logits
```

每层使用pre-norm残差；将FM噪声时刻、query-relative target时间和source age作为条件调制。Q/K归一可参考成熟DiT实现，新增调制和门控需要初始化/数值验收。慢路门控是小而非零的可学习初值，主阶段保持快路loss到慢路的梯度。

分开E和M的attention归一化，允许最新观察与旧参考分别发挥作用。Q来自同一批A/Z token；K/V由本层将E或M重新投影得到。这里的K/V是普通cross-attention计算结果，不是复用/更新原DynamicWAM各层K/V。

首版维持10次FM积分，不直接改成一步；当前另有一次sigma0前向产出contact，实际采样共11次网络前向，计入延迟。E/M的K/V在一次query内预计算并复用，条件改变则失效；训练保留梯度。这项优化要做数值等价验证。

已承诺动作在每次积分中硬钳制；动作loss不重复惩罚硬前缀。Z在各层参与动作生成；失效实体在attention、loss、输出及诊断中均携带mask，不能展示初始噪声或默认0.5接触概率。

Z仍为4角色×4未来时刻（0.1/0.2/0.4/0.8秒），每个token128维局部视觉、3维统一坐标位置、3维相对相关夹爪位置；接触是独立概率监督。4角色对多物体/工具任务只是初始覆盖：逐任务审查角色绑定及关键对象遗漏，不能因通道固定而默默丢掉交互主体。

## 4. M是什么，以及从哪里读

```text
source observation/history ──native patch embedding──> O_s
                      └──── full DynamicWAM ──> H_future(stage2), V_hat(stage2), A_ref(stage10)

M = concatenate[R_obs(O_s), R_future(H_future, V_hat), R_action(A_ref)] + metadata

fast query: E(latest observations) + cross_attention(A/Z, M)
```

“共享世界信息”是统一数据包M；“可供读取的上下文”描述M的用途，不是还要额外生成的第四种张量。R是轻量可训练读出/投影模块，逻辑上属于慢快接口。快路不需要把预测视频解码为RGB，再用一个大型视觉模型重新编码。

### 4.1 选定来源

| 组 | 具体来源 | 读出与解释 |
|---|---|---|
| 源观察O | 原生多尺度patch embedding后、进入任何联合attention之前的condition前缀 | LN+Linear(2048→384)；是因果清楚的源观察特征，不宣传成纯粹高层语义 |
| 预测未来F | 第二次联合前向最后Video block的future token，video head之前；同时取本次Euler更新后的预测clean video latent对应patch | 每cell拼接2048维hidden与192维latent patch，投影2240→384；hidden保留上下文，latent保留最终视觉预测数值 |
| 软参考动作A | 第10步结束后的16行动作 | 14→384，保留动作物理作用区间与soft-reference类型；不进入执行队列 |

原生布局通过 `grid_sizes.condition_seq_len/future_seq_len` 切分，不能凭token前几个位置猜语义。联合attention后的condition hidden可能混入候选future/action，不能当成始终有效的纯观察上下文。提取接口只给source-only predictive forward；监督forward不得导出M。

当前future latent `[48,2,12,10]`，patch `(1,2,2)` 对应60个future token，每个clean latent patch192维。保留原生格点，不再压成每slice4×4。source token数量由实际condition shape确定（若为48×2×24×20则240个），不能把预期shape当成实测。三视角拼图的边界可能穿过patch；记录camera-overlap/padding比例，跨边界token标mixed，不伪造单相机身份。

每个token保存空间位置、观测或预测类型、源时间、真实时间支持区间、生成阶段/噪声阶段、有效掩码。观察时间支持与“可作为历史上下文读取”分开；不能用±inf替代时间来源。future是否能给目标时刻参考，由物理覆盖判断。Video codec有时间压缩，两个latent不等于两个RGB帧，更不自动证明覆盖输入的全部视频时间；实际codec frame→latent映射必须测试。

### 4.2 原生2+8的限制

视频在第2次joint更新完成，动作继续细化到第10步。F不是“在最终A10下重新仿真的结果”。不增加第三次完整Video前向，也不强加无条件一致性loss。保存第二次joint的动作clean-estimate与A10差异作诊断；跨路先仅传A10软计划，避免增加两套容易混淆的动作条件。

Mfuture保留stage/sigma标记。后续实际动作与A_ref有偏差时，快路根据实际执行轨迹、当前硬前缀和旧计划的逐时间差异调整参考权重；重大身份/时间错误直接mask。连续动作偏差阈值须在开发集标定，不凭空宣称预测仍成立。保留源观察上下文，不因任一快更新而丢弃所有M。

动作参考作为有时序背景的软计划读取，保留从source到目标时刻相关动作，不只匹配target±20ms的单点。future预测的时间mask与动作上下文的可读范围应分别实现。mask限制的是直接读取，A/Z联合attention还会交换其他token的信息，不宣称各horizon完全隔离。

## 5. 时钟、频率和动作接续

### 5.1 初始运行参数（待验收）

当前native profile写100ms，但并未证明原始数据和连续执行每行实际100ms；其他路线的0.15/0.16秒不适用于本候选。首先确认动作命令的物理作用区间，再统一训练与部署网格Δ。

| 项目 | 初值 |
|---|---|
| 动作块 | 16行 |
| 快查询周期 | 2Δ；之后受控扫描1/2/4Δ |
| 快查询保护前缀 | 2行；要求端到端返回和提交均早于到期时间 |
| 慢查询周期 | 4Δ；固定刷新，每路最多一个inflight，过忙时跳过并记实际周期，不排积压请求 |
| 快采样 | 10次积分，另有contact读出成本 |

若Δ最终确认0.1秒，以上对应动作目标10Hz、快路5Hz、慢路2.5Hz。这只是待测初值，不是当前性能。若快路延迟加提交裕量超过2Δ、慢路无法在4Δ完成，该配置不成立；下一配置需重验并记录，不能自动降低世界速度或冒称原频率。

没有跳过刷新且请求正常完成的稳态下，慢参考最大年龄约为Pslow+Lslow；若希望最远Z仍获得未来参考，需要 `Pslow+Lslow+0.8 <= Hslow`。跳过刷新后年龄可能更大，因此还必须按实际source age检查独立TTL。超期参考失效；未超期但future不覆盖目标时刻时，对该部分屏蔽慢未来、由快路自行预测。Hslow来自真实codec/目标网格，不能通过重写时间戳扩大。

### 5.2 数值例子：假设Δ=0.1秒

快查询在1.00秒开始；已有队列包含合法动作。将1.00、1.10两行锁定至1.20秒，输入模型作为known prefix。新计划在1.14秒返回。

| 动作作用区间 | 返回后的执行来源 |
|---|---|
| [1.00,1.10) | 原队列，已执行/执行中，不变 |
| [1.10,1.20) | 原队列，已承诺，不变 |
| [1.20,1.30)及之后 | 新计划第2行及后续，替换旧可编辑尾部 |

新计划不是从第0行重新执行，也不把整条新计划平移到返回时刻；每行绝对时间保持不变。不平均新旧动作。必要的边界平滑通过同一前缀条件学习并检查，不能部署后任意平滑导致动作/Z含义改变。

同一query若未在 `protected_until−提交裕量` 前完成原子提交，默认拒绝；不能把1.20秒恰好返回视作安全，避免与执行tick竞争。原本未作为known condition的旧尾部可能已经执行。当前实现只检查还有无后缀，CPU反例确认会错误接收部分这种结果；裁掉过时行不能消除动作历史不一致。

### 5.3 运行时合同

队列是承诺动作的唯一拥有者，慢快路均读不可变快照。快query绑定 `episode_id, clock_id, query_id, observation_id, base_queue_version, protected_until, prefix_hash, reference_version`。在短锁内原子校验并替换尾部；不得持队列锁做模型前向。

合法版本变化（时间推进）不能简单按“版本不同”拒绝；应检查查询所属episode、是否被新查询取代，以及其保护的动作值/时间是否仍完全相同。首版每路一个inflight简化竞争。过期query默认拒绝，保留错误证据；旧队列后续执行或终止属于显式统一的故障协议，不能隐式hold/重放/暂停世界。

已承诺前缀在模型中归一化，但最终提交时直接保留队列原始值，避免归一化往返浮点误差改变硬动作。记录动作目标时间和真实提交/生效时间，区分command target与实测机器人状态。

慢query只锁定source时刻已有承诺快照，不因为计算慢就承诺整个慢计算窗口。完成时记录真实completed时间，按版本整包原子发布。快query读取最近已完成包，一次采样期间不混用两个版本。新慢包不能改变机器人队列。保存source以来实际已执行动作，与旧软参考按绝对时间对齐。

物理仿真时钟决定动作和预测时间；单调墙钟计性能/超时。通过明确映射关联二者，不能用系统日历时间或帧序号代替。物理时间用FP64或整型tick；送网络的是减去source/query后的相对时间。相机曝光时间与可用时间分开，只有在query时已到达的数据能进条件。

## 6. 数据准备

### 6.1 底座与缺口

已有.92 `domino_absolute_motion_v2`快照包含35任务、10,500 episodes，原生VAE/语言/动作/motion材料；这不等于新的DualRate训练集已经准备好。当前 `real_labels_prepared=false`，实体ROI/位姿/接触字段是否完整尚待逐样本核实。先导出任务、scene、episode、style、动力等级、时间覆盖的清单，确定L1训练预算，不能把目录名称或缓存总量当作协议证据。

来源路径：`/SSD_DISK/users/wuruihan/wam/datasets/domino_absolute_motion_v2`、原转换数据`/SSD_DISK/users/wuruihan/wam/datasets/converted/domino_exact_time`。新增标签/cache/manifests落在`.92 /SSD_DISK/users/wuruihan/DynamicWAM/outputs/datasets/`下新版本目录。

### 6.2 一条样本是(source s, query t)，不是一张图

```text
identity: task/scene/episode/style/split/clock/schema/encoder/stat hashes
source: s, source RGB/flow/state/text, timestamps, committed action prefix
availability: slow completion c, simulated-or-measured latency provenance
query: t, latest available RGB/flow/state/text, exposure/arrival timestamps
execution: physical action grid, hard prefix/mask, executed trace from s to t
targets: source video/action; query action16; Z16x134; contact16x2; field masks
```

要求s≤c≤t；source条件仅使用s时已知数据，query条件仅使用t时已知数据。例如s=.6,c=.9,t=1.0，则Z目标物理时刻为1.1/1.2/1.4/1.8秒；相应慢路参考应按这些绝对时刻读取。

离线预存原始数据索引、GT VAE latent、冻结标签特征等静态材料。主要联合阶段M必须由当前慢模型source-only纯噪声预测产生，并可反传；不能永久预存旧M冒充联合训练。慢路监督含加噪GT的那次forward只计算原生loss，不能拿来充当M。

一个source预测可供多个较晚query使用，共享可微图、平均loss；需控制显存。source sampling和query sampling分别绑定seed/cursor，完整恢复训练状态。

### 6.3 时间与监督制作

1. 先按scene/episode划分，同场景clean/randomized等style绑定同一split。未来几何/视觉统计仅训练集拟合；原生action/motion仍严格用初始化检查点统计，必须标注其来源。
2. 审计RGB曝光、动作命令作用区间、sim_time、状态采样、接触和实体identity。重复时刻需明确来源处理，禁止静默用frame_index/fps替代。运动使用原始量归一一次，现有packed统计与checkpoint不同不能直接再归一。
3. 先规定部署的视频/动作请求时间网格，再从真实记录配对GT。无法获得相应时刻的标签则mask；不能由训练未来记录反向决定部署reference的时间戳。记录codec实际frame→latent支持区间，原生8帧配置与2latent关系需实测核对。
4. 动作按命令语义重采样；hold目标和连续运动插值、夹爪事件分开处理。不得把qpos线性插值当drive target标签。
5. 视觉标签采用独立冻结ResNet18 layer3 teacher，实体区域ROI池化256维，再固定投影到128维。投影可由训练集PCA确定并冻结；teacher不与正在优化的当前编码器共享参数。模型权重、预处理、ROI规则、PCA与统计各存hash。
6. 位姿从仿真实体取得并统一坐标，相对夹爪位置用同一物理时刻计算。接触只取真实记录；缺失标未知，不填负类。若原数据缺失，通过验证过的精确重放或补采产生，不猜GT。
7. 图像标签优先实际帧/有容差的配对并记录误差；不把两帧特征线性插值称作真实目标帧编码。几何插值限同实体/有效双端/最大间隔，接触跨跳变不插值。遮挡、出视野、缺失实体按字段mask。终止附近保留可用短horizon监督，但当前代码仅有Z/contact字段mask：动作/视频不足的窗口须先新增并验证对应loss mask，不能填零参加现有loss；修复前只能保留动作/视频完整、部分Z缺失的窗口。

### 6.4 时延和前缀分布

零年龄s=c=t仅作机制预热。随后按slow计算延迟和固定刷新周期生成可到达的source-age/query组合；首轮合成分布明确标记，待真实单卡/双卡端到端延迟采样后替换。不能只均匀采样age而不考虑完成时间。

专家离线训练的known prefix必须取同一真实专家轨迹；query时的prefix与source时prefix分别构造，后者只包含当时已承诺部分。75%联合A/Z生成、25%给定专家动作预测Z可作为当前起点，比例仍需验证。动作条件模式不训练动作预测loss。

模型旧动作替换专家prefix后，原专家future不再自动正确。策略分布训练须实际执行/重放动作，到达对应状态后再产生未来和后续动作标签；失败、滑脱、恢复数据单列。错误动作不能配原轨迹同索引标签。

训练顺序：标签与可视化核验→短暂冻结慢路预热新增模块→小年龄联合→匹配实际age/latency的跨时间联合→闭环错误/恢复数据→再做采样蒸馏和物理加速。原生完整checkpoint初始化慢路；快路和接口新初始化，当前编码器用预训练视觉权重；阶段间完整继承本方案训练状态，不能每阶段回到旧DynamicWAM。步数和batch需根据真实数据/计算量确定，旧steps=4是工程smoke。

## 7. 43.54分的比较边界

[TrajWAM论文](https://arxiv.org/html/2610.25763v1)报告SR43.54、MS54.07；其设定为L1 clean、α=.1、35任务、1,750演示，测试每任务100次；DOMINO仍同步，预测32动作执行8。主表使用SAM3历史mask；附录还说明拦截任务使用仿真contact关闭轨迹adapter。其延迟在5090测，DOMINO在A800跑。

以上均为论文报告，未在本项目复现。将其作为明确目标，同时记录训练数据量、输入信息、执行步数及同步/连续差异。当前10,500条缓存与旧35×20连续评测不自动形成同协议SOTA比较。保持原生同步与连续1x两张结果表；对超越43.54的结论使用对应协议、足够样本和逐任务结果。

## 8. 实现顺序与验收

1. 先修query lease/迟到拒绝、时空metadata、无效Z输出，以及真实时间/codec producer；针对已复现错误做测试。
2. 加native显式特征返回和新readout；开关关闭时原生输出不变，开启时source-only无未来泄漏，主联合梯度到Video/Action/motion。
3. 替换当前视觉编码、加空间motion和分源cross-attention；记录实际参数/token数及内存，检查小物体与padding。
4. 小规模真实标签overlay/时间线验收和完整模型GPU工程smoke后，再确定正式recipe、八卡计划及连续1x闭环。

本提案没有宣布上述变更已经实现，没有启动GPU，也没有修改.50/.100或.92现有代码、权重、训练进程。正式实现需在.92候选新分支保留原V1，遵守harness及用户当前授权范围。
