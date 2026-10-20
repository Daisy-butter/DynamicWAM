# 动态物体 WAM 项目开发与实验约定

版本：v1.12 · 2026-10-20 · 目标投稿：CVPR 2027

本文件是后续模型、推理、训练及评测改动的共同约束。每次开始工作先阅读本文件及仓库 AGENTS.md，检查实际代码、配置和运行状态，再实施修改。新的人类指令优先；发现冲突应说明并记录。本文件不表示已完成远程代码审计，不保证模型达到 SOTA。约束通过仓库入口、启动前检查和运行清单落实；仅有文档或入口文件不能保证未来每次操作自动合规。

## 1. 目标与研究主线

首要工程目标：在采用连续仿真改动的 DOMINO Level-1 评测上建立可复现的强基线，并实现领先性能。论文目标：解释低延迟响应与动力学建模如何共同提高动态操作能力，形成可迁移的大框架。

工作聚焦三个方向：

1. 物理与系统加速：CUDA Graph、KV Cache、torch.compile 等，降低真实闭环延迟。
2. 分频加速：区分慢速语义/世界表征更新与快速状态/动作响应，明确信息刷新、缓存失效和跨频率交互机制。
3. 动力学信息高效融入：研究运动历史、状态变化与未来预测如何进入模型并影响动作，避免只堆叠额外模块。

三者应服务同一条闭环，不以绑定 DynamicWAM 或 AHA-WAM 的局部技巧作为最终框架。先形成能训练、能稳定评测的完整候选，再用消融解释收益；不在缺少基线和诊断时同时叠加多个未经验证的复杂模块。

## 2. 两条实现路径与当前任务边界

| 路径 | 主机与主要 codebase | 用户提供的当前起点（待现场复核） | 后续顺序 |
|---|---|---|---|
| Route1 | 10.12.208.50：/home/wuruihan/DynamicWAM | DynamicWAM 已有动力学基础和初版分频实现 | 先接入并验收连续仿真，再推进分频加速，随后物理/系统加速 |
| Route2 | 10.12.208.92：/home/wuruihan/AHA-WAM | AHA-WAM 已有分频基础；原架构 DOMINO Level-1 微调 baseline 正在运行 | 先完成 baseline 微调及评测，再接入动力学信息和进一步物理/系统加速 |

两台主机 /home/wuruihan 下的 DynamicWAM 和 AHA-WAM 均属于工作范围，表中是各路径主要开发位置。10.12.208.92 的 DynamicWAM/DOMINO 连续仿真实现是 Route1 本次迁移的参考源，须现场查明真实仓库、工作区与启动入口。Route2 baseline 的代码、配置、进程和已生成产物须保持可追溯；不得为了 Route1 检查而抢占其 GPU、停止其任务或原地修改其加载代码。

**Route1 连续仿真接入里程碑已由用户验收**，冻结提交 `b5907680bf4d713a2cce87d4c95b92a0561ff2f4`，标签 `route1-continuous-domino-v1-20261013`。下面的迁移步骤保留为历史验收规则。2026-10-13 用户进一步授权：在独立分支完善分频、运动上下文修正、物理加速及 RTC 训练一致性，完成 smoke 后开始正式训练；正式训练短时稳定后停止交互巡检，不创建周期巡检。冻结基线不得移动或覆盖。Route2 的原有边界不变。

已完成的接入里程碑按以下顺序验收：

1. 记录 .92 源实现及其 DOMINO 依赖的绝对路径、Git remote/branch/HEAD、dirty 状态、启动命令、协议参数和关键文件 hash；有未提交依赖时保存可还原补丁/快照，避免只记录 HEAD。
2. 在 .50 保存现有工作状态，建立独立分支或 worktree，以可审查补丁接入相同连续仿真语义；核对 .92/.50 差异及路径/设备适配，不复制运行产物覆盖现有结果。
3. 选择少量代表性 task 和 episode，先验证世界时间、目标运动、推理等待、动作保持、事件判定与观测时间戳，再核对原 DOMINO 报告结构、文件名、目录层次和 MP4 保存方式。
4. 输出小规模评测完成状态、每个 task/episode 的结果与可直接定位的 SSD MP4 绝对路径；明确这只是接入验收，不据此宣称性能提升或 SOTA。
5. 等待用户查看 MP4 并确认无误后，才将本版标为已验收并建立正式基线提交/标签。等待期间可保留工作分支、候选提交及运行快照，不能代替用户验收。

### 2.1 SSD 存储硬约束

**数据集、eval 结果、MP4、模型权重/checkpoint、训练输出、下载/编译/模型缓存等大文件，一律放在各自主机经过验证的 SSD 盘上的 wuruihan 目录树内。禁止回落到 /home/wuruihan、系统盘或未验证的默认缓存目录。** 源代码和小型 Git/实验索引可保留在 home 下。

每台主机单独确定并记录 SSD_ROOT；路径可能因挂载不同而不同，不能把 .50 的路径猜作 .92 的路径。启动前核验 realpath、实际挂载/设备类型、目录所有权和可写性、空间/inode/配额，确认解析后的目标仍在已批准的 SSD wuruihan 目录树内。SSD 不可用或余量不足时停止新任务并报告，不自动改写存储位置。

推荐在 SSD_ROOT 下按 route/run_id 划分 runs、eval、checkpoints、cache、tmp，并通过启动配置/环境变量显式设置实际会用到的输出路径（例如 HF_HOME、TORCH_HOME、TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、TMPDIR/TMP/TEMP）。逐项检查框架默认输出、下载缓存、视频编码临时文件和软链接最终落点；迁移已有大文件需单独规划，不能移动正在被 baseline 使用的文件。SAPIEN 的共享初始化锁仍须遵守第 7 节，不因按 run/worker 划分目录而割裂锁域。

不得根据历史记录推定当前位置、训练完成或 GPU 可用。登录凭据不写入本文件、Git、命令日志或实验报告。

历史对话是定位线索。每个关键事实标明来源：用户陈述、论文作者声明、源码事实、实测结果、推断或未知。未读完整的对话不得声称已全面了解；旧记录中的状态与指标必须在使用前核实协议和产物。

## 3. 相关工作与架构决策

以下为用户指定的调研清单，不代表已经核实各工作的机制或效果，也不是封闭名单：

| 方向 | 起始清单 |
|---|---|
| 分频设计 | AHA-WAM、HiRT、DuoCore-FS、TIDAL、Helix |
| WAM 加速与实时推理 | Flash-WAM、EfficientWAM、FastWAM、Realtime-VLA、Realtime-VLA-V2、RealtimeWAM、RTC、VLASH |
| 动态物体操作 | DynamicWAM、DreamVLA、DynamicVLA、DOMINO-PUMA |

做架构决策前查询论文、官方代码和配置，确认准确名称、版本和公开状态。以当前架构的瓶颈、时序依赖、可训练参数和部署预算为起点，广泛比较相关机制，再组合设计最适配的方法；不限定为把另一篇工作的模块直接迁入 DynamicWAM 或 AHA-WAM。两条 Route 可共享被验证的设计原则，具体接口与实现按各自 codebase 决定。

维护精简 related_work.md 或同等现有记录：只登记实际用于决策的来源 URL/版本/访问日期、机制、训练代价、推理条件、可复用思想及适用限制。不存在或无法定位的条目标为未核实，不编造引用。调研围绕当前可执行假设开展，不能以无边界综述推迟代码和实验。

每项设计写一份简短决策记录：问题与证据、候选机制、选择理由、计算/显存/训练成本、最小验证、淘汰标准。重点比较慢快通路交换什么信息、谁触发刷新、是否需要重训、缓存为何有效及何时失效，而非仅比较名称或报告的 FPS。

## 4. 评测协议：先冻结，再比较

原生同步 DOMINO、连续仿真同步执行、连续仿真 RTC 是不同协议，结果必须分表并写明条件。连续协议上的领先结果不能直接宣称超过原生 DOMINO 官方榜单；SOTA 声明必须限定协议，并以同协议、可复现的强基线支持。

每个正式评测冻结 protocol_id 及以下内容：仿真器提交/补丁、物理 dt、目标轨迹时钟、控制周期与动作语义、相机/观测频率、历史采样、推理与执行关系、动作块长度、RTC 拼接规则、超时/终止/成功定义、机器人与运动参数、任务/场景/seed 清单、评分版本。

连续仿真必须具备：

- 推理期间物理世界与外部目标继续推进；目标运动由物理时间驱动，不能随策略调用次数加减速。
- 分别记录墙钟、仿真物理时间与动作索引；明确保持动作、过期动作、队列耗尽、延迟提交及取消语义。
- 观测采集、提交、执行均有时间戳；执行不能使用在该时刻尚不可得的观测或推理结果。
- 等待期间持续检查成功、碰撞、越界和超时事件；不因分频或 RTC 改变任务判定。
- 监测 realtime factor、积压与丢帧；预先声明容忍阈值、异常处理及计分规则，不能默默放慢世界以迁就模型。
- 仿真与推理共享 GPU 时记录资源争用；比较模型使用一致的资源预算与调度条件。

真实墙钟延迟与人为注入延迟分开实验，不混成同一结果。更换仿真器或协议必须分配新 protocol_id，并重新建立相应基线。

## 5. 数据、训练与比较公平性

使用与目标模型格式对齐的 DOMINO Level-1 数据；核实机器人、动作空间、时间间隔、归一化、相机和切分。不得混用历史版本或其他等级而不声明。记录数据清单、预处理版本与校验值；训练/验证/正式测试隔离，测试结果不能反向选择测试场景。

训练记录基础权重来源与 SHA256、可训练模块、损失、学习率、batch、精度、步数、随机种子和恢复方式。跳过某阶段必须证明输入输出接口与所载权重兼容；明确区分初始化训练和恢复训练状态。

正式任务/seed 清单先冻结并供各方法配对复用。准备阶段筛选仅依据预先定义的环境/专家可行性，不能依据候选策略成功率筛选。基础设施故障与策略失败分别记录；保留所有尝试及重试映射，不静默剔除失败 episode。

先完成小规模冒烟与资源验证，再扩大并发及正式评测。完整 Level-1 清单以协议文件为准；若采用 35 任务×20 episode，应核实 700 个有效记录，并明确这是一轮测试，不能代替跨种子不确定性分析。

## 6. 加速和动力学模块的验收

系统加速先验证数值/行为等价及生命周期正确性：CUDA Graph 的形状、缓冲区地址、随机数与并发；KV Cache 的位置、条件变化、跨 episode 清理及失效；compile 的重编译、图断裂、预热与回退。缓存跨观测复用可能改变模型语义，必须作为算法变化独立消融。

RTC 在训练前确定具体方法及部署契约，不能把连续仿真自动等同于 RTC。若采用训练时 action-prefix conditioning，训练与推理使用同一动作归一化、clean/noise 时间方向、逐 token 时间、prefix 上限、suffix 损失及队列时间语义；超出支持范围须显式拒绝。原始 RTC 的推理期 guidance 与训练期 prefix 方法分开记述。未完成实际 controller 接线与时序验证的权重不得通过旧 controller 静默忽略 prefix；显式 d=0 消融单独标记。减少 denoising 步数需适配或蒸馏及验证，不因 CUDA Graph 支持而任意更换训练采样器。

分频设计必须明确慢快频率、共享状态版本、最大允许信息年龄、更新触发、背压与线程/进程所有权。动力学模块必须明确监督来自哪里、训练与推理分别使用什么信息、历史长度及辅助损失；特权状态只可用于明确声明的监督或诊断，不得泄漏到不允许访问它的策略。

至少报告端到端观测到动作生效延迟及 p50/p95/p99、模型推理延迟、首次编译/预热成本、稳态吞吐、显存与闭环成绩。延迟定义和计时边界固定，GPU 异步操作使用正确同步或 CUDA events。不能用吞吐或无同步的 Python 计时替代真实动作响应。

以可运行基线为起点，分别验证系统加速、分频、动力学，再测试组合与交互；比较训练数据/计算预算。对延迟改善但成功率下降、辅助预测改善但闭环无收益等结果保留并解释。先声明开发集晋级条件，最终测试避免反复调参。

## 7. GPU 卫生：强制禁区

用户提供的事故截图描述了驱动解绑、PCI remove、GPU reset 与驱动重新加载后设备不可见的情况。这是安全约束的依据，不是当前主机健康结论。

**本项目代理和脚本禁止执行 GPU reset、驱动 unbind/bind、PCI remove/rescan、驱动模块卸载/重载、设备级恢复及为恢复 GPU 而重启服务器。禁止把这些动作放入自动修复、cleanup、重试或 sudo 脚本。故障交由管理员处理。**

启动前只读检查 GPU UUID/PCI、可见卡数量、驱动状态、显存、温度、功率、活动 PID/归属，以及磁盘空间/inode/配额。缺卡、Xid/驱动异常、异常温升或占用不明时，暂停相关 GPU 的新任务，保存证据并报告；不能默认改用剩余卡掩盖异常。

2026-10-14 本次 Route1 的明确授权补充：用户在知悉历史 IOVAS 断言后要求“仔细查看并解决，注意一点不要造成GPU掉线即可。开始正式训练”。允许在固定主机、boot ID、历史日志 hash 和 GPU UUID 的证据上，逐卡进行有时限、小显存的应用级 CUDA 验证，再依次开展单卡、多卡 smoke 和正式训练。此授权不是管理员健康声明，也不表示驱动根因已修复。每次扩容须有对应设备的实际通过记录；新增驱动异常、缺卡或测试失败即停止扩容。训练启动仅设置有限的只读健康检查窗口（本版 600 秒），异常只终止本 run 登记的进程组；窗口结束不再查询 GPU 或另建巡检。设备恢复禁令仍完整有效，不能把本次历史日志授权用于其他 boot、新异常或其他设备。

按 UUID 固定设备并记录 PCI 对应关系，不能依赖重启前后的编号。只使用已确认可用且被授权使用的资源；先小并发验证再扩容。设置并记录显存、温度、积压、超时和最大重试阈值，由具体硬件及平台约束确定，不臆造统一安全数值。

2026-10-14 后续明确指令：用户在获知双卡退出新增 IOVAS 断言后，要求“再尝试一下，从本地电脑连ssh启动训练”。该指令授权重新从本机 SSH 检查并进行新的受控启动。本次采用单可见 GPU、梯度累积 14 的隔离配置，保留有效 batch 14；重新生成实际 CUDA probe、包含退出后驱动检查的 smoke 和严格重载证据，再启动正式训练。建立新的授权收据绑定此次已观察的日志，不复用旧收据，不声明驱动根因修复；不重复已失败的多卡路径。若隔离测试也新增异常则停止本次启动。此补充取代上一轮“等待用户授权”的状态，禁止设备恢复和不创建后续巡检的约束继续有效。

SAPIEN 初始化需要共享锁时，使用同主机、同锁域的绝对锁路径；验证所有 worker 的 TMPDIR/TMP/TEMP 确实指向相同共享锁位置。编译/模型缓存可按 GPU 或 worker 隔离。锁域和并发策略必须实测，不能靠无限重试处理设备上下文冲突。

最新修正（2026-10-14）：用户明确拒绝单卡正式训练，要求 GPU 正常时使用 **8 卡并行**，并要求排查自编多卡训练代码。该指令取代上面的单卡启动计划；单卡配置仅保留为未启动的历史备选。允许逐项、有时限地比较原生 CUDA、多卡通信与实际训练的启动/退出路径，并在证据支持后修正用户态代码或通信配置。每个测试独立留证，不在失败后自动循环、直接扩容或复用过期的健康收据；若出现缺卡、Xid 或运行中持续驱动故障，则停止 GPU 测试。正式训练必须通过对应 8 卡 smoke、退出后检查和权重重载。仅修正退出时序或通信路径不等于证明驱动已修复。8 卡每卡 batch 1、累积 2 的有效 batch 为 16，应如实记录，不能仍标为 14。

2026-10-14 资源释放路径补充：用户进一步授权结合最小复现检查驱动释放路径。已验证的应用级候选由本 run 的父进程临时打开选定 UUID 对应的设备句柄，在全部 worker 退出后显式关闭，最终关闭后的驱动检查也必须通过。它不修改系统 persistence、驱动或 PCI，不以长期持有句柄掩盖退出故障。启用须绑定实际通过的设备清单、boot/日志身份及 guard 源码 hash，smoke 与正式运行使用同一 guard 设置和实现；异常路径也要清理本 run 的句柄及进程。2 卡和 8 卡最小对照通过只支持这一生命周期顺序，不代替真实 8 卡训练 smoke、严格重载或长期稳定性证明。设备恢复禁令保持有效。

只终止本次 run 明确登记的 PID/进程组，优先优雅退出，再对已核实的本任务进程升级处理。禁止广泛 pkill、killall、按进程名清场或干扰其他用户。OOM/初始化失败采用有上限重试和退避，重复故障触发停止扩容及人工检查。

## 8. 长任务与结果完整性

训练/评测由持久后台 supervisor 管理，断开 SSH 或结束聊天后仍可观察；不依赖当前会话持续轮询。记录 run_id、主机、PID、启动命令、GPU UUID、日志路径，维护原子更新的 status.json、心跳、各 worker 退出码和终止原因。监督进程不得做驱动恢复。

正式结果保留 DOMINO 兼容报告和实际相机 MP4，视频与 episode 一一对应，不能用预测画面替代。当前连续仿真接入验收沿用 .92 来源的原生视频语义：已有 head-camera 采样各编码为一帧，按 10 fps 帧序播放，sidecar 保存每帧真实仿真时间及采样墙钟。**原生 10 fps 播放不等于物理时间 10 Hz 连续录像，MP4 播放时长不代表 episode 物理时长，必须在清单/报告中明确声明。** 不为迁就播放时间而伪造或重标采样时间。若后续增加物理时间对齐视频，另存为明确命名的附加产物，不替换原生文件。保留原始格式及必要别名，额外指标写附加文件；既有结果不覆盖。全部结果及视频落在该主机 SSD_ROOT 的 wuruihan 目录树内。

“与原 DOMINO 一致”须依据实际原生评测代码或已验证样例确认：报告字段/类型、成功及分数口径、任务/episode/seed 对应、目录层次、文件名与 MP4 编码/帧率/视角/时间含义；附加诊断不得破坏原解析器。MP4 应在成功、失败和异常终止时按既定规则正确收尾，清单列出实际路径、状态、大小、帧数和时长。小规模验收至少检查 ffprobe/等价工具可解析、完整解码无错误、抽帧内容正确及对应 episode 时间范围；最终视频观感由用户确认。

完成必须同时满足：任务/episode 清单对齐、退出状态已核实、终止标记可信、报告可解析、视频可解码且覆盖对应轨迹、重复/缺失检查通过、独立重算汇总一致。后台进程启动、局部测试通过或出现 summary.json 均不足以宣称评测完成。

报告至少含 SR、manipulation score 的确切聚合方式、逐任务值、样本数与不确定性、终止原因、控制跟踪误差、延迟及仿真积压。区分策略能力、控制执行误差与基础设施失效，不能把所有低 SR 归因于模型。

## 9. Git、版本与实验溯源

每次修改前记录 hostname、仓库绝对路径、remote、branch、HEAD、git status、现有 diff 及活动任务所用代码。已有未提交修改先保留并厘清归属，禁止 reset --hard、清理未跟踪文件或覆盖他人工作。

每项独立假设建分支，例如 feat/<topic>-<date>、fix/<topic>-<date>、exp/<topic>-<date>；必要时用独立 worktree。提交保持可审查，代码、配置与对应说明同步更新；不只用 ver_01/ver_02 目录表达版本。

维护 CHANGELOG.md、实验登记表及必要的设计记录，可复用现有文件而不创建重复台账。每版写明父提交、目的、涉及模块、协议影响、验证结果、已知限制、产物位置及回退目标；一条精简记录链接完整 run manifest 即可。两台服务器迁移代码通过明确提交/补丁并核对 hash，禁止无记录复制覆盖。

正式运行使用固定提交或冻结源码快照。清单记录全部相关仓库 SHA、dirty diff/未跟踪依赖快照及 hash、环境/依赖/CUDA/驱动版本、配置、数据版本、权重 hash、启动命令、协议与 seed 清单。运行期间不原地修改其加载代码；新版本开新 run_id。

权重、数据、视频、缓存和凭据不提交 Git，以清单和校验值引用。回退通过切换到已知提交的独立工作区或 revert，保留分支、失败实验与日志；删除产物不是回退步骤。发布/远端覆盖/合并按用户授权执行。

## 10. 执行效率、交付与检查

token 和沟通篇幅有限时，优先完成可运行代码、必要验证以及持久后台训练/评测脚本；报告聚焦改动、证据、状态、路径和阻塞。无需在每轮重复全文规范、重读已定位资料或生成多份同义报告。不能为节省 token 省略 GPU 安全、SSD 落盘、协议/策略边界和结果完整性检查。未获授权的新架构改动或本步要求等待用户确认的事项，不能用“优先起脚本”绕过。

开始前：读本文件 → 核实仓库和运行状态 → 定义本次假设/成功条件 → 建分支并记录基线。

执行中：最小改动 → 必要的单元/集成及闭环冒烟验证 → 资源检查 → 冻结配置 → 后台训练/评测。测试重点覆盖时间因果、队列/缓存失效、episode 清理、异常退出及报告完整性，不只检查接口能调用。

交付时：说明改了什么、为什么、在哪个提交、如何验证、结果及限制、产物路径与回退方式。未完成的训练/评测明确标为进行中；未实测的架构收益明确标为假设。

## 11. 落实方式与版本管理

本文件在相关仓库根目录以 harness.md 提供，并在各根目录 AGENTS.md 中保留以下含义的明确入口；已有 AGENTS.md 内容保留、按需追加，不覆盖原规则：

> 修改模型、训练、推理、DOMINO 仿真或评测代码，以及启动训练/评测前，必须先阅读本仓库 harness.md，按对应 Route、GPU 安全、SSD 存储、协议边界、Git 溯源及验收要求执行。Route1 已验收连续仿真基线须保留，当前允许独立分支的分频与训练工作；Route2 先完成现有 baseline 微调及评测。新的人类指令优先，冲突须记录。

远程副本同步后核对版本和 SHA256，明确唯一维护源及各副本路径；未同步的仓库如实标为未覆盖。实际同步状态与运行情况写交付记录，不能仅凭本节文字声称门禁已安装。

在现有启动入口中增加或复用轻量 preflight，启动训练/评测前检查并将结果写入 SSD 上的 run manifest：

- harness 版本/SHA256、route、run_id、仓库/DOMINO 提交及 dirty 快照、策略权重标识与配置 hash。
- SSD_ROOT 及结果/视频/checkpoint/cache/tmp 的解析后落点、容量检查、GPU UUID/PCI/占用检查、共享锁绝对路径。
- protocol_id、task/episode/seed 清单、命令、日志、PID/进程组、退出码与产物索引。
- 本次允许修改的范围、必要测试结果，以及用户要求的验收是否仍待完成。

SSD 落点、设备检查或必要协议配置不满足时启动器应报错退出，不静默绕过或回落；预检和检查器本身只能使用只读 GPU 操作。不能用“已读 harness”布尔字段代替实际检查。先覆盖本次使用的入口，再逐步统一其他入口，交付时明确覆盖范围。

开发者在提交前审查 diff 与本次范围一致，运行前确认活动 baseline 未受影响；最终留存可还原代码与配置。变更本约定时更新版本、原因及受影响的 protocol/run，不能追溯改变历史实验口径。

v0.2 变更：明确 Route1/Route2 顺序与当前连续仿真迁移边界；强化 SSD 大文件硬约束、原 DOMINO/MP4 验收、用户确认后冻结版本；加入轻量入口和运行预检要求，减少重复记录。

v0.3 变更：登记 Route1 验收标签及最新架构/训练授权；补充 RTC 训练部署契约、部署拒绝保护与训练稳定后不再交互巡检的要求。此版本只同步本次涉及的 .50 仓库，不能据此声称 .92 已同步。

v0.4 变更：记录用户对固定历史驱动异常下受控验证及启动的最新授权；明确验证证据、逐级扩容、新异常停止和有限启动检查窗口。仅同步 .50 本次仓库。

v0.5 变更：记录双卡退出异常后的最新本地 SSH 重试指令，以及单卡隔离、保持 batch 和重新验证后启动的执行范围；不放行旧的多卡异常路径。

v0.6 变更：遵循用户最新 8 卡要求，撤销单卡正式启动计划，优先排查分布式代码与退出时序；明确受控对照测试和真实 8 卡验收。

v0.7 变更：记录新的资源释放诊断授权及临时设备句柄 guard 的证据、关闭和训前验证要求，仅同步 .50 本次仓库；不放宽设备恢复禁令。

v0.8 变更：用户授权 step-12000 checkpoint 的 8 卡、35×20 连续仿真 RTC 评测。沿用训练的 clean-prefix（0–4）、H4/R4、K10、慢路每 16 动作刷新及最大缓存年龄 12 的契约；部署时间间隔依据训练数据真实 sim_time_seconds 审计，不从视频容器 FPS 推导。当前校准名义动作周期为 0.16 秒，保留演示时间间隔非恒定的限制与审计收据。

仿真评测的应用级 guard 范围须覆盖全部可见 GPU，与实际计算 worker 的 GPU 清单分开记录。原因：SAPIEN/Vulkan 在选定 cuda:0 前枚举全部物理设备，单卡 CUDA_VISIBLE_DEVICES 不足以限制该路径。单 worker、仅 GPU0 guard 的场景准备在其余 7 卡留下 IOVAS 断言；同一单 worker 配合全部 8 卡临时 guard 已完成 2 场景准备及全部句柄关闭，退出日志无新增断言。两次证据均保留，不将此描述为驱动修复。正式 8 卡评测仍需实际闭环冒烟、退出检查及视频验证。

正式评测先独立冻结 35×20 专家可行场景，再按固定清单运行策略；任何阶段失败均停止后续阶段，不自动重试。RTC 队列按绝对动作时间网格检查，物理引擎浮点 tick 与该网格分别验证，不因微小累计表示误差改变动作周期或放宽 deadline。失败 episode 的已捕获 MP4 及时间戳另存并明确标为失败，不计入有效结果。启动后仅保留限定健康检查窗口和最终退出检查，不建立周期巡检。


v0.9 变更（2026-10-14）：用户明确要求修复 NVLink 锁/状态查询错误并重启 GPU 评测。原失败发生于 guard 打开前，精确来源进程未证实；gpumask 0x0 是锁获取输出，不能据此判定掉卡。新评测入口必须先通过 /proc 验证全部 8 张卡的 UUID/PCI/设备节点，再打开全部临时 guard FD，之后才允许执行任何 NVML/nvidia-smi 遥测；遥测串行，所有查询与 worker 退出后关闭全部 FD，再验收退出日志。授权及 manifest 绑定 device_lifetime_mode=fds_before_telemetry_v1 和不变的 helper SHA。先完成一次有界遥测生命周期验证，再实际 8 卡 RTC smoke（含 MP4 解码），均无新增异常才启动同配置 35×20 正式评测。复用原已冻结的 700 个场景并重验 1400 文件 hash，使用新输出目录，不覆盖失败证据。新授权绑定当前 boot/286 条历史匹配日志身份，不复用旧收据，不清空日志、不忽略 NVLink 错误。新增异常即停止扩容；不自动重试，不宣称驱动修复；启动健康窗口 300 秒，之后仅最终退出检查。禁止设备恢复及永久 keepalive 的约束继续有效。


v1.0 变更（2026-10-14）：用户要求修复实际 RTC deadline miss 并重新 8 卡评测。允许优化 CPU 光流计算和评测调度，不改变 checkpoint、H4/R4、0.16 秒周期、慢路刷新间隔、缓存年龄、deadline 或连续物理时钟。精确帧对光流缓存须逐元素验证，跨 episode 清空；OpenCV 线程数显式固定并记录。评测采用固定 cohort：全部成员完成场景/模型预热后才开始各自 episode；所有成员结束后再处理下一集准备，避免其他 worker 的初始化/销毁干扰实时闭环。等待发生在 episode 时钟外，不能暂停正在进行的物理世界；失败立即中止 cohort，保持原 seed/场景，不重抽。验收改用 8 个任务×2 episode、最多 64 动作的实际 smoke，包含上次失败任务、慢路刷新及跨 episode 生命周期；16 个有效结果和 MP4 完整解码、全部句柄关闭及驱动证据不变后才重新启动 35×20。保存各阶段采集/处理耗时，不把这轮改动声明为已修复所有时延问题。


v1.1 变更（2026-10-14）：保留正式03在action64真实迟到及驱动干净退出的证据。下一候选仅将原四相机Color读回/RGB转换批处理为原生实现，保持相机、float32乘255/裁剪/截断、RGBA有限值检查和独立像素所有权；不得改RTC时钟门禁。必须使用匹配SAPIEN的原ABI并保存旧二进制与构建依赖/参数/hash。先通过CPU边界值等价测试，再在已有camera warmup后对同一提交帧逐像素比较原生DOMINO与新reader；该验证不新渲染、不推进物理时间，结果写收据。后续8卡冒烟采用原正式首组8任务×2episode、完整原生动作上限，移除64动作截断以覆盖更晚的慢路刷新；16报告/MP4、实际像素等价及退出健康均通过才启动正式04。原700场景和所有失败目录继续保留，任何新错误不得通过放宽deadline处理。


v1.3 变更（2026-10-15，用户明确授权为完整评测作硬件约束让步）：新增独立固定降速连续仿真协议 continuous_world_scaled_wall_motionkv_rtc_h4r4_v1，本轮固定 simulation_rate=0.25。它取代本次评测的1x墙钟约束，不追溯修改历史1x结果，也不得改称实际实时达标。物理目标时间=真实墙钟经过时间×0.25；pace、积压、deadline统一使用该映射。真实wall/model/dispatch延迟原样记录。世界在相机/模型等待期间持续按该固定速率推进；不临时停世界、不适应性调速、不补造帧、不延长H4、不丢动作、不扩大action_budget。仿真内动作间隔0.16s、H4/R4、K10、refresh16、cache age0/4/8/12、原生物体速度/事件/接触规则、checkpoint全部不变。scaled报告明确realtime_valid=false、clock_valid/continuous_valid=true；保留缩放后的全局积压上限、H4截止帽及有限真实墙钟超时。旧strict1x入口默认不变。

本次新分支继承用户的b2ec6d2及相关评测补丁；保留formal35x20_06_no_pob和所有失败证据，不排除POB/PCP，重新完整35×20且同一冻结700场景。采用实际8卡同时cohort=8，先8个任务×2完整episode（包含POB和PCP）冒烟、16MP4完整解码和退出健康，再由一次性持久pipeline自动进入正式评测；任一验收失败不启动正式、不自动重试。允许延长与固定降速对应的任务/屏障墙钟超时，不改动作/episode数量。GPU临时guard先于任何遥测；最新旧run干净退出的历史驱动身份为307（此前286不是当前快照），新授权必须绑定现场相同身份。正式启动仅300秒初始健康窗口及最终退出检查，不创建周期巡检。

降速结果用于当前权重的连续仿真能力检查，不可与旧1xRTC、同步DOMINO或排除任务的部分分数直接比较；判断分频收益需要相同场景、相同simulation_rate的基线。训练clean-prefix来自专家/teacher，部署来自策略，且训练不模拟真实采集墙钟延迟，这些限制保留。


v1.4（2026-10-15）：用户要求旧0.25x评测继续运行，在独立SSD worktree实现下一版联合分频架构与CPU/可用GPU smoke。活动评测主仓库及共享external/runtime/data保持不变。新分支以1x连续物理时钟为目标；本次不启动正式训练。新student_joint桥、接口v2、异步因果采样、RTC前缀与视频/动作联合监督必须同步导出和严格加载；不得部署时恢复官方慢路覆盖联合权重。保留H4/R4/16动作刷新及前缀0..4，丰富训练缓存年龄0..15；增加的训练支持不授权改当前评测协议或假装实际时延已达标。GPU被旧评测占用期间仅CPU smoke；GPU实测必须在独立临时guard和现场健康/资源门禁下执行，禁止抢占或自动重试。

v1.5（2026-10-15）：用户要求下一次正式训练采用完整 general 架构改动，并先完善数据、模型、RTC、真实慢快异步、加速和 smoke。本次仍在独立 worktree 开发，不修改活动评测主仓库及 Route2，不直接开始正式训练。v3 使用独立协议与严格版本化接口：0.16 秒物理动作网格、H4/R4/K10、慢路每16动作触发、只读取已完成不可变缓存，最大训练/部署缓存年龄31动作及4.96秒；失效必须显式报错，不能等待时停世界。启动先在连续世界中建立慢路，再持有已知动作并从新观测产生首个可执行后缀，计入全部启动延迟。已承诺前缀不可变，新结果原子替换尚未执行的尾部；旧计划可作为训练和推理的软条件，不能未经训练直接混合输出。

v3 数据必须区分命令 drive target 和实测 qpos，固定物理网格标签不能由时间戳重标产生。原步幅编码的未来视频 latent 不得冒充新时间网格监督；新视频缓存需要可验证的输入索引、时间、源身份和覆盖记录。专家离线配对与真实策略轨迹回放分开，未采集的 on-policy 数据不得用伪造前缀替代。Video/Action/interface 的联合梯度、源视频监督及动态运动趋势辅助监督都要实测；保留原生 motion 信息，不将稠密光流、大模型或降采样步数当作未验证的默认改动。

v3 的验收分 CPU 算法/真实数据、完整 GPU 训练/捕获/并发、1x 连续仿真报告/MP4 三层。CPU 小 Video 替身不能替代真实 Video DiT 反传、显存和并发验证。真实数据缺失或 GPU 被占用时明确报告未完成门禁；不得将候选标成正式训练就绪。有限同行审查只能证明当前未发现阻塞缺陷，不能保证不存在更好的研究方案。

v1.6（2026-10-16）：旧 0.25x 评测已完成 700/700 并干净释放，SR=182/700、MS=41.8594；这是旧 step12000 的结果，不是 v3 联合架构结果。用户授权继续完善动作条件接口并做 GPU smoke，不启动正式训练。新 proposal_feedback_v1 在每个去噪步骤读取当前带噪动作、按物理时间对齐的源计划、当前队列和硬前缀，使用噪声可信度修正快路有效 K/V，不原地改写共享源缓存，不将新规划动作声称为已执行动作。该开关及运行契约须进入权重、严格加载和训推共同路径；旧 v3 无开关保持旧行为。

允许显式 bounded_real_grid_v1 真实数据子集用于最多四步、八卡 warmup/joint smoke；必须验证原数据索引、时间、标签、训练验证划分和重新编码的完整子集 latent，记录 full_scope=false / formal_training_ready=false。该子集不能用于正式训练，子集权重不能静默成为正式初始化。正式数据门禁保持全量覆盖要求。GPU smoke 仍须新鲜现场证据、全部八卡临时 guard 先于遥测、干净退出；失败保留并定位后才能进行有明确修改的下一次试验。不得将模型反传或 CUDA Graph smoke 作为 1x 连续闭环或收益证明。

v1.6 补充（本轮历史能力审计）：可选 source_plan_completion=same_noise_native_K10_tail_stopgrad_v1 必须沿已有两次联合采样的同一噪声轨迹完成剩余八次 Action 采样，不再运行 Video、不重置动作噪声。源 K/V 条件的早期 clean estimate 与最终 source_completed_plan 是不同字段；最终计划不是执行标签，超过原物理预测窗口必须屏蔽。训练和推理均采用同样的 stop-gradient 计划自条件，源 K/V 与 source FM 保留联合梯度。warmup 增加源缓存下的原生 Action FM（权重0.1），防止共享 Action 专家只适配快路而缺乏慢路监督；Video 仍冻结。此计算恢复不等于恢复 official 权重行为，step12000 初始化与 official 初始化的效果差异仍需同协议开发对照。


## 19. Route1 v3.3 正确Stage2起点与自动训练（2026-10-16）

- 最新用户明确要求核对原Stage2起点并整体检查后自动8卡正式训练。原链是Stage1 Video + `DynamicWAM_wo_motion.pt` Action + fresh absolute-motion -> `absolute_motion_init.pt` -> Stage2 -> Stage3。SSD三份源权重均存在且张量来源已核对。正式主recipe使用该step0组合权重，新接口/source-time fresh初始化；full和step12000仅参考/对照，不混入主recipe。复用已完成且兼容的Stage1，因此不重跑PCA/蒸馏。若改压缩/教师/投影契约或发现遗忘，再据证据处理。
- 原Stage2重训会新初始化motion；这与“继承full适配时禁止无意重置运动”是不同路线。不可将wo_motion直接伪装成绝对运动checkpoint，也不可把Stage3 full称为原Stage2起点。Stage3必须接本轮正式Stage2，4步smoke权重禁止作为formal init。
- 源动作精确物理端点贯穿2joint+8Action尾部及source FM。保留native motion/state/register。offline fixed_h4_v1与recorded_real_prefix_v1分别严格校验；bootstrap为slow-only+hold，稳态fast H4。合成延迟不是on-policy覆盖。
- raw replay v2使用同一绝对monotonic时钟，不猜测修补旧相对wall记录。失败轨迹不得BC；抓手/plan-switch按可编辑的相同物理端点度量。K/V有界不保证动作跳变有界。
- 全量VAE在独立冻结工作树、SSD及临时all-adapter guard下运行；10500 episodes/1416299点全覆盖与hash通过才放行训练。一次性流水线只在完整缓存、干净资源退出、正确初始化8卡smoke和严格full/fast重载全部通过后启动Stage2，再衔接Stage3。每阶段当时生成fresh guard授权，不复用旧清单；任一错误保存并停止，不自动重试。
- 正式初始预算Stage2 80k/Stage3 40k，有效batch128（8卡×每卡16×累积1），与原版batch128对齐；以验证曲线判断收敛。训练用已验证默认通信/allocator，不改驱动；仅初始300秒稳定窗口及阶段终态验收，不另建定时巡检。真实成功replay尚不足，不能自动进入BC refinement。


## 20. Stage3 memory correction (2026-10-18)

The user requests an actual recipe/initialization audit and repair of Stage3 OOM. Preserve completed Stage2 (8x16x1=128,80k) and all failed evidence. New Stage3 uses 8 GPUs x microbatch8 x accumulation2 =128, retains objectives/LRs/40k and full data, and initializes only the audited formal Stage2 step80000. Four-update diagnostic smoke must exercise this exact microbatch/accumulation and formal data/initialization, including backward, Adam state creation and strict export reload; it may not initialize formal training. Same effective batch does not promise bitwise-identical stochastic training. Fresh health/guard/source/config identities apply to each run; old failed run may release resources cleanly but is not successful training. No driver recovery, allocator experiment, automatic retry or recurring health watcher.

2026-10-18 correction: diagnostic warmup must be below its four-step budget; attempt01 failed before training with warmup1000 and closed cleanly. New explicit corrected attempt02 uses diagnostic warmup0/formal1000; status is /SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/stage3_memory_20261018_attempt02/status.json. Preserve attempt01 evidence.


## 21. Stage3 step20000 independent deployment on .100 (2026-10-19)

The user authorizes inference-only improvements in a separate .50 worktree, Git publication to Daisy-butter including the exact DOMINO source, migration of the completed Stage3 step20000 export and matching evaluation assets, and 8-GPU 35x20 evaluation on 10.12.208.100. The active .50 Stage3 training source, configuration, processes and shared runtime dependencies must remain untouched; no GPU evaluation or GPU probe on .50. Retain its original completed exports; immutable hardlink snapshots with hashes are permitted.

This evaluation uses joint-v3, fixed H4/R4/K10, 0.16-second action grid, refresh16, maximum age31 and 1x continuous wall-clock simulation. Normalize raw state/prefix/plan in FP32 before model precision conversion. Bind the complete 20k source-time architecture with strict full/fast reload and matching normalization metadata; never substitute the old step12000 or official full backbone. Preserve the frozen35x20 scene bytes and all35 tasks, native eval_result reports and actual MP4s. Cross-host model/path changes require explicit scene compatibility evidence instead of rewriting the old manifest.

Large files may be on verified SSD or HDD under wuruihan on .100 per the latest explicit instruction; this deployment selects SSD after capacity verification. Secrets are entered through a local masked prompt only and are excluded from Git, logs and receipts. User-approved kernel-log helper is restricted to fixed read-only dmesg requests, peer UID access and a finite lifetime; stop it after the evaluation terminates. No GPU reset, driver changes or recovery. Fresh local health, actual Graph/concurrency checks, 8-task x2 complete-episode smoke, MP4 decode and clean worker/guard exit gate the one-shot formal suite; no automatic retry or recurring inspection.


2026-10-19 RTC service-latency correction: preserve failed formal35x20_02 (POB episode0 late suffix), prior accepted smoke, and clean guard exit. This is an inference-only repair: bootstrap hard prefix uses held native drive targets rather than measured qpos; native drive-target model-state semantics remain training-consistent; measured qpos remains the physical trajectory anchor and tracking diagnostic. Replace the large Queue feeder with bounded single-inflight anonymous shared mailboxes and tiny Pipe notifications, preserving complete packet values. Bound child CPU threads, spread eight starts across distinct H4 phases before episode time zero, and record complete camera/IPC service timing. No H4 increase, time scaling, stale-suffix acceptance, task exclusion, model-output blending or training-source change. The remaining expert-versus-policy prefix distribution gap is not claimed solved. Revalidate actual graph/concurrency and eight-GPU full-episode smoke under a heavy cohort before a new formal attempt.

## 22. Stage3 step20000 camera-backlog restart (2026-10-20)

The user requests fixing the camera/backlog failure and restarting all35x20, permitting the historical hardware concession to obtain a complete result before further model changes. This supersedes section21's no-time-scaling restriction only for the new explicitly named fixed0.25 joint-v3 protocol. Preserve failed formal35x20_03 and its135 valid/1 invalid saved episodes; do not reuse them as results of the new protocol. Restart all700 frozen scenes in a new output directory on .100. Do not touch .50 training, checkpoint, model architecture, weights or normalization.

Use fixed simulation_rate=0.25 for the entire new smoke/formal run, never adaptive pauses/rate changes or automatic fallback. Physical dt0.004, action grid0.16, H4/R4/K10, refresh16 and maxcache31/4.96 remain unchanged. Scale pacing, wall deadlines, optional-history timing and bounded process timeouts consistently. Preserve actual wall/model/camera latency and original250ms physical-backlog validation. Scaled reports must set realtime_valid=false; clock_valid/continuous_valid reflect actual global clock limits. A completed scaled suite is a continuous-simulation policy diagnostic, not evidence of1x realtime performance or a like-for-like comparison with earlier1x results.

Do not move mandatory observation time or recompute the committed H4 deadline to hide camera delay. Record the maximum-backlog event phase and timestamps, and validate reports before allowing cohort members to initialize their next episode. Preserve reports/MP4s for invalid attempts. The same exact20k checkpoint, full/fast model interface, native DOMINO outputs and immutable700scene hashes apply. Fresh guard-before-telemetry preflight, actual Graph/background check and8GPU8tasks x2full-episode smoke with16 decodedMP4s and clean final guard closure gate the one-shot formal35x20 launch. Include POB and PCP in smoke. Stop on error; no automatic retry, GPU recovery or recurring inspection. Initial formal stability and final exit checks remain required; no architecture changes before this evaluation finishes.


## 23. Stage3 step30000 replacement (2026-10-20)

The user explicitly replaces the running step20000 evaluation with the completed Stage3 step30000 export. This supersedes section22's exact20k requirement only. Stop only the identified .100 evaluation supervisor through its normal cleanup; preserve every20k report, MP4 and interruption receipt, and never merge20k episodes into30k results. The .50 training process/source/configuration remain protected; read and transfer completed exports only.

Use the identical fixed0.25 continuous protocol, physicaldt0.004, actiongrid0.16, H4/R4/K10, source2+8, refresh16, age31/4.96, full35 frozen700scenes, normalization and output schema. The only policy artifact change is the exact30k full/fast/backbone weights, CPU-audited with identical runtime/data contracts. Rehash source/destination files and preserve learned source-time tensors. Fresh guard-before-telemetry health, actual30k Graph/background acceptance and8GPU8tasks x2full-episode smoke with16complete MP4decodes and clean exit gate a new formal35x20 directory. No retries or1xclaim. Initial-only CPU observer exits after300s health window plus first8validreports/8aliveworkers, or failure/timeout; no post-acceptance recurring inspection.
