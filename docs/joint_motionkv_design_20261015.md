# Joint MotionKV v2：因果缓存更新与联合训练

分支 `feat/route1-joint-motionkv-20261015`；父提交 `448782a3c97dd307348f304c9a2f88400af68047`。此工作树与正在进行的 0.25x 连续评测隔离，新部署目标为 **1x 连续物理时钟**。本文描述实现及待验证假设，不是性能结果。

## 信息与接口

慢路仍运行 DynamicWAM 原生两次 video/action 联合更新，发布第二次 forward 的逐层视频 K/V。与旧版本不同，新训练和部署使用相同学生 backbone，不再在加载后用官方 teacher 覆盖慢路。快路保留原 Action DiT、原生四个运动 token 和 10 步求解器。

接口 v2 保存源 RGB、源运动、缓存来源与年龄，以及产生 K/V 的同一次 forward 对动作的干净估计 `x_sigma - sigma * velocity`。后续动作条件变化由以下信息补充：

- 当前与源 RGB 的空间特征差分，保留局部位置；当前与源光流统计差分及有效性掩码。
- 按缓存年龄对齐的源动作计划、实际已提交的 RTC 前缀及二者差值。超过源计划末端的部分明确无效，不复制最后一个动作，不读取未提交后缀。
- 源采样 sigma、状态变化、两个缓存年龄，以及各动作相对当前观测的物理目标时间。训练从原始仿真端点时间读取；部署从控制器的目标时刻构造。

更新插在原生逐层 video/action attention 消费 K/V 之前：低秩路由产生有界 K/V 残差。旧观测 attention 与新信息 attention 分开，新通路以零门控加入；这使迁移已有非零修正头时仍保持旧更新器输出（整个策略的源桥已改变，不能据此声称完整策略等价）。第一次更新可学习门控，随后新信息分支可获得梯度。支持浅层、中层、深层和全层位置配置，默认全层只是候选，未作性能择优。

这些额外输入是“旧缓存之后发生的变化”和“缓存实际动作条件与已执行计划的偏差”，并非再堆一层重复全局 attention。运动统计仍是粗粒度信息；未实现稠密光流 token、目标检测或外部运动标签。

## Recipe 与数据

采用接口暖身 → 联合优化。暖身训练原生动作模块、运动模块和新接口；联合阶段解冻 Video DiT，使跨时刻动作 suffix loss 经更新器和 K/V 回到两次慢路 forward，并加入对齐的视频及动作 flow-matching 保留损失。VAE/T5 继续冻结。官方模型只作为可选动作 teacher，不产生新学生的源缓存。

初始配置为暖身 2,000 步、联合 18,000 步，分别提供 4 步 smoke；8 GPU、每卡 batch 1、累积 2。联合学习率 video/action/interface 为 2e-6/5e-6/3e-5，视频/动作保留项初值均 0.1。这些是待校准候选，未证明足够训练或最优。无需从零重复原 Stage1；联合阶段实际恢复世界模型训练。它不是沿用旧“冻结慢路 Stage2 后直接评测”的 recipe。

缓存年龄覆盖 0..15：每个源保留四个基础槽，epoch 内在槽内因果抖动，避免将配对数量扩大四倍。原数据、动作归一化及 episode 训练/验证划分不变；源未来视频/动作标签只进入独立监督命名空间，不能成为 bridge 的观测输入。索引含 10,500 episodes、1,420,472 原样本；完整时钟校验在实际读取时执行，已完成的真实数据 smoke 仅覆盖有限样本。

## RTC 与加速边界

仍使用训练时 clean-prefix conditioning：随机 d=0..4，前缀 flow time=0、只计算 suffix loss；部署每步保持实际已提交动作不变。H4/R4、16 动作刷新、连续仿真不变。教师辅助损失使用与自身目标一致的教师前缀及重新计算的接口上下文。专家/教师前缀与策略真实前缀的分布差距仍存在；没有用离线测试冒充 on-policy 训练。

新接口明确看到物理目标时间，避免把示范中可变的动作间隔无声当作部署固定 0.16 秒。它不意味着旧数据的分布差距已消除，也不意味着相机捕获延迟已被完整模拟。1x H4 是否能准时，需要实际 GPU 与连续仿真验证；CPU 通过不改变 deadline。

保留原 K/V 复用、固定尺寸 CUDA Graph 和原生批量相机路径。新增源视觉特征显式缓存：每次慢路刷新只编码一次，episode/权重变化不可沿用；训练禁止使用该推理缓存。Graph 的所有新元数据、前缀、目标时间都具有独立静态缓冲并逐次刷新。联合训练使用非重入 activation checkpointing。尚未实测 GPU 捕获、完整模型显存和提速；torch.compile/TensorRT、减少视频 token、K10→K4/K2 蒸馏不作为已完成能力，不直接降低采样步数。

## 参考依据及取舍

| 一手来源 | 用于本实现的判断 |
|---|---|
| [AHA-WAM](https://arxiv.org/html/2606.09811v1) | 参考逐层 observation-guided K/V 修正、随机相位训练、联合 video/action FM；未声称移植其 causal video mask、rolling memory 或 TensorRT。 |
| [HiRT](https://arxiv.org/html/2410.05273v1)、[DuoCore-FS](https://arxiv.org/html/2512.20188v1)、[Helix](https://www.figure.ai/news/helix) | 慢表征必须配合快路最新视觉；频率解耦不等于永远冻结慢路。采用联合优化和时间错位数据。 |
| [TIDAL](https://arxiv.org/abs/2601.14945) | 运动变化与快速执行相连；此处使用源/当前特征及流统计差分，未复制其运动预测器。 |
| [Training-Time RTC](https://arxiv.org/html/2512.05964v1)、[RTC](https://arxiv.org/abs/2506.07339) | 训练随机延迟、干净前缀及后缀监督；保留绝对动作时间控制器，避免引入每个去噪步的推理反传。 |
| [Flash-WAM](https://arxiv.org/html/2606.05254v1)、[Efficient-WAM](https://arxiv.org/abs/2606.10040)、[Fast-WAM](https://huggingface.co/yuanty/fastwam) | 世界预测质量与动作质量、不同模态的降步预算不能混为一谈；本轮先修联合表示，压缩/蒸馏另设训练与验收。 |
| [Realtime-VLA V2](https://github.com/dexmal/realtime-vla-v2)、[VLASH](https://arxiv.org/abs/2512.01031)、[DynamicVLA](https://arxiv.org/abs/2601.22153) | 把观测、动作生效时间和执行调度看作同一部署系统；记录时延、动作时间及缓存条件，不仅看孤立模型吞吐。 |

实验需比较相同协议、相同场景、相同预算的旧 MotionKV、联合慢路、三种创新信号及组合；再比较修正层位置与学习率。已有 41.72%、不完整 1x 和新 0.25x 分数不能直接归因于本架构。尚无新权重的 SR 或 SOTA 证据。

## 执行

当前已运行 CPU 合同、梯度、优化、导出/重载和真实数据 smoke；真实 Video DiT/CUDA/8 卡训练必须等旧评测释放资源。`scripts/run_joint_motionkv_smokes.py` 默认只生成命令计划，显式执行时要求 8 个 UUID、新健康证据、干净代码提交以及旧评测完整退出；先后进行暖身与联合各 4 步、逐阶段严格重载，失败停止，不自动重试、不进入正式训练。复用 FD-before-telemetry guard，避免旧训练 supervisor 的早期未保护遥测路径。具体结果见同目录收据。
