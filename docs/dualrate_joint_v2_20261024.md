# DualRate Joint V2 实现与验收

日期：2026-10-24。唯一修改远端为10.12.208.92。

项目：`/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-joint-v2-20261024`
分支：`feat/dualrate-joint-v2-20261024`，基于已保存V1 `3d5a9a8c4342acb896e30bfe2baebf4e09b5c26a`。
原始设计副本：项目 `docs/DualRate_DynamicWAM_Design.md`；逐项细化方案：`docs/dualrate_joint_refinement_20261024.md`；约束：`harness.md` v1.19。

## 实际实现

| 部分 | 实现与连接 |
|---|---|
| 慢路 | 完整官方DynamicWAM full checkpoint，988,810,702参数原生Video/Action/motion。监督前向独立于source-only可微2联合+8动作预测。真实原生time MLP明确FP32，其余Video BF16，不使用随机底座替代。 |
| 共享世界信息M | source patch tokens；第二次联合步末Video hidden拼接clean predicted latent的空间patch；最终A10软参考。独立投影至384，并附source/completed时间、类型、codec支持区间、xy与视角覆盖。共240+60+16=316token。不是跨周期原生K/V，也不声称早期video已条件于最终A10。 |
| 快路 | 384维、6层、6头，A16与16×134的Z联合FM；AdaLN、QK归一化。当前观测和慢路分别做attention，再用初始0.1的门控混合，避免大量慢token淹没新观测。一次query内预投影条件K/V，训练可微，禁止跨query误用。 |
| 当前观测 | 真正预训练ResNet18 stem至layer3、stride16；固定BN统计。head240×320、双腕各120×160后padding与mask；16个空间flow token、原生数值motion历史、drive-target状态和精确任务语言。 |
| 动作偏差 | 只比较真实已执行/已硬承诺动作与旧软参考的同一物理区间；不读取专家未来suffix。 |
| 未来Z | 目标、support、左右夹爪各4个时间点（query+.1/.2/.4/.8s），128维局部图像上下文+xyz3+relative3；接触单独2维概率。字段级缺标mask和永久可训练能力分开；不支持的字段在整个采样过程中保持固定噪声，最终输出无效mask。 |
| 队列 | 唯一硬承诺拥有者；query lease绑定episode/clock/query/版本/前缀hash/保护截止时间，原子验证后只替换可编辑尾部。保留队列原始动作，避免归一化往返改变硬前缀。迟到拒绝，整包慢路发布、TTL与单路单inflight。 |
| 时间 | 物理时钟FP64；action Δ=.18s，快2步、慢4步、保护2步为配置目标；实际完成时间另记，不能把配置频率当作实测速度。源时间处于已执行半步时，只允许覆盖该时间的首个已承诺区间，不允许过期/未承诺动作。 |

`runtime.py`提供持续时钟回调与非阻塞poll的快慢执行框架。当前尚未通过真实DOMINO异步接线和1x延迟验收；线程存在不代表GPU并行已经成立。

## 数据

SSD根：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/datasets/dualrate_joint_v2_20261024/full_01`

- 35任务、10,500episode、207,117 source窗口、382,734 source/query样本：train345,647 / validation37,087。
- 8,270个task+scene分组绑定clean/randomized划分；83,473个重复时间记录按last-record明确处理。
- 新网格动作.18s、source stride.36s、query age.36/.72s。原缓存只有1,319个完整旧窗口严格满足该新网格，因此从原始图像重新配对，主训练进程内冻结原生VAE编码5帧condition与1+8帧future，不拼接错时latent。
- arm drive-target有界线性重采样，gripper按离散事件前向保持；当前观测只取已经曝光的帧，未来监督取误差≤.030001s的真实帧并记录实际曝光时间。future GT时间误差仅审计，不进入预测条件。
- 每episode绑定实际任务描述/资产/左右手，1,516个唯一精确prompt，包含原生scene prefix；不复用task首episode语言。
- 真实数据提供目标/夹爪pose及目标接触标签；没有support pose或depth/seg/遮挡标签。未知项始终mask。视觉目标明确是32px投影区域局部图像上下文，可能包含遮挡物，不伪称“可见对象特征”。
- future统计只来自train；visual统计来自35任务的训练样本子集并记录count。action/motion沿用官方checkpoint归一化，包含其历史数据来源，不能称为本轮重新fit的train-only统计。
- 离线completion=.18s为显式合成假设，专家前缀是离线bootstrap分布；尚无真实部署延迟分布或on-policy replay。

## 检查清单（最终结果在本文件末尾补充）

- [x] 原始markdown与讨论逐项落到代码、元数据和配置。
- [x] 完整原生checkpoint严格CPU加载，真实time MLP前后向与精度修复验证。
- [x] 慢路采样与监督分离、快路损失向原生Video/Action/motion回传CPU验证。
- [x] 真实全量索引、scene split、时钟/动作/标签mask与3任务投影检查。
- [x] query lease、硬前缀、迟到拒绝、时钟、缓存版本及无监督字段采样回归。
- [x] 完整真实文本缓存与完整Dataset读取验收：1516prompt、3任务完整样本和验证样本。
- [x] 完整真实模型GPU训练smoke、有限loss/梯度、峰值显存。
- [x] 完整工程checkpoint严格重载、优化器/RNG/cursor身份一致。
- [x] 所有owned worker/设备FD退出、kernel日志无增量。
- [x] 最终提交/tag、原home源码与V1保留核验。

非本轮通过项：正式训练、八卡DDP、CUDA Graph、真实DOMINO异步闭环、1x延迟与成功率。`formal_training_ready=false`。

## 运行记录

初始冻结 `a6b8093423c2a36a516cf6de5bbcdaa0d75738b3`，170项CPU回归通过。
文本准备text_01在编码前因脚本误取原生配置的scene_prefix层级失败；实际prompt前缀正确。全部owned进程退出、guard关闭，kernel无新增。失败目录保留，不使用自动重试。

text_02从修复提交62821bf完成1516/1516编码，worker退出0、owned进程为空、显存0MiB、guard关闭、kernel无新增；即时postflight利用率仍100%导致严格检查失败，原始记录保留。监督器已加入退出后2秒固定冷却再作一次释放查询，不循环重试。

最终训练前CPU174项通过，完整数据reader59个tensor字段真实读取、batch拼接与时钟一致性通过；source原始HDF5/flow新增stat身份sidecar（路径/size/mtime_ns，明确不是内容SHA）并于每次读取验证。

训练smoke首轮源代码 `be9b081c9ba22015022ec02d9fef5df6302bc0f5`，4步、batch1、完整慢路与新模块联合优化；配置warmup0用于直接验证联合梯度，cache_dropout0用于确保每一步都验证慢路连接；此烟测配置不代表最终正式recipe。输出 `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/dualrate_joint_v2_smoke_20261024_01`；supervisor收据 `.../outputs/diagnostics/dualrate_joint_v2_20261024/train_01`。当前阶段只记录已启动，尚不称通过。

时间语义补充：future的两个区间是请求预测目标的分组时间窗；VAE可能依赖的输入帧分别0..4、0..8另行记录；DiT hidden经过全局联合attention，不能把分组时间mask说成所有间接信息路径的因果隔离。训练目前均匀shuffle合法row，长episode权重更高；没有伪称沿用原生每episode固定10样本的均衡recipe。

train_01完整模型GPU加载成功，但在VAE预处理、优化步之前退出：适配层误用了其他Wan版本的list API；当前原生Wan2_2_VAE要求5D Tensor。完整失败和参数统计保留，owned进程/guard全部退出，kernel无新增。修复将按真实原生encode/patchify调用链回归，然后新目录train_02运行。

实际模型参数：slow988,810,702；fast20,410,530；observation4,629,440（其中ResNet trunk2,782,784）；world readout1,668,992；合计1,015,519,664可训练参数。计数不包含独立冻结的VAE/T5预处理器；不再引用V1的15.09M快路估计。

数值训练修正：原生Video参数保留BF16前向，但直接BF16 AdamW会丢失小于其量化间隔的1e-6更新。候选改用FP32 master AdamW，保存FP32主权重、momentum和variance并纳入完整恢复检查；主权重初始化于已加载模型数值，不声称恢复加载时已舍去的低位。快路与原本FP32模块保持FP32优化。source硬前缀快照也绑定episode/clock，旧episode慢请求在模型前向前拒绝。

train_02源代码38897b4，181项CPU回归通过，真实VAE编码成功；进入原生联合注意力时因现有robotwin/lila环境缺少FlashAttention退出，未发生优化步。owned进程/guard全部退出，selected GPU显存0MiB、利用率0、kernel无新增。基础环境保持原样；后续使用匹配cp310/Torch2.4/CUDA12/ABI False的官方wheel，隔离安装在SSD，并绑定依赖代码/二进制哈希。


## 最终训练烟测验收

`train_03` 完成，冻结训练源码 `5e80b18207c0a27e66192c690624de8a0818b4dc`，source SHA256 `5133a2113bb0873f8039ec47aaffadba051f810d728d26d9506cf809864b9a94`。CPU191项全部通过。

| 步 | total loss | 未裁剪梯度范数 | 非零且有限梯度张量：slow/readout/observation/fast |
|---|---:|---:|---|
| 1 | 4.245500 | 19.376562 | 475 / 20 / 68 / 156 |
| 2 | 5.905666 | 28.707747 | 475 / 20 / 68 / 156 |
| 3 | 7.417605 | 38.251057 | 477 / 20 / 68 / 156 |
| 4 | 5.729122 | 30.112726 | 477 / 20 / 68 / 156 |

全模型、真实RGB/flow/pose/contact/T5与原生VAE参与训练；没有用tiny网络、随机视觉编码器或假latent代替。每步原生监督前向、source-only可微2+8预测、新快路联合A/Z前后向与FP32 master更新全部执行。训练峰值allocated为21.0948GiB（单张A6000），步骤耗时约2.80/1.93/1.91/1.84秒，不能将训练耗时当作推理延迟。四个不同样本的loss不是收敛曲线。

FP32 master更新样本每步10,571–10,637个值改变，BF16前向样本仅139–222个即时改变，证实master正在保留小更新；统计明确只采每个master前32个数，不能说扫描全部元素。

`last.pt`约12.88GiB，SHA256 `68be8a4e38fd953055ae18c5553106afc0b1157d7f8ed325d26d3d5ae7b89a96`。严格重载恢复step4，完整模型/优化器（含FP32 master和moments）/cursor/RNG/训练状态逐tensor字节SHA256一致，changed_components为空，未额外训练；重载验证47.55秒。

最终worker退出0，owned session为空，所有guard FD关闭，8卡postflight保存；kernel前后均无匹配异常、boot未变。Supervisor全程126.47秒完成，无自动重试或持续GPU watcher。旧text_01/text_02、train_01/train_02记录全部保留。

成功训练目录：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/dualrate_joint_v2_smoke_20261024_03`，包括 `training.jsonl`、`parameters.json`、`last.pt`、`reload.json`。
监督与硬件收据：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_joint_v2_20261024/train_03`。

现用FA为从本机已安装环境复制、核验RECORD的2.7.4.post1；不是下载备选的2.8.3 wheel。隔离依赖目录及59文件SHA在训练配置中绑定，启动/退出均复核，基础环境未改。

## 复现入口

在该worktree中，训练入口为 `python -m dynamicwam.dualrate.train`，正式/DDP模式仍拒绝。GPU必须通过 `scripts/guarded_dualrate_smoke.py`：

```bash
PYTHONDONTWRITEBYTECODE=1 external/robotwin/bin/python scripts/guarded_dualrate_smoke.py \
  --config configs/dualrate_joint_v2_smoke.json \
  --run-dir /SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_joint_v2_20261024/NEW_RUN \
  --gpu GPU-dd833f16-f4e8-f4b1-6a97-3f59083d487b --max-runtime 3600 --detach
```

上面只展示入口，不应覆盖现有run。再次运行前要使用新的配置output/诊断目录，保存配置并提交，再由guard执行新鲜空闲检查；成功smoke权重不得作为正式训练初始化。数据生成入口 `scripts/prepare_dualrate_data.py` 与文本入口 `scripts/prepare_dualrate_text.py` 均保留；全文本已完成，无需重复编码。

原 `/home/wuruihan/DynamicWAM` HEAD、dirty状态和94个已核对源码hash均未变；V1保留；本轮没有连接或修改.50/.100，没有改AHA-WAM。
