# DualRate Joint V1：实现与验收边界

2026-10-23；仅 `.92`；harness v1.18。严格依照用户提供的 `DualRate_DynamicWAM_Design.md`，原稿副本及 SHA 见同目录。当前为可审查的第一版实现，`formal_training_ready=false`。

## 1. 起点与结构

代码从 `.92 /home/wuruihan/DynamicWAM` 已提交版本 `dfbec6087fee8f23b1560f37f8374d75f6888df9` 新建独立 SSD worktree；不继承主目录未提交修改。分支 `feat/dualrate-joint-v1-20261023`。原目录97个已记录源码/配置文件、现有 AHA-WAM、共享依赖和运行任务保持不变。未连接或改动 `.50/.100`。

完整官方 `DynamicWAM_full.pt` 初始化慢路，SHA `7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`。不是 Stage1 初始化，也不是 .100 FullWan5B+LoRA 路线。实际原生 Video+Action+motion 共 988,810,702 参数，完整严格加载已通过 CPU 检查。

| 模块 | 本版实现 | 输入/输出边界 |
|---|---|---|
| 慢路 | 原生12层 compact Video + 原生 Action Expert + motion | 源时刻观测、历史、任务和已承诺前缀；2次联合采样后同一动作轨迹继续8步 |
| 参考读出 | 每个观察/预测 latent 时间片4×4空间token；额外软参考动作token | 预测latent、参考动作、source/completed/version、物理支持区间、生成阶段 |
| 新鲜观测 | 三视角 stride32 空间CNN、运动区间、状态、T5 token | CNN从零训练，明确不是预训练视觉encoder；T5/VAE冻结并使用离线特征 |
| 快路 | 384维、6层、6头 Transformer，每层A/Z联合 self-attention | cross-attention读取最新观测和时间有效的参考，联合输出动作及Z的FM速度 |
| Z | 4实体×4时刻=16 token；每个134维 | 128维冻结局部视觉特征+6维几何字段；接触为独立2维BCE，不混入连续FM |

快路主体15,091,734参数；观测编码2,339,616；读出33,024。宽度、层数、损失权重、stride及16 token均为待消融的起点，尚无速度或成功率收益证据。

跨路不传原生逐层K/V。K/V只在慢路自身后8步动作采样中使用，并通过独立子类去掉缓存末端 detach，保持预测梯度；不修改原生基类。视频第二步确实积分到最终预测latent。视频在联合第2步形成，参考动作在第10步形成，`video_plan_consistency_guaranteed=false`，不能声称两者严格对应同一个最终计划。

## 2. 联合训练与数据隔离

`system.py` 的一次训练包含两条独立慢路调用：

1. 监督前向：噪声混合真实未来latent/动作，计算慢路视频和动作FM；不导出可供快路使用的缓存。
2. 纯预测前向：独立随机video/action噪声，只读取 `SourceCondition` 白名单；经2+8采样和读出给快路。快路损失沿该路径回传慢路，没有全局detach。

快路普通模式同时对 noisy A/Z 做FM；只将实际承诺动作设为干净条件。默认25%样本使用明确的“给定动作预测未来”辅助模式：完整专家动作作为干净条件，该样本不计算快路动作loss。普通模式不得读取完整专家未来动作。缓存10%随机丢弃；未来标签掩码只用于loss，不作为快路的未来可见性输入。

默认损失：慢video .001、慢action 1、快action 1、future视觉 .25、future几何 1、contact .1。慢路学习率1e-6，新模块1e-4，主阶段联合训练，warmup默认0。预测路径支持 non-reentrant activation checkpoint；内存/吞吐需真实GPU测试，不能从CPU测试推断。

所有FM使用 `x_sigma=(1-sigma)*target+sigma*noise`，速度目标 `noise-target`；慢路采样沿用vendor shift5调度，快路暂用10步线性Euler。每步重夹紧已承诺前缀；监督损失排除前缀。两路同一物理时刻的承诺必须一致，源路仍有效承诺不能在快路变成可修改项。

动作/状态按完整检查点的mean/std归一化。运动输入保持原始物理特征，native内部归一一次；快路在FP32用同一检查点统计归一，再转换到模型dtype。interval和acceleration有效性独立。实际 `.92` packed dataset 的motion统计不同，禁止直接把其预归一化值混入本模型。配置、manifest和checkpoint绑定完整原生权重与两类统计的SHA。

## 3. 时钟、标签与部署契约

source/completed/query是FP64物理秒，正常参考满足 `source <= completed <= query`。先相减再转换模型精度。预测latent的支持区间必须由真实视频帧时钟与codec映射给出，不能用latent编号猜时间，也不能把GT未来帧时间当作部署已知条件。源观察token可作为历史上下文；过期预测和目标时间不覆盖的参考不可作为未来证据。

Z顺序：目标/支撑物或容器/左夹爪/右夹爪，各自 +.1/.2/.4/.8 秒。监督producer要求单episode固定实体身份、世界坐标、冻结特征encoder身份。视觉/几何按有限间隔、双端有效的真实时间插值；contact只使用精确观测或双端一致的二元标签，不线性插值离散接触。缺失实体和字段mask loss；夹爪重复相对位置字段mask掉。默认拒绝重复时间；显式last策略整条选最后记录。统计仅拟合训练episode，验证/test不得复用相同scene拟合统计。

`timing.py` 实现部署侧完整参考原子发布、旧版本拒绝、快照隔离，以及绝对物理网格的执行队列。查询期间承诺动作不可替换；迟到裁掉过期前缀；真正的deadline miss和buffer underrun抛出错误，不隐式暂停世界、重放旧尾部或降速。

这些是已测的数据/调度契约，**尚未接入真实DOMINO后台慢路worker和renderer循环**。部署必须在实际完成后记录completed_time，用新鲜query观测，并把队列中的真实承诺前缀传入两路。首版固定刷新，先零年龄，再扫描query1/2/4控制步与refresh4/8/16；没有实测端到端延迟前不固定宣称某个高频可行。无参考时允许空world条件，但初始执行计划需显式提供，不能静默hold。

## 4. 文件与入口

核心源码在 `src/dynamicwam/dualrate/`：`slow.py`、`fast.py`、`system.py`、`factory.py`、`data.py`、`timing.py`、`train.py`。配置 `configs/dualrate_joint_v1.json`；CPU加载入口 `scripts/check_dualrate_native_cpu.py`。这些路径均相对本独立worktree。

工程训练入口（需先生成真实合格manifest）：

```bash
PYTHONPATH=$PWD/src external/robotwin/bin/python -m dynamicwam.dualrate.train \
  --config configs/dualrate_joint_v1.json --device cpu
```

GPU只接受外部已验收supervisor提供的限时、源码/配置/数据绑定、一次性guard收据；本入口不创建guard、重试或watcher。现阶段仅single-process engineering smoke，拒绝formal或WORLD_SIZE>1。全训练checkpoint原子保存模型、optimizer、epoch cursor和Python/NumPy/Torch/专用噪声RNG；resume严格校验身份，不以weights-only冒充恢复。

所有新大文件在 `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/`；候选测试日志在worktree自身SSD `outputs/`。依赖测试overlay也在SSD，没有修改共享Python环境。

## 5. 验收与下一步

已验证完整官方原生权重CPU严格加载；实际tiny原生Wan+Action+motion的2+8前后向（仅测试使用CPU SDPA、随机小模型和禁止调用的VAE占位）；快路损失跨模块梯度、GT隔离、时钟、缺失缓存、前缀、dtype、统计和checkpoint恢复回归。最终测试计数见本次收据，不能把tiny fixture当作完整预训练模型前向证据。

尚缺：真实4实体局部视觉/几何/contact标签导出、冻结局部encoder选择与物理codec区间审计、完整模型GPU前后向与显存/耗时、八卡训练、真实异步DOMINO接线和1x原生MP4冒烟。因此未启动训练/评测，不声明SOTA或实时收益。

下一次先完成真实label exporter与跨时间source/query窗口：复用 `.92 /SSD_DISK/users/wuruihan/wam/datasets/domino_absolute_motion_v2` 的原生VAE缓存，读取原始真实时钟和实体标签，验证动作归一化转换，产出train-only统计与manifest；然后先真实小批量GPU验证，再扩充八卡训练。不可用测试随机特征或GT future作为推理条件来绕过这一步。
