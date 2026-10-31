# V7 原生部署物理时间历史修复

2026-10-31。用户要求优先修复 .92 当前 Stage3 对应的部署，保持训练运行。

## 范围与训练保护

- 训练源码：`4f3b91e66d61575b675d0d12dd9be1f2caaa46e2`。
- 原训练树：`/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-compact-v7-20261030`，不修改源码、配置、权重或进程。
- 独立部署副本：`/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v7-physical-history-20261031`。
- 分支：`fix/v7-physical-history-20261031`。
- 本次保持原生同步 TOPP、零承诺前缀、快2/慢4、TTL1.44、K10、动作归一化、模型结构及训练数据定义。不是恢复连续仿真。

## 已确认的问题

训练 producer 的源/查询运动特征均按 `4 * action_interval_seconds` 选历史端点，当前为约0.72物理秒。
部署 OnlineBatchBuilder 却将 `native_event_single_v6` 识别成 ordinal history，每4条命令选一次图像。
命令的TOPP执行时长可变，因此真实dt虽未伪造，输入图像跨度仍与训练不同。慢路和快路都受影响。

## 修改

1. V6/V7 builder 的历史选择改为物理时间；仅保留旧V5的ordinal分支供历史复现。动作/A10/Z事件编号不改。
2. 在线历史调用训练的 `flow_history_selection`，共用最近帧、相同曝光去重、并列选择、容差和缺帧规则。首帧在线特例只把当前端点标有效。
3. `NativePhysicalHistory` 按原始0.06物理秒网格采集真实RGB/drive-target历史。在TOPP现有 `_update_render` 完成后采图，不增加 `scene.step`，不再次调用 `_update_render` 或 `get_obs`；避免重复运动、随机灯光更新和无关标签读取。
4. 曝光记实际SceneStepClock读数，不写名义0.06秒。与网格的偏差最多一个物理tick；漏过采样期限明确报错，不补造历史。
5. 当前查询图像仍是当前事件的真实图像。查询锚点只放入此次快照，周期历史独立保存，防止短命令/高查询率挤掉3.6秒历史。各张图像不可变，光流缓存绑定真实对象。
6. 临时渲染hook只覆盖单次take_action，finally恢复原方法；新clock/reset、倒退、采图时物理推进及遗漏逐步渲染均拒绝。
7. 记录每次真实采图时刻、要求时刻、物理tick、墙钟耗时，以及查询选中的历史端点、ID、有效mask和误差。
8. 原smoke脚本支持新的采图事件，不把它们误算成动作；增加训练源码派生证明。checkpoint仍必须来自精确冻结V7训练源码，模型/训练/数据Python文件与该版本逐字节一致；仅四个明确列出的部署文件允许不同。现有strict模型/manifest/normalization检查保留。

## 验证

76项定向CPU检查通过，4.33秒。包括同物理轨迹不同命令边界/查询频率、训练与在线端点选择逐项一致、float32物理tick、非整除传感周期、bootstrap mask、异常清理、遗漏采图、真实V7 policy调用链的模拟执行、源码派生和原有快路/时序检查。

第一轮CPU检查发现并修复了在线单帧bootstrap与训练多帧selector的接口差异。离网格查询的最近真实历史允许原训练容差，不能要求所有间隔数学上恰好0.72秒；保留实际误差而不改时间戳。

这些是CPU/fake-executor证据，不是SAPIEN渲染、最终权重GPU或SR验收。为了不争抢正在Stage3使用的GPU，本轮没有GPU任务。增加中途真实采图会增加墙钟评测耗时；原生同步协议下不因此增加物理时间，不能拿此采图实现直接宣称连续1×可用。

## 后续部署要求

- 训练正常结束后使用此独立副本。不要把修复覆盖回仍运行的训练树，也不要复用旧源码SHA的GPU授权文件。
- 重新生成当前部署源码/config授权；保留训练checkpoint的原始身份，用 `validate_deployment_source` 明确记录派生关系，不重写checkpoint身份。
- 当前脚本仍是诊断smoke入口；正式权重35×20入口需按已有计划严格区分正式/诊断权重。
- 必须实测每tick渲染hook在当前SAPIEN中的调用、相机像素/时间、原生固定动作状态/接触/评分不变，再验收最终ckpt闭环。外部DOMINO源码本轮未改。
- 此修复针对历史输入，不证明动作/Z事件标签已等价于变时长TOPP执行结果；后者仍需固定动作回放审计。
