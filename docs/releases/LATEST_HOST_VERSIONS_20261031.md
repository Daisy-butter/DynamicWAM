# 两台服务器最新实现对应表（2026-10-31）

这两版是不同的训练/评测协议，分支名称明确保留服务器与协议。发布时两台均处于 cross_time_joint 训练；本次不启动评测、不修改服务器源码/配置、不操作 GPU 或训练/评测进程。最终权重评测仍需各自完整烟测验收。

| 项目 | 10.12.208.92：Compact V7 | 10.12.208.50：Continuous RTC V8 |
|---|---|---|
| 推荐完整入口 | [release/host92-native-domino-v7-20261031](https://github.com/Daisy-butter/DynamicWAM/tree/release/host92-native-domino-v7-20261031) | [release/host50-continuous-rtc-v8-20261031](https://github.com/Daisy-butter/DynamicWAM/tree/release/host50-continuous-rtc-v8-20261031) |
| 原始部署分支 | eval/native-v7-final-20261031 | eval/dualrate-v8-continuous-final-20261031 |
| 原始部署提交 | 9f1055b70c4c6b92d0a906af7ceeb9848078318a | 919df6951d785851f5e877ab4ff3cffe383335ae |
| 原始训练分支 | feat/dualrate-compact-v7-20261030 | fix/dualrate-v8-rtc-training-20261031 |
| 原始训练提交 | 4f3b91e66d61575b675d0d12dd9be1f2caaa46e2 | 355d4af324cf8cb2f4f4fe87b6237e5528bc8bfc |
| 评测世界 | 原生同步 DOMINO：推理时物理暂停 | 连续 1× 物理推进、异步相机、RTC |
| 执行器 | 原生变时长 TOPP | 固定 45 tick（约180 ms）线性手臂端点、区间起点夹爪 |
| 动作/A10/Z 对齐 | 整数命令事件；历史和TTL使用真实物理秒 | 真实物理时间区间、全局动作网格、严格墙钟/物理截止 |
| 快路 | 单一448维/6层/7头 A/Z 联合模型，约27.69M | 单一448维/6层/7头 A/Z 联合模型，约27.75M |
| 慢路动作使用 | 对齐A10作为有界采样起点先验和训练约束 | 按区间对齐A10作为数值draft，预测不受限FM残差；夹爪不加draft |
| 慢路及信息 | Full慢路/教师冻结 | Full慢路冻结，观察/未来注意力分离及独立未来门控 |
| 部署节奏 | 快2命令；慢4命令或物理TTL1.44s刷新；零硬前缀 | fast1/slow4；保护2/3行；新观测门禁、.20s预查询预算、TTL2s |
| 训练阶段 | 1000 → 10000 → 20000，global31000；全数据 | 500 → 4000 → 8000，global12500；约半数训练回合、完整验证 |
| 初始化 | DynamicWAM_full.pt | DynamicWAM_full.pt |
| 最终权重评测 | 8任务×2完整回合烟测 → 35×20；每GPU一个环境 | 2任务×2完整回合烟测 → 35×20；每环境3GPU、两环境并行 |

## .92 包含的最新修复与入口

保留训练中的模型/数据/归一化代码；独立部署包含物理时间历史修复：按约0.72秒选择历史端点，不再把4条变时长TOPP命令当作固定0.72秒。保存真实采图时刻并逐字段验证部署源码差异。

训练配置/权重来源仍绑定原训练提交。启动脚本在服务器已有的部署工作树中：

```bash
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v7-native-eval-20261031/scripts/launch_native_v7_eval.sh --check
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v7-native-eval-20261031/scripts/launch_native_v7_eval.sh
```

对应分支中的 `docs/native_v7_final_eval.md`、`docs/compact_v7.md` 和物理历史修复说明保存部署细节与验收边界。

## .50 包含的最新修复与入口

模型、数据和执行器Python源码与运行中的训练完全相同。包含真实提交收据、同物理区间A10、原始帧运动历史、夹爪起点标签、真实承诺回放和最终权重门禁。自动评测正确识别500/4000/8000 recipe，区分训练RNG和场景seed100000。

```bash
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031/scripts/launch_continuous_v8_eval.sh --check
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031/scripts/launch_continuous_v8_eval.sh
```

backlog为500ms加既有5ms容差，观测年龄240ms；没有新增EMA、C1/C2或TOPP。对应分支中的 `docs/continuous_v8.md`、`docs/continuous_v8_final_eval.md`、`docs/continuous_executor_research_20261031.md` 保存架构、入口与执行器调研。

两条启动命令都由用户显式执行，启动后有独立有限监督、最终权重和干净退出门禁；失败不自动重试。本次发布未执行它们。脚本中的绝对路径对应现有服务器部署，不宣称克隆到任意路径即可直接运行。

## 结果与 Git 边界

旧8.142857%（57/700）属于 **V4** 的 `fix/dualrate-reference-acceptance-20261030` / `5ec849b`，不是本页V7或V8的成绩。25.43% Full属于250ms backlog的旧连续评测，不与原生TOPP或500ms连续协议混作严格同条件比较。

两个release分支分别以最新部署提交为父提交，仅新增本说明、机器可读版本对应表及README入口；源码、脚本、配置和原提交历史均保持原样。服务器活跃分支不切换。旧main和早上发布的分支不覆盖；权重、数据、MP4和大型运行日志保留在SSD。
