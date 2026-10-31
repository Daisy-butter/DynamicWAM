# DynamicWAM 实现版本索引

本次归档保留原Git提交历史，不squash、不强制覆盖远端main。范围是本次会话的.92 DualRate V1–V7，
以及.50 Route1、Full对照和连续部署修复分支。AHA-WAM与独立DreamCan路线不混作DualRate版本。
较早已在远端的Route1分支继续保留。

当前带详细说明的入口：[Compact V7发布说明](2026-10-31-dualrate-v7.md)。
当前训练原始分支保持`4f3b91e`；release分支是其文档/结果提交的后继，源代码逐文件相同。
训练服务器的HEAD、配置和进程未改变。

| 版本 | 分支 | 主要含义 |
|---|---|---|
| Joint V1 | `feat/dualrate-joint-v1-20261023` | Full慢路+紧凑A/Z快路的最初联合实现 |
| Joint V2 | `feat/dualrate-joint-v2-20261024` | 全数据和真实GPU训练/恢复基础 |
| Joint V3 | `feat/dualrate-joint-v3-20261024` | 有效实体监督、相机/部署修复与多卡烟测 |
| Hard-time V4 | `feat/dualrate-hardtime-v4-20261026` | 确定性时钟/lease、测量时序与连续分频训练 |
| V4部署修复 | `fix/dualrate-continuous-executor-20261028`、`fix/dualrate-response-20261029` | 墙钟执行器、独立慢进程和响应链路 |
| V4完成700 | `fix/dualrate-reference-acceptance-20261030` | 500ms连续setting、合法空参考验收修复，最终57/700 |
| Native V5 | `feat/dualrate-native-v5-20261030` | 原生同步TOPP、原生motor+轻量修正双分支、A10先验 |
| Native V6 | `feat/dualrate-native-v6-20261030` | 单一完整原生动作主干、A/Z事件语义及教师/续接训练 |
| Compact V7 | `feat/dualrate-compact-v7-20261030` | 单一448/6紧凑A/Z、对齐A10软约束、Graph及配对噪声修复 |

同名`fix/dualrate-response-20261029`在两台服务器有不同提交：.92的`e1145ff`保留原名，
.50的`3b6e5e3`保存在`archive/host50/fix/dualrate-response-20261029`，没有互相覆盖。
Annotated tag保留原tag对象，以下另列其最终commit。标记candidate/accepted表示历史工程状态，
不代表该版本任务成功率已通过。完整机器可读清单：[published_history_refs.json](published_history_refs.json)。

## 全部历史引用

| 引用 | commit |
|---|---|
| `refs/heads/archive/host50/fix/dualrate-response-20261029` | `3b6e5e3a5e9233fe4d6112bb7d170ddd7c6c4f08` |
| `refs/heads/audit/response-report-compat-20261029` | `e1145ff672fdc6b346bdaa3737064eb7dee1aaa6` |
| `refs/heads/eval/causal-spatial-native-h2r2-20261026` | `c610db6a776319af9bdbf435a2dfb189ef5511c4` |
| `refs/heads/eval/dualrate-backlog-budget-20261029` | `cd4d03bf0deeb150d0d96b51f7e7f911bd2cbb76` |
| `refs/heads/eval/dualrate-posttrain-20261027` | `0bac22416724f30182836a09630ea78c8557cffb` |
| `refs/heads/eval/full-baseline-hardtime-v4-20261027` | `082e8f12906be23e999d5ad3426a73a6eb72cf20` |
| `refs/heads/feat/dualrate-compact-v7-20261030` | `4f3b91e66d61575b675d0d12dd9be1f2caaa46e2` |
| `refs/heads/feat/dualrate-hardtime-v4-20261026` | `775c8e2b9dccf88e11d5d9f467a9cb85714564de` |
| `refs/heads/feat/dualrate-joint-v1-20261023` | `3d5a9a8c4342acb896e30bfe2baebf4e09b5c26a` |
| `refs/heads/feat/dualrate-joint-v2-20261024` | `c3a0b6bc0ad160f446eeeb4513c3f5cd31319a49` |
| `refs/heads/feat/dualrate-joint-v3-20261024` | `3a30a4aef1b7419fb7226176352f8e27c532f4f4` |
| `refs/heads/feat/dualrate-native-v5-20261030` | `51121fc89b9d7fac05e2bc97b0eb1894196e4509` |
| `refs/heads/feat/dualrate-native-v6-20261030` | `c03b097fd36cff8fd9aca473c85794e427fce24c` |
| `refs/heads/feat/route1-action-feedback-v31-20261016` | `dab042ffe43810c70e4430c5c80a36ede35904f5` |
| `refs/heads/feat/route1-async-joint-v3-20261015` | `d77b788206e4b0de31670c640646bdd312818d76` |
| `refs/heads/feat/route1-causal-spatial-v4-20261021` | `6fe2e65d0d8e8bfc99ff34601356b654d2085b71` |
| `refs/heads/feat/route1-continuous-port-20261013` | `b5907680bf4d713a2cce87d4c95b92a0561ff2f4` |
| `refs/heads/feat/route1-joint-motionkv-20261015` | `ac7d77e96ce27b84e6c398980087a63dc82ae4f7` |
| `refs/heads/feat/route1-motion-kv-20261013` | `3f320859dc3d9785c04a05243c02d7d617913f9c` |
| `refs/heads/feat/route1-motionkv-rtc-eval-20261014` | `abd2dc7b82471fdac7221b7d540b4754c37238e3` |
| `refs/heads/feat/route1-retention-recipe-v33-20261016` | `2314b54677474f988c61cc561820e541f2a9bef4` |
| `refs/heads/feat/route1-scaled-continuous-eval-20261015` | `448782a3c97dd307348f304c9a2f88400af68047` |
| `refs/heads/fix/dualrate-cohort-stall-20261029` | `7638cb215d6b4b7dfbc0661b14a25ce9385f5d5c` |
| `refs/heads/fix/dualrate-continuous-executor-20261028` | `c1df306ee8eca9618b47c423e9cb96ef2e24e8ae` |
| `refs/heads/fix/dualrate-reference-acceptance-20261030` | `5ec849b78e72c2dfa68c4a7db9905556c9797530` |
| `refs/heads/fix/dualrate-response-20261029` | `e1145ff672fdc6b346bdaa3737064eb7dee1aaa6` |
| `refs/heads/fix/full-baseline-async-camera-20261027` | `824460ceaa8cc69008750c895373c50347354216` |
| `refs/heads/fix/full-baseline-physx-20261027` | `1034f57286550360371a47e7aa65fcd57ef97660` |
| `refs/heads/fix/full-baseline-resume-20261027` | `f60e04b923a4f07ee320578f9dd4200b420ca97e` |
| `refs/heads/fix/route1-ddp-runtime-20261014` | `381e1ba47ebbb719f1762e300ccf9f24a3fa8ed2` |
| `refs/heads/fix/route1-device-lifetime-20261014` | `7aa9a75a1b97e15f3ca6c86a242ca155709374ce` |
| `refs/heads/fix/route1-eval-preflight-health-20261014` | `d19c31a7546bf8ecce9c041069f5e25a549bc81d` |
| `refs/heads/fix/route1-guarded-telemetry-20261014` | `7d1d9f7460830dc72bbae610b68c68f1ff72a355` |
| `refs/heads/fix/route1-normalization-parity-20261018` | `70cd2b01cd8965821000008fa587b38146dfa99f` |
| `refs/heads/fix/route1-rtc-deadline-pipeline-20261014` | `b2ec6d2b53693e17e6cf7ac100edaa9ba6144ed9` |
| `refs/heads/fix/route1-rtc-service-latency-20261019` | `cb89c75831d97c9ba7f917e7397b01765ac662f8` |
| `refs/heads/fix/route1-stage3-20k-eval-20261019` | `a0324e06e8ced23f635d091ca6e97e8cd499aa29` |
| `refs/heads/fix/route1-stage3-memory-20261018` | `b4da95704b647f128e2b6e1374b5e28c74607fad` |
| `refs/tags/archive/dualrate-v4-before-native-v5-20261030` | `e1145ff672fdc6b346bdaa3737064eb7dee1aaa6` |
| `refs/tags/archive/dualrate-v5-before-single-v6-20261030` | `51121fc89b9d7fac05e2bc97b0eb1894196e4509` |
| `refs/tags/archive/dualrate-v6-before-compact-v7-20261030` | `c03b097fd36cff8fd9aca473c85794e427fce24c` |
| `refs/tags/candidate/dualrate-hardtime-v4-20261026` | `775c8e2b9dccf88e11d5d9f467a9cb85714564de` |
| `refs/tags/candidate/dualrate-joint-v1-20261023` | `3d5a9a8c4342acb896e30bfe2baebf4e09b5c26a` |
| `refs/tags/candidate/dualrate-joint-v2-20261024` | `c3a0b6bc0ad160f446eeeb4513c3f5cd31319a49` |
| `refs/tags/candidate/dualrate-joint-v3-20261024` | `e3fb8c35c3a26d5fb1bebe3ed0660b5715280ea9` |
| `refs/tags/candidate/dualrate-joint-v3-smoke8-20261024` | `8c1344932bd7c9a04096c670e8bba4eb294c4d94` |
| `refs/tags/route1-continuous-domino-v1-20261013` | `b5907680bf4d713a2cce87d4c95b92a0561ff2f4` |
