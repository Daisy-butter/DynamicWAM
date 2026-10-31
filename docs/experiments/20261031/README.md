# 2026-10-31 已完成评测归档

这里收录的是旧 V4 连续仿真最终权重的完整700回合，以及旧官方Full连续仿真对照。
**当前Compact V7仍在训练，没有正式SR。** 结果与版本、协议绑定，不能跨run混分。

| 项目 | Full formal09 | DualRate V4 formal50_01 |
|---|---:|---:|
| 任务/回合 | 35×20 | 35×20 |
| 成功 | 178/700 | 57/700 |
| SR | 25.428571% | 8.142857% |
| Manipulation score mean | 41.349259 | 25.328458 |
| Route completion mean | 46.336848 | 32.757894 |
| 越界 | 305 | 399 |
| 预算耗尽 | 211 | 239 |
| 未抬升 | 6 | 5 |
| 配置欠账上限 | 250ms+既有5ms容差 | 500ms+既有5ms容差 |
| 实测最大欠账 | 163.379ms | 143.501ms |
| 策略GPU/环境 | 1 | 3（render/fast/slow） |
| 正常完成、worker零退出、视频全解码 | 是 | 是 |

两组实际(task,episode,seed)逐一相同，冻结prepared SHA256均为
`cff71ba1b8bf17c58082e615016e20775077787b455746e87261eaaaf60b6fb2`。
共享1×连续物理、45×float32(.004)秒动作网格、原任务预算、linear arm/native gripper、
首轮推理不计active并有2行hold、camera年龄240ms。它们都不是原生同步TOPP的V7评测。

**不作为严格同设置对照。** 欠账阈值和策略资源成本不同；本次观测到的两组最大欠账都低于250ms，
但不能据此伪称曾按完全相同配置重跑Full。500ms Full重跑由用户暂缓。SR差约17.29个百分点是
这两次运行的描述性差异，不是对某个模块的独立因果归因。

## 版本与权重

- V4模型训练commit：`775c8e2b9dccf88e11d5d9f467a9cb85714564de`；本次部署/验收commit：
  `5ec849b78e72c2dfa68c4a7db9905556c9797530`，分支`fix/dualrate-reference-acceptance-20261030`。
- V4最终ckpt SHA256：`8be3d67d3962d264e1883debec536888d71b7cb13e031d5463d17ec543522894`。
- Full checkpoint SHA256：`7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`。
- 旧版失败/中止批次不并入这两组700。

V4完整运行记录了48,259次模型重叠、46,548次慢路忙时快查询及持续新M消费。工程连通性通过，
但成功率仍低，说明“收到新参考”不足以证明“参考帮助动作”。399次越界与239次预算耗尽是
当前结果中的主要终止原因；不能直接把预算耗尽解释为预算配置错误。

## 文件与核验

- [V4总表](v4_continuous_500ms/summary.json)、[逐任务](v4_continuous_500ms/per_task.csv)、
  [700回合索引](v4_continuous_500ms/episodes.jsonl)、[最终验收及证据哈希](v4_continuous_500ms/acceptance.json)。
- [Full总表](full_baseline_250ms/summary.json)、[逐任务](full_baseline_250ms/per_task.csv)、
  [700回合索引](full_baseline_250ms/episodes.jsonl)。
- [两版逐任务对照CSV](per_task_comparison.csv)。两目录均保存冻结配置、source、guard终态和kernel终态。
- [只读重算脚本](export_results.py)逐一读取原report/metrics，重新计算成功率、MS、RC；检查35×20唯一性、
  完整标记、实际seed、已保存的视频尺寸和报告哈希。V4报告SHA还与最终验收证据匹配。
  本次没有重复解码全部视频：原终态收据已绑定全解码记录，本次保留其结果及SHA/尺寸核验。
- [V7训练快照](v7_training_snapshot.json)、[V7工程验收](v7_engineering/independent_acceptance.json)。
  V7短训自主0/2、零M复用；Full驱动无分数集成诊断的3次复用不能作为V7策略成绩。

MP4、模型权重和原始大报告留在SSD。`episodes.jsonl`保存每个视频的原路径、SHA256、尺寸、帧数，
并保存原report与metrics的哈希。实际seed来自记录，不从episode编号推算。

原V4结果根：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_reference_acceptance_20261030/attempt02/formal50_01`。
Full结果根：`/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/full_baseline_hardtime_v4_20261027/formal09`。
原生视频层级保留为`eval_result/<task>/<policy>/<task_config>/<ckpt>/<stamp>/episodeN.mp4`。

## 逐任务结果（协议不同，描述性并列）

| Task | Full成功 | Full SR | V4成功 | V4 SR |
|---|---:|---:|---:|---:|
| adjust_bottle | 6/20 | 30% | 4/20 | 20% |
| beat_block_hammer | 3/20 | 15% | 0/20 | 0% |
| click_alarmclock | 3/20 | 15% | 1/20 | 5% |
| click_bell | 1/20 | 5% | 0/20 | 0% |
| dump_bin_bigbin | 0/20 | 0% | 0/20 | 0% |
| grab_roller | 16/20 | 80% | 9/20 | 45% |
| handover_block | 3/20 | 15% | 0/20 | 0% |
| handover_mic | 8/20 | 40% | 2/20 | 10% |
| hanging_mug | 3/20 | 15% | 5/20 | 25% |
| move_can_pot | 6/20 | 30% | 0/20 | 0% |
| move_pillbottle_pad | 10/20 | 50% | 1/20 | 5% |
| move_playingcard_away | 4/20 | 20% | 0/20 | 0% |
| move_stapler_pad | 2/20 | 10% | 0/20 | 0% |
| place_a2b_left | 0/20 | 0% | 1/20 | 5% |
| place_a2b_right | 0/20 | 0% | 0/20 | 0% |
| place_bread_basket | 14/20 | 70% | 2/20 | 10% |
| place_bread_skillet | 4/20 | 20% | 0/20 | 0% |
| place_can_basket | 7/20 | 35% | 2/20 | 10% |
| place_container_plate | 8/20 | 40% | 6/20 | 30% |
| place_empty_cup | 4/20 | 20% | 5/20 | 25% |
| place_fan | 3/20 | 15% | 0/20 | 0% |
| place_mouse_pad | 6/20 | 30% | 0/20 | 0% |
| place_object_basket | 9/20 | 45% | 2/20 | 10% |
| place_object_scale | 4/20 | 20% | 0/20 | 0% |
| place_object_stand | 2/20 | 10% | 1/20 | 5% |
| place_phone_stand | 5/20 | 25% | 3/20 | 15% |
| place_shoe | 10/20 | 50% | 3/20 | 15% |
| press_stapler | 0/20 | 0% | 1/20 | 5% |
| put_bottles_dustbin | 7/20 | 35% | 1/20 | 5% |
| put_object_cabinet | 1/20 | 5% | 0/20 | 0% |
| rotate_qrcode | 4/20 | 20% | 0/20 | 0% |
| scan_object | 2/20 | 10% | 0/20 | 0% |
| shake_bottle | 9/20 | 45% | 3/20 | 15% |
| shake_bottle_horizontally | 8/20 | 40% | 4/20 | 20% |
| stamp_seal | 6/20 | 30% | 1/20 | 5% |
