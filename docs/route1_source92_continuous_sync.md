# DOMINO Level-1：连续整段执行、物理时间历史、同步查询

本版本将物理世界持续推进与非 RTC 的完整动作块策略结合。正式评测固定为官方 `DynamicWAM_full.pt`、35 tasks × 20 episodes、8 GPU、每张卡一个仿真进程和一个推理进程，保存原生 DOMINO 结果文件与 MP4。不做 4/8/16 格比较。

默认（原生 TOPP/MPLIB 约束）协议标识：`continuous_world_actual_latency_sync16_continuous_track_v2`。

另有干净消融协议：`continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1`（`--timing-mode nominal_training_timing`），见第 7 节。两套结果目录与 `semantics_id` 不得混用。

## 1. 要解决的问题及实验边界

旧真实时钟评测中，相机等待占用大量独占物理时间；逐行 TOPP 反复从停止状态起步，而按已执行行数更新历史又使四步光流对应的物理时间被拉长。这既降低闭环频率，也改变官方权重看到的运动时间分布。

本版本同时修改执行器与观测调度，属于用户指定的非 RTC 同步策略改进。它不再是“只改仿真器、原策略完全不变”的纯对照，旧版本结果独立保留，不混入新版本统计。模型结构、权重、采样器和策略观测中机器人状态的定义不变。

## 2. 完整时序

1. 在当前物理时间冻结完整多视角图像和机器人状态，等待异步读回；等待期间机器人保持已有末目标，世界继续按真实时钟前进。
2. 根据此次图像曝光时间构造历史光流，发送一次推理请求。推理期间只保持已有末目标，不执行新轨迹。
3. 模型返回完整 16×14 关节目标。验证格式，以当前实测手臂位置作为 t=0，规划覆盖全部 16 行的连续轨迹。
4. 在结果实际到达且规划结束后的当前物理时间安装整段轨迹。第一行在一个名义动作间隔之后到达，而不是直接跳到第一行。不得把新轨迹回填到已过去的物理时间。
5. 整段执行期间只更新相机历史，不发送下一次策略推理。执行全部 16 行后才开始下一次完整观测和查询；等待期间保持末目标，末速度为零。仅任务终止或原生动作预算末尾可以截断。

没有提前查询、RTC 动权重牵引、前缀冻结、过期行丢弃或异步动作块替换。相机读回与已有动作执行重叠不等于策略推理与动作执行重叠。

## 3. 连续轨迹及约束

`continuous_track_v2.py` 将当前实测姿态和全部动作行构造成逐坐标保形三次 Hermite 曲线。内部单调段使用非均匀时间网格的 PCHIP 斜率，首末斜率为零；曲线位置、速度连续，平台或方向反转处可自然降速，避免每行强制停下。`continuous_track.py` 保留解析约束验证和统一减速参考路径。

解析求出每段速度、加速度峰值，只延长违反当前 DOMINO 原生 MPLIB planner 约束的区间；每轮均重新计算非均匀斜率和全部区间的包络，以处理相邻段耦合。每段不少于 160 ms。局部迭代有上限，残余违规使用统一缩放获得可行解；同时保留已验证的统一减速曲线作为时长上界并记录所有 fallback。最终再次验证每段的约束。约束来自实际 planner 对象及其关节映射，不能解释成机器人硬件的最大能力。若轨迹超出关节位置范围，显式进行位置保护，并在每次请求中记录所有改变的坐标数和最大改变量。

这是一种满足约束的时间分配，不宣称全局时间最优。第一轮工程联调发现统一减速会让最坏单段拖慢整块，因此正式版本采用局部时间分配。合成例：从 0 到 0.5 rad、随后 15 行保持同一位置，v/a 限制均为 1 时，统一版本将全块延至 27.71 秒；局部版本只延长首段，总时长约 4.13 秒。该修正基于轨迹约束与执行耗时诊断，没有依据成功率选择参数。首轮工程联调与正式评测分目录保留。

12 个手臂坐标按原 planner 约束；两个 0–1 夹爪坐标沿用 DOMINO `set_gripper`，不虚构它们的物理角速度单位。夹爪反馈字段是原生归一化命令。手臂轨迹起点和跟踪误差读取 `get_*_arm_real_jointState()`；普通 `get_*_arm_jointState()` 返回的是 drive target，不能作为实测跟踪反馈。模型输入仍使用原接口，保持训练语义。

曲线为 C1，未声称加加速度连续、碰撞安全或实际关节精确跟随。报告保存实测手臂跟踪误差，以及轨迹名义时长、实际时长、每个局部区间和伸缩比例。`stretch` 表示总时长／名义总时长，不能理解为每段相同倍率；以 `local_interval_s` 和 `row_times_s` 为准。

## 4. 时间依据与历史输入

固定审计文件：`docs/domino_continuous_sync_timing_20261009.json`，哈希进入每个 run 的来源记录并保存副本。

- 官方 full 内嵌运动间隔均值为 0.64481509 秒，历史 stride 为 4。
- 0.64481509 / 4 = 0.16120377 秒，量化到 250 Hz 物理时钟的 40 tick，得到动作和历史采样名义间隔 **160 ms**。
- 光流端点名义间隔 **640 ms**；四个光流区间覆盖 2.56 秒，并保留加速度计算需要的更早一段。
- 140 个本地 HDF5 的独立时间抽样与上述值一致，但其数据指纹不同于官方 checkpoint 的原训练集，不能称为官方训练数据的精确复现。动作输出本身没有显式时间标签，160 ms 是由上述证据确定的名义执行契约。
- 16 行名义时长为 2.56 秒，速度/加速度约束可以延长它；不能将所有轨迹宣称为固定 2.56 秒。

历史采集主要提交 head camera，每 160 ms 物理时间尝试采样；完整多视角只用于策略查询。读回使用已有原生 GIL-free Color reader，与物理推进分离；同一组渲染缓冲读回结束前不重新提交渲染。相机单次过慢时跳过已错过的采样机会，不补造画面或时间戳。

历史端点选择以此次查询的真实曝光时间为基准，使用最多 ±80 ms 内的实际历史帧，禁止使用未来帧或同一帧充当两个端点。缺失区间显式 mask；特征计算使用真实曝光间隔，沿用原 Farneback、质量判定、特征和归一化逻辑。

## 5. 真实时钟与报告

物理步进仍为单线程唯一所有者。相机提交、CPU 后处理或进程通信造成短暂落后时，只能用已经安装的旧轨迹追赶物理时钟。最大 backlog 保持 250 ms，完整推理区间的墙钟与物理推进差保持 20 ms 门槛；超限作为基础设施错误，不能记作策略失败或偷偷重设时钟。

任务终止时不继续补走物理时间。若正在处理推理结果，终止事件可能截断推理对应的物理区间；该请求标为 `terminal_truncated`，保留实际墙钟、物理时间和 `terminal_unresolved_wall_lag_s`，不冒充完成区间。250 ms backlog 门槛仍有效。终止后的相机/推理 drain、MP4 编码、磁盘保存都在评测时钟之外。

记录独占阶段的物理时间（执行、推理、采集等）和额外的相机与动作重叠时间；后者与前者有重叠，不能相加作为总时间。需要结合真实历史端点间隔、缺失 mask、完整块覆盖和跟踪误差判断修复是否达到目的，不能仅依据 GPU 利用率判断。

## 6. 结果与可复核性

正式新结果独立写入 SSD 用户目录，保存接受的 seed、expert motion、初始状态、指令以及来源哈希。重用旧 100-episode 运行中每个任务前 20 个已接受场景，验证场景、运动和指令一致；基础设施重试必须使用同一已接受场景。

每个 episode 保留 `manifest.json`、`expert_motion.pkl`、`requests.jsonl`、`report.json`、`metrics.json`。原生结果层级保留 `episodeN.mp4`、`_result.txt`、`_metrics.json`、`_metrics_report.txt`、`_episodes_detail.json`，另外有视频时间戳与评测状态文件。

MP4 沿用原生 10 fps 观测序列编码，不代表等间隔的真实物理时间录像；实际曝光时间以 sidecar 为准。视频在报告提交前通过 FFprobe 和哈希验证；无有效 MP4 的 episode 不计为完整结果。

正式启动前进行 CPU 时序/约束/光流测试及 8 卡真实完整 episode 联调。联调用于确认实现和产物，不用其成功率调整控制参数。运行目录的 `prelaunch_validation.json` 和 `source_snapshot.tar.gz` 保存检查证据与冻结代码。

## 7. 两种 baseline：原生 MPLIB 拉伸 vs 训练名义时长消融

| | 原生 TOPP/MPLIB 约束（默认） | 训练名义时长消融 |
|---|---|---|
| `--timing-mode` | `native_mplib_limits`（默认） | `nominal_training_timing` |
| 协议 id | `..._continuous_track_v2` | `..._continuous_track_nominal_timing_v1` |
| controller | `continuous_shape_preserving_bounded_joint_track` | `continuous_shape_preserving_nominal_timing_track` |
| 连续世界 / 推理期继续 step | 是 | 是 |
| RTC / guidance / 提前查询 | 否 | 否 |
| chunk 时间 | 可被 MPLIB `joint_vel/acc_limits`（未设置时默认为 1.0）系统性拉长；实测 stretch 均值约 2.55× | **强制**按训练名义间隔执行：16×160 ms＝2.56 s，`stretch≡1` |
| 限制来源 | `continuous_session._robot_limits()` 读原生 planner；`limit_retiming=True` | 仍记录 MPLIB 限制供诊断，但 `limit_retiming=False`，不参与时间分配 |
| 用途 | 贴近原生 TOPP 约束下的连续世界无 RTC baseline | 更干净的「只去掉 RTC、仍按训练时长执行」消融 |

启动示例（名义时长消融，需新输出目录）：

```bash
cd /home/wuruihan/DynamicWAM
external/robotwin/bin/python scripts/eval_domino_continuous_sync_suite.py \
  --checkpoint external/checkpoints/DynamicWAM_full.pt \
  --output-dir /SSD_DISK/users/wuruihan/wam/outputs/evaluation/domino_continuous_sync_nominal_timing_YYYYMMDD \
  --timing-contract docs/domino_continuous_sync_timing_20261009.json \
  --timing-mode nominal_training_timing \
  --source-generation continuous_sync_nominal_timing_v1 \
  --episodes 20 --seed 100000 --gpus 0,1,2,3,4,5,6,7
```

单任务：

```bash
external/robotwin/bin/python scripts/eval_domino_continuous_sync.py \
  --timing-mode nominal_training_timing \
  --timing-contract docs/domino_continuous_sync_timing_20261009.json \
  ...  # 其余参数同默认协议，output-dir 必须新建
```

`summary.json` / `CONTINUOUS_SYNC_RESULT` 会带上 `timing_mode`、`limit_retiming` 以及 `stretch_mean` / `stretch_max`。名义协议下若任一请求 `stretch>1`，summarize 判为 invalid。
