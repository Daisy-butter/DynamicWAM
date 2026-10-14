# Route1 多卡退出异常：2026-10-14 追加诊断

**状态：尚未修复；8 卡正式训练未启动。** 分支 `fix/route1-ddp-runtime-20261014` 从 `3f320859dc3d9785c04a05243c02d7d617913f9c` 建立。冻结基线 `route1-continuous-domino-v1-20261013` 保持不变。

完整原始记录、逐 rank 时间线和日志位于 `.50`：
`/SSD_DISK/users/wuruihan/DynamicWAM/route1_motionkv_20261013/`。
汇总：`route1_runtime_rootcause_20261014.json`。本记录优先于早先 launch 文档中的单卡候选及旧放行状态。

## 最小对照结果

以下各项为两张卡、独立进程、单 UUID 可见、极小 Linear 网络或独立 CUDA 对照，未加载 DynamicWAM 模型。所有数值检查通过、进程退出码为 0；健康判定另行检查驱动日志。

| SSD 记录前缀 | 改变的因素 | 驱动异常行数 | 退出健康 |
| --- | --- | --- | --- |
| `diag_ddp_lazy2_20261014` | 不向 init_process_group 传 device_id，与旧 DeepSpeed 延迟初始化方式对照 | 249 → 252 | 失败 |
| `diag_domino_runtime_lazy2_20261014` | 原 DOMINO 环境：Torch 2.4.1+cu121、NCCL 2.20.5 | 252 → 255 | 失败 |
| `diag_cuda_mapped_host2_20261014` | 无分布式通信；两个并发 CUDA 进程各分配、校验、显式释放 4 KiB mapped host 内存 | 255 → 255 | 通过 |
| `diag_ddp_release2_20261014` | 销毁进程组前释放模型、优化器与计算引用，GC、同步、清理 allocator cache | 255 → 258 | 失败 |
| `diag_gloo2_20261014` | Gloo 后端；CUDA DDP、BF16 all_reduce、list all_gather 数值通过 | 258 → 261 | 失败 |
| `diag_direct_ddp2_20261014` | 纯标准库父进程直接启动各 rank，无 torchrun 父进程 | 261 → 264 | 失败 |

失败新增行均为 IOVAS left-over mapping 及 `pIOVAS != NULL` 断言。所记录 distributed destroy 与最终 CUDA synchronize 均返回。最后只读快照：8 卡可见、无 compute 进程；这不代表退出异常已解除。

## 与原项目的对应关系

- 原 DynamicWAM 初始版本 `feb7921` 与当前的 `pyproject.toml`、`scripts/train_all.sh` 无差异；共同使用 Accelerate/DeepSpeed。`common.py::build_accelerator` 的相关变化是 DataLoaderConfiguration，未换一套通信后端。
- 新版 DeepSpeed 0.19.2 / Torch 2.7.1 可传 `device_id`，旧 DOMINO DeepSpeed 路径不传；已用 lazy 和原环境分别验证，均未修复。
- Accelerator.end_training 本身会销毁进程组。已有应用清理改进应保留，但不能宣称其解决 IOVAS。
- 主环境为 Torch 2.7.1+cu126 / NCCL 2.26.2；已读取实际映射库路径，未发现混用 CUDA/NCCL 库的证据。加载驱动与用户态 libcuda/libnvidia-ml 同为 560.35.05。

## 能成立与不能成立的结论

模型改动不是复现此故障的必要条件。现象也不局限于 NCCL、新版运行时或 torchrun。普通并发 CUDA 与上述小型 mapped-host 对照不足以触发异常，但不能据此证明真实分布式训练健康。

相邻 NVIDIA 开源驱动源码将该断言关联到 I/O 地址空间和映射的销毁生命周期；具体映射分配者、错误释放顺序及本机精确缺陷仍未定位。不能从日志直接推出硬件损坏，也不能称它无害。`nvidia-smi` 设备查询正常与进程退出时的资源清理正确是不同检查。

参考源码：[PyTorch 2.7 distributed](https://docs.pytorch.org/docs/2.7/distributed.html)、[原 DynamicWAM common.py](https://github.com/Autumn1337/DynamicWAM/blob/main/src/dynamicwam/training/train/common.py)、[NVIDIA 560.35.03 io_vaspace.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/560.35.03/src/nvidia/src/kernel/mem_mgr/io_vaspace.c)。相邻 .03 源码不是本机 .05 二进制的逐行证明。

## 留存与后续门槛

保存带 init/backend/launcher 对照、逐阶段时间线、实际加载 libcudart 的 mapped-host 对照及安全进程归属清理的诊断脚本；CPU 模拟检查通过。原始 GPU 记录保留各自源文件 SHA，保存后的注释与报告字段修正未再运行 GPU，不覆盖旧记录。

停止 GPU 试验，没有重置、unbind、驱动重载、PCI 操作或启动单卡替代训练，也没有创建巡检。下一步需要管理员结合最小复现检查运行时/驱动资源释放，或得到有明确依据的软件修复；不能自动重跑已失败的组合、重盖旧 clearance 或跳过退出检查。恢复后仍须真实 8 卡 smoke、正常退出、导出严格 reload，再启动正式训练。
