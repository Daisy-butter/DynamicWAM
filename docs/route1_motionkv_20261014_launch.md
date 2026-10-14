# Route1 training launch evidence, 2026-10-14

The seven-GPU formal run did **not** start. Its preflight correctly rejected a
new driver-log identity after the two-GPU smoke exited. No GPU/PCI/driver reset
or recovery was attempted. The accepted continuous-simulator tag is preserved.

Artifact root: `/SSD_DISK/users/wuruihan/DynamicWAM/route1_motionkv_20261013`.

| Run | Computation and artifacts | Exit health |
| --- | --- | --- |
| `smoke1_20261014` | Stopped by supervisor's incorrect PGID ownership rule; fixed in `0137b28` using process ancestry | No new driver evidence |
| `smoke1_20261014_r2` | Four optimizer steps; validation; full and fast exports; strict CPU reload; finite weights/loss/actual gradients | Normal exit, unchanged driver evidence at next launch |
| `smoke2_20261014` | Four optimizer steps, accumulation=2/workers=2; validation; full and fast exports; strict CPU reload | **Failed**: three new IOVAS lines at exit despite process exit code 0 |
| `formal_20261014` | Preflight only; no training PID or train.log | Blocked on changed driver hash |

The new lines at kernel monotonic time `274740.877215–274740.884980` report one
left-over mapping for IOVAS `0x1b00`, then `pIOVAS != NULL` failures at lines
592 and 601. The timestamp matches the two-GPU exit at approximately 00:34:35
Beijing time. This establishes temporal association with multi-device cleanup,
not a proven NCCL root cause or permission to ignore the assertion.

Before: 232 matched lines, SHA256
`f8d21fdf93b67523c74ddd6845fe44de0b08149d0462c15ca7a26e419d550443`.
After: 235 matched lines, SHA256
`cda78cbb2878f6929c0fb5b823a4a6e18437d0d2a478eb2dbb4ba78a61131af9`.
Boot remains `ecebaa85-6ac6-4e66-b7e5-b1d6e7424d2e`. All eight GPUs remained
visible and idle in the post-failure read-only snapshot; no Xid was observed.

`formal_smoke_receipt_20261014.json` records computational/export checks only;
its creation does not establish successful exit health. The formal driver's
independent preflight is what prevented launching against changed evidence.
The supervisor now also checks driver/visibility after a smoke exits; a failure
retains the worker exit code but records an explicit failing stop reason.

Prepared fallback: `motionkv_rtc_adapt_1gpu_isolated.yaml` preserves the same
12,000 optimizer steps and effective batch 14 using one visible GPU and
accumulation=14. Its matching smoke uses four optimizer steps, two data workers
and strict export checks. These profiles are **not launched or validated**.
New user authorization and a fresh bounded probe are required before testing
this fallback against the newly observed driver identity. Old clearance files
must not be re-stamped or reused for the new assertion.

No recurring inspection or automatic evaluation has been created. RTC controller
integration, CUDA Graph GPU equivalence and policy-quality evaluation remain
separate pending work; training smoke does not establish them.

## Subsequent eight-GPU request and controlled diagnosis

The user's latest instruction rejects single-GPU formal training and requires eight GPUs when healthy. No single-GPU training was launched. New eight-GPU profiles use per-device batch 1, accumulation 2, global batch 16, and zero data workers. The accepted simulator tag remains unchanged.

The distributed-code audit checked actual installed PyTorch 2.7.1, Accelerate 1.14.0, DeepSpeed 0.19.2 and NCCL 2.26.2. Global rank/device mapping was correct; Accelerate already destroys registered process groups. A real lifecycle gap was corrected: an unnecessary zero-cursor `skip_first_batches` clone could own persistent workers outside the original loader. The new cleanup explicitly closes owned loaders/workers before releasing model/optimizer references and calling Accelerator's single shutdown. CPU tests cover cleanup, exception preservation, resume cursor and success-only synchronization. These changes are not claimed to repair the IOVAS root cause.

Diagnostic receipts under the SSD artifact root preserve complete before/after kernel evidence:

| Case | Computation | New kernel evidence |
| --- | --- | --- |
| `diag_cuda_visible2_20261014a.json` | One CUDA process, two visible GPUs, no WAM/DeepSpeed/NCCL; normal exit | NVLink status-query/RPC errors, **not IOVAS**. A separate concurrent NVLink query confounds attribution; this case does not exclude NCCL. |
| `diag_ddp_isolated2_20261014a.json` | Minimal two-rank PyTorch DDP; one UUID per worker; P2P disabled; CPU-reference gradients and reduction passed | Three IOVAS teardown lines; failed health. |
| `diag_ddp_isolated2_parentmasked_20261014b.json` | Same test, torchrun parent additionally CUDA-invisible; all computations and normal process-group destruction passed | Three IOVAS teardown lines; failed health. |

After these cases, eight GPUs remained visible and idle. The assertions reproduce without WAM, DeepSpeed or data workers, but the exact driver/runtime/communication cause remains unproven. The earlier conversational claim that pure CUDA reproduced IOVAS was corrected after inspecting the exact log delta.

The next discriminating cases separate concurrent independent CUDA contexts from NCCL allocation. NVIDIA's [NCCL 2.26.2 allocation source](https://raw.githubusercontent.com/NVIDIA/nccl/v2.26.2-1/src/misc/cudawrap.cc) has independent `NCCL_CUMEM_ENABLE` and `NCCL_CUMEM_HOST_ENABLE` switches. Its [official troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html#cumem-host-allocations) documents the legacy host-allocation fallback. Any resulting workaround needs actual eight-GPU smoke and normal-exit validation; no failed receipt is a launch clearance.

Completed discriminating cases, 2026-10-14 10:49–10:55 Beijing time:

| Case | Exit code / numerical result | Post-exit driver result |
| --- | --- | --- |
| `diag_cuda_independent2_20261014.json` | Two independent one-visible-GPU processes; file barrier proves overlapping CUDA contexts; both CPU-reference gradients passed; exit 0 | **Passed**, 243 matched lines before and after; no new kernel lines |
| `diag_ddp_legacy_alloc2_20261014.json` | Minimal DDP with both cuMem switches disabled, P2P disabled and isolated ranks; numerical checks and process-group destruction passed; exit 0 | **Failed**, 243→246; IOVAS 0x1c00 plus two assertions. NCCL logs confirm both switches accepted and SHM/direct/direct data transport. |
| `diag_ddp_socket2_20261014.json` | Same DDP plus SHM disabled and NET=Socket; all eight channel send/receive log entries use NET/Socket; CPU-reference gradient and collective checks passed; exit 0 | **Failed**, 246→249; same IOVAS 0x1c00 and two assertions. |

Socket is a distinct data transport, as described by [NVIDIA](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/env.html#nccl-shm-disable); it still uses rank-local CUDA/host buffers. Failure in both paths does not establish a particular driver bug or defective hardware. These tests establish that WAM, DeepSpeed, data workers and our training loop are not required to reproduce the assertion. They do not prove the complete training implementation free of other bugs.

**Final state for this attempt:** stopped GPU diagnostics after the Socket case. All eight devices are visible and idle in the saved post-exit snapshot; no Xid or device loss is present in this attempt's recorded delta. No eight-GPU smoke or formal training was launched, no single-GPU training was substituted, and no current eight-GPU health authorization/receipt exists. Do not relabel any failed diagnostic as healthy. No recovery action or recurring inspection was installed. A supported runtime/driver environment change or administrator assessment is needed before repeating the same failing communication cases.

Reusable minimal reproduction: `scripts/probe_multigpu_teardown.py`; it performs one explicitly selected case only, validates numerical results and post-exit health, and never retries or changes drivers. Historical script versions and run manifests remain under the SSD artifact root. Transport/allocation flags in the supervisor are diagnostic options, **not validated production workarounds**. Source is saved on the existing feature branch; rollback uses the accepted baseline tag in a separate checkout.
