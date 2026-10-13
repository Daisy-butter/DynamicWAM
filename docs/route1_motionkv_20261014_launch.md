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
