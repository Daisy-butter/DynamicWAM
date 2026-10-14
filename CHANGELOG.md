# Change log

## 2026-10-14 — Eight-GPU launch audit and deterministic teardown

Parent: `22b390081d77087aed41a614ffcc5b1c550bef9e`. The user superseded the unlaunched single-GPU plan with eight-GPU parallel training. Harness v0.6 records this scope. Eight-GPU profiles use batch 1, accumulation 2 (effective batch 16), and zero DataLoader workers for the initial validation.

Audit found no missing `destroy_process_group`: installed Accelerate already handles it. Explicitly close owned DataLoader iterators and persistent workers, avoid an unnecessary zero-cursor loader clone, release application references before Accelerator shutdown, and preserve the original training exception. CPU lifecycle tests passed. An optional per-rank UUID wrapper and explicit NCCL transport manifest support controlled diagnostics; their presence does not establish GPU health. See the launch evidence document for actual minimal CUDA/NCCL results and the formal-training state.

Actual outcome: independent overlapping CUDA processes passed; minimal DDP reproduced three IOVAS exit lines with default allocation, legacy allocation, and Socket transport. Numerical checks passed but exit health failed. Diagnostics stopped; no eight-GPU smoke or formal launch. Added reusable bounded reproduction, exact NCCL environment matching between smoke and formal, and retained all evidence on SSD. No workaround is declared successful.

## 2026-10-14 — Local SSH isolated training retry

Following the user's new retry instruction after the two-GPU exit assertion, use one visible GPU with accumulation=14 and unchanged effective batch 14. Harness v0.5 records this scope. Fresh probe, smoke including post-exit health, and strict reload are required; no old clearance is reused. Record actual outcome in `route1_motionkv_20261013` on the SSD. Model architecture and accepted simulator baseline are unchanged by this launch adjustment.

## 2026-10-14 — Controlled training launch and truthful gradient evidence

Parent: `fcdc600f66fe5698c7071d9fbb9a513a48ed6099`. Added bounded per-UUID CUDA probes, evidence-bound user authorization distinct from administrator clearance, smoke/export gates and a 600-second startup health window. Historical driver assertions are preserved; new anomalies stop the run without device recovery. DeepSpeed gradient logging now reads the actual engine norm and rejects missing/nonfinite values. CPU gate/control-flow checks passed; actual GPU training outcomes are recorded in SSD run receipts under `route1_motionkv_20261013`.

The first smoke was stopped by an overly strict process-group ownership check; torchrun workers own separate groups. Commit `0137b28` fixes ownership using verified parent ancestry. The rerun completed four single-GPU steps and strict full/fast reload; the two-GPU run completed four optimizer steps with formal accumulation=2 and data workers=2. Both exited zero and preserved all logs. Formal output name is `motionkv_rtc_adapt_20261014`; the formal launcher additionally validates the final multi-GPU reload receipt before dispatch.

Subsequent exit-health finding: two-GPU cleanup produced three new IOVAS assertions, so seven-GPU formal preflight correctly blocked launch. Computational success is not exit-health success. Added explicit post-exit smoke checks and prepared, unlaunched single-GPU accumulation=14 profiles. See [launch evidence](docs/route1_motionkv_20261014_launch.md); renewed authorization is pending for any GPU attempt against the changed driver evidence.

## 2026-10-13 — Route1 motion K/V and RTC training candidate

Parent: `b5907680bf4d713a2cce87d4c95b92a0561ff2f4`; branch `feat/route1-motion-kv-20261013`. Added action-conditioned, motion-guided bounded world K/V updates, clean-prefix/suffix RTC training, fixed-buffer CUDA Graph support and deployment refusal for unsupported controllers. Simulator and accepted MP4 baseline remain at `route1-continuous-domino-v1-20261013`.

Design, protocol limits, configs, validation gates and artifact paths: [route1_motionkv_20261013.md](docs/route1_motionkv_20261013.md). GPU smoke/training/graph measurements must be recorded separately; pending driver-health resolution is not a passed launch gate. Rollback by using the accepted tag in a separate checkout; do not reset or delete this candidate or prior artifacts.


## 2026-10-14 - MotionKV RTC continuous evaluation

Branch `feat/route1-motionkv-rtc-eval-20261014`, parent `7aa9a75a1b97e15f3ca6c86a242ca155709374ce`. Adds a timestamped clean-prefix controller and serving adapter for the completed step-12000 export; preserves the accepted continuous baseline. H4/R4, K10, immutable official slow cache with motion-conditioned updates, CUDA Graph prewarm/replay, fixed 0.16-second nominal action slots calibrated against training simulation timestamps. Native DOMINO reports, MP4/physical-time sidecars and end-to-end latency are retained; failed episode video is separately preserved. Scene preparation freezes expert-feasible seeds before policy rollout.

Checkpoint audit, 140-episode timing audit, CPU contract tests, graph/eager parity, GPU/guard lifecycle receipts and all failed attempts live at `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/motionkv_rtc_20261014`. Full checkpoint SHA256: `35b2c51f651a7f5fcbf7de9b7c0fef203c8cbb372ba7cc651e7d7968e02d70f6`. Graph gate passed (sampler median 52.66 ms versus eager 209.81 ms); these are not end-to-end latency claims.

Simulator Vulkan enumeration reaches all adapters before CUDA device selection. A single-worker/single-guard prep exposed new IOVAS teardown errors on the seven unguarded adapters (264 -> 285 anomaly lines). Application-only all-adapter temporary descriptor coverage completed the same 2-scene prep with final descriptor closure and no further assertions; failed receipts are preserved. No driver or PCI setting was changed. Actual RTC first episode completed 48 action slots/12 queries with continuous-time gates passing, but its result checker incorrectly compared absolute row times against tick-quantized per-block origins; correct comparison is the fixed global grid, with tick continuity checked separately. All later formal launch claims require fresh passed smoke and clean exit receipts.

Harness v0.8 documents the evaluation contract. Rollback target remains `route1-continuous-domino-v1-20261013`; this candidate does not establish a new performance result until the full evaluation completes.


## 2026-10-14 - Formal evaluation stopped before episodes

The 700-scene preparation succeeded, but formal startup stopped on a new NVLink/RPC lock-status error before any model episode. This delta contained no new IOVAS/pIOVAS. Added explicit prelaunch driver phase fences on a new branch, preserving the original baseline and failure evidence; CPU-only validation, no GPU restart or driver-repair claim. See [failure audit](docs/route1_eval_start_failure_20261014.md).
