# Change log

## 2026-10-14 — Local SSH isolated training retry

Following the user's new retry instruction after the two-GPU exit assertion, use one visible GPU with accumulation=14 and unchanged effective batch 14. Harness v0.5 records this scope. Fresh probe, smoke including post-exit health, and strict reload are required; no old clearance is reused. Record actual outcome in `route1_motionkv_20261013` on the SSD. Model architecture and accepted simulator baseline are unchanged by this launch adjustment.

## 2026-10-14 — Controlled training launch and truthful gradient evidence

Parent: `fcdc600f66fe5698c7071d9fbb9a513a48ed6099`. Added bounded per-UUID CUDA probes, evidence-bound user authorization distinct from administrator clearance, smoke/export gates and a 600-second startup health window. Historical driver assertions are preserved; new anomalies stop the run without device recovery. DeepSpeed gradient logging now reads the actual engine norm and rejects missing/nonfinite values. CPU gate/control-flow checks passed; actual GPU training outcomes are recorded in SSD run receipts under `route1_motionkv_20261013`.

The first smoke was stopped by an overly strict process-group ownership check; torchrun workers own separate groups. Commit `0137b28` fixes ownership using verified parent ancestry. The rerun completed four single-GPU steps and strict full/fast reload; the two-GPU run completed four optimizer steps with formal accumulation=2 and data workers=2. Both exited zero and preserved all logs. Formal output name is `motionkv_rtc_adapt_20261014`; the formal launcher additionally validates the final multi-GPU reload receipt before dispatch.

Subsequent exit-health finding: two-GPU cleanup produced three new IOVAS assertions, so seven-GPU formal preflight correctly blocked launch. Computational success is not exit-health success. Added explicit post-exit smoke checks and prepared, unlaunched single-GPU accumulation=14 profiles. See [launch evidence](docs/route1_motionkv_20261014_launch.md); renewed authorization is pending for any GPU attempt against the changed driver evidence.

## 2026-10-13 — Route1 motion K/V and RTC training candidate

Parent: `b5907680bf4d713a2cce87d4c95b92a0561ff2f4`; branch `feat/route1-motion-kv-20261013`. Added action-conditioned, motion-guided bounded world K/V updates, clean-prefix/suffix RTC training, fixed-buffer CUDA Graph support and deployment refusal for unsupported controllers. Simulator and accepted MP4 baseline remain at `route1-continuous-domino-v1-20261013`.

Design, protocol limits, configs, validation gates and artifact paths: [route1_motionkv_20261013.md](docs/route1_motionkv_20261013.md). GPU smoke/training/graph measurements must be recorded separately; pending driver-health resolution is not a passed launch gate. Rollback by using the accepted tag in a separate checkout; do not reset or delete this candidate or prior artifacts.
