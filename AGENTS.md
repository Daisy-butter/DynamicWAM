# DynamicWAM project instructions

Before every code/config/script change, read `harness.md` and applicable nested AGENTS.md.
Follow Route1 on this host. Current milestone: continuous DOMINO integration and native MP4 smoke acceptance; do not change policy/model or begin new training for this milestone.
Use `scripts/harness_preflight.py` before GPU launches. Store large artifacts under the verified local SSD wuruihan root. Never reset GPUs, alter PCI binding, reload drivers or kill unrelated processes.
Preserve existing changes. Record branch, source/patch identity, harness hash, config, checkpoint and run paths. The current milestone becomes an accepted version only after the user reviews the MP4s.

<!-- WAM-HARNESS-V02 -->
## Shared WAM harness

Before modifying DynamicWAM or AHA-WAM code/config/scripts, read `harness.md` in this directory (or the parent `/home/wuruihan/harness.md`) and follow it alongside existing local instructions and the latest explicit user request.
Route1 is DynamicWAM on .50; current milestone only ports continuous DOMINO evaluation and native MP4 export, without policy changes. Route2 is AHA-WAM on .92; finish baseline training/evaluation before architecture changes. Run paths/checkpoints/videos/caches must resolve under the local SSD wuruihan directory. Never perform GPU/driver/PCI recovery. Record harness hash and source identity in each run. User MP4 approval is required before declaring this Route1 version accepted.

Accepted milestone: user approved the five MP4s on 2026-10-13. The frozen Route1 continuous-DOMINO baseline is tagged `route1-continuous-domino-v1-20261013`; see `docs/route1_port_receipt.json`. Later work must retain this baseline and follow the latest authorized scope.

## Current authorization, 2026-10-13

The user has now authorized Route1 multirate architecture improvements, motion-conditioned K/V updates, system acceleration, RTC-consistent training, smoke and formal training. This supersedes the completed milestone's no-policy-change restriction above. Work on `feat/route1-motion-kv-20261013`; preserve the accepted baseline tag. Read harness v0.3, `docs/route1_motionkv_20261013.md`, and use `scripts/supervise_motionkv_training.py` for this training entry. Do not launch while the documented NVIDIA driver anomaly is unresolved. No automatic recovery, retries, recurring inspections or automatic evaluation. Stop interactive supervision after a short stable formal-training segment; the launched job's own durable status remains available. Route2 is untouched.

## Latest authorization, 2026-10-14

The user explicitly requested careful diagnosis followed by formal training, with GPU dropout prevention. This supersedes the blanket unresolved-history launch block above for this documented Route1 attempt only. Follow harness v0.4: bind the historical driver evidence and boot to actual per-device bounded CUDA probes; then pass one-GPU and multi-GPU training smoke plus export reload before formal training. The controlled-start receipt records user authorization, never administrator clearance or a repaired driver. New anomalies stop expansion; never attempt GPU/PCI/driver recovery. Use a bounded 600-second initial health window, then no GPU polling or recurring inspections.

Latest observed state: two-GPU smoke exited zero but cleanup produced new IOVAS assertions; seven-GPU formal launch was blocked before training. Do not reuse the old authorization against the new driver hash. Read `docs/route1_motionkv_20261014_launch.md`. A one-GPU accumulation=14 fallback is prepared only; the user has been asked for new authorization before fresh isolated validation.

Superseding user instruction, 2026-10-14 morning: retry from the local computer's SSH and start training after the reported anomaly. Follow harness v0.5: use the single-visible-GPU accumulation=14 profiles, fresh bounded probe, new evidence-bound authorization, post-exit-checked smoke and strict reload before formal launch. Do not rerun the failed multi-GPU path or claim driver repair. This replaces the pending-user-authorization state above; do not ask again for this same isolated attempt. No additional inspections after the bounded initial formal window.

Latest correction: the user rejects single-GPU formal training and requires 8 GPUs if healthy, explicitly asking to investigate our distributed code. Follow harness v0.6. No single-GPU training has started. Use bounded, separate CUDA/communication/actual-training tests to identify application causes; no blind reruns or device recovery. Require real eight-GPU smoke, post-exit health and reload before formal launch. Batch 1 per GPU with accumulation 2 means effective batch 16. Preserve failed runs and all previous instruction history.

Latest observed state at 2026-10-14 10:55 Beijing: independent simultaneous CUDA passed; minimal DDP also reproduced IOVAS with legacy allocation and Socket transport. GPU diagnostic trials stopped. Eight cards remain visible and idle in saved evidence, but no eight-GPU smoke/formal launch occurred and no single-GPU training was substituted. Do not reuse old clearance or automatically repeat failing cases. See docs/route1_motionkv_20261014_launch.md and the saved SSD diagnostic receipts. Cleanup fixes and optional transport flags are not a proven driver repair.


## Latest additional diagnosis, 2026-10-14

See `docs/route1_runtime_rootcause_20261014.md` and the same-named SSD JSON. Branch `fix/route1-ddp-runtime-20261014` preserves the accepted baseline. Lazy initialization, original DOMINO Torch 2.4.1/cu121 runtime, early application-reference release, Gloo and direct stdlib launcher all reproduced exit IOVAS assertions; independent CUDA mapped-host control passed. Model changes are not necessary to reproduce, but the exact mapping lifetime defect remains unresolved. Final snapshot: all eight GPUs visible and idle; matched driver anomaly lines 264. No eight-GPU smoke/formal run or single-GPU substitute started. GPU trials stopped again. Never reuse old clearance, retry failed cases automatically, or perform device recovery. A specifically justified software fix or administrator runtime/driver assessment must precede further expansion; formal training still requires a fresh real eight-GPU smoke, clean exit and strict reload. No recurring inspections. Keep investigation and reporting focused to conserve tokens.


## Latest authorized application fix, 2026-10-14

The user authorized inspection of driver resource release using the minimal reproduction. Exact 560.35.05 source/binary analysis identified adapter-level IOVAS lifetime. A temporary parent device-FD guard passed both two-rank and eight-rank minimal DDP, including all worker exits and every guard FD closing; driver anomaly count remained 264. See `docs/route1_device_lifetime_20261014.md` and SSD `diag_device_lifetime_guard{2,8}_20261014.json`. Work on `fix/route1-device-lifetime-20261014`, follow harness v0.7. This supports fresh real eight-GPU smoke with the same guard, strict full/fast reload and formal training only if clean. Bind guard helper SHA and enabled state in authorization, smoke and formal evidence. It is not a driver-repair or long-run-stability claim. Do not enable system persistence or retain permanent keepalive to hide teardown errors. Device recovery and recurring inspections remain prohibited; preserve historical failed evidence and accepted baseline.


## Actual eight-GPU formal start, 2026-10-14

See `docs/route1_device_lifetime_training_20261014.json`. Real eight-GPU 4-step smoke, full/fast strict reload, all workers and final guard closure passed with unchanged driver evidence (264 matched lines). Formal `formal8_guard_20261014` started at 12:02:35 Beijing from training source commit `5a44e942adc53181f4c036ba8e424e96c641b6ec`, source SHA256 `2cc8ef948e9e221598da6bec1f2f21ccc6b3fe9bb80bdda874616e7a2c185e32`. At saved 12:08:33 snapshot: step 140/12000, loss 0.496, grad 1.02, eight GPUs/effective batch 16; supervisor PID 1364813, launcher PID 1365048. The 300-second startup health window is closed. No recurring inspection or further automatic GPU/driver polling. Keep the running source frozen; do not stop the run absent a new user request or observed fault. Documentation-only commits may follow the source freeze. This is an initial snapshot, not a completion or repaired-driver claim. Later status requires a fresh user-requested read.
