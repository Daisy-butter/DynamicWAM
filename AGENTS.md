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
