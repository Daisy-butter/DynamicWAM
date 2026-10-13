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
