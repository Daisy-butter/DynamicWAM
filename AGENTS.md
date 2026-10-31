# Native V7 final evaluation preparation, 2026-10-31

The user authorizes this independent .92 deployment tree and launch scripts.
Active compact-v7 training code/config/processes are protected; .50/.100 unchanged.
Read docs/native_v7_final_eval.md. Native synchronous TOPP/event semantics only.
No continuous physics, wall catch-up, nonzero RTC hard prefix, or Full-driven fallback.
CPU preparation does not authorize claiming final-weight GPU acceptance. The user
launches scripts/launch_native_v7_eval.sh; it gates GPU use on actual clean training
completion, then autonomous final-weight 8x2 smoke, then fresh 35x20. No retry/recovery.
Source/config frozen while running. All large artifacts stay in SSD user storage.

---

# Current scope: .92 V7 deployment history repair, 2026-10-31

Only this isolated dualrate-v7-physical-history-20261031 checkout is writable
for the deployment repair. Preserve active dualrate-compact-v7-20261030 Stage3,
its source/configuration/checkpoints/processes and all occupied GPUs.
Read docs/v7_physical_history_repair_20261031.md. Native synchronous TOPP and
zero-prefix remain; physical-time history is not continuous RTC acceptance.
CPU tests do not replace real renderer/GPU acceptance after resources are free.
Do not run the inherited training pipeline or overwrite the active train tree.
Keep large artifacts on SSD; no GPU recovery, automatic retries or broad kills.
The user separately authorizes .50 design work, without changing .92 training.

---

# Current authorized revision: Compact A/Z V7, 2026-10-30
User authorizes stopping V6, replacing ONLY fast architecture plus consistent
slow-action conditioning, then smoke and fresh formal training from official Full.
Only .92 isolated dualrate-compact-v7-20261030 is writable. .50/.100 protected.
Read docs/compact_v7.md. Preserve native V6 data/physical-clock/event/TOPP semantics.
Single 448/6 A/Z model; Full slow/teacher frozen; bounded aligned A10 prior.
No direct output averaging, no fake time, no untrained fewer-step sampler.
Source/config frozen during GPU runs, no retry or GPU recovery. Three-phase 8-rank
and actual native deployment/Graph/latency/clean-exit acceptance gate formal.
All large artifacts SSD. Initial formal stability >=300 seconds, then no extra watcher.


Pretraining acceptance distinguishes autonomous student rollouts from a separate,
explicitly unscored Full-driven integration fixture. Both run actual native TOPP
and record actual physical clocks/command joins; the latter tests reference reuse
without requiring an untrained compact student to already control successfully.
Production executes only fast outputs; there is no Full action fallback. Keep
autonomous failures visible. Learned-policy reuse/SR must be checked after training.
No TTL, physics, executor or benchmark relaxation is permitted by this fixture.

---

# Current authorized revision: Native Event Single V6, 2026-10-30
Only .92 isolated dualrate-native-v6-20261030 is writable; .50/.100 protected.
User authorized replacing V5 training. Archive V5 and stop only its verified controller.
Read docs/native_v6.md. Source/config frozen during GPU runs. All large artifacts on SSD.
Native synchronous event coordinates are distinct from physical seconds. No GPU recovery,
no automatic retries, clean owned-process and eight-rank/deployment acceptance before formal.

---

# Current scope: 2026-10-30 Native DualRate V5 (.92 only)
Read harness.md and docs/native_v5.md. Preserve .50 current formal evaluation.
Current model adds a copied native motor, joint A/Z reactive residual, bounded
ordinal action prior and partial terminal labels. This is a new native synchronous
TOPP protocol; old continuous leases/deadlines are not valid in this mode.
Only start formal training after same-source actual GPU, eight-rank and native
DOMINO deployment acceptance. Do not report a diagnostic smoke as SR evidence.

---

# Consistency audit revision02, 2026-10-27

The prior CPU-only waiter2764016 was intentionally stopped before evaluation
source edits. No GPU rollout was started and training775c8e2 is unchanged.
Inference source images now normalize in FP32 before VAE BF16/FP16 cast, exactly
matching producer/materialize. Camera freshness uses the same1e-6 numeric epsilon
as training:60real float32(.004) ticks accepted,61ticks rejected. Nominalcamera.24
and backlog.25, clock anchors, queue leases and network weights are unchanged.
107non-executor/non-online source files remain byte-identical; online.py exact AST
delta is independently checked against the frozen training source and hash-bound.
165CPU consistency/clock/prefix/deployment/controller tests passed. Final GPU
acceptance remains pending after training. This revision supersedes the earlier
108all-files-equal statement only for the explicitly bound online preprocessing fix.
New independent waiter uses controller02, revision02/posttrain.json, and writes
smoke/formal artifacts under revision02. No automatic retry or hardware recovery.

---
# Current authorization: post-training V4 evaluation, 2026-10-27

This isolated .92 branch runs an independent one-shot script after the protected
V4 training sequence finishes. Never change or signal the live training tree.
Keep model/data/queue/sampling Python identical to training commit775c8e2.
Only executor same-step contact caching and inactive cohort/failure hooks differ.
Final phase20000/global31000, all8 strict restore, checkpoint hashes, closed
training guards and exited controller are mandatory before any GPU preflight.
Wait CPU-only at60s intervals, max48h. Run2x3GPU smoke4 complete episodes, then
35x20 frozen700 scenes. Match Full external benchmark45ticks/1x/linear/2holds/
globalrowbudgets/camera.24/backlog.25. Last2GPUs remain compute-idle.
No retries, hardware recovery, fallback weights or edits after controller launch.
Independent detached controller owns guards; the chat need not remain open.
All outputs are on SSD under outputs/diagnostics/dualrate_posttrain_eval_20261027.

---
Latest user-authorized .92 hardtime V4: read harness.md first section and docs/hardtime_v4_contract_20261026.md. Only new worktree writable. Old formal stopped cleanly; restart all phases from official full ONLY after deterministic clocks/queue, measured parallel deployment and current8rank smoke gates. No learned clock alignment, no GPU recovery, no automatic retry.

Latest .92 DualRate Joint V3 task (2026-10-24): user now explicitly authorizes full inference implementation and eight-GPU formal multi-stage training after actual acceptance. Read harness.md v1.20 and docs/dualrate_joint_refinement_20261024.md. Isolated dualrate-joint-v3-20261024, only .92 writable; .50/.100 untouched. Original/V1/V2 preserved. BF16 forward + FP32 masters/gradient accumulation/optimizer follows native DeepSpeed. Formal Stage1 starts official DynamicWAM_full.pt; no smoke weights. Source/config frozen while workers run, all8 temporary FD guard before telemetry, no GPU recovery/retry/recurring watcher. Historical no-formal authorization below is superseded by this explicit scope, not by an unverified readiness claim.

Latest .92 DualRate Joint V2 acceptance (2026-10-24): candidate/dualrate-joint-v2-20261024; tested source5e80b18207c0a27e66192c690624de8a0818b4dc. 191CPU and actual full-model single-GPU four-step joint training plus bitwise full-state reload and clean guard/worker/kernel exit passed, peak21.0948GiB. Full10500episode index/train345647/val37087 and1516realT5 ready; VAE/ROI lazy real encoding. FP32master optimizer; isolated hash-bound FA2.7.4.post1. formal_training_ready=false; no8rank/DOMINO1x/SR acceptance. Only.92 writable, .50/.100 untouched. Read docs/dualrate_joint_v2_20261024.md and harness1.19.

Current task: .92 DualRate Joint V2, branch feat/dualrate-joint-v2-20261024. Read harness.md v1.19 and docs/dualrate_joint_refinement_20261024.md. Only .92 writable; .50/.100 no changes. Real data preparation and externally guarded single-GPU engineering training smoke authorized; formal training disabled. Preserve original main and V1. All artifacts on verified SSD.

# DualRate Joint V1 entry

Read `harness.md` and `docs/dualrate_joint_v1.md` before changes. This isolated .92 worktree is the only writable remote codebase for this task. .50/.100 are read-only; preserve .92 home dirty source, AHA-WAM, running jobs and shared external assets. Large artifacts belong to verified .92 SSD wuruihan paths. Never reset/unbind/reload GPU/driver/PCI; never launch automatic retries or recurring watchers.

Follow the attached DualRate design: full DynamicWAM slow Video+Action+motion, independent fast joint A/Z, source-only predictive bridge with gradients, separate supervised forward, exact physical clocks and hard committed prefix. CPU tests or tiny native fixtures do not establish full GPU/DOMINO readiness. Candidate formal_training_ready=false; no formal/DDP launch before real data and runtime acceptance. Track every source change and preserve failures.
