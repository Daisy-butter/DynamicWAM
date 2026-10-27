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
