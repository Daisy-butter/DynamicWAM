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

Entry: scripts/dualrate_posttrain_eval.py --config <posttrain.json> --run-dir <controller01> --detach.
The waiting process reads saved receipts and /proc only. Failure is terminal.
Source compatibility binds all108 non-executor Python files to training; checkpoint
metadata and strict model state loading are rechecked before GPU rollout.
Native videos: formal700/eval_result/TASK/dualrate_dynamicwam/demo_clean_dynamic/last/dualrate_v4_final_formal700/episodeN.mp4.
CPU checks do not establish GPU acceptance: final-checkpoint four-episode smoke
must independently validate real Graph, overlap, new-reference consumption,
clock/queue validity, full MP4 decode and clean exit before formal dispatch.
