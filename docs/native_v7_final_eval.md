# Compact V7 final evaluation: native synchronous DOMINO

This independent .92 evaluation tree derives from physical-history deployment
commit 469ba30fc42aeb1679dd8b93fc9c89e488486360 and protects the actively trained
4f3b91e66d61575b675d0d12dd9be1f2caaa46e2 tree. Model/training files are unchanged.

Run `bash scripts/launch_native_v7_eval.sh --check` for CPU-only preflight.
Run `bash scripts/launch_native_v7_eval.sh` once to detach the server controller.
It can be launched before training ends: it waits (60-second CPU-only polls,
maximum 48 hours) for all three formal phases, final 31000-step checkpoint,
all eight strict reload receipts and clean worker/guard exits. It does not
estimate completion from midnight or accept a partially written `last.pt`.

The exact final checkpoint then runs 8 tasks x 2 full episodes, on eight GPUs
(one renderer + synchronous slow/fast policy per GPU). Graph/eager parity,
native clock/action-event ledgers, actual autonomous reference reuse/refresh,
complete MP4 decoding and clean guarded exit gate a fresh 35 x 20 evaluation.
Smoke scores never merge into the formal score. No Full-driven diagnostic,
fallback, asynchronous physics, backlog tolerance or continuous action grid is
enabled. Any engineering failure stops the pipeline without automatic retry.
Ordinary task failure/action-budget exhaustion remains a valid scored episode.

Native `env.take_action(..., action_type='qpos')` retains TOPP, native dynamic
object updates, success/contact/out-of-bounds checks and native action limits.
Fast policy replans every 2 commands; slow refreshes every 4 or physical age
>1.44s. A10 prior joins by integer event offset, without repeating expired
suffixes or modifying absolute grippers. Inference advances wall time only;
camera/history/TTL use the actual scene physical clock. Physical RGB history
uses .06s captures and .72s endpoint spacing, not every fourth TOPP command.
This is native synchronous dual-rate policy evaluation, not real-time overlap.

Prepared scenes are the frozen 35-task/700-episode source. Stored accepted
seeds and expert motion are replayed with `native_protocol=True`; continuous
velocity conversion is forbidden. Instruction mode remains native seen text,
with the existing scene prefix and exact T5 tokenizer/encoder identity. Final
source/runtime/manifest/normalization/architecture are checked against training.

Outputs: `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/native_v7_final_eval_20261031/run01`.
Top-level `status.json` is authoritative. Formal videos retain
`formal/eval_result/TASK/dualrate_compact_v7/demo_clean_dynamic/final_stage3/STAMP/episodeN.mp4`.
Per-episode physical timestamps, events, scene manifest and native metrics are
stored in `formal/TASK/episode_NNNN/`. Final aggregate: `formal/suite_completed.json`.

The launcher is prepared, not automatically started by script preparation.
CPU checks do not claim final-checkpoint GPU acceptance. Learned control quality,
on-policy generalization and benefits of reference reuse remain empirical;
the final-weight smoke gate can stop rather than silently change the model or
relax settings. Original Full native DOMINO executes longer chunks, so policy
replanning differs while simulator semantics remain native. Continuous 25.43%
is a different benchmark protocol and is not a direct score target here.
