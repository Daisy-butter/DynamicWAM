# Stage3 40k previous_plan controlled deployment

Authorized one-shot evaluation on 10.12.208.100. The .50 Stage3 run completed
40000 optimizer updates cleanly; the .100 prior 30k evaluation completed all
700 episodes and decoded videos, with 80 successes (SR 11.428571%, MS 28.986657).
Both owned worker groups and temporary guard handles closed cleanly.

## Exact experimental changes

- Full checkpoint: `checkpoint40k/multirate_step_40000.pt`, SHA256
  `7e6f747e20241177eba442ba3886e31205df72dc754c1491d00e80998457bb4e`.
- Soft previous_plan: explicit `aligned_source_plan(current, bridge,
  completed=False)`, already normalized early source actions, physical-time
  validity mask. Expired rows stay invalid. Both plan-condition and
  proposal-reference consumers, including Graph input copies, share it.
- New output label: `outputs/joint40k_35x20_01_prevsource_scaled025`.

The real installed action queue and irrevocable H4 prefix remain unchanged.
Do not substitute the completed 2+8 plan or pad expired tails. The preexisting
`previous_valid_rows` diagnostics describe the execution queue; policy timing
`previous_plan_mode=training_aligned_source_early_v1` identifies the soft network
reference. This isolates an observed condition mismatch; improvement is unproven.

## Preserved protocol

Same frozen 700 scenes, 35 tasks × 20 episodes, seeds, eight GPU workers,
continuous fixed simulation_rate=0.25, H4/R4/K10/refresh16/age31, acceleration,
RTC tail replacement/deadline rules and native DOMINO reports/MP4 paths.
Config differs from 30k only in checkpoint path and label. The 0.25 protocol is
not 1x realtime. Checkpoint timing_contract.json retains the nominal training
reference; the explicit deployment config/pipeline applies 0.25 consistently.

Full/fast strict CPU reload, exact extracted backbone, finite full weights,
source-time parameters, data identities and normalization match were verified.
State/prefix normalize in FP32 before BF16. New soft source plans must not be
normalized a second time.

## Launch and gates

`scripts/start_eval100_40k.py` binds the final clean Git HEAD and starts
`scripts/run_eval100_40k.py` once. A new guard-based preflight and actual40k
Graph/background acceptance precede eight GPUs × two complete smoke episodes.
All 16 valid reports, MP4 decodes and clean worker/guard exit are mandatory
before automatic formal700 launch. No retry and no driver recovery.

Initial-only CPU observation uses `scripts/observe_eval100_40k_initial_once.py`
after controller status exists, at most two hours. It calls
`scripts/accept_eval100_40k_initial.py` once after at least 300 seconds, eight
live workers and first eight reports. Acceptance checks native MP4 hashes,
continuous clock/RTC/performance records and saved clean driver evidence.
Then the extra observer exits, retaining only evaluation supervisors/cleanup.
Launch is not acceptance or complete700. Runtime receipts under launch40k and
the new output root are authoritative; do not edit frozen source/config.
