# V8 RTC readiness and proposed recipe — 2026-10-31

Only .50 `dualrate-v8-rtc-20261031` is writable. Previous V8 and .92 training
remain protected. Source/config are frozen during every guarded GPU run.

The actual executor remains 45 float32 .004 s physics ticks per endpoint,
linear arm position/velocity and native absolute grippers. C1/C2 trajectories
were proposed but are NOT implemented or claimed. The physical owner alone
advances time. Wall origin is immutable; backlog budget is .500+.005 s.

## Fixes

* `worker_receipt_v8`: after GPU and host plan construction, a single mailbox
  stamps actual monotonic arrival under its lock. An on-time packet may be
  processed after a bounded owner stall, but only after its wall release maps
  to real completed physics ticks and before its original physical deadline.
  No backdating, extended lease, prefix mutation or past action replay.
* Terminal closure atomically closes mailbox admission before joining workers;
  unused late results are explicitly discarded. Real active deadline failures
  remain failures. Receipt, commit and execution remain distinct events.
* Candidate response scheduling: fast minimum one action row, only a new actual
  image may trigger it; consume the latest complete M without waiting for slow.
  Slow remains four rows minimum. Admission reserve .20 s, bounded 2/3-row
  commitments. These values require measured GPU acceptance, then exact data
  event replay. Training/checkpoints bind the complete runtime contract.
* Partial fast tails retain real committed targets and at least two free rows.
  Unsupported suffixes are noise, not fabricated clean targets. Conditional-Z
  supervision can only reveal a contiguous valid action prefix.
* Arm boundary loss matches expert action increments, including the splice.
  It does not penalize all motion toward zero; grippers are excluded. Weight .1.
* Dataset commands use arm targets at interval END and absolute gripper commands
  held causally from interval START, matching the actual actuator. Endpoint
  gripper labels previously anticipated opening/closing by up to one row.
  This is V8-only; native/TOPP datasets retain their own action semantics.
* New recipe metadata replaces old evaluation's hardcoded 1k/10k/20k gate.
  Formal deployment must match its actual completed phases and exact source.

## Recipe (subject to measured eight-rank throughput/memory)

Start from original `DynamicWAM_full.pt`, SHA256
`7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`.
No diagnostic checkpoint may initialize formal training.

Freeze native slow throughout; train current encoder, structured readout and
single 448/6/7 joint A/Z fast decoder. Keep native BF16 video/FP32 time/action,
FP32 optimizer masters and accumulation. Independent draft/future dropout .2,
cache dropout .1, boundary weight .1. No extra distillation or invented Z labels.

Proposed phases: warmup 500, small_age_joint 4000, cross_time_joint 8000 updates.
Global batch32 = 8 ranks x micro2 x accumulation2. AdamW fast LR1e-4, final phase
5e-5, cosine floor .1, max grad norm1. Keep model/masters/Adam across stages;
reset phase RNG/cursor/LR for its curriculum, restore all on within-phase resume.
Actual smoke must establish finite gradients and full reload on
all eight ranks. Reduce batch only via a fresh accepted smoke/profile.

Use deterministic half of training episodes stratified by task/style; preserve
the complete validation split. Existing normalization is fit on the original
training split, including unselected training episodes, never validation.
All stages use measured parallel capture/source/query/prefix/publication events;
young-age curriculum is selected from actual available age quantiles, then all
ages and masked tails. No independently sampled fictitious clocks.

## Gates, not performance claims

Two real autonomous initial-model episodes at the preceding two-row cadence
passed full GPU/Graph/slow-process parity and clean exit: 62 active queries,
all two-step prefixes; query medians .089/.113 s, max .153 s; backlog maxima
.008/.018 s; refreshed M consumed. They ended at the diagnostic action budget,
not success. This does not establish SR or accept the new-frame cadence.

Before formal: final-source CPU adversarial queue/clock/history tests, actual
35-task bounded engineering suite (untrained, unscored), real label/time audit,
new timing indices, three-phase actual 8-rank smoke and strict reload/clean exit.
The engineering suite uses32 rows per task; it is not a shortened official eval.

Frames retain real exposure and availability. Async overload may drop a capture
slot; frame identity is episode plus real exposure/tick, not ordinal position.
No identical capture-count guarantee is made across wall scheduling workloads.
Same-command physical replay and causal sampling/masks must be checked separately.

Demonstration labels remain bounded physical-time resampling of the recorded
expert trajectory. They are not counterfactual labels for new on-policy actions.
Slow video after step2 is not guaranteed consistent with final A10; future
attention is independently revocable. Both are posttraining ablation/collection
questions, not solved by timestamp metadata. World-residual event refresh remains
disabled pending observable residual calibration and equal-compute ablation,
as ordered by the concise design document. No score floor is guaranteed.

Four failure views and on-policy/SR ablations are distinct from infrastructure
readiness. Missing raw observations must be marked unavailable, not reconstructed
from simulator truth as policy input. Formal700 and a new same-protocol Full
comparison remain posttraining work.
