# .50 Continuous + RTC V8 development — 2026-10-31

This is an isolated development candidate, not an accepted training recipe or
an SR improvement. The .92 V7 training and all previous .50 artifacts are protected.
Baseline review commit: c1edaea74c8ba1c62c25009b7a621aac7872b2ae.

## Fixed simulator contract

Reuse the tested continuous implementation from .50 `dualrate-reference-acceptance-20261030`
(5ec849b78e72c2dfa68c4a7db9905556c9797530): async camera, GIL-releasing native
PhysX call UNDER SceneStepClock, contact/passive-force reuse, independent slow
process, immutable wall origin, 1x real stepping, wall release and both deadlines.
No step skipping, time subtraction, origin rebasing, hidden pause or automatic retry.

`ContinuousStepOwner` now owns every real step. It rejects calls from another
thread, calls outside its explicit advance, reentrancy, changed step wrappers,
and any clock change other than exactly one actual timestep. The camera result
keeps its capture ID. Reports join capture/frame identities, real history endpoint
times and masks, query leases, reference versions, commits and row execution.
The online flow selector now calls the training selector; no query-count history.

The first comparison retains the existing **linear endpoint** controller: 45
actual float32 .004 s ticks per action, ~.18 s. Grippers retain absolute native
commands. Backlog allowance is explicitly .5 + .005 s, inherited from the user's
authorized relaxed continuous protocol; camera age and lease checks are unchanged.
This does NOT prove comparability with the old .25 s / 25.43% Full result. Full
must be evaluated on the same new protocol before claiming superiority.

Continuous simulation does not require linear interpolation. Streaming native
TOPP or constrained C1/C2 trajectories can share this owner, but cannot silently
replace the action contract. Fixed endpoint feasibility, actual q/qdot boundary
conditions and Z label consistency must be audited first. No trajectory smoothing
gain is claimed here; these candidates remain separately gated.

## Single fast model and explicit ablations

Use a 448-wide, 6-layer, 7-head A/Z joint model with current RGB/flow/state/text
encoding. Z remains the existing three supervised roles x four physical horizons;
no fabricated support/occlusion labels or new distillation. Full initializes the
slow model; the changed fast model requires training and is not a direct reuse of
the native V7 checkpoint. Full slow is frozen in this first candidate.

`continuous_v8.aligned_draft` joins BOTH endpoints of physical action intervals,
checks published availability, age, action-token validity and uniqueness, and
permits explicit per-row revocation. A stale or uncovered row uses zero draft;
it is never extrapolated or matched by ordinal. Gripper draft components are zero.

Residual parameterization: `x_t = draft + (1-t)*(target-draft) + t*noise`.
The model sees noisy FINAL actions and the numerical draft+validity. FM velocity
is `draft+noise-target`. The sampled final arm action equals draft plus an
unrestricted residual; no clipping toward A10 or two-model output averaging.
Known ABSOLUTE prefix values are clamped every denoising step. With no draft the
same decoder is the full-action mode. Independent draft dropout trains that mode.
This is an inductive bias, not a guarantee: a sufficiently expressive full-action
model could learn the same transformation.

`IndependentWorldBlock` separates attention normalization for pre-joint observed
source tokens and predicted future tokens. Current evidence stays direct. Future
attention has a hard-disable mask plus a learned hard-sigmoid gate that can reach
zero; history remains accessible. Draft validity is independent. Type-2 action
tokens are no longer read as generic memory; the numerical draft is explicit.
The source comes from the native pre-joint export; predicted hidden/latent tokens
remain potentially globally mixed, not magically pure local future entities.

Keep fixed refresh for the first ablations:

| Variant | Action parameterization | Future attention |
|---|---|---|
| control | full | shared history/future gate |
| residual only | residual | shared gate |
| residual + reliability | residual | separate gate |

The `action_mode` and `independent_future_gate` settings bind checkpoints. Native
event-coordinate branches are explicitly rejected by the V8 factory. CUDA Graph
copies both mask planes and draft-derived features each query, preserving the
same eager/graph sampler and per-step commitments.

## Event refresh: deliberately later

`RefreshController` implements unique-frame persistence, hysteresis, minimum and
maximum interval, one-token budget, single in-flight job and coalesced latest need.
It never changes physics time or blocks fast queries. It is **not enabled in the
production runtime** yet: current Z labels rely on training ROIs, and we do not
yet have an accepted deployment-observable, same-time world mismatch measurement.
Arbitrary latent distance or action disagreement must not masquerade as one.
Calibrate that signal and compare equal average slow compute before enabling.

## Data and train/deploy gates

`continuous_contract.validate_training_contract` rejects native V7 event data and
requires physical endpoint actions, same-time labels, actual commitments, .72 s
history, sampled slow drafts and measured parallel timings. Formal training also
requires a hash-bound receipt for simulator parity, real GPU concurrency, history,
delay queues, physical labels and all-task smoke. No old native manifest is renamed
to pass. V8 is not yet ready for formal training; .92 training remains untouched.

## Acceptance matrix from both supplied documents

| Requirement / experiment | Implementation and evidence | Remaining gate |
|---|---|---|
| E0/E1 single owner, fixed actions under load | owner guard; replay scripts; full state/timestamp byte comparison | real replay receipts, then multiple tasks/seeds and TOPP baseline |
| E2 history independent from query | immutable capture history; shared physical selector; irregular-time tests | production-log flow tensor replay across all task families |
| E3 delay/RTC | existing queue lease, prefix-content digest, wall release, dual deadlines regression | expanded real delayed/out-of-order/reset workload |
| No physical pause / double step | owner and real SceneStepClock checks on every step | prolonged concurrency and paired whole episodes |
| Arm/gripper semantics, event/contact scores | preserve production executor/checks; exact state replay | compare contact/metric event streams and constrained-trajectory feasibility |
| Residual and absent-draft mode | exact interval join, full numerical draft, separate grippers, dropout, joint A/Z; forward/backward + reload tests | continuous dataset audit and training |
| Old future independently disabled | source/future provenance, separate masks/attention and hard gate; poison-future test | good/bad-reference grouped closed-loop benefit |
| Budgeted world refresh | controller unit tests; not enabled without comparable observable residual | calibration + equal-budget E7/E8 refresh ablation |
| E4/E5 RTC on/off, prefix scan | runtime contracts retained; variable-prefix tests | paired actual 2/3-row latency/response and SR |
| E6 fresh M, E7 age, E8 memory ablations | explicit draft/full/future switches | trained-policy experiments |
| E9 source-conditioned worlds | source includes actual prefix; video2/A10 consistency NOT asserted | branch replay with executed deviations and future-label validity |
| E10 deterministic vs real concurrency | recorded-action replay plus actual fast-core load probe | full slow/fast/renderer integration and long stress |
| E11 Z ablation | unchanged joint Z representation | noZ/auxZ/jointZ matched experiments |
| Artifacts and failures | native MP4 + timestamp sidecar, checksums, isolated output, guarded clean exit | all-task episodes and four failure diagnostic plots |

The real fast-core probe uses synthetic contexts and random weights on GPU1,
render/physics on GPU0. It tests actual 448x6 K10 eager/Graph parity and compute
interference with the recorded commands; it excludes observation encoders, slow
model, IPC and task quality. Do not label this an end-to-end latency or SR test.

## First actual .50 acceptance results

Numerical source: `ad1518e0ed35f16f9c4676d7c141a5c9a70610ab`;
source SHA256 `a31f09dd64486ebd0be9e70e9dbff270ef635dd4914c99c8ac7b82857cb5b810`.
Artifacts: `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_continuous_v8_20261031`.

| Real GPU diagnostic | Camera only (`replay02`) | Camera + fast core (`replay_core03`) |
|---|---:|---:|
| Actual physics steps / saved states | 1755 / 1756 | 1755 / 1756 |
| Every saved state/timestamp vs old workers0 reference | byte-identical | byte-identical |
| Physical duration | 7.020000333 s | 7.020000333 s |
| Active wall duration | 7.024310206 s | 7.025567084 s |
| Maximum backlog | 415.707 ms | 422.653 ms |
| Published video frames | 34 | 33 |

The two native MP4s decode successfully. Both guarded jobs completed, owned
processes exited and all eight guard descriptors closed with unchanged kernel
evidence. The new fast core has **27,749,602 parameters**; real K10 eager/Graph
parity passed. Twenty background queries: median **36.765 ms**, max **47.486 ms**.
These are core-only figures with fixed synthetic contexts, NOT closed-loop
observation-to-action response. No performance/SR claim follows from them.
Final selected CPU regression suite: **194 passed**, including the new policy,
phase freezing, clock/queue, history, camera and launch/checkpoint contracts.

This case needs the explicit 500 ms budget; it did not pass the old 250 ms limit.
The capture counts differ under load. Although capture insertion is independent
of queries and the physical state trace is invariant, async rendering can miss a
scheduled slot. Therefore strict capture-log equivalence E1 is still **pending**;
one must align real exposures and account for missing frames, not relabel ordinal
frame IDs as fixed-time samples. No all-task, long-duration or full-model RTC
deployment acceptance is claimed from these two short diagnostic replays.
In particular, a >.4 s owner/physics stall can still threaten a .36 s RTC lease
even with a <.05 s fast core. The replay does not commit model outputs, so it
cannot clear that deadline gate. Full query completion/publication/activation
traces and measured prefix-distribution coverage must pass before training.

`replay01` is preserved as a failed pre-simulation launch: the imported guard's
worker-count field incorrectly used eight (guarded adapters) instead of one
(replay worker). This was repaired and regression-tested; no GPU recovery or
automatic retry occurred. All diagnostic attempts and their cleanup receipts
remain separate.
