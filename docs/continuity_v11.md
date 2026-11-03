# Continuous RTC V11: Full-anchored bounded feedback

Candidate 2026-11-03. This document records the implemented scope, not an SR guarantee.

## Evidence and choice
Full continuous linear replay was already stable. Identical-command comparisons also
accepted common-phase C2; independently projecting each joint changed the Full path
and caused contacts. Keep phase_c2 and actual contacts. Do not introduce that projector,
retime the bench, or treat collision-induced achieved-state error as proof of a policy bug.
The previous broad QP/spline proposal is deferred pending same-command evidence.

## Iteration 1: train what the controller should do
Single 448/6/7 A/Z model; frozen native Full slow and fresh-observation Full teacher.
Teacher correction is primary, not expert-minus-an-old-counterfactual-plan regression.
The primary target is the feasible fresh Full endpoint minus the actual aligned slow
baseline. Preserve near-zero corrections where the teacher agrees. Downweight local
recovery examples outside 1.5 times the trust radius. Full command increment loss includes
the known-to-new seam. Expert compact future labels are used only conditioned on expert
actions. Labels for unobserved outcomes of the student are never fabricated.

## Iteration 2: temporal continuity and real feedback
Actual q/qdot are measured by the scene owner at camera exposure. Inputs include actual
q, q minus drive target, qdot, and physical age; the original Full drive-state input is
unchanged. Missing sensors are masked. TRAIN scenes supply real achieved-state replay;
scored seeds cannot enter this pool. All selected half-data expert windows remain available,
with a declared 25 percent replay sampling weight. Replay Full baseline regeneration is
checked against the recorded baseline, with identical source and physical-interval noise.
A previous accepted plan is aligned by both FP64 interval endpoints, never ordinal position.
Pair training compares the same future interval across queries, using fresh teacher changes
rather than penalizing all reaction. Per-query same-reference correction innovation is at
most 1/3 of the 0.06-rad total arm radius. Grippers retain Full's semantics and zero correction.
The convex temporal basis and bounded innovation run identically in train and deployment.

## Iteration 3: independent refresh and deterministic contracts
Slow candidates continue publishing independently. The active reference has a minimum
4-row dwell, but cannot outlive TTL or leave fewer than two covered editable rows. The
previous accepted correction continues only with the same active reference. A new reference
resets the correction anchor; it is not falsely identified as the old physical trajectory.
Video noise is fixed per episode; action noise is keyed by physical action interval.
This removes gratuitous resampling across overlapping slow plans, not state-dependent
planning. Offline windows define their own real physical grid; their adjacent queries
share that grid. No learned timing adapter or clock relabeling is used.
Execution history is retained for the active reference, not only the newest candidate.
No extra EMA, post-hoc prefix edit, time stretching, or old-plan fallback bypass is added.

## Acceptance and training
Three commits/reviews, deterministic CPU tests, real Full source parity, new/old reference
consumption, true parallel fast/slow queries, Graph parity, physical/wall deadlines,
source/prefix identity, native MP4 full decode, actual tracking and clean owned exit.
Then all-task train-scene collection, new measured timing index, real mixed-data loading,
three-stage eight-rank smoke with strict optimizer/master/RNG reload. Only then formal
training starts from DynamicWAM_full.pt, never the diagnostic fixture. Eight ranks,
microbatch 1, accumulation 4, 500/4000/8000 updates are the initial measured-cost budget.
A finite independent supervisor gates final-checkpoint smoke and 35 x 20 eval. Failures
stop dependent stages and retain evidence; no automatic retry or GPU recovery.

## Limits
Stable commands do not imply task success. Frozen Full with RTC-conditioned prefixes and
changed refresh semantics is not automatically the native Full policy. The initial real
replay is Full-anchored behavior data, not a complete on-policy DAgger distribution. Reuse
of a stable reference trades freshness for continuity and needs measured evaluation. The
phase-C2 curve preserves endpoints/path, but is not a global velocity/acceleration/jerk
constrained optimizer. Report comparisons only against an equal benchmark/controller budget.
