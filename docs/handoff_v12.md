# V12: bounded whole-plan handoff and local feedback

Continues the V11 continuous simulation + RTC branch. Official Full is frozen;
the single 448/6 A/Z controller remains a deterministic residual model. Formal
initialization is DynamicWAM_full.pt, never the diagnostic checkpoint or V11's
partly trained weights. Historical 41.72% feedback results motivate the local
feedback design but provide no score floor for this continuous protocol.

## Three review rounds

1. Counterexample and algebra: residual bounds alone do not bound a change of
   Full references. Separate immutable raw Full B, learned delta, and explicit
   whole-plan connector. Test complete horizon, prefix, gripper and native limits.
2. Training/deployment data: one implementation constructs commands and teacher
   targets. Preserve recorded previous-query times and cross-reference inputs;
   do not replace mixed replay history with synthetic paired-query placeholders.
   Actual TRAIN rollouts must contain cross-reference events before formal launch.
3. Current-source GPU: Graph parity, real fast/slow overlap, native MP4 decode,
   physical timestamps, accepted queue deadlines, whole-plan audits, contact-on
   executor comparison, three-phase eight-rank smoke and strict optimizer reload.
   CPU tests alone do not satisfy this gate. Receipts establish actual status.

## Complete-plan controller

`handoff_v12.reconcile` aligns the last accepted full proposal to the new query's
FP64 physical intervals. A common scalar alpha across all arm joints and all
valid mutable future rows limits changes to that full plan. Normal innovation is
0.02 rad per 0.18-s query interval; a new-reference/ongoing handoff permits 0.06,
scaled by actual query spacing in [0.5,2] nominal intervals. Thus a far-tail
discontinuity is not merely deferred until it becomes executable. Missing tail
uses the new Full increments plus the final shared offset; this is explicitly
a generated connector. Without overlap, connect from the committed endpoint.
RTC prefix remains bitwise authoritative. Native limit violations fail closed.

Raw Full B is never overwritten. Q=B+delta, command=P+alpha*(Q-P), effective
baseline=P+alpha*(B-P), effective residual=alpha*delta. Logs distinguish all three.
Zero learned residual preserves the effective baseline, **not necessarily raw
Full during handoff**. It is not a native-Full success guarantee. Grippers retain
previous Full events during a handoff and switch to new Full events afterward;
there is no learned gripper correction or fractional gripper blending.

Slow scene/context inputs remain available; predicted-future tokens are masked
while their raw action plan is materially being reconciled. Compact Z/contact
outputs are invalidated when the connector changes the sampled action plan.
No truth labels enter deployment. No learned clock synchronization is introduced.

## Loss and precision diagnostics

Fast corrections apply only to the first four editable rows, bounded by 0.06 rad
in physical joint units. Main target is fresh Full teacher minus old aligned Full;
the residual head is explicitly supervised before the connector. This prevents a
saturated connector from hiding a wrong proposal through zero gradients. Additional
losses compare commands with the identically reconciled teacher and constrain
adjacent complete-action increments. Future supervision uses expert-conditioned
actions; achieved-state replay has no fabricated future/contact labels.

Teacher errors and over-radius counts are reported separately for first executable
row, near four rows, and tail, each for expert/replay/all domains. Counts refer to
valid mutable rows with any arm beyond radius, not episode or executed-action SR.
Large first-row errors remain a model/control limitation requiring evidence; tail
averages cannot justify increasing feedback amplitude. Existing scene, budget,
clock, two/three-row lease and phase-C2 executor contracts are unchanged.

Half of expert episodes, all selected windows, plus 25% TRAIN-only real feedback
sampling; three stages 500/4000/8000 updates, global batch32. Previous-query student
forward is retained, unnecessary previous-query Full-teacher forward removed.
Real collection is initially zero-residual Full-driven; it is not comprehensive
student on-policy training. Phase C2 gives shared position/velocity/acceleration
continuity at knots but is not a global jerk/velocity optimal planner. Full closed
loop MP4s and 35x20 final evaluation must establish stability and task precision.

## Reference availability gap found in real collection

In place_empty_cup, a camera delay deferred queries; the next valid source did
not finish before the previous reference's 2-s TTL. Its already accepted queue
still had predicted executable coverage, but `ReferenceLease.select` aborted
the episode before a new query was created. `ReferenceUnavailable` now defers
only that new query. Physics and the accepted queue keep running; the slow worker
remains independent. No expired reference is passed into a model, no TTL or
deadline is extended, and eventual queue underrun is still an error. Independent
event acceptance rejects deferrals when an actually usable reference exists.
Retained completed TRAIN episodes require native MP4/clock/replay hash checks
and AST proof that their successful model and dispatch paths are unchanged.
The failed partial episode is excluded and its task must be recollected.
