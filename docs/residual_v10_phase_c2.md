# V10 revision: preserve Full commands before learning feedback

Candidate on .50, same route branch. Formal training is gated on fresh receipts.

## Three reviews and selected design

1. **Causal executor review.** On the SAME successful Full basket trace, original
   linear tracking max error was 0.007313 rad, raw quintic 0.006901, old projected
   quintic 0.337568. Projection changed endpoints by up to 1.282488 rad and induced
   link/table contact. Removing collisions removed the tracking error, but is a
   diagnostic intervention only. It does not establish valid manipulation.
2. **Control/architecture review.** Use unchanged Full joint endpoints plus bounded
   clean learned corrections (arms .06 rad, gripper .05 raw units), exact immutable
   RTC prefix, original URDF joint limits. Between endpoints all joints use the
   common phase s(u)=10u^3-15u^4+6u^5, u in [0,1], on the original 45 physics ticks.
   q/v/a join continuously at zero boundary v/a. This preserves each straight
   joint-space segment, endpoints and timestamps; no per-joint retiming/projection.
   All collisions, drives, passive forces and solver parameters remain native.
   It does NOT guarantee collision freedom or arbitrary derivative bounds. Peak
   speed is 1.875 times the segment secant; stop/go and actual tracking need GPU
   comparison. Legacy `limits` fields are metadata of the rejected projector and
   are not enforced by phase_c2. The new protocol is named `_phase_c2_v10`.
3. **Learning/time review.** Fast remains one 448/6/7 compact A/Z transformer,
   deterministic zero-initialized residual. First executable row is upweighted.
   Expert and current Full teacher supervise the CLEAN residual, not a velocity
   field which leaves random action noise at initialization. Teacher/source samples
   share interval-matched noise; old reference is neither a commitment nor a fresh
   teacher. Latest RGB/spatial motion/state enter every query. A 29D input explicitly
   compares measured proprioception with the old reference drive command evaluated
   at that exact exposure, with unsupported intervals masked. This is not future
   GT and not an actual-state prediction. Teacher/expert-agreeing nontrivial
   corrections receive up to 3x weight. No loss rewards arbitrary image sensitivity.

The inference conditions no longer recompute the same V8 draft a second time.
Phase endpoint bounds run on GPU without the old serial CPU projector transfers.
Query budget is chosen before the query (.22s candidate, actual 2/3-row prefix),
never extended after dispatch. New real traces rebuild training timing/queue data;
old all-two-prefix data are not silently reused. Actual physical intervals define
action/Z joins, wall time defines readiness/release; neither is a learned clock.

Zero residual must equal RAW aligned Full commands (apart from native bounds),
not a heavily projected surrogate. This is a command identity, **not an SR lower
bound**: stale observations, changed replanning and a different executor remain.
Offline expert/teacher training also cannot establish on-policy feedback quality.
After training, compare fresh vs held-back observations at identical old reference
and physical targets, and run autonomous paired evaluations; do not infer useful
reaction from a nonzero gradient or fast inference alone.

## Research informing this choice

- [Ruckig](https://github.com/pantor/ruckig) uses phase synchronization to preserve
  straight joint-space motion; its minimum duration is not an exact duration
  contract. This implementation borrows the common-phase idea, not Ruckig itself.
- [ROS2 trajectory representation](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html)
  distinguishes interpolation continuity and warns about inconsistent derivative
  specifications. Smooth interpolation does not certify a feasible collision path.
- [Residual RL](https://arxiv.org/abs/1812.03201) supports base controller plus
  learned residual decomposition, not a guarantee that a learned residual improves
  every episode. Our labels are imitation/teacher targets, not RL rewards.
- [RTC](https://arxiv.org/abs/2506.07339) preserves already committed execution
  while replanning the rest; no averaging or retroactive rewrite of that prefix.
- [DuoCore-FS](https://arxiv.org/html/2512.20188v1) combines cached slow guidance
  with CURRENT perception and trains across time offsets. Its reported performance
  does not transfer automatically to our model, control rate or simulator.

Required before formal: native-contact Full executor replay; new autonomous
zero-residual GPU smoke with real parallel queries/new references; phase-command
and timestamp equality; all-task coverage; measured half data and real label
audit; three-stage eight-rank updates/strict resume. Record each independently.
No collision disabling, source mutation during jobs, automatic retry or GPU reset.
