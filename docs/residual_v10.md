# Continuous RTC residual V10 — candidate, 2026-11-02

**Current revision:** see [phase-C2 correction](residual_v10_phase_c2.md).
The per-joint derivative projector described below was REJECTED by real contact
replay. Production phase_c2 preserves endpoints/path and does not apply those
velocity/acceleration/jerk clamps. Historical description below is not acceptance.

Status: implementation and acceptance in progress. Formal training is NOT yet
authorized by a passing acceptance receipt. Worktree/branch remain the .50 V8
training route. Old checkpoint/results are retained; archive tag:
`archive/v8-before-residual-v10-20261102`.

## What changes, and why

V8 completed 26/700 (3.7143%). Its residual FM loss was algebraically correct,
but zero vector field left `draft + noise`, not `draft`; corrections were
unbounded. Large alternating commanded endpoints occurred even without a slow
reference change. A compact network connected to RGB does not establish useful
feedback. The old 25.43% Full run and the V8 run also have different external
budgets/protocols; neither is an automatic lower bound on this method.

The compact 448/6/7 transformer now predicts a **clean bounded residual in one
deterministic pass**, plus conditional compact future Z. No action sampling
noise, no second action generator. The action head is zero initialized. All 14
coordinates, including grippers, retain their aligned Full draft. The residual
radius is expressed in raw joint/gripper units before conversion by checkpoint
statistics. A zero correction reproduces the feasible slow baseline within
floating point round-off under the SAME executor. This is an action identity,
not a success-rate guarantee or parity with the old linear executor.

Old world/action source, current observation encoding, independent history/future
cross attention and Full slow architecture remain. Slow stays frozen. Production
still runs fast and slow on separate GPUs without suspending physical stepping
for inference. The slow action is revocable guidance; only the queue prefix is
an immutable commitment. Both physical interval endpoints must match.

No valid slow coverage for the first executable row is an explicit failure;
there is no random full-action fallback. Uncovered tail rows are masked, never
claimed to be a valid old prediction. Source refresh/TTL/horizon acceptance must
show that normal deployment avoids this condition.

## Accuracy and response supervision

Training has two separate action targets: real expert actions and a fresh Full
teacher conditioned on the latest available observation and authoritative prefix.
The teacher starts on the global grid surrounding its REAL exposure; its output
is joined to fast actions by physical intervals, not array index. Coupled noise
on overlapping intervals reduces spurious teacher changes from independent
sampling. This teacher is training-only; fast inference never waits for it.

The clean head itself is supervised, in addition to the projected executable
action, so feasibility clipping cannot eliminate its corrective gradient.
First executable rows have greater weight. Preservation loss is applied where
the target agrees with the draft; temporal difference supervision targets the
required correction rather than indiscriminately averaging all motion away.

A causal queue augmentation changes only a future protected row which has not
started and was not already committed at the old slow source. It never alters
past observations, the active row, or source commitments. Perturbations obey the
same continuation constraints. Fresh Full relabels these counterfactual futures;
expert action/Z/contact labels are not reused as their ground truth. This is
synthetic future-queue robustness, NOT on-policy data or simulated new images.

Z targets move from query + [0.1,0.2,0.4,0.8] s to first mutable action START +
those offsets. The producer makes actual labels at those times; runtime requests
the same times. Ground-truth future supervision conditions on the recorded
expert trajectory, without declaring the whole trajectory committed. Future
labels for a different executed trajectory remain masked.

## Executor and clocks

Keep exactly 45 actual float32 0.004 s steps per action. Do not rebase the wall
origin, pause physics, change exposure timestamps or retime actions to make a
trajectory feasible. The streaming quintic uses endpoint q, secant endpoint v,
and zero endpoint a, carrying these boundary states across rows. Bernstein
derivative control points certify q/v/a/jerk over the whole segment. The causal
feasibility projector is shared by training and inference. A trust box around
the projected Full trajectory limits the final correction; an empty intersection
rejects the correction, not the committed prefix. Grippers retain native
interval-start commands. Actual SAPIEN limits must equal the trained limits.

This choice is informed by [Ruckig](https://github.com/pantor/ruckig) and its
[jerk-limited online trajectory paper](https://arxiv.org/abs/2105.04830), not an
implementation of Ruckig. Ruckig's minimum duration is a lower bound, so simply
using it would not establish our exact execution timestamps. Likewise,
[TOPPRA](https://arxiv.org/abs/1707.07239) parameterizes path time and cannot be
substituted without changing action/label timing. Our fixed-duration projection
is conservative and may reduce reach per step; actual tracking and task success
must be checked, not inferred from smoother video.

Wall clock measures computational readiness and 1x release; physical clock owns
images, dynamics, action intervals and Z. Both submission deadlines remain.
The .5 s backlog allowance is not permission to relabel stale observations.
Protocol names include `fixed_c2_v10`; do not merge scores with old linear runs.

## Three review rounds and formal gates

1. Numerical architecture/executor review: zero correction, immutable prefix,
   normalization, grippers, interval matching, derivatives, trust bounds and
   finite gradients. First executor suite: 5 passed. First combined suite found
   a missing optional future-valid mask (25 passed / 1 failed); fixed explicitly.
2. Data/time review: real expert prefix feasibility, latest teacher origin,
   future labels after commitment, counterfactual label masking, measured runtime
   templates. Sampled 35 episodes / 1,961 prefix states: one infeasible under
   proposed v=2.5, a=20, jerk=400. Full index filtering and a rejection receipt
   are required, not silently modifying already committed labels.
3. Actual deployment review: production GPU eager/Graph equality, preserved
   zero-residual baseline, full asynchronous references, queue/deadline checks,
   fixed C2 commanded and measured motion, decoded native MP4s, clean GPU exit.
   Compare fresh versus stale observations at fixed draft/noise against aligned
   teacher and expert targets. Sensitivity alone is not useful adaptation.

Then run three-phase eight-rank update/strict optimizer-master-RNG restore smoke.
Only current-source receipts may enable a fresh official-Full formal sequence.
Half expert data is acceptable; validation and train/scene splits remain bound.
Durable supervision has no automatic retries or GPU reset/recovery. Post-training
closed-loop evaluation must establish stability, useful response and SR; none is
declared passed by pretraining engineering tests.

## GPU review 1 and scheduler review 2 (2026-11-02)

Frozen ec6d169 ran two unscored 32-command engineering episodes successfully:
58 fast queries, all two-row prefixes, real fast/slow overlap, native videos fully
decoded, exact zero-residual baseline within 5.1e-6 raw units. GPU guard/owned
workers exited cleanly and kernel unchanged. This was not task-performance
acceptance. Exposure-to-first-mutable action was 360 ms.

The old evaluation-only mailbox workaround held completed slow results while
fast queries were active: observed extra holds of 150--350 ms. The source now
reloads a receipt arriving during slow publication, eliminating that artificial
hold without weakening either deadline. A deterministic race test reproduces
that interleaving. Slow launch cadence changes from four rows to one: slow and
fast still have independent slots and only use actually available images.

The small causal projector is evaluated on CPU in no-grad deployment while
training retains differentiable CUDA. Bootstrap verifies actual CPU/CUDA outputs
and accept/reject decisions; transfer and projection cost remain in fast latency.
Actual articulation q/qdot are sampled separately from command q/qdot; bounded
command derivatives do not prove tracking accuracy or task success.
