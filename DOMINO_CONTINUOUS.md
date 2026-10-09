# Continuous-wall DOMINO Level-1 protocol

This is a new execution benchmark. It does not reproduce the published native
DOMINO controller or its success-rate table. Native TOPP assigns a variable
physical duration to each action row; multiplying native action counts by
100 ms is not a timing conversion.

## Physics, observations, and action time

`runtime/domino_rt/continuous_loop.py` owns a 4 ms physical clock. Model
inference runs in `ProcessPolicyWorker` using `spawn`; the owner never waits
for a returned policy result during an episode. This includes the first
formal query. One pre-episode camera/model warmup is allowed and reported
separately; warmup collection and retained policy state are forbidden.

Level-1's saved expert velocity is integrated against physical time by
`motion.py`, including when the robot holds its last target. Contact-stop,
boundary, lift, success, and transient task checks run during inference as
well as execution. Metrics are sampled at a fixed 100 ms cadence, plus the
terminal state, for every arm. This differs from native per-TOPP metrics.

Observations originate at exact 100 ms grid boundaries. A returned row `i`
is a target at `action_origin_sim_time + (i + 1) * 0.1`. The default action
origin is the observation time. A policy reusing a previous source plan may
explicitly set `metadata.action_origin_sim_time` to its source-aligned time.
The current interval is committed. New results replace only unstarted future
intervals, at the next grid boundary; already elapsed rows are discarded.
Expired results do not clear a usable old tail. Set `metadata.replace_tail`
to false when publishing only a short correction while preserving other
old future knots.

`LimitedHermiteController` maintains continuous commanded position/velocity
across interval handoffs. Arm command velocity and acceleration defaults are
2 rad/s and 20 rad/s², and limiter activation is reported. These are explicit
controller choices, not measured physical joint limits or native TOPP claims.
DOMINO `get_*_arm_jointState` exposes drive targets, not measured physical q;
the packet marks this state semantics explicitly.

`sync` queries only after its installed plan is exhausted, while physics
continues under a hold during inference. `async` may query every two control
slots if the single worker is idle; old commands continue while it computes.
Full WAM, graph-optimized full WAM, and context-feedback policies all use this
same controller. The benchmark provides asynchronous execution; it does not
claim to implement RTC's diffusion inpainting/guidance by itself.

## Wall-clock validity

There is no virtual delay, pause-world branch, initial frozen inference, or
clock reset in the timed loop. The physics owner publishes an immutable scene
snapshot to an independent renderer process. The replica restores every rigid
entity, articulation root/qpos, camera pose and light state without stepping
physics; a full scene/asset identity and a pose/FK oracle must match. It uses
the same 32-sample, depth-8 RT/OIDN rendering configuration. Only one render is
in flight and only the newest pending snapshot is retained. Image completion
never changes its source timestamp, command state or plan identity. A completed
image of an already replaced plan is discarded before policy submission.

Snapshot copies, optical flow, IPC, events, and logging remain charged to the
physics clock. Renderer and policy contention on the GPU remains measured.
This is bounded-jitter real-time simulation, not a hard-real-time guarantee.
Default maximum wall/physics backlog and inter-tick gap are both
100 ms. A violation terminates with `infrastructure_invalid`, writes its
trace, and is excluded from valid success-rate denominators. Gates may not
silently relax these limits after observing results.

Reports retain raw tick lag/gap arrays, observation/action ages, request and
worker wall intervals, physical ticks occurring during each worker interval,
late-prefix clipping, publish times, hold intervals, and remaining requests
at episode termination. The comparison of interest is actual command/observation
age and deadline compliance at matched valid task success; a serial amortized
compute formula alone is not evidence of responsive continuous control.
Reports also retain acquisition, renderer-ready, worker and actual installation
wall timestamps, camera-in-flight physical ticks, stale camera drops, and the
number of feedback-corrected rows actually executed after prefix clipping.

The legacy `--synchronous-camera` option is diagnostic only. Its approximately
95 ms camera work blocked the physics owner and failed the 100 ms backlog gate
in real-scene testing; it is not an admissible formal benchmark configuration.
Simulation concurrency is one until explicit Vulkan device binding and concurrent
clock validity have been demonstrated. `CUDA_VISIBLE_DEVICES` alone does not
establish SAPIEN Vulkan isolation.

## Fixed-scene pairing

`preparation.py` runs the expert filter for one requested seed, retrying that
same seed at most ten times. It never substitutes another seed. The immutable
accepted record stores expert motion, the selected instruction, pre-initialization
RNG, portable task-owned state, initial target pose, and DOMINO/config source
hashes. In particular, `put_object_cabinet` and `place_object_scale` need the
expert-selected `arm_tag`; reconstructing only the object motion is insufficient.
Replay bypasses the expert planner, restores state/RNG, and verifies source
identity and target starting position. The pickle is a trusted internal
artifact, never an untrusted uploaded file.

## Entrypoints

Use the configured DOMINO Python 3.10 executable without resolving its symlink.

```bash
CUDA_VISIBLE_DEVICES=0 external/robotwin/bin/python3.10 scripts/eval_domino_continuous.py \
  --config configs/absolute_motion_v2.yaml --task place_shoe --seed 0 \
  --accepted-record /absolute/records/place_shoe_0.pkl \
  --output /absolute/results/prepare_place_shoe_0 --prepare-only

CUDA_VISIBLE_DEVICES=0 external/robotwin/bin/python3.10 scripts/eval_domino_continuous.py \
  --config configs/absolute_motion_v2.yaml --task place_shoe --seed 0 \
  --accepted-record /absolute/records/place_shoe_0.pkl \
  --output /absolute/results/async_place_shoe_0 --mode async \
  --policy-config /absolute/policy.json --physics-timeout 30 --query-slots 2
```

The factory defaults to
`dynamicwam.runtime.world_context_worker:create_worker_policy(config)` and
must return an object exposing `sample(packet)`. Return a finite unnormalized
`[H,14]` absolute joint-command array, or `{actions, metadata}`. The packet
includes the three-camera composite, raw cameras/current and previous head
RGB, actual history dt, command state, instruction, flow/motion features,
physical timestamp, control period, command index, episode/query identity,
reset/warmup flags, and a deterministic sampling seed. The factory owns policy
reset and RNG handling. It must not collect warmup packets.

Each unique output directory contains `summary.json` and `events.jsonl`.
Prepare status is `prepared`; valid episodes are `complete`; runtime errors
and clock violations are explicit and exit nonzero. Existing run directories
and accepted records are never overwritten. `mock_worker:create_mock_policy`
is only a clock diagnostic and must not be reported as learned-policy quality.

CPU validation: `python tests/test_continuous_domino.py -v`.
