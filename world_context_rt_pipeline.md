# World-context continuous-physics pipeline

This is a separate experiment protocol. It does not reuse native synchronous
DOMINO scores as its control and does not claim that the generated world has
been updated after a feedback correction.

The canonical controller is
`configs/world_context_rt/controller_protocol.json`: 4 ms physics ticks,
100 ms action slots, observation requests every two slots, a sixteen-row
native WAM plan, and slow refresh beginning at age twelve. Physics continues
in its own process while a single GPU worker performs slow and fast inference
sequentially. Both maximum wall backlog and maximum tick gap are 100 ms;
violations are infrastructure-invalid. The new velocity/acceleration-limited
controller is shared across all arms. Each episode has an explicit 30 s
physical-time budget. This differs from native TOPP execution.
Protocol version 2 renders immutable snapshots in an independent scene
process. Images retain the snapshot's physical timestamp; native RT rendering
quality remains 32 samples, depth eight, and OIDN. Both camera-inflight and
actual model-worker time windows must contain physics ticks. A camera's age
at return is not evidence of model/physics overlap.

`official_eager`, `graph_zero`, and `world_context` use the same asynchronous
controller and query schedule. Their differences are the native eager solver,
the mathematically checked graph implementation of the same solver, and the
additional bounded world-context feedback, respectively. `context_off` is a
fixed development ablation.

## Registration and progression

`make_world_context_rt_manifest.py` reproduces the original episode partition
hash. Within each task, it selects the first four original training episodes
for stage 2, the next two for stage 3, and the first validation episode. It
maps original episode indices through the raw collection's `seed.txt` and
excludes every seed occurring anywhere in the official 700. Selection occurs
before preparation and never substitutes successful scenes for failed ones.

Each scene is prepared once. The accepted record includes its motion,
instruction, task-owned state, initialization RNG, and source identities.
All policy arms replay those exact bytes, verified by SHA256. A seed alone
does not establish paired dynamic trajectories.

The driver runs these stages in a fixed order:

1. CPU contract checks (including worker query-RNG reproducibility) and a
   loaded-model initial latency/parity profile.
2. Prepare the first fixed development scene once. Run the real source/replica
   RGB and FK oracle against those exact accepted bytes, with `unseen`
   instructions, before any data collection. Four retained repeats per view
   must pass both initially and after physical robot/object motion. The driver
   verifies image SHA256 and recomputes RGB errors against measured source
   repeatability; it does not add a pixel-error allowance or trust `passed`
   alone. This comes from the run's frozen source, not a reused debugging run.
3. Actual continuous-clock development collection: the first two selected
   training scenes and one heldout scene for each of three fixed tasks.
   The first scene must pass actual worker/physics overlap, independent camera
   overlap, unchanged snapshot timestamps, and the strict clock bounds before
   the remaining scenes run.
4. Offline official-WAM teacher labeling, a four-step distributed smoke,
   and a loaded smoke-checkpoint profile.
5. A fixed 200-step world-context probe, loaded probe profile, and paired
   zero/context/context-off development rollouts.
6. Remaining stage-2 zero-behavior collection and labeling; stage-2 training,
   loaded latency, and another paired development tripwire.
7. New stage-3 scenes under the exact stage-2 learned checkpoint, followed by
   labeling and training with half stage-2 zero and half learned-behavior
   examples in every local microbatch; loaded latency and development checks.
8. The fixed 35 × 20 scene request list for all three formal arms, followed
   by artifact-paired final analysis.

The development probe must improve clipped teacher error over initialization
and zero feedback, improve over disabled context by at least 1%, and improve
over same-task shuffled context. It is a development mechanism test, not an
independent confirmatory result or a task-success guarantee. The small
closed-loop tripwire permits at most one fewer success than paired zero
feedback on the six fixed development training scenes; the three heldout
scenes are used for teacher-error evaluation. These reused development
scenes do not establish independent generalization. Zero successes in both
arms stops the pipeline as an uninformative protocol/behavior outcome; it
cannot authorize long training or replacement seeds.
Each learned-policy canary must also contain at least one nonzero corrected
target consumed for positive physical time. The gate reconstructs returned
row indices, prefix clipping, actual installation and consumed intervals;
generating four residual rows which all expire cannot pass. It adds no minimum
delivery ratio or magnitude requirement. This is a necessary delivery check,
not proof that bounded actuator commands or task outcomes improved.

Loaded fast p95 must be below 100 ms; measured slow-plus-five-fast service
time per twelve slots must be below 85% of native eager service time. All
timings are recomputed from raw samples. This computation profile does not
replace the actual physics-clock gate, and a synthetic packet is labeled as
compute-only evidence. Every checkpoint is profiled after loading.

## Fixed training budget and resources

Global batch is exactly 24. With six visible cards, each rank uses batch two
and two accumulation steps. If fewer cards are visible at registration, the
largest supported rank count is chosen and accumulation is adjusted to
preserve global batch; the rank count is not changed because a GPU happens
to be busy. Selected GPU UUIDs are stored. Busy cards are waited for; unrelated
processes are never stopped. Simulation concurrency is separately fixed to
one GPU-0 renderer/worker pair. CUDA visibility alone is not accepted as proof
that six independent Vulkan renderers would use different cards. Both graphics
and compute processes are checked before acquisition. Six-card DDP training
starts only after every simulator/renderer process has closed. The first real
development scene provides a single-scene clock gate before further collection.

| Stage | Optimizer steps | Examples | LR | Initialization |
|---|---:|---:|---:|---|
| 2 | 4,200 | 100,800 | 0.00002 | fixed trained v1 feedback head |
| 3 | 1,200 | 28,800 | 0.000005 | this run's verified stage-2 export |

The complete official WAM remains frozen. Supervision uses offline teacher
queries at actually recorded on-policy observations. It does not blend expert
trajectory actions or use unobserved future video. Plan ages 0–15 are allowed
because asynchronous publication need not align with even action indices.
Stage 3 retains all stage-2 records and the same heldout episode identities.

## Launch and records

The executable entry is `scripts/run_world_context_rt_pipeline.py`. Supply
absolute paths to `--run-root`, `--project-root`, `--episodes` (packed
`episodes.jsonl`), `--raw-root`, `--official-seed-manifest`, `--base-checkpoint`,
`--init-plan-feedback`, and `--packet`. `--python` is the training interpreter;
`--sim-python` is the DOMINO/WAM deployment interpreter. Keep the venv
executable path intact rather than resolving its interpreter symlink. Using
the benchmark Python 3.10 for both avoids a cross-version runtime difference.

Use `--prepare-only` to create an immutable snapshot and inspect registration.
Then use the frozen source entry with `--run-root RUN --launch-prepared
--detach` exactly once. Alternatively omit `--prepare-only` and include
`--detach` in the initial registration command. No model phase starts during
registration. A launch marker and process lock reject duplicate launches.

The server process owns `status.json`, `commands.jsonl`, `driver.log`, per-phase
logs/completions, frozen source/input/external DOMINO identities, policy JSONs,
accepted scene records, checkpoints, and final `analysis.json`. It does not
depend on a chat heartbeat. Child processes receive explicit argv lists and
isolated CUDA UUID assignments. Timeouts stop only the driver's own process
group. There is no implicit retry, best-epoch selection, or automatic resume.
Temporary and compilation-cache directories are placed inside the SSD run
directory rather than the potentially full system `/tmp` filesystem. This
includes subprocess-local `XDG_CACHE_HOME` and NVIDIA shader cache paths;
the user's global environment is not changed.

Development, collection, training, or gate failure stops the pipeline with
`incomplete` or `pending_design_revision`, retaining all evidence. Formal
evaluation is different: every fixed requested scene is attempted once;
preparation failures and infrastructure-invalid episodes are recorded as
missing while the remaining paired scenes continue. Source/protocol/record
identity violations still stop immediately. After all formal attempts,
`complete` requires 700 valid pairs; otherwise the final state is
`evaluated_incomplete`, with valid-intersection comparisons and explicit
missing records. Missing episodes are never treated as policy failures.

Final analysis separates worker service time, request wall time, actual
observation-to-install age, wall/simulation duration, tick lag, controller
limiting, and holds. It also reports physical observation-sampling intervals,
wall-clock submission intervals, installation intervals, independent camera
age, and the actual duration of consumed nonzero correction rows. Completed
nonzero target consumption does not prove a changed physical actuator command:
velocity/acceleration limits may clip corrected and uncorrected targets to the
same output. Controller-limited tick fractions are reported separately. Completed
request statistics exclude terminal unfinished requests, whose omitted cost
is explicit. First online queries occur after explicit warmup and
are reported separately; they are not described as cold graph captures.
Success differences include paired discordance and task-cluster bootstrap
intervals, with actual task/episode counts shown.

Per-episode process isolation currently reloads the simulator and policy.
Its initialization cost is outside the timed physical episode but makes
the complete 700-scene evaluation expensive. Reusing a long-lived process
would require a separate reset-equivalence test and is not silently enabled.
