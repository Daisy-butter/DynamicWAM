# Predictive feedback: autonomous experiment protocol

This experiment tests a flat two-horizon, action-conditioned latent predictor.
It does not assume that a hierarchical JEPA model is the better architecture.
The official full WAM still generates the immutable 16-row slow plan; the
fast head uses current/history observations and a causal controller proposal.
Actual future commands and images appear only in world-model supervision.

## Data and timing contract

The physical protocol remains 4 ms physics, 100 ms command intervals, H2
queries, 16-row plans, age-12 slow refresh, independent snapshot camera,
30-second maximum episode, and 100 ms maximum wall backlog/tick gap. One
simulation/render/policy group runs at a time on the qualified GPU0. No new
rendering or teacher inference is performed to create training targets during
the rollout. The dense recorder and buffered query recorder each have an
explicit 512 MiB cap and fail on overflow; they never silently downsample.
Persistence is deferred until physical execution ends.

Each camera-ready frame retains the source physical timestamp, three RGB views
and masks, causal composite/flow/history inputs, drive targets, optional measured
articulation state with explicit validity, and the full LimitedHermite source
controller snapshot. Every actual `step_command(q, velocity)` is saved over its
4 ms interval, including holds and stale-plan execution. Commanded targets,
controller velocities, and measured articulation states remain distinct.

The two horizons follow actual query dispatch, not a hypothetical immediate
query at image capture. For source time `s` and dispatch time `d`, the shared
helper sets `install = ceil((d + 0.020) / 0.1) * 0.1` and desired future offsets
`(install - s + 0.1, install - s + 0.3)`. These are at least nominally 0.2 and
0.4 seconds, and grow with camera/query age. Real future frames are selected
in `(h0-0.1,h0+0.1]` and `(h1-0.1,h1+0.2]`; the target time remains the actual
selected frame time. Missing terminal future frames are censored.

Each sent query retains the actual dispatch controller, including its elapsed
active segment, and the complete already executed 4 ms command prefix between
source and dispatch. A two-second ring has a fixed memory bound and fails if
the prefix is no longer available. No future executed command enters this
packet. Unqueried camera frames retain `dispatch=source` with
`policy_submitted=False`, explicitly marking a hypothetical dispatch for world
supervision rather than fabricating a real query.

Four command bins contain the last applied q and duration-weighted mean
reference velocity; a gap invalidates the path. The policy receives the known
past prefix followed by a candidate rollout under the dispatch controller.
The 20 ms compute reserve is a declared conditional assumption; actual query
arrival and installation remain measured separately. Policy supervision masks
rows that will already be expired at the assumed installation, as well as
gripper coordinates and the invalid plan tail. Records with no remaining live
correction row are excluded from policy training while usable world targets
are retained and their coverage counts reported.

Offline full-WAM teacher labels join to dense data by exact capture ID, task,
seed and source time. The builder verifies raw, bundle, label, accepted-scene,
source, normalization and protocol identities. The actual future action path
is never copied into policy inputs. Previous sparse on-policy records cannot
be upgraded into executed-command trajectories by reconstruction.

## Fixed sequence and failure handling

`run_predictive_feedback_pipeline.py` reuses infrastructure helpers from the
same immutable source snapshot. It never imports executable code from an older
run or a mutable project checkout. The source/config/protocol/base/data/seed
identities, runtime executables and GPU UUIDs are registered before launch.
Completed phases require exit status plus schema, artifact SHA and evidence.
No failed phase is silently retried; no seed, final optimizer step or epoch is
selected after inspecting quality. A server process owns the sequence without
chat polling or heartbeat scheduling.

1. CPU contracts, including the delayed-dispatch profile fixture; then one
   accepted scene's RGB replica oracle and real dense collection with unchanged
   strict physical clock gates. Re-read saved numerical arrays, validate the
   entire applied 4 ms command grid and dispatch identity, and count future
   support with the same dynamic horizons used by training/deployment.
2. The fixed development 3 tasks × (2 training + 1 heldout) episodes. Reuse the
   accepted motion bytes if their external source identities match; replay the
   policies afresh. Labels are generated after each rollout.
3. Build and fully verify world pairs and teacher-policy records. Run four DDP
   optimizer steps as an execution-only smoke; this is not a scientific gate.
4. Fixed world feasibility probe, then joint policy probe. World evidence must
   beat persistence/constant velocity and pass the registered history/action
   ablations on heldout episodes. Report dynamic-region/per-horizon evidence,
   target variance and episode-cluster uncertainty. A failed gate stops the run
   with `pending_design_revision`; there is no automatic search for a pass.
5. Load the exact resulting checkpoint and profile the full fast path, including
   CPU proposals, image preparation, GPU execution and result transfers. Require
   fast p95 <100 ms, native graph action parity and the registered 0.85 service
   budget ratio. The fixture is synthetic compute evidence; the real camera
   clock gate remains separate.
6. Fixed six-episode matched closed-loop development tripwire and nonzero
   feedback-target consumption before expiry. Both arms 0/6 is uninformative
   and stops. At least one success plus at most one extra failure is a
   catastrophic-regression tripwire, not a noninferiority result. Consumption
   does not prove a counterfactual actuator change after clipping.
7. Collect the full registered 35-task training set, formal world pretraining,
   stage2 joint learning, then additional learned-policy collection and stage3.
   Re-run exact-checkpoint latency and matched small closed-loop gates.
8. Attempt all official 35×20 fixed scenes under `official_eager`, `graph_zero`
   and `predictive` using identical accepted motion/instruction bytes. Missing
   preparation or invalid clocks remain missing, never policy failures. An
   identity mismatch hard-stops. Final analysis uses the common valid paired
   intersection for success and timing; incomplete coverage produces
   `evaluated_incomplete` with explicit gaps.
9. Run `predictive_current_only` on the first five numeric manifest seeds per
   task (35×5=175), using the identical final stage3 checkpoint and runtime
   `mode=predictive, ablation=current_only`. The subset is registered at freeze,
   before outcomes. Its valid intersection with full prediction has separate
   quality/timing/paired intervals and never shrinks the three-arm 700-scene
   main analysis. Missing mechanism results remain explicit; no replacement
   seeds are selected. This tests use of prediction features, not action
   causality by itself.

## Registered budgets

The training global batch is 32: 8 cards × per-rank 2 × accumulation 2. For
4/2/1 available cards accumulation is 4/8/16. A 5/6/7-card inventory selects four
registered cards rather than changing the sample budget. GPU0 is required for
the separately qualified simulation path. Busy cards are waited for, never
terminated; no simulation process runs during DDP training.

| Phase | Optimizer steps | Global batch | Samples seen | Learning rate |
|---|---:|---:|---:|---:|
| DDP smoke | 4 | 32 | 128 | 2e-4 |
| Development world | 200 | 8 | 1,600 | 2e-4 |
| Development joint | 200 | 8 | 1,600 | 2e-5 |
| Formal world pretraining | 1,000 | 32 | 32,000 | 2e-4 |
| Stage2 joint | 3,150 | 32 | 100,800 | 2e-5 |
| Stage3 joint | 900 | 32 | 28,800 | 5e-6 |

Formal world pretraining starts from the fixed v1 initialization, not a selected
development checkpoint. Stage2 starts from that warmup; stage3 starts from the
exact stage2 export. Video/full-WAM weights and target visual encoder remain
frozen. Original expert trajectories can support separate representation
analysis, but are not substituted for continuous on-policy dynamics labels.

## Launch and inspect

After source review and unified synchronization, launch exactly once:

```bash
bash scripts/launch_predictive_feedback_20261009.sh
```

For a separate, terminal one-scene collector preflight, choose a new directory:

```bash
bash scripts/launch_predictive_feedback_20261009.sh \
  /SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/predictive_dense_preflight_20261009 dense_clock
```

`dense_clock` ends at `preflight_complete`; it cannot resume into formal
training. `development_data` similarly commits only the fixed nine-episode
dataset. A later formal run needs its own immutable source/run directory.
`status.json`, `commands.jsonl`, phase logs/completion receipts and final
`analysis.json`/`analysis.md` are the authoritative progress and outcome. TMP,
compiler and shader caches are local to SSD paths. Existing runs are never
overwritten or relaunched.
