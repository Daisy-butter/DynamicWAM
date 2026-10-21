# Causal spatial candidate: three review rounds, 2026-10-21

This is a new isolated candidate. The running .100 evaluation, .92 baseline,
accepted tags, historical results and old DOMINO controller remain protected.
The code below establishes CPU ownership, causal model/data interfaces and an
RTC queue adapter and independently tested H2 continuous episode driver. It does not establish a complete new DOMINO policy server,
CUDA timing, successful training, or improved task performance.

## Round 1: causal ownership and the H4 migration trap

Findings:

* The existing native episode loop hardcodes H4/R4 in construction, reservation,
  runtime contract checks and report validation. Changing YAML alone leaves
  `reserve(..., length=4)` active.
* A slow result can finish after a newer observation, and must never replace the
  current belief with a state reconstructed at its old capture time.
* The model reviewer confirmed that M contains only observation/text tokens;
  each request creates fresh noisy future Z and Action and jointly integrates
  both for K10. The current belief and its projected K/V are computed once per
  request, not updated ten times by the sampler.

Changes:

* `runtime/causal_spatial_contract.py` reuses the actual native
  `CommittedActionQueue` storage/atomic installation, always passing explicit H.
  It does not call or edit the old H4 episode driver. Tests exercise H1..H4 and
  the proposed H2/R2 default.
* Source tokens bind episode, capture event/time and a complete contract digest.
  Completed M publication is atomic and monotonic in source time/event; late old
  results are discarded. Publication never assigns a belief state.
* Snapshots own immutable CPU arrays. Request identity binds episode, event,
  belief revision, M publication, queue generation and absolute action origin.
  A newer belief while inference is running is permitted; a result still must
  own the active request, exact target grid and unchanged committed prefix.
* Bootstrap overlap is absent. Later overlap comes only from the installed
  queue's explicitly covered short editable suffix. No old source-plan fallback
  or extrapolated expired rows are admitted.

## Round 2: training/deployment data semantics

Cross-review found two concrete problems and corrected the design:

1. The first runtime draft used the latest four camera events. The data reviewer
   and root required the same four-point physical lookback as training: .32 s
   spacing, last available capture at each target, ending at the current event.
   Runtime now calls the shared `causal_window_indices` helper and retains the
   predecessor needed by the earliest target. Early windows repeat their first
   real image; zero-time flow/command transitions are invalid.
2. The dataset already selects the command at the left capture of an interval,
   but the initial model also shifted commands internally. The runtime reviewer
   detected this double shift; the model reviewer removed its second shift.
   Command row i now directly accompanies dt[i], from the previous selected
   capture to the current selected capture.

RGB validity and flow validity are separate. The current real image remains
valid even if there is no previous flow interval. Both paths use the same
composite splitter and current-grid Farneback measurement/validity helper;
runtime recomputes flow for the selected pair, not an unrelated adjacent pair.
The shared `flow_pair_validity` checks previous padding at the flow-warped
bilinear footprint, rather than incorrectly ANDing the same pixel coordinates.

`recorded_capture_commands` means native drive targets sampled at exposure.
They are neither measured qpos nor the complete actuator trajectory. The runtime
also exposes actual queue execution acknowledgements and measured-qpos
diagnostics, but never substitutes them or a planned suffix for a missing
capture-time command. FP32 action normalization precedes model precision
conversion and remains the policy/trainer caller's explicit responsibility.

Offline prefixes are expert H2 prefixes. Where the trainer supplies a soft
proposal, it is an explicit teacher proposal generated at the previous fast
query's recorded observation (nominally current time minus .32 s), then aligned
by physical target time. Root/data cross-review caught that using the old slow-M
source could exhaust its 16-row horizon and silently invalidate all overlap;
separate `plan_*` fields carry the previous-fast-query source. Deployment's short overlap
comes from an installed policy queue. This distribution difference is documented,
not silently declared solved; offline training is allowed, while on-policy
robustness remains a later evaluation/replay question. Failed policy actions
are not successful behavior-cloning targets.

The data reviewer identified actual latent support ends at captured future
frames 4 and 8. Root's causality review then caught that those actual future
capture times are unavailable to a live request. Therefore the model and runtime
use requested nominal action-grid endpoints 8 and 16 (initially 1.28/2.56 s,
including actual observation phase); actual VAE support ends and capture lag are
loss/audit metadata only. Future GT goes to the noisy FM trajectory/loss and is
excluded from M, belief and deployment timing inputs.

Root's final model review found that the first soft-plan encoder modified b.
The model owner moved the two editable proposal rows 2:4 into A/Z residual
conditioning, prepared once per request. `prepare_belief` now reads observations,
recorded capture commands and elapsed time only. The GPU checker requires b to
remain invariant when the soft plan changes, and requires invalid/outside-overlap
proposal rows to be exact no-ops. These GPU checks are implemented, not yet run.

## Round 3: timing, actual smoke, and honest acceptance boundaries

Primary candidate values are action dt .16 s, H2, R2, chunk16, K10, fixed 1x.
Thus each query has a nominal .32 s commitment lead, queries repeat every .32 s,
and the prediction spans 2.56 s. These are design settings, not measured speed.
The runtime age default 4.96 s and short overlap2 are configurable candidate
limits and must be bound to the trained/exported contract before deployment.

The action grid never moves. Camera exposure/reservation is fixed at the query
boundary; readiness is `max(actual_physics, physics_epoch + actual_wall_elapsed)`.
Late results fail, and no new suffix executes without a valid handoff. Remaining
wall budget is smaller when backlog already exists. Record observation time,
absolute target time and actual availability separately.

Proposed profiling target: full camera/preprocess/belief/K10/IPC/install p99.9 at
most 260 ms, with 60 ms nominal cycle headroom, plus zero actual .32 s deadline
misses in the accepted smoke. Percentiles alone do not establish a worst-case
guarantee. GPU memory publication needs the actual producer completion event.
Shared rendering must be tested: CUDA stream priority is not a preemption
guarantee, and a background slow kernel can block a short foreground path.

Reusable launch structure:

* `scripts/run_joint_motionkv_smokes.py`: fresh source/config identity, actual
  eight-rank four-update diagnostics, strict reload, clean all-FD release.
  Its historical protected receipt and recipe paths must be replaced.
* The historical `run_stage3_memory_20261018.py` one-shot smoke/audit/formal
  structure: reuse per-rank measured peaks/zero-OOM and independent formal init,
  not the old Stage2 weights, budgets, dataset paths or diagnostic weights.
* Some old local supervisor copies probe before opening the temporary guard.
  Reuse only the accepted current `fds_before_telemetry_v1` implementation with
  fresh helper/source hashes. No retry or device recovery is authorized here.

Actual GPU acceptance must test the full new model and actual data, including
backward/optimizer state creation, strict full/fast reload, K10 eager/Graph
parity, changing every new input, source refresh, and real producer/consumer
overlap. A GPU sampler check must explicitly exclude DOMINO rendering,
camera/IPC and physical execution. Old 40k sampler timing or its small-sample
Graph parity is not evidence for the new M+b+Z pipeline.

## Verification and remaining integration

`tests/test_causal_spatial_contract.py` tests the actual native CPU queue,
snapshot immutability, source publication order/concurrency, wall-clock late
rejection, absolute target alignment, prefix integrity and explicit overlap.
Its injected deterministic flow fixture tests selected-pair geometry only;
it makes no optical-flow accuracy or GPU performance claim. The default runtime
uses the exact shared OpenCV helper, not that fixture.

Local CPU results: 16/16 contract tests and 12/12 new controller tests pass.
The separate `continuous_causal_spatial.py` driver enforces H2/R2/1x, captures
native drive targets at exposure, discards bootstrap action predictions, and
reuses the existing process packet ABI. Controller tests exercise slow bootstrap,
wall/compute/camera deadline failures, prefix/request identity, partial final
blocks, report validation, installed rows2:4 overlap only and separate exposure/
availability clocks. Synthetic durations establish scheduling behavior,
not actual rendering or model latency.

`scripts/check_causal_spatial_gpu.py` is staged with actual full-size audited
initialization or actual smoke full/fast exports, two real dataset observations,
joint K10 eager/Graph input mutation, completed-M concurrent publication,
full/fast GPU output parity and measured M/b/K10 costs. Local AST and CLI checks
pass, together with 9/9 CPU guard/schema helper tests. These verify the actual
eval-supervisor manifest shape, eight-adapter inventory coverage, one guarded
compute UUID, authorization/helper/config identity and all eight opened guard
receipts, without importing Torch or querying GPUs. It has not been run on a GPU.
The dataset's `executed_actions` compatibility
name is accepted only with its exact sparse capture-drive-target provenance;
the checker maps it explicitly to `recorded_capture_commands`. The script reports
`full_DOMINO=false`, excludes camera/render/IPC/physics from its measurements,
and requires the enclosing supervisor's clean all-adapter guard release.

Still required before a new full-DOMINO claim: camera/history recording at the
declared physical cadence, normalized model packet assembly, actual M producer
ownership/completion events, new model worker and result wiring, actual session
integration of the new continuous driver/report validator, and heavy shared-
rendering complete-episode acceptance. These remain separate from the preserved
old H4 controller. Source/init/config/data/GPU/run identities must be frozen for
each real smoke and formal stage.
