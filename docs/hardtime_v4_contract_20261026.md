# DualRate V4: deterministic exposure, execution and availability clocks

User-authorized 2026-10-26 correction of the V3 serial deployment and synthetic timing recipe. Only .92 is writable. Base 3a30a4a; old formal sequence stopped through its controller, clean owned process/guard closure and unchanged kernel evidence. The new formal sequence must start from official DynamicWAM_full.pt, never old or diagnostic training weights.

## Model and time origin

Keep full native DynamicWAM slow Video/Action/motion, source-only differentiable 2+8 prediction, supervised forward separation, structured readout, fast384/6/6 joint A16+Z12, original normalization and BF16/FP32-master optimization. No learned clock alignment or new time adapter.

Slow prediction is anchored to the actual source image exposure e. Its state and history describe e or earlier; its fixed video targets are e+2k*action_dt and its action intervals start at e. The actual request snapshot is r>=image availability; completed/committed control history between e and r is already known and included as an immutable prefix, followed by any then-protected future rows. This is delayed-observation prediction with known commands, not a claim that those commands were known at exposure. Record e and r separately. Never retimestamp old images or predictions to request/completion/query time.

Deployment source exposures must actually land on the common action grid. Use integer physical steps; action_dt is an integer number of measured simulator steps (nominal .18s is45steps of the actual float32 .004s step). Preserve raw times and only allow numerical representation error. Source data use real frame timestamps as each sampled window's common origin; all source/current/actions/targets in that window share the same grid. Raw expert trajectories are resampled estimates, not newly collected on-policy execution.

Native slow motion image slots request nominal .72s intervals relative to e; image and numeric motion must use the very same real frame pairs. The four pairs share endpoints, end at e, and their actual durations are present in the native motion features, so cumulative durations specify their real historical positions. Preserve bounded endpoint matching and mask genuinely missing/invalid pairs; do not call these intervals exactly .72s. Fast retains explicit observation/target time encoding. Nearest-frame future-label error and interpolation error remain distinct from deterministic execution-clock alignment and must be audited/disclosed. A35-task audit found only219/4068 full video windows and65/3635 full motion histories supported zero-error nominal slots, whereas all were within30ms. Do not discard almost all dynamics data merely to hide the documented label sampling approximation.

## One scene clock, one monotonic wall clock

Physical time changes only after an actual successful scene.step. At activation bind (wall0,physical0). Monotonic wall time defines the expected1x position physical0+(wall-wall0), not permission to advance the scene clock without integrating physics.

Every asynchronous camera/model/transfer completion records monotonic wall readiness. Its first usable physical tick is the ceiling of that wall event on the activated physical grid. Until actual integration reaches that tick the result remains pending. At publication also record actual physical time, separating compute completion, transfer completion and publication. This prevents render blocking and catchup from making a result available in the simulated past.

Fast submission must meet both its physical queue lease and corresponding monotonic wall deadline. Recheck backlog after blocking owner operations before dispatch or commit. New queries/captures are dispatched on actual eligible ticks after catchup; missed captures are skipped and logged, never backdated. Catchup integrates every physical step using the correct action/target motion. Report bounded wall lag and maximum step gaps; do not claim that rendering never pauses wall-time simulation progress.

Bootstrap is an explicit inactive interval for first reference/fast plan setup. Bind wall0 once after activation, never reset it to hide backlog. Wall/GPU duration is diagnostic; CUDA event clocks are not world timestamps.

## Queue and asynchronous inference

Only the execution queue owns action commitments. Source prefix extraction atomically merges issued history from e to r with the commitment snapshot at r, preserving queue version/hash, and rejects gaps/expired or entirely consumed forecast horizons. Fast queries take their own current leases. Overlapping immutable absolute intervals must agree. Soft slow reference suffixes may differ from later fast plans; in-flight slow snapshots are not retroactively changed.

Renderer, fast policy and slow policy use separate GPU assignments. Slow work cannot prevent fresh fast queries. Each query snapshots the latest completely published valid M; in-flight new M does not replace it halfway through sampling. Source/publish time, prediction intervals, generation stage and plan provenance stay with the whole immutable payload. Cross-device copies must finish before publication. Only fast actions control the robot.

## Data and formal gates

Replace independent fixed latency .18 and ages .36/.72 with jointly replayed events: actual source exposure/request/publication, fast query, current observation exposure/availability, source committed endpoint and fast lease endpoint. The template must represent continued fast queries while a newer slow job is pending. Shift a complete event template to one real episode source exposure; do not independently shift its action/observation grids. Initial synthetic fixtures are engineering-only and explicitly marked; formal manifests require measured parallel deployment traces and immutable provenance.

Use the same pure temporal/queue rules in training and deployment. Keep future labels separate from conditions, task/episode identity and train-only statistics, actual missing-field masks and 12 supervised entity tokens. Do not substitute policy actions while retaining unrelated expert futures.

Required gates before official restart: deterministic clock/queue and offline/online agreement tests, complete native CPU contract and real data checks, guarded actual parallel GPU episodes with consumed new references and native MP4 decode, fresh measured timing data, all three current-source actual8rank smoke phases with strict full optimizer/master/RNG reload and clean worker/guard/kernel exit. Formal source/config stay frozen; no automatic retries or GPU recovery. Startup acceptance does not imply completed training or benchmark improvement.

## Actual initial deployment findings and acceleration

The first three-device run at 2b6925d failed the fast query at physical .72s: solo K10 query wall latency .261s fit, but a concurrent query was still running after its approximately .347s remaining wall budget. A separately configured K8 run consumed refreshed M2 but failed a later query at1.8s; its four accepted fast latencies ranged .182-.324s and one active slow latency was .700s. Both failed runs closed owned workers/guards with unchanged kernel evidence. Neither is deployment acceptance and neither permits formal training.

Keep K10 and add optional inference-only CUDA Graph for the prepared joint A/Z sampling core, including every prefix clamp and final contact forward. Query clocks, input/cache/parameter mutation checks and queue publication stay outside the graph. Noise is generated per query outside the graph and copied along with all current context/mask/condition buffers. Bootstrap prepares real-reference and empty-reference shapes, verifies actual eager/graph parity on changed prefixes/context/masks, restores query RNG and seals shapes before clock activation. Unknown active shapes fail; active recapture is forbidden. This adds no trained parameters or learned timing adapter. Real current-source parity and concurrent deployment gates remain mandatory.
