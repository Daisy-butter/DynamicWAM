# Continuous executor repair, 2026-10-28

User authorized stopping .92 formal705, repairing deployment, and testing the existing final checkpoint. Do not retrain, modify .50/.100, start formal700, or distribute tasks before this smoke is accepted.

## Cause and repair

formal705 paused scene stepping during fast inference/rendering and advanced the wall origin. Its reported 1x clocks therefore omitted real elapsed time. Preserve that run as diagnostic; do not merge its scores into a continuous evaluation.

The repaired executor keeps the original activation wall/physics mapping immutable. Physics continues while fast/slow jobs or camera work are pending. Only the fast action queue drives the robot. Queue coverage exhaustion, stale observations, and physical/wall deadline misses remain explicit failures. No old-action replay, clock credit, deadline clamp, or lease monkeypatch is permitted.

The owner snapshots scene/camera metadata and publishes four-camera work to one sequential background worker, using the exact .50 Full formal09 camera implementation. A pending capture prevents a second render update. Exposure is the real snapshot time; publication waits for actual readback wall release. Close joins the camera worker. Native bridge, simulator dynamics, solver, collisions, seeds, task budgets, and rendering quality are unchanged.

Ready-barrier rewarming occurs with the scene inactive, followed by an armed barrier. It does not replace bootstrap observation/actions or re-label their exposure. Bootstrap remains excluded from active timing, as in Full formal09.

## Training and benchmark contract

- Final cross_time_joint/last.pt SHA256: 8be3d67d3962d264e1883debec536888d71b7cb13e031d5463d17ec543522894; stages 1k/10k/20k, global step 31000.
- All protected model/data/runtime sources and runtime settings match the trained contract. Inference FP32 image normalization and numeric camera tolerance remain the separately documented corrections.
- 45 actual physics ticks/action; action_dt=0.18000000854954123; fast cadence 2 rows, slow cadence 4 rows; protection 2/3 rows; minimum query budget 0.30s; K10 sealed Graph; reference TTL 2s.
- Fixed 1x wall/physics mapping, 0.24s physical and wall observation age, 0.25s backlog plus the same 0.005s numerical scheduling allowance as Full formal09. This allowance never extends an action deadline.
- Two environments, renderer/fast/slow on three distinct GPUs per environment. Last two GPUs idle, all eight held by the lifetime guard. Resource cost differs from Full; common external benchmark remains aligned.

## Acceptance

CPU regression precedes actual two-task x two-full-episode smoke. No shortened budgets. Require strict final-checkpoint load, sealed Graph, real fast/slow overlap, fresh slow references consumed by fast queries, immutable prefix/time/queue checks, physics progress during inference and rendering, raw activation-to-terminal wall/physics consistency, complete native MP4 decode, clean owned exit/all guard closure, and unchanged kernel evidence. Raw event timestamps are checked independently; report booleans alone are insufficient.

The guard is finite and has no automatic retry or formal-evaluation transition. Freeze source/config while running. A failed attempt is preserved and must receive a diagnosed repair before a new attempt. A smoke pass proves deployment correctness for those episodes, not SR >=25.43%; that remains a full paired 700-episode comparison target.

## Measured admission reserve, smoke02

smoke01 failed safely at hanging_mug query5: query physical1.80, strict deadline2.15, actual fast readiness366.4ms exceeded its two-row window. Physical lag was2.68ms; camera finished around110ms. The fixed-clock/asynchronous repair operated correctly but exposed insufficient pre-query reserve. All owned processes and guard descriptors closed, kernel unchanged.

Use an explicit deployment_min_query_budget_seconds=0.40 instead of the training scheduler's0.30. This only chooses a prefix BEFORE acquiring the lease, within the already trained2/3-row support. It never extends an acquired deadline, pauses physics, changes cadence/horizon, or alters model weights/inputs. Actual prefix, physical query phase, timestamps, and reference ages remain supplied to the same trained model. The unchanged training runtime and effective deployment runtime are recorded separately; do not claim every scheduler scalar is identical to training. Query latency affects response even when deadlines pass, so report both query-to-execution and exposure-to-execution.

## Concurrent Python scheduling, smoke03

smoke02 preserved continuous low-lag physics for about50s but failed a strict action deadline. More seriously,117 of132 fast starts had no valid M; slow model latency grew to2.3–3.6s and expired its references. Therefore smoke02 is rejected even independently of its final deadline. All owned processes/guard closed and kernel unchanged.

The next deployment candidate sets this worker process's Python switch interval to0.5ms (default5ms), reducing GIL scheduling delay among the250Hz owner and model-dispatch threads. It also caches byte-identical language tensors separately on fast/slow GPUs before activation, removing repeated cross-device language copies. Both are execution-only changes; no clocks, prefixes, deadlines, cache TTL, model arithmetic, or checkpoint change. The0.40s pre-query admission reserve remains bounded to trained2/3-row prefixes. Actual GPU smoke must establish effectiveness; this diagnosis is not itself a pass.
