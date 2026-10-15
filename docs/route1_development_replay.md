# Development-only same-trajectory replay

The optional recorder is `runtime/domino_continuous/development_replay.py`.
Default evaluation has no recorder, additional camera requests, trajectory copies,
or replay writes. Both task and suite entrypoints accept three optional flags:

- `--development-replay-dir`: a preflight-verified SSD output directory.
- `--development-scene-allowlist`: explicit, frozen development scenes (at most 16).
- `--formal-scene-manifest`: the complete 700-scene frozen test manifest or
  `formal700_scene_exclusion` inventory.

All three flags require `--joint-v3` and a separate development `--replay-from`
scene preparation. These flags do not grant GPU launch permission or replace the
existing guard/smoke lifecycle. No collection was launched as part of CPU checks.
The allowlist schema is `schema_version: 1`, `purpose: development_replay`, with
`scenes` containing `task`, `seed`, `development_scene_id`, `manifest_sha256`
(or `scene_sha256`), and `motion_sha256`. Task/seed and either hash must be absent
from every formal scene. The requested frozen files are rehashed before device
inspection; a formal replay directory cannot be repurposed for training.

Each trajectory contains:

| File | Contents |
|---|---|
| `manifest.json` | Schema `dynamicwam_development_replay_v1`; trajectory/episode/dev-scene identities, checkpoint/source hashes, scene and exclusion receipts, artifact hashes, success/complete flags |
| `queries.json` | Actual capture/query/result times, completed action counts, prefix/previous-plan absolute endpoints, real queue publication time, async source identity and original policy diagnostics |
| `queries/NNNNNN.npz` | Actual raw composite/fast RGB, state14 drive targets, native flow RGB, native motion4x12 and masks, committed prefix, previous proposal/mask, exact target offsets and returned policy proposal |
| `frames.npz` | Every captured real composite RGB and its physical timestamp; inference/history mask |
| `executed.npz` | Each actual physics step start/end, executed drive command, velocity, measured state and hold flag |
| `targets.npz` | Only completed action indices, absolute endpoint times and actually commanded endpoint targets |

Each query NPZ also contains `motion_history_start_times` and
`motion_history_end_times` (float64). They come directly from the flow pipeline's
selected real endpoints, with NaN for absent endpoints and the original validity
masks. `queries.json` retains the full `motion_history_diagnostics`, including
missing endpoints. No evenly spaced timestamps are fabricated.

The consumed slow source is resolved by its exact capture time and completed
control count to an earlier query file. A missing source is an error; the current
observation cannot substitute for it. This includes the bootstrap hold case:
source and current both have action count zero while physical elapsed time is
positive, and the next H4 prefix is the actual hold command.

The slow producer's exact monotonic publication/consumption timestamps are retained
in policy diagnostics. `source_publication_physics_upper_bound` is explicitly the
result-receipt physical time, **not** an exact physical publication timestamp.
Converters must preserve this distinction or require additional clock mapping.
The previous action queue's publication timestamp is the controller's actual
physical installation time.

Only development collection captures full cameras for history, including during
bootstrap hold, and requests enough block-internal history for the 0.32-second
future-video grid. Its extra work remains charged to actual wall time and can fail
the unchanged RTC deadlines. Captures are never repeated to cover a missing time.
The recorder buffers bounded data in memory and writes after episode completion;
its memory cap fails explicitly instead of silently dropping samples.

Raw traces are never trainer-ready. Successful, fully validated episodes may be
marked `successful_self_imitation`; unsuccessful or incomplete trajectories are
preserved as `unavailable_failed_trajectory` and cannot provide BC targets.
The offline converter must produce same-trajectory source/current condition
latents and valid future RGB/action targets, with a verified encoder receipt.
Missing future coverage is rejected, not padded or borrowed from old expert
observations. Source bootstrap labels come from actually executed hold/control
commands, not the hypothetical priming proposal.

CPU coverage: actual controller -> fake inference policy -> record/finalize ->
strict reload, bootstrap positive elapsed time at zero action age, actual hold
commands, source lookup, formal identity/hash exclusion, failed-trajectory label
rejection, and file tampering. This is not a GPU performance or closed-loop score.
