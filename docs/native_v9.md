# Native V9: Full action baseline with compact bounded corrections

Scope: .92 only, native DOMINO (noncontinuous physics, original TOPP, original
dynamic-object motion/action budget/contact/boundary checks). Archived V7 remains
unchanged. This is a new training run from official DynamicWAM_full.pt.

## Model

Full slow/teacher stays frozen. One 448-wide, 6-layer, 7-head A/Z fast network.
No second fast action expert. A10 is joined by integer command identity and
passed numerically into the fast network. The action clean endpoint is
`aligned_A10 + radius * tanh(delta)`, inside every denoising step. Arm radius is
0.06 radians, gripper radius 0.05 in native units; conversion uses original Full
normalization. Zero-initialized delta decoder preserves aligned A10 (GPU numerical
parity required). Missing reference rows cannot be executed. This bounds change;
it does not prove a task-success lower bound, especially with stale A10.

History and predicted-future M use separately normalized cross-attention and
separate gates. The future gate sees measured executed-command deviation from
the source plan, joined by command identity (not future ground truth). A10 is not a generic M attention token. A/Z still interact within
the same transformer. Future Z is the existing three supervised roles x four
command events, with no invented support/occlusion labels.

## Native asynchronous scheduling and clocks

One simulator owner steps TOPP. Dedicated slow process on another reserved GPU
only consumes immutable observations. Fast sampling pauses physics, as native
DOMINO does; slow computation overlaps TOPP and fast computation. No wall-clock
catch-up, clock scaling, fixed-duration TOPP fiction, or deadline-based row skip.

During TOPP, existing real 60ms physical captures can request slow refresh once per four completed
command events. At most one request runs; no FIFO of old requests.
If an action is in progress, its final waypoint is the one-row slow hard prefix.
Fast queries at completed-command boundaries have zero hard prefix. Both use the
same completed-command counter. Completed M is atomically published with original
source exposure and observed completion physical time. Source identities cannot
go backwards. Fast normally consumes the latest complete valid reference and
executes one original TOPP command. Bootstrap/uncovered A10 explicitly
pauses native physics for a fresh reference; it never uses an invalid suffix.
Exceptional waits and reference reuse/overlap are measured separately.

Wall time measures compute, transport and publication delay only. Physical time
selects observations, motion endpoints and predicted-world TTL (4.0s in this newly trained contract). Integer command identity
selects action rows and Z events. Unknown future TOPP end times are not timestamps.
A10 validity is remaining integer command coverage, as in Full chunk execution;
world-feature expiry does not silently delete still-covered A10 rows. Both ages
and their separate masks are exposed in training and deployment.
The metadata .18s sampling interval remains an offline expert convention, not a
claim about native execution duration.

## Data and training

All new expert motion values and flow images are recomputed from raw frames with
the same functions/contracts as online; legacy cache timestamp matches alone are
not trusted. Original Full normalization is retained. Raw cache parity must be
tested on identical frames and real endpoints.

Stage 1: fresh-source expert bootstrap, fitting bounded corrections and Z.
If bounding a target changes the expert action prefix, the corresponding expert
Z/contact labels are masked; incompatible futures are never presented as labels. Stage
2/3: actual native rollout snapshots with their jointly recorded source/query
times, source prefix, publication time and command offset. Fresh Full endpoints at
the recorded current state supervise corrections. Replay Z/contact labels are
masked unless labels were actually collected on that trajectory; no expert suffix
is attached to an intervened trajectory. Actual replay is mandatory, and stage
selection is by explicit data domain, not synthetic elapsed-time relabelling.

Proposed quick-iteration budget: 1000/3000/5000 updates, global batch16 (8 ranks,
micro1, accumulation2), fast LR1e-4, Full frozen, native BF16 video + FP32 action,
FP32 optimization/accumulation. Use all expert tasks; actual replay starts with
35 distinct training scenes, separate from evaluation seeds. Retain strict full
state stage transfer/reload. A recorded distribution is not a promise of coverage
for all future on-policy states; report support and residual saturation.

## Acceptance (no automatic retries)

- CPU: exact command joins, real-clock translations, expired/uncovered rejection,
  independent attention gates, bounded corrections, zero-head A10 parity,
  immutable/cache/Graph buffer identity, one running slow request and owned exit.
- Actual GPU: eager/Graph parity including changed A10; zero correction parity;
  current observation/M/Z sensitivity after nonzero diagnostic head.
- Native full episodes: source partial-prefix checks, actual command/clock ledger,
  nonblocking refresh, genuine background completion, consumed new M, native MP4
  full decode, no scene access from child, original task budgets/checks.
- Real dataset: raw motion parity; replay fields/causality/hash/seed separation;
  no borrowed future labels; multi-task coverage.
- Eight-rank all-stage updates, finite losses/gradients, strict full-state reload,
  clean owned worker/guard exit, unchanged kernel health, frozen Git/config.
- Formal training/evaluation only after the corresponding gates pass. Smoke is
  engineering evidence, not performance evidence. Final native35x20 stays separate
  from the older continuous25.43% result.
