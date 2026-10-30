# Native DualRate V5 — 2026-10-30

Scope: independent .92 worktree. Archive V4 first; .50 formal evaluation and .100 remain untouched.
This revision implements native synchronous DOMINO, not continuous real-time evaluation.
Reference design: DualRate_DynamicWAM_Design.md plus explicit user revisions on 2026-10-30.

## Architecture
- Slow full DynamicWAM remains source-only predictive joint 2-video/10-action denoising.
- Fast motor copies the complete native Action Expert and absolute-motion encoder, with independent parameters.
- Native video K/V retain exact layer/head layout for the pretrained motor; this is a compatibility branch, not a claim of action-consistent updated world truth.
- Structured spatial observed/predicted world features still feed the reactive 384/6 joint A+Z branch.
- Velocity = native motor velocity + tanh(zero-initialized gate) * reactive A/Z velocity. A and Z interact at every denoising step.
- Slow reference actions are excluded from generic attention tokens. Ordinal-aligned normalized arm references shift the Gaussian prior by at most coefficient .15, exponentially decayed by actual reference age and committed deviation. They never clamp the generated suffix or average final actions. Grippers receive no such shift.
- Hard committed prefixes remain supported, but native synchronous execution has zero inference prefix.
- Full Action capacity is retained; first version uses ten steps with the native shifted FlowMatchScheduler grid. Few-step distillation is not claimed implemented.

## Time and deployment
- Native env.take_action(qpos), original TOPP, dynamic per-action movement, contact/lift/out-of-bounds and native action-count limits.
- No realtime velocity conversion, wall catch-up, extra hold actions or lease deadlines in native mode.
- Native flow pairs span four actual command observations, using their actual physical time differences (same stride semantics as the original native policy); fixed .72s nearest-frame matching is reserved for continuous mode.
- Real scene step clock is authoritative for camera/history and Z. Wall time measures compute cost only.
- Waypoint indices align A references. Unknown TOPP completion times are not fabricated as fixed .18s endpoints.
- Fast replans every two executed commands; slow refreshes every four or reference age >1.44s.
- Slow/fast inference is synchronous with the native simulator paused. This is multirate cache reuse, not evidence of continuous asynchronous performance.
- Overlapping action indices reuse Gaussian base noise to reduce sampling-induced switches, preserving each training noise marginal.
- Native joint-limit/gripper projection is recorded and invalidates Z when changed.

## Data and training
- Existing 10,500 episodes and frozen split/text/normalization identities retained.
- Ages 0/.18/.36/.54/.72/1.08/1.44/2.16 cover bootstrap and stale references.
- 545,754 train / 57,510 validation windows; 219,249 partial terminal windows retained.
- A suffix outside real data is loss-masked and receives pure noise, never fabricated targets.
- Incomplete causal-video targets disable that sample's whole video supervision; Z is masked fieldwise.
- A first-difference loss matches expert motion, rather than penalizing all velocity. It does not guarantee smooth closed-loop boundaries.
- Source-only reference pass remains differentiable; noisy GT supervised pass is never exported as M.
- Warmup freezes slow and copied motor; then joint stages train all. Motor LR 1e-5, slow 1e-6, reactive/readout 1e-4.
- BF16 native video plus FP32 time/action/new modules, FP32 optimizer masters/accumulation.
- Formal initialization is official DynamicWAM_full.pt only; diagnostic smoke weights never become formal initial weights.

## Acceptance
Require actual model copy parity, finite complete/terminal losses and module gradients, checkpoint reload,
three-stage eight-rank smoke, deterministic native clock/queue checks, real native DOMINO episode/video,
clean owned process/guard exit and unchanged kernel evidence before formal training.

## Limits
Offline expert trajectories are not on-policy TOPP rollouts; waypoint order has no guaranteed physical-duration match.
Stale joint K/V is a scenario prior. Zero native-copy error establishes identical motor computation under identical conditions, not preserved closed-loop SR.
Native scores and prior continuous 25.43% scores require separate protocol labels; no improvement or SOTA claim before evaluation.

