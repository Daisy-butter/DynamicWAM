# Joint MotionKV training candidate

This candidate lives in the isolated SSD worktree. The ongoing scaled-time evaluation and its loaded source are unchanged. GPU training has not been launched.

## Stages and gradient path

1. **Interface warmup**: explicitly initialize from `motionkv_rtc_adapt_8gpu_20261014/exports/multirate_step_12000.pt`. Copy every existing backbone/conditioner tensor exactly; initialize only the new visual-, motion-, and plan-innovation parameters and their zero-initialized gate. At identical bridge/current inputs, the gated conditioner preserves the V1 output before warmup; switching the slow source to the student is still an explicit model change. Freeze Video DiT; update native Action DiT, native motion tokens and the whole V2 interface. The slow source uses the current student backbone, under no-grad, so it has the same source model as deployment. Candidate budget: 2,000 steps; action LR 1e-5, interface LR 1e-4.
2. **Joint adaptation**: strictly load the warmup export; create a new optimizer rather than claiming resume. Train Video DiT, native Action DiT, native motion tokens and interface together. The fast suffix FM loss backpropagates through V2 updates, cached K/V and both original slow joint forwards. Add aligned native video FM and action FM (weight 0.1 each) to retain world dynamics and the native action path. Candidate budget: 18,000 steps; video/action/interface LR 2e-6/5e-6/3e-5. These are proposed starting hyperparameters, not validated optima.
3. **Optional sampler distillation** remains separate. Both supplied stages retain K10; no untrained K4 inference shortcut or unsupported prefix endpoint distillation is enabled.

The original pretrained Video DiT provides initialization; its dynamics are now adapted jointly under the native video objective. This is not a rerun of generic large-scale Stage 1, and no claim is made that the original recipe remains sufficient.

## Data, RTC and causality

The episode split and normalization remain fixed. Four cache-age strata per source retain bounded index size; deterministic epoch jitter samples causal integer ages 0..15. Source and current images use the same three-view preprocessing. Source state, native optical-flow history, source RGB, and the action estimate from the actual cache-producing forward are the only bridge inputs. Expert future video/actions remain loss targets and never enter the source bridge.

H4/R4, horizon 16, refresh 16, K10 and clean prefix lengths 0..4 are preserved. The expert FM branch conditions on expert prefix; the frozen-teacher FM anchor conditions on that teacher's prefix and rebuilds V2 plan context accordingly. Neither is an on-policy rollout; the remaining deployment-prefix distribution gap is explicit.

The new timing features expose every one of the 16 action target offsets, including d0 queries, and the **reserved committed-prefix duration**, not measured camera latency. Training uses exact simulator endpoint timestamps minus the current timestamp; prefix duration reads the d-th offset (d0=0). Deployment supplies its known command offsets (i+1)*0.16 and prefix duration d*0.16. The V2 router therefore observes variable demonstration cadence rather than silently treating it as fixed cadence. Demonstration durations vary while the present controller uses nominal 0.16-second commands, so timestamp distributions are not literally identical. Continuous simulation at 1x is assumed for this candidate; real deadline feasibility still requires hardware smoke.

## Memory, initialization and evidence

Non-reentrant activation checkpointing covers the attached slow bridge and aligned retention forward. Video VAE/text encoders remain frozen. The candidate preserves existing DDP/guard launch behavior; actual full-model GPU memory fit is unverified.

V1 -> V2 migration is an explicit training-only operation with a fixed missing-key allowlist and strict tensor shapes. A changed cache-age schedule is permitted only after recomputing the previous H/R pairing hash on identical data/episode assignments; new exports additionally carry an age-independent partition identity. Normal full/fast checkpoint loading remains strict.

`check_joint_motionkv_training_cpu.py` passes eight integration checks using real native ActionExpert, motion tokens and V2 interface with a tiny differentiable Video DiT substitute and CPU SDPA: warmup freezing, suffix-loss gradient through both slow calls, all three innovation gradients, Video optimizer update, native retention losses, checkpoint gradient equality, full/fast strict reload, prefix timing and profile parsing. This validates graph wiring; it does not validate full Video DiT CUDA memory, CUDA Graph performance or closed-loop quality. The actual packed batch also passes trainer preparation and exact prefix timestamp checks; receipt is on SSD under `outputs/data/joint_motionkv_async_20261015/actual_cpu_smoke/trainer_prepare_receipt.json`.

Profiles: `configs/multirate/joint_motionkv_warmup.yaml`, `joint_motionkv_joint.yaml`, and corresponding `_smoke.yaml` profiles (four optimizer steps each). Joint smoke consumes warmup smoke's full export. Run only after fresh guarded preflight with available GPUs; the current eight-GPU evaluation is protected.
