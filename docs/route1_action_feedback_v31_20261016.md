# Route1 v3.1: action-conditioned effective world context

The source slow snapshot is conditioned on actions committed **at that source query**, including its source RTC prefix. This snapshot is immutable after asynchronous publication. It must not be rewritten with information that did not exist at the source time.

The existing v3 code already reads evolving Action DiT tokens at every native layer and every denoising step through `MotionKVConditioner.adapt_video_kv`. It also reads the physically aligned source plan, current committed prefix, previous queue, current command state/state displacement and observed motion innovation. Therefore “the updater ignores new actions” was not an accurate description. The missing explicit connection was the discrepancy between the evolving endpoint proposal and the source/queue plans, expressed in the same physical target-time grid and with its denoising uncertainty.

## Added correction

`proposal_feedback: true` enables `sigma_conditioned_physical_plan_v1`. The query prepares source-plan alignment, valid intervals, previous queue, committed mask and exact target offsets once. Each of the K10 steps embeds 16 current normalized noisy action endpoints, differences from `(1-sigma)*source_plan` and `(1-sigma)*previous_plan`, clean committed endpoints, sigma, confidence, validity and physical-time features. Committed tokens use confidence one; the uncommitted suffix uses `1-sigma`. Pure-noise unknown suffix at sigma one contributes no proposal evidence. This is a confidence heuristic, not a calibrated posterior variance or a claim that a noisy sample is already a clean plan.

A shared 49,537-parameter MLP/gate maps these features to the existing low-rank context (192 channels). It adds no Video or Action backbone forward. The existing native layer-wise read still incorporates action/motion tokens, and the final update remains bounded by 25% of each reference K/V head RMS. The per-step context is an owned shallow derivation; neither the published world cache nor the query's base rank memory is mutated. Executed progress is represented by current command state/state delta and newly observed native motion; it is not mislabeled as measured joint feedback or a complete executed-action history.

Training, eager inference and the CUDA Graph body all call the same `predict_fast_velocity -> condition_on_proposal` path. Training supplies the same FM noisy sample and clean committed prefix; it never supplies the clean ground-truth suffix as a proposal. Source/queue actions and state share the recorded normalizer. Expired source target times are masked rather than extrapolated. This amendment corrects effective world context; it does not rerun or physically rewrite Video DiT on every fast query.

## Initialization and recipe

1. Warmup: initialize `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/motionkv_rtc_adapt_8gpu_20261014/exports/multirate_step_12000.pt`, preserving all existing backbone/interface weights and normalization. Initialize only new v3 innovation/forecast/proposal tensors, with innovation/proposal gates zero. Freeze Video, update Action/native motion/interface for 2,000 steps; LR Action 1e-5, interface 1e-4. The stored Video LR is inactive while frozen.
2. Joint: initialize the warmup step-2,000 full export. Update Video/Action/native motion/interface for 18,000 steps, LR 2e-6/5e-6/3e-5. Loss: fast suffix FM 1.0, source Video FM 0.1, source Action suffix FM 0.1, native motion forecast 0.1.
3. Real replay refinement: initialize joint step-18,000. Same-trajectory development replay, 2,000 steps, LR Video/Action/interface 1e-6/2e-6/1e-5, retaining source and forecast losses. Replay collection is still required; synthetic expert prefixes do not substitute for it.

All three use batch1/GPU x8 GPUs xaccumulation2 = effective batch16, K10/H4/R4, 0.16-second action grid, age0..31, frozen VAE/T5. This is a new warmup/joint/refinement recipe, not simply rerunning original Stage2+Stage3. Original Stage1 does not restart from scratch; Video receives task/timing adaptation in joint training. First warmup begins at the previous trained step-12000 checkpoint, whose backbone originated from the official full initialization.

The new VAE cache must retain exact physical-grid times and provenance. A separately marked finite `smoke_subset` can support only fresh <=4-step warmup/joint checks and cannot initialize/resume formal training. It does not establish complete data readiness. Old v3 checkpoints without the new flag load strictly with feedback disabled; new exports bind the feature flag to runtime metadata, and mismatches are rejected.

## Evidence boundary

`check_joint_proposal_feedback_cpu.py` verifies actual native joint/Action blocks with two tiny native Wan blocks: zero-gate parity, old strict loading, gradients to Video/Action/native motion/proposal, checkpoint parity, proposal sensitivity, pure-noise masking, immutable source, physical range masks, exact K10 eager-versus-graph-body parity, and strict new export reload. Tiny projection/identity Video RoPE and CPU SDPA are explicit substitutes. These checks do not demonstrate full Video GPU backward, CUDA capture/concurrency, actual p99 latency or closed-loop improvement.
