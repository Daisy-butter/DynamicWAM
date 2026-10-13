# Route1: motion- and action-conditioned world cache

2026-10-13; branch `feat/route1-motion-kv-20261013`; parent `b5907680bf4d713a2cce87d4c95b92a0561ff2f4`. The accepted continuous-simulator tag remains unchanged. This is a candidate architecture, not a measured improvement.

## Evidence and hypothesis

The saved full-WAM + bounded feedback result is 242/580 = 41.724% on 29 common tasks in **native synchronous DOMINO** (`plan_feedback_v1_result_analysis_20261006.md`). It is not a continuous-simulation score. That feedback changes the arm output after a 16-action world plan. Here the native Action DiT stays the actor and the correction is inserted before its original joint attention.

Source inspection establishes that video K/V depends on the joint denoising action context. Reusing it unchanged after the observation or action changes is an approximation. We hypothesize that a bounded learned update guided by current RGB, causal motion and the current denoising action state improves that approximation. The update is not mathematically equivalent to recomputing Video DiT; success and latency need measurement.

## Candidate

- Immutable official full model builds slow world K/V. Source caches are detached; the training student cannot silently change their distribution. Age coverage is 0/4/8/12 actions, world refresh every 16 actions; clock metadata uses actual source times.
- Fast observation memory keeps 4x4 spatial tokens per camera, state, instruction, cache age and the native four-interval 12-D optical-flow statistics with valid masks. These statistics are global image motion, **not** object-local flow or privileged object positions. Learned spatial routing may focus on moving objects; this remains a hypothesis.
- Each of the 12 layers constructs rank-8 assignments over cached world tokens once per fast query. Each denoising step reads its current action tokens to produce rank updates to K/V immediately before original attention. Updates have zero output initialization and componentwise tanh bounds of 0.25 times the base token/head RMS. Original cache tensors are never modified. This is a componentwise reference-scale bound, not a spectral/relative-vector-norm guarantee.
- Native state + 16 action + 4 motion + register tokens, action normalization, native dynamics projection and flow schedule remain. No separate output-residual actor. A changed action therefore changes the correction at every layer/denoising step.
- Adaptation retains 10 denoising steps. World/text/vision backbone is frozen; action expert, native motion module and new conditioner are trained from the official full checkpoint. Expert suffix flow loss weight 1, current-observation immutable teacher retention weight 0.1. Teacher conditioning uses its own clean trajectory prefix.

## RTC decision before training

Continuous simulation alone does not require RTC; asynchronous action generation does. This candidate prepares for asynchronous deployment by **training-time action-prefix conditioning**, following the training-time extension linked by the original RTC authors. Original RTC uses inference-time guidance and does not itself require retraining.

Training samples d uniformly in [0,4], injects d normalized clean actions at t=0, retains the normal flow time for remaining action/state/motion/register tokens, and computes mean suffix MSE per example. Inference overwrites the same committed prefix before each prediction and after each Euler step. d=0 is included. A 16-action chunk always leaves at least four executable suffix actions. Prefix distributions and bounds are provisional until actual deployment delay measurements; delays are not converted using a guessed physics timestep.

`rtc_prefix.py` reserves commands using actual command-start times, including an in-flight command, and rejects over-limit, underrun and late handoff. It is **not an asynchronous executor**. Before RTC evaluation, implement and validate the controller's timestamped prefix/suffix handoff, observation/chunk anchor alignment (especially an already-running command), target-end versus command-start semantics, normalization and measured delay support. The accepted synchronous controller is unchanged. RTC exports reject legacy deployment by default; explicit d=0 ablations/low-level graph tests can opt in. Until controller tests pass, do not report train/deploy RTC timing parity or asynchronous evaluation.

## Acceleration and sources

Reuse immutable text embeddings, action-ready K/V materialization and dense unpadded attention. Query-static rank assignments and RMS are computed once; per-step corrections depend on action tokens. CUDA Graph now has fixed-shape prefix buffers and 1/2/4/10-step support, with strict trained-solver contract and replay copies. K10 + MotionKV GPU graph parity and latency remain unverified. Do not enable graph in a claimed result before `check_motionkv_graph.py` passes on a real checkpoint. torch.compile and NFE reduction are deferred until this baseline is stable; stacking unmeasured optimizations would obscure the cause of failures.

Primary sources inspected 2026-10-13; details also in `docs/route1_motionkv_sources.json`:

| Source | Adopted idea / boundary |
|---|---|
| [AHA-WAM](https://github.com/serene-sivy/AHA-WAM) | Observation-guided cached-context refinement; put the update inside DynamicWAM's original joint attention and include causal motion/action conditions. |
| [RTC and Training-Time Action Conditioning](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | Distinguish inference PiGDM guidance from trained prefix conditioning; choose the latter for this candidate. |
| [VLASH](https://arxiv.org/html/2512.01031v1) | State/action/observation time alignment; shifted-state training alone does not repair object-motion context. |
| [Realtime-VLA V2](https://github.com/dexmal/realtime-vla-v2) | Training-time prefill and static serving; trajectory timing is separate from inference latency. |
| [Flash-WAM](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM) | Later modality-specific consistency distillation, after a reliable teacher; do not just cut NFE. |
| [Efficient-WAM](https://github.com/jiajun613/Efficient-WAM) | Compact/coarse dynamics representation; does not prove action-conditioned stale K/V exact. |
| [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) | Compilation, graph and invariant caching; future-free inference is not assumed safe for DOMINO dynamics. |

## Training and gates

Config: `configs/multirate/motionkv_rtc_adapt_7gpu.yaml`; 12,000 optimizer steps, batch 1/GPU, accumulation 2 (effective batch 14 at 7 ranks), action LR 1e-5 / conditioner 1e-4, warmup 200, BF16, clip 1, seed 20261013, checkpoints every 500 with 4 retained. No resume from old feedback weights; official initialization SHA256 `7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7`.

Packed data: `data/packed/domino_absolute_motion_v2`; source `data/converted/domino_exact_time`; 10,500 episodes / 35 tasks, train 9,450 and val 1,050, partition seed 20260928. Native masks handle zero-time intervals; do not replace them with assumed 30 Hz intervals. No test rollout selection is used for training.

Required order: CPU contracts → administrator resolution of the saved driver anomaly → one-GPU real-data four-step smoke plus export/reload → two-GPU four-step smoke → freeze validated source/config and receipt → seven-GPU formal start → brief finite-loss/gradient/checkpoint/log inspection. The job's own status writer persists; no scheduled inspections or automatic evaluation.

CPU checks do not establish real-model BF16/FlashAttention, GPU memory, DDP, checkpoint reload or graph equivalence. Record actual gates in the run receipt, never pre-mark them passed. Formal launcher refuses absent/failed smoke evidence. Reject this candidate for nonfinite gradients, cache/input mutation, broken dynamics/prefix retention or sustained validation regression; do not mask failures by shortening inference or changing simulator timing. Later development comparison should pair the frozen continuous baseline, d=0 candidate, RTC candidate and motion/action-update ablations at the same task/seed/resource protocol.

All artifacts/cache/tmp are under `/SSD_DISK/users/wuruihan/DynamicWAM`; training outputs under `outputs/training/<training.name>`, supervisor/evidence under `route1_motionkv_20261013`. No Route2 changes. The current GPU anomaly and actual readiness are recorded in the delivery receipt; this document is a recipe, not a launch/completion claim.
