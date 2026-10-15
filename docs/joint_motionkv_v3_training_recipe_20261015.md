# Joint MotionKV V3 recipe — 2026-10-15

This candidate is isolated in `feat/route1-async-joint-v3-20261015`. The running scaled continuous evaluation remains on its frozen main checkout. V3 GPU smoke and training are not started.

## Model and causal interface

The slow native Video/Action DiT performs two coupled denoising forwards from a source capture and publishes K/V plus the clean action estimate produced at the actual cache-producing step. Native action-motion interaction is retained. The background runtime consumes the latest completed source with a 16-action refresh and an explicit maximum source age of 31 actions / 4.96 seconds.

Current RGB patch innovation, native motion-statistic innovation, and physical-time-aligned source-plan / irrevocable-prefix mismatch update K/V through the existing bounded low-rank native-attention adapter. V3 adds a small source-K/V + source-motion + elapsed-time forecast in the original 12-statistic space. Current motion is a masked target and observed innovation, never a source forecast input. Missing motion measurements contribute zero innovation. Null world caches cannot feed forecast world evidence.

Source action endpoints and cache age stay float64 independently of model precision. Arms interpolate linearly, grippers use previous hold, and unsupported source-plan endpoints are invalid. A 1 ns tolerance absorbs floating-point endpoint roundoff. The 16-action source plan supplies no valid tail after its horizon. Runtime previous queue proposals are soft context near the hard-prefix boundary; training uses a detached source prediction as a historical proposal, not as an allegedly executed action.

## Data and objectives

All action targets use the episode-global 0.16 s command grid; the first endpoint after a real capture may be less than 0.16 s away. Recorded arm values are drive targets. Actual RGB and native motion-history capture times are preserved. Jointly sampled source publication, cache age, query phase and fast latency determine the same RTC prefix draw consumed by training. Prefix labels remain expert demonstration commands; on-policy replay is not yet available.

Native Video DiT retention uses newly encoded physical-grid future video latents at the source capture, paired with source action FM labels. Fast actor suffix FM uses current grid labels. Old variable-cadence video latents and old-cadence teacher imitation are prohibited. The dataset and launch preflight require complete manifest-bound new VAE caches; the currently built index alone cannot authorize training.

Default source video prediction spans 8 × 0.32 = 2.56 s. At source ages beyond 2.56 s, K/V is historical evidence and motion forecasting is learned extrapolation. This does not guarantee planning support over the full current horizon. Evaluate results and deadline behavior by actual cache age, especially the >2.56 s bucket. A longer video-grid recipe needs separate latents and a paired ablation; it is not silently enabled.

## Stages

1. `joint_motionkv_v3_warmup.yaml`: initialize from the actual step12000 checkpoint, preserve all known old parameters and normalizers, strictly initialize only new innovation/forecast tensors, and freeze the slow Video DiT; source bridge forwards run without gradients during warmup. Train native actor, native motion tokens and interface for a candidate 2,000 steps. Fast/interface LR 1e-4; actor LR 1e-5.
2. `joint_motionkv_v3_joint.yaml`: initialize from the warmup full export and jointly optimize slow Video DiT, native Action DiT/motion and interface for a candidate 18,000 steps. Video LR 2e-6, actor LR 5e-6, interface LR 3e-5; fast suffix FM weight1, source video/action FM weights0.1/0.1, motion forecast weight0.1. Old teacher weight0. Nonreentrant gradient checkpointing is enabled.
3. The matching `_smoke.yaml` profiles run four steps each. Passing requires real eight-rank guarded execution, finite losses/gradients, saved full/fast exports and strict reload, clean worker and temporary device-FD closure. CPU checks do not satisfy this gate.

K10/H4/R4 and chunk16 remain unchanged. Warmup and joint are explicit separate profiles. No reduction in denoising steps or formal launch is implied. Full joint checkpoint exports carry schema/interface3, source schedule, physical-grid identity and latent-manifest identity. V1/V2 recipes remain available.

## Current evidence

`outputs/diagnostics/joint_motionkv_v3_20261015/training_cpu.json` records native actor/router/bridge optimization, source-only forecast, masked/null regressions, physical alignment including nonbinary timestamps, checkpoint gradient parity, strict reload, coupled RTC sampling and bf16 preparation precision. Video DiT is a tiny differentiable substitute and attention uses CPU SDPA.

`actual_step12000_init_cpu.json` verifies real mmap initialization: 195 conditioner tensors and 480 backbone tensors preserved, normalizer unchanged, old nonzero update heads retained, new innovation gate and forecast output initialized to zero. This is a full-state schema migration audit, not full Video DiT execution. GPU, end-to-end latency and success-rate validation remain pending.


## Optional real rollout refinement, currently data blocked

`joint_motionkv_v3_replay.yaml` and `_smoke.yaml` select `pairing.kind=real_rollout` explicitly. This is a separate refinement stage, not an unreported mixture. It starts from a completed V3 joint checkpoint and keeps native source Video/Action FM, current suffix FM and motion-forecast losses. Candidate refinement is 2,000 steps with Video/Action/interface LR 1e-6 / 2e-6 / 1e-5. No such checkpoint or actual rollout corpus has yet been produced by this work.

The reader must validate same-trajectory observations, recorded source/current clocks and completed-action counts, exact historical queue on the current endpoint grid, irrevocable executed prefix, official VAE encoding, unchanged normalizer, provenance hashes and development-scene exclusion from the frozen 700 evaluation scenes. Training accepts successful replay self-imitation or same-trajectory expert correction only. Self-imitation is labelled as such; a failed policy suffix is not treated as expert supervision. Full/fast exports and exact resume bind the replay identity. Initial transfer from offline V3 additionally checks its physical-grid and parent offline holdout identities; the new replay episode split has its own identity.

Bootstrap is an explicit collection stratum: source and current observations are both real captures; simulation may advance about one second while completed action count stays zero, and H4 prefix commands are a recorded hold. The prefix history must be validated against the actual executed timeline. No synthetic relabelling of offline expert frames can satisfy this requirement. The existing tensor contract supports `delta_actions=0` and positive physical elapsed time; source plans align by seconds and the loss trains only the replaceable suffix. Synthetic CPU routing tests prove code reachability, not that this distribution has been collected or trained. Missing replay manifest/latent/provenance artifacts fail closed.


This replay stage is success-filtered self-imitation when no corrective expert supplied the labels. It introduces success-selection bias and can forget difficult offline cases; it does not by itself eliminate RTC distribution shift. Keep the short budget and reduced learning rates, compare unchanged offline validation plus separate real bootstrap/steady-state validation, and reject deployment if either degrades materially. A validated usable bootstrap sample in the training corpus is mandatory before the replay stage starts; code-only synthetic bootstrap tests cannot clear this gate.


## Source-time committed actions condition the world prediction

The slow source now carries only the prefix irrevocably known at that source query, independently of the later fast query prefix. Both coupled denoising forwards clamp those clean normalized action rows and assign their action tokens time zero; native state and motion tokens retain the ordinary denoising time. Euler updates and the published clean source plan reapply the same prefix. Source Video FM remains fully supervised while source Action FM ignores only committed rows. Offline source prefix length is jointly sampled with source publication timing; real replay uses the actual source query queue, including explicit d0 bootstrap. This follows the action-conditioned-future principle described by [Streaming-WAM](https://github.com/SJTU-DENG-Lab/Streaming-WAM), implemented through the existing native joint attention rather than an additional large encoder.

Runtime exports bind `slow_source_prefix=clean_committed_v1`; a V3 checkpoint lacking this training/deployment contract cannot be silently loaded into the new runtime. Legacy absent-prefix behavior is unchanged. CPU verification executes the actual SmallWAM joint-forward method, native ActionExpert and motion module, with two small native WanAttentionBlocks, compact projection and identity video RoPE stand-ins: d0 is bit-exact to absent prefix; both source steps use clean/t0 actions; changing the source prefix changes world K/V; gradients reach video, action and motion; source retention masks only committed actions. The raw-recorder/converter synthetic replay fixture also reaches this native joint loss with source d0 and current H4. These remain CPU/schema checks, not actual rollout collection or full-size GPU validation.
