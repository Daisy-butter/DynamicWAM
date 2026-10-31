# V8 final checkpoint evaluation — .50, 2026-10-31

Protected training: `/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-rtc-20261031`, commit `355d4af324cf8cb2f4f4fe87b6237e5528bc8bfc`.
Evaluation is isolated at `/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031`, branch `eval/dualrate-v8-continuous-final-20261031`.
All `src/dynamicwam` Python bytes must remain identical to training, SHA256 `ec50b572199312ca24e034efcfef352d39655e07d762e2eb5fe76fbeaf2cec77`.

## Launch

```bash
# CPU/filesystem checks only; no GPU use, process signal or evaluation launch
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031/scripts/launch_continuous_v8_eval.sh --check

# Explicit user launch: detached finite controller; safe to close SSH afterwards
bash /SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-v8-continuous-eval-20261031/scripts/launch_continuous_v8_eval.sh
```

Default launch waits CPU-only every 60 seconds (at most 48 hours). It requires the exact formal `500 -> 4000 -> 8000` phases, final global step 12500, complete eight-rank model/optimizer/master/RNG reload receipts, stable final bytes, clean training workers/guards/controller exit and unchanged kernel receipts. No fallback checkpoint or automatically shortened recipe. Training failure stops evaluation handoff.

Only then acquire the existing all-eight GPU guard, require idle hardware, run **2 tasks x 2 full episodes**, and validate final-checkpoint CUDA Graph, true fast/slow overlap, newly published M consumption, real physics during camera/inference, clock/queue checks, native MP4 full decoding and clean exit. Only accepted smoke enters **35 x 20 fresh episodes**. A task failure/success is an evaluation result; a timing/infrastructure failure aborts the cohort and prevents subsequent stages. There is no automatic retry or silent score retention.

The launcher is prepared, not auto-started. CPU readiness is not final-model GPU or SR acceptance. Do not edit frozen source/config once launched. The fixed controller/output directories intentionally reject repeated launch; preserve failed artifacts and prepare an explicitly reviewed new attempt if needed.

## Exact deployment contract

| Item | Value |
|---|---|
| Simulator | continuous 1x immutable wall origin; real physical steps; async camera |
| Executor | linear arm endpoints; native absolute gripper from interval start |
| Action interval | 45 x float32(.004) = .18000000854954123 seconds |
| Fast/slow | independent renderer0/fast1/slow2; two concurrent environments on six GPUs, two guarded idle |
| Fast query admission | new observation, at least one row, .20 s reserve; fixed lease, 2/3 protected rows |
| Slow | four-row refresh interval; TTL 2 s; no serialization of fast behind slow |
| Sampling | K10, accepted CUDA Graph; single joint residual A/Z fast model |
| Observation/flow | real exposure timestamps, maximum .24 s; raw-frame physical-history selector shared with training |
| Deadlines | strict physical and wall; worker receipt after actual host-ready plan |
| Backlog | .5 s plus existing .005 scheduling tolerance; no deadline tolerance |
| Smoothing | no extra EMA or untrained C1/C2/TOPP substitution |
| Benchmark | same prepared 700 scenes/seeds/instructions and native full budgets; `continuous1x_backlog500ms_v1` |

The 25.43% Full result used .25 s backlog. A same-.5 s Full evaluation is required for a strict matched comparison; this script does not launch it.

## Files and audit

Spec: `configs/continuous_v8_final_eval.json`; frozen smoke/formal templates beside it. The old controller's hardcoded V4 phase lengths are replaced by a strictly validated explicit recipe, retaining all completion/hash/cleanup checks. Stale engineering initialization, 32-row truncation, old data manifest/source/sequence bindings and compatibility report metadata are removed from this final-run configuration. The same native camera/PhysX bridge bytes are bound at the isolated evaluation path.

The entry point checks model/runtime/physical-label contracts against the actual last-stage template and manifest, source identity, renderer/FlashAttention dependencies, bridge/environment/scoring hashes, every frozen scene, ffmpeg/ffprobe and disk space. Final checkpoint payload checks remain mandatory in production; pending-weight mode exists only for CPU template preflight.

Artifacts: `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_v8_final_eval_20261031`.

- Controller: `controller01/status.json`, `controller01/controller.log`.
- Smoke: `smoke01`, then `controller01/acceptance.json`.
- Formal: `formal01/suite_status.json`, `formal01/suite_completed.json`; final aggregate audit `controller01/formal_acceptance.json`.
- Per-episode evidence: `formal01/<task>/episode_0000/report.json`.
- Native MP4: `formal01/eval_result/<task>/dualrate_dynamicwam/demo_clean_dynamic/last/v8_final_formal01_20261031/episode0.mp4` and corresponding native metrics.

Research and proposed later executor ablations are in `docs/continuous_executor_research_20261031.md`. They are not enabled by this launcher.
