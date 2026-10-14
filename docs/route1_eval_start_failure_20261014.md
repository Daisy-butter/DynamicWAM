# Formal RTC evaluation startup failure, 2026-10-14

The prepare phase completed 700/700 scenes at 19:36:52 Beijing. All 1,400 scene-manifest and expert-motion hashes were reverified. Formal evaluation stopped before any policy episode: 0/700 valid results. Its supervisor started at 19:37:12 and finished owned-process/FD cleanup at 19:38:23. No evaluation remains running.

Exact saved delta versus the formal baseline:
- `[343296.379408] NVRM: knvlinkExecGspRmRpc_IMPL: Failed to acquire locks for gpumask 0x0`
- `[343296.379410] NVRM: knvlinkCoreGetRxSublinkModeCallback: Error getting current RX sublink state!`

There were no new IOVAS/pIOVAS lines. Matched anomaly count changed 285 to 286. The new error precedes the first guard open attempt (343296.771552) by 0.392 seconds and suite process startup (343312.56) by 16.181 seconds. It cannot originate from this run's model execution or worker teardown. All eight descriptor lifecycles subsequently closed, with no further new lines after the startup snapshot. The exact issuing process/call remains unproven.

The supervisor captured its baseline before GPU telemetry, then opened the guard and launched workers without an immediate driver recheck. The first periodic startup check detected the earlier error late. The candidate fix records/checks phase boundaries against the immutable original baseline and prevents workers starting after an observed prelaunch anomaly. This is a detection/lifecycle gate correction, not a driver fix. CPU regression only; no GPU workloads or repeated failed cases were launched.

Evidence root: `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/motionkv_rtc_20261014`. Exact receipt: `formal_start_failure_audit.json`; original evidence remains in `formal35x20_eval_guard/` and `formal35x20_pipeline/`. Frozen reusable scenes: `formal35x20_prepare/frozen_manifest.json`, SHA256 `cff71ba1b8bf17c58082e615016e20775077787b455746e87261eaaaf60b6fb2`.

Original eval source commit `abd2dc7b82471fdac7221b7d540b4754c37238e3` is preserved; fixes use branch `fix/route1-eval-preflight-health-20261014`. Do not rerun the failed pipeline or advance its authorization hash to accept new errors. Further GPU attempts require a specifically justified software/runtime remedy with fresh evidence, under the existing no-device-recovery constraints.
