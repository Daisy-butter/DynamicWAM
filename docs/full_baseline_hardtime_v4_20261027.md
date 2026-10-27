# Official Full on the common V4 continuous benchmark

Scope: .50 only, .92 training untouched. Policy deployment (single GPU Full versus three-GPU DualRate) is a disclosed method difference. External scene, clock, action-grid, actuator, initialization and budget contracts are common.

Base source is read-only archive of .92 commit775c8e2b9dccf88e11d5d9f467a9cb85714564de. No training files or processes on .92 are edited. Official Full is used without new weights, prefix conditioning or RTC guidance; native K10 with two Video updates is preserved.

The baseline is explicitly delayed-install sync16: source pixels/state/motion retain their true exposure time; after the entire previous16-row chunk finishes, query the latest available observation while the physical world continues and the actuator holds its last target. A returned chunk crosses the wall/physical release gate, then all16 rows start on the next not-yet-consumed global row boundary. Its actual execution times are logged separately from its source image; there is no claim that all16 outputs describe unshifted exposure-relative future states. This policy staleness is the baseline behavior, not clock corruption.

Startup uses the same inactive warmup/initial-plan interval as V4, then two explicit held action rows followed by the complete Full16 rows. Those initial holds are prepended, not overwrites of model output. All active global rows, including inference-wait holds, consume the same task-native action budget and metrics action counter. Full receives no V4 fast300ms deadline; its declared policy behavior during inference is hold. Wall lag remains bounded at.25s, camera age at.24s, and asynchronous completed results cannot appear in simulated past.

Freeze all700 prepared scene/motion byte hashes and episode-complete instructions for later V4 evaluation; use all35 tasks with20episodes. Reuse original native report/MP4 layout. Failed infrastructure episodes do not silently count as policy failures and are never replaced by resampled seeds. No automatic retry. Full35x20 is admitted only after16 complete eight-GPU smoke episodes and clean exit.

Historical32.69% used160ms/PCHIP/initial latency included and is not this protocol. Current common grid is45 actual physics ticks (~180ms), linear arm endpoints and original grippers. This changes official Full's nominal execution time interpretation, which is disclosed; smoke establishes engineering correctness, not that this is its optimal operating point.
