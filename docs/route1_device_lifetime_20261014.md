# Route1 device lifetime correction, 2026-10-14

Branch: `fix/route1-device-lifetime-20261014`, parent `381e1ba47ebbb719f1762e300ccf9f24a3fa8ed2`. Accepted simulator baseline unchanged. All artifacts below are under `/SSD_DISK/users/wuruihan/DynamicWAM/route1_motionkv_20261013` on .50.

## Finding and bounded correction

Original full DynamicWAM training has not been independently run in this diagnostic session. Earlier modified-model smoke and minimal DDP performed arithmetic normally; formal launch was blocked by new assertions at exit, not a proven inability of the original model to train.

The exact official 560.35.05 `nv-kernel.o_binary` matches the installed source-tree binary (SHA256 `c1fe518ec42fae1c42f97cdbd8f9460117ee5daa21a881a144b6f7687fec88f6`). Static OS source and disassembly establish adapter-level IOVAS construction/destruction and the null-IOVAS failure before normal mapping unmap. See `driver_source_audit/exact_iovas_release_audit.md` and its disassembly. All cards had persistence disabled and no active persistence daemon; initial driver anomaly count was 264.

A temporary application-owned device descriptor guard keeps the selected adapters attached while workers release distributed resources. It opens only the UUID-verified `/dev/nvidiaN` character devices, uses close-on-exec, waits until workers have exited, then closes every guard descriptor. It makes no driver, persistence, PCI or recovery changes. Ordinary device open/close does interact with the driver and can initialize/deinitialize adapter state.

Both `diag_device_lifetime_guard2_20261014.json` and `diag_device_lifetime_guard8_20261014.json` passed actual numerical DDP work, worker exits, complete guard closure and subsequent health checks. Driver anomaly count remained 264. The immediately preceding direct two-rank DDP case without this guard failed with three new IOVAS lines. This is evidence for a useful release-order correction; exact allocation ownership and a universal driver root cause remain unproven.

## Training gate

The training supervisor must bind guard enabled state and helper SHA to the new diagnostic authorization, actual smoke manifest and formal receipt. All smoke workers and guard descriptors must exit before post-exit health acceptance. Old failed receipts are preserved. No single-GPU formal substitute or recurring inspection is authorized.

At this source-freeze point, real eight-GPU training smoke, strict full/fast export reload and formal startup are still pending; actual status is recorded in the new SSD run manifests. No claim of long-run stability, policy quality, RTC controller completion or SOTA follows from the minimal tests.
