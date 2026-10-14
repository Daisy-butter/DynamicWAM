# Route1 device lifetime correction, 2026-10-14

Branch: `fix/route1-device-lifetime-20261014`, parent `381e1ba47ebbb719f1762e300ccf9f24a3fa8ed2`. Accepted simulator baseline unchanged. All artifacts below are under `/SSD_DISK/users/wuruihan/DynamicWAM/route1_motionkv_20261013` on .50.

## Finding and bounded correction

Original full DynamicWAM training has not been independently run in this diagnostic session. Earlier modified-model smoke and minimal DDP performed arithmetic normally; formal launch was blocked by new assertions at exit, not a proven inability of the original model to train.

The exact official 560.35.05 `nv-kernel.o_binary` matches the installed source-tree binary (SHA256 `c1fe518ec42fae1c42f97cdbd8f9460117ee5daa21a881a144b6f7687fec88f6`). Static OS source and disassembly establish adapter-level IOVAS construction/destruction and the null-IOVAS failure before normal mapping unmap. See `driver_source_audit/exact_iovas_release_audit.md` and its disassembly. All cards had persistence disabled and no active persistence daemon; initial driver anomaly count was 264.

A temporary application-owned device descriptor guard keeps the selected adapters attached while workers release distributed resources. It opens only the UUID-verified `/dev/nvidiaN` character devices, uses close-on-exec, waits until workers have exited, then closes every guard descriptor. It makes no driver, persistence, PCI or recovery changes. Ordinary device open/close does interact with the driver and can initialize/deinitialize adapter state.

Both `diag_device_lifetime_guard2_20261014.json` and `diag_device_lifetime_guard8_20261014.json` passed actual numerical DDP work, worker exits, complete guard closure and subsequent health checks. Driver anomaly count remained 264. The immediately preceding direct two-rank DDP case without this guard failed with three new IOVAS lines. This is evidence for a useful release-order correction; exact allocation ownership and a universal driver root cause remain unproven.

## Training gate

The training supervisor must bind guard enabled state and helper SHA to the new diagnostic authorization, actual smoke manifest and formal receipt. All smoke workers and guard descriptors must exit before post-exit health acceptance. Old failed receipts are preserved. No single-GPU formal substitute or recurring inspection is authorized.

Training source frozen at `5a44e942adc53181f4c036ba8e424e96c641b6ec`, source SHA256 `2cc8ef948e9e221598da6bec1f2f21ccc6b3fe9bb80bdda874616e7a2c185e32`. The real `smoke8_guard_20261014` completed four optimizer steps, validation, full/fast exports, clean worker and guard teardown, unchanged driver evidence and strict CPU reload. Relevant CPU tests also passed: guard error cleanup, supervisor ownership/receipt gates, NCCL environment and isolated rank mapping.

Formal training started at 12:02:35 Beijing time on 2026-10-14: `formal8_guard_20261014`, supervisor PID `1364813`, training launcher PID `1365048`. Eight GPUs, per-device batch 1, accumulation 2, effective batch 16, target 12,000 optimizer steps. Checkpoints/logged training outputs: `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/motionkv_rtc_adapt_8gpu_20261014`. Supervisor and train logs: `<artifact root>/formal8_guard_20261014/`.

At the saved 12:08:33 snapshot, training had logged step 140, loss 0.496 and gradient norm 1.02. The bounded 300-second health window had closed; its final check retained anomaly count 264. No further GPU/driver polling or recurring inspection is scheduled. Training continues; this snapshot is not training completion or a long-run-stability claim. The original full-model training was not separately rerun, and RTC controller integration/policy-quality evaluation remain separate work.

Immutable initial-run evidence index: `route1_device_lifetime_training_20261014.json`, with frozen `formal_initial_status.json` and `formal_initial_health.json`. Keep the earlier failing cases. Documentation may advance independently while the running training keeps the source snapshot above.
