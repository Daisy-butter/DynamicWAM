# Deployment input normalization precision, 2026-10-18

The deployed state, native hard prefix, source RTC prefix, fast RTC prefix, and previous action queue cast raw commands to the Action DiT dtype before mean/std normalization. Offline physical-grid training normalizes raw FP32 commands first. With the current BF16 model and three real train/validation samples, the old order differs by up to 0.015625 in normalized coordinates.

Normalize these raw inputs in FP32, then cast normalized values to the existing model dtype. No statistic, checkpoint, architecture, training loss, output denormalization, or motion representation changes. Numeric motion already enters as raw FP32 and is normalized once in the native motion encoder and updater, each with the same fixed statistics.

CPU validation: three new tests execute actual HeadFlowRunner state/prefix, source prefix, and MotionKVPolicy fast-packet paths (the model sampler is a CPU stub). They verify training-order equality, immutable inputs, bootstrap/invalid-tail handling, and unchanged raw FP32 motion. Existing RTC controller tests: 26 passed. Three real dataset samples (43, 243600, 1036209) now give bitexact state/source-prefix values relative to FP32-normalize-then-BF16 training preprocessing.

This is isolated from the running Stage3 worktree; no GPU or simulator was run. GPU/closed-loop validation belongs to the later deployment acceptance. The separate audit receipt is `/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/normalization_audit_20261018.json`. Its statistical holdout limitation remains: motion statistics were fitted over all 10500 episodes before the later training/validation partition. Existing normalization statistics must not be silently refitted during Stage3.
