# Causal camera and slow flow repair — 2026-10-24

Scope: .92 isolated dualrate-joint-v3-20261024 only; parent aa96827. No GPU launch, driver operations, or .50/.100 changes.

- Condition-frame selection requires exposure + camera delay <= event + 1e-6. The existing 30 ms match tolerance bounds only the older side; it cannot admit a frame whose simulated readback has not completed. The pre-activation first-frame bootstrap remains explicit.
- The slow VAE's four flow images now come from the same physical pairs as source motion features. Exact matching cache pairs may be reused; mismatching pairs are recomputed and unmatched pairs produce zero images with invalid motion masks. No raw data or old cache is overwritten.
- New regression cases cover irregular timestamps, missing internal endpoints, causal availability, float32 clock noise, stale caches, and offline/online numeric and flow-image equality.

This is a data-path repair, not deployment acceptance. smoke10 still ended in DeadlineMiss during overlapping slow/fast work. No GPU resource scheduling or timing deadline is changed. The legacy full_01 index is not rebuilt or certified by these CPU fixtures; full real-data eligibility and frozen provenance must be revalidated before formal training. formal_training_ready=false.

Validation: 30 relevant CPU regressions passed in the existing robotwin runtime;
the final six causal/flow cases also passed (four overlapping and two additional
cache-hit cases, 32 distinct cases in total). git diff --check passed. The first
lila run had 27 passes and three missing-dependency failures (einops/sapien);
switching to the existing robotwin runtime resolved those environment failures.
pytest 8.3.5 is isolated under SSD outputs/dependencies/dualrate_pytest_review_20261024.
No GPU validation was performed. Final commit and logs are recorded in the local outputs receipt.
