# 2026-11-03 checkpoint loader repair and gated evaluation

Live audit found manual02 completed 35 native collection episodes and three
eight-rank diagnostic phases (4 updates each, all 24 strict restores). Its
trained deployment failed before rollout because the slow child passed a string
to data.sha256_file, which expects Path. Formal training never started. The
diagnostic last.pt is not a completed formal model and cannot be benchmarked.

The repair canonicalizes the checkpoint path, verifies SHA256, and strictly
loads the trained state. Numerical src, architecture, training and deployment
configuration remain byte-identical. Prior evidence is reusable only after
current rehashes of source, protected scripts, train data, replay shards,
checkpoint identities, native videos, exact phase profiles, all-rank restores,
owned clean exits, eight closed GPU descriptors and unchanged kernel evidence.
The old trained deployment failure stays failed. It is never reclassified.

A separate manual03 output reruns the repaired trained deployment. Only success
permits 500/1500/3000 formal updates, still from official DynamicWAM_full.pt and
fresh fast modules, never diagnostic weights. Existing finite guards check idle
GPUs before any job, stop on failure, and never reset/recover GPUs or retry.

The user's November 3 request authorizes the subsequent full evaluation. After
all formal phases and their strict restores/clean exits, verify the final weight
SHA and run four complete native smoke episodes. These now exercise the exact
frozen evaluation scene loader. Then four two-GPU groups run all 35x20 frozen
scenes with original native TOPP, budgets and MP4 layout. Smoke results are
excluded. Final acceptance requires all 700 unique identities, matching weights,
native timing audits, full MP4 decoding, metrics/report agreement and clean
guards. Store score and artifact hashes in formal_eval_result.json. No scheduled
task is created; the one-shot server pipeline remains independent of chat/SSH.

This supersedes only the earlier statement that final35x20 was outside the
authorized automatic sequence. No model performance guarantee is made.
