# Route1 v3 workflow and acceptance

Parent: ac7d77e; branch feat/route1-async-joint-v3-20261015. Isolated SSD worktree only; the existing scaled evaluation and Route2 are protected. No formal training in this development step.

1. Freeze the temporal contract: physical action grid, immutable RTC prefix, replacement boundary, source/capture/publication timestamps and maximum cache age. Audit historical implementation and primary literature.
2. Data: audit original clocks and action semantics; construct physical-time targets without relabeling frames. Jointly sample cache availability and action reservation. Maintain episode splits and causal observations. Require genuine trajectory provenance for policy-prefix replay.
3. Model: train Video DiT world dynamics, motion-conditioned cache trend updates, and native Action DiT dynamics together. Preserve native motion representation. Match source-plan physical times, auxiliary supervision, training and checkpoint export/loading.
4. Runtime: bounded background slow producer; immutable published cache, fast consumer, independent CUDA streams/events, episode-safe lifecycle. Controller preserves committed actions and atomically replaces only the future tail.
5. Verification: CPU temporal/queue/cache/model-gradient/reload/real-data smoke, independent adversarial review, fix findings, rerun affected tests. Then actual guarded GPU capture/backward/8-GPU/continuous-video smoke when resources are available.
6. Delivery: exact source commit, data and test receipts, unresolved gates, and a runnable next-stage command. Formal-training readiness is false until actual full-model GPU and continuous deployment checks pass.

Review roles: methods reviewer checks dynamics, causal leakage and attribution; systems reviewer checks time, concurrency and lifecycle; experienced implementer checks actual wiring, gradients, artifacts and reproducibility. Finite review cannot establish that no better research design exists; acceptance means no known blocking defect and explicit empirical evidence.

Current status (2026-10-16 Beijing): v3 candidate implementation and cross-review complete for this iteration; 20 CPU integration programs passed with source unchanged, and real step12000 migration passed. Full fixed-grid data index covers 1,416,299 samples. Source-prefix-conditioned slow dynamics, real-trajectory recorder/converter/training adapter, exact instruction lookup and explicit replay refinement are wired. GPU readiness is false: old evaluation has no clean release receipt; fixed-grid VAE encoding, real development replay and full GPU/1x smoke remain pending. No watcher, retry or formal training launched. See the final receipt and design document; this is not an accepted performance milestone.
