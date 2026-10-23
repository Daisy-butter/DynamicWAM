# DualRate Joint V1 entry

Read `harness.md` and `docs/dualrate_joint_v1.md` before changes. This isolated .92 worktree is the only writable remote codebase for this task. .50/.100 are read-only; preserve .92 home dirty source, AHA-WAM, running jobs and shared external assets. Large artifacts belong to verified .92 SSD wuruihan paths. Never reset/unbind/reload GPU/driver/PCI; never launch automatic retries or recurring watchers.

Follow the attached DualRate design: full DynamicWAM slow Video+Action+motion, independent fast joint A/Z, source-only predictive bridge with gradients, separate supervised forward, exact physical clocks and hard committed prefix. CPU tests or tiny native fixtures do not establish full GPU/DOMINO readiness. Candidate formal_training_ready=false; no formal/DDP launch before real data and runtime acceptance. Track every source change and preserve failures.
