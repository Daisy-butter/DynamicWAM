# Current experiment: official Full baseline, 2026-10-27

Read harness.md before code/config/script changes. The latest human instruction explicitly authorizes .50 writes and GPU evaluation for this isolated baseline; .92 training and .100 remain untouched. This supersedes older .92-only scope for this task.

Official DynamicWAM_full.pt only, SHA 7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7. Original native K10/Video refresh[0,1], no RTC/prefix/new learned head. Eight GPUs, one environment per GPU, 35 tasks x20 frozen scenes. V4's three-GPU policy placement is a method/resource difference, not an environment requirement; disclose costs.

Common external benchmark: measured float32 .004 physics tick, 45 ticks per .18 action row, linear arm endpoint/native gripper execution, fixed1x physical world, true camera exposures, .24 image age, .25 backlog, bootstrap before activation with explicit initial2heldrows, task-native global-row budgets including inference-hold rows. Full sync16 runs entire generated chunk then queries; returned whole chunk installs at a future global boundary, with exposure and delayed installation separately logged. No fake timing, RTC prefix conditioning, adaptive slowdown, late re-timestamping, or unlimited uncounted holds.

Preserve all historical source/results and prepared700. Work only in this isolated SSD worktree. Require deterministic CPU timing tests, guarded actual8GPU x2 full episodes, full MP4 decode, clean workers/guard closure and unchanged kernel before formal700. All8 temporary device FDs precede NVML; no reset/unbind/reload/recovery or automatic retry. Formal source/config frozen, bounded initial300s acceptance only, no recurring watcher. Necessary suite/cohort/cleanup supervision stays alive until its job ends.
