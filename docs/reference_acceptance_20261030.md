# Empty-reference acceptance repair, 2026-10-30

User authorizes a fresh .50 35x20 evaluation using the existing final checkpoint
and continuous500ms benchmark; no .92 changes or retraining in this step.

The prior formal01 failed during terminal validation of hanging_mug episode2.
The 59.940s exposure required1.631s for camera readback, including .836s/.785s
left/right-camera reads. Physics remained continuous with millisecond backlog.
Fresh-image admission correctly deferred queries; at61.848s the last published
M(source59.580s) was older than its2s TTL. Runtime correctly selected the trained
empty_world branch(reference_version=-1, source/age=null), then later consumed
new M. The old validator incorrectly required M for every active query and
aborted the cohort only after this episode had completed its162s action budget.

Repair only the CPU acceptance scripts. Missing M is accepted iff bootstrap
published a real M, no causally published M remains within TTL, and explicit
reference=-1/source=null/age=null are recorded. All original deadlines, wall
release, immutable clocks, prefixes and action budgets remain mandatory. A
new valid M must again be used. Report valid-M and empty-M counts separately.
Response acceptance uses the same condition; its .60s/.65s response and90%
two-prefix gates are unchanged. Keep real overlap/new-M consumption smoke gates.

This uses existing trained reference dropout and the already captured empty-M
CUDA Graph; it does not extend TTL, accept stale tensors, pause physics or wait
for the slow route. It does not claim to eliminate occasional renderer readback
stalls. Longer camera outages may still legitimately exhaust a queue/deadline.

Use an isolated SSD worktree and new results. Preserve all46 old reports and
videos as failed-run evidence; do not relabel them or merge old scores. CPU
regressions and replay of the original raw events precede current-source GPU
smoke. Four full episodes/MP4 decode/Graph/concurrency/clocks/clean GPU release
gate the independent no-retry formal700 controller. Model/data/runtime sources,
checkpoint and benchmark parameters are unchanged.
