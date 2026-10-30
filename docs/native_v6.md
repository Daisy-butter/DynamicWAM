# Native Event Single V6

Scope: .92 only, independent SSD worktree. V5 archived and stopped through its verified supervisor. .50/.100 protected. Start formal from official Full only after current-source CPU, real native GPU and all-three-phase eight-rank acceptance.

## Clock and command contract

Native synchronous DOMINO pauses physics during inference. Actual camera/state/flow timestamps and cache ages use FP64 SceneStepClock seconds. Commands have monotonic integer event identities; TOPP completion timestamps are recorded only after execution. Never infer command progress from elapsed_seconds/.18. Never turn unknown future TOPP durations into physical timestamps.

V6 A targets the next 16 command events; Z targets events +1/+2/+4/+8 for each supervised entity. This intentionally changes V5's fixed-second Z for the native protocol. Structured future tokens refer to command groups 1..8 and 9..16; historical tokens retain real second-domain support. The network encodes these domains separately. The original frozen WAM's nominal codec metadata is not used to assert deployment physical endpoints.

Training waypoints are sampled on a common expert trajectory grid (.18s nominal sample spacing), including partial terminal windows. Camera/query/source times are anchored to their actual recorded frames. A common event grid keeps paired query targets aligned even when camera sampling times differ. Event IDs never encode seconds. Ground-truth future labels use the actual expert record timeline at those target waypoints. Offline expert data is not on-policy TOPP replay; event semantics fix mislabelled timestamps but do not eliminate dynamics/state distribution shift.

## One fast action backbone

Copy Full's complete 768/12 Action Expert and motion module. Retain the native video K/V path and its single action decoder. Remove V5's independent 384/6 action generator and velocity summation.

Latest observation, structured O/F memory, previous soft action proposal, and noisy Z interact inside four of the native blocks through cross-attention adapters. The same native FFNs also update Z tokens; Z and contact have output heads, not another action network. Zero action gates preserve native motor computation at initialization. Verify exact full-weight parity and nonzero observation/Z influence when enabled.

## Aligned A10 supervision

The complete original Full slow model remains frozen as world model and teacher through all phases. For training, obtain its A10 from the *current query's* actual image/state/motion and exact command origin. If cached M came from an older query, run a separate no-grad fresh-query teacher; never relabel old A10 as current. Teacher construction sees no future frames/clean action suffixes.

Train the student's clean action estimate against A10 with a bounded auxiliary Huber loss (.1), weighted by teacher/expert agreement. Expert action loss stays primary; the first two executable commands have weight4. No V5 shifted-noise prior and no inference-time hard constraint to stale A10. A10 supervision encourages retention; it is not a mathematical SR lower bound and is not few-step consistency distillation.

## Precision and stability

Keep tail masks. Expert arm first-difference loss .05; adjacent queries two events apart produce a sampled previous student plan, aligned by integer IDs and used as a soft condition. Add agreement-weighted overlapping arm consistency .05. Do not average gripper commands or force stationary motion. Previous plans are proposals, never fabricated executed actions; actual history remains expert/observed. This is offline paired-query training, not on-policy recovery training.

Deploy the identical single model and previous-plan condition. Reuse overlap noise, execute native TOPP, record joint-limit projection and invalidate affected Z. Fast every2 commands; slow every4 or age>1.44 seconds. Refresh blocks in native synchronous mode while physics is paused; no continuous real-time claim. Separate continuous implementation and benchmark comparisons remain necessary.

## Training and gates

Fresh Full initialization; 8 GPUs, micro1, accumulation2. Warmup1000 trains adapters/readout/observation with motor frozen; then10k small-age and20k broad-age phases tune the motor at3e-6 and new modules at1e-4. Full teacher remains frozen, BF16 video / FP32 action-time and optimizer masters preserved. Independent stage controller, immutable identities, strict all-rank reload, no retry or GPU recovery. Smoke weights never seed formal training.

Required acceptance: same-input motor parity, latest-observation/Z influence, event/clock and wrong-origin rejection, finite losses/gradients including paired queries and teacher, full optimizer/RNG restore, genuine complete native episodes with MP4 decode, clean owned exit and unchanged kernel. Performance/SOTA is not accepted by these engineering gates; evaluate matched native Full and ablations after training.
