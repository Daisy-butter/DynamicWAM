# Consistency audit revision02, 2026-10-27

The prior CPU-only waiter2764016 was intentionally stopped before evaluation
source edits. No GPU rollout was started and training775c8e2 is unchanged.
Inference source images now normalize in FP32 before VAE BF16/FP16 cast, exactly
matching producer/materialize. Camera freshness uses the same1e-6 numeric epsilon
as training:60real float32(.004) ticks accepted,61ticks rejected. Nominalcamera.24
and backlog.25, clock anchors, queue leases and network weights are unchanged.
107non-executor/non-online source files remain byte-identical; online.py exact AST
delta is independently checked against the frozen training source and hash-bound.
165CPU consistency/clock/prefix/deployment/controller tests passed. Final GPU
acceptance remains pending after training. This revision supersedes the earlier
108all-files-equal statement only for the explicitly bound online preprocessing fix.
New independent waiter uses controller02, revision02/posttrain.json, and writes
smoke/formal artifacts under revision02. No automatic retry or hardware recovery.
