# Same-device fast-priority scheduling candidate

Parent d1ca1f4841e838a87a8a0b42341e9b1a21ca6577; only .92 isolated V3.

smoke10 query3 missed its unchanged protected-prefix deadline while slow version2
was running. Fast already used CUDA stream priority -1; stream priority alone did
not establish the required latency. Overlap is evidence, not proof of sole cause.

The new conservative scheduler reserves fast priority before submitting its worker.
The complete slow job (including VAE preprocessing/readout) runs in an operator
dispatch scope. Each dispatched slow operator drains only the slow stream before
releasing its permit. Pending fast work prevents the next slow operator from
starting; fast retains its permit through CPU action transfer and drains its own
stream before releasing. Submission failures, worker exceptions and drain errors
release reservations. No device-wide synchronize, clock/deadline/TTL change,
fallback to old actions, model/precision change or additional GPU is introduced.

This trades slow throughput for reduced slow queue depth. Custom CUDA extensions
(including direct FA2 launches) may bypass the dispatcher: actual overlap traces
and the maximum remaining kernel delay MUST be measured. It is not hard GPU
preemption, and CPU regressions do not establish deadline or freshness guarantees.
Slow-reference completion/expiry must be checked alongside fast action latency.

11 CPU scheduler/runtime regressions passed. No GPU run was launched; smoke10
is NOT declared fixed/accepted. formal_training_ready=false. Preserve smoke10 and
run a fresh guarded deployment against this frozen commit before formal training.
