# One-shot formal sequence acceptance

`scripts/run_causal_spatial_formal_sequence.py --acceptance /absolute/acceptance.json --label UNIQUE --detach`
starts a persistent CPU controller only after complete, hash-bound acceptance.
No GPU work was launched while implementing or testing this script.

The root writes the acceptance document after actual acceptance. Required fields:

* `format`: `causal_spatial_formal_acceptance_v1`; `passed`: true.
* `git_head`, `source_sha256`: exact frozen candidate identities.
* `configs`: keys `interface_adapt`, `policy_pretrain`, `joint`; each value is
  `{ "path": "absolute canonical formal recipe", "sha256": "actual hash" }`.
* `evidence.data`, `evidence.cpu`: hash-bound JSON references with `passed=true`.
  CPU additionally requires actual model full/fast roundtrip and teacher
  normalization equality.
* `evidence.smokes`: the same three phase keys, each containing `audit`,
  `guard_status`, `guard_manifest` hash-bound JSON references. Export audits must
  use `causal_spatial_export_audit_v1`, all eight ranks must complete four actual
  optimizer updates without nonfinite values/OOM, and strict full/fast reload,
  saved optimizer/RNG state and actual final checkpoint hashes must be present.
* `evidence.graph`: `result`, `guard_status`, `guard_manifest` references. It must
  test the actual joint smoke final export, use the shared production Graph,
  preserve b under proposal changes, update graph inputs without recapture,
  consume a completed background M, and pass full/fast numerical parity.
* `evidence.domino`: `completed`, `report`, `guard_status`, `guard_manifest`
  references. Same actual checkpoint as Graph; complete native 1x H2/R2 episode,
  real fast handoffs, shared production Graph, and exact Python/Torch version
  equality with Graph. At least one live `slow_launched=true` must be followed
  by a request consuming a newer `source_publication` captured at/after that
  launch observation. Native MP4 bytes/timestamp sidecar/full-frame decode
  metadata are revalidated. Success rate is not an acceptance condition.

Every reference above has `path` and `sha256`. The Graph runtime is recorded at
`result.runtime`; the DOMINO runtime at `report.worker_startup_metadata.runtime`,
both containing matching nonempty `python`, `torch` and `cuda` strings.
All guard receipts must be successful, final, fully released, source-matched,
with the current guard helper hash and all eight adapters covered. The last
accepted DOMINO guard is the first formal stage's `previous_guard`.

Formal order is interface adaptation 2,000 -> policy pretraining 30,000 -> joint
20,000 updates. These are existing recipe budgets, not a speed/quality promise.
Each subsequent `continue_from` must name the exact previous formal final
checkpoint. Diagnostic weights are forbidden. Smoke and formal model, runtime,
data, microbatch, accumulation, loss and learning-rate fields must agree.

Each stage creates a new controller label, runs the existing guarded launcher
with `--worker`, and waits on its process. Only a successful final lifecycle and
strict export audit permit the next stage. The acceptance path is passed via
`CAUSAL_SPATIAL_FORMAL_ACCEPTANCE`. There are no automatic retries, additional
GPU inspections, failed-lifecycle overrides, native-budget extensions or resume
into existing formal directories. Signals stop the owned child controller and
preserve failure receipts.

Local validation: `tests/test_causal_spatial_formal_sequence.py`, 15 CPU tests
passed, including stale identity, changed artifacts, missing smokes, wrong
phase/checkpoint, incomplete ranks, strict-reload failure, unclosed guard,
runtime mismatch, missing live M refresh and corrupted video bytes. Fixtures
are synthetic receipt data; they are not evidence of actual GPU acceptance.
