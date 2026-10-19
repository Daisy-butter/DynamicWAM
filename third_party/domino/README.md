# DOMINO source reconstruction

The evaluated DOMINO revision
`9b1f34129323957baff35bcbc866fceca7e02779` is a local descendant of the
public upstream revision
`93d485a34a6e710bb4c895d83e2bb0520a9f169e`. Because the evaluated revision is
not available from the public upstream remote, this directory retains its
complete binary-safe source delta as `evaluated.patch`.

`scripts/bootstrap_domino.py` fetches the public base, applies the patch under
`external/DOMINO`, and accepts the result only when both identities match:

- reconstructed tree SHA-256:
  `54dd0280e44ea0539274365b7feaeba24e36b7629e8679c2999476addd9c314c`;
- `script/eval_policy.py` SHA-256:
  `fec0002fe1e30833b225d3559b09b583a9a4f7a22c29bc54a3817eb70d1aea38`.

The patch SHA-256 is
`52e8b8dd18523b62daab14efd0f1acd60a5cf93fef257ab1d21294fa43f506ee`.
The upstream Apache-2.0 license is retained alongside it.


## Route1 Stage3 step20000 deployment snapshot (2026-10-19)

The active simulator source for this deployment is
`6c5a00db1bf9fbe496b19876f84d0189a7aaf86f`, based on the same public upstream
`93d485a34a6e710bb4c895d83e2bb0520a9f169e`. Use the separate
`route1_continuous_20261019.patch` and
`manifests/external_assets_stage3_20k_20261019.json`.
The earlier `evaluated.patch` and default manifest remain historical artifacts;
they are not the selected deployment snapshot.

Reconstruct in a new destination under the target user's approved data volume:

```sh
python scripts/bootstrap_domino.py --manifest manifests/external_assets_stage3_20k_20261019.json --destination /TARGET_VOLUME/wuruihan/DynamicWAM/external/DOMINO
```

The bootstrap verifies the patch, reconstructed source tree and evaluator hash.
It does not download model weights or simulator assets. Attach the separately
verified assets only after reconstruction. Never replace the source directory
of an active training or evaluation job.

CPU reconstruction was tested using the locally present exact upstream Git
object, followed by `git apply --check --index`, `git apply --index`, base-relative
whitespace checking and complete Git tree equality, including file modes.
The reconstructed Git tree is `35fe9d21ff6e332f674f2ce3b9c4eda581f784bb`;
all 1,668 tracked file contents also match the active source. An upstream
network fetch was not part of that CPU test.

See `docs/route1_stage3_20k_domino_reconstruction_20261019.json` for hashes and
`docs/route1_stage3_20k_transfer_20261019.json` for the minimal transfer plan.
The continuous physics clock, RTC scheduler, asynchronous camera reader and
native report/video writer live in the parent DynamicWAM repository under
`src/dynamicwam/runtime/domino_continuous`; this DOMINO patch alone does not
enable continuous evaluation. Use the joint-v3 suite at simulation_rate=1 and
the checkpoint-bound 0.16-second deployment profile.
