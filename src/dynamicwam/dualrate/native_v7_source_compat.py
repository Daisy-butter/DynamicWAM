"""Narrow, auditable deployment-only derivation from frozen V7 training source."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

TRAINING_COMMIT = '4f3b91e66d61575b675d0d12dd9be1f2caaa46e2'
DEPLOYMENT_ONLY = {
    'dualrate/online.py', 'dualrate/native_v7_deployment.py',
    'dualrate/native_history.py', 'dualrate/native_v7_source_compat.py',
}


def validate_deployment_source(checkpoint_source, *, repo=None):
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    prefix = 'src/dynamicwam/'
    archive = subprocess.check_output(['git', '-C', str(root), 'archive', TRAINING_COMMIT, prefix])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        baseline = {m.name[len(prefix):]: tar.extractfile(m).read()
                    for m in tar.getmembers() if m.isfile() and m.name.endswith('.py')}
    digest = hashlib.sha256()
    for path in sorted(baseline):
        digest.update(path.encode()+b'\0')
        digest.update(hashlib.sha256(baseline[path]).digest())
    if checkpoint_source != digest.hexdigest():
        raise ValueError('checkpoint is not from the exact frozen V7 training source')
    current = {p.relative_to(root/prefix).as_posix(): p.read_bytes()
               for p in (root/prefix).rglob('*.py')}
    changed = {p for p in baseline.keys() | current.keys() if baseline.get(p) != current.get(p)}
    if changed - DEPLOYMENT_ONLY:
        raise ValueError('non-deployment source changed: '+str(sorted(changed-DEPLOYMENT_ONLY)))
    if baseline.keys() - current.keys():
        raise ValueError('frozen source files removed')
    return dict(training_commit=TRAINING_COMMIT, training_source_sha256=checkpoint_source,
        protected_model_and_training_equal=True,
        deployment_changes={p: hashlib.sha256(current[p]).hexdigest() for p in sorted(changed)})
