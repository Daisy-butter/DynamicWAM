#!/usr/bin/env python3
"""Wait once on the registered physical preflight, then launch the gated DAG.

No chat polling, retry, source repair, seed replacement or gate relaxation.
The full DAG still repeats its own source/data/physics/learning/latency gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--receipt-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    preflight = a.preflight.resolve(strict=True)
    run = a.run_root.absolute()
    receipt = a.receipt_dir.absolute()
    base = Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training').resolve()
    for path in (preflight, run, receipt):
        path.resolve().relative_to(base)
    if run.exists():
        raise FileExistsError('Refusing to relaunch or overwrite existing run')
    config = read(preflight/'run_config.json')
    if config.get('format') != 'dynamicwam_predictive_feedback_pipeline' or config.get('stop_after') != 'dense_clock':
        raise ValueError('Expected the explicit one-scene physical preflight')
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not a.execute:
        receipt.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), '--preflight', str(preflight),
            '--run-root', str(run), '--receipt-dir', str(receipt), '--execute']
        write(receipt/'request.json', dict(argv=command, script_sha256=script_sha,
            preflight=str(preflight), run_root=str(run), created_at=time.time()))
        with (receipt/'supervisor.log').open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, cwd=repo)
        write(receipt/'launch.json', dict(pid=child.pid, argv=command))
        print(json.dumps(dict(pid=child.pid, receipt_dir=str(receipt), run_root=str(run))))
        return
    request = read(receipt/'request.json')
    if request['script_sha256'] != script_sha or request['run_root'] != str(run) or request['preflight'] != str(preflight):
        raise ValueError('Supervisor identity changed')
    import fcntl
    lock = (receipt/'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='waiting_preflight', preflight=str(preflight), run_root=str(run))
    try:
        deadline = time.monotonic() + 3600
        write(receipt/'status.json', state)
        while True:
            status = read(preflight/'status.json')
            if status.get('status') == 'preflight_complete':
                break
            if status.get('status') not in ('prepared', 'running', 'waiting_resources'):
                raise RuntimeError('Physical preflight did not pass: '+json.dumps(status))
            if time.monotonic() > deadline:
                raise TimeoutError('Preflight did not finish within the registered one-hour wait')
            time.sleep(15)
        # Prevent a validated executable from silently becoming another design.
        source = preflight/'source'
        verified = {}
        for old in sorted(source.rglob('*')):
            if not old.is_file() or old.suffix not in ('.py', '.sh', '.yaml', '.yml', '.json'):
                continue
            relative = old.relative_to(source)
            live = repo/relative
            digest = hashlib.sha256(old.read_bytes()).hexdigest()
            if not live.is_file() or hashlib.sha256(live.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Executable changed after physical preflight: '+str(relative))
            verified[str(relative)] = digest
        if not verified:
            raise RuntimeError('Preflight source inventory is empty')
        write(receipt/'preflight_pass.json', dict(status=status, executable_hashes=verified,
            checked_at=time.time(), documents_may_include_later_measurement_results=True))
        if run.exists():
            raise FileExistsError('New run appeared while waiting; refusing duplicate launch')
        argv = ['bash', str(repo/'scripts/launch_predictive_feedback_20261009.sh'), str(run), 'all']
        state.update(status='launching', argv=argv)
        write(receipt/'status.json', state)
        result = subprocess.run(argv, cwd=repo, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=600)
        (receipt/'pipeline_launch.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'Pipeline launch exited {result.returncode}; no retry')
        child = read(run/'driver_pid.json')
        state.update(status='handed_off', pipeline=child, finished_at=time.time(),
            next_status=str(run/'status.json'), formal_training_started=False)
        write(receipt/'status.json', state)
    except BaseException as exc:
        state.update(status='stopped', error=dict(type=type(exc).__name__, message=str(exc)),
            finished_at=time.time())
        write(receipt/'status.json', state)
        raise


if __name__ == '__main__':
    main()
