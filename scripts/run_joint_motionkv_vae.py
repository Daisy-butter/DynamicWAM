#!/usr/bin/env python3
"""One-shot guarded eight-shard VAE encoding; no watcher or formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_joint_motionkv_smokes import (
    ROOT, SSD, PROTECTED, require_release, require_fresh_authorization,
    frozen_identity, require_frozen,
)
from supervise_motionkv_training import atomic_json


def worker_commands(ids, grid, output, checkpoint, limit):
    if len(ids) != 8 or len(set(ids)) != 8 or any(not x.startswith('GPU-') for x in ids):
        raise ValueError('eight distinct GPU UUIDs required')
    if limit < 0:
        raise ValueError('episode limit must be nonnegative')
    result = []
    for rank, uuid in enumerate(ids):
        command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/encode_physical_grid_latents.py'),
                   '--physical-grid-root', str(grid), '--future-latent-root', str(output),
                   '--vae-checkpoint', str(checkpoint), '--rank', str(rank), '--world-size', '8',
                   '--device', 'cuda:0', '--guarded-launch']
        if limit:
            command += ['--episode-limit', str(limit)]
        result.append(dict(rank=rank, gpu_uuid=uuid, command=command))
    return result


def require_output_freshness(out, latent, limit):
    if out.exists():
        raise FileExistsError('preserve previous encoding receipts; choose a new output directory')
    if limit > 0 and latent.exists():
        raise FileExistsError('VAE GPU smoke requires a fresh latent directory; existing shards could skip all encoding')


def run_workers(plan_path):
    plan = json.loads(plan_path.read_text())
    # The outer approved supervisor sets this and owns worker cleanup/FD closure.
    if Path(os.environ.get('WAM_TRAINING_RUN_DIR', '/missing')).resolve() != Path(plan['guard']).resolve():
        raise RuntimeError('encoder children must run inside the selected device lifetime guard')
    require_frozen(plan['profiles'], plan['identity'])
    children, streams = [], []
    try:
        for worker in plan['workers']:
            stream = (plan_path.parent / ('rank%d.log' % worker['rank'])).open('x')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=worker['gpu_uuid'], OMP_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
            children.append(subprocess.Popen(worker['command'], cwd=ROOT, env=env,
                                             stdout=stream, stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in children]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError('encoder worker failed: ' + str(codes))
            if all(c == 0 for c in codes):
                return
            time.sleep(1)
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--gpu-uuids')
    p.add_argument('--authorization', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--physical-grid-root', type=Path)
    p.add_argument('--future-latent-root', type=Path)
    p.add_argument('--vae-checkpoint', type=Path)
    p.add_argument('--episode-limit', type=int, default=1,
                   help='1 per rank for initial smoke; explicit 0 encodes full dataset')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.worker_plan:
        run_workers(a.worker_plan)
        return
    if any(getattr(a, k) is None for k in ('gpu_uuids', 'authorization', 'output_dir',
            'physical_grid_root', 'future_latent_root', 'vae_checkpoint')):
        p.error('all GPU, authorization, data, checkpoint and output paths are required')
    out, latent = a.output_dir.resolve(), a.future_latent_root.resolve()
    if not out.is_relative_to(SSD.resolve()) or not latent.is_relative_to(SSD.resolve()):
        raise ValueError('all new artifacts must be on the project SSD')
    ids = a.gpu_uuids.split(',')
    workers = worker_commands(ids, a.physical_grid_root.resolve(), latent, a.vae_checkpoint.resolve(), a.episode_limit)
    guard = out / 'guard'
    command = [sys.executable, str(ROOT / 'scripts/supervise_motionkv_eval.py'), '--stage', 'prepare',
               '--gpus', a.gpu_uuids, '--guard-gpus', a.gpu_uuids, '--authorization', str(a.authorization.resolve()),
               '--run-dir', str(guard), '--startup-seconds', '300', '--max-runtime-seconds', '86400', '--',
               sys.executable, str(Path(__file__).resolve()), '--worker-plan', str(out / 'plan.json')]
    finalize = [sys.executable, str(ROOT / 'scripts/encode_physical_grid_latents.py'),
                '--physical-grid-root', str(a.physical_grid_root.resolve()),
                '--future-latent-root', str(latent), '--finalize']
    plan = dict(workers=workers, command=command, finalize=finalize if a.episode_limit == 0 else None,
                guard=str(guard), formal_training=False, automatic_retry=False,
                profiles=[dict(config=str(ROOT / 'configs/multirate/joint_motionkv_v3_warmup_smoke.yaml'))])
    if not a.execute:
        print(json.dumps(dict(executed=False, **plan), indent=2))
        return
    require_release(PROTECTED, require_success=False, label='protected evaluation')
    plan['authorization_sha256'] = require_fresh_authorization(a.authorization, ids)
    plan['identity'] = frozen_identity(plan['profiles'])
    require_output_freshness(out, latent, a.episode_limit)
    if not (a.physical_grid_root / 'manifest.json').is_file() or not a.vae_checkpoint.is_file():
        raise FileNotFoundError('audited grid manifest and official VAE are required')
    out.mkdir(parents=True)
    atomic_json(out / 'plan.json', plan)
    status = dict(state='running', complete_data=False, gpu_smoke_only=a.episode_limit > 0)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        require_release(guard / 'status.json', require_success=True, label='VAE encoding')
        require_frozen(plan['profiles'], plan['identity'])
        if plan['finalize']:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
            subprocess.run(finalize, cwd=ROOT, env=env, check=True)
            status['complete_data'] = True
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed_no_retry', error=repr(error))
        raise
    finally:
        atomic_json(out / 'status.json', status)


if __name__ == '__main__':
    main()
