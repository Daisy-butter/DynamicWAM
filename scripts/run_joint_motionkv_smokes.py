#!/usr/bin/env python3
"""Plan or execute two guarded eight-GPU smokes, never formal training.

Use fresh read-only health evidence after the protected evaluation releases all
GPUs. The existing FD-before-telemetry supervisor owns every CUDA child. No
waiting loop, automatic retry, driver recovery, or formal launch is installed.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import yaml

from supervise_motionkv_training import configure_nccl_transport, source_files_and_hash, file_digest, atomic_json

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/SSD_DISK/users/wuruihan/DynamicWAM')
PROTECTED = SSD / 'outputs/evaluation/motionkv_scaled_r025_20261015/formal_guard/status.json'

def training_environment(inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    # Exactly the previously validated default transport/automatic allocator.
    # Preserve and record any other explicit NCCL settings; do not invent a backend.
    nccl = configure_nccl_transport(env, 'default', 'auto')
    env.update(ACCELERATE_USE_DEEPSPEED='true', ACCELERATE_MIXED_PRECISION='bf16',
               PYTHONDONTWRITEBYTECODE='1')
    return env, nccl


def commands(ids, authorization, out, inherited=None, recipe='v2'):
    if recipe not in ('v2', 'v3', 'v3-replay'):
        raise ValueError('unknown joint recipe')
    python = str(ROOT / '.venv/bin/python')
    inherited = dict(os.environ if inherited is None else inherited)
    result = []
    phases = ('replay',) if recipe == 'v3-replay' else ('warmup', 'joint')
    for phase in phases:
        stem = 'joint_motionkv_v3' if recipe.startswith('v3') else 'joint_motionkv'
        config = ROOT / f'configs/multirate/{stem}_{phase}_smoke.yaml'
        profile = yaml.safe_load(config.read_text())
        if profile['training']['max_steps'] != 4 or profile['training'].get('resume_from'):
            raise ValueError('this launcher only accepts fresh four-step smoke profiles')
        training_run = Path(profile['paths']['training_runs']) / profile['training']['name']
        if not training_run.resolve().is_relative_to(SSD.resolve()):
            raise ValueError('training outputs must stay on the project SSD')
        guard = out / (phase + '_guard')
        command = [python, str(ROOT / 'scripts/supervise_motionkv_eval.py'),
            '--stage', 'smoke', '--run-dir', str(guard), '--gpus', ids,
            '--guard-gpus', ids, '--authorization', str(authorization),
            '--startup-seconds', '300', '--max-runtime-seconds', '7200', '--',
            '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=' + ids,
            'LD_LIBRARY_PATH=' + inherited.get('LD_LIBRARY_PATH', ''),
            'ACCELERATE_MIXED_PRECISION=bf16', 'ACCELERATE_USE_DEEPSPEED=true',
            'PYTHONDONTWRITEBYTECODE=1',
            python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
            str(ROOT / 'scripts/train_multirate.py'), '--config', str(config)]
        exports = training_run / 'exports'
        audit = [python, str(ROOT / 'scripts/check_motionkv_smoke_export.py'),
            '--config', str(config), '--full', str(exports / 'multirate_step_4.pt'),
            '--fast', str(exports / 'multirate_fast_step_4.pt'),
            '--log', str(guard / 'run.log'), '--expected-steps', '4',
            '--output', str(out / (phase + '_export_audit.json'))]
        result.append(dict(phase=phase, command=command, audit=audit, guard=str(guard),
            config=str(config), training_run=str(training_run),
            initialization=profile['paths']['init_checkpoint'],
            full_export=str(exports / 'multirate_step_4.pt')))
    if len(result) == 2 and Path(result[1]['initialization']).resolve() != Path(result[0]['full_export']).resolve():
        raise ValueError('joint smoke must initialize exactly the newly audited warmup export')
    return result


def require_release(path, *, require_success, label):
    if not path.is_file():
        raise RuntimeError(label + ': final release receipt is missing')
    value = json.loads(path.read_text())
    if (not value.get('ended_at') or value.get('final_exit_code') is None
            or value.get('driver_exit_clean') is not True
            or value.get('guard_workers_exited') is not True
            or value.get('guard_all_handles_closed') is not True):
        raise RuntimeError(label + ': workers and all temporary GPU handles have not cleanly closed')
    if require_success and (value['final_exit_code'] != 0 or value.get('stop_reason')):
        raise RuntimeError(label + ': smoke failed; later stages are forbidden')
    return value


def require_fresh_authorization(path, ids):
    if not path.is_file() or not 0 <= time.time() - path.stat().st_mtime <= 1800:
        raise RuntimeError('fresh guarded health authorization required; old evaluation clearance is not reusable')
    value = json.loads(path.read_text())
    if set(value.get('gpu_uuids', [])) != set(ids) or value.get('device_lifetime_mode') != 'fds_before_telemetry_v1':
        raise RuntimeError('authorization must bind all eight selected GPUs and FD-before-telemetry mode')
    return file_digest(path)


def frozen_identity(plan):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('freeze and commit the candidate before GPU smoke')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources, source_sha = source_files_and_hash(ROOT)
    paths = [Path(stage['config']) for stage in plan] + [ROOT / 'harness.md', ROOT / 'AGENTS.md']
    return dict(git_head=head, source_sha256=source_sha, source_files=sources,
                profile_and_instruction_sha256={str(path): file_digest(path) for path in paths})


def require_frozen(plan, identity):
    if frozen_identity(plan) != identity:
        raise RuntimeError('candidate source/config changed between smoke phases')


def require_fresh_outputs(plan, out):
    if out.exists() or any(Path(stage['training_run']).exists() for stage in plan):
        raise FileExistsError('fresh supervisor and training output directories required; preserve prior attempts')
    if not Path(plan[0]['initialization']).is_file():
        raise FileNotFoundError('warmup initialization checkpoint is missing')


def require_recipe_data(plan, recipe):
    if not recipe.startswith('v3'):
        return None
    sys.path.insert(0, str(ROOT / 'src'))
    from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
    profiles = [yaml.safe_load(Path(stage['config']).read_text()) for stage in plan]
    pairing = profiles[0]['pairing']
    roots = (pairing['physical_grid_root'], pairing['future_latent_root'])
    for profile in profiles[1:]:
        if (profile['pairing']['physical_grid_root'], profile['pairing']['future_latent_root']) != roots:
            raise ValueError('warmup/joint must use the same audited physical-time dataset')
    # A complete manifest, all shard identities and coverage are verified here;
    # a readiness boolean or a synthetic cached sample cannot release GPU smoke.
    result = dict(physical=validate_physical_grid_readiness(*roots, verify_hashes=True, require_full=True))
    if recipe == 'v3-replay':
        from dynamicwam.training.data.real_rollout_dataset import validate_replay_manifest
        _, _, replay = validate_replay_manifest(pairing['replay_root'], roots[0],
            profiles[0]['paths']['packed_dataset'], strict_provenance=True)
        result['real_replay'] = replay
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuids', required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--recipe', choices=('v2', 'v3', 'v3-replay'), default='v2')
    args = parser.parse_args()
    ids = args.gpu_uuids.split(',')
    if len(ids) != 8 or len(set(ids)) != 8 or any(not x.startswith('GPU-') for x in ids):
        raise ValueError('exactly eight distinct UUIDs required')
    out = args.output_dir.resolve()
    if not out.is_relative_to(SSD.resolve()):
        raise ValueError('smoke outputs must be on verified project SSD')
    env, nccl = training_environment()
    plan = commands(args.gpu_uuids, args.authorization.resolve(), out, env, args.recipe)
    if not args.execute:
        print(json.dumps(dict(executed=False, stages=plan, nccl_environment=nccl,
            formal_training=False, protected_release_receipt=str(PROTECTED)), indent=2))
        return
    # A failed prior evaluation can release resources cleanly; its failure is
    # preserved. New smoke stages themselves must exit zero before progression.
    require_release(PROTECTED, require_success=False, label='protected evaluation')
    authorization_sha = require_fresh_authorization(args.authorization, ids)
    identity = frozen_identity(plan)
    data_receipt = require_recipe_data(plan, args.recipe)
    require_fresh_outputs(plan, out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(state='running', stages=plan, completed=[], retry_policy='none', formal_training=False,
        frozen_source=identity, authorization_sha256=authorization_sha, nccl_environment=nccl,
        recipe=args.recipe, data_readiness=data_receipt)
    status = out / 'status.json'
    try:
        for stage in plan:
            require_frozen(plan, identity)
            if file_digest(args.authorization) != authorization_sha:
                raise RuntimeError('authorization changed during the two-stage smoke')
            atomic_json(status, receipt)
            subprocess.run(stage['command'], cwd=ROOT, env=env, check=True)
            require_release(Path(stage['guard']) / 'status.json', require_success=True, label=stage['phase'])
            require_frozen(plan, identity)
            # Schema/reload audit runs on CPU only after all device FDs close.
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
            subprocess.run(stage['audit'], cwd=ROOT, env=cpu_env, check=True)
            audit_path = out / (stage['phase'] + '_export_audit.json')
            if json.loads(audit_path.read_text()).get('passed') is not True:
                raise RuntimeError('CPU export audit did not accept smoke artifacts')
            receipt['completed'].append(stage['phase'])
        require_frozen(plan, identity)
        receipt['state'] = 'smokes_passed_no_formal_training'
    except BaseException as error:
        receipt.update(state='failed_no_retry', error=repr(error))
        raise
    finally:
        atomic_json(status, receipt)

if __name__ == '__main__':
    main()
