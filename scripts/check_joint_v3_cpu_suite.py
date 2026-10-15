#!/usr/bin/env python3
"""Bounded CPU integration suite. Never allocates CUDA or launches training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/SSD_DISK/users/wuruihan/DynamicWAM').resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-test', action='append', default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(SSD):
        raise ValueError('receipts must stay on project SSD')
    if output.exists():
        raise FileExistsError('preserve prior test attempts; use a fresh receipt')
    programs = [
        'check_joint_motion_kv_conditioner.py', 'check_joint_motionkv_training_cpu.py',
        'check_joint_motionkv_v3_training_cpu.py', 'check_joint_deployment_cpu.py',
        'check_rtc_prefix.py', 'check_motionkv_training_cpu.py',
        'check_motion_kv_runtime_contract.py', 'verify_multirate_training.py',
        'check_joint_source_prefix_cpu.py',
    ]
    tests = ['test_v3_rtc_queue.py', 'test_async_motionkv_policy.py',
             'test_motionkv_rtc_controller.py', 'test_joint_smoke_launcher.py',
             'test_joint_async_data.py', 'test_physical_grid_data.py',
             'test_v3_deployment_contract.py', 'test_v3_vae_launcher.py',
             'test_v3_smoke_plans.py', 'test_real_rollout_dataset.py',
             'test_development_replay.py', *args.extra_test]
    commands = [[str(ROOT / 'scripts' / p)] for p in programs]
    commands += [['-m', 'unittest', 'discover', '-s', 'tests', '-p', p] for p in tests]
    for p in [ROOT / 'scripts' / p for p in programs] + [ROOT / 'tests' / p for p in tests]:
        if not p.is_file():
            raise FileNotFoundError(p)
    def identity():
        digest = hashlib.sha256()
        for folder in ('src', 'scripts', 'tests', 'configs'):
            for path in sorted((ROOT / folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py', '.yaml', '.yml') and '__pycache__' not in path.parts:
                    digest.update(str(path.relative_to(ROOT)).encode())
                    digest.update(path.read_bytes())
        return digest.hexdigest()
    initial = identity()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONPATH=str(ROOT / 'src'))
    receipt = dict(device='cpu', gpu_allocations=False, passed=False, cases=[],
                   source_sha256=initial, full_video_backward=False, cuda_capture=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    for command in commands:
        start = time.monotonic()
        try:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=180)
            case = dict(command=command, exit_code=result.returncode,
                        stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as error:
            case = dict(command=command, exit_code=124, stderr=str(error), stdout='')
        case['seconds'] = round(time.monotonic() - start, 3)
        receipt['cases'].append(case)
        print(json.dumps(dict(case=command[-1], exit_code=case['exit_code'], seconds=case['seconds'])), flush=True)
        output.write_text(json.dumps(receipt, indent=2))
    receipt['source_unchanged'] = identity() == initial
    receipt['passed'] = receipt['source_unchanged'] and all(c['exit_code'] == 0 for c in receipt['cases'])
    output.write_text(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        for case in receipt['cases']:
            if case['exit_code']:
                print(case['stderr'][-3000:])
        raise SystemExit(1)
    print(json.dumps(dict(passed=True, programs=len(commands), receipt=str(output))))


if __name__ == '__main__':
    main()
