#!/usr/bin/env python3
"""Official-initialized compact student: stage2 -> stage3 -> paired Level-1 suites.

Run from an immutable source snapshot. A failed stage stops the chain; resume
training explicitly from its recorded optimizer checkpoint, not by relaunching
this script into the same directory. No process outside this pipeline is killed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamicwam.config.multirate import load_multirate_profile

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)

def verify_export(path, expected_step):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert payload['global_step'] == expected_step, path
    assert payload['modules']['architecture'] == 'compact', path
    contract = payload['config']['runtime_contract']
    assert contract['supervised_action_horizon'] == 16
    assert contract['max_age_seconds'] == .72
    weights = [(k,v) for k,v in payload['model'].items() if k.startswith('fast_policy.')]
    assert weights and all(torch.isfinite(v).all() for _,v in weights)
    assert not any('native_teacher.' in k for k in payload['model'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'step': expected_step,
            'runtime_contract': contract}

def eval_summary(arms):
    """Report pairing and actual timing separately; never infer speed from success."""
    result, manifests = {}, {}
    for name, root in arms.items():
        aggregate = json.loads((root/'aggregate.json').read_text())
        manifest = {}
        timings, modes = [], Counter()
        for task in aggregate['tasks']:
            data = json.loads(Path(task['aggregate']).read_text())
            manifest[task['task']] = [e['seed'] for e in data['episodes']]
        for log in (root/'tasks').rglob('slot*.log'):
            for line in log.read_text(errors='replace').splitlines():
                for prefix in ('SYNC-QUERY-TIMING ', 'MULTIRATE-QUERY-TIMING ', 'MULTIRATE-QUERY-DETAIL '):
                    if prefix not in line: continue
                    try: row = ast.literal_eval(line.split(prefix,1)[1])
                    except (ValueError, SyntaxError): continue
                    if not isinstance(row,dict): continue
                    if prefix.endswith('DETAIL '): modes[str(row.get('bridge_mode'))] += 1
                    else: timings.append(row)
        result[name] = {'aggregate': str(root/'aggregate.json'), 'success_count': aggregate['success_count'],
                        'requested_episodes': aggregate['requested_episodes'], 'success_rate': aggregate['success_rate'],
                        'success_rate_ci': aggregate.get('success_rate_ci'), 'timing_rows': timings,
                        'bridge_modes': dict(modes)}
        manifests[name] = manifest
    reference = next(iter(manifests.values()))
    return {'arms': result, 'episode_seeds_match': all(m == reference for m in manifests.values()),
            'episode_manifests': manifests, 'protocol': 'native DOMINO Level-1, physics paused during query; not an RTC experiment'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--episodes-per-task', type=int, default=20)
    parser.add_argument('--port', type=int, default=36261)
    args = parser.parse_args()
    root = args.run_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive sentinel prevents accidental duplicate launch into existing results.
    with (root/'pipeline.started').open('x') as stream: stream.write(str(os.getpid()))
    state = {'status': 'running', 'pid': os.getpid(), 'source': str(ROOT), 'completed': [], 'started': time.time()}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    def save():
        state['updated'] = time.time(); write_json(root/'status.json',state)
    def run(name, command):
        state.update(phase=name, command=command, log=str(root/f'{name}.log')); save()
        with (root/f'{name}.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid; save()
            code = process.wait()
        state.pop('child_pid',None)
        if code: raise RuntimeError(f'{name} exited {code}; see {root/name}.log')
        state['completed'].append(name); save()
    try:
        exports = {}
        for stage in (2,3):
            config = ROOT/'configs'/f'official_temporal_stage{stage}_7gpu.yaml'
            profile = load_multirate_profile(config)
            train = profile.training
            command = [str(ROOT/'.venv/bin/accelerate'), 'launch', '--multi_gpu', '--num_processes',
                       str(len(args.gpus.split(','))), '--num_machines','1','--mixed_precision','bf16',
                       '--main_process_port',str(args.port),'--dynamo_backend','no',
                       str(ROOT/'scripts/train_multirate.py'), '--config',str(config)]
            run(f'stage{stage}',command)
            export = Path(train['output_dir']) if 'output_dir' in train else None
            from dynamicwam.training.train.common import get_export_dir
            export = get_export_dir(train)/f"multirate_step_{train['max_steps']}.pt"
            state[f'stage{stage}_export'] = verify_export(export,train['max_steps']); save()
            exports[stage] = export
        arms = {}
        # Preserve original full as both control arms; never overwrite its weights.
        variants = [('official_sync_h16',None,16), ('official_sync_h4',None,4),
                    ('stage2_compact_h4',2,4), ('stage3_compact_h4',3,4)]
        for name, stage, horizon in variants:
            override = dict(policy_name='ciwam.adapters.domino.deploy_policy_sync_flow',
                            ckpt_setting=name,execute_horizon=horizon,eval_video_log=False)
            if stage:
                override.update(policy_name='ciwam.adapters.domino.deploy_policy_multirate_flow',
                                multirate_checkpoint=str(exports[stage]),bridge_refresh_ratio=2,
                                multirate_scheduling='independent_worker',multirate_reuse_action_noise=True)
            override_path = root/'eval_configs'/f'{name}.json';write_json(override_path,override)
            arm_root = root/'eval'/name
            run(name,[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/eval_domino.py'),
                      '--config',str(ROOT/'configs/absolute_motion_v2.yaml'),
                      '--policy-overrides',str(override_path),'--gpus',args.gpus,
                      '--run-root',str(arm_root),'--episodes-per-task',str(args.episodes_per_task),
                      '--start-seed','100000'])
            arms[name] = arm_root
            write_json(root/'evaluation_summary.json',eval_summary(arms))
        state.update(status='complete', finished=time.time());save()
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
