"""One-shot fixed-rate step40000 joint-v3 acceptance and all700 evaluation; no retries."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

root = Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
repo = root / 'repo'
sys.path.insert(0, str(repo / 'scripts'))
from supervise_motionkv_training import atomic_json, source_files_and_hash, utc

binding = json.loads((root / 'launch40k_binding.json').read_text())
expected = binding['source_head']
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == binding['controller_sha256']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == expected
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip() == 'fix/route1-previous-plan-40k-eval-20261021'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
assert shutil.disk_usage(root).free >= 20 * 1024**3
directory = root / 'launch40k'
directory.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
    WAM_KERNEL_LOG_SOCKET='/run/route1-kernel-1024-stage3-20k.sock',
    PATH=str(repo / 'external/bin') + ':' + env.get('PATH', ''),
    TMPDIR='/SSD_DISK/users/wuruihan/tmp', XDG_CACHE_HOME=str(root / 'caches'))
auth_path = root / 'authorization40k.json'
auth = json.loads(auth_path.read_text())
gpus = ','.join(auth['gpu_uuids'])
python = str(repo / 'external/robotwin/bin/python')
config = repo / 'configs/motionkv_stage3_40k_eval_scaled_20261021.yaml'
status = dict(created_at=utc(), source_head=expected, simulation_rate=0.25,
    realtime_claim=False, stages=[], no_retry=True, protected_training_changed=False)

def run(name, command):
    entry = dict(stage=name, command=command, started_at=utc())
    status['stages'].append(entry)
    status['state'] = name + '_running'
    atomic_json(directory / 'status.json', status)
    with (directory / (name + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode, finished_at=utc())
    atomic_json(directory / 'status.json', status)
    if result.returncode:
        raise RuntimeError(name + ' failed; preserve evidence and do not retry')

def clean_guard(name):
    data = json.loads((root / name / 'status.json').read_text())
    assert data['final_exit_code'] == 0
    assert all(data[key] is True for key in
        ('guard_all_handles_closed', 'guard_workers_exited', 'driver_exit_clean'))

try:
    common = ['python3', str(repo / 'scripts/supervise_motionkv_eval.py')]
    run('preflight', common + ['--run-dir', str(root / 'preflight40k'), '--gpus', gpus,
        '--guard-gpus', gpus, '--authorization', str(auth_path), '--stage', 'smoke', '--preflight-only'])
    clean_guard('preflight40k')
    graph_command = json.loads((root / 'graph03/manifest.json').read_text())['command']
    assert Path(graph_command[1]).name == 'check_motionkv_graph.py'
    graph_command[1] = str(repo / 'scripts/check_motionkv_graph40k.py')
    for flag, value in [('--checkpoint', root / 'checkpoint40k/multirate_step_40000.pt'),
                        ('--output', root / 'graph40k/graph_parity.json'), ('--deploy-config', config)]:
        assert graph_command.count(flag) == 1
        graph_command[graph_command.index(flag) + 1] = str(value)
    run('graph', common + ['--run-dir', str(root / 'graph40k'), '--gpus', auth['gpu_uuids'][0],
        '--guard-gpus', gpus, '--authorization', str(auth_path), '--stage', 'graph',
        '--single-worker', '--max-runtime-seconds', '1800', '--'] + graph_command)
    clean_guard('graph40k')
    run('graph_acceptance', [python, str(repo / 'scripts/accept_joint_deployment_graph.py'),
        '--raw', str(root / 'graph40k/graph_parity.json'), '--guard-dir', str(root / 'graph40k'),
        '--checkpoint', str(root / 'checkpoint40k/multirate_step_40000.pt'), '--config', str(config),
        '--output', str(root / 'graph_receipt40k.json')])
    command = [python, str(repo / 'scripts/run_motionkv_joint_eval_pipeline.py')]
    values = {
        'repo': repo, 'run-root': root / 'outputs/joint40k_35x20_01_prevsource_scaled025',
        'storage-root': '/SSD_DISK/users/wuruihan/DynamicWAM', 'config': config,
        'checkpoint': root / 'checkpoint40k/multirate_step_40000.pt',
        'checkpoint-audit': root / 'checkpoint40k/checkpoint_audit.json',
        'timing-contract': root / 'checkpoint40k/timing_contract.json',
        'frozen-scenes': root / 'frozen700', 'authorization': auth_path,
        'preflight-guard': root / 'preflight40k', 'graph-guard': root / 'graph40k',
        'graph-receipt': root / 'graph_receipt40k.json',
        'scene-compatibility': root / 'scene_compatibility40k.json',
        'gpus': gpus, 'expected-head': expected, 'expected-source-sha256': source_files_and_hash(repo)[1],
        'simulation-rate': 0.25,
        'frozen-scenes-sha256': 'cff71ba1b8bf17c58082e615016e20775077787b455746e87261eaaaf60b6fb2',
        'smoke-tasks': 'put_bottles_dustbin,hanging_mug,handover_block,place_can_basket,'
                       'place_object_basket,put_object_cabinet,place_container_plate,shake_bottle_horizontally',
    }
    for key, value in values.items():
        command.extend(['--' + key, str(value)])
    command.append('--stop-kernel-broker-on-exit')
    run('pipeline', command)
    status['state'] = 'completed'
except BaseException as exc:
    status.update(state='failed', error=repr(exc))
    raise
finally:
    status['updated_at'] = utc()
    atomic_json(directory / 'status.json', status)
    if Path(env['WAM_KERNEL_LOG_SOCKET']).exists():
        subprocess.run(['python3', str(root / 'tools/eval_kernel_log.py'), '--stop',
            '--socket', env['WAM_KERNEL_LOG_SOCKET']], env=env, timeout=45)
