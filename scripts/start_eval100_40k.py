"""Launch the authorized 40k/previous-source evaluation once; no observer/retry."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
BRANCH = 'fix/route1-previous-plan-40k-eval-20261021'
CHECKPOINT_SHA = '7e6f747e20241177eba442ba3886e31205df72dc754c1491d00e80998457bb4e'
RUN_NAME = 'joint40k_35x20_01_prevsource_scaled025'
DRIVER_KEYS = ('host', 'boot_id', 'driver_anomalies_sha256',
               'latest_driver_anomaly_sha256', 'anomaly_count')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def main():
    root, repo = ROOT, ROOT / 'repo'
    os.environ['WAM_KERNEL_LOG_SOCKET'] = '/run/route1-kernel-1024-stage3-20k.sock'
    sys.path.insert(0, str(root / 'tools'))
    from eval_kernel_log import driver_evidence
    for name in ('launch40k', 'launch40k_receipt.json', 'launch40k_binding.json',
                 'authorization40k.json'):
        assert not (root / name).exists(), 'Never overwrite or retry: ' + name
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert subprocess.check_output(['git', 'branch', '--show-current'],
        cwd=repo, text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'status', '--porcelain'],
        cwd=repo, text=True).strip()
    oldrun = root / 'outputs/joint30k_35x20_01_scaled025'
    oldpipeline = read(oldrun / 'pipeline/pipeline_status.json')
    assert oldpipeline['state'] in ('complete', 'completed')
    assert oldpipeline['valid_completed'] == 700 and oldpipeline['exit_code'] == 0
    assert read(oldrun / 'pipeline/formal_video_decode.json')['complete'] is True
    oldguard = read(oldrun / 'formal_guard/status.json')
    assert oldguard['final_exit_code'] == 0 and not oldguard.get('stop_reason')
    assert all(oldguard[key] is True for key in
        ('guard_workers_exited', 'guard_all_handles_closed', 'driver_exit_clean'))
    driver, oldauth = driver_evidence(), read(root / 'authorization30k.json')
    for key in DRIVER_KEYS:
        assert driver[key] == oldauth['driver'][key], 'Fresh driver evidence differs: ' + key
    assert driver['anomaly_count'] == 505
    audit = read(root / 'checkpoint40k/checkpoint_audit.json')
    assert audit['passed'] and audit['step'] == 40000
    assert audit['strict_full_reload'] and audit['strict_fast_reload']
    assert audit['full']['sha256'] == CHECKPOINT_SHA
    config = repo / 'configs/motionkv_stage3_40k_eval_scaled_20261021.yaml'
    compat = read(root / 'scene_compatibility40k.json')
    assert compat['passed'] and compat['evaluation_config_sha256'] == sha(config)
    controller = repo / 'scripts/run_eval100_40k.py'
    binding = dict(source_head=head, source_branch=BRANCH, controller_sha256=sha(controller),
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=40000, simulation_rate=.25,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_root=str(root / 'outputs' / RUN_NAME), previous30k_preserved=True,
        previous_plan_contract='aligned_source_plan_completed_false_training_fallback',
        initial_observer_started_by_launcher=False)
    save(root / 'launch40k_binding.json', binding)
    auth = dict(oldauth, driver=driver, created_at=binding['created_at'], source_head=head,
        reason='User authorized one-shot40k evaluation after completed training and all700 prior30k episodes. Only soft previous_plan is aligned to early source-plan training semantics; fresh actual40k Graph and8GPU full-episode smoke gate formal700. No driver recovery or retry.',
        user_instruction='After current training/evaluation finish, fix previous_plan and run40k on .100 with all other settings unchanged.',
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=40000, simulation_rate=.25)
    save(root / 'authorization40k.json', auth)
    with (root / 'launch40k.log').open('x') as log:
        child = subprocess.Popen(['python3', str(controller)], cwd=repo,
            env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(binding, controller_pid=child.pid, status='controller_launched',
        formal_started=False, fresh_anomaly_count=driver['anomaly_count'],
        authorization=str(root / 'authorization40k.json'))
    save(root / 'launch40k_receipt.json', receipt)
    for _ in range(50):
        if (root / 'launch40k/status.json').exists():
            break
        assert child.poll() is None, 'Controller stopped before status creation'
        time.sleep(.2)
    assert (root / 'launch40k/status.json').exists()
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
