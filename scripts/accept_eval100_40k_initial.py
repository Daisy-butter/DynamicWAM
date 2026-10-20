"""One requested initial-stability audit; no GPU polling or recurring watcher."""
from pathlib import Path
import datetime
import json
import subprocess
import sys

root=Path('/SSD_DISK/users/wuruihan/DynamicWAM/eval20k_20261019')
repo=root/'repo'; run=root/'outputs/joint40k_35x20_01_prevsource_scaled025'
read=lambda p:json.loads(p.read_text())
sys.path.insert(0,str(repo/'scripts'))
from eval_domino_motionkv_rtc import checked_episode
binding=read(root/'launch40k_binding.json')
auth=read(root/'authorization40k.json')
driver_anomaly_count=auth['driver']['anomaly_count']
status=read(run/'formal_guard/status.json')
pipeline=read(run/'pipeline/pipeline_status.json')
assert status['state']=='running' and status['startup_health_window_closed']
assert status['runtime_seconds']>=300 and status['stop_reason'] is None
assert pipeline['state']=='formal_running'
assert not list((run/'formal35x20').glob('*/error.json'))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head==binding['source_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
assert binding['checkpoint_step']==40000
assert binding['checkpoint_sha256']=='7e6f747e20241177eba442ba3886e31205df72dc754c1491d00e80998457bb4e'
rows=[]
for p in sorted((run/'formal35x20').glob('*/episode_0000/report.json')):
    d=read(p)
    assert d['clock_valid'] is True and d['continuous_valid'] is True and d['realtime_valid'] is False
    assert d['simulation_rate']==.25 and d['rows_dropped']==0
    assert d['checkpoint_sha256']==binding['checkpoint_sha256']
    checked_episode(p.parent,d['task'],d['episode_index'],.25,
        expected_protocol='continuous_world_scaled_wall_joint_motionkv_async_v3')
    rows.append(dict(task=d['task'],episode_index=d['episode_index'],success=d['success'],
        realtime_valid=False,max_backlog_s=d['max_backlog_s'],video=d['video'],
        video_frames=d['video_frames'],complete_validator_passed=True,native_video_sha256_verified=True))
assert len(rows)==8 and len({d['task'] for d in rows})==8
workers=[]
for p in sorted((run/'formal35x20/logs').glob('*.launch.json')):
    d=read(p);pid=d['pid'];proc=Path('/proc')/str(pid)
    assert proc.exists() and b'eval_domino_motionkv_rtc.py' in (proc/'cmdline').read_bytes()
    assert (proc/'stat').read_text().rsplit(')',1)[1].strip().split()[0] != 'Z'
    workers.append(dict(pid=pid,gpu=d['gpu'],task=p.name.replace('.launch.json',''),alive=True))
assert len(workers)==8 and len({w['gpu'] for w in workers})==8
for phase in read(run/'formal_guard/driver_phases.json'):
    if 'driver' in phase:
        assert phase['passed'] and phase['driver']['anomaly_count']==driver_anomaly_count and phase['delta']['anomaly_count']==0
assert read(run/'formal_guard/startup_driver.json')['anomaly_count']==driver_anomaly_count
smoke=read(run/'smoke_guard/status.json')
assert smoke['final_exit_code']==0 and all(smoke[k] for k in
    ('guard_all_handles_closed','guard_workers_exited','driver_exit_clean'))
assert read(run/'pipeline/smoke_video_decode.json')['complete'] is True
out=dict(schema_version=1,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    host='10.12.208.100',source_head=head,initial_stability_passed=True,
    full_evaluation_complete=False,formal_status=status,pipeline_pid=pipeline['pipeline_pid'],
    controller_pid=read(root/'launch40k_receipt.json')['controller_pid'],run_root=str(run),workers=workers,
    initial_reports=rows,simulation_rate=.25,realtime_claim=False,
    smoke_reports_accepted=16,smoke_videos_decoded=16,driver_anomaly_count_during_startup=driver_anomaly_count,
    driver_evidence_scope='Saved finite startup window; no extra live GPU or kernel query.',
    additional_inspection_stopped=True,recurring_monitor_created=False,
    necessary_suite_supervisor_retained=True,protected_training_changed=False)
with (root/'launch40k/initial_stability.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(dict(initial_stability_passed=True,first_reports=len(rows),alive_workers=len(workers),
    max_initial_backlog_s=max(d['max_backlog_s'] for d in rows),formal_runtime_s=status['runtime_seconds'])))
