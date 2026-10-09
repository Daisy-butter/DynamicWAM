#!/usr/bin/env python3
"""Durable real-scene gates: continuous mock -> replica RGB -> WAM, no training.

Run using the DOMINO Python. Launch in a detached process on the server; this
script, not an interactive chat, waits for every bounded subprocess. Failures
are durable and never trigger seed substitution or automatic retries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_identity(root):
    paths=[*root.glob('src/**/*.py'), *root.glob('scripts/*.py'),
           root/'configs/absolute_motion_v2.yaml',
           root/'configs/world_context_rt/controller_protocol.json']
    return {str(path.relative_to(root)):sha(path) for path in sorted(paths)}


def check_real_clock(summary, accepted_sha, *, model=False):
    if summary.get('status')!='complete' or summary.get('accepted_record',{}).get('sha256')!=accepted_sha:
        raise RuntimeError('real episode incomplete or accepted scene identity mismatch')
    report=summary['report']; controller=summary['controller']
    if (report.get('realtime_valid') is not True or report.get('asynchronous_camera') is not True or
        report.get('virtual_delay') is not False or report.get('startup_first_query_included') is not True or
        controller.get('camera_pipeline')!='independent_scene_snapshot_renderer' or
        controller.get('control_period_s')!=.1 or controller.get('physics_dt')!=.004 or
        controller.get('max_backlog_s')!=.1 or controller.get('max_tick_gap_s')!=.1 or
        report.get('camera_inflight_ticks',0)<=0 or report.get('inference_inflight_ticks',0)<=0):
        raise RuntimeError('continuous physics/camera/controller contract did not pass')
    for key in ('tick_lag_s','tick_gap_s'):
        values=report[key]
        if not values or not all(math.isfinite(float(v)) for v in values) or max(values)>.1:
            raise RuntimeError(f'real {key} exceeded unchanged 100ms gate')
    requests=report['requests']
    if not any(row.get('physics_advanced_during_worker_s',0)>0 for row in requests):
        raise RuntimeError('no physical advancement measured inside an actual worker interval')
    slow=[row for row in requests if row.get('worker_compute_s',0)>=.02]
    if not slow or any(row.get('physics_advanced_during_worker_s',0)<=0 for row in slow):
        raise RuntimeError('a long actual inference interval did not advance physics')
    if model and not any(row.get('metadata',{}).get('age_actions',0)>0 for row in requests):
        raise RuntimeError('WAM run never exercised cross-query plan reuse')
    return dict(physics_seconds=report['physics_time'],wall_seconds=report['wall_time'],
        max_tick_gap_s=max(report['tick_gap_s']),max_tick_lag_s=max(report['tick_lag_s']),
        camera_inflight_ticks=report['camera_inflight_ticks'],inference_inflight_ticks=report['inference_inflight_ticks'],
        actual_worker_intervals=len(requests),nonfresh_queries=sum(row.get('metadata',{}).get('age_actions',0)>0 for row in requests),
        executed_feedback_covered_rows=report.get('executed_feedback_covered_rows'),
        executed_nonzero_feedback_rows=report.get('executed_nonzero_feedback_rows'),
        expired_feedback_rows=report.get('expired_feedback_rows'),
        stop_reason=report['stop_reason'],success=report['success'])


def check_snapshot_events(path):
    count=0;max_roundoff=0.
    with Path(path).open() as stream:
        for line in stream:
            event=json.loads(line)
            if event.get('event')!='camera_ready':continue
            count+=1;oracle=event.get('snapshot_oracle',{})
            if oracle.get('passed') is not True or oracle.get('physics_steps')!=0:
                raise RuntimeError('camera frame lacks a successful no-step pose oracle')
            source=float(oracle['source_sim_time_seconds'])
            grid=float(event['observation_sim_time'])
            roundoff=abs(source-grid);max_roundoff=max(max_roundoff,roundoff)
            if roundoff>max(1e-7,abs(grid)*1e-7):
                raise RuntimeError('camera observation was retimestamped away from its physical snapshot')
    if not count:raise RuntimeError('no camera-ready snapshot identity evidence')
    return dict(verified_captures=count,max_engine_grid_roundoff_s=max_roundoff)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--accepted-record',type=Path,required=True)
    parser.add_argument('--policy-template',type=Path,required=True)
    parser.add_argument('--existing-mock-status',type=Path)
    parser.add_argument('--task',default='place_shoe')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--instruction-type',default='unseen')
    parser.add_argument('--phase-timeout',type=float,default=1800.)
    args=parser.parse_args()
    project=args.project_root.absolute(); run=args.run_root.absolute()
    if run.exists() and any(run.iterdir()):raise FileExistsError(f'preflight is immutable: {run}')
    run.mkdir(parents=True,exist_ok=True)
    executable=os.path.abspath(sys.executable)  # do not resolve the conda symlink
    if sys.version_info[:2]!=(3,10):raise RuntimeError('use the DOMINO Python 3.10 executable')
    accepted_sha=sha(args.accepted_record)
    source=source_identity(project)
    status=dict(format='dynamicwam_continuous_preflight',version=1,status='running',
        supervisor_pid=os.getpid(),started=time.time(),task=args.task,seed=args.seed,
        executable=executable,source_sha256=source,
        accepted_record=dict(path=str(args.accepted_record.absolute()),sha256=accepted_sha),phases={})
    status_path=run/'status.json';write(status_path,status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR=str(run/'tmp'),
             XDG_CACHE_HOME=str(run/'runtime_cache'),
             __GL_SHADER_DISK_CACHE_PATH=str(run/'runtime_cache'/'nvidia'))
    (run/'tmp').mkdir()
    (run/'runtime_cache'/'nvidia').mkdir(parents=True)
    status['runtime_environment']={key:env[key] for key in ('CUDA_VISIBLE_DEVICES','TMPDIR',
        'XDG_CACHE_HOME','__GL_SHADER_DISK_CACHE_PATH','OMP_NUM_THREADS','MKL_NUM_THREADS')}

    def idle_gpu():
        output=subprocess.check_output(['nvidia-smi','-i',args.gpu,
            '--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        if output.strip():raise RuntimeError(f'GPU preflight is contended: {output.strip()}')

    def unchanged():
        if source_identity(project)!=source:raise RuntimeError('source changed during preflight; evidence is not frozen')

    def execute(name,command):
        unchanged();idle_gpu()
        phase=dict(status='running',command=command,started=time.time())
        status['phases'][name]=phase;write(status_path,status)
        with (run/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase['child_pid']=child.pid;write(status_path,status)
            try:phase['exit_code']=child.wait(timeout=args.phase_timeout)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5.)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        phase['finished']=time.time();phase['status']='finished';write(status_path,status)
        unchanged()
        if phase['exit_code']!=0:raise RuntimeError(f'{name} failed with exit {phase["exit_code"]}; see its retained log')

    def episode_command(output,seconds):
        return [executable,str(project/'scripts/eval_domino_continuous.py'),'--config',
            str(project/'configs/absolute_motion_v2.yaml'),'--task',args.task,'--seed',str(args.seed),
            '--accepted-record',str(args.accepted_record.absolute()),'--output',str(output),
            '--physics-timeout',str(seconds)]

    try:
        if args.existing_mock_status:
            deadline=time.monotonic()+args.phase_timeout
            while True:
                existing=read(args.existing_mock_status)
                if existing.get('status')!='running':break
                if time.monotonic()>deadline:raise TimeoutError('existing mock did not complete by deadline')
                time.sleep(5.)
            if existing.get('passed') is not True or existing.get('phase')!='mock_replica_real_scene':
                raise RuntimeError(f'existing real mock failed: {existing}')
            summary_path=Path(existing['summary']);summary=read(summary_path)
            for name,digest in summary['source_sha256'].items():
                if source.get(name)!=digest:raise RuntimeError(f'existing mock used different source: {name}')
            status['phases']['mock']=dict(status='passed',reused=True,
                supervisor_status=dict(path=str(args.existing_mock_status.absolute()),sha256=sha(args.existing_mock_status)),
                summary=dict(path=str(summary_path),sha256=sha(summary_path)),
                evidence=check_real_clock(summary,accepted_sha),
                snapshot_evidence=check_snapshot_events(summary_path.parent/'events.jsonl'))
        else:
            output=run/'mock'
            execute('mock',episode_command(output,2.)+['--policy-factory',
                'dynamicwam.runtime.domino_rt.mock_worker:create_mock_policy'])
            summary_path=output/'summary.json'
            status['phases']['mock'].update(status='passed',summary=dict(path=str(summary_path),sha256=sha(summary_path)),
                evidence=check_real_clock(read(summary_path),accepted_sha),
                snapshot_evidence=check_snapshot_events(summary_path.parent/'events.jsonl'))
        write(status_path,status)

        rgb_path=run/'replica_rgb.json'
        execute('replica_rgb',[executable,str(project/'scripts/check_scene_replica.py'),
            '--config',str(project/'configs/absolute_motion_v2.yaml'),'--task',args.task,'--seed',str(args.seed),
            '--instruction-type',args.instruction_type,'--accepted-record',str(args.accepted_record.absolute()),
            '--output',str(rgb_path)])
        rgb=read(rgb_path)
        if rgb.get('passed') is not True or rgb.get('accepted_record',{}).get('sha256')!=accepted_sha:
            raise RuntimeError('replica RGB/pose repeatability oracle did not pass')
        status['phases']['replica_rgb'].update(status='passed',report=dict(path=str(rgb_path),sha256=sha(rgb_path)))
        write(status_path,status)

        policy=read(args.policy_template);policy.update(mode='graph_zero',project_root=str(project),
            deploy_config=str(project/'configs/absolute_motion_v2.yaml'),
            controller_protocol_sha256=sha(project/'configs/world_context_rt/controller_protocol.json'),
            collection_dir=str(run/'dev_collection'),task=args.task,seed=args.seed,samples_per_episode=64)
        policy.pop('feedback_checkpoint',None)
        policy_path=run/'graph_zero_policy.json';write(policy_path,policy)
        output=run/'graph_zero'
        execute('graph_zero',episode_command(output,10.)+['--policy-config',str(policy_path)])
        summary_path=output/'summary.json';summary=read(summary_path)
        if summary['controller_protocol_sha256']!=policy['controller_protocol_sha256']:
            raise RuntimeError('WAM controller protocol identity changed')
        status['phases']['graph_zero'].update(status='passed',summary=dict(path=str(summary_path),sha256=sha(summary_path)),
            evidence=check_real_clock(summary,accepted_sha,model=True),
            snapshot_evidence=check_snapshot_events(summary_path.parent/'events.jsonl'),
            policy=dict(path=str(policy_path),sha256=sha(policy_path)))
        unchanged();status.update(status='passed',passed=True)
    except BaseException:
        status.update(status='stopped_failed_gate',passed=False,error=traceback.format_exc())
    finally:
        status['finished']=time.time();write(status_path,status)
    print(json.dumps(dict(status=status['status'],status_path=str(status_path))))
    return 0 if status.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
