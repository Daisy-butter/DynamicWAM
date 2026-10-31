"""Eight native synchronous workers; one GPU/environment; finite/no retry."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from native_v7_eval_common import *


def audit_task(c,task):
    from video_artifacts import validate_video_artifacts
    out=Path(c['output']);done=read(out/task/'completed.json')
    if (done.get('complete') is not True or done['episodes']!=c['episodes'] or done['task']!=task
            or done['protocol']!=PROTOCOL or done['config_sha256']!=base.canonical_hash(c)):
        raise ValueError('task completion differs')
    reports=[];metrics=[]
    for episode in range(c['episodes']):
        d=out/task/f'episode_{episode:04d}';r=read(d/'report.json');m=read(d/'metrics.json');scene=read(d/'manifest.json')
        if (r['protocol']!=PROTOCOL or r['task']!=task or r['episode_index']!=episode
                or r['checkpoint_sha256']!=c['checkpoint_sha256'] or r['config_sha256']!=base.canonical_hash(c)
                or r['seed']!=scene['seed'] or m['seed']!=r['seed'] or not r['autonomous_fast_only']
                or r['full_driven_integration'] or not r['native_motion_unmodified']
                or not r['inference_physics_frozen'] or not r['graph_acceptance']['passed']):
            raise ValueError('episode identity/contract changed')
        if audit_events(r)!=r['clock_audit'] or not validate_video_artifacts(r):raise ValueError('clock/video audit failed')
        reports.append(r);metrics.append(m)
    native=Path(done['native_result_dir'])
    if len(list(native.glob('episode*.mp4')))!=c['episodes']:raise ValueError('native MP4 count differs')
    for name in ('_result.txt','_metrics.json','_metrics_report.txt','_episodes_detail.json'):
        if not (native/name).is_file():raise ValueError('missing native result '+name)
    return reports,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--authorization',required=True);a=p.parse_args()
    c=read(a.config);out=validate_config(c);auth=authorization(c,a.authorization);verify_artifacts(c)
    from dualrate_eval_worker import load_frozen_task
    for task in c['tasks']:load_frozen_task(c,task)
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    claim=dict(**base.proc_identity(os.getpid()),started_unix=time.time(),receipt_sha256=sha(Path(a.authorization)),config_sha256=base.canonical_hash(c))
    if claim['started_unix']>=auth['expires_at_unix']:raise ValueError('suite preparation exceeded authorization')
    path=Path(a.authorization).with_suffix('.json.started')
    with path.open('x') as f:json.dump(claim,f)
    save(out/'suite_started.json',dict(protocol=PROTOCOL,tasks=c['tasks'],episodes_per_task=c['episodes'],
        checkpoint_sha256=c['checkpoint_sha256'],source=auth['source'],resources='8 environments x 1 GPU',no_retry=True))
    active=[];pending=list(c['tasks']);finished=[];reports=[];metrics=[]
    def interrupted(sig,_):raise InterruptedError(sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    try:
        while pending or active:
            free=sorted(set(range(8))-{r['index'] for r in active})
            for index in free:
                if not pending:break
                task=pending.pop(0);logpath=out/'logs'/f'{task}.log';logpath.parent.mkdir(exist_ok=True)
                log=logpath.open('x');cache=out/'cache'/task;cache.mkdir(parents=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['gpu_uuids'][index],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
                cmd=[sys.executable,'-B',str(ROOT/'scripts/native_v7_eval_worker.py'),'--config',a.config,
                     '--authorization',a.authorization,'--task',task,'--gpu-index',str(index)]
                try:child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
                except BaseException:log.close();raise
                active.append(dict(task=task,index=index,process=child,log=log,start=time.monotonic()))
            save(out/'suite_status.json',dict(state='running',finished_tasks=finished,pending_tasks=pending,
                running=[dict(task=r['task'],pid=r['process'].pid,gpu_uuid=c['gpu_uuids'][r['index']]) for r in active],no_retry=True))
            for row in list(active):
                code=row['process'].poll()
                if code is None:
                    if time.monotonic()-row['start']>c['task_timeout_seconds']:raise TimeoutError(row['task'])
                    continue
                row['log'].close()
                if code:raise RuntimeError(f"native worker {row['task']} exited {code}; no retry")
                rr,mm=audit_task(c,row['task']);reports.extend(rr);metrics.extend(mm);finished.append(row['task']);active.remove(row)
            if active:time.sleep(2)
        expected=len(c['tasks'])*c['episodes']
        if len(reports)!=expected:raise ValueError('incomplete evaluation')
        reuses=sum(r['clock_audit']['reference_reuses'] for r in reports)
        refreshes=sum(r['clock_audit']['reference_refreshes'] for r in reports)
        if c['stage']=='smoke' and (reuses==0 or refreshes<=expected):
            raise ValueError('autonomous smoke did not demonstrate slow reference reuse/refresh')
        sys.path.insert(0,str(Path(c['domino_root'])/'script'))
        from video_artifacts import aggregate_native_metrics
        summary=aggregate_native_metrics(metrics).get_summary()
        summary={k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in summary.items()}
        verify_artifacts(c)
        refs=[dict(path=str(out/r['task']/f"episode_{r['episode_index']:04d}/report.json"),
            sha256=sha(out/r['task']/f"episode_{r['episode_index']:04d}/report.json")) for r in reports]
        save(out/'suite_completed.json',dict(complete=True,protocol=PROTOCOL,stage=c['stage'],source=auth['source'],
            checkpoint_sha256=c['checkpoint_sha256'],config_sha256=base.canonical_hash(c),valid_completed=expected,
            successes=sum(r['success'] for r in reports),success_rate_percent=100*sum(r['success'] for r in reports)/expected,
            native_metrics=summary,native_mp4_full_decode=True,all_workers_exit_zero=True,
            reference_reuses=reuses,reference_refreshes=refreshes,reports=refs,elapsed_seconds=time.monotonic()-began,no_retry=True))
        save(out/'suite_status.json',dict(state='completed',valid_completed=expected,no_retry=True))
        save(Path(a.authorization).with_suffix('.json.consumed'),dict(receipt_sha256=sha(Path(a.authorization)),output=str(out)))
    except BaseException as exc:
        save(out/'suite_status.json',dict(state='failed',error=repr(exc),traceback=traceback.format_exc(),finished_tasks=finished,no_retry=True));raise
    finally:
        for row in active:
            if row['process'].poll() is None:row['process'].terminate()
        deadline=time.monotonic()+15
        for row in active:
            try:row['process'].wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass  # Owning guard retains descriptors and reaps entire SID.
            row['log'].close()


if __name__=='__main__':main()
