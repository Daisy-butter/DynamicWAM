"""Explicit launch: wait for completed V7 -> 8x2 native smoke -> native 35x20.

CPU-only waiter; no CUDA imports before the existing guarded worker starts.
No automatic retry, training interruption, or driver recovery.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import guarded_dualrate_run as guard
from native_v7_eval_common import *
from dualrate_posttrain_eval import training_ready,require_clean_guard


def preflight(spec):
    if spec['expected_training_commit']!=TRAIN_COMMIT or spec['expected_training_source_sha256']!=TRAIN_SHA:
        raise ValueError('wrong training sequence')
    c=read(spec['template'])
    if sha(Path(spec['template']))!=spec['template_sha256']:raise ValueError('template changed')
    validate_config(c,bound=False);verify_artifacts(c)
    if c['training_config']!=str(Path(spec['training_sequence_dir'])/'cross_time_joint.config.json'):
        raise ValueError('wrong formal config')
    if sha(Path(c['training_config']))!=spec['training_config_sha256']:raise ValueError('training config changed')
    training=read(c['training_config'])
    for k in ('architecture','native_v6','native_v7','codec_id','time_contract','action_interval_seconds',
              'action_normalization_sha256','motion_normalization_sha256','init_checkpoint_sha256','manifest'):
        if c[k]!=training[k]:raise ValueError('evaluation/training configuration differs: '+k)
    sys.path.insert(0,str(ROOT/'src'))
    from dynamicwam.dualrate.native_v7_source_compat import validate_deployment_source
    compatibility=validate_deployment_source(TRAIN_SHA)
    from dualrate_eval_worker import load_frozen_task
    for task in c['tasks']:load_frozen_task(c,task)
    from dynamicwam.dualrate.renderer_runtime import validate_renderer_runtime
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay
    validate_renderer_runtime(c);validate_dependency_overlay(c)
    if set(c['gpu_uuids'])!=set(base.inventory()):raise ValueError('GPU UUID inventory differs')
    return c,compatibility


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--detach',action='store_true');p.add_argument('--controller',action='store_true');a=p.parse_args()
    spec=read(a.spec);out=Path(spec['output']);base.storage_directory(out,Path('/SSD_DISK/users/wuruihan'))
    source=guard.frozen_source(ROOT);template,compatibility=preflight(spec)
    if a.check:
        # No final checkpoint load/hash or GPU allocation while training runs.
        s=read(Path(spec['training_sequence_dir'])/'status.json')
        if s['source']['source_sha256']!=TRAIN_SHA or s.get('errors'):raise ValueError('training source/state mismatch')
        print(json.dumps(dict(passed=True,training_state=s['state'],phase=s.get('phase'),source=source,
            source_compatibility=compatibility,protocol=PROTOCOL,formal_episodes=700,gpu_used=False,
            final_checkpoint_gpu_acceptance='pending training completion and explicit launch')))
        return
    if not a.controller:out.mkdir(parents=True,exist_ok=False)
    if a.detach:
        argv=[v for v in sys.argv[1:] if v!='--detach']+['--controller']
        with (out/'controller.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*argv],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(state='dispatched',pid=child.pid,status=str(out/'status.json'))));return
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;ticks=None;began=time.monotonic()
    def status(state,**kw):save(out/'status.json',dict(state=state,controller=base.proc_identity(os.getpid()),no_retry=True,**kw))
    def stop(sig,_):raise InterruptedError(sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def run(c,name):
        nonlocal child,ticks
        if guard.frozen_source(ROOT)!=source:raise ValueError('evaluation source changed')
        cp=out/(name+'.json');save(cp,c)
        command=[sys.executable,'-B',str(ROOT/'scripts/guarded_dualrate_run.py'),'--config',str(cp),
                 '--run-dir',str(out/(name+'_guard')),'--gpus',','.join(c['gpu_uuids']),'--max-runtime',str(7*86400)]
        with (out/(name+'.supervisor.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=base.proc_identity(child.pid)['start_ticks'];status(name+'_running',supervisor_pid=child.pid,output=c['output'])
            rc=child.wait()
        child=None
        if rc:raise RuntimeError(name+' failed; automatic retry disabled')
        evidence=require_clean_guard(out/(name+'_guard'),c,source)
        summary=read(Path(c['output'])/'suite_completed.json')
        if (not summary['complete'] or summary['source']!=source or summary['checkpoint_sha256']!=c['checkpoint_sha256']
                or summary['config_sha256']!=base.canonical_hash(c) or summary['valid_completed']!=len(c['tasks'])*c['episodes']):
            raise ValueError('suite terminal evidence mismatch')
        return summary,evidence
    try:
        save(out/'source.json',source);save(out/'spec.frozen.json',spec);save(out/'compatibility.json',compatibility)
        while True:
            done=training_ready(spec)
            if done is not None:break
            if time.monotonic()-began>48*3600:raise TimeoutError('training completion wait exceeded 48h')
            status('waiting_training',training_sequence=spec['training_sequence_dir'])
            time.sleep(60)
        save(out/'training_completion.json',done)
        template.update(checkpoint=done['checkpoint']['path'],checkpoint_sha256=done['checkpoint']['sha256'])
        # Final checkpoint metadata/strict load is checked inside every guarded GPU worker.
        smoke=dict(template,stage='smoke',tasks=spec['smoke_tasks'],episodes=2,output=str(out/'smoke'),result_stamp=out.name+'_smoke')
        validate_config(smoke)
        completed,files=run(smoke,'smoke')
        files.extend([Path(smoke['output'])/'suite_completed.json',out/'training_completion.json'])
        for task in smoke['tasks']:
            files.append(Path(smoke['output'])/task/'model_acceptance.json')
        files.extend(Path(e['path']) for e in completed['reports'])
        acceptance=dict(passed=True,source=source,checkpoint_sha256=template['checkpoint_sha256'],episodes=16,
            paired_contract=paired_contract(smoke),native_mp4_full_decode=completed['native_mp4_full_decode'],
            all_workers_exit_zero=completed['all_workers_exit_zero'],reference_reuses=completed['reference_reuses'],
            reference_refreshes=completed['reference_refreshes'],
            evidence=[dict(path=str(f),sha256=sha(f)) for f in files])
        receipt=out/'smoke_acceptance.json';save(receipt,acceptance)
        formal=dict(template,output=str(out/'formal'),result_stamp=out.name+'_formal',
            acceptance_receipt=str(receipt),acceptance_receipt_sha256=sha(receipt))
        validate_formal_acceptance(formal,source)
        summary,_=run(formal,'formal');status('completed',valid_completed=700,success_rate_percent=summary['success_rate_percent'])
    except BaseException as exc:
        if child is not None and child.poll() is None and base.proc_identity(child.pid)['start_ticks']==ticks:
            child.send_signal(signal.SIGTERM)
            for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
            child.wait()  # Existing owner performs safe, complete session cleanup.
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


if __name__=='__main__':main()
