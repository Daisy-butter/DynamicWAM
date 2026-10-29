"""Finite, detached actual smoke -> response acceptance -> formal700 sequence.

No GPU polling, retries, recovery or checkpoint changes. The existing guard owns
GPU health, idle checks and cleanup. --check performs CPU-only preflight and exits.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DEFAULT_REPO=Path('/SSD_DISK/users/wuruihan/DynamicWAM/worktrees/dualrate-cohort-stall-20261029')
DEFAULT_ROOT=Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/diagnostics/dualrate_cohort_stall_20261029/final01')
SMOKE_TASKS=['place_empty_cup','place_fan']

def load_modules(repo):
    sys.path.insert(0,str(repo/'scripts'));sys.path.insert(0,str(repo/'src'))
    import guarded_dualrate_run as guard
    import dualrate_posttrain_eval as post
    import dualrate_eval_worker as worker
    import dualrate_response_acceptance as response
    for module in (guard,post,worker,response):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError('sequence imported code outside frozen repository: '+module.__name__)
    return guard,post,worker,response

def make_formal(smoke,tasks,root):
    if not isinstance(tasks,list) or len(tasks)!=35 or len(set(tasks))!=35 or not all(isinstance(x,str) for x in tasks):
        raise ValueError('exactly35 distinct task strings required')
    if not set(smoke['tasks']).issubset(tasks):raise ValueError('smoke tasks absent from formal task set')
    if smoke['tasks']!=SMOKE_TASKS:raise ValueError('failure-case smoke must be place_empty_cup then place_fan')
    if smoke.get('retained_task_receipt'):
        retained=json.loads(Path(smoke['retained_task_receipt']).read_text())
        if tasks!=retained['formal_tasks']:
            raise ValueError('retention requires the original frozen35-task order')
    else:
        # A fresh full rerun keeps the reproduced smoke cohort first.
        tasks=list(SMOKE_TASKS)+[task for task in tasks if task not in SMOKE_TASKS]
    return dict(smoke,stage='formal',tasks=tasks,episodes=20,
        output=str(root/'formal01'),result_stamp='dualrate_cohort_stall_formal01',
        acceptance_receipt=str(root/'response01_acceptance.json'),
        acceptance_receipt_sha256='PENDING_ACTUAL_GPU_RESPONSE_ACCEPTANCE')

def cpu_probe(repo,smoke,config_path):
    # Run in an isolated CPU child so the controller never initializes CUDA.
    code='''import inspect,json,sys
from pathlib import Path
from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay
c=json.load(open(sys.argv[1]));activate_dependency_overlay(validate_dependency_overlay(c))
import torch
from dynamicwam.dualrate import producer,online
import dualrate_deployment_inputs as inputs
import dualrate_slow_process as slow
repo=Path(sys.argv[2]).resolve()
for m in (producer,online,inputs,slow):
    assert Path(m.__file__).resolve().is_relative_to(repo),m.__file__
for fn in (producer._letterbox,producer.view_overlap,online.OnlineBatchBuilder.current,
           online.OnlineBatchBuilder.source,inputs.build_cached_online_batch_builder,slow._load_system):
    assert callable(fn)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(cpu_imports=True,producer_source=str(Path(producer.__file__).resolve()),cuda_initialized=False)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    env['PYTHONPATH']=os.pathsep.join((str(repo/'src'),str(repo/'scripts'),env.get('PYTHONPATH','')))
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(config_path),str(repo)],
        cwd=repo,env=env,text=True,timeout=120))

def preflight(args):
    repo=args.repo.resolve(strict=True);root=args.root.resolve()
    guard,post,worker,response=load_modules(repo)
    smoke=post.read(args.smoke_config);tasks=post.read(args.tasks_file)
    if Path(smoke['output']).resolve()!=root/'smoke01':raise ValueError('smoke output path differs')
    formal=make_formal(smoke,tasks,root)
    post.validate_pair(smoke,formal)
    for c in (smoke,formal):
        worker.validate_config(c);worker.verify_artifacts(c)
    failure_case=worker.load_frozen_task(smoke,'place_empty_cup')
    if len(failure_case)!=2 or failure_case[1][0].get('seed')!=100001:
        raise ValueError('smoke must actually include failed cup episode1 accepted seed100001')
    for path in (root/'smoke01',root/'smoke01_guard',root/'formal01',root/'formal01_guard',
                 root/'formal01.config.json',root/'response01_acceptance.json'):
        if path.exists():raise FileExistsError('new sequence paths required: '+str(path))
    source=guard.frozen_source(repo)
    probe=cpu_probe(repo,smoke,args.smoke_config)
    return repo,root,(guard,post,worker,response),smoke,formal,source,probe

def run(args):
    import fcntl
    repo,root,modules,smoke,formal,source,probe=preflight(args)
    guard,post,worker,response=modules
    out=args.run_dir.resolve();lock=(out/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();child=None;identity=None;errors=[]
    input_hashes={str(p.resolve()):post.file_sha256(p) for p in (args.smoke_config,args.tasks_file)}
    def status(state,**extra):
        post.atomic(out/'status.json',dict(state=state,controller=post.proc_identity(os.getpid()),
            elapsed_seconds=time.monotonic()-began,no_retry=True,errors=errors,**extra))
    def frozen():
        if guard.frozen_source(repo)!=source:raise RuntimeError('source changed during sequence')
        for path,digest in input_hashes.items():
            if post.file_sha256(Path(path))!=digest:raise RuntimeError('sequence input changed: '+path)
    def interrupted(sig,_frame):raise InterruptedError('sequence signal '+str(sig))
    def launch(config,path,guard_dir,limit,label):
        nonlocal child,identity
        frozen()
        command=[sys.executable,'-B',str(repo/'scripts/guarded_dualrate_run.py'),
            '--config',str(path),'--run-dir',str(guard_dir),'--storage-root',config['storage_root'],
            '--gpus',','.join(config['gpu_uuids']),'--max-runtime',str(limit)]
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        with (out/(label+'.supervisor.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            identity=post.proc_identity(child.pid)
            status(label+'_running',supervisor=identity,output=config['output'])
            # This is a finite owned subprocess wait, not a separate GPU monitor.
            rc=child.wait(timeout=limit+1200)
        child=None;identity=None
        if rc:raise RuntimeError(label+' guard failed; no retry or next phase')
        frozen();post.require_clean_guard(guard_dir,config,source)
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in previous:signal.signal(s,interrupted)
    try:
        post.atomic(out/'source.json',source);post.atomic(out/'cpu_preflight.json',probe)
        post.atomic(out/'input_hashes.json',input_hashes)
        post.atomic(out/'formal_template.json',formal)
        status('starting_smoke')
        smoke_guard=root/'smoke01_guard'
        launch(smoke,args.smoke_config,smoke_guard,args.smoke_max_runtime,'smoke')
        frozen();status('validating_actual_smoke')
        accepted=root/'response01_acceptance.json'
        response.run_gate(repo,smoke_guard,accepted)
        formal.update(acceptance_receipt=str(accepted),acceptance_receipt_sha256=post.file_sha256(accepted))
        post.validate_pair(smoke,formal)
        response.validate_for_formal(formal,guard.source_contract(source),accepted)
        guard.validate_acceptance(formal,source);worker.validate_config(formal)
        path=root/'formal01.config.json';post.atomic(path,formal)
        input_hashes[str(path)]=post.file_sha256(path)
        post.atomic(out/'input_hashes.json',input_hashes)
        status('starting_formal',acceptance_receipt=str(accepted))
        formal_guard=root/'formal01_guard'
        launch(formal,path,formal_guard,args.formal_max_runtime,'formal')
        summary,evidence,_=post.audit_suite(formal,source,700)
        post.atomic(root/'formal01_acceptance.json',dict(passed=True,valid_completed=700,
            source=source,checkpoint_sha256=formal['checkpoint_sha256'],
            evidence=[dict(path=str(p.resolve()),sha256=post.file_sha256(p)) for p in evidence]))
        status('completed',valid_completed=700,checkpoint_sha256=formal['checkpoint_sha256'])
        return 0
    except BaseException as exc:
        errors.append(repr(exc))
        if child is not None and child.poll() is None:
            try:
                if identity is not None and post.proc_identity(child.pid)['start_ticks']!=identity['start_ticks']:
                    raise RuntimeError('guard PID reused; refusing signal')
                child.send_signal(signal.SIGTERM)
                status('failed_waiting_owned_guard_cleanup',supervisor=identity)
                for s in previous:signal.signal(s,signal.SIG_IGN)
                # Never kill a cleanup guard while it still owns worker/device handles.
                child.wait()
            except BaseException as cleanup:errors.append('cleanup unconfirmed: '+repr(cleanup))
        status('failed');return 1
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
        lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=DEFAULT_REPO);p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--smoke-config',type=Path)
    p.add_argument('--tasks-file',type=Path,required=True)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--smoke-max-runtime',type=float,default=28800)
    p.add_argument('--formal-max-runtime',type=float,default=172800)
    p.add_argument('--check',action='store_true');p.add_argument('--detach',action='store_true')
    p.add_argument('--controller',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if os.name!='posix':p.error('Linux required')
    if not 0<a.smoke_max_runtime<=28800 or not 0<a.formal_max_runtime<=172800:p.error('runtime bounds exceeded')
    a.root=a.root.resolve();a.repo=a.repo.resolve(strict=True)
    a.smoke_config=a.smoke_config or a.root/'smoke01.config.json'
    a.run_dir=a.run_dir or a.root/'sequence01'
    a.smoke_config=a.smoke_config.resolve(strict=True);a.tasks_file=a.tasks_file.resolve(strict=True)
    if a.check:
        *_,probe=preflight(a);print(json.dumps(dict(passed=True,launched=False,probe=probe)));return 0
    if not a.controller:a.run_dir.mkdir(parents=True,exist_ok=False)
    elif not a.run_dir.is_dir() or any(x.name!='controller.log' for x in a.run_dir.iterdir()):
        p.error('detached controller requires a new directory')
    if a.detach and not a.controller:
        argv=[x for x in sys.argv[1:] if x!='--detach']+['--controller']
        with (a.run_dir/'controller.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*argv],cwd=a.repo,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(state='dispatched',controller_pid=child.pid,run_dir=str(a.run_dir))))
        return 0
    try:return run(a)
    except BaseException as exc:
        # Preflight precedes GPU ownership; record a failed controller even when
        # artifact/import validation fails before run() installs its handlers.
        (a.run_dir/'status.json').write_text(json.dumps(dict(state='failed',no_retry=True,
            errors=[repr(exc)],preflight_failed_before_gpu_launch=True)),encoding='utf-8')
        raise

if __name__=='__main__':raise SystemExit(main())
