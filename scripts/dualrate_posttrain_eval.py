#!/usr/bin/env python3
"""One-shot CPU-only training-completion gate -> smoke -> formal evaluation.

Reads saved receipts and /proc every60s for at most48h. It never touches a
training process, imports CUDA/NVML, retries a failed job, or estimates its ETA.
Each evaluation child is a fresh existing guard that owns GPU cleanup.
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

import guarded_dualrate_run as guard
from guarded_dualrate_smoke import atomic, canonical_hash, file_sha256, proc_identity, storage_directory

ROOT = Path(__file__).resolve().parents[1]
TRAIN_COMMIT = '775c8e2b9dccf88e11d5d9f467a9cb85714564de'
TRAIN_SHA = '9da1078527647dd7a53a25006c62802f113b4d2d5e211c2d52250a5414e63e38'
PHASES = (('warmup',1000,1000),('small_age_joint',10000,11000),('cross_time_joint',20000,31000))
POLL_SECONDS = 60.
PAIR_DIFFERENCES = {'stage','tasks','episodes','output','result_stamp','acceptance_receipt',
                    'acceptance_receipt_sha256','task_timeout_seconds','cohort_timeout_seconds'}


def read(path):
    return json.loads(Path(path).read_text())


def identity_exited(identity):
    """PID reuse means the recorded process exited; never signal that new PID."""
    try:
        actual=proc_identity(int(identity['pid']))
    except (FileNotFoundError, ProcessLookupError):
        return True
    return actual['start_ticks']!=identity['start_ticks'] or actual['state'] in ('Z','X')


def require_clean_guard(directory, config, source):
    directory=Path(directory);status=read(directory/'status.json')
    if status.get('state')!='completed' or status.get('guard_closed') is not True or status.get('errors'):
        raise ValueError('guard completion/closure/errors rejected: '+str(directory))
    owned=read(directory/'owned_exit.json');kernel=read(directory/'kernel_after.json')
    if owned.get('remaining')!=[] or owned.get('returncode')!=0:
        raise ValueError('owned workers did not exit cleanly')
    if kernel.get('unchanged') is not True or kernel.get('same_boot') is not True or kernel.get('added')!=[]:
        raise ValueError('kernel/boot changed')
    if read(directory/'source.json')!=source or read(directory/'config.frozen.json')!=config:
        raise ValueError('frozen guard source/config differs')
    auth=read(directory/'authorization.json')
    uuids=config.get('gpu_uuids') or auth.get('gpu_uuids')
    if (auth.get('identity',{}).get('config_sha256')!=canonical_hash(config)
            or len(set(uuids or []))!=8 or auth.get('gpu_uuids')!=uuids):
        raise ValueError('eight-GPU authorization binding differs')
    events=read(directory/'guard_events.json')
    opened=[(e['uuid'],e['path'],e['fd']) for e in events if e['event']=='after_open']
    closed=[(e['uuid'],e['path'],e['fd']) for e in events if e['event']=='after_close']
    if len(opened)!=8 or len(closed)!=8 or set(opened)!=set(closed) or {x[0] for x in opened}!=set(uuids):
        raise ValueError('eight guard descriptors were not closed exactly once')
    return [directory/name for name in ('status.json','owned_exit.json','kernel_before.json',
        'kernel_after.json','source.json','config.frozen.json','authorization.json','guard_events.json')]


def stable_checkpoint(path, expected_sha):
    path=Path(path).resolve(strict=True)
    before=path.stat()
    digest=file_sha256(path)
    after=path.stat()
    fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    if any(getattr(before,k)!=getattr(after,k) for k in fields) or before.st_size<=0:
        raise ValueError('checkpoint changed while being hashed')
    if digest!=expected_sha:
        raise ValueError('final checkpoint bytes differ from completion receipt')
    return dict(path=str(path),sha256=digest,size_bytes=after.st_size,
                mtime_ns=after.st_mtime_ns,stable_during_full_hash=True)


def training_ready(spec):
    """None means a healthy unfinished training controller; invalid evidence fails."""
    root=Path(spec['training_sequence_dir'])
    status=read(root/'status.json')
    source=status['source']
    if (source.get('commit')!=spec.get('expected_training_commit',TRAIN_COMMIT)
            or source.get('source_sha256')!=spec.get('expected_training_source_sha256',TRAIN_SHA)):
        raise ValueError('training source differs from requested V4')
    state=status.get('state')
    if status.get('errors') or state in ('failed','failed_waiting_guard_cleanup'):
        raise RuntimeError('training failed; evaluation will not start')
    if state!='completed':
        if state not in ('preparing','phase_running','phase_completed') or identity_exited(status['controller']):
            raise RuntimeError('unfinished training has no live recorded controller')
        return None
    # A completed status is written before process exit. Wait for exact recorded
    # controllers/guards to exit before hashing or acquiring any GPU resource.
    if not identity_exited(status['controller']):
        return None
    sequence=read(root/'sequence.frozen.json')
    rows=sequence['phases']
    if [x['phase'] for x in rows]!=[x[0] for x in PHASES] or status['sequence_id']!=sequence['sequence_id']:
        raise ValueError('training phase order/sequence identity differs')
    if read(root/'source.json')!=source:
        raise ValueError('training sequence source receipt differs')
    for name,_,_ in PHASES:
        phase_status=read(root/name/'status.json')
        if phase_status.get('state')!='completed' or phase_status.get('guard_closed') is not True:
            raise ValueError('completed training contains an unfinished phase guard')
        if not identity_exited(phase_status['supervisor']):
            return None
    evidence=[root/'status.json',root/'source.json',root/'sequence.frozen.json']
    phases=[];previous=None
    for row,(name,steps,global_step) in zip(rows,PHASES):
        template_path=Path(row['config'])
        if file_sha256(template_path)!=row['config_sha256']:
            raise ValueError('training phase template changed')
        template=read(template_path);config_path=root/(name+'.config.json');config=read(config_path)
        expected=dict(template,sequence_id=sequence['sequence_id'])
        if previous is not None:
            expected.update(stage_init_checkpoint=previous['checkpoint'],stage_init_checkpoint_sha256=previous['checkpoint_sha256'])
        if (config!=expected or config.get('phase')!=name or config.get('steps')!=steps
                or config.get('world_size')!=8 or config.get('run_kind')!='formal_distributed_training'):
            raise ValueError('training phase settings or predecessor differ')
        evidence.extend(require_clean_guard(root/name,config,source))
        output=Path(config['output']);result=read(output/'result.json');reload=read(output/'reload.json')
        if (result.get('result_kind')!='completed_phase_not_policy_quality_acceptance'
                or result.get('phase')!=name or result.get('run_kind')!=config['run_kind']
                or result.get('world_size')!=8 or result.get('completed_steps')!=steps
                or result.get('configured_steps')!=steps or result.get('global_step')!=global_step):
            raise ValueError('training result does not prove full phase completion')
        ranks=reload.get('ranks',[])
        if (reload.get('passed') is not True or reload.get('world_size')!=8
                or reload.get('equal_model') is not True or reload.get('equal_optimizer') is not True
                or reload.get('extra_training_updates')!=0 or len(ranks)!=8
                or {x.get('rank') for x in ranks}!=set(range(8))
                or any(x.get('passed') is not True or x.get('phase_step')!=steps
                       or x.get('global_step')!=global_step or x.get('before')!=x.get('after')
                       or set(x.get('before',{}))!={'model','optimizer','rank','modes','requires_grad'} for x in ranks)):
            raise ValueError('strict full optimizer/master/RNG reload failed on one or more ranks')
        if (reload.get('checkpoint_sha256')!=result.get('checkpoint_sha256')
                or reload.get('identity',{}).get('source_sha256')!=source['source_sha256']
                or result.get('identity')!=reload.get('identity')
                or reload.get('identity')!=read(root/name/'authorization.json').get('identity')):
            raise ValueError('reload checkpoint/source binding differs')
        checkpoint=stable_checkpoint(output/'last.pt',result['checkpoint_sha256'])
        if Path(reload['checkpoint']).resolve()!=Path(checkpoint['path']):
            raise ValueError('reload used a different checkpoint')
        if Path(result['checkpoint']).resolve()!=Path(checkpoint['path']):
            raise ValueError('result used a different checkpoint')
        previous=dict(phase=name,checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],global_step=global_step)
        phases.append(previous)
        evidence.extend((template_path,config_path,output/'result.json',output/'reload.json'))
    if status.get('completed_phases')!=phases:
        raise ValueError('sequence completed-phase summary differs from phase receipts')
    return dict(passed=True,sequence_id=sequence['sequence_id'],source=source,phases=phases,
        checkpoint=checkpoint,all_training_processes_exited=True,
        evidence=[dict(path=str(p.resolve()),sha256=file_sha256(p)) for p in evidence])


def validate_pair(smoke,formal):
    from dualrate_retention import expected_smoke_tasks
    if (smoke.get('run_kind')!='dualrate_eval_suite' or formal.get('run_kind')!='dualrate_eval_suite'
            or smoke.get('stage')!='smoke' or formal.get('stage')!='formal'
            or len(smoke.get('tasks',[]))!=2 or smoke.get('episodes')!=2
            or len(formal.get('tasks',[]))!=35 or formal.get('episodes')!=20
            or smoke['tasks']!=expected_smoke_tasks(formal) or smoke.get('action_limit') is not None
            or formal.get('action_limit') is not None):
        raise ValueError('full-budget 2x2 smoke then35x20 required')
    for key in set(smoke)|set(formal):
        if key not in PAIR_DIFFERENCES and smoke.get(key)!=formal.get(key):
            raise ValueError('smoke/formal contract differs: '+key)
    uuids=smoke['gpu_uuids']
    if (len(uuids)!=8 or len(set(uuids))!=8 or smoke.get('environment_groups')!=[uuids[:3],uuids[3:6]]
            or not smoke.get('benchmark_contract') or Path(smoke['output']).resolve()==Path(formal['output']).resolve()):
        raise ValueError('all-eight guard/two independent three-GPU environments required')


def bind_templates(spec,training):
    configs=[]
    for kind in ('smoke','formal'):
        path=Path(spec[kind+'_template'])
        if file_sha256(path)!=spec[kind+'_template_sha256']:
            raise ValueError('evaluation template changed: '+kind)
        config=read(path)
        if config.get('checkpoint_sha256') not in (None,'PENDING_TRAINING_COMPLETION',training['checkpoint']['sha256']):
            raise ValueError('template already bound to a different checkpoint')
        config.update(checkpoint=training['checkpoint']['path'],checkpoint_sha256=training['checkpoint']['sha256'])
        configs.append(config)
    validate_pair(*configs)
    return configs


def audit_suite(config,source,expected):
    from dualrate_eval_suite import audit_task
    from video_artifacts import probe_head_video
    output=Path(config['output']);summary=read(output/'suite_completed.json')
    if (summary.get('complete') is not True or summary.get('valid_completed')!=expected
            or summary.get('all_workers_exit_zero') is not True or summary.get('every_video_fully_decoded') is not True
            or summary.get('source')!=source or summary.get('checkpoint_sha256')!=config['checkpoint_sha256']
            or summary.get('config_sha256')!=canonical_hash(config)):
        raise ValueError('evaluation suite completion/binding failed')
    from dualrate_retention import validate_completion
    evidence=[output/'suite_completed.json',*validate_completion(config,source,summary)];decoded=[]
    for task in config['tasks']:
        reports,_=audit_task(config,task)
        for report in reports:
            path=output/task/f"episode_{report['episode_index']:04d}/report.json"
            probe=probe_head_video(Path(report['video']))
            if probe!=report['video_probe']:
                raise ValueError('independent full native MP4 decode differs')
            evidence.extend((path,Path(report['video']),Path(report['video_timestamps'])))
            decoded.append(dict(report=str(path),probe=probe))
    if len(decoded)!=expected:
        raise ValueError('wrong number of complete decoded evaluation episodes')
    return summary,evidence,decoded


def make_acceptance(smoke,directory,source,destination):
    evidence=require_clean_guard(directory,smoke,source)
    summary,paths,decoded=audit_suite(smoke,source,4);evidence.extend(paths)
    deployment=summary.get('deployment_acceptance',{})
    if (deployment.get('passed') is not True or deployment.get('fast_graph_passed') is not True
            or deployment.get('fast_queries_during_slow',0)<=0
            or deployment.get('model_overlap_fast_queries',0)<=0
            or deployment.get('refreshed_reference_consumed') is not True):
        raise ValueError('actual Graph/concurrency/new-world-reference acceptance failed')
    decode_path=Path(destination).with_name('smoke_video_decode.json')
    atomic(decode_path,dict(passed=True,count=4,decoded=decoded));evidence.append(decode_path)
    receipt=dict(passed=True,source_contract=guard.source_contract(source),
        checkpoint_sha256=smoke['checkpoint_sha256'],benchmark_contract=smoke['benchmark_contract'],
        world_size=8,complete_smoke_episodes=4,native_mp4_full_decode=True,
        hardtime_acceptance_passed=True,
        clean_worker_exit=True,guard_closed=True,kernel_unchanged=True,deployment_acceptance=deployment,
        evidence=[dict(path=str(p.resolve()),sha256=file_sha256(p)) for p in evidence])
    atomic(destination,receipt)
    return receipt


def run(spec,out,repo=ROOT):
    import fcntl
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=guard.frozen_source(repo);began=time.monotonic();child=None;child_identity=None;errors=[]
    def status(state,**extra):
        atomic(out/'status.json',dict(state=state,controller=proc_identity(os.getpid()),source=source,
            elapsed_seconds=time.monotonic()-began,no_retry=True,errors=list(errors),**extra))
    def interrupted(signum,_frame):
        raise InterruptedError('post-training evaluation interrupted: '+str(signum))
    def frozen():
        if guard.frozen_source(repo)!=source:
            raise RuntimeError('evaluation source changed while controller was waiting/running')
    def launch(config,name,max_runtime):
        nonlocal child,child_identity
        frozen();path=out/(name+'.config.json');atomic(path,config)
        directory=out/(name+'_guard')
        command=[sys.executable,'-B',str(repo/'scripts/guarded_dualrate_run.py'),
            '--config',str(path),'--run-dir',str(directory),'--storage-root',config['storage_root'],
            '--gpus',','.join(config['gpu_uuids']),'--max-runtime',str(max_runtime)]
        with (out/(name+'.supervisor.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            child_identity=proc_identity(child.pid)
            status(name+'_running',supervisor=child_identity,output=config['output'])
            rc=child.wait()
        child=None;child_identity=None
        if rc:
            raise RuntimeError(name+' guard failed; no retry or next phase')
        frozen();require_clean_guard(directory,config,source)
        return directory
    previous_signals={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in previous_signals:signal.signal(sig,interrupted)
    try:
        max_wait=float(spec.get('max_wait_seconds',172800.))
        if not 0<max_wait<=172800:
            raise ValueError('training wait bound must be positive and at most48h')
        atomic(out/'spec.frozen.json',spec);atomic(out/'source.json',source)
        status('waiting_training',poll_seconds=POLL_SECONDS)
        deadline=time.monotonic()+max_wait
        while True:
            frozen();training=training_ready(spec)
            if training is not None:break
            progress=read(Path(spec['training_sequence_dir'])/'status.json') if 'training_sequence_dir' in spec else {}
            status('waiting_training',poll_seconds=POLL_SECONDS,last_checked_wall_time=time.time(),
                   training_state=progress.get('state'),training_phase=progress.get('phase'))
            if time.monotonic()>=deadline:
                raise TimeoutError('training completion not accepted within configured48h bound')
            time.sleep(min(POLL_SECONDS,max(0.,deadline-time.monotonic())))
        atomic(out/'training_acceptance.json',training)
        smoke,formal=bind_templates(spec,training)
        for config in (smoke,formal):
            storage_directory(Path(config['output']),Path(config['storage_root']))
            if Path(config['output']).exists():raise ValueError('evaluation output must be new')
        smoke_guard=launch(smoke,'smoke',spec['smoke_max_runtime'])
        acceptance_path=out/'acceptance.json'
        make_acceptance(smoke,smoke_guard,source,acceptance_path)
        stable_checkpoint(training['checkpoint']['path'],training['checkpoint']['sha256'])
        formal.update(acceptance_receipt=str(acceptance_path.resolve()),acceptance_receipt_sha256=file_sha256(acceptance_path))
        formal_guard=launch(formal,'formal',spec['formal_max_runtime'])
        summary,evidence,_=audit_suite(formal,source,700)
        atomic(out/'formal_acceptance.json',dict(passed=True,valid_completed=700,source=source,
            checkpoint_sha256=formal['checkpoint_sha256'],guard=str(formal_guard),
            evidence=[dict(path=str(p.resolve()),sha256=file_sha256(p)) for p in evidence]))
        status('completed',valid_completed=700,checkpoint_sha256=formal['checkpoint_sha256'])
        return 0
    except BaseException as exc:
        errors.append(repr(exc))
        if child is not None and child.poll() is None:
            try:
                if proc_identity(child.pid)['start_ticks']!=child_identity['start_ticks']:
                    raise RuntimeError('guard PID reused; refusing signal')
                child.send_signal(signal.SIGTERM)
                status('failed_waiting_owned_guard_cleanup',supervisor=child_identity)
                for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
                child.wait()  # The guard retains its handles until its exact workers exit.
            except BaseException as cleanup_exc:
                errors.append('owned guard cleanup unconfirmed: '+repr(cleanup_exc))
        status('failed')
        return 1
    finally:
        for sig,handler in previous_signals.items():signal.signal(sig,handler)
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--controller',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if os.name!='posix':parser.error('Linux required')
    spec=read(args.config);out=storage_directory(args.run_dir,Path(spec['storage_root']))
    if not args.controller:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir() or any(p.name!='controller.log' for p in out.iterdir()):
        parser.error('detached controller requires its fresh output directory')
    if args.detach and not args.controller:
        argv=[x for x in sys.argv[1:] if x!='--detach']+['--controller']
        with (out/'controller.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*argv],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(state='dispatched',controller_pid=child.pid,run_dir=str(out))))
        return 0
    return run(spec,out)


if __name__=='__main__':
    raise SystemExit(main())
