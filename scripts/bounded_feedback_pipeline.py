"""One-shot native data -> three-stage smoke -> formal training -> native eval.

CPU controller survives chat/SSH closure. Every GPU child has the existing owned
guard. Any failed phase stops the pipeline, with no automatic retry.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import guarded_dualrate_run as runner
from guarded_dualrate_smoke import atomic,file_sha256,proc_identity
from dualrate_training_sequence import phase_completed

PHASES=('warmup','small_age_joint','cross_time_joint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--core',required=True)
    p.add_argument('--cpu-acceptance',required=True);p.add_argument('--not-before',required=True)
    p.add_argument('--deployment',required=True);p.add_argument('--gpus',required=True);p.add_argument('--prepared-eval',required=True)
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'src'))
    if not str(root).startswith('/SSD_DISK/users/wuruihan/'):raise ValueError('verified SSD scope required')
    root.mkdir(parents=True,exist_ok=False);base=runner.read_json(a.core);deploy=runner.read_json(a.deployment)
    from bounded_feedback_launch import wait_and_validate
    wait_and_validate(a,repo,root,base,deploy)
    gpus=a.gpus.split(',')
    if len(gpus)!=8 or len(set(gpus))!=8:raise ValueError('eight distinct reserved GPUs required')
    source=runner.frozen_source(repo);children=[];began=time.monotonic()
    def status(state,**kw):atomic(root/'status.json',dict(state=state,pid=os.getpid(),elapsed=time.monotonic()-began,no_retry=True,**kw))
    def interrupted(sig,frame):raise InterruptedError(sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    def start(command,name):
        if runner.frozen_source(repo)!=source:raise RuntimeError('frozen implementation changed')
        log=(root/(name+'.log')).open('x')
        child=subprocess.Popen(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append((child,proc_identity(child.pid)['start_ticks']));return child
    def wait(child,name):
        while child.poll() is None:
            if any(root.glob('cohort_*/abort.json')) or any(other.poll() not in (None,0) for other,_ in children):raise RuntimeError('parallel guarded phase failed; no retry')
            time.sleep(1.)
        if child.returncode:raise RuntimeError(name+' failed; no retry')
        children[:]=[x for x in children if x[0] is not child]
    def rollout(config,name,devices):
        if name.startswith('collect_') or name.startswith('formal_eval_'):
            group='collect' if name.startswith('collect_') else 'eval';directory=root/('cohort_'+group)
            if not directory.exists():
                directory.mkdir();atomic(directory/'manifest.json',dict(members=[str(i) for i in range(4)]))
            config['parallel_guard_cohort']=dict(directory=str(directory),member=name.rsplit('_',1)[1])
        path=root/(name+'.json');atomic(path,config)
        child=start([sys.executable,str(repo/'scripts/guarded_dualrate_run.py'),'--config',str(path),
            '--run-dir',str(root/(name+'_guard')),'--gpus',','.join(devices),'--max-runtime',
            str(7*86400 if config.get('purpose')=='formal_native_evaluation' else 86400)],name)
        return child
    def check_rollout(name,config):
        guard=runner.read_json(root/(name+'_guard')/'status.json')
        acceptance=runner.read_json(Path(config['output'])/'deployment_acceptance.json')
        if guard.get('state')!='completed' or not guard.get('guard_closed') or not acceptance.get('passed'):raise RuntimeError(name+' acceptance/exit failed')
        model=runner.read_json(Path(config['output'])/'model_acceptance.json')
        if model.get('passed') is not True or model.get('contract')!='native_bounded_feedback_onepass_v10':raise RuntimeError('current one-pass model/graph acceptance missing')
        if acceptance.get('protocol')!='native_topp_dualrate_bounded_feedback_v10':raise RuntimeError('native feedback protocol mismatch')
        return acceptance
    def initial_training_acceptance(child,name,configs):
        # Finite CPU-only check, then stop observing. The existing guards remain
        # responsible for the run; this adds no CUDA context or recovery action.
        import math
        started=None;first=None;launch=time.monotonic()
        files=[Path(configs[0]['output'])/f'rank_{rank}.jsonl' for rank in range(8)]
        def latest(path):
            try:
                data=path.read_bytes().splitlines()
                return json.loads(data[-1]) if data else None
            except (FileNotFoundError,ValueError):return None
        while True:
            if child.poll() is not None:raise RuntimeError('formal sequence exited before initial stability acceptance')
            rows=[latest(path) for path in files]
            if started is None and all(rows):started=time.monotonic();first=rows
            if started is not None and time.monotonic()-started>=300:break
            if time.monotonic()-launch>1800:raise RuntimeError('formal training did not reach initial stability in 30 minutes')
            time.sleep(5.)
        for rank,row in enumerate(rows):
            if row['rank']!=rank or row['step']<=first[rank]['step'] or not math.isfinite(row['grad_norm']) or not all(math.isfinite(v) for v in row['losses'].values()):
                raise RuntimeError('initial eight-rank progress/finite check failed')
            if time.time()-files[rank].stat().st_mtime>120:raise RuntimeError('initial training rank stalled')
        guardroot=root/name/'warmup';auth=runner.read_json(guardroot/'authorization.json');owner=auth['external_guard']
        if proc_identity(owner['pid'])['start_ticks']!=owner['start_ticks']:raise RuntimeError('guard identity changed')
        held={os.readlink(f) for f in Path(f"/proc/{owner['pid']}/fd").iterdir()}
        if len(owner['device_paths'])!=8 or not set(owner['device_paths'])<=held:raise RuntimeError('guard lost device lifetime descriptors')
        from guarded_dualrate_smoke import kernel_evidence
        before=runner.read_json(guardroot/'kernel_before.json')
        if kernel_evidence()!=before['lines'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=before['boot_id']:raise RuntimeError('kernel/boot changed')
        if runner.frozen_source(repo)!=source:raise RuntimeError('source changed during initial training')
        atomic(root/'initial_training_acceptance.json',dict(passed=True,seconds=time.monotonic()-started,
            first_steps=[r['step'] for r in first],steps=[r['step'] for r in rows],guard_device_descriptors=8,kernel_unchanged=True,
            source=source,observation_finished=True))
    def sequence(configs,kind,name):
        rows=[]
        for phase,c in zip(PHASES,configs):
            path=root/(name+'_'+phase+'.json');atomic(path,c)
            rows.append(dict(phase=phase,config=str(path),config_sha256=file_sha256(path),max_runtime_seconds=86400*7))
        path=root/(name+'.json');atomic(path,dict(run_kind=kind,sequence_id=configs[0]['sequence_id'],gpu_uuids=a.gpus,phases=rows))
        child=start([sys.executable,str(repo/'scripts/dualrate_training_sequence.py'),'--sequence',str(path),'--run-dir',str(root/name)],name)
        status(name,child_pid=child.pid)
        if kind=='formal_distributed_training':initial_training_acceptance(child,name,configs)
        wait(child,name)
        if runner.read_json(root/name/'status.json')['state']!='completed':raise RuntimeError(name+' incomplete')
    try:
        # Predeclared clean train scenes, not scored scenes or success-filtered
        # policy rollouts. The initial zero-residual policy remains dual-rate.
        prepared=runner.read_json(a.prepared_eval);tasks=prepared['tasks']
        if len(tasks)!=35 or len(set(tasks))!=35:raise ValueError('35 native tasks required')
        from dynamicwam.dualrate.feedback_data import training_seed_manifest
        train_seeds=training_seed_manifest(base['manifest'])
        if set(train_seeds['tasks'])!=set(tasks):raise ValueError('training task set differs')
        atomic(root/'training_scenes.json',train_seeds)
        collection=[];status('collecting_native_training_replay')
        for group in range(4):
            name=f'collect_{group}';c=json.loads(json.dumps(deploy));c.pop('checkpoint',None);c.pop('checkpoint_sha256',None)
            c.update(output=str(root/name),collect_native_replay=True,purpose='native_training_collection')
            c['training_scenes']=train_seeds
            c['deployment'].update(tasks=tasks[group::4],episodes=1,seed=300100,result_stamp=root.name+'_collect')
            collection.append((rollout(c,name,gpus[2*group:2*group+2]),name,c))
        for child,name,c in collection:wait(child,name);check_rollout(name,c)
        sys.path.insert(0,str(repo/'src'))
        from dynamicwam.dualrate.native_v9_data import build_manifest,NativeReplayDataset
        manifest=root/'dataset/train.json'
        info=build_manifest(base['manifest'],[c['output'] for _,_,c in collection],manifest)
        mm=runner.read_json(manifest);mm['bounded_feedback_dataset']=True;atomic(manifest,mm)
        from dynamicwam.dualrate.feedback_data import FeedbackDataset
        NativeReplayDataset=FeedbackDataset
        info['sha256']=file_sha256(manifest)
        if len(info['tasks'])!=35 or info['samples']<280:raise RuntimeError('native replay task/window coverage is insufficient')
        ds=NativeReplayDataset(manifest)
        from dynamicwam.dualrate.train import validate_row
        for i in ds.select_domains(['native_replay'])[::max(1,len(ds.replay)//32)]:validate_row(ds[int(i)])
        atomic(root/'dataset/acceptance.json',dict(passed=True,**info,actual_native_timing=True,borrowed_future_labels=False))
        del ds
        configs=[]
        for phase,domains in zip(PHASES,(['expert_bootstrap'],['native_replay'],['expert_bootstrap','native_replay'])):
            c=dict(base,manifest=str(manifest),run_kind='distributed_training_smoke',phase=phase,
                sequence_id=root.name+'_smoke',output=str(root/'training_smoke'/phase),steps=4,
                query_age_seconds=base['query_age_seconds'],training_domains=domains,log_every=1,
                data_distribution='expert bootstrap + actual native Full-anchored feedback collection replay, no retimed labels')
            for key in ('stage_init_checkpoint','stage_init_checkpoint_sha256','acceptance_receipt','acceptance_receipt_sha256','resume'):c.pop(key,None)
            configs.append(c)
        sequence(configs,'distributed_training_smoke','smoke_sequence')
        diagnostic=Path(configs[-1]['output'])/'last.pt'
        smoke=json.loads(json.dumps(deploy));smoke.update(output=str(root/'trained_smoke'),manifest=str(manifest),collect_native_replay=False,
            checkpoint=str(diagnostic),checkpoint_sha256=file_sha256(diagnostic),purpose='diagnostic_trained_smoke')
        smoke['deployment'].update(tasks=['adjust_bottle','click_bell'],episodes=1,seed=400000,result_stamp=root.name+'_trained_smoke')
        status('trained_deployment_smoke');child=rollout(smoke,'trained_smoke',gpus[:2]);wait(child,'trained_smoke')
        acceptance=check_rollout('trained_smoke',smoke)
        evidence=[];identity=None
        for phase,c in zip(PHASES,configs):
            _,_,result,_=phase_completed(Path(c['output']),c)
            chosen={k:result['identity'][k] for k in ('source_sha256','manifest_sha256','init_checkpoint_sha256')}
            if identity is not None and chosen!=identity:raise ValueError('phase identities differ')
            identity=chosen
            for f in [Path(c['output'])/'result.json',Path(c['output'])/'reload.json',root/'smoke_sequence'/phase/'status.json']:
                evidence.append(dict(path=str(f),sha256=file_sha256(f)))
        for folder in [root/'dataset',root/'trained_smoke',root/'trained_smoke_guard']:
            for f in folder.rglob('*'):
                if f.is_file() and f.suffix in ('.json','.mp4'):evidence.append(dict(path=str(f),sha256=file_sha256(f)))
        receipt=dict(passed=True,world_size=8,**identity,supervisor_sha256=source['distributed_supervisor_sha256'],
            source_contract=runner.source_contract(source),profiles={p:runner.profile_sha256(c) for p,c in zip(PHASES,configs)},
            evidence=evidence,native_deployment=acceptance,quality_accepted=False)
        ap=root/'acceptance.json';atomic(ap,receipt)
        from datetime import datetime
        if time.time()<datetime.fromisoformat(a.not_before).timestamp():raise RuntimeError('formal training attempted before user time')
        atomic(root/'formal_checklist.json',dict(passed=True,cpu_acceptance_sha256=file_sha256(Path(a.cpu_acceptance)),
            gpu_acceptance_sha256=file_sha256(ap),world_size=8,all_three_phases_restored=True,
            actual_native_replay=True,native_topp_mp4_accepted=True,not_before=a.not_before,
            formal_weights_from_official_full=True,smoke_weights_never_initialize_formal=True,
            policy_quality_proven=False,no_retry=True))
        formal=[]
        for phase,c,steps in zip(PHASES,configs,(500,1500,3000)):
            formal.append(dict(c,run_kind='formal_distributed_training',steps=steps,sequence_id=root.name+'_formal',
                output=str(root/'formal_training'/phase),acceptance_receipt=str(ap),acceptance_receipt_sha256=file_sha256(ap)))
        sequence(formal,'formal_distributed_training','formal_sequence')
        checkpoint=Path(formal[-1]['output'])/'last.pt';digest=file_sha256(checkpoint)
        final=json.loads(json.dumps(deploy));final.update(manifest=str(manifest),checkpoint=str(checkpoint),checkpoint_sha256=digest,
            collect_native_replay=False,purpose='final_weight_acceptance',output=str(root/'final_smoke'))
        final['deployment'].update(tasks=['adjust_bottle','click_bell'],episodes=2,seed=450000,result_stamp=root.name+'_final_smoke')
        status('final_weight_smoke');child=rollout(final,'final_smoke',gpus[:2]);wait(child,'final_smoke');check_rollout('final_smoke',final)
        status('completed_training',checkpoint=str(checkpoint),checkpoint_sha256=digest,
            formal_training_started=True,final_native_smoke_passed=True,formal_benchmark_started=False)
        return  # The user scheduled training, not an unsolicited 700-episode suite.
    except BaseException as exc:
        status('failed',error=repr(exc))
        for child,ticks in children:
            if child.poll() is None and proc_identity(child.pid)['start_ticks']==ticks:child.send_signal(signal.SIGTERM)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
        for child,_ in children:child.wait()
        raise


if __name__=='__main__':main()
