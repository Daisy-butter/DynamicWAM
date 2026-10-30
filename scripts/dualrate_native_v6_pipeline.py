"""One-shot native V6 acceptance -> fresh formal training; no automatic retries."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import guarded_dualrate_run as runner
from guarded_dualrate_smoke import atomic,file_sha256,proc_identity
from dualrate_training_sequence import phase_completed

PHASES=("warmup","small_age_joint","cross_time_joint")
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base-config',required=True)
    p.add_argument('--deploy-config',required=True);p.add_argument('--gpus',required=True)
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(__file__).resolve().parents[1]
    if not str(root).startswith('/SSD_DISK/users/wuruihan/'):raise ValueError('SSD owner scope required')
    root.mkdir(parents=True,exist_ok=False)
    source=runner.frozen_source(repo);base=runner.read_json(a.base_config);deployment=runner.read_json(a.deploy_config)
    child=None;child_ticks=None;began=time.monotonic()
    def status(state,**more):
        atomic(root/'status.json',dict(state=state,pid=os.getpid(),elapsed=time.monotonic()-began,no_retry=True,**more))
    def interrupted(sig,frame):raise InterruptedError(sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    def run(command,name):
        nonlocal child,child_ticks
        if runner.frozen_source(repo)!=source:raise RuntimeError('frozen source changed')
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_ticks=proc_identity(child.pid)['start_ticks'];status(name,child_pid=child.pid)
            rc=child.wait()
        child=None
        if rc:raise RuntimeError(name+' exited '+str(rc))
    def sequence(configs,mode,name):
        rows=[]
        for phase,c in zip(PHASES,configs):
            path=root/(name+'_'+phase+'.json');atomic(path,c)
            rows.append(dict(phase=phase,config=str(path),config_sha256=file_sha256(path),max_runtime_seconds=2400 if mode=='distributed_training_smoke' else 14*86400))
        spec=dict(run_kind=mode,sequence_id=configs[0]['sequence_id'],gpu_uuids=a.gpus,phases=rows)
        path=root/(name+'.json');atomic(path,spec)
        run([sys.executable,str(repo/'scripts/dualrate_training_sequence.py'),'--sequence',str(path),'--run-dir',str(root/name)],name)
        if runner.read_json(root/name/'status.json')['state']!='completed':raise RuntimeError('sequence incomplete')
    try:
        ages=([0.],[0.,.36,.72],[0.,.36,.72,1.08,1.44,1.8,2.16])
        configs=[]
        for phase,age in zip(PHASES,ages):
            c=dict(base,run_kind='distributed_training_smoke',phase=phase,sequence_id=root.name+'_smoke',
                output=str(root/'training_smoke'/phase),steps=2,query_age_seconds=age,
                fast_lr=1e-4,motor_lr=3e-6,slow_lr=0.,log_every=1,action_interval_seconds=.18,save_every=1000)
            for key in ('warmup_steps','stage_init_checkpoint','stage_init_checkpoint_sha256','acceptance_receipt','acceptance_receipt_sha256'):
                c.pop(key,None)
            configs.append(c)
        sequence(configs,'distributed_training_smoke','smoke_sequence')
        deployment=dict(deployment,output=str(root/'native_deployment'))
        dp=root/'native_deploy.json';atomic(dp,deployment)
        run([sys.executable,str(repo/'scripts/guarded_dualrate_run.py'),'--config',str(dp),
            '--run-dir',str(root/'native_guard'),'--gpus',a.gpus.split(',')[0],'--max-runtime','2400'],'deployment_smoke')
        guard=runner.read_json(root/'native_guard/status.json')
        accept=runner.read_json(root/'native_deployment/deployment_acceptance.json')
        parity=runner.read_json(root/'native_deployment/native_motor_parity.json')
        if guard.get('state')!='completed' or not guard.get('guard_closed') or not accept.get('passed') or not parity.get('passed'):
            raise RuntimeError('native GPU acceptance not complete')
        evidence=[]; accepted_identity=None
        for phase,c in zip(PHASES,configs):
            _,_,result,_=phase_completed(Path(c['output']),c)
            identity={k:result['identity'][k] for k in ('source_sha256','manifest_sha256','init_checkpoint_sha256')}
            if identity['source_sha256']!=source['source_sha256'] or (accepted_identity is not None and identity!=accepted_identity):
                raise RuntimeError('smoke phases disagree on source, dataset or initial checkpoint')
            accepted_identity=identity
            for f in [Path(c['output'])/'result.json',Path(c['output'])/'reload.json',root/'smoke_sequence'/phase/'status.json']:
                evidence.append(dict(path=str(f),sha256=file_sha256(f)))
        for f in (root/'native_deployment').rglob('*'):
            if f.is_file() and f.suffix in ('.json','.mp4'):evidence.append(dict(path=str(f),sha256=file_sha256(f)))
        evidence.append(dict(path=str(root/'native_guard/status.json'),sha256=file_sha256(root/'native_guard/status.json')))
        receipt=dict(passed=True,world_size=8,**accepted_identity,
            supervisor_sha256=source['distributed_supervisor_sha256'],source_contract=runner.source_contract(source),
            profiles={p:runner.profile_sha256(c) for p,c in zip(PHASES,configs)},evidence=evidence,
            native_deployment=accept,quality_accepted=False)
        ap=root/'acceptance.json';atomic(ap,receipt)
        formal=[]
        for phase,c,steps in zip(PHASES,configs,(1000,10000,20000)):
            formal.append(dict(c,run_kind='formal_distributed_training',steps=steps,
                output='/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/'+'native_v6_'+root.name+'_formal/'+phase,
                sequence_id=root.name+'_formal',acceptance_receipt=str(ap),acceptance_receipt_sha256=file_sha256(ap)))
        status('acceptance_passed',acceptance=str(ap))
        sequence(formal,'formal_distributed_training','formal_sequence')
        status('completed')
    except BaseException as e:
        if child is not None and child.poll() is None:
            if proc_identity(child.pid)['start_ticks']==child_ticks:
                child.send_signal(signal.SIGTERM)
                for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
                child.wait()
        status('failed',error=repr(e));raise

if __name__=='__main__':main()

