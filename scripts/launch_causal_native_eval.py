#!/usr/bin/env python3
"""One-shot final checkpoint -> Graph -> native 8x2 smoke -> 35x20, no retries."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from supervise_motionkv_training import atomic_json,file_digest,driver_identity,source_files_and_hash,utc
from supervise_motionkv_eval import proc_gpu_inventory
from eval_kernel_log import driver_evidence
from run_joint_motionkv_smokes import require_release
SSD=Path('/SSD_DISK/users/wuruihan/DynamicWAM')
CKPT=SSD/'outputs/training/causal_spatial_v4_joint_20261021/exports/causal_spatial_step_20000.pt'
SHA='d0b5cf4c0d94e3a02c2d604f53b7c25c097f001dc4e47d35704db2dc2c8080ec'
OLD=SSD/'outputs/diagnostics/causal_spatial_v4_20261021/formal_v4_v4_joint/guard'
PYTHON=str(ROOT/'.venv/bin/python');ROBOT=str(ROOT/'external/robotwin/bin/python')

def initial_acceptance(run,formal,proc,baseline):
    started=time.monotonic();deadline=started+7200;decoded=set()
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            if proc.returncode:raise RuntimeError('formal exited before initial acceptance')
            return dict(passed=True,formal_completed_before_acceptance=True)
        try:
            state=json.loads((formal/'status.json').read_text());guard=json.loads((run/'formal_guard/status.json').read_text())
        except FileNotFoundError:time.sleep(3);continue
        running=state.get('running',{})
        for task in running:
            receipt=formal/'receipts'/task/'episodes.json'
            if task in decoded or not receipt.is_file():continue
            episodes=json.loads(receipt.read_text())
            if not episodes:continue
            from dynamicwam.evaluation.run_domino_parallel_eval import _reported_float
            first=episodes[0]
            if not isinstance(first.get('success'),bool) or _reported_float(first,('manipulation_score','ms')) is None:raise RuntimeError('invalid first episode')
            videos=list((formal/'eval_result'/task).glob('*/*/*/*/episode0.mp4'))
            if len(videos)!=1:continue
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(videos[0]),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=600)
            decoded.add(task)
        if guard.get('startup_health_window_closed') and guard.get('runtime_seconds',0)>=300 and len(decoded)>=8 and len(running)==8:
            for row in running.values():os.kill(row['pid'],0)
            now=driver_evidence()
            if driver_identity(now)!=driver_identity(baseline):raise RuntimeError('driver delta at initial acceptance')
            return dict(passed=True,at=utc(),seconds=time.monotonic()-started,first_episodes=len(decoded),native_mp4_decoded=sorted(decoded),workers=running,
                further_observation=False,driver_unchanged=True)
        time.sleep(5)
    return dict(passed=False,state='initial_acceptance_timeout',formal_kept_running=proc.poll() is None,seconds=time.monotonic()-started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();run=Path(a.run).resolve()
    if not run.is_relative_to(SSD):raise RuntimeError('SSD output required')
    if not a.worker:
        run.mkdir(parents=True,exist_ok=False)
        with (run/'controller.log').open('x') as log:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--run',str(run),'--worker'],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(run/'launch.json',dict(pid=child.pid,created_at=utc()));print(json.dumps(dict(pid=child.pid,status=str(run/'status.json'))));return
    active=None;status=dict(state='preparing',started_at=utc(),pid=os.getpid(),automatic_retry=False)
    def update(**values):status.update(values,updated_at=utc());atomic_json(run/'status.json',status)
    def interrupt(sig,frame):raise RuntimeError('controller interrupted '+str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        update();require_release(OLD/'status.json',require_success=True,label='completed .50 training')
        cpu_root=SSD/'outputs/diagnostics/causal_native_eval_20261026'
        cpu=json.loads((cpu_root/'cpu_acceptance.json').read_text())
        export=json.loads((cpu_root/'final_export_audit.json').read_text())
        if cpu.get('passed') is not True or export.get('passed') is not True:raise RuntimeError('actual final export and native CPU gates required')
        baseline=driver_evidence()
        if driver_identity(baseline)!=driver_identity(json.loads((OLD/'driver_postexit.json').read_text())):raise RuntimeError('driver changed since training clean release')
        if file_digest(CKPT)!=SHA:raise RuntimeError('final checkpoint SHA mismatch')
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise RuntimeError('freeze source before GPU')
        source=source_files_and_hash(ROOT)[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        ids=[r['uuid'] for r in proc_gpu_inventory()];recipe=ROOT/'configs/causal_spatial/joint.yaml'
        auth=run/'authorization.json';atomic_json(auth,dict(created_at=utc(),driver=baseline,gpu_uuids=ids,
            guard_sha256=file_digest(ROOT/'scripts/device_lifetime_guard.py'),device_lifetime_mode='fds_before_telemetry_v1',
            source_sha256=source,git_head=head,config_sha256=file_digest(recipe),
            user_scope='Final .50 checkpoint native DOMINO evaluation, retain trained H2/R2 RTC prefix; no continuous simulation.'))
        # Hash actual shared DOMINO sources without mutating the checkout.
        domino=ROOT/'external/DOMINO';files={str(p.relative_to(domino)):file_digest(p) for sub in ['envs','script','task_config'] for p in (domino/sub).rglob('*') if p.is_file() and p.suffix in ('.py','.yml','.yaml')}
        atomic_json(run/'domino_source_hashes.json',files)
        update(checkpoint=str(CKPT),checkpoint_sha256=SHA,head=head,source_sha256=source)
        for stage in ['graph','smoke','formal']:
            if source_files_and_hash(ROOT)[1]!=source:raise RuntimeError('evaluation source changed')
            if any(file_digest(domino/name)!=digest for name,digest in files.items()):raise RuntimeError('shared DOMINO source changed')
            guard=run/(stage+'_guard');compute=ids[:1] if stage=='graph' else ids
            command=[PYTHON,str(ROOT/'scripts/supervise_motionkv_eval.py'),'--stage',stage,'--run-dir',str(guard),
                '--gpus',','.join(compute),'--guard-gpus',','.join(ids),'--authorization',str(auth),'--startup-seconds','300',
                '--max-runtime-seconds',str(7200 if stage=='graph' else 86400*3)]
            if stage=='graph':
                command+=['--single-worker','--',ROBOT,str(ROOT/'scripts/check_causal_spatial_gpu.py'),'--config',str(recipe),'--checkpoint',str(CKPT),'--output',str(run/'graph.json')]
            else:
                command+=['--',PYTHON,str(ROOT/'scripts/run_causal_native_suite.py'),'--run',str(run/stage),'--checkpoint',str(CKPT),'--gpus',','.join(ids)]
                if stage=='smoke':command+=['--smoke']
            atomic_json(run/(stage+'_command.json'),command)
            with (run/(stage+'_supervisor.log')).open('x') as log:
                active=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
                update(state=stage+'_running',supervisor_pid=active.pid)
                if stage=='formal':
                    receipt=initial_acceptance(run,run/'formal',active,baseline)
                    atomic_json(run/'initial_acceptance.json',receipt);update(initial_acceptance=receipt)
                    # No further health observer. Only wait for required terminal cleanup.
                code=active.wait();active=None
            require_release(guard/'status.json',require_success=code==0,label=stage)
            if code:raise RuntimeError(stage+' failed; no retry')
            if stage=='graph' and json.loads((run/'graph.json').read_text()).get('passed') is not True:raise RuntimeError('Graph not accepted')
            if stage in ('smoke','formal'):
                result=json.loads((run/stage/'aggregate.json').read_text());expected=16 if stage=='smoke' else 700
                if not result.get('complete') or result['total']!=expected:raise RuntimeError('native result count mismatch')
        update(state='completed',passed=True,finished_at=utc())
    except BaseException:
        update(state='failed_no_retry',passed=False,error=traceback.format_exc())
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=180)
            except subprocess.TimeoutExpired:update(cleanup_pending=True)
        raise

if __name__=='__main__':main()
