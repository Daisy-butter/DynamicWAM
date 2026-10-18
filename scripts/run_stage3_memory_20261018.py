"""Explicit one-shot eight-GPU Stage3 memory-fix smoke -> strict audit -> formal."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/SSD_DISK/users/wuruihan/DynamicWAM')
OUT = SSD / 'outputs/diagnostics/stage3_memory_20261018_attempt02'
OLD = SSD / 'outputs/diagnostics/retention_recipe_v33_20261016/formal_pipeline'
GRID = SSD / 'outputs/data/joint_motionkv_physical_grid_20261015'
STAGE2_AUDIT = OLD / 'stage2/export_audit.json'
OLD_RELEASE = SSD / 'outputs/diagnostics/stage3_memory_20261018/smoke/guard'
FORMAL_AUDITOR = OLD / 'audit_formal_export.py'
PYTHON = str(ROOT / '.venv/bin/python')
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from run_joint_motionkv_smokes import frozen_identity, require_frozen, require_release, training_environment
from supervise_motionkv_training import atomic_json, driver_evidence, driver_identity, file_digest, utc
from supervise_motionkv_eval import proc_gpu_inventory


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def cpu_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')


def accept_audit(path, budget, expected_full=None):
    value = read(path)
    for field in ('passed','checkpoint_reload','contracts_verified','full_fast_tensors_equal','weights_finite','log_finite'):
        require(value.get(field) is True, 'strict export audit not accepted: '+field)
    require(value.get('full_strict_reload',{}).get('passed') is True
        and value.get('fast_strict_reload',{}).get('passed') is True, 'actual full/fast strict reload missing')
    require(value['hyperparameters']['global_step']==budget, 'audited optimizer step differs')
    for item in value['artifacts'].values():
        p=Path(item['path']);require(p.resolve().is_relative_to(SSD.resolve()),'artifact outside SSD')
        require(p.is_file() and p.stat().st_size==item['bytes'] and file_digest(p)==item['sha256'],
            'audited artifact changed: '+str(p))
    if expected_full is not None:
        require(Path(value['artifacts']['full']['path']).resolve()==Path(expected_full).resolve(), 'audited full export differs')
    return value


def profiles():
    from dynamicwam.config.multirate import load_multirate_profile
    stages=[]
    for label,name,steps in [('smoke','joint_motionkv_v33_stage3_mb8_smoke_20261018.yaml',4),
                             ('formal','joint_motionkv_v33_stage3_mb8_20261018.yaml',40000)]:
        path=ROOT/'configs/multirate'/name;p=load_multirate_profile(path).training
        require(p['max_steps']==steps and not p['resume_from'] and p['phase']=='cross_time_joint','fresh joint Stage3 budget required')
        require(0 <= p['warmup_steps'] < p['max_steps'], 'warmup must fit the actual stage budget before GPU initialization')
        require(p['per_device_batch_size']==8 and p['gradient_accumulation_steps']==2,'must test exact microbatch8/accum2/global128')
        require(p['dataset'].get('smoke_subset') is False and p['dataset']['kind']=='physical_grid','both stages require complete physical-grid scope')
        require(Path(p['dataset']['physical_grid_root']).resolve()==GRID.resolve()
            and Path(p['dataset']['future_latent_root']).resolve()==(GRID/'future_latents').resolve(),'complete native grid/cache required')
        run=(Path(p['checkpoint_dir'])/p['name']).resolve()
        require(run.is_relative_to(SSD.resolve()),'training outputs must remain on SSD')
        require(steps % p['log_interval']==0,'final step must be present in logs')
        stages.append(dict(label=label,config=str(path),budget=steps,profile=p,training_run=str(run),
            initialization=p['model']['initial_checkpoint'],diagnostic_only=label=='smoke',
            full_export=str(run/'exports'/f'multirate_step_{steps}.pt'),
            fast_export=str(run/'exports'/f'multirate_fast_step_{steps}.pt')))
    smoke,formal=stages
    allowed={'name','max_steps','warmup_steps','log_interval','checkpoint_interval','validation_batches'}
    require({k:v for k,v in smoke['profile'].items() if k not in allowed}
            == {k:v for k,v in formal['profile'].items() if k not in allowed},'smoke/formal must match except explicit diagnostic budgets/log cadence')
    require(smoke['training_run']!=formal['training_run'],'diagnostic and formal outputs must differ')
    return stages


def fresh_destinations(stages):
    for stage in stages:
        require(not Path(stage['training_run']).exists(),'preserve existing training run')
        require(not (OUT/stage['label']).exists(),'preserve old controller stage receipts')


def prepare():
    require(not (OUT/'plan.json').exists(),'plan already sealed; no overwrite')
    stages=profiles();fresh_destinations(stages)
    stage2=accept_audit(STAGE2_AUDIT,80000)
    init=stage2['artifacts']['full']
    require(all(Path(s['initialization']).resolve()==Path(init['path']).resolve() for s in stages),
        'both smoke and formal must initialize formal Stage2 step80000; smoke init forbidden')
    require_release(OLD/'stage2/guard/status.json',require_success=True,label='formal Stage2')
    failed=require_release(OLD_RELEASE/'status.json',require_success=False,label='failed Stage3 clean release')
    require(failed['final_exit_code']==1,'expected explicitly corrected failed Stage3 lifecycle')
    identity=frozen_identity(stages)
    manifest=read(OLD_RELEASE/'manifest.json');ids=manifest['guard_gpu_uuids']
    require(len(ids)==8 and len(set(ids))==8,'all eight original adapter UUIDs required')
    _,nccl=training_environment()
    require(not nccl.get('NCCL_P2P_DISABLE'),'default validated NCCL required')
    plan=dict(created_at=utc(),stages=stages,frozen_source=identity,gpu_uuids=ids,
        initial_checkpoint=init,stage2_audit_sha256=file_digest(STAGE2_AUDIT),
        prior_release_sha256=file_digest(OLD_RELEASE/'status.json'),
        prior_postexit_sha256=file_digest(OLD_RELEASE/'driver_postexit.json'),
        physical_grid_manifest_sha256=file_digest(GRID/'manifest.json'),
        latent_manifest_sha256=file_digest(GRID/'future_latents/manifest.json'),
        external_dependencies={str(FORMAL_AUDITOR):file_digest(FORMAL_AUDITOR)},nccl_environment=nccl,
        startup_health_seconds=300,postwindow_gpu_polling=False,automatic_retry=False,
        full_data_hash_revalidation_pending=True,formal_initializes_from_smoke=False,
        correction='Explicit 2026-10-18 Stage3 OOM correction by microbatch8/accum2, preserving global batch128 and the model/objective.',
        formal_max_runtime_seconds=1209600,source_script_sha256=file_digest(__file__))
    atomic_json(OUT/'plan.json',plan)
    print(json.dumps(dict(prepared=True,plan=str(OUT/'plan.json'),source_commit=identity['git_head'],gpu_started=False)))


def check_frozen(plan):
    require_frozen(plan['stages'],plan['frozen_source'])
    require(file_digest(__file__)==plan['source_script_sha256'],'controller changed after freeze')
    for p,h in plan['external_dependencies'].items():require(file_digest(p)==h,'external audit helper changed')
    for p,key in [(STAGE2_AUDIT,'stage2_audit_sha256'),(OLD_RELEASE/'status.json','prior_release_sha256'),
        (OLD_RELEASE/'driver_postexit.json','prior_postexit_sha256'),(GRID/'manifest.json','physical_grid_manifest_sha256'),
        (GRID/'future_latents/manifest.json','latent_manifest_sha256')]:
        require(file_digest(p)==plan[key],'bound evidence changed: '+str(p))


def fresh_auth(plan,label,previous):
    require_release(previous/'status.json',require_success=label!='smoke',label='previous clean lifecycle')
    now=driver_evidence();require(driver_identity(now)==driver_identity(read(previous/'driver_postexit.json')),'driver changed since last final clean release')
    require({r['uuid'] for r in proc_gpu_inventory()}==set(plan['gpu_uuids']),'fresh all-adapter inventory differs')
    path=OUT/(label+'_authorization.json');require(not path.exists(),'authorization must be fresh')
    atomic_json(path,dict(created_at=utc(),driver=now,gpu_uuids=plan['gpu_uuids'],
        guard_sha256=file_digest(ROOT/'scripts/device_lifetime_guard.py'),device_lifetime_mode='fds_before_telemetry_v1',
        user_scope=plan['correction']+' Actual successful smoke and strict reload gate formal restart.',
        not_administrator_clearance=True,previous_guard=str(previous),
        previous_release_sha256=file_digest(previous/'status.json'),pipeline_plan_sha256=file_digest(OUT/'plan.json')))
    return path


def commands(stage,plan,auth,env):
    out=OUT/stage['label'];ids=','.join(plan['gpu_uuids']);guard=out/'guard'
    command=[PYTHON,str(ROOT/'scripts/supervise_motionkv_eval.py'),'--stage',stage['label'],
        '--run-dir',str(guard),'--gpus',ids,'--guard-gpus',ids,'--authorization',str(auth),
        '--startup-seconds','300','--max-runtime-seconds',str(7200 if stage['diagnostic_only'] else plan['formal_max_runtime_seconds']),'--',
        '/usr/bin/env','CUDA_VISIBLE_DEVICES='+ids,'LD_LIBRARY_PATH='+env.get('LD_LIBRARY_PATH',''),
        'ACCELERATE_MIXED_PRECISION=bf16','ACCELERATE_USE_DEEPSPEED=true','PYTHONDONTWRITEBYTECODE=1',
        PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(Path(__file__).resolve()),
        '--rank-worker','--config',stage['config'],'--memory-dir',str(guard)]
    auditor=ROOT/'scripts/check_motionkv_smoke_export.py' if stage['diagnostic_only'] else FORMAL_AUDITOR
    audit=[PYTHON,str(auditor),'--config',stage['config'],'--full',stage['full_export'],'--fast',stage['fast_export'],
        '--log',str(guard/'run.log'),'--expected-steps',str(stage['budget']),'--output',str(out/'export_audit.json')]
    return command,audit


def rank_worker(config,memory_dir):
    require(memory_dir.resolve()==Path(os.environ.get('WAM_TRAINING_RUN_DIR','/missing')).resolve(),'rank wrapper requires the active owned guard')
    sys.argv=[str(ROOT/'scripts/train_multirate.py'),'--config',str(config)]
    outcome='failed'
    try:
        runpy.run_path(str(ROOT/'scripts/train_multirate.py'),run_name='__main__')
        outcome='completed'
    except SystemExit as error:
        if error.code in (None,0):outcome='completed'
        raise
    finally:
        import torch
        report=dict(rank=int(os.environ['RANK']),local_rank=int(os.environ['LOCAL_RANK']),outcome=outcome,
            scope='owned process CUDA allocator statistics at exit; no NVML or background telemetry')
        if torch.cuda.is_initialized():
            try:
                stats=torch.cuda.memory_stats(torch.cuda.current_device())
                report.update(cuda_initialized=True,peak_allocated_bytes=stats.get('allocated_bytes.all.peak'),
                    peak_reserved_bytes=stats.get('reserved_bytes.all.peak'),num_ooms=stats.get('num_ooms'))
            except Exception as error:report.update(cuda_initialized=True,statistics_error=repr(error))
        else:report['cuda_initialized']=False
        atomic_json(memory_dir/f'memory_rank_{report["rank"]}.json',report)


def worker():
    plan=read(OUT/'plan.json');active=None;state=dict(state='validating',started_at=utc(),pid=os.getpid(),
        completed=[],automatic_retry=False,postwindow_gpu_polling=False)
    def update(**kw):state.update(kw,updated_at=utc());atomic_json(OUT/'status.json',state)
    def stop(signum,_frame):raise RuntimeError('controller received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        check_frozen(plan);fresh_destinations(plan['stages']);update()
        accept_audit(STAGE2_AUDIT,80000,plan['initial_checkpoint']['path'])
        from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
        readiness=validate_physical_grid_readiness(GRID,GRID/'future_latents',verify_hashes=True,require_full=True)
        require(readiness['samples']==1416299 and readiness['episodes']==10500 and readiness['full_scope'] is True,'full grid coverage required')
        atomic_json(OUT/'full_data_verified.json',readiness)
        import torch
        payload=torch.load(plan['initial_checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
        contract=payload['config']['runtime_contract']
        require(contract['training_scope']=='full' and contract['physical_grid']==readiness['physical_grid_identity']
            and contract['physical_grid_contract']['latent_manifest_sha256']==readiness['latent_manifest_sha256'],
            'Stage2 initialization does not bind actual full native dataset/cache')
        del payload
        env,nccl=training_environment();require(nccl==plan['nccl_environment'],'NCCL environment changed')
        previous=OLD_RELEASE
        for stage in plan['stages']:
            check_frozen(plan)
            require(file_digest(stage['initialization'])==plan['initial_checkpoint']['sha256'],'both stages must use unchanged formal Stage2 weights')
            if stage['label']=='formal':
                require(state['completed']==['smoke'],'successful diagnostic required before formal')
                accept_audit(OUT/'smoke/export_audit.json',4,plan['stages'][0]['full_export'])
            out=OUT/stage['label'];out.mkdir(exist_ok=False)
            auth=fresh_auth(plan,stage['label'],previous);command,audit=commands(stage,plan,auth,env)
            atomic_json(out/'commands.json',dict(training=command,strict_audit=audit,diagnostic_only=stage['diagnostic_only'],
                initialization=plan['initial_checkpoint'],formal_initializes_from_smoke=False))
            update(state=stage['label']+'_running',active_stage=stage['label'])
            with (out/'supervisor.log').open('x') as stream:
                active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                update(supervisor_pid=active.pid);code=active.wait();active=None
            require(code==0,stage['label']+' failed; no retry or next stage')
            previous=out/'guard';require_release(previous/'status.json',require_success=True,label=stage['label'])
            memory=[read(previous/f'memory_rank_{rank}.json') for rank in range(8)]
            require(all(m['rank']==rank and m['outcome']=='completed' and m.get('num_ooms')==0
                and m.get('peak_allocated_bytes',0)>0 and m.get('peak_reserved_bytes',0)>0
                for rank,m in enumerate(memory)),'all eight ranks need successful measured allocation peaks and zero OOMs')
            atomic_json(out/'memory_summary.json',dict(ranks=memory,max_peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in memory),
                max_peak_reserved_bytes=max(m['peak_reserved_bytes'] for m in memory),all_ranks_completed=True,all_ranks_zero_oom=True))
            check_frozen(plan);update(state=stage['label']+'_strict_cpu_audit')
            with (out/'audit.log').open('x') as stream:
                subprocess.run(audit,cwd=ROOT,env=cpu_env(),stdout=stream,stderr=subprocess.STDOUT,check=True)
            accept_audit(out/'export_audit.json',stage['budget'],stage['full_export'])
            state['completed'].append(stage['label']);update(supervisor_pid=None)
        check_frozen(plan);update(state='formal_stage3_complete',ended_at=utc(),active_stage=None)
        atomic_json(OUT/'complete.json',dict(passed=True,at=utc(),plan_sha256=file_digest(OUT/'plan.json'),
            formal_audit_sha256=file_digest(OUT/'formal/export_audit.json'),evaluation_started=False))
    except BaseException as error:
        update(state='failed_no_retry',error=repr(error),ended_at=utc())
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=180)
            except subprocess.TimeoutExpired:update(cleanup_incomplete=True,supervisor_pid=active.pid)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    mode.add_argument('--rank-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--config',type=Path,help=argparse.SUPPRESS);parser.add_argument('--memory-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.rank_worker:rank_worker(args.config,args.memory_dir);return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare();return
    if args.worker:worker();return
    plan=read(OUT/'plan.json');check_frozen(plan)
    require(not (OUT/'launch.json').exists() and not (OUT/'status.json').exists(),'one-shot controller already used; preserve evidence')
    if args.detach:
        with (OUT/'pipeline.log').open('x') as stream:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=cpu_env(),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/'launch.json',dict(pid=child.pid,at=utc(),plan_sha256=file_digest(OUT/'plan.json'),automatic_retry=False))
        print(json.dumps(dict(pid=child.pid,status=str(OUT/'status.json'),state='dispatched_not_gpu_accepted')))
    else:
        atomic_json(OUT/'launch.json',dict(pid=os.getpid(),at=utc(),plan_sha256=file_digest(OUT/'plan.json'),automatic_retry=False))
        worker()


if __name__=='__main__':main()
