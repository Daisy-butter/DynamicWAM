#!/usr/bin/env python3
"""One owned GPU lifecycle, fresh evidence, no retry or driver intervention."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from supervise_motionkv_training import atomic_json,file_digest,driver_identity,source_files_and_hash,utc
from eval_kernel_log import driver_evidence
from supervise_motionkv_eval import proc_gpu_inventory
from run_joint_motionkv_smokes import training_environment,require_release
from train_causal_spatial import load_recipe
SSD=Path('/SSD_DISK/users/wuruihan/DynamicWAM')
OUT=SSD/'outputs/diagnostics/causal_spatial_v4_20261021'
PYTHON=str(ROOT/'.venv/bin/python')

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['training','graph','domino'],required=True)
 p.add_argument('--label',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint')
 p.add_argument('--previous-guard',required=True);p.add_argument('--detach',action='store_true');p.add_argument('--worker',action='store_true')
 p.add_argument('--deploy-config',default=str(ROOT/'configs/causal_spatial/deploy.yaml'))
 p.add_argument('--software-fix-reason',default='')
 p.add_argument('--formal-scene-manifest')
 a=p.parse_args();run=OUT/a.label
 if not a.label.replace('_','').replace('-','').isalnum():raise ValueError('invalid label')
 if a.detach:
  if run.exists():raise FileExistsError('preserve previous attempt')
  run.mkdir(parents=True)
  argv=[PYTHON,str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v!='--detach']+['--worker']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  with (run/'controller.log').open('x') as f:
   child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(run/'launch.json',dict(pid=child.pid,created_at=utc(),command=argv))
  print(json.dumps(dict(pid=child.pid,status=str(run/'status.json'))));return
 if not a.worker:raise ValueError('use --detach for durable lifecycle')
 r=load_recipe(a.config);active=None;status=dict(state='preparing',pid=os.getpid(),created_at=utc(),automatic_retry=False)
 def update(**v):status.update(v,updated_at=utc());atomic_json(run/'status.json',status)
 def stop(sig,_frame):raise RuntimeError('owned controller interrupted '+str(sig))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  update();previous=Path(a.previous_guard)
  require_release(previous/'status.json',require_success=not bool(a.software_fix_reason),label='previous GPU lifecycle')
  now=driver_evidence();old=json.loads((previous/'driver_postexit.json').read_text())
  if driver_identity(now)!=driver_identity(old):raise RuntimeError('driver evidence changed since clean release; stop for investigation')
  ids=[x['uuid'] for x in proc_gpu_inventory()]
  if len(ids)!=8:raise RuntimeError('all eight adapters required')
  source=source_files_and_hash(ROOT)[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
  if dirty:raise RuntimeError('freeze and commit candidate source before GPU execution')
  if a.software_fix_reason:
   prior=json.loads((previous/'manifest.json').read_text())
   if prior['git_head']==head:raise RuntimeError('a failed lifecycle needs a specific new software fix, not a retry')
   update(previous_failure_fix_reason=a.software_fix_reason,previous_source=prior['git_head'],current_source=head)
  if a.mode=='training' and not r['diagnostic']:
   acceptance_path=os.environ.get('CAUSAL_SPATIAL_FORMAL_ACCEPTANCE')
   if not acceptance_path:raise RuntimeError('formal training requires verified one-shot sequence acceptance')
   acceptance=json.loads(Path(acceptance_path).read_text())
   if acceptance.get('format')!='causal_spatial_formal_acceptance_v1' or acceptance.get('passed') is not True:
    raise RuntimeError('formal acceptance absent or invalid')
   if acceptance.get('source_sha256')!=source or acceptance.get('git_head')!=head:
    raise RuntimeError('formal acceptance source identity mismatch')
   if acceptance['configs'][r['phase']]['sha256']!=file_digest(a.config):raise RuntimeError('formal recipe changed')
  audit=json.loads((OUT/'data_audit.json').read_text());cpu=json.loads((OUT/'cpu_integration.json').read_text())
  if not audit['passed'] or not cpu['passed']:raise RuntimeError('full data and actual CPU integration gates required')
  auth=run/'authorization.json'
  atomic_json(auth,dict(created_at=utc(),driver=now,gpu_uuids=ids,guard_sha256=file_digest(ROOT/'scripts/device_lifetime_guard.py'),
   device_lifetime_mode='fds_before_telemetry_v1',user_scope='User authorized causal-spatial architecture, real smoke, then eight-GPU formal training after acceptance.',
   not_administrator_clearance=True,source_sha256=source,git_head=head,config_sha256=file_digest(a.config)))
  env,nccl=training_environment()
  if a.mode=='training':
   cmd=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+','.join(ids),'ACCELERATE_USE_DEEPSPEED=true','ACCELERATE_MIXED_PRECISION=bf16',
    PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(ROOT/'scripts/train_causal_spatial.py'),'--config',str(Path(a.config).resolve())]
   kind='smoke' if r['diagnostic'] else 'formal';compute=ids
  elif a.mode=='graph':
   cmd=[str(ROOT/'external/robotwin/bin/python'),str(ROOT/'scripts/check_causal_spatial_gpu.py'),'--config',str(Path(a.config).resolve()),'--output',str(run/'graph.json')]
   if a.checkpoint:cmd+=['--checkpoint',a.checkpoint]
   kind='graph';compute=ids[:1]
  else:
   compute=ids[:2]
   render=next(row for row in proc_gpu_inventory() if row['uuid']==compute[0])
   # env exec preserves the worker PID/guard parent relationship. The existing
   # all-eight guard still precedes any CUDA/Vulkan initialization. Without
   # --single-worker the supervisor leaves CUDA visibility empty; set both
   # ordered compute UUIDs explicitly for this DOMINO-only worker command.
   cmd=['/usr/bin/env','CUDA_VISIBLE_DEVICES='+','.join(compute),
    'DYNAMICWAM_GPU_PCI='+render['pci_bus_id'],'DYNAMICWAM_SIM_CUDA=0','DYNAMICWAM_INFER_CUDA=1',
    str(ROOT/'external/robotwin/bin/python'),str(ROOT/'scripts/smoke_causal_spatial_domino.py'),'--config',str(Path(a.deploy_config).resolve()),
    '--checkpoint',a.checkpoint,'--output-dir',str(run/'domino'),'--allow-diagnostic','--cuda-graph']
   kind='smoke'
   if a.formal_scene_manifest:cmd+=['--formal-scene-manifest',str(Path(a.formal_scene_manifest).resolve(strict=True))]
  command=[PYTHON,str(ROOT/'scripts/supervise_motionkv_eval.py'),'--stage',kind,'--run-dir',str(run/'guard'),
   '--gpus',','.join(compute),'--guard-gpus',','.join(ids),'--authorization',str(auth),
   '--startup-seconds','300','--max-runtime-seconds',str(7200 if kind!='formal' else 1209600)]
  if len(compute)==1:command+=['--single-worker']
  command+=['--']+cmd
  atomic_json(run/'command.json',dict(command=command,nccl=nccl,source_sha256=source,head=head,recipe=r))
  with (run/'supervisor.log').open('x') as f:
   active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   update(state='running',supervisor_pid=active.pid);code=active.wait();active=None
  require_release(run/'guard/status.json',require_success=code==0,label=a.label)
  if code:raise RuntimeError('worker failed, evidence preserved; no retry')
  if source_files_and_hash(ROOT)[1]!=source:raise RuntimeError('source changed during execution')
  if a.mode=='training':
   update(state='strict_export_audit')
   command=[PYTHON,str(ROOT/'scripts/check_causal_spatial_export.py'),'--config',str(Path(a.config).resolve()),'--output',str(run/'export_audit.json')]
   with (run/'export_audit.log').open('x') as f:subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
  update(state='completed',passed=True,finished_at=utc())
 except BaseException:
  update(state='failed_no_retry',passed=False,error=traceback.format_exc())
  if active is not None and active.poll() is None:
   active.terminate()
   try:active.wait(timeout=180)
   except subprocess.TimeoutExpired:update(cleanup_pending=True)
  raise
if __name__=='__main__':main()
