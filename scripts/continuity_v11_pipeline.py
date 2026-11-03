"""Finite V11 collection -> measured data -> 8-rank smoke -> Full training -> 700 eval.
All failures stop dependent stages. No automatic retry or driver operations.
"""
import argparse,copy,hashlib,json,math,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src')]
import guarded_dualrate_run as guard
from dualrate_training_sequence import phase_completed
from dualrate_continuous_acceptance import validate_continuous_report
PHASES=['warmup','small_age_joint','cross_time_joint'];STEPS=[500,4000,8000]

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True);p.add_argument('--attempt',default='pipeline01');p.add_argument('--manual-start',required=True,action='store_true');a=p.parse_args()
 if Path(a.attempt).name!=a.attempt or not a.attempt.startswith('pipeline'):raise ValueError('local unique pipeline attempt required')
 root=Path(a.root).resolve();out=root/a.attempt;out.mkdir(exist_ok=False)
 R=REPO.parents[1];children=[];evidence={};source=guard.frozen_source(REPO);began=time.monotonic()
 def state(s,**kw):save(out/'status.json',dict(state=s,pid=os.getpid(),elapsed=time.monotonic()-began,no_retry=True,source=source,**kw))
 def ev(p):
  p=Path(p).resolve(strict=True);evidence[str(p)]=dict(path=str(p),sha256=sha(p));return p
 def stop(sig,frame):raise InterruptedError(sig)
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 base=read(root/'initial_suite.template.json')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(REPO/'src')+':'+base['python_dependency_overlay'])
 cpu=dict(env,CUDA_VISIBLE_DEVICES='')
 def launch(cmd,name,environment=env):
  if guard.frozen_source(REPO)!=source:raise RuntimeError('source changed')
  with (out/(name+'.log')).open('x') as f:
   c=subprocess.Popen(cmd,cwd=REPO,env=environment,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children.append(c);return c
 def wait(c,name):
  while c.poll() is None:
   if any(x.poll() not in (None,0) for x in children):raise RuntimeError('parallel child failed')
   time.sleep(1)
  if c.returncode:raise RuntimeError(name+' failed; no next phase')
  children.remove(c)
 def run(cmd,name,environment=env):wait(launch(cmd,name,environment),name)
 def clean(g):
  g=Path(g);s=read(g/'status.json');k=read(g/'kernel_after.json');x=read(g/'owned_exit.json')
  assert s['state']=='completed' and s['guard_closed'] and not s['errors']
  assert k['unchanged'] and k['same_boot'] and not k.get('added') and not x['remaining'] and x['returncode']==0
  for n in ('status.json','kernel_after.json','owned_exit.json','source.json','config.frozen.json'):ev(g/n)
 def guarded(c,name,gpus):
  path=out/(name+'.config.json');save(path,c)
  return launch([sys.executable,str(REPO/'scripts/guarded_dualrate_run.py'),'--config',str(path),'--run-dir',str(out/(name+'_guard')),'--gpus',','.join(gpus),'--max-runtime','86400'],name)
 try:
  state('validating_initial')
  initial=read(ev(root/(a.initial+'.status.json')));assert initial['state']=='initial_gpu_smoke_completed'
  # The added finite controller does not alter source/model/runtime code.
  assert initial['source']['source_sha256']==source['source_sha256']
  clean(root/(a.initial+'_guard'));ic=read(ev(root/(a.initial+'.config.json')))
  initial_model=read(ev(root/(a.initial+'_initial/initialization_receipt.json')))
  assert initial_model['trained_steps']==0 and not initial_model['cuda_initialized']
  all_tasks=sorted(read(root/'training_scene_pool.json')['tasks'])
  records=[];configs=[];state('collecting_train_feedback')
  # One three-GPU collector: SAPIEN creates a small graphics probe on physical
  # GPU0 even in another visibility group. Two independent guards must not
  # declare each other idle. Scored eval retains its unified eight-GPU guard.
  c=read(root/'collection.template.json');c['deployment']['tasks']=all_tasks
  c.update(output=str(out/'collection_0'),source_sha256=source['source_sha256'],training_source_sha256=source['source_sha256'],deployment_checkpoint=initial_model['checkpoint'],deployment_checkpoint_sha256=initial_model['sha256'])
  configs.append(c);records.append((guarded(c,'collection_0',base['gpu_uuids'][:3]),'collection_0',c))
  traces=[]
  for child,name,c in records:
   if child is not None:wait(child,name)
   clean(out/(name+'_guard'))
   accepted=read(ev(Path(c['output'])/'deployment_acceptance.json'));assert accepted['hardtime_acceptance_passed']
   for path in accepted['reports']:
    report=read(ev(path));validate_continuous_report(report,c)
    assert report['continuous_valid'] and report['realtime_valid'] and report['actuator_tracking']['actual_physical_tracking']
    traces.append(path)
  assert len(traces)==35
  # Fresh paired identical Full commands under both controllers, contacts ON.
  state('paired_full_tracking')
  tc=read(R/'outputs/diagnostics/residual_v10_20261102/tracking09.json')
  tc=json.loads(json.dumps(tc).replace(str(R/'worktrees/dualrate-v8-rtc-20261031'),str(REPO)))
  tc['output']=str(out/'tracking');tc['residual_v10']=base['residual_v10']
  tc['cases']=[dict(row) for row in tc['cases']]+[dict(row,label=row['label']+'_linear_v11',controller='linear') for row in tc['cases']]
  child=guarded(tc,'tracking',base['gpu_uuids']);wait(child,'tracking');clean(out/'tracking_guard')
  track=read(ev(out/'tracking/report.json'));assert track['complete'] and len(track['results'])==4
  assert all(x['max_error']<.1 for x in track['results'])
  review=out/'tracking_review.json';save(review,dict(passed=True,source_sha256=source['source_sha256'],production_contacts_enabled=True,baseline_comparison_complete=True,evidence=list(evidence.values()),scope='same Full endpoint paths under phase C2 and linear, normal contacts; not a policy SR guarantee'))
  tracking_binding=dict(path=str(review),sha256=sha(review));ev(review)
  state('preparing_current_measured_data')
  measured=out/'measured';dataset=out/'dataset'
  command=[sys.executable,str(REPO/'scripts/prepare_dualrate_exposure_data.py'),'--source',str(R/'outputs/datasets/residual_v10_20261102/half13'),'--output',str(measured),'--partial-fast-tail','--train-fraction','.5','--templates-per-source','4']
  # Include uninstrumented simultaneous two-environment deployment timings too.
  for task in ic['tasks']:traces.append(str(root/a.initial/task/'episode_0000/report.json'))
  for t in traces:command+=['--trace',t]
  run(command,'measured_data',cpu)
  run([sys.executable,str(REPO/'scripts/prepare_residual_v10_data.py'),'--source',str(measured),'--output',str(dataset),'--config',str(root/(a.initial+'.config.json'))],'bind_data',cpu)
  from dynamicwam.dualrate.feedback_replay import build_replay_index
  binding=build_replay_index([c['output'] for c in configs],out/'feedback_index.json',all_tasks)
  train=read(dataset/'train.json');train['continuity_replay']=binding;save(dataset/'train.json',train);ev(out/'feedback_index.json')
  # Actual expert physical/flow and real achieved-state replay audits.
  run([sys.executable,str(root/'audit_expert_data.py'),str(dataset),str(out/'expert_data_audit.json')],'expert_data_audit',cpu)
  run([sys.executable,str(REPO/'scripts/audit_continuity_data.py'),'--manifest',str(dataset/'train.json'),'--output',str(out/'feedback_data_audit.json')],'feedback_data_audit',cpu)
  audit=read(ev(out/'expert_data_audit.json'));assert audit['passed'] and len(audit['history_parity'])==35
  audit=read(ev(out/'feedback_data_audit.json'));assert audit['passed'] and audit['tasks']==35 and audit['paired_rows']>0
  # Tests bind deterministic time/prefix/queue and source-noise/correction rules.
  tests=['test_continuity_v11','test_residual_v10','test_dualrate_hardtime_deployment','test_dualrate_online','test_dualrate_runtime','test_dualrate_timing','test_dualrate_continuous_acceptance']
  run([sys.executable,'-m','pytest','-q',*[str(REPO/'tests'/(x+'.py')) for x in tests]],'cpu_acceptance',dict(cpu,PYTHONPATH=cpu['PYTHONPATH']+':'+str(R/'outputs/dependencies/dualrate_pytest_review_20261024')));ev(out/'cpu_acceptance.log')
  train=read(dataset/'train.json');contract=train['continuous_rtc_contract']
  acceptance=out/'continuous_data_acceptance.json';save(acceptance,dict(contract=contract,source_sha256=source['source_sha256'],same_command_state_parity=True,history_train_deploy_parity=True,injected_delay_queue_parity=True,actual_gpu_parallel_timing=True,physical_label_audit=True,all_task_smoke=True,evidence=list(evidence.values()),policy_quality_accepted=False))
  for split in ('train','validation'):
   m=read(dataset/(split+'.json'));m.update(continuous_acceptance_file=str(acceptance),continuous_acceptance_sha256=sha(acceptance));save(dataset/(split+'.json'),m)
  ev(acceptance)
  cbase=read(root/'training.template.json');ages=train['query_ages_seconds'];young=ages[:max(1,len(ages)//3)]
  cbase.update(source_sha256=source['source_sha256'],manifest=str(dataset/'train.json'),training_recipe=[dict(phase=p,steps=s) for p,s in zip(PHASES,STEPS)],data_distribution='half expert episodes, all indexed windows; paired queries + 25 percent train-only measured Full-anchored feedback replay',log_every=1)
  smokes=[]
  for phase in PHASES:
   c=copy.deepcopy(cbase);c.update(phase=phase,run_kind='distributed_training_smoke',steps=4,output=str(R/'outputs/training/continuity_v11_smoke01'/phase),sequence_id='continuity_v11_smoke01',query_age_seconds=ages if phase==PHASES[-1] else young,fast_lr=5e-5 if phase==PHASES[-1] else 1e-4)
   smokes.append(c)
  def sequence(configs,name,mode):
   rows=[]
   for c in configs:
    p=out/(name+'_'+c['phase']+'.json');save(p,c);rows.append(dict(phase=c['phase'],config=str(p),config_sha256=sha(p),max_runtime_seconds=7*86400))
   p=out/(name+'.json');save(p,dict(sequence_id=configs[0]['sequence_id'],run_kind=mode,gpu_uuids=','.join(base['gpu_uuids']),phases=rows))
   return p
  smoke_sequence=sequence(smokes,'train_smoke','distributed_training_smoke');state('eight_rank_three_stage_smoke')
  run([sys.executable,str(REPO/'scripts/dualrate_training_sequence.py'),'--sequence',str(smoke_sequence),'--run-dir',str(out/'train_smoke')],'train_smoke')
  profiles={};cost={}
  for c in smokes:
   phase=c['phase'];clean(out/'train_smoke'/phase);actual=read(out/'train_smoke'/(phase+'.config.json'))
   checkpoint,digest,result,reload=phase_completed(Path(c['output']),actual)
   assert result['identity']['source_sha256']==source['source_sha256']
   samples=[]
   for rank in range(8):
    rows=[json.loads(x) for x in ev(Path(c['output'])/f'rank_{rank}.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all(x['nonzero_finite_gradients']['fast']>0 for x in rows)
    samples.extend(x['seconds'] for x in rows[1:])
   for name in ('result.json','reload.json','parameters.json'):ev(Path(c['output'])/name)
   profiles[phase]=guard.profile_sha256(c);cost[phase]=statistics.median(samples)
  formal_acceptance=out/'formal_acceptance.json';save(formal_acceptance,dict(passed=True,world_size=8,source_sha256=source['source_sha256'],source_contract=guard.source_contract(source),supervisor_sha256=source['distributed_supervisor_sha256'],manifest_sha256=sha(dataset/'train.json'),init_checkpoint_sha256=cbase['init_checkpoint_sha256'],profiles=profiles,actuator_tracking_review=tracking_binding,evidence=list(evidence.values()),seconds_per_update=cost,estimated_hours=sum(cost[p]*s for p,s in zip(PHASES,STEPS))/3600))
  formal=[]
  for c,steps in zip(smokes,STEPS):
   c=dict(c,run_kind='formal_distributed_training',steps=steps,output=str(R/'outputs/training/continuity_v11_formal01'/c['phase']),sequence_id='continuity_v11_formal01',acceptance_receipt=str(formal_acceptance),acceptance_receipt_sha256=sha(formal_acceptance));guard.validate_acceptance(c,source);formal.append(c)
  formal_sequence=sequence(formal,'formal_sequence','formal_distributed_training');state('formal_training',estimated_hours=read(formal_acceptance)['estimated_hours'])
  child=launch([sys.executable,str(REPO/'scripts/dualrate_training_sequence.py'),'--sequence',str(formal_sequence),'--run-dir',str(out/'formal_sequence')],'formal_sequence')
  # Save a finite 300s progress acceptance, then this process simply waits.
  started=None;deadline=time.monotonic()+3600
  while time.monotonic()<deadline:
   if child.poll() is not None:raise RuntimeError('formal exited before initial progress acceptance')
   rows=[]
   for rank in range(8):
    p=Path(formal[0]['output'])/f'rank_{rank}.jsonl'
    if p.exists() and p.stat().st_size:rows.append(json.loads(p.read_text().splitlines()[-1]))
   if len(rows)==8:
    assert all(math.isfinite(x['losses']['loss']) and x['nonzero_finite_gradients']['fast']>0 for x in rows)
    if started is None:started=time.monotonic()
    if time.monotonic()-started>=300:
     save(out/'initial_training_acceptance.json',dict(passed=True,seconds=time.monotonic()-started,rank_steps=[x['step'] for x in rows],source=source));break
   time.sleep(5)
  else:raise TimeoutError('formal initial acceptance timeout')
  state('formal_running_initial_accepted',estimated_hours=read(formal_acceptance)['estimated_hours']);wait(child,'formal_sequence')
  # Exact final-checkpoint handoff using existing complete/reload/exit gates.
  from dualrate_posttrain_eval import run as postrun,validate_pair
  ec=read(root/'initial_suite.template.json');ec.update(manifest=str(dataset/'train.json'),source_sha256=source['source_sha256'],training_source_sha256=source['source_sha256'],checkpoint='PENDING_TRAINING_COMPLETION',checkpoint_sha256='PENDING_TRAINING_COMPLETION',action_limit=None,engineering_initialization_only=False)
  for k in ('acceptance_receipt','acceptance_receipt_sha256'):ec.pop(k,None)
  ec['benchmark_contract']['policy_resource_note']='V11 DualRate two environments, renderer/fast/slow dedicated GPUs each'
  spec=dict(training_sequence_dir=str(out/'formal_sequence'),training_recipe=cbase['training_recipe'],expected_training_commit=source['commit'],expected_training_source_sha256=source['source_sha256'],max_wait_seconds=172800,smoke_max_runtime=86400,formal_max_runtime=604800)
  smoke=dict(ec,stage='smoke',tasks=all_tasks[:2],episodes=2,output=str(out/'final_smoke'),result_stamp='v11_final_smoke')
  full=dict(ec,stage='formal',tasks=all_tasks,episodes=20,output=str(out/'formal700'),result_stamp='v11_formal700')
  validate_pair(smoke,full)
  for kind,c in [('smoke',smoke),('formal',full)]:
   p=out/(kind+'_eval.template.json');save(p,c);spec[kind+'_template']=str(p);spec[kind+'_template_sha256']=sha(p)
  state('posttraining_smoke_and_eval');postout=out/'posttrain';postout.mkdir();code=postrun(spec,postout,repo=REPO)
  if code:raise RuntimeError('posttraining evaluation failed')
  state('completed',valid_completed=700)
 except BaseException as e:
  for c in children:
   if c.poll() is None:c.send_signal(signal.SIGTERM)
  for c in children:
   try:c.wait(timeout=180)
   except subprocess.TimeoutExpired:pass
  state('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
