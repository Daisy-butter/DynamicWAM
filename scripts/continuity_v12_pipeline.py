"""Finite V12 collection -> measured data -> 8-rank smoke -> Full training -> 700 eval.
All failures stop dependent stages. No automatic retry or driver operations.
"""
import argparse,ast,copy,hashlib,json,math,os,signal,statistics,subprocess,sys,time,traceback
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
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--initial',required=True);p.add_argument('--attempt',default='pipeline01');p.add_argument('--reuse-prefix');p.add_argument('--retain-collection');p.add_argument('--manual-start',required=True,action='store_true');a=p.parse_args()
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
 def runtime_equivalent(prior):
  for key in ('guard_sha256','supervisor_sha256','distributed_supervisor_sha256'):
   if prior[key]!=source[key]:raise ValueError('supervision changed: '+key)
  changed=subprocess.check_output(['git','diff','--name-only',prior['commit'],source['commit']],cwd=REPO,text=True).splitlines()
  allowed={'src/dynamicwam/dualrate/continuity_data.py','src/dynamicwam/dualrate/continuity_v11.py',
           'scripts/continuity_v12_pipeline.py','scripts/audit_continuity_data.py','scripts/dualrate_domino_smoke.py',
           'src/dynamicwam/dualrate/producer.py','scripts/prepare_dualrate_exposure_data.py',
           'scripts/audit_continuity_expert_data.py','src/dynamicwam/dualrate/deployment.py',
           'src/dynamicwam/dualrate/reference_lease.py','scripts/dualrate_continuous_acceptance.py',
           'src/dynamicwam/dualrate/feedback_replay.py'}
  if any(x not in allowed and not x.startswith(('docs/','tests/')) for x in changed):raise ValueError('runtime equivalence not proven')
  file='src/dynamicwam/dualrate/continuity_v11.py'
  old=subprocess.check_output(['git','show',prior['commit']+':'+file],cwd=REPO,text=True)
  def deployment_ast(text):
   tree=ast.parse(text);tree.body=[x for x in tree.body if not isinstance(x,ast.FunctionDef) or x.name!='loss']
   return ast.dump(tree,include_attributes=False)
  assert deployment_ast(old)==deployment_ast((REPO/file).read_text())
  file='src/dynamicwam/dualrate/producer.py'
  old=subprocess.check_output(['git','show',prior['commit']+':'+file],cwd=REPO,text=True)
  def producer_runtime_ast(text):
   tree=ast.parse(text)
   tree.body=[x for x in tree.body if not isinstance(x,ast.FunctionDef) or x.name!='materialize_native_latents']
   for x in tree.body:
    if isinstance(x,ast.ClassDef) and x.name=='DominoPhysicalDataset':
     x.body=[f for f in x.body if not isinstance(f,ast.FunctionDef) or f.name!='__getitem__']
   return ast.dump(tree,include_attributes=False)
  assert producer_runtime_ast(old)==producer_runtime_ast((REPO/file).read_text())
  file='src/dynamicwam/dualrate/feedback_replay.py'
  old=subprocess.check_output(['git','show',prior['commit']+':'+file],cwd=REPO,text=True)
  def recorder_ast(text):
   tree=ast.parse(text);tree.body=[x for x in tree.body if not isinstance(x,ast.FunctionDef) or x.name!='build_replay_index']
   return ast.dump(tree,include_attributes=False)
  assert recorder_ast(old)==recorder_ast((REPO/file).read_text())
  # Retain only completed TRAIN reports: the only changed runtime path formerly
  # raised before creating a lease. Verify all successful-path AST is identical.
  for file in ('src/dynamicwam/dualrate/reference_lease.py','src/dynamicwam/dualrate/deployment.py'):
   old=subprocess.check_output(['git','show',prior['commit']+':'+file],cwd=REPO,text=True)
   def successful_path_ast(text):
    tree=ast.parse(text)
    class RemoveDeferral(ast.NodeTransformer):
     def visit_ClassDef(self,node):
      return None if node.name=='ReferenceUnavailable' else self.generic_visit(node)
     def visit_ImportFrom(self,node):
      return None if node.module=='reference_lease' and [x.name for x in node.names]==['ReferenceUnavailable'] else node
     def visit_ExceptHandler(self,node):
      return None if isinstance(node.type,ast.Name) and node.type.id=='ReferenceUnavailable' else self.generic_visit(node)
     def visit_Raise(self,node):
      if isinstance(node.exc,ast.Call) and isinstance(node.exc.func,ast.Name) and node.exc.func.id=='ReferenceUnavailable':
       node.exc=ast.Call(func=ast.Name(id='ValueError',ctx=ast.Load()),args=[ast.Constant('no causal reference with two editable physical rows')],keywords=[])
      return node
    return ast.dump(RemoveDeferral().visit(tree),include_attributes=False)
   assert successful_path_ast(old)==successful_path_ast((REPO/file).read_text())
  proof=out/('runtime_equivalence_'+prior['commit'][:8]+'.json')
  save(proof,dict(passed=True,original=prior,current=source,changed_paths=changed,unchanged_deployment_ast=True,
                 scope='successful model/dispatch paths unchanged; typed no-reference pre-query exception now defers without changing clocks, prefix or TTL'))
  ev(proof)
 def guarded(c,name,gpus):
  path=out/(name+'.config.json');save(path,c)
  return launch([sys.executable,str(REPO/'scripts/guarded_dualrate_run.py'),'--config',str(path),'--run-dir',str(out/(name+'_guard')),'--gpus',','.join(gpus),'--max-runtime','86400'],name)
 try:
  state('validating_initial')
  initial=read(ev(root/(a.initial+'.status.json')));assert initial['state']=='initial_gpu_smoke_completed'
  # The added finite controller does not alter source/model/runtime code.
  runtime_equivalent(initial['source'])
  clean(root/(a.initial+'_guard'));ic=read(ev(root/(a.initial+'.config.json')))
  initial_model=read(ev(root/(a.initial+'_initial/initialization_receipt.json')))
  assert initial_model['trained_steps']==0 and not initial_model['cuda_initialized']
  all_tasks=sorted(read(root/'training_scene_pool.json')['tasks'])
  prefix=Path(a.reuse_prefix).resolve(strict=True) if a.reuse_prefix else out
  if a.reuse_prefix:
   if prefix.parent!=root:raise ValueError('preserved prefix must be a sibling attempt')
   ps=read(ev(prefix/'status.json'));assert ps['state']=='failed'
   runtime_equivalent(ps['source'])
  records=[];configs=[];retained=[];retained_tasks=set();state('collecting_train_feedback')
  if a.reuse_prefix and (prefix/'retained_train_collection.json').exists():
   prior_retained=read(ev(prefix/'retained_train_collection.json'))
   from video_artifacts import validate_video_artifacts
   for path in prior_retained['reports']:
    path=ev(path);cfg_path=path.parents[3]/'collection_0.config.json';cfg=read(ev(cfg_path))
    r=read(path);validate_continuous_report(r,cfg);assert validate_video_artifacts(r)
    retained.append(str(path));retained_tasks.add(r['task'])
   for directory in sorted({str(Path(x).parents[3]) for x in retained}):
    configs.append(read(Path(directory)/'collection_0.config.json'))
  if a.retain_collection:
   if a.reuse_prefix:raise ValueError('choose one preserved-prefix mode')
   prior_root=Path(a.retain_collection).resolve(strict=True)
   if prior_root.parent!=root:raise ValueError('retained collection must be a sibling attempt')
   old=read(ev(prior_root/'status.json'));assert old['state']=='failed';runtime_equivalent(old['source'])
   gd=prior_root/'collection_0_guard';gs=read(ev(gd/'status.json'));ge=read(ev(gd/'owned_exit.json'));gk=read(ev(gd/'kernel_after.json'))
   assert gs['guard_closed'] and gs['errors']==["RuntimeError('worker exited 1; automatic retry disabled')"]
   assert not ge['remaining'] and ge['returncode']==1 and gk['unchanged'] and gk['same_boot'] and not gk['added']
   log=ev(gd/'worker.log').read_text()
   assert ('AssertionError: target_pose cannot be None for move action.' in log or
           'ValueError: no causal reference with two editable physical rows' in log)
   oldcfg=read(ev(prior_root/'collection_0.config.json'))
   assert oldcfg['runtime']==base['runtime'] and oldcfg['residual_v10']==base['residual_v10']
   from video_artifacts import validate_video_artifacts
   for path in sorted(Path(oldcfg['output']).glob('*/episode_*/report.json')):
    report=read(path)
    if not report.get('continuous_valid') or not report.get('realtime_valid'):continue
    validate_continuous_report(report,oldcfg);assert validate_video_artifacts(report)
    assert report['task'] not in retained_tasks
    fm=read(ev(path.parent/'feedback/manifest.json'));assert fm['complete'] and fm['identity']['split']=='train'
    retained.append(str(ev(path)));retained_tasks.add(report['task'])
   assert retained and retained_tasks<set(all_tasks)
   configs.append(oldcfg)
   save(out/'retained_train_collection.json',dict(passed=True,tasks=sorted(retained_tasks),reports=retained,scope='only complete TRAIN episodes; no scored eval results merged'))
   ev(out/'retained_train_collection.json')
  # One three-GPU collector: SAPIEN creates a small graphics probe on physical
  # GPU0 even in another visibility group. Two independent guards must not
  # declare each other idle. Scored eval retains its unified eight-GPU guard.
  if a.reuse_prefix:
   c=read(ev(prefix/'collection_0.config.json'))
   assert set(c['deployment']['tasks'])==set(all_tasks)-retained_tasks and c['residual_v10']==base['residual_v10'] and c['runtime']==base['runtime']
   configs.append(c);records.append((None,'collection_0',c))
  else:
   c=read(root/'collection.template.json');c['deployment']['tasks']=[t for t in all_tasks if t not in retained_tasks]
   c.update(output=str(out/'collection_0'),source_sha256=source['source_sha256'],training_source_sha256=source['source_sha256'],deployment_checkpoint=initial_model['checkpoint'],deployment_checkpoint_sha256=initial_model['sha256'])
   configs.append(c);records.append((guarded(c,'collection_0',base['gpu_uuids'][:3]),'collection_0',c))
  traces=list(retained)
  for child,name,c in records:
   if child is not None:wait(child,name)
   clean(prefix/(name+'_guard'))
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
  tc['cases']=[dict(row) for row in tc['cases']]+[dict(row,label=row['label']+'_linear_v12',controller='linear') for row in tc['cases']]
  if not a.reuse_prefix:
   child=guarded(tc,'tracking',base['gpu_uuids']);wait(child,'tracking')
  clean(prefix/'tracking_guard')
  track=read(ev(prefix/'tracking/report.json'));assert track['complete'] and len(track['results'])==4
  assert all(x['max_error']<.1 for x in track['results'])
  review=out/'tracking_review.json';save(review,dict(passed=True,source_sha256=source['source_sha256'],production_contacts_enabled=True,baseline_comparison_complete=True,evidence=list(evidence.values()),scope='same Full endpoint paths under phase C2 and linear, normal contacts; not a policy SR guarantee'))
  tracking_binding=dict(path=str(review),sha256=sha(review));ev(review)
  state('preparing_current_measured_data')
  measured=out/'measured';dataset=out/'dataset'
  command=[sys.executable,str(REPO/'scripts/prepare_dualrate_exposure_data.py'),'--source',str(R/'outputs/datasets/residual_v10_20261102/half13'),'--output',str(measured),'--partial-fast-tail','--partial-source-tail','--train-fraction','.5','--templates-per-source','4']
  # Include uninstrumented simultaneous two-environment deployment timings too.
  for task in ic['tasks']:traces.append(str(root/a.initial/task/'episode_0000/report.json'))
  for t in traces:command+=['--trace',t]
  run(command,'measured_data',cpu)
  run([sys.executable,str(REPO/'scripts/prepare_residual_v10_data.py'),'--source',str(measured),'--output',str(dataset),'--config',str(root/(a.initial+'.config.json'))],'bind_data',cpu)
  from dynamicwam.dualrate.feedback_replay import build_replay_index
  binding=build_replay_index([c['output'] for c in configs],out/'feedback_index.json',all_tasks)
  train=read(dataset/'train.json');train['continuity_replay']=binding;save(dataset/'train.json',train);ev(out/'feedback_index.json')
  # Actual expert physical/flow and real achieved-state replay audits.
  run([sys.executable,str(REPO/'scripts/audit_continuity_expert_data.py'),str(dataset),str(out/'expert_data_audit.json')],'expert_data_audit',cpu)
  run([sys.executable,str(REPO/'scripts/audit_continuity_data.py'),'--manifest',str(dataset/'train.json'),'--output',str(out/'feedback_data_audit.json')],'feedback_data_audit',cpu)
  audit=read(ev(out/'expert_data_audit.json'));assert audit['passed'] and len(audit['history_parity'])==35
  audit=read(ev(out/'feedback_data_audit.json'));assert audit['passed'] and audit['tasks']==35 and audit['paired_rows']>0
  # Tests bind deterministic time/prefix/queue and source-noise/correction rules.
  tests=['test_handoff_v12','test_continuity_v11_collection','test_continuity_v11','test_residual_v10','test_dualrate_hardtime_deployment','test_dualrate_online','test_dualrate_runtime','test_dualrate_timing','test_dualrate_continuous_acceptance']
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
   c=copy.deepcopy(cbase);c.update(phase=phase,run_kind='distributed_training_smoke',steps=4,output=str(R/'outputs/training/continuity_v12_smoke01'/phase),sequence_id='continuity_v12_smoke01',query_age_seconds=ages if phase==PHASES[-1] else young,fast_lr=5e-5 if phase==PHASES[-1] else 1e-4)
   smokes.append(c)
  def sequence(configs,name,mode):
   rows=[]
   for c in configs:
    p=out/(name+'_'+c['phase']+'.json');save(p,c);rows.append(dict(phase=c['phase'],config=str(p),config_sha256=sha(p),max_runtime_seconds=7*86400))
   p=out/(name+'.json');save(p,dict(sequence_id=configs[0]['sequence_id'],run_kind=mode,gpu_uuids=','.join(base['gpu_uuids']),phases=rows))
   return p
  smoke_sequence=sequence(smokes,'train_smoke','distributed_training_smoke');state('eight_rank_three_stage_smoke')
  run([sys.executable,str(REPO/'scripts/dualrate_training_sequence.py'),'--sequence',str(smoke_sequence),'--run-dir',str(out/'train_smoke')],'train_smoke')
  profiles={};cost={};precision={}
  for c in smokes:
   phase=c['phase'];clean(out/'train_smoke'/phase);actual=read(out/'train_smoke'/(phase+'.config.json'))
   checkpoint,digest,result,reload=phase_completed(Path(c['output']),actual)
   assert result['identity']['source_sha256']==source['source_sha256']
   samples=[];precision[phase]=[]
   for rank in range(8):
    rows=[json.loads(x) for x in ev(Path(c['output'])/f'rank_{rank}.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all(x['nonzero_finite_gradients']['fast']>0 for x in rows)
    samples.extend(x['seconds'] for x in rows[1:])
    precision[phase].extend({k:v for k,v in x['losses'].items() if k.startswith(('teacher_','handoff_','cross_reference_'))} for x in rows)
   for name in ('result.json','reload.json','parameters.json'):ev(Path(c['output'])/name)
   profiles[phase]=guard.profile_sha256(c);cost[phase]=statistics.median(samples)
  save(out/'teacher_precision_smoke.json',dict(scope='first/near/tail teacher discrepancy; 4 updates per rank, not policy accuracy',phases=precision));ev(out/'teacher_precision_smoke.json')
  formal_acceptance=out/'formal_acceptance.json';save(formal_acceptance,dict(passed=True,world_size=8,source_sha256=source['source_sha256'],source_contract=guard.source_contract(source),supervisor_sha256=source['distributed_supervisor_sha256'],manifest_sha256=sha(dataset/'train.json'),init_checkpoint_sha256=cbase['init_checkpoint_sha256'],profiles=profiles,actuator_tracking_review=tracking_binding,evidence=list(evidence.values()),seconds_per_update=cost,estimated_hours=sum(cost[p]*s for p,s in zip(PHASES,STEPS))/3600))
  formal=[]
  for c,steps in zip(smokes,STEPS):
   c=dict(c,run_kind='formal_distributed_training',steps=steps,output=str(R/'outputs/training/continuity_v12_formal01'/c['phase']),sequence_id='continuity_v12_formal01',acceptance_receipt=str(formal_acceptance),acceptance_receipt_sha256=sha(formal_acceptance));guard.validate_acceptance(c,source);formal.append(c)
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
  ec['benchmark_contract']['policy_resource_note']='V12 DualRate two environments, renderer/fast/slow dedicated GPUs each'
  spec=dict(training_sequence_dir=str(out/'formal_sequence'),training_recipe=cbase['training_recipe'],expected_training_commit=source['commit'],expected_training_source_sha256=source['source_sha256'],max_wait_seconds=172800,smoke_max_runtime=86400,formal_max_runtime=604800)
  smoke=dict(ec,stage='smoke',tasks=all_tasks[:2],episodes=2,output=str(out/'final_smoke'),result_stamp='v12_final_smoke')
  full=dict(ec,stage='formal',tasks=all_tasks,episodes=20,output=str(out/'formal700'),result_stamp='v12_formal700')
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
