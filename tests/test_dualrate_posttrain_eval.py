import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import dualrate_posttrain_eval as mod


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def source():
    return dict(commit=mod.TRAIN_COMMIT,source_sha256=mod.TRAIN_SHA)


def guard_fixture(directory,config,src,pid):
    uuids=['gpu'+str(i) for i in range(8)]
    identity=dict(source_sha256=src['source_sha256'],config_sha256=mod.canonical_hash(config))
    save(directory/'status.json',dict(state='completed',guard_closed=True,errors=[],supervisor=dict(pid=pid,start_ticks=10)))
    save(directory/'owned_exit.json',dict(remaining=[],returncode=0))
    save(directory/'kernel_before.json',dict(lines=[],boot_id='boot'))
    save(directory/'kernel_after.json',dict(unchanged=True,same_boot=True,added=[]))
    save(directory/'source.json',src);save(directory/'config.frozen.json',config)
    save(directory/'authorization.json',dict(identity=identity,gpu_uuids=uuids))
    events=[dict(event=event,uuid=u,path='/dev/'+u,fd=i) for event in ('after_open','after_close') for i,u in enumerate(uuids)]
    save(directory/'guard_events.json',events)
    return identity


def training_fixture(tmp_path,monkeypatch,recipe=None):
    root=tmp_path/'training';src=source();rows=[];completed=[];previous=None
    for index,(name,steps,global_step) in enumerate(mod.expected_phases({'training_recipe':recipe}) if recipe is not None else mod.PHASES):
        output=tmp_path/'weights'/name;output.mkdir(parents=True)
        checkpoint=output/'last.pt';checkpoint.write_bytes(name.encode())
        digest=mod.file_sha256(checkpoint)
        template=dict(phase=name,steps=steps,world_size=8,run_kind='formal_distributed_training',output=str(output))
        if recipe is not None:template['training_recipe']=recipe
        template_path=tmp_path/'templates'/(name+'.json');save(template_path,template)
        rows.append(dict(phase=name,config=str(template_path),config_sha256=mod.file_sha256(template_path)))
        config=dict(template,sequence_id='training1')
        if previous:
            config.update(stage_init_checkpoint=previous['checkpoint'],stage_init_checkpoint_sha256=previous['checkpoint_sha256'])
        save(root/(name+'.config.json'),config)
        identity=guard_fixture(root/name,config,src,100+index)
        save(output/'result.json',dict(result_kind='completed_phase_not_policy_quality_acceptance',phase=name,
            run_kind=config['run_kind'],world_size=8,completed_steps=steps,configured_steps=steps,
            global_step=global_step,checkpoint=str(checkpoint),checkpoint_sha256=digest,identity=identity))
        state={key:'hash' for key in ('model','optimizer','rank','modes','requires_grad')}
        ranks=[dict(rank=i,passed=True,phase_step=steps,global_step=global_step,before=state,after=state) for i in range(8)]
        save(output/'reload.json',dict(passed=True,world_size=8,equal_model=True,equal_optimizer=True,
            extra_training_updates=0,ranks=ranks,checkpoint=str(checkpoint),checkpoint_sha256=digest,identity=identity))
        previous=dict(phase=name,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,global_step=global_step)
        completed.append(previous)
    save(root/'source.json',src)
    save(root/'sequence.frozen.json',dict(sequence_id='training1',phases=rows))
    save(root/'status.json',dict(state='completed',source=src,sequence_id='training1',errors=[],
        controller=dict(pid=99,start_ticks=10),completed_phases=completed))
    monkeypatch.setattr(mod,'identity_exited',lambda identity:True)
    spec=dict(training_sequence_dir=str(root))
    if recipe is not None:spec['training_recipe']=recipe
    return spec,root


def test_waits_for_live_training_and_completed_controller_exit(tmp_path,monkeypatch):
    spec,root=training_fixture(tmp_path,monkeypatch)
    status=mod.read(root/'status.json');status['state']='phase_running';save(root/'status.json',status)
    monkeypatch.setattr(mod,'identity_exited',lambda identity:False)
    assert mod.training_ready(spec) is None
    status['state']='completed';save(root/'status.json',status)
    assert mod.training_ready(spec) is None


def test_all_three_phases_and_final_sha_required(tmp_path,monkeypatch):
    spec,root=training_fixture(tmp_path,monkeypatch)
    result=mod.training_ready(spec)
    assert result['passed'] and result['all_training_processes_exited']
    assert result['phases'][-1]['global_step']==31000
    assert result['checkpoint']['sha256']==result['phases'][-1]['checkpoint_sha256']
    assert result['checkpoint']['stable_during_full_hash']


@pytest.mark.parametrize('fault',['failed','dead','bad_rank','bad_sha','open_fd','bad_source'])
def test_training_faults_never_become_eval_ready(tmp_path,monkeypatch,fault):
    spec,root=training_fixture(tmp_path,monkeypatch)
    if fault in ('failed','dead','bad_source'):
        status=mod.read(root/'status.json')
        if fault=='failed':status['state']='failed'
        elif fault=='dead':status['state']='phase_running'
        else:status['source']['source_sha256']='other'
        save(root/'status.json',status)
    elif fault=='open_fd':
        path=root/'cross_time_joint/guard_events.json';events=mod.read(path);events.pop();save(path,events)
    else:
        path=tmp_path/'weights/cross_time_joint/reload.json';reload=mod.read(path)
        if fault=='bad_rank':reload['ranks'][7]['passed']=False
        else:reload['checkpoint_sha256']='wrong'
        save(path,reload)
    with pytest.raises((ValueError,RuntimeError)):
        mod.training_ready(spec)


def test_checkpoint_change_during_hash_is_rejected(tmp_path,monkeypatch):
    path=tmp_path/'last.pt';path.write_bytes(b'first')
    digest=mod.file_sha256(path)
    def changing_hash(p):
        p.write_bytes(b'changed size')
        return digest
    monkeypatch.setattr(mod,'file_sha256',changing_hash)
    with pytest.raises(ValueError,match='changed while being hashed'):
        mod.stable_checkpoint(path,digest)


def eval_pair(tmp_path):
    uuids=['gpu'+str(i) for i in range(8)];tasks=['task'+str(i) for i in range(35)]
    base=dict(run_kind='dualrate_eval_suite',gpu_uuids=uuids,environment_groups=[uuids[:3],uuids[3:6]],
        benchmark_contract={'action_dt':.18},checkpoint_sha256='PENDING_TRAINING_COMPLETION',checkpoint='pending',
        storage_root=str(tmp_path))
    smoke=dict(base,stage='smoke',tasks=tasks[:2],episodes=2,output=str(tmp_path/'smoke'))
    formal=dict(base,stage='formal',tasks=tasks,episodes=20,output=str(tmp_path/'formal'))
    return smoke,formal


def test_checkpoint_binding_preserves_shared_benchmark_and_rejects_drift(tmp_path):
    smoke,formal=eval_pair(tmp_path);spec={}
    for name,config in [('smoke',smoke),('formal',formal)]:
        path=tmp_path/(name+'.template.json');save(path,config)
        spec[name+'_template']=str(path);spec[name+'_template_sha256']=mod.file_sha256(path)
    result=mod.bind_templates(spec,dict(checkpoint=dict(path='/final/last.pt',sha256='finalsha')))
    assert all(x['checkpoint']=='/final/last.pt' and x['checkpoint_sha256']=='finalsha' for x in result)
    assert result[0]['benchmark_contract']==smoke['benchmark_contract']
    result[1]['benchmark_contract']={'action_dt':.16}
    with pytest.raises(ValueError,match='contract differs'):
        mod.validate_pair(*result)
    Path(spec['smoke_template']).write_text('{}')
    with pytest.raises(ValueError,match='template changed'):
        mod.bind_templates(spec,dict(checkpoint=dict(path='/final/last.pt',sha256='finalsha')))


def test_pid_reuse_means_old_training_process_is_exited(monkeypatch):
    monkeypatch.setattr(mod,'proc_identity',lambda pid:dict(pid=pid,start_ticks=22,state='R'))
    assert mod.identity_exited(dict(pid=3,start_ticks=11))
    assert not mod.identity_exited(dict(pid=3,start_ticks=22))


@pytest.mark.skipif(sys.platform=='win32',reason='controller uses the Linux advisory lock')
def test_controller_spawns_nothing_before_training_gate_then_smoke_then_formal(tmp_path,monkeypatch):
    smoke,formal=eval_pair(tmp_path);out=tmp_path/'controller';out.mkdir()
    clock=[0.];checks=[];launches=[]
    training=dict(checkpoint=dict(path='/final/last.pt',sha256='finalsha'))
    def readiness(spec):
        checks.append(clock[0])
        return None if len(checks)==1 else training
    monkeypatch.setattr(mod,'training_ready',readiness)
    monkeypatch.setattr(mod.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(mod.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(mod.guard,'frozen_source',lambda repo:source())
    monkeypatch.setattr(mod,'proc_identity',lambda pid:dict(pid=pid,start_ticks=1,state='R'))
    monkeypatch.setattr(mod,'bind_templates',lambda spec,accepted:(smoke,formal))
    monkeypatch.setattr(mod,'storage_directory',lambda path,storage:path)
    monkeypatch.setattr(mod,'stable_checkpoint',lambda *args:training['checkpoint'])
    monkeypatch.setattr(mod,'require_clean_guard',lambda *args:[])
    monkeypatch.setattr(mod,'audit_suite',lambda *args:({},[],[]))
    def acceptance(config,directory,source,destination):
        assert len(launches)==1
        save(destination,dict(passed=True))
    monkeypatch.setattr(mod,'make_acceptance',acceptance)
    class Child:
        pid=345
        def __init__(self,command,**kwargs):
            assert checks==[0.,60.]
            config=mod.read(command[command.index('--config')+1])
            launches.append(config['stage'])
            if config['stage']=='formal':
                assert config['acceptance_receipt_sha256']==mod.file_sha256(Path(config['acceptance_receipt']))
        def wait(self):return 0
    monkeypatch.setattr(mod.subprocess,'Popen',Child)
    assert mod.run(dict(max_wait_seconds=120.,smoke_max_runtime=600.,formal_max_runtime=3600.),out)==0
    assert launches==['smoke','formal']
    assert mod.read(out/'status.json')['valid_completed']==700


def accepted_smoke_fixture(tmp_path,monkeypatch,deployment=None):
    smoke,formal=eval_pair(tmp_path)
    smoke['checkpoint_sha256']=formal['checkpoint_sha256']='finalsha'
    src=dict(source(),guard_sha256='guard',supervisor_sha256='supervisor',
             distributed_supervisor_sha256='distributed',scripts_sha256='scripts')
    directory=tmp_path/'smoke_guard';guard_fixture(directory,smoke,src,100)
    if deployment is None:
        deployment=dict(passed=True,fast_graph_passed=True,fast_queries_during_slow=8,
            model_overlap_fast_queries=6,refreshed_reference_consumed=True)
    evidence=tmp_path/'audited_reports.json';save(evidence,dict(passed=True,count=4))
    monkeypatch.setattr(mod,'audit_suite',lambda *args:(dict(deployment_acceptance=deployment),[evidence],[{'report':str(i)} for i in range(4)]))
    destination=tmp_path/'acceptance.json'
    return smoke,formal,src,directory,destination


def test_generated_acceptance_is_accepted_by_actual_guard(tmp_path,monkeypatch):
    smoke,formal,src,directory,path=accepted_smoke_fixture(tmp_path,monkeypatch)
    receipt=mod.make_acceptance(smoke,directory,src,path)
    formal.update(acceptance_receipt=str(path),acceptance_receipt_sha256=mod.file_sha256(path))
    assert receipt['hardtime_acceptance_passed'] is True
    assert mod.guard.validate_acceptance(formal,src)==receipt


@pytest.mark.parametrize('fault',['wrong_count','missing_hardtime','false_hardtime','missing_graph',
                                 'false_graph','failed_deployment','no_overlap','no_refresh'])
def test_actual_guard_rejects_inconsistent_acceptance(tmp_path,monkeypatch,fault):
    smoke,formal,src,directory,path=accepted_smoke_fixture(tmp_path,monkeypatch)
    receipt=mod.make_acceptance(smoke,directory,src,path)
    if fault=='wrong_count':receipt['complete_smoke_episodes']=3
    elif fault=='missing_hardtime':del receipt['hardtime_acceptance_passed']
    elif fault=='false_hardtime':receipt['hardtime_acceptance_passed']=False
    elif fault=='missing_graph':del receipt['deployment_acceptance']['fast_graph_passed']
    elif fault=='false_graph':receipt['deployment_acceptance']['fast_graph_passed']=False
    elif fault=='failed_deployment':receipt['deployment_acceptance']['passed']=False
    elif fault=='no_overlap':receipt['deployment_acceptance']['model_overlap_fast_queries']=0
    else:receipt['deployment_acceptance']['refreshed_reference_consumed']=False
    save(path,receipt)
    formal.update(acceptance_receipt=str(path),acceptance_receipt_sha256=mod.file_sha256(path))
    with pytest.raises(ValueError):
        mod.guard.validate_acceptance(formal,src)


@pytest.mark.parametrize('graph',[False,None])
def test_acceptance_producer_cannot_publish_a_false_graph_pass(tmp_path,monkeypatch,graph):
    deployment=dict(passed=True,fast_queries_during_slow=8,model_overlap_fast_queries=6,refreshed_reference_consumed=True)
    if graph is not None:deployment['fast_graph_passed']=graph
    smoke,formal,src,directory,path=accepted_smoke_fixture(tmp_path,monkeypatch,deployment)
    with pytest.raises(ValueError,match='Graph/concurrency'):
        mod.make_acceptance(smoke,directory,src,path)
    assert not path.exists()


def test_explicit_v8_recipe_accepts_real_completion_not_old_v4(tmp_path,monkeypatch):
    recipe=[dict(phase=n,steps=s) for n,s in [('warmup',500),('small_age_joint',4000),('cross_time_joint',8000)]]
    spec,root=training_fixture(tmp_path,monkeypatch,recipe)
    result=mod.training_ready(spec)
    assert [p['global_step'] for p in result['phases']]==[500,4500,12500]
    spec['training_recipe'][-1]['steps']=7999
    with pytest.raises(ValueError):mod.training_ready(spec)


@pytest.mark.parametrize('recipe',[[],[{}],None,True, [{'phase':n,'steps':True} for n in ['warmup','small_age_joint','cross_time_joint']]])
def test_bad_explicit_recipe_rejected(recipe):
    if recipe is None:
        assert mod.expected_phases({})==mod.PHASES
    else:
        with pytest.raises(ValueError):mod.expected_phases({'training_recipe':recipe})
