import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import native_v7_eval_common as common
import guarded_dualrate_run as guard


def config():return json.loads((ROOT/'configs/native_v7_eval_template.json').read_text())


def episode():
    events=[];physical=0.
    # Commands have nonuniform TOPP durations; never q*.18 physical time.
    for q,duration in enumerate((.43,.27,.31,.12)):
        if q%2==0:
            events.append(dict(command_index=q,source_command_index=0,source_physical_time=0.,
                query_physical_time=physical,reference_age=physical,prefix_rows=0,prior_row_offset=q,
                prior_overlap_rows=16-q,prior_source_event=0,graph_ready=True,
                future_command_events=[q+n for n in (1,2,4,8)]*3,future_physical_end_times_known=False,
                previous_plan_aligned=q>0,history_sampling='physical_time',history_interval_seconds=.72,
                history_target_times=[physical-.72,physical],history_exposure_times=[0.,physical],
                history_endpoint_valid=[False,True],query_seconds=.5,fast_seconds=.1))
        events.append(dict(kind='native_topp_execute',command_index=q,physical_start=physical,
            physical_end=physical+duration,actual_duration=duration));physical+=duration
    return dict(events=events,action_count=4,physical_end=physical)


def test_nonuniform_physics_and_event_prior_are_independent():
    assert common.audit_events(episode())==dict(queries=2,reference_reuses=1,reference_refreshes=1,passed=True)


@pytest.mark.parametrize('field,value',[
    ('query_physical_time',.36),('source_physical_time',.1),('prior_row_offset',0),
    ('prefix_rows',2),('history_sampling','command_ordinal_legacy'),('future_physical_end_times_known',True),
    ('previous_plan_aligned',False),('graph_ready',False),('prior_overlap_rows',16),
])
def test_reject_wrong_clock_or_action_contract(field,value):
    r=episode();r['events'][3][field]=value
    with pytest.raises(ValueError):common.audit_events(r)


def test_capture_exposure_must_be_a_real_due_tick():
    r=episode();r['events'].insert(1,dict(kind='native_history_capture',exposure=.2,requested_exposure=.06,physics_step_seconds=.004))
    with pytest.raises(ValueError,match='physical tick'):common.audit_events(r)


@pytest.mark.parametrize('field,value',[
    ('protocol','continuous'),('time_contract','exposure_anchor_wall_release_v4'),('action_limit',32),
    ('runtime',{'action_dt':.18}),('full_driven_integration',True),('episodes',1),
    ('native_execution',dict(replan_actions=4,refresh_actions=4,max_reference_age=1.44)),
])
def test_native_config_rejects_continuous_and_fixture_settings(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):common.validate_config(c,bound=False)


def test_new_mode_dispatches_only_native_worker():
    c=config();cmd=guard.worker_command(ROOT,c,Path('config.json'),Path('auth.json'))
    assert cmd[1].endswith('native_v7_eval_suite.py')
    assert common.validate_config(c,bound=False)
    with pytest.raises(ValueError,match='not bound'):common.validate_config(c)


def test_smoke_and_formal_contract_cannot_silently_diverge():
    c=config();s=dict(c,stage='smoke',tasks=c['tasks'][:8],episodes=2,output=c['output']+'smoke',result_stamp='smoke')
    assert common.paired_contract(s)==common.paired_contract(c)
    s['native_execution']=dict(s['native_execution'],max_reference_age=2.)
    assert common.paired_contract(s)!=common.paired_contract(c)


@pytest.mark.parametrize('field,value', [('run_kind','distributed_training_smoke'),('phase_complete',False),
    ('phase_step',19999),('global_step',30999)])
def test_partial_or_smoke_weights_never_become_formal(field,value):
    c=config();p=dict(run_kind='formal_distributed_training',phase='cross_time_joint',phase_step=20000,
        global_step=31000,phase_complete=True);p[field]=value
    with pytest.raises(ValueError,match='completed formal'):common.validate_checkpoint(c,p)


def test_only_stale_utilization_may_drain(monkeypatch,tmp_path):
    rows=[dict(uuid='gpu',processes=[],used_mib=1,gpu_util_percent=1),dict(uuid='gpu',processes=[],used_mib=1,gpu_util_percent=0)]
    monkeypatch.setattr(guard.base,'telemetry',lambda *a:[rows.pop(0)])
    monkeypatch.setattr(guard.base,'require_idle',lambda r,u: None if r[0]['gpu_util_percent']==0 else (_ for _ in ()).throw(ValueError()))
    guard.postflight_idle({},tmp_path,['gpu'],clock=lambda:0,sleep=lambda _:None)
    assert json.loads((tmp_path/'postflight_idle.json').read_text())['strict_idle']


@pytest.mark.parametrize('field,value',[('processes',[{'pid':123}]),('used_mib',2048)])
def test_postflight_does_not_wait_for_foreign_process_or_allocation(monkeypatch,tmp_path,field,value):
    row=dict(uuid='gpu',processes=[],used_mib=1,gpu_util_percent=1);row[field]=value
    monkeypatch.setattr(guard.base,'telemetry',lambda *a:[row])
    monkeypatch.setattr(guard.base,'require_idle',lambda *a: (_ for _ in ()).throw(ValueError('occupied')))
    with pytest.raises(ValueError,match='occupied'):
        guard.postflight_idle({},tmp_path,['gpu'],clock=lambda:0,sleep=lambda _:pytest.fail('must fail immediately'))


def test_training_source_unchanged_by_eval_scripts():
    from dynamicwam.dualrate.native_v7_source_compat import validate_deployment_source
    assert validate_deployment_source(common.TRAIN_SHA)['protected_model_and_training_equal']
