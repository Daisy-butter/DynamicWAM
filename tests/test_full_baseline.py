"""Sync16 boundary and full owner-loop tests; no CUDA/SAPIEN model required."""
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

from dynamicwam.dualrate.full_baseline import FullChunkSchedule, query_observation_timing


def plan(offset=0.):
    return np.repeat((np.arange(16, dtype=np.float64)+offset)[:,None], 14, axis=1)


def schedule():
    s=FullChunkSchedule(np.full(14,-1.),45,2);s.bootstrap(plan());return s


def test_bootstrap_prepends_holds_without_overwriting_full_output():
    s=schedule()
    assert s.start_row==2 and s.end_row==18
    assert [s.action(i)[1] for i in range(2)]==['bootstrap_hold']*2
    np.testing.assert_array_equal(np.stack([s.action(i)[0] for i in range(2,18)]),plan())
    assert not s.can_query(18*45-1)
    assert s.can_query(18*45)


@pytest.mark.parametrize('tick,last_row,expected',[(810,17,18),(810,18,19),(811,18,19),(854,18,19),(855,18,19)])
def test_return_uses_next_unstarted_grid_row(tick,last_row,expected):
    s=schedule();s.begin_query(810)
    assert s.install(plan(20),tick=tick,last_started_row=last_row)==expected
    assert s.end_row==expected+16
    np.testing.assert_array_equal(s.action(expected)[0],np.full(14,20.))
    if expected>18:
        np.testing.assert_array_equal(s.action(18)[0],np.full(14,15.))


def test_no_second_query_while_inflight_or_previous_full_chunk_remains():
    s=schedule()
    with pytest.raises(RuntimeError):s.begin_query(809)
    s.begin_query(810)
    assert not s.can_query(9000)
    with pytest.raises(RuntimeError):s.begin_query(9000)
    s.install(plan(20),tick=905,last_started_row=20)
    assert not s.can_query(s.end_row*45-1)
    assert s.can_query(s.end_row*45)


def test_schedule_copies_returns_and_rejects_bad_or_early_plan():
    raw=plan();s=FullChunkSchedule(np.zeros(14),45);s.bootstrap(raw);raw[:]=99
    assert s.action(2)[0][0]==0
    returned=s.action(2)[0];returned[:]=55
    assert s.action(2)[0][0]==0
    s.begin_query(810)
    with pytest.raises(RuntimeError):s.install(plan(),tick=809,last_started_row=17)
    with pytest.raises(ValueError):s.install(np.zeros((15,14)),tick=810,last_started_row=17)
    assert s.inflight


def test_wait_holds_keep_final_command_without_fabricating_a_second_plan():
    s=schedule();s.begin_query(810)
    for row in range(18,28):
        action,kind=s.action(row)
        assert kind=='inference_wait_hold'
        np.testing.assert_array_equal(action,plan()[-1])


def _owner_fixture(monkeypatch, *, fail_active=False, active_delay=.4,
                   camera_seconds=.012, step_seconds=0.):
    import torch
    from dynamicwam.dualrate import full_baseline as mod
    from dynamicwam.dualrate import deployment, actuators

    class Wall:
        t=100.
        def now(self):return self.t
        def sleep(self,s):self.t+=max(0.,s)
    wall=Wall();h=float(np.float32(.004))
    monkeypatch.setattr(mod.time,'perf_counter',wall.now)
    monkeypatch.setattr(mod.time,'sleep',wall.sleep)
    # The actual camera helpers share the time module and see this same clock.
    monkeypatch.setattr(deployment,'bind_gil_releasing_physx_step',lambda *args:None)
    native=ModuleType('dynamicwam.runtime.domino_rt._sapien_color_reader');native.step_cpu_physx=lambda x:None
    monkeypatch.setitem(sys.modules,'dynamicwam.runtime.domino_rt._sapien_color_reader',native)
    monkeypatch.setattr(mod,'build_native_packet',lambda frames,policy,**kw:dict(state=frames[-1].drive_target.copy()))
    class Projection:
        serialized_bounds=[];provenance='cpu_fixture'
        def __call__(self,actions,mask):return actions,dict(changed_coordinates=0)
    monkeypatch.setattr(actuators.ActuatorProjection,'from_robot',lambda robot:Projection())
    class Robot:
        left_entity=right_entity=None
        def _entity_qf(self,*args):pass
        def set_arm_joints(self,*args):pass
        def set_gripper(self,*args):pass
    class Scene:
        ticks=0
        def step(self):self.ticks+=1;wall.t+=step_seconds
        def get_contacts(self):return []
    scene=Scene()
    env=SimpleNamespace(robot=Robot(),scene=scene,take_action_cnt=0,
                        _update_kinematic_tasks=lambda:None,_update_transient_checks=lambda:None)
    metric_steps=[]
    session=SimpleNamespace(env=env,physics_time=lambda:scene.ticks*h,timestep=lambda:h,
        _consume_events=lambda:None,stop_reason=lambda:None,
        metrics_tracker=SimpleNamespace(on_step=lambda:metric_steps.append(scene.ticks)))
    class Camera:
        pending=None;counter=0
        def begin(self,env,now):
            before=wall.t;wall.t+=camera_seconds
            self.counter+=1
            obs=dict(observation={x:dict(rgb=np.zeros((2,3,3),np.uint8)) for x in
                                  ('head_camera','left_camera','right_camera')},
                     joint_action=dict(vector=np.zeros(14,np.float32)))
            self.pending=SimpleNamespace(physics_time_s=now,capture_id=self.counter,
                capture_wall_perf_s=before,render_submitted_wall_perf_s=wall.t,
                readback_started_wall_perf_s=wall.t,readback_finished_wall_perf_s=wall.t,
                finished_wall_perf_s=wall.t,camera_readback_seconds={},observation=obs)
            return self.pending
        def ready(self,p):return True
        def finish(self,p=None):
            result=self.pending;self.pending=None;return result
    calls=[]
    class Policy:
        runtime=SimpleNamespace(device='cpu',num_inference_steps=10,video_refresh_steps=(0,1),chunk_size=16)
        def sample(self,packet):
            calls.append(scene.ticks)
            if fail_active and len(calls)>1:raise RuntimeError('intentional model failure')
            return np.zeros((16,14),np.float32)
    pools=[]
    class Future:
        def __init__(self,fn,args):
            self.failure=None
            try:
                actions,ready,info=fn(*args)
                self.ready=ready+(.4 if args[1] == 0 else active_delay)
                self.value=(actions,self.ready,dict(info,ready_wall_time=self.ready))
            except BaseException as exc:
                self.failure=exc;self.ready=wall.t+.4
        def done(self):return wall.t>=self.ready
        def result(self,timeout=None):
            if timeout is not None:wall.t=max(wall.t,self.ready)
            if self.failure is not None:raise self.failure
            return self.value
        def cancelled(self):return False
    class Pool:
        def __init__(self,**kw):self.closed=False;pools.append(self)
        def submit(self,fn,*args):return Future(fn,args)
        def shutdown(self,**kw):self.closed=True
    monkeypatch.setattr(mod,'ThreadPoolExecutor',Pool)
    return mod,session,Policy(),Camera(),pools,calls,metric_steps,h


def test_actual_owner_loop_counts_waits_budget_and_never_queries_early(monkeypatch):
    mod,session,policy,camera,pools,calls,metrics,h=_owner_fixture(monkeypatch)
    barriers=[]
    report=mod.run_episode(session,policy,camera,action_dt=45*h,action_limit=40,
                          on_ready=lambda:barriers.append(session.physics_time()))
    assert report['realtime_valid'] and report['stop_reason']=='action_budget_exhausted'
    assert report['physics_steps']==40*45 and report['completed_targets']==40
    assert report['started_targets']==session.env.take_action_cnt==40
    assert report['generated_targets_completed']==32
    assert report['hold_rows_started']==8
    assert report['hold_rows_started']+report['generated_targets_started']==40
    assert calls[0]==0 and calls[1]>=18*45 and calls[2]>=37*45
    assert barriers==[0.] and pools[0].closed and camera.pending is None
    assert len(metrics)==40 and report['rows_dropped']==0
    installed=[e for e in report['events'] if e['event']=='full_plan_installed']
    assert installed[0]['install_row']>=21
    assert all(e['publication_time']>=e['wall_release_lower_bound']-1e-7 for e in installed)
    assert all(e['installation_time']>=e['publication_time']-1e-7 for e in installed)
    assert report['events'][-1]['event']=='episode_workers_closed'
    assert report['events'][-1]['post_terminal']
    assert not hasattr(session.env.scene, '_full_step_contact_cache')
    assert all(x['calls']==40*45 for x in report['owner_stage_timing']['totals'].values())
    assert len(report['owner_stage_timing']['recent_rows'])==4


def test_model_error_drains_owned_worker_and_attaches_failure_evidence(monkeypatch):
    mod,session,policy,camera,pools,calls,metrics,h=_owner_fixture(monkeypatch,fail_active=True)
    with pytest.raises(RuntimeError,match='intentional model failure') as caught:
        mod.run_episode(session,policy,camera,action_dt=45*h,action_limit=40)
    assert pools[0].closed
    assert not hasattr(session.env.scene, '_full_step_contact_cache')
    assert not caught.value.dualrate_report['realtime_valid']
    assert caught.value.dualrate_report['stop_reason']=='infrastructure_error'
    assert any(e['event']=='full_query_started' for e in caught.value.dualrate_events)


@pytest.mark.parametrize('physical,wall_age,fresh', [(.12,.18,True),(.12,.244,False)])
def test_full_query_uses_wall_age_even_when_physical_clock_is_behind(physical, wall_age, fresh):
    from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
    bridge=WallPhysicsBridge(float(np.float32(.004)),wall_clock=lambda:100.+wall_age)
    bridge.activate(0.,100.)
    assert not bridge.caught_up(physical)
    accepted,timing=query_observation_timing(bridge,physical_now=physical,
        request_wall=100.+wall_age,exposure=0.,max_observation_age=.24)
    assert accepted is fresh
    assert timing['query_time']==physical and timing['request_wall_time']==100.+wall_age
    assert timing['observation_physical_age']==physical
    assert timing['observation_wall_age']==pytest.approx(wall_age)
    assert timing['observation_time']==0.


def test_full_query_accepts_nominal_60_actual_ticks_but_not_61():
    from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
    h=float(np.float32(.004));bridge=WallPhysicsBridge(h)
    bridge.activate(0.,100.)
    for ticks,expected in [(60,True),(61,False)]:
        accepted,_=query_observation_timing(bridge,physical_now=ticks*h,
            request_wall=100.+ticks*h,exposure=0.,max_observation_age=.24)
        assert accepted is expected


def test_owner_dispatches_full_after_complete_chunk_while_physics_catches_up(monkeypatch):
    mod,session,policy,camera,pools,calls,metrics,h=_owner_fixture(
        monkeypatch,camera_seconds=.1,step_seconds=.002)
    report=mod.run_episode(session,policy,camera,action_dt=45*h,action_limit=40)
    queries=[e for e in report['events'] if e['event']=='full_query_started']
    assert queries and all(e['observation_age']<=.2400001 for e in queries)
    assert any(e['query_wall_equivalent_time']-e['query_time']>h+1e-7 for e in queries)
    assert calls[1]>=18*45  # Two startup holds followed by all 16 model rows.
    installed=[e for e in report['events'] if e['event']=='full_plan_installed']
    assert installed and all(e['publication_time']>=e['wall_release_lower_bound']-1e-7 for e in installed)
    assert all(e['installation_time']>=e['publication_time']-1e-7 for e in installed)
    assert report['query_requires_physics_caught_up'] is False
    assert pools[0].closed and camera.pending is None


@pytest.mark.parametrize('timeout', [False, True])
def test_failure_is_signalled_before_waiting_for_worker_shutdown(monkeypatch, timeout):
    mod,session,policy,camera,pools,calls,metrics,h=_owner_fixture(
        monkeypatch, fail_active=not timeout, active_delay=.8 if timeout else .4)
    signalled=[]
    def on_failure(exc):
        assert not pools[0].closed
        assert exc.dualrate_report['stop_reason']=='infrastructure_error'
        assert any(e['event']=='full_query_started' for e in exc.dualrate_events)
        signalled.append(exc)
    with pytest.raises(Exception, match='bounded inference timeout' if timeout else 'intentional model failure') as caught:
        mod.run_episode(session,policy,camera,action_dt=45*h,action_limit=40,
                        model_query_timeout_s=.5, on_failure=on_failure)
    assert signalled==[caught.value]
    assert pools[0].closed and camera.pending is None


def test_failure_callback_error_preserves_original_failure_and_cleanup(monkeypatch):
    mod,session,policy,camera,pools,calls,metrics,h=_owner_fixture(monkeypatch,fail_active=True)
    def on_failure(exc):
        assert not pools[0].closed
        raise OSError('marker unavailable')
    with pytest.raises(RuntimeError,match='intentional model failure') as caught:
        mod.run_episode(session,policy,camera,action_dt=45*h,action_limit=40,on_failure=on_failure)
    assert caught.value.dualrate_report['failure_callback_error']=='OSError: marker unavailable'
    assert pools[0].closed and camera.pending is None


@pytest.mark.parametrize('with_provenance', [False, True])
def test_actual_native_profile_passes_strict_runtime_schema_without_cuda(monkeypatch, with_provenance):
    from copy import deepcopy
    from pathlib import Path
    from dynamicwam.inference import model_loader as loader
    config=loader.load_deploy_config(Path(__file__).resolve().parents[1]/'configs/dualrate_native_v2.yaml')
    metadata={'checkpoint_artifact_id','checkpoint_manifest','external_assets_manifest'}
    assert metadata <= config.keys()
    numerical={key:value for key,value in config.items() if key not in metadata}
    if not with_provenance:
        config=deepcopy(numerical)
    before=deepcopy(config)
    model=object();encoder=object();normalizer=object();seen=[]
    monkeypatch.setattr(loader,'_load_checkpoint_payload',lambda path: {'config':{}})
    def build_model(runtime_config,payload,*,device):
        assert runtime_config==numerical and device=='cuda'
        seen.append(runtime_config)
        return model
    monkeypatch.setattr(loader,'_build_model',build_model)
    monkeypatch.setattr(loader,'_build_t5_encoder',lambda config,device:encoder)
    monkeypatch.setattr(loader,'_build_action_normalizer',lambda config,checkpoint_config:normalizer)
    runtime=loader.build_runtime_from_config(config)
    assert seen and config==before
    assert runtime.model is model and runtime.t5_encoder is encoder and runtime.action_normalizer is normalizer
    assert runtime.num_inference_steps==10 and runtime.video_refresh_steps==(0,1)
    assert runtime.chunk_size==16 and runtime.head_flow_config==config['head_flow']
    assert runtime.action_interval_seconds==config['inference']['action_interval_ms']/1000.


@pytest.mark.parametrize('location', ['top', 'inference'])
def test_native_runtime_still_rejects_unknown_fields_before_loading(monkeypatch, location):
    from pathlib import Path
    from dynamicwam.inference import model_loader as loader
    config=loader.load_deploy_config(Path(__file__).resolve().parents[1]/'configs/dualrate_native_v2.yaml')
    target=config if location=='top' else config['inference']
    target['unknown_metadata']=True
    def forbidden(*args,**kwargs):
        pytest.fail('invalid config reached checkpoint/model loading')
    monkeypatch.setattr(loader,'_load_checkpoint_payload',forbidden)
    monkeypatch.setattr(loader,'_build_model',forbidden)
    with pytest.raises(ValueError,match='unknown_metadata'):
        loader.build_runtime_from_config(config)
