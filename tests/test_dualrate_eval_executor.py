"""Actual owner-loop checks with CPU scene/clock and inference-lifecycle fixtures."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


def fixture(monkeypatch, *, poll_failure=False, close_failure=False, swallowed_contact_failure=False):
    from dynamicwam.dualrate import deployment as mod
    from dynamicwam.dualrate.runtime import RuntimeConfig
    h=float(np.float32(.004));dt=45*h;order=[]
    class Wall:
        value=100.
        def now(self):return self.value
        def sleep(self,seconds):self.value+=max(0.,seconds)
    wall=Wall()
    monkeypatch.setattr(mod.time,'perf_counter',wall.now)
    monkeypatch.setattr(mod.time,'sleep',wall.sleep)
    monkeypatch.setattr(mod,'bind_gil_releasing_physx_step',lambda *a:None)
    native=ModuleType('dynamicwam.runtime.domino_rt._sapien_color_reader')
    native.step_cpu_physx=lambda x:None
    monkeypatch.setitem(sys.modules,'dynamicwam.runtime.domino_rt._sapien_color_reader',native)
    projection=SimpleNamespace(serialized_bounds=[],provenance='CPU fixture')
    monkeypatch.setattr(mod.ActuatorProjection,'from_robot',lambda robot:projection)
    class Robot:
        left_entity=right_entity=None
        def _entity_qf(self,*args):pass
        def set_arm_joints(self,*args):pass
        def set_gripper(self,*args):pass
    class Scene:
        ticks=0;native_reads=0
        def step(self):self.ticks+=1
        def get_contacts(self):
            self.native_reads+=1
            if swallowed_contact_failure:raise RuntimeError('native contact failure')
            return []
    scene=Scene()
    env=SimpleNamespace(scene=scene,robot=Robot(),take_action_cnt=0,_update_kinematic_tasks=lambda:None)
    env._update_transient_checks=(lambda:None) if swallowed_contact_failure else lambda:scene.get_contacts()
    def metrics():
        try:scene.get_contacts()
        except RuntimeError:
            if not swallowed_contact_failure:raise
    session=SimpleNamespace(env=env,physics_time=lambda:scene.ticks*h,timestep=lambda:h,
        _consume_events=(lambda:None) if swallowed_contact_failure else lambda:scene.get_contacts(),
        stop_reason=lambda:None,metrics_tracker=SimpleNamespace(on_step=metrics))
    class Camera:
        pending=None;counter=0
        def begin(self,env,now):
            before=wall.value;wall.value+=.012;self.counter+=1
            observation=dict(observation={name:dict(rgb=np.zeros((2,3,3),np.uint8))
                for name in ('head_camera','left_camera','right_camera')},
                joint_action=dict(vector=np.zeros(14,np.float32)))
            self.pending=SimpleNamespace(physics_time_s=now,capture_id=self.counter,
                capture_wall_perf_s=before,render_submitted_wall_perf_s=wall.value,
                readback_started_wall_perf_s=wall.value,readback_finished_wall_perf_s=wall.value,
                finished_wall_perf_s=wall.value,camera_readback_seconds={},observation=observation)
            return self.pending
        def ready(self,p):return True
        def finish(self,p=None):
            result=self.pending;self.pending=None;return result
    runtimes=[]
    class Runtime:
        slow_job=fast_job=None
        def __init__(self,*a,clock_bridge,action_projection,**kw):
            self.bridge=clock_bridge;self.action_projection=action_projection
            self.store=SimpleNamespace(snapshot=lambda now:object());runtimes.append(self)
        def start_slow(self,*a,**kw):pass
        def start_fast(self,*a,**kw):pass
        def due(self,*a):return False,False
        def poll(self):
            if poll_failure and self.bridge.active:raise RuntimeError('active inference failure')
        def fast_graph_receipt(self):return dict(enabled=False)
        def close(self):
            order.append('runtime_close')
            if close_failure:raise RuntimeError('cleanup failure')
    monkeypatch.setattr(mod,'DualRateRuntime',Runtime)
    builder=SimpleNamespace(hardtime=True,action_dt=dt,mean=np.zeros(14),scale=np.ones(14),
                            manifest={'time_contract':'exposure_anchor_wall_release_v4'})
    system=SimpleNamespace(slow=SimpleNamespace(native=SimpleNamespace(compact_wan=
        SimpleNamespace(video_model=SimpleNamespace(vae=None)))))
    return mod,session,system,builder,Camera(),RuntimeConfig(action_dt=dt),order,wall,runtimes


def test_ready_barrier_is_inactive_and_every_real_step_reuses_contacts(monkeypatch):
    mod,session,system,builder,camera,config,order,wall,runtimes=fixture(monkeypatch)
    def ready():
        assert session.physics_time()==0. and not runtimes[0].bridge.active
        wall.value+=2.;order.append('ready')
    report=mod.run_episode(session,system,builder,camera,config,action_limit=4,on_ready=ready)
    assert order==['ready','runtime_close']
    assert report['realtime_valid'] and report['physics_steps']==4*45
    assert report['contact_cache']['native_reads']==session.env.scene.native_reads==4*45
    assert report['contact_cache']['cache_hits']>=4*45
    assert report['bootstrap_wall_seconds']>=2.
    assert not hasattr(session.env.scene,'_full_step_contact_cache')
    assert 'get_contacts' not in vars(session.env.scene)
    assert report['events'][-1]['contact_cache_restored']


def test_failure_hook_precedes_worker_drain_and_preserves_original_error(monkeypatch):
    mod,session,system,builder,camera,config,order,_,_=fixture(monkeypatch,poll_failure=True,close_failure=True)
    def failed(exc):
        assert exc.dualrate_events and not exc.dualrate_report['realtime_valid']
        order.append('failure_hook')
    with pytest.raises(RuntimeError,match='active inference failure') as error:
        mod.run_episode(session,system,builder,camera,config,action_limit=4,on_failure=failed)
    assert order==['failure_hook','runtime_close']
    assert error.value.dualrate_cleanup_errors==['RuntimeError: cleanup failure']
    assert not hasattr(session.env.scene,'_full_step_contact_cache')


def test_best_effort_metrics_cannot_hide_native_contact_failure(monkeypatch):
    mod,session,system,builder,camera,config,order,_,_=fixture(monkeypatch,swallowed_contact_failure=True)
    with pytest.raises(RuntimeError,match='native contact failure'):
        mod.run_episode(session,system,builder,camera,config,action_limit=4,
                        on_failure=lambda exc:order.append('failure_hook'))
    assert order==['failure_hook','runtime_close'] and session.env.scene.native_reads==1
    assert not hasattr(session.env.scene,'_full_step_contact_cache')


def test_ready_failure_restores_cache_and_notifies_before_close(monkeypatch):
    mod,session,system,builder,camera,config,order,_,runtimes=fixture(monkeypatch)
    def failed_barrier():raise RuntimeError('peer failed before activation')
    with pytest.raises(RuntimeError,match='peer failed'):
        mod.run_episode(session,system,builder,camera,config,action_limit=4,on_ready=failed_barrier,
                        on_failure=lambda exc:order.append('failure_hook'))
    assert order==['failure_hook','runtime_close'] and not runtimes[0].bridge.active
    assert not hasattr(session.env.scene,'_full_step_contact_cache')
