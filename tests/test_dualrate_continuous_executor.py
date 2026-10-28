"""Owner-loop regressions for real continuous execution, not clock-label checks."""
from types import SimpleNamespace

import pytest

from dynamicwam.dualrate.deployment import (
    ContinuousWallPhysicsBridge, continuous_clock_evidence,
    dispatch_due_queries, action_response_statistics,
)
from dynamicwam.dualrate.timing import TimingError, DeadlineMiss, BufferUnderrun
from test_dualrate_eval_executor import fixture


def test_fixed_origin_rejects_freeze_credit_and_checks_original_elapsed():
    wall=[10.]
    bridge=ContinuousWallPhysicsBridge(.004,wall_clock=lambda:wall[0])
    bridge.activate(0.)
    wall[0]=11.
    with pytest.raises(TimingError,match='backlog'):
        bridge.check_backlog(.5,.25)
    bridge.wall_origin+=.5
    with pytest.raises(TimingError,match='origin changed'):
        bridge.check_backlog(.5,.25)


def test_full09_backlog_slack_does_not_extend_query_deadline():
    bridge=ContinuousWallPhysicsBridge(.004,wall_clock=lambda:10.254)
    bridge.activate(0.,10.)
    assert bridge.check_backlog(0.,.25)==pytest.approx(.254)
    with pytest.raises(TimingError,match='backlog'):
        bridge.check_backlog(0.,.25,wall_time=10.256)
    with pytest.raises(DeadlineMiss):
        bridge.check_deadline(.25,wall_now=10.251)


def test_raw_clock_acceptance_rejects_compressed_active_time():
    bridge=ContinuousWallPhysicsBridge(.004,wall_clock=lambda:200.)
    bridge.activate(0.,10.)
    with pytest.raises(TimingError,match='raw-clock'):
        continuous_clock_evidence(bridge,activation_wall=10.,activation_physics=0.,
            end_wall=200.,end_physics=126.,physics_steps=31500,
            max_lag=.04,min_lag=0.,max_backlog=.25)


def test_wall_stale_capture_cannot_start_query_even_when_physically_fresh():
    calls=[];events=[];deferred={}
    snapshot=(SimpleNamespace(exposure=1.,available=1.04),)
    runtime=SimpleNamespace(due=lambda now:(True,False),
        store=SimpleNamespace(snapshot=lambda now:None),start_fast=lambda *a,**k:calls.append(k))
    dispatch_due_queries(runtime,SimpleNamespace(max_observation_age=.24),
        SimpleNamespace(snapshot=lambda now:snapshot),SimpleNamespace(),now=1.1,initial=0.,
        vae=None,emit=events.append,deferred=deferred,seen_references=set(),
        exposure_walls={1.:10.},wall_clock=lambda:10.3)
    assert not calls and deferred=={'fast':1.1}
    assert events[0]['observation_age']==pytest.approx(.1)
    assert events[0]['observation_wall_age']==pytest.approx(.3)


def test_uncovered_queue_fails_without_freeze_or_fabricated_hold():
    def start(*args,**kwargs):raise BufferUnderrun('cannot commit actions absent from queue')
    runtime=SimpleNamespace(due=lambda now:(True,False),
        store=SimpleNamespace(snapshot=lambda now:None),start_fast=start)
    with pytest.raises(BufferUnderrun):
        dispatch_due_queries(runtime,SimpleNamespace(max_observation_age=.24),
            SimpleNamespace(snapshot=lambda now:(SimpleNamespace(exposure=1.,available=1.04),)),
            SimpleNamespace(completed_execution_trace=lambda *args:()),now=1.1,initial=0.,
            vae=None,emit=lambda e:None,deferred={},seen_references=set())


def test_pending_fast_and_camera_never_stop_physics(monkeypatch):
    mod,session,system,builder,camera,config,order,wall,runtimes=fixture(monkeypatch)
    base_runtime=mod.DualRateRuntime
    class PendingRuntime(base_runtime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.clock=args[2];self.started=False
            self.store=SimpleNamespace(snapshot=lambda now:SimpleNamespace(version=1))
        def due(self,now):
            return self.bridge.active and now>=2*config.action_dt and not self.started,False
        def start_fast(self,*args,**kwargs):
            if not self.bridge.active:return
            self.started=True;self.ready_wall=wall.value+.12
            self.fast_job=(SimpleNamespace(done=lambda:wall.value>=self.ready_wall),None)
        def poll(self):
            if self.fast_job is not None and self.fast_job[0].done():self.fast_job=None
    monkeypatch.setattr(mod,'DualRateRuntime',PendingRuntime)
    original_begin=camera.begin
    def begin(env,now):
        pending=original_begin(env,now)
        ready=wall.value+.08
        pending.render_submitted_wall_perf_s=ready
        pending.readback_started_wall_perf_s=ready
        pending.readback_finished_wall_perf_s=ready
        pending.finished_wall_perf_s=ready
        return pending
    camera.begin=begin
    camera.ready=lambda pending:wall.value>=pending.finished_wall_perf_s
    original_finish=camera.finish
    def finish(pending=None):
        selected=camera.pending if pending is None else pending
        wall.value=max(wall.value,selected.finished_wall_perf_s)
        return original_finish(pending)
    camera.finish=finish
    builder.max_observation_age=.24
    report=mod.run_episode(session,system,builder,camera,config,action_limit=4)
    assert report['physics_steps']==180
    assert report['physics_steps_during_fast_inflight']>=20
    assert report['physics_steps_during_camera']>=40
    evidence=report['raw_clock_acceptance']
    assert evidence['checks']['immutable_activation'] and evidence['passed']
    assert evidence['active_wall_seconds']==pytest.approx(evidence['physical_duration'],abs=.01)
    assert evidence['final_wall_time']-evidence['activation_wall_time']==evidence['active_wall_seconds']
    assert report['events'][-1]['event']=='episode_workers_closed'


def test_rewarm_and_armed_barrier_stay_outside_active_clock(monkeypatch):
    mod,session,system,builder,camera,config,order,wall,runtimes=fixture(monkeypatch)
    def ready():
        assert not runtimes[0].bridge.active and session.physics_time()==0.
        wall.value+=2.;order.append('ready')
    def armed():
        assert not runtimes[0].bridge.active and session.physics_time()==0.
        order.append('armed')
    report=mod.run_episode(session,system,builder,camera,config,action_limit=2,
        on_ready=ready,on_armed=armed)
    assert order==['ready','armed','runtime_close']
    assert len([e for e in report['events'] if e.get('phase')=='post_ready_rewarm'])==3
    assert report['raw_clock_acceptance']['passed']


def test_actual_response_reports_first_executed_mutable_row_not_gpu_latency():
    events=[dict(event='activation',wall_time=10.),
        dict(event='fast_committed',query_id=1,reference_version=2,query_time=.36,
            observation_time=.18,protected_until=.9,query_request_wall_time=10.36,
            observation_exposure_wall_time=10.18),
        dict(event='execute',start_time=.72,actual_start=.72,execution_wall_time=10.72),
        dict(event='execute',start_time=.9,actual_start=.9,execution_wall_time=10.904),
        dict(event='execute',start_time=1.08,actual_start=1.08,execution_wall_time=11.084)]
    result=action_response_statistics(events)['action_response_events']
    assert len(result)==1
    assert result[0]['query_to_execution_wall_seconds']==pytest.approx(.544)
    assert result[0]['observation_to_execution_wall_seconds']==pytest.approx(.724)
