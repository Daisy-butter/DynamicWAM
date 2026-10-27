from types import SimpleNamespace

import numpy as np
import pytest

from dynamicwam.dualrate.deployment import (
    bind_gil_releasing_physx_step,capture_timing,dispatch_due_queries,
    next_capture_deadline,warm_camera)
from dynamicwam.dualrate.producer import CONDITION_RGB_MAX_AGE_SECONDS
from dynamicwam.dualrate.online import ObservationHistory
from dynamicwam.dualrate.timing import ExecutionQueue,TimedPlan,DeadlineMiss,BufferUnderrun
from dynamicwam.runtime.domino_rt.camera_capture import _read_submitted_rgb_batch


def observation():
    return dict(observation={name:dict(rgb=np.zeros((2,3,3),np.uint8))
        for name in ('left_camera','right_camera','head_camera')},
        joint_action=dict(vector=np.zeros(14,np.float32)))


class FakeRuntime:
    def __init__(self,queue):
        self.queue=queue;self.now=.36;self.fast=[];self.slow=[]
        self.store=SimpleNamespace(snapshot=lambda now:SimpleNamespace(version=1))

    def due(self,now):return not self.fast,not self.slow

    def start_fast(self,factory,*,observation_time):
        lease=self.queue.begin_query(query_time=self.now,observation_time=observation_time,
            protect_steps=2,reference_version=1,submit_margin=.01)
        self.fast.append(lease)
        factory(lease,None)

    def start_slow(self,factory,prefix,*,source_time):
        self.slow.append((prefix,source_time));factory()


def test_stale_frames_defer_without_lease_and_fresh_arrival_keeps_actual_grid():
    queue=ExecutionQueue(TimedPlan(0.,.18,tuple((float(i),) for i in range(16)),0,0.))
    queue.action_at(.36)
    runtime=FakeRuntime(queue);history=ObservationHistory()
    history.append(observation(),exposure=.06,available=.184)
    conditions=[]
    builder=SimpleNamespace(max_observation_age=.060001,
        current=lambda frames,**kw:conditions.append(('fast',frames[-1].exposure,kw)),
        source=lambda frames,**kw:conditions.append(('slow',frames[-1].exposure,kw)))
    events=[];deferred={};seen=set()
    def dispatch():
        dispatch_due_queries(runtime,builder,history,queue,now=runtime.now,initial=0.,vae=None,
            emit=events.append,deferred=deferred,seen_references=seen)
    dispatch();dispatch()
    assert not runtime.fast and not runtime.slow and len(events)==2
    assert set(deferred)=={'fast','slow'} and not conditions
    # A completed fresh exposure arrives at a mid-action physical time.
    history.append(observation(),exposure=.36,available=.4)
    runtime.now=.4;queue.action_at(.4);dispatch()
    lease=runtime.fast[0]
    assert lease.query_time==.4 and lease.observation_time==.36
    assert lease.prefix.start_time==.36 and lease.protected_until==.72
    assert lease.prefix.actions==((2.,),(3.,))
    assert lease.protected_until-lease.submit_margin-lease.query_time==pytest.approx(.31)
    assert not deferred and seen=={1}
    assert conditions[0][2]['query_time']==.4 and conditions[1][2]['source_time']==.4
    # Deferral does not buy a new ceiling-grid action or excuse a late return.
    plan=TimedPlan(.36,.18,tuple((100.+i,) for i in range(16)),1,.4,lease)
    with pytest.raises(DeadlineMiss):queue.replace(plan,now=.711)


def test_measured_readback_age_is_current_and_keeps_its_exposure():
    measured=(0.0800000038,0.06400000304,0.11200000532,0.11600000551,0.05600000266,
              0.18000000855,0.11600000551,0.11200000532,0.10000000475,0.09600000456,
              0.11600000551,0.1200000057,0.06800000323,0.14000000665,0.11200000532,
              0.09600000456,0.11600000551,0.06000000285,0.228000010829,0.05200000247,
              0.10000000475,0.09200000437)
    assert sum(age<=.060001 for age in measured)==3
    assert all(age<=CONDITION_RGB_MAX_AGE_SECONDS for age in measured)
    queue=ExecutionQueue(TimedPlan(0.,.18,tuple((float(i),) for i in range(16)),0,0.))
    queue.action_at(.36)
    runtime=FakeRuntime(queue);history=ObservationHistory()
    history.append(observation(),exposure=0.,available=.05)
    conditions=[]
    builder=SimpleNamespace(max_observation_age=CONDITION_RGB_MAX_AGE_SECONDS,
        current=lambda frames,**kw:conditions.append(frames[-1].exposure),
        source=lambda frames,**kw:None)
    events=[];deferred={};seen=set()
    dispatch_due_queries(runtime,builder,history,queue,now=.36,initial=0.,vae=None,
        emit=events.append,deferred=deferred,seen_references=seen)
    assert set(deferred)=={'fast','slow'} and not conditions
    history.append(observation(),exposure=.248,available=.36)
    runtime.fast.clear();deferred.clear()
    dispatch_due_queries(runtime,builder,history,queue,now=.36,initial=0.,vae=None,
        emit=events.append,deferred=deferred,seen_references=seen)
    assert runtime.fast[0].observation_time==pytest.approx(.248)
    assert conditions==[.248] and not deferred


def test_deferred_queries_do_not_prevent_queue_underflow():
    q=ExecutionQueue(TimedPlan(0.,.18,((0.,),(1.,)),0,0.))
    q.action_at(.2)
    with pytest.raises(BufferUnderrun):q.action_at(.36)


def test_capture_phase_partition_and_fixed_warmup_clock(monkeypatch):
    captured=SimpleNamespace(capture_wall_perf_s=1.,render_submitted_wall_perf_s=1.02,
        readback_started_wall_perf_s=1.03,readback_finished_wall_perf_s=1.06,
        finished_wall_perf_s=1.08,camera_readback_seconds={'head':.03},physics_time_s=0.)
    timing=capture_timing(captured,owner_processed_wall=1.09)
    assert timing['total_wall_seconds']==pytest.approx(.09)
    assert sum(timing[name] for name in timing if name.endswith('_wall_seconds')
        and name!='total_wall_seconds')==pytest.approx(.09)
    assert timing['metadata_and_reader_dispatch_wall_seconds']==pytest.approx(.01)
    calls=[];events=[]
    camera=SimpleNamespace(begin=lambda env,t:calls.append(t),finish=lambda:captured)
    session=SimpleNamespace(physics_time=lambda:0.,env=object())
    monkeypatch.setattr('dynamicwam.dualrate.deployment.time.perf_counter',lambda:1.1)
    assert warm_camera(camera,session,count=3,emit=events.append) is captured
    assert calls==[0.,0.,0.] and len(events)==3
    assert all(event['event']=='camera_warmup' for event in events)
    captured.physics_time_s=.004
    with pytest.raises(Exception,match='advanced'):warm_camera(camera,session,count=1,emit=events.append)


def test_native_batch_called_once_in_original_four_camera_order():
    cameras=tuple((name,object()) for name in ('left','right','front','head'))
    calls=[]
    def batch(handles):
        calls.append(handles)
        return [np.full((2,3,3),i,np.uint8) for i in range(4)],[.001]*4
    result,*_=_read_submitted_rgb_batch(cameras,batch,lambda:0.)
    assert calls==[tuple(x[1] for x in cameras)] and tuple(result)==('left','right','front','head')
    assert all(result[name]['rgb'].flags.owndata for name,_ in cameras)
    with pytest.raises(ValueError):
        _read_submitted_rgb_batch(cameras,lambda handles:([np.zeros((2,3,4),np.uint8)]*4,[.01]*4),lambda:0.)


def test_next_capture_waits_out_a_render_longer_than_one_frame():
    # 0.09 s render plus one owner frame at exposure 0.06 s lands on 0.24 s.
    assert next_capture_deadline(0., .06, .06, .09)==.24
    assert next_capture_deadline(0., .06, .06, .05)==.18
    assert next_capture_deadline(0., .18, .06, .094)==.36
    with pytest.raises(ValueError):
        next_capture_deadline(0., .06, .06, -1.)


def test_gil_release_stays_inside_the_scene_step_clock():
    calls=[]
    class Clock:
        def __init__(self):
            self._original_step=lambda: calls.append("original")
            self.step_index=0
            self.time_seconds=0.
        def snapshot(self):
            return SimpleNamespace(time_seconds=self.time_seconds)
    class Scene:
        physx_system="physx"
        def __init__(self, clock):
            self.clock=clock
        def step(self):
            self.clock._original_step()
            self.clock.step_index+=1
            self.clock.time_seconds+=.004
    clock=Clock();scene=Scene(clock)
    env=SimpleNamespace(_scene_step_clock=clock,scene=scene)
    def step_physx(system):
        assert system=="physx"
        calls.append("physx")
    bind_gil_releasing_physx_step(env, step_physx)
    scene.step()
    assert calls==["physx"]
    assert clock.step_index==1 and clock.time_seconds==.004
    with pytest.raises(RuntimeError, match="SceneStepClock is not installed"):
        bind_gil_releasing_physx_step(SimpleNamespace(scene=scene), step_physx)


def test_owner_simulator_calls_release_the_gil():
    import threading
    import time
    import sapien.physx
    import sapien.render
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    hits=0
    stop=False
    def run():
        nonlocal hits
        while not stop:
            hits+=1
    thread=threading.Thread(target=run)
    thread.start()
    time.sleep(.01)
    before=hits
    _sapien_color_reader.sleep_releasing_gil(.15)
    progressed=hits-before
    stop=True
    thread.join()
    assert progressed>1000
    camera=sapien.render.RenderCameraComponent(8,8)
    with pytest.raises(RuntimeError,match='not added to scene'):
        _sapien_color_reader.take_picture(camera)
    _sapien_color_reader.step_cpu_physx(sapien.physx.PhysxCpuSystem())
