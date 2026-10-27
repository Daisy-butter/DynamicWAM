"""Wall/physical causal gates and exposure-prefix contracts; CPU only."""
from concurrent.futures import Future
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
from dynamicwam.dualrate.timing import (
    ExecutionQueue, TimedPlan, TimingError, DeadlineMiss, StalePlan, BufferUnderrun,
    prefix_digest,
)
from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig


def queue(dt=.1):
    return ExecutionQueue(TimedPlan(0.,dt,tuple((float(i),) for i in range(16)),1,0.))


def test_bootstrap_has_no_wall_deadline_and_activation_is_once():
    wall=[100.]
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:wall[0])
    assert bridge.ready(100.,0.)
    bridge.check_deadline(.2, .01)
    with pytest.raises(TimingError,match='bootstrap'):
        bridge.availability_time(100.)
    bridge.activate(0.)
    with pytest.raises(TimingError,match='only once'):
        bridge.activate(0.)


def test_wall_ready_during_physics_stall_cannot_publish_in_the_past():
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:100.106)
    bridge.activate(0.,100.)
    assert bridge.availability_time(100.106)==pytest.approx(.108)
    assert not bridge.ready(100.106,0.)
    assert not bridge.ready(100.106,.104)
    assert bridge.ready(100.106,.108)


def test_real_float_physics_quantum_is_not_replaced_by_nominal_time():
    h=0.004000000189989805
    bridge=WallPhysicsBridge(h,wall_clock=lambda:10.+80*h)
    bridge.activate(3.,10.)
    assert bridge.physics_dt==h
    assert bridge.availability_time(10.+45*h)==3.+45*h
    assert bridge.availability_time(10.+45.5*h)==3.+46*h


def test_wall_deadline_rejects_late_result_even_if_physics_has_not_advanced():
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:100.3)
    bridge.activate(0.,100.)
    with pytest.raises(DeadlineMiss,match='wall-clock'):
        bridge.check_deadline(.2,.01,ready_wall=100.15)
    with pytest.raises(DeadlineMiss):
        bridge.check_deadline(.2,.01,ready_wall=100.2,wall_now=100.18)


def test_backlog_is_observed_not_repaired_by_mutating_physics():
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:100.3)
    bridge.activate(0.,100.)
    with pytest.raises(TimingError,match='backlog'):
        bridge.check_backlog(0.,.25)
    assert bridge.snapshot(0.)['physical_time']==0.
    assert bridge.snapshot(0.)['expected_physical_time']==pytest.approx(.3)
    assert not bridge.caught_up(0.)
    assert bridge.caught_up(.3)


def test_exposure_prefix_joins_only_real_history_and_existing_commitments():
    q=queue()
    q.commit_prefix(query_time=0.,until=.4)
    for t in (0.,.1,.2):q.action_at(t)
    before=q._committed_end
    prefix=q.source_prefix_from_exposure(0.,.25)
    assert prefix.actions==((0.,),(1.,),(2.,),(3.,))
    assert prefix.start_time==0. and prefix.end_time==pytest.approx(.4)
    assert q._committed_end==before and prefix.queue_version==1
    assert len(prefix_digest(prefix))==64
    assert q.source_prefix_from_exposure(.2,.25).actions==((2.,),(3.,))


def test_source_prefix_cannot_substitute_queued_actions_for_missing_history():
    q=queue();q.commit_prefix(query_time=0.,until=.3)
    q.action_at(.1)  # .0 was never actually issued.
    with pytest.raises(BufferUnderrun,match='issued history'):
        q.source_prefix_from_exposure(0.,.15)


def test_source_prefix_has_no_new_reservation_and_rejects_wrong_grid_or_exhaustion():
    q=queue()
    assert q.source_prefix_from_exposure(0.,0.).actions==()
    assert q._committed_end==0
    with pytest.raises(TimingError,match='action grid'):
        q.source_prefix_from_exposure(.05,.05)
    with pytest.raises(StalePlan,match='horizon'):
        q.source_prefix_from_exposure(0.,1.6)
    q.commit_prefix(query_time=0.,until=1.6)
    with pytest.raises(StalePlan,match='horizon'):
        q.source_prefix_from_exposure(0.,0.)


def test_grid_tolerance_preserves_the_actual_exposure_value():
    q=queue();e=.3+1e-8
    prefix=q.source_prefix_from_exposure(e,e)
    assert prefix.start_time==e


def test_exposure_prefix_boundary_does_not_promote_the_next_unissued_row():
    q=queue();q.commit_prefix(query_time=0.,until=.2)
    q.action_at(0.);q.action_at(.1)
    assert q.source_prefix_from_exposure(0.,.2).actions==((0.,),(1.,))
    assert q._committed_end==2
    q.action_at(.2)
    assert q.source_prefix_from_exposure(0.,.2).actions==((0.,),(1.,),(2.,))


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast=nn.Linear(1,1)
        self.fast.config=SimpleNamespace(action_dim=1,chunk_size=16)
        self.slow=nn.Linear(1,1)


def runtime_fixture():
    p=[0.];w=[100.]
    bridge=WallPhysicsBridge(.01,wall_clock=lambda:w[0]);bridge.activate(0.,100.)
    q=queue()
    runtime=DualRateRuntime(Fixture().eval(),q,lambda:p[0],RuntimeConfig(.1),
        action_mean=[0.],action_scale=[1.],clock_bridge=bridge)
    return runtime,q,p,w


def test_ready_fast_result_waits_for_real_physics_then_commits():
    runtime,q,p,w=runtime_fixture()
    lease=q.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    job=Future()
    plan=TimedPlan(0.,.1,tuple((float(i+20),) for i in range(16)),2,0.,lease)
    job.set_result((plan,{},100.05,.05));runtime.fast_job=(job,lease)
    try:
        w[0]=100.05
        assert runtime.poll() is None and runtime.fast_job is not None
        assert q._version==1
        p[0]=.05
        assert runtime.poll()=={}
        assert q._version==2 and runtime.fast_job is None
    finally:runtime.close()


def test_ready_slow_reference_uses_owner_publication_clock_after_transfer():
    runtime,q,p,w=runtime_fixture();prefix=q.snapshot_committed(0.)
    world=dict(world_intervals=torch.tensor([[[0.,1.]]],dtype=torch.float64),
        world_types=torch.ones(1,1,dtype=torch.long),world_valid=torch.ones(1,1,dtype=torch.bool),
        source_time=torch.tensor([0.],dtype=torch.float64),completed_time=torch.tensor([0.],dtype=torch.float64),
        generation_stage='fixture')
    device_world={k:v.clone() for k,v in world.items() if isinstance(v,torch.Tensor)}
    job=Future();job.set_result((world,device_world,100.06,dict(wall_seconds=.06)))
    runtime.slow_job=(job,0.,prefix,1)
    try:
        w[0]=100.06
        runtime.poll()
        assert runtime.store.snapshot(0.) is None and runtime.slow_job is not None
        p[0]=.07
        runtime.poll()
        ref=runtime.store.snapshot(.07)
        assert ref.source_time==0. and ref.completed_time==.07
        assert runtime._published_device_world[0]==ref.version
    finally:runtime.close()


def test_wall_late_fast_result_cannot_receive_catchup_credit():
    runtime,q,p,w=runtime_fixture()
    lease=q.begin_query(query_time=0.,observation_time=0.,protect_steps=2,
        reference_version=-1,submit_margin=.01)
    job=Future();job.set_result((TimedPlan(0.,.1,((0.,),)*16,2,0.,lease),{},100.22,.22))
    runtime.fast_job=(job,lease)
    try:
        w[0]=100.22
        with pytest.raises(DeadlineMiss,match='wall-clock'):runtime.poll()
        assert p[0]==0. and q._version==1 and q._inflight is None
    finally:runtime.close()


def test_pending_fast_worker_cannot_continue_past_deadline_using_old_tail():
    runtime,q,p,w=runtime_fixture();events=[];runtime.record=events.append
    lease=q.begin_query(query_time=0.,observation_time=0.,protect_steps=2,
        reference_version=-1,submit_margin=.01)
    job=Future();runtime.fast_job=(job,lease)
    try:
        w[0]=100.2
        with pytest.raises(DeadlineMiss):runtime.poll()
        assert not job.done() and q._inflight is None and q._version==1
        with pytest.raises(TimingError,match='has failed'):runtime.poll()
    finally:
        # The timed-out worker finishes later. Close records its actual timing,
        # but never commits the returned plan or resumes execution.
        plan=TimedPlan(0.,.1,((0.,),)*16,2,0.,lease)
        job.set_result((plan,{},100.3,dict(wall_seconds=.3,gpu_done_wall_time=100.29)))
        runtime.close()
    discarded=[e for e in events if e['event']=='worker_finished_discarded']
    assert len(discarded)==1 and discarded[0]['query_id']==lease.query_id
    assert discarded[0]['ready_wall_time']==100.3 and discarded[0]['wall_seconds']==.3
    assert not discarded[0]['commit_performed_during_close'] and q._version==1
    assert not any(e['event']=='fast_committed' for e in events)


def test_host_commit_crossing_wall_deadline_is_rejected_not_reported_accepted():
    runtime,q,p,w=runtime_fixture();events=[];runtime.record=events.append
    lease=q.begin_query(query_time=0.,observation_time=0.,protect_steps=2,
        reference_version=-1,submit_margin=.01)
    job=Future();job.set_result((TimedPlan(0.,.1,((0.,),)*16,2,0.,lease),{},100.05,.05))
    runtime.fast_job=(job,lease)
    original=q.replace
    def expensive_commit(plan,*,now):
        result=original(plan,now=now)
        w[0]=100.2
        return result
    q.replace=expensive_commit
    try:
        p[0]=.05;w[0]=100.05
        with pytest.raises(DeadlineMiss):runtime.poll()
        assert q._inflight is None
        assert events[-1]['event']=='fast_rejected' and events[-1]['queue_updated_before_failure']
        assert not any(event['event']=='fast_committed' for event in events)
    finally:runtime.close()
