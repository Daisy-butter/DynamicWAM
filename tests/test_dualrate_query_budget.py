import pytest

from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig
from dynamicwam.dualrate.timing import ExecutionQueue, TimedPlan, DeadlineMiss, BufferUnderrun


def fixture(*,query=.448,wall=None,budget=.32,cap=3,protect=2,active=True,rows=16):
    w=[100.+query if wall is None else 100.+wall]
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:w[0])
    if active:bridge.activate(0.,100.)
    runtime=DualRateRuntime.__new__(DualRateRuntime)
    runtime.queue=ExecutionQueue(TimedPlan(0.,.18,tuple((float(i),) for i in range(rows)),1,0.))
    runtime.config=RuntimeConfig(.18,protect_steps=protect,min_query_budget_seconds=budget,max_protect_steps=cap)
    runtime.clock_bridge=bridge;events=[];runtime.record=events.append
    return runtime,events,w


def test_mid_row_budget_selects_three_before_lease_and_preserves_commitment():
    runtime,events,w=fixture()
    runtime.queue.action_at(.448)
    count,receipt=runtime._fast_query_protection(.448,w[0])
    assert count==3 and receipt['query_phase_seconds']==pytest.approx(.088)
    assert receipt['bounded_query_budget_enabled']
    assert runtime.queue._inflight is None and not events
    lease=runtime.queue.begin_query(query_time=.448,observation_time=.36,
        protect_steps=count,reference_version=-1,submit_margin=.01)
    assert lease.protected_until==pytest.approx(.90)
    assert lease.protected_until-.01-.448>=.32
    assert lease.prefix.actions==((2.,),(3.,),(4.,))
    with pytest.raises(DeadlineMiss):
        runtime.queue.replace(TimedPlan(.36,.18,((0.,),)*16,2,.448,lease),now=.891)


def test_wall_lag_counts_and_excess_budget_fails_without_a_lease():
    runtime,events,w=fixture(query=.36,wall=.40)
    count,receipt=runtime._fast_query_protection(.36,w[0])
    assert count==3 and receipt['wall_physics_lag_seconds']==pytest.approx(.04)
    runtime,events,w=fixture(query=.52,wall=.54,budget=.55)
    with pytest.raises(DeadlineMiss,match='bounded protected prefix'):
        runtime._fast_query_protection(.52,w[0])
    assert runtime.queue._inflight is None and runtime.queue._committed_end==0
    assert events[-1]['event']=='fast_budget_rejected'


def test_existing_commitments_and_missing_coverage_are_not_overridden():
    runtime,events,w=fixture()
    runtime.queue.commit_prefix(query_time=.448,until=1.26)
    with pytest.raises(DeadlineMiss):runtime._fast_query_protection(.448,w[0])
    assert runtime.queue._inflight is None
    runtime,events,w=fixture(rows=4)
    count,_=runtime._fast_query_protection(.448,w[0])
    with pytest.raises(BufferUnderrun):
        runtime.queue.begin_query(query_time=.448,observation_time=.36,
            protect_steps=count,reference_version=-1,submit_margin=.01)
    assert runtime.queue._inflight is None


def test_disabled_budget_preserves_legacy_protect_four_and_bootstrap_two():
    runtime,_,w=fixture(budget=0.,cap=3,protect=4)
    assert runtime._fast_query_protection(.448,w[0])[0]==4
    assert RuntimeConfig(.18,protect_steps=4).max_protect_steps is None
    runtime,_,w=fixture(active=False)
    count,receipt=runtime._fast_query_protection(.448,w[0])
    assert count==2 and not receipt['bounded_query_budget_enabled']
    runtime,_,w=fixture(protect=1,cap=2,budget=.16)
    assert runtime._fast_query_protection(.448,w[0])[0]==2
    with pytest.raises(ValueError):RuntimeConfig(.18,min_query_budget_seconds=.32)
