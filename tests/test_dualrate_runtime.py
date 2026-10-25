import time
from dataclasses import replace
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from dynamicwam.dualrate.timing import (ExecutionQueue, TimedPlan, DeadlineMiss,
    TimingError, StalePlan, CompletedReferenceStore, prefix_digest)
from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig


def initial():
    return TimedPlan(0.,.1,tuple((float(i),) for i in range(16)),1,0.)


def test_late_result_cannot_be_salvaged_by_dropping_rows():
    queue=ExecutionQueue(initial())
    lease=queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    for t in (0.,.1,.2,.3): queue.action_at(t)
    candidate=TimedPlan(0.,.1,tuple((float(i+100),) for i in range(16)),2,0.,lease)
    with pytest.raises(DeadlineMiss,match="protected"):
        queue.replace(candidate,now=.4)
    assert queue.action_at(.4)==(4.,)


def test_submission_boundary_and_margin_are_strict():
    for completion in (.19,.2,.21):
        queue=ExecutionQueue(initial())
        lease=queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,
                                reference_version=-1,submit_margin=.01)
        with pytest.raises(DeadlineMiss):
            queue.replace(replace(initial(),version=2,lease=lease),now=completion)


def test_commit_preserves_raw_prefix_and_replaces_only_tail():
    queue=ExecutionQueue(initial())
    lease=queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    values=tuple((float(i+100),) for i in range(16))
    queue.replace(TimedPlan(0.,.1,values,2,0.,lease),now=.14)
    assert queue.action_at(.15)==(1.,)
    assert queue.action_at(.2)==(102.,)
    with pytest.raises(StalePlan):
        queue.replace(TimedPlan(0.,.1,values,3,0.,lease),now=.21)


def test_clock_and_episode_and_missing_lease_rejected():
    queue=ExecutionQueue(initial(),episode_id="a")
    with pytest.raises(TimingError,match="lease"):
        queue.replace(replace(initial(),version=2),now=.05)
    lease=queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    for bad in (replace(lease,episode_id="b"),replace(lease,clock_id="other")):
        with pytest.raises(StalePlan):
            queue.replace(replace(initial(),version=2,lease=bad),now=.05)
    with pytest.raises(TimingError,match="in flight"):
        queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)


def test_reference_ttl_after_skipped_refresh_and_payload_identity():
    queue=ExecutionQueue(initial())
    prefix=queue.snapshot_committed(0.)
    store=CompletedReferenceStore(max_age=.5)
    args=dict(source_time=0.,completed_time=.1,version=1,generation_stage="joint2",
        future_times=(.2,.4,.8),future_valid=(True,True,True),payload={"value":torch.ones(1)},
        now=.1,episode_id="episode",clock_id="physical",source_queue_version=1,
        source_prefix_hash=prefix_digest(prefix))
    assert store.publish(**args)
    assert store.snapshot(.49) is not None
    assert store.snapshot(.5) is None
    args["payload"]={"source_time":torch.tensor([.01],dtype=torch.float64)}
    with pytest.raises(TimingError,match="payload clock"):
        store.publish(**args)


class RuntimeModelFixture(nn.Module):
    def __init__(self):
        super().__init__();self.fast=nn.Linear(1,1)
        self.fast.config=SimpleNamespace(action_dim=1)
    def empty_world(self,batch): return {}
    def sample_fast(self,batch,world,**kwargs):
        a=torch.arange(100.,116.).reshape(1,16,1)
        a=torch.where(batch["fast_known_mask"][...,None],batch["fast_known_actions"],a)
        return {"actions":a}


def test_actual_runtime_wires_model_output_into_leased_queue():
    now=[0.];queue=ExecutionQueue(initial())
    model=RuntimeModelFixture().eval();events=[]
    runtime=DualRateRuntime(model,queue,lambda:now[0],RuntimeConfig(.1),
        action_mean=[0.],action_scale=[1.],record=events.append)
    def batch(lease,reference):
        return dict(query_time=torch.tensor([lease.query_time],dtype=torch.float64),
            state=torch.zeros(1,1),action_times=torch.arange(16,dtype=torch.float64)[None]*.1)
    try:
        runtime.start_fast(batch,observation_time=0.)
        deadline=time.monotonic()+3
        while not runtime.fast_job[0].done():
            if time.monotonic()>deadline: raise AssertionError("fixture worker timeout")
            time.sleep(.001)
        now[0]=.14;runtime.poll()
        assert queue.action_at(.15)==(1.,)
        assert queue.action_at(.2)==(102.,)
        assert events[-1]["event"]=="fast_committed"
    finally: runtime.close()


def test_slow_prefix_from_previous_episode_is_rejected_before_model_work():
    queue=ExecutionQueue(initial(),episode_id='current')
    old=ExecutionQueue(initial(),episode_id='previous').snapshot_committed(0.)
    runtime=DualRateRuntime(RuntimeModelFixture().eval(),queue,lambda:0.,RuntimeConfig(.1),
        action_mean=[0.],action_scale=[1.])
    try:
        with pytest.raises(StalePlan,match='another episode'):
            runtime.start_slow({},old)
        assert runtime.slow_job is None
    finally:runtime.close()

def test_fast_and_slow_never_share_the_device():
    now=[0.];events=[]
    queue=ExecutionQueue(TimedPlan(0.,.18,((0.,),)*16,1,0.))
    runtime=DualRateRuntime(RuntimeModelFixture().eval(),queue,lambda:now[0],RuntimeConfig(.18),
        action_mean=[0.],action_scale=[1.],record=events.append)
    try:
        runtime.last_fast_tick=runtime.last_slow_tick=-10**12
        # Both periods are due and the device is idle: the slow reference starts,
        # and the fast query waits rather than sharing the GPU.
        assert runtime.due(.72)==(False,True)
        runtime.slow_job=('future',.72,None,2)
        runtime.last_slow_tick=4
        assert runtime.due(.72)==(False,False)
        assert runtime.fast_align_boundary
        assert events[-1]['event']=='fast_held' and events[-1]['slow_version']==2
        assert runtime.due(1.10)[0] is False
        runtime.slow_job=None
        # Mid-row resume would shrink the 0.35 s budget, and slow stays parked.
        assert runtime.due(1.50)==(False,False)
        assert runtime.due(1.62)==(True,False)
    finally:
        runtime.slow_job=None
        runtime.close()


def test_candidate_runtime_configuration_is_executable():
    import json
    from pathlib import Path
    from dynamicwam.dualrate.runtime import RuntimeConfig
    config=json.loads((Path(__file__).parents[1]/'configs/dualrate_joint_v2_smoke.json').read_text())
    runtime=RuntimeConfig(**config['runtime'])
    assert runtime.action_dt==.18 and runtime.reference_max_age==1.44
