import time
import pytest
import torch
from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
from dynamicwam.dualrate.runtime import DualRateRuntime,RuntimeConfig
from dynamicwam.dualrate.submission import FastMailbox
from dynamicwam.dualrate.timing import ExecutionQueue,TimedPlan,DeadlineMiss,StalePlan,TimingError
from test_dualrate_runtime import RuntimeModelFixture


def make():
    physical=[0.];wall=[100.]
    bridge=WallPhysicsBridge(.004,wall_clock=lambda:wall[0]);bridge.activate(0.,100.)
    queue=ExecutionQueue(TimedPlan(0.,.18,((0.,),)*16,1,0.))
    events=[]
    runtime=DualRateRuntime(RuntimeModelFixture().eval(),queue,lambda:physical[0],
        RuntimeConfig(.18,submission_contract='worker_receipt_v8'),action_mean=[0.],action_scale=[1.],
        clock_bridge=bridge,record=events.append)
    def batch(lease,reference):
        return dict(query_time=torch.tensor([lease.query_time],dtype=torch.float64),
            state=torch.zeros(1,1),action_times=torch.arange(16,dtype=torch.float64)[None]*.18)
    return runtime,physical,wall,events,batch


def wait(runtime):
    runtime.fast_job[0].result(timeout=3)


def test_owner_stall_does_not_reject_actual_on_time_delivery_or_publish_in_past():
    runtime,p,w,events,batch=make()
    try:
        # Real mailbox arrival .100, while the physical owner has not advanced.
        w[0]=100.1
        runtime.start_fast(batch,observation_time=0.);wait(runtime)
        lease=runtime.fast_job[1]
        w[0]=100.45;p[0]=.004
        assert runtime.poll() is None
        assert runtime.queue.action_at(.004)==(0.,)
        p[0]=.1
        runtime.poll()
        assert runtime.fast_job is None
        assert runtime.queue.action_at(.35)==(0.,)
        assert runtime.queue.action_at(.36)==(102.,)
        event=next(e for e in events if e['event']=='fast_committed')
        assert event['submitted_wall_time']==100.1 and event['commit_wall_time']==100.45
        assert lease.protected_until==.36
    finally:runtime.close()


def test_receipt_cannot_backdate_actual_late_submission():
    runtime,p,w,_,_=make()
    lease=runtime.queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    box=runtime._mailbox;box.register(lease)
    plan=TimedPlan(0.,.18,((0.,),)*16,2,0.,lease)
    w[0]=100.4
    with pytest.raises(DeadlineMiss):box.submit(lease,(plan,{},100.01,{}))
    assert box.peek(lease) is None
    with pytest.raises(DeadlineMiss):box.check_pending(lease)
    box.finish(lease);runtime.queue.cancel_query(lease);runtime.close()


def test_physical_activation_deadline_and_immutable_prefix_remain_strict():
    runtime,p,w,_,batch=make()
    try:
        w[0]=100.05;runtime.start_fast(batch,observation_time=0.);wait(runtime)
        p[0]=.36;w[0]=100.4
        with pytest.raises(DeadlineMiss):runtime.poll()
    finally:runtime.close()


def test_mailbox_duplicate_cancel_and_cross_episode_cannot_replace_receipt():
    runtime,p,w,_,_=make();box=runtime._mailbox
    lease=runtime.queue.begin_query(query_time=0.,observation_time=0.,protect_steps=2,reference_version=-1)
    box.register(lease);plan=TimedPlan(0.,.18,((0.,),)*16,2,0.,lease)
    value=(plan,{},100.,{})
    box.submit(lease,value)
    with pytest.raises(StalePlan):box.submit(lease,value)
    box.finish(lease)
    with pytest.raises(StalePlan):box.submit(lease,value)
    runtime.queue.cancel_query(lease);runtime.close()


def test_measured_training_trace_uses_receipt_not_delayed_owner_poll():
    from test_dualrate_exposure_index import helper,measured_trace,DT
    events=measured_trace()
    starts={e['query_id']:e for e in events if e['event']=='fast_started'}
    for e in list(events):
        if e['event']!='fast_committed':continue
        start=starts[e['query_id']];start['prefix_hash']='exact-committed-content'
        submitted=e['ready_wall_time']+.001
        e.update(submission_contract='worker_receipt_v8',submitted_wall_time=submitted,
                 commit_wall_time=100.+start['protected_until']+.1)
        events.append(dict(event='fast_submitted',query_id=e['query_id'],submitted_wall_time=submitted,
                           prefix_hash=start['prefix_hash']))
    assert helper.audit_wall_release(events,DT)>0
    next(e for e in events if e['event']=='fast_submitted')['submitted_wall_time']+=.001
    with pytest.raises(ValueError,match='receipt'):helper.audit_wall_release(events,DT)


def test_partial_tail_keeps_real_executable_targets_without_fabricating_full_chunk():
    import numpy as np
    from test_dualrate_exposure_index import helper,DT
    from dynamicwam.dualrate.exposure_timing import synthetic_templates,validate_template
    times=np.arange(160)*DT/3;templates=synthetic_templates(DT)['templates']
    full=list(helper.eligible_rows(times,templates,DT,2*DT,'same_episode',2))
    tail=list(helper.eligible_rows(times,templates,DT,2*DT,'same_episode',2,partial_fast_tail=True))
    assert len(tail)>len(full)
    assert any(times[f]+validate_template(templates[i],DT)['fast_action_start']+16*DT>times[-1] for f,i in tail)
    for f,i in tail:
        t=validate_template(templates[i],DT)
        assert times[f]+t['fast_action_start']+(t['fast_prefix_rows']+2)*DT<=times[-1]+1e-6
