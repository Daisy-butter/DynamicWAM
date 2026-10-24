import numpy as np
import pytest
import torch
from dynamicwam.dualrate.online import ObservationHistory,OnlineFrame,OnlineBatchBuilder,history_features,composite
from dynamicwam.dualrate.timing import ExecutionQueue,TimedPlan,TimingError


def observation(value=0):
    return dict(observation={c:dict(rgb=np.full((48,64,3),value,np.uint8))
        for c in ('head_camera','left_camera','right_camera')},joint_action=dict(vector=np.arange(14,dtype=np.float32)))


def test_snapshot_requires_arrival_and_immutable_copies():
    h=ObservationHistory();o=observation();h.append(o,exposure=1.,available=1.1)
    with pytest.raises(TimingError):h.snapshot(1.05)
    o['observation']['head_camera']['rgb'][:]=255
    frame=h.snapshot(1.1)[0]
    assert frame.exposure==1. and frame.available==1.1 and frame.images[0].sum()==0
    with pytest.raises(ValueError):frame.images[0][:]=3
    with pytest.raises(TimingError):h.append(observation(),exposure=1.,available=1.2)


def contract():
    return dict(compute_size=[64,64],normalization_percentile=99.,
        farneback=dict(flags=0,iterations=3,levels=3,poly_n=5,poly_sigma=1.2,pyr_scale=.5,winsize=15),
        quality=dict(method='forward_backward_consistency_v1',minimum_reliable_fraction=.1,
            relative_error=.01,absolute_error_squared=.5))


def test_flow_uses_actual_timestamps_and_twelve_observed_frames():
    rng=np.random.default_rng(42);image=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    frames=[]
    for i in range(61):
        t=.06*i + .003*i*i
        frames.append(OnlineFrame((np.roll(image,i//12,axis=1),)*3,np.zeros(14,np.float32),t,t,i))
    h=history_features(frames,contract())
    assert h['motion'].shape==(4,12)
    expected=torch.tensor([[frames[i-12].exposure,frames[i].exposure] for i in (24,36,48,60)],dtype=torch.float64)
    assert h['motion_valid'].all()
    assert torch.equal(h['motion_times'],expected)
    assert torch.equal(h['spatial_flow_times'],expected[-1])
    assert not torch.allclose(expected[:,1]-expected[:,0],torch.full((4,),.72,dtype=torch.float64))


def test_initial_missing_flow_is_masked_not_invented():
    o=observation();h=ObservationHistory();h.append(o,exposure=0.,available=0.)
    features=history_features(h.snapshot(0.),contract())
    assert not features['motion_valid'].any() and not features['motion_acceleration_valid'].any()
    assert not features['spatial_flow_valid'].any() and np.count_nonzero(features['flow_rgb'])==0


def test_online_condition_has_no_label_or_oracle_fields():
    h=ObservationHistory();h.append(observation(20),exposure=2.,available=2.01)
    manifest=dict(flow_contract=contract(),rgb_size=[240,320],rgb_view_sizes=[[240,320],[120,160],[120,160]],
                  active_entity_roles=['tracked_dynamic_target','left_gripper','right_gripper'])
    lang=dict(context=torch.zeros(512,4096,dtype=torch.bfloat16),mask=torch.ones(512,dtype=torch.bool))
    builder=OnlineBatchBuilder(manifest,action_mean=np.ones(14),action_scale=np.ones(14)*2,language=lang)
    x=builder.current(h.snapshot(2.02),query_time=2.02,action_start=1.98)
    assert x['rgb'].shape==(1,3,3,240,320)
    assert torch.equal(x['state'][0],(torch.arange(14)-1)/2)
    assert x['action_times'].dtype==x['rgb_times'].dtype==torch.float64
    assert float(x['action_times'][0,0])==1.98
    assert x['executed_actions'].shape==(1,0,14)
    assert x['future_times'].shape==(1,12) and x['query_entity_valid'].shape==(1,3)
    assert not {'future','contact','source_action','fast_action','future_loss_valid'}&x.keys()
    with pytest.raises(TimingError,match='too old'):
        builder.current(h.snapshot(2.4),query_time=2.4,action_start=2.34)


def test_completed_trace_excludes_active_and_unissued_plan():
    q=ExecutionQueue(TimedPlan(0.,.18,((1.,),(2.,),(3.,)),0,0.))
    q.action_at(.01);q.action_at(.19)
    trace=q.completed_execution_trace(0.,.2)
    assert trace==((0.,.18,(1.,)),)
    assert q.completed_execution_trace(0.,.36)==((0.,.18,(1.,)),(.18,.36,(2.,)))


def test_mid_interval_query_keeps_active_endpoint_and_global_grid():
    q=ExecutionQueue(TimedPlan(0.,.18,tuple((float(i),) for i in range(16)),0,0.))
    assert q.action_at(.2)==(1.,)
    lease=q.begin_query(query_time=.21,observation_time=.18,protect_steps=2,reference_version=4,submit_margin=.01)
    assert lease.prefix.start_time==.18 and lease.protected_until==.54
    plan=TimedPlan(.18,.18,tuple((100.+i,) for i in range(16)),1,.21,lease)
    q.replace(plan,now=.31)
    assert q.action_at(.32)==(1.,) and q.action_at(.37)==(2.,)
    assert q.action_at(.54)==(102.,)
