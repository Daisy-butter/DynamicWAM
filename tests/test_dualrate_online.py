import numpy as np
import pytest
import torch
from dynamicwam.dualrate.online import (ObservationHistory,OnlineFrame,OnlineBatchBuilder,
    history_endpoints,history_features,composite)
from dynamicwam.dualrate.producer import CAMERA_AVAILABILITY_SECONDS, CONDITION_RGB_MAX_AGE_SECONDS
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


def test_flow_uses_actual_timestamps_at_raw_frame_offsets():
    rng=np.random.default_rng(42);image=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    frames=[]
    for i in range(61):
        t=.06*i
        frames.append(OnlineFrame((np.roll(image,i//12,axis=1),)*3,np.zeros(14,np.float32),t,t,i))
    h=history_features(frames,contract())
    assert h['motion'].shape==(4,12)
    expected=torch.tensor([[frames[i-12].exposure,frames[i].exposure] for i in (24,36,48,60)],dtype=torch.float64)
    assert h['motion_valid'].all()
    assert torch.equal(h['motion_times'],expected)
    assert torch.equal(h['spatial_flow_times'],expected[-1])
    captures=[OnlineFrame((image,)*3,np.zeros(14,np.float32),.12*i,.12*i,i) for i in range(40)]
    offsets=(60,48,36,24,12,0)
    chosen,matched=history_endpoints(captures)
    assert all(matched)
    assert [frame.exposure for frame in chosen]==pytest.approx(
        [captures[-1].exposure-k*.06 for k in offsets])
    bent=[OnlineFrame((image,)*3,np.zeros(14,np.float32),.06*i+.003*i*i,.06*i+.003*i*i,i) for i in range(61)]
    chosen,matched=history_endpoints(bent)
    assert [frame.frame_id for frame in chosen]!=[bent[max(0,60-k)].frame_id for k in offsets]
    assert chosen[-1] is bent[-1] and matched[-1]
    assert not all(matched)
    for frame,offset,ok in zip(chosen,offsets,matched):
        target=bent[-1].exposure-offset*.06
        if ok:
            assert abs(frame.exposure-target)<=.06+1e-6
        else:
            assert frame is bent[-1]
            assert min(abs(item.exposure-target) for item in bent)>.06


def test_missing_flow_endpoint_cannot_validate_the_long_interval():
    image=np.zeros((64,64,3),np.uint8)
    present=(0.,.72,2.16,2.88,3.60)
    frames=[OnlineFrame((image,)*3,np.zeros(14,np.float32),t,t,i) for i,t in enumerate(present)]
    chosen,matched=history_endpoints(frames)
    assert matched==(True,True,False,True,True,True)
    assert chosen[2] is frames[-1]
    features=history_features(frames,contract())
    assert features['motion_valid'].tolist()[:2]==[False,False]
    assert float(features['motion_times'][0,0])==pytest.approx(3.60)
    assert float(features['motion_times'][0,1])==pytest.approx(3.60)
    assert float(features['motion'][0].abs().sum())==0.


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
    assert float(x['rgb_times'][0,0])==2.
    assert x['executed_actions'].shape==(1,0,14)
    aged=ObservationHistory();aged.append(observation(4),exposure=2.,available=2.12)
    seen=builder.current(aged.snapshot(2.2),query_time=2.2,action_start=2.16)
    assert float(seen['rgb_times'][0,0])==2. and float(seen['query_time'])==2.2
    assert builder.max_observation_age==CONDITION_RGB_MAX_AGE_SECONDS
    assert builder.camera_availability_seconds==CAMERA_AVAILABILITY_SECONDS
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
