from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dynamicwam.dualrate.deployment import (exact_grid_index, next_grid_capture_tick,
    dispatch_due_queries, model_overlap_statistics)
from dynamicwam.dualrate.factory import place_inference_devices
from dynamicwam.dualrate.online import ObservationHistory, OnlineBatchBuilder
from dynamicwam.dualrate.timing import ExecutionQueue, TimedPlan, TimingError


H = float(np.float32(.004))
DT = 45 * H


def observation():
    return dict(observation={name:dict(rgb=np.zeros((32,32,3),np.uint8))
        for name in ('head_camera','left_camera','right_camera')},
        joint_action=dict(vector=np.zeros(14,np.float32)))


def test_actual_physics_grid_never_relabels_off_grid_exposure():
    assert exact_grid_index(45*H,0.,DT)==1
    assert exact_grid_index(100_000*DT,0.,DT)==100_000
    with pytest.raises(TimingError,match='not on'):
        exact_grid_index(45*H+.004,0.,DT)
    with pytest.raises(TimingError,match='not on'):
        exact_grid_index(100_000*.18,0.,DT)
    assert next_grid_capture_tick(45,45,H,.09)==90
    assert next_grid_capture_tick(45,45,H,.20)==135


@pytest.mark.parametrize('ticks,accepted',[(59,True),(60,True),(61,False)])
def test_dispatch_camera_age_uses_training_numeric_epsilon(ticks,accepted):
    # Nominal 240ms is sixty real float32 physics ticks, not exactly .24.
    now=ticks*H
    history=ObservationHistory();history.append(observation(),exposure=0.,available=.12)
    calls=[];deferred={}
    runtime=SimpleNamespace(due=lambda now:(True,False),store=SimpleNamespace(snapshot=lambda now:None),
        start_fast=lambda factory,**kwargs:calls.append(kwargs))
    builder=SimpleNamespace(max_observation_age=.24)
    queue=SimpleNamespace(completed_execution_trace=lambda since,until:())
    dispatch_due_queries(runtime,builder,history,queue,now=now,initial=0.,vae=None,
        emit=lambda event:None,deferred=deferred,seen_references=set())
    assert bool(calls) is accepted
    assert ('fast' in deferred) is not accepted
    if accepted:assert calls[0]['observation_time']==0.


def test_slow_dispatch_keeps_exposure_and_request_separate():
    e=2*DT;r=e+.12
    queue=ExecutionQueue(TimedPlan(0.,DT,tuple((float(i),) for i in range(16)),0,0.))
    for t in (0.,DT,e,r):queue.action_at(t)
    queue.commit_prefix(query_time=r,until=4*DT)
    history=ObservationHistory();history.append(observation(),exposure=e,available=e+.1)
    inputs=[];calls=[]
    builder=SimpleNamespace(hardtime=True,max_observation_age=.24,
        source=lambda frames,**kw:inputs.append((frames,kw)))
    runtime=SimpleNamespace(due=lambda now:(False,True),
        start_slow=lambda factory,prefix,**kw:(calls.append((prefix,kw)),factory()))
    dispatch_due_queries(runtime,builder,history,queue,now=r,initial=0.,vae=None,
                         emit=lambda event:None,deferred={},seen_references=set())
    assert calls[0][0].start_time==e
    assert calls[0][0].actions==((2.,),(3.,))
    assert calls[0][1]['source_time']==e and calls[0][1]['request_time']==r
    assert inputs[0][1]['source_time']==e and inputs[0][1]['action_start']==e
    assert inputs[0][1]['request_time']==r
    # Reusing an already submitted exposure cannot relabel it as a new source.
    dispatch_due_queries(runtime,builder,history,queue,now=r,initial=0.,vae=None,
                         emit=lambda event:None,deferred={},seen_references=set())
    assert len(calls)==1


def test_source_builder_exposure_anchor_and_causal_availability(monkeypatch):
    e=2*DT;r=e+.12
    history=ObservationHistory();history.append(observation(),exposure=e,available=e+.1)
    frames=history.snapshot(r)
    manifest=dict(time_contract='exposure_anchor_wall_release_v4',action_interval_seconds=DT,
                  flow_contract={},active_entity_roles=['target','left','right'])
    language=dict(context=torch.zeros(512,4096),mask=torch.ones(512,dtype=torch.bool))
    builder=OnlineBatchBuilder(manifest,action_mean=np.zeros(14),action_scale=np.ones(14),language=language)
    def features(*args,**kwargs):
        return dict(flow_rgb=np.zeros((4,32,32,3),np.uint8),motion=torch.zeros(4,12),
            motion_valid=torch.zeros(4,dtype=torch.bool),motion_acceleration_valid=torch.zeros(4,dtype=torch.bool),
            motion_times=torch.full((4,2),e,dtype=torch.float64))
    monkeypatch.setattr('dynamicwam.dualrate.online.history_features',features)
    vae=SimpleNamespace(dtype=torch.float32,encode=lambda x:torch.zeros(1,48,2,24,20))
    batch=builder.source(frames,source_time=e,request_time=r,action_start=e,vae=vae)
    assert float(batch['source_time'])==e
    assert float(batch['source_request_time'])==r
    assert float(batch['source_available_time'])==e+.1
    assert float(batch['query_time'])==r and float(batch['completed_time'])==r
    assert float(batch['source_action_times'][0,0])==e
    assert float(batch['source_future_intervals'][0,-1,-1])==e+16*DT
    with pytest.raises(TimingError,match='unavailable'):
        builder.source(frames,source_time=e,request_time=e+.05,action_start=e,vae=vae)
    with pytest.raises(TimingError,match='actual grid-aligned exposure'):
        builder.source(frames,source_time=r,request_time=r,action_start=r,vae=vae)


def test_inference_placement_requires_frozen_model_and_preserves_precision():
    system=torch.nn.Module()
    for name in ('slow','world_readout','observation','fast'):
        setattr(system,name,torch.nn.Linear(2,2))
    system.register_buffer('motion_mean',torch.zeros(12))
    vae=SimpleNamespace(device=torch.device('cpu'),model=torch.nn.Linear(2,2),scale=[torch.ones(1)])
    system.slow.native=SimpleNamespace(compact_wan=SimpleNamespace(video_model=SimpleNamespace(vae=vae)))
    with pytest.raises(ValueError,match='eval, frozen'):
        place_inference_devices(system,fast_device='cpu',slow_device='cpu')
    system.eval().requires_grad_(False)
    system.slow.bfloat16()
    before={name:value.clone() for name,value in system.state_dict().items()}
    mapping=place_inference_devices(system,fast_device='cpu',slow_device='cpu')
    assert mapping==dict(fast='cpu',slow='cpu',readout='cpu',vae='cpu')
    assert all(torch.equal(value,before[name]) and value.dtype==before[name].dtype
               for name,value in system.state_dict().items())
    with pytest.raises(ValueError,match='distinct fast and slow'):
        place_inference_devices(system,fast_device='cuda:1',slow_device='cuda:1')


def test_overlap_requires_active_model_work_not_only_waiting_publication():
    events=[dict(event='activation',wall_time=10.),
        dict(event='slow_completed',version=1,model_started_wall_time=8.,gpu_done_wall_time=9.),
        dict(event='slow_completed',version=2,model_started_wall_time=10.,gpu_done_wall_time=11.),
        dict(event='fast_committed',query_id=0,model_started_wall_time=8.5,gpu_done_wall_time=8.8),
        dict(event='fast_committed',query_id=1,model_started_wall_time=10.5,gpu_done_wall_time=10.7),
        dict(event='fast_committed',query_id=2,model_started_wall_time=11.1,gpu_done_wall_time=11.3)]
    result=model_overlap_statistics(events)
    assert result['model_overlap_fast_queries']==1
    assert result['model_overlap_wall_seconds']==pytest.approx(.2)
    assert result['model_overlap_events'][0]['slow_version']==2
    assert result['model_overlap_events'][0]['query_id']==1
