"""Exercise online source pixels at the actual producer/native-VAE boundary.

The recording VAE isolates preprocessing; this is not an actual VAE GPU test.
No CUDA device/availability probe occurs in these CPU regression tests.
"""
import numpy as np
import pytest
import torch

from dynamicwam.dualrate import online, producer
from dynamicwam.dualrate.online import OnlineBatchBuilder, OnlineFrame
from dynamicwam.dualrate.timing import TimingError


class RecordingVAE:
    def __init__(self,dtype):
        self.dtype=dtype;self.inputs=[]

    def encode(self,value):
        self.inputs.append(value.detach().clone())
        b,_,t,h,w=value.shape
        return value.new_zeros(b,48,(t-1)//4+1,h//16,w//16)


def setup_source(monkeypatch):
    step=float(np.float32(.004));dt=45*step;e=2*dt
    def pixels(h,w,shift):
        return ((np.arange(h*w*3,dtype=np.uint32).reshape(h,w,3)+shift)%256).astype(np.uint8)
    images=(pixels(96,128,0),pixels(48,64,67),pixels(48,64,179))
    flows=np.stack([pixels(64,64,i*31) for i in range(4)])
    assert np.array_equal(np.unique(flows),np.arange(256,dtype=np.uint8))
    def history(*args,**kwargs):
        return dict(flow_rgb=flows.copy(),motion=torch.zeros(4,12),
            motion_valid=torch.ones(4,dtype=torch.bool),motion_acceleration_valid=torch.ones(4,dtype=torch.bool),
            motion_times=torch.tensor([[e-(4-i)*4*dt,e-(3-i)*4*dt] for i in range(4)],dtype=torch.float64),
            spatial_flow=torch.zeros(16,5),spatial_flow_valid=torch.zeros(16,dtype=torch.bool),
            spatial_flow_times=torch.tensor([e-4*dt,e],dtype=torch.float64))
    monkeypatch.setattr(online,'history_features',history)
    monkeypatch.setattr(producer,'_rgb',lambda _handle,_frame,camera:images[producer.CAMERAS.index(camera)])
    manifest=dict(time_contract='exposure_anchor_wall_release_v4',action_interval_seconds=dt,
        flow_contract={},active_entity_roles=['tracked_dynamic_target','left_gripper','right_gripper'],
        condition_rgb_max_age_seconds=.24,rgb_size=[64,64],rgb_view_sizes=[[64,64],[32,32],[32,32]])
    language=dict(context=torch.zeros(512,4096),mask=torch.ones(512,dtype=torch.bool))
    builder=OnlineBatchBuilder(manifest,action_mean=np.linspace(-1,1,14,dtype=np.float32),
        action_scale=np.linspace(.5,2,14,dtype=np.float32),language=language)
    frame=OnlineFrame(images,np.linspace(-.2,.5,14,dtype=np.float32),e,e+.1,0)
    return builder,(frame,),flows,step


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
def test_online_source_vae_input_matches_fp32_producer_normalization(monkeypatch,dtype):
    import cv2
    builder,frames,flows,_=setup_source(monkeypatch)
    e=frames[-1].exposure
    online_vae=RecordingVAE(dtype)
    builder.source(frames,source_time=e,request_time=e+.12,action_start=e,vae=online_vae)
    # Use the dataset's composite and exact pre-materialization pixel path.
    raw=np.stack([cv2.resize(image,(320,384)) for image in flows]+
                 [producer._composite(None,0,(384,320))])
    condition=torch.from_numpy(raw).permute(3,0,1,2)[None].float()/127.5-1
    future_raw=np.stack([producer._composite(None,0,(192,160))]*9)
    future=torch.from_numpy(future_raw).permute(3,0,1,2)[None].float()/127.5-1
    training_vae=RecordingVAE(dtype)
    producer.materialize_native_latents(dict(source_condition_frames=condition,source_future_frames=future),training_vae)
    assert online_vae.inputs[0].dtype==dtype
    torch.testing.assert_close(online_vae.inputs[0],training_vae.inputs[0],rtol=0,atol=0)
    if dtype!=torch.float32:
        # This test would fail before the fix, not merely verify a shared helper.
        previous=torch.from_numpy(raw).permute(3,0,1,2)[None].to(dtype)/127.5-1
        assert not torch.equal(previous,training_vae.inputs[0])


@pytest.mark.parametrize('route',['current','source'])
def test_nominal_camera_age_accepts_sixty_real_float32_steps_only(monkeypatch,route):
    builder,frames,_,step=setup_source(monkeypatch)
    e=frames[-1].exposure
    assert 60*step>.24+1e-8 and 61*step>.24+producer.NUMERIC_TIME_EPSILON
    def query(age):
        if route=='source':
            return builder.source(frames,source_time=e,request_time=e+age,action_start=e,
                                  vae=RecordingVAE(torch.float32))
        return builder.current(frames,query_time=e+age,action_start=e+builder.action_dt)
    query(60*step)
    with pytest.raises(TimingError,match='too old'):
        query(61*step)
