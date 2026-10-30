import numpy as np
import torch
import pytest
from dynamicwam.dualrate.native_data import native_rows
from dynamicwam.dualrate.native_v5 import aligned_center

def test_terminal_windows_remain_and_no_beyond_last_target():
    t=np.arange(0,3.001,.06)
    rows=native_rows(t,.36,[0.,.36])
    assert (7,0) in rows  # 2.52 source, fewer than16 future actions
    assert all(g*.36+[0,.36][a]+.18<=t[-1]+1e-8 for g,a in rows)
    assert (0,0) in rows  # zero-prefix bootstrap must actually be trained

def test_prior_ordinal_shift_expiration_and_unrestricted_correction():
    ref=torch.arange(16).float()[None,:,None].expand(1,16,14).clone()
    b=dict(fast_known_actions=torch.zeros_like(ref),fast_known_mask=torch.zeros(1,16,dtype=torch.bool),
           native_query_index=torch.tensor([2]),query_time=torch.tensor([.7],dtype=torch.float64))
    w=dict(reference_actions=ref,native_source_index=torch.tensor([0]),
           native_available=torch.tensor([True]),source_time=torch.tensor([0.],dtype=torch.float64))
    center=aligned_center(b,w)
    assert center[0,0,0]>0 and center[0,0,0]<.3
    assert center[0,14:].count_nonzero()==0
    assert center[..., [6,13]].count_nonzero()==0
    b['fast_known_mask'][0,0]=True
    assert aligned_center(b,w)[0,0].count_nonzero()==0
    w['native_available'].zero_()
    assert aligned_center(b,w).count_nonzero()==0

def test_flow_prior_correct_endpoint_has_no_hard_output_constraint():
    gt=torch.randn(1,16,14);eps=torch.randn_like(gt);center=torch.randn_like(gt)*.15
    sigma=.7;x=(1-sigma)*gt+sigma*(eps+center)
    velocity=eps+center-gt
    torch.testing.assert_close(x-sigma*velocity,gt)


def test_native_clock_replanning_and_cache_reuse(monkeypatch):
    from types import SimpleNamespace as S
    from dynamicwam.dualrate.native_deployment import NativeDualRatePolicy,ordinal_action_noise
    from dynamicwam.dualrate.actuators import ActuatorProjection
    class Builder:
        device=slow_device=torch.device('cpu')
        manifest={'time_contract':'native_sync_ordinal_v5'}
        mean=torch.zeros(14);scale=torch.ones(14)
        def source(self,frames,*,source_time,**kw):
            return dict(source_initial_state=torch.zeros(1,14),source_time=torch.tensor([source_time],dtype=torch.float64))
        def current(self,frames,*,query_time,**kw):
            return dict(state=torch.zeros(1,14),query_time=torch.tensor([query_time],dtype=torch.float64))
    class Model:
        native_v5=True
        slow=S(native=S(compact_wan=S(video_model=S(vae=None))))
        def eval(self):return self
        def predict_reference(self,batch,**kw):return dict(source_time=batch['source_time'])
        def world_readout(self,r):return r
        def sample_fast(self,*args,**kw):return {'actions':torch.zeros(1,16,14)}
    bounds=ActuatorProjection([[-10,10]]*6+[[0,1]]+[[-10,10]]*6+[[0,1]],provenance='unit fixture')
    monkeypatch.setattr(ActuatorProjection,'from_robot',classmethod(lambda cls,robot:bounds))
    class Env:
        robot=None;take_action_cnt=0;physical=0.
        def __init__(self):self._scene_step_clock=S(snapshot=lambda:S(time_seconds=self.physical))
        def take_action(self,a,action_type):
            assert action_type=='qpos'
            self.physical += [.08,.196,.052,.124,.216,.048][self.take_action_cnt]
            self.take_action_cnt+=1
    env=Env();policy=NativeDualRatePolicy(Model(),Builder())
    obs={'observation':{c:{'rgb':np.zeros((4,4,3),np.uint8)} for c in ['head_camera','left_camera','right_camera']},
         'joint_action':{'vector':np.zeros(14,np.float32)}}
    # Use actual camera names from production schema.
    from dynamicwam.dualrate.producer import CAMERAS
    obs['observation']={c:{'rgb':np.zeros((4,4,3),np.uint8)} for c in CAMERAS}
    for _ in range(6):policy.step(env,obs)
    q=[x for x in policy.records if 'query_physical_time' in x]
    assert [x['command_index'] for x in q]==[0,2,4]
    assert [x['source_command_index'] for x in q]==[0,0,4]
    assert [x['query_physical_time'] for x in q]==pytest.approx([0,.276,.452])
    a=ordinal_action_noise(0,16,14,4,torch.zeros(1))
    b=ordinal_action_noise(2,16,14,4,torch.zeros(1))
    torch.testing.assert_close(a[:,2:],b[:,:14],rtol=0,atol=0)


def test_native_flow_uses_original_command_stride_and_actual_times():
    from types import SimpleNamespace
    from dynamicwam.dualrate.online import history_endpoints
    frames=[SimpleNamespace(exposure=i*.113+(i%3)*.007,frame_id=i) for i in range(25)]
    selected,valid=history_endpoints(frames,native_ordinal=True)
    assert all(valid)
    assert [x.frame_id for x in selected]==[4,8,12,16,20,24]
    assert selected[-1].exposure-selected[-2].exposure==pytest.approx(.438)
    assert selected[-1].exposure-selected[-2].exposure!=pytest.approx(.72)



