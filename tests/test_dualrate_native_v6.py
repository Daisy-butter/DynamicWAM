import numpy as np
import torch
import pytest
from dynamicwam.dualrate.native_v6 import align_plan,validate_events,current_source

def fixture():
    return dict(source_time=torch.tensor([1.],dtype=torch.float64),completed_time=torch.tensor([1.],dtype=torch.float64),
        query_time=torch.tensor([1.37],dtype=torch.float64),native_source_index=torch.tensor([10]),native_query_index=torch.tensor([12]),
        rgb_times=torch.full((1,3),1.37,dtype=torch.float64),state_time=torch.tensor([1.37],dtype=torch.float64),
        fast_known_mask=torch.zeros(1,16,dtype=torch.bool),source_known_mask=torch.zeros(1,16,dtype=torch.bool))

def test_event_alignment_does_not_convert_elapsed_seconds_into_rows():
    p=torch.arange(16).float()[None,:,None].expand(1,16,14)
    a,m=align_plan(p,torch.tensor([10]),torch.tensor([12]))
    assert torch.equal(a[0,:14,0],torch.arange(2,16).float()) and not m[0,14:].any()
    with pytest.raises(ValueError):align_plan(p,torch.tensor([12]),torch.tensor([10]))
    with pytest.raises(ValueError):align_plan(p,torch.tensor([10.]),torch.tensor([12.]))

def test_actual_clock_and_events_are_independent_and_causal():
    b=fixture();validate_events(b)
    b['query_time']+=100.;b['rgb_times']+=100.;b['state_time']+=100.
    validate_events(b) # same command offset after an arbitrarily long TOPP action
    b['rgb_times'][0,0]-=.06
    with pytest.raises(ValueError,match='captured'):validate_events(b)

def test_native_zero_prefix_and_teacher_origin():
    b=fixture();b['teacher_initial_state']=torch.randn(1,14)
    q=current_source(b)
    assert torch.equal(q['source_time'],b['query_time'])
    assert torch.equal(q['native_source_index'],b['native_query_index'])
    assert q['source_initial_state'] is b['teacher_initial_state']
    b['fast_known_mask'][0,0]=True
    with pytest.raises(ValueError,match='zero'):validate_events(b)

def test_frozen_teacher_phase_config():
    import json
    from pathlib import Path
    from dynamicwam.dualrate.train_distributed import validate_phase_config
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/dualrate_native_v6_core.json').read_text())
    c['run_kind']='distributed_training_smoke'
    validate_phase_config(c)
    with pytest.raises(ValueError,match='frozen Full teacher'):
        validate_phase_config(dict(c,slow_lr=1e-5))
    c.pop('native_v6')
    with pytest.raises(ValueError,match='positive slow LR'):
        validate_phase_config(c)


def test_native_clock_replanning_and_cache_reuse(monkeypatch):
    from types import SimpleNamespace as S
    from dynamicwam.dualrate.native_v6_deployment import NativeDualRatePolicy,ordinal_action_noise
    from dynamicwam.dualrate.actuators import ActuatorProjection
    class Builder:
        device=slow_device=torch.device('cpu')
        manifest={'time_contract':'native_event_single_v6'}
        mean=torch.zeros(14);scale=torch.ones(14)
        def source(self,frames,*,source_time,**kw):
            return dict(source_initial_state=torch.zeros(1,14),source_time=torch.tensor([source_time],dtype=torch.float64))
        def current(self,frames,*,query_time,**kw):
            return dict(state=torch.zeros(1,14),query_time=torch.tensor([query_time],dtype=torch.float64))
    class Model:
        native_v6=True
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


