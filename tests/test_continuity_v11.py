import pytest
import torch
from dynamicwam.dualrate.continuity_v11 import align,shape_correction,defaults,augment_context


def test_alignment_uses_both_physical_endpoints():
    values=torch.arange(42.).reshape(1,3,14)
    times=torch.tensor([[[0.,.18],[.18,.36],[.36,.54]]],dtype=torch.float64)
    out,mask=align(values,times,times[:,1:])
    assert torch.equal(out,values[:,1:]) and mask.all()
    shifted=times[:,1:].clone();shifted[:,:,1]+=.001
    out,mask=align(values,times,shifted)
    assert not mask.any() and not out.any()


def test_duplicate_reference_rejected():
    t=torch.tensor([[[0.,.18],[0.,.18]]],dtype=torch.float64)
    with pytest.raises(ValueError,match='ambiguous'):
        align(torch.zeros(1,2,14),t,t)


def test_empty_real_execution_history_is_unavailable():
    t=torch.tensor([[[0.,.18]]],dtype=torch.float64)
    x,valid=align(torch.zeros(1,0,14),torch.zeros(1,0,2,dtype=torch.float64),t)
    assert x.shape==(1,1,14) and not x.any() and not valid.any()


def test_real_replay_teacher_covers_actual_prefix(tmp_path):
    from types import SimpleNamespace
    from dynamicwam.dualrate.feedback_replay import FeedbackRecorder
    t=torch.arange(16,dtype=torch.float64)*.18
    spans=torch.stack((t,t+.18),-1)[None]
    actions=torch.randn(1,16,14)
    source=dict(source_time=torch.tensor([0.],dtype=torch.float64),source_action_intervals=spans,
                source_condition_frames=torch.zeros(1,3,5,4,4))
    known=torch.arange(16)[None]<3
    b=dict(query_time=torch.tensor([.2],dtype=torch.float64),measured_state_valid=torch.tensor([True]),
           action_intervals=spans,fast_known_actions=torch.where(known[...,None],actions,0.),fast_known_mask=known,
           teacher_action_intervals=spans,executed_actions=torch.zeros(1,0,14),
           executed_action_intervals=torch.zeros(1,0,2,dtype=torch.float64),executed_action_valid=torch.zeros(1,0,dtype=torch.bool))
    ref=SimpleNamespace(completed_time=.1,payload={'reference_actions':actions})
    recorder=FeedbackRecorder(tmp_path/'ok',dict(split='train',seed=1))
    recorder.capture(1,(source,torch.zeros(16,dtype=torch.uint8)),b,dict(actions=actions,draft_valid=torch.ones(1,16,dtype=torch.bool)),ref)
    recorder.close();r=torch.load(tmp_path/'ok/query_00001.pt',weights_only=True)
    assert r['teacher_known_mask'].sum()==3 and not r['executed_action_valid'].any()
    assert torch.equal(r['teacher_known_actions'][:3],actions[0,:3])
    with pytest.raises(ValueError,match='evaluation scenes'):FeedbackRecorder(tmp_path/'bad',dict(split='train',seed=100000))


def test_bounded_temporal_basis_zero_and_grippers():
    valid=torch.ones(2,16,dtype=torch.bool)
    assert not shape_correction(torch.zeros(2,16,14),valid).any()
    x=torch.randn(2,16,14).tanh().requires_grad_()
    y=shape_correction(x,valid)
    assert y.abs().max()<=1 and not y[:,:,[6,13]].any()
    y.sum().backward();assert torch.isfinite(x.grad).all()


def test_invalid_suffix_cannot_influence_executable_rows():
    valid=torch.arange(16)[None]<7
    x=torch.randn(1,16,14).tanh();y=x.clone();y[:,7:]=1000
    assert torch.equal(shape_correction(x,valid)[:,:7],shape_correction(y,valid)[:,:7])


def test_real_sensor_missing_is_invalid_not_drive_target():
    b=defaults({'fast_known_actions':torch.ones(2,16,14)*5})
    assert not b['measured_state_valid'].any() and not b['measured_state'].any()
    assert not b['previous_valid'].any()


def test_actual_feedback_changes_context_without_replacing_native_state():
    from types import SimpleNamespace
    system=SimpleNamespace(fast=SimpleNamespace(sensor_projection=torch.nn.Linear(37,32),previous_projection=torch.nn.Linear(14,32)))
    b=defaults({'fast_known_actions':torch.zeros(1,16,14),'state':torch.ones(1,14)})
    b['measured_state_valid'][:]=True
    cond={'observation_tokens':torch.zeros(1,4,32),'observation_valid':torch.ones(1,4,dtype=torch.bool)}
    a=augment_context(system,b,cond)
    b['measured_state'][:,12]=.1
    c=augment_context(system,b,cond)
    assert not torch.equal(a['observation_tokens'],c['observation_tokens'])
    assert torch.equal(b['state'],torch.ones(1,14))


def test_real_model_interface_zero_prefix_and_teacher_gradient():
    from test_residual_v10 import fixture,generator
    from dynamicwam.dualrate.continuity_v11 import configure,CONTRACT
    s,b,w=fixture()
    configure(s,dict(contract=CONTRACT,reference_hold_steps=4,seam_weight=.1,revision_weight=.1,out_of_radius_weight=.2,revision_radius_fraction=1/3))
    b['feedback_noise_seed']=torch.tensor([17]);b['action_grid_origin']=b['source_time'].clone()
    s.residual_v10['executor']='phase_c2'
    b['teacher_motion']=b['teacher_motion']+10.
    s.eval()
    result=s.sample_fast(b,w,steps=1)
    torch.testing.assert_close(result['actions'],result['baseline_actions'],atol=1e-6,rtol=0)
    assert torch.equal(result['actions'][b['fast_known_mask']],b['fast_known_actions'][b['fast_known_mask']])
    loss=s.training_loss(b,generator=generator())
    assert torch.isfinite(loss['loss'])
    loss['loss'].backward()
    assert s.fast.action_out.weight.grad is not None
    assert torch.isfinite(s.fast.action_out.weight.grad).all()
    assert s.fast.action_out.weight.grad.abs().max()>0
    assert all(p.grad is None for p in s.slow.parameters())


def test_reference_lease_stays_atomic_but_never_extrapolates():
    from types import SimpleNamespace
    from dynamicwam.dualrate.reference_lease import ReferenceLease
    def ref(version,source):
        t=source+torch.arange(16,dtype=torch.float64)*.18
        return SimpleNamespace(version=version,source_time=source,completed_time=source+.3,expires_at=source+2,
            payload={'reference_actions':torch.zeros(1,16,14),'reference_action_intervals':torch.stack((t,t+.18),-1)[None],
                     'world_valid':torch.ones(1,16,dtype=torch.bool)})
    lease=ReferenceLease(4,.18);a=ref(1,0.);b=ref(2,.36)
    got,dev,_=lease.select(a,(1,'a'),now=.36,first=.72)
    assert got is a
    got,dev,event=lease.select(b,(2,'b'),now=.72,first=1.08)
    assert got is a and dev==(1,'a') and event['reason']=='minimum_dwell'
    got,dev,_=lease.select(b,(2,'b'),now=1.08,first=1.44)
    assert got is b and dev==(2,'b')
    with pytest.raises(ValueError,match='no causal'):
        lease.select(b,(2,'b'),now=2.5,first=2.88)


def test_same_reference_innovation_is_bounded_and_new_reference_resets():
    from types import SimpleNamespace
    from dynamicwam.dualrate.continuity_v11 import bounded_residual
    s=SimpleNamespace(radius_raw_v10=torch.ones(14)*.06,action_scale_v10=torch.ones(14),
                      continuity_v11={'revision_radius_fraction':1/3})
    base=torch.zeros(1,16,14)
    b=defaults({'fast_known_actions':base.clone()})
    b['previous_actions'][:]=.04;b['previous_actions'][:,:,[6,13]]=0
    b['previous_valid'][:]=True;b['previous_reference_same'][:]=True
    d=bounded_residual(s,b,base,torch.full_like(base,-1))
    assert (d-b['previous_actions']).abs().max()<=.0200001
    assert d.abs().max()<=.06
    torch.testing.assert_close(bounded_residual(s,b,base,base),b['previous_actions'])
    b['previous_reference_same'][:]=False
    assert not bounded_residual(s,b,base,base).any()


def test_noise_matches_physical_interval_across_query_and_source_change():
    from types import SimpleNamespace
    from dynamicwam.dualrate.continuity_noise import prediction_noise
    s=SimpleNamespace(residual_v10={'action_dt':.18},future_latent_shape=(2,2,2,2),
                      _noise=lambda shape,like,g:torch.randn(shape,dtype=like.dtype,generator=g))
    starts=10.+torch.arange(16,dtype=torch.float64)*.18
    a=torch.stack((starts,starts+.18),-1)[None]
    b=dict(feedback_noise_seed=torch.tensor([19]),action_grid_origin=torch.tensor([10.],dtype=torch.float64),
           source_initial_state=torch.zeros(1,14),source_condition_latent=torch.zeros(1,2,2,2,2))
    v,n=prediction_noise(s,b,a);v2,n2=prediction_noise(s,b,a+.36)
    assert torch.equal(v,v2) and torch.equal(n[:,2:],n2[:,:-2])
    with pytest.raises(ValueError,match='physical action grid'):prediction_noise(s,b,a+.03)


def test_sensor_units_and_age_not_drive_truth():
    from dynamicwam.dualrate.feedback_observation import measured_features
    import numpy as np
    x,valid=measured_features(np.ones(12)*.3,np.ones(12)*.4,np.ones(14)*.1,
                              np.zeros(14),np.ones(14)*2,.18)
    assert valid and torch.allclose(x[12:24],torch.ones(12)*.1)
    assert torch.allclose(x[24:36],torch.ones(12)*.2)
    with pytest.raises(ValueError):measured_features(np.ones(12),None,np.zeros(14),np.zeros(14),np.ones(14),.1)
