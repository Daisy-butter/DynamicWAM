import numpy as np
import pytest
import torch
from dynamicwam.dualrate.reference_executor import evaluate,bound_endpoints,measured_reference_discrepancy
from dynamicwam.dualrate.residual_v10 import configure,predict
from test_residual_v10 import fixture,generator


def test_phase_preserves_joint_path_endpoints_and_c2_across_reversals():
    q=np.array([.2,-.4,1.8]);targets=[np.array([2.,1.,-.5]),np.array([-.4,0.,1.5])]
    for target in targets:
        for u in np.linspace(0,1,101):
            p,v,a=evaluate(q,target,.18,u*.18)
            ratios=(p-q)/(target-q)
            assert np.ptp(ratios)<1e-12 and -.0000001<=ratios[0]<=1.0000001
            if u in (0,1):
                np.testing.assert_allclose(p,q if u==0 else target,atol=1e-12)
                np.testing.assert_allclose(v,0,atol=1e-12);np.testing.assert_allclose(a,0,atol=1e-12)
        q=target


def test_measured_reference_uses_actual_exposure_and_completed_boundary():
    ref=torch.arange(3.).reshape(1,3,1).expand(1,3,14).clone()
    world=dict(reference_actions=ref,reference_action_intervals=torch.tensor([[[0.,.18],[.18,.36],[.36,.54]]],dtype=torch.float64))
    b=dict(state=torch.ones(1,14)*.7,state_time=torch.tensor([.27],dtype=torch.float64))
    token,valid=measured_reference_discrepancy(b,world)
    assert valid.all();assert token[0,0]==.5;assert token[0,6]==1.
    torch.testing.assert_close(token[:,14:28],b['state']-token[:,:14])
    b['state_time']=torch.tensor([.36],dtype=torch.float64)
    token,valid=measured_reference_discrepancy(b,world);assert valid.all();assert token[0,0]==1.
    for t in (.09,.60):
        b['state_time']=torch.tensor([t],dtype=torch.float64)
        token,valid=measured_reference_discrepancy(b,world);assert not valid.any();assert not token.any()


def phase_fixture():
    s,b,w=fixture()
    settings=dict(s.residual_v10,executor='phase_c2',tracking_token=True,reaction_weight=2.)
    configure(s,settings,[0.]*14,[1.]*14)
    return s,b,w


def test_phase_zero_residual_preserves_raw_full_and_committed_actions():
    s,b,w=phase_fixture();s.eval()
    a,base,valid,_,_,_=predict(s,b,w)
    match=(b['action_intervals'][:,:,None]-w['reference_action_intervals'][:,None]).abs().amax(-1)<1e-7
    ref=w['reference_actions'].gather(1,match.long().argmax(-1)[...,None].expand(-1,-1,14))
    lo=ref.new_tensor(s.residual_v10['bounds'])[:,0];hi=ref.new_tensor(s.residual_v10['bounds'])[:,1]
    native=torch.minimum(torch.maximum(ref,lo),hi)
    m=valid&~b['fast_known_mask']
    torch.testing.assert_close(a[m],native[m],atol=1e-6,rtol=0)
    torch.testing.assert_close(a,base,atol=1e-6,rtol=0)
    assert torch.equal(a[b['fast_known_mask']],b['fast_known_actions'][b['fast_known_mask']])


def test_phase_trust_box_and_clean_residual_learning():
    s,b,w=phase_fixture();result=s.training_loss(b,generator=generator());result['loss'].backward()
    assert torch.isfinite(result['loss']);assert s.fast.action_out.weight.grad.abs().max()>0
    with torch.no_grad():s.fast.action_out.bias.fill_(100.)
    a,base,valid,_,_,_=predict(s,b,w)
    assert ((a-base).abs()<=s.radius_raw_v10+1e-6).all()
    assert torch.equal(a[b['fast_known_mask']],b['fast_known_actions'][b['fast_known_mask']])
