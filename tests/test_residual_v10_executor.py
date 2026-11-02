"""Numerical trajectory guarantees, not learned-policy or GPU acceptance."""
import numpy as np
import pytest
import torch
from dynamicwam.dualrate.c2_executor import ARMS, evaluate, project_chunk

DT=.18000000854954123
LIMITS=dict(velocity=2.5,acceleration=20.,jerk=400.)
BOUNDS=[[-10.,10.] if i not in (6,13) else [0.,1.] for i in range(14)]


def fixture(batch=3):
    a=torch.randn(batch,16,14)*2
    a[:,:2]=0.;a[:,:,[6,13]]=.5
    known=a.clone();mask=torch.arange(16)[None].expand(batch,-1)<2
    return a,known,mask,dict(mean=torch.zeros(14),scale=torch.ones(14),dt=DT,bounds=BOUNDS,limits=LIMITS)


def test_c2_and_continuous_derivative_bounds_on_adversarial_chunk():
    a,k,m,kw=fixture();p,valid=project_chunk(a,k,m,**kw)
    assert valid.all() and torch.equal(p[m],k[m])
    for sample in p.double().numpy():
        for i in range(2,16):
            q0,q1=sample[i-1,list(ARMS)],sample[i,list(ARMS)]
            v0=(q0-sample[i-2,list(ARMS)])/DT
            q,v,acc=evaluate(q0,q1,v0,DT,0.)
            np.testing.assert_allclose(q,q0,atol=1e-12)
            np.testing.assert_allclose(v,v0,atol=1e-12)
            np.testing.assert_allclose(acc,0.,atol=1e-12)
            q,v,acc=evaluate(q0,q1,v0,DT,DT)
            np.testing.assert_allclose(q,q1,atol=1e-12)
            np.testing.assert_allclose(v,(q1-q0)/DT,atol=1e-12)
            np.testing.assert_allclose(acc,0.,atol=1e-12)
            times=np.linspace(0,DT,101)
            samples=[evaluate(q0,q1,v0,DT,t) for t in times]
            vel=np.array([s[1] for s in samples]);acc=np.array([s[2] for s in samples])
            assert abs(vel).max()<=LIMITS['velocity']+1e-4
            assert abs(acc).max()<=LIMITS['acceleration']+1e-4
            assert abs(np.diff(acc,axis=0)/(times[1]-times[0])).max()<=LIMITS['jerk']+1e-3


def test_zero_correction_preserves_feasible_slow_plan_and_grippers():
    a,k,m,kw=fixture();base,_=project_chunk(a,k,m,**kw)
    p,ok=project_chunk(base,k,m,center=base,radius=torch.full((14,),.06),**kw)
    assert ok.all()
    torch.testing.assert_close(p,base,atol=1e-6,rtol=0)
    assert torch.equal(p[:,:,[6,13]],base[:,:,[6,13]])


def test_correction_trust_box_prefix_and_finite_backward():
    a,k,m,kw=fixture();base,_=project_chunk(a,k,m,**kw)
    delta=torch.randn_like(a,requires_grad=True)
    radius=torch.full((14,),.06)
    p,ok=project_chunk(base+delta.tanh()*radius,k,m,center=base,radius=radius,**kw)
    assert ((p-base).abs()<=radius+1e-6).all()
    assert torch.equal(p[m],k[m])
    p.square().sum().backward()
    assert torch.isfinite(delta.grad).all()


def test_reject_impossible_committed_state_and_noncontiguous_prefix():
    a,k,m,kw=fixture();k[:,1,0]=8.
    with pytest.raises(ValueError,match='no feasible'):project_chunk(a,k,m,**kw)
    m[:,4]=True
    with pytest.raises(ValueError,match='contiguous'):project_chunk(a,k,m,**kw)


def test_reject_out_of_interval_evaluation():
    with pytest.raises(ValueError,match='outside'):evaluate(np.zeros(12),np.ones(12),np.zeros(12),DT,DT+.01)
