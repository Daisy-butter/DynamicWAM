import torch
import pytest
from test_dualrate_native_v7 import inputs
from dynamicwam.dualrate.fast import FastConfig
from dynamicwam.dualrate.native_v9 import ResidualEventFast,draft
from dynamicwam.dualrate.fast_graph import eager_joint_sample,_flatten,_context

def fixture():
    m=ResidualEventFast(FastConfig(dim=24,heads=3,layers=2,future_tokens=12,future_entities=3,
        observation_dim=12,world_dim=12,test_fixture=True),[.1]*14).eval()
    kw=inputs();kw.update(draft_actions=torch.randn(1,16,14),draft_valid=torch.ones(1,16,dtype=torch.bool))
    return m,kw

def sample(m,kw):
    a=torch.randn(1,16,14);z=torch.randn(1,12,134);c=m.prepare_context(**kw)
    return eager_joint_sample(m,a,z,torch.zeros_like(a),torch.zeros(1,16,dtype=torch.bool),c,
        torch.ones_like(z,dtype=torch.bool),torch.ones(1,12,2,dtype=torch.bool),steps=10)

def test_zero_correction_preserves_A10_and_grippers():
    m,kw=fixture()
    torch.testing.assert_close(sample(m,kw)['actions'],kw['draft_actions'],atol=1e-6,rtol=0)

def test_bound_is_inside_every_denoising_step_and_receives_gradients():
    m,kw=fixture();m.action_out.weight.data.normal_(0,.03)
    kw['observation_tokens'].requires_grad_();kw['world_tokens'].requires_grad_()
    z=torch.randn(1,12,134,requires_grad=True)
    out=m(torch.randn(1,16,14),z,torch.tensor([.3]),**kw)
    assert (out['action_clean']-kw['draft_actions']).abs().max()<=.100001
    out['action_clean'].square().sum().backward()
    assert z.grad.abs().sum()>0
    assert kw['observation_tokens'].grad.abs().sum()>0 and kw['world_tokens'].grad.abs().sum()>0

def test_graph_buffers_copy_draft_and_independent_world_masks():
    m,kw=fixture();c=m.prepare_context(**kw);a=torch.randn(1,16,14);z=torch.randn(1,12,134)
    v=_flatten(a,z,a,torch.zeros(1,16,dtype=torch.bool),c,torch.ones_like(z,dtype=torch.bool),torch.ones(1,12,2,dtype=torch.bool))
    v['draft_actions']=v['draft_actions']+3
    private=_context(c,v)
    assert torch.equal(private.draft_actions,kw['draft_actions']+3)
    assert not private.slow_allowed[:,0,:,2:].any()
    assert not private.slow_allowed[:,1,:,1].any()
    kw['draft_actions'].add_(1)
    with pytest.raises(ValueError,match='modified'):m._check_cache(c)

def test_partial_slow_commitment_uses_next_command_and_real_physical_age():
    world=dict(reference_actions=torch.arange(16).float()[None,:,None].expand(1,16,14),
        native_source_index=torch.tensor([7]),source_time=torch.tensor([10.1],dtype=torch.float64))
    batch=dict(native_query_index=torch.tensor([8]),query_time=torch.tensor([11.2],dtype=torch.float64))
    a,valid=draft(world,batch,1.44)
    assert a[0,0,0]==1 and valid.sum()==15 and not valid[0,-1]
    batch['query_time']+=.5
    assert draft(world,batch,1.44)[1].sum()==15  # A10 is command-covered, M expiry is separate.
    batch['native_query_index']=torch.tensor([23])
    assert not draft(world,batch,1.44)[1].any()
