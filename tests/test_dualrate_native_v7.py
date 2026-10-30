import torch
import pytest
from dynamicwam.dualrate.fast import FastConfig
from dynamicwam.dualrate.native_v7 import CompactEventFast,aligned_reference,overlapping_noise

def inputs():
    return dict(observation_tokens=torch.randn(1,4,12),observation_valid=torch.ones(1,4,dtype=torch.bool),
        world_tokens=torch.randn(1,3,12),world_valid=torch.ones(1,3,dtype=torch.bool),
        world_intervals=torch.tensor([[[9.,10.],[11.,18.],[19.,26.]]],dtype=torch.float64),
        world_types=torch.tensor([[0,1,1]]),query_time=torch.tensor([10.4],dtype=torch.float64),
        source_time=torch.tensor([10.],dtype=torch.float64),completed_time=torch.tensor([10.],dtype=torch.float64),
        native_source_index=torch.tensor([10]),native_query_index=torch.tensor([12]),native_available=torch.tensor([True]))

def model():
    return CompactEventFast(FastConfig(dim=24,heads=3,layers=2,future_tokens=12,future_entities=3,
        observation_dim=12,world_dim=12,test_fixture=True))

def test_command_mask_and_seconds_domain_independent():
    m=model();kw=inputs();c=m.prepare_context(**kw)
    # Action events 13..18 see first future group, 19..26 second, 27+ neither.
    assert c.slow_allowed[0,0,2] and not c.slow_allowed[0,0,3]
    assert c.slow_allowed[0,6,3] and not c.slow_allowed[0,6,2]
    assert not c.slow_allowed[0,14,2:].any()
    moved=dict(kw,query_time=kw['query_time']+100.,source_time=kw['source_time']+100.,completed_time=kw['completed_time']+100.)
    moved['world_intervals']=kw['world_intervals'].clone();moved['world_intervals'][:,0]+=100.
    other=m.prepare_context(**moved)
    torch.testing.assert_close(c.time_features,other.time_features,rtol=0,atol=1e-5)
    assert torch.equal(c.slow_allowed,other.slow_allowed)
    for x,y in zip(c.kv,other.kv):
        for a,b in zip(x,y):torch.testing.assert_close(a,b,rtol=0,atol=1e-5)

def test_prior_exact_join_tail_ttl_and_no_gripper_bias():
    kw=inputs();batch=dict(query_time=kw['query_time'],native_query_index=kw['native_query_index'],fast_known_mask=torch.zeros(1,16,dtype=torch.bool))
    world=dict(source_time=kw['source_time'],completed_time=kw['completed_time'],native_source_index=kw['native_source_index'],
        reference_actions=torch.arange(16).float()[None,:,None].expand(1,16,14),native_available=torch.tensor([True]))
    center,plan,valid,w=aligned_reference(batch,world)
    assert torch.equal(plan[0,:14,0],torch.arange(2,16).float()) and valid.sum()==14
    assert not center[:,14:].any() and not center[:,:,[6,13]].any()
    assert center.abs().max()<=.45
    for changed in [dict(world,native_available=torch.tensor([False])),dict(world,source_time=world['source_time']-3)]:
        assert not aligned_reference(batch,changed)[0].any()
    with pytest.raises(ValueError):aligned_reference(dict(batch,native_query_index=torch.tensor([9])),world)

def test_overlap_noise_corresponds_to_same_commands():
    a=torch.randn(2,16,14);current=torch.tensor([12,22]);previous=torch.tensor([10,20])
    p=overlapping_noise(a,current,previous)
    assert torch.equal(p[:,2:],a[:,:14])

def test_latest_world_and_z_actually_reach_single_action_decoder():
    m=model();kw=inputs();a=torch.randn(1,16,14);z=torch.randn(1,12,134,requires_grad=True)
    kw['observation_tokens'].requires_grad_();kw['world_tokens'].requires_grad_()
    out=m(a,z,torch.tensor([.4]),**kw);out['action_velocity'].square().sum().backward()
    for t in (z,kw['observation_tokens'],kw['world_tokens']):assert t.grad.abs().sum()>0
    assert not hasattr(m,'motor')

def test_cache_mutation_and_expired_world():
    m=model();kw=inputs();c=m.prepare_context(**kw)
    kw['native_query_index'].add_(1)
    with pytest.raises(ValueError,match='modified'):m._check_cache(c)
    kw=inputs();kw['query_time']+=5.
    c=m.prepare_context(**kw)
    assert c.slow_allowed[...,0].all() and not c.slow_allowed[...,1:].any()

def test_previous_plan_is_not_hard_commitment():
    m=model();kw=inputs();a=torch.randn(1,16,14);z=torch.randn(1,12,134);sigma=torch.tensor([.4])
    base=m(a,z,sigma,**kw)['action_velocity']
    plan=torch.randn_like(a)
    changed=m(a,z,sigma,**kw,previous_actions=plan,previous_valid=torch.ones(1,16,dtype=torch.bool))['action_velocity']
    assert not torch.allclose(base,changed)
    assert changed.abs().sum()>0
