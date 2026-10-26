"""CPU-only contracts. Real CUDA parity runs under the deployment GPU guard.

No CUDA availability/device probe is made by this module or its tests.
"""
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from dynamicwam.dualrate.fast import FastConfig, FastJointWAM
from dynamicwam.dualrate.fast_graph import (
    FastGraphSampler, eager_joint_sample, _flatten, _context, _layout,
)


class CoreFixture:
    def __init__(self):self.seen=[]
    def forward_cached(self,a,z,sigma,context,*,known_actions,action_known_mask):
        self.seen.append(a.clone())
        return dict(action_velocity=a*.1+z.mean()+sigma[:,None,None],
                    future_velocity=z*.2+a.mean(),contact_logits=z[...,:2])


def test_complete_euler_sampler_retains_prefix_each_step_and_final_contact_pass():
    fast=CoreFixture();a=torch.ones(1,4,2);z=torch.ones(1,4,6)
    known=torch.full_like(a,3.);mask=torch.tensor([[True,True,False,False]])
    a=torch.where(mask[...,None],known,a)
    fields=torch.ones_like(z,dtype=torch.bool);fields[...,0]=False
    contacts=torch.ones(1,4,2,dtype=torch.bool);contacts[:,2:]=False
    context=SimpleNamespace(future_valid=torch.ones(1,4,dtype=torch.bool))
    original=(a.clone(),z.clone(),known.clone(),fields.clone())
    output=eager_joint_sample(fast,a,z,known,mask,context,fields,contacts,steps=10)
    assert len(fast.seen)==11
    assert all(torch.equal(value[mask],known[mask]) for value in fast.seen)
    assert torch.equal(output['actions'][mask],known[mask])
    assert output['future'][...,0].eq(0).all()
    assert output['contact_probability'][:,2:].eq(0).all()
    for value,before in zip((a,z,known,fields),original):assert torch.equal(value,before)


def fixture(world_count=2):
    cfg=FastConfig(dim=8,layers=1,heads=2,action_dim=2,chunk_size=4,
        future_dim=6,future_tokens=4,observation_dim=8,world_dim=8,mlp_ratio=2,test_fixture=True)
    fast=FastJointWAM(cfg).eval()
    conditions=dict(observation_tokens=torch.ones(1,2,8),observation_valid=torch.ones(1,2,dtype=torch.bool),
        world_tokens=torch.ones(1,world_count,8),world_valid=torch.ones(1,world_count,dtype=torch.bool),
        world_intervals=torch.tensor([[[0.,2.]]]*world_count,dtype=torch.float64).reshape(1,world_count,2),
        target_times=torch.tensor([[.3,.4,.5,.6,.3,.4,.6,1.]],dtype=torch.float64),
        query_time=torch.tensor([.2],dtype=torch.float64),source_time=torch.tensor([0.],dtype=torch.float64),
        completed_time=torch.tensor([.1],dtype=torch.float64))
    with torch.no_grad():context=fast.prepare_context(**conditions)
    values=_flatten(torch.ones(1,4,2),torch.ones(1,4,6),torch.zeros(1,4,2),
        torch.zeros(1,4,dtype=torch.bool),context,torch.ones(1,4,6,dtype=torch.bool),
        torch.ones(1,4,2,dtype=torch.bool))
    return fast,context,values,conditions


def test_private_graph_storage_covers_every_context_tensor_and_does_not_alias_query():
    fast,context,values,_=fixture()
    copies={name:value.clone() for name,value in values.items()}
    private=_context(context,copies)
    assert private.inputs=={} and private.versions=={}
    assert private.parameter_versions==context.parameter_versions
    assert len(private.kv)==len(context.kv)
    for original,group in zip(context.kv,private.kv):
        for a,b in zip(original,group):assert a.data_ptr()!=b.data_ptr() and torch.equal(a,b)
    copies['time_features'].add_(1)
    assert not torch.equal(private.time_features,context.time_features)
    with torch.no_grad():fast._check_cache(context)


def test_external_cache_mutation_and_parameter_mutation_are_still_rejected():
    fast,context,_,_=fixture();engine=FastGraphSampler(fast)
    with torch.no_grad():
        engine._validate(context)
        context.inputs['world_tokens'].add_(1)
        with pytest.raises(ValueError,match='modified in place'):engine._validate(context)
    fast,context,_,_=fixture();engine=FastGraphSampler(fast)
    with torch.no_grad():
        next(fast.parameters()).add_(1)
        with pytest.raises(ValueError,match='model changed'):engine._validate(context)


def test_signature_distinguishes_solver_count_real_empty_shapes_and_dtype():
    _,_,values,_=fixture();_,_,empty,_=fixture(world_count=1)
    assert _layout(values,10)!=_layout(values,8)
    assert _layout(values,10)!=_layout(empty,10)
    changed=dict(values,a=values['a'].double())
    assert _layout(values,10)!=_layout(changed,10)


def test_cpu_cannot_claim_cuda_parity_and_seal_requires_prepared_shapes():
    fast,context,values,_=fixture();engine=FastGraphSampler(fast)
    with pytest.raises(ValueError,match='CUDA device'):
        engine.sample(values['a'],values['z'],values['known'],values['mask'],context,
                      values['fields'],values['contacts'],steps=10)
    assert engine.receipt()['passed'] is False and engine.receipt()['shape_count']==0
    with pytest.raises(RuntimeError,match='prepared graph shapes'):engine.seal(2)


def test_sealed_sampler_rejects_unprepared_shape_before_any_cuda_operation():
    fast,context,values,_=fixture();engine=FastGraphSampler(fast)
    engine.sealed=True  # explicit CPU fixture for the sealed branch, not acceptance
    with pytest.raises(RuntimeError,match='after activation'):
        engine.sample(values['a'],values['z'],values['known'],values['mask'],context,
                      values['fields'],values['contacts'],steps=10)


def test_graph_sampler_is_not_a_model_child_and_changes_no_state_dict():
    fast,_,_,_=fixture();holder=nn.Module();holder.fast=fast
    before=tuple(holder.state_dict())
    holder._fast_graph_sampler=FastGraphSampler(fast)
    assert tuple(holder.state_dict())==before
    assert '_fast_graph_sampler' not in holder._modules


def test_tensor_core_rejects_invalid_step_count():
    with pytest.raises(ValueError,match='positive integer'):
        eager_joint_sample(None,None,None,None,None,None,None,None,steps=0)
