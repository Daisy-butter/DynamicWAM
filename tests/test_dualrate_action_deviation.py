"""Vectorized deviation must preserve the previous interval/gradient contract."""
from types import SimpleNamespace

import pytest
import torch

from dynamicwam.dualrate.system import DualRateSystem


def legacy(system,batch,world):
    tokens=world['world_tokens'];b,n=tokens.shape[:2];a=system.fast.config.action_dim
    delta=tokens.new_zeros(b,n,a);valid=torch.zeros(b,n,dtype=torch.bool)
    if world['reference_actions'].shape[1]==0:return delta,valid
    needed={'executed_actions','executed_action_intervals','executed_action_valid','action_intervals'}
    if not needed<=batch.keys():
        if system.test_fixture:return delta,valid
        raise ValueError('physical executed trace and current action intervals required')
    acts=batch['executed_actions'];times=batch['executed_action_intervals'];known=batch['executed_action_valid']
    if (times.dtype!=torch.float64 or times.shape!=(*acts.shape[:2],2) or
            known.dtype!=torch.bool or known.shape!=acts.shape[:2] or
            (known & (times[...,1]>batch['query_time'][:,None]+1e-8)).any()):
        raise ValueError('execution trace contains unavailable future actions')
    acts=torch.cat((acts,batch['fast_known_actions']),1)
    times=torch.cat((times,batch['action_intervals']),1)
    known=torch.cat((known,batch['fast_known_mask']),1)
    for i in range(b):
        slots=(world['world_types'][i]==2).nonzero().flatten()
        ref=world['reference_actions'][i];ri=world['reference_action_intervals'][i]
        if slots.numel()!=ref.shape[0]:raise ValueError('soft action tokens and reference plan differ')
        for j,slot in enumerate(slots):
            match=known[i] & ((times[i]-ri[j]).abs().amax(-1)<1e-7)
            if match.any():
                values=acts[i,match]
                if not torch.allclose(values,values[:1].expand_as(values),atol=1e-5,rtol=0):
                    raise ValueError('execution and commitment disagree at a physical interval')
                delta[i,slot]=values[0].to(ref.dtype)-ref[j];valid[i,slot]=True
    return delta,valid


def fixture(seed=0,*,dtype=torch.float32,token_dtype=torch.float32):
    rng=torch.Generator().manual_seed(seed);b,h,a=3,16,3
    commands=torch.randn(b,20,a,generator=rng,dtype=dtype)
    start=torch.arange(20,dtype=torch.float64)*.18
    intervals=torch.stack((start,start+.18),-1)[None].expand(b,-1,-1).clone()
    types=torch.zeros(b,23,dtype=torch.long)
    for row in types:row[torch.randperm(23,generator=rng)[:h]]=2
    batch=dict(executed_actions=commands[:,:8].clone(),executed_action_intervals=intervals[:,:8].clone(),
        executed_action_valid=torch.rand(b,8,generator=rng)>.25,query_time=torch.full((b,),1.44,dtype=torch.float64),
        fast_known_actions=commands[:,4:].clone(),action_intervals=intervals[:,4:].clone(),
        fast_known_mask=torch.rand(b,16,generator=rng)>.35)
    world=dict(world_tokens=torch.randn(b,23,5,generator=rng,dtype=token_dtype),world_types=types,
        reference_actions=torch.randn(b,h,a,generator=rng,dtype=dtype),reference_action_intervals=intervals[:,:h].clone())
    system=SimpleNamespace(test_fixture=False,fast=SimpleNamespace(config=SimpleNamespace(action_dim=a)))
    return system,batch,world


def compare(system,batch,world):
    expected=legacy(system,batch,world);actual=DualRateSystem._action_deviation(system,batch,world)
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0,equal_nan=True)
    assert torch.equal(actual[1],expected[1])
    return actual


@pytest.mark.parametrize('seed',range(5))
@pytest.mark.parametrize('dtype,token_dtype',[(torch.float32,torch.float32),(torch.float64,torch.float32),
    (torch.float32,torch.bfloat16),(torch.bfloat16,torch.bfloat16)])
def test_values_validity_scattered_slots_and_unavailable_nan_suffix(seed,dtype,token_dtype):
    system,batch,world=fixture(seed,dtype=dtype,token_dtype=token_dtype)
    batch['executed_actions'][~batch['executed_action_valid']]=torch.nan
    batch['fast_known_actions'][~batch['fast_known_mask']]=torch.nan
    compare(system,batch,world)


def test_first_match_gradients_preserve_execution_precedence_and_reference_gradient():
    system,batch,world=fixture(dtype=torch.float64,token_dtype=torch.float64)
    batch['executed_action_valid'].fill_(True);batch['fast_known_mask'].fill_(True)
    leaves=[batch['executed_actions'],batch['fast_known_actions'],world['reference_actions']]
    for value in leaves:value.requires_grad_()
    expected,_=legacy(system,batch,world);actual,_=DualRateSystem._action_deviation(system,batch,world)
    weight=torch.arange(expected.numel(),dtype=torch.float64).reshape_as(expected)/17
    eg=torch.autograd.grad((expected*weight).sum(),leaves)
    ag=torch.autograd.grad((actual*weight).sum(),leaves)
    for left,right in zip(eg,ag):torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert ag[1][:,:4].eq(0).all()  # duplicate commitment loses to already issued trace


@pytest.mark.parametrize('case',['none_known','zero_candidates','empty_reference','missing_fixture_trace'])
def test_empty_paths_do_not_create_new_gradient_edges(case):
    system,batch,world=fixture();world['reference_actions'].requires_grad_()
    if case=='none_known':
        batch['executed_action_valid'].zero_();batch['fast_known_mask'].zero_()
    elif case=='zero_candidates':
        for key in ('executed_actions','executed_action_intervals','executed_action_valid',
                    'fast_known_actions','action_intervals','fast_known_mask'):batch[key]=batch[key][:,:0]
    elif case=='empty_reference':world['reference_actions']=world['reference_actions'][:,:0]
    else:
        system.test_fixture=True;batch.pop('executed_actions')
    actual=compare(system,batch,world)
    assert not actual[0].requires_grad and not actual[1].any()


@pytest.mark.parametrize('case',['future_trace','wrong_dtype','wrong_shape','wrong_mask','slot_count',
    'conflict','nan','second_row_layout','missing_trace'])
def test_rejections_match_original(case):
    system,batch,world=fixture()
    batch['executed_action_valid'].fill_(True);batch['fast_known_mask'].fill_(True)
    if case=='future_trace':batch['query_time'].zero_()
    elif case=='wrong_dtype':batch['executed_action_intervals']=batch['executed_action_intervals'].float()
    elif case=='wrong_shape':batch['executed_action_intervals']=batch['executed_action_intervals'][:,:1]
    elif case=='wrong_mask':batch['executed_action_valid']=batch['executed_action_valid'].long()
    elif case=='slot_count':world['world_types'][0].zero_()
    elif case=='conflict':batch['fast_known_actions'][0,0,0]+=.1
    elif case=='nan':batch['executed_actions'][0,0,0]=torch.nan
    elif case=='second_row_layout':
        batch['fast_known_actions'][0,0,0]+=.1;world['world_types'][1].zero_()
    else:batch.pop('executed_actions')
    with pytest.raises(ValueError) as expected:legacy(system,batch,world)
    with pytest.raises(ValueError) as actual:DualRateSystem._action_deviation(system,batch,world)
    assert str(actual.value)==str(expected.value)


@pytest.mark.parametrize('offset',[.5e-7,2e-7])
def test_matching_uses_both_physical_interval_endpoints(offset):
    system,batch,world=fixture()
    world['reference_action_intervals'][:,:,1]+=offset
    compare(system,batch,world)


@pytest.mark.parametrize('difference',[.5e-5,2e-5,float('inf')])
def test_duplicate_tolerance_and_infinity_match_allclose(difference):
    system,batch,world=fixture()
    batch['executed_action_valid'].fill_(True);batch['fast_known_mask'].fill_(True)
    batch['executed_actions'][0,4,0]=0.
    batch['fast_known_actions'][0,0,0]=difference
    if difference==float('inf'):
        batch['executed_actions'][0,4,0]=difference
    if difference==2e-5:
        with pytest.raises(ValueError):legacy(system,batch,world)
        with pytest.raises(ValueError):DualRateSystem._action_deviation(system,batch,world)
    else:compare(system,batch,world)
