"""Physical reference reads, latest evidence, and unavailable-label contracts."""
import torch
from dataclasses import replace

from dynamicwam.dualrate.fast import FastJointWAM

import test_dualrate_fast as fast_tests
from test_dualrate_system import make_system, generator, copy_batch


def fast_fixture():
    fixture = fast_tests.FastJointTests()
    fixture.setUp()
    return fixture


def test_reference_reads_follow_absolute_target_time_not_token_position():
    f = fast_fixture()
    kw = dict(f.kw)
    kw['world_intervals'] = torch.tensor([[[10., 10.4], [10.4, 10.8]]] * 2,
                                        dtype=torch.float64)
    kw['target_times'] = torch.tensor([[10.6, 10.2, 10.7, 10.1] * 2] * 2,
                                     dtype=torch.float64)
    cache = f.model.prepare_context(**kw)
    assert cache.slow_allowed[0, 0].tolist() == [False, False, True]
    assert cache.slow_allowed[0, 1].tolist() == [False, True, False]
    assert cache.slow_allowed[0, 2].tolist() == [False, False, True]
    assert cache.slow_allowed[0, 3].tolist() == [False, True, False]
    expected = f.model(f.a, f.z, f.sigma, **kw)
    reordered = dict(kw)
    for name in ('world_tokens', 'world_valid', 'world_intervals'):
        reordered[name] = kw[name].flip(1)
    actual = f.model(f.a, f.z, f.sigma, **reordered)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)


def test_drive_target_endpoint_time_and_reference_boundary_history():
    system,batch=make_system()
    world=system.world_readout(system.predict_reference(batch,generator=generator()))
    condition=system.conditions(batch,world)
    torch.testing.assert_close(condition['target_times'][:,:4],batch['action_intervals'][...,1])
    f=fast_fixture();kw=dict(f.kw)
    kw['world_types']=torch.full((2,2),2,dtype=torch.long)
    kw['world_intervals']=torch.tensor([[[9.5,10.2],[10.2,10.4]]]*2,dtype=torch.float64)
    kw['target_times']=torch.tensor([[10.2,10.3,10.4,10.5]*2]*2,dtype=torch.float64)
    cache=f.model.prepare_context(**kw)
    assert cache.slow_allowed[0,0].tolist()==[False,True,False]
    assert cache.slow_allowed[0,1].tolist()==[False,True,True]
    # Earlier actions remain context for later futures, never averaged into
    # executable output or confused with the immutable queue prefix.
    assert cache.slow_allowed[0,2].tolist()==[False,True,True]


def test_slow_gate_can_reduce_reference_weight_from_latest_evidence():
    f = fast_fixture()
    block = f.model.blocks[0]
    captured = []
    hook = block.slow_gate.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone()))
    f.run_model()
    f.run_model(observation_tokens=f.kw['observation_tokens'] + 1)
    hook.remove()
    width = f.cfg.dim
    # Time, sigma, action-deviation and reference are unchanged. Only the new
    # evidence section of the gate input is allowed to change in block one.
    torch.testing.assert_close(captured[0][..., :width], captured[1][..., :width])
    torch.testing.assert_close(captured[0][..., -1], captured[1][..., -1])
    delta = captured[1][0, 0, width:2*width] - captured[0][0, 0, width:2*width]
    assert delta.abs().sum() > 0
    # Demonstrate representable downweighting, not a claim that the untrained
    # gate has already learned which particular observation implies mismatch.
    with torch.no_grad():
        block.slow_gate.weight[0, width:2*width] = -delta
    old = block.slow_gate(captured[0])[0, 0].sigmoid()
    new = block.slow_gate(captured[1])[0, 0].sigmoid()
    assert new < old


def test_missing_support_and_contact_values_never_become_zero_supervision():
    system, batch = make_system()
    system.future_field_support[4:8] = False
    system.contact_field_support[4:] = False
    batch['future_loss_valid'][:, 4:8] = False
    batch['contact_valid'][:, 4:] = False
    changed = copy_batch(batch)
    changed['future'][:, 4:8] = torch.nan
    changed['contact'][:, 4:] = torch.nan
    lhs = system.training_loss(batch, generator())
    rhs = system.training_loss(changed, generator())
    for name in lhs:
        torch.testing.assert_close(lhs[name], rhs[name], rtol=0, atol=0)


def test_wholly_unsupported_roles_do_not_inject_noise_into_action_attention():
    system, batch = make_system()
    system.future_field_support[4:8] = False
    system.contact_field_support[4:8] = False
    batch['query_entity_valid'] = torch.ones(1, 4, dtype=torch.bool)
    world = system.world_readout(system.predict_reference(batch, generator=generator()))
    conditions = system.conditions(batch, world)
    assert not conditions['future_valid'][:, 4:8].any()
    assert conditions['future_valid'][:, :4].all()
    a, z = torch.randn(1, 4, 2), torch.randn(1, 16, 10)
    first = system.fast(a, z, torch.ones(1), **conditions)
    z[:, 4:8] = torch.nan
    second = system.fast(a, z, torch.ones(1), **conditions)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_static_unsupported_fields_are_zero_padding_and_noise_cannot_change_actions():
    system,batch=make_system()
    system.future_field_support[:,0]=False
    batch['future_loss_valid'][:,:,0]=False
    seen=[]
    hook=system.fast.register_forward_pre_hook(lambda module,args:seen.append(args[1].detach().clone()))
    system.training_loss(batch,generator())
    hook.remove()
    assert not seen[0][...,0].any()
    world=system.world_readout(system.predict_reference(batch,generator=generator()))
    expected=system.sample_fast(batch,world,steps=2,generator=generator())
    original=system._noise
    def corrupted_unused_noise(shape,like,rng):
        value=original(shape,like,rng)
        if tuple(shape[-2:])==(16,10):value[...,0]=torch.nan
        return value
    system._noise=corrupted_unused_noise
    actual=system.sample_fast(batch,world,steps=2,generator=generator())
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)


def test_frozen_slow_warmup_skips_only_useless_supervised_pass():
    system, batch = make_system()
    for parameter in system.slow.parameters():
        parameter.requires_grad_(False)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('frozen slow supervision must not run')
    system.slow.supervised_forward = forbidden
    result = system.training_loss(batch, generator())
    assert result['slow_video'] == 0 and result['slow_action'] == 0
    result['loss'].backward()
    assert all(parameter.grad is None for parameter in system.slow.parameters())
    assert system.world_readout.future_projection.weight.grad.abs().sum() > 0
    assert system.fast.action_out.weight.grad.abs().sum() > 0


def test_execution_deviation_ignores_clean_future_and_uses_physical_intervals():
    system, batch = make_system()
    ref = system.world_readout(system.predict_reference(batch, generator=generator()))
    batch['executed_actions'] = torch.zeros(1, 0, 2)
    batch['executed_action_intervals'] = torch.zeros(1, 0, 2, dtype=torch.float64)
    batch['executed_action_valid'] = torch.zeros(1, 0, dtype=torch.bool)
    delta, valid = system._action_deviation(batch, ref)
    slots = (ref['world_types'][0] == 2).nonzero().flatten()
    assert valid[0, slots].tolist() == [True, False, False, False]
    changed = copy_batch(batch)
    changed['fast_action'].fill_(10000)
    changed['fast_known_actions'][:, 1:].fill_(torch.nan)
    after, after_valid = system._action_deviation(changed, ref)
    torch.testing.assert_close(after, delta, rtol=0, atol=0)
    assert torch.equal(after_valid, valid)


def test_physical_three_role_model_joint_gradient_and_sampling_layout():
    system, batch = make_system()
    indices=torch.tensor(list(range(4))+list(range(8,16)))
    system.fast=FastJointWAM(replace(system.fast.config,future_entities=3,future_tokens=12))
    system.future_field_support=system.future_field_support[indices].clone()
    system.contact_field_support=system.contact_field_support[indices].clone()
    for key in ('future','future_loss_valid','contact','contact_valid','future_times'):
        batch[key]=batch[key][:,indices]
    batch['query_entity_valid']=torch.ones(1,3,dtype=torch.bool)
    result=system.training_loss(batch,generator())
    result['loss'].backward()
    assert system.slow.native.video_weight.grad.abs().sum()>0
    assert system.slow.native.action_weight.grad.abs().sum()>0
    assert system.world_readout.future_projection.weight.grad.abs().sum()>0
    world=system.world_readout(system.predict_reference(batch,generator=generator()))
    output=system.sample_fast(batch,world,steps=2,generator=generator())
    assert output['future'].shape==(1,12,10)
    assert output['future_valid'].shape==(1,12)
    assert output['contact_probability'].shape==(1,12,2)
