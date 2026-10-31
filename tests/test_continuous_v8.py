"""CPU architecture/contracts only, not a GPU/SR acceptance claim."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import pytest
import torch
from dynamicwam.dualrate.continuous_owner import ContinuousStepOwner
from dynamicwam.dualrate.continuous_v8 import ContinuousResidualFast, configure, aligned_draft
from dynamicwam.dualrate.refresh import RefreshConfig, RefreshController
from dynamicwam.dualrate.timing import TimingError
from test_dualrate_system import make_system,generator


def fixture():
    system,batch=make_system()
    batch['fast_action_valid']=torch.ones(batch['fast_action'].shape[:2],dtype=torch.bool)
    system.fast=ContinuousResidualFast(system.fast.config)
    configure(system,dict(draft_max_age=2.,draft_dropout=.2,future_dropout=.2,
                         action_mode='residual',independent_future_gate=True))
    world=system.world_readout(system.predict_reference(batch,generator=generator()))
    return system,batch,world


def test_physics_owner_exact_ticks_no_hidden_step_or_foreign_thread():
    ticks=[0]
    def step():ticks[0]+=1
    scene=SimpleNamespace(step=step,get_timestep=lambda:.004)
    clock=lambda:ticks[0]*.004
    with ContinuousStepOwner(scene,clock) as owner:
        for _ in range(45):owner.advance()
        assert owner.receipt()['physical_time']==.18
        with pytest.raises(TimingError):scene.step()
        errors=[]
        def foreign():
            try:owner.advance()
            except TimingError as exc:errors.append(exc)
        t=threading.Thread(target=foreign);t.start();t.join()
        assert len(errors)==1 and ticks[0]==45
        ticks[0]+=1
        with pytest.raises(TimingError,match='outside owner'):owner.advance()
    assert scene.step is step


def test_draft_exact_interval_tail_expiry_absence_and_revocation():
    system,batch,world=fixture()
    draft,valid=aligned_draft(batch,world,max_age=2.)
    assert valid.all()
    torch.testing.assert_close(draft,world['reference_actions'])
    altered=dict(batch,action_intervals=batch['action_intervals']+.05)
    a,m=aligned_draft(altered,world,max_age=2.)
    assert not m.any() and not a.any()
    altered=dict(batch,query_time=batch['query_time']+3)
    assert not aligned_draft(altered,world,max_age=2.)[1].any()
    a,m=aligned_draft(batch,system.empty_world(batch),max_age=2.)
    assert a.shape==draft.shape and not m.any() and not a.any()
    revoked=dict(batch,draft_allowed=torch.zeros_like(valid))
    assert not aligned_draft(revoked,world,max_age=2.)[0].any()
    dropped=system.mask_reference(world,batch,torch.tensor([False]))
    assert not aligned_draft(batch,dropped,max_age=2.)[1].any()


def test_future_off_preserves_history_and_draft_and_ignores_poisoned_future():
    system,batch,world=fixture();system.eval()
    batch=dict(batch,world_future_allowed=torch.tensor([False]))
    cond=system.conditions(batch,world)
    ctx=system.fast.prepare_context(**cond)
    assert ctx.slow_allowed[:,0,...,1:].any()
    assert not ctx.slow_allowed[:,1,...,1:].any()
    assert cond['draft_valid'].all()
    poisoned=dict(world,world_tokens=world['world_tokens'].clone())
    poisoned['world_tokens'][world['world_types']==1]+=1234
    a=system.sample_fast(batch,world,steps=2,generator=generator())
    b=system.sample_fast(batch,poisoned,steps=2,generator=generator())
    torch.testing.assert_close(a['actions'],b['actions'],atol=0,rtol=0)
    torch.testing.assert_close(a['future'],b['future'],atol=0,rtol=0)
    torch.testing.assert_close(a['actions'][batch['fast_known_mask']],
        batch['fast_known_actions'][batch['fast_known_mask']],atol=0,rtol=0)


def test_same_model_full_mode_train_backward_and_state_reload():
    system,batch,world=fixture()
    system.train()
    result=system.training_loss(batch,generator(),action_conditioned=torch.tensor([False]))
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert system.fast.draft_embedding.weight.grad is not None
    assert all(p.grad is None for p in system.slow.parameters())
    system.eval()
    expected=system.sample_fast(batch,system.empty_world(batch),steps=2,generator=generator())
    other,_,_=fixture();other.load_state_dict(system.state_dict(),strict=True);other.eval()
    actual=other.sample_fast(batch,other.empty_world(batch),steps=2,generator=generator())
    torch.testing.assert_close(expected['actions'],actual['actions'],atol=0,rtol=0)


def test_query_cache_and_graph_flatten_copy_both_independent_masks():
    from dynamicwam.dualrate.fast_graph import _flatten,_context
    system,batch,world=fixture()
    ctx=system.fast.prepare_context(**system.conditions(batch,world))
    values=_flatten(batch['fast_action'],batch['future'],batch['fast_known_actions'],
        batch['fast_known_mask'],ctx,batch['future_loss_valid'],batch['contact_valid'])
    copied={k:v.clone() for k,v in values.items()}
    other=_context(ctx,copied)
    assert other.slow_allowed.shape[1]==2
    torch.testing.assert_close(other.slow_allowed,ctx.slow_allowed)
    ctx.inputs['action_draft'].add_(1)
    with pytest.raises(ValueError,match='modified'):system.fast._check_cache(ctx)


def test_refresh_persistence_unique_frames_budget_coalescing_and_max_wait():
    control=RefreshController(RefreshConfig(high=.3,low=.1))
    for _ in range(8):control.observe(now=.72,frame_id=0,residual=.4,comparable=True)
    assert control.due(.72) is None
    control.observe(now=.76,frame_id=1,residual=.4,comparable=True)
    control.observe(now=.8,frame_id=2,residual=.4,comparable=True)
    assert control.started(.8)['reason']=='persistent_world_mismatch'
    for frame in range(3,9):control.observe(now=frame*.4,frame_id=frame,residual=.4,comparable=True)
    assert control.due(3.6) is None
    control.completed()
    assert control.started(3.6)['frame_id']==8
    control.completed()
    control.observe(now=3.7,frame_id=9,residual=.01,comparable=True)
    assert control.due(4.4) is None
    assert control.due(5.77)=='maximum_reference_wait'
    with pytest.raises(ValueError):control.due(3.)


def test_native_data_cannot_be_relabelled_as_continuous():
    from dynamicwam.dualrate.continuous_contract import validate_training_contract
    with pytest.raises(ValueError,match='audited'):
        validate_training_contract(dict(action_interval_seconds=.18,time_contract='native_event_single_v6'),{})


@pytest.mark.parametrize('known_count',[0,2,3])
def test_full_and_residual_modes_preserve_same_absolute_commitments(known_count):
    system,batch,world=fixture();system.eval()
    batch=dict(batch,fast_known_mask=torch.arange(4)[None]<known_count)
    for mode in ('full','residual'):
        system.continuous_v8['action_mode']=mode
        result=system.sample_fast(batch,world,steps=2,generator=generator())
        torch.testing.assert_close(result['actions'][batch['fast_known_mask']],
            batch['fast_known_actions'][batch['fast_known_mask']],atol=0,rtol=0)
        assert torch.isfinite(result['actions']).all()


def test_grippers_absolute_and_invalid_tail_never_uses_nearest_reference():
    intervals=torch.tensor([[[0.,.18],[.18,.36],[.36,.54]]],dtype=torch.float64)
    world=dict(reference_action_intervals=intervals,reference_actions=torch.ones(1,3,14)*4,
        world_types=torch.full((1,3),2),world_intervals=intervals,
        world_valid=torch.ones(1,3,dtype=torch.bool),source_time=torch.tensor([0.],dtype=torch.float64),
        completed_time=torch.tensor([.1],dtype=torch.float64))
    batch=dict(action_intervals=intervals+.18,query_time=torch.tensor([.2],dtype=torch.float64))
    draft,valid=aligned_draft(batch,world,max_age=2.)
    assert valid.tolist()==[[True,True,False]]
    assert (draft[0,:2,0]==4).all()  # No .15 clipping or forced-small residual.
    assert not draft[:,:,[6,13]].any() and not draft[:,2].any()


def test_shared_training_history_selector_irregular_physical_clock():
    from dynamicwam.dualrate.online import history_endpoints
    from dynamicwam.dualrate.producer import flow_history_selection
    times=[0.,.41,.73,1.1,1.44,1.98,2.16,2.88]
    frames=tuple(SimpleNamespace(exposure=t,frame_id=i) for i,t in enumerate(times))
    got,mask=history_endpoints(frames)
    indices,expected=flow_history_selection(times,len(times)-1)
    assert [f.frame_id for f in got]==indices and list(mask)==expected
    assert got[-2].exposure==2.16


def test_each_phase_keeps_full_slow_frozen_and_fast_live():
    from dynamicwam.dualrate.train_distributed import set_phase,PHASES
    system,_,_=fixture()
    for phase in PHASES:
        set_phase(system,phase)
        assert not system.slow.training
        assert not any(p.requires_grad for p in system.slow.parameters())
        assert all(p.requires_grad for p in system.fast.parameters())


def test_checkpoint_semantics_do_not_depend_only_on_parameter_shapes():
    from dynamicwam.dualrate.continuous_contract import validate_policy_contract
    validate_policy_contract({}, {})
    cfg={'continuous_v8':{'action_mode':'residual'},'runtime':{'submission_contract':'worker_receipt_v8'},
         'runtime_contract':{'submission_contract':'worker_receipt_v8'}}
    validate_policy_contract(cfg,cfg)
    with pytest.raises(ValueError):validate_policy_contract(cfg,{})
    with pytest.raises(ValueError):validate_policy_contract(cfg,{'continuous_v8':{'action_mode':'full'}})


def test_shared_gate_control_variant_and_learned_zero_future_gate():
    system,batch,world=fixture();system.eval()
    for block in system.fast.blocks:
        with torch.no_grad():block.future_gate.bias.fill_(-10)
    altered=dict(world,world_tokens=world['world_tokens'].clone())
    altered['world_tokens'][world['world_types']==1]+=100
    a=system.sample_fast(batch,world,steps=2,generator=generator())
    b=system.sample_fast(batch,altered,steps=2,generator=generator())
    torch.testing.assert_close(a['actions'],b['actions'],atol=0,rtol=0)
    system.fast=ContinuousResidualFast(system.fast.config,independent_future_gate=False)
    configure(system,dict(draft_max_age=2.,draft_dropout=.2,future_dropout=.2,
        action_mode='full',independent_future_gate=False))
    system.eval()
    result=system.sample_fast(batch,world,steps=2,generator=generator())
    assert torch.isfinite(result['actions']).all()


def test_guard_replay_is_one_worker_but_holds_eight_gpu_descriptors():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).parents[1]/'scripts/guarded_dualrate_run.py').read_text())
    expressions=[k.value for node in ast.walk(tree) if isinstance(node,ast.Call)
                 for k in node.keywords if k.arg=='world_size' and isinstance(k.value,ast.IfExp)]
    assert len(expressions)==1
    expr=ast.Expression(expressions[0])
    value=eval(compile(expr,'guard_contract','eval'),{'mode':'dualrate_failure_replay','TRAIN_MODES':set()})
    assert value==1
