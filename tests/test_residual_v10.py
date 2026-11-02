"""Small numerical fixtures; no real Full/GPU/robot performance claim."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dynamicwam.dualrate.residual_v10 import ResidualFast,configure,predict,anchor_future
from dynamicwam.dualrate.continuous_v8 import configure as configure_v8
from dynamicwam.dualrate.slow import StructuredWorldReadout
from dynamicwam.dualrate.fast import CurrentObservationEncoder
from test_dualrate_system import make_system,generator
from test_residual_v10_executor import BOUNDS,LIMITS


def fixture():
    s,b=make_system()
    s.slow.native.config.action_dim=14
    s.fast=ResidualFast(replace(s.fast.config,action_dim=14))
    s.world_readout=StructuredWorldReadout(latent_channels=2,action_dim=14,world_dim=12,native_hidden_dim=2,test_fixture=True)
    s.observation=CurrentObservationEncoder(dim=12,views=1,motion_dim=3,state_dim=14,text_dim=4,test_fixture=True)
    for name in ('source_initial_state','state'):b[name]=torch.zeros(1,14)
    for name in ('source_action','fast_action','source_known_actions','fast_known_actions'):
        b[name]=torch.zeros(1,4,14);b[name][:,:,[6,13]]=.5
    b['source_action'][:,2:,:6]=.03;b['fast_action'][:,2:,:6]=.03
    for name in ('source_known_mask','fast_known_mask'):b[name]=torch.tensor([[True,True,False,False]])
    b['state_time']=b['rgb_times'].amax(-1)
    b['fast_action_valid']=torch.ones(1,4,dtype=torch.bool)
    configure_v8(s,dict(draft_max_age=2.,draft_dropout=0.,future_dropout=0.,action_mode='residual',independent_future_gate=True))
    configure(s,dict(action_dt=.1,bounds=BOUNDS,limits=LIMITS,arm_radius=.06,gripper_radius=.05,
        execute_weight=4.,preserve_tolerance=.02,future_anchor='first_mutable_start',deterministic=True,teacher_weight=.5,
        queue_perturb_probability=.5,queue_perturb_radius=.04),[0.]*14,[1.]*14)
    b.update(executed_actions=b['source_action'].clone(),executed_action_intervals=b['source_action_intervals'].clone(),
             executed_action_valid=torch.zeros(1,4,dtype=torch.bool))
    b['future_times']=anchor_future(b,4)
    for name in ('condition_latent','text','initial_state','motion','motion_valid','motion_acceleration_valid',
                 'future_intervals','temporal_support','view_overlap','token_valid','input_type',
                 'known_actions','known_mask','action_times','action_intervals'):
        b['teacher_'+name]=b['source_'+name].clone()
    world=s.world_readout(s.predict_reference(b,generator=generator()))
    return s,b,world


def test_exact_zero_residual_and_randomness_independence():
    s,b,w=fixture();s.eval()
    a=s.sample_fast(b,w,steps=1,generator=generator(3))
    c=s.sample_fast(b,w,steps=1,generator=generator(999))
    torch.testing.assert_close(a['actions'],a['baseline_actions'],atol=1e-6,rtol=0)
    assert torch.equal(a['actions'],c['actions'])
    assert torch.equal(a['actions'][b['fast_known_mask']],b['fast_known_actions'][b['fast_known_mask']])
    with pytest.raises(ValueError):s.sample_fast(b,w,steps=10)


def test_future_times_reject_query_anchor_even_when_prefix_is_valid():
    s,b,w=fixture();b=dict(b,future_times=b['query_time'][:,None]+torch.tensor([.1,.2,.4,.8]*4))
    with pytest.raises(ValueError,match='first mutable'):s.sample_fast(b,w)


def test_clean_residual_supervision_has_nonzero_finite_gradient():
    s,b,w=fixture();result=s.training_loss(b,generator=generator())
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    g=s.fast.action_out.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().max()>0
    assert all(p.grad is None for p in s.slow.parameters())


def test_current_observation_connection_after_head_learning():
    s,b,w=fixture();s.eval()
    with torch.no_grad():s.fast.action_out.weight.normal_(std=.01)
    original=predict(s,b,w)[4]['normalized_correction']
    changed=dict(b,rgb=1.-b['rgb'])
    updated=predict(s,changed,w)[4]['normalized_correction']
    assert (updated-original).abs().max()>0
    # This is sensitivity only. Whether the change is HELPFUL requires paired
    # actual teacher targets and closed-loop intervention acceptance.


def test_refuse_missing_slow_baseline_not_a_random_full_action_fallback():
    s,b,w=fixture()
    with pytest.raises(ValueError,match='slow coverage'):s.sample_fast(b,s.empty_world(b))


def test_queue_counterfactual_only_changes_unstarted_unlocked_future_and_relabels():
    from dynamicwam.dualrate.residual_v10 import future_queue_augmentation,current_teacher_batch
    s,b,w=fixture();s.residual_v10['queue_perturb_probability']=1.
    b['source_known_mask'][:,1]=False
    b['executed_action_intervals'][:,0]=torch.tensor([2.,2.1],dtype=torch.float64)
    b['executed_action_valid'][:,0]=True
    changed,selected=future_queue_augmentation(s,b,generator())
    assert selected.all()
    assert torch.equal(changed['fast_known_actions'][:,0],b['fast_known_actions'][:,0])
    assert torch.equal(changed['source_known_actions'],b['source_known_actions'])
    assert (changed['fast_known_actions'][:,1]-b['fast_known_actions'][:,1]).abs().max()>0
    teacher=current_teacher_batch(changed)
    assert torch.equal(teacher['source_known_actions'][:,1],changed['fast_known_actions'][:,1])
    result=s.training_loss(b,generator=generator())
    assert result['queue_augmented']==1
    assert result['future_visual']==result['future_geometry']==result['contact']==0


def test_source_commitment_cannot_be_overwritten_by_queue_augmentation():
    from dynamicwam.dualrate.residual_v10 import future_queue_augmentation
    s,b,w=fixture();s.residual_v10['queue_perturb_probability']=1.
    b['executed_action_intervals'][:,0]=torch.tensor([2.,2.1],dtype=torch.float64)
    b['executed_action_valid'][:,0]=True
    changed,selected=future_queue_augmentation(s,b,generator())
    assert not selected.any() and torch.equal(changed['fast_known_actions'],b['fast_known_actions'])
