import numpy as np
import pytest
import torch
from test_dualrate_native_v7 import inputs
from dynamicwam.dualrate.fast import FastConfig
from dynamicwam.dualrate.bounded_feedback import BoundedFeedback,InteractionEncoder,outputs
from dynamicwam.dualrate.fast_graph import _flatten,_context
from dynamicwam.dualrate.native_interaction import InteractionHistory,read_arm_state

torch.set_num_threads(1)

def fixture():
    torch.manual_seed(42)
    m=BoundedFeedback(FastConfig(dim=24,heads=3,layers=2,future_tokens=12,future_entities=3,
        observation_dim=12,world_dim=12,test_fixture=True),[.1]*14).eval()
    kw=inputs();kw.update(draft_actions=torch.randn(1,16,14),draft_valid=torch.ones(1,16,dtype=torch.bool))
    return m,kw

def test_zero_identity_nonzero_bounds_gripper_and_fresh_passthrough():
    m,k=fixture();c=m.prepare_context(**k)
    assert torch.equal(m.predict(c)['actions'],k['draft_actions'])
    m.action_out.weight.data.normal_(0,2);m.action_out.bias.data.fill_(3)
    result=m.predict(m.prepare_context(**k))['actions']
    assert (result-k['draft_actions']).abs().max()<=.100001
    assert torch.equal(result[:,:,[6,13]],k['draft_actions'][:,:,[6,13]])
    k['native_query_index']=k['native_source_index'].clone()
    assert torch.equal(m.predict(m.prepare_context(**k))['actions'],k['draft_actions'])

def test_observation_world_plan_tokens_and_achieved_history_reach_action():
    m,k=fixture();m.action_out.weight.data.normal_(0,.1)
    for name in ('observation_tokens','world_tokens'):k[name].requires_grad_()
    k['interaction_history']=torch.randn(1,8,64,requires_grad=True)
    k['interaction_valid']=torch.ones(1,8,dtype=torch.bool)
    out=m.predict(m.prepare_context(**k));out['actions'].square().sum().backward()
    for name in ('observation_tokens','world_tokens','interaction_history'):assert k[name].grad.abs().sum()>0
    assert m.plan_tokens.grad.abs().sum()>0

def test_no_resampling_no_hidden_residual_accumulation():
    m,k=fixture();m.action_out.weight.data.normal_(0,.1);c=m.prepare_context(**k)
    a=m.predict(c)['actions'];torch.randn(1000)
    assert torch.equal(a,m.predict(c)['actions'])
    assert torch.equal(k['draft_actions'],c.draft_actions)

def test_future_labels_are_not_action_inputs_but_actions_condition_forecast():
    m,k=fixture();c=m.prepare_context(**k);out=m.predict(c)
    z1,_=m.forecast_for(out['hidden'],out['actions'],c)
    z2,_=m.forecast_for(out['hidden'],out['actions']+1,c)
    assert not torch.equal(z1,z2)
    assert torch.equal(out['actions'],m.predict(c)['actions'])

def test_graph_context_has_correction_mask_and_original_mutation_guard():
    m,k=fixture();c=m.prepare_context(**k);a=k['draft_actions'];z=torch.zeros(1,12,134)
    values=_flatten(a,z,a,torch.zeros(1,16,dtype=torch.bool),c,torch.ones_like(z,dtype=torch.bool),torch.ones(1,12,2,dtype=torch.bool))
    assert 'correction_valid' in values
    private=_context(c,{key:value.clone() for key,value in values.items()})
    assert torch.equal(m.predict(c)['actions'],m.predict(private)['actions'])
    k['observation_tokens'].add_(1)
    with pytest.raises(ValueError,match='modified'):m._check_cache(c)

def test_masked_interactions_are_inert_and_empty_is_finite():
    m=InteractionEncoder(12).eval();x=torch.randn(2,8,64);valid=torch.zeros(2,8,dtype=torch.bool)
    x[:]=float('nan');out,mask=m(x,valid)
    assert torch.isfinite(out).all() and not out.any() and not mask.any()

def test_command_achieved_is_real_state_not_reference_disagreement():
    h=InteractionHistory(np.zeros(14),np.ones(14))
    h.append(index=0,start=0.,end=.372,command=np.ones(14),before=np.zeros(12),after=np.full(12,.4),velocity=np.full(12,.2),source_time=0.)
    t=h.tensors(.372,1,'cpu');x=t['interaction_history'][0,-1]
    assert torch.allclose(x[14:26],torch.zeros(12))
    assert torch.allclose(x[26:38],torch.full((12,),.4))
    assert torch.allclose(x[50:62],torch.full((12,),.6))
    assert t['interaction_valid'].sum()==1
    with pytest.raises(ValueError):h.tensors(.1,1,'cpu')
    with pytest.raises(ValueError):h.append(index=2,start=.372,end=.5,command=np.ones(14),before=np.zeros(12),after=np.ones(12),velocity=np.zeros(12),source_time=0.)

def test_same_command_noise_is_matched_across_teachers():
    from dynamicwam.dualrate.native_v7_deployment import ordinal_action_noise
    a=ordinal_action_noise(3,16,14,7,torch.zeros(1))
    b=ordinal_action_noise(5,16,14,7,torch.zeros(1))
    assert torch.equal(a[:,2:],b[:,:14])

def test_masked_draft_tail_cannot_change_valid_action():
    m,k=fixture();m.action_out.weight.data.normal_(0,.1);k['draft_valid'][:,4:]=False
    a=m.predict(m.prepare_context(**k))['actions'][:,:4]
    k['draft_actions'][:,4:]+=100
    b=m.predict(m.prepare_context(**k))['actions'][:,:4]
    torch.testing.assert_close(a,b,atol=1e-7,rtol=0)

def test_half_subset_is_whole_episodes_per_task_style_not_per_unique_seed(tmp_path):
    import json
    from dynamicwam.dualrate.feedback_data import subset_manifest
    from dynamicwam.dualrate.data import sha256_file
    episodes=[]
    for task in ('a','b'):
        for style in ('clean','random'):
            for i in range(4):
                episodes.append(dict(task=task,style=style,scene_id=f'{task}:{i}',split='train',episode_uid=f'{task}/{style}/{i}'))
    (tmp_path/'episodes.json').write_text(json.dumps(episodes))
    np.save(tmp_path/'rows.npy',np.array([(i,j,a) for i in range(16) for j in range(2) for a in range(3)]))
    for f in ('text.json','prompts.json','stats.json'):(tmp_path/f).write_text('{}')
    m=dict(split='train',episodes_file='episodes.json',rows_file='rows.npy',rows_sha256=sha256_file(tmp_path/'rows.npy'),text_index='text.json',prompts_file='prompts.json',source_file_stats='stats.json')
    (tmp_path/'train.json').write_text(json.dumps(m))
    r=subset_manifest(tmp_path/'train.json',tmp_path/'half'/'train.json')
    assert len(r['selected_episode_ids'])==8 and r['rows']==48
    assert all(x==dict(total=4,selected=2) for x in r['strata'].values())

def test_feedback_loss_has_directional_gradient_outside_radius_and_no_future_leak():
    from types import SimpleNamespace
    from dynamicwam.dualrate.bounded_feedback import training_loss
    m,k=fixture()
    class System:
        fast=m
        slow=torch.nn.Linear(1,1)
        native_v6=dict(execute_weight=4.)
        loss_config=SimpleNamespace(visual_dim=128,future_visual=.25,future_geometry=1.,contact=.1)
        def predict_reference(self,batch,generator=None):
            a=k['draft_actions'].clone()
            if bool(batch['native_source_index'][0]==batch['native_query_index'][0]):a[:,:,0]+=.6
            return dict(reference_actions=a)
        def world_readout(self,old):return old
        def conditions(self,batch,world):return k
    b=dict(source_time=k['source_time'],completed_time=k['completed_time'],query_time=k['query_time'],
        native_query_index=k['native_query_index'],native_source_index=k['native_source_index'],
        fast_known_mask=torch.zeros(1,16,dtype=torch.bool),fast_action=k['draft_actions'].clone(),
        fast_action_valid=torch.ones(1,16,dtype=torch.bool),future=torch.zeros(1,12,134),
        future_loss_valid=torch.ones(1,12,134,dtype=torch.bool),contact=torch.zeros(1,12,2),
        contact_valid=torch.ones(1,12,2,dtype=torch.bool),feedback_noise_seed=torch.tensor([42]))
    result=training_loss(System(),b);result['loss'].backward()
    assert result['reachable_fraction']==0 and result['supervised_action_fraction']==1
    assert m.action_out.bias.grad[0]<0 and torch.isfinite(result['loss'])
