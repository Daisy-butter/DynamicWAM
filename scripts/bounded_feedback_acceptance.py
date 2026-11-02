"""Actual GPU gate for the one-pass model, including changed-input graph replay."""
import json
from pathlib import Path

def graph_acceptance(system,config):
    import torch
    from dynamicwam.dualrate.producer import DominoPhysicalDataset,materialize_native_latents
    from dynamicwam.dualrate.train import _device_batch
    from dynamicwam.dualrate.native_v9 import draft
    m=json.loads(Path(config['manifest']).read_text());mp=m.get('expert_manifest',config['manifest'])
    ds=DominoPhysicalDataset(mp)
    index=next(i for i,r in enumerate(ds.rows) if int(r[2])>0)
    b=materialize_native_latents(_device_batch(torch.utils.data.default_collate([ds[index]]),torch.device('cuda:0')),system.slow.native.compact_wan.video_model.vae)
    with torch.no_grad():
        world=system.world_readout(system.predict_reference(b));world['native_available']=torch.tensor([True],device='cuda:0')
        eager=system.sample_fast(b,world)
        prior,valid=draft(world,b,config['native_v9']['reference_ttl'])
        zero=not bool(system.fast.action_out.weight.any() or system.fast.action_out.bias.any())
        if zero and not torch.equal(eager['actions'],prior):raise ValueError('zero residual is not exact Full baseline')
        if not torch.equal(eager['actions'][:,:,[6,13]],prior[:,:,[6,13]]):raise ValueError('gripper events changed')
        weight=system.fast.action_out.weight.detach().clone()
        system.fast.action_out.weight.copy_(torch.linspace(-.03,.03,weight.numel(),device=weight.device).reshape_as(weight))
        engine=system.enable_fast_graph();base=system.sample_fast(b,world)
        interaction=dict(b,interaction_history=torch.ones(1,8,64,device='cuda:0'),interaction_valid=torch.zeros(1,8,device='cuda:0',dtype=torch.bool),
            interaction_intervals=b['query_time'][:,None,None].expand(1,8,2).clone(),
            interaction_command_indices=torch.full((1,8),-1,device='cuda:0',dtype=torch.long))
        interaction['interaction_valid'][:,-1]=True
        interaction['interaction_command_indices'][:,-1]=b['native_query_index']-1
        interaction['interaction_intervals'][:,-1,0]-=.1
        cases={
            'observation':(dict(b,rgb=b['rgb']*.5),world),
            'world':(b,dict(world,world_tokens=world['world_tokens']*.5)),
            'draft':(b,dict(world,reference_actions=world['reference_actions']+.125)),
            'interaction':(interaction,world),
        }
        differences={}
        for name,(bb,ww) in cases.items():
            got=system.sample_fast(bb,ww)
            system._fast_graph_sampler=None;expected=system.sample_fast(bb,ww);system._fast_graph_sampler=engine
            for key in got:torch.testing.assert_close(got[key],expected[key],atol=2e-5,rtol=2e-5)
            difference=float((got['actions']-base['actions']).abs().max())
            if difference<=1e-8:raise ValueError(name+' disconnected from action')
            differences[name]=difference
        graph=engine.seal(expected_shapes=1)
        system._fast_graph_sampler=None;del engine
        system.fast.action_out.weight.copy_(weight)
        engine=system.enable_fast_graph();restored=system.sample_fast(b,world)
        for key in eager:torch.testing.assert_close(eager[key],restored[key],atol=2e-5,rtol=2e-5)
        final=engine.seal(expected_shapes=1)
    return dict(passed=True,contract='native_bounded_feedback_onepass_v10',zero_correction_parity=zero,
        exact_gripper_preservation=True,sensitivities=differences,nonzero_graph=graph,graph=final,
        privileged_future_is_input=False)
