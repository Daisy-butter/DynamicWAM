"""Same-trajectory source completion and distinct conditioning, CPU native tiny test."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from check_joint_motionkv_v3_training_cpu import v3_fixture,cpu_attention
from check_joint_source_prefix_cpu import TinyCompact
from dynamicwam.models.small_wam import SmallWAMActionModel,SmallWAMActionConfig
from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.native_wam_teacher import sample_native_wam
from dynamicwam.models.rtc_prefix import PrefixCondition
from dynamicwam.inference.native_fast_graph import NativeFastGraph
from dynamicwam.inference.multirate_model_loader import build_fast_conditioner
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable

def run():
    torch.set_num_threads(2);torch.manual_seed(20261016);checks=[]
    with ExitStack() as stack:
        for name in ('dynamicwam.models.small_wam.flash_attention','dynamicwam.vendor.wan.modules.model.flash_attention',
                     'dynamicwam.models.action_core_ops.flash_attention','dynamicwam.models.action_core_ops.dense_action_attention'):
            stack.enter_context(patch(name,cpu_attention))
        model,batch=v3_fixture();old=model.backbone
        cfg=SmallWAMActionConfig(SimpleNamespace(dim=12,num_heads=2,num_layers=2,head_dim=6),14,14,16,24,48,2,False,4,(0.,)*12,(1.,)*12)
        model.backbone=SmallWAMActionModel(cfg,TinyCompact())
        model.backbone.action_expert.load_state_dict(old.action_expert.state_dict())
        model.backbone.absolute_motion_tokens.load_state_dict(old.absolute_motion_tokens.state_dict())
        model.fast_conditioner=JointMotionKVConditioner(**dict(model.fast_conditioner.constructor_spec,proposal_feedback=True,complete_source_plan=True))
        model.bridge=CausalWorldBridge(model.schedule,complete_source_plan=True)
        core=model.backbone;source=batch['source'];noise=batch['source_noise'];conditioner=model.fast_conditioner
        with torch.no_grad():
            conditioner.innovation_gate.fill_(.1);conditioner.proposal_gate.fill_(.2);conditioner.completed_plan_gate.fill_(.2)
            for p in (*conditioner.delta_k,*conditioner.delta_v):p.weight.normal_(std=.02)
        old_bridge=CausalWorldBridge(model.schedule)(core,source,noise,detach_cache=False)
        with patch.object(core,'_forward_video_action',wraps=core._forward_video_action) as joints,patch.object(core,'forward_action_with_video_cache',wraps=core.forward_action_with_video_cache) as tails:
            bridge=model.bridge(core,source,noise,detach_cache=False)
        assert joints.call_count==2 and tails.call_count==8
        native=sample_native_wam(core,source,noise,model.schedule)
        assert torch.equal(native,bridge.source_completed_plan)
        assert torch.equal(bridge.source_action_plan,old_bridge.source_action_plan)
        assert torch.equal(bridge.source_action_sigma,old_bridge.source_action_sigma)
        assert all(torch.equal(a,b) for a,b in zip(bridge.video_k+bridge.video_v,old_bridge.video_k+old_bridge.video_v))
        assert not torch.equal(bridge.source_completed_plan,bridge.source_action_plan)
        assert not bridge.source_completed_plan.requires_grad and bridge.video_k[0].requires_grad
        checks.append('exact_native_same_noise_K10_plan_2joint8tail_distinct_earlyestimate_originalKV_attached')
        prefix=PrefixCondition(torch.randn_like(noise.actions),torch.tensor([0,4]))
        src=replace(source,committed_prefix=prefix.actions,committed_prefix_lengths=prefix.lengths)
        with patch.object(core,'_build_action_tokens',wraps=core._build_action_tokens) as tokens,patch.object(core,'_build_action_time_embeddings',wraps=core._build_action_time_embeddings) as times:
            prefixed=model.bridge(core,src,noise,detach_cache=False)
        assert len(tokens.call_args_list)==10 and len(times.call_args_list)==10
        for call in tokens.call_args_list:assert torch.equal(call.args[1][1,:4],prefix.actions[1,:4])
        for call in times.call_args_list:
            t=call.args[0];assert t[1,1:5].eq(0).all() and t[1,17:21].gt(0).all()
        assert torch.equal(prefixed.source_completed_plan[1,:4],prefix.actions[1,:4])
        checks.append('all10source_steps_use_clean_H4_and_preserve_native_motion_denoising_times')
        current=replace(batch['current'],committed_prefix=prefix.actions,committed_prefix_lengths=prefix.lengths,
            previous_plan=torch.zeros_like(noise.actions),previous_plan_valid=torch.zeros(noise.actions.shape[:2],dtype=torch.bool))
        text=torch.stack(source.text_embeddings);memory=model.prepare_fast_memory(current,bridge,text)
        changed=replace(bridge,source_completed_plan=bridge.source_completed_plan+1)
        changed_memory=model.prepare_fast_memory(current,changed,text)
        assert not torch.equal(memory.rank_context[0][0],changed_memory.rank_context[0][0])
        # Sample1's source plan has expired; arbitrary final plan cannot leak.
        assert torch.equal(memory.rank_context[0][1],changed_memory.rank_context[0][1])
        times=torch.ones(2)*500
        out=model.predict_fast_velocity(noise.actions,times,current,bridge,memory,prefix=prefix)
        other=model.predict_fast_velocity(noise.actions,times,current,changed,changed_memory,prefix=prefix)
        assert not torch.equal(out[0],other[0]) and torch.equal(out[1],other[1])
        null=replace(bridge,null_bridge=True);null_changed=replace(changed,null_bridge=True)
        a=model.prepare_fast_memory(current,null,text);b=model.prepare_fast_memory(current,null_changed,text)
        assert all(torch.equal(x,y) for x,y in zip(a.rank_context,b.rank_context))
        times0=torch.ones(2)*1000
        first0=model.predict_fast_velocity(noise.actions,times0,current,bridge,memory,prefix=prefix)
        changed0=model.predict_fast_velocity(noise.actions,times0,current,changed,changed_memory,prefix=prefix)
        assert not torch.equal(first0[0],changed0[0])
        checks.append('separate_completed_prior_changes_fastoutput_including_sigma1_only_on_valid_physical_targets_null_and_expired_masked')
        configure_multirate_trainable(model,'warmup');model.joint_video_weight=0.;model.joint_action_weight=.1
        model.zero_grad(set_to_none=True)
        with patch.object(core,'_forward_video_action',wraps=core._forward_video_action) as warm_joints:
            warm=model(batch)
        assert warm_joints.call_count==2 and warm['native_metrics']['retention/source_cached_kv']==1
        assert 'loss/action_retention' in warm['native_metrics'] and 'loss/video_retention' not in warm['native_metrics']
        warm['native_loss'].backward()
        assert all(p.grad is None for p in core.compact_wan.parameters())
        assert sum(float(p.grad.abs().sum()) for p in core.action_expert.parameters() if p.grad is not None)>0
        changed_target=dict(batch,source_action_target=batch['source_action_target']+1.)
        assert not torch.equal(warm['native_metrics']['loss/action_retention'],model(changed_target)['native_metrics']['loss/action_retention'])
        model.joint_video_weight=.1
        checks.append('warmup_bare_sourceKV_ActionFM_uses_source_labels_trains_native_actor_freezesVideo_noextraVideo_forward')
        configure_multirate_trainable(model,'joint')
        model.zero_grad(set_to_none=True);model.gradient_checkpointing=False
        first=model(batch)['native_loss'];first.backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        for module in (core.compact_wan,core.action_expert,core.absolute_motion_tokens,conditioner.completed_plan_encoder,conditioner.proposal_encoder):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
        model.zero_grad(set_to_none=True);model.gradient_checkpointing=True
        second=model(batch)['native_loss'];second.backward()
        assert torch.equal(first,second)
        assert all(torch.allclose(p.grad,grads[n],atol=1e-6,rtol=1e-5) for n,p in model.named_parameters() if n in grads)
        checks.append('stopgrad_proposal_keeps_joint_Video_Action_motion_interface_gradients_and_checkpoint_parity')
        legacy_spec=dict(conditioner.constructor_spec,complete_source_plan=False)
        legacy=build_fast_conditioner(legacy_spec)
        previous_spec=dict(legacy.constructor_spec);previous_spec.pop('complete_source_plan')
        strict=build_fast_conditioner(previous_spec);strict.load_state_dict(legacy.state_dict(),strict=True)
        assert not strict.complete_source_plan
        enabled=build_fast_conditioner(conditioner.constructor_spec);enabled.load_state_dict(conditioner.state_dict(),strict=True)
        checks.append('old_v31_strict_disabled_and_new_flag_strict_enabled_reload')
        model.eval().requires_grad_(False)
        with torch.no_grad():
            memory=model.prepare_fast_memory(current,bridge,text)
            eager=model.sample_fast_actions(current,bridge,memory,noise.actions,prefix=prefix)
            graph=NativeFastGraph(model,enabled=False)
            obs,world=graph._canonical_context(current,bridge,noise.actions,prefix)
            graph._allocate(obs,world,text,None,noise.actions,prefix)
            assert graph.bridge.source_completed_plan.data_ptr()!=bridge.source_completed_plan.data_ptr()
            assert torch.equal(eager,graph._forward())
            graph._copy_inputs(current,changed,text,None,noise.actions,prefix)
            assert torch.equal(graph.bridge.source_completed_plan,changed.source_completed_plan)
        checks.append('graph_owns_and_updates_finalplan_tensor_eager_graph_body_K10_exact')
        print(json.dumps(dict(passed=True,checks=checks,loss=float(first),device='cpu',extra_source_action_forwards=8,
            extra_video_forwards=0,stop_gradient_proposal=True,limitation='tiny native Wan blocks/projection/identityVideoRoPE; no fullGPU latency memory or closedloop validation')))
if __name__=='__main__':run()
