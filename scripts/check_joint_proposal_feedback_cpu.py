"""Causal evolving-plan feedback with real native joint/actor blocks; CPU only."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from check_joint_motionkv_v3_training_cpu import v3_fixture,cpu_attention
from check_joint_source_prefix_cpu import TinyCompact
from dynamicwam.models.small_wam import SmallWAMActionModel,SmallWAMActionConfig
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.rtc_prefix import PrefixCondition,apply_prefix
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable
from dynamicwam.inference.multirate_model_loader import export_fast_only_payload,load_fast_only_modules,build_fast_conditioner
from dynamicwam.inference.native_fast_graph import NativeFastGraph


def run():
    torch.set_num_threads(2);torch.manual_seed(20261016);checks=[]
    with ExitStack() as stack:
        for name in ("dynamicwam.models.small_wam.flash_attention","dynamicwam.vendor.wan.modules.model.flash_attention",
                     "dynamicwam.models.action_core_ops.flash_attention","dynamicwam.models.action_core_ops.dense_action_attention"):
            stack.enter_context(patch(name,cpu_attention))
        model,batch=v3_fixture();old=model.backbone
        cfg=SmallWAMActionConfig(SimpleNamespace(dim=12,num_heads=2,num_layers=2,head_dim=6),14,14,16,24,48,2,False,4,(0.,)*12,(1.,)*12)
        model.backbone=SmallWAMActionModel(cfg,TinyCompact())
        model.backbone.action_expert.load_state_dict(old.action_expert.state_dict())
        model.backbone.absolute_motion_tokens.load_state_dict(old.absolute_motion_tokens.state_dict())
        # The fixture carries source Video FM weight .1, hence it must use
        # the joint phase; warmup profiles correctly require Video weight zero.
        configure_multirate_trainable(model,"joint")
        legacy=deepcopy(model)
        disabled_spec=dict(legacy.fast_conditioner.constructor_spec);disabled_spec.pop("proposal_feedback",None)
        reloaded=build_fast_conditioner(disabled_spec);reloaded.load_state_dict(legacy.fast_conditioner.state_dict(),strict=True)
        assert not reloaded.proposal_feedback
        proposal=JointMotionKVConditioner(**dict(model.fast_conditioner.constructor_spec,proposal_feedback=True))
        result=proposal.load_state_dict(model.fast_conditioner.state_dict(),strict=False)
        assert len(result.missing_keys)==5 and not result.unexpected_keys
        model.fast_conditioner=proposal
        zero=model(batch)["native_loss"];reference=legacy(batch)["native_loss"]
        assert torch.equal(zero,reference)
        checks.append("old_v3_spec_strict_loads_disabled_and_enabled_zero_gate_is_bit_exact")
        configure_multirate_trainable(model,"joint")
        zero.backward();assert proposal.proposal_gate.grad.abs()>0
        with torch.no_grad():proposal.proposal_gate.fill_(.2)
        model.zero_grad(set_to_none=True)
        output=model(batch);output["native_loss"].backward()
        for module in (model.backbone.compact_wan,model.backbone.action_expert,model.backbone.absolute_motion_tokens,proposal.proposal_encoder):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append("joint_suffix_loss_reaches_proposal_encoder_and_native_video_action_motion")
        model.zero_grad(set_to_none=True);model.gradient_checkpointing=False
        a=model(batch)["native_loss"];a.backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True);model.gradient_checkpointing=True
        b=model(batch)["native_loss"];b.backward()
        assert torch.equal(a,b)
        assert all(torch.allclose(p.grad,grads[n],atol=1e-6,rtol=1e-5) for n,p in model.named_parameters() if n in grads)
        checks.append("nonreentrant_checkpoint_output_and_gradient_parity")
        bridge=model.bridge(model.backbone,batch["source"],batch["source_noise"],detach_cache=False)
        old,valid=proposal.aligned_source_plan(batch["current"],bridge)
        prefix=PrefixCondition(torch.randn_like(old),torch.tensor([0,4]))
        current=replace(batch["current"],committed_prefix=prefix.actions,committed_prefix_lengths=prefix.lengths,
            previous_plan=old,previous_plan_valid=valid)
        text=torch.stack(batch["source"].text_embeddings)
        memory=model.prepare_fast_memory(current,bridge,text)
        base_world=[x.detach().clone() for x in bridge.video_k+bridge.video_v]
        base_ranks=[x.detach().clone() for x in memory.rank_context]
        sample=apply_prefix(torch.randn_like(old),prefix).requires_grad_(True)
        times=torch.tensor([500.,500.])
        derived=proposal.condition_on_proposal(sample,times,memory,num_train_timesteps=1000)
        second=proposal.condition_on_proposal(sample+.5,times,memory,num_train_timesteps=1000)
        assert not torch.equal(derived.rank_context[0],second.rank_context[0])
        delta=(derived.rank_context[0]-memory.rank_context[0]).square().sum()
        delta.backward();assert sample.grad[:,4:].abs().sum()>0
        assert all(torch.equal(a,b) for a,b in zip(base_world,bridge.video_k+bridge.video_v))
        assert all(torch.equal(a,b) for a,b in zip(base_ranks,memory.rank_context))
        checks.append("changing_current_noisy_proposal_changes_context_with_suffix_gradients_without_mutating_source_or_memory")
        f,c=proposal.proposal_features(sample,torch.ones(2)*1000,memory,num_train_timesteps=1000)
        assert c[0].eq(0).all() and c[1,:4].eq(1).all() and c[1,4:].eq(0).all()
        noisy=sample.detach().clone();noisy[:,:]+=100;noisy=apply_prefix(noisy,prefix)
        first=proposal.condition_on_proposal(sample,torch.ones(2)*1000,memory,num_train_timesteps=1000)
        changed=proposal.condition_on_proposal(noisy,torch.ones(2)*1000,memory,num_train_timesteps=1000)
        assert torch.equal(first.rank_context[0],changed.rank_context[0])
        assert torch.equal(first.rank_context[0][0],memory.rank_context[0][0])
        assert not torch.equal(first.rank_context[0][1],memory.rank_context[0][1])
        nan_prefix=prefix.actions.clone();nan_prefix[0]=float('nan');nan_prefix[1,4:]=float('nan')
        safe_current=replace(current,committed_prefix=nan_prefix)
        safe_memory=model.prepare_fast_memory(safe_current,bridge,text)
        assert all(torch.equal(a,b) for a,b in zip(memory.rank_context,safe_memory.rank_context))
        assert f[1,:,14:28].eq(0).all()  # Source plan beyond physical horizon is masked.
        checks.append("pure_noise_suffix_has_zero_evidence_clean_prefix_full_strength_no_clean_suffix_leak_and_expired_source_mask")
        model.eval().requires_grad_(False)
        with torch.no_grad():
            memory=model.prepare_fast_memory(current,bridge,text)
            noise=sample.detach()
            eager=model.sample_fast_actions(current,bridge,memory,noise,prefix=prefix)
            graph=NativeFastGraph(model,enabled=False)
            obs,world=graph._canonical_context(current,bridge,noise,prefix)
            graph._allocate(obs,world,text,None,noise,prefix)
            unrolled=graph._forward()
            assert torch.equal(eager,unrolled)
            assert torch.equal(eager[1,:4],prefix.actions[1,:4])
        checks.append("K10_eager_and_graph_body_bit_exact_on_CPU_with_hard_prefix_preserved")
        spec=dict(action_expert=dict(dim=24,ffn_dim=48,num_layers=2,state_dim=14,action_dim=14,chunk_size=16),
            wan_attention=dict(dim=12,num_heads=2,head_dim=6),motion=dict(history_count=4,mean=[0.]*12,scale=[1.]*12),
            fast_conditioner=proposal.constructor_spec)
        fast=export_fast_only_payload(model.state_dict(),module_spec=spec,global_step=2)
        loaded=load_fast_only_modules(fast)
        assert loaded.fast_conditioner.proposal_feedback
        assert all(torch.equal(x,loaded.fast_conditioner.state_dict()[k]) for k,x in proposal.state_dict().items())
        checks.append("new_fast_export_strict_reload_preserves_feedback_parameters_and_flag")
        params=sum(p.numel() for p in proposal.proposal_encoder.parameters())+proposal.proposal_gate.numel()
        real_params=(4*14+8)*192+192+192*192+192+1
        print(json.dumps(dict(passed=True,checks=checks,loss=float(a),device="cpu",added_parameters_fixture=params,
            added_parameters_real_192dim=real_params,extra_backbone_forwards=0,
            limitation="two tiny native Wan blocks, compact projection and identity Video RoPE; graph body CPU parity is not CUDA capture/concurrency validation")))

if __name__=="__main__":run()
