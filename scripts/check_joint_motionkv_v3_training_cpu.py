"""V3 actual actor/router/bridge CPU smoke; tiny Video DiT stand-in, no CUDA."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from check_joint_motionkv_training_cpu import joint_fixture,cpu_attention
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.rtc_prefix import PrefixCondition, apply_prefix
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable,multirate_optimizer_groups,MultirateTrainer,migrate_joint_interface_initialization,prepare_multirate_batch
from dynamicwam.inference.multirate_model_loader import install_full_checkpoint,export_fast_only_payload,load_fast_only_modules,fast_only_matches


def v3_fixture():
    model,batch=joint_fixture()
    spec=dict(model.fast_conditioner.constructor_spec,interface_version=3)
    model.fast_conditioner=JointMotionKVConditioner(**spec)
    with torch.no_grad():
        model.fast_conditioner.innovation_gate.fill_(.1)
        for head in (*model.fast_conditioner.delta_k,*model.fast_conditioner.delta_v): head.weight.normal_(std=.02)
    source=batch["source"]
    offsets=torch.arange(1,17,dtype=torch.float64)[None].expand(2,-1)*.16
    batch["source"]=replace(source,action_target_offsets=offsets,
        committed_prefix=torch.zeros_like(batch["clean_actions"]),committed_prefix_lengths=torch.zeros(2,dtype=torch.long))
    batch["current"]=replace(batch["current"],action_target_offsets=offsets,
        age=replace(batch["current"].age,delta_sim_seconds=torch.tensor([.08,2.8])))
    batch["source_aligned"]=dict(batch["aligned"],rtc_prefix=PrefixCondition(batch["source"].committed_prefix,batch["source"].committed_prefix_lengths))
    batch["source_video_target"]=batch["video_target"].clone()
    batch["source_action_target"]=batch["action_target"].clone()
    model.motion_prediction_weight=.1
    model.native_teacher_weight=0.
    model.external_native_teacher=False
    batch.pop("native_teacher_actions")
    return model,batch


def run():
    torch.set_num_threads(2)
    checks=[]
    cfg=dict(action_learning_rate=1e-3,fast_learning_rate=1e-3,video_learning_rate=1e-4,weight_decay=0.)
    with patch("dynamicwam.models.action_core_ops.flash_attention",cpu_attention), patch("dynamicwam.models.action_core_ops.dense_action_attention",cpu_attention):
        model,batch=v3_fixture()
        configure_multirate_trainable(model,"joint")
        optimizer=torch.optim.AdamW(multirate_optimizer_groups(model,cfg,"joint"))
        original={k:v.detach().clone() for k,v in model.fast_conditioner.motion_forecast.state_dict().items()}
        losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output=model(batch);loss=output["native_loss"]
            assert torch.isfinite(loss)
            assert "loss/motion_forecast" in output["native_metrics"]
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert sum(float(p.grad.abs().sum()) for p in model.backbone.compact_wan.video_model.wan_model.parameters())>0
            assert sum(float(p.grad.abs().sum()) for p in model.backbone.action_expert.parameters() if p.grad is not None)>0
            assert sum(float(p.grad.abs().sum()) for p in model.backbone.absolute_motion_tokens.parameters() if p.grad is not None)>0
            if step:
                for name in ("motion_forecast","visual_innovation","motion_innovation","plan_innovation"):
                    assert sum(float(p.grad.abs().sum()) for p in getattr(model.fast_conditioner,name).parameters() if p.grad is not None)>0,name
            optimizer.step();losses.append(float(loss))
        assert any(not torch.equal(original[k],v) for k,v in model.fast_conditioner.motion_forecast.state_dict().items())
        checks.append("two_teacher_free_joint_steps_update_slow_native_actor_motion_and_all_innovations_forecast")
        for null in (False,True):
            optimizer.zero_grad(set_to_none=True)
            probe_batch=dict(batch,null_bridge=torch.full((2,),null,dtype=torch.bool))
            model(probe_batch)["native_loss"].backward()
            missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing,missing
        checks.append("all_trainable_parameters_have_gradients_with_present_or_null_cache_ddp_graph")
        before=model(batch)["native_metrics"]
        changed=dict(batch,aligned={**batch["aligned"],"future_latent":batch["aligned"]["future_latent"]+100},
            video_target=batch["video_target"]+100,action_target=batch["action_target"]+100)
        after=model(changed)["native_metrics"]
        assert torch.equal(before["loss/video_retention"],after["loss/video_retention"])
        assert torch.equal(before["loss/action_retention"],after["loss/action_retention"])
        checks.append("source_time_video_action_retention_ignores_misaligned_current_targets")
        bridge=model.bridge(model.backbone,batch["source"],batch["source_noise"],detach_cache=False)
        plan=torch.arange(16,dtype=torch.float32)[None,:,None].expand(2,-1,14).clone()
        probe=replace(bridge,source_action_plan=plan)
        aligned,valid=model.fast_conditioner.aligned_source_plan(batch["current"],probe)
        assert torch.allclose(aligned[0,0,[0,5,7,12]],torch.full((4,),.5),atol=1e-6)
        assert torch.equal(aligned[0,0,[6,13]],torch.zeros(2))
        assert not valid[1].any() and torch.equal(aligned[1],torch.zeros_like(aligned[1]))
        altered=replace(batch["current"],age=replace(batch["current"].age,delta_actions=torch.tensor([999.,999.])))
        other,other_valid=model.fast_conditioner.aligned_source_plan(altered,probe)
        assert torch.equal(other,aligned) and torch.equal(other_valid,valid)
        checks.append("physical_source_plan_alignment_linear_arms_hold_grippers_masks_expiry_ignores_integer_age")
        absolute_source=torch.tensor(.1,dtype=torch.float64)
        absolute_current=torch.tensor(.58,dtype=torch.float64)
        source_grid=torch.arange(1,17,dtype=torch.float64)[None].expand(2,-1)*.16
        current_grid=torch.arange(4,20,dtype=torch.float64)[None].expand(2,-1)*.16
        precise_bridge=replace(probe,source_action_target_offsets=source_grid-absolute_source)
        precise_current=replace(batch["current"],
            action_target_offsets=current_grid-absolute_current,
            age=replace(batch["current"].age,delta_sim_seconds=(absolute_current-absolute_source-1e-14).expand(2)))
        precise,_=model.fast_conditioner.aligned_source_plan(precise_current,precise_bridge)
        assert torch.equal(precise[:,0,[6,13]],torch.full((2,2),3.))
        checks.append("nonbinary_absolute_timestamps_keep_exact_endpoint_gripper_with_one_nanosecond_roundoff_tolerance")
        current=batch["current"]
        prediction=model.fast_conditioner.predict_current_motion(current,bridge)
        perturbed=replace(current,motion=current.motion+50)
        assert torch.equal(prediction,model.fast_conditioner.predict_current_motion(perturbed,bridge))
        other_time=replace(current,age=replace(current.age,delta_sim_seconds=torch.tensor([.3,3.])))
        assert not torch.equal(prediction,model.fast_conditioner.predict_current_motion(other_time,bridge))
        checks.append("motion_forecast_reads_only_source_and_elapsed_not_supervision_target")
        captured=[]
        hook=model.fast_conditioner.motion_innovation[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        invalid=replace(current,motion=torch.full_like(current.motion,float('nan')),
            motion_interval_valid_mask=torch.zeros_like(current.motion_interval_valid_mask),
            motion_acceleration_valid_mask=torch.zeros_like(current.motion_acceleration_valid_mask))
        model.fast_conditioner._motion_delta(invalid,bridge,torch.float32);hook.remove()
        assert torch.equal(captured[0],torch.zeros_like(captured[0]))
        null_bridge=replace(bridge,null_bridge=torch.ones(2,dtype=torch.bool))
        huge_world=replace(null_bridge,video_v=tuple(v+1000 for v in bridge.video_v))
        assert torch.equal(model.fast_conditioner.predict_current_motion(current,null_bridge),
            model.fast_conditioner.predict_current_motion(current,huge_world))
        checks.append("invalid_motion_has_zero_innovation_and_null_cache_world_values_cannot_leak_into_forecast")
        optimizer.zero_grad(set_to_none=True);model.gradient_checkpointing=False
        plain=model(batch)["native_loss"];plain.backward()
        expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True);model.gradient_checkpointing=True
        cached=model(batch)["native_loss"];cached.backward()
        assert torch.equal(plain,cached)
        assert all(torch.allclose(p.grad,expected[n],atol=1e-6,rtol=1e-5) for n,p in model.named_parameters() if n in expected)
        checks.append("checkpoint_gradient_parity_including_source_fm_and_motion_forecast")
        cloned,_=v3_fixture();full=dict(format="dynamicwam_multirate_checkpoint",version=2,model=model.state_dict())
        install_full_checkpoint(cloned,full,allow_v2_init=False)
        spec=dict(action_expert=dict(dim=24,ffn_dim=48,num_layers=2,state_dim=14,action_dim=14,chunk_size=16),wan_attention=dict(dim=12,num_heads=2,head_dim=6),motion=dict(history_count=4,mean=[0.]*12,scale=[1.]*12),fast_conditioner=model.fast_conditioner.constructor_spec)
        fast=export_fast_only_payload(model.state_dict(),module_spec=spec,global_step=2)
        modules=load_fast_only_modules(fast)
        assert fast_only_matches(model.state_dict(),fast)
        assert all(torch.equal(v,modules.fast_conditioner.state_dict()[k]) for k,v in model.fast_conditioner.state_dict().items())
        checks.append("strict_full_and_fast_reload_preserve_v3_forecast_and_time_context")
        trainer=MultirateTrainer.__new__(MultirateTrainer);trainer.config={"rtc_prefix_max":4,"native_source_bridge":"student_joint"}
        prepared=dict(batch,action_target_times=batch["current"].action_target_offsets,
            observation_sim_time=torch.zeros(2,dtype=torch.float64),
            async_timing={"rtc_prefix_lengths":torch.tensor([1,4])},rtc_sample_prefix_actions=batch["clean_actions"]+1.)
        with patch("torch.randint",side_effect=AssertionError("independent prefix draw forbidden")):
            trainer._attach_rtc_prefix(prepared)
        assert torch.equal(prepared["rtc_prefix"].lengths,torch.tensor([1,4]))
        assert torch.equal(prepared["rtc_prefix"].actions,batch["clean_actions"]+1.)
        checks.append("rtc_prefix_consumes_same_async_timing_draw_without_resampling")
        # Synthetic preparation regression, not a fabricated VAE-cache receipt.
        source,current=batch["source"],batch["current"]
        raw=dict(source_state=source.state,source_motion=source.motion,source_condition_latent=source.condition_latent,
            source_motion_interval_valid_mask=source.motion_interval_valid_mask,
            source_motion_acceleration_valid_mask=source.motion_acceleration_valid_mask,
            source_rgb=source.rgb,source_rgb_valid_mask=source.rgb_valid_mask,
            text_embeddings=source.text_embeddings,source_sim_time=absolute_source.expand(2),
            bridge_noise_shape_latent=batch["source_noise"].future_latent,
            fast_state=current.state,fast_motion=current.motion,fast_rgb=current.rgb,fast_rgb_valid_mask=current.rgb_valid_mask,
            fast_motion_interval_valid_mask=current.motion_interval_valid_mask,
            fast_motion_acceleration_valid_mask=current.motion_acceleration_valid_mask,
            fast_action_sequence=batch["clean_actions"],fast_future_latent=batch["source_noise"].future_latent,
            fast_condition_latent=source.condition_latent,state_delta=current.age.state_delta,
            delta_sim_seconds=(absolute_current-absolute_source).expand(2),delta_actions=torch.tensor([3.,3.]),
            fast_sim_time=absolute_current.expand(2),physical_grid_version=1,
            async_timing={"rtc_prefix_lengths":torch.tensor([1,4])},rtc_prefix_actions=batch["clean_actions"],
            source_rtc_prefix_actions=torch.zeros_like(batch["clean_actions"]),source_rtc_prefix_lengths=torch.zeros(2,dtype=torch.long),
            joint_targets=dict(source_action_target_times=source_grid,fast_action_target_times=current_grid,
                source_future_latent=batch["source_noise"].future_latent,source_action_sequence=batch["clean_actions"]))
        prepared=prepare_multirate_batch(raw,timesteps=torch.linspace(1000,0,20),sigmas=torch.linspace(1,0,20),
            device="cpu",dtype=torch.bfloat16,phase="cross_time_joint",null_bridge_probability=0.,sim_scale=1.,action_scale=16.,
            native_multirate=True,joint_interface_version=3)
        assert prepared["current"].age.delta_sim_seconds.dtype==torch.float64
        assert prepared["source"].sim_time.dtype==torch.float64
        assert torch.equal(prepared["current"].age.delta_sim_seconds,raw["delta_sim_seconds"])
        assert prepared["source"].action_target_offsets.dtype==torch.float64
        assert prepared["current"].action_target_offsets.dtype==torch.float64
        checks.append("bf16_training_preparation_preserves_float64_physical_timestamps_and_offsets")
        replay_raw=dict(raw,previous_plan=torch.full_like(batch["clean_actions"],.37),
            previous_plan_valid_mask=torch.ones(batch["clean_actions"].shape[:2],dtype=torch.bool),
            replay_identity=["synthetic_fixture_only"]*2)
        replay_prepared=prepare_multirate_batch(replay_raw,timesteps=torch.linspace(1000,0,20),sigmas=torch.linspace(1,0,20),
            device="cpu",dtype=torch.float32,phase="cross_time_joint",null_bridge_probability=0.,sim_scale=1.,action_scale=16.,
            native_multirate=True,joint_interface_version=3)
        trainer._attach_rtc_prefix(replay_prepared)
        with patch.object(model,"prepare_fast_memory",wraps=model.prepare_fast_memory) as routed:
            replay_output=model(replay_prepared)
        seen=routed.call_args.args[0]
        assert torch.equal(seen.previous_plan,replay_raw["previous_plan"])
        assert torch.equal(seen.committed_prefix,replay_raw["rtc_prefix_actions"])
        assert torch.isfinite(replay_output["native_loss"])
        checks.append("recorded_replay_previous_queue_and_actual_hard_prefix_reach_native_loss_without_source_substitution")
        bootstrap_raw=dict(replay_raw,delta_actions=torch.zeros(2),delta_sim_seconds=torch.ones(2,dtype=torch.float64),
            fast_sim_time=torch.full((2,),1.1,dtype=torch.float64),
            async_timing={"rtc_prefix_lengths":torch.full((2,),4,dtype=torch.long)},
            rtc_prefix_actions=source.state[:,None].expand(-1,16,-1).clone(),
            joint_targets={**replay_raw["joint_targets"],"fast_action_target_times":torch.arange(7,23,dtype=torch.float64)[None].expand(2,-1)*.16})
        bootstrap=prepare_multirate_batch(bootstrap_raw,timesteps=torch.linspace(1000,0,20),sigmas=torch.linspace(1,0,20),
            device="cpu",dtype=torch.float32,phase="cross_time_joint",null_bridge_probability=0.,sim_scale=1.,action_scale=16.,
            native_multirate=True,joint_interface_version=3)
        trainer._attach_rtc_prefix(bootstrap)
        assert bootstrap["current"].age.delta_actions.eq(0).all()
        assert bootstrap["current"].age.delta_sim_seconds.eq(1).all()
        assert bootstrap["rtc_prefix"].lengths.eq(4).all()
        assert torch.isfinite(model(bootstrap)["native_loss"])
        checks.append("synthetic_bootstrap_training_path_allows_elapsed_hold_with_zero_completed_actions_and_H4_prefix")
    print(json.dumps(dict(passed=True,checks=checks,losses=losses,device="cpu",video="tiny differentiable substitute",attention="CPU SDPA")))

if __name__=="__main__":run()
