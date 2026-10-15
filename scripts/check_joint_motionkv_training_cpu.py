"""CPU integration of attached slow bridge, V2 innovations, native actor and FM.

The tiny differentiable video projection replaces the large Video DiT and CPU
SDPA replaces FlashAttention. This verifies graph connectivity and contracts;
it does not claim real Video DiT memory fit or GPU runtime validation.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import types
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_motionkv_training_cpu import fixture, cpu_attention
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable, multirate_optimizer_groups, migrate_joint_interface_initialization, MultirateTrainer
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import install_full_checkpoint, export_fast_only_payload, load_fast_only_modules, fast_only_matches


def joint_fixture():
    model, current, old_bridge, text, batch = fixture()
    current = replace(current, action_target_offsets=(torch.arange(1,17)[None].expand(2,-1)*.16).double())
    batch["current"] = current
    model.fast_conditioner = JointMotionKVConditioner(dim=24, layers=2, heads=2,
        head_dim=12, text_dim=16, state_dim=14, world_layers=(0,1),
        world_heads=2, world_head_dim=6, world_dim=16, kv_rank=3)
    # Emulate the nonzero trained V1 correction heads before V2 migration.
    for projection in (*model.fast_conditioner.delta_k, *model.fast_conditioner.delta_v):
        nn.init.normal_(projection.weight, std=.02)
    if hasattr(model.fast_conditioner, "innovation_gate"):
        with torch.no_grad(): model.fast_conditioner.innovation_gate.fill_(.1)
    core = model.backbone
    core.compact_wan = nn.Module()
    core.compact_wan.video_model = nn.Module()
    core.compact_wan.video_model.precision = torch.float32
    core.compact_wan.video_model.wan_model = nn.Linear(12,12)
    def joint(self, packet, *, include_video_prediction, build_video_cache, detach_cache=True):
        future, actions = packet["future_latent"], packet["noisy_actions"]
        b = future.shape[0]
        tokens = future.reshape(b,1,12).expand(b,7,12) + actions.mean((1,2))[:,None,None] * .1
        tokens = self.compact_wan.video_model.wan_model(tokens)
        keys = tuple((tokens * (i+1)).reshape(b,7,2,6) for i in range(2))
        values = tuple((tokens * (i+2)).reshape(b,7,2,6) for i in range(2))
        cache = dict(video_k=keys, video_v=values, seq_lens=torch.full((b,),7), grid_sizes={})
        action = self.forward_action_with_video_cache(dict(packet,video_cache=cache))["action_pred"]
        result = {"action_pred":action}
        if include_video_prediction:
            result["video_pred"] = tokens.mean(1).reshape_as(future)
        if build_video_cache:
            if detach_cache:
                cache = dict(cache, video_k=tuple(t.detach() for t in keys),video_v=tuple(t.detach() for t in values))
            result["video_cache"] = cache
        return result
    core._forward_video_action = types.MethodType(joint, core)
    model.native_source_bridge = "student_joint"
    model.gradient_checkpointing = True
    model.joint_action_weight = .1
    model.joint_video_weight = .1
    b = current.state.shape[0]
    source = SourceObservation(torch.randn(b,12,1,1,1), current.state.detach().clone(),
        current.motion.detach().clone(), current.motion_interval_valid_mask,
        current.motion_acceleration_valid_mask,list(text),0,0.,
        rgb=current.rgb.detach().clone()*.95,rgb_valid_mask=current.rgb_valid_mask)
    noise = SourceNoise(torch.randn(b,12,1,1,1),torch.randn(b,16,14))
    aligned = dict(source.as_model_batch(), future_latent=torch.randn_like(noise.future_latent),
        noisy_actions=batch["fast_noisy_actions"], action_t=batch["fast_action_t"],video_t=batch["fast_action_t"])
    batch.update(source=source,source_noise=noise,aligned=aligned,
        action_target=torch.randn_like(batch["clean_actions"]),video_target=torch.randn_like(noise.future_latent))
    batch.pop("native_source_bridge")
    return model,batch


def run():
    torch.set_num_threads(2)
    results=[]
    config=dict(action_learning_rate=1e-3,fast_learning_rate=1e-3,video_learning_rate=1e-4,weight_decay=0.)
    with patch("dynamicwam.models.action_core_ops.flash_attention",cpu_attention), patch("dynamicwam.models.action_core_ops.dense_action_attention",cpu_attention):
        model,batch=joint_fixture()
        configure_multirate_trainable(model,"warmup")
        groups=multirate_optimizer_groups(model,config,"warmup")
        assert [g["name"] for g in groups]==["action_core","fast_conditioner"]
        model(batch)["native_loss"].backward()
        assert all(p.grad is None for p in model.backbone.compact_wan.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.fast_conditioner.plan_innovation.parameters())>0
        results.append("warmup_frozen_video_with_trainable_actual_source_plan_interface")
        model.zero_grad(set_to_none=True)
        configure_multirate_trainable(model,"joint")
        groups=multirate_optimizer_groups(model,config,"joint")
        assert [g["name"] for g in groups]==["action_core","fast_conditioner","video_expert"]
        # No video/action preservation anchor: slow gradient must come via cache.
        model.joint_video_weight=model.joint_action_weight=0.
        output=model(batch)
        output["native_loss"].backward()
        video=list(model.backbone.compact_wan.video_model.wan_model.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in video)
        assert sum(float(p.grad.abs().sum()) for p in video)>0
        for name in ("visual_innovation","motion_innovation","plan_innovation"):
            parameters=list(getattr(model.fast_conditioner,name).parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),name
            assert sum(float(p.grad.abs().sum()) for p in parameters)>0,name
        results.append("suffix_loss_reaches_slow_video_through_real_two_pass_bridge_and_all_innovations")
        before=[p.detach().clone() for p in video]
        optimizer=torch.optim.AdamW(groups)
        optimizer.step()
        assert any(not torch.equal(old,p) for old,p in zip(before,video))
        optimizer.zero_grad(set_to_none=True)
        model.joint_video_weight=model.joint_action_weight=.1
        output=model(batch)
        assert "loss/video_retention" in output["native_metrics"] and "loss/action_retention" in output["native_metrics"]
        output["native_loss"].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        results.append("joint_optimizer_updates_video_and_native_fm_retention_is_finite")
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing=False
        plain=model(batch)["native_loss"]
        plain.backward()
        expected={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing=True
        captured=model(batch)["native_loss"]
        captured.backward()
        assert torch.equal(plain,captured)
        assert all(torch.allclose(p.grad,expected[n],atol=1e-6,rtol=1e-5) for n,p in model.named_parameters() if n in expected)
        results.append("nonreentrant_checkpoint_retains_nested_bridge_outputs_and_gradients")
        state=model.state_dict()
        prefixes=("fast_conditioner.visual_innovation.","fast_conditioner.motion_innovation.","fast_conditioner.plan_innovation.","fast_conditioner.innovation_kind","fast_conditioner.innovation_gate")
        old={k:v.clone() for k,v in state.items() if not k.startswith(prefixes)}
        payload=dict(format="dynamicwam_multirate_checkpoint",model=old,config={"model":{"fast_conditioner":{"motion_kv":True}}})
        filled,missing=migrate_joint_interface_initialization(model,payload)
        assert missing and all(torch.equal(old[k],filled["model"][k]) for k in old)
        bad=copy.deepcopy(payload); bad["model"].pop(next(k for k in old if k.startswith("backbone.")))
        try: migrate_joint_interface_initialization(model,bad)
        except RuntimeError: pass
        else: raise AssertionError("noninterface missingweight accepted")
        results.append("explicit_v1_migration_preserves_every_existing_weight_and_rejects_missing_backbone")
        reloaded,_=joint_fixture()
        full=dict(format="dynamicwam_multirate_checkpoint",version=2,model=model.state_dict())
        install_full_checkpoint(reloaded,full,allow_v2_init=False)
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in full["model"].items())
        spec=dict(action_expert=dict(dim=24,ffn_dim=48,num_layers=2,state_dim=14,action_dim=14,chunk_size=16),
            wan_attention=dict(dim=12,num_heads=2,head_dim=6),
            motion=dict(history_count=4,mean=[0.]*12,scale=[1.]*12),
            fast_conditioner=model.fast_conditioner.constructor_spec)
        fast=export_fast_only_payload(model.state_dict(),module_spec=spec,global_step=2)
        modules=load_fast_only_modules(fast)
        assert fast_only_matches(model.state_dict(),fast)
        assert all(torch.equal(v,modules.fast_conditioner.state_dict()[k]) for k,v in model.fast_conditioner.state_dict().items())
        assert all(torch.equal(v,modules.action_expert.state_dict()[k]) for k,v in model.backbone.action_expert.state_dict().items())
        results.append("strict_full_and_fast_only_reload_preserves_all_student_parameters")
        trainer=MultirateTrainer.__new__(MultirateTrainer)
        trainer.config={"rtc_prefix_max":4,"native_source_bridge":"student_joint"}
        prepared=dict(batch,action_target_times=torch.tensor([[2.+.13*(i+1) for i in range(16)], [5.+.19*(i+1) for i in range(16)]],dtype=torch.float64),observation_sim_time=torch.tensor([2.,5.],dtype=torch.float64))
        prepared["current"] = replace(prepared["current"], action_target_offsets=prepared["action_target_times"]-prepared["observation_sim_time"][:,None])
        with patch("torch.randint",return_value=torch.tensor([0,4])):
            trainer._attach_rtc_prefix(prepared)
        assert torch.allclose(prepared["current"].observation_delay_seconds,torch.tensor([0.,.76],dtype=torch.float64))
        assert torch.equal(prepared["rtc_prefix"].actions,prepared["clean_actions"])
        assert torch.equal(prepared["current"].action_target_offsets,
            prepared["action_target_times"]-prepared["observation_sim_time"][:,None])
        results.append("prefix_reservation_and_all_action_offsets_use_exact_endpoint_times_including_d0")
    root=Path(__file__).resolve().parents[1]
    for phase in ("warmup","joint"):
        for suffix in ("","_smoke"):
            p=load_multirate_profile(root/f"configs/multirate/joint_motionkv_{phase}{suffix}.yaml").training
            assert p["native_source_bridge"]=="student_joint" and p["rtc_prefix_max"]==4
            assert p["dataset"]["cache_age_jitter"] and p["dataset"]["cache_age_actions"]==list(range(16))
    old=load_multirate_profile(root/"configs/multirate/motionkv_rtc_adapt_8gpu.yaml").training
    assert old["native_source_bridge"]=="immutable_official" and "interface_version" not in old["model"]["fast_conditioner"]
    results.append("four_stage_profiles_and_legacy_profile_parse_with_h4_k10_contract")
    print(json.dumps({"passed":True,"checks":results,"device":"cpu","video":"tiny differentiable substitute","attention":"CPU SDPA"}))

if __name__=="__main__": run()
