"""Causal V4 model checks with native Action and tiny native WAN blocks.

The projection geometry is reduced for CPU; these tests do not claim full-size
GPU fit, BF16 kernel safety, deadline performance or task quality.
"""
from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from torch import nn
from torch.nn import functional as F
from dynamicwam.models.causal_spatial import CausalSpatialModel,CausalSpatialConfig
from dynamicwam.models.small_wam import SmallWAMActionModel,SmallWAMActionConfig
from dynamicwam.models.rtc_prefix import PrefixCondition
from dynamicwam.vendor.wan.modules.model import WanAttentionBlock


def cpu_attention(q,k,v,**kwargs):
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)


class TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = torch.float32
        self.wan_model = nn.Module()
        self.wan_model.patch_size = (1,2,2)
        self.wan_model.patch_embedding = nn.Conv3d(4,12,(1,2,2),stride=(1,2,2))
        self.wan_model.blocks = nn.ModuleList(WanAttentionBlock(12,24,2) for _ in range(2))

    def _patch_video_latent(self,x):
        p = self.wan_model.patch_embedding(x)
        return p.flatten(2).transpose(1,2),torch.tensor(p.shape[2:])[None].expand(x.shape[0],-1)


class TinyCompact(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_model = TinyWrapper()
        self.text = nn.Linear(16,12)
        self.time = nn.Linear(1,12)
        self.mod = nn.Linear(12,72)

    def prepare_text_context(self,embeddings):
        x = torch.stack(embeddings) if not torch.is_tensor(embeddings) else embeddings
        return self.text(x)

    def prepare_time_embeddings(self,t,n):
        if t.ndim == 1:
            t = t[:,None].expand(-1,n)
        e = self.time(t[...,None]/1000)
        return e,self.mod(e).reshape(t.shape[0],n,6,12)


def fixture(checkpointing=False):
    torch.manual_seed(901)
    native = SmallWAMActionConfig(SimpleNamespace(dim=12,num_heads=2,num_layers=2,head_dim=6),
        14,14,16,24,48,2,False,4,(0.,)*12,(1.,)*12)
    backbone = SmallWAMActionModel(native,TinyCompact())
    nn.init.normal_(backbone.action_expert.decoder.action_head[0].weight,std=.05)
    cfg = CausalSpatialConfig(z_dim=24,belief_dim=16,belief_grid=(2,2),latent_channels=4,
        history_length=2,gradient_checkpointing=checkpointing)
    model = CausalSpatialModel(backbone,cfg)
    b = 2
    actions = torch.randn(b,16,14)
    batch = dict(source_condition_latent=torch.randn(b,4,2,8,8),
        source_text_embeddings=torch.randn(b,5,16),source_time=torch.tensor([0.,.2],dtype=torch.float64),
        source_id=torch.tensor([1,2]),observation_age=torch.tensor([.16,.32]),
        spatial_rgb=torch.randn(b,2,3,3,16,16),spatial_flow=torch.randn(b,2,3,2,16,16),
        spatial_valid=torch.ones(b,2,3,1,16,16,dtype=torch.bool),
        spatial_flow_valid=torch.ones(b,2,3,1,16,16,dtype=torch.bool),
        spatial_dt=torch.tensor([[0.,.16],[0.,.24]]),recorded_capture_commands=torch.randn(b,2,14),
        command_valid=torch.ones(b,2,dtype=torch.bool),initial_state=torch.randn(b,14),
        absolute_motion_features=torch.randn(b,4,12),
        absolute_motion_interval_valid_mask=torch.ones(b,4,dtype=torch.bool),
        absolute_motion_acceleration_valid_mask=torch.ones(b,4,dtype=torch.bool),
        action_target_offsets=torch.arange(1,17)[None].expand(b,-1).double()*.16,
        future_target_offsets=torch.tensor([[.64,1.92],[.64,1.92]],dtype=torch.float64),
        future_latent=torch.randn(b,4,2,4,4),noisy_actions=actions,
        action_t=torch.tensor([500.,250.]),video_t=torch.tensor([500.,250.]),
        rtc_prefix=PrefixCondition(torch.randn_like(actions),torch.tensor([2,2])))
    batch["spatial_flow_valid"][:,0] = False
    batch["command_valid"][:,0] = False
    return model,batch


class CausalSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.patch = patch("dynamicwam.vendor.wan.modules.model.flash_attention",cpu_attention)
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def test_native_geometry_and_bidirectional_gradients(self):
        model,batch = fixture()
        a = batch["noisy_actions"].clone().requires_grad_()
        z = batch["future_latent"].clone().requires_grad_()
        out = model(dict(batch,noisy_actions=a,future_latent=z))
        self.assertEqual(out["action_pred"].shape,a.shape)
        self.assertEqual(out["video_pred"].shape,z.shape)
        da = torch.autograd.grad(out["video_pred"].square().mean(),a,retain_graph=True)[0]
        dz = torch.autograd.grad(out["action_pred"].square().mean(),z,retain_graph=True)[0]
        self.assertGreater(float(da[:,2:].abs().sum()),0.)
        self.assertGreater(float(dz.abs().sum()),0.)
        self.assertEqual(float(da[:,:2].abs().sum()),0.)
        out["action_pred"].square().mean().backward()
        motion_grad = sum(float(p.grad.abs().sum()) for p in model.backbone.absolute_motion_tokens.parameters() if p.grad is not None)
        self.assertGreater(motion_grad,0.)

    def test_memory_has_no_candidate_or_future_input_and_fast_reuses_it(self):
        model,batch = fixture()
        source = batch["source_condition_latent"].requires_grad_()
        a = batch["noisy_actions"].requires_grad_()
        z = batch["future_latent"].requires_grad_()
        memory = model.encode_memory(source,batch["source_text_embeddings"],batch["source_time"])
        grads = torch.autograd.grad(sum(x.sum() for x in memory.values),(source,a,z),allow_unused=True)
        self.assertIsNotNone(grads[0]); self.assertIsNone(grads[1]); self.assertIsNone(grads[2])
        frozen = memory.detached()
        saved = [x.clone() for x in frozen.keys+frozen.values]
        with patch.object(model,"encode_memory",side_effect=AssertionError("fast reran slow encoder")):
            model(batch,frozen)
            model(dict(batch,noisy_actions=a+1,future_latent=z-1),frozen)
        for before,after in zip(saved,frozen.keys+frozen.values):
            self.assertTrue(torch.equal(before,after))

    def test_belief_predicts_before_current_observation(self):
        model,batch = fixture()
        original = model.prepare_belief(batch)
        rgb = batch["spatial_rgb"].clone();rgb[:,-1] += 2
        altered = model.prepare_belief(dict(batch,spatial_rgb=rgb))
        self.assertTrue(torch.equal(original["prediction"][:,-1],altered["prediction"][:,-1]))
        self.assertFalse(torch.equal(original["target"][:,-1],altered["target"][:,-1]))
        self.assertFalse(torch.equal(original["tokens"],altered["tokens"]))
        self.assertFalse(bool(original["prediction_valid"][:,0].any()))

    def test_invalid_flow_and_previous_plan_cannot_leak(self):
        model,batch = fixture()
        flow_valid = torch.zeros_like(batch["spatial_flow_valid"])
        previous_valid = torch.zeros_like(batch["noisy_actions"][...,0],dtype=torch.bool)
        one = dict(batch,spatial_flow_valid=flow_valid,previous_plan=batch["noisy_actions"],previous_plan_valid=previous_valid)
        two = dict(one,spatial_flow=torch.full_like(batch["spatial_flow"],float("nan")),
                   previous_plan=torch.full_like(batch["noisy_actions"],float("nan")))
        a,b = model.prepare_belief(one),model.prepare_belief(two)
        self.assertTrue(torch.equal(a["tokens"],b["tokens"]))
        self.assertTrue(torch.equal(a["target"],b["target"]))
        self.assertFalse(bool(a["prediction_valid"][...,3:].any()))
        with torch.no_grad(): model.plan_gate.fill_(.5)
        pa,pb = model.prepare_plan(one),model.prepare_plan(two)
        self.assertTrue(all(torch.equal(pa[key],pb[key]) for key in pa))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in pb.values()))

    def test_plan_zero_gate_does_not_add_ghost_attention_tokens(self):
        model,batch = fixture()
        plain = model.prepare_belief(batch)
        conditioned = dict(batch,previous_plan=batch["noisy_actions"],
            previous_plan_valid=torch.ones(2,16,dtype=torch.bool))
        prior = model.prepare_belief(conditioned)
        self.assertTrue(torch.equal(plain["tokens"],prior["tokens"]))
        absent,zero = model(batch),model(conditioned)
        self.assertTrue(torch.equal(absent["action_pred"],zero["action_pred"]))
        self.assertTrue(torch.equal(absent["video_pred"],zero["video_pred"]))
        with torch.no_grad():model.plan_gate.fill_(.5)
        changed = model.prepare_belief(conditioned)
        self.assertTrue(torch.equal(plain["tokens"],changed["tokens"]))
        for before,after in zip(plain["kv"],changed["kv"]):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        altered = model(conditioned)
        self.assertFalse(torch.equal(absent["action_pred"],altered["action_pred"]))
        self.assertFalse(torch.equal(absent["video_pred"],altered["video_pred"]))
        changed_tail = conditioned["previous_plan"].clone();changed_tail[:,4:] += 999
        after_tail = model(dict(conditioned,previous_plan=changed_tail))
        self.assertTrue(torch.equal(altered["action_pred"],after_tail["action_pred"]))

    def test_checkpointed_gradients_match(self):
        eager,batch = fixture(False)
        checked = copy.deepcopy(eager)
        checked.config = replace(checked.config,gradient_checkpointing=True)
        outputs = []
        for model in (eager,checked):
            out = model(batch)
            loss = out["action_pred"].square().mean()+out["video_pred"].square().mean()
            loss.backward();outputs.append(out)
        self.assertTrue(torch.allclose(outputs[0]["action_pred"],outputs[1]["action_pred"],atol=1e-6,rtol=1e-5))
        for (name,a),(other,b) in zip(eager.named_parameters(),checked.named_parameters()):
            self.assertEqual(name,other)
            self.assertEqual(a.grad is None,b.grad is None,name)
            if a.grad is not None:
                self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-5,rtol=1e-4),name)

    def test_fast_only_strict_roundtrip_and_k10_prefix(self):
        model,batch = fixture()
        model.eval()
        memory = model.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"]).detached()
        fast = model.make_fast_model().eval()
        self.assertFalse(any("compact_wan" in k for k in fast.state_dict()))
        fast.load_state_dict(model.fast_state_dict(),strict=True)
        with torch.no_grad():
            a,b = model(batch,memory),fast(batch,memory)
            self.assertTrue(torch.equal(a["action_pred"],b["action_pred"]))
            self.assertTrue(torch.equal(a["video_pred"],b["video_pred"]))
            with patch.object(fast,"forward",wraps=fast.forward) as calls:
                sampled = fast.sample(batch,action_noise=batch["noisy_actions"],
                    future_noise=batch["future_latent"],sigmas=torch.linspace(1,0,11),memory=memory)
            self.assertEqual(calls.call_count,10)
            for call in calls.call_args_list:
                packet = call.args[0]
                self.assertTrue(torch.equal(packet["noisy_actions"][:,:2],batch["rtc_prefix"].actions[:,:2]))
                self.assertTrue(torch.equal(packet["action_t"],packet["video_t"]))
            self.assertTrue(torch.equal(sampled["actions"][:,:2],batch["rtc_prefix"].actions[:,:2]))
        with self.assertRaises(RuntimeError):
            fast.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"])

    def test_physical_geometry_and_version_reject(self):
        model,batch = fixture()
        memory = model.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"])
        with self.assertRaises(ValueError):
            model(batch,replace(memory,model_version="wrong"))
        with self.assertRaises(ValueError):
            model(dict(batch,future_target_offsets=torch.zeros(2,3)),memory)

    def test_sampling_validates_once_and_restores_native_training_checks(self):
        model,batch = fixture()
        model.eval()
        memory = model.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"])
        args = dict(action_noise=batch["noisy_actions"],future_noise=batch["future_latent"],
                    sigmas=torch.linspace(1,0,11),memory=memory)
        motion = model.backbone.absolute_motion_tokens
        with patch.object(motion,"validate_inputs",wraps=motion.validate_inputs) as check:
            eager = model.sample(batch,**args)
            self.assertEqual(check.call_count,1)
            check.reset_mock()
            captured = model.sample(batch,**args,validate_input=False)
            self.assertEqual(check.call_count,0)
        self.assertTrue(torch.equal(eager["actions"],captured["actions"]))
        self.assertTrue(torch.equal(eager["future_latent"],captured["future_latent"]))
        self.assertFalse(model.backbone._motion_inputs_prevalidated)
        with patch.object(model,"_sample_prevalidated",side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError,"injected"):
                model.sample(batch,**args,validate_input=False)
        self.assertFalse(model.backbone._motion_inputs_prevalidated)
        invalid = dict(batch,absolute_motion_interval_valid_mask=torch.zeros_like(batch["absolute_motion_interval_valid_mask"]))
        with self.assertRaisesRegex(ValueError,"acceleration"):
            model.sample(invalid,**args)
        model.train()
        with self.assertRaisesRegex(ValueError,"eval Graph"):
            model.sample(batch,**args,validate_input=False)
        with self.assertRaisesRegex(ValueError,"acceleration"):
            model(invalid,memory)

    def test_real_native_wrapper_bf16_observation_path(self):
        from dynamicwam.training.models.wan_model import WanVideoModel
        from dynamicwam.training.models.compact_wan import CompactWANModel
        # Actual wrapper/native blocks, reduced width only; no teacher download.
        wrapper = WanVideoModel(dict(model_type="t2v",patch_size=(1,2,2),text_len=5,
            in_dim=48,out_dim=48,dim=12,ffn_dim=24,freq_dim=16,text_dim=16,
            num_heads=2,num_layers=2),device="cpu",precision="bfloat16")
        model,batch = fixture()
        model.backbone.compact_wan = CompactWANModel(model.backbone.config.compact_wan,wrapper)
        memory = model.encode_memory(torch.randn(2,48,2,8,8),batch["source_text_embeddings"])
        self.assertEqual(memory.keys[0].shape,(2,32,2,6))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in memory.keys+memory.values))


if __name__ == "__main__":
    unittest.main()
