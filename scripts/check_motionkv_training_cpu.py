"""Tiny integrated training check with real native action/motion modules.

Only FlashAttention dispatch is replaced by CPU SDPA. This executes the actual
ActionCoreOps, ActionExpert, MotionKVConditioner, MultirateWAM adaptation loss,
RTC prefix times, backward, and two optimizer updates. It does not instantiate
Video DiT, produce teacher trajectories, or establish GPU/closed-loop quality.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.inference.fast_runtime import StandaloneActionCore
from dynamicwam.models.action_expert import ActionExpert, ActionExpertConfig
from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.models.motion_kv_conditioner import MotionKVConditioner
from dynamicwam.models.multirate_types import BridgeAge, BridgeTensorBundle, CurrentObservation
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.models.rtc_prefix import PrefixCondition, action_token_times, validate_prefix_condition


def cpu_attention(q, k, v, *, k_lens=None, window_size=(-1, -1), **unused):
    assert q.device.type == k.device.type == v.device.type == "cpu"
    assert tuple(window_size) == (-1, -1)
    mask = None
    if k_lens is not None:
        mask = (torch.arange(k.shape[1])[None] < k_lens[:, None])[:, None, None, :]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
        v.transpose(1, 2), attn_mask=mask).transpose(1, 2)


def fixture():
    torch.manual_seed(613)
    cfg = ActionExpertConfig(dim=24, ffn_dim=48, num_layers=2,
                             state_dim=14, action_dim=14, chunk_size=16)
    action = ActionExpert(cfg, dict(dim=12, num_heads=2, head_dim=6))
    # Official loaded weights have a trained decoder; a scratch ActionExpert
    # zeros this head and would make all upstream first-step gradients zero.
    with torch.no_grad():
        action.decoder.action_head[0].weight.normal_(std=.03)
    motion = AbsoluteMotionTokenModule(dim=24, history_count=4,
                                      feature_mean=(0.,) * 12, feature_scale=(1.,) * 12)
    core = StandaloneActionCore(action, motion,
        dict(wan_attention=dict(precision="float32", window_sizes=[(-1, -1)] * 2)))
    conditioner = MotionKVConditioner(dim=24, layers=2, heads=2, head_dim=12,
        text_dim=16, state_dim=14, world_layers=(0, 1), world_heads=2,
        world_head_dim=6, world_dim=16, kv_rank=3)
    model = MultirateWAM(core, conditioner, native_multirate=True,
                         fast_steps=10, native_teacher_weight=.5)
    model.native_phase = "adapt"
    model.external_native_teacher = True
    model.rtc_prefix_max = 6
    model.train()
    b, n = 2, 7
    text = torch.randn(b, 5, 16)
    current = CurrentObservation(rgb=torch.randn(b, 3, 3, 32, 32),
        rgb_valid_mask=torch.ones(b, 3, 32, 32, dtype=torch.bool),
        state=torch.randn(b, 14), motion=torch.randn(b, 4, 12, requires_grad=True),
        motion_interval_valid_mask=torch.tensor([[True, True, False, True], [True, False, True, True]]),
        motion_acceleration_valid_mask=torch.tensor([[False, True, False, True], [True, False, False, True]]),
        text_embeddings=list(text), age=BridgeAge(torch.tensor([.4, 1.2]), torch.tensor([4., 12.]),
            torch.ones(b, dtype=torch.bool), torch.randn(b, 14)))
    bridge = BridgeTensorBundle(tuple(torch.randn(b, n, 2, 6) for _ in range(2)),
        tuple(torch.randn(b, n, 2, 6) for _ in range(2)), torch.tensor([n, n - 2]), {}, 0, 0., False)
    clean, noise = torch.randn(b, 16, 14), torch.randn(b, 16, 14)
    sigma = torch.tensor([.3, .6]).reshape(b, 1, 1)
    noisy = ((1 - sigma) * clean + sigma * noise).requires_grad_()
    prefix = PrefixCondition(clean.detach().clone(), torch.tensor([0, 6]))
    validate_prefix_condition(prefix, max_prefix=6, reference=noisy)
    batch = dict(native_source_bridge=bridge, current=current, fast_text=text,
        fast_text_mask=torch.ones(b, 5, dtype=torch.bool), null_bridge=torch.zeros(b, dtype=torch.bool),
        clean_actions=clean, fast_noisy_actions=noisy, fast_action_t=torch.tensor([300., 600.]),
        fast_target=noise - clean, native_teacher_actions=torch.randn_like(clean),
        teacher_flow_noise=torch.randn_like(clean), fast_sigma=sigma, rtc_prefix=prefix)
    return model, current, bridge, text, batch


def nonzero_finite_grad(parameters):
    gradients = [p.grad for p in parameters]
    return (all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
            and sum(float(g.abs().sum()) for g in gradients) > 0.)


def run():
    torch.set_num_threads(2)
    checks, loss_values = [], []
    with patch("dynamicwam.models.action_core_ops.flash_attention", cpu_attention), \
         patch("dynamicwam.models.action_core_ops.dense_action_attention", cpu_attention):
        model, current, bridge, text, batch = fixture()
        original_cache = tuple(t.clone() for t in (*bridge.video_k, *bridge.video_v))
        memory = model.prepare_fast_memory(current, bridge, text, batch["fast_text_mask"])
        native_batch = dict(video_cache=bridge.as_video_cache(),
            noisy_actions=batch["fast_noisy_actions"], action_t=batch["fast_action_t"],
            initial_state=current.state, absolute_motion_features=current.motion,
            absolute_motion_interval_valid_mask=current.motion_interval_valid_mask,
            absolute_motion_acceleration_valid_mask=current.motion_acceleration_valid_mask)
        original = model.backbone.forward_action_with_video_cache(native_batch)["action_pred"]
        d0 = PrefixCondition(batch["clean_actions"], torch.zeros(2, dtype=torch.long))
        actual = model.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                                             current, bridge, memory, prefix=d0)
        assert torch.equal(actual, original), "zero-init d0 changed native prediction"
        checks.append("zero_init_d0_exact_real_native_action_output")
        core_tokens, _ = model.backbone._build_action_tokens(native_batch,
            batch["fast_noisy_actions"], current.state)
        assert core_tokens.shape[1] == 1 + 16 + 4 + 4
        seen_times = []
        build_time = model.backbone._build_action_time_embeddings
        def record_time(times, sequence_length):
            seen_times.append(times.detach().clone())
            return build_time(times, sequence_length)
        with patch.object(model.backbone, "_build_action_time_embeddings", side_effect=record_time):
            model.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                                        current, bridge, memory, prefix=batch["rtc_prefix"])
        expected = action_token_times(batch["fast_action_t"], batch["rtc_prefix"], core_tokens.shape[1])
        assert torch.equal(seen_times[0], expected)
        assert bool((seen_times[0][1, 1:7] == 0).all())
        assert bool((seen_times[0][1, 17:21] == 600).all()), "native motion tokens were retimed"
        checks.append("prefix_per_token_times_reach_real_action_core_and_motion_is_retained")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        first_delta = model.fast_conditioner.delta_v[0].weight.detach().clone()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            current.motion.grad = None
            batch["fast_noisy_actions"].grad = None
            outputs = model(batch, mode="multirate_train")
            loss = outputs["native_loss"]
            assert bool(torch.isfinite(loss)), "nonfinite integrated loss"
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                       for p in model.fast_conditioner.parameters()), "unused/nonfinite new-module parameter"
            assert nonzero_finite_grad(model.backbone.action_expert.parameters())
            assert nonzero_finite_grad(model.backbone.absolute_motion_tokens.parameters())
            assert current.motion.grad is not None and bool(torch.isfinite(current.motion.grad).all())
            assert current.motion.grad.abs().sum() > 0
            assert batch["fast_noisy_actions"].grad is not None
            assert bool(torch.isfinite(batch["fast_noisy_actions"].grad).all())
            assert batch["fast_noisy_actions"].grad[:, 6:].abs().sum() > 0
            assert torch.equal(batch["fast_noisy_actions"].grad[1, :6],
                               torch.zeros_like(batch["fast_noisy_actions"].grad[1, :6]))
            if step == 1:
                assert nonzero_finite_grad(model.fast_conditioner.motion_encoder.parameters())
                assert nonzero_finite_grad(model.fast_conditioner.action_k.parameters())
                assert nonzero_finite_grad(model.fast_conditioner.action_v.parameters())
            loss_values.append(float(loss.detach()))
            optimizer.step()
        assert not torch.equal(first_delta, model.fast_conditioner.delta_v[0].weight)
        checks.append("two_optimizer_updates_finite_with_action_motion_and_all_new_parameter_gradients")
        checks.append("committed_prefix_has_zero_noisy_input_gradient")

        model.eval()
        with torch.no_grad():
            memory = model.prepare_fast_memory(current, bridge, text, batch["fast_text_mask"])
            actions = model.sample_fast_actions(current, bridge, memory,
                torch.randn_like(batch["clean_actions"]), prefix=batch["rtc_prefix"])
            assert torch.equal(actions[1, :6], batch["rtc_prefix"].actions[1, :6])
            assert bool(torch.isfinite(actions).all())
            checks.append("actual_ten_step_solver_preserves_committed_prefix")
            keys, values = [x.clone() for x in bridge.video_k], [x.clone() for x in bridge.video_v]
            for key, value in zip(keys, values):
                key[1, 5:] += 500.
                value[1, 5:] -= 800.
            padded = replace(bridge, video_k=tuple(keys), video_v=tuple(values))
            padding_memory = model.prepare_fast_memory(current, padded, text, batch["fast_text_mask"])
            before = model.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                current, bridge, memory, prefix=batch["rtc_prefix"])
            after = model.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                current, padded, padding_memory, prefix=batch["rtc_prefix"])
            assert torch.equal(before, after), "padded world tokens affected native joint attention"
            checks.append("padded_world_cache_is_excluded_in_real_joint_attention")
        assert all(torch.equal(old, new) for old, new in zip(original_cache,
            (*bridge.video_k, *bridge.video_v)))
        checks.append("world_cache_unchanged_after_training_and_sampling")
    print(json.dumps(dict(passed=True, checks=checks, optimizer_steps=2,
        losses=loss_values, device="cpu", attention_backend="SDPA replacement of FlashAttention only")))


if __name__ == "__main__":
    run()
