"""CPU contracts for action-conditioned world K/V correction; no GPU launch."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.models.motion_kv_conditioner import MotionKVConditioner
from dynamicwam.models.multirate_types import BridgeAge, BridgeTensorBundle, CurrentObservation


def fixture(dtype=torch.float32):
    torch.manual_seed(7231)
    model = MotionKVConditioner(dim=24, layers=2, heads=2, head_dim=12,
        text_dim=16, state_dim=14, world_layers=(0, 1), world_heads=2,
        world_head_dim=6, world_dim=16, kv_rank=3, motion_mean=[.1] * 12,
        motion_scale=[2.] * 12).to(dtype=dtype).eval()
    b, n = 2, 7
    current = CurrentObservation(rgb=torch.randn(b, 3, 3, 32, 32),
        rgb_valid_mask=torch.ones(b, 3, 32, 32, dtype=torch.bool),
        state=torch.randn(b, 14).to(dtype), motion=torch.randn(b, 4, 12),
        motion_interval_valid_mask=torch.tensor([[True, True, False, True], [True, False, True, True]]),
        motion_acceleration_valid_mask=torch.tensor([[False, True, False, True], [True, False, False, True]]),
        text_embeddings=[], age=BridgeAge(torch.tensor([.4, 1.2]), torch.tensor([4., 12.]),
            torch.ones(b, dtype=torch.bool), torch.randn(b, 14).to(dtype)))
    bridge = BridgeTensorBundle(tuple(torch.randn(b, n, 2, 6).to(dtype) for _ in range(2)),
        tuple(torch.randn(b, n, 2, 6).to(dtype) for _ in range(2)),
        torch.tensor([n, n - 2]), {}, 0, 0., False)
    return model, current, bridge, torch.randn(b, 5, 16).to(dtype), torch.randn(b, 22, 24).to(dtype)


def active_outputs(model):
    with torch.no_grad():
        for layer in (*model.delta_k, *model.delta_v):
            layer.weight.normal_(std=.2)


def run():
    torch.set_num_threads(2)
    checks = []
    model, current, bridge, text, action = fixture()
    originals = [x.clone() for x in (*bridge.video_k, *bridge.video_v)]
    memory = model.prepare_memory(current, text, bridge=bridge)
    loss = 0
    for i in range(2):
        k, v = model.adapt_video_kv(i, action, memory, bridge.video_k[i], bridge.video_v[i])
        assert torch.equal(k, bridge.video_k[i]) and torch.equal(v, bridge.video_v[i])
        loss = loss + (k * torch.randn_like(k)).sum() + (v * torch.randn_like(v)).sum()
    loss.backward()
    assert all(p.weight.grad is not None and p.weight.grad.abs().sum() > 0 for p in (*model.delta_k, *model.delta_v))
    assert all(p.grad is not None for p in model.parameters()), "a trainable parameter is unused"
    checks.append("zero_init_exact_and_output_gradients")
    active_outputs(model)
    memory = model.prepare_memory(current, text, bridge=bridge)
    key, value = model.adapt_video_kv(0, action, memory, bridge.video_k[0], bridge.video_v[0])
    changed_action = model.adapt_video_kv(0, action + torch.randn_like(action), memory, bridge.video_k[0], bridge.video_v[0])[0]
    assert not torch.allclose(key, changed_action, atol=1e-7, rtol=1e-7)
    changed_motion = replace(current, motion=current.motion + 2.)
    motion_memory = model.prepare_memory(changed_motion, text, bridge=bridge)
    motion_key = model.adapt_video_kv(0, action, motion_memory, bridge.video_k[0], bridge.video_v[0])[0]
    assert not torch.allclose(key, motion_key, atol=1e-7, rtol=1e-7)
    checks.append("motion_and_current_action_change_cache")
    spatial = current.rgb.clone()
    spatial[:, :, :, :16, :16] += 3.
    spatial_memory = model.prepare_memory(replace(current, rgb=spatial), text, bridge=bridge)
    assert not torch.allclose(memory.rank_context[0], spatial_memory.rank_context[0])
    checks.append("spatial_rgb_tokens_are_used")
    bad_motion = current.motion.clone()
    bad_motion[~current.motion_interval_valid_mask] = float("nan")
    bad_motion[..., 9:12][~current.motion_acceleration_valid_mask] = float("inf")
    invalid_memory = model.prepare_memory(replace(current, motion=bad_motion), text, bridge=bridge)
    assert torch.equal(memory.rank_context[0], invalid_memory.rank_context[0])
    assert torch.equal(memory.assignments[0], invalid_memory.assignments[0])
    checks.append("invalid_motion_values_masked_before_normalization")
    null_bridge = replace(bridge, null_bridge=torch.ones(2, dtype=torch.bool))
    null_memory = model.prepare_memory(current, text, bridge=null_bridge)
    null_key, null_value = model.adapt_video_kv(0, action, null_memory, bridge.video_k[0], bridge.video_v[0])
    assert torch.equal(null_key, bridge.video_k[0]) and torch.equal(null_value, bridge.video_v[0])
    perturbed = replace(null_bridge, video_k=tuple(x + 50. for x in bridge.video_k),
                         video_v=tuple(x - 80. for x in bridge.video_v))
    other_null = model.prepare_memory(current, text, bridge=perturbed)
    assert torch.equal(null_memory.rank_context[0], other_null.rank_context[0])
    assert not bool(null_memory.assignments[0].any())
    checks.append("null_bridge_drops_old_context_and_correction")
    for output, base in ((key, bridge.video_k[0]), (value, bridge.video_v[0])):
        bound = base.float().square().mean(-1).sqrt() * model.max_relative_update
        measured = (output.float() - base.float()).square().mean(-1).sqrt()
        assert bool((measured <= bound + 2e-7).all()) and bool(torch.isfinite(output).all())
        assert torch.equal(output[1, 5:], base[1, 5:])
    assert all(torch.equal(a, b) for a, b in zip(originals, (*bridge.video_k, *bridge.video_v)))
    checks.append("bounded_finite_padding_and_no_cache_mutation")
    key_snapshot = key.detach().clone()
    model.adapt_video_kv(0, -action, memory, bridge.video_k[0], bridge.video_v[0])
    assert torch.equal(key, key_snapshot)
    checks.append("returned_outputs_have_independent_storage")
    restored, *_ = fixture()
    restored.motion_mean.zero_()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.motion_mean, model.motion_mean)
    checks.append("strict_checkpoint_restores_motion_statistics")
    for dtype in (torch.bfloat16, torch.float16):
        half_model, half_current, half_bridge, half_text, half_action = fixture(dtype)
        active_outputs(half_model)
        half_memory = half_model.prepare_memory(half_current, half_text, bridge=half_bridge)
        hk, hv = half_model.adapt_video_kv(0, half_action, half_memory,
                                          half_bridge.video_k[0], half_bridge.video_v[0])
        assert hk.dtype == dtype and hv.dtype == dtype and bool(torch.isfinite(hk).all() & torch.isfinite(hv).all())
    checks.append("fp16_bf16_finite_and_dtype_preserved")
    print({"passed": True, "checks": checks, "count": len(checks)})


if __name__ == "__main__":
    run()
