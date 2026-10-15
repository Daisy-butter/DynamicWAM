"""CPU-only V2 innovation, source-gradient and checkpoint contract smoke."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise
from check_motion_kv_conditioner import fixture, active_outputs


def joint_fixture():
    old, current, bridge, text, action = fixture()
    model = JointMotionKVConditioner(**old.constructor_spec)
    result = model.load_state_dict(old.state_dict(), strict=False)
    assert not result.unexpected_keys
    allowed = ("visual_innovation.", "motion_innovation.", "plan_innovation.", "innovation_kind", "innovation_gate")
    assert result.missing_keys and all(x.startswith(allowed) for x in result.missing_keys)
    current = replace(current, committed_prefix=torch.randn(2, 16, 14), committed_prefix_lengths=torch.tensor([4, 2]),
        observation_delay_seconds=torch.tensor([.03, .05]),
        action_target_offsets=torch.arange(1,17).float()[None].expand(2,-1) * .16)
    bridge = replace(bridge, interface_version=2, source_action_plan=torch.randn(2, 16, 14, requires_grad=True),
        source_action_sigma=.9, source_motion=current.motion.clone() + .5,
        source_motion_interval_valid_mask=current.motion_interval_valid_mask.clone(),
        source_motion_acceleration_valid_mask=current.motion_acceleration_valid_mask.clone(),
        source_rgb=current.rgb.clone() + .1, source_rgb_valid_mask=current.rgb_valid_mask)
    bridge.video_k = tuple(x.detach().requires_grad_() for x in bridge.video_k)
    bridge.video_v = tuple(x.detach().requires_grad_() for x in bridge.video_v)
    return model, current, bridge, text, action


def run():
    torch.set_num_threads(2)
    checks = []
    model, current, bridge, text, action = joint_fixture()
    memory = model.prepare_memory(current, text, bridge=bridge)
    for i in range(model.layers):
        k, v = model.adapt_video_kv(i, action, memory, bridge.video_k[i], bridge.video_v[i])
        assert torch.equal(k, bridge.video_k[i]) and torch.equal(v, bridge.video_v[i])
    assert model.apply_layer(0, action, memory) is action
    checks.append("zero_init_identity_native_action_motion_path_preserved")
    # Actual V1 migration has already-trained nonzero correction heads. Its
    # function must be preserved, not just an artificial all-zero-head fixture.
    legacy, *_ = fixture()
    active_outputs(legacy)
    migrated, migration_current, migration_bridge, migration_text, migration_actions = joint_fixture()
    result = migrated.load_state_dict(legacy.state_dict(), strict=False)
    assert all(k.startswith(("visual_innovation.","motion_innovation.","plan_innovation.",
                            "innovation_kind","innovation_gate")) for k in result.missing_keys)
    old_memory = legacy.prepare_memory(migration_current, migration_text, bridge=migration_bridge)
    new_memory = migrated.prepare_memory(migration_current, migration_text, bridge=migration_bridge)
    for layer in range(legacy.layers):
        old_kv = legacy.adapt_video_kv(layer,migration_actions,old_memory,migration_bridge.video_k[layer],migration_bridge.video_v[layer])
        new_kv = migrated.adapt_video_kv(layer,migration_actions,new_memory,migration_bridge.video_k[layer],migration_bridge.video_v[layer])
        assert all(torch.equal(old,new) for old,new in zip(old_kv,new_kv))
    fresh_prefixes=("visual_innovation.","motion_innovation.","plan_innovation.","innovation_kind","innovation_gate")
    for name,p in migrated.named_parameters():
        p.requires_grad_(name.startswith(fresh_prefixes))
    trainable=[p for p in migrated.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(trainable,lr=.02)
    before={n:p.detach().clone() for n,p in migrated.named_parameters() if p.requires_grad}
    weights=[torch.randn_like(x) for x in (migration_bridge.video_k[0],migration_bridge.video_v[0])]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        mm=migrated.prepare_memory(migration_current,migration_text,bridge=migration_bridge)
        kv=migrated.adapt_video_kv(0,migration_actions,mm,migration_bridge.video_k[0],migration_bridge.video_v[0])
        sum((x*w).sum() for x,w in zip(kv,weights)).backward()
        assert migrated.innovation_gate.grad.abs()>0
        if step == 1:
            for name in ("visual_innovation","motion_innovation","plan_innovation"):
                assert sum(p.grad.abs().sum() for p in getattr(migrated,name).parameters())>0
        optimizer.step()
    for name in ("visual_innovation.1.weight","motion_innovation.0.weight","plan_innovation.0.weight","innovation_gate"):
        assert not torch.equal(before[name],dict(migrated.named_parameters())[name])
    checks.append("nonzero_v1_migration_exact_then_gate_and_all_innovations_learn_in_two_updates")

    active_outputs(model)
    with torch.no_grad():
        model.innovation_gate.fill_(.1)
    model.zero_grad()
    memory = model.prepare_memory(current, text, bridge=bridge)
    loss = sum(sum((x * torch.randn_like(x)).sum() for x in model.adapt_video_kv(i, action, memory,
        bridge.video_k[i], bridge.video_v[i])) for i in range(model.layers))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for name in ("visual_innovation.1.weight", "motion_innovation.0.weight", "plan_innovation.0.weight"):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in (*bridge.video_k, *bridge.video_v))
    assert bridge.source_action_plan.grad is not None and bridge.source_action_plan.grad.abs().sum() > 0
    checks.append("slow_cache_plan_and_three_innovation_paths_receive_gradients")
    def context(c=current, b=bridge):
        return model.prepare_memory(c, text, bridge=b).rank_context[0]
    baseline = context()
    perturbations = [replace(bridge, source_rgb=bridge.source_rgb + 2),
        replace(bridge, source_motion=bridge.source_motion + 2),
        replace(bridge, source_action_plan=bridge.source_action_plan + 2)]
    assert all(not torch.allclose(baseline, context(b=b)) for b in perturbations)
    assert not torch.allclose(baseline, context(c=replace(current, committed_prefix=current.committed_prefix + 2)))
    assert not torch.allclose(baseline, context(c=replace(current, observation_delay_seconds=.25)))
    assert not torch.allclose(baseline, context(c=replace(current, action_target_offsets=current.action_target_offsets * 1.5)))
    d0=replace(current, committed_prefix_lengths=torch.zeros_like(current.committed_prefix_lengths), observation_delay_seconds=0.)
    assert not torch.allclose(context(c=d0),context(c=replace(d0, action_target_offsets=d0.action_target_offsets * 1.5)))
    checks.append("source_rgb_motion_plan_committed_prefix_and_delay_affect_context")
    poisoned = current.committed_prefix.clone()
    valid = torch.arange(16)[None] < current.committed_prefix_lengths[:, None]
    poisoned[~valid] = float("nan")
    assert torch.equal(baseline, context(c=replace(current, committed_prefix=poisoned)))
    checks.append("uncommitted_suffix_never_leaks_into_context")
    # A source age near the end of its horizon may not repeat/clamp the last
    # source action to invent plan overlap beyond that horizon.
    old = bridge.source_action_plan.detach().clone()
    older = replace(current, age=replace(current.age, delta_actions=torch.tensor([16, 20])))
    changed = replace(bridge, source_action_plan=old + 100)
    assert torch.equal(context(c=older), context(c=older, b=changed))
    checks.append("expired_plan_rows_explicitly_masked")
    for bad in (replace(bridge, interface_version=1), replace(bridge, source_rgb=None), replace(bridge, source_action_plan=None)):
        try: context(b=bad)
        except ValueError: pass
        else: raise AssertionError("legacy or missing context accepted")
    try: context(c=replace(current, committed_prefix=None))
    except ValueError: pass
    else: raise AssertionError("implicit RTC prefix accepted")
    try: context(c=replace(current, action_target_offsets=None))
    except ValueError: pass
    else: raise AssertionError("implicit physical action target cadence accepted")
    checks.append("strict_v2_context_prefix_and_physical_target_clock_contract")
    restored = JointMotionKVConditioner(**model.constructor_spec)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(baseline, restored.prepare_memory(current, text, bridge=bridge).rank_context[0])
    checks.append("versioned_strict_checkpoint_roundtrip")
    one_layer = JointMotionKVConditioner(**dict(model.constructor_spec, correction_layers=[1]))
    one_layer.load_state_dict(model.state_dict(), strict=True)
    m = one_layer.prepare_memory(current, text, bridge=bridge)
    k,v = one_layer.adapt_video_kv(0, action, m, bridge.video_k[0], bridge.video_v[0])
    assert torch.equal(k, bridge.video_k[0]) and torch.equal(v, bridge.video_v[0])
    checks.append("layer_placement_ablation_exact_inactive_layers")

    for precision in (torch.bfloat16, torch.float16):
        reduced = JointMotionKVConditioner(**model.constructor_spec).to(dtype=precision)
        reduced.load_state_dict(model.state_dict(), strict=True)
        reduced_bridge = replace(bridge, video_k=tuple(x.to(precision) for x in bridge.video_k),
            video_v=tuple(x.to(precision) for x in bridge.video_v))
        reduced_memory = reduced.prepare_memory(current, text.to(precision), bridge=reduced_bridge)
        rk,rv = reduced.adapt_video_kv(0, action.to(precision), reduced_memory,
            reduced_bridge.video_k[0], reduced_bridge.video_v[0])
        assert rk.dtype == precision and rv.dtype == precision
        assert torch.isfinite(rk).all() and torch.isfinite(rv).all()
    checks.append("fp16_bf16_innovation_path_finite_and_dtype_preserved")

    frozen = JointMotionKVConditioner(**model.constructor_spec).eval().requires_grad_(False)
    frozen.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        cached = frozen.prepare_source_cache(bridge)
        expected = frozen.prepare_memory(current, text, bridge=bridge)
        actual = frozen.prepare_memory(current, text, bridge=cached)
        assert all(torch.equal(a,b) for a,b in zip(expected.rank_context, actual.rank_context))
        calls = []
        handle = frozen.encoder.register_forward_hook(lambda *unused: calls.append(1))
        frozen.prepare_memory(current, text, bridge=cached)
        handle.remove()
        assert len(calls) == 1, "cached source RGB was encoded again"
        changed = replace(cached, source_rgb=bridge.source_rgb + 1)
        refreshed = frozen.prepare_source_cache(changed)
        assert not torch.equal(cached.source_visual_tokens, refreshed.source_visual_tokens)
        assert torch.equal(frozen.prepare_memory(current,text,bridge=replace(changed,
            source_visual_tokens=None,source_visual_valid=None)).rank_context[0],
            frozen.prepare_memory(current,text,bridge=refreshed).rank_context[0])
        before = refreshed.source_visual_tokens.clone()
        next(frozen.encoder.parameters()).add_(.01)
        revised = frozen.prepare_source_cache(refreshed)
        assert not torch.equal(before, revised.source_visual_tokens)
    try: model.prepare_memory(current, text, bridge=cached)
    except RuntimeError: pass
    else: raise AssertionError("inference source cache accepted during training")
    checks.append("explicit_source_visual_cache_equivalent_skips_encoding_and_invalidates")

    from types import SimpleNamespace
    from dynamicwam.inference.native_fast_graph import NativeFastGraph, _BRIDGE_CONTEXT_TENSORS
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from dynamicwam.models.rtc_prefix import PrefixCondition
    graph_model = SimpleNamespace(native_multirate=True, fast_schedule=BridgeSchedule.from_flow_match(),
        backbone=SimpleNamespace(action_attention_spec=SimpleNamespace(precision=torch.float32)))
    graph = NativeFastGraph(graph_model, enabled=False)
    noise = torch.randn(2,16,14)
    prefix = PrefixCondition(current.committed_prefix, current.committed_prefix_lengths)
    canonical_current, canonical_bridge = graph._canonical_context(current, cached, noise, prefix)
    graph._allocate(canonical_current, canonical_bridge, text, None, noise, prefix)
    for name in _BRIDGE_CONTEXT_TENSORS:
        src, dst = getattr(canonical_bridge,name), getattr(graph.bridge,name)
        if src is not None:
            assert torch.equal(src,dst) and src.data_ptr() != dst.data_ptr(), name
            assert getattr(graph.conditioning_bridge,name).data_ptr() == dst.data_ptr(), name
    assert graph.current.committed_prefix.data_ptr() != prefix.actions.data_ptr()
    changed_fields = {name: (getattr(cached,name).clone() + .25) for name in
        ("source_action_plan", "source_motion", "source_rgb", "source_visual_tokens")}
    newer = replace(cached, **changed_fields, source_action_sigma=.3)
    newer_prefix = PrefixCondition(prefix.actions + .2, torch.tensor([0,3]))
    newer_current = replace(current, observation_delay_seconds=torch.tensor([.2,.3]),
        action_target_offsets=current.action_target_offsets * 1.25)
    graph._copy_inputs(newer_current, newer, text+1, None, noise+1, newer_prefix)
    for name in changed_fields:
        assert torch.equal(getattr(graph.bridge,name), getattr(newer,name)), name
    assert torch.equal(graph.current.committed_prefix, newer_prefix.actions)
    assert torch.equal(graph.current.committed_prefix_lengths, newer_prefix.lengths)
    assert torch.equal(graph.current.observation_delay_seconds, newer_current.observation_delay_seconds)
    assert torch.equal(graph.current.action_target_offsets, newer_current.action_target_offsets)
    assert graph.current.action_target_offsets.data_ptr() != newer_current.action_target_offsets.data_ptr()
    assert float(graph.bridge.source_action_sigma) == float(torch.tensor(.3))
    assert graph._signature(canonical_current,canonical_bridge,text,None,noise,prefix) != graph._signature(
        canonical_current,replace(canonical_bridge,source_visual_tokens=None,source_visual_valid=None),text,None,noise,prefix)
    checks.append("graph_static_context_owns_all_v2_inputs_and_copies_dynamic_updates")

    class EagerProbe(nn.Module):
        native_multirate = True
        rtc_prefix_max = 4
        def prepare_fast_memory(self, obs, world, text, mask):
            self.seen = obs
            return obs.committed_prefix
        def sample_fast_actions(self, obs, world, memory, noise, prefix):
            assert torch.equal(obs.committed_prefix_lengths, prefix.lengths)
            return memory.clone()
    eager_model = EagerProbe().eval()
    eager_graph = NativeFastGraph(eager_model, enabled=False)
    assert torch.equal(eager_graph.sample(current,bridge,text,None,noise,prefix), prefix.actions)
    reset = eager_graph.sample(current,bridge,text,None,noise,None)
    assert torch.equal(reset,torch.zeros_like(noise))
    assert not eager_model.seen.committed_prefix_lengths.any()
    checks.append("eager_entry_attaches_canonical_prefix_and_resets_d0")

    class Joint(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.tensor(.2))
        def _forward_video_action(self, batch, *, include_video_prediction, build_video_cache, detach_cache):
            future, a = batch["future_latent"], batch["noisy_actions"]
            out = {"video_pred": future * self.weight, "action_pred": a * self.weight}
            if build_video_cache:
                k = (future.flatten(1).mean(1) + a.flatten(1).mean(1) + self.weight).view(-1,1,1,1)
                if detach_cache: k = k.detach()
                out["video_cache"] = {"video_k":[k], "video_v":[k * 2], "seq_lens":torch.ones(a.shape[0],dtype=torch.long), "grid_sizes":{}}
            return out
    backbone = Joint()
    source = SourceObservation(torch.randn(2,1,1,2,2), current.state, current.motion,
        current.motion_interval_valid_mask, current.motion_acceleration_valid_mask, [], 3, .5,
        rgb=current.rgb, rgb_valid_mask=current.rgb_valid_mask)
    noise = SourceNoise(torch.randn(2,1,2,2,2), torch.randn(2,16,14))
    attached = CausalWorldBridge()(backbone, source, noise, detach_cache=False)
    assert attached.interface_version == 2 and attached.source_rgb is current.rgb
    (attached.video_k[0].sum() + attached.source_action_plan.sum()).backward()
    assert backbone.weight.grad is not None and backbone.weight.grad.abs() > 0
    detached = CausalWorldBridge()(backbone, source, noise, detach_cache=True)
    assert not detached.source_action_plan.requires_grad and not detached.video_k[0].requires_grad
    checks.append("two_joint_forward_cache_and_plan_attached_or_detached_consistently")
    print({"passed":True, "count":len(checks), "checks":checks})


if __name__ == "__main__":
    run()
