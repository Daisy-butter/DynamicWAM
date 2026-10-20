"""CPU check of the actual serving repair, training fallback and Graph copies.

Tiny native fixture only: actual full CUDA capture/concurrent slow publication
must pass separately. No checkpoint, training or simulator change is made.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_joint_motionkv_v3_training_cpu import v3_fixture, cpu_attention
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.rtc_prefix import PrefixCondition
from dynamicwam.inference.native_fast_graph import NativeFastGraph
from dynamicwam.runtime.domino_continuous.motionkv_policy import (
    with_training_previous_plan, PREVIOUS_PLAN_MODE,
)


class CapturedTrainingCurrent(Exception):
    pass


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261021)
    checks = []
    with ExitStack() as stack:
        for name in ("dynamicwam.models.action_core_ops.flash_attention",
                     "dynamicwam.models.action_core_ops.dense_action_attention"):
            stack.enter_context(patch(name, cpu_attention))
        model, batch = v3_fixture()
        model.gradient_checkpointing = False
        model.fast_conditioner = JointMotionKVConditioner(
            **dict(model.fast_conditioner.constructor_spec,
                   proposal_feedback=True, complete_source_plan=True))
        conditioner = model.fast_conditioner
        bridge = model.bridge(model.backbone, batch["source"], batch["source_noise"], detach_cache=True)
        plan = torch.arange(16 * 14, dtype=torch.float32).reshape(1, 16, 14).expand(2, -1, -1).clone() / 100
        bridge = replace(bridge, source_action_plan=plan,
                         source_completed_plan=plan + 100)
        prefix = PrefixCondition(torch.randn_like(plan), torch.tensor([4, 4]))
        base = replace(batch["current"], committed_prefix=prefix.actions,
                       committed_prefix_lengths=prefix.lengths,
                       previous_plan=None, previous_plan_valid=None)
        offsets = torch.arange(1, 17, dtype=torch.float64)[None].expand(2, -1) * .16
        base = replace(base, action_target_offsets=offsets)
        cases = []
        for age in (0, 4, 8, 12, 16, 31):
            current = replace(base, age=replace(base.age,
                delta_sim_seconds=torch.full((2,), age * .16, dtype=torch.float64),
                delta_actions=age))
            fixed = with_training_previous_plan(current, bridge, conditioner)
            expected, valid = conditioner.aligned_source_plan(current, bridge)
            assert torch.equal(fixed.previous_plan, expected)
            assert torch.equal(fixed.previous_plan_valid, valid)
            assert valid.sum(1).tolist() == [max(0, 16-age)] * 2
            assert not fixed.previous_plan.requires_grad
            assert fixed.committed_prefix is current.committed_prefix
            assert fixed.action_target_offsets is current.action_target_offsets
            assert current.previous_plan is None
            cases.append((current, fixed))
        checks.append("regular_ages_0_4_8_12_16_31_exact_training_alignment_masks_no_tail_fill")
        # Time must control alignment even when integer age gives another answer.
        current = replace(base, action_target_offsets=offsets-.03,
            age=replace(base.age, delta_sim_seconds=torch.tensor([.11, .79], dtype=torch.float64),
                        delta_actions=torch.tensor([0, 0])))
        fixed = with_training_previous_plan(current, bridge, conditioner)
        assert torch.allclose(fixed.previous_plan[0, 0, [0, 5, 7, 12]],
                              (plan[0, 0]+plan[0, 1])[ [0, 5, 7, 12]]/2)
        assert torch.equal(fixed.previous_plan[0, 0, [6, 13]], plan[0, 0, [6, 13]])
        assert fixed.previous_plan_valid.sum(1).tolist() == [15, 11]
        assert torch.equal(fixed.previous_plan[:, -1], torch.zeros_like(plan[:, -1]))
        checks.append("fractional_physical_times_linear_arms_held_grippers_ignore_integer_age")

        # Compare to the *actual* training fallback before the first consumer,
        # including the native training path's detached source reference.
        captured = {}
        def capture(current, actual_bridge, *args, **kwargs):
            captured.update(current=current, bridge=actual_bridge)
            raise CapturedTrainingCurrent()
        source_batch = dict(batch, current=cases[1][0], rtc_prefix=prefix)
        with patch.object(model.bridge, "forward", return_value=bridge), patch.object(model, "prepare_fast_memory", side_effect=capture):
            try:
                model(source_batch)
            except CapturedTrainingCurrent:
                pass
            else:
                raise AssertionError("training did not reach prepare_fast_memory")
        assert torch.equal(captured["current"].previous_plan, cases[1][1].previous_plan)
        assert torch.equal(captured["current"].previous_plan_valid, cases[1][1].previous_plan_valid)
        checks.append("actual_training_forward_fallback_identical_to_explicit_serving_tensors")

        model.eval().requires_grad_(False)
        text = torch.stack(batch["source"].text_embeddings)
        features = []
        hook = conditioner.plan_innovation.register_forward_pre_hook(
            lambda module, args: features.append(args[0].detach().clone()))
        with torch.no_grad():
            memory = model.prepare_fast_memory(cases[1][1], bridge, text)
        hook.remove()
        now = cases[1][1]
        previous, valid = now.previous_plan, now.previous_plan_valid
        assert torch.equal(memory.proposal_reference[2], previous)
        assert torch.equal(memory.proposal_reference[3], valid)
        assert torch.equal(features[0][..., -2].bool(), valid)
        near = features[0][..., -1]
        assert torch.allclose(features[0][..., -16:-2], .1*near[...,None]*previous)
        proposal, _ = conditioner.proposal_features(
            batch["fast_noisy_actions"], batch["fast_action_t"], memory,
            num_train_timesteps=1000)
        assert torch.equal(proposal[..., 14:28], proposal[..., 28:42])
        checks.append("plan_innovation_and_proposal_reference_share_early_plan_masks_and_delta_semantics")

        # Fixed-shape Graph input storage must change *values* on every query,
        # including an entirely expired horizon and a new published bridge.
        graph = NativeFastGraph(model, enabled=False)
        noise = batch["source_noise"].actions
        with torch.no_grad():
            graph._allocate(cases[0][1], bridge, text, None, noise, prefix)
            ptrs = (graph.current.previous_plan.data_ptr(), graph.current.previous_plan_valid.data_ptr())
            assert ptrs[0] != cases[0][1].previous_plan.data_ptr()
            signature = graph._signature(cases[0][1], bridge, text, None, noise, prefix)
            for _, now in cases:
                assert signature == graph._signature(now, bridge, text, None, noise, prefix)
                graph._copy_inputs(now, bridge, text, None, noise, prefix)
                assert torch.equal(graph.current.previous_plan, now.previous_plan)
                assert torch.equal(graph.current.previous_plan_valid, now.previous_plan_valid)
                assert ptrs == (graph.current.previous_plan.data_ptr(), graph.current.previous_plan_valid.data_ptr())
                assert torch.equal(graph.current.committed_prefix, prefix.actions)
                assert torch.equal(graph.prefix.actions, prefix.actions)
            expired_output = graph._forward()
            assert bool(torch.isfinite(expired_output).all())
            assert torch.equal(expired_output[:, :4], prefix.actions[:, :4])
            eager_memory = model.prepare_fast_memory(cases[-1][1], bridge, text)
            eager_output = model.sample_fast_actions(
                cases[-1][1], bridge, eager_memory, noise, prefix=prefix)
            assert torch.equal(expired_output, eager_output)
            renewed = replace(bridge, source_action_plan=plan-7)
            now = with_training_previous_plan(cases[1][0], renewed, conditioner)
            graph._copy_inputs(now, renewed, text, None, noise, prefix)
            assert torch.equal(graph.current.previous_plan, now.previous_plan)
            assert torch.equal(graph.bridge.source_action_plan, renewed.source_action_plan)
            assert torch.equal(bridge.source_action_plan, plan)
        checks.append("graph_static_owned_buffers_update_age_expired_mask_and_new_bridge_without_recapture")
        checks.append("fully_expired_context_actual_graph_body_matches_eager_finite_and_hard_prefix_unchanged")
        print(json.dumps(dict(passed=True, checks=checks, previous_plan_mode=PREVIOUS_PLAN_MODE,
            device="cpu", limitation="tiny native fixture; full GPU capture and closed loop checked separately")))


if __name__ == "__main__":
    run()
