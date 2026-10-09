"""CPU contracts for the native action-expert multirate path.

Run with ``PYTHONPATH=src python scripts/check_native_multirate.py``. These
checks use a tiny backbone and do not instantiate WAN, allocate CUDA memory,
or establish task success or deployment latency.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import (
    BridgeAge, BridgeTensorBundle, CurrentObservation, SourceNoise, SourceObservation,
)
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.models.world_aware_conditioner import WorldAwareConditioner


class TinyBackbone(nn.Module):
    """Differentiable action path with recorded joint solver inputs."""

    def __init__(self):
        super().__init__()
        self.action_expert = nn.ModuleDict({"input": nn.Linear(4, 32), "output": nn.Linear(32, 4)})
        self.absolute_motion_tokens = nn.Linear(12, 32)
        self.compact_wan = nn.Linear(4, 4)
        self.action_attention_spec = SimpleNamespace(precision=torch.float32, window_sizes=((-1, -1),) * 2)
        self.joint_times = []
        self.action_calls = 0

    def _forward_video_action(self, batch, *, include_video_prediction, build_video_cache, detach_cache=True):
        self.joint_times.append(batch["action_t"].detach().clone())
        result = {"action_pred": self.action_expert["output"](self.action_expert["input"](batch["noisy_actions"]))}
        if include_video_prediction:
            result["video_pred"] = torch.zeros_like(batch["future_latent"]) + self.compact_wan.weight.mean()
        if build_video_cache:
            b = batch["noisy_actions"].shape[0]
            item = self.compact_wan.weight.mean().expand(b, 6, 2, 8)
            if detach_cache:
                item = item.detach()
            result["video_cache"] = dict(video_k=[item, item], video_v=[item, item],
                seq_lens=torch.full((b,), 6, dtype=torch.long),
                grid_sizes={"condition_seq_len": 2}, all_video_tokens_valid=True)
        return result

    def forward_action_with_video_cache(self, batch, *, fast_residual=None, null_bridge=False):
        self.action_calls += 1
        hidden = self.action_expert["input"](batch["noisy_actions"])
        hidden = hidden + self.absolute_motion_tokens(batch["absolute_motion_features"].mean(1))[:, None]
        if fast_residual is not None:
            for layer in range(2):
                hidden = fast_residual(layer, hidden)
        return {"action_pred": self.action_expert["output"](hidden)}


def fixture(*, steps=2):
    batch = 2
    conditioner = WorldAwareConditioner(dim=32, layers=2, heads=4, head_dim=8,
        text_dim=8, state_dim=4, world_layers=(0, 1), world_heads=2,
        world_head_dim=8, world_queries=4, world_dim=16)
    age = BridgeAge(torch.tensor([0., .6]), torch.tensor([0., 12.]),
                    torch.ones(batch, dtype=torch.bool), torch.zeros(batch, 4))
    text = torch.randn(batch, 3, 8)
    current = CurrentObservation(torch.randn(batch, 3, 3, 32, 32), torch.randn(batch, 4),
        torch.randn(batch, 2, 12), torch.ones(batch, 2, dtype=torch.bool),
        torch.ones(batch, 2, dtype=torch.bool), list(text), age)
    bridge = BridgeTensorBundle(tuple(torch.randn(batch, 6, 2, 8) for _ in range(2)),
        tuple(torch.randn(batch, 6, 2, 8) for _ in range(2)), torch.tensor([6, 4]),
        {"condition_seq_len": 2}, 0, 0.)
    model = MultirateWAM(TinyBackbone(), conditioner, native_multirate=True, fast_steps=steps)
    return model, current, bridge, text


class NativeMultirateContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261004)

    def test_fast_schedule_is_independent_from_slow_joint_prefix(self):
        for fast_steps in (1, 2, 4):
            model, current, bridge, text = fixture(steps=fast_steps)
            original = model.schedule.action_sigmas.clone()
            self.assertEqual(len(model.schedule.action_sigmas), 10)
            self.assertEqual(len(model.schedule.video_sigmas), 2)
            self.assertEqual(len(model.fast_schedule.action_sigmas), fast_steps)
            self.assertIs(model.bridge.schedule, model.schedule)
            noise = SourceNoise(torch.randn(2, 4, 2, 2, 2), torch.randn(2, 16, 4))
            source = SourceObservation(torch.randn(2, 4, 2, 2, 2), current.state, current.motion,
                current.motion_interval_valid_mask, current.motion_acceleration_valid_mask,
                current.text_embeddings, 0, 0.)
            model.bridge(model.backbone, source, noise, detach_cache=True)
            expected = model.schedule.action_timesteps[:2]
            torch.testing.assert_close(torch.stack(model.backbone.joint_times)[:, 0], expected)
            memory = model.prepare_fast_memory(current, bridge, text)
            actions = model.sample_fast_actions(current, bridge, memory, noise.actions)
            self.assertEqual(model.backbone.action_calls, fast_steps)
            self.assertEqual(actions.shape, (2, 16, 4))
            torch.testing.assert_close(model.schedule.action_sigmas, original, rtol=0, atol=0)

    def test_native_bridge_rejects_shortened_slow_schedule(self):
        model, *_ = fixture()
        with self.assertRaises((ValueError, RuntimeError)):
            MultirateWAM(model.backbone, model.fast_conditioner,
                BridgeSchedule.from_flow_match(action_steps=2), native_multirate=True, fast_steps=2)

    def test_world_router_uses_fresh_observation_and_preserves_zero_residual(self):
        model, current, bridge, text = fixture()
        conditioner = model.fast_conditioner
        memory = model.prepare_fast_memory(current, bridge, text)
        self.assertEqual(len(memory.keys), 2)
        self.assertEqual(memory.keys[0].shape, (2, 62, 16))
        tokens = torch.randn(2, 21, 32)
        for layer in range(2):
            torch.testing.assert_close(conditioner.apply_layer(layer, tokens, memory), tokens, rtol=0, atol=0)
        changed = dataclasses.replace(current, rgb=current.rgb + 1)
        fresh = model.prepare_fast_memory(changed, bridge, text)
        self.assertFalse(torch.allclose(memory.keys[0][:, -8:], fresh.keys[0][:, -8:]))
        changed_world = dataclasses.replace(bridge, video_v=tuple(v + torch.randn_like(v) for v in bridge.video_v))
        altered = model.prepare_fast_memory(current, changed_world, text)
        self.assertFalse(torch.allclose(memory.values[0][:, -8:], altered.values[0][:, -8:]))

    def test_null_and_padded_world_values_cannot_leak_into_router(self):
        model, current, bridge, text = fixture()
        bridge.null_bridge = torch.tensor([False, True])
        before = model.prepare_fast_memory(current, bridge, text)
        values = tuple(v.clone() for v in bridge.video_v)
        for value in values:
            value[1] += 10000
        altered = model.prepare_fast_memory(current, dataclasses.replace(bridge, video_v=values), text)
        torch.testing.assert_close(before.values[0][1], altered.values[0][1], rtol=0, atol=0)
        self.assertFalse(before.valid[1, -8:].any())
        self.assertTrue(torch.isfinite(before.values[0]).all())
        bridge.null_bridge = False
        before = model.prepare_fast_memory(current, bridge, text)
        values = tuple(v.clone() for v in bridge.video_v)
        for value in values:
            value[1, 4:] += 10000
        altered = model.prepare_fast_memory(current, dataclasses.replace(bridge, video_v=values), text)
        torch.testing.assert_close(before.values[0][1], altered.values[0][1], rtol=0, atol=0)

    def test_native_sampler_retains_gradients_to_action_motion_and_residual(self):
        model, current, bridge, text = fixture(steps=2)
        memory = model.prepare_fast_memory(current, bridge, text)
        output = model.sample_fast_actions(current, bridge, memory, torch.randn(2, 16, 4))
        output.square().mean().backward()
        for module in (model.backbone.action_expert, model.backbone.absolute_motion_tokens):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        for block in model.fast_conditioner.blocks:
            self.assertGreater(block.out_proj.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.compact_wan.parameters()))

    def test_both_native_training_stages_freeze_world_weights(self):
        from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable
        for stage in ("adapt", "distill"):
            model, *_ = fixture()
            model.native_phase = stage
            model.external_native_teacher = True
            configure_multirate_trainable(model, "warmup")
            self.assertTrue(all(p.requires_grad for p in model.backbone.action_expert.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.backbone.absolute_motion_tokens.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.fast_conditioner.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.backbone.compact_wan.parameters()))

    def test_native_training_requires_external_teacher_and_detached_source_bridge(self):
        model, current, bridge, text = fixture()
        model.native_phase = "distill"
        with self.assertRaisesRegex(RuntimeError, "external teacher"):
            model({})
        model.external_native_teacher = True
        with self.assertRaisesRegex(RuntimeError, "fixed teacher actions"):
            model({})
        batch = dict(native_teacher_actions=torch.zeros(2, 16, 4))
        with self.assertRaisesRegex(RuntimeError, "official source bridge"):
            model(batch)
        bridge.video_k[0].requires_grad_(True)
        batch["native_source_bridge"] = bridge
        with self.assertRaisesRegex(RuntimeError, "detached"):
            model(batch)

    def test_distillation_uses_deployed_solver_and_does_not_mutate_official_bridge(self):
        model, current, bridge, text = fixture(steps=2)
        model.native_phase = "distill"
        model.native_teacher_weight = 1.
        model.native_expert_weight = .1
        model.native_fm_anchor_weight = 0.
        model.external_native_teacher = True
        initial = torch.randn(2, 16, 4)
        source_initial = torch.randn(2, 16, 4)
        teacher = torch.randn(2, 16, 4, requires_grad=True)
        batch = dict(native_teacher_actions=teacher, native_source_bridge=bridge,
            null_bridge=torch.tensor([False, True]), current=current, fast_text=text,
            source_noise=SourceNoise(torch.randn(2, 4, 2, 2, 2), source_initial),
            native_fast_noise=initial,
            clean_actions=torch.randn(2, 16, 4))
        source_snapshot = source_initial.clone()
        cache_snapshot = tuple(t.clone() for t in (*bridge.video_k, *bridge.video_v))
        with patch.object(model, "sample_fast_actions", wraps=model.sample_fast_actions) as sampler:
            output = model(batch)
        self.assertEqual(sampler.call_count, 1)
        self.assertIs(sampler.call_args.args[3], initial)
        self.assertIsNot(sampler.call_args.args[3], source_initial)
        self.assertEqual(model.backbone.action_calls, 2)
        self.assertFalse(model.backbone.joint_times, "student must never rebuild the official source bridge")
        self.assertIs(bridge.null_bridge, False)
        torch.testing.assert_close(source_initial, source_snapshot, rtol=0, atol=0)
        for original, after in zip(cache_snapshot, (*bridge.video_k, *bridge.video_v)):
            torch.testing.assert_close(original, after, rtol=0, atol=0)
        expected = (output["native_actions"] - teacher.detach()).square().mean()
        expected = expected + .1 * (output["native_actions"] - batch["clean_actions"]).square().mean()
        torch.testing.assert_close(output["native_loss"], expected)
        output["native_loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(model.backbone.action_expert["output"].weight.grad.abs().sum(), 0)
        self.assertGreater(model.fast_conditioner.blocks[0].out_proj.weight.grad.abs().sum(), 0)

    def test_router_has_no_tensor_to_host_boolean_in_query_path(self):
        model, current, bridge, text = fixture()
        with patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host synchronization")):
            memory = model.prepare_fast_memory(current, bridge, text)
        self.assertEqual(memory.valid.shape, (2, 62))

    def test_native_fast_export_roundtrip_keeps_router_and_independent_solver(self):
        from dynamicwam.inference.multirate_model_loader import (
            build_fast_only_modules, export_fast_only_payload,
        )
        from dynamicwam.inference.fast_runtime import build_fast_runtime
        model, current, bridge, text = fixture(steps=2)
        spec = dict(architecture="native", native_multirate=True, fast_steps=2,
            schedule=model.schedule.to_config(), fast_schedule=model.fast_schedule.to_config(),
            action_expert=dict(dim=32, ffn_dim=64, num_layers=2, state_dim=4, action_dim=4, chunk_size=16),
            wan_attention=dict(dim=16, num_heads=2, head_dim=8, precision="float32"),
            motion=dict(history_count=2, mean=[0.] * 12, scale=[1.] * 12),
            fast_conditioner=model.fast_conditioner.constructor_spec)
        modules = build_fast_only_modules(spec)
        modules.fast_conditioner.load_state_dict(model.fast_conditioner.state_dict(), strict=True)
        state = {}
        for name, module in modules.items():
            prefix = "fast_conditioner." if name == "fast_conditioner" else f"backbone.{name}."
            state.update({prefix + key: value for key, value in module.state_dict().items()})
        # Prove a full source export cannot accidentally retain the video model.
        state["backbone.compact_wan.weight"] = torch.randn(4, 4)
        payload = export_fast_only_payload(state, module_spec=spec, global_step=3)
        self.assertNotIn("backbone.compact_wan.weight", payload["model"])
        loaded = build_fast_runtime(payload)
        self.assertTrue(loaded.native_multirate)
        self.assertEqual(len(loaded.schedule.action_sigmas), 10)
        self.assertEqual(len(loaded.fast_schedule.action_sigmas), 2)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        self.assertFalse(any(m.training for m in loaded.modules()))
        model.eval()
        with torch.no_grad():
            before = model.prepare_fast_memory(current, bridge, text)
            after = loaded.prepare_fast_memory(current, bridge, text)
        for expected, actual in zip(before.keys + before.values, after.keys + after.values):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key, value in payload["model"].items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

    def test_native_cannot_silently_use_legacy_twelve_action_worker_cadence(self):
        from dynamicwam.runtime.ciwam.wam.multirate_policy import validate_native_scheduling
        contract = {"native_multirate": {"world_refresh_actions": 16}}
        validate_native_scheduling(contract, "controlled_ratio")
        with self.assertRaisesRegex(RuntimeError, "16-action source cadence"):
            validate_native_scheduling(contract, "independent_worker")
        validate_native_scheduling({}, "independent_worker")

    def test_training_export_persists_both_solver_contracts(self):
        from dynamicwam.training.train.multirate_trainer import MultirateTrainer
        model, *_ = fixture(steps=2)
        trainer = MultirateTrainer.__new__(MultirateTrainer)
        trainer.model = model
        trainer.global_step = 2
        trainer.motion_checkpoint_metadata = {"history_count": 2}
        trainer.motion_statistics = dict(mean=[0.] * 12, scale=[1.] * 12)
        trainer.action_normalization = {"stats": {}}
        trainer.accelerator = SimpleNamespace(is_main_process=True,
            get_state_dict=lambda m: m.state_dict(), unwrap_model=lambda m: m)
        trainer._checkpoint_config = lambda: {"runtime_contract": {"native_multirate": {"fast_steps": 2}}}
        trainer.config = dict(native_phase="distill", native_fast_steps=2,
            model=dict(action_expert=dict(dim=32, ffn_dim=64, num_layers=2,
                state_dim=4, action_dim=4, chunk_size=16),
                compact_wan=dict(dim=16, num_heads=2, head_dim=8, precision="float32"),
                fast_conditioner=model.fast_conditioner.constructor_spec),
            flow_matching=dict(shift=5., sigma_min=0., extra_one_step=True, training_timesteps=1000))
        saved = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("dynamicwam.training.train.multirate_trainer.get_export_dir", return_value=Path(directory)), \
                 patch("dynamicwam.training.train.multirate_trainer._atomic_torch_save",
                       side_effect=lambda payload, path: saved.append(payload)):
                trainer.export_multirate()
        self.assertEqual(len(saved), 2)
        for payload in saved:
            self.assertEqual(payload["modules"]["schedule"], model.schedule.to_config())
            self.assertEqual(payload["modules"]["fast_schedule"], model.fast_schedule.to_config())
            self.assertEqual(payload["modules"]["fast_steps"], 2)

    def test_native_recipes_cover_the_full_deployed_age_cycle(self):
        from dynamicwam.config.multirate import load_multirate_profile
        root = Path(__file__).resolve().parents[1] / "configs" / "multirate"
        files = sorted(root.glob("native_*stage[23]*.yaml"))
        self.assertGreaterEqual(len(files), 6)
        for path in files:
            config = load_multirate_profile(path).training
            self.assertTrue(config["model"]["native_multirate"])
            self.assertEqual(config["dataset"]["execute_horizon"], 4)
            self.assertEqual(config["dataset"]["every_fast_queries"], 4)
            self.assertEqual(config["dataset"]["age_weighting"], "uniform")
            self.assertIsNone(config["max_bridge_age_seconds"])
            self.assertEqual(config["null_bridge_probability"], .1)
            self.assertTrue(config["model"]["native_teacher_checkpoint"])
            self.assertEqual(config["phase"], "fast_warmup")
            self.assertEqual(config["loss"]["video_weight"], 0.)
            if config["native_phase"] == "adapt":
                self.assertEqual(config["native_fast_steps"], 10)
            else:
                self.assertIn(config["native_fast_steps"], (1, 2, 4))

    def test_native_fast_world_and_flow_noises_are_independent_and_reproducible(self):
        from verify_multirate_training import _batch
        from dynamicwam.training.train.multirate_trainer import prepare_multirate_batch
        schedule = BridgeSchedule.from_flow_match()

        def prepare():
            return prepare_multirate_batch(_batch(), timesteps=schedule.action_timesteps,
                sigmas=schedule.action_sigmas, device="cpu", dtype=torch.float32,
                phase="warmup", null_bridge_probability=.1, sim_scale=1., action_scale=16.,
                native_teacher=True, native_multirate=True,
                generator=torch.Generator().manual_seed(92604))

        before, repeated = prepare(), prepare()
        noises = (before["source_noise"].actions, before["native_fast_noise"], before["teacher_flow_noise"])
        repeated_noises = (repeated["source_noise"].actions, repeated["native_fast_noise"], repeated["teacher_flow_noise"])
        for index, left in enumerate(noises):
            torch.testing.assert_close(left, repeated_noises[index], rtol=0, atol=0)
            for right in noises[index + 1:]:
                self.assertNotEqual(left.data_ptr(), right.data_ptr())
                self.assertFalse(torch.equal(left, right))
        torch.testing.assert_close(before["source_noise"].future_latent,
                                   repeated["source_noise"].future_latent, rtol=0, atol=0)

    def test_teacher_and_student_share_fast_noise_while_source_bridge_keeps_world_noise(self):
        from dynamicwam.training.train.multirate_trainer import MultirateTrainer
        model, current, bridge, text = fixture(steps=2)
        source = SourceObservation(torch.randn(2, 4, 2, 2, 2), current.state, current.motion,
            current.motion_interval_valid_mask, current.motion_acceleration_valid_mask,
            current.text_embeddings, 0, 0.)
        source_noise = SourceNoise(torch.randn(2, 4, 2, 2, 2), torch.randn(2, 16, 4))
        fast_noise = torch.randn_like(source_noise.actions)
        prepared = dict(source=source, source_noise=source_noise, native_fast_noise=fast_noise,
            current=current, aligned={"condition_latent": source.condition_latent + 1},
            fast_text=text, clean_actions=torch.randn_like(fast_noise))
        trainer = MultirateTrainer.__new__(MultirateTrainer)
        trainer.config = dict(native_phase="distill", flow_matching=dict(
            shift=5., sigma_min=0., extra_one_step=True, training_timesteps=1000))
        trainer.native_teacher = TinyBackbone().requires_grad_(False)
        source_calls = []

        def make_source(teacher, observation, noise, **kwargs):
            source_calls.append((observation, noise))
            return bridge

        trainer.native_source_bridge_builder = make_source
        teacher_noises = []

        def sample_teacher(teacher, observation, noise, schedule):
            teacher_noises.append(noise)
            self.assertIs(observation.condition_latent, prepared["aligned"]["condition_latent"])
            return noise.actions * .5

        with patch("dynamicwam.models.native_wam_teacher.sample_native_wam", side_effect=sample_teacher):
            trainer._attach_native_teacher(prepared)
        self.assertEqual(len(teacher_noises), 1)
        self.assertIs(teacher_noises[0].actions, fast_noise)
        self.assertIsNot(teacher_noises[0].actions, source_noise.actions)
        self.assertIs(source_calls[0][0], source)
        self.assertIs(source_calls[0][1], source_noise)
        self.assertIs(prepared["native_source_bridge"], bridge)
        model.native_phase = "distill"
        model.external_native_teacher = True
        model.native_teacher_weight = 1.
        with patch.object(model, "sample_fast_actions", wraps=model.sample_fast_actions) as sampler:
            output = model(prepared)
        self.assertIs(sampler.call_args.args[3], teacher_noises[0].actions)
        self.assertTrue(torch.isfinite(output["native_loss"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
