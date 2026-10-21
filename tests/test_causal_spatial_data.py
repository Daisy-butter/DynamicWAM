"""CPU causality/geometry tests; torch/OpenCV integration tests require runtime deps."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import unittest
import sys

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "src/dynamicwam/training/data/causal_spatial_dataset.py"
sys.path.insert(0, str(PATH.parents[3]))
SPEC = importlib.util.spec_from_file_location("causal_spatial_data_under_test", PATH)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None


class PhysicalWindowTests(unittest.TestCase):
    def test_actual_time_and_no_future_equal_timestamp_frame(self):
        indices, times, dt, valid = data.causal_window_indices([0, .1, .2, .2, .33, .7], 2,
            window_size=3, stride_seconds=.1)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_allclose(times, [0, .1, .2])
        np.testing.assert_allclose(dt, [0, .1, .1])
        np.testing.assert_array_equal(valid, [False, True, True])

    def test_bootstrap_repeats_are_not_dynamics_targets(self):
        indices, _, dt, valid = data.causal_window_indices([0, .11, .2], 1,
            window_size=4, stride_seconds=.32)
        np.testing.assert_array_equal(indices, [0, 0, 0, 1])
        np.testing.assert_allclose(dt, [0, 0, 0, .11])
        np.testing.assert_array_equal(valid, [False, False, False, True])

    def test_capture_jitter_does_not_become_nominal_dt(self):
        _, times, dt, _ = data.causal_window_indices([0, .13, .31, .5, .67, .85, 1.04], 6,
            window_size=4, stride_seconds=.32)
        np.testing.assert_allclose(times, [0, .31, .67, 1.04])
        np.testing.assert_allclose(dt, [0, .31, .36, .37])

    def test_invalid_clock_rejected(self):
        for times in ([0, float("nan")], [0, -.1]):
            with self.assertRaises(ValueError):
                data.causal_window_indices(times, 1)
        with self.assertRaises(ValueError):
            data.causal_window_indices([0, .1], 2)

    def test_latents_use_actual_group_support_end(self):
        offsets = data.future_support_end_offsets([1.1, 1.2, 1.31, 1.42, 1.51, 1.6, 1.71, 1.82], 1.)
        np.testing.assert_allclose(offsets, [.42, .82])
        with self.assertRaises(ValueError):
            data.future_support_end_offsets(np.arange(8), 4)

    def test_contract_is_honest_and_record_buckets_unmodified(self):
        class Base:
            pairing_identity = "verified"
            max_supported_cache_age_actions = 31
            records = [1, 2]
            def __len__(self): return 2
            def set_epoch(self, epoch): self.epoch = epoch
        base = Base()
        wrapped = data.CausalSpatialDataset(base)
        wrapped.set_force_fresh_memory(True)
        wrapped.set_epoch(5)
        self.assertEqual(base.max_supported_cache_age_actions, 31)
        self.assertEqual(wrapped.records, [1, 2])
        self.assertEqual(base.epoch, 5)
        self.assertFalse(wrapped.contract["on_policy_ready"])
        self.assertFalse(wrapped.contract["counterfactual_ready"])
        self.assertEqual(wrapped.contract["command_history_semantics"], data.COMMAND_HISTORY_SEMANTICS)

    def test_flow_mask_checks_warped_previous_image_not_same_pixel(self):
        previous = np.ones((5, 6), dtype=bool); previous[:, :2] = False
        current = np.ones((5, 6), dtype=bool)
        delta = np.zeros((5, 6, 2), dtype=np.float32); delta[..., 0] = 1.5
        mask = data.flow_pair_validity(delta, current, previous, current)
        self.assertFalse(mask[:, :4].any())
        self.assertTrue(mask[:, 4:].all())

    def test_plan_stays_recent_when_memory_is_31_actions_old(self):
        times = np.arange(70) * .16
        eligible = np.stack([np.arange(70), np.arange(100, 170)], axis=-1)
        position = data.causal_plan_position(times, eligible, 31)
        self.assertEqual(position, 29)
        self.assertAlmostEqual(times[31] - times[position], .32)
        self.assertIsNone(data.causal_plan_position(times, eligible, 1))


@dataclass
class Record:
    episode_index: int = 0
    source_sample_id: int = 11
    fast_sample_id: int = 15
    source_frame: int = 1
    fast_frame: int = 5


@unittest.skipUnless(HAS_TORCH, "torch is required for real dataset tensor integration")
class DatasetIntegrationTests(unittest.TestCase):
    def make_dataset(self, **kwargs):
        import torch
        class Normalizer:
            def normalize(self, value):
                assert value.dtype == torch.float32
                return value * 2
        class Base:
            pairing_identity = "fixture-grid"
            normalizer = Normalizer()
            def __len__(self): return 2
            def _make_record(self, index): return self.record
            def _episode(self, index):
                return dict(times=np.arange(20) * .16,
                    eligible=np.stack([np.arange(20), np.arange(10, 30)], axis=-1),
                    action_times=(np.arange(20)[:, None] + np.arange(1, 17)[None, :]) * .16,
                    commands=np.repeat(np.arange(20, dtype=np.float32)[:, None], 14, axis=1))
            def _read_frame(self, record):
                self.read_frames.append(record.fast_frame)
                return np.full((4, 4, 3), record.fast_frame, dtype=np.uint8)
            def __getitem__(self, index):
                return dict(
                    source_condition_latent=torch.full((48, 2, 24, 20), -1.),
                    fast_condition_latent=torch.full((48, 2, 24, 20), 5.),
                    source_state=torch.ones(14), fast_state=torch.ones(14) * 5,
                    source_motion=torch.ones(4, 12), fast_motion=torch.ones(4, 12) * 5,
                    source_motion_interval_valid_mask=torch.ones(4, dtype=torch.bool),
                    fast_motion_interval_valid_mask=torch.ones(4, dtype=torch.bool),
                    source_motion_acceleration_valid_mask=torch.ones(4, dtype=torch.bool),
                    fast_motion_acceleration_valid_mask=torch.ones(4, dtype=torch.bool),
                    fast_action_sequence=torch.arange(16 * 14, dtype=torch.float32).reshape(16, 14),
                    fast_future_latent=torch.ones(48, 2, 12, 10) * 99,
                    text_embedding=torch.ones(2 + index, 8),
                    joint_targets=dict(
                        source_action_target_times=torch.arange(2, 18, dtype=torch.float64) * .16,
                        fast_action_target_times=torch.arange(6, 22, dtype=torch.float64) * .16,
                        fast_video_actual_times=.8 + torch.arange(1, 9, dtype=torch.float64) * .32 - self.future_lag))
        base = Base()
        base.read_frames = []
        base.record = Record()
        base.future_lag = .02
        class Packed:
            def __getitem__(self, sample_id):
                frame = sample_id - 10
                return dict(episode_index=0, condition_latent=torch.full((48, 2, 24, 20), float(frame)),
                    initial_state=torch.full((14,), float(frame)), absolute_motion_features=torch.full((4, 12), float(frame)),
                    absolute_motion_interval_valid_mask=torch.ones(4, dtype=torch.bool),
                    absolute_motion_acceleration_valid_mask=torch.ones(4, dtype=torch.bool))
        base.packed = Packed()
        def observations(frame):
            return (np.stack([frame.transpose(2, 0, 1)] * 3).astype(np.float32),
                np.stack([frame] * 3), np.ones((3, 4, 4), dtype=bool))
        def flow(previous, current):
            delta = np.zeros((4, 4, 2), dtype=np.float32)
            delta[..., 0] = float(current[0, 0, 0]) - float(previous[0, 0, 0])
            return delta, np.ones((4, 4), dtype=bool)
        return data.CausalSpatialDataset(base, observation_backend=observations, flow_backend=flow, **kwargs)

    def test_only_causal_images_and_actual_transition_commands(self):
        import torch
        ds = self.make_dataset()
        sample = ds[0]
        self.assertEqual(ds.base.read_frames, [0, 1, 3, 5])
        self.assertEqual(tuple(sample["spatial_rgb"].shape), (4, 3, 3, 4, 4))
        self.assertEqual(tuple(sample["spatial_flow"].shape), (4, 3, 2, 4, 4))
        self.assertTrue(sample["spatial_valid"][0].all())
        self.assertFalse(sample["spatial_flow_valid"][0].any())
        torch.testing.assert_close(sample["spatial_dt"], torch.tensor([0, .16, .32, .32], dtype=torch.float64))
        torch.testing.assert_close(sample["executed_actions"][:, 0], torch.tensor([0., 0., 2., 6.]))
        self.assertFalse(sample["previous_plan_valid"].any())
        self.assertNotIn("future_latent", sample)
        self.assertNotIn("noisy_actions", sample)

    def test_h2_prefix_and_future_supervision(self):
        import torch
        sample = self.make_dataset()[0]
        torch.testing.assert_close(sample["rtc_prefix"][:2], sample["clean_actions"][:2])
        self.assertEqual(sample["rtc_prefix_lengths"], 2)
        self.assertEqual(sample["rtc_prefix"][2:].count_nonzero(), 0)
        torch.testing.assert_close(sample["future_target_offsets"], torch.tensor([1.28, 2.56], dtype=torch.float64))
        torch.testing.assert_close(sample["future_support_lag"], torch.tensor([.02, .02], dtype=torch.float64))
        self.assertEqual(tuple(sample["clean_future"].shape), (48, 2, 12, 10))
        self.assertEqual(sample["source_condition_latent"].mean(), -1.)
        self.assertEqual(sample["clean_future"].mean(), 99.)

    def test_fresh_memory_overrides_complete_source_bundle(self):
        import torch
        ds = self.make_dataset(force_fresh_memory=True)
        sample = ds[0]
        self.assertEqual(float(sample["observation_age"]), 0.)
        self.assertEqual(sample["source_id"], sample["current_id"])
        torch.testing.assert_close(sample["source_condition_latent"], sample["clean_current_condition"])
        torch.testing.assert_close(sample["source_state"], sample["initial_state"])
        torch.testing.assert_close(sample["source_motion"], sample["absolute_motion_features"])
        torch.testing.assert_close(sample["source_action_target_offsets"], sample["action_target_offsets"])

    def test_collate_variable_language_and_exact_shapes(self):
        ds = self.make_dataset()
        batch = data.collate_causal_spatial([ds[0], ds[1]])
        self.assertEqual(tuple(batch["spatial_valid"].shape), (2, 4, 3, 1, 4, 4))
        self.assertEqual(tuple(batch["clean_future"].shape), (2, 48, 2, 12, 10))
        self.assertEqual(len(batch["source_text_embeddings"]), 2)
        self.assertEqual(batch["source_text_embeddings"][1].shape[0], 3)
        self.assertEqual(batch["command_history_semantics"], data.COMMAND_HISTORY_SEMANTICS)

    def test_plan_observation_is_recent_and_independent_of_slow_memory(self):
        ds = self.make_dataset(force_fresh_memory=True)
        sample = ds[0]
        self.assertEqual(sample["source_id"], 15)
        self.assertEqual(sample["plan_id"], 13)
        self.assertTrue(sample["plan_context_valid"])
        self.assertAlmostEqual(float(sample["plan_time"]), .48)
        self.assertEqual(float(sample["plan_condition_latent"].mean()), 3.)
        self.assertFalse(any(key.startswith("plan_") and "future" in key for key in sample))

    def test_bootstrap_plan_is_absent_not_current_observation(self):
        ds = self.make_dataset()
        ds.base.record = Record(source_sample_id=10, fast_sample_id=10, source_frame=0, fast_frame=0)
        sample = ds[0]
        self.assertFalse(sample["plan_context_valid"])
        self.assertEqual(sample["plan_id"], -1)
        self.assertEqual(sample["plan_condition_latent"].count_nonzero(), 0)
        self.assertEqual(sample["plan_action_target_times"].count_nonzero(), 0)

    def test_future_capture_jitter_does_not_condition_query_timestamps(self):
        import torch
        ds = self.make_dataset()
        first = ds[0]
        ds.base.future_lag = .09
        second = ds[0]
        torch.testing.assert_close(first["future_target_offsets"], second["future_target_offsets"])
        self.assertFalse(torch.equal(first["future_actual_support_offsets"], second["future_actual_support_offsets"]))
        ds.base.future_lag = .17
        with self.assertRaises(ValueError): ds[0]


@unittest.skipUnless(HAS_CV2, "OpenCV is required for the actual dense-flow test")
class DenseFlowTests(unittest.TestCase):
    def test_translation_direction_current_grid_and_validity(self):
        import cv2
        rng = np.random.default_rng(20261021)
        previous = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1, 0, 3], [0, 1, 0]]), (80, 80))
        flow, valid = data.farneback_pair(previous, current)
        crop = np.zeros((80, 80), dtype=bool); crop[16:-16, 16:-16] = True
        self.assertGreater(int((valid & crop).sum()), 1000)
        self.assertAlmostEqual(float(np.median(flow[..., 0][valid & crop])), 3., delta=.3)
        self.assertAlmostEqual(float(np.median(flow[..., 1][valid & crop])), 0., delta=.3)
        self.assertTrue(np.isfinite(flow).all())


if __name__ == "__main__":
    unittest.main()
