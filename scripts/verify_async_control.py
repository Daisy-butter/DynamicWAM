#!/usr/bin/env python3
"""Unit checks for inference-only fast-slow control. No GPU or DOMINO required."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamicwam.evaluation.summarize_async_control import (  # noqa: E402
    summarize_control,
)
from dynamicwam.runtime.ciwam.wam.async_control import (  # noqa: E402
    AsyncControlConfig,
    FastSlowScheduler,
    default_async_config,
    load_async_control_config,
)
from dynamicwam.runtime.ciwam.wam.rtc import (  # noqa: E402
    apply_hard_prefix,
    boundary_delta,
    leftover_prefix,
    stitch_chunk,
)
from dynamicwam.runtime.ciwam.wam.world_buffer import WorldBridge  # noqa: E402
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler  # noqa: E402


def _dummy_cache(layers: int = 2) -> dict:
    return {
        "video_k": [torch.zeros(1, 4, 2, 8) for _ in range(layers)],
        "video_v": [torch.zeros(1, 4, 2, 8) for _ in range(layers)],
    }


def _chunk(start: float, length: int = 16, dim: int = 14) -> np.ndarray:
    values = np.arange(start, start + length, dtype=np.float32)
    return np.repeat(values[:, None], dim, axis=1)


class WorldBridgeTests(unittest.TestCase):
    def test_stale_until_written_then_ages(self) -> None:
        bridge = WorldBridge()
        self.assertTrue(bridge.stale(8))
        bridge.write(_dummy_cache(), simulator_time=1.5)
        self.assertFalse(bridge.stale(8))
        self.assertEqual(bridge.age_steps, 0)
        for _ in range(7):
            bridge.tick()
        self.assertFalse(bridge.stale(8))
        bridge.tick()
        self.assertTrue(bridge.stale(8))

    def test_reset_clears_cache(self) -> None:
        bridge = WorldBridge()
        bridge.write(_dummy_cache())
        bridge.tick()
        bridge.reset()
        self.assertFalse(bridge.has_cache())
        self.assertIsNone(bridge.frozen_motion())
        self.assertEqual(bridge.age_steps, 0)

    def test_motion_stays_aligned_with_video_cache(self) -> None:
        bridge = WorldBridge()
        features = np.zeros((4, 12), dtype=np.float32)
        features[-1, 2] = 4.0
        bridge.write(
            _dummy_cache(),
            motion_features=features,
            motion_interval_valid_mask=np.ones(4, dtype=np.bool_),
            motion_acceleration_valid_mask=np.ones(4, dtype=np.bool_),
        )
        frozen = bridge.frozen_motion()
        assert frozen is not None
        drifted = features.copy()
        drifted[-1, 2] = 12.0
        self.assertTrue(bridge.motion_drifted(drifted, ratio=0.5))
        self.assertFalse(bridge.motion_drifted(features, ratio=0.5))


class RTCTests(unittest.TestCase):
    def test_leftover_prefix_respects_delay(self) -> None:
        pending = [np.full(14, float(index), dtype=np.float32) for index in range(5)]
        prefix = leftover_prefix(pending, 2)
        assert prefix is not None
        self.assertEqual(prefix.shape, (2, 14))
        self.assertTrue(np.array_equal(prefix[0], pending[0]))
        self.assertIsNone(leftover_prefix(pending, 0))
        self.assertIsNone(leftover_prefix([], 2))

    def test_stitch_and_boundary_delta(self) -> None:
        leftover = _chunk(3.0, length=2)
        generated = _chunk(10.0, length=16)
        stitched = stitch_chunk(generated, leftover)
        self.assertTrue(np.array_equal(stitched[:2], leftover))
        self.assertTrue(np.array_equal(stitched[2:], generated[2:]))
        self.assertGreater(boundary_delta(stitched, leftover), 0.0)

    def test_hard_prefix_matches_add_noise(self) -> None:
        scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(num_inference_steps=4, training=False)
        prefix = torch.ones(1, 2, 14)
        noise = torch.full((1, 2, 14), 0.5)
        noisy = torch.randn(1, 16, 14)
        timestep = scheduler.timesteps[0]
        apply_hard_prefix(noisy, prefix, noise, scheduler, timestep)
        expected = scheduler.add_noise(prefix, noise, scheduler.timesteps[1])
        self.assertTrue(torch.allclose(noisy[:, :2], expected))


class SchedulerTests(unittest.TestCase):
    def test_open_loop_is_shorter_than_sync_chunk(self) -> None:
        async_cfg = default_async_config()
        scheduler = FastSlowScheduler(async_cfg, chunk_size=16)
        replan_gaps = []
        last_replan = 0
        for step in range(16):
            if scheduler.needs_replan():
                if step:
                    replan_gaps.append(step - last_replan)
                last_replan = step
                path = scheduler.choose_path()
                scheduler.commit_chunk(
                    _chunk(float(step)),
                    path=path,
                    leftover=scheduler.leftover(),
                    video_cache=_dummy_cache() if path == "slow" else None,
                )
            scheduler.pop_action()
        self.assertEqual(replan_gaps, [async_cfg.fast_horizon] * 3)
        self.assertLess(async_cfg.fast_horizon, 16)

    def test_default_config_refreshes_world_every_replan(self) -> None:
        scheduler = FastSlowScheduler(default_async_config(), chunk_size=16)
        paths = []
        for step in range(16):
            if scheduler.needs_replan():
                path = scheduler.choose_path()
                paths.append(path)
                scheduler.commit_chunk(
                    _chunk(float(step)),
                    path=path,
                    leftover=scheduler.leftover(),
                    video_cache=_dummy_cache() if path == "slow" else None,
                )
            scheduler.pop_action()
        self.assertEqual(paths, ["slow", "slow", "slow", "slow"])

    def test_replans_every_fast_horizon_and_prefers_fast_path(self) -> None:
        config = AsyncControlConfig(
            fast_horizon=4,
            max_world_age_steps=8,
            inference_delay=2,
        )
        scheduler = FastSlowScheduler(config, chunk_size=16)
        slow = 0
        fast = 0
        max_pending_after_replan = 0
        for step in range(24):
            if scheduler.needs_replan():
                leftover = scheduler.leftover()
                path = scheduler.choose_path()
                if path == "slow":
                    slow += 1
                    cache = _dummy_cache()
                else:
                    fast += 1
                    cache = None
                delta = scheduler.commit_chunk(
                    _chunk(float(step)),
                    path=path,
                    leftover=leftover,
                    video_cache=cache,
                    simulator_time=float(step),
                )
                self.assertGreaterEqual(delta, 0.0)
                max_pending_after_replan = max(
                    max_pending_after_replan, len(scheduler.pending)
                )
            scheduler.pop_action()
        self.assertEqual(slow, 3)
        self.assertEqual(fast, 3)
        self.assertEqual(max_pending_after_replan, 16)
        self.assertEqual(scheduler.steps_since_replan, 4)

    def test_commit_freezes_leftover_prefix(self) -> None:
        scheduler = FastSlowScheduler(
            AsyncControlConfig(inference_delay=2),
            chunk_size=16,
        )
        first = _chunk(0.0)
        scheduler.commit_chunk(
            first,
            path="slow",
            leftover=None,
            video_cache=_dummy_cache(),
        )
        for _ in range(4):
            scheduler.pop_action()
        leftover = scheduler.leftover()
        assert leftover is not None
        self.assertEqual(leftover.shape[0], 2)
        second = _chunk(40.0)
        scheduler.commit_chunk(
            second,
            path="fast",
            leftover=leftover,
        )
        executed = scheduler.pop_action()
        self.assertTrue(np.array_equal(executed, leftover[0]))

    def test_zero_delay_does_not_keep_leftover(self) -> None:
        scheduler = FastSlowScheduler(default_async_config(), chunk_size=16)
        scheduler.commit_chunk(
            _chunk(0.0),
            path="slow",
            leftover=None,
            video_cache=_dummy_cache(),
        )
        for _ in range(4):
            scheduler.pop_action()
        self.assertIsNone(scheduler.leftover())
        second = _chunk(40.0)
        scheduler.commit_chunk(
            second,
            path="slow",
            leftover=None,
            video_cache=_dummy_cache(),
        )
        executed = scheduler.pop_action()
        self.assertTrue(np.array_equal(executed, second[0]))


class ConfigTests(unittest.TestCase):
    def test_sidecar_roundtrip_and_defaults(self) -> None:
        defaults = default_async_config()
        self.assertEqual(defaults.fast_horizon, 4)
        self.assertEqual(defaults.max_world_age_steps, 4)
        self.assertEqual(defaults.inference_delay, 0)
        self.assertEqual(defaults.rtc_mode, "hard_prefix")
        self.assertEqual(defaults.motion_refresh_ratio, 0.5)
        sidecar = ROOT / "configs" / "absolute_motion_v2_async.yml"
        loaded = load_async_control_config(sidecar)
        self.assertEqual(loaded, defaults)

    def test_rejects_unknown_keys_and_bad_delay(self) -> None:
        with self.assertRaises(ValueError):
            load_async_control_config({"fast_horizon": 4, "vlm": True})
        with self.assertRaises(ValueError):
            AsyncControlConfig(fast_horizon=4, inference_delay=5)

    def test_sidecar_is_not_a_canonical_profile(self) -> None:
        text = (ROOT / "configs" / "absolute_motion_v2_async.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("schema_version", text)
        self.assertNotIn("policy_registration_name", text)
        official = (ROOT / "configs" / "absolute_motion_v2.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "ciwam.adapters.domino.deploy_policy_sync_flow",
            official,
        )

    def test_temp_file_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "async.yml"
            path.write_text("fast_horizon: 3\nmax_world_age_steps: 6\n", encoding="utf-8")
            loaded = load_async_control_config(path)
            self.assertEqual(loaded.fast_horizon, 3)
            self.assertEqual(loaded.max_world_age_steps, 6)
            self.assertEqual(loaded.inference_delay, 0)


class SummaryTests(unittest.TestCase):
    def test_fast_share_and_open_loop(self) -> None:
        metrics = summarize_control(
            [
                {
                    "open_loop_horizon": 4,
                    "slow_forwards": 2,
                    "fast_forwards": 6,
                    "mean_fast_ms": 80.0,
                    "mean_slow_ms": 170.0,
                    "max_boundary_delta": 0.04,
                },
                {
                    "open_loop_horizon": 4,
                    "slow_forwards": 2,
                    "fast_forwards": 6,
                    "mean_fast_ms": 90.0,
                    "mean_slow_ms": 180.0,
                    "max_boundary_delta": 0.02,
                },
            ]
        )
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["slow_forwards"], 4)
        self.assertEqual(metrics["fast_forwards"], 12)
        self.assertAlmostEqual(metrics["fast_share"], 0.75)
        self.assertEqual(metrics["open_loop_horizon"], 4)
        self.assertAlmostEqual(metrics["max_boundary_delta"], 0.04)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
