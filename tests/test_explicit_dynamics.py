"""Unit tests for image-plane Newtonian rollout."""

from __future__ import annotations

import math
import unittest

import torch

from dynamicwam.absolute_motion import MOTION_FEATURE_DIM
from dynamicwam.explicit_dynamics import (
    NEWTON_ACCELERATION_FEATURE_INDICES,
    NEWTON_FEATURE_DIM,
    rollout_image_plane_newton,
)


def _rollout(
    features: torch.Tensor,
    interval_valid: torch.Tensor,
    acceleration_valid: torch.Tensor,
):
    return rollout_image_plane_newton(
        features,
        interval_valid,
        acceleration_valid,
        action_interval_seconds=0.1,
        horizon_steps=16,
        window_count=4,
    )


def _set_interval(
    features: torch.Tensor,
    index: int,
    *,
    velocity: tuple[float, float],
    delta_t: float = 0.5,
) -> None:
    vx, vy = velocity
    speed = math.hypot(vx, vy)
    features[0, index, 4] = delta_t
    features[0, index, 5] = vx
    features[0, index, 6] = vy
    features[0, index, 7] = speed
    features[0, index, 8] = speed


def _uniform_history(
    *,
    velocity: tuple[float, float],
    history_count: int = 4,
    valid_count: int | None = None,
    p99_speed: float | None = None,
    delta_t: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.zeros(1, history_count, MOTION_FEATURE_DIM)
    interval_valid = torch.zeros(1, history_count, dtype=torch.bool)
    if valid_count is None:
        valid_count = history_count
    start = history_count - valid_count
    for index in range(start, history_count):
        _set_interval(features, index, velocity=velocity, delta_t=delta_t)
        interval_valid[0, index] = True
        if p99_speed is not None:
            features[0, index, 8] = p99_speed
    acceleration_valid = interval_valid.clone()
    if valid_count > 0:
        acceleration_valid[0, start] = False
    return features, interval_valid, acceleration_valid


class ExplicitDynamicsRolloutTests(unittest.TestCase):
    def test_constant_acceleration_end_state(self) -> None:
        features = torch.zeros(1, 4, MOTION_FEATURE_DIM)
        interval_valid = torch.ones(1, 4, dtype=torch.bool)
        acceleration_valid = torch.tensor([[False, True, True, True]])
        v0 = torch.tensor([2.0, -1.0])
        acceleration = torch.tensor([0.5, 0.25])
        delta_t = 0.5
        remaining = [2.0, 1.5, 1.0, 0.5]
        for index, duration in enumerate(remaining):
            t_center = -duration + 0.5 * delta_t
            velocity = (v0 + acceleration * t_center).tolist()
            _set_interval(features, index, velocity=tuple(velocity), delta_t=delta_t)
        newton, window_valid, accel_valid = _rollout(
            features,
            interval_valid,
            acceleration_valid,
        )
        self.assertEqual(tuple(newton.shape), (1, 4, NEWTON_FEATURE_DIM))
        self.assertTrue(bool(window_valid.all()))
        self.assertTrue(bool(accel_valid.all()))
        t_end = torch.tensor([0.4, 0.8, 1.2, 1.6])
        vx, vy = 2.0, -1.0
        ax, ay = 0.5, 0.25
        for index, time in enumerate(t_end.tolist()):
            px = vx * time + 0.5 * ax * time * time
            py = vy * time + 0.5 * ay * time * time
            self.assertAlmostEqual(float(newton[0, index, 0]), px, places=4)
            self.assertAlmostEqual(float(newton[0, index, 1]), py, places=4)

    def test_invalid_history_is_zero(self) -> None:
        features, interval_valid, acceleration_valid = _uniform_history(
            velocity=(4.0, 4.0),
            valid_count=0,
        )
        newton, window_valid, accel_valid = _rollout(
            features,
            interval_valid,
            acceleration_valid,
        )
        self.assertFalse(bool(window_valid.any()))
        self.assertFalse(bool(accel_valid.any()))
        self.assertTrue(torch.equal(newton, torch.zeros_like(newton)))

    def test_single_interval_keeps_constant_velocity(self) -> None:
        features, interval_valid, acceleration_valid = _uniform_history(
            velocity=(1.5, -2.0),
            valid_count=1,
        )
        newton, window_valid, accel_valid = _rollout(
            features,
            interval_valid,
            acceleration_valid,
        )
        self.assertTrue(bool(window_valid.all()))
        self.assertFalse(bool(accel_valid.any()))
        t_end = torch.tensor([0.4, 0.8, 1.2, 1.6])
        speed = math.hypot(1.5, -2.0)
        for index, time in enumerate(t_end.tolist()):
            self.assertAlmostEqual(float(newton[0, index, 4]), 1.5 * time, places=5)
            self.assertAlmostEqual(float(newton[0, index, 5]), -2.0 * time, places=5)
            self.assertAlmostEqual(float(newton[0, index, 11]), speed, places=5)
            for feature_index in NEWTON_ACCELERATION_FEATURE_INDICES:
                self.assertEqual(float(newton[0, index, feature_index]), 0.0)

    def test_object_scale_uses_p99_speed(self) -> None:
        features, interval_valid, acceleration_valid = _uniform_history(
            velocity=(3.0, 0.0),
            valid_count=1,
            p99_speed=12.0,
        )
        newton, _, _ = _rollout(features, interval_valid, acceleration_valid)
        self.assertAlmostEqual(float(newton[0, -1, 4]), 12.0 * 1.6, places=5)
        self.assertAlmostEqual(float(newton[0, -1, 5]), 0.0, places=5)
        self.assertAlmostEqual(float(newton[0, -1, 11]), 12.0, places=5)

    def test_constant_turn_matches_circular_displacement(self) -> None:
        omega = 0.5
        v0 = torch.tensor([4.0, 0.0])
        features = torch.zeros(1, 4, MOTION_FEATURE_DIM)
        interval_valid = torch.ones(1, 4, dtype=torch.bool)
        acceleration_valid = torch.tensor([[False, True, True, True]])
        delta_t = 0.5
        remaining = [2.0, 1.5, 1.0, 0.5]
        for index, duration in enumerate(remaining):
            t_center = -duration + 0.5 * delta_t
            angle = omega * t_center
            velocity = (
                math.cos(angle) * float(v0[0]) - math.sin(angle) * float(v0[1]),
                math.sin(angle) * float(v0[0]) + math.cos(angle) * float(v0[1]),
            )
            _set_interval(features, index, velocity=velocity, delta_t=delta_t)
        newton, _, accel_valid = _rollout(features, interval_valid, acceleration_valid)
        self.assertTrue(bool(accel_valid.all()))
        t_end = torch.tensor([0.4, 0.8, 1.2, 1.6])
        for index, time in enumerate(t_end.tolist()):
            px = math.sin(omega * time) / omega * float(v0[0])
            py = (1.0 - math.cos(omega * time)) / omega * float(v0[0])
            self.assertAlmostEqual(float(newton[0, index, 2]), px, places=3)
            self.assertAlmostEqual(float(newton[0, index, 3]), py, places=3)
            self.assertAlmostEqual(float(newton[0, index, 9]), omega * time, places=3)
        self.assertLess(float(newton[0, 0, 7]), float(newton[0, 0, 6]))

    def test_zero_omega_constant_turn_matches_constant_velocity(self) -> None:
        features, interval_valid, acceleration_valid = _uniform_history(
            velocity=(2.0, -1.0),
        )
        newton, _, accel_valid = _rollout(features, interval_valid, acceleration_valid)
        self.assertTrue(bool(accel_valid.all()))
        torch.testing.assert_close(
            newton[0, :, 2:4],
            newton[0, :, 4:6],
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertAlmostEqual(float(newton[0, -1, 9]), 0.0, places=5)

    def test_least_squares_uses_all_windows(self) -> None:
        features = torch.zeros(1, 4, MOTION_FEATURE_DIM)
        interval_valid = torch.ones(1, 4, dtype=torch.bool)
        acceleration_valid = torch.tensor([[False, True, True, True]])
        v0 = torch.tensor([1.0, 0.0])
        acceleration = torch.tensor([2.0, 0.0])
        delta_t = 0.5
        remaining = [2.0, 1.5, 1.0, 0.5]
        for index, duration in enumerate(remaining):
            t_center = -duration + 0.5 * delta_t
            velocity = (v0 + acceleration * t_center).tolist()
            _set_interval(features, index, velocity=tuple(velocity), delta_t=delta_t)
        last_only = features.clone()
        last_valid = torch.tensor([[False, False, False, True]])
        last_accel = torch.tensor([[False, False, False, True]])
        full, _, full_accel = _rollout(features, interval_valid, acceleration_valid)
        last, _, last_accel_mask = _rollout(last_only, last_valid, last_accel)
        self.assertTrue(bool(full_accel.all()))
        self.assertFalse(bool(last_accel_mask.any()))
        expected = 1.0 * 1.6 + 0.5 * 2.0 * 1.6 * 1.6
        self.assertAlmostEqual(float(full[0, -1, 0]), expected, places=4)
        self.assertNotAlmostEqual(float(full[0, -1, 0]), float(last[0, -1, 4]), places=3)

    def test_linear_motion_has_high_tangential_fraction(self) -> None:
        features = torch.zeros(1, 4, MOTION_FEATURE_DIM)
        interval_valid = torch.ones(1, 4, dtype=torch.bool)
        acceleration_valid = torch.tensor([[False, True, True, True]])
        v0 = torch.tensor([2.0, 0.0])
        acceleration = torch.tensor([1.0, 0.0])
        delta_t = 0.5
        remaining = [2.0, 1.5, 1.0, 0.5]
        for index, duration in enumerate(remaining):
            t_center = -duration + 0.5 * delta_t
            velocity = (v0 + acceleration * t_center).tolist()
            _set_interval(features, index, velocity=tuple(velocity), delta_t=delta_t)
        newton, _, _ = _rollout(features, interval_valid, acceleration_valid)
        self.assertGreater(float(newton[0, 0, 10]), 0.9)
        self.assertLess(float(newton[0, 0, 6]), 1e-3)


if __name__ == "__main__":
    unittest.main()
