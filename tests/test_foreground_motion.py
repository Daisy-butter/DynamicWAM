"""Tests for vision-only moving-blob displacement statistics."""

from __future__ import annotations

import unittest

import numpy as np

from dynamicwam.absolute_motion import (
    FOREGROUND_MIN_PIXELS,
    displacement_statistics,
    flow_to_rgb,
    rgb_to_flow,
    select_foreground_blob,
)


def _blank(height: int = 64, width: int = 64) -> np.ndarray:
    return np.zeros((height, width, 2), dtype=np.float32)


def _paint_square(
    flow: np.ndarray,
    *,
    top: int,
    left: int,
    size: int,
    velocity: tuple[float, float],
) -> None:
    flow[top : top + size, left : left + size, 0] = velocity[0]
    flow[top : top + size, left : left + size, 1] = velocity[1]


class ForegroundDisplacementTests(unittest.TestCase):
    def test_object_mean_ignores_static_background(self) -> None:
        flow = _blank()
        _paint_square(flow, top=20, left=20, size=8, velocity=(3.0, 0.0))
        stats, centroid = displacement_statistics(flow, magnitude_percentile=99.0)
        self.assertAlmostEqual(float(stats[0]), 3.0, places=4)
        self.assertAlmostEqual(float(stats[1]), 0.0, places=4)
        self.assertGreater(float(stats[0]), float(flow[..., 0].mean()) * 10.0)
        self.assertIsNotNone(centroid)
        assert centroid is not None
        self.assertAlmostEqual(float(centroid[0]), 23.5, places=3)
        self.assertAlmostEqual(float(centroid[1]), 23.5, places=3)

    def test_higher_energy_blob_wins_without_prior(self) -> None:
        flow = _blank()
        _paint_square(flow, top=8, left=8, size=5, velocity=(10.0, 0.0))
        _paint_square(flow, top=40, left=40, size=10, velocity=(2.0, 0.0))
        stats, centroid = displacement_statistics(flow, magnitude_percentile=99.0)
        self.assertAlmostEqual(float(stats[0]), 10.0, places=4)
        assert centroid is not None
        self.assertLess(float(centroid[0]), 20.0)

    def test_previous_centroid_sticks_to_nearby_eligible_blob(self) -> None:
        flow = _blank()
        _paint_square(flow, top=8, left=8, size=6, velocity=(8.0, 0.0))
        _paint_square(flow, top=40, left=40, size=6, velocity=(6.0, 0.0))
        _mask, chosen = select_foreground_blob(
            flow,
            previous_centroid=np.asarray((43.0, 43.0)),
        )
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertGreater(float(chosen[0]), 35.0)

    def test_static_scene_falls_back_to_global_mean(self) -> None:
        flow = _blank()
        stats, centroid = displacement_statistics(flow, magnitude_percentile=99.0)
        self.assertTrue(np.allclose(stats, 0.0))
        self.assertIsNone(centroid)

    def test_too_few_pixels_falls_back_to_global_mean(self) -> None:
        flow = _blank()
        count = FOREGROUND_MIN_PIXELS - 1
        flow.reshape(-1, 2)[:count, 0] = 10.0
        stats, centroid = displacement_statistics(flow, magnitude_percentile=99.0)
        expected = 10.0 * count / flow.shape[0] / flow.shape[1]
        self.assertAlmostEqual(float(stats[0]), expected, places=5)
        self.assertIsNone(centroid)

    def test_rgb_roundtrip_recovers_moving_blob_direction(self) -> None:
        flow = _blank()
        _paint_square(flow, top=16, left=16, size=12, velocity=(4.0, -2.0))
        rgb = flow_to_rgb(flow, normalization_percentile=99.0)
        scale = float(np.percentile(np.linalg.norm(flow, axis=-1), 99.0))
        recovered = rgb_to_flow(rgb, magnitude_scale=scale)
        stats, _centroid = displacement_statistics(
            recovered,
            magnitude_percentile=99.0,
        )
        self.assertGreater(float(stats[0]), 3.0)
        self.assertLess(float(stats[1]), -1.0)


if __name__ == "__main__":
    unittest.main()
