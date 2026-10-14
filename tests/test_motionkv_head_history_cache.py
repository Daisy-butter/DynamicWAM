"""CPU regression: pair reuse preserves the full exact-time motion payload."""
from __future__ import annotations

import unittest
from unittest.mock import Mock
import numpy as np

from dynamicwam.runtime.domino_continuous.camera_stream import PhysicalTimeHeadHistory


CONFIG = {
    "count": 4, "policy_stride": 4, "compute_size": [64, 64],
    "normalization_percentile": 99.,
    "farneback": {"pyr_scale": .5, "levels": 3, "winsize": 15,
                  "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0},
    "quality": {"method": "forward_backward_consistency_v1", "relative_error": .01,
                "absolute_error_squared": .5, "minimum_reliable_fraction": .1},
}


class HeadHistoryCacheTests(unittest.TestCase):
    def histories(self, **kwargs):
        return [PhysicalTimeHeadHistory(CONFIG, frame_period_s=.16, **kwargs) for _ in range(2)]

    def assert_payload_equal(self, left, right):
        for name in ("flow_rgb", "motion_features", "interval_valid_mask", "acceleration_valid_mask"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))

    def test_actual_farneback_matches_uncached_for_irregular_and_missing_endpoints(self):
        cached, reference = self.histories()
        rng = np.random.default_rng(205)
        base = rng.integers(0, 256, size=(96, 112, 3), dtype=np.uint8)
        # Real optical flow and quality filtering, with irregular endpoints and
        # a missing interval. Reference clears the cache before every query.
        for index in range(47):
            if index in (17, 18, 19, 20, 21):
                continue
            timestamp = index * .16 + (0.007 if index % 3 else 0.)
            image = np.roll(base, index % 10, axis=1)
            for history in (cached, reference):
                history.push(image, simulator_time_seconds=timestamp)
            if index % 4 == 0 or index == 46:
                reference._pair_cache.clear()
                self.assert_payload_equal(cached.observation(), reference.observation())
                # Returned tensors must not alias retained cached RGB arrays.
                output = cached.observation()
                output.flow_rgb[:] = 0
                self.assert_payload_equal(cached.observation(), reference.observation())
        retained = {sample[2] for sample in cached._frames}
        self.assertTrue(all(a in retained and b in retained for a, b in cached._pair_cache))
        self.assertLessEqual(len(cached._pair_cache), cached._pair_cache_limit)

    def test_opencv_single_thread_preserves_actual_flow_payload(self):
        import cv2
        original_threads = cv2.getNumThreads()
        rng = np.random.default_rng(619)
        base = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        outputs = []
        try:
            for thread_count in (96, 1):
                cv2.setNumThreads(thread_count)
                history, _ = self.histories()
                for index in range(25):
                    frame = np.roll(base, (index, 2 * index), axis=(0, 1))
                    history.push(frame, simulator_time_seconds=index * .16)
                history._pair_cache.clear()
                outputs.append(history.observation())
            self.assert_payload_equal(*outputs)
        finally:
            cv2.setNumThreads(original_threads)

    def test_shifted_h4_query_computes_only_one_new_pair(self):
        history, _ = self.histories()
        actual_compute = history._compute
        history._compute = Mock(wraps=actual_compute)
        rng = np.random.default_rng(1)
        base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        for index in range(21):
            history.push(np.roll(base, index, axis=1), simulator_time_seconds=index * .16)
        history.observation()
        self.assertEqual(history._compute.call_count, 5)
        for index in range(21, 25):
            history.push(np.roll(base, index, axis=1), simulator_time_seconds=index * .16)
        history.observation()
        self.assertEqual(history._compute.call_count, 6)
        self.assertEqual(history.diagnostics["flow_pair_cache_hits"], 4)
        self.assertEqual(history.diagnostics["flow_pair_cache_misses"], 1)

    def test_reset_and_small_frame_ring_cannot_reuse_stale_pairs(self):
        cached, reference = self.histories(max_frames=6)
        for episode in range(2):
            for history in (cached, reference):
                history.reset()
                self.assertFalse(history._pair_cache)
            for index in range(12):
                image = np.full((64, 64, 3), episode * 80 + index, dtype=np.uint8)
                for history in (cached, reference):
                    history.push(image, simulator_time_seconds=index * .64)
                reference._pair_cache.clear()
                self.assert_payload_equal(cached.observation(), reference.observation())
                retained = {sample[2] for sample in cached._frames}
                self.assertTrue(all(a in retained and b in retained for a, b in cached._pair_cache))


if __name__ == "__main__":
    unittest.main()
