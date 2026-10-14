"""CPU-only native Color conversion and strict asynchronous batch contract."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np

from dynamicwam.runtime.domino_continuous.camera_capture import _read_submitted_rgb_batch


class NativeRGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from unittest.mock import patch
        # Import otherwise issues hidden nvidia-smi discovery calls. Prevent
        # those probes: numerical conversion requires no driver/device access.
        with patch("os.system", return_value=1):
            import sapien.pysapien
        path = os.environ.get("NATIVE_RGB_BATCH_LIBRARY")
        if path:
            spec = importlib.util.spec_from_file_location("_sapien_color_reader", path)
            cls.native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.native)
        else:
            from dynamicwam.runtime.domino_continuous import _sapien_color_reader
            cls.native = _sapien_color_reader

    def assert_equivalent(self, rgba):
        actual = self.native.rgba_to_rgb_u8(rgba)
        with np.errstate(over="ignore", invalid="ignore"):
            expected = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.owndata and actual.flags.c_contiguous)
        rgba[:] = 0
        np.testing.assert_array_equal(actual, expected)

    def test_random_float32_bit_patterns(self):
        rng = np.random.default_rng(736)
        rgba = rng.integers(0, 2**32, size=(257, 263, 4), dtype=np.uint32).view(np.float32)
        rgba[~np.isfinite(rgba)] = 0
        self.assert_equivalent(rgba)

    def test_truncation_boundaries_extremes_and_subnormal_values(self):
        centers = np.arange(256, dtype=np.float32) / np.float32(255)
        values = np.concatenate([centers, np.nextafter(centers, -np.inf),
            np.nextafter(centers, np.inf), np.array([-1e30, 1e30, -0.,
                np.finfo(np.float32).tiny, np.nextafter(np.float32(0), np.float32(1))], np.float32)])
        rgba = np.resize(values, (1, len(values), 4)).astype(np.float32)
        self.assert_equivalent(rgba)

    def test_nonfinite_including_alpha_is_rejected(self):
        for channel in range(4):
            for value in (np.nan, np.inf, -np.inf):
                rgba = np.zeros((2, 3, 4), np.float32)
                rgba[1, 2, channel] = value
                with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                    self.native.rgba_to_rgb_u8(rgba)

    def test_bad_shape_dtype_and_strides_are_rejected(self):
        for rgba in (np.zeros((2, 3, 3), np.float32), np.zeros((0, 3, 4), np.float32),
                     np.zeros((2, 3, 4), np.float64), np.zeros((4, 6, 4), np.float32)[::2, ::2]):
            with self.assertRaises((TypeError, RuntimeError)):
                self.native.rgba_to_rgb_u8(rgba)

    def test_batch_worker_is_one_call_in_preserved_camera_order(self):
        names = ("left", "right", "front", "head")
        handles = tuple(object() for _ in names)
        calls = []
        def read(received):
            calls.append(received)
            return [np.full((2, 3, 3), index, np.uint8) for index in range(4)], [.001] * 4
        output, _, _, durations = _read_submitted_rgb_batch(tuple(zip(names, handles)), read, lambda: 1.)
        self.assertEqual(calls, [handles])
        self.assertEqual(tuple(output), names)
        self.assertEqual(tuple(durations), names)
        for index, name in enumerate(names):
            np.testing.assert_array_equal(output[name]["rgb"], index)

    def test_selective_capture_uses_same_batch_reader_for_head_and_all_views(self):
        from dynamicwam.runtime.domino_continuous.test_camera_capture import Env
        from dynamicwam.runtime.domino_continuous.camera_stream import SelectiveAsyncCameraCapture
        env, calls = Env(), []
        def reader(cameras):
            calls.append(tuple(camera.name for camera in cameras))
            return [self.native.rgba_to_rgb_u8(camera.submitted) for camera in cameras], [0.] * len(cameras)
        with SelectiveAsyncCameraCapture(batch_reader=reader) as capture:
            capture.begin(env, 1., head_only=True)
            head = capture.finish()
            self.assertEqual(head.camera_names, ("head_camera",))
            capture.begin(env, 2., head_only=False)
            full = capture.finish()
            self.assertEqual(full.camera_names, tuple(env.cameras.names()))
        self.assertEqual(calls, [("head_camera",), tuple(env.cameras.names())])

    def test_batch_worker_rejects_bad_native_payload_without_fallback(self):
        cameras = (("head", object()),)
        for result in (([], []), ([np.zeros((2, 3, 4), np.uint8)], [.01]),
                       ([np.zeros((2, 3, 3), np.float32)], [.01]),
                       ([np.zeros((2, 3, 3), np.uint8)], [np.nan])):
            with self.assertRaises(ValueError):
                _read_submitted_rgb_batch(cameras, lambda _: result, lambda: 1.)


if __name__ == "__main__": unittest.main()
