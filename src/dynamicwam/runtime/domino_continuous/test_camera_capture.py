"""Real-thread CPU tests for frozen asynchronous four-camera observations."""
from __future__ import annotations

from concurrent.futures import TimeoutError
import threading
import time
import unittest

import numpy as np

try:
    from .camera_capture import AsyncCameraCapture
except ImportError:
    from camera_capture import AsyncCameraCapture


class Camera:
    def __init__(self, name, owner, log, source):
        self.name, self.owner, self.log = name, owner, log
        self.source = source
        self.submitted = None

    def take_picture(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker attempted render submission")
        self.log.append(("take_picture", self.name, threading.get_ident()))
        self.submitted = self.source.copy()


class Manager:
    def __init__(self, owner, log):
        self.owner, self.log = owner, log
        self.collect_wrist_camera = self.collect_head_camera = True
        self.source = np.array([[[0.1, -0.2, 1.2, 0.5], [0.001, .999, .3, 1.]]], dtype=np.float32)
        self.left_camera = Camera("left_camera", owner, log, self.source)
        self.right_camera = Camera("right_camera", owner, log, self.source)
        self.static_camera_name = ["front_camera", "head_camera"]
        self.static_camera_list = [Camera(name, owner, log, self.source) for name in self.static_camera_name]
        self.config = {name: {"intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4),
                              "cam2world_gl": np.eye(4)} for name in self.names()}

    def names(self):
        return ["left_camera", "right_camera", *self.static_camera_name]

    def update_picture(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker accessed camera manager")
        self.log.append(("update_picture", "", threading.get_ident()))
        for camera in [self.left_camera, self.right_camera, *self.static_camera_list]:
            camera.take_picture()

    def get_config(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker accessed camera transforms")
        self.log.append(("get_config", "", threading.get_ident()))
        return self.config


class Robot:
    def __init__(self, owner, log):
        self.owner, self.log = owner, log
        self.left = [0., 1., 2., 3., 4., 5., .4]
        self.right = [6., 7., 8., 9., 10., 11., .6]

    def access(self, name, value):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker accessed mutable robot state")
        self.log.append((name, "", threading.get_ident()))
        return value

    def get_left_arm_jointState(self):
        return self.access("left_qpos", self.left)

    def get_right_arm_jointState(self):
        return self.access("right_qpos", self.right)

    def get_left_gripper_val(self):
        return self.access("left_gripper", self.left[-1])

    def get_right_gripper_val(self):
        return self.access("right_gripper", self.right[-1])


class Env:
    def __init__(self):
        self.owner = threading.get_ident()
        self.log = []
        self.data_type = {"rgb": True, "qpos": True, "endpose": True,
                          "depth": False, "pointcloud": False, "third_view": False,
                          "mesh_segmentation": False, "actor_segmentation": False}
        self.cameras = Manager(self.owner, self.log)
        self.robot = Robot(self.owner, self.log)
        self.pose = {"left": np.arange(7, dtype=float), "right": np.arange(7, dtype=float) + 20}
        self.now_obs = {"sentinel": "untouched"}
        self.physics = 2.0

    def _update_render(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker updated scene rendering")
        self.log.append(("update_render", "", threading.get_ident()))

    def get_arm_pose(self, arm):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker accessed entity transforms")
        self.log.append(("arm_pose", arm, threading.get_ident()))
        return self.pose[arm]

    def step(self):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker stepped physics")
        self.physics += .004
        self.robot.left[0] += 1
        self.robot.right[0] += 1
        self.pose["left"][0] += 1
        self.cameras.config["head_camera"]["cam2world_gl"][0, 3] += 1
        self.cameras.source[0, 0, 0] += .01


class Reader:
    def __init__(self, *, blocked=False, failure=None, bad_result=None):
        self.started = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.failure, self.bad_result = failure, bad_result
        self.calls = []

    def __call__(self, camera):
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release fake native reader")
        self.calls.append((camera.name, threading.get_ident()))
        camera.log.append(("read_color", camera.name, threading.get_ident()))
        if self.failure is not None:
            raise self.failure
        rgba = camera.submitted.copy()
        return self.bad_result(rgba) if self.bad_result else rgba


class CameraCaptureTests(unittest.TestCase):
    def test_frozen_state_timestamp_pixels_and_physics_continue_while_pending(self):
        env, reader = Env(), Reader(blocked=True)
        with AsyncCameraCapture(reader=reader) as capture:
            pending = capture.begin(env, physics_time_s=env.physics)
            self.assertTrue(reader.started.wait(1))
            self.assertFalse(capture.ready(pending))
            self.assertIsNone(capture.poll(pending))
            for _ in range(25):
                env.step()
            self.assertGreater(env.physics, 2.09)
            self.assertFalse(capture.ready(pending))
            reader.release.set()
            result = capture.finish(pending, timeout=1)
        observation = result.observation
        self.assertEqual(result.physics_time_s, 2.0)
        self.assertEqual(observation["joint_action"]["vector"].tolist(),
                         [0., 1., 2., 3., 4., 5., .4, 6., 7., 8., 9., 10., 11., .6])
        self.assertEqual(observation["endpose"]["left_endpose"][0], 0)
        self.assertEqual(observation["observation"]["head_camera"]["cam2world_gl"][0, 3], 0)
        original_rgba = np.array([[[0.1, -0.2, 1.2, 0.5], [0.001, .999, .3, 1.]]], dtype=np.float32)
        expected = (original_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
        self.assertEqual(list(observation["observation"]), env.cameras.names())
        for camera in result.camera_names:
            np.testing.assert_array_equal(observation["observation"][camera]["rgb"], expected)
        self.assertEqual(env.now_obs, {"sentinel": "untouched"})
        self.assertEqual(observation["pointcloud"], [])
        self.assertEqual(result.camera_names, tuple(env.cameras.names()))
        self.assertLessEqual(result.capture_wall_perf_s, result.render_submitted_wall_perf_s)
        self.assertLessEqual(result.render_submitted_wall_perf_s, result.readback_started_wall_perf_s)
        self.assertLessEqual(result.readback_finished_wall_perf_s, result.finished_wall_perf_s)

    def test_exact_submission_order_and_worker_only_reads_color(self):
        env, reader = Env(), Reader()
        with AsyncCameraCapture(reader=reader) as capture:
            capture.finish(capture.begin(env, env.physics), timeout=1)
        operations = [entry[0] for entry in env.log]
        self.assertEqual(operations[:6], ["update_render", "update_picture", *(["take_picture"] * 4)])
        first_read = operations.index("read_color")
        self.assertLess(operations.index("right_qpos"), first_read)
        self.assertEqual([name for name, _ in reader.calls], env.cameras.names())
        for operation, _name, thread_id in env.log:
            self.assertEqual(thread_id == env.owner, operation != "read_color")

    def test_overlapping_begin_rejected_even_when_result_ready_but_unconsumed(self):
        env, reader = Env(), Reader(blocked=True)
        with AsyncCameraCapture(reader=reader) as capture:
            pending = capture.begin(env, env.physics)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                capture.begin(env, env.physics)
            reader.release.set()
            pending._future.result(timeout=1)
            self.assertTrue(capture.ready(pending))
            self.assertTrue(capture.ready(pending))
            with self.assertRaisesRegex(RuntimeError, "pending"):
                capture.begin(env, env.physics)
            capture.finish(pending)
            self.assertFalse(capture.busy)
            env.step()
            second = capture.begin(env, env.physics)
            self.assertEqual(second.capture_id, pending.capture_id + 1)
            self.assertEqual(capture.finish(second, timeout=1).physics_time_s, env.physics)
        self.assertEqual(sum(row[0] == "update_render" for row in env.log), 2)

    def test_timeout_keeps_future_owned_until_drained(self):
        env, reader = Env(), Reader(blocked=True)
        with AsyncCameraCapture(reader=reader) as capture:
            pending = capture.begin(env, env.physics)
            with self.assertRaises(TimeoutError):
                capture.finish(pending, timeout=.01)
            self.assertTrue(capture.busy)
            with self.assertRaisesRegex(RuntimeError, "pending"):
                capture.begin(env, env.physics)
            reader.release.set()
            self.assertEqual(capture.drain(timeout=1).physics_time_s, env.physics)
            self.assertIsNone(capture.drain())

    def test_terminal_close_waits_for_all_reads_before_environment_destruction(self):
        env, reader = Env(), Reader(blocked=True)
        capture = AsyncCameraCapture(reader=reader)
        pending = capture.begin(env, env.physics)
        self.assertTrue(reader.started.wait(1))
        timer = threading.Timer(.03, reader.release.set)
        timer.start()
        try:
            capture.close()
        finally:
            timer.join()
        self.assertTrue(pending.done)
        self.assertEqual(len(reader.calls), 4)
        self.assertFalse(capture.busy)
        self.assertTrue(all(not thread.is_alive() for thread in capture._executor._threads))
        capture.close()  # Idempotent teardown.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            capture.begin(env, env.physics)

    def test_lifecycle_must_stay_on_physics_owner(self):
        env, reader = Env(), Reader()
        errors = []
        with AsyncCameraCapture(reader=reader) as capture:
            pending = capture.begin(env, env.physics)
            def foreign_thread():
                for operation in (lambda: capture.ready(pending), lambda: capture.finish(pending),
                                  capture.close, lambda: capture.begin(env, 3)):
                    try:
                        operation()
                    except RuntimeError as exc:
                        errors.append(str(exc))
            thread = threading.Thread(target=foreign_thread)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 4)
            self.assertTrue(all("owner" in error for error in errors))
            capture.finish(pending, timeout=1)

    def test_native_failure_is_propagated_and_close_joins(self):
        env, reader = Env(), Reader(failure=RuntimeError("native transfer failed"))
        capture = AsyncCameraCapture(reader=reader)
        capture.begin(env, env.physics)
        with self.assertRaisesRegex(RuntimeError, "native transfer failed"):
            capture.close()
        self.assertFalse(capture.busy)
        self.assertTrue(all(not thread.is_alive() for thread in capture._executor._threads))

    def test_invalid_native_array_never_enters_observation(self):
        corruptions = [lambda a: a.astype(np.float64), lambda a: a[:, :, :3].copy(),
                       lambda a: a.view(), lambda a: np.full_like(a, np.nan)]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                env, reader = Env(), Reader(bad_result=corrupt)
                with AsyncCameraCapture(reader=reader) as capture:
                    pending = capture.begin(env, env.physics)
                    with self.assertRaises((TypeError, ValueError)):
                        capture.finish(pending, timeout=1)
                    self.assertFalse(capture.busy)
                    self.assertEqual(env.now_obs, {"sentinel": "untouched"})

    def test_unsupported_modes_rejected_before_any_render(self):
        for mode in ("depth", "mesh_segmentation", "actor_segmentation", "pointcloud", "third_view", "observer"):
            with self.subTest(mode=mode):
                env = Env()
                env.data_type[mode] = True
                with AsyncCameraCapture(reader=Reader()) as capture:
                    with self.assertRaisesRegex(ValueError, "does not support"):
                        capture.begin(env, env.physics)
                self.assertEqual(env.log, [])

    def test_no_camera_can_be_silently_removed_and_state_is_required(self):
        mutations = [lambda e: e.data_type.update(rgb=False), lambda e: e.data_type.update(qpos=False),
                     lambda e: setattr(e.cameras, "collect_head_camera", False),
                     lambda e: e.cameras.static_camera_list.pop(),
                     lambda e: e.cameras.static_camera_name.__setitem__(0, "head_camera")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                env = Env()
                mutate(env)
                with AsyncCameraCapture(reader=Reader()) as capture:
                    with self.assertRaises(ValueError):
                        capture.begin(env, env.physics)
                self.assertEqual(env.log, [])

    def test_disabled_endpose_preserves_original_empty_dictionary(self):
        env = Env()
        env.data_type["endpose"] = False
        with AsyncCameraCapture(reader=Reader()) as capture:
            result = capture.finish(capture.begin(env, 0.0), timeout=1)
        self.assertEqual(result.observation["endpose"], {})

    def test_consumed_or_foreign_tokens_cannot_be_reused(self):
        env = Env()
        with AsyncCameraCapture(reader=Reader()) as capture:
            token = capture.begin(env, 0.0)
            capture.finish(token, timeout=1)
            for operation in (capture.ready, capture.poll, capture.finish):
                with self.assertRaisesRegex(RuntimeError, "consumed"):
                    operation(token)


if __name__ == "__main__":
    unittest.main()
