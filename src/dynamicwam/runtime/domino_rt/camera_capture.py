"""Submit the original four RGB cameras on the physics owner, read asynchronously.

Only already-submitted Color buffers are touched by the worker. Camera poses,
robot state, metadata, and the simulator timestamp are frozen by ``begin`` on
the owner thread. The caller updates ``env.now_obs`` and its flow history on
that same owner thread, using ``CapturedObservation.physics_time_s``.

The native reader must return an owning float32 HWC4 NumPy array and release
the GIL while waiting/downloading. No Python ``camera.get_picture`` fallback
is provided: that fallback would reintroduce the physics-blocking failure.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class CapturedObservation:
    observation: dict[str, Any]
    physics_time_s: float
    capture_wall_perf_s: float
    render_submitted_wall_perf_s: float
    readback_started_wall_perf_s: float
    readback_finished_wall_perf_s: float
    finished_wall_perf_s: float
    camera_names: tuple[str, ...]
    camera_readback_seconds: dict[str, float]


@dataclass(frozen=True)
class PendingCapture:
    capture_id: int
    physics_time_s: float
    capture_wall_perf_s: float
    render_submitted_wall_perf_s: float
    camera_names: tuple[str, ...]
    _metadata: dict[str, Any] = field(repr=False)
    _future: Future = field(repr=False)

    @property
    def done(self) -> bool:
        """Readiness check only; the owner still consumes via poll/finish."""
        return self._future.done()


def _read_submitted_colors(cameras: tuple[tuple[str, Any], ...],
                           read_color: Callable[[Any], np.ndarray],
                           clock: Callable[[], float]) -> tuple[dict, float, float, dict]:
    """Worker entry: no env, robot, physics, camera config, or render submission."""
    started = clock()
    rgb = {}
    durations = {}
    for name, camera in cameras:
        before = clock()
        rgba = read_color(camera)
        after = clock()
        if not isinstance(rgba, np.ndarray) or rgba.dtype != np.float32:
            raise TypeError(f"native Color reader requires owning float32 NumPy; {name} returned {type(rgba)}")
        if rgba.ndim != 3 or rgba.shape[-1] != 4 or min(rgba.shape[:2]) < 1:
            raise ValueError(f"native Color reader requires nonempty HWC4; {name} returned {rgba.shape}")
        if not rgba.flags.owndata:
            raise ValueError(f"native Color reader must return an owning array for {name}")
        if not np.isfinite(rgba).all():
            raise ValueError(f"native Color reader returned nonfinite Color values for {name}")
        # Exactly Camera.get_rgba followed by Camera.get_rgb; do not change
        # rounding, gamma, channels, view order, or the ray-tracing settings.
        rgba_u8 = (rgba * 255).clip(0, 255).astype("uint8")
        rgb[name] = {"rgb": rgba_u8[:, :, :3]}
        durations[name] = after - before
    return rgb, started, clock(), durations


class AsyncCameraCapture:
    """At most one snapshot; one reader thread; all lifecycle calls on owner.

    During a pending snapshot the caller may advance physics but must not call
    ``_update_render``, ``update_picture``, or destroy the scene. ``close``
    waits for the native reader, including after terminal events/errors.
    """

    def __init__(self, *, reader: Callable[[Any], np.ndarray] | None = None,
                 expected_camera_count: int = 4,
                 clock: Callable[[], float] = time.perf_counter):
        if expected_camera_count < 1:
            raise ValueError("expected camera count must be positive")
        if reader is None:
            from _sapien_color_reader import read_color
            reader = read_color
        self._reader = reader
        self._clock = clock
        self._expected_camera_count = expected_camera_count
        self._owner_thread = threading.get_ident()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="domino-color-readback")
        self._pending: PendingCapture | None = None
        self._next_id = 0
        self._closed = False

    def _owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("camera capture lifecycle must run on its physics-owner thread")

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def pending(self) -> PendingCapture | None:
        return self._pending

    def _camera_handles(self, env) -> tuple[tuple[str, Any], ...]:
        data = env.data_type
        unsupported = [key for key in ("depth", "mesh_segmentation", "actor_segmentation",
                                       "pointcloud", "third_view", "observer") if data.get(key, False)]
        if unsupported:
            raise ValueError(f"asynchronous RGB capture does not support enabled modes: {unsupported}")
        if not data.get("rgb", False) or not data.get("qpos", False):
            raise ValueError("official DynamicWAM capture requires rgb and qpos")
        manager = env.cameras
        if not manager.collect_wrist_camera or not manager.collect_head_camera:
            raise ValueError("official observation requires both wrists and head camera")
        names = list(manager.static_camera_name)
        handles = list(manager.static_camera_list)
        if len(names) != len(handles):
            raise ValueError("static camera names and handles have different lengths")
        cameras = [("left_camera", manager.left_camera), ("right_camera", manager.right_camera)]
        cameras.extend(zip(names, handles))
        camera_names = [name for name, _ in cameras]
        if len(camera_names) != self._expected_camera_count or len(set(camera_names)) != len(camera_names):
            raise ValueError(f"expected {self._expected_camera_count} distinct original cameras, got {camera_names}")
        if "head_camera" not in camera_names:
            raise ValueError("original static camera set does not include head_camera")
        return tuple(cameras)

    def begin(self, env: Any, physics_time_s: float) -> PendingCapture:
        """Submit render and freeze its complete metadata before physics resumes."""
        self._owner()
        if self._closed:
            raise RuntimeError("camera capture is closed")
        if self._pending is not None:
            raise RuntimeError("consume or drain the pending capture before beginning another")
        timestamp = float(physics_time_s)
        if not math.isfinite(timestamp):
            raise ValueError("capture physics timestamp must be finite")
        cameras = self._camera_handles(env)
        captured_wall = self._clock()
        env._update_render()
        env.cameras.update_picture()
        render_submitted_wall = self._clock()
        observation = {
            "observation": deepcopy(env.cameras.get_config()),
            "pointcloud": [], "joint_action": {}, "endpose": {},
        }
        names = tuple(name for name, _ in cameras)
        if set(observation["observation"]) != set(names):
            raise ValueError("camera configuration keys differ from submitted original views")
        if env.data_type.get("endpose", False):
            observation["endpose"] = deepcopy({
                "left_endpose": env.get_arm_pose("left"),
                "left_gripper": env.robot.get_left_gripper_val(),
                "right_endpose": env.get_arm_pose("right"),
                "right_gripper": env.robot.get_right_gripper_val(),
            })
        # Match DOMINO's qpos field layout and concatenation semantics.
        left = deepcopy(env.robot.get_left_arm_jointState())
        right = deepcopy(env.robot.get_right_arm_jointState())
        observation["joint_action"] = {
            "left_arm": left[:-1], "left_gripper": left[-1],
            "right_arm": right[:-1], "right_gripper": right[-1],
            "vector": np.array(list(left) + list(right)),
        }
        future = self._executor.submit(_read_submitted_colors, cameras, self._reader, self._clock)
        pending = PendingCapture(self._next_id, timestamp, captured_wall, render_submitted_wall,
                                 names, observation, future)
        self._next_id += 1
        self._pending = pending
        return pending

    def _require(self, pending: PendingCapture | None) -> PendingCapture:
        self._owner()
        selected = self._pending if pending is None else pending
        if selected is None or selected is not self._pending:
            raise RuntimeError("capture is missing, already consumed, or belongs to another reader")
        return selected

    def poll(self, pending: PendingCapture | None = None) -> CapturedObservation | None:
        selected = self._require(pending)
        if not selected.done:
            return None
        return self.finish(selected)

    def ready(self, pending: PendingCapture | None = None) -> bool:
        """Owner-side readiness without consuming the frozen frame."""
        return self._require(pending).done

    def finish(self, pending: PendingCapture | None = None, *,
               timeout: float | None = None) -> CapturedObservation:
        """Consume ready output on owner; timeout retains a still-running future."""
        selected = self._require(pending)
        try:
            rgb, started, ended, durations = selected._future.result(timeout=timeout)
        except BaseException:
            # A timeout must not allow another rendering submission while
            # the reader still owns the previous submitted frame.
            if selected.done:
                self._pending = None
            raise
        observation = selected._metadata
        for name in selected.camera_names:
            observation["observation"][name].update(rgb[name])
        self._pending = None
        return CapturedObservation(observation, selected.physics_time_s,
                                   selected.capture_wall_perf_s, selected.render_submitted_wall_perf_s,
                                   started, ended, self._clock(), selected.camera_names, durations)

    def drain(self, *, timeout: float | None = None) -> CapturedObservation | None:
        """Wait/consume before terminal teardown; never submits another render."""
        self._owner()
        return None if self._pending is None else self.finish(timeout=timeout)

    def close(self) -> None:
        """Always join outstanding native reads before scene/environment release."""
        self._owner()
        if self._closed:
            return
        self._closed = True
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        self._owner()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
