"""Freeze original camera state on the physics owner, then read asynchronously.

By default, the worker only touches already-submitted Color buffers. Camera poses,
robot state, metadata, and the simulator timestamp are frozen by ``begin`` on
the owner thread. The caller updates ``env.now_obs`` and its flow history on
that same owner thread, using ``CapturedObservation.physics_time_s``.

Production uses one native batch call that releases the GIL across all Color
downloads and exact float32-to-uint8 conversion. It returns owning RGB arrays.
The explicit float32 single-camera reader remains a reference/testing API; no
fallback from a missing or failing native batch reader is allowed.

Opt-in ``async_submit`` also runs the four sequential native take-picture calls
on that single worker. ``_update_render`` and all metadata reads stay on owner:
SAPIEN copies entity poses to its renderer there, before physics can resume.
No render-state update or scene lifecycle change may overlap the worker.
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
    render_update_wall_seconds: float = 0.
    camera_submit_seconds: dict[str, float] = field(default_factory=dict)
    async_submit: bool = False
    capture_id: int | None = None


@dataclass(frozen=True)
class PendingCapture:
    capture_id: int
    physics_time_s: float
    capture_wall_perf_s: float
    # None in async mode: submission has not necessarily completed at begin().
    render_submitted_wall_perf_s: float | None
    camera_names: tuple[str, ...]
    _metadata: dict[str, Any] = field(repr=False)
    _future: Future = field(repr=False)
    render_update_wall_seconds: float = 0.
    camera_submit_seconds: dict[str, float] = field(default_factory=dict)
    async_submit: bool = False

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


def _read_submitted_rgb_batch(cameras, batch_reader, clock):
    """One native call; scene lifecycle and render-state updates stay on owner."""
    started = clock()
    images, elapsed = batch_reader(tuple(camera for _, camera in cameras))
    if len(images) != len(cameras) or len(elapsed) != len(cameras):
        raise ValueError("native RGB batch count differs from submitted cameras")
    rgb, durations = {}, {}
    for (name, _), image, duration in zip(cameras, images, elapsed):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1
                or not image.flags.owndata or not image.flags.c_contiguous):
            raise ValueError(f"native RGB batch requires owning packed uint8 HWC3 for {name}")
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid native readback/conversion duration for {name}")
        rgb[name] = {"rgb": image}
        durations[name] = duration
    return rgb, started, clock(), durations


def _submit_pictures(cameras, take_picture, clock):
    durations = {}
    for name, camera in cameras:
        before = clock()
        take_picture(camera)
        durations[name] = clock() - before
    return clock(), durations


def _submit_and_read(cameras, take_picture, read_worker, reader, clock):
    """Exclusive renderer worker; only frozen render handles, never ECS/env."""
    submitted, submit_durations = _submit_pictures(cameras, take_picture, clock)
    return (*read_worker(cameras, reader, clock), submitted, submit_durations)


def validate_native_rgb_conversion(native):
    """Numerical acceptance after guarded native import; no render/device call."""
    rng = np.random.default_rng(736)
    rgba = rng.integers(0, 2**32, size=(257, 263, 4), dtype=np.uint32).view(np.float32)
    rgba[~np.isfinite(rgba)] = 0
    centers = np.arange(256, dtype=np.float32) / np.float32(255)
    values = np.concatenate([centers, np.nextafter(centers, -np.inf),
                             np.nextafter(centers, np.inf)])
    cases = [rgba, np.resize(values, (1, len(values), 4)).astype(np.float32)]
    for case in cases:
        with np.errstate(over='ignore', invalid='ignore'):
            expected = (case * 255).clip(0, 255).astype('uint8')[:, :, :3]
        actual = native.rgba_to_rgb_u8(case)
        if (actual.dtype != np.uint8 or not actual.flags.owndata
                or not actual.flags.c_contiguous or not np.array_equal(actual, expected)):
            raise ValueError('native RGB conversion differs from native DOMINO pixels')
    rejected = 0
    for channel in range(4):
        for value in (np.nan, np.inf, -np.inf):
            bad = np.zeros((2, 3, 4), np.float32);bad[0, 0, channel] = value
            try:
                native.rgba_to_rgb_u8(bad)
            except RuntimeError as exc:
                if 'nonfinite' not in str(exc):raise
                rejected += 1
            else:
                raise ValueError('native RGB conversion accepted nonfinite Color')
    return dict(passed=True, exact_finite_cases=len(cases), nonfinite_cases_rejected=rejected,
                contract=native.rgb_batch_contract, rendering_tested=False)


class AsyncCameraCapture:
    """At most one snapshot; one reader thread; all lifecycle calls on owner.

    During a pending snapshot the caller may advance physics but must not call
    ``_update_render``, ``update_picture``, change renderer parameters/entities,
    or destroy the scene. ``close`` joins rendering and reading, including
    after terminal events/errors. A supplied worker initializer can establish
    the guarded renderer's thread-local CUDA device before any native call.
    """

    def __init__(self, *, reader: Callable[[Any], np.ndarray] | None = None,
                 batch_reader: Callable | None = None,
                 expected_camera_count: int = 4,
                 clock: Callable[[], float] = time.perf_counter,
                 async_submit: bool = False,
                 worker_initializer: Callable[[], None] | None = None):
        if expected_camera_count < 1:
            raise ValueError("expected camera count must be positive")
        if reader is not None and batch_reader is not None:
            raise ValueError("choose either explicit reference reader or native batch reader")
        if type(async_submit) is not bool:
            raise TypeError("async_submit must be a bool")
        if worker_initializer is not None and not callable(worker_initializer):
            raise TypeError("worker_initializer must be callable")
        if reader is None and batch_reader is None:
            from _sapien_color_reader import read_rgb_batch
            batch_reader = read_rgb_batch
        self._reader = batch_reader if batch_reader is not None else reader
        self._read_worker = (_read_submitted_rgb_batch if batch_reader is not None
                             else _read_submitted_colors)
        self._clock = clock
        self._expected_camera_count = expected_camera_count
        self._owner_thread = threading.get_ident()
        self._async_submit = async_submit
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="domino-color-readback",
                                            initializer=worker_initializer)
        self._pending: PendingCapture | None = None
        self._next_id = 0
        self._closed = False

    def _submit_readback(self, cameras):
        return self._executor.submit(self._read_worker, cameras, self._reader, self._clock)

    def _owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("camera capture lifecycle must run on its physics-owner thread")

    @property
    def busy(self) -> bool:
        return self._pending is not None

    @property
    def async_submit(self) -> bool:
        return self._async_submit

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
        """Freeze renderer and metadata before dispatching any background work."""
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
        render_update_wall = self._clock() - captured_wall
        # update_picture calls Camera.take_picture, whose SAPIEN binding holds
        # the GIL for the whole render wait. Inference workers are Python threads.
        from dynamicwam.runtime.domino_rt import _sapien_color_reader
        camera_submit_seconds = {}
        render_submitted_wall = None
        if not self._async_submit:
            render_submitted_wall, camera_submit_seconds = _submit_pictures(
                cameras, _sapien_color_reader.take_picture, self._clock)
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
        if self._async_submit:
            future = self._executor.submit(_submit_and_read, cameras,
                _sapien_color_reader.take_picture, self._read_worker, self._reader, self._clock)
        else:
            future = self._submit_readback(cameras)
        pending = PendingCapture(self._next_id, timestamp, captured_wall, render_submitted_wall,
                                 names, observation, future, render_update_wall, camera_submit_seconds,
                                 self._async_submit)
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
            output = selected._future.result(timeout=timeout)
            if selected.async_submit:
                rgb, started, ended, durations, submitted, submit_durations = output
            else:
                rgb, started, ended, durations = output
                submitted = selected.render_submitted_wall_perf_s
                submit_durations = selected.camera_submit_seconds
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
                                   selected.capture_wall_perf_s, submitted,
                                   started, ended, self._clock(), selected.camera_names, durations,
                                   selected.render_update_wall_seconds, dict(submit_durations),
                                   selected.async_submit, selected.capture_id)

    def drain(self, *, timeout: float | None = None) -> CapturedObservation | None:
        """Wait/consume before terminal teardown; never submits another render."""
        self._owner()
        return None if self._pending is None else self.finish(timeout=timeout)

    def close(self) -> None:
        """Join native submission/reads before scene/environment release."""
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
