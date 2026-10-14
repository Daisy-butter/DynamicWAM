"""Selective asynchronous RGB capture and causal physical-time head history.

This module does not step physics or schedule actions. The owner may continue
its current trajectory while a native Color read is pending. As with the base
reader, it must not update render state again until that read is consumed.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
import math
from typing import Any

import numpy as np

try:
    from .camera_capture import AsyncCameraCapture, PendingCapture, _read_submitted_colors
except ImportError:  # Direct local CPU test execution.
    from camera_capture import AsyncCameraCapture, PendingCapture, _read_submitted_colors


class SelectiveAsyncCameraCapture(AsyncCameraCapture):
    """Full frozen policy snapshots; head-only snapshots for motion history.

    Head-only snapshots deliberately contain no robot state or camera config;
    their sole consumer is the history/video stream. Full query snapshots use
    the unchanged base implementation, including all original views and qpos.
    """

    def begin(self, env: Any, physics_time_s: float, *, head_only: bool = False) -> PendingCapture:
        if not head_only:
            return super().begin(env, physics_time_s)
        self._owner()
        if self._closed:
            raise RuntimeError("camera capture is closed")
        if self._pending is not None:
            raise RuntimeError("consume or drain the pending capture before beginning another")
        timestamp = float(physics_time_s)
        if not math.isfinite(timestamp):
            raise ValueError("capture physics timestamp must be finite")
        # Reuse validation of the complete original sensor configuration. No
        # robot pose/joint reads happen here or in the Color worker.
        cameras = tuple((name, handle) for name, handle in self._camera_handles(env)
                        if name == "head_camera")
        if len(cameras) != 1:
            raise ValueError("exactly one original head camera is required")
        captured_wall = self._clock()
        env._update_render()
        # DOMINO Camera.update_picture submits every camera, so do not use it
        # for history. This preserves the head sensor and its render settings.
        cameras[0][1].take_picture()
        submitted_wall = self._clock()
        metadata = {"observation": {"head_camera": {}}, "pointcloud": [],
                    "joint_action": {}, "endpose": {}}
        future = self._executor.submit(_read_submitted_colors, cameras, self._reader, self._clock)
        pending = PendingCapture(self._next_id, timestamp, captured_wall, submitted_wall,
                                 ("head_camera",), metadata, future)
        self._next_id += 1
        self._pending = pending
        return pending


def _motion_ops():
    from dynamicwam.absolute_motion import (
        MotionObservationBatch, build_motion_features, compute_flow_observation,
        flow_compute_contract,
    )
    return MotionObservationBatch, build_motion_features, compute_flow_observation, flow_compute_contract


class PhysicalTimeHeadHistory:
    """Select actual head frames near a query-relative physical-time grid.

    The original HeadFlowBuffer's count=4, stride=4 means four flow intervals
    plus one preceding interval for acceleration. Here each interval targets
    ``policy_stride * frame_period_s`` instead of four completed readbacks.
    Missed captures are never duplicated or relabelled with synthetic times.
    An absent/out-of-tolerance endpoint produces an invalid (black) flow pair.

    Endpoint selection is nearest in time, with earlier-frame tie breaking.
    Frames later than the query's frozen timestamp are never candidates. A
    tolerance below half an interval prevents one frame serving two slots.
    Every valid displacement/velocity uses the selected *actual* timestamps.
    ``diagnostics`` distinguishes missing endpoints from flow-quality masks.
    """

    def __init__(self, config: dict[str, Any], *, frame_period_s: float,
                 endpoint_tolerance_s: float | None = None, max_frames: int = 256):
        Batch, build, compute, contract_fn = _motion_ops()
        contract = contract_fn(config)
        self._Batch, self._build, self._compute = Batch, build, compute
        self.history_count = int(config["count"])
        self.policy_stride = int(contract["policy_stride"])
        self.compute_size = tuple(int(v) for v in contract["compute_size"])
        self.normalization_percentile = float(contract["normalization_percentile"])
        self.farneback = dict(contract["farneback"])
        self.quality = dict(contract["quality"])
        self.frame_period_s = float(frame_period_s)
        if not math.isfinite(self.frame_period_s) or self.frame_period_s <= 0:
            raise ValueError("frame_period_s must be finite and positive")
        if self.history_count < 1 or self.policy_stride < 1:
            raise ValueError("history_count and policy_stride must be positive")
        self.interval_s = self.policy_stride * self.frame_period_s
        self.endpoint_tolerance_s = (self.frame_period_s if endpoint_tolerance_s is None
                                     else float(endpoint_tolerance_s))
        if (not math.isfinite(self.endpoint_tolerance_s) or self.endpoint_tolerance_s < 0
                or self.endpoint_tolerance_s >= self.interval_s / 2):
            raise ValueError("endpoint_tolerance_s must be nonnegative and below half a flow interval")
        if int(max_frames) != max_frames or max_frames < self.history_count + 2:
            raise ValueError("max_frames must be an integer covering all history endpoints")
        self._frames: deque[tuple[np.ndarray, float, int]] = deque(maxlen=int(max_frames))
        self._retention_s = ((self.history_count + 1) * self.interval_s
                             + 2 * self.endpoint_tolerance_s + self.frame_period_s)
        self._next_frame_id = 0
        self._diagnostics: dict[str, Any] = {}
        # Consecutive H4 queries normally share four of their five exact
        # endpoint pairs. Cache only immutable pixel-pair results, never a
        # motion feature or timestamp: velocity/acceleration are rebuilt below.
        self._pair_cache: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self._pair_cache_limit = 2 * (self.history_count + 1)

    @property
    def latest_timestamp(self) -> float | None:
        return self._frames[-1][1] if self._frames else None

    @property
    def diagnostics(self) -> dict[str, Any]:
        return deepcopy(self._diagnostics)

    def reset(self) -> None:
        self._frames.clear()
        self._next_frame_id = 0
        self._diagnostics = {}
        self._pair_cache.clear()

    def push(self, frame_rgb_u8: np.ndarray, *, simulator_time_seconds: float) -> None:
        if (not isinstance(frame_rgb_u8, np.ndarray) or frame_rgb_u8.dtype != np.uint8
                or frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[-1] != 3
                or min(frame_rgb_u8.shape[:2]) < 1):
            raise ValueError("expected nonempty HWC uint8 RGB")
        timestamp = float(simulator_time_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("simulator_time_seconds must be finite")
        if self._frames and timestamp <= self._frames[-1][1]:
            raise ValueError("head capture timestamps must increase strictly; no duplicated frames")
        # Keep the original pixels: flow's own INTER_AREA resize remains the
        # single authoritative conversion, preserving the native visual path.
        self._frames.append((np.ascontiguousarray(frame_rgb_u8).copy(), timestamp,
                             self._next_frame_id))
        self._next_frame_id += 1
        while self._frames and self._frames[0][1] < timestamp - self._retention_s:
            self._frames.popleft()
        retained = {sample[2] for sample in self._frames}
        for key in tuple(self._pair_cache):
            if key[0] not in retained or key[1] not in retained:
                del self._pair_cache[key]

    def observation(self, *, reference_time_s: float | None = None):
        if not self._frames:
            raise RuntimeError("motion observation requested before the first head frame")
        reference = self.latest_timestamp if reference_time_s is None else float(reference_time_s)
        if not math.isfinite(reference):
            raise ValueError("reference_time_s must be finite")
        samples = [sample for sample in self._frames if sample[1] <= reference]
        if not samples:
            raise RuntimeError("no captured frame exists at or before this query")
        # A policy observation and flow history must end at the same snapshot.
        if abs(samples[-1][1] - reference) > 1e-9:
            raise ValueError("reference_time_s must identify a captured query snapshot")
        targets = [reference - offset * self.interval_s
                   for offset in range(self.history_count + 1, -1, -1)]
        selected = []
        for target in targets:
            candidate = min(samples, key=lambda sample: (abs(sample[1] - target), sample[1]))
            selected.append(candidate if abs(candidate[1] - target) <= self.endpoint_tolerance_s + 1e-9
                            else None)
        flow_rgb, displacement, starts, ends, valid, temporal, quality = [], [], [], [], [], [], []
        cache_hits = cache_misses = 0
        for previous, current in zip(selected, selected[1:]):
            is_temporal = (previous is not None and current is not None
                           and previous[2] != current[2] and current[1] > previous[1])
            if is_temporal:
                key = (previous[2], current[2])
                cached = self._pair_cache.get(key)
                if cached is None:
                    cached = self._compute(
                        previous[0], current[0], compute_size=self.compute_size,
                        normalization_percentile=self.normalization_percentile,
                        farneback=self.farneback, quality=self.quality)
                    self._pair_cache[key] = cached
                    cache_misses += 1
                    if len(self._pair_cache) > self._pair_cache_limit:
                        self._pair_cache.popitem(last=False)
                else:
                    self._pair_cache.move_to_end(key)
                    cache_hits += 1
                rgb, stats, _fraction, quality_valid = cached
            else:
                rgb = np.zeros((*self.compute_size, 3), dtype=np.uint8)
                stats, quality_valid = np.zeros(4, dtype=np.float32), False
            # Missing times are a masked placeholder, not an observation. They
            # are never exposed as valid velocity/acceleration or real frames.
            starts.append(previous[1] if previous is not None else reference)
            ends.append(current[1] if current is not None else reference)
            flow_rgb.append(rgb)
            displacement.append(stats)
            temporal.append(bool(is_temporal))
            quality.append(bool(quality_valid))
            valid.append(bool(is_temporal and quality_valid))
        features, interval_valid, acceleration_valid = self._build(
            np.stack(displacement), np.asarray(starts, dtype=np.float64),
            np.asarray(ends, dtype=np.float64), np.asarray(valid, dtype=np.bool_))
        self._diagnostics = {
            "reference_time_s": reference,
            "frame_period_s": self.frame_period_s, "target_interval_s": self.interval_s,
            "endpoint_tolerance_s": self.endpoint_tolerance_s,
            "retained_frames": len(self._frames),
            "flow_pair_cache_hits": cache_hits, "flow_pair_cache_misses": cache_misses,
            "endpoint_target_times_s": targets,
            "endpoint_capture_times_s": [s[1] if s is not None else None for s in selected],
            "endpoint_frame_ids": [s[2] if s is not None else None for s in selected],
            "endpoint_offsets_s": [s[1] - target if s is not None else None
                                   for s, target in zip(selected, targets)],
            "interval_dt_s": [end - start if ok else None
                              for start, end, ok in zip(starts[1:], ends[1:], temporal[1:])],
            "temporal_valid_mask": temporal[1:], "quality_valid_mask": quality[1:],
            "interval_valid_mask": interval_valid[1:].tolist(),
            "acceleration_valid_mask": acceleration_valid[1:].tolist(),
        }
        return self._Batch(flow_rgb=np.stack(flow_rgb[1:]), motion_features=features[1:],
                           interval_valid_mask=interval_valid[1:],
                           acceleration_valid_mask=acceleration_valid[1:])
