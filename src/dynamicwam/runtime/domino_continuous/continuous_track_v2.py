"""Candidate continuous chunk trajectory with local, constraint-driven retiming.

Each interval starts at the data-derived nominal duration. Only intervals
whose exact command envelopes violate supplied arm limits are lengthened.
Nonuniform PCHIP slopes are recomputed after every update, preserving C1 and
coordinate-wise shape bounds. This is feasible retiming, not a claim of
globally time-optimal control or collision/actuator tracking safety.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np

try:
    from .continuous_track import ContinuousJointTrack as _UniformTrack, _unit_interval_envelopes
except ImportError:
    from continuous_track import ContinuousJointTrack as _UniformTrack, _unit_interval_envelopes


def _nonuniform_coefficients(knots: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    delta = np.diff(knots, axis=0)
    secants = delta / intervals[:, None]
    derivatives = np.zeros_like(knots)
    if len(intervals) > 1:
        left, right = secants[:-1], secants[1:]
        same_direction = ((left > 0) & (right > 0)) | ((left < 0) & (right < 0))
        total_h = intervals[:-1] + intervals[1:]
        weight_left = ((intervals[:-1] + 2 * intervals[1:]) / (3 * total_h))[:, None]
        weight_right = ((2 * intervals[:-1] + intervals[1:]) / (3 * total_h))[:, None]
        smaller = np.minimum(np.abs(left), np.abs(right))
        ratio_left = np.divide(smaller, np.abs(left), out=np.zeros_like(left), where=same_direction)
        ratio_right = np.divide(smaller, np.abs(right), out=np.zeros_like(right), where=same_direction)
        denominator = weight_left * ratio_left + weight_right * ratio_right
        magnitude = np.divide(smaller, denominator, out=np.zeros_like(smaller), where=same_direction)
        derivatives[1:-1] = np.where(same_direction, np.sign(left) * magnitude, 0.)
    m0 = derivatives[:-1] * intervals[:, None]
    m1 = derivatives[1:] * intervals[:, None]
    return np.stack((m0 + m1 - 2 * delta, 3 * delta - 2 * m0 - m1, m0, knots[:-1]), axis=1)


def _segment_envelopes(coeff: np.ndarray, intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = (coeff[:, index, :] for index in range(3))
    unit_speed = np.maximum(np.abs(c), np.abs(3 * a + 2 * b + c))
    root = np.divide(-b, 3 * a, out=np.zeros_like(a), where=a != 0)
    valid = (a != 0) & (root > 0) & (root < 1)
    root = np.clip(root, 0., 1.)
    unit_speed = np.maximum(unit_speed, np.where(valid, np.abs((3 * a * root + 2 * b) * root + c), 0.))
    unit_acceleration = np.maximum(np.abs(2 * b), np.abs(6 * a + 2 * b))
    return unit_speed / intervals[:, None], (unit_acceleration / intervals[:, None]) / intervals[:, None]


class ContinuousJointTrack(_UniformTrack):
    """API-compatible local retiming candidate; row_times are authoritative.

    ``stretch`` is total-duration/nominal-duration, and ``interval_s`` is the
    mean interval for backwards-compatible diagnostics only. They do not
    describe a uniform clock. Consumers must sample by elapsed time and use
    row_times for target-boundary events. Each actual interval is >= nominal.

    A bounded iteration recomputes the nonuniform weighted harmonic slopes
    after lengthening violating intervals. A final uniform residual scale
    guarantees feasibility even without convergence. The validated uniform
    reference remains an upper bound: if local retiming is slower, use that
    reference, reporting the fallback explicitly.
    """
    def __init__(self, current_qpos: Any, targets: Any, action_interval_s: float,
                 velocity_limits: Any, acceleration_limits: Any,
                 position_limits: Any | None = None, constrained_axes: Any | None = None,
                 *, max_iterations: int = 32, limit_retiming: bool = True) -> None:
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(limit_retiming, bool):
            raise ValueError("limit_retiming must be a bool")
        super().__init__(current_qpos, targets, action_interval_s, velocity_limits,
                         acceleration_limits, position_limits, constrained_axes,
                         limit_retiming=limit_retiming)
        if not limit_retiming:
            # Keep Hermite shape at training-nominal 160 ms intervals; no MPLIB stretch.
            intervals = np.full(len(self._targets), self.action_interval_s)
            self.local_interval_s = intervals
            self.row_times = np.cumsum(intervals)
            self.duration = float(self.row_times[-1])
            self.stretch = 1.0
            self.interval_s = self.action_interval_s
            speed, acceleration = _segment_envelopes(self._coeff, intervals)
            self.max_velocity = speed.max(axis=0)
            self.max_acceleration = acceleration.max(axis=0)
            self._local_diagnostics = {
                "local_interval_s": intervals.tolist(),
                "local_stretch": [1.0] * len(intervals),
                "max_local_stretch": 1.0,
                "mean_action_interval_s": self.interval_s,
                "local_iterations": 0,
                "local_converged": True,
                "global_fallback": False,
                "global_fallback_kind": None,
                "residual_global_scale": 1.0,
                "residual_ratio_before_final_scale": 1.0,
                "uniform_reference_duration_s": self.duration,
                "uniform_reference_stretch": 1.0,
                "per_segment_max_velocity": speed.tolist(),
                "per_segment_max_acceleration": acceleration.tolist(),
                "limit_retiming": False,
                "timing_mode": "nominal_training_timing",
            }
            for array in (self.local_interval_s, self.row_times, self.max_velocity, self.max_acceleration):
                array.setflags(write=False)
            return
        reference_coeff = self._coeff
        reference_interval = self.interval_s
        reference_duration = self.duration
        reference_stretch = self.stretch
        intervals = np.full(len(self._targets), self.action_interval_s)
        axes = self.constrained_axes
        safety = 1. + 128 * np.finfo(np.float64).eps
        converged = False
        fallback_kind = None

        def factors(speed, acceleration):
            v_ratio = np.max(speed[:, axes] / self._velocity_limits[axes], axis=1, initial=0.)
            a_ratio = np.max(acceleration[:, axes] / self._acceleration_limits[axes], axis=1, initial=0.)
            return np.maximum(1., np.maximum(v_ratio, np.sqrt(a_ratio)))

        with np.errstate(divide="raise", invalid="raise", over="raise"):
            try:
                for iteration in range(int(max_iterations)):
                    coeff = _nonuniform_coefficients(self._knots, intervals)
                    speed, acceleration = _segment_envelopes(coeff, intervals)
                    scale = factors(speed, acceleration)
                    if float(scale.max()) <= 1. + 1e-12:
                        converged = True
                        break
                    intervals *= np.where(scale > 1., scale * safety, 1.)
                # Recompute even after the last allowed update. Constraints
                # couple adjacent intervals through their endpoint slopes.
                coeff = _nonuniform_coefficients(self._knots, intervals)
                speed, acceleration = _segment_envelopes(coeff, intervals)
                residual_scale = float(factors(speed, acceleration).max())
                converged = residual_scale <= 1. + 1e-12
                fallback = not converged
                # Scaling every h by the same factor leaves normalized cubic
                # coefficients invariant, scales v by 1/s and a by 1/s^2.
                polish_scale = residual_scale * safety if residual_scale > 1. else 1.
                intervals *= polish_scale
                speed, acceleration = _segment_envelopes(coeff, intervals)
            except (FloatingPointError, OverflowError, ZeroDivisionError):
                # The base constructor has already certified this reference.
                coeff = reference_coeff
                intervals = np.full(len(self._targets), reference_interval)
                speed, acceleration = _segment_envelopes(coeff, intervals)
                fallback, polish_scale, residual_scale = True, 1., None
                fallback_kind = "validated_uniform_reference_numeric"
                converged = False

        if fallback_kind is None and fallback:
            fallback_kind = "uniform_residual"
        if float(np.sum(intervals)) > reference_duration * (1. + 1e-12):
            coeff = reference_coeff
            intervals = np.full(len(self._targets), reference_interval)
            speed, acceleration = _segment_envelopes(coeff, intervals)
            fallback, fallback_kind = True, "validated_uniform_reference"
        if not (np.isfinite(coeff).all() and np.isfinite(intervals).all() and np.isfinite(speed).all() and np.isfinite(acceleration).all()):
            raise RuntimeError("local trajectory contains nonfinite values")
        if np.any(intervals < self.action_interval_s):
            raise RuntimeError("retiming compressed a training action interval")
        if np.any(speed[:, axes] > self._velocity_limits[axes] * safety) or np.any(acceleration[:, axes] > self._acceleration_limits[axes] * safety):
            raise RuntimeError("retimed trajectory violates constrained velocity/acceleration limits")

        _, _, position_min, position_max = _unit_interval_envelopes(coeff)
        tolerance = 256 * np.finfo(np.float64).eps * np.maximum(1., np.max(np.abs(self._knots), axis=0))
        if np.any(position_min < np.min(self._knots, axis=0) - tolerance) or np.any(position_max > np.max(self._knots, axis=0) + tolerance):
            raise RuntimeError("nonuniform PCHIP overshoots knot bounds")
        if position_limits is not None:
            bounds = np.asarray(position_limits)
            if np.any(position_min < bounds[:, 0] - tolerance) or np.any(position_max > bounds[:, 1] + tolerance):
                raise RuntimeError("nonuniform PCHIP violates position limits")
        self._coeff = coeff
        self.local_interval_s = intervals
        self.row_times = np.cumsum(intervals)
        self.duration = float(self.row_times[-1])
        self.stretch = self.duration / self.nominal_duration
        self.interval_s = self.duration / len(intervals)
        self.max_velocity = speed.max(axis=0)
        self.max_acceleration = acceleration.max(axis=0)
        self._position_min, self._position_max = position_min, position_max
        self._local_diagnostics = {
            "local_interval_s": intervals.tolist(),
            "local_stretch": (intervals / self.action_interval_s).tolist(),
            "max_local_stretch": float(np.max(intervals / self.action_interval_s)),
            "mean_action_interval_s": self.interval_s,
            "local_iterations": iteration + 1,
            "local_converged": converged,
            "global_fallback": fallback,
            "global_fallback_kind": fallback_kind,
            "residual_global_scale": polish_scale,
            "residual_ratio_before_final_scale": residual_scale,
            "uniform_reference_duration_s": reference_duration,
            "uniform_reference_stretch": reference_stretch,
            "per_segment_max_velocity": speed.tolist(),
            "per_segment_max_acceleration": acceleration.tolist(),
            "limit_retiming": True,
            "timing_mode": "native_mplib_limits",
        }
        for array in (self._coeff, self.local_interval_s, self.row_times, self.max_velocity, self.max_acceleration):
            array.setflags(write=False)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = float(elapsed)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed <= 0.:
            return self._current.copy(), np.zeros(14)
        if elapsed >= self.duration:
            return self._targets[-1].copy(), np.zeros(14)
        index = min(int(np.searchsorted(self.row_times, elapsed, side="right")), len(self._targets) - 1)
        start = self.row_times[index - 1] if index else 0.
        interval = self.local_interval_s[index]
        u = min(1., max(0., (elapsed - start) / interval))
        a, b, c, d = self._coeff[index]
        return ((a * u + b) * u + c) * u + d, ((3 * a * u + 2 * b) * u + c) / interval

    @property
    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics
        result.update(self._local_diagnostics)
        result.update(trajectory="shape_preserving_c1_hermite_local_retime_v2",
                      action_interval_s=None, interval_semantics="nonuniform; use local_interval_s and row_times_s",
                      stretch_semantics="total_duration / nominal_duration")
        return result
