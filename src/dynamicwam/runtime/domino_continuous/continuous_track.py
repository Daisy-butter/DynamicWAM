"""Shape-preserving, time-scaled joint trajectories for a complete action chunk.

The 14 coordinates are [left arm (6), left gripper, right arm (6), right
gripper]. Limits must use the units of their constrained coordinates.
This module bounds commanded positions and, on constrained axes, velocities
and one-sided accelerations; it does not establish actuator tracking or
collision safety. Unconstrained normalized grippers retain their actuator's
own control semantics and do not impose invented physical timing limits.
There is no action dropping, RTC prefix, or inference scheduling here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DIMENSIONS = 14


def _vector(value: Any, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (DIMENSIONS,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({DIMENSIONS},) array")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive in every coordinate")
    return array


def _unit_interval_envelopes(coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic extrema of q, dq/du and d2q/du2 on every segment, u in [0, 1]."""
    a, b, c, d = (coeff[:, index, :] for index in range(4))
    q_end = a + b + c + d
    q_min = np.minimum(d, q_end).copy()
    q_max = np.maximum(d, q_end).copy()
    speed = np.maximum(np.abs(c), np.abs(3.0 * a + 2.0 * b + c))
    acceleration = np.maximum(np.abs(2.0 * b), np.abs(6.0 * a + 2.0 * b))

    # A quadratic velocity reaches its only interior extremum at -b/(3a).
    nonzero_a = a != 0.0
    u = np.divide(-b, 3.0 * a, out=np.zeros_like(a), where=nonzero_a)
    interior = nonzero_a & (u > 0.0) & (u < 1.0)
    speed = np.maximum(speed, np.where(interior, np.abs((3.0 * a * u + 2.0 * b) * u + c), 0.0))

    # Position extrema are roots of 3a*u^2 + 2b*u + c. Shape preservation
    # gives an additional independent bound, but we verify the polynomial.
    aa, bb = 3.0 * a, 2.0 * b
    discriminant = bb * bb - 4.0 * aa * c
    quadratic = nonzero_a & (discriminant >= 0.0)
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
    root_numerator = -0.5 * (bb + np.copysign(sqrt_discriminant, bb))
    root1 = np.divide(root_numerator, aa, out=np.zeros_like(aa), where=quadratic)
    root2 = np.divide(c, root_numerator, out=np.zeros_like(c), where=quadratic & (root_numerator != 0.0))
    linear = (~nonzero_a) & (bb != 0.0)
    linear_root = np.divide(-c, bb, out=np.zeros_like(c), where=linear)
    for root, valid in ((root1, quadratic), (root2, quadratic & (root_numerator != 0.0)), (linear_root, linear)):
        inside = valid & (root > 0.0) & (root < 1.0)
        q = ((a * root + b) * root + c) * root + d
        q_min = np.minimum(q_min, np.where(inside, q, q_min))
        q_max = np.maximum(q_max, np.where(inside, q, q_max))
    return speed.max(axis=0), acceleration.max(axis=0), q_min.min(axis=0), q_max.max(axis=0)


class ContinuousJointTrack:
    """C1 continuous, shape-preserving Hermite playback of every target row.

    A supplied current pose is the t=0 knot; target row i is at nominal time
    (i+1)*action_interval_s. Interior slopes use the uniform-grid PCHIP
    harmonic mean. The first and last slopes are zero to join stationary
    synchronous hold phases. Interior slopes only become zero at plateaus
    and direction reversals, rather than being reset at every target row.

    One global time stretch enforces velocity and acceleration limits on
    constrained_axes (all axes by default), unless limit_retiming=False for the
    nominal-training-timing ablation (stretch stays 1; envelopes are still reported). An excluded axis needs positive
    finite limit placeholders but they do not constrain timing or imply
    actuator velocity/acceleration bounds. Reported limits are null there.
    Accelerations can jump at knots (the curve is C1, not jerk limited).
    Position limits, when provided, have shape (14, 2); +/-inf is allowed.
    Sampling before/after the trajectory holds the first/last pose.
    """

    def __init__(
        self,
        current_qpos: Any,
        targets: Any,
        action_interval_s: float,
        velocity_limits: Any,
        acceleration_limits: Any,
        position_limits: Any | None = None,
        constrained_axes: Any | None = None,
        *,
        limit_retiming: bool = True,
    ) -> None:
        self._current = _vector(current_qpos, "current_qpos")
        self._targets = np.array(targets, dtype=np.float64, copy=True)
        if self._targets.ndim != 2 or self._targets.shape[0] < 1 or self._targets.shape[1] != DIMENSIONS:
            raise ValueError(f"targets must have shape (N, {DIMENSIONS}), N >= 1")
        if not np.isfinite(self._targets).all():
            raise ValueError("targets must be finite")
        self.action_interval_s = float(action_interval_s)
        if not math.isfinite(self.action_interval_s) or self.action_interval_s <= 0.0:
            raise ValueError("action_interval_s must be finite and strictly positive")
        self._velocity_limits = _vector(velocity_limits, "velocity_limits", positive=True)
        self._acceleration_limits = _vector(acceleration_limits, "acceleration_limits", positive=True)
        if constrained_axes is None:
            self.constrained_axes = np.ones(DIMENSIONS, dtype=np.bool_)
        else:
            axes = np.asarray(constrained_axes)
            if axes.shape != (DIMENSIONS,) or axes.dtype.kind != "b":
                raise ValueError("constrained_axes must be a boolean (14,) array")
            self.constrained_axes = axes.copy()
        self._knots = np.vstack((self._current, self._targets))
        limits = None
        if position_limits is not None:
            limits = np.array(position_limits, dtype=np.float64, copy=True)
            if limits.shape != (DIMENSIONS, 2) or np.isnan(limits).any() or np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError("position_limits must be ordered (14, 2) bounds without NaN")
            if np.any(self._knots < limits[:, 0]) or np.any(self._knots > limits[:, 1]):
                raise ValueError("current_qpos or targets violate supplied position_limits")

        # Use normalized tangents dq/du, avoiding division by dt until limits
        # are evaluated. This is PCHIP on a uniformly spaced time grid.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                delta = np.diff(self._knots, axis=0)
                tangents = np.zeros_like(self._knots)
                left, right = delta[:-1], delta[1:]
                same_direction = ((left > 0.0) & (right > 0.0)) | ((left < 0.0) & (right < 0.0))
                small, large = np.minimum(np.abs(left), np.abs(right)), np.maximum(np.abs(left), np.abs(right))
                ratio = np.divide(small, large, out=np.zeros_like(small), where=same_direction)
                harmonic = np.sign(left) * (small / (0.5 + 0.5 * ratio))
                tangents[1:-1] = np.where(same_direction, harmonic, 0.0)
                m0, m1 = tangents[:-1], tangents[1:]
                self._coeff = np.stack((m0 + m1 - 2.0 * delta, 3.0 * delta - 2.0 * m0 - m1, m0, self._knots[:-1]), axis=1)
                unit_speed, unit_acceleration, position_min, position_max = _unit_interval_envelopes(self._coeff)
                nominal_speed = unit_speed / self.action_interval_s
                nominal_acceleration = (unit_acceleration / self.action_interval_s) / self.action_interval_s
                constrained_speed_ratio = nominal_speed[self.constrained_axes] / self._velocity_limits[self.constrained_axes]
                constrained_acceleration_ratio = nominal_acceleration[self.constrained_axes] / self._acceleration_limits[self.constrained_axes]
                self.limit_retiming = bool(limit_retiming)
                if self.limit_retiming:
                    self.stretch = max(1.0, float(np.max(constrained_speed_ratio, initial=0.0)), math.sqrt(float(np.max(constrained_acceleration_ratio, initial=0.0))))
                    if self.stretch > 1.0:
                        self.stretch *= 1.0 + 64.0 * np.finfo(np.float64).eps
                else:
                    # Ablation: keep training-nominal wall time; still report envelopes.
                    self.stretch = 1.0
                self.interval_s = self.action_interval_s * self.stretch
                self.nominal_duration = self.action_interval_s * len(self._targets)
                self.duration = self.interval_s * len(self._targets)
                self.row_times = np.arange(1, len(self._targets) + 1, dtype=np.float64) * self.interval_s
                self.max_velocity = unit_speed / self.interval_s
                self.max_acceleration = (unit_acceleration / self.interval_s) / self.interval_s
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise ValueError("trajectory values exceed supported numeric range") from exc
        if not all(math.isfinite(value) for value in (self.duration, self.nominal_duration, self.interval_s, self.stretch)) or not np.isfinite(self._coeff).all():
            raise ValueError("trajectory values exceed supported numeric range")
        self._position_min = position_min
        self._position_max = position_max
        self._nominal_speed = nominal_speed
        self._nominal_acceleration = nominal_acceleration
        verification_slack = 1.0 + 64.0 * np.finfo(np.float64).eps
        axes = self.constrained_axes
        if self.limit_retiming and (np.any(self.max_velocity[axes] > self._velocity_limits[axes] * verification_slack) or np.any(self.max_acceleration[axes] > self._acceleration_limits[axes] * verification_slack)):
            raise RuntimeError("time-scaled trajectory violates constrained velocity/acceleration limits")

        # Tolerance only accommodates floating-point polynomial evaluation;
        # supplied knot bounds above are strict and no target is clipped.
        tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.max(np.abs(self._knots), axis=0))
        if np.any(position_min < np.min(self._knots, axis=0) - tolerance) or np.any(position_max > np.max(self._knots, axis=0) + tolerance):
            raise RuntimeError("shape-preserving trajectory unexpectedly overshoots its knots")
        if limits is not None and (np.any(position_min < limits[:, 0] - tolerance) or np.any(position_max > limits[:, 1] + tolerance)):
            raise RuntimeError("trajectory extrema violate supplied position_limits")
        for array in (self._current, self._targets, self._knots, self._coeff, self._velocity_limits, self._acceleration_limits, self.constrained_axes, self.row_times, self.max_velocity, self.max_acceleration):
            array.setflags(write=False)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        """Return position and velocity in coordinate units and units/second."""
        elapsed = float(elapsed)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed <= 0.0:
            return self._current.copy(), np.zeros(DIMENSIONS, dtype=np.float64)
        if elapsed >= self.duration:
            return self._targets[-1].copy(), np.zeros(DIMENSIONS, dtype=np.float64)
        row_position = elapsed / self.interval_s
        index = min(int(math.floor(row_position)), len(self._targets) - 1)
        u = min(1.0, max(0.0, row_position - index))
        a, b, c, d = self._coeff[index]
        position = ((a * u + b) * u + c) * u + d
        velocity = ((3.0 * a * u + 2.0 * b) * u + c) / self.interval_s
        return position, velocity

    @property
    def diagnostics(self) -> dict[str, Any]:
        """JSON-serializable exact command envelopes, excluding tracking error."""
        return {
            "trajectory": "shape_preserving_c1_hermite_uniform_stretch",
            "target_count": len(self._targets),
            "nominal_action_interval_s": self.action_interval_s,
            "action_interval_s": self.interval_s,
            "nominal_duration_s": self.nominal_duration,
            "duration_s": self.duration,
            "stretch": self.stretch,
            "limit_retiming": self.limit_retiming,
            "row_times_s": self.row_times.tolist(),
            "nominal_max_velocity": self._nominal_speed.tolist(),
            "nominal_max_acceleration": self._nominal_acceleration.tolist(),
            "max_velocity": self.max_velocity.tolist(),
            "max_acceleration": self.max_acceleration.tolist(),
            "constrained_axes": self.constrained_axes.tolist(),
            "velocity_limits": [float(value) if constrained else None for value, constrained in zip(self._velocity_limits, self.constrained_axes)],
            "acceleration_limits": [float(value) if constrained else None for value, constrained in zip(self._acceleration_limits, self.constrained_axes)],
            "position_min": self._position_min.tolist(),
            "position_max": self._position_max.tolist(),
        }
