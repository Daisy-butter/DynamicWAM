"""Explicit absolute-time command protocol, independent of native TOPP.

Row i is a desired absolute joint position at observation_time+(i+1)*period.
Only future, not-yet-started intervals may be changed by a returned plan.
The common controller interpolates Hermite knots and limits command velocity
and acceleration. These limits apply to commanded targets, not physical q.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


class TimedPlanQueue:
    def __init__(self, period: float, dimension: int = 14) -> None:
        if period <= 0 or dimension != 14:
            raise ValueError("positive period and 14-dimensional ALOHA commands required")
        self.period = float(period)
        self.dimension = dimension
        self.knots: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
        self.pending: dict[str, Any] | None = None
        self.expired_results = 0

    def stage(self, result: dict[str, Any], now: float) -> dict[str, Any]:
        actions = np.asarray(result['actions'], dtype=np.float64)
        observation = float(result['observation_sim_time'])
        origin = float(result.get('metadata', {}).get('action_origin_sim_time', observation))
        if actions.ndim != 2 or actions.shape[1] != self.dimension or not len(actions):
            raise ValueError("actions must be nonempty [H,14]")
        if not np.isfinite(actions).all() or not math.isfinite(origin):
            raise ValueError("nonfinite result")
        origin_slot = round(origin / self.period)
        if abs(origin - origin_slot * self.period) > 1e-7:
            raise ValueError("observation must originate on the global control grid")
        # Even exact-boundary arrivals are installed by the next boundary loop.
        boundary = math.ceil((now - 1e-9) / self.period)
        drop = max(0, boundary - origin_slot)
        record = dict(request_id=result['request_id'], observation_sim_time=observation,
                      action_origin_sim_time=origin,
                      receive_sim_time=now, install_boundary=boundary,
                      install_sim_time=boundary * self.period,
                      clipped_prefix=min(drop, len(actions)), returned_rows=len(actions),
                      expired=drop >= len(actions), metadata=result.get('metadata', {}))
        if record['expired']:
            self.expired_results += 1
            return record
        self.pending = dict(actions=actions.copy(), origin_slot=origin_slot,
                            boundary=boundary, record=record)
        return record

    def publish(self, slot: int) -> dict[str, Any] | None:
        pending = self.pending
        if pending is None or pending['boundary'] > slot:
            return None
        self.pending = None
        record = pending['record']
        drop = max(0, slot - pending['origin_slot'])
        record['clipped_prefix'] = min(drop, len(pending['actions']))
        if drop >= len(pending['actions']):
            record['expired'] = True
            record['installed_rows'] = 0
            self.expired_results += 1
            return record
        if record['metadata'].get('replace_tail', True):
            self.knots = {end: value for end, value in self.knots.items() if end <= slot}
        for index, row in enumerate(pending['actions']):
            end = pending['origin_slot'] + index + 1
            if end > slot:
                self.knots[end] = (row.copy(), record)
        record['actual_install_sim_time'] = slot * self.period
        record['installed_rows'] = sum(end > slot and rec is record for end, (_, rec) in self.knots.items())
        return record

    def pop(self, end_slot: int) -> tuple[np.ndarray, dict[str, Any]] | None:
        self.knots = {end: value for end, value in self.knots.items() if end >= end_slot}
        return self.knots.pop(end_slot, None)

    def peek(self, end_slot: int) -> np.ndarray | None:
        value = self.knots.get(end_slot)
        return None if value is None else value[0]

    def has_future(self, after_slot: int) -> bool:
        return any(end > after_slot for end in self.knots)


class LimitedHermiteController:
    """Continuous commanded q and bounded command v/a across plan replacement."""
    def __init__(self, initial: np.ndarray, *, period: float = .1,
                 max_velocity: float = 2., max_acceleration: float = 20.,
                 max_gripper_velocity: float = 10.) -> None:
        self.q = np.asarray(initial, dtype=np.float64).copy()
        if self.q.shape != (14,) or not np.isfinite(self.q).all():
            raise ValueError("initial command must be finite [14]")
        if min(period, max_velocity, max_acceleration, max_gripper_velocity) <= 0:
            raise ValueError("controller limits must be positive")
        self.period = float(period)
        self.v = np.zeros(14)
        self.vmax = np.full(14, float(max_velocity)); self.vmax[[6, 13]] = max_gripper_velocity
        self.amax = np.full(14, float(max_acceleration)); self.amax[[6, 13]] = max_acceleration * 5
        self.start_q = self.q.copy(); self.start_v = self.v.copy()
        self.goal = self.q.copy(); self.end_v = self.v.copy()
        self.elapsed = 0.
        self.limited_ticks = 0
        self.ticks = 0

    def begin(self, target: np.ndarray | None, next_target: np.ndarray | None = None) -> None:
        self.start_q = self.q.copy(); self.start_v = self.v.copy()
        self.goal = self.q.copy() if target is None else np.asarray(target, dtype=np.float64).copy()
        self.goal[[6, 13]] = np.clip(self.goal[[6, 13]], 0., 1.)
        slope = np.zeros(14) if next_target is None else (np.asarray(next_target) - self.start_q) / (2 * self.period)
        self.end_v = np.clip(slope, -self.vmax, self.vmax)
        self.elapsed = 0.

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        self.elapsed = min(self.period, self.elapsed + dt)
        t = self.elapsed / self.period
        desired = ((2*t**3-3*t**2+1)*self.start_q + (t**3-2*t**2+t)*self.period*self.start_v
                   + (-2*t**3+3*t**2)*self.goal + (t**3-t**2)*self.period*self.end_v)
        desired_v = (desired - self.q) / dt
        velocity = np.clip(desired_v, -self.vmax, self.vmax)
        velocity = np.clip(velocity, self.v - self.amax*dt, self.v + self.amax*dt)
        self.limited_ticks += int(np.any(np.abs(velocity - desired_v) > 1e-8))
        self.q += velocity * dt
        self.v = velocity
        self.ticks += 1
        return self.q.copy(), self.v.copy()
