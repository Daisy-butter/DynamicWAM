"""A wall/physics correspondence, never a clock that advances the simulator.

Physics remains the count of completed real steps. Wall-ready results are held
until those steps reach their earliest admissible tick, preventing catch-up from
using a result in the simulated past. Bootstrap is explicitly inactive.
"""
from __future__ import annotations

import math
import time
from .timing import TimingError, DeadlineMiss


class WallPhysicsBridge:
    def __init__(self, physics_dt, *, wall_clock=time.perf_counter, numeric_tolerance=1e-7):
        if (not math.isfinite(physics_dt) or physics_dt <= 0 or
                not 0 <= numeric_tolerance < physics_dt * .01):
            raise ValueError("positive real physics quantum and sub-tick numerical tolerance required")
        self.physics_dt = float(physics_dt)
        self.numeric_tolerance = float(numeric_tolerance)
        self.wall_clock = wall_clock
        self.active = False
        self.wall_origin = self.physics_origin = None

    def wall_now(self):
        value = float(self.wall_clock())
        if not math.isfinite(value):
            raise TimingError("nonfinite monotonic wall clock")
        return value

    def activate(self, physics_time, wall_time=None):
        if self.active:
            raise TimingError("wall/physics bridge may activate only once")
        wall = self.wall_now() if wall_time is None else float(wall_time)
        if not math.isfinite(physics_time) or not math.isfinite(wall):
            raise TimingError("finite activation clocks required")
        self.physics_origin, self.wall_origin = float(physics_time), wall
        self.active = True

    def expected_physics(self, wall_time=None):
        if not self.active:
            raise TimingError("bootstrap has no active wall/physics correspondence")
        wall = self.wall_now() if wall_time is None else float(wall_time)
        if not math.isfinite(wall) or wall < self.wall_origin - self.numeric_tolerance:
            raise TimingError("wall timestamp precedes activation")
        return self.physics_origin + (wall - self.wall_origin)

    def availability_time(self, ready_wall):
        """Earliest admissible real step, NOT a measured completion timestamp."""
        expected = self.expected_physics(ready_wall)
        ticks = max(0, math.ceil((expected - self.physics_origin - self.numeric_tolerance) / self.physics_dt))
        return self.physics_origin + ticks * self.physics_dt

    def ready(self, ready_wall, physical_now):
        if not math.isfinite(physical_now):
            raise TimingError("nonfinite physical clock")
        return not self.active or physical_now + self.numeric_tolerance >= self.availability_time(ready_wall)

    def check_backlog(self, physical_now, max_backlog, *, wall_time=None):
        if not self.active:
            return 0.
        if not math.isfinite(physical_now) or not math.isfinite(max_backlog) or max_backlog <= 0:
            raise ValueError("finite physical clock and positive backlog bound required")
        lag = self.expected_physics(wall_time) - physical_now
        if lag > max_backlog or lag < -self.physics_dt - self.numeric_tolerance:
            raise TimingError(f"wall/physics backlog {lag:.9f}s outside [-one step,{max_backlog}]")
        return lag

    def caught_up(self, physical_now, *, wall_time=None):
        return not self.active or abs(self.expected_physics(wall_time) - physical_now) <= self.physics_dt + self.numeric_tolerance

    def check_deadline(self, protected_until, submit_margin=0., *, ready_wall=None, wall_now=None):
        if not self.active:
            return
        if not math.isfinite(protected_until) or not math.isfinite(submit_margin) or submit_margin < 0:
            raise ValueError("invalid protected wall deadline")
        deadline = self.wall_origin + (protected_until - self.physics_origin) - submit_margin
        now = self.wall_now() if wall_now is None else float(wall_now)
        for value in (now,) if ready_wall is None else (float(ready_wall), now):
            if not math.isfinite(value) or value >= deadline:
                raise DeadlineMiss("query exceeded protected prefix wall-clock submission deadline")

    def snapshot(self, physical_now, *, wall_time=None):
        wall = self.wall_now() if wall_time is None else float(wall_time)
        result = dict(wall_time=wall, physical_time=float(physical_now), active=self.active)
        if self.active:
            expected = self.expected_physics(wall)
            result.update(expected_physical_time=expected, backlog_seconds=expected-physical_now,
                          physics_step_seconds=self.physics_dt)
        return result
