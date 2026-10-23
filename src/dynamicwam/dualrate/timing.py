"""Deployment-only physical-clock reference publication and executable plans.

No thread, CUDA context, simulator or watchdog is created here. Integration must
call these transitions at the real simulation/controller clock and propagate
DeadlineMiss/BufferUnderrun; no implicit hold or old-plan replay is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from functools import wraps
import threading
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor


class TimingError(RuntimeError):
    pass


class DeadlineMiss(TimingError):
    pass


class BufferUnderrun(TimingError):
    pass


class StalePlan(TimingError):
    pass


def _finite(*values: float) -> None:
    if not all(math.isfinite(x) for x in values):
        raise ValueError("clock values must be finite")


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


@dataclass(frozen=True)
class ReferenceSnapshot:
    source_time: float
    completed_time: float
    version: int
    generation_stage: str
    future_times: tuple[float, ...]
    future_valid: tuple[bool, ...]
    payload: Mapping[str, Tensor]

    def supports(self, target_times: Sequence[float], *, query_time: float) -> tuple[bool, ...]:
        """Only actual future support; gaps/expired points are not extrapolated."""
        _finite(query_time, *target_times)
        if query_time < self.completed_time:
            raise TimingError("reference not completed at query time")
        result = []
        for target in target_times:
            supported = False
            if target >= query_time and self.future_times:
                for index, when in enumerate(self.future_times):
                    if abs(target - when) <= 1e-9:
                        supported = self.future_valid[index]
                        break
                    if index and self.future_times[index-1] < target < when:
                        supported = self.future_valid[index-1] and self.future_valid[index]
                        break
            result.append(supported)
        return tuple(result)


def _copy_payload(payload: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if not payload or not all(isinstance(k, str) and isinstance(v, Tensor) for k,v in payload.items()):
        raise ValueError("reference payload must be nonempty named tensors")
    if any(value.requires_grad for value in payload.values()):
        raise ValueError("deployment cache cannot hold autograd graphs; use live bridge in training")
    return MappingProxyType({key: value.detach().clone() for key,value in payload.items()})


class CompletedReferenceStore:
    """Atomic publication; source and version both monotonic. Readers get snapshots.

    Tensor payloads are copied on publication and query so mutation of caller or
    consumer buffers cannot alter the stored completed reference. Training should
    use the live differentiable bridge instead of this deployment cache.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._reference: ReferenceSnapshot | None = None

    def publish(self, *, source_time: float, completed_time: float, version: int,
                generation_stage: str, future_times: Sequence[float],
                future_valid: Sequence[bool], payload: Mapping[str, Tensor], now: float,
                complete: bool = True) -> bool:
        _finite(source_time, completed_time, now, *future_times)
        if not complete or not source_time <= completed_time <= now:
            raise TimingError("cannot publish an in-flight or time-inconsistent reference")
        if not isinstance(version, int) or version < 0 or not generation_stage:
            raise ValueError("reference version and generation stage required")
        if len(future_times) != len(future_valid) or any(b <= a for a,b in zip(future_times, future_times[1:])):
            raise ValueError("future support must have unique increasing physical times")
        if any(t < source_time for t in future_times):
            raise ValueError("future reference predates its source")
        copied = _copy_payload(payload)
        reference = ReferenceSnapshot(source_time, completed_time, version, generation_stage,
                                      tuple(future_times), tuple(bool(x) for x in future_valid), copied)
        with self._lock:
            old = self._reference
            if old is not None and (version <= old.version or source_time < old.source_time):
                return False  # late completion of old-source work is explicitly rejected
            self._reference = reference
        return True

    def snapshot(self, query_time: float) -> ReferenceSnapshot | None:
        _finite(query_time)
        with self._lock:
            reference = self._reference
            if reference is None:
                return None
            if query_time < reference.completed_time:
                raise TimingError("query cannot consume a not-yet-complete reference")
            return ReferenceSnapshot(reference.source_time, reference.completed_time,
                                     reference.version, reference.generation_stage,
                                     reference.future_times, reference.future_valid,
                                     _copy_payload(reference.payload))


@dataclass(frozen=True)
class TimedPlan:
    start_time: float
    dt: float
    actions: tuple[tuple[float, ...], ...]
    version: int
    query_time: float

    def __post_init__(self):
        _finite(self.start_time, self.dt, self.query_time)
        # A mid-step observation can carry the current, already-committed row
        # whose start precedes query_time. replace() will preserve that row.
        if self.dt <= 0 or not isinstance(self.version, int) or self.version < 0:
            raise ValueError("invalid plan time/grid/version")
        rows = tuple(tuple(float(x) for x in row) for row in self.actions)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("actions must be a nonempty rectangular sequence")
        _finite(*(x for row in rows for x in row))
        object.__setattr__(self, "actions", rows)

    @property
    def end_time(self) -> float:
        return self.start_time + self.dt * len(self.actions)


@dataclass(frozen=True)
class CommittedPrefix:
    start_time: float
    end_time: float
    dt: float
    actions: tuple[tuple[float, ...], ...]
    queue_version: int


class ExecutionQueue:
    """Grid-aligned replacement with immutable executed and committed intervals.

    Row k controls [origin+k*dt, origin+(k+1)*dt). A mid-row return cannot
    replace the row already executing. New plans must cover the first free row;
    all expired rows are discarded, and committed overlap must match exactly.
    """
    def __init__(self, initial_plan: TimedPlan):
        self._lock = threading.RLock()
        self.origin = initial_plan.start_time
        self.dt = initial_plan.dt
        self._rows = dict(enumerate(initial_plan.actions))
        self._version = initial_plan.version
        self._width = len(initial_plan.actions[0])
        self._committed_end = 0
        self._last_time = self.origin

    def _floor(self, value: float) -> int:
        return math.floor((value-self.origin)/self.dt + 1e-9)

    def _ceil(self, value: float) -> int:
        return math.ceil((value-self.origin)/self.dt - 1e-9)

    def _coverage(self, now: float) -> None:
        _finite(now)
        if now < self._last_time - 1e-9:
            raise TimingError("execution clock moved backwards")
        for index in range(self._floor(self._last_time), self._floor(now)+1):
            if index not in self._rows:
                raise BufferUnderrun(f"no action at {self.origin + index*self.dt:.9f}s")

    @_serialized
    def action_at(self, now: float) -> tuple[float, ...]:
        self._coverage(now)
        index = self._floor(now)
        self._last_time = now
        self._committed_end = max(self._committed_end, index+1)
        self._rows = {i: row for i,row in self._rows.items() if i >= index}
        return self._rows[index]

    @_serialized
    def commit_prefix(self, *, query_time: float, until: float) -> CommittedPrefix:
        """Reserve the actions that will execute while the next query is in flight."""
        _finite(query_time, until)
        if until < query_time:
            raise ValueError("prefix end precedes query")
        self._coverage(query_time)
        first = self._floor(query_time)
        last = max(self._committed_end, self._ceil(until))
        if query_time > self.origin + first*self.dt + 1e-9:
            last = max(last, first+1)
        for index in range(first, last):
            if index not in self._rows:
                raise BufferUnderrun("cannot commit actions absent from the queue")
        self._committed_end = last
        self._last_time = query_time
        self._rows = {i: row for i,row in self._rows.items() if i >= first}
        return CommittedPrefix(self.origin + first*self.dt, self.origin + last*self.dt,
                               self.dt, tuple(self._rows[i] for i in range(first,last)), self._version)

    @_serialized
    def replace(self, plan: TimedPlan, *, now: float) -> int:
        """Return number of expired/committed candidate rows discarded on acceptance."""
        _finite(now)
        if plan.version <= self._version:
            raise StalePlan("old plan version cannot overwrite execution queue")
        if plan.query_time > now + 1e-9:
            raise TimingError("plan returned before its query")
        if abs(plan.dt - self.dt) > 1e-10 or len(plan.actions[0]) != self._width:
            raise ValueError("plan action grid/width changed")
        offset = (plan.start_time-self.origin)/self.dt
        first = round(offset)
        if abs(offset-first) > 1e-7:
            raise ValueError("plan targets are not aligned to execution clock")
        # At a boundary the existing buffer may end exactly now; a timely new row
        # is legal. Missing earlier intervals remain a real underrun.
        if now < self._last_time - 1e-9:
            raise TimingError("execution clock moved backwards")
        for index in range(self._floor(self._last_time), self._ceil(now)):
            if index not in self._rows:
                raise BufferUnderrun("execution ran out before new plan returned")
        free = max(self._committed_end, self._ceil(now))
        incoming = {first+i: row for i,row in enumerate(plan.actions)}
        for index, row in incoming.items():
            if self._floor(now) <= index < free and index in self._rows and row != self._rows[index]:
                raise TimingError("new plan changes a currently executing or committed action")
        if free not in incoming:
            raise DeadlineMiss("new plan has no action at the first uncommitted target time")
        updated = {i: row for i,row in self._rows.items() if self._floor(now) <= i < free}
        updated.update({i: row for i,row in incoming.items() if i >= free})
        self._rows = updated
        self._version = plan.version
        self._last_time = now
        return max(0, free-first)
