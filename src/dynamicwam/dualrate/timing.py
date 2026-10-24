"""Deployment-only physical-clock reference publication and executable plans.

No thread, CUDA context, simulator or watchdog is created here. Integration must
call these transitions at the real simulation/controller clock and propagate
DeadlineMiss/BufferUnderrun; no implicit hold or old-plan replay is provided.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    episode_id: str = "episode"
    clock_id: str = "physical"
    expires_at: float = math.inf
    source_queue_version: int = -1
    source_prefix_hash: str = ""

    def supports(self, target_times: Sequence[float], *, query_time: float) -> tuple[bool, ...]:
        """Only actual future support; gaps/expired points are not extrapolated."""
        _finite(query_time, *target_times)
        if query_time < self.completed_time:
            raise TimingError("reference not completed at query time")
        if query_time >= self.expires_at:
            return tuple(False for _ in target_times)
        if {"world_intervals", "world_types", "world_valid"} <= self.payload.keys():
            intervals = self.payload["world_intervals"].detach().cpu()[0]
            valid = (self.payload["world_valid"] & (self.payload["world_types"] == 1)).detach().cpu()[0]
            return tuple(bool(target >= query_time and
                (((intervals[:,0] <= target) & (target <= intervals[:,1])) & valid).any())
                for target in target_times)
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
    def __init__(self, *, episode_id="episode", clock_id="physical", max_age=2., test_fixture=False):
        self._lock = threading.Lock()
        self._reference: ReferenceSnapshot | None = None
        if not episode_id or not clock_id or not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("reference episode/clock and finite positive TTL required")
        self.episode_id, self.clock_id, self.max_age = str(episode_id), str(clock_id), float(max_age)
        self.test_fixture = bool(test_fixture)

    def publish(self, *, source_time: float, completed_time: float, version: int,
                generation_stage: str, future_times: Sequence[float],
                future_valid: Sequence[bool], payload: Mapping[str, Tensor], now: float,
                complete: bool = True, episode_id=None, clock_id=None,
                source_queue_version=-1, source_prefix_hash="") -> bool:
        _finite(source_time, completed_time, now, *future_times)
        if not complete or not source_time <= completed_time <= now:
            raise TimingError("cannot publish an in-flight or time-inconsistent reference")
        for key,value in (("source_time",source_time),("completed_time",completed_time)):
            if key in payload and (payload[key].shape != (1,) or payload[key].dtype != torch.float64 or
                                   float(payload[key].item()) != value):
                raise TimingError("reference payload clock differs from publication metadata")
        if not isinstance(version, int) or version < 0 or not generation_stage:
            raise ValueError("reference version and generation stage required")
        if not self.test_fixture and (episode_id != self.episode_id or clock_id != self.clock_id or
                source_queue_version < 0 or len(source_prefix_hash) != 64):
            raise TimingError("reference source queue/episode/clock provenance missing or changed")
        if len(future_times) != len(future_valid) or any(b <= a for a,b in zip(future_times, future_times[1:])):
            raise ValueError("future support must have unique increasing physical times")
        if any(t < source_time for t in future_times):
            raise ValueError("future reference predates its source")
        copied = _copy_payload(payload)
        reference = ReferenceSnapshot(source_time, completed_time, version, generation_stage,
                                      tuple(future_times), tuple(bool(x) for x in future_valid), copied,
                                      self.episode_id, self.clock_id, source_time+self.max_age,
                                      source_queue_version, source_prefix_hash)
        if now >= reference.expires_at:
            return False
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
            if query_time >= reference.expires_at:
                return None
            return ReferenceSnapshot(reference.source_time, reference.completed_time,
                                     reference.version, reference.generation_stage,
                                     reference.future_times, reference.future_valid,
                                     _copy_payload(reference.payload), reference.episode_id, reference.clock_id,
                                     reference.expires_at, reference.source_queue_version, reference.source_prefix_hash)


@dataclass(frozen=True)
class TimedPlan:
    start_time: float
    dt: float
    actions: tuple[tuple[float, ...], ...]
    version: int
    query_time: float
    lease: QueryLease | None = None

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
    episode_id: str = "episode"
    clock_id: str = "physical"


def prefix_digest(prefix: CommittedPrefix) -> str:
    return hashlib.sha256(json.dumps((prefix.start_time, prefix.end_time, prefix.dt,
        prefix.actions,prefix.episode_id,prefix.clock_id), allow_nan=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class QueryLease:
    episode_id: str
    clock_id: str
    query_id: int
    query_time: float
    observation_time: float
    base_queue_version: int
    protected_until: float
    submit_margin: float
    prefix: CommittedPrefix
    prefix_hash: str
    reference_version: int

    def __post_init__(self):
        _finite(self.query_time, self.observation_time, self.protected_until, self.submit_margin)
        if (not self.episode_id or not self.clock_id or self.query_id < 0 or
                self.observation_time > self.query_time or self.submit_margin < 0 or
                self.query_time >= self.protected_until - self.submit_margin or
                self.protected_until != self.prefix.end_time or
                self.base_queue_version != self.prefix.queue_version or
                self.prefix_hash != prefix_digest(self.prefix)):
            raise ValueError("invalid query lease/immutable prefix")


class ExecutionQueue:
    """Grid-aligned replacement with immutable executed and committed intervals.

    Row k controls [origin+k*dt, origin+(k+1)*dt). A mid-row return cannot
    replace the row already executing. New plans must cover the first free row;
    all expired rows are discarded, and committed overlap must match exactly.
    """
    def __init__(self, initial_plan: TimedPlan, *, episode_id="episode", clock_id="physical",
                 allow_unleased_test_fixture=False):
        self._lock = threading.RLock()
        self.origin = initial_plan.start_time
        self.dt = initial_plan.dt
        self._rows = dict(enumerate(initial_plan.actions))
        self._version = initial_plan.version
        self._width = len(initial_plan.actions[0])
        self._committed_end = 0
        self._last_time = self.origin
        self.episode_id, self.clock_id = str(episode_id), str(clock_id)
        if not self.episode_id or not self.clock_id:
            raise ValueError("episode and physical clock identity required")
        self._test_fixture = bool(allow_unleased_test_fixture)
        self._query_counter = 0
        self._inflight = None
        self._executed = {}

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
        self._executed[index] = self._rows[index]
        self._last_time = now
        self._committed_end = max(self._committed_end, index+1)
        self._rows = {i: row for i,row in self._rows.items() if i >= index}
        return self._rows[index]

    @_serialized
    def executed_trace(self, since: float, until: float) -> tuple:
        """Only actions actually issued by action_at, never planned future rows."""
        _finite(since, until)
        if since > until or until > self._last_time + 1e-9:
            raise TimingError("execution trace requested outside observed time")
        return tuple((self.origin + i*self.dt, self.origin+(i+1)*self.dt, row)
                     for i,row in sorted(self._executed.items())
                     if self.origin+i*self.dt < until and self.origin+(i+1)*self.dt > since)

    @_serialized
    def snapshot_committed(self, now: float) -> CommittedPrefix:
        self._coverage(now)
        first = self._floor(now)
        # A row already underway is immutable even if not previously reserved.
        last = max(self._committed_end, first + int(now > self.origin + first*self.dt + 1e-9))
        return CommittedPrefix(self.origin+first*self.dt, self.origin+last*self.dt,
                               self.dt, tuple(self._rows[i] for i in range(first,last)), self._version,
                               self.episode_id,self.clock_id)

    @_serialized
    def begin_query(self, *, query_time: float, observation_time: float, protect_steps: int,
                    reference_version: int, submit_margin: float = 0.) -> QueryLease:
        if self._inflight is not None:
            raise TimingError("one fast query may be in flight; cancel/finish it first")
        if protect_steps < 1 or observation_time > query_time:
            raise ValueError("invalid protection or unavailable observation")
        until = self.origin + (self._floor(query_time) + protect_steps)*self.dt
        if query_time >= until-submit_margin:
            raise DeadlineMiss("no submission budget before protected boundary")
        prefix = self.commit_prefix(query_time=query_time, until=until)
        lease = QueryLease(self.episode_id, self.clock_id, self._query_counter, query_time,
            observation_time, self._version, prefix.end_time, submit_margin, prefix,
            prefix_digest(prefix), int(reference_version))
        self._query_counter += 1
        self._inflight = lease
        return lease

    @_serialized
    def cancel_query(self, lease: QueryLease) -> None:
        if self._inflight != lease:
            raise StalePlan("query no longer owns the in-flight slot")
        self._inflight = None

    def _validate_lease(self, plan, now):
        lease = plan.lease
        if lease is None:
            if self._test_fixture:
                return
            raise TimingError("production plan requires its query lease")
        if (lease != self._inflight or lease.episode_id != self.episode_id or
                lease.clock_id != self.clock_id or plan.query_time != lease.query_time):
            raise StalePlan("query/episode/clock lease is stale or mismatched")
        if now >= lease.protected_until - lease.submit_margin:
            raise DeadlineMiss("query exceeded protected prefix submission deadline")
        prefix = lease.prefix
        first = round((prefix.start_time-self.origin)/self.dt)
        for offset, value in enumerate(prefix.actions):
            i = first+offset
            actual = self._executed.get(i, self._rows.get(i))
            if actual != value:
                raise TimingError("immutable committed prefix history changed")

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
                               self.dt, tuple(self._rows[i] for i in range(first,last)), self._version,
                               self.episode_id,self.clock_id)

    @_serialized
    def replace(self, plan: TimedPlan, *, now: float) -> int:
        """Return number of expired/committed candidate rows discarded on acceptance."""
        _finite(now)
        self._validate_lease(plan, now)
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
        if plan.lease is not None:
            # Preserve original unnormalized queue bytes/values. The normalized
            # prefix is a model condition; its round trip is not an execution edit.
            p = plan.lease.prefix
            pi = round((p.start_time-self.origin)/self.dt)
            incoming.update({pi+i: row for i,row in enumerate(p.actions)})
        for index, row in incoming.items():
            if self._floor(now) <= index < free and index in self._rows and row != self._rows[index]:
                raise TimingError("new plan changes a currently executing or committed action")
        if free not in incoming:
            raise DeadlineMiss("new plan has no action at the first uncommitted target time")
        updated = {i: row for i,row in self._rows.items() if self._floor(now) <= i < free}
        updated.update({i: row for i,row in incoming.items() if i >= free})
        self._rows = updated
        self._version = plan.version
        self._inflight = None
        self._last_time = now
        return max(0, free-first)
