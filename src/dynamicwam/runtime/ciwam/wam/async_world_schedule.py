"""Causal, bounded, nonblocking slow-world publication for MotionKV V3.

Threading does not guarantee GPU overlap or less contention. The producer must
finish its own CUDA event before returning; this scheduler publishes only those
immutable values. It never waits for a refresh from the fast query path.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time

from dynamicwam.runtime.ciwam.wam.fast_slow_worker import LatestWorldWorker

ASYNC_WORLD_SCHEDULE = "asynchronous_latest_completed_v1"


@dataclass(frozen=True)
class WorldRequest:
    packet: object
    source_index: int
    source_time: float
    submitted_monotonic: float


@dataclass(frozen=True)
class PublishedWorld:
    value: object
    source_index: int
    source_time: float
    submitted_monotonic: float
    completed_monotonic: float


class AsyncWorldScheduler:
    def __init__(self, build, *, snapshot, refresh_actions=16, max_age_actions=31,
                 max_age_seconds=4.96, action_interval_seconds=.16, clock=time.monotonic):
        self.build, self.snapshot, self.clock = build, snapshot, clock
        self.refresh_actions, self.max_age_actions = int(refresh_actions), int(max_age_actions)
        self.max_age_seconds, self.action_interval_seconds = float(max_age_seconds), float(action_interval_seconds)
        if (self.refresh_actions < 1 or self.max_age_actions < self.refresh_actions or
            not math.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0 or
            not math.isfinite(self.action_interval_seconds) or self.action_interval_seconds <= 0):
            raise ValueError("invalid asynchronous cache budget")
        self.worker = LatestWorldWorker(self._build)
        self.current = None
        self.last_request = None
        self.sequence = 0

    def _build(self, request):
        value = self.build(request.packet)
        return PublishedWorld(value, request.source_index, request.source_time,
                              request.submitted_monotonic, self.clock())

    def reset(self):
        self.worker.reset()
        self.current = self.last_request = None

    def close(self, timeout=5.):
        return self.worker.close(timeout)

    def completed_count(self):
        return self.worker.completed_count()

    def request(self, packet):
        index, timestamp = int(packet["action_index"]), float(packet["sim_time"])
        if index < 0 or not math.isfinite(timestamp):
            raise ValueError("invalid asynchronous source timestamp")
        if self.last_request is not None and (index < self.last_request.source_index or timestamp < self.last_request.source_time):
            raise ValueError("asynchronous source observations cannot move backwards")
        request = WorldRequest(self.snapshot(packet), index, timestamp, self.clock())
        self.sequence += 1
        self.worker.request(self.sequence, request)
        self.last_request = request
        return self.sequence

    def _accept(self, ready, *, action_index, observation_time):
        if ready is None:
            return False
        candidate = ready.value
        # A later observation cannot condition an older request, even if its
        # GPU work happened to finish first. Existing accepted memory stays.
        if candidate.source_index > action_index or candidate.source_time > observation_time + 1e-9:
            return False
        if self.current is not None and (candidate.source_index < self.current.source_index or
                candidate.source_time <= self.current.source_time):
            return False
        self.current = candidate
        return True

    def bootstrap(self, packet, *, timeout=120.):
        if self.current is not None:
            raise RuntimeError("bootstrap is only allowed before the first source")
        sequence = self.request(packet)
        ready = self.worker.wait_for(sequence, timeout=timeout)
        if not self._accept(ready, action_index=int(packet["action_index"]), observation_time=float(packet["sim_time"])):
            raise RuntimeError("bootstrap did not publish its causal source")
        return self.current

    def poll(self, packet):
        index, timestamp = int(packet["action_index"]), float(packet["sim_time"])
        accepted = self._accept(self.worker.latest(), action_index=index, observation_time=timestamp)
        if self.current is None:
            raise RuntimeError("asynchronous fast path requires completed bootstrap")
        age, seconds = index - self.current.source_index, timestamp - self.current.source_time
        if age < 0 or seconds < -1e-9:
            raise RuntimeError("future slow context cannot condition this observation")
        if age > self.max_age_actions or seconds > self.max_age_seconds + 1e-8:
            raise RuntimeError(f"asynchronous world cache expired: age_actions={age}, age_seconds={seconds:.6f}")
        last = self.last_request
        due = (last is None or index - last.source_index >= self.refresh_actions or
               timestamp - last.source_time >= self.refresh_actions * self.action_interval_seconds)
        submitted = False
        if due:
            self.request(packet)
            submitted = True
        with self.worker.condition:
            inflight = self.worker.active is not None
            queued = self.worker.pending is not None
            generation = self.worker.generation
        return self.current, dict(published_new=accepted, refresh_submitted=submitted,
            source_action_index=self.current.source_index, source_sim_time=self.current.source_time,
            source_submitted_monotonic=self.current.submitted_monotonic,
            source_published_monotonic=self.current.completed_monotonic,
            source_consumed_monotonic=self.clock(),
            cache_age_actions=age, cache_age_seconds=seconds,
            source_build_queue_wall_seconds=self.current.completed_monotonic-self.current.submitted_monotonic,
            refresh_inflight=inflight, refresh_queued=queued,
            generation=generation, slow_schedule=ASYNC_WORLD_SCHEDULE)
