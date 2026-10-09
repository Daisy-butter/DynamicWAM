"""Bounded latest-request producer with immutable completed snapshots.

One in-flight request plus one pending request; no queue of obsolete frames.
The CUDA completion barrier is on the producer thread, never on a normal fast
query. The consumer keeps its snapshot alive until its output reaches CPU.
No in-place tensor reuse is permitted. This differs deliberately from a pool
whose storage would require per-consumer CUDA events.
"""
from __future__ import annotations
from dataclasses import dataclass
from threading import Condition, Thread
import time


@dataclass(frozen=True)
class CompletedWorld:
    generation: int
    source_index: int
    value: object
    ready_monotonic: float


class LatestWorldWorker:
    def __init__(self, build):
        self.build = build
        self.condition = Condition()
        self.generation = 0
        self.pending = None
        self.active = None
        self.ready = None
        self.error = None
        self.closed = False
        self.published = 0
        self.dropped = 0
        self.thread = Thread(target=self._run, name="dynamicwam-world", daemon=True)
        self.thread.start()

    def request(self, source_index, packet):
        with self.condition:
            if self.closed:
                raise RuntimeError("world worker is closed")
            source_index = int(source_index)
            # Coalesce an already running request as well as queued/ready work.
            candidates = [item[1] for item in (self.active, self.pending)
                          if item is not None and item[0] == self.generation]
            if self.ready is not None:
                candidates.append(self.ready.source_index)
            if candidates and max(candidates) >= source_index:
                return False
            self.pending = (self.generation, source_index, packet)
            self.condition.notify_all()
            return True

    def latest(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("world producer failed") from self.error
            return self.ready

    def wait_for(self, source_index, timeout=120.):
        deadline = time.monotonic() + timeout
        with self.condition:
            generation = self.generation
            while True:
                if self.generation != generation or self.closed:
                    raise RuntimeError("episode reset or shutdown while waiting for world")
                if self.error is not None:
                    raise RuntimeError("world producer failed") from self.error
                if self.ready is not None and self.ready.source_index >= source_index:
                    return self.ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("world producer did not complete")
                self.condition.wait(remaining)

    def reset(self):
        with self.condition:
            self.generation += 1
            self.pending = self.ready = self.error = None
            self.published = 0  # Only accepted completions in this episode.
            self.condition.notify_all()

    def close(self, timeout=5.):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def completed_count(self):
        with self.condition:
            return self.published

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                generation, source_index, packet = self.pending
                self.active = self.pending
                self.pending = None
            try:
                value = self.build(packet)
            except Exception as exc:
                with self.condition:
                    self.active = None
                    if generation == self.generation and not self.closed:
                        self.error = exc
                    self.condition.notify_all()
                continue
            with self.condition:
                self.active = None
                if self.closed:
                    return
                if generation != self.generation or (self.ready is not None and self.ready.source_index >= source_index):
                    self.dropped += 1
                else:
                    self.ready = CompletedWorld(generation, source_index, value, time.monotonic())
                    self.published += 1
                self.condition.notify_all()
