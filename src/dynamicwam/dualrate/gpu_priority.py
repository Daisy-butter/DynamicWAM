"""Cooperative same-device scheduling; does not modify simulation time.

Slow CUDA operators are drained individually so fast work never waits behind
an entire queued slow forward. This conservative path trades slow throughput
for bounded queue depth; actual worst-case operator latency still needs GPU QA.
"""
from contextlib import contextmanager
from threading import Condition
from torch.utils._python_dispatch import TorchDispatchMode


class FastPriorityGate:
    def __init__(self):
        self.condition = Condition()
        self.pending = 0
        self.busy = False

    def reserve_fast(self):
        with self.condition:
            self.pending += 1
            self.condition.notify_all()

    def cancel_fast(self):
        with self.condition:
            self.pending -= 1
            self.condition.notify_all()

    @contextmanager
    def fast(self, drain=lambda: None):
        with self.condition:
            self.condition.wait_for(lambda: not self.busy)
            self.busy = True
        try:
            yield
        finally:
            try:
                drain()
            finally:
                with self.condition:
                    self.busy = False
                    self.pending -= 1
                    self.condition.notify_all()

    @contextmanager
    def slow(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.busy and not self.pending)
            self.busy = True
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()


class SlowOperatorSlices(TorchDispatchMode):
    def __init__(self, gate, stream):
        super().__init__()
        self.gate, self.stream = gate, stream

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with self.gate.slow():
            try:
                return func(*args, **(kwargs or {}))
            finally:
                # Only this slow stream, never a device-wide synchronize.
                # Also drain a partially dispatched operator before releasing
                # the permit on an exception.
                self.stream.synchronize()
