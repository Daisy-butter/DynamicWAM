"""Bounded, observational timing of the continuous physics owner.

No timers drive execution and no GC policy is changed. Wall-minus-thread-CPU
is only a scheduling/native-wait indicator, not proof of its particular cause.
"""
from collections import deque
from contextlib import contextmanager
import gc
import threading
import time


class LoopDiagnostics:
    def __init__(self, *, wall_clock=None, cpu_clock=None, tick_capacity=128,
                 long_span_seconds=.01):
        self.wall = wall_clock or time.perf_counter
        self.cpu = cpu_clock or time.thread_time
        self.owner = threading.get_ident()
        self.ticks = deque(maxlen=tick_capacity)
        self.long_spans = deque(maxlen=64)
        self.gc_events = deque(maxlen=32)
        self.threshold = float(long_span_seconds)
        self.aggregates = {}
        self.tick = None
        self.tick_count = 0
        self.gc_totals = dict(count=0, wall_seconds=0., thread_cpu_seconds=0., max_wall_seconds=0.)
        self._gc_start = None
        self._callback = self._gc_callback
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError('loop diagnostics GC callback already installed')
        gc.callbacks.append(self._callback)
        self.installed = True

    def close(self):
        if self.installed:
            for index, callback in enumerate(gc.callbacks):
                if callback is self._callback:
                    del gc.callbacks[index]
                    break
            self.installed = False

    def _gc_callback(self, phase, info):
        if threading.get_ident() != self.owner:
            return
        if phase == 'start':
            self._gc_start = (self.wall(), self.cpu(), info.get('generation'))
        elif phase == 'stop' and self._gc_start is not None:
            started, cpu_started, generation = self._gc_start
            self._gc_start = None
            wall, cpu = self.wall() - started, self.cpu() - cpu_started
            event = dict(started_wall=started, generation=generation,
                         wall_seconds=wall, thread_cpu_seconds=cpu,
                         collected=info.get('collected'), uncollectable=info.get('uncollectable'))
            self.gc_events.append(event)
            self.gc_totals['count'] += 1
            self.gc_totals['wall_seconds'] += wall
            self.gc_totals['thread_cpu_seconds'] += cpu
            self.gc_totals['max_wall_seconds'] = max(self.gc_totals['max_wall_seconds'], wall)

    @contextmanager
    def span(self, name, *, physical_time=None):
        started, cpu_started = self.wall(), self.cpu()
        try:
            yield
        finally:
            wall, cpu = self.wall() - started, self.cpu() - cpu_started
            item = dict(wall_seconds=wall, thread_cpu_seconds=cpu)
            aggregate = self.aggregates.setdefault(name, dict(
                count=0, wall_seconds=0., thread_cpu_seconds=0., max_wall_seconds=0.))
            aggregate['count'] += 1
            aggregate['wall_seconds'] += wall
            aggregate['thread_cpu_seconds'] += cpu
            aggregate['max_wall_seconds'] = max(aggregate['max_wall_seconds'], wall)
            if self.tick is not None:
                self.tick['spans'][name] = item
            if wall >= self.threshold:
                self.long_spans.append(dict(name=name, started_wall=started,
                    physical_time=physical_time, tick=None if self.tick is None else self.tick['tick'], **item))

    def begin_tick(self, tick, physical_time, lag):
        if self.tick is not None:
            raise RuntimeError('diagnostic real ticks must not overlap')
        self.tick = dict(tick=int(tick), physical_before=float(physical_time),
            lag_before=float(lag), started_wall=self.wall(), started_cpu=self.cpu(), spans={})

    def finish_tick(self, physical_time, lag, *, complete=True):
        if self.tick is None:
            return
        item, self.tick = self.tick, None
        item.update(physical_after=float(physical_time), lag_after=float(lag), complete=bool(complete),
                    wall_seconds=self.wall()-item['started_wall'],
                    thread_cpu_seconds=self.cpu()-item.pop('started_cpu'))
        self.ticks.append(item)
        self.tick_count += 1

    def snapshot(self, *, include_ticks=False):
        result = dict(schema='continuous_owner_loop_timing_v1',
            interpretation='thread_cpu excludes other threads; wall minus CPU may include native waits or descheduling',
            tick_count=self.tick_count, tick_capacity=self.ticks.maxlen,
            long_span_threshold_seconds=self.threshold,
            phase_totals={name: dict(value) for name, value in self.aggregates.items()},
            long_spans=list(self.long_spans), owner_gc_totals=dict(self.gc_totals),
            owner_gc_events=list(self.gc_events))
        if include_ticks:
            result['recent_ticks'] = list(self.ticks)
        return result
