"""Real runtime/queue contracts with a deterministic delayed CPU worker.

No learned clock correction and no deployment-clock credits are involved. The
fixture controls worker launch, preprocessing and model durations independently.
"""
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig
from dynamicwam.dualrate.timing import BufferUnderrun, DeadlineMiss, ExecutionQueue, TimedPlan


H = 0.004000000189989805  # actual float32 SAPIEN quantum
DT = 45 * H


class ManualPool:
    def __init__(self):
        self.pending = []

    def submit(self, callback):
        result = Future()
        self.pending.append((callback, result))
        return result

    def run_one(self):
        callback, result = self.pending.pop(0)
        try:
            result.set_result(callback())
        except BaseException as exc:
            result.set_exception(exc)
        return result

    def shutdown(self, **kwargs):
        while self.pending:
            self.run_one()


class TinySystem(nn.Module):
    def __init__(self, wall):
        super().__init__()
        self.fast = nn.Linear(1, 1)
        self.slow = nn.Linear(1, 1)
        self.fast.config = SimpleNamespace(action_dim=1, chunk_size=16)
        self.wall = wall
        self.model_seconds = .02
        self.seen = None

    def empty_world(self, batch):
        return {}

    def sample_fast(self, batch, world, **kwargs):
        self.seen = {key: value.clone() for key, value in batch.items()}
        self.wall[0] += self.model_seconds
        return {"actions": torch.full((1, 16, 1), 100.)}


@contextmanager
def live_runtime(*, physical=0., lag=0., rows=16):
    p, w, events = [physical], [100. + physical + lag], []
    bridge = WallPhysicsBridge(H, wall_clock=lambda: w[0])
    bridge.activate(0., 100.)
    queue = ExecutionQueue(TimedPlan(0., DT, tuple((float(i),) for i in range(rows)), 1, 0.))
    queue.action_at(0.)
    system = TinySystem(w).eval()
    runtime = DualRateRuntime(system, queue, lambda: p[0],
        RuntimeConfig(DT, min_query_budget_seconds=.3, max_protect_steps=3),
        action_mean=[0.], action_scale=[1.], clock_bridge=bridge, record=events.append)
    runtime.pool.shutdown(wait=True)
    runtime.pool = ManualPool()
    try:
        yield runtime, queue, p, w, events
    finally:
        runtime.close()


def batch_for(lease):
    return {
        "query_time": torch.tensor([lease.query_time], dtype=torch.float64),
        "action_times": torch.tensor(
            [[lease.prefix.start_time + i * DT for i in range(16)]], dtype=torch.float64),
    }


@pytest.mark.parametrize("phase,lag,expected", [(0., 0., 2), (.08, 0., 3), (0., .08, 3)])
def test_real_phase_and_lag_choose_the_trained_two_or_three_row_lease(phase, lag, expected):
    with live_runtime(physical=phase, lag=lag) as (runtime, queue, p, w, events):
        runtime.start_fast(lambda lease, ref: batch_for(lease), observation_time=0.)
        lease = runtime.fast_job[1]
        assert lease.query_time == phase
        assert lease.prefix.start_time == 0.
        assert len(lease.prefix.actions) == expected
        assert lease.protected_until == expected * DT
        start = events[-1]
        assert start["query_phase_seconds"] == pytest.approx(phase)
        assert start["wall_physics_lag_seconds"] == pytest.approx(lag)
        assert lease.protected_until - lease.submit_margin - (w[0] - 100.) >= .3


def test_owner_snapshot_survives_worker_wait_and_advancing_execution():
    with live_runtime(physical=.08, lag=.05) as (runtime, queue, p, w, events):
        observed = []
        def preprocess(lease, reference):
            observed.append((lease.query_time, lease.prefix.actions, p[0], w[0]))
            w[0] += .06
            return batch_for(lease)
        runtime.start_fast(preprocess, observation_time=0.)
        captured = runtime.fast_job[1]
        p[0] = DT
        queue.action_at(p[0])
        w[0] = 100.25
        runtime.pool.run_one()
        plan, output, ready, metrics = runtime.fast_job[0].result()
        assert observed == [(.08, captured.prefix.actions, DT, 100.25)]
        assert plan.query_time == .08 and plan.start_time == 0.
        assert metrics["wall_seconds"] == pytest.approx(.20)
        assert metrics["worker_started_wall_time"] == 100.25
        assert metrics["model_started_wall_time"] == pytest.approx(100.31)
        assert ready == pytest.approx(100.33)
        assert runtime.system.seen["query_time"].item() == .08
        assert runtime.system.seen["fast_known_mask"].sum().item() == 3
        assert plan.actions[:3] == captured.prefix.actions
        assert runtime.poll() is None  # physical time has not reached real wall-ready
        assert queue._version == 1
        p[0] = 84 * H
        assert runtime.poll() is output
        assert queue._version == 2
        assert queue.action_at(2 * DT) == (2.,)
        assert queue.action_at(3 * DT) == (100.,)


def test_worker_wait_and_preprocessing_are_charged_even_with_a_short_model():
    with live_runtime() as (runtime, queue, p, w, events):
        runtime.system.model_seconds = .08
        def preprocess(lease, reference):
            w[0] += .20
            return batch_for(lease)
        runtime.start_fast(preprocess, observation_time=0.)
        p[0] = .08
        w[0] = 100.08
        runtime.pool.run_one()
        plan, output, ready, metrics = runtime.fast_job[0].result()
        assert ready == pytest.approx(100.36)
        assert metrics["wall_seconds"] == pytest.approx(.36)
        assert ready - metrics["model_started_wall_time"] == pytest.approx(.08)
        with pytest.raises(DeadlineMiss, match="wall-clock"):
            runtime.poll()
        assert p[0] == .08 and queue._version == 1 and queue._inflight is None
        assert runtime.clock_bridge.wall_origin == 100.
        assert not any(item["event"] == "fast_committed" for item in events)


def test_budget_cap_rejection_acquires_no_lease_and_does_not_postpone_next_query():
    with live_runtime(physical=.17, lag=.25) as (runtime, queue, p, w, events):
        before = (dict(queue._rows), queue._committed_end, queue._query_counter, runtime.last_fast_tick)
        with pytest.raises(DeadlineMiss, match="bounded protected prefix"):
            runtime.start_fast(lambda lease, ref: batch_for(lease), observation_time=0.)
        assert (dict(queue._rows), queue._committed_end, queue._query_counter, runtime.last_fast_tick) == before
        assert runtime.fast_job is None and queue._inflight is None and runtime.due(p[0])[0]
        assert not runtime.pool.pending
        # Real catch-up to the same wall instant restores budget; it does not
        # shift either origin or claim that the deferred request ran earlier.
        p[0] = .42
        runtime.start_fast(lambda lease, ref: batch_for(lease), observation_time=0.)
        assert runtime.fast_job[1].query_time == .42
        assert runtime.clock_bridge.wall_origin == 100.


def test_queue_tail_cannot_be_extended_with_invented_holds_for_a_three_row_lease():
    with live_runtime(lag=.08, rows=2) as (runtime, queue, p, w, events):
        before = (dict(queue._rows), queue._committed_end, queue._query_counter, runtime.last_fast_tick)
        with pytest.raises(BufferUnderrun, match="absent from the queue"):
            runtime.start_fast(lambda lease, ref: batch_for(lease), observation_time=0.)
        assert (dict(queue._rows), queue._committed_end, queue._query_counter, runtime.last_fast_tick) == before
        assert queue._inflight is None and runtime.fast_job is None and not runtime.pool.pending
