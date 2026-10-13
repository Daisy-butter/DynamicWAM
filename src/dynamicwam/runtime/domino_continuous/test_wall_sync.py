"""Behavioral CPU tests for actual-time synchronous DOMINO execution.

The fake worker has a wall-clock completion deadline: only elapsed clock
time can complete a request. Advancing physics cannot complete inference.
No test supplies a virtual latency to the controller under test.
"""
from __future__ import annotations

import inspect
import unittest

import numpy as np

try:
    from .wall_sync import RealtimeViolation, run_wall_sync_episode, sampling_seed
except ImportError:  # Also allow ``python test_wall_sync.py`` in a frozen bundle.
    from wall_sync import RealtimeViolation, run_wall_sync_episode, sampling_seed


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)

    def sleep(self, seconds):
        if seconds < 0:
            raise AssertionError("negative sleep")
        self.sleeps.append(seconds)
        self.advance(seconds)


class Segment:
    def __init__(self, row, ticks):
        self.row = row
        self.remaining = ticks

    @property
    def finished(self):
        return self.remaining == 0


class FakeSession:
    def __init__(self, clock, *, dt=.01, capture_cost=0.0, plan_cost=0.0,
                 step_cost=0.0, terminal_tick=None, terminal_reason="out_of_bounds",
                 segment_ticks=None):
        self.clock = clock
        self.dt = dt
        self.capture_cost = capture_cost
        self.plan_cost = plan_cost
        self.step_cost = step_cost
        self.terminal_tick = terminal_tick
        self.terminal_reason = terminal_reason
        self.segment_ticks = segment_ticks or (lambda row: 2 + row % 3)
        self.worker = None
        self.ticks = 0
        self.samples = []
        self.packet_calls = self.history_calls = 0
        self.plans = []
        self.steps = []
        self.metrics = []
        self.native_action_count = 0

    def timestep(self):
        return self.dt

    def physics_time(self):
        return self.ticks * self.dt

    def object_position(self):
        return np.array([2 * self.physics_time(), 0.0, 0.0])

    def capture_packet(self):
        self.packet_calls += 1
        packet = {"captured_physics_s": self.physics_time()}
        self.samples.append((self.physics_time(), self.clock()))
        self.clock.advance(self.capture_cost)
        return packet

    def capture_history(self):
        self.history_calls += 1
        self.samples.append((self.physics_time(), self.clock()))
        self.clock.advance(self.capture_cost)

    def plan(self, action):
        row = int(action[0])
        self.plans.append((row, self.physics_time(), self.clock()))
        self.clock.advance(self.plan_cost)
        return Segment(row, self.segment_ticks(row))

    def begin_native_action(self):
        self.native_action_count += 1

    def step(self, segment):
        self.steps.append({
            "physics_before": self.physics_time(), "wall": self.clock(),
            "row": None if segment is None else segment.row,
            "worker_busy": bool(self.worker and self.worker.busy),
            "worker_compute_pending": bool(
                self.worker and self.worker.busy and self.clock() < self.worker.ready_at - 1e-9
            ),
        })
        self.ticks += 1
        if segment is not None:
            segment.remaining -= 1
        self.clock.advance(self.step_cost)

    def record_target_metrics(self):
        self.metrics.append((self.physics_time(), self.native_action_count))

    def stop_reason(self):
        if self.terminal_tick is not None and self.ticks >= self.terminal_tick:
            return self.terminal_reason
        return None


class AsyncFakeSession(FakeSession):
    def begin_capture(self, *, for_inference):
        if getattr(self, "capture_pending", False):
            raise AssertionError("overlapping capture")
        self.capture_pending = True
        self.capture_for_inference = for_inference
        self.frozen_capture = (self.physics_time(), self.clock())
        self.capture_deadline = self.clock() + self.capture_cost

    def capture_ready(self):
        return self.clock() + 1e-9 >= self.capture_deadline

    def finish_capture(self):
        if not self.capture_ready():
            raise AssertionError("readback not complete")
        self.capture_pending = False
        self.samples.append(self.frozen_capture)
        if self.capture_for_inference:
            self.packet_calls += 1
            return {"captured_physics_s": self.frozen_capture[0]}
        self.history_calls += 1
        return None


class FakeWorker:
    def __init__(self, clock, session, *, durations=(.05,), first_poll_stall=0.0,
                 blocked_ready_poll=0.0, wrong_request_id=False, bad_shape=False,
                 nonfinite=False):
        self.clock = clock
        self.session = session
        session.worker = self
        self.durations = durations
        self.first_poll_stall = first_poll_stall
        self.blocked_ready_poll = blocked_ready_poll
        self.wrong_request_id = wrong_request_id
        self.bad_shape = bad_shape
        self.nonfinite = nonfinite
        self.busy = False
        self.ready_at = 0.0
        self.calls = []
        self.poll_count = 0

    def submit(self, packet):
        if self.busy:
            raise AssertionError("second inference request in flight")
        self.busy = True
        self.request_id = len(self.calls)
        self.started = self.clock()
        self.ready_at = self.started + self.durations[min(self.request_id, len(self.durations) - 1)]
        self.calls.append({"packet": dict(packet), "wall": self.started,
                           "physics": self.session.physics_time(),
                           "started_targets": self.session.native_action_count})
        return self.request_id

    def poll(self):
        self.poll_count += 1
        if self.poll_count == 1 and self.first_poll_stall:
            self.clock.advance(self.first_poll_stall)
            return None
        if self.clock() + 1e-9 < self.ready_at:
            return None
        self.clock.advance(self.blocked_ready_poll)
        self.busy = False
        actions = np.repeat(np.arange(16)[:, None], 14, axis=1).astype(float)
        if self.bad_shape:
            actions = actions[:15]
        if self.nonfinite:
            actions[0, 0] = np.nan
        return {"request_id": self.request_id + int(self.wrong_request_id),
                "actions": actions, "started": self.started, "finished": self.ready_at}


class WallSyncTests(unittest.TestCase):
    def run_case(self, *, action_limit=32, durations=(.05,), session_args=None,
                 worker_args=None, runner_args=None):
        clock = FakeClock()
        session = FakeSession(clock, **(session_args or {}))
        worker = FakeWorker(clock, session, durations=durations, **(worker_args or {}))
        records = []
        report = run_wall_sync_episode(
            session, worker, task="test_task", episode_seed=123,
            action_limit=action_limit, sink=records.append, clock=clock,
            sleep=clock.sleep, **(runner_args or {}),
        )
        self.assertEqual(records, report["requests"])
        return session, worker, report

    def run_async_capture_case(self, **session_args):
        clock = FakeClock()
        session = AsyncFakeSession(clock, **session_args)
        worker = FakeWorker(clock, session)
        report = run_wall_sync_episode(session, worker, task="test", episode_seed=1,
            action_limit=2, clock=clock, sleep=clock.sleep)
        return session, worker, report

    def test_long_camera_wait_steps_physics_without_loosening_backlog_gate(self):
        session, worker, report = self.run_async_capture_case(capture_cost=1.04)
        self.assertEqual(report["completed_targets"], 2)
        self.assertAlmostEqual(report["realtime_factor"], 1.0)
        self.assertLess(report["max_backlog_s"], .02)
        self.assertEqual(worker.calls[0]["packet"]["captured_physics_s"], 0)
        self.assertAlmostEqual(worker.calls[0]["physics"], 1.04)
        self.assertEqual(session.packet_calls, 1)
        self.assertEqual(session.history_calls, 1)
        self.assertTrue(all(c["asynchronous_readback"] and c["delivered"] for c in report["captures"]))
        self.assertTrue(all(c["physics_ticks_during_capture"] >= 100 for c in report["captures"]))

    def test_terminal_during_camera_wait_does_not_submit_or_finish_stale_frame(self):
        session, worker, report = self.run_async_capture_case(capture_cost=1.04, terminal_tick=3)
        self.assertEqual(report["stop_reason"], "out_of_bounds")
        self.assertEqual(worker.calls, [])
        self.assertEqual(session.samples, [])
        self.assertEqual(len(session.metrics), 1)
        self.assertFalse(report["captures"][0]["delivered"])
        self.assertEqual(report["captures"][0]["physics_ticks_during_capture"], 3)
        self.assertTrue(session.capture_pending)  # owner must drain before scene destruction

    def test_world_steps_while_model_is_actually_pending_including_first_query(self):
        session, worker, report = self.run_case(durations=(.07, .11))
        self.assertEqual(len(worker.calls), 2)
        self.assertFalse(report["virtual_delay"])
        self.assertFalse(report["startup_delay_excluded"])
        self.assertTrue(any(s["worker_compute_pending"] for s in session.steps))
        self.assertEqual(session.steps[0]["row"], None)
        self.assertTrue(session.steps[0]["worker_compute_pending"])
        for record, expected in zip(report["requests"], (.07, .11)):
            self.assertAlmostEqual(record["inference_physics_advance_s"], expected, places=8)
            self.assertAlmostEqual(record["inference_wall_elapsed_s"], expected, places=8)
            self.assertAlmostEqual(record["object_displacement_while_inference"][0], 2 * expected)
            self.assertGreater(record["physics_ticks_while_inference_outstanding"], 0)

    def test_queries_only_after_full_chunk_and_no_model_during_execution(self):
        session, worker, report = self.run_case(action_limit=48)
        self.assertEqual([c["started_targets"] for c in worker.calls], [0, 16, 32])
        self.assertEqual([r for r, _, _ in session.plans], list(range(16)) * 3)
        self.assertTrue(all(not s["worker_busy"] for s in session.steps if s["row"] is not None))
        for record in report["requests"]:
            self.assertEqual(record["started_rows"], list(range(16)))
            self.assertEqual(record["completed_rows"], list(range(16)))
            self.assertEqual(record["rows_dropped"], 0)
            self.assertFalse(record["guidance"])
            self.assertFalse(record["prefetch"])

    def test_slow_result_still_executes_row_zero_and_all_sixteen(self):
        session, _, report = self.run_case(action_limit=16, durations=(2.05,))
        self.assertEqual([p[0] for p in session.plans], list(range(16)))
        self.assertEqual(report["completed_targets"], 16)
        self.assertEqual(report["rows_dropped"], 0)
        self.assertGreater(report["requests"][0]["inference_physics_advance_s"], 1.6)

    def test_action_budget_not_physics_timeout_and_final_target_finishes(self):
        session, _, report = self.run_case(
            action_limit=17, durations=(.5,), session_args={"segment_ticks": lambda _: 50})
        self.assertEqual(report["started_targets"], 17)
        self.assertEqual(report["completed_targets"], 17)
        self.assertEqual(session.native_action_count, 17)
        self.assertEqual(report["requests"][-1]["completed_rows"], [0])
        self.assertEqual(report["stop_reason"], "action_budget_exhausted")
        self.assertGreater(report["physics_time_s"], 17 * .1)
        self.assertTrue(report["requests"][-1]["targets"][-1]["completed"])

    def test_history_once_per_native_action_and_never_during_hold(self):
        session, worker, report = self.run_case(durations=(.08, .2))
        self.assertEqual(len(session.samples), 32)
        self.assertEqual(session.packet_calls, 2)
        self.assertEqual(session.history_calls, 30)
        self.assertEqual(len(session.plans), 32)
        for call, record in zip(worker.calls, report["requests"]):
            samples_in_inference = [t for t, _ in session.samples
                                    if call["physics"] + 1e-9 < t < record["receive_physics_s"] - 1e-9]
            self.assertEqual(samples_in_inference, [])
        self.assertTrue(worker.calls[0]["packet"]["_reset_policy"])
        self.assertFalse(worker.calls[1]["packet"]["_reset_policy"])

    def test_capture_and_topp_overhead_advance_under_hold_before_new_command(self):
        session, _, report = self.run_case(
            action_limit=2, durations=(.04,),
            session_args={"capture_cost": .03, "plan_cost": .04})
        self.assertAlmostEqual(report["phase_physics_seconds"]["capture"], .06)
        self.assertAlmostEqual(report["phase_physics_seconds"]["planning"], .08)
        first = report["requests"][0]
        self.assertEqual(first["observation_physics_s"], 0)
        self.assertAlmostEqual(first["submit_physics_s"], .03)
        self.assertAlmostEqual(first["receive_physics_s"], .07)
        self.assertAlmostEqual(first["targets"][0]["start_physics_s"], .11)
        before_first = [s for s in session.steps if s["physics_before"] < .11 - 1e-9]
        self.assertTrue(all(s["row"] is None for s in before_first))
        self.assertAlmostEqual(report["realtime_factor"], 1.0)

    def test_ready_result_cannot_install_fresh_commands_into_overdue_slots(self):
        session, _, report = self.run_case(
            action_limit=1, durations=(.02,), worker_args={"first_poll_stall": .08})
        request = report["requests"][0]
        self.assertGreaterEqual(request["receive_physics_s"] + 1e-9, .08)
        overdue_steps = [s for s in session.steps if s["physics_before"] < .08 - 1e-9]
        self.assertTrue(overdue_steps)
        self.assertTrue(all(s["row"] is None for s in overdue_steps))
        self.assertAlmostEqual(request["inference_clock_error_s"], 0.0, places=8)
        self.assertEqual(request["physics_ticks_while_inference_outstanding"],
                         sum(s["worker_busy"] for s in session.steps))

    def test_terminal_during_initial_inference_records_metrics_without_execution(self):
        session, worker, report = self.run_case(
            durations=(.2,), session_args={"terminal_tick": 3})
        self.assertEqual(report["stop_reason"], "out_of_bounds")
        self.assertEqual(report["started_targets"], 0)
        self.assertEqual(session.plans, [])
        self.assertEqual(len(session.metrics), 1)
        self.assertFalse(report["requests"][0]["installed"])
        self.assertTrue(report["requests"][0]["terminal_before_result"])
        self.assertTrue(worker.busy)  # Evaluator must drain it before the next episode.

    def test_terminal_during_capture_or_planning_does_not_start_target(self):
        for costs, expected_calls in (({"capture_cost": .04}, 0),
                                      ({"plan_cost": .04}, 1)):
            with self.subTest(costs=costs):
                session, worker, report = self.run_case(
                    durations=(0.0,), session_args={"terminal_tick": 3, **costs})
                self.assertEqual(report["stop_reason"], "out_of_bounds")
                self.assertEqual(report["started_targets"], 0)
                self.assertEqual(len(worker.calls), expected_calls)
                self.assertEqual(len(session.metrics), 1)

    def test_success_can_end_inside_a_topp_segment(self):
        session, _, report = self.run_case(
            durations=(0.0,), session_args={"terminal_tick": 2,
                "terminal_reason": "success", "segment_ticks": lambda _: 20})
        self.assertTrue(report["success"])
        self.assertEqual(report["started_targets"], 1)
        self.assertEqual(report["completed_targets"], 0)
        self.assertEqual(len(session.metrics), 1)

    def test_capture_backlog_violation_is_not_silently_replayed(self):
        with self.assertRaisesRegex(RealtimeViolation, "backlog"):
            self.run_case(session_args={"capture_cost": .3})

    def test_slow_physics_fails_the_realtime_gate(self):
        with self.assertRaisesRegex(RealtimeViolation, "backlog"):
            self.run_case(session_args={"step_cost": .02}, durations=(1.0,),
                          runner_args={"max_backlog_s": .06})

    def test_blocking_result_poll_fails_inference_clock_gate(self):
        with self.assertRaisesRegex(RealtimeViolation, "inference clock mismatch"):
            self.run_case(worker_args={"blocked_ready_poll": .06})

    def test_wrong_response_or_invalid_full_chunk_fails_loudly(self):
        for setting in ("wrong_request_id", "bad_shape", "nonfinite"):
            with self.subTest(setting=setting):
                with self.assertRaises(RuntimeError):
                    self.run_case(worker_args={setting: True})

    def test_invalid_limits_and_busy_worker_rejected(self):
        for setting in ({"action_limit": 0}, {"runner_args": {"max_backlog_s": 0}},
                        {"runner_args": {"infer_clock_tolerance_s": 0}},
                        {"session_args": {"dt": 0}}):
            with self.subTest(setting=setting):
                with self.assertRaises(ValueError):
                    self.run_case(**setting)
        clock = FakeClock()
        session = FakeSession(clock)
        worker = FakeWorker(clock, session)
        worker.busy = True
        with self.assertRaisesRegex(RuntimeError, "drained"):
            run_wall_sync_episode(session, worker, task="x", episode_seed=1,
                                  action_limit=1, clock=clock, sleep=clock.sleep)

    def test_no_virtual_delay_argument_and_query_seed_is_stable(self):
        self.assertFalse(any("delay" in name for name in inspect.signature(run_wall_sync_episode).parameters))
        self.assertEqual(sampling_seed("x", 2, 3), sampling_seed("x", 2, 3))
        self.assertNotEqual(sampling_seed("x", 2, 3), sampling_seed("x", 2, 4))


if __name__ == "__main__":
    unittest.main()
