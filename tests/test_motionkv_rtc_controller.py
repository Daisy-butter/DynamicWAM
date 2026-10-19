"""CPU behavioural tests: actual elapsed time controls worker completion."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
import numpy as np

from dynamicwam.runtime.domino_continuous.continuous_rtc import (
    CommittedActionQueue, PROTOCOL, SCALED_PROTOCOL, protocol_for_rate, RealtimeViolation, run_continuous_rtc_episode,
)
from dynamicwam.runtime.domino_continuous.continuous_track_v2 import ContinuousJointTrack


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Session:
    limit_retiming = False
    action_interval_s = .1
    def __init__(self, clock, *, terminal_tick=None, capture_duration=.01, planning_cost=0., step_cost=0., interval_s=.1):
        self.clock = clock
        self.ticks = self.native_actions = self.metrics = 0
        self.terminal_tick, self.capture_duration = terminal_tick, capture_duration
        self.planning_cost, self.step_cost = planning_cost, step_cost
        self.action_interval_s = interval_s
        self.q = np.zeros(14)
        self.frames, self.commands = [], []
        self.pending = None
        self._q_limits = np.array([[-10., 10.]] * 14)
    def timestep(self): return .01
    def physics_time(self): return self.ticks * self.timestep()
    def object_position(self): return np.array([self.physics_time(), 0., 0.])
    def stop_reason(self): return "success" if self.terminal_tick is not None and self.ticks >= self.terminal_tick else None
    def current_qpos(self): return self.q.copy()
    def current_drive_targets(self): return self.q.copy()
    def history_diagnostics(self): return {"count": len(self.frames)}
    def begin_native_action(self): self.native_actions += 1
    def record_target_metrics(self): self.metrics += 1
    def begin_capture(self, *, for_inference):
        if self.pending is not None: raise AssertionError("overlapping camera")
        self.pending = (self.physics_time(), self.clock(), for_inference)
    def capture_ready(self): return self.clock() + 1e-9 >= self.pending[1] + self.capture_duration
    def finish_capture(self):
        pending = self.pending
        self.pending = None
        self.frames.append(pending)
        return {"frame_physics_s": pending[0]} if pending[2] else None
    def plan_chunk(self, actions):
        self.clock.sleep(self.planning_cost)
        return ContinuousJointTrack(self.q, actions, self.action_interval_s, np.ones(14), np.ones(14), limit_retiming=False)
    def step_command(self, position, velocity):
        self.commands.append((self.physics_time(), None if position is None else position.copy()))
        if position is not None: self.q = position.copy()
        self.ticks += 1
        self.clock.sleep(self.step_cost)


class Worker:
    def __init__(self, clock, durations=(.05,), *, mutate_prefix=False, poll_stall=0., wrong_id=False):
        self.clock, self.durations = clock, durations
        self.mutate_prefix, self.poll_stall, self.wrong_id = mutate_prefix, poll_stall, wrong_id
        self.busy = False
        self.calls = []
    def submit(self, packet):
        if self.busy: raise AssertionError("overlapping inference")
        self.busy = True
        self.calls.append(packet.copy())
        self.index = len(self.calls) - 1
        self.started = self.clock()
        self.ready = self.started + self.durations[min(self.index, len(self.durations) - 1)]
        return self.index
    def poll(self):
        if self.clock() + 1e-9 < self.ready: return None
        if self.index > 0: self.clock.sleep(self.poll_stall)
        self.busy = False
        rows = np.full((16, 14), .01 * (self.index + 1))
        d = self.calls[-1]["_rtc_prefix_length"]
        rows[:d] = self.calls[-1]["_rtc_prefix_actions"]
        if d and self.mutate_prefix: rows[0, 0] += 1
        return {"request_id": self.index + int(self.wrong_id), "actions": rows,
                "started": self.started, "finished": self.ready}


class QueueTests(unittest.TestCase):
    def make(self):
        queue = CommittedActionQueue(interval_s=.1)
        actions = np.arange(16 * 14).reshape(16, 14) / 1000.
        queue.bootstrap(actions)
        return queue, actions
    def test_atomic_suffix_replacement_keeps_exact_prefix(self):
        queue, actions = self.make()
        request = queue.reserve(0, 1.)
        output = actions.copy()
        output[4:] += 1
        queue.install(output, request, ready_physics_s=1.2)
        np.testing.assert_array_equal(queue.take(0, 4), actions[:4])
        np.testing.assert_array_equal(queue.take(4, 12), output[4:])
    def test_late_result_rejected_without_mutation(self):
        queue, actions = self.make()
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            queue.install(actions, queue.reserve(0, 1.), ready_physics_s=1.401)
        np.testing.assert_array_equal(queue.take(0, 16), actions)
    def test_prefix_overflow_and_underrun_rejected(self):
        queue, _ = self.make()
        with self.assertRaises(ValueError): queue.reserve(0, 0., length=5)
        with self.assertRaises(RealtimeViolation): queue.reserve(14, 0.)
    def test_stale_reservation_rejected(self):
        queue, actions = self.make()
        request = queue.reserve(0, 0.)
        queue.rows[0][0] += .5
        with self.assertRaisesRegex(RuntimeError, "stale"):
            queue.install(actions, request, ready_physics_s=.1)
    def test_nan_output_rejected(self):
        queue, actions = self.make()
        actions[5, 2] = np.nan
        with self.assertRaises(ValueError): queue.install(actions, queue.reserve(0, 0.), ready_physics_s=.1)


class ControllerTests(unittest.TestCase):
    def run_case(self, *, actions=20, durations=(.05,), session_args=None, worker_args=None, controller_args=None):
        clock = Clock()
        session = Session(clock, **(session_args or {}))
        worker = Worker(clock, durations, **(worker_args or {}))
        records = []
        report = run_continuous_rtc_episode(session, worker, task="test", episode_seed=123,
            action_limit=actions, clock=clock, sleep=clock.sleep, sink=records.append,
            action_interval_s=session.action_interval_s, frame_period_s=session.action_interval_s, **(controller_args or {}))
        self.assertEqual(records, report["requests"])
        return session, worker, report
    def test_continuous_world_and_h4_prefix_overlap(self):
        session, worker, report = self.run_case()
        self.assertEqual(report["protocol"], PROTOCOL)
        self.assertEqual(report["completed_targets"], 20)
        self.assertEqual(session.native_actions, 20)
        self.assertTrue(report["execution_inference_overlap"])
        self.assertAlmostEqual(report["realtime_factor"], 1.)
        self.assertEqual([p["_rtc_prefix_length"] for p in worker.calls], [0, 4, 4, 4, 4])
        self.assertEqual([p["_rtc_observation_action_index"] for p in worker.calls], [0, 0, 4, 8, 12])
        self.assertTrue(all(r["inference_physics_advance_s"] > 0 for r in report["requests"]))
        for block in report["blocks"]:
            self.assertAlmostEqual(block["end_physics_s"] - block["start_physics_s"], block["count"] * .1)
    def test_slow_bootstrap_is_measured_world_moves_under_hold(self):
        session, _, report = self.run_case(actions=8, durations=(1.1, .05))
        self.assertGreater(report["requests"][0]["inference_physics_advance_s"], 1.)
        self.assertGreater(report["object_displacement"][0], 1.8)
        self.assertFalse(report["startup_delay_excluded"])
        self.assertTrue(all(command is None for t, command in session.commands if t < 1.))
    def test_late_nonbootstrap_rejected_instead_of_dropping_rows(self):
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(durations=(.05, .5))
    def test_poll_delay_cannot_backdate_suffix_at_boundary(self):
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(durations=(.05, .35), worker_args={"poll_stall": .06})
    def test_prefix_mutation_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            self.run_case(worker_args={"mutate_prefix": True})
    def test_terminal_during_initial_inference_keeps_pending_for_owner_drain(self):
        session, worker, report = self.run_case(durations=(1.,), session_args={"terminal_tick": 4})
        self.assertTrue(report["success"])
        self.assertEqual(report["started_targets"], 0)
        self.assertTrue(worker.busy)
        self.assertTrue(report["requests"][0]["terminal_before_result"])
    def test_terminal_during_capture_never_submits_stale_frame(self):
        _, worker, report = self.run_case(session_args={"terminal_tick": 3, "capture_duration": .2})
        self.assertEqual(worker.calls, [])
        self.assertFalse(report["captures"][0]["delivered"])
    def test_final_partial_block_has_exact_action_budget(self):
        session, _, report = self.run_case(actions=9)
        self.assertEqual(report["completed_targets"], 9)
        self.assertEqual(session.native_actions, 9)
        self.assertEqual(report["blocks"][-1]["count"], 1)
    def test_wrong_response_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "out-of-order"):
            self.run_case(worker_args={"wrong_id": True})
    def test_calibrated_160ms_clock_and_cache_refresh_boundary(self):
        session, worker, report = self.run_case(actions=24, session_args={"interval_s": .16})
        self.assertEqual(report["completed_targets"], 24)
        self.assertAlmostEqual(report["action_interval_s"], .16)
        self.assertEqual([p["_rtc_observation_action_index"] for p in worker.calls], [0, 0, 4, 8, 12, 16])
        self.assertEqual([p["_rtc_cache_age_actions"] for p in worker.calls], [0, 0, 4, 8, 12, 0])
        for request in report["requests"][1:]:
            self.assertAlmostEqual(request["observation_to_suffix_effect_s"], .64)
            self.assertGreaterEqual(request["observation_to_first_command_wall_s"] + 1e-9, .64)
        self.assertGreater(report["skipped_history_grid_slots"], 0)
        self.assertGreater(report["capture_gap_physics_max_s"], report["history_frame_period_s"])
        self.assertEqual(report["started_targets"], session.native_actions)
    def test_retiming_refused(self):
        clock = Clock()
        session, worker = Session(clock), Worker(clock)
        session.limit_retiming = True
        with self.assertRaisesRegex(ValueError, "retiming"):
            run_continuous_rtc_episode(session, worker, task="x", episode_seed=1, action_limit=4)
    def test_physics_overload_does_not_slow_down_world_silently(self):
        with self.assertRaisesRegex(RealtimeViolation, "backlog"):
            self.run_case(session_args={"step_cost": .03}, durations=(.5,))

    def test_ready_capture_not_starved_by_catch_up_lag(self):
        """Render-induced lag must not drain before a ready inference capture.

        begin_capture blocks like update_picture (~100ms+). That creates wall
        lag while the async readback is already complete. Catch-up used to tick
        away the whole lag before finish_capture, burning the H4 budget before
        the worker even started. Ready captures must preempt catch-up.
        """
        clock = Clock()

        class LaggySession(Session):
            def begin_capture(self, *, for_inference):
                super().begin_capture(for_inference=for_inference)
                # Stay under max_backlog / clock-tolerance after finish drains
                # lag, but large enough that starved catch-up would record many
                # physics_ticks_during_capture.
                self.clock.sleep(0.15)

        session = LaggySession(clock, capture_duration=0.01)
        worker = Worker(clock, (.05,))
        report = run_continuous_rtc_episode(session, worker, task="test", episode_seed=7,
            action_limit=12, clock=clock, sleep=clock.sleep,
            action_interval_s=session.action_interval_s, frame_period_s=session.action_interval_s)
        self.assertEqual(report["completed_targets"], 12)
        inference = [c for c in report["captures"] if c.get("for_inference") and c.get("delivered")]
        self.assertGreaterEqual(len(inference), 2)
        # 0.15s lag / 0.01s dt = 15 ticks if catch-up ignores readiness.
        for capture in inference[1:]:
            self.assertLess(capture["physics_ticks_during_capture"], 5,
                            msg=f"capture still starved: {capture}")


    def test_declared_rate_validation(self):
        self.assertEqual(protocol_for_rate(1), PROTOCOL)
        self.assertEqual(protocol_for_rate(.25), SCALED_PROTOCOL)
        for invalid in (0, -.25, 1.01, float("inf"), float("nan")):
            with self.assertRaises(ValueError): protocol_for_rate(invalid)

    def test_default_strict_mode_equals_explicit_rate_one(self):
        default = self.run_case(actions=8)[2]
        explicit = self.run_case(actions=8, controller_args={"simulation_rate": 1.})[2]
        self.assertEqual(default, explicit)
        self.assertTrue(default["realtime_valid"])

    def test_scaled_wall_paces_world_without_retiming_actions(self):
        session, worker, report = self.run_case(actions=24, durations=(.12, .5),
            session_args={"interval_s": .16}, controller_args={"simulation_rate": .25})
        self.assertEqual(report["protocol"], SCALED_PROTOCOL)
        self.assertEqual(report["clock"], "scaled_wall_clock")
        self.assertFalse(report["realtime_valid"])
        self.assertTrue(report["continuous_valid"] and report["clock_valid"])
        self.assertFalse(report["virtual_delay"])
        self.assertAlmostEqual(report["realtime_factor"], .25)
        self.assertEqual(report["completed_targets"], 24)
        self.assertEqual([p["_rtc_cache_age_actions"] for p in worker.calls], [0, 0, 4, 8, 12, 0])
        for block in report["blocks"]:
            self.assertAlmostEqual(block["end_physics_s"] - block["start_physics_s"], block["count"] * .16)
            self.assertAlmostEqual(block["end_wall_s"] - block["start_wall_s"], block["count"] * .16 / .25)
        for request in report["requests"]:
            self.assertGreater(request["dispatch_to_receive_physics_s"], 0)
            self.assertGreater(request["object_displacement_while_inference"][0], 0)
            self.assertAlmostEqual(request["object_displacement_dispatch_to_receive"][0], request["dispatch_to_receive_physics_s"])
            self.assertGreaterEqual(request["receive_wall_s"] + 1e-8, request["model_finished_wall_s"])
        for request in report["requests"][1:]:
            self.assertAlmostEqual(request["observation_to_suffix_effect_s"], .64)
            self.assertGreaterEqual(request["observation_to_first_command_wall_s"] + 1e-8, .64 / .25)
            self.assertLessEqual(.25 * request["install_wall_s"] + report["initial_physics_s"],
                                 request["suffix_not_before_physics_s"] + 1e-8)

    def test_scaled_wall_accepts_compute_latency_only_within_fixed_h4(self):
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(actions=8, durations=(.05, .7))
        _, _, report = self.run_case(actions=8, durations=(.05, .7), controller_args={"simulation_rate": .25})
        self.assertEqual(report["completed_targets"], 8)
        self.assertGreater(report["requests"][1]["inference_wall_elapsed_s"], .4)
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(actions=8, durations=(.05, 1.7), controller_args={"simulation_rate": .25})

    def test_scaled_deadline_cannot_backdate_poll_delay(self):
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(actions=8, durations=(.05, 1.3), worker_args={"poll_stall": .4},
                          controller_args={"simulation_rate": .25})

    def test_declared_slow_rate_handles_physics_load_but_still_bounds_overload(self):
        _, _, report = self.run_case(actions=8, durations=(.1,), session_args={"step_cost": .03},
                                      controller_args={"simulation_rate": .25})
        self.assertEqual(report["completed_targets"], 8)
        self.assertLessEqual(report["max_backlog_s"], .25)
        with self.assertRaisesRegex(RealtimeViolation, "backlog"):
            self.run_case(session_args={"step_cost": .08}, durations=(.5,), controller_args={"simulation_rate": .25})

    def test_real_dispatch_latency_includes_post_capture_catch_up(self):
        clock = Clock()
        class LaggySession(Session):
            def begin_capture(self, *, for_inference):
                super().begin_capture(for_inference=for_inference)
                self.clock.sleep(.15)
        session = LaggySession(clock, capture_duration=.01, step_cost=.005)
        worker = Worker(clock, (.05,))
        report = run_continuous_rtc_episode(session, worker, task="x", episode_seed=1,
            action_limit=8, clock=clock, sleep=clock.sleep, simulation_rate=.25,
            action_interval_s=.1, frame_period_s=.1)
        self.assertTrue(any(r["dispatch_wall_s"] < r["submit_wall_s"] for r in report["requests"]))
        for request in report["requests"]:
            self.assertEqual(request["submit_timestamp_semantics"], "legacy_post_capture_catch_up")
            self.assertAlmostEqual(request["dispatch_to_receive_wall_s"], request["receive_wall_s"] - request["dispatch_wall_s"])
            self.assertGreaterEqual(request["dispatch_to_receive_wall_s"], request["inference_wall_elapsed_s"])
            self.assertGreaterEqual(request["model_started_wall_s"] + 1e-8, request["dispatch_wall_s"])


    def test_scaled_report_passes_evaluator_and_rejects_clock_tampering(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from eval_domino_motionkv_rtc import validate_rtc_report
        session, _, report = self.run_case(actions=24, durations=(.12, .5),
            session_args={"interval_s": .16}, controller_args={"simulation_rate": .25})
        report["physics_dt_s"] = session.timestep()  # Added by the real evaluator wrapper.
        validate_rtc_report(report)
        mismatch = copy.deepcopy(report)
        mismatch["simulation_rate"] = 1.
        with self.assertRaises(ValueError): validate_rtc_report(mismatch)
        late = copy.deepcopy(report)
        request = late["requests"][1]
        request["install_wall_s"] = (request["suffix_not_before_physics_s"] - late["initial_physics_s"]) / .25 + .1
        with self.assertRaises(ValueError): validate_rtc_report(late)


if __name__ == "__main__":
    unittest.main()
