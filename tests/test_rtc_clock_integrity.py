"""CPU regression for real camera backlog, fixed scaled v3 and immutable grids."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_motionkv_rtc_controller import Clock, Session, Worker
from dynamicwam.runtime.domino_continuous.continuous_rtc import (
    JOINT_V3_PROTOCOL, JOINT_V3_SCALED_PROTOCOL, RealtimeViolation,
    protocol_for_rate, run_continuous_rtc_episode,
)


class SubmitStallSession(Session):
    def __init__(self, clock, *, first_submit_s=.3, **kwargs):
        super().__init__(clock, **kwargs)
        self.first_submit_s = first_submit_s
        self.capture_count = 0

    def begin_capture(self, *, for_inference):
        super().begin_capture(for_inference=for_inference)
        if self.capture_count == 0:
            self.clock.sleep(self.first_submit_s)
        self.capture_count += 1


class ClockIntegrityTests(unittest.TestCase):
    def run_case(self, *, rate=1., first_submit_s=.0, durations=(.05,), step_cost=0.):
        clock = Clock()
        session = SubmitStallSession(clock, interval_s=.16, first_submit_s=first_submit_s,
                                     step_cost=step_cost)
        worker = Worker(clock, durations=durations)
        report = run_continuous_rtc_episode(session, worker, task="clock", episode_seed=3,
            action_limit=24, action_interval_s=.16, frame_period_s=.16, joint_v3=True,
            simulation_rate=rate, clock=clock, sleep=clock.sleep)
        return session, worker, report

    def test_only_explicit_joint_scaled_protocol_is_accepted(self):
        self.assertEqual(protocol_for_rate(1., joint_v3=True), JOINT_V3_PROTOCOL)
        self.assertEqual(protocol_for_rate(.25, joint_v3=True), JOINT_V3_SCALED_PROTOCOL)
        for value in (.5, .1, 0., float("nan")):
            with self.assertRaises(ValueError):
                protocol_for_rate(value, joint_v3=True)

    def test_recovered_render_peak_cannot_claim_clock_valid(self):
        _, _, report = self.run_case(first_submit_s=.3)
        self.assertEqual(report["completed_targets"], 24)
        self.assertAlmostEqual(report["max_backlog_s"], .3)
        self.assertFalse(report["clock_valid"])
        self.assertFalse(report["continuous_valid"])
        self.assertFalse(report["realtime_valid"])
        self.assertEqual(report["clock_validation_failures"], ["max_backlog_s"])
        event = report["max_backlog_event"]
        self.assertEqual(event["phase"], "camera_submit")
        self.assertEqual(event["query_index"], 0)
        self.assertEqual(event["action_origin"], 0)
        self.assertEqual(event["physics_s"], 0.)
        self.assertAlmostEqual(event["backlog_s"], event["wall_s"] - event["physics_s"])
        self.assertEqual(report["first_backlog_violation"], event)
        self.assertTrue(event["pending_capture"])
        self.assertAlmostEqual(report["captures"][0]["post_submit_backlog_s"], .3)

    def test_scaled_world_keeps_same_physical_targets_and_deadlines(self):
        session, worker, report = self.run_case(rate=.25, first_submit_s=.3, durations=(.12, .5))
        self.assertEqual(report["protocol"], JOINT_V3_SCALED_PROTOCOL)
        self.assertTrue(report["joint_v3"])
        self.assertTrue(report["clock_valid"] and report["continuous_valid"])
        self.assertFalse(report["realtime_valid"])
        self.assertIsNone(report["first_backlog_violation"])
        self.assertEqual(report["clock_validation_failures"], [])
        self.assertAlmostEqual(report["max_backlog_s"], .075)
        self.assertEqual(report["completed_targets"], 24)
        self.assertEqual(session.native_actions, 24)
        self.assertAlmostEqual(report["realtime_factor"], .25)
        self.assertEqual(report["bootstrap_mode"], "prime_then_hold")
        self.assertFalse(report["startup_delay_excluded"])
        self.assertFalse(report["requests"][0]["bootstrap_actions_executed"])
        grid = report["action_grid_start_physics_s"]
        for packet, request in zip(worker.calls[1:], report["requests"][1:]):
            origin = request["action_origin"]
            self.assertEqual(packet["_rtc_prefix_length"], 4)
            self.assertAlmostEqual(request["observation_physics_s"], grid + origin * .16)
            self.assertAlmostEqual(request["suffix_not_before_physics_s"], grid + (origin + 4) * .16)
            np.testing.assert_allclose(packet["sim_time"] + packet["_rtc_action_target_offsets"],
                grid + (origin + np.arange(1, 17)) * .16, rtol=0, atol=1e-8)
            self.assertLessEqual(.25 * request["install_wall_s"] + report["initial_physics_s"],
                                 request["suffix_not_before_physics_s"] + 1e-8)
            self.assertEqual(request["rows_dropped"], 0)
        for block in report["blocks"]:
            self.assertAlmostEqual(block["start_physics_s"], grid + block["origin"] * .16)
            self.assertAlmostEqual(block["end_physics_s"] - block["start_physics_s"], block["count"] * .16)

    def test_scaled_mode_still_rejects_late_suffix(self):
        with self.assertRaisesRegex(RealtimeViolation, "late"):
            self.run_case(rate=.25, durations=(.05, 2.7))

    def test_backlog_failure_retains_phase_evidence(self):
        with self.assertRaisesRegex(RealtimeViolation, "backlog") as caught:
            self.run_case(step_cost=.031, durations=(.5,))
        diagnostics = caught.exception.diagnostics
        self.assertGreater(diagnostics["observed_max_backlog_s"], .25)
        self.assertIsNotNone(diagnostics["max_backlog_event"])
        self.assertIsNotNone(diagnostics["first_backlog_violation"])

    def test_clock_limits_cannot_be_silently_relaxed(self):
        for extra in ({"max_backlog_s": .251}, {"infer_clock_tolerance_s": .021}):
            clock = Clock()
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                run_continuous_rtc_episode(Session(clock), Worker(clock), task="clock", episode_seed=0,
                    action_limit=4, clock=clock, sleep=clock.sleep, **extra)


if __name__ == "__main__":
    unittest.main()
