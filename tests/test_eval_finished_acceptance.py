"""CPU acceptance ordering; fake video bytes do not establish GPU/video acceptance."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import eval_domino_motionkv_rtc as evaluation
import video_artifacts


def report_fixture(rate=.25):
    def request(index):
        observation = 10. if index == 0 else 10.4
        received = .4 if index == 0 else 2.
        dispatch = 0. if index == 0 else 1.6
        source_age = observation - 10.
        async_info = dict(slow_schedule="asynchronous_latest_completed_v1",
            source_action_index=0, source_sim_time=10., cache_age_actions=0,
            cache_age_seconds=source_age, source_submitted_monotonic=100.,
            source_published_monotonic=100.1, source_consumed_monotonic=100.2 + index,
            generation=0)
        return dict(query_index=index, prefix_length=0 if index == 0 else 4,
            action_origin=0, cache_age_actions=0, simulation_rate=rate,
            observation_wall_s=dispatch, dispatch_wall_s=dispatch, receive_wall_s=received,
            dispatch_to_receive_wall_s=received-dispatch,
            observation_physics_s=observation, submit_physics_s=observation,
            receive_physics_s=10. + rate * received, global_scaled_clock_error_s=0.,
            installed=True, install_physics_s=10. + rate * received,
            install_wall_s=received, bootstrap_actions_executed=False,
            suffix_not_before_physics_s=observation+.64, suffix_effective_physics_s=observation+.64,
            action_target_offsets=[i*.16 for i in range(1,17)],
            policy_diagnostics=dict(motion_interface_version=3, source_bridge="student_joint",
                asynchronous_world=async_info, cache_age_actions=0,
                cache_age_seconds=source_age, bootstrap_only=index == 0,
                fast_sampler_ms=0. if index == 0 else 100.))
    return dict(protocol=evaluation.evaluation_protocol(rate, True), joint_v3=True,
        bootstrap_mode="prime_then_hold", simulation_rate=rate,
        realtime_valid=rate == 1., clock_valid=True, continuous_valid=True,
        controller="committed_clean_prefix_rtc_fixed_grid_h4r4",
        clock="actual_wall_clock" if rate == 1. else "scaled_wall_clock",
        virtual_delay=False, startup_delay_excluded=False,
        rtc_strategy="trained_clean_prefix_conditioning", rtc_prefix_max=4,
        execute_horizon=4, slow_refresh_actions=16, limit_retiming=False,
        rows_dropped=0, guided_requests=0, asynchronous_camera_readback=True,
        clock_limits=dict(max_backlog_s=.25, inference_clock_error_s=.02),
        max_backlog_s=.01, max_inference_clock_error_s=.001,
        action_interval_s=.16, initial_physics_s=10., action_grid_start_physics_s=10.4,
        physics_time_s=1., wall_time_s=4., task="fixture", episode_index=0,
        seed=7, requested_seed=7, success=False, video_required=True,
        cohort_scheduling=dict(finished=None), requests=[request(0), request(1)], blocks=[])


class FinishedAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.report = report_fixture()
        self.events = []
        self.wall = 4.
        outer = self
        class Cohort:
            def wait(self, episode, phase):
                outer.events.append("barrier")
                outer.wall += 100.
                return dict(episode=episode, phase=phase)
        self.cohort = Cohort()
        self.video_check = patch.object(video_artifacts, "validate_video_artifacts", return_value=True)
        self.video_check.start()
        self.addCleanup(self.video_check.stop)
        self.addCleanup(self.tmp.cleanup)
        self.save_fixture()

    def save_fixture(self):
        evaluation.save(self.directory / "report.json", self.report)
        evaluation.save(self.directory / "manifest.json", {k:self.report[k]
            for k in ("task", "episode_index", "seed", "requested_seed")})
        evaluation.save(self.directory / "metrics.json", dict(seed=7, success=False, manipulation_score=10.))
        (self.directory / "episode0.mp4").write_bytes(b"preexisting-video-sentinel")

    def finish(self, replay=None):
        evaluation.finish_checked_episode(self.directory, "fixture", 0, .25,
            protocol=evaluation.evaluation_protocol(.25, True), report=self.report,
            cohort=self.cohort, replay_recorder=replay)

    def test_complete_validation_precedes_replay_and_barrier_without_changing_clocks(self):
        original = evaluation.checked_episode
        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            self.events.append("validated")
            return result
        outer = self
        class Replay:
            directory = outer.directory / "replay"
            def finalize(self, **kwargs):
                outer.events.append("replay")
                outer.assertEqual(kwargs, dict(success=False, complete=True))
        before = {k:self.report[k] for k in ("initial_physics_s", "physics_time_s", "wall_time_s")}
        with patch.object(evaluation, "checked_episode", side_effect=checked):
            self.finish(Replay())
        self.assertEqual(self.events, ["validated", "replay", "barrier"])
        saved = evaluation.read(self.directory / "report.json")
        self.assertEqual({k:saved[k] for k in before}, before)
        self.assertEqual(saved["cohort_scheduling"]["finished"]["phase"], "finished")
        self.assertEqual(self.wall, 104.)

    def test_invalid_peak_preserves_all_saved_evidence_and_never_releases(self):
        self.report["max_backlog_s"] = .33278791466727853
        self.save_fixture()
        before = {p.name:p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, "clock limits"):
            self.finish()
        self.assertEqual(self.events, [])
        self.assertEqual({p.name:p.read_bytes() for p in self.directory.iterdir()}, before)

    def test_actual_false_clock_flags_preserve_report_before_rejection(self):
        self.report["clock_valid"] = False
        self.save_fixture()
        with self.assertRaises(ValueError):
            self.finish()
        self.assertEqual(self.events, [])
        self.assertFalse(evaluation.read(self.directory / "report.json")["clock_valid"])

    def test_full_prefix_validation_runs_before_barrier(self):
        self.report["requests"][1]["prefix_length"] = 3
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "prefix length"):
            self.finish()
        self.assertEqual(self.events, [])

    def test_video_validation_failure_does_not_release_peers(self):
        with patch.object(video_artifacts, "validate_video_artifacts", return_value=False):
            with self.assertRaisesRegex(ValueError, "video validation"):
                self.finish()
        self.assertEqual(self.events, [])
        self.assertTrue((self.directory / "episode0.mp4").exists())

    def test_metric_identity_failure_does_not_release_peers(self):
        metric = evaluation.read(self.directory / "metrics.json")
        metric["seed"] = 8
        evaluation.save(self.directory / "metrics.json", metric)
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            self.finish()
        self.assertEqual(self.events, [])

    def test_main_has_no_early_finished_barrier_or_clock_flag_raise(self):
        tree = ast.parse(Path(evaluation.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        waits = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "wait"
            and len(n.args)>1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "finished"]
        self.assertEqual(waits, [])
        finalizers = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "finish_checked_episode"]
        self.assertEqual(len(finalizers), 1)
        old_gates = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "valid_clock_identity"]
        self.assertEqual(old_gates, [])
        video = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "finalize"
            and any(k.arg == "episode_end_physics_s" for k in n.keywords))
        endpoint = next(k.value for k in video.keywords if k.arg == "episode_end_physics_s")
        self.assertIsInstance(endpoint, ast.Name)
        self.assertEqual(endpoint.id, "episode_end_physics_s")


class ScaledContractTests(unittest.TestCase):
    def test_explicit_quarter_rate_uses_separate_joint_protocol(self):
        self.assertEqual(evaluation.evaluation_protocol(), evaluation.PROTOCOL)
        self.assertEqual(evaluation.evaluation_protocol(1., True), evaluation.JOINT_V3_PROTOCOL)
        self.assertEqual(evaluation.evaluation_protocol(.25, True), evaluation.JOINT_V3_SCALED_PROTOCOL)
        for rate in (.5, 0., float("nan")):
            with self.assertRaises(ValueError):
                evaluation.evaluation_protocol(rate, True)
        report = report_fixture()
        evaluation.validate_rtc_report(report)
        self.assertFalse(report["realtime_valid"])
        self.assertEqual(report["action_interval_s"], .16)
        self.assertEqual(report["rtc_prefix_max"], 4)

    def test_scaled_cannot_claim_one_x_or_escape_physical_cache_age(self):
        report = report_fixture()
        report["realtime_valid"] = True
        with self.assertRaisesRegex(ValueError, "clock identity"):
            evaluation.validate_rtc_report(report)
        report = report_fixture()
        report["requests"][1]["policy_diagnostics"]["cache_age_seconds"] = 5.
        with self.assertRaisesRegex(ValueError, "source age"):
            evaluation.validate_rtc_report(report)

    def test_scaled_wall_deadline_stays_on_physical_h4(self):
        report = report_fixture()
        # 4.2 wall seconds corresponds to physical 11.05, beyond H4=11.04.
        report["requests"][1]["install_wall_s"] = 4.2
        with self.assertRaisesRegex(ValueError, "declared-clock deadline"):
            evaluation.validate_rtc_report(report)


if __name__ == "__main__":
    unittest.main()
