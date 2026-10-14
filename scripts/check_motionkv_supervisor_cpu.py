#!/usr/bin/env python3
"""CPU-only regression checks: smoke process teardown and health acceptance."""
import importlib.util
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "motionkv_supervisor", Path(__file__).with_name("supervise_motionkv_training.py"))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class SmokePostexitTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(mode="smoke", uuids=["GPU-A"],
            expected_gpus=8, max_temp_c=85, max_used_mib=100)
        self.baseline = dict(host="host", boot_id="boot", anomaly_count=168,
            driver_anomalies_sha256="known", latest_driver_anomaly_sha256="last")

    def test_clean_exit_checks_cleanup_then_driver(self):
        order = []
        def gpu(*args, **kwargs):
            order.append("gpu")
            self.assertNotIn("own_pgid", kwargs)
            return {"devices": ["GPU-A"], "processes": []}
        def driver():
            order.append("driver")
            return self.baseline
        with patch.object(supervisor, "gpu_preflight", side_effect=gpu), patch.object(
                supervisor, "driver_evidence", side_effect=driver):
            result = supervisor.smoke_postexit_health(self.args, self.baseline)
        self.assertTrue(result["passed"])
        self.assertEqual(order, ["gpu", "driver"])

    def test_exit_zero_with_new_teardown_assertion_is_not_accepted(self):
        newer = dict(self.baseline, anomaly_count=171, driver_anomalies_sha256="new")
        with tempfile.TemporaryDirectory() as folder, patch.object(
                supervisor, "gpu_preflight", return_value={"devices": [], "processes": []}), patch.object(
                supervisor, "driver_evidence", return_value=newer):
            status = {"stop_reason": None}
            supervisor.record_process_exit(self.args, self.baseline, Path(folder), status, 0)
            saved = json.loads((Path(folder) / "status.json").read_text())
            health = json.loads((Path(folder) / "postexit_health.json").read_text())
        self.assertEqual(saved["state"], "process_exited")
        self.assertEqual(saved["exit_code"], 0)
        self.assertEqual(saved["stop_reason"], "postexit_health_failed")
        self.assertFalse(saved["postexit_health_passed"])
        self.assertEqual(health["driver"], newer)

    def test_remaining_compute_process_records_gpu_evidence(self):
        failure = RuntimeError("selected GPU still has compute PID")
        failure.gpu_evidence = {"devices": ["GPU-A"], "processes": [{"pid": "123"}]}
        with patch.object(supervisor, "gpu_preflight", side_effect=failure), patch.object(
                supervisor, "driver_evidence", return_value=self.baseline):
            result = supervisor.smoke_postexit_health(self.args, self.baseline)
        self.assertFalse(result["passed"])
        self.assertEqual(result["gpu"], failure.gpu_evidence)
        self.assertEqual(result["driver"], self.baseline)

    def test_missing_gpu_snapshot_is_preserved(self):
        row = "GPU-A,0000:1b:00.0,A6000,30,1,49140,70,300\n"
        with patch.object(supervisor, "command", return_value=row):
            with self.assertRaises(RuntimeError) as ctx:
                supervisor.gpu_preflight(["GPU-A"], 8, 85, 100)
        self.assertEqual(len(ctx.exception.gpu_evidence["devices"]), 1)

    def test_driver_query_failure_is_fail_closed(self):
        with patch.object(supervisor, "gpu_preflight", return_value={}), patch.object(
                supervisor, "driver_evidence", side_effect=TimeoutError("dmesg read")):
            result = supervisor.smoke_postexit_health(self.args, self.baseline)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["driver"])

    def test_formal_exit_does_not_poll_after_startup_window(self):
        self.args.mode = "formal"
        with tempfile.TemporaryDirectory() as folder, patch.object(
                supervisor, "gpu_preflight") as gpu, patch.object(supervisor, "driver_evidence") as driver:
            status = {"stop_reason": None, "startup_health_window_closed": True}
            supervisor.record_process_exit(self.args, self.baseline, Path(folder), status, 7)
            self.assertFalse((Path(folder) / "postexit_health.json").exists())
        gpu.assert_not_called()
        driver.assert_not_called()
        self.assertEqual(status["exit_code"], 7)
        self.assertNotIn("postexit_health_passed", status)


class DeviceGuardTests(unittest.TestCase):
    def diagnostic(self):
        return dict(guard_source_sha256="helper", guard_all_handles_closed=True,
            device_guard_events=[
                dict(event="after_open", uuid="GPU-A", fd=7, path="/dev/nvidia0", monotonic=1),
                dict(event="after_close", uuid="GPU-A", fd=7, path="/dev/nvidia0", monotonic=2)])

    def test_diagnostic_requires_actual_matching_final_close(self):
        expected = {"enabled": True, "helper_sha256": "helper"}
        supervisor.validate_guard_diagnostic(self.diagnostic(), ["GPU-A"], expected)
        for change in ("missing_close", "wrong_fd", "wrong_source", "not_closed", "close_before_open"):
            data = self.diagnostic()
            if change == "missing_close":
                data["device_guard_events"].pop()
            elif change == "wrong_fd":
                data["device_guard_events"][-1]["fd"] = 8
            elif change == "wrong_source":
                data["guard_source_sha256"] = "other"
            elif change == "not_closed":
                data["guard_all_handles_closed"] = False
            else:
                data["device_guard_events"][-1]["monotonic"] = .5
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                supervisor.validate_guard_diagnostic(data, ["GPU-A"], expected)

    def test_finished_workers_precede_close_and_settling(self):
        events = []
        guard = SimpleNamespace(handles=[7])
        def close():
            events.append("close")
            guard.handles.clear()
        guard.close = close
        def register(*args):
            events.append("register")
        with tempfile.TemporaryDirectory() as folder, patch.object(
                supervisor, "register_run_processes", side_effect=register), patch.object(
                supervisor, "live_owned_processes", return_value=[]), patch.object(
                supervisor.time, "sleep", side_effect=lambda secs: events.append(secs)), patch.object(
                supervisor, "gpu_preflight") as gpu, patch.object(supervisor, "driver_evidence") as driver:
            status = {}
            supervisor.finish_device_guard(guard, Path(folder), status, None, {})
        self.assertEqual(events, ["register", 3, "close", 5])
        self.assertTrue(status["guard_workers_exited"])
        self.assertTrue(status["guard_all_handles_closed"])
        gpu.assert_not_called()
        driver.assert_not_called()

    def test_exception_still_closes_guard_and_fails(self):
        guard = SimpleNamespace(handles=[7])
        def close():
            guard.handles.clear()
        guard.close = close
        with tempfile.TemporaryDirectory() as folder, patch.object(
                supervisor, "register_run_processes", side_effect=RuntimeError("cannot establish ownership")), patch.object(
                supervisor.time, "sleep"):
            status = {}
            with self.assertRaisesRegex(RuntimeError, "cannot establish ownership"):
                supervisor.finish_device_guard(guard, Path(folder), status, None, {})
        self.assertFalse(status["guard_workers_exited"])
        self.assertTrue(status["guard_all_handles_closed"])
        self.assertFalse(guard.handles)

    def test_reused_pid_is_never_treated_as_owned(self):
        with patch.object(supervisor, "process_identity", return_value={"pid": 123, "state": "R", "starttime": 9}):
            self.assertEqual(supervisor.live_owned_processes({123: {"starttime": 8}}), [])

    def test_formal_receipt_requires_guard_identity_and_closed_smoke(self):
        expected = {"enabled": True, "helper_sha256": "helper"}
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoint = run / "exports" / "weights.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"CPU fixture")
            event_path = run / "device_guard_events.json"
            supervisor.atomic_json(event_path, self.diagnostic()["device_guard_events"])
            supervisor.atomic_json(run / "postexit_health.json", {"passed": True})
            manifest = dict(mode="smoke", source_sha256="source", nccl_p2p="default",
                nccl_alloc="auto", nccl_environment={}, rank_visibility="all",
                device_lifetime_guard=expected, rank_topology={"global_rank_gpu_uuids": ["GPU-A"]},
                paths={"training_output": str(checkpoint.parent)})
            status = dict(state="process_exited", exit_code=0, stop_reason=None,
                guard_all_handles_closed=True, guard_workers_exited=True,
                device_guard_events_sha256=supervisor.file_digest(event_path),
                postexit_health_passed=True, postexit_health_sha256=supervisor.file_digest(run / "postexit_health.json"))
            supervisor.atomic_json(run / "manifest.json", manifest)
            supervisor.atomic_json(run / "status.json", status)
            receipt = dict(passed=True, gpu_smoke=True, checkpoint_reload=True,
                formal_config_sha256="config", source_sha256="source", smoke_run_dir=str(run),
                smoke_manifest_sha256=supervisor.file_digest(run / "manifest.json"),
                smoke_status_sha256=supervisor.file_digest(run / "status.json"),
                checkpoint_path=str(checkpoint), checkpoint_sha256=supervisor.file_digest(checkpoint),
                checkpoint_reload_receipt={}, device_lifetime_guard=expected)
            receipt_path = run / "receipt.json"
            reload = dict(passed=True, checkpoint_sha256=receipt["checkpoint_sha256"], source_sha256="source")
            supervisor.atomic_json(receipt_path, receipt)
            with patch.object(supervisor, "inside_ssd", side_effect=lambda value, *args, **kwargs: Path(value)), \
                    patch.object(supervisor, "referenced_receipt", return_value=reload):
                supervisor.validate_smoke(receipt_path, "config", "source", run,
                    expected_device_lifetime_guard=expected)
                for different in (supervisor.guard_identity(), {"enabled": True, "helper_sha256": "changed"}):
                    with self.assertRaisesRegex(ValueError, "device lifetime guard"):
                        supervisor.validate_smoke(receipt_path, "config", "source", run,
                            expected_device_lifetime_guard=different)
                status["guard_all_handles_closed"] = False
                supervisor.atomic_json(run / "status.json", status)
                receipt["smoke_status_sha256"] = supervisor.file_digest(run / "status.json")
                supervisor.atomic_json(receipt_path, receipt)
                with self.assertRaisesRegex(ValueError, "guard must close"):
                    supervisor.validate_smoke(receipt_path, "config", "source", run,
                        expected_device_lifetime_guard=expected)

    def test_all_popen_calls_explicitly_exclude_guard_descriptors(self):
        tree = ast.parse(Path(supervisor.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            flags = {entry.arg: entry.value for entry in call.keywords}
            self.assertIsInstance(flags["close_fds"], ast.Constant)
            self.assertIs(flags["close_fds"].value, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
