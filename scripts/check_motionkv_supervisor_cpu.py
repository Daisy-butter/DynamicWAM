#!/usr/bin/env python3
"""CPU-only regression checks: smoke process teardown and health acceptance."""
import importlib.util
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
