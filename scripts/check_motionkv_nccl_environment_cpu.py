#!/usr/bin/env python3
"""CPU-only launch-contract regression tests; never imports torch or uses GPUs.

Every receipt/checkpoint below is a disposable synthetic unit-test fixture, not
training evidence. Real filesystem hashes exercise the supervisor's receipt
chain without loading a model or mocking GPU operations.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SUPERVISOR = Path(__file__).with_name("supervise_motionkv_training.py")
spec = importlib.util.spec_from_file_location("motionkv_supervisor_cpu_test", SUPERVISOR)
supervisor = importlib.util.module_from_spec(spec)
torch_before_import = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
spec.loader.exec_module(supervisor)
torch_after_import = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class NCCLEnvironmentContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="motionkv-nccl-cpu-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run, self.output = self.root / "smoke", self.root / "training"
        self.run.mkdir()
        self.output.mkdir()
        self.checkpoint = self.output / "synthetic-checkpoint.bin"
        self.checkpoint.write_bytes(b"CPU TEST FIXTURE ONLY: not a model checkpoint")
        self.uuids = ["GPU-00000000-0000-0000-0000-000000000001",
                      "GPU-00000001-0000-0000-0000-000000000001"]
        self.nccl_environment = supervisor.configure_nccl_transport({
            "NCCL_DEBUG": "WARN", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}, "disable", "legacy")
        self.manifest = dict(mode="smoke", source_sha256="synthetic-source-hash",
            device_lifetime_guard=dict(enabled=False, helper_sha256=None),
            paths=dict(training_output=str(self.output)), nccl_p2p="disable", nccl_alloc="legacy",
            nccl_environment=dict(self.nccl_environment), rank_visibility="isolated",
            rank_topology=dict(launcher_cuda_masked=True, global_rank_gpu_uuids=self.uuids),
            environment=dict(CUDA_VISIBLE_DEVICES="", WAM_ASSIGNED_GPU_UUIDS=",".join(self.uuids)))
        health = self.run / "postexit_health.json"
        save(health, dict(passed=True, fixture_only=True))
        self.status = dict(state="process_exited", exit_code=0, stop_reason=None,
            postexit_health_passed=True, postexit_health_sha256=sha(health))
        reload_path = self.root / "synthetic-reload.json"
        save(reload_path, dict(passed=True, checkpoint_sha256=sha(self.checkpoint),
            source_sha256="synthetic-source-hash", fixture_only=True))
        self.receipt = self.root / "synthetic-receipt.json"
        self.receipt_value = dict(passed=True, gpu_smoke=True, checkpoint_reload=True,
            device_lifetime_guard=dict(enabled=False, helper_sha256=None),
            fixture_only=True, formal_config_sha256="synthetic-config-hash",
            source_sha256="synthetic-source-hash", smoke_run_dir=str(self.run),
            checkpoint_path=str(self.checkpoint), checkpoint_sha256=sha(self.checkpoint),
            checkpoint_reload_receipt=dict(path=str(reload_path), sha256=sha(reload_path)))
        self.persist()

    def persist(self):
        """Refresh real hashes so negative cases exercise semantics, not stale hashes."""
        save(self.run / "manifest.json", self.manifest)
        save(self.run / "status.json", self.status)
        self.receipt_value.update(smoke_manifest_sha256=sha(self.run / "manifest.json"),
            smoke_status_sha256=sha(self.run / "status.json"))
        save(self.receipt, self.receipt_value)

    def validate(self, **changes):
        kwargs = dict(expected_transport="disable", expected_rank_visibility="isolated",
            expected_alloc="legacy", expected_nccl_environment=self.nccl_environment)
        kwargs.update(changes)
        return supervisor.validate_smoke(self.receipt, "synthetic-config-hash",
            "synthetic-source-hash", self.root, **kwargs)

    def test_import_is_guarded_and_does_not_load_torch(self):
        self.assertEqual(torch_before_import, torch_after_import)
        tree = ast.parse(SUPERVISOR.read_text(encoding="utf-8"))
        self.assertTrue(any(isinstance(node, ast.If) and
            ast.unparse(node.test) == "__name__ == '__main__'" for node in tree.body))
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and
            "torch" in ast.unparse(node) for node in tree.body))

    def test_auto_removes_both_inherited_cumem_settings(self):
        env = dict(NCCL_CUMEM_ENABLE="1", NCCL_CUMEM_HOST_ENABLE="0",
            NCCL_P2P_DISABLE="1", NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
            UNRELATED_SETTING="preserved")
        actual = supervisor.configure_nccl_transport(env, "default", "auto")
        self.assertEqual(actual, dict(NCCL_DEBUG="WARN", TORCH_NCCL_ASYNC_ERROR_HANDLING="1"))
        self.assertNotIn("NCCL_CUMEM_ENABLE", env)
        self.assertNotIn("NCCL_CUMEM_HOST_ENABLE", env)
        self.assertNotIn("NCCL_P2P_DISABLE", env)
        self.assertEqual(env["UNRELATED_SETTING"], "preserved")

    def test_legacy_sets_exactly_two_allocator_overrides(self):
        env = dict(UNRELATED_SETTING="preserved")
        actual = supervisor.configure_nccl_transport(env, "default", "legacy")
        self.assertEqual(actual, dict(NCCL_CUMEM_ENABLE="0", NCCL_CUMEM_HOST_ENABLE="0"))
        self.assertEqual(env["UNRELATED_SETTING"], "preserved")

    def test_invalid_modes_are_rejected(self):
        for transport, allocation in (("invalid", "auto"), ("default", "invalid")):
            with self.subTest(transport=transport, allocation=allocation), self.assertRaises(ValueError):
                supervisor.configure_nccl_transport({}, transport, allocation)

    def test_matching_real_fixture_hash_chain_passes(self):
        result = self.validate()
        self.assertEqual(result, dict(path=str(self.receipt), sha256=sha(self.receipt)))

    def test_allocator_selection_and_missing_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "allocation must match"):
            self.validate(expected_alloc="auto")
        self.manifest.pop("nccl_alloc")
        self.persist()
        with self.assertRaisesRegex(ValueError, "allocation must match"):
            self.validate()

    def test_actual_allocator_environment_must_match_declared_mode(self):
        self.manifest["nccl_environment"]["NCCL_CUMEM_HOST_ENABLE"] = "1"
        self.persist()
        with self.assertRaisesRegex(ValueError, "allocation and recorded environment disagree"):
            self.validate()

    def test_every_nccl_environment_key_including_debug_is_compared(self):
        for name, value in (("NCCL_DEBUG", "INFO"), ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0"),
                ("NCCL_SHM_DISABLE", "1"), ("NCCL_NET", "Socket")):
            expected = dict(self.nccl_environment)
            expected[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "environment must exactly match"):
                self.validate(expected_nccl_environment=expected)
        expected = dict(self.nccl_environment)
        expected.pop("NCCL_DEBUG")
        with self.assertRaisesRegex(ValueError, "environment must exactly match"):
            self.validate(expected_nccl_environment=expected)
        with self.assertRaisesRegex(ValueError, "environment must exactly match"):
            self.validate(expected_nccl_environment=None)

    def test_transport_and_rank_visibility_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "P2P transport must match"):
            self.validate(expected_transport="default")
        with self.assertRaisesRegex(ValueError, "rank visibility must match"):
            self.validate(expected_rank_visibility="all")

    def test_masked_launcher_is_part_of_isolated_contract(self):
        self.manifest["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(self.uuids)
        self.persist()
        with self.assertRaisesRegex(ValueError, "CUDA-masked torchrun parent"):
            self.validate()

    def test_default_auto_all_visibility_with_minimal_environment_passes(self):
        self.manifest.update(nccl_p2p="default", nccl_alloc="auto", rank_visibility="all",
            nccl_environment={})
        self.persist()
        result = self.validate(expected_transport="default", expected_rank_visibility="all",
            expected_alloc="auto", expected_nccl_environment=None)
        self.assertEqual(result["sha256"], sha(self.receipt))

    def test_checkpoint_hash_still_checked_after_environment_passes(self):
        self.checkpoint.write_bytes(b"changed synthetic fixture")
        with self.assertRaisesRegex(ValueError, "checkpoint identity/location mismatch"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
