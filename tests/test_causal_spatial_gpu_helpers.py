"""CPU-only receipt/interface tests; do not initialize or query CUDA."""
from pathlib import Path
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("causal_gpu_checker",ROOT/"scripts/check_causal_spatial_gpu.py")
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


class GuardedCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/"guard";self.run.mkdir()
        (self.root/"scripts").mkdir()
        (self.root/"scripts/device_lifetime_guard.py").write_text("# fixture only\n")
        self.config=self.root/"recipe.json";self.config.write_text("{}")
        self.ids=[f"GPU-{i}" for i in range(8)]
        self.auth=dict(guard_sha256=checker.digest(self.root/"scripts/device_lifetime_guard.py"),
            source_sha256="source-fixture",device_lifetime_mode="fds_before_telemetry_v1",
            gpu_uuids=self.ids,config_sha256=checker.digest(self.config))
        self.authorization=self.root/"authorization.json"
        self.authorization.write_text(json.dumps(self.auth))
        self.manifest=dict(device_lifetime_mode=self.auth["device_lifetime_mode"],
            guard_gpu_uuids=self.ids,proc_gpu_inventory=[dict(uuid=x) for x in self.ids],
            worker_gpu_uuids=self.ids[:1],device_guard_sha256=self.auth["guard_sha256"],
            source_sha256=self.auth["source_sha256"],authorization=str(self.authorization),
            authorization_sha256=checker.digest(self.authorization),stage="graph",preflight_only=False)
        self.events=[dict(event="after_open",uuid=x,fd=i+10) for i,x in enumerate(self.ids)]
        self.write()
        self.env=patch.dict(os.environ,dict(WAM_TRAINING_RUN_DIR=str(self.run),
            WAM_SSD_ROOT=str(self.root),CUDA_VISIBLE_DEVICES=self.ids[0]))
        self.env.start();self.addCleanup(self.env.stop)
        root_patch=patch.object(checker,"ROOT",self.root);root_patch.start();self.addCleanup(root_patch.stop)

    def write(self):
        (self.run/"manifest.json").write_text(json.dumps(self.manifest))
        (self.run/"device_guard_events.json").write_text(json.dumps(self.events))

    def check(self):return checker.guarded_context(self.root/"report.json",self.config)

    def test_current_eval_supervisor_schema_is_accepted(self):
        manifest,run=self.check()
        self.assertEqual(manifest,self.manifest);self.assertEqual(run,self.run)

    def test_duplicate_or_incomplete_guard_inventory_is_rejected(self):
        for ids in (self.ids[:7],self.ids[:7]+self.ids[:1]):
            self.manifest["guard_gpu_uuids"]=ids;self.write()
            with self.assertRaisesRegex(RuntimeError,"eight adapters"):self.check()

    def test_actual_inventory_must_match_guard_coverage(self):
        self.manifest["proc_gpu_inventory"][-1]["uuid"]="GPU-other";self.write()
        with self.assertRaisesRegex(RuntimeError,"actual eight-adapter"):self.check()

    def test_compute_mask_must_match_one_authorized_guarded_worker(self):
        for mask in ("","GPU-other",",".join(self.ids[:2])):
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=mask):
                with self.assertRaisesRegex(RuntimeError,"one guarded compute UUID"):self.check()

    def test_changed_guard_helper_authorization_or_config_is_rejected(self):
        for file in (self.root/"scripts/device_lifetime_guard.py",self.authorization,self.config):
            original=file.read_bytes();file.write_bytes(original+b" ")
            with self.assertRaises(RuntimeError):self.check()
            file.write_bytes(original)

    def test_closed_or_not_opened_guard_receipt_is_rejected(self):
        original=copy.deepcopy(self.events)
        for events in (original[:-1],original+[dict(event="before_close",uuid=self.ids[0])]):
            self.events=events;self.write()
            with self.assertRaisesRegex(RuntimeError,"descriptors must be open"):self.check()

    def test_preflight_or_other_stage_cannot_claim_actual_graph_execution(self):
        for values in (dict(stage="smoke"),dict(stage="graph",preflight_only=True)):
            self.manifest.update(values);self.write()
            with self.assertRaisesRegex(RuntimeError,"execution-stage"):self.check()

    def test_result_cannot_escape_supervisor_storage(self):
        with self.assertRaisesRegex(RuntimeError,"verified SSD"):
            checker.guarded_context(self.root.parent/"outside-report.json",self.config)

    def test_inference_whitelist_excludes_future_labels_and_realized_future_clocks(self):
        forbidden={"clean_future","clean_actions","future_actual_support_offsets",
                   "future_support_lag","future_actual_times","plan_condition_latent"}
        self.assertFalse(forbidden.intersection(checker.INPUT_KEYS))
        self.assertIn("future_target_offsets",checker.INPUT_KEYS)


if __name__=="__main__":unittest.main()
