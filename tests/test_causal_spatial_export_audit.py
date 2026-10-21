"""CPU-only adversarial receipt tests; no Torch or CUDA import required."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_causal_spatial_export.py"
SPEC = importlib.util.spec_from_file_location("export_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ExportReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.recipe = dict(steps=4, accumulation=4, phase="joint", diagnostic=True)
        self.receipt = dict(world_size=8, completed=True, error_type=None, step=4,
            optimizer_updates=4, nonfinite_loss_count=0, nonfinite_grad_count=0,
            oom_count=0, finite_loss_checks=16, finite_grad_checks=4,
            last_grad_norm=.02, peak_allocated=100, peak_reserved=120,
            last_metrics=dict(total=.1, action=.02),
            gradient_evidence=dict(video=.1, observer=.1, future_and_interfaces=.1,
                                   action=.1, motion=.1, plan=0.))
        for rank in range(8):
            self.write_rank(rank)

    def write_rank(self, rank, **updates):
        value = dict(copy.deepcopy(self.receipt), rank=rank, **updates)
        (self.run / f"rank_{rank}_resources.json").write_text(json.dumps(value))

    def test_eight_real_rank_receipts_required(self):
        self.assertEqual(len(audit.audit_ranks(self.run, self.recipe)), 8)
        (self.run / "rank_7_resources.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "exactly ranks"):
            audit.audit_ranks(self.run, self.recipe)

    def test_missing_checks_oom_and_failed_exit_rejected(self):
        for change in (dict(finite_grad_checks=3), dict(finite_loss_checks=15),
                       dict(oom_count=1), dict(completed=False), dict(world_size=1),
                       dict(last_grad_norm=None), dict(last_grad_norm=float("nan"))):
            with self.subTest(change=change):
                self.write_rank(6, **change)
                with self.assertRaises(RuntimeError):
                    audit.audit_ranks(self.run, self.recipe)
                self.write_rank(6)

    def test_dormant_required_pathway_rejected(self):
        groups = dict(self.receipt["gradient_evidence"], observer=0.)
        self.write_rank(2, gradient_evidence=groups)
        with self.assertRaisesRegex(RuntimeError, "observer"):
            audit.audit_ranks(self.run, self.recipe)

    def test_all_smoke_updates_and_final_validation_required(self):
        rows = [dict(step=i, phase="joint", elapsed_seconds=i, total=.1) for i in range(1,5)]
        (self.run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (self.run / "validation.jsonl").write_text(json.dumps(dict(step=4, metrics=dict(total=.1))))
        self.assertTrue(audit.audit_logs(self.run, self.recipe)["all_recorded_metrics_finite"])
        (self.run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows[::3])))
        with self.assertRaisesRegex(RuntimeError, "every update"):
            audit.audit_logs(self.run, self.recipe)

    def test_nan_validation_rejected_even_with_final_step(self):
        rows = [dict(step=i, phase="joint", total=.1) for i in range(1,5)]
        (self.run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (self.run / "validation.jsonl").write_text(json.dumps(dict(step=4, metrics=dict(total=float("nan")))))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            audit.audit_logs(self.run, self.recipe)

    def test_teacher_training_sha_is_required_and_content_bound(self):
        path = self.run / "teacher.pt"
        path.write_bytes(b"local fixture: no checkpoint deserialization")
        recipe = dict(teacher_checkpoint=str(path))
        initial = dict(teacher_sha256=audit.digest(path))
        self.assertTrue(audit.audit_teacher_identity(recipe, initial)["training_time_identity_bound"])
        with self.assertRaisesRegex(RuntimeError, "missing or malformed"):
            audit.audit_teacher_identity(recipe, {})
        path.write_bytes(b"changed teacher after training")
        with self.assertRaisesRegex(RuntimeError, "changed since training"):
            audit.audit_teacher_identity(recipe, initial)


if __name__ == "__main__":
    unittest.main()
