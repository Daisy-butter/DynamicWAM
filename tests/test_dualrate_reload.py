"""Full-state reload acceptance uses explicit CPU fixtures, never GPU claims."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch import nn

from dynamicwam.dualrate.train import (EpochCursor, capture_rng, parameter_counts,
    save_training_checkpoint, state_sha256, verify_checkpoint_reload)


class ReloadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        random.seed(19)
        self.model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.1), nn.Linear(4, 2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.cursor = EpochCursor(9, 2, 19)
        self.generator = torch.Generator().manual_seed(20)
        self.identity = {"fixture": "cpu-reload-not-formal-training"}
        self.cursor.next_indices()
        loss = self.model(torch.randn(2, 3, generator=self.generator)).square().mean()
        loss.backward()
        self.optimizer.step()

    def test_strict_reload_preserves_all_state_without_another_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt = Path(directory)/"last.pt", Path(directory)/"reload.json"
            save_training_checkpoint(path, self.model, self.optimizer, 1, self.cursor,
                                     self.identity, generator=self.generator)
            rng = state_sha256(capture_rng(self.generator))
            optimizer = state_sha256(self.optimizer.state_dict())
            result = verify_checkpoint_reload(path, self.model, self.optimizer, self.cursor,
                self.identity, expected_step=1, report_path=receipt, generator=self.generator)
            self.assertTrue(result["passed"])
            self.assertEqual(result["extra_training_updates"], 0)
            self.assertEqual(result["before"], result["after"])
            self.assertEqual(result["cursor_before"], result["cursor_after"])
            self.assertEqual(result["restored_step"], 1)
            self.assertEqual(state_sha256(capture_rng(self.generator)), rng)
            self.assertEqual(state_sha256(self.optimizer.state_dict()), optimizer)
            self.assertEqual(json.loads(receipt.read_text())["passed"], True)
            self.assertFalse(result["formal_training_ready"])

    def test_any_model_tensor_difference_produces_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt = Path(directory)/"last.pt", Path(directory)/"reload.json"
            save_training_checkpoint(path, self.model, self.optimizer, 1, self.cursor,
                                     self.identity, generator=self.generator)
            value = torch.load(path, weights_only=True)
            value["model"]["2.bias"][0] += 1
            torch.save(value, path)
            with self.assertRaisesRegex(RuntimeError, "reload mismatch"):
                verify_checkpoint_reload(path, self.model, self.optimizer, self.cursor,
                    self.identity, expected_step=1, report_path=receipt, generator=self.generator)
            report = json.loads(receipt.read_text())
            self.assertFalse(report["passed"])
            self.assertIn("model", report["changed_components"])

    def test_optimizer_and_step_mismatch_are_not_hidden_by_model_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path, receipt = Path(directory)/"last.pt", Path(directory)/"reload.json"
            save_training_checkpoint(path, self.model, self.optimizer, 1, self.cursor,
                                     self.identity, generator=self.generator)
            value = torch.load(path, weights_only=True)
            state = next(iter(value["optimizer"]["state"].values()))
            state["exp_avg"].add_(.5)
            torch.save(value, path)
            with self.assertRaises(RuntimeError):
                verify_checkpoint_reload(path, self.model, self.optimizer, self.cursor,
                    self.identity, expected_step=2, report_path=receipt, generator=self.generator)
            report = json.loads(receipt.read_text())
            self.assertFalse(report["passed"])
            self.assertIn("optimizer", report["changed_components"])
            self.assertEqual(report["before"]["model"], report["after"]["model"])
            self.assertNotEqual(report["expected_step"], report["restored_step"])

    def test_hash_is_shape_dtype_value_and_container_exact(self):
        value = torch.tensor([1., -0.], dtype=torch.float32)
        base = state_sha256({"a": value, 2: [3, (4,)]})
        self.assertEqual(base, state_sha256({2: [3, (4,)], "a": value.clone()}))
        for altered in (value.reshape(1, 2), value.double(), torch.tensor([1., 0.])):
            self.assertNotEqual(base, state_sha256({"a": altered, 2: [3, (4,)]}))
        self.assertNotEqual(state_sha256([1, 2]), state_sha256((1, 2)))
        self.assertEqual(state_sha256(torch.tensor(1., dtype=torch.bfloat16)),
                         state_sha256(torch.tensor(1., dtype=torch.bfloat16)))
        self.assertEqual(state_sha256(torch.empty(0)), state_sha256(torch.empty(0)))

    def test_counts_use_real_tensors_and_report_shared_modules(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(3, 2)
                self.right = nn.Linear(3, 2)
                self.right.weight = self.left.weight
                self.left.bias.requires_grad_(False)
        model = Shared()
        counts = parameter_counts(model)
        self.assertEqual(counts["total"]["total"], 10)  # shared weight 6 + two biases 2+2
        self.assertEqual(counts["total"]["trainable"], 8)
        self.assertEqual(counts["total"]["frozen"], 2)
        self.assertEqual(counts["modules"]["left"]["total"], 8)
        self.assertEqual(counts["shared_parameter_tensors_across_modules"], 1)


if __name__ == "__main__":
    unittest.main()
