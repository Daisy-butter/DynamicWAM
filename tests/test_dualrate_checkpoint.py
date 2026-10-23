import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from dynamicwam.dualrate.train import (
    EpochCursor, atomic_checkpoint, canonical_hash, capture_rng,
    load_training_checkpoint, restore_rng, save_training_checkpoint,
    _validate_normalization, _gpu_authorization,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        random.seed(23)
        self.model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.2), nn.Linear(5, 2))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.cursor = EpochCursor(7, 3, 23)
        self.generator = torch.Generator().manual_seed(24)
        self.identity = {"source_sha256": "a" * 64, "manifest_sha256": "b" * 64}

    def step(self):
        rows = self.cursor.next_indices()
        x = torch.tensor([[float(i), 1, 2] for i in rows])
        target = torch.randn(len(rows), 2, generator=self.generator)
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model(x) - target).square().mean()
        loss.backward()
        self.optimizer.step()
        return rows, loss.detach().clone()

    def test_complete_resume_matches_uninterrupted_update(self):
        self.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(path, self.model, self.optimizer, 1, self.cursor, self.identity, generator=self.generator)
            next_rows, next_loss = self.step()
            expected = {k: v.clone() for k, v in self.model.state_dict().items()}
            random_expected = random.random()
            # Deliberately disrupt all state before reload.
            for p in self.model.parameters():
                p.data.zero_()
            torch.manual_seed(0)
            random.seed(0)
            self.generator.manual_seed(0)
            step = load_training_checkpoint(path, self.model, self.optimizer, self.cursor, self.identity, generator=self.generator)
            actual_rows, actual_loss = self.step()
            self.assertEqual(step, 1)
            self.assertEqual(next_rows, actual_rows)
            torch.testing.assert_close(next_loss, actual_loss, rtol=0, atol=0)
            for k, v in self.model.state_dict().items():
                torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
            self.assertEqual(random.random(), random_expected)

    def test_identity_and_cursor_rejected_before_weights_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(path, self.model, self.optimizer, 0, self.cursor, self.identity, generator=self.generator)
            before = {k: v.clone() for k, v in self.model.state_dict().items()}
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, self.model, self.optimizer, self.cursor, {"different": 1}, generator=self.generator)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, self.model, self.optimizer, EpochCursor(8, 3, 23), self.identity, generator=self.generator)
            for k, v in self.model.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_cursor_epoch_resume_and_no_global_rng_consumption(self):
        state = torch.get_rng_state()
        first = EpochCursor(7, 3, 23)
        epoch = first.next_indices() + first.next_indices() + first.next_indices()
        self.assertEqual(sorted(epoch), list(range(7)))
        second = EpochCursor(7, 3, 23)
        second.load_state_dict(first.state_dict())
        self.assertEqual(first.next_indices(), second.next_indices())
        self.assertEqual(first.epoch, 1)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_checkpoint(path, {"old": torch.tensor([1])})
            original = path.read_bytes()
            with mock.patch("torch.save", side_effect=RuntimeError("write failed")):
                with self.assertRaises(RuntimeError):
                    atomic_checkpoint(path, {"new": torch.tensor([2])})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_storage_floor_prevents_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with mock.patch("shutil.disk_usage") as usage:
                usage.return_value.free = 10
                with self.assertRaises(OSError):
                    atomic_checkpoint(path, {"x": torch.zeros(2)}, min_free_bytes=10)
            self.assertFalse(path.exists())

    def test_numpy_and_generator_rng_roundtrip(self):
        state = capture_rng(self.generator)
        a = torch.rand(4, generator=self.generator)
        b = torch.rand(4)
        restore_rng(state, self.generator)
        torch.testing.assert_close(torch.rand(4, generator=self.generator), a, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(4), b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            restore_rng(state, None)

    def test_normalization_binding_and_raw_motion_contract(self):
        config = dict(init_checkpoint_sha256="a" * 64, action_normalization_sha256="b" * 64, motion_normalization_sha256="c" * 64)
        manifest = dict(native_checkpoint_sha256="a" * 64, action_normalization_sha256="b" * 64, motion_normalization_sha256="c" * 64, action_representation="checkpoint_mean_std", motion_representation="raw_absolute_motion_v2", split="train")
        _validate_normalization(config, manifest)
        for key in manifest:
            with self.assertRaises(ValueError):
                _validate_normalization(config, dict(manifest, **{key: "wrong"}))

    def test_gpu_authorization_fails_before_any_cuda_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"authorized": False}), encoding="utf-8")
            with mock.patch("torch.cuda.init", side_effect=AssertionError("CUDA must not initialize")):
                with self.assertRaises(ValueError):
                    _gpu_authorization(path, self.identity, Path(directory))

    def test_canonical_config_hash_order_invariance(self):
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
