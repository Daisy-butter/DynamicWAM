"""CPU precision tests; the fixture does not claim native GPU acceptance."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from dynamicwam.dualrate.optimizer import FP32MasterAdamW
from dynamicwam.dualrate.train import (EpochCursor, capture_rng, load_training_checkpoint,
    save_training_checkpoint, state_sha256, verify_checkpoint_reload)


class TinyMixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        self.high = nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float32))


class MasterAdamWTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)

    def test_sub_bfloat_ulp_updates_accumulate_while_plain_bfloat_stalls(self):
        low = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        plain = nn.Parameter(low.detach().clone())
        optimizer = FP32MasterAdamW([low], lr=1e-4, betas=(0., 0.), weight_decay=0., record_updates=True)
        baseline = torch.optim.AdamW([plain], lr=1e-4, betas=(0., 0.), weight_decay=0., foreach=False)
        for step in range(64):
            low.grad, plain.grad = torch.ones_like(low), torch.ones_like(plain)
            optimizer.step()
            baseline.step()
            if step == 0:
                self.assertEqual(low.item(), 1.)
                self.assertGreater(optimizer.last_step_statistics["sampled_max_abs_master_update"], 0)
                self.assertEqual(optimizer.last_step_statistics["sampled_changed_forward_values"], 0)
            master = optimizer.state[low]["master"]
            self.assertEqual(master.dtype, torch.float32)
            torch.testing.assert_close(low, master.to(torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(plain.item(), 1.)
        self.assertLess(low.item(), 1.)
        self.assertAlmostEqual(master.item(), 1. - 64e-4, delta=2e-6)
        self.assertEqual(optimizer.state[low]["exp_avg"].dtype, torch.float32)
        self.assertEqual(optimizer.state[low]["exp_avg_sq"].dtype, torch.float32)

    def test_fp32_parameters_match_native_adamw_exactly(self):
        p = nn.Parameter(torch.randn(5))
        baseline_p = nn.Parameter(p.detach().clone())
        low = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([{"params": [p, low], "lr": .003}], weight_decay=.02)
        baseline = torch.optim.AdamW([baseline_p], lr=.003, weight_decay=.02, foreach=False)
        for index in range(5):
            gradient = torch.randn_like(p)
            p.grad, baseline_p.grad, low.grad = gradient.clone(), gradient.clone(), torch.ones_like(low)
            optimizer.param_groups[0]["lr"] = baseline.param_groups[0]["lr"] = .003 / (index + 1)
            optimizer.step()
            baseline.step()
            torch.testing.assert_close(p, baseline_p, rtol=0, atol=0)
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[p][name], baseline.state[baseline_p][name], rtol=0, atol=0)
        self.assertNotIn("master", optimizer.state[p])

    def test_zero_grad_and_no_gradient_do_not_apply_stale_master_step(self):
        p = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([p], lr=.01, weight_decay=.1)
        p.grad = torch.ones_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = state_sha256(optimizer.state_dict())
        self.assertIsNone(p.grad)
        self.assertTrue(all(master is None or master.grad is None for group in optimizer._masters for master in group))
        optimizer.step()
        self.assertEqual(before, state_sha256(optimizer.state_dict()))
        p.grad = torch.ones_like(p)
        optimizer.zero_grad(set_to_none=False)
        self.assertEqual(p.grad.abs().sum().item(), 0)

    def test_checkpoint_resume_retains_sub_ulp_masters_and_matches_next_updates(self):
        model = TinyMixed()
        optimizer = FP32MasterAdamW([{"params": model.parameters(), "name": "real-fixture"}],
            lr=1e-4, weight_decay=0., record_updates=True)
        cursor, generator = EpochCursor(9, 2, 18), torch.Generator().manual_seed(20)
        identity = {"kind": "explicit-cpu-fixture"}
        def update():
            cursor.next_indices()
            optimizer.zero_grad(set_to_none=True)
            model.low.grad = torch.ones_like(model.low)
            model.high.grad = torch.randn(model.high.shape, generator=generator)
            optimizer.step()
        for _ in range(4):
            update()
        self.assertTrue(torch.equal(model.low, torch.ones_like(model.low)))
        self.assertTrue((optimizer.state[model.low]["master"] < 1).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_training_checkpoint(path, model, optimizer, 4, cursor, identity, generator=generator)
            receipt = verify_checkpoint_reload(path, model, optimizer, cursor, identity, expected_step=4,
                report_path=Path(directory)/"reload.json", generator=generator)
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["optimizer_precision"]["master_dtype"], "torch.float32")
            for _ in range(32):
                update()
            expected = dict(model=state_sha256(model.state_dict()), optimizer=state_sha256(optimizer.state_dict()),
                cursor=state_sha256(cursor.state_dict()), rng=state_sha256(capture_rng(generator)))
            self.assertEqual(load_training_checkpoint(path, model, optimizer, cursor, identity, generator=generator), 4)
            for _ in range(32):
                update()
            actual = dict(model=state_sha256(model.state_dict()), optimizer=state_sha256(optimizer.state_dict()),
                cursor=state_sha256(cursor.state_dict()), rng=state_sha256(capture_rng(generator)))
            self.assertEqual(actual, expected)

    def test_loader_rejects_lossy_moments_masters_or_mismatched_forward_values(self):
        p = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        optimizer = FP32MasterAdamW([p], lr=1e-4)
        p.grad = torch.ones_like(p)
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        for field in ("master", "moment", "forward"):
            altered = copy.deepcopy(saved)
            if field == "master":
                altered["master_params"][0][0] = altered["master_params"][0][0].bfloat16()
            elif field == "moment":
                next(iter(altered["adamw"]["state"].values()))["exp_avg"] = torch.zeros(2, dtype=torch.bfloat16)
            else:
                altered["master_params"][0][0].add_(1)
            before = state_sha256(optimizer.state_dict())
            with self.assertRaises(ValueError):
                optimizer.load_state_dict(altered)
            self.assertEqual(before, state_sha256(optimizer.state_dict()))
        with self.assertRaises(ValueError):
            optimizer.load_state_dict(torch.optim.AdamW([p]).state_dict())


if __name__ == "__main__":
    unittest.main()
