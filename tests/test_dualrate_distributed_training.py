"""Real two-process CPU DDP contract tests; not eight-GPU acceptance."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamicwam.dualrate.optimizer import FP32MasterAdamW
from dynamicwam.dualrate.train import state_sha256, file_sha256
from dynamicwam.dualrate.train_distributed import (DistributedEpochCursor, load_checkpoint,
    save_checkpoint, set_phase, verify_reload, selected_rows, set_learning_rate,
    validate_resume_authorization)


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.slow = nn.Linear(3, 3).bfloat16()
        self.world_readout = nn.Linear(3, 3)
        self.observation = nn.Linear(3, 3)
        self.fast = nn.Linear(3, 1)
        self.optional = nn.Linear(3, 1)

    def forward(self, x, use_optional):
        y = self.fast(self.world_readout(self.slow(x.bfloat16()).float()) + self.observation(x))
        if use_optional:
            y = y + self.optional(x)
        return {"loss": y.square().mean()}


def _config():
    return dict(sequence_id="cpu-fixture-not-a-formal-run", world_size=2,
        run_kind="distributed_training_smoke", phase="warmup", steps=2,
        init_checkpoint_sha256="fixture", action_normalization_sha256="fixture",
        motion_normalization_sha256="fixture", architecture={}, codec_id="fixture",
        optimizer_kind="fp32_master_adamw", gradient_accumulation_dtype="float32",
        slow_lr=1e-4, fast_lr=1e-3, lr_warmup_steps=0, min_lr_ratio=.1)


def _ddp_worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(13)
        model = Fixture()
        optimizer = FP32MasterAdamW([
            dict(params=model.slow.parameters(), lr=1e-4, name="slow"),
            dict(params=[p for name,p in model.named_parameters() if not name.startswith("slow.")], lr=1e-3, name="new")])
        config = _config()
        identity = dict(source_sha256="fixture", manifest_sha256="fixture", init_checkpoint_sha256="fixture",
                        config_sha256="phase1", device_type="cpu", torch_version=str(torch.__version__))
        set_phase(model, "warmup")
        wrapped = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        generator = torch.Generator().manual_seed(100+rank)
        cursor = DistributedEpochCursor(25, 2, 91, rank, 2)
        device = torch.device("cpu")
        path = Path(folder) / "last.pt"

        def update(step):
            optimizer.zero_grad()
            set_learning_rate(optimizer, config, step)
            for _ in range(2):
                ids = cursor.next_indices()
                x = torch.randn((len(ids), 3), generator=generator)
                wrapped(x, rank == 0)["loss"].backward()
                optimizer.accumulate_gradients(.5)
            optimizer.clip_accumulated_grad_norm_(1.)
            optimizer.step()
            optimizer.zero_grad()

        update(0)
        save_checkpoint(path, model, optimizer, cursor, generator, device, config, identity, 1, 1)
        verify_reload(path, model, optimizer, cursor, generator, device, config, identity, 1, 1, Path(folder))
        update(1)
        expected = (state_sha256(model.state_dict()), state_sha256(optimizer.state_dict()), cursor.state_dict())
        assert load_checkpoint(path, model, optimizer, cursor, generator, config, identity) == (1, 1)
        update(1)
        actual = (state_sha256(model.state_dict()), state_sha256(optimizer.state_dict()), cursor.state_dict())
        assert actual == expected, "next update differs after full per-rank resume"
        save_checkpoint(path, model, optimizer, cursor, generator, device, config, identity, 2, 2)
        verify_reload(path, model, optimizer, cursor, generator, device, config, identity, 2, 2, Path(folder))
        # Rewrap after unfreezing; DDP's parameter set must remain static within phase.
        del wrapped
        config = dict(config, phase="small_age_joint", stage_init_checkpoint=str(path))
        identity = dict(identity, config_sha256="phase2")
        set_phase(model, config["phase"])
        wrapped = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        new_cursor = DistributedEpochCursor(50, 2, 92, rank, 2)
        before = state_sha256(optimizer.state_dict())
        assert load_checkpoint(path, model, optimizer, new_cursor, generator, config, identity, transition=True) == (0, 2)
        assert state_sha256(optimizer.state_dict()) == before
        assert new_cursor.position == 0
        cursor = new_cursor
        update(0)
        assert optimizer.state[model.slow.weight]["exp_avg"].dtype == torch.float32
        formal = dict(config, run_kind="formal_distributed_training")
        try:
            load_checkpoint(path, model, optimizer, cursor, generator, formal, identity, transition=True)
        except ValueError as exc:
            assert "smoke" in str(exc)
        else:
            raise AssertionError("smoke state accepted as formal init")
        hashes = [None, None]
        dist.all_gather_object(hashes, state_sha256(model.state_dict()))
        assert len(set(hashes)) == 1
        Path(folder, f"rank_{rank}_passed").write_text("two-process CPU DDP, exact resume and phase transition passed")
    finally:
        dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_guarded_resume_path_and_bytes_bind_before_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder).resolve()
            path = output / "last.pt"
            path.write_bytes(b"explicit full-state checkpoint fixture")
            receipt = dict(resume_checkpoint=str(path), resume_checkpoint_sha256=file_sha256(path))
            validate_resume_authorization(str(path), receipt, output)
            validate_resume_authorization(None, {}, output)
            with self.assertRaises(ValueError):
                validate_resume_authorization(None, receipt, output)
            with self.assertRaises(ValueError):
                validate_resume_authorization(str(path), {}, output)
            with self.assertRaises(ValueError):
                validate_resume_authorization("last.pt", receipt, output)
            path.write_bytes(b"changed after authorization")
            with self.assertRaises(ValueError):
                validate_resume_authorization(str(path), receipt, output)

    def test_global_sampler_disjoint_and_exact_resume(self):
        cursors = [DistributedEpochCursor(103, 3, 71, rank, 8) for rank in range(8)]
        seen = set()
        for _ in range(4):
            batches = [c.next_indices() for c in cursors]
            values = sum(batches, [])
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(seen.intersection(values))
            seen.update(values)
        self.assertEqual(len(seen), 96)
        saved = cursors[4].state_dict()
        expected = cursors[4].next_indices()
        cursors[4].load_state_dict(saved)
        self.assertEqual(cursors[4].next_indices(), expected)
        changed = dict(saved, rank=3)
        with self.assertRaises(ValueError):
            cursors[4].load_state_dict(changed)

    def test_age_filter_selects_rows_without_changing_times(self):
        import numpy as np
        class Dataset:
            manifest = dict(query_ages_seconds=[.36, .72])
            rows = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
        self.assertEqual(selected_rows(Dataset(), [.36]).tolist(), [0, 2])
        with self.assertRaises(ValueError):
            selected_rows(Dataset(), [.18])

    def test_fp32_accumulation_and_partial_checkpoint_rejection(self):
        low, high = nn.Parameter(torch.ones(1, dtype=torch.bfloat16)), nn.Parameter(torch.ones(1))
        optimizer = FP32MasterAdamW([low, high], lr=.01, weight_decay=0.)
        for _ in range(300):
            low.grad, high.grad = torch.ones_like(low), torch.ones_like(high)
            optimizer.accumulate_gradients()
            self.assertIsNone(low.grad)
            self.assertIsNone(high.grad)
        self.assertTrue(all(grad.item() == 300. for _,grad in optimizer.accumulated_gradients()))
        with self.assertRaises(RuntimeError):
            optimizer.state_dict()
        norm = optimizer.clip_accumulated_grad_norm_(1.)
        self.assertAlmostEqual(float(norm), math_sqrt(2*300**2), delta=1e-3)
        optimizer.step()
        optimizer.state_dict()
        self.assertEqual(optimizer.state[low]["exp_avg"].dtype, torch.float32)

    def test_real_two_process_ddp_checkpoint_and_next_update(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(_ddp_worker, args=(str(Path(folder)/"rendezvous"), folder), nprocs=2, join=True)
            self.assertTrue(all(Path(folder, f"rank_{rank}_passed").exists() for rank in range(2)))


def math_sqrt(value):
    return value**.5


if __name__ == "__main__":
    unittest.main()
