import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from dynamicwam.training.data.paired_index import action_target_times, causal_cache_offsets, jittered_cache_age
from dynamicwam.training.data.paired_world_action_dataset import PairedWorldActionDataset, paired_collate_fn


class PackedFixture:
    def __init__(self, root):
        self.root = root
        self.lang_groups = {0: [0]}

    def __getitem__(self, index):
        return dict(episode_index=0, condition_latent=torch.full((1, 1), float(index)),
            future_latent=torch.full((1, 2), float(index + 1000)),
            initial_state=torch.full((14,), float(index)),
            absolute_motion_features=torch.full((4, 12), float(index)),
            absolute_motion_interval_valid_mask=torch.ones(4, dtype=torch.bool),
            absolute_motion_acceleration_valid_mask=torch.ones(4, dtype=torch.bool),
            action_sequence=torch.full((16, 14), float(index + 2000)))

    def _load_language_embedding(self, *args):
        return torch.zeros(2, 8)


class JointAsyncDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [dict(episode_index=0, condition_frame_idx=i, sample_id=i,
            split="clean", task_name="fixture", episode_name="0", lang_group_id=0) for i in range(60)]
        (self.root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        # Nonuniform clock catches accidental replacement with frame-rate time.
        self.clock = dict(frame_index=torch.arange(120), sim_step_index=torch.arange(120),
            sim_time_seconds=torch.arange(120, dtype=torch.float64).square() * .01)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return PairedWorldActionDataset(self.root, self.root, splits=["clean"], execute_horizon=4,
            every_fast_queries=4, global_downsample_rate=3, chunk_size=16, seconds_per_frame=.03,
            packed=PackedFixture(self.root), episode_partition={0: "train"},
            clock_reader=lambda r: self.clock,
            frame_reader=lambda r: np.full((360, 320, 3), r.fast_frame, dtype=np.uint8), **kwargs)

    def test_timestamp_semantics_nonuniform_and_fail_closed(self):
        times = action_target_times({0: 1., 3: 1.3, 6: 2.2}, 0, chunk_size=2, global_downsample_rate=3)
        self.assertEqual(times, [1.3, 2.2])
        with self.assertRaises(ValueError):
            action_target_times({0: 1., 3: 1.3, 6: 1.2}, 0, chunk_size=2, global_downsample_rate=3)
        with self.assertRaises(ValueError):
            action_target_times({0: 1., 3: 1.3}, 0, chunk_size=2, global_downsample_rate=3)

    def test_cache_support_rejects_noncausal_and_duplicates(self):
        self.assertEqual(causal_cache_offsets(cache_age_actions=[3, 0, 1], global_downsample_rate=3), [0, 3, 9])
        for ages in ([0, -1], [0, 0], [1], [0, 1.5], [0, True]):
            with self.assertRaises(ValueError):
                causal_cache_offsets(cache_age_actions=ages, global_downsample_rate=3)

    def test_jitter_is_deterministic_epoch_sensitive_and_bounded(self):
        def draw(epoch):
            return jittered_cache_age(source_sample_id=12, base_age=8, slot_width=4,
                valid_ages=[8, 9, 10, 11], seed=2, epoch=epoch)
        first = [draw(e) for e in range(20)]
        self.assertEqual(first, [draw(e) for e in range(20)])
        self.assertGreater(len(set(first)), 1)
        self.assertTrue(set(first) <= {8, 9, 10, 11})

    def test_jitter_preserves_record_budget_and_holdout(self):
        old = self.dataset()
        new = self.dataset(cache_age_jitter=True, joint_supervision=True, cache_age_seed=8)
        self.assertEqual(len(old), len(new))
        self.assertEqual(new.legacy_pairing_identity, old.pairing_identity)
        self.assertEqual(new.episode_partition_identity, old.episode_partition_identity)
        self.assertNotEqual(new.pairing_identity, old.pairing_identity)
        self.assertEqual(new.max_supported_cache_age_actions, 15)
        for epoch in range(3):
            new.set_epoch(epoch)
            for index in range(len(new)):
                record = new._record_for_epoch(index)
                self.assertLess(record.fast_frame, 60)
                self.assertGreaterEqual(record.fast_frame, record.source_frame)
                self.assertEqual(record.fast_sample_id, record.fast_frame)

    def test_joint_labels_separate_from_causal_observations(self):
        dataset = self.dataset(joint_supervision=True)
        sample = dataset[1]  # source0 -> fast12, cache age4
        self.assertEqual(sample["source_sample_id"], 0)
        self.assertEqual(sample["fast_sample_id"], 12)
        self.assertEqual(sample["cache_age_actions"], 4)
        self.assertEqual(sample["joint_targets"]["source_action_sequence"][0, 0].item(), 2000)
        self.assertEqual(sample["fast_action_sequence"][0, 0].item(), 2012)
        self.assertEqual(sample["source_condition_latent"].item(), 0)
        self.assertFalse(torch.equal(sample["source_rgb"], sample["fast_rgb"]))
        self.assertGreater(sample["fast_rgb"].sum().item(), sample["source_rgb"].sum().item())
        self.assertAlmostEqual(sample["joint_targets"]["source_action_target_times"][0].item(), .09)
        self.assertAlmostEqual(sample["joint_targets"]["fast_action_target_times"][0].item(), 2.25)
        collated = paired_collate_fn([sample, sample])
        self.assertEqual(collated["joint_targets"]["fast_action_target_times"].dtype, torch.float64)
        self.assertEqual(collated["source_rgb"].shape[0], 2)
        self.assertEqual(collated["fast_motion"].shape, (2, 4, 12))

    def test_legacy_surface_and_clock_requirement(self):
        old = self.dataset()
        self.assertNotIn("joint_targets", old[1])
        # Legacy subclass constructors intentionally skip the parent initializer.
        del old.cache_age_jitter
        del old.joint_supervision
        self.assertNotIn("joint_targets", old[1])
        with self.assertRaises(ValueError):
            self.dataset(joint_supervision=True, clock_mode="action_only")
        with self.assertRaises(ValueError):
            self.dataset(cache_age_jitter=True, cache_age_actions=[0, 4, 8, 12])


if __name__ == "__main__":
    unittest.main()


