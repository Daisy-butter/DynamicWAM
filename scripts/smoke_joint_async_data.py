#!/usr/bin/env python3
"""Bounded CPU-only real-data loading smoke; no simulator or GPU APIs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import torch
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.paired_world_action_dataset import PairedWorldActionDataset, paired_collate_fn


class MetadataSubset:
    def __init__(self, packed, root):
        self._packed, self.root = packed, root
        self.lang_groups = packed.lang_groups

    def __getitem__(self, index):
        return self._packed[index]

    def _load_language_embedding(self, *args):
        return self._packed._load_language_embedding(*args)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(Path("/SSD_DISK/users/wuruihan").resolve()):
        raise ValueError("smoke output must stay on approved SSD")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    rows = []
    with (args.packed_root / "samples.jsonl").open() as handle:
        for _ in range(64):
            rows.append(json.loads(next(handle)))
    subset = out / "metadata_subset"
    subset.mkdir()
    (subset / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    packed = PackedAbsoluteMotionDataset(args.packed_root, max_open_shards=1)
    dataset = PairedWorldActionDataset(args.packed_root, args.source_root, splits=["clean", "randomized"],
        execute_horizon=4, every_fast_queries=4, global_downsample_rate=3, chunk_size=16,
        seconds_per_frame=1 / 30, cache_age_jitter=True, cache_age_seed=20261015,
        joint_supervision=True, max_open_shards=1, max_open_clocks=1,
        packed=MetadataSubset(packed, subset),
        episode_partition={int(row["episode_index"]): "train" for row in rows})
    evidence = []
    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        for index in range(4):
            item = dataset[index]
            original_fast = packed[int(item["fast_sample_id"])]
            for key, packed_key in (("fast_motion", "absolute_motion_features"),
                                    ("fast_action_sequence", "action_sequence")):
                if not torch.equal(item[key], original_fast[packed_key]):
                    raise AssertionError("native motion or normalized action changed")
            times = item["joint_targets"]["fast_action_target_times"]
            if not (torch.all(times[1:] > times[:-1]) and times[0] > item["fast_sim_time"]):
                raise AssertionError("invalid real action target clock")
            evidence.append(dict(epoch=epoch, source_sample_id=item["source_sample_id"],
                fast_sample_id=item["fast_sample_id"], cache_age_actions=item["cache_age_actions"],
                source_time=item["source_sim_time"], fast_time=item["fast_sim_time"],
                first_target_time=times[0].item(), last_target_time=times[-1].item(),
                fast_motion_sha256=digest(item["fast_motion"]),
                source_rgb_shape=list(item["source_rgb"].shape), fast_rgb_shape=list(item["fast_rgb"].shape)))
    dataset.set_epoch(0)
    batch = paired_collate_fn([dataset[0], dataset[1]])
    torch.save(batch, out / "actual_cpu_batch.pt")
    dataset.close()
    receipt = dict(passed=True, device="cpu", rgb_sample_pairs_loaded=10, rgb_frame_reads_upper_bound=20, packed_source_candidates=len(rows),
        bounded_records=len(dataset), samples_checked=len(evidence), epoch_values=[0, 1],
        native_motion_and_normalized_action_equal=True, evidence=evidence,
        dataset_identity=packed.dataset_identity, batch=str(out / "actual_cpu_batch.pt"),
        limitations="bounded first-episode data smoke; full corpus clock validation remains on load")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("passed", "samples_checked", "bounded_records", "batch")}))


if __name__ == "__main__":
    main()
