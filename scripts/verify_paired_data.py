"""Pair index, shared language id, and causal batch assembly."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from dynamicwam.models.multirate_types import reject_leaky_source_fields
from dynamicwam.training.data.paired_index import steady_state_offsets
from dynamicwam.training.data.paired_world_action_dataset import (
    CLOCK_KEYS,
    build_pair_records,
    clock_time_seconds,
    episode_lang_id,
    load_episode_clock,
    paired_collate_fn,
    write_pair_manifest,
)
from dynamicwam.training.train.common import build_training_flow_scheduler
from dynamicwam.training.train.multirate_trainer import prepare_multirate_batch


def _row(episode: int, frame: int, sample: int, split: str = "clean") -> dict:
    return {
        "episode_index": episode,
        "condition_frame_idx": frame,
        "sample_id": sample,
        "lang_group_id": 7,
        "split": split,
        "task_name": "move_can_pot",
        "episode_name": f"episode{episode}",
    }


def _sample(batch: int = 2) -> dict:
    text = [torch.zeros(4, 8), torch.zeros(2, 8)]
    return {
        "source_condition_latent": torch.zeros(batch, 2, 1, 2, 2),
        "source_state": torch.zeros(batch, 4),
        "source_motion": torch.zeros(batch, 4, 12),
        "source_motion_interval_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "source_motion_acceleration_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "bridge_noise_shape_latent": torch.full((batch, 2, 1, 2, 2), 3.0),
        "fast_rgb": torch.zeros(batch, 3, 3, 16, 16),
        "fast_state": torch.ones(batch, 4),
        "fast_motion": torch.zeros(batch, 4, 12),
        "fast_motion_interval_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "fast_motion_acceleration_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "fast_action_sequence": torch.zeros(batch, 4, 4),
        "fast_future_latent": torch.ones(batch, 2, 1, 2, 2),
        "fast_condition_latent": torch.zeros(batch, 2, 1, 2, 2),
        "text_embeddings": text,
        "delta_sim_seconds": torch.tensor([0.0, 0.48]),
        "delta_actions": torch.tensor([0.0, 12.0]),
        "state_delta": torch.ones(batch, 4),
        "offset_label": ["steady_state", "stress"],
    }


def main() -> None:
    lang_ids = [10, 11, 12]
    if episode_lang_id(lang_ids) != 10:
        raise SystemExit("episode language id was not pinned to the first row")
    if lang_ids[1 % len(lang_ids)] == episode_lang_id(lang_ids):
        raise SystemExit("the pin did not differ from a later sample index")

    offsets = steady_state_offsets(
        execute_horizon=4,
        every_fast_queries=4,
        global_downsample_rate=3,
    )
    if offsets != [0, 12, 24, 36]:
        raise SystemExit(f"steady-state offsets were {offsets}")

    samples = [
        _row(0, 0, 1),
        _row(0, 12, 2),
        _row(1, 0, 3, split="holdout"),
        _row(1, 12, 4, split="holdout"),
    ]
    records, drops = build_pair_records(
        samples,
        offsets=[0, 12],
        stress_offsets=[48],
        splits=["clean"],
        chunk_size=1,
        global_downsample_rate=3,
        lang_ids_by_group={7: lang_ids},
        total_frames_by_episode={0: 40, 1: 40},
    )
    keys = {(record.episode_index, record.source_frame, record.fast_frame) for record in records}
    if keys != {(0, 0, 0), (0, 0, 12), (0, 12, 12)}:
        raise SystemExit(f"unexpected pairs: {sorted(keys)}")
    if any(record.episode_index == 1 for record in records):
        raise SystemExit("a holdout episode was paired")
    if any(record.lang_id != 10 for record in records):
        raise SystemExit("pairs did not share the episode language id")
    if drops["missing_fast_frame"] < 1 or drops["split_filtered"] < 1:
        raise SystemExit(f"drop report did not count rejected pairs: {drops}")
    labels = {record.offset: record.offset_label for record in records}
    if labels[0] != "steady_state":
        raise SystemExit("offset 0 was not labeled steady-state")

    past, past_drops = build_pair_records(
        [_row(0, 0, 1), _row(0, 12, 2)],
        offsets=[0, 12],
        splits=["clean"],
        chunk_size=1,
        global_downsample_rate=3,
        lang_ids_by_group={7: lang_ids},
        total_frames_by_episode={0: 8},
    )
    if any(record.fast_frame == 12 for record in past):
        raise SystemExit("an action window past the episode was clamped into the set")
    if past_drops["action_window_past_end"] < 1:
        raise SystemExit("past-end pairs were not reported")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clock_path = root / "clock.pt"
        torch.save(
            {
                "frame_index": torch.tensor([0, 12]),
                "sim_step_index": torch.tensor([0, 30]),
                "sim_time_seconds": torch.tensor([0.0, 0.4]),
                "contact": torch.tensor([1.0]),
            },
            clock_path,
        )
        clock = load_episode_clock(clock_path)
        if set(clock) != set(CLOCK_KEYS):
            raise SystemExit(f"clock loader kept privileged fields: {sorted(clock)}")
        if abs(clock_time_seconds(clock, 12) - 0.4) > 1e-6:
            raise SystemExit("simulator time lookup failed")
        write_pair_manifest(root / "pairs.jsonl", records)
        lines = (root / "pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if len(lines) != len(records):
            raise SystemExit("pair manifest length does not match the records")

    scheduler = build_training_flow_scheduler(
        {
            "flow_matching": {
                "shift": 5.0,
                "sigma_min": 0.0,
                "extra_one_step": True,
                "training_timesteps": 16,
            }
        }
    )
    prepared = prepare_multirate_batch(
        _sample(),
        timesteps=scheduler.timesteps,
        sigmas=scheduler.sigmas,
        device="cpu",
        dtype=torch.float32,
        phase="fast_warmup",
        null_bridge_probability=0.0,
        sim_scale=1.0,
        action_scale=16.0,
        null_bridge=False,
    )
    source_batch = prepared["source"].as_model_batch()
    reject_leaky_source_fields(source_batch)
    if "future_latent" in source_batch or "action_sequence" in source_batch:
        raise SystemExit("the causal source received future labels")
    if torch.allclose(prepared["source_noise"].future_latent, torch.full_like(prepared["source_noise"].future_latent, 3.0)):
        raise SystemExit("bridge noise reused the shape-donor values")
    if "future_latent" not in prepared["aligned"]:
        raise SystemExit("the regularizer lost its future latent")
    if prepared["current"].age.features(sim_scale=1.0, action_scale=16.0).ndim != 2:
        raise SystemExit("batched age features were flattened")
    if prepared["fast_text"].shape[0] != 2 or prepared["fast_text_mask"].shape != (2, 4):
        raise SystemExit("text padding did not keep one shared embedding per row")
    collated = paired_collate_fn(
        [
            {
                "source_condition_latent": torch.zeros(2, 1, 2, 2),
                "source_state": torch.zeros(4),
                "source_motion": torch.zeros(4, 12),
                "source_motion_interval_valid_mask": torch.ones(4, dtype=torch.bool),
                "source_motion_acceleration_valid_mask": torch.ones(4, dtype=torch.bool),
                "bridge_noise_shape_latent": torch.zeros(2, 1, 2, 2),
                "fast_rgb": torch.zeros(3, 3, 16, 16),
                "fast_rgb_valid_mask": torch.ones(3, 16, 16, dtype=torch.bool),
                "fast_state": torch.zeros(4),
                "fast_motion": torch.zeros(4, 12),
                "fast_motion_interval_valid_mask": torch.ones(4, dtype=torch.bool),
                "fast_motion_acceleration_valid_mask": torch.ones(4, dtype=torch.bool),
                "fast_action_sequence": torch.zeros(4, 4),
                "fast_future_latent": torch.zeros(2, 1, 2, 2),
                "fast_condition_latent": torch.zeros(2, 1, 2, 2),
                "text_embedding": torch.zeros(3, 8),
                "delta_sim_seconds": 0.0,
                "delta_actions": 0,
                "state_delta": torch.zeros(4),
                "offset_label": "steady_state",
            }
        ]
    )
    if collated["fast_rgb"].ndim != 5:
        raise SystemExit("collate did not stack RGB as a batch")
    print("verified paired data")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_paired_data failed: {exc}", file=sys.stderr)
        raise
