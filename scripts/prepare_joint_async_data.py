#!/usr/bin/env python3
"""Freeze a compact asynchronous pairing index without reading video/latents.

This index describes all source candidates and their deterministic epoch-jitter
rule. Exact action timestamps remain in the immutable interception sidecars and
are validated when loaded; metadata completion is not a full clock-data audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.training.data.multirate_sampling import partition_episodes, pairing_identity


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def grouped_samples(path):
    """Packed v2 records are ordered by episode, which bounds temporary RAM."""
    previous = None
    group = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episode = int(row["episode_index"])
            if previous is not None and previous != episode:
                seen.add(previous)
                yield previous, group
                group = []
            if episode in seen:
                raise ValueError("packed samples are not episode-contiguous")
            previous = episode
            group.append(row)
        if group:
            yield previous, group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--every-fast-queries", type=int, default=4)
    parser.add_argument("--prefix-max", type=int, default=4)
    parser.add_argument("--partition-seed", type=int, default=20260928)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--jitter-seed", type=int, default=20261015)
    args = parser.parse_args()
    packed, source = args.packed_root.resolve(), args.source_root.resolve()
    destination = args.output_dir.resolve()
    allowed = Path("/SSD_DISK/users/wuruihan").resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError("joint data artifacts must stay below the verified SSD wuruihan root")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing data preparation")
    if min(args.execute_horizon, args.every_fast_queries) < 1:
        raise ValueError("query and refresh periods must be positive")
    metadata = json.loads((packed / "dataset.json").read_text())
    chunk = int(metadata["data"]["action_chunk_size"])
    stride = int(metadata["data"]["global_downsample_rate"])
    if not 0 <= args.prefix_max < chunk:
        raise ValueError("RTC prefix must leave a supervised action suffix")
    episodes = [json.loads(line) for line in (packed / "episodes.jsonl").open()]
    episode_rows = [dict(episode_index=int(row["episode_id"]), split=row["split"],
        task_name=row["task_name"], episode_name=row["episode_name"]) for row in episodes
        if int(row["sample_count"]) > 0]
    assignments = partition_episodes(episode_rows, seed=args.partition_seed,
        validation_fraction=args.validation_fraction)
    base_ages = [i * args.execute_horizon for i in range(args.every_fast_queries)]
    age_support = list(range(args.execute_horizon * args.every_fast_queries))
    partition_identity = pairing_identity(episode_rows, assignments, seed=args.partition_seed,
        validation_fraction=args.validation_fraction, clock_mode="required", offsets=[], stress_offsets=[])
    legacy_identity = pairing_identity(episode_rows, assignments, seed=args.partition_seed,
        validation_fraction=args.validation_fraction, clock_mode="required",
        offsets=[age * stride for age in base_ages], stress_offsets=[])
    sampling_identity = hashlib.sha256(json.dumps(dict(base_pairing_identity=legacy_identity,
        strategy="causal_slot_epoch_jitter_v1", slot_width=args.execute_horizon,
        support=age_support, seed=args.jitter_seed), sort_keys=True).encode()).hexdigest()
    destination.mkdir(parents=True)
    counts = Counter()
    task_counts = defaultdict(Counter)
    sample_hash = hashlib.sha256()
    # One sequential scan, no decoder, VAE, CUDA or simulator import.
    with (destination / "episode_index.jsonl").open("w", encoding="utf-8") as out:
        for episode, rows in grouped_samples(packed / "samples.jsonl"):
            frames = {int(row["condition_frame_idx"]): int(row["sample_id"]) for row in rows}
            if len(frames) != len(rows):
                raise ValueError(f"duplicate condition frame in episode {episode}")
            first = rows[0]
            subset = assignments[episode]
            eligible = {str(age): sum(frame + age * stride in frames for frame in frames)
                        for age in base_ages}
            clock_path = source / first["split"] / first["task_name"] / "interception" / (first["episode_name"] + ".pt")
            if not clock_path.is_file():
                raise FileNotFoundError(clock_path)
            # Compact full source lattice, not 16 materialized rows per source.
            pairs = sorted(frames.items())
            serialized = json.dumps(pairs, separators=(",", ":"))
            sample_hash.update(serialized.encode())
            row = dict(episode_index=episode, split=first["split"], task_name=first["task_name"],
                episode_name=first["episode_name"], subset=subset, source_candidates=pairs,
                eligible_base_pair_counts=eligible, clock_path=str(clock_path),
                clock_bytes=clock_path.stat().st_size, clock_validation="deferred_to_sample_loading")
            out.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            counts[f"{subset}_episodes"] += 1
            counts["episodes"] += 1
            counts["samples"] += len(rows)
            counts["materialized_slot_pairs"] += sum(eligible.values())
            task_counts[first["task_name"]][subset] += 1
    if counts["samples"] != int(metadata["sample_count"]):
        raise ValueError("source sample count differs from packed metadata")
    receipt = dict(schema_version=1, status="metadata_complete", dataset_identity=dict(
        dataset_fingerprint=metadata["dataset_fingerprint"], metadata_sha256=digest(packed / "dataset.json"),
        action_stats_sha256=digest(packed / "action_stats.json"), motion_stats_sha256=digest(packed / "motion_stats.json")),
        packed_root=str(packed), source_root=str(source), counts=dict(counts), task_split_counts=task_counts,
        episode_partition_identity=partition_identity, legacy_pairing_identity=legacy_identity,
        pairing_identity=sampling_identity, source_lattice_sha256=sample_hash.hexdigest(),
        episode_index_sha256=digest(destination / "episode_index.jsonl"),
        pairing=dict(strategy="causal_slot_epoch_jitter_v1", base_ages=base_ages, cache_age_actions=age_support,
            cache_age_seed=args.jitter_seed, epoch_changes=True, global_downsample_rate=stride,
            execute_horizon=args.execute_horizon, every_fast_queries=args.every_fast_queries,
            partition_seed=args.partition_seed, validation_fraction=args.validation_fraction),
        rtc=dict(prefix_max=args.prefix_max, chunk_size=chunk, action_target_time_source="sim_time_seconds",
            target_timestamps_are="endpoint_times_not_command_start_times", clock_mode="required",
            prefix_source="trainer_policy_or_expert_mixture_must_be_logged"),
        clock_audit="presence_only; exact uniqueness, monotonicity and label alignment checked on load",
        reads="metadata_and_clock_file_stats_only; no_images_latents_or_GPU",
        source_labels="joint_targets namespace; never runtime conditioning")
    (destination / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=receipt["status"], output=str(destination), counts=dict(counts),
        pairing_identity=sampling_identity), sort_keys=True))


if __name__ == "__main__":
    main()
