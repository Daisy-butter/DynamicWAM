"""Rebuild packed 12-D kinematics from cached flow RGB with blob statistics."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from dynamicwam.absolute_motion import (
    MOTION_FEATURE_DIM,
    MOTION_FEATURE_NAMES,
    MOTION_STATISTICS_VERSION,
    TEMPORAL_CONTRACT,
    build_checkpoint_motion_metadata,
    build_flow_cache_parameters,
    load_exact_flow_cache,
    raw_pairs,
    rebuild_cache_motion_features,
    validate_motion_statistics,
    write_exact_flow_cache,
)
from dynamicwam.config import load_profile
from dynamicwam.explicit_dynamics import (
    NEWTON_ACCELERATION_FEATURE_INDICES,
    NEWTON_FEATURE_DIM,
    NEWTON_FEATURE_NAMES,
    NEWTON_SOURCE,
    NEWTON_STATISTICS_VERSION,
    NEWTON_STATS_FILE,
    NEWTON_TEMPORAL_CONTRACT,
    rollout_image_plane_newton,
)
from dynamicwam.training.data.packed_dataset import (
    TRAIN_DATASET_METADATA,
    TRAIN_DATASET_MOTION_STATS,
    TRAIN_DATASET_SHARD_DIR,
    _hash_json_payload,
    _iter_jsonl,
    _read_json,
)

CACHE_MARKER = "blob_foreground_v1.json"
CACHE_PROGRESS = "blob_foreground_v1.progress.jsonl"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rewrite_one_cache(
    job: tuple[str, str, int],
) -> str:
    path_value, params_json, stride = job
    cache_path = Path(path_value)
    arrays = load_exact_flow_cache(
        cache_path,
        expected_params_json=params_json,
    )
    features = rebuild_cache_motion_features(arrays, raw_stride=stride)
    arrays["motion_features"] = features
    write_exact_flow_cache(
        cache_path,
        arrays=arrays,
        params_json=params_json,
    )
    return str(cache_path)


def _rewrite_flow_caches(
    *,
    cache_root: Path,
    params_json: str,
    stride: int,
    workers: int,
) -> int:
    marker = cache_root / CACHE_MARKER
    if marker.is_file():
        raise RuntimeError(
            "flow caches already have blob kinematics; refusing to invert "
            f"with overwritten p99 scales: {marker}"
        )
    all_paths = sorted(cache_root.glob("*/*/videos/*.flow.npz"))
    if not all_paths:
        raise FileNotFoundError(f"no flow caches found under {cache_root}")
    progress_path = cache_root / CACHE_PROGRESS
    done: set[str] = set()
    if progress_path.is_file():
        done = {
            line.strip()
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [path for path in all_paths if str(path) not in done]
    completed = len(all_paths) - len(pending)
    if pending:
        jobs = [(str(path), params_json, int(stride)) for path in pending]
        with (
            ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool,
            progress_path.open("a", encoding="utf-8") as progress,
        ):
            futures = [pool.submit(_rewrite_one_cache, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                rewritten = future.result()
                progress.write(rewritten + "\n")
                progress.flush()
                completed += 1
                if index == 1 or index == len(jobs) or index % 200 == 0:
                    print(
                        f"rewrote flow cache {completed}/{len(all_paths)}",
                        flush=True,
                    )
    elif completed:
        print(f"all {completed} flow caches already rewritten", flush=True)
    _write_json(
        marker,
        {
            "version": 1,
            "count": completed,
            "raw_stride": int(stride),
        },
    )
    progress_path.unlink(missing_ok=True)
    return completed


def _episode_feature_table(
    *,
    cache_root: Path,
    packed_root: Path,
    params_json: str,
) -> dict[int, np.ndarray]:
    episodes = list(_iter_jsonl(packed_root / "episodes.jsonl"))
    table: dict[int, np.ndarray] = {}
    for index, episode in enumerate(episodes, start=1):
        cache_path = (
            cache_root
            / str(episode["split"])
            / str(episode["task_name"])
            / "videos"
            / f"{episode['episode_name']}.flow.npz"
        )
        arrays = load_exact_flow_cache(
            cache_path,
            expected_params_json=params_json,
        )
        table[int(episode["episode_id"])] = arrays["motion_features"]
        if index == 1 or index == len(episodes) or index % 500 == 0:
            print(
                f"loaded rewritten cache features {index}/{len(episodes)}",
                flush=True,
            )
    return table


def _sample_features(
    *,
    episode_features: np.ndarray,
    condition_index: int,
    history_count: int,
    policy_stride: int,
    global_downsample_rate: int,
) -> np.ndarray:
    pairs = raw_pairs(
        int(condition_index),
        history_count=int(history_count),
        policy_stride=int(policy_stride),
        global_downsample_rate=int(global_downsample_rate),
    )
    return np.stack(
        [episode_features[current] for _previous, current in pairs],
        axis=0,
    ).astype(np.float32)


def _accumulate_motion(
    features: np.ndarray,
    interval_valid: np.ndarray,
    acceleration_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    valid = np.repeat(interval_valid[..., None], MOTION_FEATURE_DIM, axis=-1)
    valid[..., 9:12] = acceleration_valid[..., None]
    flat = values.reshape(-1, MOTION_FEATURE_DIM)
    mask = valid.reshape(-1, MOTION_FEATURE_DIM)
    count = mask.sum(axis=0, dtype=np.int64)
    total = np.where(mask, flat, 0.0).sum(axis=0, dtype=np.float64)
    total_square = np.where(mask, flat * flat, 0.0).sum(axis=0, dtype=np.float64)
    return count, total, total_square


def _accumulate_newton(
    features: torch.Tensor,
    interval_valid: torch.Tensor,
    acceleration_valid: torch.Tensor,
    *,
    action_interval_seconds: float,
    horizon_steps: int,
    window_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    newton, window_valid, window_accel = rollout_image_plane_newton(
        features,
        interval_valid,
        acceleration_valid,
        action_interval_seconds=action_interval_seconds,
        horizon_steps=horizon_steps,
        window_count=window_count,
    )
    values = newton.detach().cpu().numpy().astype(np.float64)
    valid = np.repeat(
        window_valid.detach().cpu().numpy()[..., None],
        NEWTON_FEATURE_DIM,
        axis=-1,
    )
    accel = window_accel.detach().cpu().numpy()[..., None]
    valid[..., list(NEWTON_ACCELERATION_FEATURE_INDICES)] = accel
    flat = values.reshape(-1, NEWTON_FEATURE_DIM)
    mask = valid.reshape(-1, NEWTON_FEATURE_DIM)
    count = mask.sum(axis=0, dtype=np.int64)
    total = np.where(mask, flat, 0.0).sum(axis=0, dtype=np.float64)
    total_square = np.where(mask, flat * flat, 0.0).sum(axis=0, dtype=np.float64)
    return count, total, total_square


def _stats_payload(
    *,
    feature_names: tuple[str, ...],
    schema_version: int,
    temporal_contract: str,
    count: np.ndarray,
    total: np.ndarray,
    total_square: np.ndarray,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if np.any(count <= 0):
        raise RuntimeError(f"some features have no valid observations: {count.tolist()}")
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    standard_deviation = np.sqrt(variance)
    minimum_scale = 1e-6
    scale = np.maximum(standard_deviation, minimum_scale)
    payload = {
        "schema_version": int(schema_version),
        "feature_names": list(feature_names),
        "count": count.astype(np.int64).tolist(),
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": float(minimum_scale),
        "temporal_contract": temporal_contract,
    }
    payload.update(extra)
    return payload


def _rewrite_packed_shards(
    *,
    packed_root: Path,
    feature_table: dict[int, np.ndarray],
    history_count: int,
    policy_stride: int,
    global_downsample_rate: int,
    newton: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    shard_dir = packed_root / TRAIN_DATASET_SHARD_DIR
    shard_paths = sorted(shard_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no packed shards found in {shard_dir}")
    motion_count = np.zeros(MOTION_FEATURE_DIM, dtype=np.int64)
    motion_total = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)
    motion_square = np.zeros(MOTION_FEATURE_DIM, dtype=np.float64)
    newton_count = np.zeros(NEWTON_FEATURE_DIM, dtype=np.int64)
    newton_total = np.zeros(NEWTON_FEATURE_DIM, dtype=np.float64)
    newton_square = np.zeros(NEWTON_FEATURE_DIM, dtype=np.float64)
    for index, shard_path in enumerate(shard_paths, start=1):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensors = {key: handle.get_tensor(key) for key in keys}
        episode_indices = tensors["episode_indices"].cpu().numpy().astype(np.int64)
        condition_indices = (
            tensors["condition_frame_indices"].cpu().numpy().astype(np.int64)
        )
        rebuilt = np.zeros(
            (episode_indices.shape[0], history_count, MOTION_FEATURE_DIM),
            dtype=np.float32,
        )
        for row, (episode_id, condition_index) in enumerate(
            zip(episode_indices.tolist(), condition_indices.tolist())
        ):
            rebuilt[row] = _sample_features(
                episode_features=feature_table[int(episode_id)],
                condition_index=int(condition_index),
                history_count=history_count,
                policy_stride=policy_stride,
                global_downsample_rate=global_downsample_rate,
            )
        tensors["absolute_motion_features"] = torch.from_numpy(rebuilt)
        temporary = shard_path.with_name(f".{shard_path.name}.tmp.{os.getpid()}")
        try:
            save_file(tensors, str(temporary))
            temporary.replace(shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        interval = tensors["absolute_motion_interval_valid_masks"].numpy()
        accel = tensors["absolute_motion_acceleration_valid_masks"].numpy()
        shard_count, shard_total, shard_square = _accumulate_motion(
            rebuilt,
            interval,
            accel,
        )
        motion_count += shard_count
        motion_total += shard_total
        motion_square += shard_square
        n_count, n_total, n_square = _accumulate_newton(
            tensors["absolute_motion_features"],
            tensors["absolute_motion_interval_valid_masks"],
            tensors["absolute_motion_acceleration_valid_masks"],
            action_interval_seconds=float(newton["action_interval_seconds"]),
            horizon_steps=int(newton["horizon_steps"]),
            window_count=int(newton["window_count"]),
        )
        newton_count += n_count
        newton_total += n_total
        newton_square += n_square
        print(f"rewrote packed shard {index}/{len(shard_paths)}: {shard_path.name}", flush=True)
        del tensors
    motion_stats = _stats_payload(
        feature_names=MOTION_FEATURE_NAMES,
        schema_version=MOTION_STATISTICS_VERSION,
        temporal_contract=TEMPORAL_CONTRACT,
        count=motion_count,
        total=motion_total,
        total_square=motion_square,
        extra={},
    )
    newton_stats = _stats_payload(
        feature_names=NEWTON_FEATURE_NAMES,
        schema_version=NEWTON_STATISTICS_VERSION,
        temporal_contract=NEWTON_TEMPORAL_CONTRACT,
        count=newton_count,
        total=newton_total,
        total_square=newton_square,
        extra={
            "source": NEWTON_SOURCE,
            "window_count": int(newton["window_count"]),
            "horizon_steps": int(newton["horizon_steps"]),
            "action_interval_seconds": float(newton["action_interval_seconds"]),
        },
    )
    return motion_stats, newton_stats


def run(*, config_path: str, workers: int) -> None:
    profile = load_profile(config_path)
    raw = profile.raw
    packed_root = Path(raw["paths"]["packed_dataset"])
    cache_root = Path(raw["paths"]["head_flow_cache"])
    flow = raw["method"]["head_flow"]
    video = raw["method"]["video"]
    params = build_flow_cache_parameters(
        head_flow_config=flow,
        global_downsample_rate=int(video["global_downsample_rate"]),
    )
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    stride = int(params["raw_stride"])
    history_count = int(flow["count"])
    policy_stride = int(flow["policy_stride"])
    downsample = int(video["global_downsample_rate"])
    print(f"rewriting flow caches under {cache_root}", flush=True)
    cache_count = _rewrite_flow_caches(
        cache_root=cache_root,
        params_json=params_json,
        stride=stride,
        workers=workers,
    )
    print(f"rewrote {cache_count} flow caches", flush=True)
    feature_table = _episode_feature_table(
        cache_root=cache_root,
        packed_root=packed_root,
        params_json=params_json,
    )
    newton = profile._explicit_dynamics()
    motion_stats, newton_stats = _rewrite_packed_shards(
        packed_root=packed_root,
        feature_table=feature_table,
        history_count=history_count,
        policy_stride=policy_stride,
        global_downsample_rate=downsample,
        newton=newton,
    )
    _write_json(packed_root / TRAIN_DATASET_MOTION_STATS, motion_stats)
    _write_json(packed_root / NEWTON_STATS_FILE, newton_stats)
    motion_stats = validate_motion_statistics(
        _read_json(packed_root / TRAIN_DATASET_MOTION_STATS)
    )
    metadata = _read_json(packed_root / TRAIN_DATASET_METADATA)
    absolute_motion = build_checkpoint_motion_metadata(
        history_count=history_count,
        statistics=motion_stats,
        statistics_sha256=_hash_json_payload(motion_stats),
        head_flow_config=flow,
    )
    absolute_motion["statistics_file"] = TRAIN_DATASET_MOTION_STATS
    metadata["absolute_motion"] = absolute_motion
    fingerprint_payload = {
        "format": metadata["format"],
        "version": metadata["version"],
        "manifest_sha256": metadata["manifest_sha256"],
        "sample_count": metadata["sample_count"],
        "episode_count": metadata["episode_count"],
        "shard_size": metadata["shard_size"],
        "data": metadata["data"],
        "latent": metadata["latent"],
        "language": {
            "dtype": metadata["language"]["dtype"],
            "items_per_shard": int(raw["packing"]["language_items_per_shard"]),
        },
        "sampling": metadata["sampling"],
        "absolute_motion": absolute_motion,
    }
    metadata["dataset_fingerprint"] = _hash_json_payload(fingerprint_payload)
    _write_json(packed_root / TRAIN_DATASET_METADATA, metadata)
    print(
        "updated motion_stats newton_stats and dataset fingerprint "
        f"{metadata['dataset_fingerprint']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    run(config_path=arguments.config, workers=int(arguments.workers))


if __name__ == "__main__":
    main()
