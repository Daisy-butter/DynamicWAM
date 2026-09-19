"""Scan the packed Level-1 corpus and write newton_stats.json without repacking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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


def _accumulate(
    newton: torch.Tensor,
    interval_valid: torch.Tensor,
    acceleration_valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = newton.detach().cpu().numpy().astype(np.float64)
    valid = np.repeat(
        interval_valid.detach().cpu().numpy()[..., None],
        NEWTON_FEATURE_DIM,
        axis=-1,
    )
    accel = acceleration_valid.detach().cpu().numpy()[..., None]
    valid[..., list(NEWTON_ACCELERATION_FEATURE_INDICES)] = accel
    flat = values.reshape(-1, NEWTON_FEATURE_DIM)
    mask = valid.reshape(-1, NEWTON_FEATURE_DIM)
    count = mask.sum(axis=0, dtype=np.int64)
    total = np.where(mask, flat, 0.0).sum(axis=0, dtype=np.float64)
    total_square = np.where(mask, flat * flat, 0.0).sum(axis=0, dtype=np.float64)
    return count, total, total_square


def run(*, config_path: str) -> Path:
    profile = load_profile(config_path)
    raw = profile.raw
    packed_root = Path(raw["paths"]["packed_dataset"])
    shard_dir = packed_root / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"packed shard directory is missing: {shard_dir}")
    newton = profile._explicit_dynamics()
    from safetensors import safe_open

    metadata_path = packed_root / "dataset.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "dynamicwam_absolute_motion_dataset":
            raise RuntimeError(f"packed dataset format is unexpected: {metadata_path}")
    count = np.zeros(NEWTON_FEATURE_DIM, dtype=np.int64)
    total = np.zeros(NEWTON_FEATURE_DIM, dtype=np.float64)
    total_square = np.zeros(NEWTON_FEATURE_DIM, dtype=np.float64)
    shard_paths = sorted(shard_dir.glob("shard_*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no packed shards found in {shard_dir}")
    for index, shard_path in enumerate(shard_paths, start=1):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            features = handle.get_tensor("absolute_motion_features")
            interval_valid = handle.get_tensor("absolute_motion_interval_valid_masks")
            acceleration_valid = handle.get_tensor(
                "absolute_motion_acceleration_valid_masks"
            )
        rollout, window_valid, window_accel = rollout_image_plane_newton(
            features,
            interval_valid,
            acceleration_valid,
            action_interval_seconds=float(newton["action_interval_seconds"]),
            horizon_steps=int(newton["horizon_steps"]),
            window_count=int(newton["window_count"]),
        )
        shard_count, shard_total, shard_square = _accumulate(
            rollout,
            window_valid,
            window_accel,
        )
        count += shard_count
        total += shard_total
        total_square += shard_square
        print(
            f"scanned newton shard {index}/{len(shard_paths)}: {shard_path.name}",
            flush=True,
        )

    if np.any(count <= 0):
        raise RuntimeError(
            f"some newton features have no valid observations: {count.tolist()}"
        )
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    standard_deviation = np.sqrt(variance)
    minimum_scale = 1e-6
    scale = np.maximum(standard_deviation, minimum_scale)
    payload = {
        "schema_version": NEWTON_STATISTICS_VERSION,
        "feature_names": list(NEWTON_FEATURE_NAMES),
        "count": count.astype(np.int64).tolist(),
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "scale": scale.tolist(),
        "minimum_scale": float(minimum_scale),
        "temporal_contract": NEWTON_TEMPORAL_CONTRACT,
        "source": NEWTON_SOURCE,
        "window_count": int(newton["window_count"]),
        "horizon_steps": int(newton["horizon_steps"]),
        "action_interval_seconds": float(newton["action_interval_seconds"]),
    }
    output_path = packed_root / NEWTON_STATS_FILE
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    output_path = run(config_path=arguments.config)
    print(output_path)


if __name__ == "__main__":
    main()
