"""Image-plane Newtonian rollout from packed absolute-motion features.

The 12-D history descriptors already encode past displacement, velocity, and
finite-difference acceleration. Rolling the last interval forward would be
nearly tautological. This module fits object-scale constant-acceleration,
constant-velocity, and constant-turn models on the full history, then predicts
the future action-chunk horizon, leaving the learned policy to generate joints.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynamicwam.absolute_motion import MOTION_FEATURE_DIM

NEWTON_STATISTICS_VERSION = 2
NEWTON_STATS_FILE = "newton_stats.json"
NEWTON_TEMPORAL_CONTRACT = "image_plane_newton_rollout_v2"
NEWTON_SOURCE = "absolute_motion_features"

NEWTON_FEATURE_NAMES = (
    "ca_end_displacement_x_pixels",
    "ca_end_displacement_y_pixels",
    "ct_end_displacement_x_pixels",
    "ct_end_displacement_y_pixels",
    "cv_end_displacement_x_pixels",
    "cv_end_displacement_y_pixels",
    "ca_history_rmse_pixels_per_second",
    "ct_history_rmse_pixels_per_second",
    "cv_history_rmse_pixels_per_second",
    "omega_times_horizon",
    "tangential_acceleration_fraction",
    "object_speed_pixels_per_second",
)
NEWTON_FEATURE_DIM = len(NEWTON_FEATURE_NAMES)
NEWTON_ACCELERATION_FEATURE_INDICES = (0, 1, 2, 3, 6, 7, 9, 10)

_MEAN_VELOCITY_SLICE = slice(5, 7)
_DELTA_T_INDEX = 4
_P99_SPEED_INDEX = 8
_VELOCITY_EPS = 1e-6
_OMEGA_EPS = 1e-4
_DET_EPS = 1e-8

CHECKPOINT_NEWTON_METADATA_KEYS = frozenset(
    {
        "window_count",
        "horizon_steps",
        "action_interval_seconds",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "temporal_contract",
        "source",
        "statistics_sha256",
    }
)


def hash_json_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def newton_step_times(
    *,
    action_interval_seconds: float,
    horizon_steps: int,
) -> torch.Tensor:
    interval = float(action_interval_seconds)
    steps = int(horizon_steps)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("action_interval_seconds must be positive and finite")
    if steps <= 0:
        raise ValueError("horizon_steps must be positive")
    return torch.arange(1, steps + 1, dtype=torch.float32) * interval


def _window_step_groups(
    *,
    horizon_steps: int,
    window_count: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    horizon_steps = int(horizon_steps)
    window_count = int(window_count)
    if horizon_steps <= 0 or window_count <= 0:
        raise ValueError("horizon_steps and window_count must be positive")
    if horizon_steps % window_count != 0:
        raise ValueError(
            "horizon_steps must be divisible by window_count: "
            f"{horizon_steps} % {window_count}"
        )
    window_size = horizon_steps // window_count
    ends = torch.arange(1, window_count + 1, dtype=torch.int64) * window_size
    starts = ends - window_size
    return window_size, starts, ends


def _object_scale_velocity(features: torch.Tensor) -> torch.Tensor:
    mean_velocity = features[:, :, _MEAN_VELOCITY_SLICE]
    p99_speed = features[:, :, _P99_SPEED_INDEX].clamp(min=0.0)
    speed = torch.linalg.vector_norm(mean_velocity, dim=-1)
    well_defined = speed > _VELOCITY_EPS
    direction = mean_velocity / speed.clamp(min=_VELOCITY_EPS).unsqueeze(-1)
    scaled = direction * p99_speed.unsqueeze(-1)
    return torch.where(well_defined.unsqueeze(-1), scaled, mean_velocity)


def _interval_center_times(delta_t: torch.Tensor) -> torch.Tensor:
    duration = delta_t.clamp(min=0.0)
    remaining = duration.flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,))
    t_end = duration - remaining
    t_start = t_end - duration
    return 0.5 * (t_start + t_end)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    extra_dims: int = 0,
) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    while extra_dims > 0:
        weights = weights.unsqueeze(-1)
        extra_dims -= 1
    total = (values * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=_VELOCITY_EPS)
    return total / count


def _fit_linear_velocity(
    velocity: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = valid.to(dtype=velocity.dtype)
    ones = torch.ones_like(times)
    s0 = weights.sum(dim=1)
    s1 = (weights * times).sum(dim=1)
    s2 = (weights * times.square()).sum(dim=1)
    sy = (weights.unsqueeze(-1) * velocity).sum(dim=1)
    sty = (weights.unsqueeze(-1) * times.unsqueeze(-1) * velocity).sum(dim=1)
    det = s0 * s2 - s1.square()
    enough = s0 >= 2.0
    invertible = enough & (det.abs() > _DET_EPS)
    v0_ls = (
        s2.unsqueeze(-1) * sy - s1.unsqueeze(-1) * sty
    ) / det.clamp(min=_DET_EPS).unsqueeze(-1)
    acceleration_ls = (
        s0.unsqueeze(-1) * sty - s1.unsqueeze(-1) * sy
    ) / det.clamp(min=_DET_EPS).unsqueeze(-1)
    v0_mean = _masked_mean(velocity, valid, extra_dims=1)
    v0 = torch.where(invertible.unsqueeze(-1), v0_ls, v0_mean)
    acceleration = torch.where(
        invertible.unsqueeze(-1),
        acceleration_ls,
        torch.zeros_like(acceleration_ls),
    )
    return v0, acceleration, invertible


def _heading_omega(
    velocity: torch.Tensor,
    times: torch.Tensor,
    valid: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    previous = velocity[:, :-1, :]
    current = velocity[:, 1:, :]
    dt = times[:, 1:] - times[:, :-1]
    pair_valid = (
        valid[:, :-1]
        & valid[:, 1:]
        & (dt.abs() > _VELOCITY_EPS)
        & (torch.linalg.vector_norm(previous, dim=-1) > _VELOCITY_EPS)
        & (torch.linalg.vector_norm(current, dim=-1) > _VELOCITY_EPS)
    )
    cross = previous[..., 0] * current[..., 1] - previous[..., 1] * current[..., 0]
    dot = (previous * current).sum(dim=-1)
    heading_rate = torch.atan2(cross, dot) / dt.clamp(min=_VELOCITY_EPS)
    omega = _masked_mean(heading_rate, pair_valid)
    has_pairs = pair_valid.to(dtype=velocity.dtype).sum(dim=1) > 0.0
    return torch.where(has_pairs, omega, fallback)


def _rotate_velocity(
    velocity: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    return torch.stack((cos * vx - sin * vy, sin * vx + cos * vy), dim=-1)


def _constant_turn_displacement(
    velocity: torch.Tensor,
    omega: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    perpendicular = torch.stack((-velocity[..., 1], velocity[..., 0]), dim=-1)
    angle = omega * time
    small = omega.abs() < _OMEGA_EPS
    safe_omega = torch.where(small, torch.ones_like(omega), omega)
    sin_over_omega = torch.where(small, time, torch.sin(angle) / safe_omega)
    one_minus_cos_over_omega = torch.where(
        small,
        0.5 * omega * time.square(),
        (1.0 - torch.cos(angle)) / safe_omega,
    )
    return (
        sin_over_omega.unsqueeze(-1) * velocity
        + one_minus_cos_over_omega.unsqueeze(-1) * perpendicular
    )


def _history_rmse(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    error = torch.linalg.vector_norm(predicted - observed, dim=-1)
    weights = valid.to(dtype=error.dtype)
    count = weights.sum(dim=1).clamp(min=_VELOCITY_EPS)
    return torch.sqrt((error.square() * weights).sum(dim=1) / count)


def rollout_image_plane_newton(
    features: torch.Tensor,
    interval_valid: torch.Tensor,
    acceleration_valid: torch.Tensor,
    *,
    action_interval_seconds: float,
    horizon_steps: int,
    window_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict future image-plane motion with CA, CV, and constant-turn models.

    Returns
    -------
    newton_features
        `[B, window_count, 12]` float32 descriptors.
    window_interval_valid
        `[B, window_count]` true when at least one history interval is valid.
    window_acceleration_valid
        `[B, window_count]` true when at least two history intervals are valid
        and a linear acceleration fit is identified.
    """

    if features.ndim != 3 or features.shape[-1] != MOTION_FEATURE_DIM:
        raise ValueError(
            "absolute_motion_features must be [B, K, "
            f"{MOTION_FEATURE_DIM}], got {tuple(features.shape)}"
        )
    batch_size, history_count, _ = features.shape
    expected_mask = (batch_size, history_count)
    if tuple(interval_valid.shape) != expected_mask:
        raise ValueError(
            f"interval_valid must be {expected_mask}, got {tuple(interval_valid.shape)}"
        )
    if tuple(acceleration_valid.shape) != expected_mask:
        raise ValueError(
            "acceleration_valid must be "
            f"{expected_mask}, got {tuple(acceleration_valid.shape)}"
        )
    if bool((acceleration_valid & ~interval_valid).any()):
        raise ValueError("acceleration cannot be valid for an invalid interval")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("absolute_motion_features contains non-finite values")

    features = features.to(dtype=torch.float32)
    interval_valid = interval_valid.to(dtype=torch.bool)
    acceleration_valid = acceleration_valid.to(dtype=torch.bool)
    device = features.device

    _, _, window_ends = _window_step_groups(
        horizon_steps=horizon_steps,
        window_count=window_count,
    )
    times = newton_step_times(
        action_interval_seconds=action_interval_seconds,
        horizon_steps=horizon_steps,
    ).to(device=device)
    t_end = times[window_ends.to(device=device) - 1]

    velocity = _object_scale_velocity(features)
    velocity = torch.where(interval_valid.unsqueeze(-1), velocity, torch.zeros_like(velocity))
    centers = _interval_center_times(features[:, :, _DELTA_T_INDEX])
    v0, acceleration, invertible = _fit_linear_velocity(
        velocity,
        centers,
        interval_valid,
    )
    v_cv = _masked_mean(velocity, interval_valid, extra_dims=1)
    v0_norm = torch.linalg.vector_norm(v0, dim=-1)
    fallback_omega = (v0[:, 0] * acceleration[:, 1] - v0[:, 1] * acceleration[:, 0]) / (
        v0_norm.square().clamp(min=_VELOCITY_EPS)
    )
    fallback_omega = torch.where(
        v0_norm > _VELOCITY_EPS,
        fallback_omega,
        torch.zeros_like(fallback_omega),
    )
    omega = _heading_omega(velocity, centers, interval_valid, fallback_omega)
    omega = torch.where(invertible, omega, torch.zeros_like(omega))

    has_interval = interval_valid.any(dim=1)
    has_acceleration = invertible
    history_indices = torch.arange(history_count, device=device).expand(batch_size, -1)
    last_index = torch.where(
        interval_valid,
        history_indices,
        torch.full_like(history_indices, -1),
    ).max(dim=1).values
    gather_index = last_index.clamp(min=0)
    batch_index = torch.arange(batch_size, device=device)
    last_velocity = velocity[batch_index, gather_index]
    last_center = centers[batch_index, gather_index]
    v0_ct = _rotate_velocity(last_velocity, omega * (-last_center))
    v0 = torch.where(has_interval.unsqueeze(-1), v0, torch.zeros_like(v0))
    v0_ct = torch.where(has_interval.unsqueeze(-1), v0_ct, torch.zeros_like(v0_ct))
    v_cv = torch.where(has_interval.unsqueeze(-1), v_cv, torch.zeros_like(v_cv))
    last_velocity = torch.where(
        has_interval.unsqueeze(-1),
        last_velocity,
        torch.zeros_like(last_velocity),
    )
    acceleration = torch.where(
        has_acceleration.unsqueeze(-1),
        acceleration,
        torch.zeros_like(acceleration),
    )
    omega = torch.where(has_acceleration, omega, torch.zeros_like(omega))
    v0_ct = torch.where(has_acceleration.unsqueeze(-1), v0_ct, last_velocity)

    t_end = t_end.view(1, window_count)
    v0_windows = v0.unsqueeze(1)
    v0_ct_windows = v0_ct.unsqueeze(1)
    v_cv_windows = v_cv.unsqueeze(1)
    acceleration_windows = acceleration.unsqueeze(1)
    omega_windows = omega.unsqueeze(1)
    time = t_end.expand(batch_size, -1)

    ca_end_displacement = (
        v0_windows * time.unsqueeze(-1)
        + 0.5 * acceleration_windows * time.square().unsqueeze(-1)
    )
    ct_end_displacement = _constant_turn_displacement(
        v0_ct_windows,
        omega_windows,
        time,
    )
    cv_end_displacement = v_cv_windows * time.unsqueeze(-1)

    ca_history = v0.unsqueeze(1) + acceleration.unsqueeze(1) * centers.unsqueeze(-1)
    ct_history = _rotate_velocity(
        v0_ct.unsqueeze(1),
        omega.unsqueeze(1) * centers,
    )
    cv_history = v_cv.unsqueeze(1).expand_as(velocity)
    ca_rmse = _history_rmse(ca_history, velocity, interval_valid)
    ct_rmse = _history_rmse(ct_history, velocity, interval_valid)
    cv_rmse = _history_rmse(cv_history, velocity, interval_valid)
    ca_rmse = torch.where(has_acceleration, ca_rmse, torch.zeros_like(ca_rmse))
    ct_rmse = torch.where(has_acceleration, ct_rmse, torch.zeros_like(ct_rmse))
    cv_rmse = torch.where(has_interval, cv_rmse, torch.zeros_like(cv_rmse))

    speed = torch.linalg.vector_norm(last_velocity, dim=-1)
    accel_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    direction = v0 / v0_norm.clamp(min=_VELOCITY_EPS).unsqueeze(-1)
    tangential = (acceleration * direction).sum(dim=-1).abs() / (
        accel_norm + _VELOCITY_EPS
    )
    tangential = torch.where(has_acceleration, tangential, torch.zeros_like(tangential))
    omega_horizon = omega.unsqueeze(1) * time

    newton = torch.zeros(
        (batch_size, window_count, NEWTON_FEATURE_DIM),
        dtype=torch.float32,
        device=device,
    )
    newton[:, :, 0:2] = ca_end_displacement
    newton[:, :, 2:4] = ct_end_displacement
    newton[:, :, 4:6] = cv_end_displacement
    newton[:, :, 6] = ca_rmse.unsqueeze(1)
    newton[:, :, 7] = ct_rmse.unsqueeze(1)
    newton[:, :, 8] = cv_rmse.unsqueeze(1)
    newton[:, :, 9] = omega_horizon
    newton[:, :, 10] = tangential.unsqueeze(1)
    newton[:, :, 11] = speed.unsqueeze(1)

    window_interval_valid = has_interval[:, None].expand(-1, window_count).clone()
    window_acceleration_valid = has_acceleration[:, None].expand(-1, window_count).clone()
    newton = torch.where(
        window_interval_valid[..., None],
        newton,
        torch.zeros_like(newton),
    )
    feature_valid = window_interval_valid[..., None].expand_as(newton).clone()
    feature_valid[..., list(NEWTON_ACCELERATION_FEATURE_INDICES)] = (
        window_acceleration_valid[..., None]
    )
    newton = torch.where(feature_valid, newton, torch.zeros_like(newton))
    return newton, window_interval_valid, window_acceleration_valid


def validate_newton_statistics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("newton statistics must be a mapping")
    expected = {
        "schema_version",
        "feature_names",
        "count",
        "mean",
        "standard_deviation",
        "scale",
        "minimum_scale",
        "temporal_contract",
        "source",
        "window_count",
        "horizon_steps",
        "action_interval_seconds",
    }
    if set(payload) != expected:
        raise ValueError(
            "newton statistics keys differ from the contract: "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    if payload["schema_version"] != NEWTON_STATISTICS_VERSION:
        raise ValueError(
            f"newton statistics schema_version must be {NEWTON_STATISTICS_VERSION}"
        )
    if tuple(payload["feature_names"]) != NEWTON_FEATURE_NAMES:
        raise ValueError("newton statistics feature order differs from the model")
    if (
        payload["temporal_contract"] != NEWTON_TEMPORAL_CONTRACT
        or payload["source"] != NEWTON_SOURCE
    ):
        raise ValueError("newton statistics contract is not image-plane rollout v2")
    arrays = {}
    for key in ("count", "mean", "standard_deviation", "scale"):
        array = np.asarray(payload[key])
        if array.shape != (NEWTON_FEATURE_DIM,):
            raise ValueError(f"newton statistics {key} must have 12 values")
        arrays[key] = array
    if np.any(arrays["count"].astype(np.int64) <= 0):
        raise ValueError("every newton feature requires at least one valid sample")
    for key in ("mean", "standard_deviation", "scale"):
        array = arrays[key].astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"newton statistics {key} contains non-finite values")
    minimum_scale = float(payload["minimum_scale"])
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("newton statistics minimum_scale must be positive")
    if np.any(arrays["scale"].astype(np.float64) < minimum_scale):
        raise ValueError("newton feature scales fall below minimum_scale")
    window_count = int(payload["window_count"])
    horizon_steps = int(payload["horizon_steps"])
    action_interval_seconds = float(payload["action_interval_seconds"])
    if window_count <= 0 or horizon_steps <= 0:
        raise ValueError("newton window_count and horizon_steps must be positive")
    if horizon_steps % window_count != 0:
        raise ValueError("newton horizon_steps must be divisible by window_count")
    if not np.isfinite(action_interval_seconds) or action_interval_seconds <= 0.0:
        raise ValueError("newton action_interval_seconds must be positive")
    normalized = dict(payload)
    normalized["window_count"] = window_count
    normalized["horizon_steps"] = horizon_steps
    normalized["action_interval_seconds"] = action_interval_seconds
    normalized["feature_names"] = list(NEWTON_FEATURE_NAMES)
    return normalized


def validate_checkpoint_newton_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint explicit-dynamics metadata must be a mapping")
    if set(payload) != CHECKPOINT_NEWTON_METADATA_KEYS:
        raise ValueError(
            "checkpoint explicit-dynamics metadata keys differ from v2: "
            f"missing={sorted(CHECKPOINT_NEWTON_METADATA_KEYS - set(payload))}, "
            f"unknown={sorted(set(payload) - CHECKPOINT_NEWTON_METADATA_KEYS)}"
        )
    window_count = int(payload["window_count"])
    horizon_steps = int(payload["horizon_steps"])
    action_interval_seconds = float(payload["action_interval_seconds"])
    if window_count <= 0 or horizon_steps <= 0:
        raise ValueError("checkpoint newton window_count/horizon_steps must be positive")
    if horizon_steps % window_count != 0:
        raise ValueError("checkpoint newton horizon_steps must divide into windows")
    if not np.isfinite(action_interval_seconds) or action_interval_seconds <= 0.0:
        raise ValueError("checkpoint newton action_interval_seconds must be positive")
    if tuple(payload["feature_names"]) != NEWTON_FEATURE_NAMES:
        raise ValueError("checkpoint newton feature order differs from v2")
    if (
        payload["temporal_contract"] != NEWTON_TEMPORAL_CONTRACT
        or payload["source"] != NEWTON_SOURCE
    ):
        raise ValueError("checkpoint newton contract is not image-plane rollout v2")
    mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    if (
        mean.shape != (NEWTON_FEATURE_DIM,)
        or scale.shape != (NEWTON_FEATURE_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("checkpoint newton normalization is invalid")
    digest = payload["statistics_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("checkpoint newton statistics_sha256 is invalid")
    return {
        "window_count": window_count,
        "horizon_steps": horizon_steps,
        "action_interval_seconds": action_interval_seconds,
        "feature_names": list(NEWTON_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "temporal_contract": NEWTON_TEMPORAL_CONTRACT,
        "source": NEWTON_SOURCE,
        "statistics_sha256": str(digest),
    }


def build_checkpoint_newton_metadata(
    *,
    statistics: Any,
    statistics_sha256: str,
) -> dict[str, Any]:
    statistics = validate_newton_statistics(statistics)
    metadata = {
        "window_count": int(statistics["window_count"]),
        "horizon_steps": int(statistics["horizon_steps"]),
        "action_interval_seconds": float(statistics["action_interval_seconds"]),
        "feature_names": list(statistics["feature_names"]),
        "feature_mean": [float(value) for value in statistics["mean"]],
        "feature_scale": [float(value) for value in statistics["scale"]],
        "temporal_contract": statistics["temporal_contract"],
        "source": statistics["source"],
        "statistics_sha256": str(statistics_sha256),
    }
    return validate_checkpoint_newton_metadata(metadata)


def load_newton_statistics(root: str | Path) -> dict[str, Any]:
    path = Path(root) / NEWTON_STATS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"newton statistics are missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return validate_newton_statistics(payload)


def load_newton_checkpoint_metadata(root: str | Path) -> dict[str, Any]:
    statistics = load_newton_statistics(root)
    return build_checkpoint_newton_metadata(
        statistics=statistics,
        statistics_sha256=hash_json_payload(statistics),
    )
