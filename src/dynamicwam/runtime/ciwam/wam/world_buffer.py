"""Persistent slow-path world state shared across action-chunk replans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Latest-interval mean displacement magnitude in the 12-D kinematic descriptor.
_DISPLACEMENT_MAGNITUDE_INDEX = 2


@dataclass
class WorldBridge:
    """Store one aligned world snapshot from a slow WAM refresh.

    Video KV and kinematic tokens were trained as a pair. The fast path may
    reuse this snapshot and update proprioception only; it must not mix a
    stale video cache with freshly computed motion tokens.
    """

    video_cache: dict[str, Any] | None = None
    condition_latent: Any = None
    motion_features: np.ndarray | None = None
    motion_interval_valid_mask: np.ndarray | None = None
    motion_acceleration_valid_mask: np.ndarray | None = None
    simulator_time: float | None = None
    age_steps: int = 0
    _refresh_count: int = field(default=0, repr=False)

    def has_cache(self) -> bool:
        return self.video_cache is not None

    def stale(self, max_age_steps: int) -> bool:
        if max_age_steps <= 0:
            raise ValueError(f"max_age_steps must be positive, got {max_age_steps}")
        return (not self.has_cache()) or self.age_steps >= int(max_age_steps)

    def write(
        self,
        video_cache: dict[str, Any],
        *,
        simulator_time: float | None = None,
        condition_latent: Any = None,
        motion_features: np.ndarray | None = None,
        motion_interval_valid_mask: np.ndarray | None = None,
        motion_acceleration_valid_mask: np.ndarray | None = None,
    ) -> None:
        if not isinstance(video_cache, dict) or "video_k" not in video_cache:
            raise ValueError("video_cache must contain layer-wise video_k entries")
        keys = video_cache.get("video_k")
        values = video_cache.get("video_v")
        if not keys or not values or len(keys) != len(values):
            raise ValueError("video_cache is missing matching video_k / video_v layers")
        self.video_cache = video_cache
        self.condition_latent = condition_latent
        self.motion_features = (
            None if motion_features is None else np.asarray(motion_features).copy()
        )
        self.motion_interval_valid_mask = (
            None
            if motion_interval_valid_mask is None
            else np.asarray(motion_interval_valid_mask).copy()
        )
        self.motion_acceleration_valid_mask = (
            None
            if motion_acceleration_valid_mask is None
            else np.asarray(motion_acceleration_valid_mask).copy()
        )
        self.simulator_time = (
            None if simulator_time is None else float(simulator_time)
        )
        self.age_steps = 0
        self._refresh_count += 1

    def frozen_motion(self) -> dict[str, np.ndarray] | None:
        if (
            self.motion_features is None
            or self.motion_interval_valid_mask is None
            or self.motion_acceleration_valid_mask is None
        ):
            return None
        return {
            "motion_features": self.motion_features,
            "motion_interval_valid_mask": self.motion_interval_valid_mask,
            "motion_acceleration_valid_mask": self.motion_acceleration_valid_mask,
        }

    def motion_drifted(
        self,
        motion_features: np.ndarray,
        *,
        ratio: float,
        min_pixels: float = 2.0,
    ) -> bool:
        """True when the latest interval's displacement no longer matches the snapshot."""
        if ratio <= 0.0 or self.motion_features is None:
            return False
        current = np.asarray(motion_features, dtype=np.float64)
        stored = np.asarray(self.motion_features, dtype=np.float64)
        if current.ndim != 2 or stored.ndim != 2:
            return True
        if current.shape[1] <= _DISPLACEMENT_MAGNITUDE_INDEX:
            return True
        new_mag = float(current[-1, _DISPLACEMENT_MAGNITUDE_INDEX])
        old_mag = float(stored[-1, _DISPLACEMENT_MAGNITUDE_INDEX])
        delta = abs(new_mag - old_mag)
        return delta >= min_pixels and delta >= ratio * max(abs(old_mag), 1.0)

    def tick(self) -> None:
        if self.video_cache is not None:
            self.age_steps += 1

    def reset(self) -> None:
        self.video_cache = None
        self.condition_latent = None
        self.motion_features = None
        self.motion_interval_valid_mask = None
        self.motion_acceleration_valid_mask = None
        self.simulator_time = None
        self.age_steps = 0
        self._refresh_count = 0
