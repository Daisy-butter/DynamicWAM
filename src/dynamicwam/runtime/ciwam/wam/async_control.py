"""Sidecar async-control config and the fast/slow replan scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .rtc import boundary_delta, leftover_prefix, stitch_chunk
from .world_buffer import WorldBridge

ASYNC_POLICY_REGISTRATION_NAME = "ciwam.adapters.domino.deploy_policy_async_flow"
DEFAULT_ASYNC_CONFIG_NAME = "absolute_motion_v2_async.yml"


@dataclass(frozen=True)
class AsyncControlConfig:
    fast_horizon: int = 4
    max_world_age_steps: int = 4
    inference_delay: int = 0
    rtc_mode: str = "hard_prefix"
    max_guidance_weight: float = 10.0
    motion_refresh_ratio: float = 0.5

    def __post_init__(self) -> None:
        for name in ("fast_horizon", "max_world_age_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.inference_delay < 0:
            raise ValueError(
                f"inference_delay must be >= 0, got {self.inference_delay}"
            )
        if self.inference_delay > self.fast_horizon:
            raise ValueError(
                "inference_delay must be <= fast_horizon: "
                f"{self.inference_delay} > {self.fast_horizon}"
            )
        if self.rtc_mode != "hard_prefix":
            raise ValueError(
                f"only rtc_mode='hard_prefix' is supported, got {self.rtc_mode!r}"
            )
        if not np.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0.0:
            raise ValueError("max_guidance_weight must be a positive finite number")
        if not np.isfinite(self.motion_refresh_ratio) or self.motion_refresh_ratio < 0.0:
            raise ValueError("motion_refresh_ratio must be a non-negative finite number")


def default_async_config() -> AsyncControlConfig:
    return AsyncControlConfig()


def load_async_control_config(source: str | Path | dict[str, Any] | None) -> AsyncControlConfig:
    """Load sidecar async settings. Official DynamicWAM profiles are not used."""
    if source is None:
        return default_async_config()
    if isinstance(source, dict):
        payload = dict(source)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"async control config not found: {path}")
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("async control config requires PyYAML") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"async control config must be a mapping: {path}")
        payload = loaded
    allowed = {
        "fast_horizon",
        "max_world_age_steps",
        "inference_delay",
        "rtc_mode",
        "max_guidance_weight",
        "motion_refresh_ratio",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown async control keys: {unknown}")
    defaults = default_async_config()
    return AsyncControlConfig(
        fast_horizon=int(payload.get("fast_horizon", defaults.fast_horizon)),
        max_world_age_steps=int(
            payload.get("max_world_age_steps", defaults.max_world_age_steps)
        ),
        inference_delay=int(payload.get("inference_delay", defaults.inference_delay)),
        rtc_mode=str(payload.get("rtc_mode", defaults.rtc_mode)),
        max_guidance_weight=float(
            payload.get("max_guidance_weight", defaults.max_guidance_weight)
        ),
        motion_refresh_ratio=float(
            payload.get("motion_refresh_ratio", defaults.motion_refresh_ratio)
        ),
    )


def resolve_async_config_path(
    usr_args: dict[str, Any],
    *,
    project_root: str,
) -> Path | None:
    raw = usr_args.get("dynamicwam_async_config")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = Path(project_root).expanduser() / path
        return path.resolve()
    candidate = Path(project_root).expanduser() / "configs" / DEFAULT_ASYNC_CONFIG_NAME
    if candidate.is_file():
        return candidate.resolve()
    return None


@dataclass
class FastSlowScheduler:
    """Decide when to replan and whether the slow world path must run."""

    config: AsyncControlConfig
    chunk_size: int
    bridge: WorldBridge = field(default_factory=WorldBridge)
    pending: deque[np.ndarray] = field(default_factory=deque)
    steps_since_replan: int = 0

    def __post_init__(self) -> None:
        if self.chunk_size < 2:
            raise ValueError(f"chunk_size must be >= 2, got {self.chunk_size}")

    def reset(self) -> None:
        self.bridge.reset()
        self.pending.clear()
        self.steps_since_replan = 0

    def needs_replan(self) -> bool:
        return (not self.pending) or self.steps_since_replan >= self.config.fast_horizon

    def choose_path(self, motion_features: np.ndarray | None = None) -> Literal["slow", "fast"]:
        if self.bridge.stale(self.config.max_world_age_steps):
            return "slow"
        if motion_features is not None and self.bridge.motion_drifted(
            motion_features,
            ratio=self.config.motion_refresh_ratio,
        ):
            return "slow"
        return "fast"

    def leftover(self) -> np.ndarray | None:
        return leftover_prefix(list(self.pending), self.config.inference_delay)

    def commit_chunk(
        self,
        actions: np.ndarray,
        *,
        path: Literal["slow", "fast"],
        leftover: np.ndarray | None,
        video_cache: dict[str, Any] | None = None,
        simulator_time: float | None = None,
        motion_features: np.ndarray | None = None,
        motion_interval_valid_mask: np.ndarray | None = None,
        motion_acceleration_valid_mask: np.ndarray | None = None,
    ) -> float:
        if actions.ndim != 2 or int(actions.shape[0]) != self.chunk_size:
            raise RuntimeError(
                "absolute-motion policy returned the wrong action chunk: "
                f"expected ({self.chunk_size}, A), got {actions.shape}"
            )
        stitched = stitch_chunk(actions, leftover)
        delta = boundary_delta(stitched, leftover)
        self.pending = deque(stitched)
        self.steps_since_replan = 0
        if path == "slow":
            if video_cache is None:
                raise RuntimeError("slow path must write a video cache into WorldBridge")
            self.bridge.write(
                video_cache,
                simulator_time=simulator_time,
                motion_features=motion_features,
                motion_interval_valid_mask=motion_interval_valid_mask,
                motion_acceleration_valid_mask=motion_acceleration_valid_mask,
            )
        return delta

    def pop_action(self) -> np.ndarray:
        if not self.pending:
            raise RuntimeError("no pending action after the fast-slow replan")
        action = self.pending.popleft()
        self.steps_since_replan += 1
        self.bridge.tick()
        return action
