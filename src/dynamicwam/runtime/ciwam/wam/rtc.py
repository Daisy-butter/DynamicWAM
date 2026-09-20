"""Inference-time hard-prefix Real-Time Chunking for flow-matching chunks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def leftover_prefix(
    pending_actions: list[np.ndarray] | tuple[np.ndarray, ...],
    inference_delay: int,
) -> np.ndarray | None:
    """Return the qpos prefix that must stay frozen across a replan."""
    if inference_delay < 0:
        raise ValueError(f"inference_delay must be >= 0, got {inference_delay}")
    if not pending_actions or inference_delay == 0:
        return None
    prefix_len = min(int(inference_delay), len(pending_actions))
    prefix = np.stack(
        [np.asarray(action, dtype=np.float32) for action in pending_actions[:prefix_len]],
        axis=0,
    )
    if prefix.ndim != 2:
        raise ValueError(f"leftover prefix must be [P, A], got {prefix.shape}")
    return prefix


def stitch_chunk(actions: np.ndarray, leftover: np.ndarray | None) -> np.ndarray:
    """Overwrite the generated chunk prefix with the executed leftover qpos."""
    chunk = np.asarray(actions, dtype=np.float32)
    if leftover is None or leftover.size == 0:
        return chunk
    prefix = np.asarray(leftover, dtype=np.float32)
    if prefix.ndim != 2:
        raise ValueError(f"leftover must be [P, A], got {prefix.shape}")
    if chunk.ndim != 2:
        raise ValueError(f"actions must be [T, A], got {chunk.shape}")
    if prefix.shape[1] != chunk.shape[1]:
        raise ValueError(
            "leftover action dim differs from the generated chunk: "
            f"{prefix.shape} vs {chunk.shape}"
        )
    if prefix.shape[0] > chunk.shape[0]:
        raise ValueError(
            "leftover is longer than the generated chunk: "
            f"{prefix.shape[0]} > {chunk.shape[0]}"
        )
    stitched = chunk.copy()
    stitched[: prefix.shape[0]] = prefix
    return stitched


def boundary_delta(chunk: np.ndarray, leftover: np.ndarray | None) -> float:
    """Max |Δq| between the last frozen action and the first free action."""
    if leftover is None or leftover.size == 0:
        return 0.0
    prefix_len = int(np.asarray(leftover).shape[0])
    if prefix_len >= int(chunk.shape[0]):
        return 0.0
    previous = np.asarray(chunk[prefix_len - 1], dtype=np.float64)
    nxt = np.asarray(chunk[prefix_len], dtype=np.float64)
    return float(np.max(np.abs(nxt - previous)))


def _timestep_index(scheduler: Any, timestep: torch.Tensor | float) -> int:
    if isinstance(timestep, torch.Tensor):
        value = timestep.detach().float().cpu()
        if value.ndim:
            value = value.reshape(-1)[0]
    else:
        value = torch.tensor(float(timestep))
    return int(torch.argmin((scheduler.timesteps - value).abs()).item())


def next_scheduler_timestep(scheduler: Any, timestep: torch.Tensor | float) -> torch.Tensor | None:
    """Return the sigma timestep after ``scheduler.step`` at ``timestep``."""
    index = _timestep_index(scheduler, timestep)
    if index + 1 >= len(scheduler.timesteps):
        return None
    return scheduler.timesteps[index + 1]


def apply_hard_prefix(
    noisy_actions: torch.Tensor,
    prefix: torch.Tensor,
    prefix_noise: torch.Tensor,
    scheduler: Any,
    timestep: torch.Tensor | float,
) -> torch.Tensor:
    """Re-inject the known leftover at the current flow-matching noise level."""
    if noisy_actions.ndim != 3:
        raise ValueError(
            f"noisy_actions must be [B, T, A], got {tuple(noisy_actions.shape)}"
        )
    if prefix.ndim != 3 or prefix_noise.ndim != 3:
        raise ValueError("prefix and prefix_noise must be [B, P, A]")
    prefix_len = int(prefix.shape[1])
    if prefix_len == 0:
        return noisy_actions
    if prefix.shape[0] != noisy_actions.shape[0]:
        raise ValueError("prefix batch size must match noisy_actions")
    if prefix_len > int(noisy_actions.shape[1]):
        raise ValueError("prefix is longer than the noisy action chunk")
    next_time = next_scheduler_timestep(scheduler, timestep)
    if next_time is None:
        noisy_actions[:, :prefix_len] = prefix.to(
            device=noisy_actions.device,
            dtype=noisy_actions.dtype,
        )
        return noisy_actions
    noised = scheduler.add_noise(
        prefix.to(device=noisy_actions.device, dtype=noisy_actions.dtype),
        prefix_noise.to(device=noisy_actions.device, dtype=noisy_actions.dtype),
        next_time,
    )
    noisy_actions[:, :prefix_len] = noised
    return noisy_actions
