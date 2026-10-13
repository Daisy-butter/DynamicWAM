"""Training-time RTC primitives for DynamicWAM's clean-at-zero flow convention.

The three tensor functions have no data-dependent Python branches or host sync.
Call ``validate_prefix_condition`` before entering a compiled/captured region;
resample lengths within its validated range without changing tensor shapes.
The timestamp queue helper is an OUTSIDE-GRAPH deployment contract, not an
asynchronous executor. It must be wired to the actual controller before RTC eval.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class PrefixCondition:
    actions: Tensor  # [B, H, D], clean NORMALIZED actions; padding after d ignored.
    lengths: Tensor  # [B], integral; 0 <= d < H.


def validate_prefix_condition(
    condition: PrefixCondition,
    *,
    max_prefix: int | None = None,
    reference: Tensor | None = None,
) -> None:
    """Validate outside CUDA Graph/compile; this deliberately synchronizes.

    ``max_prefix`` is the trained/deployed delay limit, not a clipping threshold.
    Full-prefix samples are rejected: every sample must supervise a suffix.
    """
    actions, lengths = condition.actions, condition.lengths
    if actions.ndim != 3 or min(actions.shape) < 1:
        raise ValueError("prefix actions must have nonempty shape [B,H,D]")
    if not actions.is_floating_point():
        raise TypeError("prefix actions must be floating point")
    if lengths.shape != (actions.shape[0],):
        raise ValueError("prefix lengths must have shape [B]")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("prefix lengths must be int32 or int64")
    if actions.device != lengths.device:
        raise ValueError("prefix actions and lengths must share a device")
    horizon = actions.shape[1]
    limit = horizon - 1 if max_prefix is None else max_prefix
    if not isinstance(limit, int) or not 0 <= limit < horizon:
        raise ValueError("max_prefix must satisfy 0 <= max_prefix < horizon")
    values = lengths.detach().cpu().tolist()
    if any(length < 0 or length > limit for length in values):
        raise ValueError(f"prefix length outside [0,{limit}]; no truncation is allowed")
    mask = prefix_mask(condition).unsqueeze(-1).expand_as(actions)
    if not bool(torch.isfinite(actions.masked_select(mask)).all()):
        raise ValueError("committed prefix contains nonfinite actions")
    if reference is not None and (
        reference.shape != actions.shape
        or reference.device != actions.device
        or reference.dtype != actions.dtype
    ):
        raise ValueError("reference and prefix actions must match shape/device/dtype")


def prefix_mask(condition: PrefixCondition) -> Tensor:
    """Return [B,H] mask with only tensor operations."""
    return torch.arange(condition.actions.shape[1], device=condition.actions.device)[None, :] < condition.lengths[:, None]


def apply_prefix(noisy: Tensor, condition: PrefixCondition) -> Tensor:
    """Overwrite committed action positions; suffix and its gradient survive."""
    return torch.where(prefix_mask(condition).unsqueeze(-1), condition.actions, noisy)


def action_token_times(action_t: Tensor, condition: PrefixCondition, total_tokens: int) -> Tensor:
    """[B] -> [B,total_tokens] for state + H actions + 4 motion + registers.

    Prefix action tokens receive t=0 (clean); all other tokens retain action_t.
    In particular d=0 exactly broadcasts the native common timestep. This does
    not erase or retime the four native dynamics tokens.
    """
    horizon = condition.actions.shape[1]
    if action_t.ndim != 1 or action_t.shape[0] != condition.actions.shape[0]:
        raise ValueError("action_t must have shape [B]")
    if total_tokens < 1 + horizon + 4:
        raise ValueError("token layout requires state + H actions + four motion tokens")
    indices = torch.arange(total_tokens, device=action_t.device)[None, :]
    clean = (indices >= 1) & (indices < 1 + condition.lengths[:, None]) & (indices <= horizon)
    return torch.where(clean, torch.zeros_like(action_t[:, None]), action_t[:, None])


def masked_suffix_mse(pred: Tensor, target: Tensor, condition: PrefixCondition) -> Tensor:
    """Mean per-example suffix MSE, then batch mean (not token-weighted batch).

    The caller MUST validate lengths first. No clamp silently accepts d >= H;
    the independent validator rejects it before graph capture/execution.
    """
    mask = ~prefix_mask(condition)
    # where also excludes nonfinite/irrelevant prefix targets from forward loss.
    error = torch.where(mask.unsqueeze(-1), pred.float() - target.float(), 0.0)
    per_sample = error.square().sum(dim=(1, 2)) / (mask.sum(dim=1) * pred.shape[-1])
    return per_sample.mean()


@dataclass(frozen=True)
class CommittedQueuePrefix:
    condition: PrefixCondition  # batch one, padded to requested H.
    prefix_effective_times: Tensor  # [d], actual absolute controller timestamps.
    anchor_time: float  # start of current action; may precede query_time.
    suffix_not_before: float  # exclusive end of the last irrevocable action.
    estimated_ready_time: float


def committed_prefix_from_queue(
    actions: Tensor,
    effective_times: Tensor,
    *,
    query_time: float,
    estimated_ready_time: float,
    queue_end_time: float,
    horizon: int,
    max_prefix: int,
) -> CommittedQueuePrefix:
    """Freeze a timestamped queue through estimated completion, outside graph.

    actions [N,D] are already normalized. effective_times [N] mark command START
    boundaries; queue_end_time explicitly ends the last command. Intervals are
    [start,next_start). Include the currently executing command in the input.
    It remains irrevocable even for zero predicted latency. Future commands
    starting strictly before readiness are also frozen. A command exactly at
    readiness can be replaced if it has not begun at query time. No uniform
    action period, physics tick, inferred last duration, or resampling is used.

    Caller must align the new model chunk to returned anchor_time, retain exact
    prefix timestamps, and commit suffix only at suffix_not_before or later.
    If inference finishes later than estimated, recompute/reject its handoff;
    this reservation alone cannot guarantee that future result is still timely.
    """
    if actions.ndim != 2 or min(actions.shape) < 1 or not actions.is_floating_point():
        raise ValueError("queue actions must be nonempty floating [N,D]")
    if effective_times.ndim != 1 or effective_times.shape[0] != actions.shape[0]:
        raise ValueError("effective_times must have shape [N]")
    if not effective_times.is_floating_point():
        raise TypeError("effective_times must be floating absolute timestamps")
    times = effective_times.detach().to(device="cpu", dtype=torch.float64).tolist()
    scalars = (query_time, estimated_ready_time, queue_end_time)
    if not all(math.isfinite(float(t)) for t in times + list(scalars)):
        raise ValueError("queue timestamps must be finite")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("queue effective times must be strictly increasing")
    if queue_end_time <= times[-1]:
        raise ValueError("queue_end_time must end the final action interval")
    if not times[0] <= query_time < queue_end_time:
        raise ValueError("queue does not cover current execution time")
    if estimated_ready_time < query_time:
        raise ValueError("estimated ready time precedes observation/query time")
    if estimated_ready_time > queue_end_time:
        raise ValueError("queue underrun: inference readiness exceeds queue coverage")
    if not isinstance(horizon, int) or horizon < 1 or not isinstance(max_prefix, int) or not 0 <= max_prefix < horizon:
        raise ValueError("require 0 <= max_prefix < horizon")
    first = bisect_right(times, query_time) - 1
    stop = max(first + 1, bisect_left(times, estimated_ready_time))
    length = stop - first
    if length > max_prefix or length >= horizon:
        raise ValueError(f"committed prefix {length} exceeds trained limit {max_prefix}; refusing truncation")
    padded = actions.new_zeros((1, horizon, actions.shape[-1]))
    padded[0, :length] = actions[first:stop]
    condition = PrefixCondition(padded, torch.tensor([length], dtype=torch.long, device=actions.device))
    validate_prefix_condition(condition, max_prefix=max_prefix)
    return CommittedQueuePrefix(
        condition=condition,
        prefix_effective_times=effective_times[first:stop].clone(),
        anchor_time=times[first],
        suffix_not_before=times[stop] if stop < len(times) else float(queue_end_time),
        estimated_ready_time=float(estimated_ready_time),
    )


def validate_suffix_handoff(
    reservation: CommittedQueuePrefix,
    *,
    actual_ready_time: float,
    suffix_effective_time: float,
) -> None:
    """Controller-side check; a late handoff must be rebuilt, never backdated."""
    if not math.isfinite(actual_ready_time) or not math.isfinite(suffix_effective_time):
        raise ValueError("handoff timestamps must be finite")
    if suffix_effective_time < reservation.suffix_not_before:
        raise ValueError("suffix would overwrite a committed/executing action")
    if actual_ready_time > reservation.suffix_not_before:
        raise ValueError("late result invalidates queue reservation; rebuild prefix")
    if suffix_effective_time < actual_ready_time:
        raise ValueError("suffix cannot become effective before inference is ready")
