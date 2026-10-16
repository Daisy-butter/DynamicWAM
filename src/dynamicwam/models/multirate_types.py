"""Tensor contracts for the causal bridge and the fast action path.

These objects carry tensors and identity fields only. They do not own threads,
CUDA events, or simulator handles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "future_latent",
        "action_sequence",
        "video_frames",
        "clean_future_latent",
        "expert_actions",
    }
)


@dataclass
class SourceObservation:
    """Observations available at the slow source time. No future labels."""

    condition_latent: torch.Tensor
    state: torch.Tensor
    motion: torch.Tensor
    motion_interval_valid_mask: torch.Tensor
    motion_acceleration_valid_mask: torch.Tensor
    text_embeddings: list[torch.Tensor]
    source_id: int | torch.Tensor
    sim_time: float | torch.Tensor
    # Causal source image snapshot for innovation features; never future labels.
    rgb: torch.Tensor | None = None
    rgb_valid_mask: torch.Tensor | None = None
    action_target_offsets: torch.Tensor | None = None
    committed_prefix: torch.Tensor | None = None
    committed_prefix_lengths: torch.Tensor | None = None

    def as_model_batch(self) -> dict[str, object]:
        batch = {
            "condition_latent": self.condition_latent,
            "initial_state": self.state,
            "absolute_motion_features": self.motion,
            "absolute_motion_interval_valid_mask": self.motion_interval_valid_mask,
            "absolute_motion_acceleration_valid_mask": (
                self.motion_acceleration_valid_mask
            ),
            "text_embeddings": self.text_embeddings,
        }
        # Explicit source marker distinguishes the native slow/source-FM path
        # from fast calls, which already carry physical time via the interface.
        batch["source_query"] = True
        if self.action_target_offsets is not None:
            batch["source_action_target_offsets"] = self.action_target_offsets
        if self.committed_prefix is not None or self.committed_prefix_lengths is not None:
            if self.committed_prefix is None or self.committed_prefix_lengths is None:
                raise ValueError("source committed prefix requires actions and lengths together")
            from dynamicwam.models.rtc_prefix import PrefixCondition
            batch["rtc_prefix"] = PrefixCondition(self.committed_prefix, self.committed_prefix_lengths)
        return batch


@dataclass
class SourceNoise:
    future_latent: torch.Tensor
    actions: torch.Tensor


@dataclass
class BridgeTensorBundle:
    """Layer-wise video K/V from the second joint forward."""

    video_k: tuple[torch.Tensor, ...]
    video_v: tuple[torch.Tensor, ...]
    seq_lens: torch.Tensor
    grid_sizes: dict[str, torch.Tensor | int]
    source_id: int | torch.Tensor
    sim_time: float | torch.Tensor
    null_bridge: bool | torch.Tensor = False
    # Host-side flag only. Never derive this from a CUDA reduction: publishing
    # the bridge must not sync the producer stream. The pack path is correct for
    # both padded and unpadded rows, so the default stays False.
    all_video_tokens_valid: bool = False
    # Version 2 metadata describes the conditions that actually produced K/V.
    # source_action_plan is a clean estimate from the cache-producing forward,
    # not the expert target or a later fast plan. Sigma exposes its uncertainty.
    interface_version: int = 1
    source_action_plan: torch.Tensor | None = None
    source_action_sigma: torch.Tensor | float | None = None
    # Separate final proposal from the SAME native K10 trajectory. It is not
    # the action condition that produced video K/V. Explicit stop-gradient.
    source_completed_plan: torch.Tensor | None = None
    source_action_target_offsets: torch.Tensor | None = None
    source_motion: torch.Tensor | None = None
    source_motion_interval_valid_mask: torch.Tensor | None = None
    source_motion_acceleration_valid_mask: torch.Tensor | None = None
    source_rgb: torch.Tensor | None = None
    source_rgb_valid_mask: torch.Tensor | None = None
    # Explicit inference-only derived source cache. Rebuild on every source
    # install or weight change; training must always encode attached source RGB.
    source_visual_tokens: torch.Tensor | None = None
    source_visual_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.video_k) != len(self.video_v):
            raise ValueError("video K/V layer counts differ")
        dropped = (
            False
            if torch.is_tensor(self.null_bridge)
            else bool(self.null_bridge)
        )
        if not dropped and not self.video_k:
            raise ValueError("a present bridge needs one K/V pair per layer")
        self.all_video_tokens_valid = bool(self.all_video_tokens_valid)

    def as_video_cache(self) -> dict[str, object]:
        return {
            "video_k": list(self.video_k),
            "video_v": list(self.video_v),
            "seq_lens": self.seq_lens,
            "grid_sizes": self.grid_sizes,
            "all_video_tokens_valid": self.all_video_tokens_valid,
        }


@dataclass
class BridgeAge:
    delta_sim_seconds: float
    delta_actions: int
    bridge_valid: bool
    state_delta: torch.Tensor

    def features(self, *, sim_scale: float, action_scale: float) -> torch.Tensor:
        if sim_scale <= 0.0 or action_scale <= 0.0:
            raise ValueError("age scales must be positive")
        delta = self.state_delta
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
            squeeze = True
        elif delta.ndim == 2:
            squeeze = False
        else:
            raise ValueError(
                f"state_delta must be [D] or [B, D], got {tuple(delta.shape)}"
            )
        batch = int(delta.shape[0])

        def column(value: float | torch.Tensor) -> torch.Tensor:
            if torch.is_tensor(value):
                tensor = value.to(device=delta.device, dtype=delta.dtype).reshape(-1)
                if tensor.numel() == 1:
                    tensor = tensor.expand(batch)
                if int(tensor.numel()) != batch:
                    raise ValueError(
                        f"age field has {int(tensor.numel())} rows for a batch of {batch}"
                    )
                return tensor.unsqueeze(1)
            return delta.new_full((batch, 1), float(value))

        if torch.is_tensor(self.bridge_valid):
            valid = column(self.bridge_valid.to(dtype=delta.dtype))
        else:
            valid = column(1.0 if self.bridge_valid else 0.0)
        header = torch.cat(
            (
                column(self.delta_sim_seconds) / float(sim_scale),
                column(self.delta_actions) / float(action_scale),
                valid,
            ),
            dim=1,
        )
        features = torch.cat((header, delta.to(dtype=header.dtype)), dim=1)
        return features.squeeze(0) if squeeze else features


@dataclass
class CurrentObservation:
    """Fast-path observation at time t. RGB is view-major ``[B,3,3,H,W]``."""

    rgb: torch.Tensor
    state: torch.Tensor
    motion: torch.Tensor
    motion_interval_valid_mask: torch.Tensor
    motion_acceleration_valid_mask: torch.Tensor
    text_embeddings: list[torch.Tensor]
    age: BridgeAge
    rgb_valid_mask: torch.Tensor | None = None
    # Only actions already committed before this observation; suffix rows must
    # be ignored. They use the identical normalized action space as Action DiT.
    committed_prefix: torch.Tensor | None = None
    committed_prefix_lengths: torch.Tensor | None = None
    observation_delay_seconds: float | torch.Tensor = 0.0
    previous_plan: torch.Tensor | None = None
    previous_plan_valid: torch.Tensor | None = None
    # Planned physical effect times, relative to this observation. Available
    # queue metadata, not future observations; required by the V2 interface.
    action_target_offsets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.rgb.ndim != 5 or int(self.rgb.shape[1]) != 3:
            raise ValueError(
                "current RGB must be [B, views=3, C, H, W], "
                f"got {tuple(self.rgb.shape)}"
            )


def pad_text_embeddings(
    embeddings: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length text rows to ``[B, T, C]`` with a true-valid mask."""

    if not embeddings:
        raise ValueError("text embeddings are empty")
    width = int(embeddings[0].shape[-1])
    length = max(int(item.shape[0]) for item in embeddings)
    text = embeddings[0].new_zeros(len(embeddings), length, width)
    mask = torch.zeros(
        len(embeddings),
        length,
        dtype=torch.bool,
        device=embeddings[0].device,
    )
    for index, item in enumerate(embeddings):
        if item.ndim != 2 or int(item.shape[-1]) != width:
            raise ValueError(
                f"text embedding {index} has shape {tuple(item.shape)}, expected [T, {width}]"
            )
        text[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return text, mask


def reject_leaky_source_fields(payload: dict[str, object]) -> None:
    present = FORBIDDEN_SOURCE_FIELDS.intersection(payload)
    if present:
        raise ValueError(
            "causal bridge source received future labels: " + ", ".join(sorted(present))
        )
