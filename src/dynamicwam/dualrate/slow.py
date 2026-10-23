"""Differentiable, source-only native 2+8 sampling and spatial world readout.

The internal native K/V only accelerates the remaining eight slow action steps.
It is never the inter-rate interface. The public reference contains the final
two-step predicted video latent, observed latent, final action plan and clocks.
No RGB decode/re-encode is required. Auxiliary VAE/text encoders remain outside.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SourceCondition:
    """Source-time conditions; motion remains raw for native normalization.

    This whitelist intentionally excludes future latent, future labels, clean
    suffix actions, current-query observations and supervision-forward caches.
    Text embeddings may be a padded tensor or the native list of T5 tensors.
    """
    condition_latent: Tensor
    text_embeddings: Tensor | list[Tensor]
    initial_state: Tensor
    absolute_motion_features: Tensor
    absolute_motion_interval_valid_mask: Tensor
    absolute_motion_acceleration_valid_mask: Tensor

    @classmethod
    def parse(cls, source: SourceCondition | Mapping) -> SourceCondition:
        if isinstance(source, cls):
            return source
        expected = {field.name for field in fields(cls)}
        if not isinstance(source, Mapping) or set(source) != expected:
            raise ValueError("source must contain exactly the SourceCondition whitelist")
        return cls(**source)

    def as_batch(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class DifferentiableVideoCacheMixin:
    """Exact native cache computation, omitting only its terminal detach calls."""
    differentiable_video_cache = True

    def _video_attention_kv_for_cache(self, video_tokens, video_modulation,
                                     layer_idx, grid_sizes, freqs):
        layer = self.compact_wan.video_model.wan_model.blocks[layer_idx]
        attn = layer.self_attn
        norm_video = layer.norm1(video_tokens).float() * (
            1 + video_modulation[1].squeeze(2)
        ) + video_modulation[0].squeeze(2)
        batch, seq_len = norm_video.shape[:2]
        with torch.autocast("cuda", dtype=self.compact_wan.video_model.precision,
                            enabled=norm_video.is_cuda):
            key = attn.norm_k(attn.k(norm_video.to(attn.k.weight.dtype))).view(
                batch, seq_len, attn.num_heads, attn.head_dim)
            value = attn.v(norm_video.to(attn.v.weight.dtype)).view(
                batch, seq_len, attn.num_heads, attn.head_dim)
            key = self.compact_wan.apply_multiscale_rope(key, grid_sizes, freqs)
        return key, value


@lru_cache(maxsize=None)
def differentiable_native_class(base_class: type[nn.Module]) -> type[nn.Module]:
    """Construct this subclass before strict-loading the full native checkpoint.

    State-dict keys are unchanged. Save state dictionaries, not pickled models.
    The original class and methods are never monkeypatched.
    """
    if issubclass(base_class, DifferentiableVideoCacheMixin):
        return base_class
    return type("Differentiable" + base_class.__name__,
                (DifferentiableVideoCacheMixin, base_class), {"__module__": __name__})


def adopt_native_for_training(native: nn.Module) -> nn.Module:
    """Use an already-loaded native model without duplicating its large weights.

    The new module container shares parameter objects with the supplied model;
    callers must relinquish the old container for training. Its class/methods
    stay unchanged. There is no global hook, cache mutation or extra checkpoint.
    """
    if isinstance(native, DifferentiableVideoCacheMixin):
        return native
    result = copy.copy(native)
    result.__class__ = differentiable_native_class(type(native))
    for name in ("_modules", "_parameters", "_buffers"):
        setattr(result, name, getattr(native, name).copy())
    return result


def _prefix(actions: Tensor, known_actions: Tensor, known_mask: Tensor) -> Tensor:
    if known_actions.shape != actions.shape or known_mask.shape != actions.shape[:2]:
        raise ValueError("known actions/mask must match [B,H,D] / [B,H]")
    if known_mask.dtype != torch.bool:
        raise ValueError("known_mask must be boolean")
    if (known_mask[:, 1:] & ~known_mask[:, :-1]).any():
        raise ValueError("known actions must form a contiguous prefix")
    if not torch.isfinite(known_actions[known_mask]).all():
        raise ValueError("known action prefix must be finite")
    # Values outside the mask are deliberately ignored, even when NaN-filled.
    return torch.where(known_mask[..., None], known_actions, actions)


def _clock(value, batch: int, device, name: str) -> Tensor:
    value = torch.as_tensor(value, dtype=torch.float64, device=device)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite scalar or [B]")
    return value


def _time_layout(native, sigma: Tensor, known_mask: Tensor, dtype) -> Tensor:
    batch, horizon = known_mask.shape
    motion_count = int(native.config.motion_history_count)
    registers = int(native.action_expert.config.num_registers)
    # Preserve native state/motion/register noise conditioning. Only explicitly
    # committed action tokens become clean (t=0). Empty prefix is native parity.
    times = sigma.to(device=known_mask.device, dtype=dtype).reshape(batch, 1).expand(
        batch, 1 + horizon + motion_count + registers).clone()
    times[:, 1:1 + horizon] = times[:, 1:1 + horizon].masked_fill(known_mask, 0)
    return times


def _euler(sample: Tensor, velocity: Tensor, delta: Tensor) -> Tensor:
    # Match native vendor CPU-scalar multiplication in BF16/FP16 as well as FP32.
    product = velocity.float() * delta.to(device=velocity.device, dtype=torch.float32)
    return sample + product.to(velocity.dtype)


class SlowReferenceAdapter(nn.Module):
    """Native two joint video/action steps plus eight action-only refinements.

    Prediction inputs contain source observations and independently drawn noise
    only; supervision accepts noisy targets in a separate method. Main training
    must enable gradients on native Video/Action/motion parameters. This module
    never detaches the returned reference, including under activation checkpoint.
    """
    def __init__(self, native: nn.Module, *, gradient_checkpointing: bool = False,
                 shift: float = 5.0, sigma_min: float = 0.0,
                 extra_one_step: bool = True, num_train_timesteps: int = 1000):
        super().__init__()
        from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler
        self.native = adopt_native_for_training(native)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_train_timesteps = int(num_train_timesteps)
        self.schedule_config = dict(shift=shift, sigma_min=sigma_min,
                                    extra_one_step=extra_one_step,
                                    num_train_timesteps=num_train_timesteps,
                                    action_steps=10, video_steps=2)
        for kind, steps in (("action", 10), ("video", 2)):
            scheduler = FlowMatchScheduler(num_inference_steps=steps,
                num_train_timesteps=num_train_timesteps, shift=shift,
                sigma_min=sigma_min, extra_one_step=extra_one_step)
            scheduler.set_timesteps(num_inference_steps=steps, training=False)
            self.register_buffer(kind + "_sigmas", scheduler.sigmas.float(), persistent=True)
            self.register_buffer(kind + "_timesteps", scheduler.timesteps.float(), persistent=True)

    def _call(self, function, batch, **kwargs):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(function, batch, use_reentrant=False,
                              preserve_rng_state=True, **kwargs)
        return function(batch, **kwargs)

    def _batch(self, source, actions, known_actions, known_mask, action_t, *, validate_prefix=True,
               time_dtype=None):
        source = SourceCondition.parse(source)
        if actions.ndim != 3 or actions.shape[1] != self.native.config.chunk_size:
            raise ValueError("actions must use the native [B,chunk_size,action_dim] layout")
        if actions.shape[-1] != self.native.config.action_dim:
            raise ValueError("action dimension differs from the native checkpoint")
        values = source.as_batch()
        values["noisy_actions"] = (_prefix(actions, known_actions, known_mask)
            if validate_prefix else torch.where(known_mask[..., None], known_actions, actions))
        values["action_t"] = _time_layout(self.native, action_t, known_mask,
                                          actions.dtype if time_dtype is None else time_dtype)
        return values

    def supervised_forward(self, source, noisy_video: Tensor, noisy_actions: Tensor,
                           video_sigma: Tensor, action_sigma: Tensor,
                           known_actions: Tensor, known_mask: Tensor) -> dict:
        """Return velocities for L_sv/L_sa; the caller must mask prefix action loss.

        Sigma is normalized [0,1], not the native 0..1000 embedding time. Clean
        target/noise mixing and masks are owned by the training objective.
        """
        batch_size = noisy_actions.shape[0]
        av = self._sigma(action_sigma, batch_size, noisy_actions.device)
        vv = self._sigma(video_sigma, batch_size, noisy_video.device)
        batch = self._batch(source, noisy_actions, known_actions, known_mask,
                            av * self.num_train_timesteps)
        batch.update(future_latent=noisy_video,
                     video_t=vv.to(noisy_video.dtype) * self.num_train_timesteps)
        result = self._call(self.native._forward_video_action, batch,
                            include_video_prediction=True, build_video_cache=False)
        # Deliberately no cache or world-reference return from this code path.
        return {"video_pred": result["video_pred"], "action_pred": result["action_pred"]}

    @staticmethod
    def _sigma(value, batch_size, device):
        value = torch.as_tensor(value, dtype=torch.float32, device=device)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.shape != (batch_size,) or not torch.isfinite(value).all() or (
                (value < 0) | (value > 1)).any():
            raise ValueError("sigma must be finite [B] in [0,1]")
        return value

    def sample_reference(self, source, video_noise: Tensor, action_noise: Tensor,
                         known_actions: Tensor, known_mask: Tensor,
                         action_times: Tensor, future_intervals: Tensor,
                         source_time, completed_time, *, version: int = 0) -> dict:
        source = SourceCondition.parse(source)
        if video_noise.ndim != 5 or source.condition_latent.ndim != 5:
            raise ValueError("video noise and observed latent must be [B,C,T,H,W]")
        batch_size = action_noise.shape[0]
        if video_noise.shape[:2] != source.condition_latent.shape[:2] or (
                video_noise.shape[0] != batch_size):
            raise ValueError("observed/predicted video batches/channels differ")
        source_time = _clock(source_time, batch_size, action_noise.device, "source_time")
        completed_time = _clock(completed_time, batch_size, action_noise.device, "completed_time")
        if (completed_time < source_time).any():
            raise ValueError("completion cannot precede source observation")
        intervals = torch.as_tensor(future_intervals, device=action_noise.device,
                                    dtype=torch.float64)
        times = torch.as_tensor(action_times, device=action_noise.device, dtype=torch.float64)
        if intervals.shape != (batch_size, video_noise.shape[2], 2):
            raise ValueError("future_intervals must explicitly describe each codec latent [B,T,2]")
        if not torch.isfinite(intervals).all() or (intervals[..., 1] < intervals[..., 0]).any():
            raise ValueError("future latent physical support is invalid")
        if (intervals[..., 0] < source_time[:, None]).any() or (
                intervals[:, 1:, 0] < intervals[:, :-1, 0]).any():
            raise ValueError("future supports must be ordered and no earlier than source")
        if times.shape != action_noise.shape[:2] or not torch.isfinite(times).all():
            raise ValueError("action_times must explicitly match [B,H]")
        if (times < source_time[:, None]).any() or (times[:, 1:] <= times[:, :-1]).any():
            raise ValueError("action targets must increase from source time")
        actions = _prefix(action_noise, known_actions, known_mask)
        future = video_noise
        cache = None
        for step in range(10):
            action_t = self.action_timesteps[step].expand(batch_size)
            batch = self._batch(source, actions, known_actions, known_mask, action_t,
                                validate_prefix=False, time_dtype=action_noise.dtype)
            if step < 2:
                # Preserve native scalar video timestep semantics for both observed
                # and future video tokens. Known action prefix alone uses t=0.
                batch.update(future_latent=future,
                    video_t=self.video_timesteps[step].to(video_noise.dtype).expand(batch_size))
                result = self._call(self.native._forward_video_action, batch,
                    include_video_prediction=True, build_video_cache=(step == 1))
                next_video_sigma = (self.video_sigmas[step + 1] if step == 0
                                    else self.video_sigmas.new_zeros(()))
                future = _euler(future, result["video_pred"],
                                next_video_sigma - self.video_sigmas[step])
                if step == 1:
                    cache = result["video_cache"]
                    if torch.is_grad_enabled() and any(p.requires_grad for p in self.native.parameters()):
                        if not any(v.requires_grad for v in cache["video_k"] + cache["video_v"]):
                            raise RuntimeError("native video cache detached the predictive gradient")
            else:
                batch["video_cache"] = cache
                result = self._call(self.native.forward_action_with_video_cache, batch)
            next_action_sigma = (self.action_sigmas[step + 1] if step < 9
                                 else self.action_sigmas.new_zeros(()))
            actions = torch.where(known_mask[..., None], known_actions,
                _euler(actions, result["action_pred"],
                       next_action_sigma - self.action_sigmas[step]))
        return dict(condition_latent=source.condition_latent, predicted_video_latent=future,
            reference_actions=actions, action_times=times, future_intervals=intervals,
            source_time=source_time, completed_time=completed_time, version=int(version),
            generation_stage="video_joint_step_2_final_action_step_10",
            video_plan_consistency_guaranteed=False,
            provenance="source_only_independent_noise_prediction")


class StructuredWorldReadout(nn.Module):
    """4x4 spatial tokens per latent slice, plus typed soft action-reference tokens.

    Latent time intervals are external codec provenance, never guessed from the
    number of latent slices. Observation tokens remain available as source context
    at all target times; they do not claim to depict those future target times.
    """
    def __init__(self, latent_channels: int = 48, action_dim: int = 14,
                 world_dim: int = 384, grid_size: int = 4,
                 action_time_tolerance: float = 0.02):
        super().__init__()
        if grid_size < 1 or action_time_tolerance <= 0:
            raise ValueError("grid size and action time tolerance must be positive")
        self.grid_size = int(grid_size)
        self.action_time_tolerance = float(action_time_tolerance)
        self.visual_projection = nn.Conv2d(latent_channels, world_dim, kernel_size=1)
        self.action_projection = nn.Linear(action_dim, world_dim)
        self.type_embedding = nn.Embedding(3, world_dim)
        self.spatial_embedding = nn.Parameter(torch.randn(1, grid_size ** 2, world_dim) * .02)
        self.temporal_projection = nn.Linear(2, world_dim)

    def _visual(self, latent: Tensor, kind: int, intervals: Tensor, source_time: Tensor):
        batch, channels, frames, height, width = latent.shape
        image = latent.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        image = F.adaptive_avg_pool2d(image, (self.grid_size, self.grid_size))
        image = image.to(self.visual_projection.weight.dtype)
        tokens = self.visual_projection(image).flatten(2).transpose(1, 2)
        tokens = tokens + self.spatial_embedding + self.type_embedding.weight[kind]
        relative = (intervals - source_time[:, None, None]).to(tokens.dtype)
        # Observation intervals are unbounded only for attention visibility. Their
        # feature time embedding represents source time (zero), not infinity.
        if kind == 0:
            relative = torch.zeros_like(relative)
        time = self.temporal_projection(relative).reshape(batch * frames, 1, -1)
        tokens = (tokens + time).reshape(batch, frames * self.grid_size ** 2, -1)
        return tokens, intervals.repeat_interleave(self.grid_size ** 2, dim=1)

    def forward(self, reference: Mapping) -> dict:
        if reference.get("provenance") != "source_only_independent_noise_prediction":
            raise ValueError("world readout only accepts predictive source-only references")
        observed = reference["condition_latent"]
        predicted = reference["predicted_video_latent"]
        batch, _, observed_frames, _, _ = observed.shape
        source_time = reference["source_time"]
        obs_intervals = source_time.new_empty(batch, observed_frames, 2)
        obs_intervals[..., 0] = -torch.inf
        obs_intervals[..., 1] = torch.inf
        obs_tokens, obs_intervals = self._visual(observed, 0, obs_intervals, source_time)
        future_tokens, future_intervals = self._visual(
            predicted, 1, reference["future_intervals"], source_time)
        actions = reference["reference_actions"].to(self.action_projection.weight.dtype)
        action_intervals = torch.stack((reference["action_times"] - self.action_time_tolerance,
                                        reference["action_times"] + self.action_time_tolerance), dim=-1)
        action_tokens = self.action_projection(actions) + self.type_embedding.weight[2]
        relative_actions = (action_intervals - source_time[:, None, None]).to(action_tokens.dtype)
        action_tokens = action_tokens + self.temporal_projection(relative_actions)
        tokens = torch.cat((obs_tokens, future_tokens, action_tokens), dim=1)
        intervals = torch.cat((obs_intervals, future_intervals, action_intervals), dim=1)
        types = torch.cat((torch.zeros(obs_tokens.shape[1], dtype=torch.long, device=tokens.device),
                           torch.ones(future_tokens.shape[1], dtype=torch.long, device=tokens.device),
                           torch.full((action_tokens.shape[1],), 2, dtype=torch.long, device=tokens.device)))
        return dict(world_tokens=tokens, world_intervals=intervals,
                    world_valid=torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
                    world_types=types.expand(batch, -1), source_time=source_time,
                    completed_time=reference["completed_time"], version=reference["version"],
                    generation_stage=reference["generation_stage"],
                    video_plan_consistency_guaranteed=False)
