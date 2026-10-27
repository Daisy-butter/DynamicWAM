"""Differentiable source-only 2+8 sampling with native-grid feature readout.

The internal native K/V only accelerates the remaining eight slow action steps.
It is never the inter-rate interface. The public reference contains pre-joint
observation tokens, joint2 future hidden/clean latent, final action plan and clocks.
No RGB decode/re-encode is required. Auxiliary VAE/text encoders remain outside.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Mapping

import torch
from torch import Tensor, nn
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
    if isinstance(value, Tensor) and value.dtype != torch.float64:
        raise ValueError(f"{name} tensor must be FP64 before physical time arithmetic")
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
                         source_time, completed_time, *, action_intervals: Tensor,
                         source_metadata: Mapping, future_metadata: Mapping,
                         version: int = 0) -> dict:
        source = SourceCondition.parse(source)
        if video_noise.ndim != 5 or source.condition_latent.ndim != 5:
            raise ValueError("video noise and observed latent must be [B,C,T,H,W]")
        batch_size = action_noise.shape[0]
        actions = _prefix(action_noise, known_actions, known_mask)
        if video_noise.shape[:2] != source.condition_latent.shape[:2] or (
                video_noise.shape[0] != batch_size):
            raise ValueError("observed/predicted video batches/channels differ")
        source_time = _clock(source_time, batch_size, action_noise.device, "source_time")
        completed_time = _clock(completed_time, batch_size, action_noise.device, "completed_time")
        if (completed_time < source_time).any():
            raise ValueError("completion cannot precede source observation")
        if any(not isinstance(value, Tensor) or value.dtype != torch.float64
               for value in (future_intervals, action_times, action_intervals)):
            raise ValueError("future/action intervals and times must explicitly be FP64 tensors")
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
        if (times[:, 1:] <= times[:, :-1]).any():
            raise ValueError("action targets must increase")
        action_support = torch.as_tensor(action_intervals, device=action_noise.device, dtype=torch.float64)
        if action_support.shape != (*times.shape, 2) or not torch.isfinite(action_support).all():
            raise ValueError("action_intervals must explicitly describe physical [B,H,2] support")
        if (action_support[..., 1] <= action_support[..., 0]).any() or not torch.equal(action_support[..., 0], times):
            raise ValueError("action_times must be exact starts of positive physical action intervals")
        if not torch.allclose(action_support[:, 1:, 0], action_support[:, :-1, 1], atol=1e-8, rtol=0):
            raise ValueError("action support must be contiguous, without gaps or overlaps")
        # An asynchronous source may arrive during the first executing row.
        # Preserve only that immutable still-active row; do not admit expired
        # history or an uncommitted pre-source target into future sampling.
        active_first = torch.zeros_like(known_mask)
        active_first[:, 0] = known_mask[:, 0] & (action_support[:, 0, 1] > source_time)
        if ((times < source_time[:, None]) & ~active_first).any():
            raise ValueError("pre-source action must be the committed first interval still covering source")
        patch_size = tuple(self.native.compact_wan.video_model.wan_model.patch_size)
        if patch_size != (1, 2, 2):
            raise ValueError("this readout requires the verified native temporal1/spatial2 patch contract")
        source_grid = latent_patch_grid(source.condition_latent, patch_size)
        future_grid = latent_patch_grid(video_noise, patch_size)
        source_metadata = validate_spatial_metadata(source_metadata, source_grid, source_time, future=False)
        future_metadata = validate_spatial_metadata(future_metadata, future_grid, source_time, future=True)
        if source_metadata["codec_id"] != future_metadata["codec_id"]:
            raise ValueError("source/future codec identities differ")
        if not torch.equal(future_metadata["temporal_support"], intervals):
            raise ValueError("future_intervals differ from declared codec support")
        future = video_noise
        cache = None
        feature_export = None
        joint_action_estimate = None
        for step in range(10):
            action_t = self.action_timesteps[step].expand(batch_size)
            batch = self._batch(source, actions, known_actions, known_mask, action_t,
                                validate_prefix=False, time_dtype=action_noise.dtype)
            if step < 2:
                # Preserve native scalar video timestep semantics for both observed
                # and future video tokens. Known action prefix alone uses t=0.
                batch.update(future_latent=future,
                    video_t=self.video_timesteps[step].to(video_noise.dtype).expand(batch_size))
                if step == 1:
                    batch["predictive_reference"] = True
                result = self._call(self.native._forward_video_action, batch,
                    include_video_prediction=True, build_video_cache=(step == 1),
                    export_predictive_features=(step == 1))
                next_video_sigma = (self.video_sigmas[step + 1] if step == 0
                                    else self.video_sigmas.new_zeros(()))
                future = _euler(future, result["video_pred"],
                                next_video_sigma - self.video_sigmas[step])
                if step == 1:
                    cache = result["video_cache"]
                    feature_export = {key: result[key] for key in
                        ("source_tokens", "future_hidden", "feature_grid_sizes")}
                    joint_action_estimate = torch.where(known_mask[..., None], known_actions,
                        _euler(actions, result["action_pred"], -self.action_sigmas[step]))
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
        if feature_export is None or joint_action_estimate is None:
            raise RuntimeError("second native joint step did not export prediction features")
        return dict(**feature_export, condition_latent=source.condition_latent, predicted_video_latent=future,
            reference_actions=actions, action_times=times, future_intervals=intervals,
            action_intervals=action_support, source_metadata=source_metadata,
            future_metadata=future_metadata, patch_size=patch_size,
            test_fixture=source_metadata["codec_id"].startswith("test_fixture"),
            prediction_sigmas=torch.stack((self.video_sigmas[1], self.action_sigmas[1])).to(actions.device),
            # These are diagnostic estimates, NOT an additional action condition.
            joint_action_estimate=joint_action_estimate.detach(),
            joint_final_action_mse=(joint_action_estimate.detach().float() - actions.detach().float()).square().mean(dim=(1, 2)),
            source_time=source_time, completed_time=completed_time, version=int(version),
            generation_stage="video_joint_step_2_final_action_step_10",
            video_plan_consistency_guaranteed=False,
            provenance="source_only_independent_noise_prediction")


def latent_patch_grid(latent: Tensor, patch_size=(1, 2, 2)) -> tuple[int, int, int]:
    """Native Conv3d layout; the producer must NOT pool before this operation."""
    if latent.ndim != 5 or any(size <= 0 for size in latent.shape):
        raise ValueError("latent must be a nonempty [B,C,T,H,W] tensor")
    if len(patch_size) != 3 or any(p < 1 for p in patch_size):
        raise ValueError("invalid native patch size")
    if any(n % p for n, p in zip(latent.shape[2:], patch_size)):
        raise ValueError("latent dimensions must exactly divide the native patch size")
    return tuple(n // p for n, p in zip(latent.shape[2:], patch_size))


def latent_patches(latent: Tensor, patch_size=(1, 2, 2)) -> Tensor:
    """[B,N,C*pt*ph*pw], ordered time then row then column, like native Conv3d."""
    time, height, width = latent_patch_grid(latent, patch_size)
    batch, channels = latent.shape[:2]
    pt, ph, pw = patch_size
    return latent.reshape(batch, channels, time, pt, height, ph, width, pw).permute(
        0, 2, 4, 6, 1, 3, 5, 7).reshape(batch, time * height * width, channels * pt * ph * pw)


def validate_spatial_metadata(metadata: Mapping, grid, source_time: Tensor, *, future: bool) -> dict:
    """Validate externally audited codec/time/view metadata without inventing it.

    view_overlap is head/left-wrist/right-wrist/padding nominal patch coverage,
    not a claim that the VAE receptive field contains only one camera. Input type
    is RGB=0, flow=1, mixed=2, and follows codec support rather than latent index.
    """
    required = {"codec_id", "temporal_support", "view_overlap", "valid", "input_type"}
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("spatial metadata requires exact codec/time/view/valid/input_type fields")
    if not isinstance(metadata["codec_id"], str) or not metadata["codec_id"].strip():
        raise ValueError("explicit audited codec_id is required")
    result = dict(metadata)
    batch, time, height, width = source_time.shape[0], *grid
    count = time * height * width
    support = result["temporal_support"]
    if not isinstance(support, Tensor) or support.dtype != torch.float64:
        raise ValueError("codec temporal_support must be explicit FP64 physical seconds")
    support = support.to(source_time.device)
    if support.shape != (batch, time, 2) or not torch.isfinite(support).all() or (
            support[..., 1] < support[..., 0]).any():
        raise ValueError("invalid codec temporal support shape/order")
    if (support[:, 1:, 1] < support[:, :-1, 1]).any():
        raise ValueError("codec slices must have ordered support ends")
    if future and (support[..., 0] < source_time[:, None]).any():
        raise ValueError("future target supports cannot precede source time")
    if not future and (support[..., 1] > source_time[:, None]).any():
        raise ValueError("source codec contains observations unavailable at source time")
    view = result["view_overlap"]
    valid, kind = result["valid"], result["input_type"]
    if not isinstance(view, Tensor) or not view.dtype.is_floating_point or view.shape != (batch, count, 4):
        raise ValueError("view_overlap must be [B,Nnative,4] floating coverage fractions")
    if not torch.isfinite(view).all() or ((view < 0) | (view > 1)).any() or not torch.allclose(
            view.float().sum(-1), torch.ones_like(view[..., 0], dtype=torch.float32), atol=1e-5, rtol=0):
        raise ValueError("view/padding coverage must be finite fractions summing to one")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.shape != (batch, count):
        raise ValueError("valid must explicitly mask every native token")
    if (valid.to(view.device) & (view[..., 3] >= 1 - 1e-6)).any():
        raise ValueError("fully padded patches cannot be valid source/future evidence")
    if not isinstance(kind, Tensor) or kind.dtype != torch.long or kind.shape != (batch, time) or (
            (kind < 0) | (kind > 2)).any():
        raise ValueError("input_type must be [B,T] long RGB=0/flow=1/mixed=2")
    result.update(temporal_support=support, view_overlap=view.to(source_time.device),
                  valid=valid.to(source_time.device), input_type=kind.to(source_time.device))
    return result


class StructuredWorldReadout(nn.Module):
    """Native-grid O(prejoint) / F(hidden+clean patch) / A10 soft reference.

    Finite physical support and read eligibility are separate. Consumers must use
    world_types: source context is historical, future evidence needs target-time
    coverage, and action references provide temporally ordered soft plan context.
    """
    def __init__(self, latent_channels: int = 48, action_dim: int = 14,
                 world_dim: int = 384, *, native_hidden_dim: int = 2048,
                 patch_size=(1, 2, 2), test_fixture: bool = False):
        super().__init__()
        if tuple(patch_size) != (1, 2, 2):
            raise ValueError("only the verified native patch1x2x2 contract is supported")
        self.test_fixture = bool(test_fixture)
        self.patch_size = tuple(patch_size)
        self.native_hidden_dim = int(native_hidden_dim)
        self.latent_channels = int(latent_channels)
        patch_width = latent_channels * 4
        self.source_norm = nn.LayerNorm(native_hidden_dim)
        self.future_norm = nn.LayerNorm(native_hidden_dim)
        self.source_projection = nn.Linear(native_hidden_dim, world_dim)
        self.future_projection = nn.Linear(native_hidden_dim + patch_width, world_dim)
        self.action_projection = nn.Linear(action_dim, world_dim)
        self.type_embedding = nn.Embedding(3, world_dim)
        self.input_type_embedding = nn.Embedding(3, world_dim)
        self.spatial_projection = nn.Linear(2, world_dim)
        self.temporal_projection = nn.Linear(2, world_dim)
        self.view_projection = nn.Linear(4, world_dim)
        self.stage_projection = nn.Linear(2, world_dim)

    def _metadata_features(self, tokens, metadata, grid, source_time, kind):
        batch, time, height, width = tokens.shape[0], *grid
        intervals = metadata["temporal_support"].repeat_interleave(height * width, dim=1)
        dtype = tokens.dtype
        # FP64 subtraction happens before conversion to the model precision.
        relative = (intervals - source_time[:, None, None]).to(dtype)
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=tokens.device, dtype=dtype) + .5) / height,
            (torch.arange(width, device=tokens.device, dtype=dtype) + .5) / width,
            indexing="ij")
        position = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2).repeat(batch, time, 1)
        types = metadata["input_type"].repeat_interleave(height * width, dim=1)
        return (tokens + self.type_embedding.weight[kind]
                + self.input_type_embedding(types)
                + self.temporal_projection(relative)
                + self.spatial_projection(position)
                + self.view_projection(metadata["view_overlap"].to(dtype))), intervals, position

    def forward(self, reference: Mapping) -> dict:
        if reference.get("provenance") != "source_only_independent_noise_prediction":
            raise ValueError("world readout only accepts predictive source-only references")
        required = {"source_tokens", "future_hidden", "feature_grid_sizes", "source_metadata",
                    "future_metadata", "action_intervals", "prediction_sigmas", "patch_size"}
        if not required <= reference.keys():
            raise ValueError("native feature export and explicit temporal/spatial metadata are required")
        if reference.get("test_fixture", False) and not self.test_fixture:
            raise ValueError("a test-fixture reference cannot enter a production readout")
        if tuple(reference["patch_size"]) != self.patch_size:
            raise ValueError("readout/native patch size differs")
        source_time = reference["source_time"]
        observed, predicted = reference["condition_latent"], reference["predicted_video_latent"]
        obs_grid = latent_patch_grid(observed, self.patch_size)
        future_grid = latent_patch_grid(predicted, self.patch_size)
        layout = reference["feature_grid_sizes"]
        for name, grid in (("condition", obs_grid), ("future", future_grid)):
            if tuple(layout[name + "_grid_shape"]) != grid or int(layout[name + "_seq_len"]) != grid[0] * grid[1] * grid[2]:
                raise ValueError("native feature/latent grid layout mismatch")
        obs_meta = validate_spatial_metadata(reference["source_metadata"], obs_grid, source_time, future=False)
        future_meta = validate_spatial_metadata(reference["future_metadata"], future_grid, source_time, future=True)
        if obs_meta["codec_id"] != future_meta["codec_id"]:
            raise ValueError("source/future codec identity mismatch")
        source_feature, future_feature = reference["source_tokens"], reference["future_hidden"]
        batch = observed.shape[0]
        for feature, grid in ((source_feature, obs_grid), (future_feature, future_grid)):
            if feature.shape != (batch, grid[0] * grid[1] * grid[2], self.native_hidden_dim):
                raise ValueError("native hidden feature shape mismatch")
        if predicted.shape[1] != self.latent_channels:
            raise ValueError("clean predicted latent channels differ from readout")
        dtype = self.source_projection.weight.dtype
        obs_tokens = self.source_projection(self.source_norm(source_feature.to(dtype)))
        # Keep the native standardized VAE numerical scale; do not normalize away
        # the predicted latent magnitude or spatially pool adjacent cells.
        future_tokens = self.future_projection(torch.cat((
            self.future_norm(future_feature.to(dtype)),
            latent_patches(predicted, self.patch_size).to(dtype)), dim=-1))
        obs_tokens, obs_intervals, obs_position = self._metadata_features(
            obs_tokens, obs_meta, obs_grid, source_time, 0)
        future_tokens, future_intervals, future_position = self._metadata_features(
            future_tokens, future_meta, future_grid, source_time, 1)
        sigmas = reference["prediction_sigmas"].to(device=future_tokens.device, dtype=dtype)
        if sigmas.shape != (2,) or not torch.isfinite(sigmas).all() or ((sigmas < 0) | (sigmas > 1)).any():
            raise ValueError("prediction sigma stage must contain video/action normalized sigmas")
        future_tokens = future_tokens + self.stage_projection(sigmas)[None, None]
        actions = reference["reference_actions"].to(self.action_projection.weight.dtype)
        action_intervals = reference["action_intervals"]
        if action_intervals.dtype != torch.float64 or action_intervals.shape != (*actions.shape[:2], 2) or not torch.isfinite(action_intervals).all():
            raise ValueError("soft actions require explicit FP64 physical support")
        if (action_intervals[..., 1] <= action_intervals[..., 0]).any():
            raise ValueError("soft action support must be positive")
        action_tokens = self.action_projection(actions) + self.type_embedding.weight[2]
        action_tokens = action_tokens + self.temporal_projection(
            (action_intervals - source_time[:, None, None]).to(action_tokens.dtype))
        tokens = torch.cat((obs_tokens, future_tokens, action_tokens), dim=1)
        intervals = torch.cat((obs_intervals, future_intervals, action_intervals), dim=1)
        types = torch.cat((torch.zeros(obs_tokens.shape[1], dtype=torch.long, device=tokens.device),
                           torch.ones(future_tokens.shape[1], dtype=torch.long, device=tokens.device),
                           torch.full((action_tokens.shape[1],), 2, dtype=torch.long, device=tokens.device)))
        valid = torch.cat((obs_meta["valid"], future_meta["valid"],
                           torch.ones(actions.shape[:2], dtype=torch.bool, device=tokens.device)), dim=1)
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        # Action tokens have no image footprint: use finite zero coordinates and
        # a separate spatial-valid mask, rather than inventing a camera identity.
        positions = torch.cat((obs_position, future_position,
                                tokens.new_zeros(batch, actions.shape[1], 2)), dim=1)
        views = torch.cat((obs_meta["view_overlap"], future_meta["view_overlap"],
                            tokens.new_zeros(batch, actions.shape[1], 4)), dim=1)
        return dict(world_tokens=tokens, world_intervals=intervals, world_valid=valid,
                    world_types=types.expand(batch, -1), world_positions=positions,
                    world_view_overlap=views, world_spatial_valid=(types != 2).expand(batch, -1),
                    reference_actions=reference["reference_actions"],
                    reference_action_intervals=action_intervals,
                    source_time=source_time, completed_time=reference["completed_time"],
                    version=reference["version"], codec_id=obs_meta["codec_id"],
                    generation_stage=reference["generation_stage"],
                    video_plan_consistency_guaranteed=False)
