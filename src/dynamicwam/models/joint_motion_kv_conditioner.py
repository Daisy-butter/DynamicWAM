"""Versioned cross-time corrections of jointly trainable action-conditioned K/V.

New information is represented as causal innovation: source-to-current RGB
patch differences, motion-statistic change, and source-plan/committed-prefix
mismatch. These signals select spatial low-rank updates before native joint
attention, never replace native Action DiT motion tokens. All source tensors
remain differentiable during joint training; no hidden inference state exists.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from dynamicwam.models.motion_kv_conditioner import MotionKVConditioner

JOINT_MOTIONKV_INTERFACE_VERSION = 2


class JointMotionKVConditioner(MotionKVConditioner):
    def __init__(self, *, action_dim=14, interface_version=2,
                 correction_layers=None, require_source_rgb=True, **kwargs):
        if int(interface_version) != JOINT_MOTIONKV_INTERFACE_VERSION:
            raise ValueError("joint MotionKV requires interface version 2")
        super().__init__(**kwargs)
        self.interface_version = int(interface_version)
        self.action_dim = int(action_dim)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.correction_layers = tuple(range(self.layers)) if correction_layers is None else tuple(int(i) for i in correction_layers)
        if not self.correction_layers or len(set(self.correction_layers)) != len(self.correction_layers) or min(self.correction_layers) < 0 or max(self.correction_layers) >= self.layers:
            raise ValueError("correction_layers must be unique native layer indices")
        self.require_source_rgb = bool(require_source_rgb)
        d = self.world_dim
        self.visual_innovation = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.motion_innovation = nn.Sequential(nn.Linear(14, d), nn.SiLU(), nn.Linear(d, d))
        self.plan_innovation = nn.Sequential(nn.Linear(3 * self.action_dim + 6, d), nn.SiLU(), nn.Linear(d, d))
        self.innovation_kind = nn.Parameter(torch.randn(1, 3, d) * .02)
        # New information must not alter the V1 softmax normalizer at migration.
        # Loaded nonzero K/V heads train this gate first, then innovation paths.
        self.innovation_gate = nn.Parameter(torch.zeros(()))
        self.constructor_spec.update(action_dim=self.action_dim, interface_version=self.interface_version,
            correction_layers=list(self.correction_layers), require_source_rgb=self.require_source_rgb)

    @torch.inference_mode()
    def prepare_source_cache(self, bridge):
        """Return an owned derived bridge, recomputed even if it was cached.

        Caller installs this once per new source after freezing inference
        weights. There is no model-global memoization or cross-episode state.
        Re-running this method is the explicit invalidation path for a changed
        source snapshot or changed weights. Training rejects these cached rows.
        """
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("source visual caching requires frozen eval weights")
        if bridge.interface_version != self.interface_version or bridge.source_rgb is None:
            raise ValueError("source visual caching requires a version 2 RGB snapshot")
        visual, valid = self.encoder(bridge.source_rgb, bridge.source_rgb_valid_mask)
        return replace(bridge, source_visual_tokens=visual.clone(), source_visual_valid=valid.clone())

    @staticmethod
    def _column(value, batch, device, dtype):
        value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError("joint context scalar batch mismatch")
        return value[:, None]

    def _motion_delta(self, current, bridge, dtype):
        source = bridge.source_motion
        source_interval = bridge.source_motion_interval_valid_mask
        source_accel = bridge.source_motion_acceleration_valid_mask
        if source is None or source_interval is None or source_accel is None:
            raise ValueError("joint MotionKV requires causal source motion metadata")
        if source.shape != current.motion.shape:
            raise ValueError("source/current motion history layouts differ")
        interval = current.motion_interval_valid_mask.bool() & source_interval.bool()
        accel = current.motion_acceleration_valid_mask.bool() & source_accel.bool() & interval
        valid = torch.cat((interval[..., None].expand(-1, -1, 9), accel[..., None].expand(-1, -1, 3)), -1)
        # NaN/Inf from unavailable intervals must not enter subtraction.
        now = torch.where(valid, current.motion.float(), torch.zeros_like(current.motion.float()))
        old = torch.where(valid, source.float(), torch.zeros_like(source.float()))
        delta = (now - old) / self.motion_scale.float()[None, None]
        delta = delta.sign() * torch.log1p(delta.abs())
        features = torch.cat((delta, interval[..., None].float(), accel[..., None].float()), -1)
        return self.motion_innovation(features.to(dtype)) + self.motion_position.to(dtype)

    def _plan_tokens(self, current, bridge, dtype):
        plan = bridge.source_action_plan
        prefix, lengths = current.committed_prefix, current.committed_prefix_lengths
        if plan is None or bridge.source_action_sigma is None:
            raise ValueError("joint MotionKV requires actual source action-plan metadata")
        if prefix is None or lengths is None:
            raise ValueError("joint MotionKV requires explicit committed prefix, including length zero")
        if plan.ndim != 3 or prefix.shape != plan.shape or plan.shape[-1] != self.action_dim:
            raise ValueError("plan and prefix must share [B,chunk,action_dim]")
        batch, chunk, _ = plan.shape
        target_offsets = current.action_target_offsets
        if target_offsets is None or target_offsets.shape != (batch, chunk):
            raise ValueError("joint MotionKV requires explicit [B,chunk] physical action target offsets")
        device = plan.device
        age = self._column(current.age.delta_actions, batch, device, torch.long)
        index = age + torch.arange(chunk, device=device)[None]
        available = (index >= 0) & (index < chunk)
        old = plan.gather(1, index.clamp(0, chunk - 1)[..., None].expand(-1, -1, self.action_dim))
        old = torch.where(available[..., None], old, torch.zeros_like(old))
        length = self._column(lengths, batch, device, torch.long)
        committed = torch.arange(chunk, device=device)[None] < length
        safe_prefix = torch.where(committed[..., None], prefix, torch.zeros_like(prefix))
        difference = torch.where((available & committed)[..., None], safe_prefix - old, torch.zeros_like(old))
        position = torch.arange(chunk, device=device, dtype=dtype)[None, :, None].expand(batch, -1, -1) / max(chunk - 1, 1)
        sigma = self._column(bridge.source_action_sigma, batch, device, dtype)[:, None].expand(-1, chunk, -1)
        delay = self._column(current.observation_delay_seconds, batch, device, dtype)[:, None].expand(-1, chunk, -1) / self.sim_scale
        features = torch.cat((old.to(dtype), safe_prefix.to(dtype), difference.to(dtype),
            available[..., None].to(dtype), committed[..., None].to(dtype), position, sigma, delay,
            target_offsets.to(device=device, dtype=dtype)[..., None] / self.sim_scale), -1)
        return self.plan_innovation(features)

    def _rank_observation(self, seeds, observed, observed_valid, *, current, bridge):
        if bridge is None or bridge.interface_version != self.interface_version:
            raise ValueError("joint MotionKV cannot silently consume a legacy bridge")
        base = super()._rank_observation(seeds, observed, observed_valid, current=current, bridge=bridge)
        dtype = observed.dtype
        if bridge.source_visual_tokens is not None:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("source visual cache is inference-only; training must encode source RGB")
            old_visual, old_valid = bridge.source_visual_tokens, bridge.source_visual_valid
            if old_valid is None or old_visual.shape != observed[:, :48].shape or old_valid.shape != observed_valid[:, :48].shape:
                raise ValueError("source visual cache layout mismatch")
            innovation_valid = observed_valid[:, :48] & old_valid
        elif bridge.source_rgb is None:
            if self.require_source_rgb:
                raise ValueError("joint MotionKV requires causal source RGB for spatial innovation")
            # Explicit ablation remains in the parameter graph with no image
            # contribution, rather than silently substituting the current image.
            old_visual = observed[:, :48] * 0
            innovation_valid = observed_valid[:, :48] & False
        else:
            old_visual, old_valid = self.encoder(bridge.source_rgb, bridge.source_rgb_valid_mask)
            innovation_valid = observed_valid[:, :48] & old_valid
        spatial_delta = torch.where(innovation_valid[..., None], observed[:, :48] - old_visual,
                                    torch.zeros_like(old_visual))
        visual = self.visual_innovation(spatial_delta) + self.innovation_kind[:, :1].to(dtype)
        motion = self._motion_delta(current, bridge, dtype) + self.innovation_kind[:, 1:2].to(dtype)
        plan = self._plan_tokens(current, bridge, dtype) + self.innovation_kind[:, 2:3].to(dtype)
        extra = torch.ones((observed.shape[0], motion.shape[1] + plan.shape[1]), device=observed.device, dtype=torch.bool)
        innovation = self.observation_norm(torch.cat((visual, motion, plan), 1))
        valid = torch.cat((innovation_valid, extra), 1)
        residual = self._attention(base, self.observation_k(innovation), self.observation_v(innovation), valid)
        return base + self.innovation_gate.tanh().to(dtype) * residual

    def adapt_video_kv(self, layer_idx, action_tokens, memory, base_key, base_value):
        key, value = super().adapt_video_kv(layer_idx, action_tokens, memory, base_key, base_value)
        if int(layer_idx) not in self.correction_layers:
            # Fixed zero multiplier keeps training's parameter-use graph stable
            # for layer-position ablations while preserving bit-exact base K/V.
            return base_key + (key - base_key) * 0, base_value + (value - base_value) * 0
        return key, value
