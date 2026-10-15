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
        if int(interface_version) not in (2, 3):
            raise ValueError("joint MotionKV requires interface version 2 or 3")
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
        plan_width = 3 * self.action_dim + 6 if self.interface_version == 2 else 4 * self.action_dim + 8
        self.plan_innovation = nn.Sequential(nn.Linear(plan_width, d), nn.SiLU(), nn.Linear(d, d))
        if self.interface_version == 3:
            # Tiny forecast in the existing native statistic space. Inputs are
            # exclusively source world K/V, source motion/masks and elapsed time.
            self.motion_forecast = nn.Sequential(nn.Linear(self.world_head_dim + 14 * self.motion_history_count + 3, d),
                nn.SiLU(), nn.Linear(d, 12 * self.motion_history_count))
            nn.init.zeros_(self.motion_forecast[-1].weight)
            nn.init.zeros_(self.motion_forecast[-1].bias)
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
            raise ValueError("source visual caching requires a matching versioned RGB snapshot")
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

    def _normalize_motion(self, motion, interval, acceleration):
        valid = torch.cat((interval[..., None].expand(-1, -1, 9),
                           (acceleration & interval)[..., None].expand(-1, -1, 3)), -1)
        safe = torch.where(valid, motion.float(), self.motion_mean.float()[None, None])
        normalized = (safe - self.motion_mean.float()) / self.motion_scale.float()
        return normalized.sign() * torch.log1p(normalized.abs()), valid

    def predict_current_motion(self, current, bridge):
        if self.interface_version != 3:
            raise ValueError("motion forecasting requires interface version 3")
        source = bridge.source_motion
        if source is None or bridge.source_motion_interval_valid_mask is None or bridge.source_motion_acceleration_valid_mask is None:
            raise ValueError("motion forecasting requires source-only motion metadata")
        normalized, _ = self._normalize_motion(source,
            bridge.source_motion_interval_valid_mask.bool(), bridge.source_motion_acceleration_valid_mask.bool())
        masks = torch.stack((bridge.source_motion_interval_valid_mask,
            bridge.source_motion_acceleration_valid_mask & bridge.source_motion_interval_valid_mask), -1).float()
        value = bridge.video_v[-1]
        valid = torch.arange(value.shape[1], device=value.device)[None] < bridge.seq_lens[:, None]
        null = self._column(bridge.null_bridge, source.shape[0], value.device, torch.bool)
        valid = valid & ~null
        world = torch.where(valid[..., None, None], value.float(), torch.zeros_like(value.float())).mean(2)
        world = world.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        elapsed = self._column(current.age.delta_sim_seconds, source.shape[0], source.device, torch.float32) / self.sim_scale
        time = torch.cat((elapsed, elapsed.square(), torch.log1p(elapsed.clamp_min(0))), -1)
        evidence = torch.cat((world, torch.cat((normalized, masks), -1).flatten(1), time), -1)
        residual = self.motion_forecast(evidence.to(self.motion_forecast[0].weight.dtype)).reshape_as(normalized)
        # The source history is a meaningful zero-init baseline. Learned drift
        # replaces a hand-coded flow extrapolation without requiring dense flow.
        return normalized + residual.float()

    def motion_prediction_loss(self, current, bridge):
        prediction = self.predict_current_motion(current, bridge)
        target, valid = self._normalize_motion(current.motion.detach(),
            current.motion_interval_valid_mask.bool(), current.motion_acceleration_valid_mask.bool())
        error = torch.where(valid, prediction - target, torch.zeros_like(prediction))
        return (error.square().sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)).mean()

    def aligned_source_plan(self, current, bridge):
        """Align source targets by physical time; arms linear, grippers held."""
        plan, source_offsets = bridge.source_action_plan, bridge.source_action_target_offsets
        offsets = current.action_target_offsets
        if plan is None or source_offsets is None or offsets is None or source_offsets.shape != plan.shape[:2] or offsets.shape != plan.shape[:2]:
            raise ValueError("V3 source plan requires matching physical endpoint offsets")
        b, horizon, width = plan.shape
        source_offsets = source_offsets.to(device=plan.device, dtype=torch.float64)
        query = offsets.to(device=plan.device, dtype=torch.float64) + self._column(current.age.delta_sim_seconds,b,plan.device,torch.float64)
        # Tensor-only search is capture-safe and never substitutes integer age.
        index = (query[..., None] + 1e-9 >= source_offsets[:, None, :]).sum(-1) - 1
        lower, upper = index.clamp(0,horizon-1), (index+1).clamp(0,horizon-1)
        lo = plan.gather(1,lower[...,None].expand(-1,-1,width))
        hi = plan.gather(1,upper[...,None].expand(-1,-1,width))
        t0, t1 = source_offsets.gather(1,lower), source_offsets.gather(1,upper)
        weight = ((query-t0)/(t1-t0).clamp_min(1e-12)).clamp(0,1).to(plan.dtype)
        result = lo + (hi-lo)*weight[...,None]
        if self.action_dim == 14:
            index_d = torch.arange(width,device=plan.device)
            keep = (index_d == 6) | (index_d == 13)
            result = torch.where(keep[None,None],lo,result)
        available = (query >= source_offsets[:,:1]-1e-9) & (query <= source_offsets[:,-1:]+1e-9)
        return torch.where(available[...,None],result,torch.zeros_like(result)), available

    def _motion_delta(self, current, bridge, dtype):
        if self.interface_version == 3:
            prediction = self.predict_current_motion(current, bridge)
            target, valid = self._normalize_motion(current.motion,
                current.motion_interval_valid_mask.bool(), current.motion_acceleration_valid_mask.bool())
            interval = current.motion_interval_valid_mask.bool()
            acceleration = current.motion_acceleration_valid_mask.bool() & interval
            innovation = torch.where(valid, target - prediction, torch.zeros_like(prediction))
            features = torch.cat((innovation, interval[..., None].float(), acceleration[..., None].float()), -1)
            return self.motion_innovation(features.to(dtype)) + self.motion_position.to(dtype)
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
        if self.interface_version == 3:
            old, available = self.aligned_source_plan(current, bridge)
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
        if self.interface_version == 3:
            previous, previous_valid = current.previous_plan, current.previous_plan_valid
            if previous is None or previous_valid is None or previous.shape != plan.shape or previous_valid.shape != plan.shape[:2]:
                raise ValueError("V3 requires explicit previous proposal and validity, including empty bootstrap")
            # Soft queue context decays away from the hard-prefix boundary. No
            # old suffix target is imposed: only committed prefix is immutable.
            near = torch.exp(-(target_offsets.to(device=device,dtype=dtype) - delay.squeeze(-1)*self.sim_scale).clamp_min(0)/.64)
            safe_previous = torch.where(previous_valid[...,None],previous,torch.zeros_like(previous))
            features = torch.cat((features, .1*near[...,None]*safe_previous.to(dtype),
                previous_valid[...,None].to(dtype), near[...,None]), -1)
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
