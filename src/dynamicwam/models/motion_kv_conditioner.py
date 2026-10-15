"""Motion- and action-conditioned low-rank corrections of frozen world K/V.

The RGB encoder retains 4x4 tokens per camera. Native four-interval optical-flow
statistics supply causal motion evidence, not privileged object locations.
Each query builds small rank memories and spatial assignments once; every
action denoising step updates those memories using that step's action tokens.
The original Action DiT, including its native motion tokens, remains the actor.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.compact_fast_policy import CompactVision
from dynamicwam.models.fast_conditioner import FastConditionMemory
from dynamicwam.models.fast_observation_encoder import TextTokenPool


class MotionKVMemory(FastConditionMemory):
    """Owned per-query computations; all tensors remain attached for training.

    Extra fields are tuples indexed by native action/world layer. Assignments
    are [B,Nv,R], rank_context [B,R,D], world_valid [B,Nv], and RMS buffers
    [B,Nv,H,1]. keys/values expose rank_context for existing memory consumers.
    No field aliases a mutable CUDA graph output from a previous query.
    """

    def __init__(self, assignments, rank_context, world_valid, key_rms, value_rms):
        context = tuple(rank_context)
        valid = torch.ones(context[0].shape[:2], device=context[0].device, dtype=torch.bool)
        super().__init__(context, context, valid)
        self.assignments = tuple(assignments)
        self.rank_context = context
        self.world_valid = tuple(world_valid)
        self.key_rms = tuple(key_rms)
        self.value_rms = tuple(value_rms)


class MotionKVConditioner(nn.Module):
    """Bounded corrections immediately before the native joint attention.

    Reuses WorldAwareConditioner's spatial CompactVision and TextTokenPool.
    Legacy world_layers/world_queries are accepted in the constructor so old
    configuration plumbing remains usable; corrections explicitly cover ALL
    native layers and use kv_rank slots. There is no separate action-residual
    branch or unused legacy branch parameters.
    """

    def __init__(self, *, dim=768, layers=12, heads=8, head_dim=96,
                 text_dim=4096, state_dim=14, sim_scale=1.0, action_scale=16.0,
                 world_aware=True, world_layers=(3, 7, 11), world_heads=16,
                 world_head_dim=128, world_queries=16, world_dim=192,
                 encoder="compact_gn", motion_kv=True, kv_rank=8,
                 max_relative_update=.25, motion_mean=None, motion_scale=None,
                 motion_history_count=4):
        super().__init__()
        if not world_aware or not motion_kv or encoder != "compact_gn":
            raise ValueError("motion KV requires world_aware/motion_kv and compact_gn")
        if min(dim, layers, heads, head_dim, world_heads, world_head_dim,
               world_queries, world_dim, kv_rank, motion_history_count) < 1:
            raise ValueError("motion KV dimensions must be positive")
        if dim != heads * head_dim or world_dim % heads:
            raise ValueError("native dimensions or compact attention heads disagree")
        if not math.isfinite(max_relative_update) or not 0 < max_relative_update <= 1:
            raise ValueError("max_relative_update must be finite and in (0,1]")
        if sim_scale <= 0 or action_scale <= 0:
            raise ValueError("age scales must be positive")
        self.dim, self.layers, self.heads = int(dim), int(layers), int(heads)
        self.world_heads, self.world_head_dim = int(world_heads), int(world_head_dim)
        self.world_dim, self.kv_rank = int(world_dim), int(kv_rank)
        self.state_dim, self.motion_history_count = int(state_dim), int(motion_history_count)
        self.sim_scale, self.action_scale = float(sim_scale), float(action_scale)
        self.max_relative_update = float(max_relative_update)
        self.world_aware, self.motion_kv = True, True
        self.world_layers = tuple(range(self.layers))
        mean = torch.as_tensor([0.] * 12 if motion_mean is None else motion_mean, dtype=torch.float32)
        scale = torch.as_tensor([1.] * 12 if motion_scale is None else motion_scale, dtype=torch.float32)
        if mean.shape != (12,) or scale.shape != (12,):
            raise ValueError("native motion normalization needs exactly twelve features")
        if not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("invalid motion normalization")
        self.register_buffer("motion_mean", mean.clone())
        self.register_buffer("motion_scale", scale.clone())
        d = self.world_dim
        self.encoder = CompactVision(d, encoder)
        self.text_pool = TextTokenPool(text_dim=text_dim, dim=d, queries=4)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.age_mlp = nn.Sequential(nn.Linear(3 + state_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.motion_encoder = nn.Sequential(nn.Linear(14, d), nn.SiLU(), nn.Linear(d, d))
        self.motion_position = nn.Parameter(torch.randn(1, self.motion_history_count, d) * .02)
        self.rank_queries = nn.Parameter(torch.randn(1, self.kv_rank, d) * .02)
        self.observation_norm = nn.LayerNorm(d)
        self.observation_k = nn.Linear(d, d, bias=False)
        self.observation_v = nn.Linear(d, d, bias=False)
        # Mean over heads preserves each native head-channel position while
        # avoiding a Nv x 2048 x 192 projection on every fast observation.
        self.cache_key = nn.ModuleList(nn.Linear(world_head_dim, d, bias=False) for _ in range(layers))
        self.cache_value = nn.ModuleList(nn.Linear(world_head_dim, d, bias=False) for _ in range(layers))
        self.rank_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.action_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.action_k = nn.ModuleList(nn.Linear(dim, d, bias=False) for _ in range(layers))
        self.action_v = nn.ModuleList(nn.Linear(dim, d, bias=False) for _ in range(layers))
        self.action_q = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(layers))
        self.output_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        width = world_heads * world_head_dim
        self.delta_k = nn.ModuleList(nn.Linear(d, width, bias=False) for _ in range(layers))
        self.delta_v = nn.ModuleList(nn.Linear(d, width, bias=False) for _ in range(layers))
        for projection in (*self.delta_k, *self.delta_v):
            nn.init.zeros_(projection.weight)
        self.constructor_spec = dict(dim=dim, layers=layers, heads=heads, head_dim=head_dim,
            text_dim=text_dim, state_dim=state_dim, sim_scale=sim_scale, action_scale=action_scale,
            world_aware=True, world_layers=list(world_layers), world_heads=world_heads,
            world_head_dim=world_head_dim, world_queries=world_queries, world_dim=world_dim,
            encoder=encoder, motion_kv=True, kv_rank=kv_rank,
            max_relative_update=max_relative_update, motion_history_count=motion_history_count)

    def _attention(self, query, key, value, valid=None):
        b = query.shape[0]
        h, width = self.heads, self.world_dim // self.heads
        q = query.reshape(b, -1, h, width).transpose(1, 2)
        k = key.reshape(b, -1, h, width).transpose(1, 2)
        v = value.reshape(b, -1, h, width).transpose(1, 2)
        mask = None if valid is None else valid[:, None, None, :]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return result.transpose(1, 2).reshape(b, -1, self.world_dim)

    def _motion_tokens(self, current, dtype):
        features = current.motion.float()
        interval = current.motion_interval_valid_mask.bool()
        acceleration = current.motion_acceleration_valid_mask.bool() & interval
        expected = (current.state.shape[0], self.motion_history_count)
        if features.shape != (*expected, 12) or interval.shape != expected or acceleration.shape != expected:
            raise ValueError("motion KV needs native [B,history,12] motion and masks")
        valid = torch.cat((interval[..., None].expand(-1, -1, 9),
                           acceleration[..., None].expand(-1, -1, 3)), -1)
        # Replace invalid values before arithmetic, including invalid NaN/Inf.
        safe = torch.where(valid, features, self.motion_mean.float()[None, None])
        normalized = (safe - self.motion_mean.float()) / self.motion_scale.float()
        normalized = normalized.sign() * torch.log1p(normalized.abs())
        inputs = torch.cat((normalized, interval[..., None].float(), acceleration[..., None].float()), -1)
        return self.motion_encoder(inputs.to(dtype)) + self.motion_position.to(dtype)

    def _observation_tokens(self, current, text, text_mask=None, *, bridge=None):
        visual, visual_valid = self.encoder(current.rgb, current.rgb_valid_mask)
        dtype, device, batch = visual.dtype, visual.device, visual.shape[0]
        language = self.text_pool(text, text_mask)
        state = self.state_encoder(current.state.to(dtype))[:, None]
        age_features = current.age.features(sim_scale=self.sim_scale, action_scale=self.action_scale)
        if age_features.ndim == 1:
            age_features = age_features[None].expand(batch, -1)
        age = self.age_mlp(age_features.to(device=device, dtype=dtype))[:, None]
        motion = self._motion_tokens(current, dtype)
        observed = torch.cat((visual, language, state, age, motion), 1)
        # Invalid intervals still carry explicit validity/position tokens.
        extra = torch.ones((batch, language.shape[1] + 2 + self.motion_history_count),
                           device=device, dtype=torch.bool)
        observed_valid = torch.cat((visual_valid, extra), 1)
        return observed, observed_valid

    def _rank_observation(self, seeds, observed, observed_valid, *, current, bridge):
        observed = self.observation_norm(observed)
        return seeds + self._attention(seeds, self.observation_k(observed),
                                       self.observation_v(observed), observed_valid)

    def prepare_memory(self, current, text, text_mask=None, *, bridge=None):
        if bridge is None or len(bridge.video_k) != self.layers or len(bridge.video_v) != self.layers:
            raise ValueError("motion KV requires one frozen world K/V pair per native layer")
        observed, observed_valid = self._observation_tokens(current, text, text_mask, bridge=bridge)
        dtype, device, batch = observed.dtype, observed.device, observed.shape[0]
        seeds = self.rank_queries.to(dtype).expand(batch, -1, -1)
        rank_observation = self._rank_observation(seeds, observed, observed_valid,
                                                current=current, bridge=bridge)
        null = torch.as_tensor(bridge.null_bridge, device=device, dtype=torch.bool).reshape(-1)
        if null.numel() == 1:
            null = null.expand(batch)
        if null.numel() != batch:
            raise ValueError("null bridge mask batch mismatch")
        lengths = bridge.seq_lens.to(device=device).reshape(batch)
        assignments, contexts, validity, key_rms, value_rms = [], [], [], [], []
        for layer, (key, value) in enumerate(zip(bridge.video_k, bridge.video_v, strict=True)):
            if key.shape != value.shape or key.ndim != 4 or key.shape[0] != batch or key.shape[2:] != (self.world_heads, self.world_head_dim):
                raise ValueError("world cache shape differs from motion KV configuration")
            keep = (torch.arange(key.shape[1], device=device)[None] < lengths[:, None]) & ~null[:, None]
            safe_key = torch.where(keep[..., None, None], key, torch.zeros_like(key))
            safe_value = torch.where(keep[..., None, None], value, torch.zeros_like(value))
            compressed_key = self.cache_key[layer](safe_key.float().mean(2).to(dtype))
            compressed_value = self.cache_value[layer](safe_value.float().mean(2).to(dtype))
            scores = torch.matmul(compressed_key, rank_observation.transpose(1, 2)) / math.sqrt(self.world_dim)
            assignment = scores.float().softmax(-1).to(dtype) * keep[..., None].to(dtype)
            mass = assignment.float().sum(1).clamp_min(1e-6)
            read = torch.matmul(assignment.transpose(1, 2), compressed_value).float() / mass[..., None]
            contexts.append(self.rank_norm[layer](rank_observation + read.to(dtype)))
            assignments.append(assignment)
            validity.append(keep)
            # Frozen reference defines a per-token/per-head bound. A zero
            # reference head receives zero correction; no arbitrary RMS floor.
            key_rms.append(key.detach().float().square().mean(-1, keepdim=True).sqrt())
            value_rms.append(value.detach().float().square().mean(-1, keepdim=True).sqrt())
        return MotionKVMemory(assignments, contexts, validity, key_rms, value_rms)

    def adapt_video_kv(self, layer_idx, action_tokens, memory, base_key, base_value):
        layer = int(layer_idx)
        context = memory.rank_context[layer]
        tokens = self.action_norm[layer](action_tokens.to(self.action_k[layer].weight.dtype))
        read = self._attention(self.action_q[layer](context), self.action_k[layer](tokens),
                               self.action_v[layer](tokens))
        updated = self.output_norm[layer](context + read)
        shape = (*base_key.shape[:2], self.world_heads, self.world_head_dim)
        assignment = memory.assignments[layer]
        delta_k = torch.matmul(assignment, self.delta_k[layer](updated)).reshape(shape)
        delta_v = torch.matmul(assignment, self.delta_v[layer](updated)).reshape(shape)
        keep = memory.world_valid[layer][..., None, None]
        # Elementwise bounded low-rank parameterization guarantees RMS(delta)
        # <= max_relative_update * RMS(base) for every token and native head.
        dk = delta_k.float().tanh() * (self.max_relative_update * memory.key_rms[layer])
        dv = delta_v.float().tanh() * (self.max_relative_update * memory.value_rms[layer])
        dk = torch.where(keep, dk, torch.zeros_like(dk))
        dv = torch.where(keep, dv, torch.zeros_like(dv))
        return base_key + dk.to(base_key.dtype), base_value + dv.to(base_value.dtype)

    def apply_layer(self, layer_idx, tokens, memory):
        """No post-attention actuator residual; native action/motion path stays."""
        return tokens
