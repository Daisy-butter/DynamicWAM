"""Compact joint action/interaction-future FM model, with no native KV updater.

Absolute times are seconds and must arrive as float64 tensors. Time differences
are computed before any model-dtype conversion. Ground-truth future labels,
visibility and contact never enter this model's conditioning interface.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastConfig:
    dim: int = 384
    layers: int = 6
    heads: int = 6
    action_dim: int = 14
    chunk_size: int = 16
    future_dim: int = 134
    future_tokens: int = 16
    observation_dim: int = 384
    world_dim: int = 384
    future_entities: int = 4
    mlp_ratio: int = 4

    def __post_init__(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.heads or self.future_tokens % self.future_entities:
            raise ValueError("dim must be divisible by heads; future_tokens by future_entities")


def _absolute_time(value: Tensor, name: str) -> Tensor:
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must be float64 absolute seconds")
    return value


class _TimeFeatures(nn.Module):
    """Continuous physical time, independent of arbitrary absolute clock origin."""

    def __init__(self, inputs: int, dim: int):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor([0.25, 1.0, 4.0, 16.0]))
        self.projection = nn.Sequential(nn.Linear(inputs * 9, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, relative_seconds: Tensor) -> Tensor:
        x = torch.nan_to_num(relative_seconds.float(), nan=0.0, posinf=60.0, neginf=-60.0).clamp(-60, 60)
        phase = x.unsqueeze(-1) * self.frequencies.float() * math.pi
        features = torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)
        return self.projection(features.to(self.projection[0].weight.dtype))


class _JointBlock(nn.Module):
    def __init__(self, cfg: FastConfig):
        super().__init__()
        self.heads = cfg.heads
        self.norm_self = nn.LayerNorm(cfg.dim)
        self.norm_cross = nn.LayerNorm(cfg.dim)
        self.norm_ff = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.self_output = nn.Linear(cfg.dim, cfg.dim)
        self.query = nn.Linear(cfg.dim, cfg.dim)
        self.context_kv = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.cross_output = nn.Linear(cfg.dim, cfg.dim)
        self.ff = nn.Sequential(nn.Linear(cfg.dim, cfg.dim * cfg.mlp_ratio), nn.GELU(), nn.Linear(cfg.dim * cfg.mlp_ratio, cfg.dim))

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

    @staticmethod
    def _merge(x: Tensor) -> Tensor:
        return x.transpose(1, 2).flatten(2)

    def forward(self, x: Tensor, token_valid: Tensor, context: Tensor, allowed: Tensor) -> Tensor:
        # Both action and Z tokens participate in every denoising step.
        q, k, v = self.qkv(self.norm_self(x)).chunk(3, dim=-1)
        mixed = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v), attn_mask=token_valid[:, None, None, :])
        x = x + self.self_output(self._merge(mixed))
        k, v = self.context_kv(context).chunk(2, dim=-1)
        mixed = F.scaled_dot_product_attention(self._heads(self.query(self.norm_cross(x))), self._heads(k), self._heads(v), attn_mask=allowed[:, None])
        x = x + self.cross_output(self._merge(mixed))
        x = x + self.ff(self.norm_ff(x))
        return torch.where(token_valid[..., None], x, torch.zeros_like(x))


class FastJointWAM(nn.Module):
    """Predict FM velocities for A and Z jointly; sample/inpaint outside forward.

    ``future_valid`` is an OPTIONAL query-time known slot/request mask. It must
    never be populated from future visibility, projection or label availability.
    Label masks belong only to the loss. Future order is entity-major/time-minor.

    ``known_actions`` and ``action_known_mask`` encode the already committed
    prefix (or explicitly selected action-conditioned training mode). A sampler
    must also clamp those actions at every integration step. Slow reference
    actions are soft world tokens, never this hard constraint.

    World visibility below gates direct cross-attention by target time; joint
    A/Z attention may propagate information between valid generated tokens.
    Entirely expired or not-yet-completed references have no information path.
    """

    def __init__(self, config: FastConfig):
        super().__init__()
        self.config = config
        c = config
        self.action_in = nn.Linear(c.action_dim, c.dim)
        self.future_in = nn.Linear(c.future_dim, c.dim)
        self.observation_in = nn.Linear(c.observation_dim, c.dim)
        self.world_in = nn.Linear(c.world_dim, c.dim)
        self.type_embedding = nn.Embedding(4, c.dim)  # action, future, observation, world
        self.known_embedding = nn.Embedding(2, c.dim)
        self.action_position = nn.Embedding(c.chunk_size, c.dim)
        self.entity_embedding = nn.Embedding(c.future_entities, c.dim)
        self.future_position = nn.Embedding(c.future_tokens // c.future_entities, c.dim)
        self.query_time_embedding = _TimeFeatures(4, c.dim)
        self.world_time_embedding = _TimeFeatures(2, c.dim)
        self.sigma_embedding = _TimeFeatures(1, c.dim)
        self.blocks = nn.ModuleList([_JointBlock(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.dim)
        self.action_out = nn.Linear(c.dim, c.action_dim)
        self.future_out = nn.Linear(c.dim, c.future_dim)
        self.contact_out = nn.Linear(c.dim, 2)

    def forward(
        self, actions: Tensor, future: Tensor, sigma: Tensor, *,
        observation_tokens: Tensor, observation_valid: Tensor,
        world_tokens: Tensor, world_valid: Tensor, world_intervals: Tensor,
        target_times: Tensor, query_time: Tensor, source_time: Tensor,
        completed_time: Tensor, future_valid: Optional[Tensor] = None,
        action_known_mask: Optional[Tensor] = None,
        known_actions: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        c = self.config
        batch = actions.shape[0]
        device = actions.device
        length = c.chunk_size + c.future_tokens
        if actions.shape != (batch, c.chunk_size, c.action_dim) or future.shape != (batch, c.future_tokens, c.future_dim):
            raise ValueError("action/future shape does not match FastConfig")
        if sigma.shape != (batch,) or target_times.shape != (batch, length):
            raise ValueError("sigma/target_times shape mismatch")
        if observation_tokens.shape[:2] != observation_valid.shape or world_tokens.shape[:2] != world_valid.shape:
            raise ValueError("context validity masks must match their token arrays")
        if world_intervals.shape != (*world_valid.shape, 2):
            raise ValueError("world_intervals must be [B, M, 2]")
        for name, value in (("query_time", query_time), ("source_time", source_time), ("completed_time", completed_time)):
            _absolute_time(value, name)
            if value.shape != (batch,):
                raise ValueError(f"{name} must be [B]")
        _absolute_time(target_times, "target_times")
        _absolute_time(world_intervals, "world_intervals")
        if future_valid is None:
            future_valid = torch.ones((batch, c.future_tokens), dtype=torch.bool, device=device)
        if future_valid.shape != (batch, c.future_tokens):
            raise ValueError("future_valid must be [B, Z]")
        if (known_actions is None) != (action_known_mask is None):
            raise ValueError("known_actions and action_known_mask must be supplied together")
        if action_known_mask is None:
            action_known_mask = torch.zeros((batch, c.chunk_size), dtype=torch.bool, device=device)
        else:
            if known_actions.shape != actions.shape or action_known_mask.shape != actions.shape[:2]:
                raise ValueError("known action constraint shape mismatch")
            actions = torch.where(action_known_mask[..., None], known_actions, actions)
        future = torch.where(future_valid[..., None], future, torch.zeros_like(future))
        action_ids = torch.arange(c.chunk_size, device=device)
        future_ids = torch.arange(c.future_tokens, device=device)
        nt = c.future_tokens // c.future_entities
        a = self.action_in(actions.to(self.action_in.weight.dtype)) + self.type_embedding.weight[0] + self.action_position(action_ids) + self.known_embedding(action_known_mask.long())
        z = self.future_in(future.to(self.future_in.weight.dtype)) + self.type_embedding.weight[1] + self.entity_embedding(future_ids // nt) + self.future_position(future_ids % nt)
        age = query_time - source_time
        latency = completed_time - source_time
        relative = torch.stack((target_times - query_time[:, None], target_times - source_time[:, None], age[:, None].expand(-1, length), latency[:, None].expand(-1, length)), dim=-1)
        x = torch.cat((a, z), dim=1) + self.query_time_embedding(relative) + self.sigma_embedding(sigma[:, None])[:, None]
        token_valid = torch.cat((torch.ones_like(action_known_mask), future_valid.bool()), dim=1)

        # Null context ensures finite SDPA even with no observations/cache.
        obs = self.observation_in(torch.where(observation_valid[..., None], observation_tokens, torch.zeros_like(observation_tokens)).to(self.observation_in.weight.dtype)) + self.type_embedding.weight[2]
        available = (completed_time <= query_time) & (source_time <= completed_time)
        eligible = world_valid.bool() & available[:, None]
        start, end = world_intervals.unbind(-1)
        eligible = eligible & (end >= start) & (end >= query_time[:, None])
        world = self.world_in(torch.where(eligible[..., None], world_tokens, torch.zeros_like(world_tokens)).to(self.world_in.weight.dtype)) + self.type_embedding.weight[3]
        world = world + self.world_time_embedding(world_intervals - query_time[:, None, None])
        context = torch.cat((x.new_zeros((batch, 1, c.dim)), obs, world), dim=1)
        world_allowed = eligible[:, None] & (target_times[:, :, None] >= start[:, None]) & (target_times[:, :, None] <= end[:, None])
        allowed = torch.cat((torch.ones((batch, length, 1), device=device, dtype=torch.bool), observation_valid[:, None].expand(-1, length, -1).bool(), world_allowed), dim=-1)
        for block in self.blocks:
            x = block(x, token_valid, context, allowed)
        x = self.norm(x)
        action_velocity = self.action_out(x[:, :c.chunk_size])
        future_velocity = self.future_out(x[:, c.chunk_size:])
        contact = self.contact_out(x[:, c.chunk_size:])
        return {
            "action_velocity": torch.where(action_known_mask[..., None], torch.zeros_like(action_velocity), action_velocity),
            "future_velocity": torch.where(future_valid[..., None], future_velocity, torch.zeros_like(future_velocity)),
            "contact_logits": torch.where(future_valid[..., None], contact, torch.zeros_like(contact)),
        }


class CurrentObservationEncoder(nn.Module):
    """Trainable FROM-SCRATCH spatial RGB encoder, not a pretrained backbone.

    RGB is normalized floating point [B, views, 3, H, W] in [0, 1]. Each
    non-overlapping 32x32 image region remains a token, with explicit x/y and
    camera identity. No global pooling. ``motion`` is the 4x12 native motion
    feature history; interval endpoints and invalid history are supplied by the
    caller. Language tokens are frozen-encoder outputs, not trainable labels.
    """

    def __init__(self, dim: int = 384, views: int = 3, motion_dim: int = 12, state_dim: int = 14, text_dim: int = 4096):
        super().__init__()
        self.dim, self.views = dim, views
        self.pretrained = False
        hidden = max(16, dim // 4)
        self.rgb = nn.Sequential(nn.Conv2d(3, hidden, 4, 4), nn.GELU(), nn.Conv2d(hidden, hidden * 2, 4, 4), nn.GELU(), nn.Conv2d(hidden * 2, dim, 2, 2))
        self.xy = nn.Linear(2, dim)
        self.view_embedding = nn.Embedding(views, dim)
        self.type_embedding = nn.Embedding(4, dim)  # RGB, motion, state, language
        self.time_embedding = _TimeFeatures(2, dim)
        self.motion = nn.Linear(motion_dim, dim)
        self.motion_acceleration_embedding = nn.Embedding(2, dim)
        self.state = nn.Linear(state_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, rgb: Tensor, motion: Tensor, state: Tensor, text_tokens: Tensor, *,
        query_time: Tensor, rgb_times: Tensor, motion_times: Tensor,
        motion_valid: Tensor, text_valid: Tensor, rgb_valid: Optional[Tensor] = None,
        state_time: Optional[Tensor] = None,
        motion_acceleration_valid: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (self.views, 3) or min(rgb.shape[-2:]) < 32:
            raise ValueError("RGB must be [B, views, 3, H>=32, W>=32]")
        if not rgb.is_floating_point():
            raise TypeError("RGB must be normalized floating point in [0,1]")
        batch = rgb.shape[0]
        _absolute_time(query_time, "query_time")
        _absolute_time(rgb_times, "rgb_times")
        _absolute_time(motion_times, "motion_times")
        if rgb_times.shape != (batch, self.views) or motion_times.shape != (*motion.shape[:2], 2):
            raise ValueError("RGB timestamps are [B,V]; motion intervals are [B,T,2]")
        if motion_valid.shape != motion.shape[:2] or text_valid.shape != text_tokens.shape[:2]:
            raise ValueError("motion/text masks must match tokens")
        if state_time is None:
            state_time = query_time
        _absolute_time(state_time, "state_time")
        rv = (rgb_times <= query_time[:, None]) & torch.isfinite(rgb_times)
        if rgb_valid is not None:
            rv = rv & rgb_valid.bool()
        image = torch.where(rv[..., None, None, None], rgb, torch.zeros_like(rgb))
        patches = self.rgb(image.flatten(0, 1).to(self.rgb[0].weight.dtype))
        height, width = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2).reshape(batch, self.views, height * width, self.dim)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=rgb.device), torch.linspace(-1, 1, width, device=rgb.device), indexing="ij")
        xy = self.xy(torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(self.xy.weight.dtype))
        image_dt = (rgb_times - query_time[:, None]).unsqueeze(-1).expand(-1, -1, 2)
        patches = patches + xy + self.view_embedding.weight[None, :, None] + self.time_embedding(image_dt)[:, :, None] + self.type_embedding.weight[0]
        patches = patches.flatten(1, 2)
        image_valid = rv[..., None].expand(-1, -1, height * width).flatten(1)
        mv = motion_valid.bool() & torch.isfinite(motion_times).all(-1) & (motion_times[..., 0] <= motion_times[..., 1]) & (motion_times[..., 1] <= query_time[:, None])
        # Missing acceleration validity never implies valid observed acceleration.
        # Caller normalizes in FP32 first; invalid normalized fields become zero.
        if motion_acceleration_valid is None:
            motion_acceleration_valid = torch.zeros_like(mv)
        if motion_acceleration_valid.shape != mv.shape:
            raise ValueError("motion_acceleration_valid must match [B,T] intervals")
        av = motion_acceleration_valid.bool() & mv
        fields = mv[..., None].expand_as(motion).clone()
        fields[..., 9:12] &= av[..., None]
        motion_features = self.motion(torch.where(fields, motion, torch.zeros_like(motion)).to(self.motion.weight.dtype)) + self.time_embedding(motion_times - query_time[:, None, None]) + self.type_embedding.weight[1] + self.motion_acceleration_embedding(av.long())
        sv = torch.isfinite(state_time) & (state_time <= query_time)
        state_dt = (state_time - query_time)[:, None].expand(-1, 2)
        state_features = self.state(torch.where(sv[:, None], state, torch.zeros_like(state)).to(self.state.weight.dtype)) + self.time_embedding(state_dt) + self.type_embedding.weight[2]
        text_features = self.text(torch.where(text_valid[..., None], text_tokens, torch.zeros_like(text_tokens)).to(self.text.weight.dtype)) + self.type_embedding.weight[3]
        tokens = self.norm(torch.cat((patches, motion_features, state_features[:, None], text_features), dim=1))
        valid = torch.cat((image_valid, mv, sv[:, None], text_valid.bool()), dim=1)
        return torch.where(valid[..., None], tokens, torch.zeros_like(tokens)), valid
