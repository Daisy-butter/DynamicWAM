"""Once-per-observation world fusion and an independent compact action flow.

This model never owns a WAN, VAE or text encoder. World K/V are inputs, not
parameters. The denoising loop only reads the prepared small memory. Ten steps
remain the default; reducing them is a separate experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.models.flow_schedule import BridgeSchedule, euler_update
from dynamicwam.models.multirate_types import BridgeTensorBundle, CurrentObservation
from dynamicwam.models.fast_observation_encoder import TextTokenPool


@dataclass(frozen=True)
class CompactFastConfig:
    dim: int = 384
    heads: int = 6
    layers: int = 4
    memory_tokens: int = 32
    world_layers: tuple[int, ...] = (3, 7, 11)
    world_heads: int = 16
    world_head_dim: int = 128
    text_dim: int = 4096
    state_dim: int = 14
    action_dim: int = 14
    chunk_size: int = 16
    sim_scale: float = 1.0
    action_scale: float = 16.0
    encoder: str = "compact_gn"
    temporal_world_memory: bool = False
    temporal_seconds_per_action: float = 0.0  # zero keeps legacy action-index semantics

    feedback_enabled: bool = False
    feedback_channels: int = 48

    def __post_init__(self):
        if not math.isfinite(self.temporal_seconds_per_action) or self.temporal_seconds_per_action < 0:
            raise ValueError("invalid temporal seconds scale")
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("compact dim must be divisible by heads")
        if min(self.layers, self.memory_tokens, self.chunk_size) < 1:
            raise ValueError("compact depths and lengths must be positive")
        if not self.world_layers or min(self.world_layers) < 0:
            raise ValueError("select at least one non-negative world layer")
        if min(self.world_heads, self.world_head_dim, self.state_dim, self.action_dim, self.text_dim) < 1:
            raise ValueError("compact input and world dimensions must be positive")
        if self.sim_scale <= 0 or self.action_scale <= 0:
            raise ValueError("compact age scales must be positive")
        if len(set(self.world_layers)) != len(self.world_layers):
            raise ValueError("world layers must be unique")


class CompactVision(nn.Module):
    """Spatial, mask-aware encoder; optional offline ResNet18 initialization.

    compact_gn is deliberately labelled scratch. resnet18 accepts a strict
    torchvision state dict through load_pretrained, never downloads weights.
    Pretraining is an explicit training choice, not inferred from normalization.
    """
    def __init__(self, dim: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "compact_gn":
            stages = []
            channels = (3, 32, 64, 128, 192)
            for a, b in zip(channels, channels[1:]):
                stages += [nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, b), nn.SiLU()]
            self.body = nn.Sequential(*stages)
            output_dim = 192
        elif kind == "resnet18":
            from torchvision.models import resnet18
            net = resnet18(weights=None)
            self.body = nn.Sequential(*list(net.children())[:-2])
            output_dim = 512
        else:
            raise ValueError(f"unknown fast encoder {kind!r}")
        self.projection = nn.Linear(output_dim, dim)
        self.view_position = nn.Parameter(torch.randn(1, 3, 1, dim) * .02)
        self.spatial_position = nn.Parameter(torch.randn(1, 1, 16, dim) * .02)
        self.train(True)

    def load_pretrained(self, path: str):
        if self.kind != "resnet18":
            raise ValueError("pretrained encoder weights require resnet18")
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.body.load_state_dict(nn.Sequential(*list(model.children())[:-2]).state_dict(), strict=True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, rgb, mask=None):
        batch, views, channels, height, width = rgb.shape
        if views != 3 or channels != 3:
            raise ValueError("expected three RGB views")
        x = rgb.reshape(batch * views, 3, height, width)
        if mask is None:
            mask = torch.ones((batch, views, height, width), device=rgb.device, dtype=torch.bool)
        if mask.shape != (batch, views, height, width):
            raise ValueError("RGB valid mask shape mismatch")
        m = mask.reshape(batch * views, 1, height, width).float()
        x = self.body((x * m).to(dtype=next(self.body.parameters()).dtype))
        m = F.interpolate(m, size=x.shape[-2:], mode="area").to(x.dtype)
        mass = F.adaptive_avg_pool2d(m, (4, 4))
        pooled = F.adaptive_avg_pool2d(x * m, (4, 4)) / mass.clamp_min(1e-6)
        tokens = self.projection(pooled.flatten(2).transpose(1, 2))
        tokens = tokens.reshape(batch, 3, 16, -1) + self.view_position + self.spatial_position
        valid = (mass.flatten(1) > 0).reshape(batch, 48)
        return tokens.reshape(batch, 48, -1), valid


def _bridge_keep(null_bridge, reference):
    """Per-row multiplier. A dropped row contributes no world residual and stays in the graph."""
    if torch.is_tensor(null_bridge):
        keep = (~null_bridge.to(dtype=torch.bool)).to(dtype=reference.dtype)
        if keep.ndim == 0 or int(keep.numel()) == 1:
            return keep.reshape(())
        return keep.reshape(-1, *([1] * (reference.ndim - 1)))
    return reference.new_zeros(()) if null_bridge else reference.new_ones(())


def attend(q, k, v, heads, valid=None):
    q = q.to(k.dtype)
    v = v.to(k.dtype)
    b, nq, d = q.shape
    q = q.view(b, nq, heads, d // heads).transpose(1, 2)
    k = k.reshape(b, -1, heads, d // heads).transpose(1, 2)
    v = v.reshape(b, -1, heads, d // heads).transpose(1, 2)
    mask = None if valid is None else valid[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, nq, d)


class CompactActionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.self_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.memory_kv = nn.Linear(dim, dim * 2)
        self.cross_out = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, x, memory):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.self_out(attend(q, k, v, self.heads))
        x = x + self.cross_out(attend(self.query(self.norm2(x)), *memory, self.heads))
        return x + self.ffn(self.norm3(x))


@dataclass
class CompactMemory:
    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    state: torch.Tensor
    tokens: torch.Tensor | None = None


class CompactFastPolicy(nn.Module):
    def __init__(self, config: CompactFastConfig, *, motion_history_count: int,
                 motion_mean, motion_scale, schedule: BridgeSchedule | None = None):
        super().__init__()
        self.config = config
        self.schedule = schedule or BridgeSchedule.from_flow_match()
        c, d = config, config.dim
        self.encoder = CompactVision(d, c.encoder)
        self.text_pool = TextTokenPool(c.text_dim, d, queries=4)
        self.motion = AbsoluteMotionTokenModule(dim=d, history_count=motion_history_count,
                                                feature_mean=tuple(motion_mean), feature_scale=tuple(motion_scale))
        self.state_encoder = nn.Linear(c.state_dim, d)
        self.age_encoder = nn.Sequential(nn.Linear(c.state_dim + 3, d), nn.SiLU(), nn.Linear(d, d))
        self.fusion_queries = nn.Parameter(torch.randn(1, c.memory_tokens, d) * .02)
        self.current_kv = nn.Linear(d, d * 2)
        self.current_out = nn.Linear(d, d)
        world_dim = c.world_heads * c.world_head_dim
        self.world_queries = nn.ModuleList(nn.Linear(d, world_dim) for _ in c.world_layers)
        self.world_outputs = nn.ModuleList(nn.Linear(world_dim, d) for _ in c.world_layers)
        if c.temporal_world_memory:
            self.world_time_encoder = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        self.fusion_norm = nn.LayerNorm(d)
        self.fusion_ffn = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))
        self.action_encoder = nn.Linear(c.action_dim, d)
        self.action_position = nn.Parameter(torch.randn(1, c.chunk_size, d) * .02)
        self.time_encoder = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(CompactActionBlock(d, c.heads) for _ in range(c.layers))
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, c.action_dim)
        if c.feedback_enabled:
            from dynamicwam.models.action_feedback import ActionConditionedForecast
            self.feedback_dynamics = ActionConditionedForecast(d,c.heads,c.action_dim,c.chunk_size,c.feedback_channels)
            self.feedback_projection = nn.Linear(c.feedback_channels,d)
            self.feedback_position = nn.Parameter(torch.randn(1,32,d)*.02)
        self.last_feedback_future = None
        self.feedback_disabled = False

    def prepare_memory(self, current: CurrentObservation, bridge: BridgeTensorBundle,
                       text, text_mask=None) -> CompactMemory:
        dtype = self.state_encoder.weight.dtype
        if self.config.feedback_enabled:
            # Null/stale WAM memory has no meaningful age or source-state delta.
            # Canonicalize it identically in training and online fallback.
            keep = _bridge_keep(bridge.null_bridge, current.state).reshape(-1)
            if keep.numel() == 1: keep = keep.expand(current.state.shape[0])
            age = current.age
            current = replace(current,age=replace(age,
                delta_sim_seconds=torch.as_tensor(age.delta_sim_seconds,device=current.state.device)*keep,
                delta_actions=torch.as_tensor(age.delta_actions,device=current.state.device)*keep,
                bridge_valid=keep.bool(),state_delta=age.state_delta*keep[:,None]))
        visual, valid = self.encoder(current.rgb, current.rgb_valid_mask)
        state = self.state_encoder(current.state.to(dtype))
        motion = self.motion(current.motion, current.motion_interval_valid_mask,
                             current.motion_acceleration_valid_mask, dtype=dtype, device=state.device)
        age = self.age_encoder(current.age.features(sim_scale=self.config.sim_scale,
                            action_scale=self.config.action_scale).to(device=state.device, dtype=dtype))
        if age.ndim == 1:
            age = age.unsqueeze(0).expand(state.shape[0], -1)
        conditions = torch.cat((visual.to(dtype), self.text_pool(text, text_mask).to(dtype),
                                motion, state[:, None], age[:, None]), dim=1)
        valid = torch.cat((valid, torch.ones((valid.shape[0], conditions.shape[1]-valid.shape[1]),
                                             device=valid.device, dtype=torch.bool)), dim=1)
        k, v = self.current_kv(conditions).chunk(2, dim=-1)
        memory = self.fusion_queries.to(dtype).expand(state.shape[0], -1, -1) + state[:, None]
        memory = memory + self.current_out(attend(memory, k, v, self.config.heads, valid))
        temporal_mask, world_keep = None, 1.0
        if self.config.temporal_world_memory:
            from dynamicwam.models.temporal_world_memory import temporal_world_layout
            temporal_age = current.age.delta_actions
            if self.config.temporal_seconds_per_action > 0:
                temporal_age = torch.as_tensor(current.age.delta_sim_seconds, device=memory.device) / self.config.temporal_seconds_per_action
            features, temporal_mask, world_keep = temporal_world_layout(
                bridge, temporal_age, queries=self.config.memory_tokens,
                horizon_actions=self.config.chunk_size, device=memory.device, dtype=dtype)
            # Rebase forecast intervals to this query, after fusing current RGB.
            memory = memory + self.world_time_encoder(features)
        # Always execute the selected projections, even for a dropped bridge,
        # so distributed training has one stable parameter graph.
        for i, layer in enumerate(self.config.world_layers):
            if layer >= len(bridge.video_k):
                raise ValueError(f"bridge lacks configured world layer {layer}")
            wk, wv = bridge.video_k[layer], bridge.video_v[layer]
            b, n, heads, head_dim = wk.shape
            if (heads, head_dim) != (self.config.world_heads, self.config.world_head_dim):
                raise ValueError("world bridge head layout mismatch")
            q = self.world_queries[i](memory).reshape(b, -1, heads, head_dim).transpose(1, 2)
            valid_world = torch.arange(n, device=wk.device)[None] < bridge.seq_lens[:, None]
            value = F.scaled_dot_product_attention(q, wk.to(q.dtype).transpose(1, 2),
                         wv.to(q.dtype).transpose(1, 2),
                         attn_mask=temporal_mask if temporal_mask is not None else valid_world[:, None, None])
            value = value.transpose(1, 2).flatten(2)
            correction = self.world_outputs[i](value)
            memory = memory + correction * world_keep * _bridge_keep(bridge.null_bridge, correction)
        memory = memory + self.fusion_ffn(self.fusion_norm(memory))
        return CompactMemory(tuple(block.memory_kv(memory).chunk(2, dim=-1) for block in self.blocks), state, memory)

    def predict_velocity(self, actions, timestep, memory: CompactMemory):
        c = self.config
        if actions.shape[1:] != (c.chunk_size, c.action_dim):
            raise ValueError("compact action shape mismatch")
        dtype = self.action_encoder.weight.dtype
        t = timestep.float().reshape(actions.shape[0], 1)
        frequency = torch.exp(-math.log(10000.0) * torch.arange(c.dim // 2, device=t.device) / (c.dim // 2))
        phase = t * frequency[None]
        emb = torch.cat((phase.cos(), phase.sin()), dim=-1)
        if emb.shape[-1] < c.dim:
            emb = F.pad(emb, (0, 1))
        time = self.time_encoder(emb.to(dtype))
        x = self.action_encoder(actions.to(dtype)) + self.action_position.to(dtype)
        x = x + time[:, None] + memory.state[:, None]
        for block, kv in zip(self.blocks, memory.layer_kv):
            x = block(x, kv)
        return self.out(self.out_norm(x))

    def feedback_memory(self, memory, future):
        if memory.tokens is None: raise RuntimeError('feedback requires compact context tokens')
        tokens = torch.cat((memory.tokens, self.feedback_projection(future) + self.feedback_position),dim=1)
        return CompactMemory(tuple(b.memory_kv(tokens).chunk(2,dim=-1) for b in self.blocks),memory.state,tokens)

    def sample_feedback(self, noise, memory):
        # Distilled endpoint map; do not apply this solver to old FM checkpoints.
        t = self.schedule.action_timesteps[0].to(noise).reshape(1).expand(noise.shape[0])
        proposal = noise - self.predict_velocity(noise,t,memory)
        future = self.feedback_dynamics(memory.tokens,proposal)
        corrected_memory = memory if self.feedback_disabled else self.feedback_memory(memory,future)
        actions = noise - self.predict_velocity(noise,t,corrected_memory)
        # The published forecast corresponds to the FINAL selected action plan.
        self.last_feedback_future = self.feedback_dynamics(memory.tokens,actions).detach()
        return actions, proposal, future

    def sample_actions(self, noise, memory, *, denormalize=None):
        if self.config.feedback_enabled:
            actions,_,_=self.sample_feedback(noise,memory)
            return denormalize(actions) if denormalize is not None else actions
        actions = noise
        for index in range(len(self.schedule.action_sigmas)):
            sigma, nxt, t = self.schedule.action_step(index)
            t = t.to(device=actions.device).reshape(1).expand(actions.shape[0])
            actions = euler_update(actions, self.predict_velocity(actions, t, memory), sigma, nxt)
        return actions if denormalize is None else denormalize(actions)
