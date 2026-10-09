"""Current-observation routing of a cached world prior, once per query.

The raw WAM cache remains unchanged. A bounded set of current-observation
queries reads selected world layers and supplies low-rank residual context to
the native action expert. Cache age is a feature, not an invented timestamp
for a VAE latent bin. No candidate-action counterfactual claim is made here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.compact_fast_policy import CompactVision
from dynamicwam.models.fast_conditioner import FastConditioner, FastConditionMemory
from dynamicwam.models.fast_observation_encoder import TextTokenPool


class NativeWorldResidual(nn.Module):
    """Small query/memory attention with an unchanged native-width output."""

    def __init__(self, dim, residual_dim, heads):
        super().__init__()
        if residual_dim % heads:
            raise ValueError("world_dim must divide by conditioner heads")
        self.heads = int(heads)
        self.residual_dim = int(residual_dim)
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, residual_dim, bias=False)
        self.out_proj = nn.Linear(residual_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, tokens, key, value, key_valid=None):
        batch, length, _ = tokens.shape
        d = self.residual_dim // self.heads
        query = self.q_proj(self.norm(tokens.to(self.q_proj.weight.dtype)))
        query = query.reshape(batch, length, self.heads, d).transpose(1, 2)
        key = key.reshape(batch, -1, self.heads, d).transpose(1, 2)
        value = value.reshape(batch, -1, self.heads, d).transpose(1, 2)
        mask = None if key_valid is None else key_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key.to(query.dtype), value.to(query.dtype),
                                                  attn_mask=mask)
        mixed = attended.transpose(1, 2).reshape(batch, length, self.residual_dim)
        return tokens + self.out_proj(mixed).to(tokens.dtype)


class WorldAwareConditioner(FastConditioner):
    """Native-width residual outputs with a compact observation/world encoder."""

    def __init__(self, *, dim=768, layers=12, heads=8, head_dim=96,
                 text_dim=4096, state_dim=14, sim_scale=1.0, action_scale=16.0,
                 world_aware=True, world_layers=(3, 7, 11), world_heads=16,
                 world_head_dim=128, world_queries=16, world_dim=192,
                 encoder="compact_gn"):
        if not world_aware:
            raise ValueError("WorldAwareConditioner requires world_aware=true")
        if encoder != "compact_gn":
            raise ValueError("native multirate encoder must use compact_gn")
        if min(world_queries, world_dim, world_heads, world_head_dim) < 1:
            raise ValueError("invalid world routing dimensions")
        if not world_layers or len(set(world_layers)) != len(world_layers):
            raise ValueError("select distinct world layers")
        if min(world_layers) < 0:
            raise ValueError("negative world layer")
        super().__init__(dim=dim, layers=layers, heads=heads, head_dim=head_dim,
                         text_dim=text_dim, state_dim=state_dim,
                         sim_scale=sim_scale, action_scale=action_scale)
        self.world_aware = True
        self.world_layers = tuple(int(i) for i in world_layers)
        self.world_heads = int(world_heads)
        self.world_head_dim = int(world_head_dim)
        self.world_queries = int(world_queries)
        self.world_dim = int(world_dim)
        self.layer_k = nn.ModuleList(nn.Linear(dim, world_dim, bias=False) for _ in range(layers))
        self.layer_v = nn.ModuleList(nn.Linear(dim, world_dim, bias=False) for _ in range(layers))
        self.blocks = nn.ModuleList(NativeWorldResidual(dim, world_dim, heads) for _ in range(layers))
        self.encoder = CompactVision(world_dim, encoder)
        self.visual_out = nn.Linear(world_dim, dim)
        self.text_pool = TextTokenPool(text_dim=text_dim, dim=world_dim, queries=4)
        self.text_out = nn.Linear(world_dim, dim)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.context_query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, world_dim))
        self.world_query_position = nn.Parameter(torch.randn(1, world_queries, world_dim) * .02)
        width = world_heads * world_head_dim
        self.world_q = nn.ModuleList(nn.Linear(world_dim, width, bias=False) for _ in world_layers)
        self.world_out = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, world_dim),
                                                   nn.SiLU(), nn.Linear(world_dim, dim)) for _ in world_layers)
        self.world_position = nn.Parameter(torch.randn(len(world_layers), 1, 1, dim) * .02)
        self.constructor_spec = dict(dim=dim, layers=layers, heads=heads, head_dim=head_dim,
            text_dim=text_dim, state_dim=state_dim, sim_scale=sim_scale, action_scale=action_scale,
            world_aware=True, world_layers=list(world_layers), world_heads=world_heads,
            world_head_dim=world_head_dim, world_queries=world_queries, world_dim=world_dim,
            encoder=encoder)

    def prepare_memory(self, current, text, text_mask=None, *, bridge=None):
        if bridge is None:
            raise ValueError("world-aware conditioning requires the source bridge")
        visual, visual_valid = self.encoder(current.rgb, current.rgb_valid_mask)
        visual = self.visual_out(visual)
        pooled = self.text_out(self.text_pool(text, text_mask))
        age = self._age_token(current.age, reference=visual)
        state = self.state_encoder(current.state.to(self.state_encoder[0].weight.dtype)).unsqueeze(1)
        observed = torch.cat((visual, pooled, age, state), dim=1)
        extra = torch.ones((visual.shape[0], pooled.shape[1] + 2), device=visual.device, dtype=torch.bool)
        observed_valid = torch.cat((visual_valid, extra), dim=1)
        weights = observed_valid.to(observed.dtype).unsqueeze(-1)
        summary = (observed * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1)
        queries = self.context_query(summary) + self.world_query_position
        batch = visual.shape[0]
        null = torch.as_tensor(bridge.null_bridge, device=visual.device, dtype=torch.bool).reshape(-1)
        if null.numel() == 1:
            null = null.expand(batch)
        if null.numel() != batch:
            raise ValueError("world null mask batch mismatch")
        routed = []
        for local_layer, world_layer in enumerate(self.world_layers):
            key = bridge.video_k[world_layer]
            value = bridge.video_v[world_layer]
            if key.ndim != 4 or key.shape[2:] != (self.world_heads, self.world_head_dim):
                raise ValueError("world K/V shape differs from conditioner contract")
            q = self.world_q[local_layer](queries).reshape(batch, self.world_queries,
                self.world_heads, self.world_head_dim).transpose(1, 2)
            key = key.to(device=q.device, dtype=q.dtype).transpose(1, 2)
            value = value.to(device=q.device, dtype=q.dtype).transpose(1, 2)
            lengths = bridge.seq_lens.to(device=q.device).reshape(batch)
            valid = torch.arange(key.shape[2], device=q.device)[None] < lengths[:, None]
            # SDPA handles null rows as zeros. Keep current observation/text
            # available so no action residual ever attends an all-masked row.
            valid = valid & ~null[:, None]
            read = F.scaled_dot_product_attention(q, key, value, attn_mask=valid[:, None, None, :])
            read = read.transpose(1, 2).reshape(batch, self.world_queries, -1)
            item = self.world_out[local_layer](read) + self.world_position[local_layer]
            routed.append(item * (~null).to(item.dtype)[:, None, None])
        world = torch.cat(routed, dim=1)
        tokens = torch.cat((observed, world), dim=1)
        world_valid = (~null)[:, None].expand(batch, world.shape[1])
        valid = torch.cat((observed_valid, world_valid), dim=1)
        return FastConditionMemory(tuple(proj(tokens) for proj in self.layer_k),
                                   tuple(proj(tokens) for proj in self.layer_v), valid)
