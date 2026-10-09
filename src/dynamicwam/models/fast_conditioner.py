"""Per-layer residual that reads current vision, text, and bridge age.

World attention stays on the cached video K/V. This residual uses its own
768-d projections. Only the output projection is zero-initialized.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamicwam.models.fast_observation_encoder import (
    FastObservationEncoder,
    TextTokenPool,
)
from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation


class FastResidualBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        if heads * head_dim != dim:
            raise ValueError(f"{heads} * {head_dim} != {dim}")
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dim = tokens.shape
        projection_input = tokens.to(dtype=self.q_proj.weight.dtype)
        query = self.q_proj(self.norm(projection_input)).view(
            batch, length, self.heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_valid is not None:
            if key_valid.shape[:2] != (batch, key.shape[2]):
                raise ValueError(
                    f"fast memory mask {tuple(key_valid.shape)} does not match keys {tuple(key.shape)}"
                )
            attn_mask = key_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key.to(query.dtype),
            value.to(query.dtype),
            attn_mask=attn_mask,
        )
        mixed = attended.transpose(1, 2).reshape(batch, length, dim)
        residual = self.out_proj(mixed).to(dtype=tokens.dtype)
        return tokens + residual


class FastConditionMemory:
    def __init__(
        self,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
        valid: torch.Tensor | None = None,
    ) -> None:
        if len(keys) != len(values) or not keys:
            raise ValueError("fast memory needs one K/V pair per action layer")
        self.keys = keys
        self.values = values
        self.valid = valid


class FastConditioner(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 768,
        layers: int = 12,
        heads: int = 8,
        head_dim: int = 96,
        text_dim: int = 4096,
        state_dim: int = 14,
        sim_scale: float = 1.0,
        action_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.layers = int(layers)
        self.state_dim = int(state_dim)
        self.sim_scale = float(sim_scale)
        self.action_scale = float(action_scale)
        self.encoder = FastObservationEncoder(dim=dim)
        self.text_pool = TextTokenPool(text_dim=text_dim, dim=dim, queries=4)
        age_in = 3 + state_dim
        self.age_mlp = nn.Sequential(
            nn.Linear(age_in, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.layer_k = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.layer_v = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(layers))
        self.blocks = nn.ModuleList(
            FastResidualBlock(dim, heads, head_dim) for _ in range(layers)
        )

    def prepare_memory(
        self,
        current: CurrentObservation,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> FastConditionMemory:
        visual, visual_valid = self.encoder(current.rgb, current.rgb_valid_mask)
        pooled = self.text_pool(text, text_mask)
        age = self._age_token(current.age, reference=visual)
        if age.shape[0] != visual.shape[0]:
            raise ValueError("age batch does not match the current RGB batch")
        memory = torch.cat((visual, pooled, age), dim=1)
        extra = torch.ones(
            (visual_valid.shape[0], pooled.shape[1] + age.shape[1]),
            device=visual_valid.device,
            dtype=torch.bool,
        )
        valid = torch.cat((visual_valid, extra), dim=1)
        keys = tuple(proj(memory) for proj in self.layer_k)
        values = tuple(proj(memory) for proj in self.layer_v)
        # Keep the mask on device, including the all-valid case. Do not reduce
        # it to a Python bool in the denoising loop.
        return FastConditionMemory(keys, values, valid)

    def apply_layer(
        self,
        layer_idx: int,
        tokens: torch.Tensor,
        memory: FastConditionMemory,
    ) -> torch.Tensor:
        return self.blocks[layer_idx](
            tokens,
            memory.keys[layer_idx],
            memory.values[layer_idx],
            memory.valid,
        )

    def _age_token(self, age: BridgeAge, *, reference: torch.Tensor) -> torch.Tensor:
        features = age.features(
            sim_scale=self.sim_scale,
            action_scale=self.action_scale,
        ).to(device=reference.device, dtype=reference.dtype)
        width = 3 + self.state_dim
        if features.ndim == 1:
            if int(features.shape[0]) != width:
                raise ValueError(
                    f"age features must be {width}-D, got {tuple(features.shape)}"
                )
            features = features.unsqueeze(0).expand(reference.shape[0], -1)
        elif features.ndim != 2 or int(features.shape[1]) != width:
            raise ValueError(
                f"age features must be [B, {width}], got {tuple(features.shape)}"
            )
        elif int(features.shape[0]) != int(reference.shape[0]):
            raise ValueError("age batch does not match the current RGB batch")
        return self.age_mlp(features).unsqueeze(1)
