"""Current three-view encoder and text pooling for the fast action path.

BatchNorm follows ``train()`` and ``eval()``. Deployment calls ``eval()`` after
loading, so inference reads the accumulated running stats and does not update them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastObservationEncoder(nn.Module):
    """Map ``[B, 3, 3, H, W]`` views to 16 spatial tokens per view."""

    def __init__(self, dim: int = 768, views: int = 3, grid: int = 4) -> None:
        super().__init__()
        if dim <= 0 or views != 3 or grid != 4:
            raise ValueError("the fast encoder emits 3 views x 4 x 4 tokens")
        self.dim = int(dim)
        self.views = int(views)
        self.grid = int(grid)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.view_embedding = nn.Parameter(torch.zeros(views, dim))
        coords = torch.linspace(-1.0, 1.0, grid)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        spatial = torch.stack((xx, yy), dim=-1).reshape(1, grid * grid, 2)
        self.register_buffer("spatial_coords", spatial, persistent=False)
        self.spatial_proj = nn.Linear(2, dim)
        nn.init.normal_(self.view_embedding, std=0.02)

    def forward(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 5 or int(rgb.shape[1]) != self.views:
            raise ValueError(
                f"expected RGB [B, {self.views}, 3, H, W], got {tuple(rgb.shape)}"
            )
        batch, views, channels, height, width = rgb.shape
        if channels != 3:
            raise ValueError(f"expected 3 RGB channels, got {channels}")
        if mask is None:
            mask = torch.ones(
                (batch, views, height, width),
                device=rgb.device,
                dtype=torch.bool,
            )
        if tuple(mask.shape) != (batch, views, height, width):
            raise ValueError(
                f"RGB valid mask must be {(batch, views, height, width)}, got {tuple(mask.shape)}"
            )
        flat = rgb.reshape(batch * views, channels, height, width)
        valid = mask.reshape(batch * views, 1, height, width).to(device=flat.device, dtype=flat.dtype)
        flat = flat * valid
        features = self.stem(flat.to(dtype=self.spatial_proj.weight.dtype))
        valid = F.interpolate(valid, size=features.shape[-2:], mode="area").to(features.dtype)
        mass = F.adaptive_avg_pool2d(valid, (self.grid, self.grid))
        pooled = F.adaptive_avg_pool2d(features * valid, (self.grid, self.grid)) / mass.clamp_min(1e-6)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, views, self.grid * self.grid, self.dim)
        spatial = self.spatial_proj(self.spatial_coords.to(dtype=tokens.dtype))
        view = self.view_embedding.to(dtype=tokens.dtype).view(1, views, 1, self.dim)
        tokens = (tokens + spatial.view(1, 1, self.grid * self.grid, self.dim) + view).reshape(
            batch, views * self.grid * self.grid, self.dim
        )
        token_valid = (mass.flatten(1) > 0).reshape(batch, views * self.grid * self.grid)
        return tokens, token_valid


class TextTokenPool(nn.Module):
    """Pool variable-length text embeddings into a fixed token count."""

    def __init__(self, text_dim: int, dim: int, queries: int = 4) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.text_proj = nn.Linear(text_dim, dim)
        self.scale = dim**-0.5

    def forward(
        self,
        text: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text.ndim != 3:
            raise ValueError(f"text embeddings must be [B, T, C], got {tuple(text.shape)}")
        projected = self.text_proj(text.to(dtype=self.text_proj.weight.dtype))
        query = self.queries.to(dtype=projected.dtype).unsqueeze(0).expand(
            projected.shape[0], -1, -1
        )
        logits = torch.matmul(query, projected.transpose(1, 2)) * self.scale
        if mask is not None:
            if mask.shape[:2] != projected.shape[:2]:
                raise ValueError(
                    f"text mask {tuple(mask.shape)} does not match {tuple(projected.shape)}"
                )
            logits = logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, projected)
