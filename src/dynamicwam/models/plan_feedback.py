"""Bounded visual feedback around an unchanged native WAM action plan.

There is no diffusion, world-cache mutation or future extrapolation here.
The planner owns semantics and gripper events; feedback adjusts arm targets
inside the remaining four-command window. Zero feedback recovers the plan.
"""
from __future__ import annotations

import torch
from torch import nn

from dynamicwam.models.compact_fast_policy import CompactVision
from dynamicwam.models.fast_observation_encoder import TextTokenPool


def plan_window(plan, age_actions, horizon=4):
    """Batched exact slicing; invalid ages fail instead of wrapping/padding."""
    if plan.ndim != 3 or plan.shape[1:] != (16, 14) or horizon != 4:
        raise ValueError('expected a 16x14 plan and a four-command window')
    age = torch.as_tensor(age_actions, device=plan.device).reshape(-1)
    if age.numel() == 1:
        age = age.expand(plan.shape[0])
    if age.numel() != plan.shape[0] or not bool(((age >= 0) & (age <= 12) & (age % 4 == 0)).all()):
        raise ValueError('plan age must be 0, 4, 8 or 12 for each batch row')
    indices = age.long()[:, None] + torch.arange(horizon, device=plan.device)[None]
    return plan.gather(1, indices[..., None].expand(-1, -1, 14))


class PlanFeedbackHead(nn.Module):
    def __init__(self, dim=192, heads=6, layers=2, residual_limit=0.1):
        super().__init__()
        if dim % heads or layers < 1 or not 0 < residual_limit <= 0.2:
            raise ValueError('invalid bounded feedback configuration')
        self.constructor_spec = dict(dim=dim, heads=heads, layers=layers, residual_limit=residual_limit)
        self.residual_limit = float(residual_limit)
        self.vision = CompactVision(dim, 'compact_gn')
        self.text = TextTokenPool(text_dim=4096, dim=dim, queries=4)
        self.state = nn.Sequential(nn.Linear(42, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.plan = nn.Linear(14, dim)
        self.position = nn.Parameter(torch.randn(1, 16, dim) * 0.02)
        self.age = nn.Embedding(4, dim)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(dim, heads,
            dim_feedforward=dim * 3, dropout=0.0, activation='gelu', batch_first=True,
            norm_first=True) for _ in range(layers))
        self.output = nn.Linear(dim, 14)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        arm_mask = torch.ones(14)
        arm_mask[[6, 13]] = 0
        self.register_buffer('arm_mask', arm_mask)

    def forward(self, rgb, rgb_valid_mask, state, source_state, plan, age_actions, text, text_mask=None):
        window = plan_window(plan, age_actions)
        age = torch.as_tensor(age_actions, device=plan.device).reshape(-1).long()
        if age.numel() == 1:
            age = age.expand(plan.shape[0])
        visual, valid = self.vision(rgb, rgb_valid_mask)
        language = self.text(text, text_mask)
        # Planned preceding command estimates tracking error without assigning
        # a fictitious physical duration to each DOMINO TOPP command.
        preceding = plan.gather(1, (age - 1).clamp_min(0)[:, None, None].expand(-1, 1, 14)).squeeze(1)
        preceding = torch.where((age == 0)[:, None], source_state, preceding)
        state_token = self.state(torch.cat((state, state - preceding, state - source_state), -1).float())[:, None]
        future = self.plan(plan.float()) + self.position
        memory = torch.cat((visual, language, state_token, future), 1)
        extra = torch.ones((plan.shape[0], language.shape[1] + 17), device=plan.device, dtype=torch.bool)
        valid = torch.cat((valid, extra), 1)
        query = self.plan(window.float()) + self.age(age // 4)[:, None]
        for block in self.blocks:
            query = block(query, memory, memory_key_padding_mask=~valid)
        residual = self.output(query).float().tanh() * self.residual_limit
        # The fresh first window is already generated from this observation.
        # Gripper timing remains entirely under the native WAM planner.
        return residual * self.arm_mask * (age > 0).float()[:, None, None]
