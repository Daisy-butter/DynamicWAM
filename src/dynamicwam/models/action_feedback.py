"""Action-conditioned prediction of compact future VAE patches.

Training uses observed futures ONLY with the actions that produced them. The
policy consumes predictions from its own proposals, never those future labels.
This is a learned local dynamics approximation, not a counterfactual oracle.
"""
import torch
from torch import nn
from torch.nn import functional as F


def future_patch_target(latent, channels=48):
    if latent.ndim != 5 or latent.shape[1] != channels or latent.shape[2] != 2:
        raise ValueError(f'expected future latent [B,{channels},2,H,W], got {latent.shape}')
    return F.adaptive_avg_pool3d(latent.float(), (2, 4, 4)).flatten(2).transpose(1, 2)


class ActionConditionedForecast(nn.Module):
    def __init__(self, dim, heads, action_dim, chunk_size, channels=48):
        super().__init__()
        self.heads = heads
        self.action = nn.Linear(action_dim, dim)
        self.action_position = nn.Parameter(torch.randn(1, chunk_size, dim) * .02)
        self.queries = nn.Parameter(torch.randn(1, 32, dim) * .02)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.latent = nn.Linear(dim, channels)

    def forward(self, context, actions):
        from dynamicwam.models.compact_fast_policy import attend
        a = self.action(actions.to(context.dtype)) + self.action_position
        kv = self.kv(self.norm(torch.cat((context, a), dim=1))).chunk(2, dim=-1)
        q = self.queries + context.mean(dim=1, keepdim=True)
        q = q + self.out(attend(self.query(q), *kv, self.heads))
        return self.latent(q + self.ffn(q))


def dynamics_loss(prediction, target):
    # Explicit temporal-change term prevents static background alone from
    # dominating both supervised future bins. Both terms have observed labels.
    p, y = prediction.float(), target.float()
    return F.mse_loss(p, y) + 2 * F.mse_loss(p[:,16:]-p[:,:16], y[:,16:]-y[:,:16])
