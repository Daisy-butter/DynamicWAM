"""Time-indexed queries over WAM's cached latent grid, without a CUDA sync.

Latent bins describe forecast *intervals*, not independent object states:
VAE receptive fields and joint attention mix information. Their native order
is retained; a stale bin may remain historical context, never future support.
"""
from __future__ import annotations

import torch


def temporal_world_layout(bridge, age_actions, *, queries, horizon_actions, device, dtype):
    layout = bridge.grid_sizes
    nc, nf = int(layout['condition_seq_len']), int(layout['future_seq_len'])
    bins, height, width = tuple(int(v) for v in layout['future_grid_shape'])
    if bins * height * width != nf or bins < 1 or nc < 1:
        raise ValueError('temporal memory requires a valid native condition/future grid')
    if queries % (bins + 1):
        raise ValueError('memory_tokens must divide evenly into condition + future bins')
    n = bridge.video_k[0].shape[1]
    if nc + nf != n:
        raise ValueError('temporal memory does not accept an ambiguous padded grid')
    b = bridge.video_k[0].shape[0]
    age = torch.as_tensor(age_actions, device=device, dtype=dtype).reshape(-1)
    if age.numel() == 1:
        age = age.expand(b)
    if age.numel() != b:
        raise ValueError('bridge age batch differs from temporal memory')
    # Group 0 reads condition/history; other groups read one forecast bin each.
    group = torch.arange(queries, device=device) // (queries // (bins + 1))
    token_group = torch.cat((torch.zeros(nc, device=device, dtype=torch.long),
                            torch.arange(1, bins + 1, device=device).repeat_interleave(height * width)))
    end = group.to(dtype) * (float(horizon_actions) / bins)
    start = (group.to(dtype) - 1).clamp_min(0) * (float(horizon_actions) / bins)
    future = group > 0
    remaining_start = (start[None] - age[:, None]) / horizon_actions
    remaining_end = (end[None] - age[:, None]) / horizon_actions
    remaining_fraction = ((end[None] - age[:, None]).clamp_min(0) /
                          (float(horizon_actions) / bins)).clamp_max(1)
    features = torch.stack((future.expand(b, -1).to(dtype), remaining_start,
                            remaining_end, remaining_fraction * future[None]), dim=-1)
    alive = (~future)[None] | (end[None] > age[:, None])
    mask = (group[:, None] == token_group[None])[None].expand(b, -1, -1)
    mask = mask & (torch.arange(n, device=device)[None, None] < bridge.seq_lens[:, None, None])
    # Expired groups use a safe nonempty condition read, then their world
    # correction is explicitly zeroed. Avoid all-masked-row backend NaNs.
    mask = torch.where(alive[:, :, None], mask, (token_group == 0)[None, None])
    return features, mask[:, None], alive.to(dtype)[:, :, None]
