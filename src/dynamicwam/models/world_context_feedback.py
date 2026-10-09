"""Read the native plan's generated world reference without changing its sampler.

The values come from the second joint denoising call, which also supplies the
eight remaining native action steps. They are intermediate, action-conditioned
features, not a decoded clean future or a rollout under corrected actions.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.temporal_plan_feedback import (
    TemporalPlanFeedbackHead, _ages, remaining_window,
)


WORLD_CONTEXT_SPEC = dict(version=1, layer_index=11, pool_height=4, pool_width=4,
    width=2048, source='second_joint_video_value_cache',
    condition_and_future=True, pooling='per_native_time_bin_spatial_average',
    coordinates='is_future_native_bin_fraction_y_x',
    time_semantics='native_latent_order_not_physical_future_coverage',
    storage_dtype='bfloat16')


@torch.no_grad()
def pack_world_context(cache):
    """Return detached, owned tensors; never expose mutable graph cache buffers.

    No new backbone call, K projection, host CUDA synchronization, or random
    draw is needed. Spatial pooling preserves the original temporal bins.
    """
    spec = WORLD_CONTEXT_SPEC
    layout = cache['grid_sizes']
    layer = spec['layer_index']
    if len(cache['video_v']) <= layer:
        raise ValueError('native world context requires layer 11 video values')
    value = cache['video_v'][layer]
    if value.ndim != 4 or value.shape[-2] * value.shape[-1] != spec['width']:
        raise ValueError('expected native values [B,N,16,128]')
    if cache.get('all_video_tokens_valid') is not True:
        raise ValueError('world pooling requires the native dense token layout')
    b, n = value.shape[:2]
    sizes = [tuple(int(x) for x in layout[f'{part}_grid_shape'])
             for part in ('condition', 'future')]
    counts = [int(layout[f'{part}_seq_len']) for part in ('condition', 'future')]
    if (sum(counts) != n or any(len(shape) != 3 or min(shape) < 1 or
            shape[0] * shape[1] * shape[2] != count for shape, count in zip(sizes, counts))):
        raise ValueError('world grid and video value sequence disagree')
    values, coordinates = [], []
    h, w = spec['pool_height'], spec['pool_width']
    yy, xx = torch.meshgrid((torch.arange(h, device=value.device).float() + .5) / h,
        (torch.arange(w, device=value.device).float() + .5) / w, indexing='ij')
    offset = 0
    for part, ((t, ih, iw), count) in enumerate(zip(sizes, counts)):
        grid = value[:, offset:offset+count].detach().reshape(b*t, ih, iw, spec['width'])
        # FP32 averaging is deterministic with respect to stored BF16 values.
        pooled = F.adaptive_avg_pool2d(grid.permute(0, 3, 1, 2).float(), (h, w))
        values.append(pooled.permute(0, 2, 3, 1).reshape(b, t*h*w, spec['width']))
        coordinate = torch.stack((torch.full((t, h, w), float(part), device=value.device),
            ((torch.arange(t, device=value.device).float()+.5)/t)[:, None, None].expand(t,h,w),
            yy.expand(t,h,w), xx.expand(t,h,w)), -1).reshape(1,t*h*w,4)
        coordinates.append(coordinate.expand(b,-1,-1))
        offset += count
    result = torch.cat(values, 1).to(torch.bfloat16).contiguous().clone()
    return dict(world_values=result,
        world_coordinates=torch.cat(coordinates, 1).contiguous().clone(),
        world_mask=torch.ones(result.shape[:2], device=value.device, dtype=torch.bool))


class WorldContextFeedbackHead(TemporalPlanFeedbackHead):
    """Function-preserving world readout around the existing bounded head.

    Current observations route a reference-world read into the existing 16
    head-camera tokens. The final world projection is exactly zero initially;
    no extra valid memory tokens change the old decoder's attention denominator.
    """
    ARCHITECTURE_VERSION = 1

    def __init__(self, dim=192, heads=6, layers=2, residual_limit=.1):
        super().__init__(dim=dim, heads=heads, layers=layers, residual_limit=residual_limit)
        self.world_projection = nn.Sequential(nn.LayerNorm(WORLD_CONTEXT_SPEC['width']),
            nn.Linear(WORLD_CONTEXT_SPEC['width'], dim))
        self.world_coordinates = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.world_query_norm = nn.LayerNorm(dim)
        self.world_attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.world_output = nn.Linear(dim, dim)
        nn.init.zeros_(self.world_output.weight)
        nn.init.zeros_(self.world_output.bias)

    @torch.no_grad()
    def initialize_from_plan_feedback(self, state):
        # Parent migration enumerates all shared keys, so migrate through its
        # exact old class rather than mistakenly treating world keys as legacy.
        with torch.random.fork_rng(devices=[]):
            legacy = TemporalPlanFeedbackHead(**self.constructor_spec)
        report = legacy.initialize_from_plan_feedback(state)
        current = self.state_dict()
        for name, tensor in legacy.state_dict().items():
            current[name].copy_(tensor)
        self.world_output.weight.zero_(); self.world_output.bias.zero_()
        return dict(report, world_readout_zero=True, memory_tokens=69,
            world_context_spec=dict(WORLD_CONTEXT_SPEC))

    def _validate_world(self, plan, values, coordinates, mask, context_valid):
        b = plan.shape[0]
        if (values.ndim != 3 or values.shape[0] != b or values.shape[1] < 1 or
                values.shape[2] != WORLD_CONTEXT_SPEC['width'] or
                tuple(coordinates.shape) != (*values.shape[:2],4) or
                tuple(mask.shape) != tuple(values.shape[:2]) or mask.dtype != torch.bool or
                tuple(context_valid.shape) != (b,) or context_valid.dtype != torch.bool):
            raise ValueError('invalid world-reference tensors/masks')
        if any(x.device != plan.device for x in (values,coordinates,mask,context_valid)):
            raise ValueError('world reference and plan must share a device')
        if not bool(torch.isfinite(values).all() & torch.isfinite(coordinates).all() &
                    ((~context_valid) | mask.any(1)).all()):
            raise ValueError('nonfinite world reference or an empty valid context')

    def forward(self, rgb, rgb_valid_mask, state, source_state, plan, age_actions,
                text, previous_head, previous_head_mask, history_dt, history_valid,
                plan_age_seconds, world_values, world_coordinates, world_mask,
                context_valid, text_mask=None, validate_inputs=True):
        if validate_inputs:
            self.validate_inputs(rgb=rgb,rgb_valid_mask=rgb_valid_mask,state=state,
                source_state=source_state,plan=plan,age_actions=age_actions,text=text,
                text_mask=text_mask,previous_head=previous_head,previous_head_mask=previous_head_mask,
                history_dt=history_dt,history_valid=history_valid,plan_age_seconds=plan_age_seconds)
            self._validate_world(plan,world_values,world_coordinates,world_mask,context_valid)
        window, tail_valid = remaining_window(plan,age_actions,validate=False)
        age = _ages(plan,age_actions).long()
        visual, visual_valid = self._visual_tokens(
            rgb,rgb_valid_mask,previous_head,previous_head_mask,history_valid)
        language = self.text(text,text_mask)
        preceding = plan.gather(1,(age-1).clamp_min(0)[:,None,None].expand(-1,1,14)).squeeze(1)
        preceding = torch.where((age==0)[:,None],source_state,preceding)
        state_token = self.state(torch.cat((state,state-preceding,state-source_state),-1).float())
        dt = torch.where(history_valid,history_dt.float(),torch.zeros_like(history_dt).float())
        time_delta = self.time(torch.stack((torch.log1p(dt),history_valid.float()),-1))
        state_token = state_token + time_delta * history_valid[:,None].to(state_token.dtype)
        # Any expired/mismatched bundle must be explicitly disabled by its
        # owner. A harmless dummy avoids all-masked attention backend NaNs.
        active = context_valid[:,None] & world_mask
        safe = active.clone(); safe[:,0] = safe[:,0] | (~active.any(1))
        encoded = self.world_projection(world_values.float()) + self.world_coordinates(world_coordinates.float())
        encoded = torch.where(active[...,None],encoded,torch.zeros_like(encoded))
        plan_query = self.plan(window.float())
        query = self.world_query_norm(visual[:,:16] + state_token[:,None] +
            (plan_query * tail_valid[...,None]).sum(1,keepdim=True) / tail_valid.sum(1)[:,None,None].clamp_min(1) +
            self.age(age)[:,None])
        routed,_ = self.world_attention(query,encoded,encoded,key_padding_mask=~safe,need_weights=False)
        delta = self.world_output(routed) * context_valid[:,None,None].to(routed.dtype)
        delta = delta * visual_valid[:,:16,None].to(delta.dtype)
        visual = torch.cat((visual[:,:16]+delta,visual[:,16:]),1)
        memory = torch.cat((visual,language,state_token[:,None],self.plan(plan.float())+self.position),1)
        extra_valid = torch.ones((plan.shape[0],language.shape[1]+17),device=plan.device,dtype=torch.bool)
        valid = torch.cat((visual_valid,extra_valid),1)
        query = plan_query + self.age(age)[:,None]
        for block in self.blocks:
            query = block(query,memory,memory_key_padding_mask=~valid,tgt_key_padding_mask=~tail_valid)
        residual = self.output(query).float().tanh()*self.residual_limit
        return residual*self.arm_mask*(tail_valid & (age>0)[:,None])[...,None].float()
