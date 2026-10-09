"""Causal short-history feedback around an immutable native sixteen-action plan.

The video/action backbone and its plan are not modified. A shared visual body
encodes the current three views and one preceding head view in one batch.
All horizons use the same four-row head; unsupported tail rows are zero.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.compact_fast_policy import CompactVision
from dynamicwam.models.fast_observation_encoder import TextTokenPool


def _ages(plan, age_actions):
    age = torch.as_tensor(age_actions, device=plan.device).reshape(-1)
    if age.numel() == 1:
        age = age.expand(plan.shape[0])
    return age


def remaining_window(plan, age_actions, validate=True):
    """Return a four-row gather and validity; padded rows are never targets.

    Clamping is only for a safe gather, not permission to supervise repeated
    final commands. Callers must use the returned mask in every objective.
    ``validate=False`` is reserved for externally validated static graph input.
    """
    if plan.ndim != 3 or tuple(plan.shape[1:]) != (16, 14):
        raise ValueError('expected plan [B,16,14]')
    age = _ages(plan, age_actions)
    if age.numel() != plan.shape[0]:
        raise ValueError('one plan age is required per batch row')
    if validate and not bool((torch.isfinite(age) & (age == age.round()) &
                              (age >= 0) & (age < 16)).all()):
        raise ValueError('plan ages must be integers in [0,15]')
    indices = age.long()[:, None] + torch.arange(4, device=plan.device)[None]
    valid = indices < 16
    window = plan.gather(1, indices.clamp(max=15)[..., None].expand(-1, -1, 14))
    return window, valid


class TemporalPlanFeedbackHead(nn.Module):
    ARCHITECTURE_VERSION = 3

    def __init__(self, dim=192, heads=6, layers=2, residual_limit=0.1):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads or layers < 1 or not 0 < residual_limit <= .2:
            raise ValueError('invalid temporal feedback configuration')
        self.constructor_spec = dict(dim=dim, heads=heads, layers=layers,
                                     residual_limit=residual_limit)
        self.residual_limit = float(residual_limit)
        self.vision = CompactVision(dim, 'compact_gn')
        self.text = TextTokenPool(text_dim=4096, dim=dim, queries=4)
        self.state = nn.Sequential(nn.Linear(42, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.temporal = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        for branch in (self.time, self.temporal):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self.plan = nn.Linear(14, dim)
        self.position = nn.Parameter(torch.randn(1, 16, dim) * .02)
        self.age = nn.Embedding(16, dim)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(
            dim, heads, dim_feedforward=dim * 3, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for _ in range(layers))
        self.output = nn.Linear(dim, 14)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        arm = torch.ones(14)
        arm[[6, 13]] = 0
        self.register_buffer('arm_mask', arm)

    @torch.no_grad()
    def initialize_from_plan_feedback(self, state):
        """Migrate a v1 PlanFeedbackHead state without changing its memory layout.

        The caller owns checkpoint provenance/spec/normalizer verification.
        This method validates all legacy tensor keys before modifying this
        module. Only ages 0/4/8/12 preserve the old function mathematically;
        interpolation gives other rows an initialization, not a guarantee.
        """
        if not isinstance(state, Mapping):
            raise TypeError('expected the legacy PlanFeedbackHead state_dict')
        current = self.state_dict()
        shared = {key for key in current if not key.startswith(('time.', 'temporal.'))}
        if set(state) != shared:
            raise ValueError(f'legacy feedback keys differ: missing={sorted(shared-set(state))}, '
                             f'unexpected={sorted(set(state)-shared)}')
        for key in sorted(shared):
            incoming, reference = state[key], current[key]
            expected_shape = (4, reference.shape[1]) if key == 'age.weight' else tuple(reference.shape)
            if not torch.is_tensor(incoming) or tuple(incoming.shape) != expected_shape:
                raise ValueError(f'legacy feedback tensor shape mismatch: {key}')
            if incoming.dtype != reference.dtype or not bool(torch.isfinite(incoming).all()):
                raise ValueError(f'legacy feedback tensor dtype/nonfinite mismatch: {key}')
        if not torch.equal(state['arm_mask'].to(self.arm_mask.device), self.arm_mask):
            raise ValueError('legacy arm mask must preserve the gripper exclusions')
        for key in sorted(shared - {'age.weight'}):
            current[key].copy_(state[key])
        original = state['age.weight'].to(device=self.age.weight.device)
        for age in range(16):
            if age >= 12:
                value = original[3]
            elif age % 4 == 0:
                value = original[age//4]
            else:
                left, fraction = age//4, (age % 4)/4.
                value = torch.lerp(original[left], original[left+1], fraction)
            self.age.weight[age].copy_(value)
        # Reinitialization must also be safe if called on an already-used head.
        for branch in (self.time, self.temporal):
            branch[-1].weight.zero_()
            branch[-1].bias.zero_()
        return dict(architecture_version=self.ARCHITECTURE_VERSION,
            shared_tensors_copied=len(shared)-1, preserved_ages=[0, 4, 8, 12],
            interpolated_ages=[1, 2, 3, 5, 6, 7, 9, 10, 11], held_ages=[13, 14, 15],
            memory_tokens=69, temporal_last_layer_zero=True, time_last_layer_zero=True,
            age_extension_function_guarantee=False,
            scope='legacy supported ages preserve the function up to floating-point kernel rounding; other ages require validation')

    def validate_inputs(self, rgb, rgb_valid_mask, state, source_state, plan,
                        age_actions, text, previous_head, previous_head_mask,
                        history_dt, history_valid, plan_age_seconds, text_mask=None):
        if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (3, 3):
            raise ValueError('expected current RGB [B,3,3,H,W]')
        b, _, _, h, w = rgb.shape
        expected = {'rgb_valid_mask': (b, 3, h, w), 'state': (b, 14),
                    'source_state': (b, 14), 'plan': (b, 16, 14),
                    'previous_head': (b, 3, h, w),
                    'previous_head_mask': (b, h, w),
                    'history_dt': (b,), 'history_valid': (b,),
                    'plan_age_seconds': (b,)}
        values = dict(rgb=rgb, rgb_valid_mask=rgb_valid_mask, state=state,
                      source_state=source_state, plan=plan, previous_head=previous_head,
                      previous_head_mask=previous_head_mask, history_dt=history_dt,
                      history_valid=history_valid, plan_age_seconds=plan_age_seconds,
                      text=text)
        for name, shape in expected.items():
            value = values[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if text.ndim != 3 or text.shape[0] != b or text.shape[1] < 1 or text.shape[2] != 4096:
            raise ValueError('text must be [B,T,4096] with T positive')
        if text_mask is not None:
            if tuple(text_mask.shape) != tuple(text.shape[:2]) or text_mask.dtype != torch.bool:
                raise ValueError('text_mask must be boolean [B,T]')
            values['text_mask'] = text_mask
        for name in ('rgb_valid_mask', 'previous_head_mask', 'history_valid'):
            if values[name].dtype != torch.bool:
                raise ValueError(f'{name} must be boolean')
        if any(value.device != plan.device for value in values.values()):
            raise ValueError('feedback inputs must share the plan device')
        age = _ages(plan, age_actions)
        if age.numel() != b:
            raise ValueError('one integer age is required per row')
        # Combine numerical checks before the single host read; never run this
        # Tensor-to-bool path inside CUDA graph capture.
        good = (torch.isfinite(age) & (age == age.round()) & (age >= 0) & (age < 16)).all()
        good = good & (history_dt >= 0).all() & (plan_age_seconds >= 0).all()
        good = good & ((~history_valid) | (history_dt > 0)).all()
        for value in values.values():
            if value.is_floating_point():
                good = good & torch.isfinite(value).all()
        if text_mask is not None:
            good = good & text_mask.any(dim=1).all()
        if not bool(good):
            raise ValueError('nonfinite input, invalid age/time, or empty language mask')

    def _visual_tokens(self, rgb, rgb_mask, previous_head, previous_mask, history_valid):
        b, _, _, h, w = rgb.shape
        prior_mask = previous_mask & history_valid[:, None, None]
        mask = torch.cat((rgb_mask, prior_mask[:, None]), dim=1)
        pixels = torch.cat((rgb, previous_head[:, None]), dim=1)
        pixels = torch.where(mask[:, :, None], pixels, torch.zeros_like(pixels))
        features = self.vision.body(pixels.reshape(b * 4, 3, h, w).to(
            dtype=next(self.vision.body.parameters()).dtype))
        m = F.interpolate(mask.reshape(b * 4, 1, h, w).float(),
                          size=features.shape[-2:], mode='area').to(features.dtype)
        mass = F.adaptive_avg_pool2d(m, (4, 4))
        pooled = F.adaptive_avg_pool2d(features * m, (4, 4)) / mass.clamp_min(1e-6)
        tokens = self.vision.projection(pooled.flatten(2).transpose(1, 2)).reshape(b, 4, 16, -1)
        current = tokens[:, :3] + self.vision.view_position + self.vision.spatial_position
        prior = tokens[:, 3] + self.vision.view_position[:, 0] + self.vision.spatial_position[:, 0]
        validity = (mass.flatten(1) > 0).reshape(b, 4, 16)
        history_tokens = self.temporal(torch.cat((current[:, 0], prior, current[:, 0] - prior), -1))
        history_mask = validity[:, 0] & validity[:, 3] & history_valid[:, None]
        history_tokens = torch.where(history_mask[..., None], history_tokens, torch.zeros_like(history_tokens))
        # Zero-initialized information is added to existing head-view tokens.
        # Appending zero-valued valid tokens would still alter softmax weights.
        current = torch.cat((current[:, :1] + history_tokens[:, None], current[:, 1:]), dim=1)
        return current.flatten(1, 2), validity[:, :3].flatten(1)

    def forward(self, rgb, rgb_valid_mask, state, source_state, plan, age_actions,
                text, previous_head, previous_head_mask, history_dt, history_valid,
                plan_age_seconds, text_mask=None, validate_inputs=True):
        if validate_inputs:
            self.validate_inputs(rgb=rgb, rgb_valid_mask=rgb_valid_mask, state=state,
                source_state=source_state, plan=plan, age_actions=age_actions,
                text=text, text_mask=text_mask, previous_head=previous_head,
                previous_head_mask=previous_head_mask, history_dt=history_dt,
                history_valid=history_valid, plan_age_seconds=plan_age_seconds)
        window, tail_valid = remaining_window(plan, age_actions, validate=False)
        age = _ages(plan, age_actions).long()
        visual, visual_valid = self._visual_tokens(
            rgb, rgb_valid_mask, previous_head, previous_head_mask, history_valid)
        language = self.text(text, text_mask)
        preceding = plan.gather(1, (age - 1).clamp_min(0)[:, None, None].expand(-1, 1, 14)).squeeze(1)
        preceding = torch.where((age == 0)[:, None], source_state, preceding)
        state_token = self.state(torch.cat((state, state - preceding, state - source_state), -1).float())
        dt = torch.where(history_valid, history_dt.float(), torch.zeros_like(history_dt).float())
        # Training covers actual image-history lags. Physical plan age remains
        # audited telemetry: expert row timing cannot identify the deployed
        # controller's cumulative TOPP duration, so do not condition on it.
        time_features = torch.stack((torch.log1p(dt), history_valid.float()), dim=-1)
        time_delta = self.time(time_features) * history_valid[:, None].to(state_token.dtype)
        state_token = (state_token + time_delta)[:, None]
        future = self.plan(plan.float()) + self.position
        memory = torch.cat((visual, language, state_token, future), dim=1)
        extra_valid = torch.ones((plan.shape[0], language.shape[1] + 17),
                                 dtype=torch.bool, device=plan.device)
        valid = torch.cat((visual_valid, extra_valid), dim=1)
        query = self.plan(window.float()) + self.age(age)[:, None]
        for block in self.blocks:
            query = block(query, memory, memory_key_padding_mask=~valid,
                          tgt_key_padding_mask=~tail_valid)
        residual = self.output(query).float().tanh() * self.residual_limit
        enabled = tail_valid & (age > 0)[:, None]
        return residual * self.arm_mask * enabled[..., None].float()
