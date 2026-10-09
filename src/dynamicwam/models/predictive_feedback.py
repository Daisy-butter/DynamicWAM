"""Short-horizon prediction in a fixed spatial representation.

World supervision uses actually executed action paths. Policy inference accepts
only deployable proposals; it never accepts the future training action label.
Each query is grounded in new RGB, not a claimed rollout of corrected history.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.temporal_plan_feedback import (
    TemporalPlanFeedbackHead, _ages, remaining_window,
)


ABLATIONS = frozenset(('none', 'nohistory', 'noaction', 'current_only', 'shuffle_future'))


class _SpatialBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                 nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, x, valid):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # A permanently valid zero key avoids an all-masked softmax for a
        # missing camera packet, without making invalid image content visible.
        k = torch.cat((k, torch.zeros_like(k[:, :, :1])), dim=2)
        v = torch.cat((v, torch.zeros_like(v[:, :, :1])), dim=2)
        key_valid = torch.cat((valid, torch.ones_like(valid[:, :1])), dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, None, None])
        x = x + self.output(y.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(x)


class LatentPredictor(nn.Module):
    """Predict two physical horizons, each conditioned on four action bins.

    Bin channels are normalized joint-command endpoint (14) and mean command
    velocity (14). The caller owns their physical-time construction and units.
    The final layer starts at zero, yielding the exact current-token baseline.
    """
    def __init__(self, dim=192, heads=6):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError('latent dimensions must be divisible by heads')
        self.dim = dim
        self.history = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.history_time = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action = nn.Sequential(nn.Linear(4 * 28, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.horizon = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(_SpatialBlock(dim, heads) for _ in range(2))
        self.output = nn.Linear(dim, dim)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def validate_inputs(self, current_tokens, previous_head_tokens, history_dt,
                        history_valid, token_valid, action_path, future_dt,
                        previous_valid=None, future_valid=None):
        if current_tokens.ndim != 3 or tuple(current_tokens.shape[1:]) != (48, self.dim):
            raise ValueError('current_tokens must be [B,48,D]')
        b = current_tokens.shape[0]
        expected = dict(previous_head_tokens=(b, 16, self.dim), history_dt=(b,),
                        history_valid=(b,), token_valid=(b, 48),
                        action_path=(b, 2, 4, 28), future_dt=(b, 2))
        values = dict(previous_head_tokens=previous_head_tokens, history_dt=history_dt,
                      history_valid=history_valid, token_valid=token_valid,
                      action_path=action_path, future_dt=future_dt)
        if previous_valid is not None:
            expected['previous_valid'] = (b, 16)
            values['previous_valid'] = previous_valid
        if future_valid is not None:
            expected['future_valid'] = (b, 2)
            values['future_valid'] = future_valid
        for name, shape in expected.items():
            value = values[name]
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
            if value.device != current_tokens.device:
                raise ValueError('latent inputs must share a device')
        for name in ('token_valid', 'history_valid', 'previous_valid', 'future_valid'):
            if name in values and values[name].dtype != torch.bool:
                raise ValueError(f'{name} must be boolean')
        good = (history_dt >= 0).all() & ((~history_valid) | (history_dt > 0)).all()
        valid = torch.ones_like(future_dt, dtype=torch.bool) if future_valid is None else future_valid
        good = good & torch.where(valid, future_dt > 0, future_dt == 0).all()
        good = good & ((~valid.all(dim=1)) | (future_dt[:, 1] > future_dt[:, 0])).all()
        for value in (current_tokens, *values.values()):
            if value.is_floating_point():
                good = good & torch.isfinite(value).all()
        if not bool(good):
            raise ValueError('invalid latent inputs or nonpositive/unordered physical horizons')

    def forward(self, current_tokens, previous_head_tokens, history_dt, history_valid,
                token_valid, action_path, future_dt, previous_valid=None,
                validate_inputs=True, ablation='none', future_valid=None):
        if ablation not in ABLATIONS:
            raise ValueError(f'unknown predictive ablation {ablation!r}')
        if validate_inputs:
            self.validate_inputs(current_tokens, previous_head_tokens, history_dt,
                history_valid, token_valid, action_path, future_dt, previous_valid, future_valid)
        if ablation == 'current_only':
            return current_tokens[:, None].expand(-1, 2, -1, -1)
        b, _, d = current_tokens.shape
        enabled = history_valid if ablation != 'nohistory' else torch.zeros_like(history_valid)
        prior_valid = token_valid[:, :16] if previous_valid is None else previous_valid
        history_mask = enabled[:, None] & token_valid[:, :16] & prior_valid
        prior = torch.where(history_mask[..., None], previous_head_tokens, torch.zeros_like(previous_head_tokens))
        head = current_tokens[:, :16]
        temporal = self.history(torch.cat((head, prior, head - prior), -1))
        dt = torch.where(enabled, history_dt, torch.zeros_like(history_dt)).float()
        temporal = temporal + self.history_time(torch.stack((torch.log1p(dt), enabled.float()), -1))[:, None]
        temporal = torch.where(history_mask[..., None], temporal, torch.zeros_like(temporal))
        context = torch.cat((head + temporal, current_tokens[:, 16:]), dim=1)
        context = torch.where(token_valid[..., None], context, torch.zeros_like(context))
        path = torch.zeros_like(action_path) if ablation == 'noaction' else action_path
        conditions = self.action(path.float().flatten(2))
        horizon = future_dt.float()
        conditions = conditions + self.horizon(torch.stack((horizon, torch.log1p(horizon)), -1))
        x = (context[:, None] + conditions[:, :, None]).reshape(b * 2, 48, d)
        valid = token_valid[:, None].expand(-1, 2, -1).reshape(b * 2, 48)
        for block in self.blocks:
            x = block(x, valid)
        delta = self.output(x).reshape(b, 2, 48, d)
        if ablation == 'shuffle_future':
            # Held-out batch permutation; batch-one diagnosis exchanges horizons.
            delta = delta.roll(1, dims=0) if b > 1 else delta.flip(1)
        delta = torch.where(token_valid[:, None, :, None], delta, torch.zeros_like(delta))
        if future_valid is not None:
            delta = torch.where(future_valid[:, :, None, None], delta, torch.zeros_like(delta))
        return current_tokens[:, None] + delta


class PredictiveFeedbackHead(TemporalPlanFeedbackHead):
    ARCHITECTURE_VERSION = 1

    def __init__(self, dim=192, heads=6, layers=2, residual_limit=.1):
        super().__init__(dim=dim, heads=heads, layers=layers, residual_limit=residual_limit)
        self.latent_predictor = LatentPredictor(dim, heads)
        # No bias: a zero future delta must remain exactly zero after training.
        self.future_readout = nn.Linear(2 * dim, dim, bias=False)
        nn.init.zeros_(self.future_readout.weight)
        self._freeze_vision()

    def _freeze_vision(self):
        self.vision.float()
        self.vision.requires_grad_(False)
        self.vision.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_vision()
        return self

    @torch.no_grad()
    def initialize_from_plan_feedback(self, state):
        if not isinstance(state, Mapping):
            raise TypeError('expected the legacy PlanFeedbackHead state_dict')
        current = self.state_dict()
        extra = ('time.', 'temporal.', 'latent_predictor.', 'future_readout.')
        shared = {key for key in current if not key.startswith(extra)}
        if set(state) != shared:
            raise ValueError(f'legacy feedback keys differ: missing={sorted(shared-set(state))}, unexpected={sorted(set(state)-shared)}')
        for key in sorted(shared):
            value, reference = state[key], current[key]
            shape = (4, reference.shape[1]) if key == 'age.weight' else tuple(reference.shape)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != reference.dtype:
                raise ValueError(f'legacy feedback shape/dtype mismatch: {key}')
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite legacy tensor: {key}')
        if not torch.equal(state['arm_mask'].to(self.arm_mask.device), self.arm_mask):
            raise ValueError('legacy gripper mask mismatch')
        for key in shared - {'age.weight'}:
            current[key].copy_(state[key])
        original = state['age.weight'].to(self.age.weight.device)
        for age in range(16):
            left = min(age // 4, 3)
            value = original[left] if age >= 12 or age % 4 == 0 else torch.lerp(original[left], original[left + 1], (age % 4) / 4.)
            self.age.weight[age].copy_(value)
        for branch in (self.time, self.temporal):
            branch[-1].weight.zero_()
            branch[-1].bias.zero_()
        self.latent_predictor.reset_output()
        self.future_readout.weight.zero_()
        self._freeze_vision()
        return dict(architecture_version=self.ARCHITECTURE_VERSION, shared_tensors_copied=len(shared)-1,
            preserved_ages=[0, 4, 8, 12], interpolated_ages=[1, 2, 3, 5, 6, 7, 9, 10, 11],
            held_ages=[13, 14, 15], memory_tokens=69, vision_frozen=True,
            target_representation='fixed_v1_spatial_tokens', future_initialization='copy_current',
            future_readout_zero=True, age_extension_function_guarantee=False)

    @torch.no_grad()
    def encode_observations(self, rgb, rgb_valid_mask, previous_head, previous_head_mask,
                            history_valid=None):
        """One four-view CNN pass; no policy/text inputs or target actions."""
        b, _, _, h, w = rgb.shape
        prior_mask = previous_head_mask
        if history_valid is not None:
            prior_mask = prior_mask & history_valid[:, None, None]
        mask = torch.cat((rgb_valid_mask, prior_mask[:, None]), 1)
        pixels = torch.cat((rgb, previous_head[:, None]), 1)
        pixels = torch.where(mask[:, :, None], pixels, torch.zeros_like(pixels))
        # Training labels and deployed observations must inhabit the same
        # fixed numerical space even when the predictor/decoder use AMP.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            features = self.vision.body(pixels.reshape(b * 4, 3, h, w).float())
            m = F.interpolate(mask.reshape(b * 4, 1, h, w).float(), size=features.shape[-2:], mode='area')
            mass = F.adaptive_avg_pool2d(m, (4, 4))
            pooled = F.adaptive_avg_pool2d(features * m, (4, 4)) / mass.clamp_min(1e-6)
            tokens = self.vision.projection(pooled.flatten(2).transpose(1, 2)).reshape(b, 4, 16, -1)
            current = tokens[:, :3] + self.vision.view_position + self.vision.spatial_position
            previous = tokens[:, 3] + self.vision.view_position[:, 0] + self.vision.spatial_position[:, 0]
        validity = (mass.flatten(1) > 0).reshape(b, 4, 16)
        return dict(current_tokens=current.flatten(1, 2), previous_head_tokens=previous,
                    token_valid=validity[:, :3].flatten(1), previous_valid=validity[:, 3])

    @torch.no_grad()
    def encode_targets(self, rgb, mask):
        """One observed instant, three views; callers flatten B x horizons."""
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            return self.vision(rgb.float(), mask)

    def predict_latents(self, current_tokens, previous_head_tokens, history_dt, history_valid,
                        token_valid, action_path, future_dt, previous_valid=None,
                        validate_inputs=True, ablation='none', future_valid=None):
        return self.latent_predictor(current_tokens, previous_head_tokens, history_dt,
            history_valid, token_valid, action_path, future_dt, previous_valid,
            validate_inputs=validate_inputs, ablation=ablation, future_valid=future_valid)

    def validate_inputs(self, rgb, rgb_valid_mask, state, source_state, plan, age_actions,
                        text, previous_head, previous_head_mask, history_dt, history_valid,
                        plan_age_seconds, proposal_path, proposal_dt, text_mask=None):
        super().validate_inputs(rgb, rgb_valid_mask, state, source_state, plan,
            age_actions, text, previous_head, previous_head_mask, history_dt,
            history_valid, plan_age_seconds, text_mask)
        b = plan.shape[0]
        for name, value, shape in (('proposal_path', proposal_path, (b, 2, 4, 28)),
                                   ('proposal_dt', proposal_dt, (b, 2))):
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.device != plan.device:
                raise ValueError(f'{name} must be {shape} on the plan device')
        good = torch.isfinite(proposal_path).all() & torch.isfinite(proposal_dt).all()
        good = good & (proposal_dt > 0).all() & (proposal_dt[:, 1] > proposal_dt[:, 0]).all()
        if not bool(good):
            raise ValueError('invalid proposal path or nonpositive/unordered horizons')

    def forward(self, rgb, rgb_valid_mask, state, source_state, plan, age_actions,
                text, previous_head, previous_head_mask, history_dt, history_valid,
                plan_age_seconds, proposal_path, proposal_dt, text_mask=None,
                validate_inputs=True, ablation='none', return_aux=False):
        if validate_inputs:
            self.validate_inputs(rgb, rgb_valid_mask, state, source_state, plan,
                age_actions, text, previous_head, previous_head_mask, history_dt,
                history_valid, plan_age_seconds, proposal_path, proposal_dt, text_mask)
        encoded = self.encode_observations(rgb, rgb_valid_mask, previous_head,
                                           previous_head_mask, history_valid)
        return self.forward_encoded(**encoded, state=state, source_state=source_state,
            plan=plan, age_actions=age_actions, text=text, text_mask=text_mask,
            history_dt=history_dt, history_valid=history_valid, plan_age_seconds=plan_age_seconds,
            proposal_path=proposal_path, proposal_dt=proposal_dt,
            validate_inputs=False, ablation=ablation, return_aux=return_aux)

    def forward_encoded(self, current_tokens, previous_head_tokens, token_valid,
                        previous_valid, state, source_state, plan, age_actions, text,
                        history_dt, history_valid, plan_age_seconds, proposal_path,
                        proposal_dt, text_mask=None, validate_inputs=True,
                        ablation='none', return_aux=False):
        """Decode a proposal from already encoded observations; never runs CNN.

        Encoded tensors are supplied by ``encode_observations`` from THIS head's
        fixed visual space. The caller owns any cache identity across queries.
        Reforecasting a corrected proposal uses ``predict_latents`` with these
        same four tensors and does not run this decoder or visual body again.
        """
        encoded = dict(current_tokens=current_tokens, previous_head_tokens=previous_head_tokens,
                       token_valid=token_valid, previous_valid=previous_valid)
        if validate_inputs:
            self.latent_predictor.validate_inputs(current_tokens, previous_head_tokens,
                history_dt, history_valid, token_valid, proposal_path, proposal_dt, previous_valid)
            b = current_tokens.shape[0]
            for name, value, shape in (('state', state, (b, 14)),
                    ('source_state', source_state, (b, 14)), ('plan', plan, (b, 16, 14)),
                    ('plan_age_seconds', plan_age_seconds, (b,))):
                if not torch.is_tensor(value) or tuple(value.shape) != shape or value.device != current_tokens.device:
                    raise ValueError(f'{name} must be {shape} on the encoded device')
            remaining_window(plan, age_actions, validate=True)
            if text.ndim != 3 or text.shape[0] != b or text.shape[1] < 1 or text.shape[2] != 4096 or text.device != plan.device:
                raise ValueError('text must be [B,T,4096] on the encoded device')
            good = (plan_age_seconds >= 0).all()
            for value in (state, source_state, plan, plan_age_seconds, text):
                good = good & torch.isfinite(value).all()
            if text_mask is not None:
                if text_mask.shape != text.shape[:2] or text_mask.dtype != torch.bool or text_mask.device != plan.device:
                    raise ValueError('invalid encoded language mask')
                good = good & text_mask.any(dim=1).all()
            if not bool(good):
                raise ValueError('nonfinite encoded policy inputs or empty language mask')
        predicted = self.predict_latents(**encoded, history_dt=history_dt,
            history_valid=history_valid, action_path=proposal_path, future_dt=proposal_dt,
            validate_inputs=False, ablation=ablation)
        visual = encoded['current_tokens']
        prior = encoded['previous_head_tokens']
        history_on = history_valid if ablation != 'nohistory' else torch.zeros_like(history_valid)
        valid_history = encoded['token_valid'][:, :16] & encoded['previous_valid'] & history_on[:, None]
        temporal = self.temporal(torch.cat((visual[:, :16], prior, visual[:, :16] - prior), -1))
        temporal = torch.where(valid_history[..., None], temporal, torch.zeros_like(temporal))
        future_delta = predicted - visual[:, None]
        readout = self.future_readout(future_delta.permute(0, 2, 1, 3).flatten(2))
        visual = torch.cat((visual[:, :16] + temporal, visual[:, 16:]), 1) + readout
        window, tail_valid = remaining_window(plan, age_actions, validate=False)
        age = _ages(plan, age_actions).long()
        language = self.text(text, text_mask)
        preceding = plan.gather(1, (age - 1).clamp_min(0)[:, None, None].expand(-1, 1, 14)).squeeze(1)
        preceding = torch.where((age == 0)[:, None], source_state, preceding)
        state_token = self.state(torch.cat((state, state - preceding, state - source_state), -1).float())
        dt = torch.where(history_on, history_dt.float(), torch.zeros_like(history_dt).float())
        time_delta = self.time(torch.stack((torch.log1p(dt), history_on.float()), -1)) * history_on[:, None].to(state_token.dtype)
        future = self.plan(plan.float()) + self.position
        memory = torch.cat((visual, language, (state_token + time_delta)[:, None], future), 1)
        extra = torch.ones((plan.shape[0], language.shape[1] + 17), dtype=torch.bool, device=plan.device)
        valid = torch.cat((encoded['token_valid'], extra), 1)
        query = self.plan(window.float()) + self.age(age)[:, None]
        for block in self.blocks:
            query = block(query, memory, memory_key_padding_mask=~valid, tgt_key_padding_mask=~tail_valid)
        residual = self.output(query).float().tanh() * self.residual_limit
        residual = residual * self.arm_mask * (tail_valid & (age > 0)[:, None])[..., None].float()
        if return_aux:
            return dict(residual=residual, predicted_latents=predicted, **encoded)
        return residual
