"""A/Z joint flow matching; physical clocks are subtracted in FP64.

Future labels and visibility never enter conditioning. QueryContext is an
ephemeral differentiable per-query cache, never an inter-rate native KV updater.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vision import ResNet18Stage3


@dataclass(frozen=True)
class FastConfig:
    dim: int = 384
    layers: int = 6
    heads: int = 6
    action_dim: int = 14
    chunk_size: int = 16
    future_dim: int = 134
    future_tokens: int = 16
    observation_dim: int = 384
    world_dim: int = 384
    future_entities: int = 4
    mlp_ratio: int = 4
    slow_gate_init: float = 0.1
    test_fixture: bool = False

    def __post_init__(self):
        for name, value in vars(self).items():
            if name not in ("test_fixture", "slow_gate_init") and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads or self.future_tokens % self.future_entities:
            raise ValueError("dim must be divisible by heads; future_tokens by future_entities")
        if not 0 < self.slow_gate_init < 1:
            raise ValueError("slow_gate_init must be strictly between zero and one")


def _absolute_time(value, name):
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must be float64 absolute seconds")
    return value


class _TimeFeatures(nn.Module):
    def __init__(self, inputs, dim):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor([.25, 1., 4., 16.]))
        self.projection = nn.Sequential(nn.Linear(inputs * 9, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, seconds):
        x = torch.nan_to_num(seconds.float(), nan=0., posinf=60., neginf=-60.).clamp(-60, 60)
        phase = x.unsqueeze(-1) * self.frequencies.float() * math.pi
        features = torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        return self.projection(features.to(self.projection[0].weight.dtype))


class _QKNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        x = value.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(value.dtype) * self.weight.to(value.dtype)


def _version(value):
    try:
        return value._version
    except RuntimeError:  # Inference tensors have no version counter.
        return None


@dataclass(frozen=True)
class QueryContext:
    owner: int
    inputs: dict
    versions: dict
    parameter_versions: tuple
    kv: tuple
    current_allowed: Tensor
    slow_allowed: Tensor
    time_features: Tensor
    action_deviation: Tensor
    future_valid: Tensor


class _JointBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.norm_self = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.norm_current = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.norm_slow = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(cfg.dim, elementwise_affine=False)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.self_output = nn.Linear(cfg.dim, cfg.dim)
        # Shared projections, separate E/M attention normalization and masks.
        self.query = nn.Linear(cfg.dim, cfg.dim)
        self.context_kv = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.cross_output = nn.Linear(cfg.dim, cfg.dim)
        self.q_norm, self.k_norm = _QKNorm(cfg.dim // cfg.heads), _QKNorm(cfg.dim // cfg.heads)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, 6 * cfg.dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        # The gate must be able to respond to *current* evidence, not only age
        # and executed-plan disagreement. The latest-observation attention
        # response is query-specific and already available in this block.
        self.slow_gate = nn.Linear(2 * cfg.dim + 1, 1)
        nn.init.zeros_(self.slow_gate.weight)
        nn.init.constant_(self.slow_gate.bias, math.log(cfg.slow_gate_init / (1 - cfg.slow_gate_init)))
        self.ff = nn.Sequential(nn.Linear(cfg.dim, cfg.dim * cfg.mlp_ratio), nn.GELU(), nn.Linear(cfg.dim * cfg.mlp_ratio, cfg.dim))

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, x.shape[-1] // self.heads).transpose(1, 2)

    @staticmethod
    def _merge(x):
        return x.transpose(1, 2).flatten(2)

    def prepare_context(self, current, slow):
        result = []
        for context in (current, slow):
            k, v = self.context_kv(context).chunk(2, -1)
            k, v = self.k_norm(self._heads(k)), self._heads(v)
            result.extend((F.pad(k, (0, 0, 1, 0)), F.pad(v, (0, 0, 1, 0))))
        return tuple(result)

    def _cross(self, x, k, v, allowed):
        q = self.q_norm(self._heads(self.query(x)))
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        output = self.cross_output(self._merge(mixed))
        # Sentinel is enabled iff no real key is readable; reuse that bit rather
        # than reducing the full static key mask at every denoising step.
        return torch.where(~allowed[..., :1], output, torch.zeros_like(output))

    def forward(self, x, valid, kv, current_allowed, slow_allowed, condition, deviation):
        sa, sc, ca, cc, fa, fc = self.modulation(condition).chunk(6, -1)
        q, k, v = self.qkv(self.norm_self(x) * (1 + sc) + sa).chunk(3, -1)
        mixed = F.scaled_dot_product_attention(self.q_norm(self._heads(q)), self.k_norm(self._heads(k)), self._heads(v), attn_mask=valid[:, None, None, :])
        x = x + self.self_output(self._merge(mixed))
        latest = self._cross(self.norm_current(x) * (1 + cc) + ca, kv[0], kv[1], current_allowed)
        x = x + latest
        slow = self._cross(self.norm_slow(x) * (1 + cc) + ca, kv[2], kv[3], slow_allowed)
        gate = self.slow_gate(torch.cat((condition, self.norm_current(latest),
            deviation[:, None, None].expand(-1, x.shape[1], 1)), -1)).sigmoid()
        x = x + gate * slow
        x = x + self.ff(self.norm_ff(x) * (1 + fc) + fa)
        return torch.where(valid[..., None], x, torch.zeros_like(x))


class FastJointWAM(nn.Module):
    """World type 0 is history, 1 future prediction, 2 soft action reference.

    Caller applies episode/TTL validity to observed history. Future M must cover
    a target and have remaining support. Soft Aref describes source-to-target
    actions, including known executed history. Direct reads are time masked;
    joint A/Z attention may propagate information between generated tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.action_in = nn.Linear(c.action_dim, c.dim)
        self.future_in = nn.Linear(c.future_dim, c.dim)
        self.observation_in = nn.Linear(c.observation_dim, c.dim)
        self.world_in = nn.Linear(c.world_dim, c.dim)
        self.world_delta_in = nn.Linear(c.action_dim + 1, c.dim)
        self.world_type_embedding = nn.Embedding(3, c.dim)
        self.type_embedding = nn.Embedding(4, c.dim)
        self.known_embedding = nn.Embedding(2, c.dim)
        self.action_position = nn.Embedding(c.chunk_size, c.dim)
        self.entity_embedding = nn.Embedding(c.future_entities, c.dim)
        self.future_position = nn.Embedding(c.future_tokens // c.future_entities, c.dim)
        self.query_time_embedding = _TimeFeatures(4, c.dim)
        self.world_time_embedding = _TimeFeatures(2, c.dim)
        self.sigma_embedding = _TimeFeatures(1, c.dim)
        self.blocks = nn.ModuleList([_JointBlock(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.dim)
        self.action_out = nn.Linear(c.dim, c.action_dim)
        self.future_out = nn.Linear(c.dim, c.future_dim)
        self.contact_out = nn.Linear(c.dim, 2)

    def prepare_context(self, *, observation_tokens, observation_valid, world_tokens,
                        world_valid, world_intervals, target_times, query_time,
                        source_time, completed_time, world_types=None,
                        world_action_delta=None, world_action_delta_valid=None, future_valid=None):
        c = self.config
        inputs = dict(observation_tokens=observation_tokens, observation_valid=observation_valid,
            world_tokens=world_tokens, world_valid=world_valid, world_intervals=world_intervals,
            target_times=target_times, query_time=query_time, source_time=source_time,
            completed_time=completed_time, world_types=world_types, world_action_delta=world_action_delta,
            world_action_delta_valid=world_action_delta_valid, future_valid=future_valid)
        if any(_version(value) is None for value in inputs.values() if isinstance(value, Tensor)):
            raise ValueError("query cache requires versioned tensors; use no_grad rather than inference_mode")
        batch, length = query_time.shape[0], c.chunk_size + c.future_tokens
        if observation_tokens.shape != (*observation_valid.shape, c.observation_dim) or world_tokens.shape != (*world_valid.shape, c.world_dim):
            raise ValueError("context token/mask dimensions do not match FastConfig")
        if observation_tokens.shape[0] != batch or world_tokens.shape[0] != batch or target_times.shape != (batch, length):
            raise ValueError("context batch/target_times mismatch")
        if world_intervals.shape != (*world_valid.shape, 2):
            raise ValueError("world_intervals must be [B,M,2]")
        for name in ("query_time", "source_time", "completed_time", "target_times", "world_intervals"):
            _absolute_time(inputs[name], name)
        for name in ("query_time", "source_time", "completed_time"):
            if inputs[name].shape != (batch,):
                raise ValueError(f"{name} must be [B]")
        if any(not torch.isfinite(inputs[name]).all() for name in ("query_time", "source_time", "completed_time", "target_times")):
            raise ValueError("query, source, completion and requested target clocks must be finite")
        if not c.test_fixture and (world_types is None or world_action_delta is None or world_action_delta_valid is None):
            raise ValueError("production context requires world_types and explicit action deviation/mask")
        if world_types is None:
            world_types = torch.ones_like(world_valid, dtype=torch.long)
        if world_types.shape != world_valid.shape or world_types.dtype != torch.long or ((world_types < 0) | (world_types > 2)).any():
            raise ValueError("world_types must be int64 [B,M] in {0,1,2}")
        if world_action_delta is None:
            world_action_delta = world_tokens.new_zeros((*world_valid.shape, c.action_dim))
        if world_action_delta_valid is None:
            world_action_delta_valid = torch.zeros_like(world_valid, dtype=torch.bool)
        if world_action_delta.shape != (*world_valid.shape, c.action_dim) or world_action_delta_valid.shape != world_valid.shape:
            raise ValueError("world action deviation/mask shape mismatch")
        if future_valid is None:
            future_valid = torch.ones((batch, c.future_tokens), device=query_time.device, dtype=torch.bool)
        if future_valid.shape != (batch, c.future_tokens):
            raise ValueError("future_valid must be [B,Z]")
        age, latency = query_time - source_time, completed_time - source_time
        relative = torch.stack((target_times - query_time[:, None], target_times - source_time[:, None], age[:, None].expand(-1, length), latency[:, None].expand(-1, length)), -1)
        times = self.query_time_embedding(relative)
        current = self.observation_in(torch.where(observation_valid[..., None], observation_tokens, torch.zeros_like(observation_tokens)).to(self.observation_in.weight.dtype)) + self.type_embedding.weight[2]
        available = torch.isfinite(query_time) & torch.isfinite(source_time) & torch.isfinite(completed_time) & (completed_time <= query_time) & (source_time <= completed_time)
        start, end = world_intervals.unbind(-1)
        interval_valid = torch.isfinite(world_intervals).all(-1) & (end >= start)
        if not c.test_fixture and not interval_valid.all():
            raise ValueError("world intervals must have finite ordered physical support")
        eligible = world_valid.bool() & available[:, None] & interval_valid
        observed, predicted, action = (world_types == i for i in range(3))
        obs_allowed = observed[:, None] & (end[:, None] <= query_time[:, None, None])
        pred_allowed = predicted[:, None] & (end[:, None] >= query_time[:, None, None]) & (target_times[..., None] >= start[:, None]) & (target_times[..., None] <= end[:, None])
        # Keep the relevant source-to-target plan history. At an exact boundary
        # the ending command belongs to the target; the next row has not begun.
        action_allowed = action[:, None] & (end[:, None] >= source_time[:, None, None]) & (start[:, None] < target_times[..., None] - 1e-8)
        allowed = eligible[:, None] & (obs_allowed | pred_allowed | action_allowed)
        readable = allowed.any(1)
        slow = self.world_in(torch.where(readable[..., None], world_tokens, torch.zeros_like(world_tokens)).to(self.world_in.weight.dtype)) + self.type_embedding.weight[3] + self.world_type_embedding(world_types)
        slow = slow + self.world_time_embedding(world_intervals - query_time[:, None, None])
        dv = world_action_delta_valid.bool() & readable & action
        delta = torch.where(dv[..., None], world_action_delta, torch.zeros_like(world_action_delta)).to(self.world_delta_in.weight.dtype)
        df = self.world_delta_in(torch.cat((delta, dv[..., None].to(delta.dtype)), -1))
        slow = slow + torch.where(action[..., None], df, torch.zeros_like(df))
        deviation = delta.float().abs().sum((1, 2)) / (dv.sum(1) * c.action_dim).clamp_min(1)
        current_allowed = observation_valid[:, None].bool().expand(-1, length, -1)
        current_allowed = torch.cat((~current_allowed.any(-1, keepdim=True), current_allowed), -1)
        allowed = torch.cat((~allowed.any(-1, keepdim=True), allowed), -1)
        kv = tuple(block.prepare_context(current, slow) for block in self.blocks)
        return QueryContext(id(self), inputs, {k: _version(v) for k, v in inputs.items() if isinstance(v, Tensor)},
            tuple((id(p), _version(p)) for p in self.parameters()), kv, current_allowed, allowed,
            times, deviation.to(times.dtype), future_valid.bool())

    def _check_cache(self, cache, conditions=None):
        if cache.owner != id(self):
            raise ValueError("query cache belongs to a different model")
        if cache.parameter_versions != tuple((id(p), _version(p)) for p in self.parameters()):
            raise ValueError("model changed since query cache construction")
        for key, value in cache.inputs.items():
            if isinstance(value, Tensor) and _version(value) != cache.versions[key]:
                raise ValueError(f"query cache input {key} was modified in place")
            if conditions is not None and conditions.get(key) is not value:
                raise ValueError(f"query cache cannot be reused with changed {key}")

    def forward(self, actions, future, sigma, *, context_cache=None, action_known_mask=None, known_actions=None, **conditions):
        cache = self.prepare_context(**conditions) if context_cache is None else context_cache
        if context_cache is not None:
            self._check_cache(cache, conditions)
        return self.forward_cached(actions, future, sigma, cache, action_known_mask=action_known_mask, known_actions=known_actions)

    def forward_cached(self, actions, future, sigma, cache, *, future_valid=None, action_known_mask=None, known_actions=None):
        self._check_cache(cache)
        c, batch, device = self.config, actions.shape[0], actions.device
        if actions.shape != (batch, c.chunk_size, c.action_dim) or future.shape != (batch, c.future_tokens, c.future_dim) or sigma.shape != (batch,):
            raise ValueError("action/future/sigma shape does not match FastConfig")
        if cache.time_features.shape[0] != batch:
            raise ValueError("cache batch mismatch")
        if future_valid is None:
            future_valid = cache.future_valid
        elif future_valid is not cache.inputs["future_valid"]:
            raise ValueError("future validity is part of the query cache identity")
        if (known_actions is None) != (action_known_mask is None):
            raise ValueError("known_actions and action_known_mask must be supplied together")
        if action_known_mask is None:
            action_known_mask = torch.zeros((batch, c.chunk_size), dtype=torch.bool, device=device)
        else:
            if known_actions.shape != actions.shape or action_known_mask.shape != actions.shape[:2]:
                raise ValueError("known action constraint shape mismatch")
            actions = torch.where(action_known_mask[..., None], known_actions, actions)
        future = torch.where(future_valid[..., None], future, torch.zeros_like(future))
        ai, zi = torch.arange(c.chunk_size, device=device), torch.arange(c.future_tokens, device=device)
        nt = c.future_tokens // c.future_entities
        a = self.action_in(actions.to(self.action_in.weight.dtype)) + self.type_embedding.weight[0] + self.action_position(ai) + self.known_embedding(action_known_mask.long())
        z = self.future_in(future.to(self.future_in.weight.dtype)) + self.type_embedding.weight[1] + self.entity_embedding(zi // nt) + self.future_position(zi % nt)
        condition = cache.time_features + self.sigma_embedding(sigma[:, None])[:, None]
        x = torch.cat((a, z), 1) + condition
        valid = torch.cat((torch.ones_like(action_known_mask), future_valid.bool()), 1)
        for block, kv in zip(self.blocks, cache.kv):
            x = block(x, valid, kv, cache.current_allowed, cache.slow_allowed, condition, cache.action_deviation)
        x = self.norm(x)
        action, z = x[:, :c.chunk_size], x[:, c.chunk_size:]
        av, zv, contact = self.action_out(action), self.future_out(z), self.contact_out(z)
        return {"action_velocity": torch.where(action_known_mask[..., None], torch.zeros_like(av), av),
            "future_velocity": torch.where(future_valid[..., None], zv, torch.zeros_like(zv)),
            "contact_logits": torch.where(future_valid[..., None], contact, torch.zeros_like(contact))}


class CurrentObservationEncoder(nn.Module):
    """Pretrained stride16 spatial RGB, local flow, global motion, state/text.

    RGB is resized without cropping by the caller, in [0,1]; pixel_valid marks
    external letterbox padding. Bottom/right stride padding retains partial edge
    cells. Coordinates are mask-weighted centers in the supplied image plane.
    Global motion and state arrive checkpoint-normalized. Local flow is raw
    (dx,dy,vx,vy,reliable_fraction), normalized here once in FP32.
    """
    def __init__(self, dim=384, views=3, motion_dim=12, state_dim=14, text_dim=4096,
                 *, vision_weights_path=None, vision_weights_sha256=None,
                 flow_mean=None, flow_scale=None, test_fixture=False):
        super().__init__()
        self.dim, self.views, self.test_fixture = dim, views, bool(test_fixture)
        self.rgb = ResNet18Stage3(vision_weights_path, vision_weights_sha256, test_fixture=test_fixture)
        self.pretrained = self.rgb.pretrained
        if not test_fixture and (flow_mean is None or flow_scale is None):
            raise ValueError("local flow requires checkpoint mean/scale for motion indices 0,1,5,6")
        mean = torch.as_tensor([0.] * 4 if flow_mean is None else flow_mean, dtype=torch.float32)
        scale = torch.as_tensor([1.] * 4 if flow_scale is None else flow_scale, dtype=torch.float32)
        if mean.shape != (4,) or scale.shape != (4,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("flow_mean/flow_scale must be finite four-vectors with positive scales")
        self.register_buffer("flow_mean", mean)
        self.register_buffer("flow_scale", scale)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        self.rgb_projection = nn.Conv2d(256, dim, 1)
        self.xy = nn.Linear(2, dim)
        self.coverage = nn.Linear(1, dim)
        self.view_embedding = nn.Embedding(views, dim)
        self.type_embedding = nn.Embedding(5, dim)  # RGB, global motion, state, language, local flow
        self.time_embedding = _TimeFeatures(2, dim)
        self.motion = nn.Linear(motion_dim, dim)
        self.spatial_flow = nn.Linear(5, dim)
        self.motion_acceleration_embedding = nn.Embedding(2, dim)
        self.state = nn.Linear(state_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rgb, motion, state, text_tokens, *, query_time, rgb_times,
                motion_times, motion_valid, text_valid, rgb_valid=None,
                rgb_pixel_valid=None, state_time=None, motion_acceleration_valid=None,
                spatial_flow=None, spatial_flow_valid=None, spatial_flow_times=None):
        if rgb.ndim != 5 or rgb.shape[1:3] != (self.views, 3) or min(rgb.shape[-2:]) < 16:
            raise ValueError("RGB must be [B,V,3,H>=16,W>=16]")
        if not rgb.is_floating_point():
            raise TypeError("RGB must be floating point in [0,1]")
        batch, _, _, ih, iw = rgb.shape
        for name, value in (("query_time", query_time), ("rgb_times", rgb_times), ("motion_times", motion_times)):
            _absolute_time(value, name)
        if query_time.shape != (batch,) or rgb_times.shape != (batch, self.views) or motion_times.shape != (*motion.shape[:2], 2):
            raise ValueError("query/RGB/motion timestamp shape mismatch")
        if motion_valid.shape != motion.shape[:2] or text_valid.shape != text_tokens.shape[:2]:
            raise ValueError("motion/text masks must match tokens")
        if not self.test_fixture and any(x is None for x in (state_time, rgb_pixel_valid, spatial_flow, spatial_flow_valid, spatial_flow_times)):
            raise ValueError("production observation requires real state time, pixel mask, explicit local-flow values/mask/times")
        if state_time is None:
            state_time = query_time
        _absolute_time(state_time, "state_time")
        if state_time.shape != (batch,):
            raise ValueError("state_time must be [B]")
        if rgb_pixel_valid is None:
            rgb_pixel_valid = torch.ones((batch, self.views, ih, iw), dtype=torch.bool, device=rgb.device)
        if rgb_pixel_valid.shape != (batch, self.views, ih, iw):
            raise ValueError("rgb_pixel_valid must be [B,V,H,W]")
        rv = torch.isfinite(rgb_times) & (rgb_times <= query_time[:, None])
        if rgb_valid is not None:
            if rgb_valid.shape != rv.shape:
                raise ValueError("rgb_valid must be [B,V]")
            rv = rv & rgb_valid.bool()
        pixels = rgb_pixel_valid.bool() & rv[..., None, None]
        image = torch.where(pixels[:, :, None], rgb.float(), torch.zeros_like(rgb, dtype=torch.float32)).flatten(0, 1)
        if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError("valid RGB pixels must be finite in [0,1]")
        image = (image - self.image_mean.float()) / self.image_std.float()
        flat_mask = pixels.flatten(0, 1)[:, None]
        image = torch.where(flat_mask, image, torch.zeros_like(image))
        pad = (0, (-iw) % 16, 0, (-ih) % 16)
        image, pixel_float = F.pad(image, pad), F.pad(flat_mask.float(), pad)
        patches = self.rgb_projection(self.rgb(image.to(self.rgb.conv1.weight.dtype)))
        h, w = patches.shape[-2:]
        coverage = F.avg_pool2d(pixel_float, 16, 16)
        yy, xx = torch.meshgrid((torch.arange(ih, device=rgb.device, dtype=torch.float32) + .5) * (2 / ih) - 1,
                               (torch.arange(iw, device=rgb.device, dtype=torch.float32) + .5) * (2 / iw) - 1, indexing="ij")
        coordinates = torch.stack((xx, yy))[None] * flat_mask
        coordinates = F.avg_pool2d(F.pad(coordinates, pad), 16, 16) / coverage.clamp_min(1e-8)
        xy = self.xy(coordinates.flatten(2).transpose(1, 2).to(self.xy.weight.dtype))
        cv = self.coverage(coverage.flatten(2).transpose(1, 2).to(self.coverage.weight.dtype))
        patches = (patches.flatten(2).transpose(1, 2) + xy + cv).reshape(batch, self.views, h * w, self.dim)
        image_dt = (rgb_times - query_time[:, None]).unsqueeze(-1).expand(-1, -1, 2)
        patches = patches + self.view_embedding.weight[None, :, None] + self.time_embedding(image_dt)[:, :, None] + self.type_embedding.weight[0]
        image_valid = (coverage.flatten(1) > 0).reshape(batch, self.views * h * w)
        mv = motion_valid.bool() & torch.isfinite(motion_times).all(-1) & (motion_times[..., 0] < motion_times[..., 1]) & (motion_times[..., 1] <= query_time[:, None])
        if motion_acceleration_valid is None:
            motion_acceleration_valid = torch.zeros_like(mv)
        if motion_acceleration_valid.shape != mv.shape:
            raise ValueError("motion_acceleration_valid must match [B,T]")
        av = motion_acceleration_valid.bool() & mv
        fields = mv[..., None].expand_as(motion).clone()
        fields[..., 9:12] &= av[..., None]
        motion_features = self.motion(torch.where(fields, motion, torch.zeros_like(motion)).to(self.motion.weight.dtype)) + self.time_embedding(motion_times - query_time[:, None, None]) + self.type_embedding.weight[1] + self.motion_acceleration_embedding(av.long())
        sv = torch.isfinite(state_time) & (state_time <= query_time)
        state_dt = (state_time - query_time)[:, None].expand(-1, 2)
        state_features = self.state(torch.where(sv[:, None], state, torch.zeros_like(state)).to(self.state.weight.dtype)) + self.time_embedding(state_dt) + self.type_embedding.weight[2]
        text_features = self.text(torch.where(text_valid[..., None], text_tokens, torch.zeros_like(text_tokens)).to(self.text.weight.dtype)) + self.type_embedding.weight[3]
        if spatial_flow is None:
            spatial_flow = motion.new_zeros((batch, 16, 5))
            spatial_flow_valid = torch.zeros((batch, 16), device=rgb.device, dtype=torch.bool)
            spatial_flow_times = query_time[:, None].expand(-1, 2)
        if spatial_flow.shape != (batch, 16, 5) or spatial_flow_valid.shape != (batch, 16) or spatial_flow_times.shape != (batch, 2):
            raise ValueError("local flow must be [B,16,5], mask [B,16], times [B,2]")
        _absolute_time(spatial_flow_times, "spatial_flow_times")
        flow_time_valid = torch.isfinite(spatial_flow_times).all(-1) & (spatial_flow_times[:, 0] < spatial_flow_times[:, 1]) & (spatial_flow_times[:, 1] <= query_time)
        fv = spatial_flow_valid.bool() & flow_time_valid[:, None] & (spatial_flow[..., 4] > 0)
        clean_flow = torch.where(fv[..., None], spatial_flow.float(), torch.zeros_like(spatial_flow, dtype=torch.float32))
        if not torch.isfinite(clean_flow).all() or (clean_flow[..., 4] > 1).any():
            raise ValueError("valid local flow must be finite with reliability fraction in (0,1]")
        normalized = torch.cat(((clean_flow[..., :4] - self.flow_mean.float()) / self.flow_scale.float(), clean_flow[..., 4:]), -1)
        normalized = torch.where(fv[..., None], normalized, torch.zeros_like(normalized))
        fy, fx = torch.meshgrid(torch.arange(4, device=rgb.device, dtype=torch.float32) * .5 - .75, torch.arange(4, device=rgb.device, dtype=torch.float32) * .5 - .75, indexing="ij")
        flow_xy = self.xy(torch.stack((fx, fy), -1).reshape(16, 2).to(self.xy.weight.dtype))
        local = self.spatial_flow(normalized.to(self.spatial_flow.weight.dtype)) + flow_xy + self.view_embedding.weight[0] + self.type_embedding.weight[4] + self.time_embedding(spatial_flow_times - query_time[:, None])[:, None]
        tokens = self.norm(torch.cat((patches.flatten(1, 2), motion_features, state_features[:, None], text_features, local), 1))
        valid = torch.cat((image_valid, mv, sv[:, None], text_valid.bool(), fv), 1)
        return torch.where(valid[..., None], tokens, torch.zeros_like(tokens)), valid
