"""Frozen native fast-query CUDA graph with explicit static input ownership.

This captures observation routing and the complete trained action solver.
It changes launch scheduling only. Every query copies observation, age, text,
noise and world K/V; replays never silently retain an old action or old bridge.
Capture errors are fatal when enabled. CPU/explicitly disabled use is eager.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import time

import torch
from dynamicwam.models.flow_schedule import scheduler_delta_product

from dynamicwam.models.action_core_ops import materialize_attention_cache
from dynamicwam.models.multirate_types import BridgeAge
from dynamicwam.models.rtc_prefix import PrefixCondition, apply_prefix, validate_prefix_condition


def _tensor_signature(value):
    if value is None:
        return None
    return (tuple(value.shape), str(value.dtype), str(value.device))


class NativeFastGraph:
    def __init__(self, model, *, enabled=True, warmup=3):
        if not getattr(model, "native_multirate", False):
            raise ValueError("native graph requires the native multirate architecture")
        if warmup < 1:
            raise ValueError("CUDA graph warmup must be positive")
        self.model = model
        self.requested_enabled = bool(enabled)
        self.warmup = int(warmup)
        self.graph = None
        self.signature = None
        self.stats = dict(enabled=False, captures=0, replays=0, eager_calls=0,
                          last_capture_ms=None, last_signature=None)

    def invalidate(self):
        self.graph = None
        self.signature = None
        for name in ("current", "bridge", "conditioning_bridge", "text", "text_mask", "noise", "prefix", "output", "timesteps", "deltas", "video_cache"):
            if hasattr(self, name):
                delattr(self, name)

    def _check_model(self):
        if self.model.training or any(m.training for m in self.model.modules()):
            raise RuntimeError("native graph requires eval mode for every module")
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("native graph requires frozen inference weights")

    def _signature(self, current, bridge, text, text_mask, noise, prefix):
        # Values remain dynamic, including both clocks; only shape/dtype and a
        # host-side bool branch are part of the capture contract.
        null = bridge.null_bridge
        null_signature = _tensor_signature(null) if torch.is_tensor(null) else ("host_bool", bool(null))
        return (tuple(_tensor_signature(v) for v in (current.rgb, current.rgb_valid_mask,
                    current.state, current.motion, current.motion_interval_valid_mask,
                    current.motion_acceleration_valid_mask, current.age.state_delta, text, text_mask, noise,
                    prefix.actions, prefix.lengths)),
                tuple(_tensor_signature(v) for v in (*bridge.video_k, *bridge.video_v, bridge.seq_lens)),
                null_signature, bool(bridge.all_video_tokens_valid),
                tuple(float(v) for v in self.model.fast_schedule.action_sigmas.cpu()),
                tuple(float(v) for v in self.model.fast_schedule.action_timesteps.cpu()))

    @staticmethod
    def _age_inputs(current, device):
        dtype = current.age.state_delta.dtype
        batch = current.state.shape[0]
        def column(value, target_dtype):
            tensor = torch.as_tensor(value, device=device, dtype=target_dtype).reshape(-1)
            if tensor.numel() == 1:
                tensor = tensor.expand(batch)
            if tensor.numel() != batch:
                raise ValueError("native graph age batch mismatch")
            return tensor
        return BridgeAge(column(current.age.delta_sim_seconds, dtype),
                         column(current.age.delta_actions, dtype),
                         column(current.age.bridge_valid, torch.bool), current.age.state_delta)

    def _allocate(self, current, bridge, text, text_mask, noise, prefix):
        device = noise.device
        def clone(value):
            return None if value is None else value.detach().to(device=device).clone()
        age = self._age_inputs(current, device)
        self.current = replace(current, rgb=clone(current.rgb), rgb_valid_mask=clone(current.rgb_valid_mask),
            state=clone(current.state), motion=clone(current.motion),
            motion_interval_valid_mask=clone(current.motion_interval_valid_mask),
            motion_acceleration_valid_mask=clone(current.motion_acceleration_valid_mask),
            age=BridgeAge(clone(age.delta_sim_seconds), clone(age.delta_actions),
                          clone(age.bridge_valid), clone(age.state_delta)))
        self.bridge = replace(bridge, video_k=tuple(clone(t) for t in bridge.video_k),
            video_v=tuple(clone(t) for t in bridge.video_v), seq_lens=clone(bridge.seq_lens),
            null_bridge=clone(bridge.null_bridge) if torch.is_tensor(bridge.null_bridge) else bool(bridge.null_bridge))
        # The router canonicalizes null flags with torch.as_tensor(device=...);
        # pre-materialize that scalar so capture does not copy a host bool. Keep
        # the action core's host bool branch to avoid two attention evaluations.
        self.conditioning_bridge = replace(self.bridge, null_bridge=torch.as_tensor(
            self.bridge.null_bridge, device=device, dtype=torch.bool))
        self.text, self.text_mask, self.noise = clone(text), clone(text_mask), clone(noise)
        self.prefix = PrefixCondition(clone(prefix.actions), clone(prefix.lengths))
        self.video_cache = materialize_attention_cache(self.bridge.as_video_cache(), device=device,
            dtype=self.model.backbone.action_attention_spec.precision)
        # All solver constants are materialized before capture; constructing a
        # CPU terminal sigma and copying it inside the graph is forbidden.
        self.timesteps, self.deltas = [], []
        for index in range(len(self.model.fast_schedule.action_sigmas)):
            sigma, next_sigma, timestep = self.model.fast_schedule.action_step(index)
            self.timesteps.append(timestep.to(device=device, dtype=noise.dtype).reshape(1).expand(noise.shape[0]))
            self.deltas.append((next_sigma - sigma).to(device=device))

    def _copy_inputs(self, current, bridge, text, text_mask, noise, prefix):
        for name in ("rgb", "state", "motion", "motion_interval_valid_mask", "motion_acceleration_valid_mask"):
            getattr(self.current, name).copy_(getattr(current, name))
        if self.current.rgb_valid_mask is not None:
            self.current.rgb_valid_mask.copy_(current.rgb_valid_mask)
        age = self._age_inputs(current, noise.device)
        for name in ("delta_sim_seconds", "delta_actions", "bridge_valid", "state_delta"):
            getattr(self.current.age, name).copy_(getattr(age, name))
        self.text.copy_(text)
        if self.text_mask is not None:
            self.text_mask.copy_(text_mask)
        self.noise.copy_(noise)
        # Refresh EVERY replay, including prefix=None (canonical d=0).
        self.prefix.actions.copy_(prefix.actions)
        self.prefix.lengths.copy_(prefix.lengths)
        self.bridge.seq_lens.copy_(bridge.seq_lens)
        if torch.is_tensor(self.bridge.null_bridge):
            self.bridge.null_bridge.copy_(bridge.null_bridge)
            self.conditioning_bridge.null_bridge.copy_(bridge.null_bridge)
        for dest, source in zip((*self.bridge.video_k, *self.bridge.video_v),
                                (*bridge.video_k, *bridge.video_v), strict=True):
            dest.copy_(source)
        # Materialization may cast a raw FP32 source cache to BF16. Refresh the
        # attention-ready copies too when they do not share the static storage.
        for raw, ready in zip((*self.bridge.video_k, *self.bridge.video_v),
                              (*self.video_cache["video_k"], *self.video_cache["video_v"]), strict=True):
            if raw.data_ptr() != ready.data_ptr():
                ready.copy_(raw)

    @contextmanager
    def _validated_motion(self):
        core = self.model.backbone
        previous = getattr(core, "_motion_inputs_prevalidated", False)
        core._motion_inputs_prevalidated = True
        try:
            yield
        finally:
            core._motion_inputs_prevalidated = previous

    def _forward(self):
        memory = self.model.prepare_fast_memory(self.current, self.conditioning_bridge, self.text, self.text_mask)
        actions = self.noise
        for timestep, delta in zip(self.timesteps, self.deltas, strict=True):
            velocity = self.model.predict_fast_velocity(actions, timestep, self.current,
                self.bridge, memory, video_cache=self.video_cache, prefix=self.prefix)
            actions = actions + scheduler_delta_product(velocity, delta)
            actions = apply_prefix(actions, self.prefix)
        return actions

    def _capture(self):
        start = time.perf_counter()
        device = self.noise.device
        active = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(active)
        with torch.cuda.stream(side), self._validated_motion():
            for _ in range(self.warmup):
                self._forward()
        active.wait_stream(side)
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            graph = torch.cuda.CUDAGraph()
            with self._validated_motion(), torch.cuda.graph(graph):
                self.output = self._forward()
        self.graph = graph
        self.stats["captures"] += 1
        self.stats["last_capture_ms"] = (time.perf_counter() - start) * 1000

    @torch.inference_mode()
    def sample(self, current, bridge, text, text_mask, noise, prefix=None):
        self._check_model()
        if prefix is None:
            prefix = PrefixCondition(torch.zeros_like(noise),
                torch.zeros(noise.shape[0], device=noise.device, dtype=torch.long))
        # Value checks are deliberately outside capture. Invalid delays fail;
        # they must never be clipped silently to the trained maximum.
        validate_prefix_condition(prefix,
            max_prefix=int(getattr(self.model, "rtc_prefix_max", 0)), reference=noise)
        use_graph = self.requested_enabled and noise.is_cuda
        self.stats["enabled"] = use_graph
        if not use_graph:
            self.stats["eager_calls"] += 1
            memory = self.model.prepare_fast_memory(current, bridge, text, text_mask)
            return self.model.sample_fast_actions(current, bridge, memory, noise, prefix=prefix)
        if len(self.model.fast_schedule.action_sigmas) not in (1, 2, 4, 10):
            raise ValueError("native CUDA graph supports trained 1/2/4/10-step inference only")
        if not bridge.all_video_tokens_valid:
            raise ValueError("native CUDA graph requires a certified dense world cache")
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError("native graph must run under the normal explicit-dtype inference policy")
        self.model.backbone.absolute_motion_tokens.validate_inputs(current.motion,
            current.motion_interval_valid_mask, current.motion_acceleration_valid_mask)
        signature = self._signature(current, bridge, text, text_mask, noise, prefix)
        try:
            if signature != self.signature:
                self.invalidate()
                self._allocate(current, bridge, text, text_mask, noise, prefix)
                self._capture()
                self.signature = signature
                self.stats["last_signature"] = repr(signature)
            else:
                self._copy_inputs(current, bridge, text, text_mask, noise, prefix)
            with torch.cuda.device(noise.device):
                self.graph.replay()
            self.stats["replays"] += 1
            # The caller owns its result; the next replay must not mutate a
            # previously returned action plan while a controller consumes it.
            return self.output.clone()
        except Exception as error:
            self.invalidate()
            raise RuntimeError("native fast CUDA graph failed; eager fallback is disabled") from error
