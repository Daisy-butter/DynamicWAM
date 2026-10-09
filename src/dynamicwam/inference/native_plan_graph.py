"""Exact native 10/2 planner with a captured eight-step action suffix.

The two joint steps, action trajectory, world cache and parameter precisions
remain native. This is an optional inference engine, not a new policy/solver.
VAE encoding and text encoding belong to the caller, outside this interface.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import torch

from dynamicwam.models.action_core_ops import materialize_attention_cache
from dynamicwam.models.flow_schedule import euler_update, scheduler_delta_product


def _signature(tensor):
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


class NativePlanSampler:
    def __init__(self, backbone, schedule, enabled=True, warmup=3, omit_unused_joint_outputs=False):
        if len(schedule.action_sigmas) != 10 or len(schedule.video_sigmas) != 2:
            raise ValueError("native plan sampler requires the official 10 action / 2 video schedule")
        if warmup < 1:
            raise ValueError("positive graph warmup required")
        self.backbone, self.schedule = backbone, schedule
        # Production HeadFlowRunner fixes scheduler embeddings to branch
        # parameter precision BEFORE integration. Video state promotes to FP32
        # after its first prediction; that must not silently change video_t.
        self.action_dtype = next(backbone.action_expert.parameters()).dtype
        self.video_dtype = backbone.compact_wan.video_model.precision
        self.enabled, self.warmup = bool(enabled), int(warmup)
        self.omit_unused_joint_outputs = bool(omit_unused_joint_outputs)
        self.graph = None
        self.signature = None
        self.stats = dict(captures=0, replays=0, eager_calls=0, graph_enabled=False,
                          last_capture_ms=None, last_signature=None,
                          scope="two eager joint steps plus eight captured native action steps")

    def invalidate(self):
        self.graph, self.signature = None, None
        for key in ("static_batch", "static_actions", "static_cache", "timesteps", "deltas", "output"):
            if hasattr(self, key):
                delattr(self, key)

    @contextmanager
    def _validated_motion(self):
        previous = getattr(self.backbone, "_motion_inputs_prevalidated", False)
        self.backbone._motion_inputs_prevalidated = True
        try:
            yield
        finally:
            self.backbone._motion_inputs_prevalidated = previous

    def _suffix(self):
        actions = self.static_actions
        for timestep, delta in zip(self.timesteps, self.deltas, strict=True):
            batch = dict(self.static_batch, noisy_actions=actions, action_t=timestep,
                         video_cache=self.static_cache)
            velocity = self.backbone.forward_action_with_video_cache(batch)["action_pred"]
            actions = actions + scheduler_delta_product(velocity, delta)
        return actions

    def _capture(self, batch, actions, cache):
        device = actions.device
        self.static_actions = actions.detach().clone()
        keys = ("initial_state", "absolute_motion_features", "absolute_motion_interval_valid_mask",
                "absolute_motion_acceleration_valid_mask")
        self.static_batch = {key: batch[key].detach().clone() for key in keys}
        self.static_cache = dict(cache, video_k=[x.detach().clone() for x in cache["video_k"]],
                                 video_v=[x.detach().clone() for x in cache["video_v"]],
                                 seq_lens=cache["seq_lens"].detach().clone())
        self.static_cache = materialize_attention_cache(self.static_cache, device=device,
                                 dtype=self.backbone.action_attention_spec.precision)
        self.timesteps, self.deltas = [], []
        for index in range(2, 10):
            sigma, next_sigma, timestep = self.schedule.action_step(index)
            self.timesteps.append(timestep.to(device=device, dtype=self.action_dtype).reshape(1).expand(actions.shape[0]))
            self.deltas.append((next_sigma - sigma).to(device=device))
        start = time.perf_counter()
        active = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(active)
        with torch.cuda.stream(side), self._validated_motion():
            for _ in range(self.warmup):
                self._suffix()
        active.wait_stream(side)
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            graph = torch.cuda.CUDAGraph()
            with self._validated_motion(), torch.cuda.graph(graph):
                self.output = self._suffix()
        self.graph = graph
        self.stats["captures"] += 1
        self.stats["last_capture_ms"] = (time.perf_counter() - start) * 1000

    def _copy(self, batch, actions, cache):
        self.static_actions.copy_(actions)
        for key, value in self.static_batch.items():
            value.copy_(batch[key])
        self.static_cache["seq_lens"].copy_(cache["seq_lens"])
        for key in ("video_k", "video_v"):
            for destination, source in zip(self.static_cache[key], cache[key], strict=True):
                destination.copy_(source)

    @torch.inference_mode()
    def sample(self, source, noise):
        if any(module.training for module in self.backbone.modules()):
            raise RuntimeError("native plan sampler is restricted to eval mode")
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise RuntimeError("native plan sampler requires frozen inference weights")
        if noise.actions.is_cuda and torch.is_autocast_enabled("cuda"):
            raise RuntimeError("native plan sampler preserves explicit native precision; disable autocast")
        batch = source.as_model_batch()
        actions, future, cache = noise.actions, noise.future_latent, None
        # Identical to sample_native_wam: continue this very action trajectory
        # after BOTH joint updates; never reset it to fresh Gaussian noise.
        for index in range(2):
            sigma, next_sigma, timestep = self.schedule.action_step(index)
            step = dict(batch, noisy_actions=actions,
                action_t=timestep.to(device=actions.device, dtype=self.action_dtype).reshape(1).expand(actions.shape[0]), future_latent=future,
                video_t=self.schedule.video_timesteps[index].to(device=future.device,
                    dtype=self.video_dtype).reshape(1).expand(actions.shape[0]))
            # The first cache is discarded; the second video's prediction is
            # never integrated into the action-only suffix. Keep the legacy
            # default unchanged and certify this optional pruning vs the runner.
            output = self.backbone._forward_video_action(step,
                include_video_prediction=(index == 0 or not self.omit_unused_joint_outputs),
                build_video_cache=(index == 1 or not self.omit_unused_joint_outputs), detach_cache=True)
            if index == 1:
                cache = output["video_cache"]
            if index == 0:
                vs, vn, _ = self.schedule.video_pair(index)
                future = euler_update(future, output["video_pred"], vs, vn)
            actions = euler_update(actions, output["action_pred"], sigma, next_sigma)
        use_graph = self.enabled and actions.is_cuda
        self.stats["graph_enabled"] = use_graph
        if not use_graph:
            self.stats["eager_calls"] += 1
            for index in range(2, 10):
                sigma, next_sigma, timestep = self.schedule.action_step(index)
                output = self.backbone.forward_action_with_video_cache(dict(batch,
                    noisy_actions=actions, action_t=timestep.to(device=actions.device,
                        dtype=self.action_dtype).reshape(1).expand(actions.shape[0]),
                    video_cache=cache))
                actions = euler_update(actions, output["action_pred"], sigma, next_sigma)
            return actions, cache
        if not cache.get("all_video_tokens_valid", False):
            raise RuntimeError("native graph suffix requires a certified dense native cache")
        self.backbone.absolute_motion_tokens.validate_inputs(source.motion,
            source.motion_interval_valid_mask, source.motion_acceleration_valid_mask)
        signature = tuple(_signature(value) for value in (actions, source.state, source.motion,
            source.motion_interval_valid_mask, source.motion_acceleration_valid_mask,
            *cache["video_k"], *cache["video_v"], cache["seq_lens"]))
        try:
            if self.signature != signature:
                self.invalidate()
                self._capture(batch, actions, cache)
                self.signature = signature
                self.stats["last_signature"] = repr(signature)
            else:
                self._copy(batch, actions, cache)
            with torch.cuda.device(actions.device):
                self.graph.replay()
            self.stats["replays"] += 1
            return self.output.clone(), cache
        except Exception as error:
            self.invalidate()
            raise RuntimeError("native plan CUDA graph failed; no silent eager fallback") from error
