"""Causal-spatial H2/R2 conditioning with the original synchronous executor.

DOMINO owns all physics. Actual capture clocks condition motion, belief and M
age. The .16-second requested model offsets and action-index queue are nominal
prediction coordinates; variable-duration native TOPP does NOT execute those
physical endpoints. This compatibility protocol makes no continuous/1x claim.
"""
from __future__ import annotations

import time
import numpy as np

from .causal_spatial_policy import CausalSpatialPolicy, capture_metadata


NATIVE_PROTOCOL = "native_domino_causal_spatial_h2r2_nominal_targets_v1"
MODEL_OFFSETS = np.arange(1, 17, dtype=np.float64) * .16


class NativeRTCPrefixQueue:
    """Immutable next two commands; replace only the uncommitted suffix.

Indices count native take_action calls, never physics ticks. There are no wall
deadlines because inference runs while the simulator is paused.
"""
    def __init__(self, initial_targets):
        targets = np.asarray(initial_targets, dtype=np.float64)
        if targets.shape != (14,) or not np.isfinite(targets).all():
            raise ValueError("native bootstrap needs finite fourteen-dimensional drive targets")
        self.rows = {0: targets.copy(), 1: targets.copy()}
        self.origin = 0
        self.generation = 0
        self.last_query_origin = None
        self._reservation = None

    def query(self):
        if self.origin % 2 or self.last_query_origin == self.origin or self._reservation is not None:
            raise RuntimeError("native H2/R2 requires one query at each even action origin")
        try:
            prefix = np.stack([self.rows[self.origin + i] for i in range(2)])
        except KeyError as exc:
            raise RuntimeError("native committed prefix is missing") from exc
        previous = np.zeros((16, 14), dtype=np.float64)
        valid = np.zeros(16, dtype=bool)
        if self.generation:
            # Shift the prior plan by the two completed action indices. Only
            # its replaceable overlap rows 2:4 enter the training mask.
            for i in (2, 3):
                if self.origin + i in self.rows:
                    previous[i] = self.rows[self.origin + i]
                    valid[i] = True
        self._reservation = (self.origin, self.generation, prefix.copy())
        return prefix.copy(), previous, valid

    def install(self, actions):
        if self._reservation is None:
            raise RuntimeError("native result has no owning prefix reservation")
        origin, generation, prefix = self._reservation
        if origin != self.origin or generation != self.generation:
            raise RuntimeError("native result belongs to a stale queue generation")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (16, 14) or not np.isfinite(actions).all():
            raise ValueError("native result must be finite [16,14]")
        if not np.array_equal(np.stack([self.rows[origin+i] for i in range(2)]), prefix):
            raise RuntimeError("native committed prefix changed during inference")
        if not np.allclose(actions[:2], prefix, atol=2e-5, rtol=1e-5):
            raise ValueError("native sampler changed the committed prefix")
        self.rows = {origin+i: (prefix[i].copy() if i < 2 else actions[i].copy()) for i in range(16)}
        self.generation += 1
        self.last_query_origin = origin
        self._reservation = None

    def current(self):
        if self._reservation is not None or (self.origin % 2 == 0 and self.last_query_origin != self.origin):
            raise RuntimeError("native action requires its H2 query to finish")
        if self.origin not in self.rows:
            raise RuntimeError("native action queue underrun")
        return self.rows[self.origin].copy()

    def acknowledge(self, native_action_count):
        if int(native_action_count) != self.origin + 1:
            raise RuntimeError("one native take_action call must advance exactly one action")
        self.current()
        self.rows.pop(self.origin)
        self.origin += 1


def validate_native_packet(packet):
    capture_metadata(packet)
    if packet.get("_native_protocol") != NATIVE_PROTOCOL:
        raise ValueError("not a native synchronous causal-spatial packet")
    origin = packet["_native_action_origin"]
    if type(origin) is not int or origin < 0 or origin % 2:
        raise ValueError("native queries require nonnegative even action origins")
    if packet.get("_native_prefix_length") != 2:
        raise ValueError("native compatibility sampling must retain trained H2 conditioning")
    if not np.isfinite(float(packet["sim_time"])):
        raise ValueError("native capture physics time must be finite")
    prefix = np.asarray(packet["_native_prefix_actions"], dtype=np.float64)
    offsets = np.asarray(packet["_native_model_target_offsets"], dtype=np.float64)
    previous = np.asarray(packet["_native_previous_actions"], dtype=np.float32)
    valid = np.asarray(packet["_native_previous_valid"], dtype=bool)
    if prefix.shape != (2, 14) or offsets.shape != (16,) or previous.shape != (16, 14) or valid.shape != (16,):
        raise ValueError("invalid native causal-spatial packet geometry")
    if not np.isfinite(prefix).all() or not np.isfinite(previous[valid]).all():
        raise ValueError("nonfinite native prefix/proposal")
    if not np.array_equal(offsets, MODEL_OFFSETS):
        raise ValueError("native model offsets must preserve the requested .16-second horizon")
    overlap = np.zeros(16, dtype=bool)
    if origin:
        overlap[2:4] = valid[2:4]
    return origin, prefix.copy(), offsets.copy(), np.where(overlap[:, None], previous, 0), overlap


class NativeCausalSpatialPolicy(CausalSpatialPolicy):
    """Reuse frozen loader, normalization, M, K10 model and production Graph."""
    def _new_episode(self):
        super()._new_episode()
        self._native_last_physics_s = None

    def observe_native(self, packet):
        capture_metadata(packet)
        if packet.get("_native_protocol") != NATIVE_PROTOCOL:
            raise ValueError("native observation protocol mismatch")
        physics = float(packet["sim_time"])
        if not np.isfinite(physics) or (self._native_last_physics_s is not None and physics <= self._native_last_physics_s):
            raise ValueError("native observations must retain advancing actual physics clocks")
        observation = self._observe(packet)
        self._native_last_physics_s = physics
        return observation

    def _native_memory(self, packet, observation, origin):
        """Wait only for native compatibility; never relax the 4.96s age bound."""
        waited = []
        self._slow_status()
        now = float(packet["sim_time"])
        current = self.memories.current
        needs_memory = current is None or now-current.source.observation_physics_s > self.contract.max_memory_age_s
        if needs_memory and self._future is not None:
            started = time.perf_counter()
            self._future.result(timeout=120.)
            self._slow_status()
            waited.append(dict(reason="await_inflight_memory", milliseconds=1000*(time.perf_counter()-started)))
            current = self.memories.current
        if current is None or now-current.source.observation_physics_s > self.contract.max_memory_age_s:
            reason = "bootstrap_current_observation_memory" if current is None else "native_topp_memory_age_refresh"
            started = time.perf_counter()
            if not self._submit_memory(packet, observation, origin):
                raise RuntimeError("native synchronous M refresh unexpectedly already in flight")
            self._future.result(timeout=120.)
            self._slow_status()
            waited.append(dict(reason=reason, milliseconds=1000*(time.perf_counter()-started)))
        return self.memories.snapshot(observation_physics_s=now, request_wall_s=time.perf_counter()), waited

    def sample(self, packet):
        import torch
        started = time.perf_counter()
        if self._text is None or self.vae is None:
            raise RuntimeError("prepare_instruction must finish before native sampling")
        origin, prefix, offsets, previous, valid = validate_native_packet(packet)
        observation = self.observe_native(packet)
        memory, memory_waits = self._native_memory(packet, observation, origin)
        batch = self._model_batch(packet, observation, memory, prefix, offsets, previous, valid)
        generator = torch.Generator(device=self.device).manual_seed(int(packet["_sampling_seed"]))
        noise = (torch.randn((1, 16, 14), device=self.device, generator=generator),
                 torch.randn((1, 48, 2, 12, 10), device=self.device, generator=generator))
        fast_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self._graph is None:
                output = self.model.sample(batch, action_noise=noise[0], future_noise=noise[1],
                                           sigmas=self.sigmas, memory=memory.payload)
            else:
                output = self._graph.replay(batch, memory.payload, noise)
        self._sync_stream()
        actions = self.normalizer.denormalize(output["actions"].float()).squeeze(0).cpu().numpy().astype(np.float64)
        if actions.shape != (16, 14) or not np.isfinite(actions).all():
            raise RuntimeError("malformed native causal-spatial plan")
        actions[:2] = prefix
        ready = time.perf_counter()
        launched = self._submit_memory(packet, observation, origin) if origin-self._source_origin >= 16 else False
        self.last_policy_timing = dict(protocol=NATIVE_PROTOCOL, policy="native_causal_spatial", H=2, R=2, K=10,
            chunk_size=16, prefix_length=2, action_origin=origin, cuda_graph=self._graph is not None,
            graph_captures=0 if self._graph is None else self._graph.captures,
            graph_replays=0 if self._graph is None else self._graph.replays,
            capture_physics_s=float(packet["sim_time"]),
            model_target_offsets_s=offsets.tolist(), future_target_offsets_s=offsets[[7,15]].tolist(),
            model_time_semantics="nominal_requested_offsets_not_native_topp_execution_endpoints",
            previous_plan_alignment="native_action_index_shift_not_measured_physics_endpoint_alignment",
            observation_clock="actual_native_scene_clock", continuous_world=False,
            source_event_id=memory.source.event_id, source_sim_time=memory.source.observation_physics_s,
            source_publication=memory.publication, source_completed_wall_s=memory.completed_wall_s,
            source_consumed_wall_s=fast_started,
            memory_age_seconds=float(packet["sim_time"])-memory.source.observation_physics_s,
            memory_waits=memory_waits, slow_launched=launched, slow_in_flight=self._future is not None,
            slow_dispatch_order="after_fast_actions_cpu_ready", last_slow=self._slow_status(),
            spatial_window_times=observation.spatial_times.tolist(), soft_plan_valid_rows=int(valid.sum()),
            fast_model_ms=1000*(ready-fast_started), complete_request_ms=1000*(time.perf_counter()-started))
        return actions

    def reset_native_episode(self):
        receipt = self.drain_episode()
        self._new_episode()
        return receipt
