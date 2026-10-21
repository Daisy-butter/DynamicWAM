"""CPU ownership and timing for the causal spatial policy.

This module is a usable request/installation adapter, not a DOMINO episode
driver. Camera exposure, continuous stepping, model invocation, normalization,
and CUDA completion events remain the caller's responsibility. All queue values
are raw absolute 14D drive targets; model-space values must be denormalized
before installation. No source-plan or previous-plan fallback exists here.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import threading
from typing import Any

import numpy as np

from .domino_continuous.continuous_rtc import CommittedActionQueue, PrefixReservation
from .domino_continuous.wall_sync import RealtimeViolation


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _array(value, *, shape=None, dtype=np.float32):
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite observation/action")
    # Bytes ownership prevents a caller from simply re-enabling writes.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class CausalSpatialContract:
    version: str = "causal_spatial_v4_h2r2_v1"
    action_interval_s: float = .16
    committed_actions: int = 2
    replan_actions: int = 2
    action_horizon: int = 16
    denoising_steps: int = 10
    window_size: int = 4
    window_stride_s: float = .32
    overlap_actions: int = 2
    max_memory_age_s: float = 4.96
    physics_dt_s: float = .004
    simulation_rate: float = 1.

    def __post_init__(self):
        for name in ("committed_actions", "replan_actions", "action_horizon",
                     "denoising_steps", "window_size", "overlap_actions"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.replan_actions <= self.committed_actions <= 4:
            raise ValueError("require 1 <= R <= H <= native queue capacity 4")
        if self.action_horizon != 16 or self.denoising_steps != 10:
            raise ValueError("native Action contract requires chunk16 and K10")
        if self.window_size < 1 or not 0 <= self.overlap_actions <= 16-self.committed_actions:
            raise ValueError("invalid finite window/overlap")
        for name in ("action_interval_s", "max_memory_age_s", "physics_dt_s", "window_stride_s"):
            if _finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.simulation_rate != 1.:
            raise ValueError("causal spatial primary protocol requires fixed 1x")
        ticks = self.action_interval_s / self.physics_dt_s
        if ticks < 1 or not math.isclose(ticks, round(ticks), abs_tol=1e-8):
            raise ValueError("action interval must use an integral physical tick count")

    @property
    def commitment_s(self):
        return self.committed_actions * self.action_interval_s

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutedAction:
    episode: str
    index: int
    target_physics_s: float
    completed_physics_s: float
    action: np.ndarray
    measured_qpos: np.ndarray | None


@dataclass(frozen=True)
class WindowSnapshot:
    episode: str
    event_id: int
    revision: int
    observation_physics_s: float
    available_wall_s: float
    spatial_rgb: np.ndarray
    flow: np.ndarray
    valid: np.ndarray
    flow_valid: np.ndarray
    dt: np.ndarray
    recorded_capture_commands: np.ndarray
    command_valid: np.ndarray
    observation_valid: np.ndarray
    spatial_times: np.ndarray
    spatial_event_ids: np.ndarray
    contract_version: str

    @property
    def executed_actions(self):
        """Compatibility name only: capture commands, NOT dense executed history."""
        return self.recorded_capture_commands

    @property
    def executed_valid(self):
        return self.command_valid

    def model_inputs(self):
        """Unbatched, raw CPU inputs; explicit caller adds batch/normalization."""
        return dict(spatial_rgb=self.spatial_rgb, spatial_flow=self.flow,
            spatial_valid=self.valid, spatial_flow_valid=self.flow_valid, spatial_dt=self.dt,
            recorded_capture_commands=self.recorded_capture_commands, command_valid=self.command_valid)


class FiniteObservationWindow:
    """Same physical lookback/flow as offline data, without hidden b state.

    The owner records actual native drive targets at exposure. Interval row i
    contains the target at the PREVIOUS selected capture, aligned with dt[i].
    These sparse capture commands are not dense controls or measured qpos.
    Execution acknowledgements remain separate diagnostics; they never silently
    replace missing capture targets or observations.
    """
    def __init__(self, episode: str, contract=CausalSpatialContract(), *, flow_backend=None, flow_workers=3):
        if type(flow_workers) is not int or not 1 <= flow_workers <= 3:
            raise ValueError("flow_workers must be an integer within 1..3")
        self.episode, self.contract = str(episode), contract
        self._frames = deque()
        self._flows = OrderedDict()
        self._flow_backend = flow_backend
        self._flow_workers = flow_workers
        self._last_flow_workers = 0
        self._revision = 0

    def append(self, *, event_id, observation_physics_s, available_wall_s,
               spatial_rgb, raw_rgb, valid, capture_drive_targets,
               executed: ExecutedAction | None = None):
        t = _finite(observation_physics_s, "observation time")
        w = _finite(available_wall_s, "observation availability")
        if type(event_id) is not int or event_id < 0:
            raise ValueError("nonnegative integer camera event id required")
        previous = self._frames[-1] if self._frames else None
        if previous and (event_id <= previous[0] or t <= previous[1] or w < previous[2]):
            raise ValueError("camera events must advance once in time and sequence")
        rgb = _array(spatial_rgb)
        if rgb.ndim != 4 or rgb.shape[:2] != (3, 3) or min(rgb.shape[2:]) < 1:
            raise ValueError("spatial_rgb must be [3 views,3 channels,H,W]")
        shape = rgb.shape[-2:]
        if previous and previous[3].shape != rgb.shape:
            raise ValueError("camera shape changed within episode")
        mask = _array(valid, shape=(3, 1, *shape))
        if ((mask < 0) | (mask > 1)).any():
            raise ValueError("image validity must be in [0,1]")
        if np.asarray(raw_rgb).dtype != np.uint8:
            raise ValueError("raw RGB for the shared flow helper must be uint8")
        raw = _array(raw_rgb, shape=(3,*shape,3), dtype=np.uint8)
        action = _array(capture_drive_targets, shape=(14,))
        if executed is not None:
            if executed.episode != self.episode or executed.completed_physics_s > t + 1e-8:
                raise ValueError("execution receipt belongs to future/another episode")
            if not math.isfinite(executed.completed_physics_s):
                raise ValueError("invalid execution receipt time")
        self._frames.append((event_id, t, w, rgb, raw, mask, action, executed))
        # Retain the last capture at/before the earliest requested time, plus
        # every capture after it. This preserves last-available selection.
        cutoff = t-(self.contract.window_size-1)*self.contract.window_stride_s
        while len(self._frames) > 1 and self._frames[1][1] <= cutoff+1e-10:
            self._frames.popleft()
        # Immutable exposure ids also own image geometry and padding masks.
        # A pair cannot be selected again after either endpoint is evicted.
        live_ids = {frame[0] for frame in self._frames}
        for key in tuple(self._flows):
            if key[0] not in live_ids or key[1] not in live_ids:
                del self._flows[key]
        self._revision += 1
        return self.snapshot()

    def append_composite(self, frame, **metadata):
        from dynamicwam.observations.causal_spatial_window import composite_observation
        rgb,raw,valid = composite_observation(frame)
        return self.append(spatial_rgb=rgb,raw_rgb=raw,valid=valid[:,None],**metadata)

    def snapshot(self):
        if not self._frames:
            raise RuntimeError("a real observation is required before a belief snapshot")
        from dynamicwam.observations.causal_spatial_window import (
            causal_window_indices, farneback_pair, flow_pair_validity)
        frames = list(self._frames); last = frames[-1]
        indices,times,dt,transitions = causal_window_indices([f[1] for f in frames], len(frames)-1,
            window_size=self.contract.window_size,stride_seconds=self.contract.window_stride_s)
        selected = [frames[i] for i in indices]
        rgb = _array(np.stack([f[3] for f in selected]))
        valid = _array(np.stack([f[5] for f in selected]),dtype=np.bool_)
        steps,views,_,height,width = rgb.shape
        flow = np.zeros((steps,views,2,height,width),dtype=np.float32)
        flow_valid = np.zeros((steps,views,1,height,width),dtype=bool)
        backend = self._flow_backend or farneback_pair
        missing = {}
        for i in range(1,steps):
            if not transitions[i]:
                continue
            for view in range(views):
                key = (selected[i-1][0],selected[i][0],view,height,width)
                if key not in self._flows:
                    missing[key] = (selected[i-1][4][view],selected[i][4][view],
                                    valid[i-1,view,0],valid[i,view,0])

        def compute_pair(inputs):
            # Read only immutable CPU images/masks. Workers never access the
            # mutable cache or any model, Torch/CUDA state, or shared output.
            previous,current,previous_valid,current_valid = inputs
            delta,mask = backend(previous,current)
            delta = _array(delta,shape=(height,width,2))
            mask = _array(mask,shape=(height,width),dtype=np.bool_)
            mask = flow_pair_validity(delta,mask,previous_valid,current_valid)
            return (_array(np.where(mask[...,None],delta,0).transpose(2,0,1)),
                    _array(mask,dtype=np.bool_))

        self._last_flow_workers = 0
        if missing:
            workers = min(self._flow_workers,len(missing))
            if workers > 1:
                if backend is farneback_pair:
                    import cv2
                    if cv2.getNumThreads() != 1:
                        # Direct CPU callers may not run production's thread
                        # setup. Avoid nested OpenCV pools without changing
                        # process-global configuration in this helper.
                        workers = 1
            self._last_flow_workers = workers
            if workers > 1:
                # Per-snapshot lifetime: shutdown joins all workers even when a
                # pair raises. Ordered map plus main-thread writeback makes
                # completion order irrelevant and prevents partial publication.
                with ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="causal-spatial-flow") as executor:
                    results = list(executor.map(compute_pair,missing.values()))
            else:
                results = [compute_pair(inputs) for inputs in missing.values()]
            for key,result in zip(missing,results):
                self._flows[key] = result
        for i in range(1,steps):
            if not transitions[i]:
                continue
            for view in range(views):
                key = (selected[i-1][0],selected[i][0],view,height,width)
                self._flows.move_to_end(key)
                pair_flow,mask = self._flows[key]
                flow[i,view] = pair_flow
                flow_valid[i,view,0] = mask
        while len(self._flows) > 64:
            self._flows.popitem(last=False)
        commands = np.stack([selected[max(0,i-1)][6] for i in range(steps)])
        commands[~transitions] = 0.
        return WindowSnapshot(self.episode, last[0], self._revision, last[1], last[2],
            rgb,_array(flow),valid,_array(flow_valid,dtype=np.bool_),_array(dt,dtype=np.float64),
            _array(commands),_array(transitions,dtype=np.bool_),
            _array(np.ones(steps,dtype=bool),dtype=np.bool_),_array(times,dtype=np.float64),
            _array([f[0] for f in selected],dtype=np.int64),self.contract.identity)


@dataclass(frozen=True)
class MemorySource:
    episode: str
    request_id: int
    event_id: int
    observation_physics_s: float
    available_wall_s: float
    contract_version: str


@dataclass(frozen=True)
class PublishedMemory:
    source: MemorySource
    publication: int
    completed_wall_s: float
    payload: Any


class LatestObservationMemory:
    """Publish complete observation-only memory, without touching the belief.

    Payload tensors are producer-owned immutable values. This CPU layer checks
    provenance/availability, not neural attention masks or CUDA events; callers
    must pass the actual completion event's ready result, never a prediction.
    """
    def __init__(self, episode: str, contract=CausalSpatialContract()):
        self.episode, self.contract = str(episode), contract
        self._pending = {}; self._request = 0; self._publication = 0
        self._lock = threading.Lock()
        self.current: PublishedMemory | None = None

    def begin(self, observation: WindowSnapshot):
        if observation.episode != self.episode:
            raise ValueError("memory source belongs to another episode")
        if observation.contract_version != self.contract.identity:
            raise ValueError("memory source window contract mismatch")
        with self._lock:
            token = MemorySource(self.episode, self._request, observation.event_id,
                observation.observation_physics_s, observation.available_wall_s, self.contract.identity)
            self._pending[self._request] = token; self._request += 1
            return token

    def publish(self, source, payload, *, completed_wall_s, completion_ready):
        # Complete-token validation and latest-source comparison are atomic even
        # if multiple producer callbacks finish simultaneously.
        with self._lock:
            return self._publish(source, payload, completed_wall_s, completion_ready)

    def _publish(self, source, payload, completed_wall_s, completion_ready):
        if source.episode != self.episode or self._pending.get(source.request_id) != source:
            raise ValueError("unknown/stale memory source token")
        if completion_ready is not True:
            raise RuntimeError("memory cannot publish before actual producer completion")
        w = _finite(completed_wall_s, "memory completion")
        if w < source.available_wall_s:
            raise ValueError("memory completes before source observation availability")
        del self._pending[source.request_id]
        if self.current and (source.event_id <= self.current.source.event_id or
                             source.observation_physics_s <= self.current.source.observation_physics_s):
            return False
        self._publication += 1
        self.current = PublishedMemory(source, self._publication, w, payload)
        return True

    def snapshot(self, *, observation_physics_s, request_wall_s):
        with self._lock:
            current = self.current
        if current is None:
            raise RuntimeError("no completed observation memory")
        t = _finite(observation_physics_s, "request observation")
        w = _finite(request_wall_s, "request wall")
        age = t-current.source.observation_physics_s
        if age < -1e-8 or age > self.contract.max_memory_age_s + 1e-8:
            raise ValueError("memory is from the future or exceeds configured age")
        if current.completed_wall_s > w:
            raise ValueError("memory was not available at this request snapshot")
        return current


@dataclass(frozen=True)
class SnapshotVersion:
    episode: str
    request_id: int
    event_id: int
    belief_revision: int
    memory_publication: int
    queue_generation: int
    action_origin: int
    contract_version: str


@dataclass(frozen=True)
class CausalRequest:
    version: SnapshotVersion
    observation: WindowSnapshot
    memory: PublishedMemory
    request_wall_s: float
    prefix_actions: np.ndarray
    overlap_actions: np.ndarray
    overlap_valid: np.ndarray
    action_target_times: np.ndarray
    action_target_offsets: np.ndarray
    future_target_offsets: np.ndarray
    reservation: PrefixReservation


class CausalPlanQueue:
    """Reuse the native queue with EXPLICIT H, without the H4 episode driver.

    Call request at each R boundary; execute/acknowledge rows continuously while
    inference runs; install a result before its immutable H deadline. Initial
    held targets are explicit bootstrap input, never a generated source plan.
    """
    def __init__(self, *, episode, initial_actions, grid_start_physics_s,
                 physics_epoch_s, wall_epoch_s, contract=CausalSpatialContract()):
        self.episode, self.contract = str(episode), contract
        self.grid_start = _finite(grid_start_physics_s, "grid start")
        self.physics_epoch = _finite(physics_epoch_s, "physics epoch")
        self.wall_epoch = _finite(wall_epoch_s, "wall epoch")
        if self.grid_start < self.physics_epoch:
            raise ValueError("action grid starts before episode physical epoch")
        # Legacy storage advertises capacity 4; only reserve's explicit length
        # commits rows. The old H4 episode loop is deliberately not invoked.
        self.native = CommittedActionQueue(interval_s=contract.action_interval_s, prefix_max=4)
        self.native.bootstrap(initial_actions)
        self.completed_actions = 0
        self._request_id = 0; self._last_origin = -1; self._installations = 0
        self._active: CausalRequest | None = None

    def request(self, observation, memory, *, request_wall_s):
        c = self.contract; origin = self.completed_actions
        if self._active is not None:
            raise RuntimeError("only one fast request may be in flight")
        if origin % c.replan_actions or self._last_origin == origin:
            raise ValueError("request must occur once at each R boundary")
        if observation.episode != self.episode or memory.source.episode != self.episode:
            raise ValueError("snapshot belongs to another episode")
        if memory.source.contract_version != c.identity or observation.contract_version != c.identity:
            raise ValueError("memory contract version differs from the active policy")
        boundary = self.grid_start + origin*c.action_interval_s
        if abs(observation.observation_physics_s-boundary) > c.physics_dt_s*.1:
            raise ValueError("mandatory observation moved off the absolute action grid")
        w = _finite(request_wall_s, "request wall")
        if w < max(observation.available_wall_s, memory.completed_wall_s, self.wall_epoch):
            raise ValueError("snapshot inputs were not available at request time")
        age = observation.observation_physics_s-memory.source.observation_physics_s
        if not -1e-8 <= age <= c.max_memory_age_s+1e-8:
            raise ValueError("memory from future or too old")
        reservation = self.native.reserve(origin, boundary, length=c.committed_actions)
        target_times = self.grid_start + (origin+np.arange(1,17))*c.action_interval_s
        previous, valid = self.native.proposal(origin)
        overlap_valid = np.zeros(16, dtype=bool)
        if self._installations:
            start = c.committed_actions; end = start+c.overlap_actions
            overlap_valid[start:end] = valid[start:end]
        previous[~overlap_valid] = 0.
        version = SnapshotVersion(self.episode, self._request_id, observation.event_id,
            observation.revision, memory.publication, self.native.generation, origin, c.identity)
        request = CausalRequest(version, observation, memory, w,
            _array(reservation.actions), _array(previous), _array(overlap_valid, dtype=np.bool_),
            _array(target_times, dtype=np.float64),
            _array(target_times-observation.observation_physics_s, dtype=np.float64),
            _array(target_times[[7,15]]-observation.observation_physics_s,dtype=np.float64),reservation)
        self._active = request; self._request_id += 1; self._last_origin = origin
        return request

    def install(self, *, version, actions, action_target_times, ready_physics_s, ready_wall_s):
        request = self._active
        if request is None or version != request.version:
            raise RuntimeError("result snapshot/version does not own active request")
        times = _array(action_target_times, shape=(16,), dtype=np.float64)
        if not np.allclose(times, request.action_target_times, rtol=0., atol=1e-9):
            raise ValueError("result action targets moved off the reserved grid")
        physical = _finite(ready_physics_s, "result physics")
        wall = _finite(ready_wall_s, "result wall")
        if wall < request.request_wall_s or physical < request.observation.observation_physics_s-1e-8:
            raise ValueError("result precedes its request")
        ready = max(physical, self.physics_epoch+wall-self.wall_epoch)
        if ready-request.memory.source.observation_physics_s > self.contract.max_memory_age_s+1e-8:
            raise RealtimeViolation("memory aged beyond trained bound before installation")
        self.native.install(actions, request.reservation, ready_physics_s=ready)
        self._active = None; self._installations += 1
        return {"request_id": version.request_id, "ready_clock_s": ready,
                "deadline_physics_s": request.reservation.deadline_physics_s,
                "suffix_first_target_s": request.action_target_times[self.contract.committed_actions],
                "replacement_semantics": "immutable_prefix_then_atomic_unexecuted_tail"}

    def next_action(self):
        origin = self.completed_actions
        if self._active and origin >= self._active.version.action_origin+self.contract.committed_actions:
            raise RealtimeViolation("no valid suffix at committed H boundary")
        if origin % self.contract.replan_actions == 0 and self._last_origin != origin:
            raise RuntimeError("a fresh observation/request is required at this R boundary")
        return _array(self.native.take(origin, 1)[0], dtype=np.float64)

    def next_block(self, count=None):
        """Owned R-row targets for the continuous trajectory planner."""
        count=self.contract.replan_actions if count is None else count
        if type(count) is not int or not 1 <= count <= self.contract.replan_actions:
            raise ValueError("planned block must stay within the next R actions")
        self.next_action()
        if self._active and self.completed_actions+count > self._active.version.action_origin+self.contract.committed_actions:
            raise RealtimeViolation("trajectory would cross the pending committed boundary")
        return _array(self.native.take(self.completed_actions,count),dtype=np.float64)

    def acknowledge_executed(self, *, action_index, completed_physics_s, measured_qpos=None):
        if action_index != self.completed_actions:
            raise ValueError("executed action acknowledgements must be contiguous and unique")
        action = self.next_action()
        target = self.grid_start+(action_index+1)*self.contract.action_interval_s
        t = _finite(completed_physics_s, "actual execution completion")
        if t < target-self.contract.physics_dt_s*.1:
            raise ValueError("cannot acknowledge an unexecuted future target")
        measured = None if measured_qpos is None else _array(measured_qpos, shape=(14,))
        self.completed_actions += 1
        return ExecutedAction(self.episode, action_index, target, t, action, measured)
