"""Continuous DOMINO driver: only fast plans reach the robot's execution queue."""
from __future__ import annotations
import math
import threading
import time
import numpy as np
import torch
from contextlib import nullcontext
from .timing import ExecutionQueue, TimedPlan, TimingError, DeadlineMiss, BufferUnderrun, prefix_digest
from .runtime import DualRateRuntime
from .online import ObservationHistory
from .actuators import ActuatorProjection
from .physical_clock import WallPhysicsBridge
from .contact_cache import StepContactCache
from .producer import NUMERIC_TIME_EPSILON

PROTOCOL='dualrate_v4_exposure_anchor_wall_release_fast_only'


class ContinuousWallPhysicsBridge(WallPhysicsBridge):
    """Fixed activation mapping; Full formal09's 5ms backlog tolerance only.

    This tolerance is not a query deadline extension. Neither wall origin nor
    elapsed wall time may be adjusted during an active episode.
    """
    def __init__(self, *args, backlog_scheduling_tolerance=.005, **kwargs):
        super().__init__(*args, **kwargs)
        value = float(backlog_scheduling_tolerance)
        if not math.isfinite(value) or not 0 <= value <= .005:
            raise ValueError('backlog scheduling tolerance must be within [0,.005]')
        self.backlog_scheduling_tolerance = value
        self._activation_origins = None

    def activate(self, physics_time, wall_time=None):
        super().activate(physics_time, wall_time)
        self._activation_origins = (self.physics_origin, self.wall_origin)

    def assert_fixed_origins(self):
        if self.active and self._activation_origins != (self.physics_origin, self.wall_origin):
            raise TimingError('active wall/physics origin changed')

    def expected_physics(self, wall_time=None):
        self.assert_fixed_origins()
        return super().expected_physics(wall_time)

    def check_backlog(self, physical_now, max_backlog, *, wall_time=None):
        # Delegate all signed-lag and finite-value checks to the original clock.
        if not math.isfinite(max_backlog) or max_backlog <= 0:
            raise ValueError('positive finite backlog bound required')
        return super().check_backlog(physical_now,
            max_backlog + self.backlog_scheduling_tolerance, wall_time=wall_time)


def continuous_clock_evidence(bridge, *, activation_wall, activation_physics,
                              end_wall, end_physics, physics_steps, max_lag,
                              min_lag, max_backlog):
    """Validate raw, non-rebased clocks before declaring continuous real time."""
    bridge.assert_fixed_origins()
    wall = float(end_wall) - float(activation_wall)
    physics = float(end_physics) - float(activation_physics)
    dt = bridge.physics_dt
    raw_lag = wall - physics
    limit = float(max_backlog) + bridge.backlog_scheduling_tolerance
    checks = dict(
        immutable_activation=(bridge.wall_origin == activation_wall and
                              bridge.physics_origin == activation_physics),
        finite_clocks=all(math.isfinite(x) for x in (wall, physics, max_lag, min_lag)),
        physics_steps_match=abs(physics-physics_steps*dt) <= 1e-7,
        raw_wall_not_behind=raw_lag >= -dt-bridge.numeric_tolerance,
        raw_backlog_in_bound=raw_lag <= limit,
        observed_backlog_in_bound=max_lag <= limit,
        observed_no_future_steps=min_lag >= -dt-bridge.numeric_tolerance,
        positive_active_duration=wall >= 0 and physics >= 0)
    if not all(checks.values()):
        raise TimingError('continuous raw-clock acceptance failed: '+str(checks))
    return dict(passed=all(checks.values()),checks=checks,
        activation_wall_time=float(activation_wall),activation_physics_time=float(activation_physics),
        final_wall_time=float(end_wall),final_physics_time=float(end_physics),
        active_wall_seconds=wall,physical_duration=physics,raw_end_backlog_s=raw_lag,
        observed_min_backlog_s=min_lag,observed_max_backlog_s=max_lag,
        configured_max_backlog_s=float(max_backlog),
        backlog_scheduling_tolerance_s=bridge.backlog_scheduling_tolerance,
        measured_physics_per_wall_second=physics/wall if wall>0 else None)


def exact_grid_index(when, origin, interval, *, tolerance=1e-6):
    """Identify an existing grid instant, never relabel an off-grid exposure."""
    if not all(math.isfinite(x) for x in (when,origin,interval)) or interval<=0:
        raise TimingError('invalid physical grid')
    index=round((when-origin)/interval)
    if abs(when-(origin+index*interval))>tolerance:
        raise TimingError('actual timestamp is not on the declared physical grid')
    return index


def next_grid_capture_tick(tick, action_ticks, physics_dt, render_wall, *, recovery_seconds=.06):
    """Skip whole capture slots if rendering needs more than one action interval."""
    if action_ticks<1 or physics_dt<=0 or render_wall<0 or recovery_seconds<0:
        raise ValueError('invalid capture tick schedule')
    intervals=max(1,math.ceil((render_wall+recovery_seconds)/(action_ticks*physics_dt)-1e-9))
    return int(tick)+intervals*int(action_ticks)


def action_response_statistics(events):
    """First executed mutable row per query; exclude superseded/terminal plans.

    Queue replacement preserves rows before protected_until, so the newest
    committed plan whose mutable frontier reaches an executed row owns that row.
    This measures actual response, not merely GPU sampling time.
    """
    activation=next((e['wall_time'] for e in events if e['event']=='activation'),math.inf)
    commits=[];seen=set();responses=[]
    for event in events:
        if event.get('post_terminal'):continue
        if event['event']=='fast_committed':
            commits.append(event)
        elif event['event']=='execute':
            owner=next((e for e in reversed(commits)
                if e['protected_until']<=event['start_time']+1e-9),None)
            if owner is None or owner['query_id'] in seen:continue
            seen.add(owner['query_id'])
            query_wall=owner.get('query_request_wall_time')
            exposure_wall=owner.get('observation_exposure_wall_time')
            if query_wall is None or query_wall<activation:continue
            executed_wall=event['execution_wall_time']
            if executed_wall<query_wall:raise TimingError('action executed before its query')
            responses.append(dict(query_id=owner['query_id'],reference_version=owner['reference_version'],
                observation_time=owner['observation_time'],query_time=owner['query_time'],
                first_mutable_action_time=owner['protected_until'],actual_execution_time=event['actual_start'],
                execution_wall_time=executed_wall,query_request_wall_time=query_wall,
                observation_exposure_wall_time=exposure_wall,
                query_to_execution_wall_seconds=executed_wall-query_wall,
                observation_to_execution_wall_seconds=None if exposure_wall is None else executed_wall-exposure_wall))
    return dict(action_response_events=responses,action_response_measurement='first_executed_mutable_row_by_queue_provenance')


def model_overlap_statistics(events):
    """Overlap of measured model-call wall intervals, not a CUDA kernel profile."""
    activation=next((e['wall_time'] for e in events if e['event']=='activation'),math.inf)
    def interval(event):
        start=event.get('model_started_wall_time');end=event.get('gpu_done_wall_time')
        if start is None or end is None:return None
        if not math.isfinite(start) or not math.isfinite(end) or end<start:
            raise TimingError('invalid measured model interval')
        return max(start,activation),end
    slow=[(e,interval(e)) for e in events if e['event']=='slow_completed']
    fast=[(e,interval(e)) for e in events if e['event']=='fast_committed']
    overlaps=[]
    for f,fi in fast:
        if fi is None:continue
        for s,si in slow:
            if si is None:continue
            overlap=min(fi[1],si[1])-max(fi[0],si[0])
            if overlap>0:
                overlaps.append(dict(query_id=f['query_id'],slow_version=s['version'],
                    overlap_wall_seconds=overlap,fast_model_wall_interval=list(fi),slow_model_wall_interval=list(si)))
    return dict(model_overlap_fast_queries=len({e['query_id'] for e in overlaps}),
        model_overlap_wall_seconds=sum(e['overlap_wall_seconds'] for e in overlaps),
        model_overlap_events=overlaps,model_overlap_measurement='wall_intervals_ending_after_cuda_stream_sync')


def bind_gil_releasing_physx_step(env, step_physx) -> None:
    """Release the GIL under the installed scene-step clock, not around it.

    SceneStepClock counts time only after its saved ``scene.step`` returns.
    A direct PhysX call leaves that counter at zero, so wall time runs ahead
    and the backlog gate fires before the first capture.
    """
    clock = getattr(env, "_scene_step_clock", None)
    if clock is None or getattr(clock, "_original_step", None) is None:
        raise RuntimeError("SceneStepClock is not installed")
    physx = env.scene.physx_system

    def releasing_step():
        step_physx(physx)

    clock._original_step = releasing_step


def next_capture_deadline(initial, now, frame_period, render_wall) -> float:
    """Legacy synchronous-camera cadence helper, unused by the async executor.

    The active executor uses actual background render duration and integer
    action-grid slots through ``next_grid_capture_tick``.
    """
    if frame_period <= 0 or render_wall < 0 or not math.isfinite(render_wall):
        raise ValueError("capture schedule requires a positive frame and a finite render wait")
    # One extra frame covers owner stepping and readback. Spacing by the render
    # wait alone still left about 0.01 s unrecovered per capture, which crossed
    # the 0.25 s backlog during an episode. The image-age gate stays 0.24 s.
    earliest = float(now) + max(float(frame_period), float(render_wall)) + float(frame_period)
    steps = math.ceil((earliest - float(initial)) / float(frame_period) - 1e-9)
    return float(initial) + steps * float(frame_period)


def capture_timing(captured, *, owner_processed_wall):
    """Partition real wall latency; do not relabel exposure or availability."""
    points=[float(captured.capture_wall_perf_s),float(captured.render_submitted_wall_perf_s),
        float(captured.readback_started_wall_perf_s),float(captured.readback_finished_wall_perf_s),
        float(captured.finished_wall_perf_s),float(owner_processed_wall)]
    if any(not math.isfinite(x) for x in points) or any(b<a for a,b in zip(points,points[1:])):
        raise TimingError('capture wall timestamps are invalid')
    names=('render_submit','metadata_and_reader_dispatch','readback_and_conversion',
           'readback_complete_to_owner_finish','owner_history_and_video')
    parts={name+'_wall_seconds':b-a for name,a,b in zip(names,points,points[1:])}
    return dict(parts,total_wall_seconds=points[-1]-points[0],
        camera_readback_seconds=dict(captured.camera_readback_seconds),
        wall_timestamps=points,clock='process_monotonic_perf_counter')


def warm_camera(camera, session, *, count, emit):
    """Fixed pre-activation warmup, with no duplicate frames in model history."""
    if isinstance(count,bool) or int(count)!=count or count<1:
        raise ValueError('camera warmup count must be a positive integer')
    source=float(session.physics_time());captured=None
    for index in range(int(count)):
        camera.begin(session.env,source)
        captured=camera.finish()
        if float(session.physics_time())!=source or captured.physics_time_s!=source:
            raise TimingError('camera warmup advanced the inactive episode clock')
        emit(dict(event='camera_warmup',index=index,count=count,exposure=source,
                  timing=capture_timing(captured,owner_processed_wall=time.perf_counter())))
    return captured


def reference_trace_since(reference, initial, query_time):
    """Trim only history that cannot match any current soft-action reference.

    The store snapshot is CPU-owned; never introduce a device synchronization
    on the physics owner. Unknown/legacy layouts retain the original full trace.
    The matching epsilon is the one used by DualRateSystem._action_deviation.
    This changes no executed command, timestamp, prefix, or model condition.
    """
    if reference is None:
        return initial
    payload = getattr(reference, 'payload', None)
    if not hasattr(payload, 'get'):
        return initial
    intervals = payload.get('reference_action_intervals')
    if (not isinstance(intervals, torch.Tensor) or intervals.device.type != 'cpu'
            or intervals.dtype != torch.float64 or intervals.ndim != 3
            or intervals.shape[0] != 1 or intervals.shape[2] != 2
            or intervals.shape[1] == 0 or not torch.isfinite(intervals).all()
            or (intervals[..., 1] <= intervals[..., 0]).any()):
        return initial
    earliest = float(intervals[..., 0].min()) - 1e-7
    return max(initial, min(query_time, earliest))


def dispatch_due_queries(runtime,builder,history,queue,*,now,initial,vae,emit,deferred,seen_references,
                         exposure_walls=None,wall_clock=time.perf_counter):
    """Wait for genuinely fresh available pixels while physics/queue continue.

    Deferral never advances the route's last-query tick. A later lease uses its
    actual physical time and the original floor-grid/protect_steps deadline.

    Dispatch can occur during catch-up. Real wall availability, observation age,
    and the trained bounded 2/3-row lease still apply; physics never waits here.
    """
    fast_due,slow_due=runtime.due(now)
    if not (fast_due or slow_due):return
    snapshot=history.snapshot(now)
    age=now-snapshot[-1].exposure
    wall_age = None if exposure_walls is None else float(wall_clock())-exposure_walls[snapshot[-1].exposure]
    fresh=(0<=age<=builder.max_observation_age+NUMERIC_TIME_EPSILON and
           (wall_age is None or 0<=wall_age<=builder.max_observation_age+NUMERIC_TIME_EPSILON))
    for route,due in (('fast',fast_due),('slow',slow_due)):
        if not due:continue
        if not fresh:
            if route not in deferred:
                deferred[route]=now
                emit(dict(event='query_deferred',route=route,physical_time=now,
                    observation_time=snapshot[-1].exposure,observation_available=snapshot[-1].available,
                    observation_age=age,observation_wall_age=wall_age,max_observation_age=builder.max_observation_age,
                    reason='no_fresh_completed_capture'))
            continue
        began=None
        if route in deferred:
            began=deferred.pop(route)
            emit(dict(event='query_resumed',route=route,physical_time=now,
                deferred_seconds=now-began,observation_time=snapshot[-1].exposure,
                observation_age=age))
        if route=='fast':
            ref=runtime.store.snapshot(now)
            # Reference publication and physical advancement are owner-only, so
            # start_fast below sees this same reference at this same query time.
            # Old actions before its support cannot affect action-deviation M.
            trace=queue.completed_execution_trace(reference_trace_since(ref,initial,now),now)
            try:
                runtime.start_fast(lambda lease,reference,snapshot=snapshot,trace=trace:
                    builder.current(snapshot,query_time=lease.query_time,action_start=lease.prefix.start_time,
                                    executed=trace),observation_time=snapshot[-1].exposure)
            except DeadlineMiss as exc:
                # Raised before begin_query when ceil((phase+lag+budget+margin)/dt)
                # exceeds max_protect_steps. Re-defer; catch-up shrinks lag.
                if 'bounded protected prefix' not in str(exc):
                    raise
                deferred[route]=began if began is not None else now
                emit(dict(event='query_deferred',route=route,physical_time=now,
                    observation_time=snapshot[-1].exposure,observation_available=snapshot[-1].available,
                    observation_age=age,observation_wall_age=wall_age,max_observation_age=builder.max_observation_age,
                    reason='insufficient_prefix_budget'))
                continue
            if ref:seen_references.add(ref.version)
        else:
            if getattr(builder,'hardtime',False):
                exposure=snapshot[-1].exposure
                exact_grid_index(exposure,queue.origin,queue.dt)
                if exposure<=getattr(runtime,'last_source_exposure',-math.inf)+1e-9:
                    continue
                prefix=queue.source_prefix_from_exposure(exposure,now,horizon=16)
                runtime.start_slow(lambda snapshot=snapshot,now=now,prefix=prefix,exposure=exposure:
                    builder.source(snapshot,source_time=exposure,request_time=now,
                                   action_start=prefix.start_time,vae=vae),
                    prefix,source_time=exposure,request_time=now,source_available_time=snapshot[-1].available)
                runtime.last_source_exposure=exposure
            else:
                prefix=queue.snapshot_committed(now)
                runtime.start_slow(lambda snapshot=snapshot,now=now,prefix=prefix:
                    builder.source(snapshot,source_time=now,action_start=prefix.start_time,vae=vae),
                    prefix,source_time=now)



def run_episode(session, system, builder, camera, runtime_config, *, action_limit,
                frame_period=None, max_backlog=.25, record=None, seed=0, video=None,
                camera_warmup_captures=3, on_ready=None, on_armed=None, on_failure=None,
                wall_scheduling_tolerance=0., physical_scheduling_tolerance=0.,
                backlog_scheduling_tolerance=.005, control_scope=None):
    """One owner thread steps 250 Hz physics; two workers preprocess and infer.

    Initial reference and fast plan are established before activation, explicitly
    recorded as bootstrap latency. Once activated every operation is wall-clock
    charged. Background rendering overlaps physics; catch-up is bounded and never skips
    scene steps or makes a result available before its wall-mapped release tick.
    Target rows are endpoints over [.18i,.18(i+1)); arms interpolate linearly from
    the preceding committed endpoint, grippers keep native command semantics.
    """
    if not getattr(builder,'hardtime',False):
        raise ValueError('deployment requires exposure_anchor_wall_release_v4 data contract')
    frame_period=runtime_config.action_dt if frame_period is None else float(frame_period)
    if action_limit<1 or frame_period<=0 or max_backlog<=0:
        raise ValueError('invalid bounded deployment limits')
    events=[];bridge=None;owner_thread=threading.get_ident();post_terminal=False
    contact_cache=None;main_error=None;exposure_walls={};query_requests={}
    def emit(event):
        if event.get('event')=='fast_started':
            exposure=event['observation_time']
            event=dict(event,observation_exposure_wall_time=exposure_walls.get(exposure))
            query_requests[event['query_id']]=event
        elif event.get('event')=='fast_committed':
            request=query_requests.get(event['query_id'],{})
            event=dict(event,query_request_wall_time=request.get('request_wall_time'),
                observation_exposure_wall_time=request.get('observation_exposure_wall_time'),
                first_mutable_action_time=event['protected_until'])
        if post_terminal:
            event=dict(event,post_terminal=True)
        elif bridge is not None and bridge.active:
            if threading.get_ident()==owner_thread:
                event=dict(event,clock_pair=bridge.snapshot(clock()))
            else:
                # Worker diagnostics cannot sample the advancing scene clock.
                # Only owner transitions define physical publication/commit time.
                event=dict(event,worker_wall_time=bridge.wall_now())
        events.append(event)
        if record:record(event)
    history=ObservationHistory();runtime=None;pending=None;completed_capture=None
    initial=float(session.physics_time());dt=float(session.timestep())
    if abs(dt-.004)>1e-8:
        raise TimingError('unverified DOMINO physics timestep')
    action_ticks=round(runtime_config.action_dt/dt)
    if action_ticks<1 or abs(action_ticks*dt-runtime_config.action_dt)>1e-10:
        raise TimingError('action grid must lie on integer physics ticks')
    if abs(builder.action_dt-runtime_config.action_dt)>1e-10:
        raise TimingError('data and controller action grids differ')
    capture_rows=round(frame_period/runtime_config.action_dt)
    if capture_rows<1 or abs(capture_rows*runtime_config.action_dt-frame_period)>1e-10:
        raise TimingError('source exposure schedule must use whole action intervals')
    capture_ticks=capture_rows*action_ticks
    clock=lambda:float(session.physics_time())
    if float(wall_scheduling_tolerance) != 0 or float(physical_scheduling_tolerance) != 0:
        raise ValueError('continuous executor uses the trained strict query deadlines')
    bridge=ContinuousWallPhysicsBridge(dt,wall_clock=time.perf_counter,
        backlog_scheduling_tolerance=backlog_scheduling_tolerance)
    def checked_now():
        now=clock()
        return now,bridge.check_backlog(now,max_backlog)
    bootstrap_started=time.perf_counter()
    captured=warm_camera(camera,session,count=camera_warmup_captures,emit=emit)
    history.append(captured.observation,exposure=captured.physics_time_s,available=initial)
    exposure_walls[captured.physics_time_s]=captured.capture_wall_perf_s
    if video:video.append(captured.observation['observation']['head_camera']['rgb'],
                         physics_time_s=initial,capture_wall_perf_s=captured.capture_wall_perf_s)
    held=tuple(float(x) for x in captured.observation['joint_action']['vector'])
    queue=ExecutionQueue(TimedPlan(initial,runtime_config.action_dt,(held,)*16,0,initial),
        episode_id=f'domino-{seed}',clock_id='sapien-scene-step-clock')
    prefix=queue.commit_prefix(query_time=initial,until=initial+runtime_config.protect_steps*queue.dt)
    runtime=DualRateRuntime(system,queue,clock,runtime_config,action_mean=builder.mean,
        action_scale=builder.scale,seed=seed,record=emit,clock_bridge=bridge,
        action_projection=ActuatorProjection.from_robot(session.env.robot))
    emit(dict(event='actuator_contract',bounds=runtime.action_projection.serialized_bounds,
              provenance=runtime.action_projection.provenance,projection='uncommitted_fast_suffix_only'))
    vae=system.slow.native.compact_wan.video_model.vae
    frames=history.snapshot(initial)
    try:
        runtime.start_slow(lambda:builder.source(frames,source_time=initial,request_time=initial,
                           action_start=prefix.start_time,vae=vae),
                           prefix,source_time=initial,request_time=initial,source_available_time=initial)
        runtime.last_source_exposure=initial
        while runtime.slow_job is not None:
            runtime.poll();time.sleep(.001)
        if runtime.store.snapshot(initial) is None:
            raise TimingError('initial slow reference was not published')
        runtime.start_fast(lambda lease,ref:builder.current(frames,query_time=lease.query_time,
            action_start=lease.prefix.start_time),observation_time=frames[-1].exposure)
        while runtime.fast_job is not None:
            runtime.poll();time.sleep(.001)
        graph_acceptance=runtime.fast_graph_receipt()
        if runtime_config.fast_cuda_graph and not (graph_acceptance.get('passed') and graph_acceptance.get('sealed')):
            raise TimingError('fast CUDA Graph must pass and seal before activation')
        # Shared executor optimization: unchanged checks on every real physics
        # step, one native contact snapshot reused by read-only consumers.
        contact_cache=StepContactCache(session.env.scene,clock).install()
        if on_ready is not None:
            on_ready()
            # The peer/model ready wait can leave the renderer idle. Rewarm the
            # still frozen scene without replacing the bootstrap observation/plan.
            warm_camera(camera,session,count=camera_warmup_captures,
                emit=lambda event:emit(dict(event,phase='post_ready_rewarm')))
        if on_armed is not None:on_armed()
        bootstrap_seconds=time.perf_counter()-bootstrap_started
        wall0=time.perf_counter();bridge.activate(initial,wall0)
        emit(dict(event='activation',physics_time=initial,wall_time=wall0,bootstrap_wall_seconds=bootstrap_seconds,
                  bootstrap_policy='initial_reference_then_fast_plan_before_activation',
                  held_prefix_seconds=runtime_config.protect_steps*queue.dt,
                  physics_step_seconds=dt,action_ticks=action_ticks,
                  wall_scheduling_tolerance=0.,physical_scheduling_tolerance=0.,
                  backlog_scheduling_tolerance=float(bridge.backlog_scheduling_tolerance),
                  immutable_wall_origin=True,inflight_catchup_hold=False,soft_horizon_reserve_hold=False))
        max_lag=0.;min_lag=0.;last_row=-1;completed_rows=0;metrics_row=-1
        previous=np.asarray(held,np.float64);target=previous.copy();next_capture_tick=capture_ticks
        from dynamicwam.runtime.domino_rt import _sapien_color_reader
        bind_gil_releasing_physx_step(session.env, _sapien_color_reader.step_cpu_physx)
        accepted_captures=1;skip_captures=0;seen_references=set();terminal=None;deferred={}
        exposure_walls={captured.physics_time_s:captured.capture_wall_perf_s}
        physics_steps=0;fast_inflight_steps=0;camera_inflight_steps=0;max_step_wall_gap=0.;last_step_wall=time.perf_counter()
        while terminal is None:
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
            if tick>=action_limit*action_ticks:
                terminal='action_budget_exhausted';break
            runtime.poll()
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            if pending is not None and camera.ready(pending):
                completed_capture=camera.finish(pending);pending=None
                render_wall=completed_capture.render_submitted_wall_perf_s-completed_capture.capture_wall_perf_s
                source_tick=exact_grid_index(completed_capture.physics_time_s,initial,dt,tolerance=1e-8)
                next_capture_tick=max(next_capture_tick,next_grid_capture_tick(source_tick,capture_ticks,dt,render_wall))
                now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            if completed_capture is not None and bridge.ready(completed_capture.finished_wall_perf_s,now):
                c=completed_capture;completed_capture=None
                exact_grid_index(c.physics_time_s,initial,queue.dt)
                history.append(c.observation,exposure=c.physics_time_s,available=now)
                exposure_walls[c.physics_time_s]=c.capture_wall_perf_s
                accepted_captures+=1
                if video:video.append(c.observation['observation']['head_camera']['rgb'],
                    physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)
                emit(dict(event='capture',exposure=c.physics_time_s,available=now,
                    readback_ready_wall=c.readback_finished_wall_perf_s,owner_ready_wall=c.finished_wall_perf_s,
                    availability_lower_bound=bridge.availability_time(c.finished_wall_perf_s),
                    timing=capture_timing(c,owner_processed_wall=time.perf_counter())))
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            # Neither route is gated on the other or on caught-up-only dispatch.
            # Freshness and the bounded lease decide whether a due query can start.
            dispatch_due_queries(runtime,builder,history,queue,now=now,initial=initial,vae=vae,
                emit=emit,deferred=deferred,seen_references=seen_references,
                exposure_walls=exposure_walls,wall_clock=time.perf_counter)
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
            if tick>=next_capture_tick and tick%capture_ticks==0:
                skip_captures+=max(0,(tick-next_capture_tick)//capture_ticks)
                if pending is None and completed_capture is None:
                    pending=camera.begin(session.env,now)
                    owner_submit_wall=time.perf_counter()-pending.capture_wall_perf_s
                    after,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
                    if after!=now or pending.physics_time_s!=now:
                        raise TimingError('capture advanced scene time or relabelled exposure')
                    emit(dict(event='capture_enqueued',capture_id=pending.capture_id,exposure=now,
                        exposure_tick=tick,owner_submit_wall_seconds=owner_submit_wall))
                    # The real render duration is available only when the background
                    # capture finishes; adjust cadence there, never from enqueue time.
                    next_capture_tick=tick+capture_ticks
                else:
                    skip_captures+=1;next_capture_tick=tick+capture_ticks
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            # Only advance when the corresponding real-time tick is due.
            wait=(now-initial)+dt-(time.perf_counter()-wall0)
            if wait>0:
                time.sleep(min(wait,.001));continue
            # Poll once more before this actual tick consumes committed coverage.
            # A job becoming ready during owner work must meet the same deadline.
            runtime.poll()
            now,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            # While a query or camera runs, execute the immutable queue prefix and
            # keep stepping the scene. Exhausted coverage or missed deadlines fail
            # explicitly; they never pause/rebase the benchmark clock.
            tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
            row=tick//action_ticks
            if row>=action_limit:
                terminal='action_budget_exhausted';break
            action=np.asarray(queue.action_at(now),np.float64)
            if row!=last_row:
                previous=target.copy();target=action.copy();last_row=row
                session.env.take_action_cnt=int(getattr(session.env,'take_action_cnt',0))+1
                emit(dict(event='execute',row=row,start_time=initial+row*queue.dt,end_time=initial+(row+1)*queue.dt,
                          actual_start=now,physical_tick=tick,execution_wall_time=time.perf_counter(),target=target.tolist(),controller='endpoint_linear_arm_native_gripper'))
            elif not np.array_equal(action,target):
                raise TimingError('active committed endpoint changed')
            fraction=(tick%action_ticks+1)/action_ticks
            position=previous+(target-previous)*fraction;velocity=(target-previous)/queue.dt
            position[[6,13]]=target[[6,13]];velocity[[6,13]]=0
            robot=session.env.robot
            # Same single-state passive-force reuse as the paired Full baseline.
            # The scope ends before kinematic updates and the real physics step.
            with control_scope() if control_scope is not None else nullcontext():
                robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
                robot.set_arm_joints(position[:6],velocity[:6],'left')
                robot.set_arm_joints(position[7:13],velocity[7:13],'right')
                robot.set_gripper(position[6],'left');robot.set_gripper(position[13],'right')
            session.env._update_kinematic_tasks()
            if runtime.fast_job is not None and not runtime.fast_job[0].done():
                fast_inflight_steps+=1
            if pending is not None and not camera.ready(pending):
                camera_inflight_steps+=1
            session.env.scene.step()
            if abs(clock()-(now+dt))>1e-8:
                raise TimingError('scene step skipped or rewrote physical time')
            physics_steps+=1
            step_wall=time.perf_counter();max_step_wall_gap=max(max_step_wall_gap,step_wall-last_step_wall)
            last_step_wall=step_wall
            _,lag=checked_now();max_lag=max(max_lag,lag);min_lag=min(min_lag,lag)
            session.env._update_transient_checks();session._consume_events()
            terminal=session.stop_reason()
            row_complete=(tick+1)%action_ticks==0
            if row_complete:completed_rows+=1
            if (row_complete or terminal is not None) and metrics_row<row:
                if session.metrics_tracker is not None:session.metrics_tracker.on_step()
                metrics_row=row
            if contact_cache.error is not None:
                # Native best-effort metrics must not swallow infrastructure failures.
                raise contact_cache.error
            # Release the GIL after every real tick, including catch-up loops.
            # No positive delay, scene pause, clock rebase, or skipped tick.
            time.sleep(0)
        inflight=False;fast_during_slow=0
        for event in events:
            if event['event']=='activation':inflight=False
            elif event['event']=='slow_started':inflight=True
            elif event['event']=='slow_completed':inflight=False
            elif event['event']=='fast_started' and inflight:fast_during_slow+=1
        end_wall=time.perf_counter();end_physics=clock()
        final_lag=bridge.check_backlog(end_physics,max_backlog,wall_time=end_wall)
        max_lag=max(max_lag,final_lag);min_lag=min(min_lag,final_lag)
        clock_evidence=continuous_clock_evidence(bridge,activation_wall=wall0,activation_physics=initial,
            end_wall=end_wall,end_physics=end_physics,physics_steps=physics_steps,
            max_lag=max_lag,min_lag=min_lag,max_backlog=max_backlog)
        emit(dict(event='terminal_clock',**clock_evidence))
        report=dict(protocol=PROTOCOL,time_contract=builder.manifest['time_contract'],
            realtime_valid=clock_evidence['passed'],continuous_valid=clock_evidence['passed'],simulation_rate=1.,
            success=terminal=='success',stop_reason=terminal,bootstrap_wall_seconds=bootstrap_seconds,
            bootstrap_latency_excluded_from_active_episode=True,action_interval_s=queue.dt,
            physical_duration=clock_evidence['physical_duration'],active_wall_seconds=clock_evidence['active_wall_seconds'],max_backlog_s=max_lag,
            raw_clock_acceptance=clock_evidence,
            physics_steps=physics_steps,physics_step_seconds=dt,max_physics_step_wall_gap_s=max_step_wall_gap,
            physics_steps_during_fast_inflight=fast_inflight_steps,
            physics_steps_during_camera=camera_inflight_steps,
            no_skipped_physics_steps=clock_evidence['checks']['physics_steps_match'],wall_release_gating=True,wall_and_physics_deadlines=True,
            fast_queries_during_slow=fast_during_slow,source_exposures_on_action_grid=True,
            action_limit=action_limit,started_targets=last_row+1,completed_targets=completed_rows,
            captures=accepted_captures,skipped_capture_requests=skip_captures,
            camera_warmup_captures=camera_warmup_captures,unfinished_query_deferrals=deferred,
            reference_versions_consumed=sorted(seen_references),events=events,
            executable_action_source='fast_only',on_policy_training_distribution=False,
            fast_graph_acceptance=runtime.fast_graph_receipt(),
            contact_cache=contact_cache.statistics(),
            **model_overlap_statistics(events),**action_response_statistics(events))
        return report
    except BaseException as exc:
        main_error=exc
        exc.dualrate_events=events
        exc.dualrate_report=dict(protocol=PROTOCOL,time_contract=builder.manifest['time_contract'],
            realtime_valid=False,continuous_valid=False,success=False,
            error=f'{type(exc).__name__}: {exc}',events=events)
        if on_failure is not None:
            try:on_failure(exc)
            except BaseException as callback_exc:
                emit(dict(event='failure_callback_error',error=f'{type(callback_exc).__name__}: {callback_exc}'))
        raise
    finally:
        # Stop the episode clock before draining unfinished CPU/GPU/readback work.
        post_terminal=True;cleanup_errors=[]
        try:
            if runtime is not None:runtime.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if pending is not None:
                c=camera.finish(pending)
                if video:video.append(c.observation['observation']['head_camera']['rgb'],
                    physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)
            elif completed_capture is not None and video:
                c=completed_capture
                video.append(c.observation['observation']['head_camera']['rgb'],
                    physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if contact_cache is not None:contact_cache.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        emit(dict(event='episode_workers_closed',
            cleanup_errors=[f'{type(e).__name__}: {e}' for e in cleanup_errors],
            contact_cache_restored=contact_cache is None or not contact_cache.installed))
        if cleanup_errors and main_error is None:
            exc=cleanup_errors[0];exc.dualrate_events=events
            raise exc
        if cleanup_errors and main_error is not None:
            main_error.dualrate_cleanup_errors=[f'{type(e).__name__}: {e}' for e in cleanup_errors]

