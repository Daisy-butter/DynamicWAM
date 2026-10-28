"""Continuous DOMINO driver: only fast plans reach the robot's execution queue."""
from __future__ import annotations
import math
import threading
import time
import numpy as np
import torch
from .timing import ExecutionQueue, TimedPlan, TimingError, DeadlineMiss, BufferUnderrun, prefix_digest
from .runtime import DualRateRuntime
from .online import ObservationHistory
from .actuators import ActuatorProjection
from .physical_clock import WallPhysicsBridge
from .contact_cache import StepContactCache
from .producer import NUMERIC_TIME_EPSILON

PROTOCOL='dualrate_v4_exposure_anchor_wall_release_fast_only'


def queue_can_commit_protect(queue, *, query_time, protect_steps):
    """Eval-only read-only mirror of commit_prefix coverage (timing.py stays frozen).

    Soft-horizon reserve hold must not invent actions; this only asks whether the
    existing queue already has the contiguous rows begin_query would reserve.
    """
    if protect_steps < 1:
        raise ValueError('protect_steps must be >= 1')
    if not math.isfinite(query_time):
        raise ValueError('query_time must be finite')
    with queue._lock:
        first = queue._floor(query_time)
        until = queue.origin + (first + protect_steps) * queue.dt
        last = max(queue._committed_end, queue._ceil(until))
        if query_time > queue.origin + first * queue.dt + 1e-9:
            last = max(last, first + 1)
        return all(index in queue._rows for index in range(first, last))

def credit_wall_while_phys_frozen(bridge, *, now, frozen_wall_seconds):
    """Eval-only: phys held at `now` while wall advanced; do not count as backlog.

    camera.begin render (~0.10s) and inflight/soft-horizon phys freezes burn wall
    with phys fixed. Without a credit, pre-existing catch-up lag (~0.14s) plus one
    render trips max_backlog=0.25 (formal703 ep1). Shift wall_origin by the frozen
    wall interval only — preserves real lag, does not raise the backlog ceiling.
    Caller must also add the same duration to run_episode wall0 so wait()-paced
    catch-up stays coherent (smoke43 negative-lag race).
    """
    frozen = float(frozen_wall_seconds)
    if not math.isfinite(frozen) or frozen < 0:
        raise ValueError('frozen_wall_seconds must be a finite non-negative duration')
    if frozen > 0 and bridge.active:
        bridge.wall_origin += frozen






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
    """Schedule the next submission after this render can be absorbed at 1x.

    ``take_picture`` freezes physics for the whole wait. Submitting again on
    the raw 0.06 s grid stacks that wait faster than the scene can catch up
    and trips the backlog gate. The returned time stays on the capture grid.
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


def dispatch_due_queries(runtime,builder,history,queue,*,now,initial,vae,emit,deferred,seen_references):
    """Wait for genuinely fresh available pixels while physics/queue continue.

    Deferral never advances the route's last-query tick. A later lease uses its
    actual physical time and the original floor-grid/protect_steps deadline.

    Eval-only: when catch-up-hold leaves residual wall/phys lag, a deferred
    fresh-capture resume can hit training's bounded-budget ceil at protect=4
    while max_protect_steps=3 (smoke38). Training lag≈0 so it never trips;
    keep the same formula/cap — defer until phase+lag fits instead of abort.
    """
    fast_due,slow_due=runtime.due(now)
    if not (fast_due or slow_due):return
    snapshot=history.snapshot(now)
    age=now-snapshot[-1].exposure
    fresh=0<=age<=builder.max_observation_age+NUMERIC_TIME_EPSILON
    for route,due in (('fast',fast_due),('slow',slow_due)):
        if not due:continue
        if not fresh:
            if route not in deferred:
                deferred[route]=now
                emit(dict(event='query_deferred',route=route,physical_time=now,
                    observation_time=snapshot[-1].exposure,observation_available=snapshot[-1].available,
                    observation_age=age,max_observation_age=builder.max_observation_age,
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
            if ref:seen_references.add(ref.version)
            trace=queue.completed_execution_trace(initial,now)
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
                    observation_age=age,max_observation_age=builder.max_observation_age,
                    reason='insufficient_prefix_budget'))
                continue
            except BufferUnderrun as exc:
                # Soft horizon exhausted for next protect (formal700: camera defer
                # ~2.25s ate queued future rows). Do not abort: re-defer and let
                # run_episode soft-horizon reserve hold keep the last pose.
                deferred[route]=began if began is not None else now
                emit(dict(event='query_deferred',route=route,physical_time=now,
                    observation_time=snapshot[-1].exposure,observation_available=snapshot[-1].available,
                    observation_age=age,max_observation_age=builder.max_observation_age,
                    reason='insufficient_soft_horizon',error=type(exc).__name__,
                    error_message=str(exc)))
                continue
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



class TolerantWallPhysicsBridge(WallPhysicsBridge):
    """Eval-only wall scheduling tolerance (measured continuous-sim slack).

    Training WallPhysicsBridge.check_deadline is strict (wall≈phys). Eval admits
    ready_wall / wall_now up to wall_scheduling_tolerance past the mapped hard
    wall deadline (config-wired; default 0.017, cap 0.02). Does not touch
    physical_clock.py / timing queues. Paired with TolerantDualRateRuntime and
    run_episode inflight catch-up hold.
    """
    def __init__(self, physics_dt, *, wall_scheduling_tolerance=0.017, **kwargs):
        super().__init__(physics_dt, **kwargs)
        tol = float(wall_scheduling_tolerance)
        if not (0 <= tol <= 0.02):
            raise ValueError('wall_scheduling_tolerance must be in [0, 0.02]')
        self.wall_scheduling_tolerance = tol

    def check_deadline(self, protected_until, submit_margin=0., *, ready_wall=None, wall_now=None):
        if not self.active:
            return
        if not math.isfinite(protected_until) or not math.isfinite(submit_margin) or submit_margin < 0:
            raise ValueError("invalid protected wall deadline")
        deadline = (self.wall_origin + (protected_until - self.physics_origin)
                    - submit_margin + self.wall_scheduling_tolerance)
        now = self.wall_now() if wall_now is None else float(wall_now)
        for value in (now,) if ready_wall is None else (float(ready_wall), now):
            if not math.isfinite(value) or value >= deadline:
                raise DeadlineMiss("query exceeded protected prefix wall-clock submission deadline")



class TolerantDualRateRuntime(DualRateRuntime):
    """Eval-only DualRateRuntime: commit-before-inflight + small measured phys slack.

    Training DualRateRuntime polls inflight before commit with STRICT deadlines
    (no phys/wall tol) because wall≈phys — training has no Isaac render backlog.
    Eval continuous-sim previously catch-up-stepped while a leased fast worker
    ran, so phys advanced faster than wall and burned protected_until (smoke35:
    ~26ms over_hard while clearing ~36ms backlog in one ~0.41s query). Raising
    phys_tol then let commits land past the mutability frontier → TimingError
    (smoke34); converting that to DeadlineMiss only renamed the failure.

    Primary fix is in run_episode: hold catch-up while fast_job is inflight so
    the lease window restores training's wall≈phys invariant. Remaining eval
    slack is measurement jitter only (wall_tol≤0.02, phys_tol≤0.02):

      * TolerantWall admits ready_wall within wall_tol of mapped hard;
      * catch-up wall admit on commit/inflight when phys still < hard+phys_tol
        (wall can lead while phys is held);
      * never rewind replace clock (smoke20); safe hard-eps clamp only when
        hard-eps >= _last_time; else true now + tolerant validate.

    runtime.py / timing.py / physical_clock.py remain protected (训推一致).
    """
    def __init__(self, *args, physical_scheduling_tolerance=0.011, **kwargs):
        super().__init__(*args, **kwargs)
        tol = float(physical_scheduling_tolerance)
        if not (0 <= tol <= 0.02):
            raise ValueError('physical_scheduling_tolerance must be in [0, 0.02]')
        self.physical_scheduling_tolerance = tol

    def _fast_query_protection(self, query, requested_wall):
        """Same bounded-budget lease as training, with formal09 scheduling headroom.

        Training ceil((phase+lag+min_budget+margin)/dt) can leave only ~0.322s of
        remaining prefix when protect_steps=2. Occasional eval fast wall times
        reach ~0.35s (smoke09 q21), so the lease is born already too tight and
        inflight wall deadline fires while the worker is still finishing.

        Keep runtime.py protected; only bump effective protect_steps when the
        remaining prefix would be < min_query_budget + submit_margin + wall_tol
        (and still <= max_protect_steps). Does not change model/timing queues.
        """
        count, details = DualRateRuntime._fast_query_protection(self, query, requested_wall)
        cfg = self.config
        if not (cfg.min_query_budget_seconds > 0 and self.clock_bridge is not None and self.clock_bridge.active):
            return count, details
        if cfg.max_protect_steps is None:
            return count, details
        dt = self.queue.dt
        wall_tol = 0.017
        if self.clock_bridge is not None and hasattr(self.clock_bridge, 'wall_scheduling_tolerance'):
            wall_tol = float(self.clock_bridge.wall_scheduling_tolerance)
        # Measured smoke09–12: protect=2 rem∈[0.322,0.346] but fast wall tail 0.328–0.360s (worker still running). Training max_protect_steps=3 is the contract remedy — bump when remaining cannot cover a full protect_steps prefix net of margin (+ wall_tol).
        need = cfg.protect_steps * dt - cfg.submit_margin + wall_tol  # ≈0.357 @dt=0.18,tol=7ms
        while count < cfg.max_protect_steps:
            until = self.queue.origin + (math.floor((query - self.queue.origin) / dt + 1e-9) + count) * dt
            remaining = until - cfg.submit_margin - query
            if remaining >= need:
                break
            count += 1
        details = dict(details)
        details['effective_protect_steps'] = count
        details['eval_protect_headroom_seconds'] = need
        return count, details

    def poll(self):
        """Commit finished fast work before applying inflight deadlines.

        Coherent eval scheme (aligned to training begin_query/protect/submit_margin,
        wall bridge, replace monotonic clock; eval-only continuous-sim slack):

        1. Wall: TolerantWallPhysicsBridge.check_deadline admits ready_wall with
           measured wall_scheduling_tolerance (training check is strict).
        2. Physical replace: ALWAYS pass true monotonic now (never rewind /
           hard-eps). Training ExecutionQueue.replace forbids now < _last_time
           (smoke20 TimingError when eval clamped now=hard-1e-9 after execute).
        3. Catch-up / wall-tol window: if ready_wall already passed (1) but
           physical now sits in [hard, hard+phys_tol], temporarily wrap
           queue._validate_lease to apply the same phys_tol, then restore.
           Outside that window: DeadlineMiss (honest late).
        4. Inflight (still-running only): physical gate uses hard+phys_tol;
           wall gate uses TolerantWall. Commit-before-inflight preserves
           finished work whose ready_wall met the deadline while the owner
           was catching up (training polls inflight first — no Isaac backlog).
        """
        if self._failure is not None:
            raise TimingError('runtime has failed; close owned workers before continuing') from self._failure

        # 1) Publish finished slow jobs (same as parent).
        if self.slow_job is not None and self.slow_job[0].done():
            job,source,prefix,version=self.slow_job
            world,device_world,ready_wall,metrics=job.result()
            completed=float(self.clock())
            if self.clock_bridge is None or self.clock_bridge.ready(ready_wall,completed):
                self.slow_job=None
                payload={k:v for k,v in world.items() if isinstance(v,torch.Tensor)}
                payload['completed_time']=torch.tensor([completed],dtype=torch.float64)
                intervals=payload["world_intervals"][0][payload["world_types"][0]==1]
                times=tuple(sorted(set(intervals.flatten().tolist())))
                accepted=self.store.publish(source_time=source,completed_time=completed,version=version,
                    generation_stage=str(world["generation_stage"]),future_times=times,
                    future_valid=(True,)*len(times),payload=payload,now=completed,
                    episode_id=self.queue.episode_id,clock_id=self.queue.clock_id,
                    source_queue_version=prefix.queue_version,source_prefix_hash=prefix_digest(prefix))
                if accepted:
                    self._published_device_world=(version,device_world)
                self.record(dict(event="slow_completed",source_time=source,completed_time=completed,
                    published_time=completed,publication_wall_time=self.wall_clock(),
                    version=version,accepted=accepted,**metrics))

        # 2) Commit finished fast jobs BEFORE inflight deadline (eval catch-up).
        if self.fast_job is not None and self.fast_job[0].done():
            job,lease=self.fast_job
            submitted=False
            discarded=0
            try:
                plan,output,ready_wall,metrics=job.result()
                if not isinstance(metrics,dict):
                    metrics=dict(wall_seconds=metrics)
                now=float(self.clock())
                if self.clock_bridge is not None:
                    try:
                        self.clock_bridge.check_deadline(
                            lease.protected_until, lease.submit_margin, ready_wall=ready_wall)
                    except DeadlineMiss:
                        # Owner catch-up / backlog: wall can lead physics so ready_wall
                        # trips TolerantWall while physical now is still inside
                        # hard+phys_tol (smoke31 q26: over_hard_wall~0.12s with
                        # backlog~0.11s but over_hard_phys=10ms). Admit for commit;
                        # never rewind. Truly late phys still raises below / via replace.
                        hard = lease.protected_until - lease.submit_margin
                        if not (self.clock_bridge.active
                                and now < hard + self.physical_scheduling_tolerance):
                            raise
                    if not self.clock_bridge.ready(ready_wall, now):
                        return None
                # Commit clock policy (t111u, smoke20/34):
                # - Never pass replace a now < _last_time (smoke20 rewind TimingError).
                # - If hard <= now <= hard+phys_tol and hard-eps >= _last_time: safe
                #   non-rewind clamp replace(now=hard-eps) so mutability window matches
                #   training deadline (avoids smoke34 TimingError changing executing row).
                # - If hard-eps < _last_time: keep true now + temporary tolerant validate;
                #   if replace still hits executing-row TimingError, treat as DeadlineMiss.
                replace_now = now
                _restore = None
                if self.clock_bridge is not None and self.clock_bridge.active:
                    hard = lease.protected_until - lease.submit_margin
                    slack = self.physical_scheduling_tolerance
                    last = float(getattr(self.queue, '_last_time', now))
                    if now >= hard:
                        if (now - hard) > slack:
                            raise DeadlineMiss(
                                "query exceeded protected prefix submission deadline")
                        candidate = hard - 1e-9
                        if candidate + 1e-9 >= last:
                            replace_now = candidate
                        else:
                            _q = self.queue
                            _orig = _q._validate_lease
                            _hard = hard
                            _tol = slack
                            def _tolerant_validate(plan, now_, *, _orig=_orig, _q=_q,
                                                   _hard=_hard, _tol=_tol):
                                from .timing import DeadlineMiss, StalePlan, TimingError
                                lease = plan.lease
                                if lease is None:
                                    if getattr(_q, '_test_fixture', False):
                                        return
                                    raise TimingError("production plan requires its query lease")
                                if (lease != _q._inflight or lease.episode_id != _q.episode_id or
                                        lease.clock_id != _q.clock_id or
                                        plan.query_time != lease.query_time):
                                    raise StalePlan(
                                        "query/episode/clock lease is stale or mismatched")
                                if now_ >= _hard + _tol:
                                    raise DeadlineMiss(
                                        "query exceeded protected prefix submission deadline")
                                return _orig(plan, min(now_, _hard - 1e-9))
                            _q._validate_lease = _tolerant_validate
                            _restore = (_q, _orig)
                try:
                    try:
                        discarded = self.queue.replace(plan, now=replace_now)
                    except TimingError as exc:
                        if (replace_now == now and now >= (lease.protected_until - lease.submit_margin)
                                and "currently executing or committed action" in str(exc)):
                            raise DeadlineMiss(
                                "query exceeded protected prefix submission deadline") from exc
                        raise
                finally:
                    if _restore is not None:
                        _restore[0]._validate_lease = _restore[1]
                submitted = True
                if self.clock_bridge is not None:
                    try:
                        self.clock_bridge.check_deadline(
                            lease.protected_until, lease.submit_margin, ready_wall=ready_wall)
                    except DeadlineMiss:
                        hard = lease.protected_until - lease.submit_margin
                        if not (self.clock_bridge.active
                                and float(self.clock()) < hard + self.physical_scheduling_tolerance):
                            raise
                self.fast_job = None
            except BaseException as exc:
                self.fast_job = None
                self._discarded_jobs.append(('fast', (job, lease), dict(
                    reason='failed_submission', queue_updated_before_failure=submitted)))
                if not submitted:
                    self._cancel_fast(lease)
                self._failure = exc
                self.record(dict(event="fast_rejected", query_id=lease.query_id,
                                 error=type(exc).__name__,
                                 queue_updated_before_failure=submitted))
                raise
            self.record(dict(event="fast_committed", query_id=lease.query_id,
                             discarded_rows=discarded,
                             protected_until=lease.protected_until,
                             submit_margin=lease.submit_margin,
                             query_time=lease.query_time,
                             commit_time=float(self.clock()),
                             observation_time=lease.observation_time,
                             commit_wall_time=self.wall_clock(),
                             ready_wall_time=ready_wall,
                             reference_version=lease.reference_version, **metrics))
            return output

        # 3) Inflight deadline only while the worker is still running.
        if self.fast_job is not None and (self.clock_bridge is None or self.clock_bridge.active):
            _, lease = self.fast_job
            try:
                if float(self.clock()) >= (lease.protected_until - lease.submit_margin
                                           + self.physical_scheduling_tolerance):
                    raise DeadlineMiss('in-flight query exceeded physical submission deadline')
                if self.clock_bridge is not None:
                    try:
                        self.clock_bridge.check_deadline(lease.protected_until, lease.submit_margin)
                    except DeadlineMiss:
                        hard = lease.protected_until - lease.submit_margin
                        # Same catch-up skew as commit: wall ahead of phys under backlog.
                        if not (float(self.clock()) < hard + self.physical_scheduling_tolerance):
                            raise
            except BaseException as exc:
                self._cancel_fast(lease)
                self._failure = exc
                self.record(dict(event='fast_rejected', query_id=lease.query_id,
                                 error=type(exc).__name__, reason='inflight_deadline',
                                 worker_done=self.fast_job[0].done()))
                raise
        return None

def run_episode(session, system, builder, camera, runtime_config, *, action_limit,
                frame_period=None, max_backlog=.25, record=None, seed=0, video=None,
                camera_warmup_captures=3, on_ready=None, on_failure=None,
                wall_scheduling_tolerance=0.017, physical_scheduling_tolerance=0.011):
    """One owner thread steps 250 Hz physics; two workers preprocess and infer.

    Initial reference and fast plan are established before activation, explicitly
    recorded as bootstrap latency. Once activated every operation is wall-clock
    charged. Rendering can block physics; catch-up is bounded and never skips
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
    contact_cache=None;main_error=None
    def emit(event):
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
    bridge=TolerantWallPhysicsBridge(dt,wall_clock=time.perf_counter,wall_scheduling_tolerance=float(wall_scheduling_tolerance))
    def checked_now():
        now=clock()
        return now,bridge.check_backlog(now,max_backlog)
    bootstrap_started=time.perf_counter()
    captured=warm_camera(camera,session,count=camera_warmup_captures,emit=emit)
    history.append(captured.observation,exposure=captured.physics_time_s,available=initial)
    if video:video.append(captured.observation['observation']['head_camera']['rgb'],
                         physics_time_s=initial,capture_wall_perf_s=captured.capture_wall_perf_s)
    held=tuple(float(x) for x in captured.observation['joint_action']['vector'])
    queue=ExecutionQueue(TimedPlan(initial,runtime_config.action_dt,(held,)*16,0,initial),
        episode_id=f'domino-{seed}',clock_id='sapien-scene-step-clock')
    prefix=queue.commit_prefix(query_time=initial,until=initial+runtime_config.protect_steps*queue.dt)
    runtime=TolerantDualRateRuntime(system,queue,clock,runtime_config,action_mean=builder.mean,
        action_scale=builder.scale,seed=seed,record=emit,clock_bridge=bridge,
        physical_scheduling_tolerance=float(physical_scheduling_tolerance),
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
        if on_ready is not None:on_ready()
        bootstrap_seconds=time.perf_counter()-bootstrap_started
        wall0=time.perf_counter();bridge.activate(initial,wall0)
        emit(dict(event='activation',physics_time=initial,wall_time=wall0,bootstrap_wall_seconds=bootstrap_seconds,
                  bootstrap_policy='initial_reference_then_fast_plan_before_activation',
                  held_prefix_seconds=runtime_config.protect_steps*queue.dt,
                  physics_step_seconds=dt,action_ticks=action_ticks,
                  wall_scheduling_tolerance=float(bridge.wall_scheduling_tolerance),
                  physical_scheduling_tolerance=float(runtime.physical_scheduling_tolerance),
                  inflight_catchup_hold=True,soft_horizon_reserve_hold=True))
        max_lag=0.;last_row=-1;completed_rows=0;metrics_row=-1
        previous=np.asarray(held,np.float64);target=previous.copy();next_capture_tick=capture_ticks
        from dynamicwam.runtime.domino_rt import _sapien_color_reader
        bind_gil_releasing_physx_step(session.env, _sapien_color_reader.step_cpu_physx)
        accepted_captures=1;skip_captures=0;seen_references=set();terminal=None;deferred={};soft_horizon_hold_wall=None;soft_horizon_starve_seconds=30.0
        physics_steps=0;max_step_wall_gap=0.;last_step_wall=time.perf_counter()
        while terminal is None:
            now,lag=checked_now();max_lag=max(max_lag,lag)
            tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
            if tick>=action_limit*action_ticks:
                terminal='action_budget_exhausted';break
            runtime.poll()
            now,lag=checked_now();max_lag=max(max_lag,lag)
            if pending is not None and camera.ready(pending):
                completed_capture=camera.finish(pending);pending=None
                now,lag=checked_now();max_lag=max(max_lag,lag)
            if completed_capture is not None and bridge.ready(completed_capture.finished_wall_perf_s,now):
                c=completed_capture;completed_capture=None
                exact_grid_index(c.physics_time_s,initial,queue.dt)
                history.append(c.observation,exposure=c.physics_time_s,available=now)
                accepted_captures+=1
                if video:video.append(c.observation['observation']['head_camera']['rgb'],
                    physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)
                emit(dict(event='capture',exposure=c.physics_time_s,available=now,
                    readback_ready_wall=c.readback_finished_wall_perf_s,owner_ready_wall=c.finished_wall_perf_s,
                    availability_lower_bound=bridge.availability_time(c.finished_wall_perf_s),
                    timing=capture_timing(c,owner_processed_wall=time.perf_counter())))
            now,lag=checked_now();max_lag=max(max_lag,lag)
            # smoke14 BufferUnderrun: dispatch was gated on caught_up only. With backlog
            # (0.04–0.13s) a deferred fast route waiting on fresh pixels never resumed —
            # capture completed (age would be fresh) but catch-up skipped dispatch until
            # exposure aged past max_observation_age (0.24s). Training deferral contract
            # ("physics/queue continue while waiting for fresh pixels") requires dispatch
            # while deferred even mid-catch-up; caught_up still gates new non-deferred dues.
            if bridge.caught_up(now) or deferred:
                dispatch_due_queries(runtime,builder,history,queue,now=now,initial=initial,vae=vae,
                                     emit=emit,deferred=deferred,seen_references=seen_references)
                now,lag=checked_now();max_lag=max(max_lag,lag)
                tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
                if tick>=next_capture_tick and tick%capture_ticks==0:
                    skip_captures+=max(0,(tick-next_capture_tick)//capture_ticks)
                    if pending is None and completed_capture is None:
                        pending=camera.begin(session.env,now)
                        render_wall=time.perf_counter()-pending.capture_wall_perf_s
                        # formal703: render ~0.10s on top of ~0.14s lag → backlog 0.254>0.25
                        # smoke43: must also advance wall0 or wait()-driven catch-up races
                        # phys ahead of the credited bridge (negative lag ~-0.005).
                        credit_wall_while_phys_frozen(bridge,now=now,frozen_wall_seconds=render_wall)
                        wall0+=render_wall
                        after,lag=checked_now();max_lag=max(max_lag,lag)
                        if after!=now or pending.physics_time_s!=now:
                            raise TimingError('capture advanced scene time or relabelled exposure')
                        emit(dict(event='capture_submitted',capture_id=pending.capture_id,exposure=now,
                            exposure_tick=tick,owner_submit_wall_seconds=render_wall))
                        next_capture_tick=next_grid_capture_tick(tick,capture_ticks,dt,render_wall)
                    else:
                        skip_captures+=1;next_capture_tick=tick+capture_ticks
            now,lag=checked_now();max_lag=max(max_lag,lag)
            # Only advance when the corresponding real-time tick is due.
            wait=(now-initial)+dt-(time.perf_counter()-wall0)
            if wait>0:
                time.sleep(min(wait,.001));continue
            # Commit finished fast work before any hold decision so replace can
            # refill the soft horizon (smoke40: hold-before-poll left a done job
            # uncommitted, then wall backlog blew past 0.25s).
            runtime.poll()
            now,lag=checked_now();max_lag=max(max_lag,lag)
            # Training wall≈phys: no Isaac backlog burn during a leased fast query.
            # Eval catch-up otherwise advances phys faster than wall and exhausts
            # protected_until while the worker is still running (smoke35 ~26ms
            # over_hard clearing backlog mid-query). Hold phys (poll only) until
            # the fast worker finishes; resume catch-up after commit/reject.
            # Wall may lead while held — TolerantWall + catch-up wall admit cover it.
            if (runtime.fast_job is not None and not runtime.fast_job[0].done()):
                # Phys freeze while worker runs: credit wall+wall0 so lag does not
                # walk up to max_backlog (pairs with capture credit; smoke43 sync).
                freeze_t0=time.perf_counter()
                time.sleep(0.0005)
                frozen=time.perf_counter()-freeze_t0
                credit_wall_while_phys_frozen(bridge,now=now,frozen_wall_seconds=frozen)
                wall0+=frozen
                continue
            # Soft-horizon reserve hold (formal700 BufferUnderrun): keep the last
            # protect_need rows for the next begin_query. Hold when consuming one
            # more row would leave fewer than protect_need (protect_need+1 probe).
            # Only when no fast job is inflight/pending — otherwise poll/replace
            # refills. Pause wall↔phys while held so multi-second camera wait does
            # not trip max_backlog (smoke40). Do not invent/replay actions.
            protect_need=int(runtime_config.max_protect_steps or runtime_config.protect_steps)
            if (runtime.fast_job is None and
                    not queue_can_commit_protect(queue,query_time=now,protect_steps=protect_need+1)):
                if soft_horizon_hold_wall is None:
                    soft_horizon_hold_wall=time.perf_counter()
                    emit(dict(event='soft_horizon_hold',physical_time=now,
                              protect_steps=protect_need,reserve_probe_steps=protect_need+1,
                              reason='insufficient_future_rows_for_protect',
                              deferred_routes=sorted(deferred)))
                elif time.perf_counter()-soft_horizon_hold_wall>soft_horizon_starve_seconds:
                    raise BufferUnderrun(
                        f'soft-horizon reserve hold starved after '
                        f'{soft_horizon_starve_seconds:.3f}s waiting for new plan')
                # Re-anchor wall correspondence to the frozen phys pose.
                bridge.wall_origin=bridge.wall_now()-(now-bridge.physics_origin)
                wall0=time.perf_counter()-(now-initial)
                # Phys is frozen, so the normal capture grid cannot advance. If no
                # camera work is in flight, force a fresh capture at the held pose
                # (smoke41 hold#4: stuck off-grid with a stale in-flight frame and
                # starved). Age-0 exposure at `now` lets deferred routes resume.
                if pending is None and completed_capture is None:
                    pending=camera.begin(session.env,now)
                    render_wall=time.perf_counter()-pending.capture_wall_perf_s
                    credit_wall_while_phys_frozen(bridge,now=now,frozen_wall_seconds=render_wall)
                    wall0+=render_wall
                    after,lag=checked_now();max_lag=max(max_lag,lag)
                    if after!=now or pending.physics_time_s!=now:
                        raise TimingError('hold capture advanced scene time or relabelled exposure')
                    emit(dict(event='capture_submitted',capture_id=pending.capture_id,exposure=now,
                              exposure_tick=tick,owner_submit_wall_seconds=render_wall,
                              reason='soft_horizon_hold_refresh'))
                    next_capture_tick=next_grid_capture_tick(tick,capture_ticks,dt,render_wall)
                time.sleep(0.0005)
                continue
            soft_horizon_hold_wall=None
            now,lag=checked_now();max_lag=max(max_lag,lag)
            tick=exact_grid_index(now,initial,dt,tolerance=1e-8)
            row=tick//action_ticks
            if row>=action_limit:
                terminal='action_budget_exhausted';break
            action=np.asarray(queue.action_at(now),np.float64)
            if row!=last_row:
                previous=target.copy();target=action.copy();last_row=row
                session.env.take_action_cnt=int(getattr(session.env,'take_action_cnt',0))+1
                emit(dict(event='execute',row=row,start_time=initial+row*queue.dt,end_time=initial+(row+1)*queue.dt,
                          actual_start=now,physical_tick=tick,target=target.tolist(),controller='endpoint_linear_arm_native_gripper'))
            elif not np.array_equal(action,target):
                raise TimingError('active committed endpoint changed')
            fraction=(tick%action_ticks+1)/action_ticks
            position=previous+(target-previous)*fraction;velocity=(target-previous)/queue.dt
            position[[6,13]]=target[[6,13]];velocity[[6,13]]=0
            robot=session.env.robot
            robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
            robot.set_arm_joints(position[:6],velocity[:6],'left')
            robot.set_arm_joints(position[7:13],velocity[7:13],'right')
            robot.set_gripper(position[6],'left');robot.set_gripper(position[13],'right')
            session.env._update_kinematic_tasks()
            session.env.scene.step()
            if abs(clock()-(now+dt))>1e-8:
                raise TimingError('scene step skipped or rewrote physical time')
            physics_steps+=1
            step_wall=time.perf_counter();max_step_wall_gap=max(max_step_wall_gap,step_wall-last_step_wall)
            last_step_wall=step_wall
            _,lag=checked_now();max_lag=max(max_lag,lag)
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
        inflight=False;fast_during_slow=0
        for event in events:
            if event['event']=='activation':inflight=False
            elif event['event']=='slow_started':inflight=True
            elif event['event']=='slow_completed':inflight=False
            elif event['event']=='fast_started' and inflight:fast_during_slow+=1
        report=dict(protocol=PROTOCOL,time_contract=builder.manifest['time_contract'],
            realtime_valid=True,continuous_valid=True,simulation_rate=1.,
            success=terminal=='success',stop_reason=terminal,bootstrap_wall_seconds=bootstrap_seconds,
            bootstrap_latency_excluded_from_active_episode=True,action_interval_s=queue.dt,
            physical_duration=clock()-initial,active_wall_seconds=time.perf_counter()-wall0,max_backlog_s=max_lag,
            physics_steps=physics_steps,physics_step_seconds=dt,max_physics_step_wall_gap_s=max_step_wall_gap,
            no_skipped_physics_steps=True,wall_release_gating=True,wall_and_physics_deadlines=True,
            fast_queries_during_slow=fast_during_slow,source_exposures_on_action_grid=True,
            action_limit=action_limit,started_targets=last_row+1,completed_targets=completed_rows,
            captures=accepted_captures,skipped_capture_requests=skip_captures,
            camera_warmup_captures=camera_warmup_captures,unfinished_query_deferrals=deferred,
            reference_versions_consumed=sorted(seen_references),events=events,
            executable_action_source='fast_only',on_policy_training_distribution=False,
            fast_graph_acceptance=runtime.fast_graph_receipt(),
            contact_cache=contact_cache.statistics(),
            **model_overlap_statistics(events))
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

