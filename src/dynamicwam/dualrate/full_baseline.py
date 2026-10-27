"""Official Full sync16 policy in the common V4 continuous execution harness.

The sampler is unmodified: no known-action conditioning or RTC guidance. Its
untimed A16 is installed whole after completion on the next untouched execution
row. Exposure, query and installation times remain separate. Explicit wait holds
consume the same global physical-row budget as generated commands.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np

PROTOCOL = 'continuous_full_sync16_v4_grid180_linear_bootstrap_v2'


def camera_submission_timing(captured):
    """Owner-side render partition; timestamps and image contents stay intact."""
    return dict(update_render_wall_seconds=float(getattr(captured, 'render_update_wall_seconds', 0.)),
                camera_take_picture_wall_seconds=dict(getattr(captured, 'camera_submit_seconds', {})))


def warm_full_camera(camera, session, *, count, emit, event='camera_warmup',
                     last_two_limit=None):
    """Finite inactive renders, never published to model history or video."""
    from .deployment import capture_timing
    from .timing import TimingError
    if type(count) is not int or count < 1 or (last_two_limit is not None and count < 2):
        raise ValueError('positive fixed warmup count (at least two for stability) required')
    source = float(session.physics_time()); durations = []; captured = None
    for index in range(count):
        camera.begin(session.env, source)
        captured = camera.finish()
        if float(session.physics_time()) != source or captured.physics_time_s != source:
            raise TimingError('camera warmup advanced the inactive episode clock')
        timing = capture_timing(captured, owner_processed_wall=time.perf_counter())
        durations.append(timing['total_wall_seconds'])
        emit(dict(event=event, index=index, count=count, exposure=source,
                  timing=timing, submission_timing=camera_submission_timing(captured)))
    if last_two_limit is not None and any(x > last_two_limit for x in durations[-2:]):
        raise TimingError('post-barrier camera warmup failed last-two latency bound')
    return captured


def query_observation_timing(bridge, *, physical_now, request_wall, exposure,
                             max_observation_age):
    """Check source freshness in both clocks without relabelling either one."""
    wall_equivalent = bridge.expected_physics(request_wall)
    physical_age = physical_now-exposure
    wall_age = wall_equivalent-exposure
    age = max(physical_age, wall_age)
    # A sub-tick numerical tolerance includes 60 actual float32(.004) ticks
    # at a nominal .24 s boundary; it does not grant another physical tick.
    fresh = (0 <= physical_age and
             age <= max_observation_age+bridge.numeric_tolerance)
    return fresh, dict(query_time=physical_now, request_wall_time=request_wall,
        query_wall_equivalent_time=wall_equivalent, observation_time=exposure,
        observation_physical_age=physical_age, observation_wall_age=wall_age,
        observation_age=age, max_observation_age=max_observation_age,
        freshness_time_basis='max_actual_physics_and_wall_equivalent')


class FullChunkSchedule:
    """Owner-only integer-grid schedule, also exercised without CUDA/SAPIEN."""

    def __init__(self, held, action_ticks, bootstrap_hold_steps=2):
        self.held = np.array(held, dtype=np.float64, copy=True)
        if self.held.shape != (14,) or not np.isfinite(self.held).all():
            raise ValueError('finite fourteen-actuator held target required')
        if type(action_ticks) is not int or action_ticks < 1:
            raise ValueError('positive integer action ticks required')
        if type(bootstrap_hold_steps) is not int or bootstrap_hold_steps < 0:
            raise ValueError('nonnegative integer startup hold required')
        self.action_ticks = action_ticks
        self.bootstrap_hold_steps = bootstrap_hold_steps
        self.actions = None
        self.start_row = self.end_row = None
        self.inflight = False
        self.query_id = -1

    @staticmethod
    def checked_actions(actions):
        value = np.array(actions, dtype=np.float64, copy=True)
        if value.shape != (16, 14) or not np.isfinite(value).all():
            raise ValueError('official Full must return finite A16x14')
        value.setflags(write=False)
        return value

    def bootstrap(self, actions):
        if self.actions is not None or self.inflight:
            raise RuntimeError('bootstrap may occur only once')
        self.actions = self.checked_actions(actions)
        self.start_row = self.bootstrap_hold_steps
        self.end_row = self.start_row + 16
        self.query_id = 0

    def can_query(self, tick):
        return (self.actions is not None and not self.inflight and
                tick >= self.end_row * self.action_ticks)

    def begin_query(self, tick):
        if not self.can_query(tick):
            raise RuntimeError('sync16 query requires complete previous block and free slot')
        self.inflight = True
        self.query_id += 1
        return self.query_id

    def install(self, actions, *, tick, last_started_row):
        if not self.inflight:
            raise RuntimeError('no outstanding query to install')
        if type(tick) is not int or type(last_started_row) is not int or tick < 0:
            raise ValueError('integer physical tick and last row required')
        if tick < self.end_row * self.action_ticks:
            raise RuntimeError('cannot replace an unexecuted sync16 suffix')
        row = max((tick + self.action_ticks - 1) // self.action_ticks,
                  last_started_row + 1)
        self.held = self.actions[-1].copy()
        self.actions = self.checked_actions(actions)
        self.start_row, self.end_row = row, row + 16
        self.inflight = False
        return row

    def action(self, row):
        if self.actions is None:
            raise RuntimeError('bootstrap not installed')
        if self.start_row <= row < self.end_row:
            return self.actions[row-self.start_row].copy(), 'generated'
        if row >= self.end_row:
            return self.actions[-1].copy(), 'inference_wait_hold'
        return self.held.copy(), ('bootstrap_hold' if self.query_id == 0 else 'publication_wait_hold')


def build_native_packet(frames, policy, *, action_dt, pair_cache):
    """Native camera composite/state; V4 causal real-exposure flow selection."""
    from .online import history_features
    from dynamicwam.runtime.ciwam.adapters.domino.composite import composite_robotwin_frame

    last = frames[-1]
    h = history_features(frames, policy.runtime.head_flow_config, action_dt=action_dt,
                         pair_cache=pair_cache, include_spatial=False)
    obs = {'observation': {name: {'rgb': image} for name, image in
                           zip(('head_camera','left_camera','right_camera'), last.images)}}
    composite_config = policy.runtime.observation_config
    return dict(frame=composite_robotwin_frame(obs, composite_config),
                state=np.array(last.drive_target, copy=True), flow_frames=h['flow_rgb'],
                motion_features=h['motion'].numpy(),
                motion_interval_valid_mask=h['motion_valid'].numpy(),
                motion_acceleration_valid_mask=h['motion_acceleration_valid'].numpy())


def run_episode(session, policy, camera, *, action_dt, action_limit, frame_period=None,
                max_backlog=.25, max_observation_age=.24, seed=0, video=None, record=None,
                camera_warmup_captures=3, bootstrap_hold_steps=2,
                model_query_timeout_s=30., on_ready=None, on_armed=None, on_failure=None,
                passive_force_dedup=False):
    """Run one bounded episode; a single inference worker never owns the scene.

    ``on_ready`` precedes three frozen-state camera rewarm captures; ``on_armed``
    then waits for every peer to finish rewarming before activation. The caller owns model/camera
    lifetime and video finalization. Both success and exceptions drain this
    episode's worker and camera before returning. Exceptions carry
    ``dualrate_events`` and ``dualrate_report`` for durable failure evidence.
    ``on_failure(exc)`` signals the process supervisor before waiting for any
    outstanding worker, so a stalled device cannot hide a bounded timeout.
    """
    import torch
    from .actuators import ActuatorProjection
    from .deployment import (exact_grid_index, next_grid_capture_tick,
                             capture_timing, bind_gil_releasing_physx_step)
    from .online import ObservationHistory, ObservationPairCache
    from .physical_clock import WallPhysicsBridge
    from .contact_cache import StepContactCache
    from .passive_force import passive_force_scope
    from .timing import TimingError

    if type(passive_force_dedup) is not bool:
        raise ValueError('passive_force_dedup must be a boolean')
    passive_force_counts=dict(native_evaluations=0,reused_calls=0)
    if (type(action_limit) is not int or action_limit < 1 or
            not all(math.isfinite(x) and x > 0 for x in
                    (action_dt, max_backlog, max_observation_age, model_query_timeout_s))):
        raise ValueError('positive bounded execution settings required')
    frame_period = action_dt if frame_period is None else float(frame_period)
    initial = float(session.physics_time()); physics_dt = float(session.timestep())
    if abs(physics_dt-.004) > 1e-8:
        raise TimingError('unverified DOMINO physics timestep')
    action_ticks = round(action_dt/physics_dt)
    if action_ticks < 1 or abs(action_ticks*physics_dt-action_dt) > 1e-10:
        raise TimingError('action interval must be an integer number of actual physics ticks')
    capture_rows = round(frame_period/action_dt)
    if capture_rows < 1 or abs(capture_rows*action_dt-frame_period) > 1e-10:
        raise TimingError('captures must lie on the common action grid')
    if (policy.runtime.num_inference_steps != 10 or
            tuple(policy.runtime.video_refresh_steps) != (0,1) or policy.runtime.chunk_size != 16):
        raise ValueError('official baseline requires native K10/video[0,1]/A16')
    capture_ticks = capture_rows*action_ticks
    clock = lambda: float(session.physics_time())
    # Bind one callable here so owner/worker diagnostics and release gates use
    # the same clock, including deterministic tests replacing perf_counter.
    bridge = WallPhysicsBridge(physics_dt, wall_clock=time.perf_counter)
    history = ObservationHistory(); pair_cache = ObservationPairCache()
    events = []; owner = threading.get_ident(); report = None; post_terminal = False
    main_error = None
    def emit(event):
        if post_terminal:
            event = dict(event, post_terminal=True)
        elif bridge.active and threading.get_ident() == owner:
            event = dict(event, clock_pair=bridge.snapshot(clock()))
        events.append(event)
        if record:
            record(event)
    def checked_now():
        nonlocal max_lag
        now = clock()
        wall = bridge.wall_now()
        if bridge.active:
            max_lag = max(max_lag, bridge.expected_physics(wall)-now)
        return now, bridge.check_backlog(now, max_backlog, wall_time=wall)

    projection = ActuatorProjection.from_robot(session.env.robot)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='official-full-inference')
    future = None; pending = completed_capture = None
    query_request = None; queries = 0; max_lag = 0.; max_gap = 0.
    physics_steps = completed_rows = generated_rows = generated_completed = hold_rows = 0
    last_row = -1; metrics_row = -1; accepted_captures = skip_captures = 0
    terminal = None; deferred_since = None; bootstrap_seconds = 0.; wall0 = None
    row_kind = None; schedule = None; inference_metrics = []
    contact_cache = None
    owner_stage_totals = {name: dict(calls=0, wall_seconds=0., max_wall_seconds=0.)
                          for name in ('actuator', 'kinematic', 'physx', 'transient', 'events')}
    owner_stage_recent = deque(maxlen=4)
    bootstrap_started = time.perf_counter()

    def stage_finished(row, name, started):
        finished = time.perf_counter(); duration = finished-started
        total = owner_stage_totals[name]
        total['calls'] += 1; total['wall_seconds'] += duration
        total['max_wall_seconds'] = max(total['max_wall_seconds'], duration)
        if not owner_stage_recent or owner_stage_recent[-1]['row'] != row:
            owner_stage_recent.append(dict(row=row, stages={}))
        stages = owner_stage_recent[-1]['stages']
        stage = stages.setdefault(name, dict(calls=0, wall_seconds=0., max_wall_seconds=0.))
        stage['calls'] += 1; stage['wall_seconds'] += duration
        stage['max_wall_seconds'] = max(stage['max_wall_seconds'], duration)
        return finished

    def infer(frames, query_id):
        worker_started = time.perf_counter()
        packet = build_native_packet(frames, policy, action_dt=action_dt, pair_cache=pair_cache)
        prepared = time.perf_counter()
        device = torch.device(policy.runtime.device)
        device_ids = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        sample_seed = int(np.random.SeedSequence([int(seed), int(query_id)]).generate_state(1)[0])
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(sample_seed)
            actions = policy.sample(packet)
            if device.type == 'cuda':
                torch.cuda.current_stream(device).synchronize()
        model_done = time.perf_counter()
        actions = FullChunkSchedule.checked_actions(actions)
        projected, receipt = projection(torch.from_numpy(actions.copy()).float(),
                                        torch.zeros(16, dtype=torch.bool))
        result = projected.numpy().copy()
        ready = time.perf_counter()
        return result, ready, dict(query_id=query_id, sample_seed=sample_seed,
            worker_started_wall_time=worker_started, model_started_wall_time=prepared,
            gpu_done_wall_time=model_done, ready_wall_time=ready,
            preprocessing_wall_seconds=prepared-worker_started,
            model_wall_seconds=model_done-prepared, worker_wall_seconds=ready-worker_started,
            projection=receipt, sampler='official_HeadFlowRunner_K10_video01_no_prefix_no_guidance')

    def poll_result(now, tick):
        nonlocal future, query_request
        if future is None:
            return
        if time.perf_counter()-query_request['request_wall_time'] > model_query_timeout_s:
            raise TimingError('official Full query exceeded bounded inference timeout')
        if not future.done():
            return
        actions, ready_wall, info = future.result()
        if not bridge.ready(ready_wall, now):
            return
        install_row = schedule.install(actions, tick=tick, last_started_row=last_row)
        inference_metrics.append(info)
        emit(dict({**query_request, **info}, event='full_plan_installed',
                  publication_time=now, publication_wall_time=time.perf_counter(),
                  wall_release_lower_bound=bridge.availability_time(ready_wall),
                  install_row=install_row, installation_time=initial+install_row*action_dt,
                  execution_end_time=initial+(install_row+16)*action_dt,
                  rows=16, rows_dropped=0, execution_time_semantics='post_ready_full_chunk',
                  source_time_preserved=True))
        future = None; query_request = None

    def make_report(valid, error=None):
        return dict(protocol=PROTOCOL, time_contract='full_sync16_common_v4_physical_grid',
            realtime_valid=valid, continuous_valid=valid, simulation_rate=1.,
            success=terminal == 'success' and valid, stop_reason=terminal if valid else 'infrastructure_error',
            error=error, bootstrap_wall_seconds=bootstrap_seconds,
            bootstrap_latency_excluded_from_active_episode=True, bootstrap_hold_steps=bootstrap_hold_steps,
            action_interval_s=action_dt, physical_duration=clock()-initial,
            active_wall_seconds=0. if wall0 is None else time.perf_counter()-wall0,
            max_backlog_s=max_lag, max_backlog_limit_s=max_backlog,
            physics_steps=physics_steps, physics_step_seconds=physics_dt,
            max_physics_step_wall_gap_s=max_gap, no_skipped_physics_steps=valid,
            wall_release_gating=True, source_exposures_on_action_grid=True,
            query_requires_physics_caught_up=False,
            query_freshness_time_basis='max_actual_physics_and_wall_equivalent',
            action_limit=action_limit, started_targets=last_row+1, completed_targets=completed_rows,
            generated_targets_started=generated_rows, generated_targets_completed=generated_completed,
            hold_rows_started=hold_rows, holds_consume_action_budget=True,
            inference_calls=queries, rows_dropped=0, guided_requests=0,
            execution_inference_overlap=False, sampler_prefix_conditioning=False,
            controller='endpoint_linear_arm_native_gripper', captures=accepted_captures,
            skipped_capture_requests=skip_captures, camera_warmup_captures=camera_warmup_captures,
            post_barrier_camera_warmup_captures=3,
            post_barrier_warmup_last_two_limit_s=max_observation_age,
            history_frame_period_s=frame_period, max_observation_age_s=max_observation_age,
            inference_metrics=inference_metrics, events=events,
            passive_force_dedup=passive_force_dedup,
            passive_force_counts=dict(passive_force_counts) if passive_force_dedup else None,
            owner_stage_timing=dict(totals=owner_stage_totals, recent_rows=list(owner_stage_recent),
                                   clock='process_monotonic_perf_counter', recent_row_limit=4),
            contact_cache=None if contact_cache is None else contact_cache.statistics(),
            executable_action_source='official_full_sync16',
            model_forecast_time_not_relabelled=True)

    try:
        captured = warm_full_camera(camera, session, count=camera_warmup_captures, emit=emit)
        history.append(captured.observation, exposure=captured.physics_time_s, available=initial)
        accepted_captures = 1
        if video:
            video.append(captured.observation['observation']['head_camera']['rgb'],
                         physics_time_s=initial, capture_wall_perf_s=captured.capture_wall_perf_s)
        held = tuple(float(x) for x in captured.observation['joint_action']['vector'])
        schedule = FullChunkSchedule(held, action_ticks, bootstrap_hold_steps)
        future = pool.submit(infer, history.snapshot(initial), 0)
        actions, ready_wall, info = future.result(timeout=model_query_timeout_s)
        future = None; schedule.bootstrap(actions); queries = 1; inference_metrics.append(info)
        emit(dict(event='full_bootstrap_plan', observation_time=initial, query_time=initial,
                  install_row=bootstrap_hold_steps, rows=16, prefix_overwritten=False, **info))
        emit(dict(event='actuator_contract', bounds=projection.serialized_bounds,
                  provenance=projection.provenance, projection='generated_full_commands_only'))
        contact_cache = StepContactCache(session.env.scene, clock).install()
        if on_ready:
            on_ready()
        # Cohort setup can leave a ready renderer idle for tens of seconds.
        # Rewarm only at the unchanged inactive scene; preserve the original
        # bootstrap observation, sampled A16, history, video, and sample seeds.
        rewarmed = warm_full_camera(camera, session, count=3, emit=emit,
                                    event='camera_post_barrier_warmup', last_two_limit=max_observation_age)
        if on_armed:
            on_armed()
        if clock() != initial:
            raise TimingError('pre-activation barriers advanced the inactive episode clock')
        bootstrap_seconds = time.perf_counter()-bootstrap_started
        wall0 = time.perf_counter(); bridge.activate(initial, wall0)
        emit(dict(event='activation', physical_time=initial, wall_time=wall0,
                  bootstrap_wall_seconds=bootstrap_seconds, bootstrap_policy='full_plan_before_activation',
                  rewarm_to_activation_wall_seconds=wall0-rewarmed.finished_wall_perf_s,
                  held_prefix_seconds=bootstrap_hold_steps*action_dt, action_ticks=action_ticks,
                  physics_step_seconds=physics_dt))
        from dynamicwam.runtime.domino_rt import _sapien_color_reader
        bind_gil_releasing_physx_step(session.env, _sapien_color_reader.step_cpu_physx)
        previous = np.asarray(held, np.float64); target = previous.copy()
        next_capture_tick = capture_ticks; last_step_wall = time.perf_counter()
        while terminal is None:
            now, lag = checked_now(); max_lag = max(max_lag, lag)
            tick = exact_grid_index(now, initial, physics_dt, tolerance=1e-8)
            if tick >= action_limit*action_ticks:
                terminal = 'action_budget_exhausted'; break
            poll_result(now, tick)
            if pending is not None and camera.ready(pending):
                completed_capture = camera.finish(pending); pending = None
                now, lag = checked_now(); max_lag = max(max_lag, lag)
            if completed_capture is not None and bridge.ready(completed_capture.finished_wall_perf_s, now):
                c = completed_capture; completed_capture = None
                exact_grid_index(c.physics_time_s, initial, action_dt)
                history.append(c.observation, exposure=c.physics_time_s, available=now)
                accepted_captures += 1
                if video:
                    video.append(c.observation['observation']['head_camera']['rgb'],
                                 physics_time_s=c.physics_time_s, capture_wall_perf_s=c.capture_wall_perf_s)
                emit(dict(event='capture', exposure=c.physics_time_s, available=now,
                          availability_lower_bound=bridge.availability_time(c.finished_wall_perf_s),
                          timing=capture_timing(c, owner_processed_wall=time.perf_counter()),
                          submission_timing=camera_submission_timing(c)))
            now, lag = checked_now(); max_lag = max(max_lag, lag)
            tick = exact_grid_index(now, initial, physics_dt, tolerance=1e-8)
            # Full has no RTC protected-prefix deadline. Once its complete
            # chunk has executed, a published fresh image can start inference
            # while physics catches up. Camera submissions still wait below;
            # model outputs still cross the wall/physics release gate.
            if schedule.can_query(tick):
                snapshot = history.snapshot(now)
                fresh, query_timing = query_observation_timing(bridge,
                    physical_now=now, request_wall=time.perf_counter(),
                    exposure=snapshot[-1].exposure, max_observation_age=max_observation_age)
                if fresh:
                    query_id = schedule.begin_query(tick)
                    query_request = dict(query_id=query_id,
                        observation_available=snapshot[-1].available, **query_timing)
                    emit(dict(event='full_query_started', **query_request))
                    if deferred_since is not None:
                        emit(dict(event='query_resumed', physical_time=now,
                                  deferred_seconds=now-deferred_since, **query_timing))
                        deferred_since = None
                    future = pool.submit(infer, snapshot, query_id); queries += 1
                elif deferred_since is None:
                    deferred_since = now
                    emit(dict(event='query_deferred', physical_time=now, **query_timing,
                              observation_available=snapshot[-1].available,
                              reason='no_fresh_completed_capture'))
            now, lag = checked_now(); max_lag = max(max_lag, lag)
            if bridge.caught_up(now):
                now, lag = checked_now(); max_lag = max(max_lag, lag)
                tick = exact_grid_index(now, initial, physics_dt, tolerance=1e-8)
                if tick >= next_capture_tick and tick % capture_ticks == 0:
                    skip_captures += max(0, (tick-next_capture_tick)//capture_ticks)
                    if pending is None and completed_capture is None:
                        pending = camera.begin(session.env, now)
                        render_wall = time.perf_counter()-pending.capture_wall_perf_s
                        # Persist the expensive call before enforcing backlog,
                        # so a breach retains its exposure and render partition.
                        emit(dict(event='capture_submitted', capture_id=pending.capture_id,
                                  exposure=now, exposure_tick=tick, owner_submit_wall_seconds=render_wall,
                                  submission_timing=camera_submission_timing(pending)))
                        after, lag = checked_now(); max_lag = max(max_lag, lag)
                        if after != now or pending.physics_time_s != now:
                            raise TimingError('capture advanced scene time or relabelled exposure')
                        next_capture_tick = next_grid_capture_tick(tick, capture_ticks, physics_dt, render_wall)
                    else:
                        skip_captures += 1; next_capture_tick = tick+capture_ticks
            now, lag = checked_now(); max_lag = max(max_lag, lag)
            wait = now-initial+physics_dt-(time.perf_counter()-wall0)
            if wait > 0:
                time.sleep(min(wait, .001)); continue
            tick = exact_grid_index(now, initial, physics_dt, tolerance=1e-8)
            poll_result(now, tick)
            row = tick//action_ticks
            if row >= action_limit:
                terminal = 'action_budget_exhausted'; break
            action, kind = schedule.action(row)
            if row != last_row:
                previous = target.copy(); target = action.copy(); last_row = row; row_kind = kind
                generated_rows += int(kind == 'generated'); hold_rows += int(kind != 'generated')
                session.env.take_action_cnt = int(getattr(session.env, 'take_action_cnt', 0))+1
                emit(dict(event='execute', row=row, row_kind=kind, query_id=schedule.query_id,
                          start_time=initial+row*action_dt, end_time=initial+(row+1)*action_dt,
                          actual_start=now, physical_tick=tick, target=target.tolist(),
                          controller='endpoint_linear_arm_native_gripper'))
            elif not np.array_equal(action, target):
                raise TimingError('active committed endpoint changed')
            stage_started = time.perf_counter()
            fraction = (tick % action_ticks+1)/action_ticks
            position = previous+(target-previous)*fraction; velocity = (target-previous)/action_dt
            position[[6,13]] = target[[6,13]]; velocity[[6,13]] = 0
            robot = session.env.robot
            with passive_force_scope(robot,enabled=passive_force_dedup) as force_counts:
                robot._entity_qf(robot.left_entity); robot._entity_qf(robot.right_entity)
                robot.set_arm_joints(position[:6], velocity[:6], 'left')
                robot.set_arm_joints(position[7:13], velocity[7:13], 'right')
                robot.set_gripper(position[6], 'left'); robot.set_gripper(position[13], 'right')
            if force_counts is not None:
                for name in passive_force_counts:
                    passive_force_counts[name]+=force_counts[name]
            stage_started = stage_finished(row, 'actuator', stage_started)
            session.env._update_kinematic_tasks()
            stage_started = stage_finished(row, 'kinematic', stage_started)
            session.env.scene.step()
            stage_finished(row, 'physx', stage_started)
            if abs(clock()-(now+physics_dt)) > 1e-8:
                raise TimingError('scene step skipped or rewrote physical time')
            physics_steps += 1; step_wall = time.perf_counter()
            max_gap = max(max_gap, step_wall-last_step_wall); last_step_wall = step_wall
            _, lag = checked_now(); max_lag = max(max_lag, lag)
            stage_started = time.perf_counter()
            session.env._update_transient_checks()
            stage_started = stage_finished(row, 'transient', stage_started)
            session._consume_events(); terminal = session.stop_reason()
            stage_finished(row, 'events', stage_started)
            row_complete = (tick+1) % action_ticks == 0
            if row_complete:
                completed_rows += 1; generated_completed += int(row_kind == 'generated')
            if (row_complete or terminal is not None) and metrics_row < row:
                if session.metrics_tracker is not None:
                    session.metrics_tracker.on_step()
                metrics_row = row
            if contact_cache.error is not None:
                # Native metrics can catch read errors; an infrastructure
                # contact failure must still abort the whole episode.
                raise contact_cache.error
        report = make_report(True)
        return report
    except BaseException as exc:
        main_error = exc
        report = make_report(False, f'{type(exc).__name__}: {exc}')
        exc.dualrate_events = events; exc.dualrate_report = report
        if on_failure is not None:
            try:
                on_failure(exc)
            except BaseException as callback_exc:
                report['failure_callback_error'] = f'{type(callback_exc).__name__}: {callback_exc}'
        raise
    finally:
        # No physics is integrated during post-terminal drains. A real worker
        # exception is propagated, never counted as a successful terminal path.
        post_terminal = True
        cleanup_errors = []
        try:
            pool.shutdown(wait=True, cancel_futures=True)
            if future is not None and not future.cancelled():
                future.result()
                emit(dict(event='full_query_drained', discarded=True, query_id=schedule.query_id if schedule else 0))
        except BaseException as exc:
            cleanup_errors.append(exc)
        # Always drain the camera even if the inference future failed.
        try:
            if pending is not None:
                c = camera.finish(pending)
                if video:
                    video.append(c.observation['observation']['head_camera']['rgb'],
                                 physics_time_s=c.physics_time_s, capture_wall_perf_s=c.capture_wall_perf_s)
            elif completed_capture is not None and video:
                c = completed_capture
                video.append(c.observation['observation']['head_camera']['rgb'],
                             physics_time_s=c.physics_time_s, capture_wall_perf_s=c.capture_wall_perf_s)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if contact_cache is not None:
                contact_cache.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        emit(dict(event='episode_workers_closed', inference_thread_joined=True,
                  cleanup_errors=[f'{type(e).__name__}: {e}' for e in cleanup_errors]))
        if cleanup_errors and main_error is None:
            exc = cleanup_errors[0]
            exc.dualrate_events = events
            exc.dualrate_report = make_report(False, f'{type(exc).__name__}: {exc}')
            raise exc
        if cleanup_errors and main_error is not None:
            main_error.dualrate_report['cleanup_errors'] = [f'{type(e).__name__}: {e}' for e in cleanup_errors]
