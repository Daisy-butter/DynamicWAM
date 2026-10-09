"""Bounded-jitter wall-clock physics with an independent inference process.

There is no virtual-delay or pause-world option. CPU camera/IPC/event work is
charged to the same clock; excessive backlog is an infrastructure violation.
Catch-up ticks retain the prior command and never backdate a newly received
plan. Startup model/camera warmup precedes this loop, first query does not.
"""
from __future__ import annotations
import bisect
import hashlib
import math
import time
from typing import Any, Callable
import numpy as np
from .timed_control import LimitedHermiteController, TimedPlanQueue
from .trajectory_recorder import controller_snapshot,ExecutedCommandRing


class RealtimeViolation(RuntimeError):
    pass


def run_continuous_episode(session: Any, worker: Any, *, task: str, seed: int,
                           mode: str = 'async', control_period: float = .1,
                           query_slots: int = 2, physics_timeout: float = 30.,
                           max_backlog: float = .1, max_tick_gap: float = .1,
                           max_velocity: float = 2., max_acceleration: float = 20.,
                           sink: Callable[[dict], None] = lambda item: None,
                           clock: Callable[[], float] = time.perf_counter,
                           sleep: Callable[[float], None] = time.sleep,
                           trajectory_recorder: Any | None = None) -> dict:
    if mode not in ('sync', 'async') or query_slots < 1 or physics_timeout <= 0:
        raise ValueError('invalid continuous protocol')
    measured_dt = float(session.timestep())
    dt = .004  # SAPIEN returns float32 0.00400000019; use exact protocol grid.
    ticks_per_slot = round(control_period / dt)
    if abs(measured_dt - dt) > 1e-8 or abs(ticks_per_slot*dt-control_period) > 1e-8:
        raise ValueError('protocol requires 4 ms physics and integral action slots')
    if worker.busy:
        raise RuntimeError('worker must be ready and idle at episode start')
    queue = TimedPlanQueue(control_period)
    controller = LimitedHermiteController(session.command_state(), period=control_period,
        max_velocity=max_velocity, max_acceleration=max_acceleration)
    origin_sim = session.physics_time()
    original_sink=sink
    previous_recording=bool(getattr(session,'trajectory_recording',False))
    if trajectory_recorder is not None:
        if not getattr(session,'asynchronous_camera',False):
            raise ValueError('trajectory recording requires the independent snapshot camera')
        trajectory_recorder.begin(task=task,seed=seed,origin_sim_time=origin_sim,
            initial_drive_targets=controller.q,physics_timeout=physics_timeout)
        session.trajectory_recording=True
        # Disk-backed event sinks must not perturb the collection behavior.
        sink=trajectory_recorder.event
    origin_wall = clock()
    last_tick_wall = origin_wall
    initial_position = session.object_position().copy()
    ticks = 0; queries = 0; inference_ticks = 0; camera_ticks = 0; installed = 0; hold_slots = 0
    next_query_slot = 0; active_record = None; active_target = False
    last_boundary = -1; inflight: dict | None = None
    lags: list[float] = []; gaps: list[float] = []; action_ages: list[float] = []
    tick_walls: list[float] = []
    requests: list[dict] = []; commands: list[dict] = []
    invalid_reason = None; stop_reason = None
    async_camera=bool(getattr(session,'asynchronous_camera',False))
    ready_packet=None; stale_camera_frames=0; overwritten_ready_frames=0
    executed_commands=ExecutedCommandRing()

    def physical() -> float:
        return ticks * dt

    def lag_check() -> float:
        lag = clock() - origin_wall - physical()
        if lag > max_backlog:
            raise RealtimeViolation(f'wall_backlog {lag:.6f} > {max_backlog:.6f}')
        return lag

    def tick() -> None:
        nonlocal ticks, last_tick_wall, inference_ticks, camera_ticks
        delay = origin_wall + (ticks+1)*dt - clock()
        if delay > 0:
            sleep(delay)
        lag = lag_check()
        before = clock(); gap = before - last_tick_wall
        if gap > max_tick_gap:
            raise RealtimeViolation(f'physics_tick_gap {gap:.6f} > {max_tick_gap:.6f}')
        q, v = controller.step(dt)
        saved_q=q.copy();saved_v=v.copy()
        if trajectory_recorder is not None:
            engine_before=session.physics_time()
        session.step_command(q, v)
        ticks += 1
        if worker.busy:
            inference_ticks += 1
        if getattr(session,'camera_busy',False):
            camera_ticks += 1
        last_tick_wall = before
        lags.append(lag); gaps.append(gap)
        tick_walls.append(before)
        actual = session.physics_time() - origin_sim
        if abs(actual - physical()) > dt * .1:
            raise RealtimeViolation(f'scene_clock drift expected={physical()} observed={actual}')
        executed_commands.append(saved_q,saved_v,tick_index=ticks-1)
        if trajectory_recorder is not None:
            trajectory_recorder.tick(saved_q,saved_v,tick_index=ticks-1,start_sim_time=(ticks-1)*dt,
                end_sim_time=physical(),engine_start_time=engine_before,engine_end_time=actual+origin_sim,
                hold=not active_target,record=active_record)

    def begin_interval(slot: int, *, publish_new: bool) -> None:
        nonlocal installed, active_target, active_record, hold_slots, last_boundary
        if publish_new:
            publication = queue.publish(slot)
            if publication is not None:
                if not publication['expired']:
                    publication['actual_install_wall'] = clock()
                installed += int(not publication['expired'])
                sink(dict(event='publish', **publication))
        item = queue.pop(slot+1)
        active_target = item is not None
        if item is None:
            hold_slots += 1; active_record = None
            controller.begin(None)
        else:
            action, active_record = item
            controller.begin(action, queue.peek(slot+2))
            action_ages.append(physical()-active_record['observation_sim_time'])
        record = dict(event='control_interval', slot=slot,
            start_sim_time=physical(), end_sim_time=(slot+1)*control_period,
            catch_up=not publish_new, hold=not active_target,
            request_id=None if active_record is None else active_record['request_id'],
            observation_age_s=None if active_record is None else physical()-active_record['observation_sim_time'])
        if active_record is not None:
            source_row=round(slot-active_record['action_origin_sim_time']/control_period)
            feedback_rows=int(active_record['metadata'].get('feedback_rows',0))
            record['returned_row_index']=source_row
            record['feedback_row_covered']=0<=source_row<feedback_rows
            magnitudes=active_record['metadata'].get('feedback_row_l2',[])
            record['feedback_row_l2']=float(magnitudes[source_row]) if 0<=source_row<len(magnitudes) else None
        commands.append(record); sink(record)
        last_boundary = slot

    def submit_packet(packet: dict, *, acquisition_started: float, capture_finished: float) -> None:
        nonlocal queries, inflight
        queries+=1
        stamp=float(packet['observation_sim_time']); capture_slot=int(packet['command_index'])
        noise_seed=int.from_bytes(hashlib.sha256(f'{task}:{seed}:{capture_slot}'.encode()).digest()[:4],'little')
        packet.update(_reset_policy=queries==1,_sampling_seed=noise_seed,
                      task_name=task,episode_seed=seed)
        dispatch=physical();source_controller=dict(packet['controller_state'])
        source_controller.update(dispatch_controller=controller_snapshot(controller,source_physics_time=dispatch,
            future_knots=[dict(end_slot=end,q=value[0]) for end,value in sorted(queue.knots.items())]),
            executed_prefix=executed_commands.prefix(stamp,dispatch))
        packet.update(controller_state=source_controller,dispatch_sim_time=dispatch)
        if trajectory_recorder is not None:trajectory_recorder.mark_query(packet)
        submitted=clock(); packet['policy_submitted_wall']=submitted
        request_id=worker.submit(packet)
        inflight=dict(request_id=request_id,observation_sim_time=stamp,dispatch_sim_time=dispatch,
            submitted_wall=submitted,start_ticks=ticks,
            acquisition_started_wall=acquisition_started,capture_finished_wall=capture_finished)
        for name in ('capture_id','snapshot_finished_wall','renderer_started_wall','camera_ready_wall'):
            if name in packet:inflight[name]=packet[name]
        sink(dict(event='submit',**inflight,capture_and_clock_age_wall_s=submitted-(origin_wall+stamp),
            observation_age_at_submit_s=physical()-stamp,
            capture_timing=packet.get('_capture_timing',{}),sampling_seed=noise_seed,initial=queries==1))

    try:
        while physical() < physics_timeout - 1e-9 and not session.stop_reason():
            lag_check()
            # Camera and result handling never give a future command to elapsed ticks.
            # Resume the owner clock before polling/capturing/publishing new work.
            if clock() - origin_wall - physical() >= dt * 1.01:
                # Previously published trajectories keep their physical schedule
                # while catching up; only new observations/results are deferred.
                if ticks % ticks_per_slot == 0 and last_boundary != ticks // ticks_per_slot:
                    begin_interval(ticks // ticks_per_slot, publish_new=False)
                tick()
                continue
            slot = ticks // ticks_per_slot
            boundary = ticks % ticks_per_slot == 0
            if async_camera:
                camera_result=session.poll_capture()
                if camera_result is not None:
                    if trajectory_recorder is not None:
                        trajectory_recorder.observation(camera_result,received_sim_time=physical())
                        # This auxiliary measurement is for offline targets;
                        # the policy keeps its original observation contract.
                        camera_result.pop('_trajectory_measured_joint_state',None)
                        camera_result.pop('_trajectory_drive_targets',None)
                    overwritten_ready_frames+=int(ready_packet is not None)
                    ready_packet=camera_result
                    sink(dict(event='camera_ready',capture_id=camera_result['capture_id'],
                        observation_sim_time=camera_result['observation_sim_time'],receive_sim_time=physical(),
                        acquisition_started_wall=camera_result['acquisition_started_wall'],
                        renderer_started_wall=camera_result['renderer_started_wall'],
                        capture_finished_wall=camera_result['capture_finished_wall'],
                        camera_ready_wall=camera_result['camera_ready_wall'],
                        snapshot_oracle=camera_result.get('snapshot_oracle')))
            result = worker.poll()
            if result is not None:
                if inflight is None or result['request_id'] != inflight['request_id']:
                    raise RuntimeError('unexpected inference response identity')
                record = queue.stage(result, physical())
                record.update(submitted_wall=inflight['submitted_wall'],
                    dispatch_sim_time=inflight['dispatch_sim_time'],
                    acquisition_started_wall=inflight['acquisition_started_wall'],
                    capture_finished_wall=inflight['capture_finished_wall'],
                    worker_started_wall=float(result['worker_started_wall']),
                    worker_finished_wall=float(result['worker_finished_wall']),
                    received_wall=clock(),
                    request_wall_s=clock()-inflight['submitted_wall'],
                    inference_physics_advanced=physical()-inflight['observation_sim_time'],
                    worker_compute_s=float(result['worker_finished_wall'])-float(result['worker_started_wall']))
                record['observation_to_return_wall_s'] = record['received_wall'] - inflight['acquisition_started_wall']
                record['observation_to_worker_finish_wall_s'] = record['worker_finished_wall'] - inflight['acquisition_started_wall']
                for name in ('capture_id','snapshot_finished_wall','renderer_started_wall','camera_ready_wall'):
                    if name in inflight:record[name]=inflight[name]
                worker_ticks = bisect.bisect_right(tick_walls, result['worker_finished_wall']) - bisect.bisect_left(tick_walls, result['worker_started_wall'])
                record['physics_ticks_during_worker'] = worker_ticks
                record['physics_advanced_during_worker_s'] = worker_ticks*dt
                requests.append(record); sink(dict(event='result', **record)); inflight = None
            if boundary and last_boundary != slot:
                begin_interval(slot, publish_new=True)
                should_query = (async_camera or not worker.busy) and slot >= next_query_slot
                if mode == 'sync':
                    should_query = should_query and not active_target and not queue.has_future(slot)
                if should_query:
                    stamp = physical()
                    capture_metadata=dict(sim_time=stamp,observation_sim_time=stamp,
                        command_index=slot,control_period_s=control_period,
                        active_plan_id=None if active_record is None else active_record['metadata'].get('plan_id'),
                        controller_hold=not active_target)
                    capture_metadata['controller_state']=controller_snapshot(controller,source_physics_time=stamp,
                        future_knots=[dict(end_slot=end,q=value[0]) for end,value in sorted(queue.knots.items())])
                    if async_camera:
                        capture_id=session.begin_capture(capture_metadata)
                        sink(dict(event='capture_begin',capture_id=capture_id,**capture_metadata,
                                  capture_submitted_wall=clock()))
                        next_query_slot=slot+query_slots
                    else:
                        before_capture = session.physics_time()
                        acquisition_started = clock()
                        packet = session.capture_packet()
                        capture_finished = clock()
                        if abs(session.physics_time()-before_capture) > 1e-8:
                            raise RuntimeError('capture advanced scene; timestamp contract broken')
                        packet.update(capture_metadata)
                        submit_packet(packet,acquisition_started=acquisition_started,capture_finished=capture_finished)
                        next_query_slot=slot+query_slots
            if async_camera and ready_packet is not None and not worker.busy:
                may_submit=mode=='async' or (not active_target and not queue.has_future(slot))
                if may_submit:
                    packet=ready_packet;ready_packet=None
                    current_plan=None if active_record is None else active_record['metadata'].get('plan_id')
                    if packet.get('active_plan_id') != current_plan:
                        stale_camera_frames+=1
                        sink(dict(event='camera_drop',reason='plan_changed_since_capture',
                            capture_id=packet['capture_id'],observation_sim_time=packet['observation_sim_time'],
                            captured_plan_id=packet.get('active_plan_id'),active_plan_id=current_plan))
                        next_query_slot=min(next_query_slot,slot+1)
                    else:
                        submit_packet(packet,acquisition_started=packet['acquisition_started_wall'],
                                      capture_finished=packet['capture_finished_wall'])
            tick()
        stop_reason = session.stop_reason() or 'physics_timeout'
    except RealtimeViolation as exc:
        invalid_reason = str(exc); stop_reason = 'infrastructure_invalid'
        sink(dict(event='clock_violation', reason=invalid_reason, physics_time=physical(), wall_time=clock()-origin_wall))
    except BaseException as exc:
        if trajectory_recorder is not None:trajectory_recorder.abort(exc)
        raise
    finally:
        if trajectory_recorder is not None:session.trajectory_recording=previous_recording
    wall = clock()-origin_wall
    # The source of a still-running result is never installed after episode termination.
    report=dict(protocol='domino_continuous_wall_v1', mode=mode, task=task, seed=seed,
        control_period_s=control_period, physics_dt=dt, physics_time=physical(), wall_time=wall,
        realtime_factor=physical()/wall if wall else None, physics_ticks=ticks,
        realtime_valid=invalid_reason is None, realtime_invalid_reason=invalid_reason,
        stop_reason=stop_reason, success=stop_reason == 'success' and invalid_reason is None,
        inference_calls=queries, inference_inflight_ticks=inference_ticks,
        asynchronous_camera=async_camera,camera_inflight_ticks=camera_ticks,
        stale_camera_frames=stale_camera_frames,overwritten_ready_camera_frames=overwritten_ready_frames,
        renderer_superseded_frames=getattr(getattr(session,'renderer',None),'superseded',0),
        installed_results=installed, expired_results=queue.expired_results,
        terminal_inflight=None if inflight is None else dict(inflight),
        hold_slots=hold_slots, controller_limited_ticks=controller.limited_ticks,
        tick_lag_s=lags, tick_gap_s=gaps, action_observation_age_s=action_ages,
        requests=requests, control_intervals=commands,
        object_displacement=(session.object_position()-initial_position).tolist(),
        controller=dict(kind='bounded_hermite_command', max_velocity=max_velocity,
            max_acceleration=max_acceleration, native_topp_equivalent=False),
        clock_limits=dict(max_backlog_s=max_backlog,max_tick_gap_s=max_tick_gap),
        query_slots=query_slots, physics_timeout_s=physics_timeout,
        executed_feedback_covered_rows=sum(bool(row.get('feedback_row_covered')) for row in commands),
        executed_nonzero_feedback_rows=sum(bool(row.get('feedback_row_covered')) and (row.get('feedback_row_l2') or 0)>0 for row in commands),
        expired_feedback_rows=sum(min(int(row['metadata'].get('feedback_rows',0)),row['clipped_prefix']) for row in requests),
        startup_first_query_included=True, virtual_delay=False)
    if trajectory_recorder is not None:
        report['trajectory_recording']=trajectory_recorder.finish(report,
            pending_camera_at_end=bool(getattr(session,'camera_busy',False)))
        trajectory_recorder.flush_events(original_sink)
    return report
