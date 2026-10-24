"""Continuous DOMINO driver: only fast plans reach the robot's execution queue."""
from __future__ import annotations
import math
import time
import numpy as np
from .timing import ExecutionQueue, TimedPlan, TimingError
from .runtime import DualRateRuntime
from .online import ObservationHistory
from .actuators import ActuatorProjection

PROTOCOL='dualrate_v3_continuous_fast_only_dt018_v1'


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
    """
    fast_due,slow_due=runtime.due(now)
    if not (fast_due or slow_due):return
    snapshot=history.snapshot(now)
    age=now-snapshot[-1].exposure
    fresh=0<=age<=builder.max_observation_age+1e-8
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
        if route in deferred:
            began=deferred.pop(route)
            emit(dict(event='query_resumed',route=route,physical_time=now,
                deferred_seconds=now-began,observation_time=snapshot[-1].exposure,
                observation_age=age))
        if route=='fast':
            ref=runtime.store.snapshot(now)
            if ref:seen_references.add(ref.version)
            trace=queue.completed_execution_trace(initial,now)
            runtime.start_fast(lambda lease,reference,snapshot=snapshot,trace=trace:
                builder.current(snapshot,query_time=lease.query_time,action_start=lease.prefix.start_time,
                                executed=trace),observation_time=snapshot[-1].exposure)
        else:
            prefix=queue.snapshot_committed(now)
            runtime.start_slow(lambda snapshot=snapshot,now=now,prefix=prefix:
                builder.source(snapshot,source_time=now,action_start=prefix.start_time,vae=vae),
                prefix,source_time=now)


def run_episode(session, system, builder, camera, runtime_config, *, action_limit,
                frame_period=.06, max_backlog=.25, record=None, seed=0, video=None,
                camera_warmup_captures=3):
    """One owner thread steps 250 Hz physics; two workers preprocess and infer.

    Initial reference and fast plan are established before activation, explicitly
    recorded as bootstrap latency. Once activated every operation is wall-clock
    charged; there is no pause, catch-up grace, time scaling or slow-action fallback.
    Target rows are endpoints over [.18i,.18(i+1)); arms interpolate linearly from
    the preceding committed endpoint, grippers keep native command semantics.
    """
    if action_limit<1 or frame_period<=0 or max_backlog<=0:
        raise ValueError('invalid bounded deployment limits')
    events=[]
    def emit(event):
        events.append(event)
        if record:record(event)
    history=ObservationHistory();runtime=None;pending=None
    initial=float(session.physics_time());dt=float(session.timestep())
    if abs(dt-.004)>1e-8:
        raise TimingError('unverified DOMINO physics timestep')
    if abs(runtime_config.action_dt/.004-round(runtime_config.action_dt/.004))>1e-7:
        raise TimingError('action grid must lie on integer physics ticks')
    clock=lambda:float(session.physics_time())
    bootstrap_started=time.perf_counter()
    captured=warm_camera(camera,session,count=camera_warmup_captures,emit=emit)
    history.append(captured.observation,exposure=captured.physics_time_s,available=initial)
    if video:video.append(captured.observation['observation']['head_camera']['rgb'],
                         physics_time_s=initial,capture_wall_perf_s=captured.capture_wall_perf_s)
    held=tuple(float(x) for x in captured.observation['joint_action']['vector'])
    queue=ExecutionQueue(TimedPlan(initial,runtime_config.action_dt,(held,)*16,0,initial),
        episode_id=f'domino-{seed}',clock_id='sapien-scene-step-clock')
    prefix=queue.commit_prefix(query_time=initial,until=initial+runtime_config.protect_steps*queue.dt)
    runtime=DualRateRuntime(system,queue,clock,runtime_config,action_mean=builder.mean,
        action_scale=builder.scale,seed=seed,record=emit,
        action_projection=ActuatorProjection.from_robot(session.env.robot))
    emit(dict(event='actuator_contract',bounds=runtime.action_projection.serialized_bounds,
              provenance=runtime.action_projection.provenance,projection='uncommitted_fast_suffix_only'))
    vae=system.slow.native.compact_wan.video_model.vae
    frames=history.snapshot(initial)
    try:
        runtime.start_slow(lambda:builder.source(frames,source_time=initial,action_start=prefix.start_time,vae=vae),
                           prefix,source_time=initial)
        while runtime.slow_job is not None:
            runtime.poll();time.sleep(.001)
        if runtime.store.snapshot(initial) is None:
            raise TimingError('initial slow reference was not published')
        runtime.start_fast(lambda lease,ref:builder.current(frames,query_time=lease.query_time,
            action_start=lease.prefix.start_time),observation_time=frames[-1].exposure)
        while runtime.fast_job is not None:
            runtime.poll();time.sleep(.001)
        bootstrap_seconds=time.perf_counter()-bootstrap_started
        emit(dict(event='activation',physics_time=initial,bootstrap_wall_seconds=bootstrap_seconds,
                  bootstrap_policy='initial_reference_then_fast_plan_before_activation',
                  held_prefix_seconds=runtime_config.protect_steps*queue.dt))
        wall0=time.perf_counter();max_lag=0.;last_row=-1;completed_rows=0;metrics_row=-1
        previous=np.asarray(held,np.float64);target=previous.copy();next_capture=initial+frame_period
        accepted_captures=1;skip_captures=0;seen_references=set();terminal=None;deferred={}
        while terminal is None:
            now=clock();elapsed=time.perf_counter()-wall0;lag=elapsed-(now-initial)
            if now>=initial+action_limit*queue.dt-1e-9:
                terminal='action_budget_exhausted';break
            max_lag=max(max_lag,lag)
            if lag>max_backlog:
                raise TimingError(f'global wall/physics backlog {lag:.6f}s exceeds {max_backlog}s')
            runtime.poll()
            if pending is not None and camera.ready(pending):
                c=camera.finish(pending);pending=None
                history.append(c.observation,exposure=c.physics_time_s,available=clock())
                accepted_captures+=1
                if video:video.append(c.observation['observation']['head_camera']['rgb'],
                    physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)
                emit(dict(event='capture',exposure=c.physics_time_s,available=clock(),
                    timing=capture_timing(c,owner_processed_wall=time.perf_counter())))
            now=clock()
            dispatch_due_queries(runtime,builder,history,queue,now=now,initial=initial,vae=vae,
                                 emit=emit,deferred=deferred,seen_references=seen_references)
            now=clock()
            if now>=next_capture-1e-9:
                if pending is None:
                    pending=camera.begin(session.env,now)
                    emit(dict(event='capture_submitted',capture_id=pending.capture_id,exposure=now,
                        owner_submit_wall_seconds=time.perf_counter()-pending.capture_wall_perf_s))
                else:
                    skip_captures+=1
                next_capture=initial+(math.floor((now-initial)/frame_period+1e-9)+1)*frame_period
            # Only advance when the corresponding real-time tick is due.
            wait=(now-initial)+dt-(time.perf_counter()-wall0)
            if wait>0:
                time.sleep(min(wait,.001));continue
            row=math.floor((now-initial)/queue.dt+1e-9)
            if row>=action_limit:
                terminal='action_budget_exhausted';break
            action=np.asarray(queue.action_at(now),np.float64)
            if row!=last_row:
                previous=target.copy();target=action.copy();last_row=row
                session.env.take_action_cnt=int(getattr(session.env,'take_action_cnt',0))+1
                emit(dict(event='execute',row=row,start_time=initial+row*queue.dt,end_time=initial+(row+1)*queue.dt,
                          actual_start=now,target=target.tolist(),controller='endpoint_linear_arm_native_gripper'))
            elif not np.array_equal(action,target):
                raise TimingError('active committed endpoint changed')
            fraction=min(1.,max(0.,(now-initial-row*queue.dt+dt)/queue.dt))
            position=previous+(target-previous)*fraction;velocity=(target-previous)/queue.dt
            position[[6,13]]=target[[6,13]];velocity[[6,13]]=0
            robot=session.env.robot
            robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
            robot.set_arm_joints(position[:6],velocity[:6],'left')
            robot.set_arm_joints(position[7:13],velocity[7:13],'right')
            robot.set_gripper(position[6],'left');robot.set_gripper(position[13],'right')
            session.env._update_kinematic_tasks();session.env.scene.step()
            session.env._update_transient_checks();session._consume_events()
            terminal=session.stop_reason()
            row_complete=clock()>=initial+(row+1)*queue.dt-1e-9
            if row_complete:completed_rows+=1
            if (row_complete or terminal is not None) and metrics_row<row:
                if session.metrics_tracker is not None:session.metrics_tracker.on_step()
                metrics_row=row
        report=dict(protocol=PROTOCOL,realtime_valid=True,continuous_valid=True,simulation_rate=1.,
            success=terminal=='success',stop_reason=terminal,bootstrap_wall_seconds=bootstrap_seconds,
            bootstrap_latency_excluded_from_active_episode=True,action_interval_s=queue.dt,
            physical_duration=clock()-initial,active_wall_seconds=time.perf_counter()-wall0,max_backlog_s=max_lag,
            action_limit=action_limit,started_targets=last_row+1,completed_targets=completed_rows,
            captures=accepted_captures,skipped_capture_requests=skip_captures,
            camera_warmup_captures=camera_warmup_captures,unfinished_query_deferrals=deferred,
            reference_versions_consumed=sorted(seen_references),events=events,
            executable_action_source='fast_only',on_policy_training_distribution=False)
        return report
    except BaseException as exc:
        exc.dualrate_events=events
        raise
    finally:
        # Stop the episode clock before draining unfinished CPU/GPU/readback work.
        if runtime is not None:runtime.close()
        if pending is not None:
            c=camera.finish(pending)
            if video:video.append(c.observation['observation']['head_camera']['rgb'],
                physics_time_s=c.physics_time_s,capture_wall_perf_s=c.capture_wall_perf_s)

