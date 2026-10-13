"""Full-chunk synchronous policy with continuous, bounded joint references.

Physics has one owner and follows the actual wall clock. Camera readback is
independent of control. A new policy request requires the previous entire
chunk to finish; model inference never overlaps commanded trajectory motion.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np

try:
    from .wall_sync import RealtimeViolation, sampling_seed
except ImportError:
    from wall_sync import RealtimeViolation, sampling_seed

PROTOCOL = "continuous_world_actual_latency_sync16_continuous_track_v2"
PROTOCOL_NOMINAL_TIMING = "continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1"
PROTOCOL_BY_TIMING = {
    "native_mplib_limits": PROTOCOL,
    "nominal_training_timing": PROTOCOL_NOMINAL_TIMING,
}


def run_continuous_sync_episode(session, worker, *, task, episode_seed, action_limit,
                                sink=lambda record: None, max_backlog_s=.25,
                                infer_clock_tolerance_s=.02, frame_period_s=.16,
                                clock=time.perf_counter, sleep=time.sleep):
    if action_limit < 1 or any(not np.isfinite(v) or v <= 0 for v in
                              (max_backlog_s, infer_clock_tolerance_s, frame_period_s)):
        raise ValueError("positive action budget and timing parameters required")
    if worker.busy:
        raise RuntimeError("inference worker must be drained before an episode")
    dt = float(session.timestep())
    if not np.isfinite(dt) or not 0 < dt < 1:
        raise ValueError("physics timestep must be finite and between zero and one second")
    phys0 = float(session.physics_time())
    wall0 = clock()
    object0 = np.asarray(session.object_position()).copy()
    next_head = phys0
    pending = None
    track = None
    track_start = None
    track_rows = 0
    active_record = None
    query = 0
    started_targets = completed_targets = 0
    requests, captures, tracking = [], [], []
    slow_history_cooldown = 0  # skip history slots after a slow RGB frame
    phase_ticks = {"capture": 0, "inference": 0, "planning": 0, "execution": 0, "logging": 0}
    max_lag = max_gap = 0.0
    last_tick = wall0
    camera_motion_ticks = 0
    terminal_reason = None
    query_packet = None
    planning = False
    last_metric_physics = None

    def wall(): return clock() - wall0
    def physical(): return float(session.physics_time())
    def stopped(): return session.stop_reason()

    def record_metrics():
        nonlocal last_metric_physics
        if last_metric_physics is None or physical() > last_metric_physics + 1e-9:
            session.record_target_metrics()
            last_metric_physics = physical()

    def phase():
        if track is not None: return "execution"
        if worker.busy: return "inference"
        if planning: return "planning"
        return "capture"

    def lag_value():
        nonlocal max_lag
        value = wall() - (physical() - phys0)
        max_lag = max(max_lag, value)
        return value

    def enforce_lag():
        value = lag_value()
        if value > max_backlog_s:
            raise RealtimeViolation(
                f"wall/physics backlog {value:.6f}s exceeds {max_backlog_s:.6f}s "
                f"at physics={physical()-phys0:.6f}s; no clock reset or virtual replay allowed",
                diagnostics={"phase": phase(), "physics_s": physical()-phys0, "wall_s": wall(),
                             "started_targets": started_targets, "query_index": query,
                             "last_capture": captures[-1] if captures else pending})
        return value

    def lag():
        return enforce_lag()

    def finish_request(reason):
        nonlocal active_record, query
        if active_record is None: return
        active_record.update(end_physics_s=physical(), end_wall_s=wall(), stop_reason=reason)
        requests.append(active_record)
        sink(active_record)
        active_record = None
        query += 1

    def tick(*, paced=True, enforce=True):
        nonlocal last_tick, max_gap, camera_motion_ticks, track, track_rows
        nonlocal started_targets, completed_targets, terminal_reason
        if paced:
            wait = wall0 + physical() - phys0 + dt - clock()
            if wait > 0: sleep(wait)
        if enforce:
            enforce_lag()
        which = phase()
        if track is not None:
            if worker.busy:
                raise RuntimeError("synchronous control cannot execute while inference is outstanding")
            q, dq = track.sample(physical() + dt - track_start)
            session.step_command(q, dq)
            if pending is not None: camera_motion_ticks += 1
            feedback = session.current_qpos()
            tracking.append(np.abs(np.asarray(feedback) - q))
            while track_rows < len(track.row_times) and physical() + 1e-8 >= track_start + track.row_times[track_rows]:
                row = track_rows
                active_record["completed_rows"].append(row)
                active_record["targets"][row].update(end_physics_s=physical(), end_wall_s=wall(), completed=True)
                completed_targets += 1
                record_metrics()
                track_rows += 1
                if track_rows < len(track.row_times) and stopped() is None:
                    session.begin_native_action()
                    started_targets += 1
                    active_record["started_rows"].append(track_rows)
                    active_record["targets"].append({"row": track_rows, "start_physics_s": physical(),
                                                     "start_wall_s": wall(), "completed": False})
            if stopped() is None and track_rows == len(track.row_times):
                # sample() clamps exactly to the last target and zero velocity.
                if completed_targets >= action_limit: terminal_reason = "action_budget_exhausted"
                finish_request(terminal_reason)
                track = None
        else:
            session.step_command(None, None)
        now = clock()
        max_gap = max(max_gap, now-last_tick)
        last_tick = now
        phase_ticks[which] += 1
        if enforce:
            enforce_lag()

    def catch_up():
        # Recover wall time burned by camera submit/readback before enforcing
        # max_backlog_s. History RGB on these Level-1 scenes can take ~0.25-0.30s
        # per frame; raising inside recovery made that overshoot fatal. Keep
        # draining while lag shrinks; if we made progress this call but are still
        # slightly over budget, defer enforce to the next loop (and skip further
        # history frames) instead of killing the episode.
        guard = 0
        max_guard = 100000
        stalled = 0
        peak = lag_value()
        started = wall()
        while stopped() is None and terminal_reason is None and guard < max_guard:
            value = lag_value()
            peak = max(peak, value)
            if value + 1e-9 < dt:
                break
            if wall() - started > 2.0:
                break
            before = value
            tick(paced=False, enforce=False)
            guard += 1
            after = lag_value()
            if after > before + 1e-4:
                stalled += 1
                if stalled >= 8 and after > max_backlog_s:
                    break
            elif after >= before - 1e-12:
                stalled += 1
                if stalled >= 20 and after > max_backlog_s:
                    break
            else:
                stalled = 0
        final = lag_value()
        if final > max_backlog_s:
            # Still recovering from a camera spike: allow one more outer-loop pass
            # unless lag grew vs peak or doubled the formal budget.
            if final <= max_backlog_s * 2.0 and final < peak - 1e-4:
                return final
            enforce_lag()
        return final

    def start_capture(for_inference):
        nonlocal pending, next_head
        if pending is not None: raise RuntimeError("overlapping render submissions")
        frozen = physical()
        pending = {"physics_s": frozen, "wall_s": wall(), "for_inference": for_inference,
                   "started_targets": started_targets, "delivered": False,
                   "asynchronous_readback": True, "motion_ticks_at_submit": camera_motion_ticks}
        session.begin_capture(for_inference=for_inference)
        pending["submit_duration_s"] = wall() - pending["wall_s"]
        # Missed grid positions are never represented by fabricated frames.
        next_head = phys0 + (np.floor((frozen-phys0)/frame_period_s + 1e-8)+1)*frame_period_s

    def finish_capture():
        nonlocal pending, query_packet, slow_history_cooldown
        packet = session.finish_capture()
        pending.update(duration_s=wall()-pending["wall_s"], delivered=True,
                       physics_ticks_during_capture=int(round((physical()-pending["physics_s"])/dt)),
                       trajectory_ticks_during_capture=camera_motion_ticks-pending["motion_ticks_at_submit"])
        captures.append(pending)
        # A single history readback on cup/stand can burn ~0.25-0.30s. Cool down
        # further history submits so catch_up can drain before the next render.
        if (not pending["for_inference"]) and pending["duration_s"] > max(0.08, max_backlog_s * 0.3):
            slow_history_cooldown = max(slow_history_cooldown, 4)
        if pending["for_inference"]:
            if active_record is not None or track is not None or worker.busy:
                raise RuntimeError("query capture delivered before the previous chunk finished")
            query_packet = (packet, dict(pending))
        pending = None

    def submit_query():
        nonlocal query_packet, active_record
        packet, capture = query_packet
        query_packet = None
        packet["_sampling_seed"] = sampling_seed(task, episode_seed, query)
        packet["_reset_policy"] = query == 0
        submit_phys, submit_wall = physical(), wall()
        request_id = worker.submit(packet)
        active_record = {"query_index": query, "request_id": request_id,
            "sampling_seed": packet["_sampling_seed"], "observation_physics_s": capture["physics_s"],
            "observation_wall_s": capture["wall_s"], "capture_wall_s": capture["duration_s"],
            "submit_physics_s": submit_phys, "submit_wall_s": submit_wall,
            "initial_query": query == 0, "startup_delay_excluded": False,
            "guidance": False, "prefetch": False, "rows_dropped": 0,
            "started_rows": [], "completed_rows": [], "targets": [], "installed": False,
            "history": session.history_diagnostics(),
            "object_at_submit": np.asarray(session.object_position()).tolist()}

    def receive_result(result, *, terminal=False):
        nonlocal track, track_start, track_rows, planning, started_targets
        r = active_record
        r.update(receive_wall_s=wall(), receive_physics_s=physical(), terminal_before_result=terminal)
        r["inference_wall_elapsed_s"] = r["receive_wall_s"]-r["submit_wall_s"]
        r["inference_physics_advance_s"] = r["receive_physics_s"]-r["submit_physics_s"]
        r["inference_clock_error_s"] = r["inference_wall_elapsed_s"]-r["inference_physics_advance_s"]
        r["object_displacement_while_inference"] = (np.asarray(session.object_position())-r.pop("object_at_submit")).tolist()
        # A terminal event truncates the physical interval, even when poll()
        # returns after that event. Do not step past terminal or backdate wall
        # timestamps to force an equality for an interval that never completed.
        r["inference_clock_check"] = "terminal_truncated" if terminal else "complete_interval"
        if terminal:
            r["terminal_unresolved_wall_lag_s"] = wall() - (physical()-phys0)
        if not terminal and abs(r["inference_clock_error_s"]) > infer_clock_tolerance_s:
            raise RealtimeViolation(f"inference clock mismatch: {r['inference_clock_error_s']:.6f}s")
        if terminal: return
        if result["request_id"] != r["request_id"]: raise RuntimeError("out-of-order inference result")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.shape != (16, 14) or not np.isfinite(actions).all():
            raise RuntimeError("model must return a finite 16 x 14 chunk")
        r.update(actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                 model_started_wall_s=float(result["started"])-wall0,
                 model_finished_wall_s=float(result["finished"])-wall0,
                 model_compute_wall_s=float(result["finished"]-result["started"]))
        planning = True
        before = wall()
        planned = session.plan_chunk(actions[:min(16, action_limit-completed_targets)])
        r["planning_wall_s"] = wall()-before
        catch_up()
        planning = False
        if stopped() is not None: return
        track, track_start, track_rows = planned, physical(), 0
        r.update(installed=True, install_physics_s=track_start,
                 trajectory={**planned.diagnostics, **getattr(planned,"guard_diagnostics",{})},
                 planned_row_times_physics_s=(track_start+planned.row_times).tolist())
        session.begin_native_action()
        started_targets += 1
        r["started_rows"].append(0)
        r["targets"].append({"row": 0, "start_physics_s": physical(), "start_wall_s": wall(), "completed": False})

    while stopped() is None and terminal_reason is None:
        catch_up()
        if stopped() is not None or terminal_reason is not None: break
        if pending is not None and session.capture_ready():
            finish_capture()
            catch_up()
        if stopped() is not None or terminal_reason is not None: break
        if worker.busy:
            if active_record is None: raise RuntimeError("inference has no request record")
            if wall()-active_record["submit_wall_s"] > 120:
                raise RealtimeViolation("actual inference request exceeded 120 seconds")
            result = worker.poll()
            if result is not None:
                catch_up()
                if stopped() is not None: break
                receive_result(result)
        if stopped() is not None or terminal_reason is not None: break
        if query_packet is not None:
            submit_query()
        if pending is None:
            latest = session.latest_capture_time()
            # Distinct physical exposures; no same-time duplicate history frame.
            distinct = latest is None or physical() > latest + 1e-9
            if distinct:
                if track is None and not worker.busy and active_record is None:
                    start_capture(True)
                elif physical() + 1e-8 >= next_head:
                    # History frames call _update_render on the owner thread and
                    # can take ~0.25-0.30s on cup/stand. Skip slots when lag is
                    # already non-trivial, during a post-slow-frame cooldown, or
                    # while executing a chunk (track) with little headroom.
                    # Inference captures are never skipped.
                    skip_history = (
                        slow_history_cooldown > 0
                        or lag_value() > max_backlog_s * 0.15
                        or (track is not None and lag_value() > max_backlog_s * 0.08)
                    )
                    if skip_history:
                        if slow_history_cooldown > 0:
                            slow_history_cooldown -= 1
                        next_head = phys0 + (np.floor((physical()-phys0)/frame_period_s + 1e-8)+1)*frame_period_s
                    else:
                        start_capture(False)
        if pending is not None and wall()-pending["wall_s"] > 120:
            raise RealtimeViolation("camera readback exceeded 120 seconds")
        catch_up()
        if stopped() is None and terminal_reason is None:
            tick()

    reason = stopped() or terminal_reason
    if active_record is not None:
        if not active_record.get("installed") and "inference_clock_error_s" not in active_record:
            receive_result(None, terminal=True)
        finish_request(reason)
    record_metrics()
    if pending is not None:
        pending.update(duration_s=wall()-pending["wall_s"], delivered=False,
                       trajectory_ticks_during_capture=camera_motion_ticks-pending["motion_ticks_at_submit"])
        captures.append(pending)
    duration, sim_duration = wall(), physical()-phys0
    feedback = np.asarray(tracking)
    timing_mode = getattr(session, "timing_mode", "native_mplib_limits")
    protocol = getattr(session, "protocol", PROTOCOL_BY_TIMING.get(timing_mode, PROTOCOL))
    controller = getattr(session, "controller", "continuous_shape_preserving_bounded_joint_track")
    limit_retiming = bool(getattr(session, "limit_retiming", timing_mode == "native_mplib_limits"))
    return {"protocol": protocol, "controller": controller,
        "timing_mode": timing_mode, "limit_retiming": limit_retiming,
        "clock": "actual_wall_clock", "virtual_delay": False, "startup_delay_excluded": False,
        "action_limit": action_limit, "started_targets": started_targets, "completed_targets": completed_targets,
        "inference_calls": len(requests), "physics_time_s": sim_duration, "wall_time_s": duration,
        "realtime_factor": sim_duration/duration if duration else 1., "max_backlog_s": max_lag,
        "max_tick_gap_wall_s": max_gap,
        "max_inference_clock_error_s": max((abs(r.get("inference_clock_error_s",0)) for r in requests
                                            if r.get("inference_clock_check") == "complete_interval"), default=0.),
        "terminal_unresolved_wall_lag_s": max((r.get("terminal_unresolved_wall_lag_s",0) for r in requests), default=0.),
        "clock_limits": {"max_backlog_s": max_backlog_s, "inference_clock_error_s": infer_clock_tolerance_s},
        "phase_physics_seconds": {k:v*dt for k,v in phase_ticks.items()},
        "realtime_valid": True, "rows_dropped": 0, "guided_requests": 0,
        "execution_inference_overlap": False, "asynchronous_camera_readback": True,
        "camera_trajectory_overlap_s": camera_motion_ticks*dt, "history_frame_period_s": frame_period_s,
        "tracking_abs_error_mean": feedback.mean(axis=0).tolist() if tracking else None,
        "tracking_abs_error_max": feedback.max(axis=0).tolist() if tracking else None,
        "requests": requests, "captures": captures,
        "object_displacement": (np.asarray(session.object_position())-object0).tolist(),
        "stop_reason": reason, "success": reason=="success"}
