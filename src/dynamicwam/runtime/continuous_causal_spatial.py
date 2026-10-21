"""Independent actual-1x H2/R2 controller for causal spatial WAM.

The native continuous camera/trajectory/MP4 session is reused. Only the action
queue storage is shared with the historical H4 driver; every reservation here
passes H=2 explicitly. The policy process owns finite observations and complete
observation-only memory publications. This owner owns exposure and installation
timestamps, absolute drive-target commitments, and the real wall-clock deadline.
"""
from __future__ import annotations

import hashlib
import math
import time
import numpy as np

from .domino_continuous.continuous_rtc import CommittedActionQueue
from .domino_continuous.wall_sync import RealtimeViolation, sampling_seed

PROTOCOL = "continuous_world_actual_latency_causal_spatial_h2r2_v1"

def protocol_for_rate(simulation_rate):
    if not math.isfinite(simulation_rate) or simulation_rate != 1.:
        raise ValueError("causal spatial primary protocol requires fixed actual 1x")
    return PROTOCOL

def run_continuous_causal_spatial_episode(session, worker, *, task, episode_seed, action_limit,
                               sink=lambda record: None, action_interval_s=.16,
                               execute_horizon=2, prefix_max=2, max_backlog_s=.25,
                               infer_clock_tolerance_s=.02, frame_period_s=.16,
                               request_timeout_s=120., simulation_rate=1., clock=time.perf_counter,
                               sleep=time.sleep):
    """Execute H2/R2 on the measured wall/physics clock.

    Bootstrap initializes observation memory while native drive targets hold.
    Every subsequent exposure and immutable two-action deadline stays on one
    absolute .16-second action grid. The owner drains worker/camera after return.
    No fixed-rate concession or deadline extension is available.
    """
    protocol = protocol_for_rate(simulation_rate)
    simulation_rate = float(simulation_rate)
    if worker.busy:
        raise RuntimeError("inference worker must be drained before an episode")
    if action_limit < 1 or execute_horizon != 2 or prefix_max != 2:
        raise ValueError("require positive action limit and explicit H2/R2")
    if not math.isfinite(action_interval_s) or not math.isclose(action_interval_s, .16, abs_tol=1e-10):
        raise ValueError("causal spatial policy requires the .16-second training grid")
    if any(not math.isfinite(v) or v <= 0 for v in
           (max_backlog_s, infer_clock_tolerance_s, frame_period_s, request_timeout_s)):
        raise ValueError("timing limits must be finite and positive")
    if max_backlog_s > .25 or infer_clock_tolerance_s > .02:
        raise ValueError("RTC clock limits cannot exceed .25s backlog or .02s inference error")
    if getattr(session, "limit_retiming", True):
        raise ValueError("RTC requires nominal_training_timing; joint retiming changes trained prefix clock")
    if not np.isclose(float(session.action_interval_s), action_interval_s, atol=1e-10):
        raise ValueError("session action spacing does not match RTC checkpoint")
    dt = float(session.timestep())
    if not 0 < dt < action_interval_s or not np.isclose(round(action_interval_s / dt) * dt,
                                                       action_interval_s, atol=1e-8):
        raise ValueError("trained action interval must contain an integral number of physics ticks")
    wall0, phys0 = clock(), float(session.physics_time())
    object0 = np.asarray(session.object_position()).copy()
    queue = CommittedActionQueue(interval_s=action_interval_s)
    captures, requests, blocks, tracking = [], [], [], []
    active = pending = reservation = track = None
    grid0 = None
    track_start = None
    block_origin = block_rows = 0
    completed = started = query = 0
    last_handoff_boundary = 0
    last_handoff_query = 0
    next_head = phys0
    skipped_history_slots = 0
    terminal_reason = None
    max_lag = max_gap = overlap_ticks = 0.
    max_backlog_event = first_backlog_violation = None
    last_tick = wall0
    phase_ticks = {"capture": 0, "inference": 0, "execution": 0, "execution_and_inference": 0}

    def wall(): return clock() - wall0
    def physical(): return float(session.physics_time())
    def stopped(): return session.stop_reason() or terminal_reason

    def fail(message, **details):
        diagnostics = {"physics_s": physical(), "wall_s": wall(), "completed_targets": completed,
                       "started_targets": started, "query_index": query,
                       "active_request": active, "pending_capture": pending,
                       "completed_requests": requests, "completed_captures": captures,
                       "physics_runtime_timing": getattr(session, "runtime_timing_summary", lambda: {})(),
                       "simulation_rate": simulation_rate, "initial_physics_s": phys0,
                       "scaled_wall_elapsed_s": simulation_rate * wall(),
                       "observed_max_backlog_s": max_lag, "max_backlog_event": max_backlog_event,
                       "first_backlog_violation": first_backlog_violation, **details}
        raise RealtimeViolation(message, diagnostics=diagnostics)

    def lag(phase=None):
        nonlocal max_lag, max_backlog_event, first_backlog_violation
        now_wall, now_physics = wall(), physical()
        value = simulation_rate * now_wall - (now_physics - phys0)
        if value > max_lag:
            max_lag = value
            event = {"backlog_s": value, "wall_s": now_wall, "physics_s": now_physics,
                     "scaled_wall_elapsed_s": simulation_rate * now_wall,
                     "phase": phase or ("camera_pending" if pending is not None else
                         "inference_pending" if active is not None else
                         "execution" if track is not None else "hold_or_handoff"),
                     "query_index": pending.get("query_index") if pending is not None else
                         active.get("query_index") if active is not None else query,
                     "action_origin": pending["action_origin"] if pending is not None else
                         active["action_origin"] if active is not None else completed,
                     "completed_targets": completed, "started_targets": started,
                     "pending_capture": pending is not None, "worker_busy": bool(worker.busy)}
            max_backlog_event = event
            if value > max_backlog_s + 1e-8 and first_backlog_violation is None:
                first_backlog_violation = dict(event)
        return value

    def guard_actions(actions):
        result = queue.validate(actions)
        bounds = getattr(session, "_q_limits", None)
        if bounds is not None:
            result = np.clip(result, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
        return result

    def tick(*, paced=True):
        nonlocal last_tick, max_gap, overlap_ticks, started, completed, track, block_rows, terminal_reason
        # Stop catch-up at a handoff boundary so ready results are polled before
        # any future command is issued. Physics never executes a late suffix.
        if grid0 is not None and track is None:
            return False
        if paced:
            # Preserve strict-mode floating-point operation order at boundaries.
            wait = (wall0 + physical() - phys0 + dt - clock()) if simulation_rate == 1 else (
                wall0 + (physical() - phys0 + dt) / simulation_rate - clock())
            if wait > 0:
                sleep(wait)
        if track is not None:
            which = "execution_and_inference" if worker.busy else "execution"
            q, dq = track.sample(physical() + dt - track_start)
            if "first_command_wall_s" not in blocks[-1]:
                block = blocks[-1]
                block.update(first_command_wall_s=wall(), first_command_physics_s=physical())
                source = requests[block["source_query_index"]]
                source.update(first_suffix_command_wall_s=wall(), first_suffix_command_physics_s=physical(),
                    observation_to_first_command_wall_s=wall() - source["observation_wall_s"],
                    observation_to_first_command_physics_s=physical() - source["observation_physics_s"])
            session.step_command(q, dq)
            measured = np.asarray(session.current_qpos())
            tracking.append(np.abs(measured - q))
            overlap_ticks += int(worker.busy)
            while block_rows < len(track.row_times) and physical() + 1e-8 >= track_start + track.row_times[block_rows]:
                completed += 1
                block_rows += 1
                session.record_target_metrics()
                if block_rows < len(track.row_times) and stopped() is None:
                    session.begin_native_action()
                    started += 1
            if block_rows == len(track.row_times):
                blocks[-1].update(end_physics_s=physical(), end_wall_s=wall(), completed=True)
                track = None
                if completed >= action_limit:
                    terminal_reason = "action_budget_exhausted"
        else:
            which = "inference" if worker.busy else "capture"
            session.step_command(None, None)
        now = clock()
        max_gap = max(max_gap, now - last_tick)
        last_tick = now
        phase_ticks[which] += 1
        return True

    def catch_up():
        count = 0
        previous_lag, growing = lag(), 0
        while stopped() is None and lag() >= dt - 1e-9:
            # A ready camera must be consumed before more physics burns the
            # committed H2 budget. Observation time is already frozen; every
            # extra tick here only shrinks the inference window.
            if pending is not None and session.capture_ready():
                break
            if not tick(paced=False):
                break
            count += 1
            current_lag = lag()
            growing = growing + 1 if current_lag >= previous_lag - 1e-9 else 0
            if growing >= 8 and current_lag > max_backlog_s:
                # Outstanding async capture: unpaced physics contends with GPU
                # readback, so lag can grow even though the stall is recoverable.
                # Yield to the main loop (finish_capture / camera timeout).
                # True infra death with no pending IO still hard-fails.
                if pending is not None:
                    break
                fail("RTC physical backlog grows during catch-up", max_backlog_s=max_backlog_s)
            previous_lag = current_lag
            if count > 10000:
                fail("RTC physical clock cannot catch up")
        # Pending capture (ready or not) owns temporary backlog; camera timeout
        # still bounds a wedged readback. Only fail when nothing is in flight.
        if lag() > max_backlog_s and pending is None:
            fail("RTC wall/physics backlog exceeds declared limit", max_backlog_s=max_backlog_s)

    def capture(for_inference):
        nonlocal pending, next_head, reservation
        if pending is not None:
            fail("RTC camera ownership conflict at action boundary")
        frozen = physical()
        if for_inference:
            if active is not None or worker.busy:
                fail("RTC previous inference still outstanding at next query")
            reservation = None if grid0 is None else queue.reserve(completed, frozen, length=prefix_max)
        drive = np.asarray(session.current_drive_targets(), dtype=np.float64)
        if drive.shape != (14,) or not np.isfinite(drive).all():
            fail("invalid capture-time native drive targets")
        pending = {"physics_s": frozen, "wall_s": wall(), "for_inference": for_inference,
                   "capture_drive_targets": drive.tolist(),
                   "delivered": False, "asynchronous_readback": True,
                   "action_origin": completed, "query_index": query if for_inference else None}
        # Keep the exposure and reservation on the immutable action boundary.
        # Catching up here would move the observation and extend frozen+H2;
        # rendering overhead remains real lag, never hidden by re-labelling.
        pending["pre_submit_backlog_s"] = lag("camera_before_submit")
        session.begin_capture(for_inference=for_inference)
        pending["submit_duration_s"] = wall() - pending["wall_s"]
        pending["post_submit_backlog_s"] = lag("camera_submit")
        next_head = frozen + frame_period_s

    def finish_capture():
        nonlocal pending, active, query
        record = pending
        packet = session.finish_capture()
        record.update(duration_s=wall() - record["wall_s"], delivered=True,
                      physics_ticks_during_capture=int(round((physical() - record["physics_s"]) / dt)))
        record["capture_stages"] = getattr(session, "last_capture_diagnostics", {})
        record["finish_backlog_s"] = lag("camera_finish")
        captures.append(record)
        pending = None
        if not record["for_inference"] or stopped() is not None:
            before_catch_up = wall()
            catch_up()
            record["post_capture_catch_up_ms"] = 1000 * (wall() - before_catch_up)
            return
        if reservation is not None and physical() > reservation.deadline_physics_s + 1e-9:
            fail("RTC query capture exceeded committed prefix deadline")
        prefix = np.empty((0, 14), dtype=np.float64) if reservation is None else reservation.actions.copy()
        previous_actions, previous_valid = queue.proposal(record["action_origin"])
        # Only two editable rows from an already installed real policy output
        # are eligible soft context. Initial hold is never a proposed plan.
        overlap_valid = np.zeros(16, dtype=bool)
        if queue.generation:
            overlap_valid[prefix_max:prefix_max+2] = previous_valid[prefix_max:prefix_max+2]
        previous_valid = overlap_valid
        previous_actions[~previous_valid] = 0.
        # Endpoints are on the immutable queue grid, relative to actual capture.
        # Initial source priming is a hypothetical source plan, never executed.
        target_offsets = np.arange(1, 17, dtype=np.float64) * action_interval_s
        if grid0 is not None:
            target_offsets += grid0 + record["action_origin"] * action_interval_s - record["physics_s"]
        # The DOMINO packet is the raw head-over-two-wrists composite; letterbox
        # conversion belongs to the slow policy preprocessor, not this packet.
        if "frame" in packet:
            packet["fast_frame"] = np.array(packet["frame"], copy=True)
        packet["sim_time"] = record["physics_s"]
        packet["_rtc_capture_drive_targets"] = np.asarray(record["capture_drive_targets"],dtype=np.float64)
        packet["_rtc_capture_wall_perf_s"] = wall0+record["wall_s"]
        packet["_rtc_capture_available_wall_perf_s"] = clock()
        packet["_rtc_contract"] = PROTOCOL
        packet.update(_sampling_seed=sampling_seed(task, episode_seed, query), _reset_policy=query == 0,
                      _rtc_prefix_actions=prefix, _rtc_prefix_length=len(prefix),
                      _rtc_previous_actions=previous_actions,
                      _rtc_previous_valid=previous_valid,
                      _rtc_queue_generation=queue.generation,
                      _rtc_action_target_offsets=target_offsets,
                      _rtc_bootstrap_only=bool(query == 0),
                      _rtc_observation_action_index=record["action_origin"],
                      _rtc_cache_age_actions=record["action_origin"] % 16,
                      _rtc_observation_physics_s=record["physics_s"],
                      _rtc_action_interval_s=action_interval_s)
        # Dispatch before draining render lag, then poll while catching up.
        # A ready result cannot excuse excessive global backlog or extend the
        # immutable two-action deadline.
        # The legacy submit_* fields below are after render-lag draining.
        # Preserve them for old reports, but measure true end-to-end dispatch
        # separately: the model may finish before post-capture catch-up does.
        dispatch_wall, dispatch_physics = wall(), physical()
        object_at_dispatch = np.asarray(session.object_position()).tolist()
        request_id = worker.submit(packet)
        before_catch_up = wall()
        early = None
        previous_lag, growing, count = lag(), 0, 0
        stopped_for_ready_lag = False
        while stopped() is None and lag() >= dt - 1e-9:
            if early is None:
                early = worker.poll()
            if (
                early is not None
                and lag() <= max_backlog_s + 1e-9
            ):
                stopped_for_ready_lag = True
                break
            if reservation is not None and physical() + dt > reservation.deadline_physics_s + 1e-9:
                break
            if not tick(paced=False):
                break
            count += 1
            current_lag = lag()
            growing = growing + 1 if current_lag >= previous_lag - 1e-9 else 0
            if growing >= 8 and current_lag > max_backlog_s:
                fail("RTC physical backlog grows during catch-up", max_backlog_s=max_backlog_s)
            previous_lag = current_lag
            if count > 10000:
                fail("RTC physical clock cannot catch up")
        record["post_capture_catch_up_ms"] = 1000 * (wall() - before_catch_up)
        record["post_capture_catch_up_capped_for_deadline"] = (
            reservation is not None
            and lag() >= dt - 1e-9
            and not stopped_for_ready_lag
        )
        record["post_capture_catch_up_stopped_for_ready_result"] = bool(stopped_for_ready_lag)
        if lag() > max_backlog_s and early is None:
            fail("RTC wall/physics backlog exceeds declared limit", max_backlog_s=max_backlog_s)
        submit_wall, submit_physics = wall(), physical()
        if reservation is not None and submit_physics > reservation.deadline_physics_s + 1e-9:
            fail("RTC query capture exceeded committed prefix deadline")
        active = {"query_index": query, "request_id": request_id,
                  "sampling_seed": packet["_sampling_seed"], "initial_query": query == 0,
                  "observation_physics_s": record["physics_s"], "observation_wall_s": record["wall_s"],
                  "capture_wall_s": record["duration_s"], "submit_wall_s": submit_wall,
                  "submit_physics_s": submit_physics, "prefix_length": len(prefix),
                  "submit_timestamp_semantics": "legacy_post_capture_catch_up",
                  "dispatch_wall_s": dispatch_wall, "dispatch_physics_s": dispatch_physics,
                  "dispatch_timestamp_semantics": "before_worker_submit",
                  "object_at_dispatch": object_at_dispatch,
                  "simulation_rate": simulation_rate,
                  "action_origin": record["action_origin"], "memory_refresh_actions": 16,
                  "prefix_actions_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                  "previous_proposal_sha256": hashlib.sha256(previous_actions.tobytes()).hexdigest(),
                  "previous_proposal_valid_rows": int(previous_valid.sum()),
                  "queue_generation_at_dispatch": queue.generation,
                  "bootstrap_only": bool(query == 0),
                  "action_target_offsets": target_offsets.tolist(),
                  "replacement_semantics": "immutable_prefix_then_atomic_unexecuted_tail",
                  "suffix_not_before_physics_s": None if reservation is None else reservation.deadline_physics_s,
                  "installed": False, "rows_dropped": 0, "guidance": False,
                  "rtc_strategy": "trained_clean_prefix_conditioning", "prefetch": reservation is not None,
                  "startup_delay_excluded": False, "history": session.history_diagnostics(),
                  "object_at_submit": np.asarray(session.object_position()).tolist()}
        query += 1
        if early is not None:
            finish_request(early, terminal=stopped() is not None)

    def finish_request(result=None, *, terminal=False):
        nonlocal active, reservation, grid0, last_handoff_boundary, last_handoff_query
        r = active
        # AsyncWorldScheduler publishes absolute time.monotonic() timestamps.
        # Keep this separate from episode-relative wall(), including fake clocks.
        r.update(receive_wall_s=wall(), receive_physics_s=physical(), terminal_before_result=terminal)
        r["inference_wall_elapsed_s"] = r["receive_wall_s"] - r["submit_wall_s"]
        r["inference_physics_advance_s"] = r["receive_physics_s"] - r["submit_physics_s"]
        r["inference_clock_error_s"] = r["inference_wall_elapsed_s"] - r["inference_physics_advance_s"]
        r["inference_clock_check"] = "terminal_truncated" if terminal else "complete_interval"
        r["inference_interval_semantics"] = "legacy_post_catch_up_to_receive"
        r["scaled_inference_clock_error_s"] = simulation_rate * r["inference_wall_elapsed_s"] - r["inference_physics_advance_s"]
        r["dispatch_to_receive_wall_s"] = r["receive_wall_s"] - r["dispatch_wall_s"]
        r["dispatch_to_receive_physics_s"] = r["receive_physics_s"] - r["dispatch_physics_s"]
        r["scaled_dispatch_interval_error_s"] = simulation_rate * r["dispatch_to_receive_wall_s"] - r["dispatch_to_receive_physics_s"]
        r["global_scaled_clock_error_s"] = simulation_rate * r["receive_wall_s"] - (r["receive_physics_s"] - phys0)
        r["clock_gate_semantics"] = "strict_actual_interval" if simulation_rate == 1 else "declared_scaled_wall_global_bound"
        current_object = np.asarray(session.object_position())
        r["object_displacement_while_inference"] = (current_object - r.pop("object_at_submit")).tolist()
        r["object_displacement_dispatch_to_receive"] = (current_object - r.pop("object_at_dispatch")).tolist()
        if not terminal:
            if result["request_id"] != r["request_id"]:
                raise RuntimeError("out-of-order RTC inference response")
            if simulation_rate == 1:
                if abs(r["inference_clock_error_s"]) > infer_clock_tolerance_s:
                    fail("RTC inference clock mismatch", error_s=r["inference_clock_error_s"])
            elif abs(r["global_scaled_clock_error_s"]) > max_backlog_s:
                fail("RTC scaled-wall clock mismatch", error_s=r["global_scaled_clock_error_s"])
            raw = queue.validate(result["actions"])
            actions = guard_actions(raw)
            r.update(model_started_wall_s=float(result["started"]) - wall0,
                     model_finished_wall_s=float(result["finished"]) - wall0,
                     model_compute_wall_s=float(result["finished"] - result["started"]),
                     actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                     position_guard_clipped_coordinates=int(np.count_nonzero(actions != raw)),
                     policy_diagnostics=result.get("policy_timing", result.get("policy_diagnostics", result.get("metadata", {}))))
            if "ipc_timing" in result:
                r["ipc_timing"] = result["ipc_timing"]
            metadata = r["policy_diagnostics"]
            if metadata.get("policy") == "causal_spatial" and reservation is not None:
                source_time = float(metadata["source_sim_time"])
                produced = float(metadata["source_completed_wall_s"])
                consumed = float(metadata["source_consumed_wall_s"])
                age_at_install = max(physical(),phys0+wall())-source_time
                if not all(math.isfinite(v) for v in (source_time,produced,consumed,age_at_install)):
                    fail("nonfinite observation memory provenance")
                if produced > consumed or source_time > r["observation_physics_s"]+1e-8:
                    fail("observation memory was consumed before complete or comes from the future")
                if not 0 <= age_at_install <= 4.96+1e-8:
                    fail("observation memory exceeded its trained age at installation",memory_age_s=age_at_install)
                r["memory_age_at_install_s"] = age_at_install
            if reservation is None:
                # Bootstrap is observation-memory-only; its returned actions
                # are never installed. A fresh observation replans while H2 hold rows
                # execute on the now-established physical queue grid.
                # A hold continues the native command, not measured qpos.
                # Tracking lag (especially a closing gripper) must not turn
                # an unchanged commitment into a different target prefix.
                hold = np.asarray(session.current_drive_targets(), dtype=np.float64)
                if hold.shape != (14,) or not np.isfinite(hold).all():
                    raise RuntimeError("invalid causal spatial bootstrap drive targets")
                held = np.repeat(hold[None], 16, axis=0)
                actions = guard_actions(held)
                r["bootstrap_hold_source"] = "native_drive_targets"
                r["bootstrap_hold_position_guard_clipped_coordinates"] = int(np.count_nonzero(actions != held))
                r["bootstrap_actions_executed"] = False
                queue.bootstrap(actions)
                # Keep grid unset until the first track has been planned. All
                # bootstrap/planning wall latency therefore advances under hold.
                last_handoff_boundary = 0
            else:
                # Check prefix before clipping so guarding cannot hide mutation.
                if not np.allclose(raw[:len(reservation.actions)], reservation.actions, atol=2e-5, rtol=1e-5):
                    raise RuntimeError("RTC policy did not preserve the committed prefix")
                # At a handoff boundary catch_up deliberately yields to poll.
                # Its frozen physical timestamp cannot excuse a result that is
                # already late on the actual wall clock.
                ready_clock_s = max(physical(), phys0 + simulation_rate * wall())
                try:
                    queue.install(actions, reservation, ready_physics_s=ready_clock_s)
                except RealtimeViolation as exc:
                    exc.diagnostics.update(active_request=r, physics_s=physical(), wall_s=wall(),
                        completed_targets=completed, started_targets=started,
                        completed_requests=requests, completed_captures=captures)
                    raise
                last_handoff_boundary = reservation.origin + len(reservation.actions)
            r.update(installed=True, install_physics_s=physical(), install_wall_s=wall(),
                     suffix_effective_physics_s=physical() if reservation is None else reservation.deadline_physics_s,
                     observation_to_suffix_effect_s=(physical() if reservation is None else reservation.deadline_physics_s) - r["observation_physics_s"])
            last_handoff_query = r["query_index"]
        r["stop_reason"] = stopped() if terminal else None
        requests.append(r)
        sink(r)
        active = reservation = None

    def start_block():
        nonlocal track, track_start, grid0, block_rows, block_origin, started
        if completed and last_handoff_boundary != completed:
            fail("RTC late suffix: no valid H2 result at committed boundary",
                 deadline_physics_s=grid0 + completed * action_interval_s)
        if pending is not None:
            fail("RTC history capture overlaps the next query boundary")
        count = min(execute_horizon, action_limit - completed)
        before = wall()
        planned = session.plan_chunk(queue.take(completed, count))
        planning_wall = wall() - before
        if not np.allclose(planned.row_times, np.arange(1, count + 1) * action_interval_s, atol=1e-8):
            raise RuntimeError("RTC trajectory retimed fixed training action intervals")
        if grid0 is None:
            catch_up()
            if stopped() is not None:
                return
            grid0 = physical()
            requests[0].update(suffix_effective_physics_s=grid0,
                first_action_effect_wall_s=wall(),
                observation_to_suffix_effect_s=grid0 - requests[0]["observation_physics_s"])
        track, track_start, block_rows, block_origin = planned, grid0 + completed * action_interval_s, 0, completed
        if abs(physical() - track_start) > dt * .1:
            fail("RTC track start is off the immutable action grid", track_start_s=track_start)
        blocks.append({"origin": completed, "count": count, "start_physics_s": physical(),
                       "start_wall_s": wall(), "planning_wall_s": planning_wall, "completed": False,
                       "source_query_index": last_handoff_query,
                       "row_times_physics_s": (track_start + planned.row_times).tolist(),
                       "trajectory": {**planned.diagnostics, **getattr(planned, "guard_diagnostics", {})}})
        session.begin_native_action()
        started += 1
        if action_limit - completed > execute_horizon:
            capture(True)

    capture(True)
    while stopped() is None:
        # Service ready IO before catch-up so render lag cannot starve submit.
        if pending is not None and session.capture_ready():
            finish_capture()
        if stopped() is not None:
            break
        if worker.busy:
            if active is None:
                raise RuntimeError("RTC worker has no request record")
            if wall() - active["submit_wall_s"] > request_timeout_s:
                fail("RTC inference timeout")
            result = worker.poll()
            if result is not None:
                catch_up()
                finish_request(result, terminal=stopped() is not None)
        if stopped() is not None:
            break
        catch_up()
        if stopped() is not None:
            break
        if pending is not None and session.capture_ready():
            finish_capture()
        if stopped() is not None:
            break
        if track is None and queue.rows:
            start_block()
        if stopped() is not None:
            break
        # Do not let optional history occupy the next mandatory query exposure.
        # Missed frames stay missing (and carry their actual timestamp/masks).
        if track is not None and physical() + 1e-8 >= next_head:
            if (pending is None and track_start + track.duration - physical() > 2 * frame_period_s + 1e-8):
                capture(False)
            else:
                while next_head <= physical() + 1e-8:
                    skipped_history_slots += 1
                    next_head += frame_period_s
        if pending is not None and wall() - pending["wall_s"] > request_timeout_s:
            fail("RTC camera readback timeout")
        catch_up()
        if stopped() is None:
            tick()

    if active is not None:
        finish_request(terminal=True)
    session.record_target_metrics()
    if pending is not None:
        pending.update(duration_s=wall() - pending["wall_s"], delivered=False,
                       physics_ticks_during_capture=int(round((physical() - pending["physics_s"]) / dt)))
        captures.append(pending)
    lag("episode_complete")
    duration, sim_duration = wall(), physical() - phys0
    max_infer_error = max((abs(r.get("inference_clock_error_s", 0.)) for r in requests
        if r.get("inference_clock_check") == "complete_interval"), default=0.)
    clock_failures = []
    if not 0 <= max_lag <= max_backlog_s + 1e-8:
        clock_failures.append("max_backlog_s")
    if simulation_rate == 1 and not 0 <= max_infer_error <= infer_clock_tolerance_s + 1e-8:
        clock_failures.append("max_inference_clock_error_s")
    clock_valid = not clock_failures
    feedback = np.asarray(tracking)
    capture_gaps = np.diff([r["physics_s"] for r in captures if r["delivered"]])
    return {"protocol": protocol, "controller": "committed_clean_prefix_rtc_fixed_grid_h2r2",
            "causal_spatial": True,
            "bootstrap_mode": "memory_only_then_native_hold",
            "timing_mode": "nominal_training_timing", "limit_retiming": False,
            "clock": "actual_wall_clock" if simulation_rate == 1 else "scaled_wall_clock",
            "simulation_rate": simulation_rate, "initial_physics_s": phys0,
            "clock_valid": clock_valid, "continuous_valid": clock_valid,
            "clock_validation_failures": clock_failures,
            "clock_semantics": "physics_elapsed_target=simulation_rate*actual_wall_elapsed",
            "virtual_delay": False, "startup_delay_excluded": False,
            "rtc_strategy": "trained_clean_prefix_conditioning", "rtc_prefix_max": prefix_max,
            "action_interval_s": action_interval_s, "execute_horizon": execute_horizon,
            "slow_refresh_actions": 16, "action_grid_start_physics_s": grid0,
            "denoising_steps": 10, "physics_dt_s": dt,
            "memory_semantics": "observation_only_latest_complete",
            "belief_semantics": "finite_observed_window_no_candidate_actions",
            "future_semantics": "fresh_joint_future_action_K10_per_request",
            "capture_commands_semantics": "native_drive_targets_sampled_at_exposure",
            "overlap_rows": [2,4],
            "action_limit": action_limit, "started_targets": started, "completed_targets": completed,
            "inference_calls": len(requests), "physics_time_s": sim_duration, "wall_time_s": duration,
            "realtime_factor": sim_duration / duration if duration else 1., "max_backlog_s": max_lag,
            "max_tick_gap_wall_s": max_gap, "max_backlog_event": max_backlog_event,
            "first_backlog_violation": first_backlog_violation,
            "max_inference_clock_error_s": max_infer_error,
            "max_scaled_inference_clock_error_s": max((abs(r.get("scaled_inference_clock_error_s", 0.)) for r in requests
                if r.get("inference_clock_check") == "complete_interval"), default=0.),
            "max_dispatch_to_receive_wall_s": max((r.get("dispatch_to_receive_wall_s", 0.) for r in requests), default=0.),
            "clock_limits": {"max_backlog_s": max_backlog_s, "inference_clock_error_s": infer_clock_tolerance_s,
                "inference_clock_gate": "strict_actual_interval" if simulation_rate == 1 else "declared_scaled_wall_global_bound"},
            "phase_physics_seconds": {key: value * dt for key, value in phase_ticks.items()},
            "realtime_valid": simulation_rate == 1 and clock_valid, "rows_dropped": 0, "guided_requests": 0,
            "execution_inference_overlap": overlap_ticks > 0, "execution_inference_overlap_s": overlap_ticks * dt,
            "asynchronous_camera_readback": True, "history_frame_period_s": frame_period_s,
            "history_frame_period_semantics": "requested grid; missed frames are not fabricated",
            "skipped_history_grid_slots": skipped_history_slots,
            "capture_gap_physics_mean_s": float(capture_gaps.mean()) if len(capture_gaps) else None,
            "capture_gap_physics_max_s": float(capture_gaps.max()) if len(capture_gaps) else None,
            "tracking_abs_error_mean": feedback.mean(axis=0).tolist() if tracking else None,
            "tracking_abs_error_max": feedback.max(axis=0).tolist() if tracking else None,
            "physics_runtime_timing": getattr(session, "runtime_timing_summary", lambda: {})(),
            "requests": requests, "captures": captures, "blocks": blocks,
            "object_displacement": (np.asarray(session.object_position()) - object0).tolist(),
            "stop_reason": stopped(), "success": stopped() == "success"}



def validate_causal_spatial_report(report):
    """Recompute the H2/1x timing contract from saved measurements.

    This checks the controller protocol, not task success, policy quality,
    native MP4 decoding, or GPU cleanup. Those remain evaluator gates.
    """
    expected = dict(protocol=PROTOCOL,controller="committed_clean_prefix_rtc_fixed_grid_h2r2",
        causal_spatial=True,bootstrap_mode="memory_only_then_native_hold",
        timing_mode="nominal_training_timing",limit_retiming=False,simulation_rate=1.,
        clock="actual_wall_clock",rtc_prefix_max=2,execute_horizon=2,
        denoising_steps=10,action_interval_s=.16,virtual_delay=False,
        startup_delay_excluded=False,rows_dropped=0,guided_requests=0)
    for key,value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"causal spatial report contract mismatch: {key}")
    if not all(report.get(key) is True for key in ("clock_valid","continuous_valid","realtime_valid")):
        raise ValueError("causal spatial report failed real-time clock validity")
    if report.get("clock_validation_failures"):
        raise ValueError("causal spatial clock failure list is nonempty")
    def finite(value,name):
        number=float(value)
        if not math.isfinite(number): raise ValueError(f"nonfinite {name}")
        return number
    limit=finite(report["clock_limits"]["max_backlog_s"],"backlog limit")
    tolerance=finite(report["clock_limits"]["inference_clock_error_s"],"inference tolerance")
    if not 0 < limit <= .25 or not 0 < tolerance <= .02:
        raise ValueError("clock limits were loosened")
    if not 0 <= finite(report["max_backlog_s"],"maximum backlog") <= limit+1e-8:
        raise ValueError("reported global backlog exceeded limit")
    if report.get("first_backlog_violation") is not None:
        raise ValueError("a global backlog violation was recorded")
    if not 0 <= finite(report["max_inference_clock_error_s"],"maximum inference error") <= tolerance+1e-8:
        raise ValueError("reported inference clock mismatch")
    initial=finite(report["initial_physics_s"],"physics epoch")
    grid=report["action_grid_start_physics_s"]
    if grid is not None: grid=finite(grid,"action grid")
    requests=report["requests"]
    origins=[]
    for index,record in enumerate(requests):
        if record["query_index"] != index:
            raise ValueError("request indices are not contiguous")
        bootstrap=index==0
        if bool(record["bootstrap_only"]) != bootstrap or record["prefix_length"] != (0 if bootstrap else 2):
            raise ValueError("memory bootstrap or H2 prefix changed")
        if record.get("rows_dropped",0) or record.get("guidance"):
            raise ValueError("rows were dropped or guidance substituted")
        if record["inference_clock_check"]=="complete_interval":
            elapsed=finite(record["receive_wall_s"],"receive wall")-finite(record["submit_wall_s"],"submit wall")
            advance=finite(record["receive_physics_s"],"receive physics")-finite(record["submit_physics_s"],"submit physics")
            if abs(elapsed-advance)>tolerance+1e-8:
                raise ValueError("inference interval does not track actual wall time")
            if abs(float(record["inference_clock_error_s"])-(elapsed-advance))>1e-8:
                raise ValueError("inference error receipt was altered")
        if not record.get("installed"): continue
        if bootstrap:
            if record.get("bootstrap_actions_executed") is not False or record.get("bootstrap_hold_source")!="native_drive_targets":
                raise ValueError("bootstrap candidate actions were executed")
            continue
        origin=record["action_origin"];origins.append(origin)
        if type(origin) is not int or origin%2:
            raise ValueError("request origin is not on R2")
        observation=finite(record["observation_physics_s"],"observation physics")
        if grid is None or abs(observation-(grid+.16*origin))>1e-8:
            raise ValueError("exposure moved off immutable action grid")
        deadline=observation+.32
        if abs(finite(record["suffix_not_before_physics_s"],"deadline")-deadline)>1e-8:
            raise ValueError("H2 deadline was changed")
        ready=max(finite(record["install_physics_s"],"install physics"),
                  initial+finite(record["install_wall_s"],"install wall"))
        if ready>deadline+1e-8:
            raise ValueError("suffix installed after the physical or wall H2 deadline")
        metadata=record.get("policy_diagnostics",{})
        if metadata.get("policy")=="causal_spatial":
            source=finite(metadata["source_sim_time"],"memory source")
            if source>observation+1e-8 or not 0<=ready-source<=4.96+1e-8:
                raise ValueError("memory source age is invalid at installation")
            if finite(metadata["source_completed_wall_s"],"memory completed")>finite(metadata["source_consumed_wall_s"],"memory consumed"):
                raise ValueError("incomplete memory publication was consumed")
            if (metadata.get("H"),metadata.get("R"),metadata.get("K"))!=(2,2,10):
                raise ValueError("policy H/R/K differs from controller contract")
        if abs(finite(record["suffix_effective_physics_s"],"suffix effect")-deadline)>1e-8:
            raise ValueError("suffix became effective outside its commitment")
        if not np.allclose(record["action_target_offsets"],np.arange(1,17)*.16,atol=1e-9,rtol=0):
            raise ValueError("action target offsets differ from physical grid")
    if origins != list(range(0,2*len(origins),2)):
        raise ValueError("installed R2 requests skipped or duplicated a boundary")
    completed=report["completed_targets"];started=report["started_targets"]
    if not 0<=completed<=started<=report["action_limit"]:
        raise ValueError("native action budget counters invalid")
    count=0
    for block in report["blocks"]:
        if block["origin"]!=count or not 1<=block["count"]<=2:
            raise ValueError("execution block does not implement contiguous R2")
        if grid is None or abs(block["start_physics_s"]-(grid+.16*count))>1e-8:
            raise ValueError("execution block moved off the action grid")
        expected_times=grid+.16*(count+np.arange(1,block["count"]+1))
        if not np.allclose(block["row_times_physics_s"],expected_times,atol=1e-8,rtol=0):
            raise ValueError("execution endpoints were retimed")
        if block["completed"] and abs(block["end_physics_s"]-expected_times[-1])>1e-8:
            raise ValueError("completed block endpoint differs from training grid")
        count+=block["count"]
    return True

