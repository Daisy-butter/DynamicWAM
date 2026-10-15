"""Actual-wall-clock RTC for the trained clean-prefix MotionKV model.

One owner advances physics and commits actions. A spawned worker computes a
16-row chunk while four already-committed rows execute at the calibrated
spacing. The returned suffix replaces only future rows, atomically. Late
results fail explicitly: extending a four-row reservation is not trained RTC.
The initial d=0 bootstrap holds commands while the world continues to move.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np

try:
    from .wall_sync import RealtimeViolation, sampling_seed
except ImportError:
    from wall_sync import RealtimeViolation, sampling_seed

PROTOCOL = "continuous_world_actual_latency_motionkv_rtc_h4r4_v1"


@dataclass(frozen=True)
class PrefixReservation:
    origin: int
    actions: np.ndarray
    observation_physics_s: float
    deadline_physics_s: float


class CommittedActionQueue:
    """CPU queue; absolute unnormalised targets, indexed by execution position."""

    def __init__(self, *, interval_s=.16, horizon=16, prefix_max=4):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("action interval must be positive")
        if horizon != 16 or prefix_max != 4:
            raise ValueError("this checkpoint contract requires horizon=16 and prefix_max=4")
        self.interval_s, self.horizon, self.prefix_max = interval_s, horizon, prefix_max
        self.rows: dict[int, np.ndarray] = {}
        self.generation = 0

    @staticmethod
    def validate(actions):
        array = np.asarray(actions, dtype=np.float64)
        if array.shape != (16, 14) or not np.isfinite(array).all():
            raise ValueError("RTC worker must return finite [16,14] absolute actions")
        return array.copy()

    def bootstrap(self, actions):
        if self.rows:
            raise RuntimeError("queue bootstrap cannot overwrite an existing plan")
        self.rows = {i: a.copy() for i, a in enumerate(self.validate(actions))}

    def take(self, origin, count):
        try:
            return np.stack([self.rows[i] for i in range(origin, origin + count)])
        except KeyError as exc:
            raise RealtimeViolation("RTC queue underrun", diagnostics={"origin": origin, "count": count}) from exc

    def reserve(self, origin, observation_physics_s, length=4):
        if not isinstance(length, int) or not 0 < length <= self.prefix_max:
            raise ValueError("committed prefix exceeds trained limit; no truncation")
        prefix = self.take(origin, length).copy()
        prefix.setflags(write=False)
        return PrefixReservation(origin, prefix, float(observation_physics_s),
                                 float(observation_physics_s) + length * self.interval_s)

    def install(self, actions, reservation, *, ready_physics_s):
        output = self.validate(actions)
        d = len(reservation.actions)
        if not math.isfinite(ready_physics_s) or ready_physics_s > reservation.deadline_physics_s + 1e-9:
            raise RealtimeViolation("RTC late suffix: inference exceeded committed prefix",
                diagnostics={"ready_physics_s": ready_physics_s,
                             "deadline_physics_s": reservation.deadline_physics_s,
                             "prefix_length": d, "origin": reservation.origin})
        if not np.array_equal(self.take(reservation.origin, d), reservation.actions):
            raise RuntimeError("RTC stale reservation: committed queue was changed")
        # Normalise/denormalise in the policy can introduce float32 roundoff.
        if not np.allclose(output[:d], reservation.actions, atol=2e-5, rtol=1e-5):
            raise RuntimeError("RTC worker changed the committed action prefix")
        replacement = dict(self.rows)
        for row in range(d, self.horizon):
            replacement[reservation.origin + row] = output[row].copy()
        # Expired predictions beyond this chunk must never be reused.
        replacement = {i: a for i, a in replacement.items()
                       if reservation.origin <= i < reservation.origin + self.horizon}
        self.rows = replacement
        self.generation += 1


def run_continuous_rtc_episode(session, worker, *, task, episode_seed, action_limit,
                               sink=lambda record: None, action_interval_s=.16,
                               execute_horizon=4, prefix_max=4, max_backlog_s=.25,
                               infer_clock_tolerance_s=.02, frame_period_s=.16,
                               request_timeout_s=120., clock=time.perf_counter,
                               sleep=time.sleep):
    """Run a strict H4/R4 episode, retaining the native session's video/events.

    Session must use fixed training timing; physical v/a retiming would change
    the action/prefix timestamps and is deliberately rejected. On terminal the
    owner evaluator drains the worker and camera before destroying the scene.
    """
    if worker.busy:
        raise RuntimeError("inference worker must be drained before an episode")
    if action_limit < 1 or execute_horizon != 4 or prefix_max != 4:
        raise ValueError("require positive action limit, H4 and trained prefix_max=4")
    if not math.isfinite(action_interval_s) or action_interval_s <= 0:
        raise ValueError("calibrated training action interval must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in
           (max_backlog_s, infer_clock_tolerance_s, frame_period_s, request_timeout_s)):
        raise ValueError("timing limits must be finite and positive")
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
                       "physics_runtime_timing": getattr(session, "runtime_timing_summary", lambda: {})(), **details}
        raise RealtimeViolation(message, diagnostics=diagnostics)

    def lag():
        nonlocal max_lag
        value = wall() - (physical() - phys0)
        max_lag = max(max_lag, value)
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
            wait = wall0 + physical() - phys0 + dt - clock()
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
            tracking.append(np.abs(np.asarray(session.current_qpos()) - q))
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
            # committed H4 budget. Observation time is already frozen; every
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
            reservation = None if grid0 is None else queue.reserve(completed, frozen)
        pending = {"physics_s": frozen, "wall_s": wall(), "for_inference": for_inference,
                   "delivered": False, "asynchronous_readback": True,
                   "action_origin": completed, "query_index": query if for_inference else None}
        session.begin_capture(for_inference=for_inference)
        pending["submit_duration_s"] = wall() - pending["wall_s"]
        next_head = frozen + frame_period_s

    def finish_capture():
        nonlocal pending, active, query
        record = pending
        packet = session.finish_capture()
        record.update(duration_s=wall() - record["wall_s"], delivered=True,
                      physics_ticks_during_capture=int(round((physical() - record["physics_s"]) / dt)))
        record["capture_stages"] = getattr(session, "last_capture_diagnostics", {})
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
        # The DOMINO packet is the raw head-over-two-wrists composite; letterbox
        # conversion belongs to the slow policy preprocessor, not this packet.
        if "frame" in packet:
            packet["fast_frame"] = np.array(packet["frame"], copy=True)
        packet["sim_time"] = record["physics_s"]
        packet.update(_sampling_seed=sampling_seed(task, episode_seed, query), _reset_policy=query == 0,
                      _rtc_prefix_actions=prefix, _rtc_prefix_length=len(prefix),
                      _rtc_observation_action_index=record["action_origin"],
                      _rtc_cache_age_actions=record["action_origin"] % 16,
                      _rtc_observation_physics_s=record["physics_s"],
                      _rtc_action_interval_s=action_interval_s)
        # Fire MotiionKV first, then drain render lag. smoke15 solo POB: model
        # finished in ~70ms while deadline-capped catch_up still ran ~683ms and
        # burned the H4 window, so the ready result sat unconsumed -> late suffix.
        # smoke09 immediate mid-drain early-install left illegal backlog under
        # 8-GPU contention. Compromise: poll during drain, but only stop early
        # when the result is ready AND residual lag is within max_backlog_s.
        # Otherwise keep the physics-deadline cap. No H4 widen.
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
                  "action_origin": record["action_origin"], "cache_age_actions": record["action_origin"] % 16,
                  "prefix_actions_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
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
        r.update(receive_wall_s=wall(), receive_physics_s=physical(), terminal_before_result=terminal)
        r["inference_wall_elapsed_s"] = r["receive_wall_s"] - r["submit_wall_s"]
        r["inference_physics_advance_s"] = r["receive_physics_s"] - r["submit_physics_s"]
        r["inference_clock_error_s"] = r["inference_wall_elapsed_s"] - r["inference_physics_advance_s"]
        r["inference_clock_check"] = "terminal_truncated" if terminal else "complete_interval"
        r["object_displacement_while_inference"] = (np.asarray(session.object_position()) - r.pop("object_at_submit")).tolist()
        if not terminal:
            if result["request_id"] != r["request_id"]:
                raise RuntimeError("out-of-order RTC inference response")
            if abs(r["inference_clock_error_s"]) > infer_clock_tolerance_s:
                fail("RTC inference clock mismatch", error_s=r["inference_clock_error_s"])
            raw = queue.validate(result["actions"])
            actions = guard_actions(raw)
            r.update(model_started_wall_s=float(result["started"]) - wall0,
                     model_finished_wall_s=float(result["finished"]) - wall0,
                     model_compute_wall_s=float(result["finished"] - result["started"]),
                     actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                     position_guard_clipped_coordinates=int(np.count_nonzero(actions != raw)),
                     policy_diagnostics=result.get("policy_timing", result.get("policy_diagnostics", result.get("metadata", {}))))
            if reservation is None:
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
                ready_clock_s = max(physical(), phys0 + wall())
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
            fail("RTC late suffix: no valid H4 result at committed boundary",
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
    duration, sim_duration = wall(), physical() - phys0
    feedback = np.asarray(tracking)
    capture_gaps = np.diff([r["physics_s"] for r in captures if r["delivered"]])
    return {"protocol": PROTOCOL, "controller": "committed_clean_prefix_rtc_fixed_grid_h4r4",
            "timing_mode": "nominal_training_timing", "limit_retiming": False,
            "clock": "actual_wall_clock", "virtual_delay": False, "startup_delay_excluded": False,
            "rtc_strategy": "trained_clean_prefix_conditioning", "rtc_prefix_max": prefix_max,
            "action_interval_s": action_interval_s, "execute_horizon": execute_horizon,
            "slow_refresh_actions": 16, "action_grid_start_physics_s": grid0,
            "action_limit": action_limit, "started_targets": started, "completed_targets": completed,
            "inference_calls": len(requests), "physics_time_s": sim_duration, "wall_time_s": duration,
            "realtime_factor": sim_duration / duration if duration else 1., "max_backlog_s": max_lag,
            "max_tick_gap_wall_s": max_gap,
            "max_inference_clock_error_s": max((abs(r.get("inference_clock_error_s", 0.)) for r in requests
                if r.get("inference_clock_check") == "complete_interval"), default=0.),
            "clock_limits": {"max_backlog_s": max_backlog_s, "inference_clock_error_s": infer_clock_tolerance_s},
            "phase_physics_seconds": {key: value * dt for key, value in phase_ticks.items()},
            "realtime_valid": True, "rows_dropped": 0, "guided_requests": 0,
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
