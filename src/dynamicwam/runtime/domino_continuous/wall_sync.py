"""Original full-chunk synchronous policy on a continuously paced world.

No simulated inference latency is accepted by this interface. Inference runs
in the actual model process while the physics owner keeps stepping. Camera
and TOPP overhead stay on the same wall clock; overdue steps retain the last
command before a newly available command may be installed.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

import numpy as np


class RealtimeViolation(RuntimeError):
    """Infrastructure could not honor the clock; never a policy failure."""

    def __init__(self, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def sampling_seed(task: str, episode_seed: int, query: int) -> int:
    return int.from_bytes(hashlib.sha256(f"wall-sync-v1:{task}:{episode_seed}:{query}".encode()).digest()[:4], "little")


def run_wall_sync_episode(
    session: Any,
    worker: Any,
    *,
    task: str,
    episode_seed: int,
    action_limit: int,
    sink: Callable[[dict], None] = lambda record: None,
    max_backlog_s: float = 0.25,
    infer_clock_tolerance_s: float = 0.02,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """One actual-time rollout, including the first real policy query.

    Model setup/dummy warmup and camera shader warmup belong before this
    function. Original action budget and per-target history/metric cadence
    are preserved; a final sample is added for a terminal inference hold.
    ``clock``/``sleep`` dependencies are only for CPU scheduling tests.
    """
    if action_limit < 1 or max_backlog_s <= 0 or infer_clock_tolerance_s <= 0:
        raise ValueError("positive action limit and clock tolerances are required")
    dt = float(session.timestep())
    if not 0 < dt < 1:
        raise ValueError("invalid physics timestep")
    if worker.busy:
        raise RuntimeError("worker must be drained before starting an episode")
    physics_origin = float(session.physics_time())
    origin = clock()
    initial_position = np.asarray(session.object_position()).copy()
    last_tick_wall = origin
    max_gap = max_lag = 0.0
    lag_samples: list[float] = []
    requests: list[dict] = []
    phase_ticks = {"capture": 0, "inference": 0, "planning": 0, "execution": 0, "logging": 0}
    completed_targets = started_targets = 0
    query = 0
    reason: str | None = None
    current_phase = "starting"
    captures: list[dict] = []

    def elapsed() -> float:
        return clock() - origin

    def physical() -> float:
        return float(session.physics_time()) - physics_origin

    def backlog() -> float:
        nonlocal max_lag
        lag = elapsed() - physical()
        max_lag = max(max_lag, lag)
        if lag > max_backlog_s:
            raise RealtimeViolation(
                f"wall/physics backlog {lag:.6f}s exceeds {max_backlog_s:.6f}s "
                f"at physics={physical():.6f}s; no clock reset or virtual replay allowed",
                diagnostics={"phase": current_phase, "physics_s": physical(),
                             "wall_s": elapsed(), "started_targets": started_targets,
                             "query_index": query, "last_capture": captures[-1] if captures else None},
            )
        return lag

    def terminal() -> str | None:
        return session.stop_reason()

    def tick(segment: Any, phase: str, paced: bool = True) -> None:
        nonlocal last_tick_wall, max_gap, current_phase
        current_phase = phase
        if paced:
            deadline = origin + physical() + dt
            wait = deadline - clock()
            if wait > 0:
                sleep(wait)
        backlog()
        session.step(segment)
        now = clock()
        max_gap = max(max_gap, now - last_tick_wall)
        last_tick_wall = now
        phase_ticks[phase] += 1
        lag_samples.append(max(0.0, backlog()))

    def catch_up_hold(phase: str) -> None:
        nonlocal current_phase
        current_phase = phase
        # The camera and TOPP APIs block. Advance their already elapsed time
        # under the previous drive target, never using the future new command.
        # This is only bounded fixed-step scheduling jitter. The full model
        # inference runs concurrently with physics in the loop below.
        while terminal() is None and backlog() + 1e-9 >= dt:
            tick(None, phase, paced=False)

    def capture(*, for_inference: bool = True) -> tuple[dict | None, float, float, float]:
        nonlocal current_phase
        current_phase = "capture"
        obs_phys, obs_wall = float(session.physics_time()), elapsed()
        capture_ticks_before = phase_ticks["capture"]
        paced_camera_ticks = 0
        # A submitted render snapshot is immutable while its Color buffers
        # are read by a GIL-releasing native worker. Only the owner steps
        # physics; no new render submission is allowed until readback drains.
        asynchronous = hasattr(session, "begin_capture")
        packet = None
        delivered = False
        if asynchronous:
            session.begin_capture(for_inference=for_inference)
            capture_started = clock()
            while terminal() is None:
                catch_up_hold("capture")
                if terminal() is not None:
                    break
                if session.capture_ready():
                    packet = session.finish_capture()
                    delivered = True
                    break
                if clock() - capture_started > 120:
                    raise RealtimeViolation("camera readback exceeded 120 seconds")
                tick(None, "capture")
                paced_camera_ticks += 1
        elif for_inference:
            packet = session.capture_packet()
            delivered = True
        else:
            session.capture_history()
            delivered = True
        cost = elapsed() - obs_wall
        captures.append({"physics_s": obs_phys, "wall_s": obs_wall,
                         "duration_s": cost, "for_inference": for_inference,
                         "started_targets": started_targets, "delivered": delivered,
                         "asynchronous_readback": asynchronous,
                         "paced_ticks_while_camera_pending": paced_camera_ticks,
                         "physics_ticks_during_capture": phase_ticks["capture"] - capture_ticks_before})
        catch_up_hold("capture")
        return packet, obs_phys, obs_wall, cost

    while reason is None and started_targets < action_limit:
        packet, obs_phys, obs_wall, capture_wall = capture()
        if terminal() is not None:
            session.record_target_metrics()
            reason = terminal()
            break
        assert packet is not None
        packet["_sampling_seed"] = sampling_seed(task, episode_seed, query)
        packet["_reset_policy"] = query == 0
        submit_phys = float(session.physics_time())
        submit_wall = elapsed()
        request_id = worker.submit(packet)
        record = {
            "query_index": query, "request_id": request_id,
            "sampling_seed": packet["_sampling_seed"],
            "observation_physics_s": obs_phys, "observation_wall_s": obs_wall,
            "capture_wall_s": capture_wall,
            "submit_physics_s": submit_phys, "submit_wall_s": submit_wall,
            "initial_query": query == 0, "startup_delay_excluded": False,
            "rows_dropped": 0, "guidance": False, "prefetch": False,
            "started_rows": [], "completed_rows": [], "targets": [],
            "physics_ticks_while_inference_outstanding": 0,
            "installed": False,
        }
        object_at_submit = np.asarray(session.object_position()).copy()
        inference_ticks_before = phase_ticks["inference"]
        result = None
        while terminal() is None:
            # Results are not used in past physics slots even when a poll
            # observes a result queued while a bounded scheduling lag clears.
            catch_up_hold("inference")
            if terminal() is not None:
                break
            result = worker.poll()
            if result is not None:
                break
            if elapsed() - submit_wall > 120:
                raise RealtimeViolation("actual inference request exceeded 120 seconds")
            tick(None, "inference")
        record["physics_ticks_while_inference_outstanding"] = phase_ticks["inference"] - inference_ticks_before
        record["receive_wall_s"] = elapsed()
        record["receive_physics_s"] = float(session.physics_time())
        record["inference_wall_elapsed_s"] = record["receive_wall_s"] - submit_wall
        record["inference_physics_advance_s"] = record["receive_physics_s"] - submit_phys
        record["inference_clock_error_s"] = (
            record["inference_wall_elapsed_s"] - record["inference_physics_advance_s"]
        )
        record["object_displacement_while_inference"] = (
            np.asarray(session.object_position()) - object_at_submit
        ).tolist()
        record["terminal_before_result"] = terminal() is not None
        if abs(record["inference_clock_error_s"]) > infer_clock_tolerance_s:
            raise RealtimeViolation(f"inference clock mismatch: {record['inference_clock_error_s']:.6f}s")
        if terminal() is not None:
            session.record_target_metrics()
            reason = terminal()
        else:
            assert result is not None
            if result["request_id"] != request_id:
                raise RuntimeError("out-of-order inference result")
            actions = np.asarray(result["actions"], dtype=np.float64)
            if actions.shape != (16, 14) or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid official full chunk: {actions.shape}")
            record["actions_sha256"] = hashlib.sha256(actions.tobytes()).hexdigest()
            record["model_started_wall_s"] = float(result["started"]) - origin
            record["model_finished_wall_s"] = float(result["finished"]) - origin
            record["model_compute_wall_s"] = float(result["finished"] - result["started"])
            record["installed"] = True
            for row, action in enumerate(actions):
                if terminal() is not None or started_targets >= action_limit:
                    break
                if worker.busy:
                    raise RuntimeError("baseline must not infer during chunk execution")
                # Row zero keeps the query snapshot; other rows update motion
                # history exactly once before native target execution.
                if row:
                    capture(for_inference=False)
                if terminal() is not None:
                    session.record_target_metrics()
                    break
                plan_wall = elapsed()
                segment = session.plan(action)
                plan_cost = elapsed() - plan_wall
                catch_up_hold("planning")
                if terminal() is not None:
                    session.record_target_metrics()
                    break
                session.begin_native_action()
                started_targets += 1
                record["started_rows"].append(row)
                target = {"row": row, "planning_wall_s": plan_cost,
                          "start_physics_s": float(session.physics_time()), "start_wall_s": elapsed()}
                while not segment.finished and terminal() is None:
                    tick(segment, "execution")
                session.record_target_metrics()
                target.update(end_physics_s=float(session.physics_time()), end_wall_s=elapsed(),
                              completed=bool(segment.finished))
                record["targets"].append(target)
                if segment.finished:
                    completed_targets += 1
                    record["completed_rows"].append(row)
            reason = terminal()
            if reason is None and started_targets >= action_limit:
                reason = "action_budget_exhausted"
        record["end_physics_s"] = float(session.physics_time())
        record["end_wall_s"] = elapsed()
        record["stop_reason"] = reason
        if reason is None and record["completed_rows"] != list(range(16)):
            raise RuntimeError("non-terminal chunk did not execute all 16 rows")
        requests.append(record)
        sink(record)
        if reason is None:
            catch_up_hold("logging")
            if terminal() is not None:
                session.record_target_metrics()
                reason = terminal()
        query += 1
    wall = elapsed()
    physics_s = physical()
    errors = [abs(r["inference_clock_error_s"]) for r in requests]
    return {
        "protocol": "continuous_world_actual_latency_sync16_native_topp_v1",
        "controller": "native_per_target_topp", "clock": "actual_wall_clock",
        "virtual_delay": False, "startup_delay_excluded": False,
        "action_limit": action_limit, "started_targets": started_targets,
        "completed_targets": completed_targets, "inference_calls": len(requests),
        "physics_time_s": physics_s, "wall_time_s": wall,
        "realtime_factor": physics_s / wall if wall else 1.0,
        "max_backlog_s": max_lag, "max_tick_gap_wall_s": max_gap,
        "p95_tick_lag_s": float(np.percentile(lag_samples, 95)) if lag_samples else 0.0,
        "max_inference_clock_error_s": max(errors, default=0.0),
        "clock_limits": {"max_backlog_s": max_backlog_s, "inference_clock_error_s": infer_clock_tolerance_s},
        "phase_physics_seconds": {k: v * dt for k, v in phase_ticks.items()},
        "realtime_valid": True,
        "rows_dropped": 0, "guided_requests": 0, "execution_inference_overlap": False,
        "asynchronous_camera_readback": hasattr(session, "begin_capture"),
        "requests": requests,
        "captures": captures,
        "object_displacement": (np.asarray(session.object_position()) - initial_position).tolist(),
        "stop_reason": reason, "success": reason == "success",
    }
