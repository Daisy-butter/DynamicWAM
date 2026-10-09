"""DOMINO adapter for a fixed execute horizon and bridge refresh ratio.

This is a separate entry from the synchronous policy and from the RTC sidecar.
Controlled runs do not refresh the bridge because motion drifted. Each callback
executes one action, and a new query starts only after H_exec completed steps.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from ...execution import NativeStepper, query_timing_summary
from ...flow import HeadFlowBuffer
from ...wam.multirate_control import ControlledMultirateController
from ...wam.multirate_policy import MultiratePolicy
from .composite import composite_robotwin_frame, extract_state, head_camera_frame

logger = logging.getLogger(__name__)


def fast_composite_from_observation(
    observation: dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    """Head over two wrists. Wrists are area-downsampled before the fast encoder."""

    obs = observation["observation"]
    head_size = tuple(int(value) for value in config["head_size"])
    wrist_size = tuple(int(value) for value in config["wrist_size"])
    head = np.asarray(obs["head_camera"]["rgb"])
    if head.shape[:2] != head_size:
        raise ValueError(
            f"DOMINO head camera shape differs from DynamicWAM: "
            f"expected {head_size}, got {head.shape[:2]}"
        )
    if head.dtype != np.uint8:
        head = np.clip(head, 0, 255).astype(np.uint8)
    left = _resize_area(np.asarray(obs["left_camera"]["rgb"]), wrist_size)
    right = _resize_area(np.asarray(obs["right_camera"]["rgb"]), wrist_size)
    bottom = np.concatenate([left, right], axis=1)
    frame = np.concatenate([head, bottom], axis=0)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resize_area(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


class DynamicWAMMultirateDeployModel:
    def __init__(self, usr_args: dict[str, Any]) -> None:
        checkpoint = usr_args.get("multirate_checkpoint")
        if not checkpoint:
            raise ValueError("multirate DOMINO eval requires usr_args['multirate_checkpoint']")
        lead = int(usr_args.get("bridge_lead_actions", 0))
        scheduling = str(usr_args.get("multirate_scheduling", "controlled_ratio"))
        # A mid-chunk snapshot has to run on the world worker. The controlled
        # ratio path builds the bridge inside the waited query.
        if lead > 0 and scheduling == "controlled_ratio":
            scheduling = "independent_worker"
        self.policy = MultiratePolicy(
            usr_args["dynamicwam_deploy_config"],
            str(checkpoint),
            project_root=usr_args["dynamicwam_root"],
            fast_conditioner=usr_args.get("fast_conditioner"),
            reuse_action_noise=bool(usr_args.get("multirate_reuse_action_noise", False)),
            scheduling=scheduling,
            bridge_wait_budget_ms=float(usr_args.get("bridge_wait_budget_ms", 150.0)),
        )
        self.policy.setup()
        compact = self.policy.fast_model.fast_policy
        if compact is not None and compact.config.feedback_enabled:
            compact.feedback_disabled = bool(usr_args.get('multirate_disable_feedback',False))
        self.flow = HeadFlowBuffer(self.policy.runtime.head_flow_config)
        self.controller = ControlledMultirateController(
            execute_horizon=int(usr_args.get("execute_horizon", 4)),
            every_fast_queries=int(usr_args.get("bridge_refresh_ratio", 4)),
            chunk_size=int(self.policy.runtime.chunk_size),
            bridge_lead_actions=lead,
        )
        self.pace_action_interval = bool(usr_args.get("pace_action_interval", False))
        self.stepper: NativeStepper | None = None
        self.query_ms: list[float] = []
        self.query_includes_bridge: list[bool] = []
        self.query_traces: list[dict] = []

    def bind_env(self, task_env) -> None:
        if self.stepper is None or self.stepper.env is not task_env:
            self.stepper = NativeStepper(
                task_env,
                action_interval_seconds=self.policy.runtime.action_interval_seconds,
            )
            self.policy.set_instruction(task_env.get_instruction())

    def finish_episode(self) -> None:
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            logger.info(
                "multirate episode: %s slow_builds=%d fast_queries=%d",
                summary,
                self.controller.slow_builds,
                self.controller.fast_queries,
            )
            print(f"SYNC-EPISODE-SUMMARY {summary}", flush=True)
            timing = query_timing_summary(self.query_ms, int(summary["steps"]))
            bridge_ms = [
                value
                for value, includes_bridge in zip(self.query_ms, self.query_includes_bridge)
                if includes_bridge
            ]
            fast_ms = [
                value
                for value, includes_bridge in zip(self.query_ms, self.query_includes_bridge)
                if not includes_bridge
            ]
            timing.update(
                slow_builds=self.controller.slow_builds,
                fast_queries=self.controller.fast_queries,
                actual_slow_builds=self.policy.actual_slow_builds,
                blocking_refreshes=self.policy.blocking_refreshes,
                over_age_queries=self.policy.over_age_queries,
                scheduling=self.policy.scheduling,
                execute_horizon=self.controller.execute_horizon,
                refresh_ratio=self.controller.every_fast_queries,
                bridge_lead_actions=self.controller.bridge_lead_actions,
                bridge_wait_budget_ms=self.policy.bridge_wait_budget_ms,
                pace_action_interval=self.pace_action_interval,
                inline_bridge_queries=sum(t["bridge_mode"].startswith("inline_") for t in self.query_traces),
                waited_bridge_queries=sum(t["bridge_mode"].startswith("wait_") for t in self.query_traces),
                world_wait_ms_sum=sum(t["world_wait_ms"] for t in self.query_traces),
                checkpoint=self.policy.checkpoint_path,
                reuse_action_noise=self.policy.reuse_action_noise,
                bridge_query_count=len(bridge_ms),
                bridge_query_ms_mean=(sum(bridge_ms) / len(bridge_ms)) if bridge_ms else None,
                fast_only_query_count=len(fast_ms),
                fast_only_query_ms_mean=(sum(fast_ms) / len(fast_ms)) if fast_ms else None,
            )
            print(f"MULTIRATE-QUERY-TIMING {timing}", flush=True)
        self.query_ms.clear()
        self.query_includes_bridge.clear()
        self.query_traces.clear()
        self.stepper = None
        self.flow.reset()
        self.controller.reset()
        self.policy.reset()


def get_model(usr_args: dict[str, Any]) -> DynamicWAMMultirateDeployModel:
    return DynamicWAMMultirateDeployModel(usr_args)


def eval(
    TASK_ENV,
    model: DynamicWAMMultirateDeployModel,
    observation: dict[str, Any],
) -> None:
    model.bind_env(TASK_ENV)
    observation_config = model.policy.runtime.observation_config
    frame = composite_robotwin_frame(observation, observation_config)
    fast_frame = fast_composite_from_observation(observation, observation_config)
    scene_clock = getattr(TASK_ENV, "_scene_step_clock", None)
    if scene_clock is None:
        raise RuntimeError("multirate deployment requires TASK_ENV._scene_step_clock")
    simulator_time = float(scene_clock.snapshot().time_seconds)
    model.flow.push(
        head_camera_frame(observation),
        simulator_time_seconds=simulator_time,
    )
    step_started = time.perf_counter()
    packet = _observation_packet(
        model,
        frame,
        fast_frame,
        observation,
        simulator_time,
    )
    if model.controller.needs_bridge_snapshot():
        model.policy.submit_bridge_snapshot(packet, model.controller)
    did_query = False
    if model.controller.needs_query():
        did_query = True
        started = time.perf_counter()
        actions = model.policy.query(packet, model.controller)
        model.query_ms.append((time.perf_counter() - started) * 1000.0)
        trace = dict(model.policy.last_query_trace, query_ms=model.query_ms[-1])
        model.query_traces.append(trace)
        # Classification follows actual work/wait in this query. The controlled
        # R clock cannot classify an independent worker or an over-age refresh.
        model.query_includes_bridge.append(trace["bridge_mode"] != "reuse")
        print(f"MULTIRATE-QUERY-DETAIL {trace}", flush=True)
        model.controller.commit(actions)
        logger.info(
            "multirate query slow_builds=%d fast_queries=%d",
            model.controller.slow_builds,
            model.controller.fast_queries,
        )
    if model.stepper is None:
        raise RuntimeError("DOMINO environment was not bound before execution")
    model.stepper.execute(model.controller.take_action())
    model.controller.mark_executed()
    # Pace only the steps that are not already waiting on a fast sample, so the
    # four actions before a same-cadence decision last about 4 * action_interval.
    if model.pace_action_interval and not did_query:
        remaining = model.stepper.action_interval_s - (time.perf_counter() - step_started)
        if remaining > 0.0:
            time.sleep(remaining)


def _observation_packet(
    model: DynamicWAMMultirateDeployModel,
    frame: np.ndarray,
    fast_frame: np.ndarray,
    observation: dict[str, Any],
    simulator_time: float,
) -> dict[str, Any]:
    motion = model.flow.observation()
    return {
        "frame": frame,
        "fast_frame": fast_frame,
        "state": extract_state(observation),
        "flow_frames": motion.flow_rgb,
        "motion_features": motion.motion_features,
        "motion_interval_valid_mask": motion.interval_valid_mask,
        "motion_acceleration_valid_mask": motion.acceleration_valid_mask,
        "sim_time": simulator_time,
    }


def reset_model(model: DynamicWAMMultirateDeployModel) -> None:
    model.finish_episode()
