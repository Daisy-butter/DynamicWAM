"""DOMINO adapter with fast-slow replanning. Official Level-1 stays on sync."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ...execution import NativeStepper
from ...flow import HeadFlowBuffer
from ...wam.async_control import (
    ASYNC_POLICY_REGISTRATION_NAME,
    FastSlowScheduler,
    load_async_control_config,
    resolve_async_config_path,
)
from ...wam.policy import DynamicWAMPolicy
from .composite import composite_robotwin_frame, extract_state, head_camera_frame

logger = logging.getLogger(__name__)


class DynamicWAMAsyncDeployModel:
    def __init__(self, usr_args: Dict[str, Any]) -> None:
        project_root = usr_args["dynamicwam_root"]
        self.policy = DynamicWAMPolicy(
            usr_args["dynamicwam_deploy_config"],
            project_root=project_root,
        )
        self.policy.setup()
        config_path = resolve_async_config_path(usr_args, project_root=project_root)
        self.control = load_async_control_config(config_path)
        self.scheduler = FastSlowScheduler(
            self.control,
            chunk_size=int(self.policy.runtime.chunk_size),
        )
        self.flow = HeadFlowBuffer(self.policy.runtime.head_flow_config)
        self.stepper: Optional[NativeStepper] = None
        self._slow_forwards = 0
        self._fast_forwards = 0
        self._slow_latency_ms: list[float] = []
        self._fast_latency_ms: list[float] = []
        self._boundary_deltas: list[float] = []

    def bind_env(self, task_env) -> None:
        if self.stepper is None or self.stepper.env is not task_env:
            self.stepper = NativeStepper(
                task_env,
                action_interval_seconds=(self.policy.runtime.action_interval_seconds),
            )
            self.policy.set_instruction(task_env.get_instruction())

    def finish_episode(self) -> None:
        if self.stepper is not None:
            summary = self._episode_summary()
            logger.info("async-flow episode: %s", summary)
            print(
                f"SYNC-EPISODE-SUMMARY {self._sync_compatible_summary(summary)}",
                flush=True,
            )
            print(f"ASYNC-EPISODE-SUMMARY {summary}", flush=True)
        self.stepper = None
        self.flow.reset()
        self.policy.reset()
        self.scheduler.reset()
        self._slow_forwards = 0
        self._fast_forwards = 0
        self._slow_latency_ms.clear()
        self._fast_latency_ms.clear()
        self._boundary_deltas.clear()

    def _sync_compatible_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "steps": summary["steps"],
            "kinds": summary["kinds"],
            "overrun_steps": summary["overrun_steps"],
        }

    def _episode_summary(self) -> Dict[str, Any]:
        stepper_summary = (
            self.stepper.telemetry.summary()
            if self.stepper is not None
            else {"steps": 0, "kinds": {}, "overrun_steps": 0}
        )
        return {
            **stepper_summary,
            "policy": ASYNC_POLICY_REGISTRATION_NAME,
            "open_loop_horizon": self.control.fast_horizon,
            "max_world_age_steps": self.control.max_world_age_steps,
            "inference_delay": self.control.inference_delay,
            "rtc_mode": self.control.rtc_mode,
            "slow_forwards": self._slow_forwards,
            "fast_forwards": self._fast_forwards,
            "mean_slow_ms": _mean(self._slow_latency_ms),
            "mean_fast_ms": _mean(self._fast_latency_ms),
            "max_boundary_delta": (
                max(self._boundary_deltas) if self._boundary_deltas else 0.0
            ),
        }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def get_model(usr_args: Dict[str, Any]) -> DynamicWAMAsyncDeployModel:
    return DynamicWAMAsyncDeployModel(usr_args)


def eval(TASK_ENV, model: DynamicWAMAsyncDeployModel, observation: Dict[str, Any]) -> None:
    model.bind_env(TASK_ENV)
    frame = composite_robotwin_frame(
        observation,
        model.policy.runtime.observation_config,
    )
    scene_clock = getattr(TASK_ENV, "_scene_step_clock", None)
    if scene_clock is None:
        raise RuntimeError(
            "absolute-motion deployment requires TASK_ENV._scene_step_clock"
        )
    simulator_time = float(scene_clock.snapshot().time_seconds)
    model.flow.push(
        head_camera_frame(observation),
        simulator_time_seconds=simulator_time,
    )
    if model.scheduler.needs_replan():
        motion = model.flow.observation()
        packet = {
            "frame": frame,
            "state": extract_state(observation),
            "flow_frames": motion.flow_rgb,
            "motion_features": motion.motion_features,
            "motion_interval_valid_mask": motion.interval_valid_mask,
            "motion_acceleration_valid_mask": (motion.acceleration_valid_mask),
        }
        leftover = model.scheduler.leftover()
        path = model.scheduler.choose_path(packet["motion_features"])
        started = time.perf_counter()
        if path == "slow":
            actions, video_cache = model.policy.sample_slow(
                packet,
                leftover=leftover,
                inference_delay=model.control.inference_delay,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            model._slow_forwards += 1
            model._slow_latency_ms.append(latency_ms)
            delta = model.scheduler.commit_chunk(
                actions,
                path="slow",
                leftover=leftover,
                video_cache=video_cache,
                simulator_time=simulator_time,
                motion_features=packet["motion_features"],
                motion_interval_valid_mask=packet["motion_interval_valid_mask"],
                motion_acceleration_valid_mask=packet[
                    "motion_acceleration_valid_mask"
                ],
            )
        else:
            video_cache = model.scheduler.bridge.video_cache
            frozen = model.scheduler.bridge.frozen_motion()
            if frozen is None:
                raise RuntimeError(
                    "fast path requires a frozen motion snapshot aligned with the "
                    "video cache"
                )
            packet.update(frozen)
            actions = model.policy.sample_fast(
                packet,
                video_cache,
                leftover=leftover,
                inference_delay=model.control.inference_delay,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            model._fast_forwards += 1
            model._fast_latency_ms.append(latency_ms)
            delta = model.scheduler.commit_chunk(
                actions,
                path="fast",
                leftover=leftover,
            )
        model._boundary_deltas.append(delta)
    if model.stepper is None:
        raise RuntimeError("DOMINO environment was not bound before execution")
    model.stepper.execute(model.scheduler.pop_action())


def reset_model(model: DynamicWAMAsyncDeployModel) -> None:
    model.finish_episode()
