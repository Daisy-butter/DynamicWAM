"""Original DOMINO take_action/metrics/video, causal-spatial H2/R2 planning."""
from __future__ import annotations

import json
import time
import numpy as np

from dynamicwam.inference.native_causal_spatial_policy import (
    MODEL_OFFSETS, NATIVE_PROTOCOL, NativeCausalSpatialPolicy, NativeRTCPrefixQueue,
)
from dynamicwam.runtime.ciwam.execution import NativeStepper, query_timing_summary
from dynamicwam.runtime.domino_continuous.camera_stream import PhysicalTimeHeadHistory
from .composite import composite_robotwin_frame, extract_state, head_camera_frame


class NativeCausalSpatialDeployModel:
    def __init__(self, usr_args):
        config = usr_args.get("causal_spatial_deploy_config", usr_args.get("dynamicwam_deploy_config"))
        if not config or not usr_args.get("causal_spatial_checkpoint"):
            raise ValueError("native causal-spatial requires deployment config and explicit checkpoint")
        self.policy = NativeCausalSpatialPolicy(config, usr_args["causal_spatial_checkpoint"],
            device=usr_args.get("causal_spatial_device", "cuda:0"), allow_diagnostic=False,
            cuda_graph=bool(usr_args.get("causal_spatial_cuda_graph", True)))
        import torch
        from dynamicwam.inference.causal_spatial_policy import _current_cuda_identity
        self.gpu_identity = _current_cuda_identity(torch, self.policy.device.index or 0,
                                                  next(self.policy.model.parameters()).device)
        print("NATIVE-CAUSAL-MODEL " + json.dumps(dict(protocol=NATIVE_PROTOCOL,
            checkpoint_sha256=self.policy.checkpoint_sha256, gpu_identity=self.gpu_identity,
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda))), flush=True)
        self.flow = PhysicalTimeHeadHistory(self.policy.config["head_flow"],
            frame_period_s=.16, endpoint_tolerance_s=.08)
        self.sampling_seed = int(usr_args.get("policy_sampling_seed", 20261026))
        self.stepper = None
        self.queue = None
        self.query_ms = []
        self.memory_wait_count = 0
        self._closed = False

    def bind_env(self, env, observation):
        if self.stepper is not None:
            if self.stepper.env is not env:
                raise RuntimeError("reset_model is required before binding another native environment")
            return
        if int(env.take_action_cnt) != 0:
            raise RuntimeError("native H2 bootstrap must begin at action zero")
        self.policy.prepare_instruction(env.get_instruction())
        self.stepper = NativeStepper(env, action_interval_seconds=.16)
        self.queue = NativeRTCPrefixQueue(extract_state(observation))

    def finish_episode(self):
        drained = self.policy.reset_native_episode()
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            print(f"SYNC-EPISODE-SUMMARY {summary}", flush=True)
            query_timing = query_timing_summary(self.query_ms, summary["steps"])
            query_timing.update(execute_horizon=2, prefix_length=2, protocol=NATIVE_PROTOCOL,
                query_period_actions=2, action_rows_per_eval_callback=1)
            print(f"SYNC-QUERY-TIMING {query_timing}", flush=True)
            summary.update(protocol=NATIVE_PROTOCOL, H=2, R=2, K=10, chunk_size=16,
                bootstrap_hold_actions=min(2, summary["steps"]), memory_wait_count=self.memory_wait_count,
                timing=query_timing, slow_drain=drained)
            print("NATIVE-CAUSAL-EPISODE-SUMMARY " + json.dumps(summary), flush=True)
        self.stepper = None
        self.queue = None
        self.query_ms.clear()
        self.memory_wait_count = 0
        self.flow.reset()

    def close(self):
        if not self._closed:
            self.policy.close()
            self._closed = True


def get_model(usr_args):
    return NativeCausalSpatialDeployModel(usr_args)


def eval(TASK_ENV, model, observation):
    model.bind_env(TASK_ENV, observation)
    scene_clock = getattr(TASK_ENV, "_scene_step_clock", None)
    if scene_clock is None:
        raise RuntimeError("native causal-spatial requires the actual scene step clock")
    actual_physics = float(scene_clock.snapshot().time_seconds)
    origin = int(TASK_ENV.take_action_cnt)
    if origin != model.queue.origin:
        raise RuntimeError("native action count and committed queue diverged")
    # get_obs has already completed and the native caller does not step physics
    # while policy code runs. Wall exposure is not measured by this adapter.
    available_wall = time.perf_counter()
    model.flow.push(head_camera_frame(observation), simulator_time_seconds=actual_physics)
    motion = model.flow.observation()
    packet = dict(frame=composite_robotwin_frame(observation, model.policy.config["observation"]),
        state=extract_state(observation), flow_frames=motion.flow_rgb, motion_features=motion.motion_features,
        motion_interval_valid_mask=motion.interval_valid_mask,
        motion_acceleration_valid_mask=motion.acceleration_valid_mask, sim_time=actual_physics,
        _native_protocol=NATIVE_PROTOCOL, _native_action_origin=origin,
        _rtc_capture_drive_targets=extract_state(observation).astype(np.float64),
        _rtc_capture_wall_perf_s=available_wall, _rtc_capture_available_wall_perf_s=available_wall,
        _native_wall_clock_semantics="adapter_entry_availability_not_measured_camera_exposure")
    if origin % 2 == 0:
        prefix, previous, valid = model.queue.query()
        packet.update(_native_prefix_length=2, _native_prefix_actions=prefix,
            _native_model_target_offsets=MODEL_OFFSETS.copy(), _native_previous_actions=previous,
            _native_previous_valid=valid,
            _sampling_seed=model.sampling_seed+100000*int(getattr(TASK_ENV, "test_num", 0))+origin)
        started = time.perf_counter()
        actions = model.policy.sample(packet)
        model.query_ms.append(1000*(time.perf_counter()-started))
        model.queue.install(actions)
        timing = dict(model.policy.last_policy_timing)
        timing.update(physics_before_query_s=actual_physics,
            physics_after_query_s=float(scene_clock.snapshot().time_seconds),
            action_count_after_query=int(TASK_ENV.take_action_cnt),
            flow_diagnostics=model.flow.diagnostics,
            observation_wall_clock_semantics=packet["_native_wall_clock_semantics"])
        model.memory_wait_count += len(timing["memory_waits"])
        print("NATIVE-CAUSAL-QUERY " + json.dumps(timing), flush=True)
    else:
        model.policy.observe_native(packet)
    if float(scene_clock.snapshot().time_seconds) != actual_physics or int(TASK_ENV.take_action_cnt) != origin:
        raise RuntimeError("native inference/preprocessing advanced simulator physics or action budget")
    model.stepper.execute(model.queue.current())
    model.queue.acknowledge(TASK_ENV.take_action_cnt)


def reset_model(model):
    model.finish_episode()


def close_model(model):
    model.close()
