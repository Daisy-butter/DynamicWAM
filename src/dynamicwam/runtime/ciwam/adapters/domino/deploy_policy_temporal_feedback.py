"""Native DOMINO execution: history is previous command, not previous query."""
from __future__ import annotations

from collections import deque
import math
import time
from ...execution import NativeStepper, query_timing_summary
from ...flow import HeadFlowBuffer
from ...wam.temporal_feedback_policy import TemporalFeedbackPolicy
from .composite import composite_robotwin_frame, extract_state, head_camera_frame
from .deploy_policy_multirate_flow import fast_composite_from_observation


class TemporalFeedbackDeployModel:
    def __init__(self, args):
        self.policy = TemporalFeedbackPolicy(args['dynamicwam_deploy_config'], project_root=args['dynamicwam_root'],
            checkpoint=args.get('temporal_feedback_checkpoint'),
            feedback_enabled=not args.get('temporal_feedback_disabled', False),
            history_enabled=not args.get('temporal_history_disabled', False),
            graph_enabled=bool(args.get('plan_graph_enabled', True)), execute_horizon=int(args.get('execute_horizon', 1)))
        self.policy.setup()
        self.flow = HeadFlowBuffer(self.policy.runtime.head_flow_config)
        self.pending = deque()
        self.stepper = None
        self.query_ms = []
        self.previous_wall = self.previous_sim = None
        self.previous_head = self.previous_head_time = None
        self.index = 0

    def finish(self):
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            print(f'SYNC-EPISODE-SUMMARY {summary}', flush=True)
            timing = query_timing_summary(self.query_ms, int(summary['steps']))
            timing.update(execute_horizon=self.policy.execute_horizon, checkpoint=self.policy.checkpoint_path,
                base_checkpoint_sha256=self.policy.base_sha256, feedback_enabled=self.policy.feedback_enabled,
                history_enabled=self.policy.history_enabled)
            print(f'TEMPORAL-FEEDBACK-TIMING {timing}', flush=True)
        self.pending.clear()
        self.query_ms.clear()
        self.flow.reset()
        self.policy.reset()
        self.stepper = None
        self.previous_wall = self.previous_sim = None
        self.previous_head = self.previous_head_time = None
        self.index = 0


def get_model(usr_args):
    return TemporalFeedbackDeployModel(usr_args)


def reset_model(model):
    model.finish()


def eval(TASK_ENV, model, observation):
    entered = time.perf_counter()
    if model.stepper is None:
        model.stepper = NativeStepper(TASK_ENV, action_interval_seconds=model.policy.runtime.action_interval_seconds)
        model.policy.set_instruction(TASK_ENV.get_instruction())
    clock = getattr(TASK_ENV, '_scene_step_clock', None)
    if clock is None:
        raise RuntimeError('temporal feedback requires the simulator clock')
    sim = float(clock.snapshot().time_seconds)
    if not math.isfinite(sim):
        raise ValueError('nonfinite simulator clock')
    head = head_camera_frame(observation)
    model.flow.push(head, simulator_time_seconds=sim)
    if not model.pending:
        config = model.policy.runtime.observation_config
        refresh = model.policy.plan is None or model.policy.age == 16
        fast_frame = fast_composite_from_observation(observation, config)
        dt = 0. if model.previous_head_time is None else sim - model.previous_head_time
        if dt < 0:
            raise RuntimeError('simulator clock moved backwards')
        packet = dict(frame=composite_robotwin_frame(observation, config) if refresh else fast_frame,
            fast_frame=fast_frame, state=extract_state(observation), action_index=model.index, sim_time=sim,
            previous_head=head if model.previous_head is None else model.previous_head,
            history_dt=dt, history_valid=model.previous_head is not None and dt > 0)
        if refresh:
            motion = model.flow.observation()
            packet.update(flow_frames=motion.flow_rgb, motion_features=motion.motion_features,
                motion_interval_valid_mask=motion.interval_valid_mask,
                motion_acceleration_valid_mask=motion.acceleration_valid_mask)
        start = time.perf_counter()
        actions = model.policy.query(packet)
        returned = time.perf_counter()
        model.query_ms.append((returned-start)*1000)
        detail = dict(model.policy.last_trace, query_ms=model.query_ms[-1],
            adapter_preprocess_ms=(start-entered)*1000, observation_to_action_ms=(returned-entered)*1000,
            decision_interval_wall_ms=None if model.previous_wall is None else (entered-model.previous_wall)*1000,
            decision_interval_sim_ms=None if model.previous_sim is None else (sim-model.previous_sim)*1000)
        print(f'TEMPORAL-FEEDBACK-QUERY {detail}', flush=True)
        model.previous_wall, model.previous_sim = entered, sim
        model.pending.extend(actions)
    # Save before execution; observation buffers may be reused/mutated by render.
    # This runs for every action, including the three nonquery entries under H4.
    model.previous_head, model.previous_head_time = head.copy(), sim
    model.stepper.execute(model.pending.popleft())
    model.index += 1
