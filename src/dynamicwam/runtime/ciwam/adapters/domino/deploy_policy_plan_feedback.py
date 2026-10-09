"""Native synchronous DOMINO Level-1 execution for plan feedback (not RTC)."""
from __future__ import annotations
from collections import deque
import time
from ...execution import NativeStepper, query_timing_summary
from ...flow import HeadFlowBuffer
from ...wam.plan_feedback_policy import PlanFeedbackPolicy
from .composite import composite_robotwin_frame, extract_state, head_camera_frame
from .deploy_policy_multirate_flow import fast_composite_from_observation


class PlanFeedbackDeployModel:
    def __init__(self, args):
        self.policy = PlanFeedbackPolicy(args['dynamicwam_deploy_config'], project_root=args['dynamicwam_root'],
            checkpoint=args.get('plan_feedback_checkpoint'), feedback_enabled=not args.get('plan_feedback_disabled', False),
            graph_enabled=bool(args.get('plan_graph_enabled', True)))
        self.policy.setup()
        self.flow = HeadFlowBuffer(self.policy.runtime.head_flow_config)
        self.pending = deque()
        self.stepper = None
        self.query_ms = []
        self.previous_wall = self.previous_sim = None
        self.index = 0

    def finish(self):
        if self.stepper is not None:
            summary = self.stepper.telemetry.summary()
            print(f'SYNC-EPISODE-SUMMARY {summary}', flush=True)
            timing = query_timing_summary(self.query_ms, int(summary['steps']))
            timing.update(execute_horizon=4, checkpoint=self.policy.checkpoint_path,
                base_checkpoint_sha256=self.policy.base_sha256, feedback_enabled=self.policy.feedback_enabled)
            print(f'PLAN-FEEDBACK-TIMING {timing}', flush=True)
        self.pending.clear()
        self.query_ms.clear()
        self.flow.reset()
        self.policy.reset()
        self.stepper = None
        self.previous_wall = self.previous_sim = None
        self.index = 0


def get_model(usr_args):
    return PlanFeedbackDeployModel(usr_args)


def reset_model(model):
    model.finish()


def eval(TASK_ENV, model, observation):
    entered = time.perf_counter()
    if model.stepper is None:
        model.stepper = NativeStepper(TASK_ENV, action_interval_seconds=model.policy.runtime.action_interval_seconds)
        model.policy.set_instruction(TASK_ENV.get_instruction())
    clock = getattr(TASK_ENV, '_scene_step_clock', None)
    if clock is None:
        raise RuntimeError('plan feedback requires the simulator clock')
    sim = float(clock.snapshot().time_seconds)
    model.flow.push(head_camera_frame(observation), simulator_time_seconds=sim)
    if not model.pending:
        config = model.policy.runtime.observation_config
        refresh = model.policy.plan is None or model.policy.age == 16
        fast_frame = fast_composite_from_observation(observation, config)
        packet = dict(frame=composite_robotwin_frame(observation, config) if refresh else fast_frame,
            fast_frame=fast_frame, state=extract_state(observation),
            action_index=model.index, sim_time=sim)
        # Retain every frame in history, but compute optical flow only when the
        # unchanged native planner consumes it. Fast visual feedback needs RGB.
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
        print(f'PLAN-FEEDBACK-QUERY {detail}', flush=True)
        model.previous_wall, model.previous_sim = entered, sim
        model.pending.extend(actions)
    model.stepper.execute(model.pending.popleft())
    model.index += 1
