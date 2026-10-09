"""Full native WAM plan plus causal, time-aware bounded arm feedback."""
from __future__ import annotations

import math
import numpy as np
import torch

from dynamicwam.inference.feedback_graph import FeedbackGraph
from dynamicwam.inference.native_plan_graph import NativePlanSampler
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise, pad_text_embeddings
from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
from dynamicwam.models.temporal_feedback_contract import FORMAT, VERSION, runtime_contract
from dynamicwam.models import onpolicy_feedback_contract as onpolicy_contract
from dynamicwam.observations.multiview import fast_views_from_composite, letterbox_view, imagenet_normalize
from .policy import DynamicWAMPolicy
from .plan_feedback_policy import file_sha256


def fixed_feedback_text(text, max_tokens=512):
    """Keep one graph shape across instructions; padding remains masked.

    WAM's original text path is unchanged. This affects only the small head,
    whose graph would otherwise capture again for each instruction length.
    """
    embeddings, mask = pad_text_embeddings(text)
    padding = max_tokens - embeddings.shape[1]
    if padding < 0:
        raise ValueError('instruction exceeds the WAN 512-token feedback budget')
    return (torch.nn.functional.pad(embeddings, (0, 0, 0, padding)),
            torch.nn.functional.pad(mask, (0, padding), value=False))


class TemporalFeedbackPolicy:
    def __init__(self, config, *, project_root, checkpoint=None, feedback_enabled=True,
                 graph_enabled=True, execute_horizon=1, history_enabled=True):
        if execute_horizon not in (1, 2, 4):
            raise ValueError('temporal feedback execute_horizon must be 1, 2 or 4')
        self.base = DynamicWAMPolicy(config, project_root=project_root)
        self.checkpoint_path = checkpoint
        self.feedback_enabled = bool(feedback_enabled)
        self.history_enabled = bool(history_enabled)
        self.graph_enabled = bool(graph_enabled)
        self.execute_horizon = execute_horizon
        self.supported_horizons = (1, 2, 4)
        self.reset()

    @property
    def runtime(self):
        return self.base.runtime

    def setup(self):
        from dynamicwam.inference.model_loader import load_deploy_config
        from dynamicwam.action_normalization import RoboTwinQposNormalizer
        self.base.setup()
        runner = self.base.runner
        cfg = load_deploy_config(self.base.deploy_config_path)
        schedule = BridgeSchedule.from_flow_match(shift=self.runtime.flow_match_shift,
            sigma_min=self.runtime.flow_match_sigma_min, extra_one_step=self.runtime.flow_match_extra_one_step)
        self.base_sha256 = file_sha256(cfg['checkpoint_path'])
        payload = None
        if self.checkpoint_path:
            payload = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            identity = (payload.get('format'), payload.get('version'))
            if identity not in ((FORMAT, VERSION), (onpolicy_contract.FORMAT, onpolicy_contract.VERSION)):
                raise ValueError('expected a temporal plan-feedback export')
            if payload['base_checkpoint']['sha256'] != self.base_sha256:
                raise ValueError('temporal feedback was trained against another WAM checkpoint')
            normal = payload['action_normalization']
            saved = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
            for name in ('_mean', '_std'):
                if not torch.equal(getattr(saved, name), getattr(self.runtime.action_normalizer, name)):
                    raise ValueError('temporal feedback/planner normalization mismatch')
        elif self.feedback_enabled:
            raise ValueError('trained temporal feedback checkpoint is required')
        with torch.random.fork_rng(devices=[]):
            self.head = TemporalPlanFeedbackHead(**(payload['spec'] if payload else {}))
        if payload:
            if payload['format'] == onpolicy_contract.FORMAT:
                mode = onpolicy_contract.validate_runtime_contract(payload['runtime_contract'], schedule,
                    self.head.residual_limit, self.execute_horizon)
                self.supported_horizons = tuple(payload['runtime_contract']['execute_horizons'])
                # A static checkpoint must remain static even when the adapter
                # supplies valid causal history. Explicit no-history ablation
                # of a temporal checkpoint is still allowed.
                self.history_enabled = self.history_enabled and mode == 'temporal'
            elif payload['runtime_contract'] != runtime_contract(schedule, self.head.residual_limit):
                raise ValueError('temporal feedback train/deploy contract mismatch')
            self.head.load_state_dict(payload['model'], strict=True)
        self.head.to(runner.device).float().eval().requires_grad_(False)
        self.sampler = NativePlanSampler(runner.model, schedule, enabled=self.graph_enabled,
                                        omit_unused_joint_outputs=True)
        self.feedback_graph = FeedbackGraph(self.head, enabled=self.graph_enabled)
        self.reset()

    def reset(self):
        self.base.reset()
        self.plan = self.source_state = None
        self.age = 0
        self.source_sim_time = None
        self.last_trace = {}

    def set_instruction(self, instruction):
        self.base.set_instruction(instruction)
        self.plan = self.source_state = None
        self.age = 0
        self.source_sim_time = None

    @torch.inference_mode()
    def query(self, packet):
        if self.feedback_enabled and self.execute_horizon not in self.supported_horizons:
            raise ValueError('feedback execute horizon is unsupported by this checkpoint')
        runner = self.base.runner
        sim_time = float(packet['sim_time'])
        if not math.isfinite(sim_time):
            raise ValueError('finite actual simulator time required')
        state = runner._normalize_state(self.base._state_tensor(packet['state']))
        refresh = self.plan is None or self.age == 16
        if refresh:
            text = runner._get_text_embeddings()
            future, condition = runner._initialize_video_latent(self.base._frame_tensor(packet['frame']), packet['flow_frames'])
            motion = runner._motion_batch(packet['motion_features'], packet['motion_interval_valid_mask'],
                                          packet['motion_acceleration_valid_mask'])
            source = SourceObservation(condition_latent=condition, state=state,
                motion=motion['absolute_motion_features'], motion_interval_valid_mask=motion['absolute_motion_interval_valid_mask'],
                motion_acceleration_valid_mask=motion['absolute_motion_acceleration_valid_mask'],
                text_embeddings=text, source_id=int(packet.get('action_index', 0)), sim_time=sim_time)
            self.plan, _ = self.sampler.sample(source, SourceNoise(future_latent=future, actions=runner._fresh_noisy_actions()))
            self.plan = self.plan.float()
            self.source_state = state.clone()
            self.source_sim_time = sim_time
            self.age = 0
            self.text, self.text_mask = fixed_feedback_text(text)
        if self.age not in range(16) or self.age + self.execute_horizon > 16:
            raise RuntimeError('feedback window left its immutable reference plan')
        elapsed = sim_time - self.source_sim_time
        if elapsed < 0:
            raise ValueError('simulator time moved backwards inside a plan')
        window = self.plan[:, self.age:self.age + self.execute_horizon]
        residual = torch.zeros_like(window)
        history_dt = float(packet.get('history_dt', 0.))
        history_valid = bool(packet.get('history_valid', False)) and self.history_enabled
        if not math.isfinite(history_dt) or history_dt < 0 or (history_valid and history_dt <= 0):
            raise ValueError('invalid previous-command observation time')
        if self.feedback_enabled and self.age > 0:
            rgb, mask = fast_views_from_composite(packet.get('fast_frame', packet['frame']))
            if history_valid:
                prev, prev_mask = letterbox_view(packet['previous_head'])
                prev = imagenet_normalize(prev).transpose(2, 0, 1)
            else:
                prev, prev_mask = rgb[0], mask[0]
            def tensor(array):
                return torch.as_tensor(np.ascontiguousarray(array), device=runner.device).unsqueeze(0)
            inputs = dict(rgb=tensor(rgb), rgb_valid_mask=tensor(mask), state=state.float(),
                source_state=self.source_state.float(), plan=self.plan, text=self.text, text_mask=self.text_mask,
                previous_head=tensor(prev), previous_head_mask=tensor(prev_mask),
                age_actions=torch.tensor([self.age], device=runner.device),
                history_dt=torch.tensor([history_dt], device=runner.device, dtype=torch.float32),
                history_valid=torch.tensor([history_valid], device=runner.device, dtype=torch.bool),
                plan_age_seconds=torch.tensor([elapsed], device=runner.device, dtype=torch.float32))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=state.is_cuda):
                residual = self.feedback_graph(**inputs)[:, :self.execute_horizon]
        actions = self.runtime.action_normalizer.denormalize(window + residual.float())
        # One result copy synchronizes actions and trace values together.
        combined = torch.cat((actions, residual.float()), dim=-1).squeeze(0).float().cpu().numpy()
        if not np.isfinite(combined).all():
            raise FloatingPointError('nonfinite temporal feedback action or residual')
        result = combined[:, :14].copy()
        self.last_trace = dict(mode='refresh' if refresh else 'feedback', age_actions=self.age,
            plan_age_seconds=elapsed, execute_horizon=self.execute_horizon, history_dt=history_dt,
            history_valid=history_valid, history_enabled=self.history_enabled,
            feedback_enabled=self.feedback_enabled, residual_abs_max=float(np.abs(combined[:, 14:]).max()),
            planner_graph=dict(self.sampler.stats), feedback_graph=dict(self.feedback_graph.stats))
        self.age += self.execute_horizon
        return result
