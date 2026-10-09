"""Native full WAM planning every 16 commands, bounded feedback every four."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from dynamicwam.inference.native_plan_graph import NativePlanSampler
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise, pad_text_embeddings
from dynamicwam.models.plan_feedback import PlanFeedbackHead, plan_window
from dynamicwam.observations.multiview import fast_views_from_composite
from .policy import DynamicWAMPolicy


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class PlanFeedbackPolicy:
    def __init__(self, config, *, project_root, checkpoint=None, feedback_enabled=True, graph_enabled=True):
        self.base = DynamicWAMPolicy(config, project_root=project_root)
        self.checkpoint_path = checkpoint
        self.feedback_enabled = bool(feedback_enabled)
        self.graph_enabled = bool(graph_enabled)
        self.reset()

    @property
    def runtime(self):
        return self.base.runtime

    def setup(self):
        from dynamicwam.inference.model_loader import load_deploy_config
        self.base.setup()
        runner = self.base.runner
        cfg = load_deploy_config(self.base.deploy_config_path)
        schedule = BridgeSchedule.from_flow_match(
            shift=self.runtime.flow_match_shift, sigma_min=self.runtime.flow_match_sigma_min,
            extra_one_step=self.runtime.flow_match_extra_one_step)
        base_path = cfg['checkpoint_path']
        self.base_sha256 = file_sha256(base_path)
        payload = None
        if self.checkpoint_path:
            payload = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
            if payload.get('format') != 'dynamicwam_plan_feedback_checkpoint' or payload.get('version') != 1:
                raise ValueError('expected a plan-feedback export')
            if payload['base_checkpoint']['sha256'] != self.base_sha256:
                raise ValueError('feedback was trained against a different native WAM')
            contract = payload['runtime_contract']
            expected = dict(source_plan='immutable_official_full_16', execute_horizon=4,
                refresh_actions=16, ages=[0, 4, 8, 12], age0_correction=False,
                gripper_correction=False, residual_units='normalized_qpos',
                preprocess='three_view_160_imagenet_mask_v1', schedule=schedule.to_config(),
                precision=dict(video='bfloat16', action='float32', solver='float32',
                    feedback_parameters='float32', feedback_autocast='bfloat16'))
            if any(contract.get(k) != v for k, v in expected.items()):
                raise ValueError('feedback runtime contract does not match this policy')
            from dynamicwam.action_normalization import RoboTwinQposNormalizer
            normal = payload['action_normalization']
            saved = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
            for name in ('_mean', '_std'):
                if not torch.equal(getattr(saved, name), getattr(self.runtime.action_normalizer, name)):
                    raise ValueError('feedback action normalization differs from planner')
        elif self.feedback_enabled:
            raise ValueError('feedback deployment requires a trained checkpoint')
        with torch.random.fork_rng(devices=[]):
            self.head = PlanFeedbackHead(**(payload['spec'] if payload else {}))
        if payload:
            self.head.load_state_dict(payload['model'], strict=True)
            if contract['residual_limit'] != self.head.residual_limit:
                raise ValueError('feedback residual bound differs from training')
        self.head.to(runner.device).float().eval().requires_grad_(False)
        self.sampler = NativePlanSampler(runner.model, schedule, enabled=self.graph_enabled)
        self.reset()

    def reset(self):
        self.base.reset()
        self.plan = None
        self.age = 0
        self.source_state = None
        self.last_trace = {}

    def set_instruction(self, instruction):
        self.base.set_instruction(instruction)
        self.plan = None
        self.age = 0

    @torch.inference_mode()
    def query(self, packet):
        runner = self.base.runner
        state = runner._normalize_state(self.base._state_tensor(packet['state']))
        refresh = self.plan is None or self.age == 16
        if refresh:
            text = runner._get_text_embeddings()
            future, condition = runner._initialize_video_latent(self.base._frame_tensor(packet['frame']), packet['flow_frames'])
            motion = runner._motion_batch(packet['motion_features'], packet['motion_interval_valid_mask'], packet['motion_acceleration_valid_mask'])
            source = SourceObservation(condition_latent=condition, state=state,
                motion=motion['absolute_motion_features'], motion_interval_valid_mask=motion['absolute_motion_interval_valid_mask'],
                motion_acceleration_valid_mask=motion['absolute_motion_acceleration_valid_mask'],
                text_embeddings=text, source_id=int(packet.get('action_index', 0)), sim_time=float(packet.get('sim_time', 0.)))
            noise = SourceNoise(future_latent=future, actions=runner._fresh_noisy_actions())
            self.plan, _ = self.sampler.sample(source, noise)
            self.plan = self.plan.float()
            self.source_state = state.clone()
            self.age = 0
            self.text, self.text_mask = pad_text_embeddings(text)
        window = plan_window(self.plan, self.age)
        residual = torch.zeros_like(window)
        if self.feedback_enabled and self.age > 0:
            rgb, mask = fast_views_from_composite(packet.get('fast_frame', packet['frame']))
            rgb = torch.from_numpy(rgb).unsqueeze(0).to(runner.device)
            mask = torch.from_numpy(mask).unsqueeze(0).to(runner.device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=rgb.is_cuda):
                residual = self.head(rgb, mask, state.float(), self.source_state.float(), self.plan,
                    self.age, self.text, self.text_mask)
        # Accumulate in FP32. Never mutate the reference plan, so corrections
        # cannot compound across future feedback windows.
        actions = self.runtime.action_normalizer.denormalize(window + residual.float())
        result = self.base._numpy_chunk(actions)
        self.last_trace = dict(mode='refresh' if refresh else 'feedback', age_actions=self.age,
            feedback_enabled=self.feedback_enabled, residual_abs_max=float(residual.abs().max()),
            planner_graph=dict(self.sampler.stats))
        self.age += 4
        return result
