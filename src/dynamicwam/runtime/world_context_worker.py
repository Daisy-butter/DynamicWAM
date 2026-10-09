"""Process-local WAM planner and observation-routed reusable world context.

Only the simulator owns physical time. This worker never advances or rewinds
the environment. Returned rows are absolute joint targets at packet time plus
(row+1)*period; the controller, not this model, discards expired rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
import uuid

import numpy as np
import torch

from dynamicwam.inference.native_plan_graph import NativePlanSampler
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise
from dynamicwam.models.world_context_feedback import WorldContextFeedbackHead, pack_world_context, WORLD_CONTEXT_SPEC
from dynamicwam.observations.multiview import fast_views_from_composite, letterbox_view, imagenet_normalize
from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy
from dynamicwam.runtime.ciwam.wam.temporal_feedback_policy import fixed_feedback_text
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_copy(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'unsupported record value {type(value)}')


def exclusive_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(cpu_copy(value), stream)
    return dict(path=str(path.absolute()), sha256=file_sha256(path))


class WorldContextWorker:
    def __init__(self, config):
        self.config = dict(config)
        self.mode = config.get('mode', 'graph_zero')
        if self.mode not in ('official_eager', 'graph_zero', 'world_context', 'context_off'):
            raise ValueError('unknown continuous policy arm')
        self.period = float(config.get('control_period_s', .1))
        self.refresh_age = int(config.get('refresh_age', 12))
        if self.period != .1 or self.refresh_age != 12:
            raise ValueError('v1 requires explicit 100ms slots and refresh at age12')
        self.protocol_sha = str(config['controller_protocol_sha256'])
        if len(self.protocol_sha) != 64:
            raise ValueError('controller protocol identity required')
        self.base = DynamicWAMPolicy(config['deploy_config'], project_root=config['project_root'])
        self.base.setup()
        from dynamicwam.inference.model_loader import load_deploy_config
        cfg = load_deploy_config(self.base.deploy_config_path)
        self.base_sha = file_sha256(cfg['checkpoint_path'])
        expected_base = config.get('base_checkpoint_sha256')
        if expected_base and expected_base != self.base_sha:
            raise ValueError('official full checkpoint identity mismatch')
        runtime = self.base.runtime
        self.schedule = BridgeSchedule.from_flow_match(shift=runtime.flow_match_shift,
            sigma_min=runtime.flow_match_sigma_min, extra_one_step=runtime.flow_match_extra_one_step)
        self.sampler = NativePlanSampler(self.base.runner.model, self.schedule,
            enabled=self.mode != 'official_eager', omit_unused_joint_outputs=self.mode != 'official_eager')
        self.head = None
        self.checkpoint_sha = None
        self.history_mode = 'temporal'
        self.world_mode = 'context'
        if config.get('feedback_checkpoint'):
            path = config['feedback_checkpoint']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            self.checkpoint_sha = file_sha256(path)
            if payload['base_checkpoint']['sha256'] != self.base_sha:
                raise ValueError('head and WAM checkpoint identities differ')
            with torch.random.fork_rng(devices=[]):
                self.head = WorldContextFeedbackHead(**payload['spec'])
            if payload.get('format') == 'dynamicwam_plan_feedback_checkpoint':
                if self.mode != 'graph_zero':
                    raise ValueError('legacy head is initialization only, not a continuous-policy checkpoint')
                self.head.initialize_from_plan_feedback(payload['model'])
            else:
                from dynamicwam.models import world_context_feedback_contract as contract
                if payload.get('format') != contract.FORMAT or payload.get('version') != contract.VERSION:
                    raise ValueError('expected world-context checkpoint')
                if payload.get('controller_protocol_sha256') != self.protocol_sha:
                    raise ValueError('training/deployment controller protocol differs')
                self.history_mode,self.world_mode=contract.validate_runtime_contract(
                    payload['runtime_contract'],self.schedule,self.head.residual_limit,2)
                if (payload.get('world_context_spec') != WORLD_CONTEXT_SPEC or
                    payload.get('history_mode') != self.history_mode or payload.get('world_mode') != self.world_mode):
                    raise ValueError('export metadata and runtime contract disagree')
                self.head.load_state_dict(payload['model'], strict=True)
            from dynamicwam.action_normalization import RoboTwinQposNormalizer
            normal = payload['action_normalization']
            saved = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
            for key in ('_mean', '_std'):
                if not torch.equal(getattr(saved, key), getattr(runtime.action_normalizer, key)):
                    raise ValueError('head normalization differs from planner')
            self.head.to(self.base.runner.device).float().eval().requires_grad_(False)
        if self.mode in ('world_context', 'context_off') and self.head is None:
            raise ValueError('feedback arm requires a trained protocol-matched head')
        self.reset()

    def reset(self):
        self.base.reset()
        self.plan = self.source_state = self.context = None
        self.source_time = None
        self.source_action_index = None
        self.plan_id = None
        self.instruction = None
        self.record_count = 0
        self.bundle_ref = None
        self.source_noise = None

    def _source(self, packet, *, sampling_seed=None):
        runner = self.base.runner
        seed = packet.get('_sampling_seed') if sampling_seed is None else sampling_seed
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('an explicit nonnegative int64 sampling seed is required')
        # Query identity, not warmup or previous query count, owns the noise.
        generator = torch.Generator(device=runner.device).manual_seed(seed)
        state = runner._normalize_state(self.base._state_tensor(packet['state']))
        text = runner._get_text_embeddings()
        future, condition = runner._initialize_video_latent(self.base._frame_tensor(packet['frame']), packet['flow_frames'], generator=generator)
        motion = runner._motion_batch(packet['motion_features'], packet['motion_interval_valid_mask'],
                                      packet['motion_acceleration_valid_mask'])
        source = SourceObservation(condition_latent=condition, state=state,
            motion=motion['absolute_motion_features'], motion_interval_valid_mask=motion['absolute_motion_interval_valid_mask'],
            motion_acceleration_valid_mask=motion['absolute_motion_acceleration_valid_mask'], text_embeddings=text,
            source_id=int(packet['command_index']), sim_time=float(packet['sim_time']))
        return source, SourceNoise(future_latent=future, actions=runner._fresh_noisy_actions(generator=generator)), text

    def _inputs(self, packet, age):
        runner = self.base.runner
        rgb, mask = fast_views_from_composite(packet.get('fast_frame', packet['frame']))
        history_dt = float(packet.get('history_dt', 0.))
        previous = packet.get('previous_head_rgb', packet.get('previous_head'))
        valid = previous is not None and history_dt > 0 and math.isfinite(history_dt)
        if valid:
            prev, prev_mask = letterbox_view(previous)
            prev = imagenet_normalize(prev).transpose(2, 0, 1)
        else:
            prev, prev_mask = rgb[0], mask[0]
            history_dt = 0.
        def tensor(x):
            return torch.as_tensor(np.ascontiguousarray(x), device=runner.device).unsqueeze(0)
        return dict(rgb=tensor(rgb), rgb_valid_mask=tensor(mask),
            state=runner._normalize_state(self.base._state_tensor(packet['state'])).float(),
            source_state=self.source_state, plan=self.plan, text=self.text, text_mask=self.text_mask,
            previous_head=tensor(prev), previous_head_mask=tensor(prev_mask),
            age_actions=torch.tensor([age], device=runner.device),
            history_dt=torch.tensor([history_dt], device=runner.device),
            history_valid=torch.tensor([valid], device=runner.device),
            plan_age_seconds=torch.tensor([float(packet['sim_time']) - self.source_time], device=runner.device))

    def _record(self, packet, inputs, age):
        directory = self.config.get('collection_dir')
        if packet.get('_warmup_only') or not directory or self.record_count >= int(self.config.get('samples_per_episode', 64)):
            return
        root = Path(directory).absolute()
        if self.bundle_ref is None:
            self.bundle_ref = exclusive_tensor(root/'bundles'/f'{self.plan_id}.pt', dict(
                format='dynamicwam_world_context_bundle', version=1,
                plan=self.plan, source_state=self.source_state, **self.context,
                plan_id=self.plan_id, source_sim_time=self.source_time,
                source_action_index=self.source_action_index, published_sim_time=None,
                source_noise=self.source_noise,
                sampling_seed=self.source_sampling_seed,
                text=self.text, text_mask=self.text_mask,
                base_checkpoint_sha256=self.base_sha, controller_protocol_sha256=self.protocol_sha,
                world_context_spec=WORLD_CONTEXT_SPEC))
        # Teacher inference is deliberately deferred until this physical rollout
        # ends. Its cost must never change the behavior that generated the data.
        row = dict(format='dynamicwam_world_context_raw_record', version=1,
            task=str(self.config['task']), seed=int(self.config['seed']),
            inputs={k:v for k,v in inputs.items() if k not in ('plan','source_state','text','text_mask')},
            packet=packet, world_context_ref=self.bundle_ref,
            plan_id=self.plan_id, age=age, command_index=int(packet['command_index']),
            source_action_index=self.source_action_index,
            sim_time=float(packet['sim_time']), history_dt=float(inputs['history_dt'].item()),
            current_sim_time=float(packet['sim_time']), source_sim_time=self.source_time,
            behavior=self.mode, behavior_checkpoint_sha256=self.checkpoint_sha,
            base_checkpoint_sha256=self.base_sha, controller_protocol_sha256=self.protocol_sha)
        identity = exclusive_tensor(root/'raw'/f'{self.record_count:06d}.pt', row)
        with (root/'raw_index.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(**identity, plan_id=self.plan_id, age=age,
                                         command_index=int(packet['command_index'])))+'\n')
        self.record_count += 1

    @torch.inference_mode()
    def sample(self, packet):
        if packet.get('_reset_policy', False):
            self.reset()
        instruction = str(packet['instruction'])
        if instruction != self.instruction:
            self.base.set_instruction(instruction)
            self.instruction = instruction
            self.plan = None
        now = float(packet['sim_time'])
        if not math.isfinite(now) or now < 0 or float(packet['control_period_s']) != self.period:
            raise ValueError('invalid packet physical time/control protocol')
        age = 16 if self.plan is None else int(math.floor((now-self.source_time)/self.period + 1e-6))
        if age < 0:
            raise ValueError('observation predates the active plan')
        acknowledged=packet.get('active_plan_id')
        branch_mismatch=acknowledged is not None and acknowledged != self.plan_id
        refresh = self.plan is None or age >= self.refresh_age or branch_mismatch
        if refresh:
            source, noise, text = self._source(packet)
            self.source_sampling_seed = packet['_sampling_seed']
            self.plan, cache = self.sampler.sample(source, noise)
            self.source_noise = dict(actions=noise.actions.detach().clone(),
                                     future_latent=noise.future_latent.detach().clone())
            self.plan = self.plan.float().clone()
            self.context = pack_world_context(cache) if (self.head is not None or self.config.get('collection_dir')) else {}
            self.source_state = source.state.float().clone()
            self.source_time = now
            self.source_action_index = int(packet['command_index'])
            self.plan_id = uuid.uuid4().hex
            self.bundle_ref = None
            self.text, self.text_mask = fixed_feedback_text(text)
            age = 0
        inputs = self._inputs(packet, age) if (self.head is not None or self.config.get('collection_dir')) else None
        tail = self.plan[:, age:].clone()
        residual = torch.zeros_like(tail[:, :min(4, tail.shape[1])])
        if age > 0 and self.mode in ('world_context', 'context_off'):
            if self.history_mode == 'static':
                inputs=dict(inputs,history_valid=torch.zeros_like(inputs['history_valid']),
                            history_dt=torch.zeros_like(inputs['history_dt']))
            world = dict(self.context)
            world['context_valid'] = torch.tensor([self.mode == 'world_context' and self.world_mode=='context'], device=self.plan.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                residual = self.head(**inputs, **world)[:, :min(4, tail.shape[1])].float()
            # Four corrected knots cover the next two-row query plus one query
            # delay; the rest stays the immutable WAM plan.
            tail[:, :residual.shape[1]] += residual
        record_start=time.perf_counter()
        if age > 0:
            self._record(packet, inputs, age)
        recording_wall_s=time.perf_counter()-record_start if self.config.get('collection_dir') else 0.
        actions = self.base.runtime.action_normalizer.denormalize(tail).squeeze(0).float().cpu().numpy()
        # Report the generated correction separately from the controller's
        # eventual prefix clipping. The existing action transfer has already
        # synchronized this query; analysis attributes these norms only to
        # intervals that actually consume the corresponding corrected row.
        feedback_row_l2 = residual.squeeze(0).float().cpu().norm(dim=-1).tolist()
        if not np.isfinite(actions).all():
            raise FloatingPointError('nonfinite continuous control result')
        return dict(actions=actions, metadata=dict(mode='refresh' if refresh else 'feedback',
            policy_arm=self.mode, plan_id=self.plan_id, source_sim_time=self.source_time,
            action_origin_sim_time=self.source_time + age * self.period,
            age_actions=age, feedback_rows=int(residual.shape[1]) if self.mode in ('world_context','context_off') and age else 0,
            feedback_row_l2=feedback_row_l2,feedback_row_l2_units='normalized_action',
            branch_mismatch_refresh=branch_mismatch,
            context_semantics='second_joint_reference_world_not_updated_future',
            context_bytes=sum(v.numel()*v.element_size() for v in self.context.values()),
            controller_protocol_sha256=self.protocol_sha, base_checkpoint_sha256=self.base_sha,
            checkpoint_sha256=self.checkpoint_sha, recording_wall_s=recording_wall_s,
            source_sampling_seed=self.source_sampling_seed, query_sampling_seed=packet.get('_sampling_seed'),
            planner_graph=dict(self.sampler.stats)))


def create_worker_policy(config):
    return WorldContextWorker(config)
