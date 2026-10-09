"""Diagnostic data from actual native-plan execution; never a speed benchmark.

The policy always executes its unchanged zero-feedback plan. A frozen current
WAM labels selected observed states without acting. Torch RNG is restored and
the reference plan is checked for mutation before an action is returned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import deploy_policy_temporal_feedback as native
from .composite import composite_robotwin_frame
from dynamicwam.models.multirate_types import SourceObservation, SourceNoise


def shifted_noise(actions, age):
    if not 0 < age < 16 or tuple(actions.shape[1:]) != (16, 14):
        raise ValueError('shadow labels require a nonfresh native 16x14 plan')
    return torch.cat((actions[:, age:], torch.randn_like(actions[:, :age])), dim=1)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


class ShadowFeedbackDeployModel(native.TemporalFeedbackDeployModel):
    def __init__(self, args):
        if not args.get('shadow_output_root') or not args.get('fixed_episode_seeds'):
            raise ValueError('shadow collection requires a new directory and explicit fixed seeds')
        configured = dict(args, temporal_feedback_disabled=True, temporal_feedback_checkpoint=None,
                          execute_horizon=2)
        super().__init__(configured)
        self.output_root = Path(args['shadow_output_root'])
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.fixed_seeds = [int(seed) for seed in args['fixed_episode_seeds']]
        if len(set(self.fixed_seeds)) != len(self.fixed_seeds):
            raise ValueError('duplicate shadow episode seed')
        self.task = args['task_name']
        self.max_samples = int(args.get('shadow_samples_per_episode', 64))
        if not 1 <= self.max_samples <= 256:
            raise ValueError('bounded shadow sample count required')
        self.original_query = self.policy.query
        self.policy.query = self.shadow_query
        self.samples_by_episode = {}
        self.source_noise = None
        self.observation = self.environment = None

    @torch.inference_mode()
    def shadow_query(self, packet):
        policy, runner = self.policy, self.policy.base.runner
        fresh = policy.plan is None or policy.age == 16
        age = 0 if fresh else policy.age
        # Instrument the actual planner's noise draws; no extra RNG consumption.
        if fresh:
            video_fn, action_fn = runner._initialize_video_latent, runner._fresh_noisy_actions
            captured = {}
            def video(*args, **kwargs):
                future, condition = video_fn(*args, **kwargs)
                captured['future_latent'] = future.clone()
                return future, condition
            def action():
                result = action_fn()
                captured['actions'] = result.clone()
                return result
            runner._initialize_video_latent, runner._fresh_noisy_actions = video, action
            try:
                result = self.original_query(packet)
            finally:
                runner._initialize_video_latent, runner._fresh_noisy_actions = video_fn, action_fn
            self.source_noise = SourceNoise(**captured)
            return result
        result = self.original_query(packet)
        episode = int(self.environment.ep_num)
        if not 0 <= episode < len(self.fixed_seeds):
            raise RuntimeError('environment episode escaped the fixed diagnostic seed list')
        ordinal = self.samples_by_episode.get(episode, 0)
        # Deterministic prefix sampling of all nonfresh H2 queries, not selection
        # based on residual, success, teacher agreement, or image contents.
        if ordinal >= self.max_samples:
            return result
        if self.source_noise is None:
            raise RuntimeError('missing noise from the actually executed source plan')
        reference = policy.plan.clone()
        source_state = policy.source_state.clone()
        clock_before = float(self.environment._scene_step_clock.snapshot().time_seconds)
        if clock_before != float(packet['sim_time']):
            raise RuntimeError('shadow input observation clock differs from the simulator')
        state = runner._normalize_state(policy.base._state_tensor(packet['state']))
        motion = self.flow.observation()
        frame = composite_robotwin_frame(self.observation, policy.runtime.observation_config)
        cuda_devices = [reference.device.index] if reference.is_cuda else []
        label_seed = int.from_bytes(hashlib.sha256(
            f'{self.task}/{self.fixed_seeds[episode]}/{self.index}'.encode()).digest()[:4], 'little')
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(label_seed)
            _, condition = runner._initialize_video_latent(policy.base._frame_tensor(frame), motion.flow_rgb)
            fields = runner._motion_batch(motion.motion_features, motion.interval_valid_mask,
                                           motion.acceleration_valid_mask)
            source = SourceObservation(condition_latent=condition, state=state,
                motion=fields['absolute_motion_features'],
                motion_interval_valid_mask=fields['absolute_motion_interval_valid_mask'],
                motion_acceleration_valid_mask=fields['absolute_motion_acceleration_valid_mask'],
                text_embeddings=runner._get_text_embeddings(), source_id=int(self.index),
                sim_time=float(packet['sim_time']))
            noise = SourceNoise(future_latent=self.source_noise.future_latent.clone(),
                actions=shifted_noise(self.source_noise.actions, age))
            teacher, _ = policy.sampler.sample(source, noise)
        if (not torch.equal(reference, policy.plan) or not torch.equal(source_state, policy.source_state)
                or policy.age != age + 2):
            raise RuntimeError('shadow teacher mutated the executed native plan or its age')
        if float(self.environment._scene_step_clock.snapshot().time_seconds) != clock_before:
            raise RuntimeError('shadow teacher advanced the simulator')
        expected = policy.runtime.action_normalizer.denormalize(reference[:, age:age+2])
        np.testing.assert_array_equal(result, expected.squeeze(0).float().cpu().numpy())
        token_count = int(policy.text_mask[0].sum())
        if (not bool(policy.text_mask[0, :token_count].all())
                or bool(policy.text_mask[0, token_count:].any())):
            raise RuntimeError('shadow compact text requires a contiguous valid prefix')
        record = dict(format='dynamicwam_on_policy_shadow', version=1, task=self.task,
            episode_index=episode, seed=self.fixed_seeds[episode], instruction=str(self.environment.get_instruction()),
            command_index=int(self.index), age=age, source_sim_time=float(policy.source_sim_time),
            sim_time=float(packet['sim_time']), history_dt=float(packet['history_dt']),
            history_valid=bool(packet['history_valid']), label_seed=label_seed,
            base_checkpoint_sha256=policy.base_sha256, source_state=policy.source_state,
            state=state, plan=reference, teacher=teacher.float(),
            teacher_inputs=dict(condition_latent=condition, state=state, motion=source.motion,
                motion_interval_valid_mask=source.motion_interval_valid_mask,
                motion_acceleration_valid_mask=source.motion_acceleration_valid_mask),
            teacher_noise_inputs=dict(future_latent=noise.future_latent, actions=noise.actions),
            schedule=policy.sampler.schedule.to_config(),
            executed_actions=result, fast_frame=packet['fast_frame'], previous_head=packet['previous_head'],
            text=policy.text[:, :token_count], text_mask=policy.text_mask[:, :token_count],
            state_semantics='joint_drive_targets', behavior='immutable_official_full_16_zero_feedback',
            history_semantics='actual_previous_command',
            provenance='command targets and RGB after actual zero-feedback plan execution; not measured joint positions or expert rollout',
            teacher_noise='same actual source future noise; shifted overlapping action noise; independent tail; one label, not a variance estimate',
            timing_scope='diagnostic shadow computation pauses simulation; not policy latency')
        path = self.output_root/f'episode{episode:03d}_query{ordinal:04d}.pt'
        with path.open('xb') as stream:
            torch.save(cpu_copy(record), stream)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with (self.output_root/'index.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(path=path.name, sha256=digest, episode=episode,
                seed=record['seed'], command_index=self.index, age=age))+'\n')
        self.samples_by_episode[episode] = ordinal + 1
        print(f'SHADOW-FEEDBACK-SAMPLE {episode} {self.index} {age} {path.name}', flush=True)
        return result


def get_model(usr_args):
    return ShadowFeedbackDeployModel(usr_args)


def reset_model(model):
    model.finish()
    model.source_noise = None


def eval(TASK_ENV, model, observation):
    model.environment, model.observation = TASK_ENV, observation
    return native.eval(TASK_ENV, model, observation)
