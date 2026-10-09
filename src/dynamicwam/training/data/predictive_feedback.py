"""Causal future-image pairs from complete, hashed continuous-control recordings.

Actual future commands are exclusively world-model supervision. This dataset
does not expose them as policy inputs and never substitutes a proposed plan.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.observations.multiview import imagenet_normalize
from dynamicwam.runtime.domino_rt.proposal_path import aggregate_command_path, forecast_horizons

FORMAT = 'dynamicwam_predictive_feedback_dataset'
VERSION = 1
MINIMUM_NOMINAL_HORIZONS = (.2, .4)
WINDOW_OFFSETS = ((-.1, .1), (-.1, .2))
HORIZON_RULE = 'ceil((dispatch+.02)/.1)*.1-source+(.1,.3)'
PATH_BINS = 4
TIME_TOLERANCE = 1e-8


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def checked_bytes(identity, root):
    if not isinstance(identity, dict) or not {'path', 'sha256'} <= identity.keys():
        raise ValueError('artifact requires a path and SHA256')
    expected = identity['sha256']
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('invalid SHA256')
    path = Path(identity['path'])
    path = (path if path.is_absolute() else Path(root) / path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'changed immutable artifact: {path}')
    if 'bytes' in identity and len(raw) != identity['bytes']:
        raise ValueError('artifact byte count differs')
    return raw, path


def checked_json(identity, root):
    raw, path = checked_bytes(identity, root)
    return json.loads(raw), path


def select_future_frames(times, source_index, desired_horizons=None):
    """Select distinct real frames in disjoint horizon windows; never pad tails."""
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0) or not 0 <= source_index < len(times)):
        raise ValueError('observation times must be finite and strictly increasing')
    delta = times - times[source_index]
    desired_horizons = np.asarray(MINIMUM_NOMINAL_HORIZONS if desired_horizons is None
        else desired_horizons, dtype=np.float64)
    if (desired_horizons.shape != (2,) or not np.isfinite(desired_horizons).all() or
            desired_horizons[0] <= .1 or np.diff(desired_horizons)[0] < .2-TIME_TOLERANCE):
        raise ValueError('desired future horizons must define positive disjoint windows')
    result = np.full(2, -1, dtype=np.int64)
    for k, (desired, (left, right)) in enumerate(zip(desired_horizons, WINDOW_OFFSETS)):
        lo, hi = desired + left, desired + right
        candidates = np.flatnonzero((delta > lo + TIME_TOLERANCE) & (delta <= hi + TIME_TOLERANCE))
        if len(candidates):
            # Stable tie-breaking chooses the earlier actual camera frame.
            result[k] = candidates[np.argmin(np.abs(delta[candidates] - desired))]
    if result[0] >= 0 and result[0] == result[1]:
        raise RuntimeError('a future frame was reused for two horizons')
    return result


def _rgb_tensor(rgb):
    return torch.from_numpy(imagenet_normalize(rgb.astype(np.float32) / 255.).transpose(0, 3, 1, 2).copy())


POLICY_FORMAT = 'dynamicwam_predictive_feedback_record'
POLICY_INPUTS = frozenset(('rgb','rgb_valid_mask','previous_head','previous_head_mask',
    'history_dt','history_valid','state','source_state','plan','age_actions','text','text_mask',
    'plan_age_seconds','proposal_path','proposal_dt'))


class PredictiveFeedbackDataset(Dataset):
    """Episode-isolated pairs; actual future paths never enter policy inputs.

    ``mode=world`` requires observed futures. ``mode=policy`` requires an
    independently hashed current-teacher label and may have no observed future;
    its world loss is then masked, never endpoint padded.
    """
    def __init__(self, manifest_path, *, partition, reference_checkpoint, max_cached_episodes=2,
                 mode='world'):
        if partition not in ('train', 'validation'):
            raise ValueError('unknown partition')
        if mode not in ('world','policy'):
            raise ValueError('unknown predictive training mode')
        self.mode = mode
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = sha256(self.path)
        self.manifest = json.loads(self.path.read_text())
        m, root = self.manifest, self.path.parent
        if m.get('format') != FORMAT or m.get('version') != VERSION:
            raise ValueError('unknown predictive dataset format')
        if (m.get('base_checkpoint', {}).get('sha256') != reference_checkpoint['base_checkpoint']['sha256'] or
                m.get('action_normalization') != reference_checkpoint['action_normalization']):
            raise ValueError('predictive data and v1 normalization/base identity differ')
        self.protocol_sha256 = m['controller_protocol_sha256']
        if (not isinstance(self.protocol_sha256, str) or len(self.protocol_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.protocol_sha256)):
            raise ValueError('controller identity is required')
        normal = m['action_normalization']
        normalizer = RoboTwinQposNormalizer.from_stats(normal['stats'], epsilon=normal['epsilon'])
        self.mean, self.scale = normalizer._mean.numpy(), normalizer._std.numpy()
        official, _ = checked_json(m['official_seed_manifest'], root)
        tasks = official.get('tasks', {})
        if (official.get('source_arm') != 'official_sync_h16' or len(tasks) != 35 or
                any(len(seeds) != 20 or len(set(seeds)) != 20 for seeds in tasks.values())):
            raise ValueError('authoritative official 700 exclusion manifest is required')
        self.max_cached = max(1, int(max_cached_episodes))
        self._cache = OrderedDict()
        self.episodes, self.records = [], []
        seen, all_episode_ids = set(), {'train': [], 'validation': []}
        coverage = {'train': Counter(), 'validation': Counter()}
        for entry in m['episodes']:
            task, seed, split = entry['task'], entry['seed'], entry['partition']
            behavior=entry.get('behavior','graph_zero')
            behavior_checkpoint=entry.get('behavior_checkpoint_sha256')
            if behavior not in ('graph_zero','predictive') or (behavior=='predictive' and
                    (not isinstance(behavior_checkpoint,str) or len(behavior_checkpoint)!=64)):
                raise ValueError('unidentified collection behavior')
            if (split not in all_episode_ids or type(seed) is not int or task not in tasks or
                    seed in tasks[task] or (task, seed) in seen):
                raise ValueError('duplicate, cross-partition, or official-evaluation episode')
            seen.add((task, seed)); all_episode_ids[split].append((task, seed))
            completion, completion_path = checked_json(entry['trajectory_completion'], root)
            if (completion.get('format') != 'dynamicwam_continuous_trajectory_completion' or
                    completion.get('version') != 1 or completion.get('status') != 'complete'):
                raise ValueError('dense recording has no successful commit marker')
            trajectory, trajectory_path = checked_json(completion['manifest'], completion_path.parent)
            metadata, provenance = trajectory.get('metadata', {}), trajectory.get('provenance', {})
            if (trajectory.get('format') != 'dynamicwam_continuous_trajectory' or
                    trajectory.get('version') != 1 or trajectory.get('status') != 'complete' or
                    metadata.get('task') != task or metadata.get('seed') != seed or
                    metadata.get('realtime_valid') is not True or
                    provenance.get('controller_protocol_sha256') != self.protocol_sha256 or
                    trajectory.get('physics_dt') != .004 or trajectory.get('control_period') != .1 or
                    trajectory.get('view_order') != ['head', 'left_wrist', 'right_wrist'] or
                    not provenance.get('source_sha256')):
                raise ValueError('dense recording task/clock/source contract differs')
            episode = dict(task=task, seed=seed, partition=split, trajectory=trajectory,
                root=trajectory_path.parent, manifest_sha256=completion['manifest']['sha256'],
                completion_sha256=entry['trajectory_completion']['sha256'])
            for artifact in trajectory['artifacts'].values():
                checked_bytes(artifact, trajectory_path.parent)
            observations, commands = self._read_episode(episode)
            times = observations['observation_sim_time']
            policies = {}
            if entry.get('policy_index'):
                raw_index,index_path = checked_bytes(entry['policy_index'],root)
                episode['policy_index_sha256'] = entry['policy_index']['sha256']
                for line in raw_index.decode('utf-8').splitlines():
                    if not line.strip():continue
                    identity = json.loads(line)
                    capture = identity.get('capture_id')
                    if type(capture) is not int or capture in policies or capture not in observations['capture_id']:
                        raise ValueError('policy label is duplicated or has no actual capture')
                    raw_label,label_path = checked_bytes(identity,index_path.parent)
                    label = torch.load(io.BytesIO(raw_label),map_location='cpu',weights_only=True)
                    self._validate_policy(label,task,seed,capture,episode,observations)
                    first_live_row = int(round((float(label['inputs']['proposal_dt'][0])-.1)/.1))
                    available_rows = min(4,16-int(label['inputs']['age_actions']))
                    live_rows = max(0,available_rows-first_live_row)
                    coverage[split]['policy_expired_prefix_rows'] += min(first_live_row,available_rows)
                    coverage[split]['policy_no_live_rows'] += int(live_rows == 0)
                    policies[capture] = dict(path=str(label_path),sha256=identity['sha256'],
                        first_live_row=first_live_row,live_rows=live_rows)
                coverage[split]['policy_labels'] += len(policies)
            episode_index = len(self.episodes); self.episodes.append(episode)
            coverage[split]['observations'] += len(times)
            for current in range(len(times)):
                dispatch = float(observations['dispatch_sim_time'][current])
                submitted = bool(observations['policy_submitted'][current])
                desired = forecast_horizons(times[current], dispatch)
                future = select_future_frames(times, current, desired)
                coverage[split]['actual_dispatch' if submitted else 'unsubmitted_source_dispatch_default'] += 1
                paths, valid = [], []
                for index in future:
                    if index < 0:
                        paths.append(np.zeros((4, 28), np.float32)); valid.append(False)
                    else:
                        path, complete = aggregate_command_path(commands['start_sim_time'], commands['end_sim_time'],
                            commands['q_command'], commands['velocity_command'], times[current], times[index], self.mean, self.scale)
                        paths.append(path); valid.append(complete)
                        coverage[split]['incomplete_path'] += int(not complete)
                for h, flag in enumerate(valid):
                    coverage[split][f'horizon_{h}_valid'] += int(flag)
                policy_ref = policies.get(int(observations['capture_id'][current]))
                if not any(valid):
                    coverage[split]['no_supported_future'] += 1
                if (mode == 'world' and not any(valid)) or (mode == 'policy' and
                        (not policy_ref or policy_ref['live_rows']==0)):
                    continue
                coverage[split]['pairs'] += 1
                if split == partition:
                    self.records.append(dict(episode_index=episode_index, current=current, future=future,
                        actual_action_path=np.stack(paths), future_valid=np.asarray(valid, dtype=np.bool_),
                        future_dt=np.asarray([times[i] - times[current] if flag else 0.
                            for i, flag in zip(future, valid)], dtype=np.float32), task=task, seed=seed,
                        observation_sim_time=float(times[current]),
                        dispatch_sim_time=dispatch, policy_submitted=submitted,
                        desired_future_dt=desired.astype(np.float32),
                        capture_id=int(observations['capture_id'][current]),policy_ref=policy_ref,
                        behavior=behavior,behavior_checkpoint_sha256=behavior_checkpoint,
                        record_id=f"{episode['manifest_sha256']}:{current}"))
            self._cache[episode_index] = (observations, commands)
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        if not self.records or any(not entries for entries in all_episode_ids.values()):
            raise ValueError('nonempty train/heldout episodes and supported future pairs are required')
        self.partition = partition
        self.provenance = dict(manifest_sha256=self.manifest_sha256, controller_protocol_sha256=self.protocol_sha256,
            partition=partition, episodes=all_episode_ids, coverage={k:dict(v) for k,v in coverage.items()},
            trajectory_manifest_sha256=[e['manifest_sha256'] for e in self.episodes],
            policy_index_sha256=[e.get('policy_index_sha256') for e in self.episodes],
            training_behaviors=sorted({r['behavior'] for r in self.records}),
            state_semantics='joint_drive_targets; not measured q', training_scope=mode,
            policy_row_mask='tail_valid and row >= round((proposal_dt[0]-.1)/.1); no_live_records excluded from policy only',
            minimum_nominal_horizons=list(MINIMUM_NOMINAL_HORIZONS),
            horizon_rule=HORIZON_RULE, window_offsets=[list(x) for x in WINDOW_OFFSETS], path_bins=4,
            dispatch='actual submission time when policy_submitted; otherwise source-time hypothetical dispatch',
            action_path='actual commands from source to selected future; source-to-dispatch prefix was already observed; never a policy input',
            gripper_velocity='controller reference; not applied by position-only gripper API',
            timestamp='canonical_capture_and_command_grid; exact engine times checked as provenance')

    def _validate_policy(self,label,task,seed,capture,episode,observations):
        if (label.get('format') != POLICY_FORMAT or label.get('version') != 1 or
                label.get('task') != task or label.get('seed') != seed or label.get('capture_id') != capture or
                label.get('base_checkpoint_sha256') != self.manifest['base_checkpoint']['sha256'] or
                label.get('controller_protocol_sha256') != self.protocol_sha256 or
                label.get('trajectory_manifest_sha256') != episode['manifest_sha256']):
            raise ValueError('policy label does not identify this real episode/capture/base')
        provenance = label.get('label_provenance',{})
        if (provenance.get('teacher_only') is not True or provenance.get('current_observation') is not True or
                provenance.get('generation') != 'offline_after_rollout' or
                not provenance.get('source_sha256') or not label.get('proposal_provenance')):
            raise ValueError('policy teacher/proposal provenance is incomplete')
        inputs=label.get('inputs',{})
        if set(inputs) != POLICY_INPUTS or any(not torch.is_tensor(v) for v in inputs.values()):
            raise ValueError('policy inputs must contain only the deployable tensor contract')
        expected=dict(rgb=(3,3,160,160),rgb_valid_mask=(3,160,160),previous_head=(3,160,160),
            previous_head_mask=(160,160),history_dt=(),history_valid=(),state=(14,),source_state=(14,),
            plan=(16,14),age_actions=(),plan_age_seconds=(),proposal_path=(2,4,28),proposal_dt=(2,))
        for name,shape in expected.items():
            if tuple(inputs[name].shape) != shape:raise ValueError('policy tensor shape differs: '+name)
        if inputs['text'].ndim!=2 or inputs['text'].shape[-1]!=4096 or inputs['text_mask'].shape!=inputs['text'].shape[:1]:
            raise ValueError('policy text layout differs')
        if any(not bool(torch.isfinite(v).all()) for v in inputs.values() if v.is_floating_point()):
            raise ValueError('nonfinite policy input')
        if any(inputs[k].dtype!=torch.bool for k in ('rgb_valid_mask','previous_head_mask','history_valid','text_mask')):
            raise ValueError('policy masks must be boolean')
        if (inputs['age_actions'].dtype not in (torch.int32,torch.int64) or not 0<int(inputs['age_actions'])<16 or
                not bool((inputs['proposal_dt']>0).all()) or inputs['proposal_dt'][1]<=inputs['proposal_dt'][0] or
                float(inputs['history_dt'])<0 or (bool(inputs['history_valid']) and float(inputs['history_dt'])<=0)):
            raise ValueError('policy age/history/proposal time differs')
        teacher=label.get('teacher')
        if not torch.is_tensor(teacher) or tuple(teacher.shape) not in ((4,14),(16,14)) or not bool(torch.isfinite(teacher).all()):
            raise ValueError('policy teacher must be a finite current four/sixteen-row plan')
        i=int(np.flatnonzero(observations['capture_id']==capture)[0])
        stamp=float(label.get('observation_sim_time',-1))
        label_dispatch=float(label.get('dispatch_sim_time',-1))
        if not np.isfinite(stamp) or abs(stamp-float(observations['observation_sim_time'][i]))>1e-8:
            raise ValueError('teacher observation timestamp does not match dense capture')
        dispatch = float(observations['dispatch_sim_time'][i])
        if (not bool(observations['policy_submitted'][i]) or not np.isfinite(label_dispatch) or
                abs(label_dispatch-dispatch)>1e-8 or
                not np.allclose(inputs['proposal_dt'].numpy(),
                    forecast_horizons(observations['observation_sim_time'][i],dispatch),atol=1e-6,rtol=0)):
            raise ValueError('policy dispatch or deployable proposal horizons differ from dense capture')
        # Dense images are uint8; retain the original runtime float inputs after
        # verifying the sole permitted quantization difference.
        rgb=_rgb_tensor(observations['rgb'][i])
        if (not torch.equal(inputs['rgb_valid_mask'],torch.from_numpy(observations['rgb_valid_mask'][i])) or
                float((inputs['rgb'].float()-rgb).abs().max())>1/(255*.224)+1e-5):
            raise ValueError('policy label uses a different current image')
        state=(observations['policy_state'][i]-self.mean)/self.scale
        if not np.allclose(inputs['state'].numpy(),state,atol=2e-5,rtol=1e-5):
            raise ValueError('policy label does not use the captured command state')

    @staticmethod
    def _read_episode(episode):
        trajectory, root = episode['trajectory'], episode['root']
        result = []
        for name in ('observations', 'commands'):
            raw, _ = checked_bytes(trajectory['artifacts'][name], root)
            with np.load(io.BytesIO(raw), allow_pickle=False) as storage:
                required = (('rgb', 'rgb_valid_mask', 'observation_sim_time', 'source_physics_time',
                    'capture_roundoff_s', 'received_sim_time', 'dispatch_sim_time', 'policy_submitted',
                    'drive_targets', 'policy_state', 'capture_id', 'snapshot_sequence')
                    if name == 'observations' else ('q_command', 'velocity_command', 'start_sim_time',
                    'end_sim_time', 'engine_start_time', 'engine_end_time', 'tick_index'))
                if any(k not in storage for k in required):
                    raise ValueError('dense recording lacks required numerical fields')
                result.append({k:np.array(storage[k], copy=True) for k in required})
        obs, cmd = result
        f, t = len(obs['observation_sim_time']), len(cmd['start_sim_time'])
        if (not f or not t or trajectory['frames'] != f or trajectory['ticks'] != t or
                obs['rgb'].shape != (f,3,160,160,3) or obs['rgb'].dtype != np.uint8 or
                obs['rgb_valid_mask'].shape != (f,3,160,160) or obs['rgb_valid_mask'].dtype != np.bool_ or
                any(obs[k].shape != (f,14) for k in ('drive_targets','policy_state')) or
                any(cmd[k].shape != (t,14) for k in ('q_command','velocity_command'))):
            raise ValueError('dense image/command layout differs')
        if (any(obs[k].shape != (f,) for k in ('observation_sim_time','source_physics_time',
                'capture_roundoff_s','received_sim_time','dispatch_sim_time','policy_submitted','capture_id','snapshot_sequence')) or
                obs['policy_submitted'].dtype != np.bool_ or
                any(cmd[k].shape != (t,) for k in ('start_sim_time','end_sim_time','engine_start_time','engine_end_time','tick_index')) or
                len(set(obs['capture_id'].tolist()))!=f or len(set(obs['snapshot_sequence'].tolist()))!=f):
            raise ValueError('dense timestamps/capture identities differ')
        if any(not np.isfinite(v).all() for group in result for v in group.values() if v.dtype.kind == 'f'):
            raise ValueError('nonfinite dense trajectory')
        times = obs['observation_sim_time']
        if (np.any(np.diff(times) <= 0) or times[0] < 0 or
                np.any(obs['received_sim_time'] < times - 1e-7) or
                np.any(np.abs(obs['source_physics_time'] - times) > np.maximum(1e-7, np.abs(times)*1e-7)) or
                not np.allclose(obs['capture_roundoff_s'], obs['source_physics_time']-times, atol=1e-10, rtol=0)):
            raise ValueError('camera capture clock was retimestamped or reordered')
        submitted = obs['policy_submitted']; dispatch = obs['dispatch_sim_time']
        if (np.any(dispatch < times-1e-8) or np.any(dispatch-times > 2.) or
                np.any(np.abs(dispatch[~submitted]-times[~submitted]) > 1e-8) or
                np.any(dispatch[submitted] < obs['received_sim_time'][submitted]-1e-8) or
                np.any(dispatch[submitted] > float(trajectory['metadata']['physical_end_time'])+1e-8)):
            raise ValueError('dispatch clock is inconsistent with actual policy submission')
        origin = float(trajectory['metadata']['origin_sim_time'])
        if (not np.array_equal(cmd['tick_index'], np.arange(t)) or
                not np.allclose(cmd['start_sim_time'], np.arange(t)*.004, atol=1e-8, rtol=0) or
                not np.allclose(cmd['end_sim_time'], (np.arange(t)+1)*.004, atol=1e-8, rtol=0) or
                np.any(np.abs(cmd['engine_start_time']-origin-cmd['start_sim_time']) > .0004) or
                np.any(np.abs(cmd['engine_end_time']-origin-cmd['end_sim_time']) > .0004) or
                abs(cmd['end_sim_time'][-1]-float(trajectory['metadata']['physical_end_time'])) > 1e-8 or
                times[-1] > cmd['end_sim_time'][-1]+1e-8):
            raise ValueError('dense command intervals do not cover actual complete physics')
        return obs, cmd

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]; episode_index = row['episode_index']
        if episode_index not in self._cache:
            self._cache[episode_index] = self._read_episode(self.episodes[episode_index])
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)
        self._cache.move_to_end(episode_index)
        obs, _ = self._cache[episode_index]
        i = row['current']; history_valid = i > 0
        rgb = _rgb_tensor(obs['rgb'][i]); mask = torch.from_numpy(obs['rgb_valid_mask'][i].copy())
        if history_valid:
            previous = _rgb_tensor(obs['rgb'][i-1])[0]
            previous_mask = torch.from_numpy(obs['rgb_valid_mask'][i-1,0].copy())
            dt = obs['observation_sim_time'][i]-obs['observation_sim_time'][i-1]
        else:
            previous, previous_mask, dt = torch.zeros_like(rgb[0]), torch.zeros_like(mask[0]), 0.
        future_rgb = torch.zeros(2,3,3,160,160); future_mask = torch.zeros(2,3,160,160,dtype=torch.bool)
        for h, (j, valid) in enumerate(zip(row['future'], row['future_valid'])):
            if valid:
                future_rgb[h] = _rgb_tensor(obs['rgb'][j])
                future_mask[h] = torch.from_numpy(obs['rgb_valid_mask'][j].copy())
        result = dict(rgb=rgb, rgb_valid_mask=mask, previous_head=previous, previous_head_mask=previous_mask,
            history_dt=torch.tensor(dt,dtype=torch.float32), history_valid=torch.tensor(history_valid),
            future_rgb=future_rgb, future_rgb_valid_mask=future_mask,
            actual_action_path=torch.from_numpy(row['actual_action_path'].copy()),
            future_dt=torch.from_numpy(row['future_dt'].copy()), future_valid=torch.from_numpy(row['future_valid'].copy()),
            record_id=row['record_id'], task=row['task'], seed=row['seed'],
            observation_sim_time=row['observation_sim_time'], dispatch_sim_time=row['dispatch_sim_time'],
            policy_submitted=row['policy_submitted'], desired_future_dt=torch.from_numpy(row['desired_future_dt'].copy()),
            future_frame_indices=row['future'].tolist())
        if self.mode=='policy':
            raw,_=checked_bytes(row['policy_ref'],self.path.parent)
            label=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            result.update(policy_inputs=label['inputs'],teacher=label['teacher'][:4].float(),
                policy_record_sha256=row['policy_ref']['sha256'])
        return result


def predictive_collate_fn(rows):
    if not rows:
        raise ValueError('empty predictive batch')
    def collate(values):
        if torch.is_tensor(values[0]):return torch.stack(values)
        if isinstance(values[0],dict):return {k:collate([v[k] for v in values]) for k in values[0]}
        return values
    return {key:collate([r[key] for r in rows]) for key in rows[0]}
