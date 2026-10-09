#!/usr/bin/env python3
"""Read-only CPU audit of real temporal pairs and stubbed training contracts.

No real teacher is constructed and no CUDA context is initialized. Writes one
new JSON report only; training data/checkpoints are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch

import train_temporal_feedback as trainer
from dynamicwam.training.data.temporal_feedback_dataset import TemporalPairSampler, HISTORY_RAW_FRAME_LAGS, causal_history_frame
from dynamicwam.training.data.multirate_sampling import ProcessGroupAlignedSampler
from dynamicwam.training.data.paired_world_action_dataset import (
    clock_time_seconds, episode_media_paths)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite timing statistic')
    return dict(count=int(len(values)), mean=float(values.mean()), **{
        name: float(np.quantile(values, q)) for name, q in
        [('min', 0), ('p05', .05), ('p25', .25), ('p50', .5), ('p75', .75), ('p95', .95), ('max', 1)]})


def choose_indices(dataset, count, seed):
    """Round-robin tasks/ages; episode uniform within each available stratum."""
    by_task = defaultdict(lambda: defaultdict(list))
    for episode, groups in dataset.records.groups.items():
        task = dataset.records.episodes[episode]['task_name']
        for age in groups:
            by_task[task][age].append(episode)
    tasks = sorted(by_task)
    if count < max(len(tasks), 16):
        raise ValueError('audit count cannot cover all tasks and ages')
    rng, selected = random.Random(seed), []
    for i in range(count):
        task, age = tasks[i % len(tasks)], i % 16
        if not by_task[task].get(age):
            raise RuntimeError(f'no age {age} for task {task}')
        episode = rng.choice(by_task[task][age])
        start, stop = dataset.records.groups[episode][age]
        selected.append(rng.randrange(start, stop))
    return selected


def clock_identity(dataset, record, clock):
    digest = hashlib.sha256()
    for key in ('frame_index', 'sim_step_index', 'sim_time_seconds'):
        tensor = clock[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    video, path = episode_media_paths(dataset.source_root, split=record.split,
        task_name=record.task_name, episode_name=record.episode_name)
    stat = path.stat()
    return dict(episode_index=record.episode_index, path=str(path), size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns, clock_fields_sha256=digest.hexdigest(),
        video=dict(path=str(video), size_bytes=video.stat().st_size,
                   mtime_ns=video.stat().st_mtime_ns))


def audit_subset(dataset, *, count, seed):
    selected = choose_indices(dataset, count, seed)
    rows, identities = [], {}
    history, plan_ages = [], []
    task_counts, age_counts, lag_counts = Counter(), Counter(), Counter()
    zero_dt_candidates = Counter()
    by_lag_dt = defaultdict(list)
    by_age_dt, by_age_plan, by_task_dt = defaultdict(list), defaultdict(list), defaultdict(list)
    started = time.perf_counter()
    for n, index in enumerate(selected):
        record = dataset.records[index]
        sample = dataset[index]  # deliberately num_workers=0, no GPU
        clock = dataset._clock_for(record)
        source_time = clock_time_seconds(clock, record.source_frame)
        current_time = clock_time_seconds(clock, record.fast_frame)
        first_clock_frame = int(clock['frame_index'].min())
        zero_dt_lags = [lag for lag in HISTORY_RAW_FRAME_LAGS
            if record.fast_frame-lag >= first_clock_frame and
            clock_time_seconds(clock, record.fast_frame-lag) == current_time]
        zero_dt_candidates.update(zero_dt_lags)
        previous_frame = int(sample['history_frame_index'])
        frame_offset = int(sample['history_frame_offset'])
        assert (previous_frame, frame_offset, sample['history_dt']) == causal_history_frame(
            record, clock, seed=dataset.history_seed)
        age = int(sample['delta_actions'])
        assert record.fast_frame - record.source_frame == age * 3
        assert 0 <= age <= 15 and current_time >= source_time
        assert math.isclose(sample['delta_sim_seconds'], current_time - source_time, abs_tol=1e-7)
        if sample['history_valid']:
            assert 0 <= previous_frame < record.fast_frame
            assert frame_offset in HISTORY_RAW_FRAME_LAGS
            assert record.fast_frame - previous_frame == frame_offset
            previous_time = clock_time_seconds(clock, previous_frame)
            assert previous_time < current_time
            assert math.isclose(sample['history_dt'], current_time - previous_time, abs_tol=1e-7)
            assert sample['previous_head_mask'].any()
            history.append(sample['history_dt'])
            by_age_dt[age].append(sample['history_dt'])
            by_task_dt[record.task_name].append(sample['history_dt'])
            by_lag_dt[frame_offset].append(sample['history_dt'])
            lag_counts[frame_offset] += 1
        else:
            assert sample['history_dt'] == 0
            assert previous_frame == -1 and frame_offset == 0
            assert not sample['previous_head_mask'].any()
        assert tuple(sample['previous_head'].shape) == (3, 160, 160)
        assert tuple(sample['previous_head_mask'].shape) == (160, 160)
        assert torch.isfinite(sample['previous_head']).all()
        assert sample['previous_head_mask'].dtype == torch.bool
        assert torch.isfinite(sample['fast_action_sequence']).all()
        assert tuple(sample['fast_action_sequence'].shape) == (16, 14)
        if record.episode_index not in identities:
            identities[record.episode_index] = clock_identity(dataset, record, clock)
        plan_ages.append(sample['delta_sim_seconds'])
        by_age_plan[age].append(sample['delta_sim_seconds'])
        age_counts[age] += 1
        task_counts[record.task_name] += 1
        rows.append(dict(index=index, episode_index=record.episode_index, task=record.task_name,
            source_sample_id=record.source_sample_id, current_sample_id=record.fast_sample_id,
            source_frame=record.source_frame, current_frame=record.fast_frame,
            previous_frame=previous_frame, age_actions=age, history_valid=sample['history_valid'],
            history_frame_offset=frame_offset,
            zero_dt_candidate_lags=zero_dt_lags,
            history_dt=sample['history_dt'], plan_age_seconds=sample['delta_sim_seconds']))
        if n % 32 == 31:
            print(json.dumps(dict(subset=dataset.subset, read=n+1, total=count)), flush=True)
    assert set(age_counts) == set(range(16))
    return dict(subset=dataset.subset, samples=count, total_pairs=len(dataset),
        index_bytes=dataset.index_bytes, total_age_counts=dataset.records.age_counts,
        task_counts=dict(task_counts), age_counts=dict(age_counts),
        history_lag_counts=dict(lag_counts),
        zero_dt_candidate_counts=dict(zero_dt_candidates),
        per_lag_history_dt={lag: quantiles(values) for lag, values in by_lag_dt.items()},
        history_valid_count=len(history), history_dt=quantiles(history),
        plan_age_seconds=quantiles(plan_ages),
        per_age={age: dict(history_dt=quantiles(by_age_dt[age]),
                          plan_age_seconds=quantiles(by_age_plan[age])) for age in range(16)},
        per_task_history_dt={task: quantiles(values) for task, values in by_task_dt.items()},
        sampled_pairs=rows, clocks=list(identities.values()),
        seconds=time.perf_counter()-started)


def cpu_contract_checks(dataset):
    """Real helper graph, stubbed sampler: no learned teacher/GPU computation."""
    b = 16
    g = torch.Generator().manual_seed(991)
    batch = dict(source_condition_latent=torch.zeros(b, 2, 1, 2, 2),
        source_state=torch.zeros(b, 14), source_motion=torch.zeros(b, 1, 4),
        source_motion_interval_valid_mask=torch.ones(b, 1, dtype=torch.bool),
        source_motion_acceleration_valid_mask=torch.ones(b, 1, dtype=torch.bool),
        bridge_noise_shape_latent=torch.zeros(b, 2, 2, 2, 2), fast_rgb=torch.zeros(b, 3, 3, 8, 8),
        fast_rgb_valid_mask=torch.ones(b, 3, 8, 8, dtype=torch.bool), fast_state=torch.ones(b, 14),
        fast_motion=torch.zeros(b, 1, 4), fast_motion_interval_valid_mask=torch.ones(b, 1, dtype=torch.bool),
        fast_motion_acceleration_valid_mask=torch.ones(b, 1, dtype=torch.bool),
        fast_action_sequence=torch.randn(b, 16, 14, generator=g),
        fast_future_latent=torch.zeros(b, 2, 2, 2, 2), fast_condition_latent=torch.zeros(b, 2, 1, 2, 2),
        text_embeddings=[torch.zeros(2, 4096) for _ in range(b)], delta_sim_seconds=torch.arange(b).float()/10,
        delta_actions=torch.arange(b).float(), state_delta=torch.ones(b, 14),
        previous_head=torch.ones(b, 3, 8, 8), previous_head_mask=torch.ones(b, 8, 8, dtype=torch.bool),
        history_dt=torch.ones(b)/10, history_valid=torch.ones(b, dtype=torch.bool))
    class Teacher:
        compact_wan = SimpleNamespace(video_model=SimpleNamespace(precision=torch.bfloat16))
        def eval(self):
            return self
    calls = []
    def sample(teacher, source, noise, schedule):
        assert source.condition_latent.dtype == noise.future_latent.dtype == torch.bfloat16
        assert source.state.dtype == noise.actions.dtype == torch.float32
        assert not torch.is_autocast_enabled()
        calls.append(True)
        return noise.actions + source.state[:, None, :] / 10
    cases = {}
    with patch.object(trainer, 'sample_native_wam', sample):
        for mode, dropout in [('temporal', 0.), ('static', 0.), ('temporal', 1.), ('temporal', .15)]:
            key = f'{mode}_{dropout}'
            cases[key] = trainer.prepare_targets(batch, Teacher(), None, torch.tensor([1000., 0.]),
                torch.tensor([1., 0.]), 'cpu', generator=torch.Generator().manual_seed(121),
                history_mode=mode, history_dropout=dropout)
    reference = cases['temporal_0.0']
    assert reference[0]['history_valid'].all()
    assert not cases['static_0.0'][0]['history_valid'].any()
    assert not cases['temporal_1.0'][0]['history_valid'].any()
    assert cases['temporal_0.15'][0]['history_valid'].dtype == torch.bool
    for case in cases.values():
        for left, right in zip(reference[1:], case[1:]):
            assert torch.equal(left, right), 'history intervention changed teacher target'
    inputs, base, target, expert = reference
    ages = inputs['age_actions']
    residual = torch.full_like(base, .013, requires_grad=True)
    loss, metrics = trainer.feedback_objective(residual, base, target, expert, ages,
        limit=.1, expert_weight=.25, l2_weight=.01)
    loss.backward()
    active = ((ages > 0)[:, None, None] &
        (ages[:, None] + torch.arange(4)[None] < 16)[:, :, None]).expand_as(base).clone()
    active[..., [6, 13]] = False
    assert not residual.grad[~active].count_nonzero()
    assert residual.grad[active].abs().sum() > 0
    expected = ((base-target).square()[active]).mean()
    torch.testing.assert_close(metrics['zero_teacher_mse'], expected)
    altered_target = target.clone()
    altered_target[~active] = 1e6
    _, altered = trainer.feedback_objective(residual, base, altered_target, expert, ages,
        limit=.1, expert_weight=.25, l2_weight=.01)
    for key in ('loss', 'zero_teacher_mse', 'feedback_teacher_mse', 'teacher_clip_fraction'):
        torch.testing.assert_close(metrics[key], altered[key])
    base_sampler = TemporalPairSampler(dataset.records, samples_per_episode=10,
        validation=True, seed=881)
    aligned = ProcessGroupAlignedSampler(base_sampler, batch_size=1, num_processes=7)
    global_indices = list(aligned)
    shards = [list(trainer.RankSampler(aligned, rank, 7)) for rank in range(7)]
    assert len(set(map(len, shards))) == 1
    reconstructed = [index for group in zip(*shards) for index in group]
    assert reconstructed == global_indices
    assert {dataset.records[i].offset//3 for i in global_indices[:16]} == set(range(16))
    return dict(passed=True, teacher_stub_calls=len(calls), dtype_contract=True,
        fixed_targets_under_history_intervention=True, all_age_arm_tail_gradient_mask=True,
        metrics_ignore_identity_gripper_tail=True, ranks=7, rank_samples=len(shards[0]),
        rank_interleaving_exact=True, validation_prefix_all_ages=True,
        note='Repeated draws remain possible by design; rank positions shard without padding or omission.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256, help='Samples per subset; at least 256')
    parser.add_argument('--seed', type=int, default=20261006)
    args = parser.parse_args()
    if args.samples < 256:
        parser.error('read at least 256 real pairs per subset')
    if args.output.exists():
        raise FileExistsError('audit reports are immutable; choose a new output path')
    torch.set_num_threads(4)
    started = time.perf_counter()
    config = trainer.load_multirate_profile(args.config).training
    train = trainer.make_dataset(config)
    val = trainer.make_dataset(config, validation=True, train=train)
    train_episodes, val_episodes = set(train.records.groups), set(val.records.groups)
    assert not train_episodes & val_episodes
    assert train.episode_partition == val.episode_partition
    assert train.pairing_identity == val.pairing_identity
    base_path = Path(config['model']['native_teacher_checkpoint'])
    payload = trainer.load_checkpoint_payload(base_path)
    normalizer = payload['config']['action_normalization']
    assert train.packed.action_stats_sha256 == normalizer['stats_sha256']
    assert payload['format'] == 'dynamicwam_absolute_motion_checkpoint'
    del payload
    source_files = ['scripts/check_temporal_feedback_data.py', 'scripts/train_temporal_feedback.py',
        'src/dynamicwam/training/data/temporal_feedback_dataset.py',
        'src/dynamicwam/training/data/paired_world_action_dataset.py',
        'src/dynamicwam/training/data/paired_index.py',
        'src/dynamicwam/models/temporal_plan_feedback.py',
        'src/dynamicwam/models/temporal_feedback_contract.py',
        'src/dynamicwam/training/train/multirate_trainer.py']
    result = dict(format='dynamicwam_temporal_dataset_audit', version=2, passed=False,
        source_root=str(ROOT), source_hashes={name: trainer.sha256(ROOT/name) for name in source_files},
        config=dict(path=str(args.config.resolve()), sha256=trainer.sha256(args.config)),
        dataset=dict(root=str(train.packed.root), source_root=str(train.source_root),
            identity=train.packed.dataset_identity, pairing_identity=train.pairing_identity,
            action_stats_sha256=train.packed.action_stats_sha256,
            history_raw_frame_lags=list(train.history_raw_frame_lags),
            history_selection='pair_identity_sha256_v1', history_seed=train.history_seed,
            metadata_hashes={name: trainer.sha256(train.packed.root/name) for name in
                ('dataset.json', 'action_stats.json', 'motion_stats.json', 'samples.jsonl')},
            train_episode_count=len(train_episodes), validation_episode_count=len(val_episodes),
            episode_overlap=[], episode_partition=train.episode_partition),
        base_checkpoint=dict(path=str(base_path), sha256=trainer.sha256(base_path)),
        action_normalization=normalizer, normalization_match=True, num_workers=0,
        cpu_contracts=cpu_contract_checks(val))
    result['subsets'] = {dataset.subset: audit_subset(dataset, count=args.samples,
        seed=args.seed+(dataset.subset=='validation')) for dataset in (train, val)}
    result.update(passed=True, seconds=time.perf_counter()-started,
        cuda_initialized=torch.cuda.is_initialized())
    assert not result['cuda_initialized']
    train.close()
    val.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, output=str(args.output), seconds=result['seconds'],
        train_pairs=len(train), validation_pairs=len(val))), flush=True)


if __name__ == '__main__':
    main()
