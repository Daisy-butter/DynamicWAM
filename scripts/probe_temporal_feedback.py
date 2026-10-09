#!/usr/bin/env python3
"""Fixed-target, episode-held-out temporal/static feedback feasibility probe.

The frozen official teacher runs once. Both heads receive identical cached
inputs/targets, initialization, minibatch order and optimizer budgets. This is
an offline diagnostic, not evidence of closed-loop success or causality.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FORMAT = 'dynamicwam_temporal_feedback_probe_cache'
VERSION = 3
EXPECTED_INITIALIZATION_SHA256 = 'be393d437a5c750b4642e8f685a86971a86cea07810b8e1a3f357014cb3bc17d'
EXPECTED_HISTORY_LAGS = (1, 2, 3, 6, 9, 12, 18, 24, 36)
DT_EDGES = (0., .08, .16, .32, .64, 1., 2., 4., 8., float('inf'))
PLAN_EDGES = (0., .5, 1., 2., 4., 8., 16., float('inf'))
SOURCE_FILES = (
    'scripts/probe_temporal_feedback.py', 'scripts/train_temporal_feedback.py',
    'src/dynamicwam/models/temporal_plan_feedback.py',
    'src/dynamicwam/models/temporal_feedback_contract.py',
    'src/dynamicwam/training/data/temporal_feedback_dataset.py',
    'src/dynamicwam/models/native_wam_teacher.py',
    'src/dynamicwam/models/flow_schedule.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def episode_key(row):
    meta = row['record']
    return (meta['split'], meta['task_name'], meta['episode_name'])


def assert_disjoint(train, validation):
    overlap = {episode_key(row) for row in train} & {episode_key(row) for row in validation}
    if overlap:
        raise ValueError(f'train/heldout episode leakage: {sorted(overlap)[:5]}')
    if not train or not validation:
        raise ValueError('both probe partitions must contain samples')


def ratio(rows, numerator='sse', denominator='count'):
    count = sum(row[denominator] for row in rows)
    return sum(row[numerator] for row in rows) / count if count else None


def paired_episode_ci(left, right, *, repetitions=2000, seed=20261006):
    """Right minus left, reweighting by active coordinates after episode draws."""
    if len(left) != len(right):
        raise ValueError('paired metric rows differ in length')
    grouped = collections.defaultdict(lambda: [0., 0., 0.])
    for a, b in zip(left, right):
        if a['record_id'] != b['record_id'] or a['count'] != b['count'] or a['episode'] != b['episode']:
            raise ValueError('paired metrics must use the same records and active mask')
        group = grouped[tuple(a['episode'])]
        group[0] += a['sse']; group[1] += b['sse']; group[2] += a['count']
    groups = [g for g in grouped.values() if g[2] > 0]
    if len(groups) < 2:
        return dict(episodes=len(groups), difference=None if not groups else (groups[0][1]-groups[0][0])/groups[0][2],
                    ci95=None, method='paired episode-cluster percentile bootstrap; too few episodes')
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        draw = rng.choices(groups, k=len(groups))
        samples.append(sum(g[1]-g[0] for g in draw)/sum(g[2] for g in draw))
    return dict(episodes=len(groups), difference=sum(g[1]-g[0] for g in groups)/sum(g[2] for g in groups),
                ci95=[quantile(samples, .025), quantile(samples, .975)], repetitions=repetitions, seed=seed,
                method='paired episode-cluster percentile bootstrap; active-coordinate weighted')


def mean_episode_ci(rows, *, seed):
    zero = [dict(row, sse=0.) for row in rows]
    result = paired_episode_ci(zero, rows, seed=seed)
    result['mean'] = result.pop('difference')
    return result


def history_permutation(rows, valid, *, seed):
    """Permute heldout history only; never take a donor from the train cache.

    Keep current observations, plan, state and measured dt fixed. Cross-episode
    image shuffling tests useful association, not a motion-only causal effect.
    """
    indices = [i for i, flag in enumerate(valid) if flag]
    if len({episode_key(rows[i]) for i in indices}) < 2:
        raise ValueError('shuffle control needs valid histories from at least two heldout episodes')
    rng = random.Random(seed)
    best, score = indices[:], -1
    for _ in range(200):
        trial = indices[:]; rng.shuffle(trial)
        current = sum(episode_key(rows[i]) != episode_key(rows[j]) for i, j in zip(indices, trial))
        if current > score:
            best, score = trial, current
        if score == len(indices):
            break
    permutation = list(range(len(rows)))
    for i, j in zip(indices, best):
        permutation[i] = j
    return permutation, dict(valid_rows=len(indices), changed_episode_rows=score,
                             changed_episode_fraction=score/len(indices), seed=seed,
                             scope='heldout valid histories; previous RGB/mask only; dt remains measured recipient dt')


def gate_result(static, fresh, shuffled, zero, initial, *, minimum_gain=.01):
    if any(v is None or not math.isfinite(v) for v in (static, fresh, shuffled, zero, initial)):
        return dict(passed=False, reason='nonfinite or empty active heldout metric')
    gain = (static-fresh)/static if static > 0 else None
    improvement = gain is not None and gain >= minimum_gain
    shuffle = fresh <= shuffled
    return dict(passed=bool(improvement and shuffle and fresh <= zero and fresh <= initial), relative_arm_mse_gain=gain,
                required_relative_gain=minimum_gain, temporal_beats_static=bool(improvement),
                fresh_not_worse_than_shuffled=bool(shuffle), static_target_mse=static,
                fresh_target_mse=fresh, shuffled_target_mse=shuffled,
                fresh_not_worse_than_zero=bool(fresh <= zero), fresh_not_worse_than_initial=bool(fresh <= initial),
                zero_target_mse=zero, initial_target_mse=initial,
                scope='point-estimate feasibility gate, not a significance or closed-loop noninferiority claim')


def bin_name(value, edges):
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f'[{low:g},{high:g})'
    return 'outside_bins'


def deployment_contract_identity(contract, *, checkpoint_format, checkpoint_version):
    """A changed model/initialization needs its own preserved diagnostic cache."""
    if (checkpoint_version != 3 or tuple(contract.get('history_training_raw_lags', ())) != EXPECTED_HISTORY_LAGS
            or contract.get('history_sampling') != 'pair_identity_sha256_v1'
            or contract.get('plan_age_seconds') != 'telemetry_only'
            or contract.get('architecture') != 'pretrained_static_head_additive_zero_temporal_v3'
            or contract.get('age_initialization') != 'legacy_0_4_8_12_exact_linear_between_hold_after12'):
        raise ValueError('probe requires contract v3: pretrained additive temporal head and real multi-lag history')
    return dict(format=checkpoint_format, version=checkpoint_version,
                history_training_raw_lags=list(EXPECTED_HISTORY_LAGS),
                history_sampling=contract['history_sampling'], plan_age_seconds=contract['plan_age_seconds'],
                architecture=contract['architecture'], age_initialization=contract['age_initialization'])


def initialization_file_identity(path):
    identity = dict(path=str(Path(path).resolve()), sha256=sha256(path))
    if identity['sha256'] != EXPECTED_INITIALIZATION_SHA256:
        raise ValueError('probe must initialize from the preregistered trained v1 checkpoint SHA')
    return identity


def validate_fixed_budget(args):
    expected = dict(train_samples=512, validation_samples=256, teacher_batch_size=1,
                    batch_size=32, epochs=20, lr=2e-5, seed=20261006,
                    expert_weight=.25, residual_l2=.01, history_dropout=.15)
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError(f'fixed v3 development probe budget mismatch: {key}; expected {value}')


def initialize_head_from_cache(args, cache, *, trainer=None):
    import torch
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
    trainer = trainer or importlib.import_module('train_temporal_feedback')
    expected = initialization_file_identity(args.init_plan_feedback)
    if cache['metadata']['initialization_checkpoint'] != expected:
        raise ValueError('initialization checkpoint identity differs from cache generation')
    payload = torch.load(args.init_plan_feedback, map_location='cpu', weights_only=True, mmap=True)
    model = TemporalPlanFeedbackHead()
    config = trainer.load_multirate_profile(args.config).training
    trainer.validate_plan_initialization(payload,
        base_identity=cache['metadata']['base_checkpoint'],
        dataset_identity=cache['metadata']['dataset_identity'],
        legacy_pairing_identity=cache['metadata']['legacy_pairing_identity'],
        normalization=cache['metadata']['action_normalization'],
        schedule=trainer.BridgeSchedule.from_training_config(config), spec=model.constructor_spec)
    mapping = model.initialize_from_plan_feedback(payload['model'])
    return model, dict(kind='function_preserving_plan_feedback_v1', **expected,
                       source_format=payload['format'], source_version=payload['version'],
                       source_global_step=payload['global_step'], mapping=mapping)


def source_hashes():
    return {name: sha256(ROOT/name) for name in SOURCE_FILES}


def record_metadata(record, index):
    fields = ('episode_index', 'source_sample_id', 'fast_sample_id', 'source_frame', 'fast_frame',
              'lang_group_id', 'lang_id', 'offset', 'offset_label', 'split', 'task_name', 'episode_name')
    result = {key: getattr(record, key) for key in fields}
    result['dataset_index'] = int(index)
    return result


def cache_coverage(rows, task_universe=None):
    result = dict(samples=len(rows), episodes=len({episode_key(r) for r in rows}),
                  tasks=dict(collections.Counter(r['record']['task_name'] for r in rows)),
                  ages=dict(collections.Counter(str(int(r['inputs']['age_actions'])) for r in rows)),
                  history_valid=sum(bool(r['inputs']['history_valid']) for r in rows))
    tasks=sorted(task_universe if task_universe is not None else result['tasks'])
    result['task_universe']=tasks
    result['missing_tasks']=[task for task in tasks if task not in result['tasks']]
    result['task_age_counts']={task:{str(age):sum(r['record']['task_name']==task
        and int(r['inputs']['age_actions'])==age for r in rows) for age in range(16)} for task in tasks}
    result['empty_task_age_cells']=sum(count==0 for ages in result['task_age_counts'].values() for count in ages.values())
    lags=(0,)+EXPECTED_HISTORY_LAGS
    result['history_raw_frame_lags']={str(lag):sum(r['record']['history_frame_offset']==lag for r in rows) for lag in lags}
    result['history_lag_by_age']={str(age):{
        str(lag):sum(int(r['inputs']['age_actions'])==age and r['record']['history_frame_offset']==lag for r in rows)
        for lag in lags} for age in range(16)}
    result['history_lag_by_task']={task:{str(lag):sum(
        r['record']['task_name']==task and r['record']['history_frame_offset']==lag for r in rows)
        for lag in lags} for task in tasks}
    for field in ('history_dt', 'plan_age_seconds'):
        values=[float(r['inputs'][field]) for r in rows]
        result[field]=dict(min=min(values),p50=quantile(values,.5),p95=quantile(values,.95),max=max(values))
    valid_dt=[float(r['inputs']['history_dt']) for r in rows if bool(r['inputs']['history_valid'])]
    result['history_dt_valid_only']=dict(count=len(valid_dt),min=min(valid_dt) if valid_dt else None,
        p50=quantile(valid_dt,.5),p95=quantile(valid_dt,.95),max=max(valid_dt) if valid_dt else None)
    return result


def cache_inputs(inputs, sample, texts):
    import torch
    result = {}
    for key, value in inputs.items():
        if key in {'text', 'text_mask'}:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'cached head input {key} must be a batched tensor')
        tensor = value[sample].detach().cpu().contiguous()
        if key in {'rgb', 'previous_head'}:
            tensor = tensor.half()
        result[key] = tensor.clone()
    mask = inputs['text_mask'][sample].detach().cpu().bool()
    active = mask.nonzero().flatten()
    length = int(active[-1]) + 1 if len(active) else 1
    text = inputs['text'][sample, :length].detach().cpu().contiguous()
    mask = mask[:length].contiguous()
    identity = hashlib.sha256(str((text.dtype, tuple(text.shape))).encode()
                              + text.view(torch.uint8).numpy().tobytes()+mask.numpy().tobytes()).hexdigest()
    if identity not in texts:
        texts[identity] = dict(text=text.clone(), text_mask=mask.clone())
    return result, identity


def collect_cache_split(dataset, config, trainer, args, teacher, schedule, timesteps, sigmas, device, *, validation, texts):
    import torch
    from types import SimpleNamespace
    loader_args = SimpleNamespace(batch_size=args.teacher_batch_size, workers=0, seed=args.seed,
                                 validation_batches=math.ceil(args.validation_samples/args.teacher_batch_size))
    loader, sampler = trainer.make_loader(dataset, config, loader_args, 0, 1, validation=validation)
    sampler.set_epoch(0)
    indices = iter(sampler)
    wanted = args.validation_samples if validation else args.train_samples
    generator = torch.Generator(device=device).manual_seed(args.seed + (10000 if validation else 0))
    result, seen = [], set()
    for batch in loader:
        current = list(itertools.islice(indices, args.teacher_batch_size))
        if len(current) != args.teacher_batch_size:
            raise RuntimeError('cache sampler and loader became misaligned')
        inputs, base, target, expert = trainer.prepare_targets(batch, teacher, schedule, timesteps, sigmas,
                                                              device, generator=generator)
        for i, index in enumerate(current):
            if index in seen:
                continue
            meta = record_metadata(dataset.records[index], index)
            meta.update(history_frame_offset=int(batch['history_frame_offset'][i]),
                        history_frame_index=int(batch['history_frame_index'][i]))
            valid=bool(inputs['history_valid'][i])
            if valid:
                if (meta['history_frame_offset'] not in EXPECTED_HISTORY_LAGS
                        or meta['history_frame_index'] < 0
                        or meta['fast_frame']-meta['history_frame_index'] != meta['history_frame_offset']
                        or float(inputs['history_dt'][i]) <= 0):
                    raise ValueError('cached temporal history disagrees with real causal frame metadata')
            elif (meta['history_frame_index'],meta['history_frame_offset'],float(inputs['history_dt'][i])) != (-1,0,0.):
                raise ValueError('missing history must have sentinel frame -1, lag 0 and dt 0')
            cached, text_id = cache_inputs(inputs, i, texts)
            result.append(dict(record=meta, record_id=json_hash(meta), inputs=cached, text_id=text_id,
                               base=base[i].detach().cpu().float().clone(),
                               teacher=target[i].detach().cpu().float().clone(),
                               expert=expert[i].detach().cpu().float().clone()))
            seen.add(index)
            if len(result) == wanted:
                return result
        if len(result) % 32 < args.teacher_batch_size:
            print(json.dumps(dict(kind='probe_cache', partition='heldout' if validation else 'train',
                                  samples=len(result), requested=wanted)), flush=True)
    raise RuntimeError(f'probe sampler supplied {len(result)} unique records, need {wanted}')


def build_cache(args, cache_path):
    import torch
    trainer = importlib.import_module('train_temporal_feedback')
    config = trainer.load_multirate_profile(args.config).training
    schedule = trainer.BridgeSchedule.from_training_config(config)
    contract = trainer.runtime_contract(schedule, .1)
    deployment_identity=deployment_contract_identity(contract,checkpoint_format=trainer.FORMAT,
                                                     checkpoint_version=trainer.VERSION)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('official teacher cache creation needs CUDA; --head-only supports CPU')
    torch.cuda.set_device(device)
    trainer.seed_all(args.seed)
    train = trainer.make_dataset(config)
    validation = trainer.make_dataset(config, validation=True, train=train)
    if tuple(train.history_raw_frame_lags)!=EXPECTED_HISTORY_LAGS or tuple(validation.history_raw_frame_lags)!=EXPECTED_HISTORY_LAGS:
        raise ValueError('dataset multi-lag sampling differs from the declared deployment contract')
    base_path = Path(config['model']['native_teacher_checkpoint']).resolve()
    payload = trainer.load_checkpoint_payload(base_path)
    normalization = payload['config']['action_normalization']
    cfg = config['dataset']
    legacy_pairing = trainer.make_pairing_identity(list(train.records.episodes.values()),
        train.episode_partition, seed=cfg['partition_seed'], validation_fraction=cfg['validation_fraction'],
        clock_mode=cfg['clock_mode'], offsets=[age*cfg['global_downsample_rate'] for age in (0,4,8,12)],
        stress_offsets=())
    initial_metadata = dict(initialization_checkpoint=initialization_file_identity(args.init_plan_feedback),
        base_checkpoint=dict(path=str(base_path),sha256=sha256(base_path)),
        dataset_identity=train.packed.dataset_identity, legacy_pairing_identity=legacy_pairing,
        action_normalization=normalization)
    initial_head, initialization = initialize_head_from_cache(args, dict(metadata=initial_metadata), trainer=trainer)
    del initial_head
    teacher = trainer.build_teacher(config, payload, train, device)
    del payload
    flow = trainer.build_training_flow_scheduler(config)
    timesteps = flow.timesteps.to(device=device, dtype=torch.float32)
    sigmas = flow.sigmas.to(device=device, dtype=torch.float32)
    texts = {}
    train_rows = collect_cache_split(train, config, trainer, args, teacher, schedule, timesteps, sigmas,
                                    device, validation=False, texts=texts)
    val_rows = collect_cache_split(validation, config, trainer, args, teacher, schedule, timesteps, sigmas,
                                  device, validation=True, texts=texts)
    assert_disjoint(train_rows, val_rows)
    metadata = dict(format=FORMAT, version=VERSION, seed=args.seed,
                    teacher_batch_size=args.teacher_batch_size, train_samples=args.train_samples,
                    validation_samples=args.validation_samples, config_path=str(args.config.resolve()),
                    config_sha256=sha256(args.config), resolved_config_sha256=json_hash(config),
                    source_hashes=source_hashes(), dataset_identity=train.packed.dataset_identity,
                    pairing_identity=train.pairing_identity,
                    initialization_checkpoint=initial_metadata['initialization_checkpoint'],
                    initialization=initialization, legacy_pairing_identity=legacy_pairing,
                    action_normalization=normalization, runtime_contract=contract,
                    deployment_contract=deployment_identity,
                    history_partition_seed=int(config['dataset']['partition_seed']),
                    base_checkpoint=initial_metadata['base_checkpoint'],
                    split=dict(train_episodes=len({episode_key(r) for r in train_rows}),
                               heldout_episodes=len({episode_key(r) for r in val_rows}), overlap=0,
                               train_record_sha256=json_hash([r['record'] for r in train_rows]),
                               heldout_record_sha256=json_hash([r['record'] for r in val_rows])),
                    storage='RGB/current+previous cached FP16 and restored FP32 for both arms; targets/plan/state FP32; text exact deduplicated',
                    teacher_target_noise='one fixed draw per cached pair; no re-sampling between arms or epochs',
                    budget=dict(teacher_pairs=len(train_rows)+len(val_rows), text_entries=len(texts)))
    task_universe=sorted({value['task_name'] for value in train.records.episodes.values()
                          if value['split'] in config['dataset']['splits']})
    metadata['coverage']=dict(train=cache_coverage(train_rows,task_universe),heldout=cache_coverage(val_rows,task_universe))
    cache = dict(metadata=metadata, train=train_rows, validation=val_rows, texts=texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix('.tmp')
    torch.save(cache, temporary); temporary.replace(cache_path)
    manifest = dict(metadata, cache_sha256=sha256(cache_path), cache_bytes=cache_path.stat().st_size)
    cache_path.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    train.close(); validation.close()
    del teacher, train, validation
    torch.cuda.empty_cache()
    return cache


def load_cache(args, path):
    import torch
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(path) != manifest['cache_sha256']:
        raise ValueError('probe cache content hash changed')
    for key, expected in dict(format=FORMAT, version=VERSION, seed=args.seed,
                              train_samples=args.train_samples, validation_samples=args.validation_samples,
                              teacher_batch_size=args.teacher_batch_size,
                              initialization_checkpoint=initialization_file_identity(args.init_plan_feedback),
                              config_sha256=sha256(args.config), source_hashes=source_hashes()).items():
        if manifest.get(key) != expected:
            raise ValueError(f'probe cache provenance mismatch: {key}')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    from dynamicwam.models.temporal_feedback_contract import FORMAT as CHECKPOINT_FORMAT, VERSION as CHECKPOINT_VERSION
    identity=deployment_contract_identity(cache['metadata']['runtime_contract'],
        checkpoint_format=CHECKPOINT_FORMAT,checkpoint_version=CHECKPOINT_VERSION)
    if cache['metadata'].get('deployment_contract')!=identity:
        raise ValueError('cache was not produced under the current v3 initialization/history contract')
    for key, value in cache['metadata'].items():
        if manifest.get(key) != value:
            raise ValueError(f'cache and manifest disagree: {key}')
    assert_disjoint(cache['train'], cache['validation'])
    return cache


def collate_cache(rows, texts, device, *, history='fresh', donors=None):
    import torch
    inputs = {key: torch.stack([row['inputs'][key] for row in rows]).to(device)
              for key in rows[0]['inputs']}
    for key in ('rgb', 'previous_head'):
        inputs[key] = inputs[key].float()
    if history == 'static':
        inputs['history_valid'] = torch.zeros_like(inputs['history_valid'])
    elif history == 'shuffled':
        if donors is None or len(donors) != len(rows):
            raise ValueError('shuffled inference needs matched donor rows')
        for key in ('previous_head', 'previous_head_mask'):
            inputs[key] = torch.stack([row['inputs'][key] for row in donors]).to(device)
        inputs['previous_head'] = inputs['previous_head'].float()
    elif history != 'fresh':
        raise ValueError('unknown history intervention')
    entries = [texts[row['text_id']] for row in rows]
    length = max(len(entry['text']) for entry in entries)
    first = entries[0]['text']
    inputs['text'] = torch.zeros(len(rows), length, first.shape[-1], dtype=first.dtype, device=device)
    inputs['text_mask'] = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
    for i, entry in enumerate(entries):
        n = len(entry['text'])
        inputs['text'][i, :n] = entry['text'].to(device)
        inputs['text_mask'][i, :n] = entry['text_mask'].to(device)
    targets = {key: torch.stack([row[key] for row in rows]).to(device=device, dtype=torch.float32)
               for key in ('base', 'teacher', 'expert')}
    return inputs, targets


def reachable_target(targets, ages, *, limit, expert_weight):
    import torch
    td = targets['teacher']-targets['base']
    ed = targets['expert']-targets['base']
    target = (td.clamp(-limit, limit)+expert_weight*ed.clamp(-limit, limit))/(1+expert_weight)
    tail = ages[:, None]+torch.arange(4, device=ages.device)[None, :] < 16
    active = (ages[:, None, None] > 0) & tail[:, :, None]
    arm = torch.ones(14, dtype=torch.bool, device=ages.device); arm[[6, 13]] = False
    mask = active & arm[None, None, :]
    return target, mask, td, ed


def metric_rows(residual, targets, inputs, rows, *, limit, expert_weight):
    import torch
    target, mask, td, ed = reachable_target(targets, inputs['age_actions'], limit=limit, expert_weight=expert_weight)
    pred = residual.float()
    def total(value):
        return (value*mask).flatten(1).sum(1).detach().cpu().tolist()
    columns = dict(sse=total((pred-target).square()), count=total(torch.ones_like(pred)),
                   teacher_sse=total((pred-td.clamp(-limit,limit)).square()),
                   expert_sse=total((pred-ed.clamp(-limit,limit)).square()),
                   unclipped_teacher_sse=total((pred-td).square()), unclipped_expert_sse=total((pred-ed).square()),
                   target_sum=total(target), target_sumsq=total(target.square()),
                   prediction_sum=total(pred), prediction_sumsq=total(pred.square()),
                   teacher_clipped=total((td.abs()>limit).float()), expert_clipped=total((ed.abs()>limit).float()),
                   sign_conflict=total((td*ed<0).float()), abs_residual_sum=total(pred.abs()))
    result=[]
    for i, row in enumerate(rows):
        result.append(dict(record_id=row['record_id'], episode=list(episode_key(row)),
                           age=int(inputs['age_actions'][i]), history_valid=bool(row['inputs']['history_valid']),
                           task_name=row['record']['task_name'],
                           history_frame_offset=row['record']['history_frame_offset'],
                           history_frame_index=row['record']['history_frame_index'],
                           history_dt=float(inputs['history_dt'][i]), plan_age_seconds=float(inputs['plan_age_seconds'][i]),
                           **{key:value[i] for key,value in columns.items()}))
    return result


def summarize_metrics(rows):
    mse = ratio(rows)
    mean = ratio(rows, 'target_sum')
    pred_mean = ratio(rows, 'prediction_sum')
    return dict(rows=len(rows), active_coordinates=sum(r['count'] for r in rows), target_mse=mse,
                zero_target_mse=ratio(rows,'target_sumsq'),
                teacher_clipped_mse=ratio(rows,'teacher_sse'), expert_clipped_mse=ratio(rows,'expert_sse'),
                teacher_unclipped_mse=ratio(rows,'unclipped_teacher_sse'), expert_unclipped_mse=ratio(rows,'unclipped_expert_sse'),
                target_mean=mean, target_variance=None if mean is None else max(0.,ratio(rows,'target_sumsq')-mean**2),
                prediction_variance=None if pred_mean is None else max(0.,ratio(rows,'prediction_sumsq')-pred_mean**2),
                mean_abs_residual=ratio(rows,'abs_residual_sum'), teacher_clip_fraction=ratio(rows,'teacher_clipped'),
                expert_clip_fraction=ratio(rows,'expert_clipped'), target_sign_conflict_fraction=ratio(rows,'sign_conflict'),
                variance_scope='across cached active coordinates; not conditional teacher sampling variance')


def stratified_metrics(rows):
    groups = dict(all=rows, tail_ages13_15=[r for r in rows if r['age'] >= 13],
                  legacy_ages0_4_8_12=[r for r in rows if r['age'] in (0,4,8,12)],
                  valid_history=[r for r in rows if r['history_valid']],
                  invalid_history=[r for r in rows if not r['history_valid']])
    for age in range(16):
        groups[f'age_{age}'] = [r for r in rows if r['age']==age]
    for lag in (0,)+EXPECTED_HISTORY_LAGS:
        groups[f'history_raw_lag_{lag}']=[r for r in rows if r['history_frame_offset']==lag]
    for task in sorted({row['task_name'] for row in rows}):
        groups[f'task_{task}']=[r for r in rows if r['task_name']==task]
    for name, field, edges in [('history_dt','history_dt',DT_EDGES),('physical_plan_age','plan_age_seconds',PLAN_EDGES)]:
        for label in [f'[{low:g},{high:g})' for low,high in zip(edges,edges[1:])]:
            groups[f'{name}_{label}']=[r for r in rows if bin_name(r[field],edges)==label]
    return {key:summarize_metrics(values) for key,values in groups.items()}


def train_head(model, cache, args, device, *, history):
    import torch
    trainer=importlib.import_module('train_temporal_feedback')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    trace=[]; rows=cache['train']; model.train()
    for epoch in range(args.epochs):
        order=list(range(len(rows))); random.Random(args.seed+epoch).shuffle(order)
        numerator=0.; count=0
        for start in range(0,len(order),args.batch_size):
            selected=[rows[i] for i in order[start:start+args.batch_size]]
            inputs,targets=collate_cache(selected,cache['texts'],device,history=history)
            # Draw for both arms to keep RNG consumption matched. Static masks
            # remain false; temporal training uses the formal dropout contract.
            keep=torch.rand(len(selected),device=device)>=args.history_dropout
            inputs['history_valid']=inputs['history_valid'] & keep
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                residual=model(**inputs)
            loss,_=trainer.feedback_objective(residual,targets['base'],targets['teacher'],targets['expert'],
                inputs['age_actions'],limit=model.residual_limit,expert_weight=args.expert_weight,l2_weight=args.residual_l2)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite probe loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            numerator+=float(loss.detach())*len(selected);count+=len(selected)
        trace.append(dict(epoch=epoch+1,mean_training_objective=numerator/count,history=history))
        print(json.dumps(dict(kind='probe_train',**trace[-1])),flush=True)
    return trace


def evaluate(model, cache, args, device, *, history, permutation=None, zero=False, prediction_path=None):
    import torch
    model.eval(); result=[]; rows=cache['validation']; raw=[]
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            selected=rows[start:start+args.batch_size]
            donors=None if permutation is None else [rows[j] for j in permutation[start:start+args.batch_size]]
            inputs,targets=collate_cache(selected,cache['texts'],device,history=history,donors=donors)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                residual=torch.zeros_like(targets['base']) if zero else model(**inputs)
            if prediction_path is not None:
                raw.append(residual.detach().cpu().float())
            result+=metric_rows(residual,targets,inputs,selected,limit=model.residual_limit,expert_weight=args.expert_weight)
    if prediction_path is not None:
        torch.save(dict(record_ids=[row['record_id'] for row in rows],
                        residual=torch.cat(raw), ages=torch.tensor([row['age'] for row in result]),
                        legacy_ages=[0,4,8,12], initialization_checkpoint=cache['metadata']['initialization_checkpoint'],
                        description='Untrained migrated head predictions; fixed final input rounding as both trained arms'),
                   prediction_path)
    return result


def run_heads(args, cache, cache_path):
    import torch
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
    from dynamicwam.models.temporal_feedback_contract import FORMAT as CHECKPOINT_FORMAT, VERSION as CHECKPOINT_VERSION
    device=torch.device(args.device)
    torch.manual_seed(args.seed)
    initial,initialization=initialize_head_from_cache(args,cache)
    state={key:value.detach().cpu().clone() for key,value in initial.state_dict().items()}
    init_path=args.run_dir/'initial_state.pt';torch.save(state,init_path)
    spec=initial.constructor_spec
    valid=[bool(row['inputs']['history_valid']) for row in cache['validation']]
    permutation,shuffle_info=history_permutation(cache['validation'],valid,seed=args.seed+33000)
    initial=initial.to(device=device,dtype=torch.float32)
    initial_prediction_path=args.run_dir/'initial_predictions.pt'
    predictions=dict(initial=evaluate(initial,cache,args,device,history='fresh',prediction_path=initial_prediction_path),
                     zero=evaluate(initial,cache,args,device,history='fresh',zero=True))
    initial_static=evaluate(initial,cache,args,device,history='static')
    max_initial_metric_difference=max(abs(a['sse']-b['sse']) for a,b in zip(predictions['initial'],initial_static))
    if max_initial_metric_difference != 0.:
        raise ValueError('zero additive branches must preserve the same initial fresh/static predictions')
    del initial
    traces={}
    for name,history in [('static','static'),('temporal','fresh')]:
        torch.manual_seed(args.seed)
        model=TemporalPlanFeedbackHead(**spec).to(device=device,dtype=torch.float32)
        model.load_state_dict(state,strict=True)
        traces[name]=train_head(model,cache,args,device,history=history)
        predictions[name]=evaluate(model,cache,args,device,history=history)
        if name=='temporal':predictions['shuffled']=evaluate(model,cache,args,device,history='shuffled',permutation=permutation)
        torch.save(dict(format=CHECKPOINT_FORMAT,version=CHECKPOINT_VERSION,spec=spec,
                        model={key:value.detach().cpu() for key,value in model.state_dict().items()},
                        base_checkpoint=cache['metadata']['base_checkpoint'],
                        action_normalization=cache['metadata']['action_normalization'],
                        runtime_contract=cache['metadata']['runtime_contract'],
                        dataset_identity=cache['metadata']['dataset_identity'],
                        pairing_identity=cache['metadata']['pairing_identity'],
                        phase='analysis_probe',global_step=args.epochs*math.ceil(len(cache['train'])/args.batch_size),
                        cache_sha256=sha256(cache_path),source_hashes=source_hashes(),
                        initial_state_sha256=sha256(init_path),arm=name,
                        initialization=initialization,
                        usage='fixed-budget probe and small pretraining canary only; not a formal stage2 result'),
                   args.run_dir/f'probe_{name}.pt')
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    metrics={key:stratified_metrics(rows) for key,rows in predictions.items()}
    gate=gate_result(metrics['static']['all']['target_mse'],metrics['temporal']['all']['target_mse'],
                     metrics['shuffled']['all']['target_mse'],metrics['zero']['all']['target_mse'],
                     metrics['initial']['all']['target_mse'],minimum_gain=.01)
    report=dict(format='dynamicwam_temporal_feedback_probe_report',version=1,gate=gate,
                created_unix=time.time(),cache_path=str(cache_path.resolve()),cache_sha256=sha256(cache_path),
                cache_metadata=cache['metadata'],initial_state_sha256=sha256(init_path),spec=spec,
                initialization=initialization, initial_fresh_static_max_metric_difference=max_initial_metric_difference,
                initial_predictions=dict(path=str(initial_prediction_path.resolve()),sha256=sha256(initial_prediction_path)),
                deployment_checkpoint=dict(path=str((args.run_dir/'probe_temporal.pt').resolve()),
                                           sha256=sha256(args.run_dir/'probe_temporal.pt')),
                args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                target='(clipped teacher delta + expert_weight * clipped expert delta) / (1 + expert_weight); excludes L2',
                mask='age_actions>0 AND action slot before plan end AND arm dimensions excluding grippers 6,13',
                metrics=metrics,shuffle=shuffle_info,
                comparisons=dict(temporal_minus_static=paired_episode_ci(predictions['static'],predictions['temporal'],seed=args.seed),
                                 fresh_minus_shuffled=paired_episode_ci(predictions['shuffled'],predictions['temporal'],seed=args.seed+1),
                                 temporal_minus_zero=paired_episode_ci(predictions['zero'],predictions['temporal'],seed=args.seed+2),
                                 temporal_minus_initial=paired_episode_ci(predictions['initial'],predictions['temporal'],seed=args.seed+3)),
                mean_confidence_intervals={name:dict(all=mean_episode_ci(rows,seed=args.seed+10),
                    tail_ages13_15=mean_episode_ci([row for row in rows if row['age']>=13],seed=args.seed+11),
                    legacy_ages0_4_8_12=mean_episode_ci([row for row in rows if row['age'] in (0,4,8,12)],seed=args.seed+12))
                    for name,rows in predictions.items()},
                training_trace=traces,raw_heldout_rows=predictions,
                limitations=['one fixed target-noise draw per pair, no estimate of conditional teacher variance',
                             'cross-episode history shuffle is an association check, not causal motion isolation',
                             'heldout offline error and point gate do not establish closed-loop success or noninferiority',
                             '256 heldout pairs cannot fill all 35 tasks x 16 age cells; task-age/lag coverage includes empty cells',
                             'fixed final epoch only; no heldout checkpoint selection or repeated hyperparameter search'])
    report['development_series'] = 'v1 single-lag and v2 multi-lag failures preserved; v3 tests fixed pretrained initialization. These are sequential development probes, not independent confirmatory experiments.'
    (args.run_dir/'probe_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(kind='probe_complete',gate=gate,report=str(args.run_dir/'probe_report.json'))),flush=True)
    return 0 if gate['passed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--init-plan-feedback',type=Path,required=True)
    parser.add_argument('--train-samples',type=int,default=512)
    parser.add_argument('--validation-samples',type=int,default=256)
    parser.add_argument('--teacher-batch-size',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--expert-weight',type=float,default=.25)
    parser.add_argument('--residual-l2',type=float,default=.01)
    parser.add_argument('--history-dropout',type=float,default=.15)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--cache-only',action='store_true')
    mode.add_argument('--head-only',action='store_true')
    args=parser.parse_args()
    try:
        validate_fixed_budget(args)
        initialization_file_identity(args.init_plan_feedback)
    except ValueError as error:
        parser.error(str(error))
    if min(args.train_samples,args.validation_samples,args.teacher_batch_size,args.batch_size,args.epochs)<1 or args.lr<=0:
        parser.error('probe budgets and learning rate must be positive')
    if args.expert_weight<0 or args.residual_l2<0:parser.error('objective weights must be nonnegative')
    if args.history_dropout!=.15:parser.error('the shared temporal runtime contract requires history dropout .15')
    args.run_dir=args.run_dir.resolve();args.run_dir.mkdir(parents=True,exist_ok=True)
    if (args.run_dir/'probe_report.json').exists():raise FileExistsError('use a fresh probe run directory')
    cache_path=args.run_dir/'fixed_targets.pt'
    if args.head_only:
        cache=load_cache(args,cache_path)
    else:
        if cache_path.exists():raise FileExistsError('existing cache requires explicit --head-only')
        cache=build_cache(args,cache_path)
    if args.cache_only:return 0
    return run_heads(args,cache,cache_path)


if __name__=='__main__':
    raise SystemExit(main())
