#!/usr/bin/env python3
"""Immutable temporal-feedback experiment; gates precede costly training.

Preparation failures leave an arm incomplete, never count as failed episodes,
and never prevent the other formal arms from running. No implicit retries,
seed replacement, checkpoint selection, or resume is supported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dynamicwam.config import load_profile
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.evaluation.run_domino_level1_suite import load_episode_seed_manifest, valid_task_aggregate, write_json
from make_fixed_seed_domino_eval import generate
from run_native_pipeline import assert_gpus_idle, sha256, source_identity

TASKS = ('place_shoe', 'place_fan', 'adjust_bottle')
PACKET = Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/feedback_preflight_20261003/synthetic_packet.npz')
TRAIN_FILES = ('scripts/train_temporal_feedback.py', 'src/dynamicwam/models/temporal_plan_feedback.py',
    'src/dynamicwam/models/temporal_feedback_contract.py',
    'src/dynamicwam/training/data/temporal_feedback_dataset.py',
    'src/dynamicwam/training/data/paired_world_action_dataset.py',
    'src/dynamicwam/models/flow_schedule.py', 'src/dynamicwam/models/native_wam_teacher.py',
    'src/dynamicwam/models/causal_world_bridge.py')


class EvaluationIncomplete(RuntimeError):
    """Missing preparation/output does not measure policy success."""


def snapshot_identity(root):
    identity = source_identity(root)
    for path in sorted((ROOT/'tests').rglob('*.py')):
        identity['files'][path.relative_to(ROOT).as_posix()] = sha256(path)
    identity['sha256'] = hashlib.sha256(json.dumps(identity['files'], sort_keys=True).encode()).hexdigest()
    return identity


def literal_constant(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f'missing literal {name} in {path}')


def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measured(series):
    values = series.get('samples_ms', [])
    if len(values) < 20 or any(not math.isfinite(float(v)) or v <= 0 for v in values):
        raise RuntimeError('latency requires twenty finite positive raw samples')
    mean, p95 = sum(values) / len(values), percentile(values, .95)
    for key, value in (('mean_ms', mean), ('p95_ms', p95)):
        if not math.isclose(value, series.get(key, float('nan')), rel_tol=1e-6):
            raise RuntimeError(f'reported latency {key} differs from raw samples')
    return mean, p95


def same_path(left, right):
    """Allow directory aliases when verifying identity, never rewrite a launcher."""
    if not left or not right:
        return False
    left, right = Path(left).absolute(), Path(right).absolute()
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def runtime_identity_evidence(path, executable):
    rows = []
    for line in Path(path).read_text(errors='replace').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'executable' in value:
            rows.append(value)
    if len(rows) != 1:
        raise RuntimeError('expected one unambiguous deployment interpreter identity')
    result = rows[0]
    if not same_path(result.get('executable'), executable) or result.get('cuda_initialized') is not False:
        raise RuntimeError('deployment identity probe used another launcher or initialized CUDA')
    if any(not result.get(key) for key in ('python', 'torch', 'cuda', 'prefix')):
        raise RuntimeError('deployment identity probe omitted required runtime fields')
    return result


def runtime_evidence(report, expected):
    if expected is None:
        raise RuntimeError('latency gate requires an independently probed DOMINO deployment interpreter')
    if not isinstance(report, dict):
        raise RuntimeError('latency profile omitted deployment runtime identity')
    for key in ('python', 'torch', 'cuda', 'cudnn'):
        if key not in report or report[key] != expected[key]:
            raise RuntimeError(f'latency used a different deployment runtime: {key}')
    for key in ('executable', 'prefix'):
        if not same_path(report.get(key), expected.get(key)):
            raise RuntimeError(f'latency used a different deployment runtime: {key}')
    capability = report.get('gpu_capability')
    if (not report.get('gpu_name') or not isinstance(capability, list) or len(capability) != 2
            or any(type(v) is not int or v < 0 for v in capability)):
        raise RuntimeError('latency profile omitted the actual CUDA device identity')
    return dict(report)


def latency_evidence(path, base_sha, checkpoint=None, packet=None, expected_runtime=None):
    report = json.loads(Path(path).read_text())
    runtime = runtime_evidence(report.get('runtime'), expected_runtime)
    if report.get('base_checkpoint_sha256') != base_sha:
        raise RuntimeError('latency profile used a different official checkpoint')
    files = literal_constant(ROOT / 'scripts/profile_temporal_feedback.py', 'FILES')
    for name in files:
        if report.get('source_hashes', {}).get(name) != sha256(ROOT / name):
            raise RuntimeError(f'latency profile source mismatch: {name}')
    if packet is not None and report.get('packet_sha256') != sha256(packet):
        raise RuntimeError('latency profile used a different observation fixture')
    if checkpoint is not None:
        if (not report.get('checkpoint') or Path(report['checkpoint']).resolve() != Path(checkpoint).resolve()
                or report.get('checkpoint_sha256') != sha256(checkpoint)):
            raise RuntimeError('latency profile used a different temporal checkpoint')
    anchors = report.get('native_zero_anchors', [])
    keys = {(row.get('horizon'), row.get('variation')) for row in anchors}
    if (len(anchors) != 9 or keys != {(h, v) for h in (1, 2, 4) for v in range(3)}
            or any(row.get('exact') is not True for row in anchors)
            or report.get('zero_feedback_native_full_plan_exact') is not True):
        raise RuntimeError('all nine actual official-runner exact anchors are required')
    official, _ = measured(report['official_eager'])
    planner, _ = measured(report['native_graph_planner'])
    measured(report['feedback_query'])
    horizons = {}
    for horizon in (1, 2, 4):
        row = report['by_horizon'][str(horizon)]
        fast, p95 = measured(row['fast'])
        cycle = planner + (16 // horizon - 1) * fast
        ratio = cycle / official
        passed = ratio <= .85 and p95 < 15.
        for key, value in (('ratio_to_official', ratio), ('ms_per_action', cycle / 16),
                           ('ratio_to_graph_zero', cycle / planner)):
            if not math.isclose(row.get(key, float('nan')), value, rel_tol=1e-6):
                raise RuntimeError(f'forged latency budget {horizon}/{key}')
        if row.get('passed') is not passed:
            raise RuntimeError('reported horizon gate disagrees with recomputed budget')
        horizons[str(horizon)] = dict(passed=passed, ratio_to_official=ratio,
            ratio_to_graph_zero=cycle / planner, fast_p95_ms=p95, ms_per_action=cycle / 16)
    selected = next((h for h in (1, 2, 4) if horizons[str(h)]['passed']), None)
    if selected is None or report.get('selected_horizon') != selected or report.get('passed') is not True:
        raise RuntimeError('no honestly selected horizon meets the preset 15% budget saving and fast p95 <15ms')
    if not math.isfinite(float(report.get('cold_feedback_query_ms', float('nan')))):
        raise RuntimeError('cold graph cost must be reported separately')
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), selected_horizon=selected,
        by_horizon=horizons, cold_feedback_query_ms=report['cold_feedback_query_ms'],
        runtime=runtime, scope=report.get('scope'), limitations=report.get('limitations'))


def checkpoint_evidence(path, *, steps, base, training, initialization=None):
    import torch
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
    from dynamicwam.models.temporal_feedback_contract import FORMAT, VERSION, runtime_contract
    torch.set_num_threads(1)
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    expected = dict(format=FORMAT, version=VERSION, global_step=steps, base_checkpoint=base,
        world_size=7, global_batch=14, history_mode='temporal', history_dropout=.15,
        phase='temporal_feedback_refine' if initialization else 'temporal_feedback_adapt')
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'temporal export contract mismatch: {key}')
    head = TemporalPlanFeedbackHead()
    contract = runtime_contract(BridgeSchedule.from_training_config(training), head.residual_limit)
    if payload.get('spec') != head.constructor_spec or payload.get('runtime_contract') != contract:
        raise RuntimeError('temporal head/deployment contract mismatch')
    head.load_state_dict(payload['model'], strict=True)
    for name, value in payload['model'].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'nonfinite feedback tensor: {name}')
        if value.is_floating_point() and value.dtype != torch.float32:
            raise RuntimeError(f'exported master parameter is not FP32: {name}')
    for name in TRAIN_FILES:
        if payload.get('source_hashes', {}).get(name) != sha256(ROOT / name):
            raise RuntimeError(f'training source mismatch: {name}')
    wanted_init = dict(path=str(Path(initialization).resolve()), sha256=sha256(initialization)) if initialization else None
    if payload.get('initialization') != wanted_init:
        raise RuntimeError('stage3 must initialize from the verified stage2; stage2 must be fresh')
    recipe = payload.get('config', {})
    wanted = dict(steps=steps, lr=5e-5 if initialization else 2e-4, batch_size=1, gradient_accumulation=2,
        validate_every=500, validation_batches=32, expert_weight=.25, residual_l2=.01,
        history_mode='temporal', history_dropout=.15)
    if any(recipe.get(k) != v for k, v in wanted.items()):
        raise RuntimeError('exported recipe differs from the declared pipeline budget')
    official = torch.load(base['path'], map_location='cpu', weights_only=True, mmap=True)
    if payload.get('action_normalization') != official['config']['action_normalization']:
        raise RuntimeError('temporal normalization differs from the immutable official planner')
    for key in ('dataset_identity', 'pairing_identity'):
        if not payload.get(key):
            raise RuntimeError(f'missing {key}')
    for key in ('train_ages', 'validation_ages'):
        ages = {int(k): int(v) for k, v in payload['data'][key].items()}
        if set(ages) != set(range(16)) or min(ages.values()) <= 0:
            raise RuntimeError(f'missing deployed age coverage: {key}')
    if (payload['data'].get('history_raw_frame_lags') != contract['history_training_raw_lags']
            or payload['data'].get('history_selection') != contract['history_sampling']):
        raise RuntimeError('training did not use the contracted multi-lag physical history')
    validation = payload['validation']
    if not validation or any(not math.isfinite(float(v)) for v in validation.values()):
        raise RuntimeError('nonfinite or missing held-out validation')
    if any(validation.get(f'age_{age}/samples', 0) <= 0 for age in range(16)):
        raise RuntimeError('held-out validation missed a deployed age')
    if validation.get('age_0/active_arm_coordinates') != 0:
        raise RuntimeError('identity age zero must not dilute validation errors')
    for age in range(1, 16):
        if validation.get(f'age_{age}/active_arm_coordinates') != 12 * min(4, 16-age) * validation[f'age_{age}/samples']:
            raise RuntimeError('validation failed arm-only, unpadded-tail weighting')
    rows = [json.loads(line) for line in (path.parent.parent / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    initialized = [r for r in rows if r.get('kind') == 'initialized']
    complete = [r for r in rows if r.get('kind') == 'complete']
    if (len(initialized) != 1 or initialized[0].get('teacher_trainable_parameters') != 0
            or initialized[0].get('global_batch') != 14 or not complete or complete[-1].get('step') != steps
            or initialized[0].get('teacher_precision') != contract['precision']):
        raise RuntimeError('missing frozen/full-precision teacher or completed training evidence')
    gradients = [r for r in rows if r.get('kind') == 'train']
    if (not gradients or any(not math.isfinite(float(r['grad_norm'])) or r['grad_tensors'] <= 0 for r in gradients)
            or not any(r['grad_norm'] > 0 for r in gradients)):
        raise RuntimeError('missing finite nonzero feedback gradient evidence')
    return dict(path=str(path), sha256=sha256(path), global_step=steps, global_batch=14,
        base_checkpoint=base, spec=payload['spec'], teacher_trainable_parameters=0,
        validation=validation, dataset_identity=payload['dataset_identity'], pairing_identity=payload['pairing_identity'])


def data_evidence(path, *, config, base):
    report = json.loads(Path(path).read_text())
    audit_base = report.get('base_checkpoint', {})
    if (report.get('format') != 'dynamicwam_temporal_dataset_audit' or report.get('version') != 2
            or report.get('passed') is not True or report.get('cuda_initialized') is not False
            or report.get('normalization_match') is not True or audit_base.get('sha256') != base['sha256']
            or Path(audit_base.get('path', '')).resolve() != Path(base['path']).resolve()
            or report.get('config', {}).get('sha256') != sha256(config)):
        raise RuntimeError('real-data audit format/config/base/CPU/normalization gate failed')
    sources = report.get('source_hashes', {})
    required = {'scripts/check_temporal_feedback_data.py', 'scripts/train_temporal_feedback.py',
        'src/dynamicwam/training/data/temporal_feedback_dataset.py',
        'src/dynamicwam/models/temporal_plan_feedback.py', 'src/dynamicwam/models/temporal_feedback_contract.py'}
    if not required.issubset(sources) or any(value != sha256(ROOT/name) for name, value in sources.items()):
        raise RuntimeError('real-data audit used stale or incomplete source')
    dataset = report['dataset']
    lags = [1, 2, 3, 6, 9, 12, 18, 24, 36]
    if (dataset.get('episode_overlap') != [] or dataset.get('history_raw_frame_lags') != lags
            or min(dataset.get('train_episode_count', 0), dataset.get('validation_episode_count', 0)) <= 0
            or not dataset.get('identity') or not dataset.get('pairing_identity')):
        raise RuntimeError('real-data audit lacks disjoint episodes or contracted physical history lags')
    for name in ('dataset.json', 'action_stats.json', 'motion_stats.json', 'samples.jsonl'):
        if dataset.get('metadata_hashes', {}).get(name) != sha256(Path(dataset['root'])/name):
            raise RuntimeError(f'packed dataset metadata changed: {name}')
    if report.get('cpu_contracts', {}).get('passed') is not True:
        raise RuntimeError('real-data/teacher-stub contracts did not pass')
    for partition in ('train', 'validation'):
        subset = report['subsets'][partition]
        if subset.get('samples', 0) < 256 or set(map(int, subset['age_counts'])) != set(range(16)):
            raise RuntimeError('real-data audit missed minimum samples or action ages')
        if set(map(int, subset['history_lag_counts'])) != set(lags):
            raise RuntimeError('real sampled histories do not cover every declared training lag')
        if not subset.get('clocks'):
            raise RuntimeError('missing source episode clock identities')
        for clock in subset['clocks']:
            digest = clock.get('clock_fields_sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise RuntimeError('missing computed physical clock field hash')
            for item in (clock, clock['video']):
                stat = Path(item['path']).stat()
                if stat.st_size != item['size_bytes'] or stat.st_mtime_ns != item['mtime_ns']:
                    raise RuntimeError('source clock/video changed after the real-data audit')
        for row in subset['sampled_pairs']:
            dt = row['history_dt']
            if not math.isfinite(dt) or dt < 0 or (row['history_valid'] and (dt <= 0 or row['previous_frame'] >= row['current_frame'])):
                raise RuntimeError('history is noncausal or has invalid physical time')
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), passed=True,
        dataset_identity=dataset['identity'], pairing_identity=dataset['pairing_identity'],
        history_raw_frame_lags=lags, train_history_dt=report['subsets']['train']['history_dt'],
        validation_history_dt=report['subsets']['validation']['history_dt'])


def probe_evidence(path, *, config, base, smoke):
    report = json.loads(Path(path).read_text())
    if report.get('format') != 'dynamicwam_temporal_feedback_probe_report' or report.get('version') != 1:
        raise RuntimeError('unexpected temporal probe report')
    metadata = report['cache_metadata']
    declared_contract = metadata.get('deployment_contract', {})
    if (metadata.get('format') != 'dynamicwam_temporal_feedback_probe_cache' or metadata.get('version') != 2
            or declared_contract.get('format') != 'dynamicwam_temporal_plan_feedback_checkpoint'
            or declared_contract.get('version') != 2
            or declared_contract.get('history_training_raw_lags') != [1, 2, 3, 6, 9, 12, 18, 24, 36]
            or declared_contract.get('history_sampling') != 'pair_identity_sha256_v1'
            or declared_contract.get('plan_age_seconds') != 'telemetry_only'
            or metadata.get('base_checkpoint') != base or metadata.get('config_sha256') != sha256(config)
            or metadata.get('dataset_identity') != smoke['dataset_identity']
            or metadata.get('pairing_identity') != smoke['pairing_identity']
            or metadata.get('train_samples') != 512 or metadata.get('validation_samples') != 256):
        raise RuntimeError('probe cache/teacher/data identity mismatch')
    split = metadata.get('split', {})
    if split.get('overlap') != 0 or min(split.get('train_episodes', 0), split.get('heldout_episodes', 0)) <= 0:
        raise RuntimeError('probe requires disjoint nonempty held-out episodes')
    for name in literal_constant(ROOT / 'scripts/probe_temporal_feedback.py', 'SOURCE_FILES'):
        if metadata.get('source_hashes', {}).get(name) != sha256(ROOT / name):
            raise RuntimeError(f'probe source mismatch: {name}')
    if report.get('cache_sha256') != sha256(report['cache_path']):
        raise RuntimeError('probe cached teacher targets changed')
    for arm in ('static', 'temporal', 'shuffled'):
        summary = report['metrics'][arm]['all']
        if summary.get('rows') != 256 or summary.get('active_coordinates', 0) <= 0:
            raise RuntimeError('probe needs exactly 256 held-out rows and nonempty active arm coordinates')
    static, fresh, shuffled = [report['metrics'][k]['all']['target_mse'] for k in ('static', 'temporal', 'shuffled')]
    if any(not math.isfinite(float(v)) or v < 0 for v in (static, fresh, shuffled)) or static <= 0:
        raise RuntimeError('probe has no finite active-arm error denominator')
    gain = (static-fresh) / static
    zero = report['metrics']['temporal']['all'].get('zero_target_mse')
    if zero is None or not math.isfinite(float(zero)) or zero < 0:
        raise RuntimeError('probe omitted a finite zero-correction error on the same clipped reachable target')
    for arm in ('static', 'shuffled'):
        reference = report['metrics'][arm]['all'].get('zero_target_mse', float('nan'))
        if not math.isclose(reference, zero, rel_tol=1e-7, abs_tol=1e-12):
            raise RuntimeError('probe arms do not share the same zero-correction target/mask baseline')
    if fresh > zero:
        raise RuntimeError(f'temporal feasibility gate failed against unchanged native plan: temporal_target_mse={fresh}, '
            f'zero_correction_target_mse={zero}; beating a worse static head is insufficient')
    if gain < .01 or fresh > shuffled or report.get('gate', {}).get('passed') is not True:
        raise RuntimeError(f'temporal feasibility gate failed: relative gain={gain}, fresh={fresh}, shuffled={shuffled}')
    declared = report['gate'].get('relative_arm_mse_gain', float('nan'))
    if not math.isclose(gain, declared, rel_tol=1e-6, abs_tol=1e-10):
        raise RuntimeError('reported probe gate differs from raw metrics')
    deployment = report.get('deployment_checkpoint', {})
    if not deployment.get('path') or sha256(deployment['path']) != deployment.get('sha256'):
        raise RuntimeError('missing explicitly labeled deployable probe checkpoint')
    wanted = dict(train_samples=512, validation_samples=256, teacher_batch_size=1, batch_size=32,
        epochs=20, lr=2e-4, seed=20261006, history_dropout=.15)
    if any(report.get('args', {}).get(key) != value for key, value in wanted.items()):
        raise RuntimeError('probe recipe differs from the predeclared fixed budget')
    checkpoint_identity = probe_checkpoint_evidence(report, config)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), passed=True,
        relative_arm_mse_gain=gain, fresh_target_mse=fresh, shuffled_target_mse=shuffled,
        zero_correction_target_mse=zero, temporal_not_worse_than_zero=fresh <= zero,
        zero_comparison_scope='same held-out clipped teacher/expert reachable residual target; active arm coordinates only',
        deployment_checkpoint=deployment, checkpoint_contract=checkpoint_identity, split=split,
        scope='one prespecified offline feasibility gate; no closed-loop accuracy inference')


def probe_checkpoint_evidence(report, config):
    import torch
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
    from dynamicwam.models.temporal_feedback_contract import FORMAT, VERSION, runtime_contract
    payload = torch.load(report['deployment_checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
    metadata = report['cache_metadata']
    expected = dict(format=FORMAT, version=VERSION, phase='analysis_probe', arm='temporal', global_step=320,
        cache_sha256=report['cache_sha256'], initial_state_sha256=report['initial_state_sha256'])
    for key in ('base_checkpoint', 'action_normalization', 'runtime_contract', 'dataset_identity', 'pairing_identity', 'source_hashes'):
        expected[key] = metadata[key]
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f'deployable probe checkpoint identity mismatch: {key}')
    head = TemporalPlanFeedbackHead()
    if payload.get('spec') != head.constructor_spec or report.get('spec') != head.constructor_spec:
        raise RuntimeError('probe head specification differs from deployed head')
    current_contract = runtime_contract(BridgeSchedule.from_training_config(load_multirate_profile(config).training), head.residual_limit)
    if payload.get('runtime_contract') != current_contract:
        raise RuntimeError('probe runtime contract differs from the current train/deploy contract')
    head.load_state_dict(payload['model'], strict=True)
    for name, value in payload['model'].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) or (value.is_floating_point() and value.dtype != torch.float32):
            raise RuntimeError(f'invalid probe head master tensor: {name}')
    initial_path = Path(report['args']['run_dir'])/'initial_state.pt'
    if sha256(initial_path) != report['initial_state_sha256']:
        raise RuntimeError('probe common initialization artifact changed')
    return dict(phase='analysis_probe', global_step=320, usage='early canary only; never stage2 initialization',
        initial_state_sha256=report['initial_state_sha256'])


def task_evidence(directory, seeds, checkpoint, base_sha, *, feedback, history, horizon):
    directory = Path(directory)
    aggregate_path = directory / 'aggregate.json'
    aggregate = valid_task_aggregate(aggregate_path, len(seeds), expected_start_seed=100000, expected_episode_seeds=seeds)
    if aggregate is None:
        raise EvaluationIncomplete(f'incomplete/preparation-failed or seed-mismatched task: {directory}')
    episodes = aggregate['episodes']
    if any(type(ep.get('success')) is not bool for ep in episodes) or aggregate['success_count'] != sum(ep['success'] for ep in episodes):
        raise EvaluationIncomplete(f'inconsistent success accounting: {directory}')
    rows, queries = [], []
    for log in sorted((directory / 'logs').glob('slot*.log')):
        for line in log.read_text(errors='replace').splitlines():
            if line.startswith('TEMPORAL-FEEDBACK-TIMING '):
                rows.append(ast.literal_eval(line.split(' ', 1)[1]))
            elif line.startswith('TEMPORAL-FEEDBACK-QUERY '):
                queries.append(ast.literal_eval(line.split(' ', 1)[1]))
    if len(rows) != len(seeds):
        raise EvaluationIncomplete(f'missing episode timing/loader evidence: {directory}')
    for row in rows:
        if (Path(row.get('checkpoint') or '').resolve() != Path(checkpoint).resolve()
                or row.get('base_checkpoint_sha256') != base_sha or row.get('feedback_enabled') is not feedback
                or row.get('history_enabled') is not history or row.get('execute_horizon') != horizon):
            raise RuntimeError(f'wrong loaded temporal evaluation arm: {directory}')
        if any(not math.isfinite(float(row.get(k, 0))) or row.get(k, 0) <= 0
               for k in ('query_count', 'executed_steps', 'query_ms_sum')):
            raise RuntimeError(f'nonfinite/empty action timing: {directory}')
    count, steps, ms = [sum(row[key] for row in rows) for key in ('query_count', 'executed_steps', 'query_ms_sum')]
    if len(queries) != count:
        raise EvaluationIncomplete(f'query/episode timing records disagree: {directory}')
    response = {}
    for key in ('observation_to_action_ms', 'adapter_preprocess_ms', 'decision_interval_wall_ms', 'decision_interval_sim_ms'):
        values = [row[key] for row in queries if row.get(key) is not None]
        if any(not math.isfinite(float(v)) or v < 0 for v in values):
            raise RuntimeError(f'nonfinite/missing reaction timing {key}: {directory}')
        if not values and key in ('observation_to_action_ms', 'adapter_preprocess_ms'):
            raise RuntimeError(f'missing query latency {key}: {directory}')
        response[key] = dict(count=len(values), mean=sum(values)/len(values) if values else None,
            p50=percentile(values, .5) if values else None, p95=percentile(values, .95) if values else None)
    return dict(aggregate=str(aggregate_path), sha256=sha256(aggregate_path), seeds=list(seeds),
        reported_episodes=len(seeds), success_count=aggregate['success_count'],
        episodes=[dict(seed=ep['seed'], success=ep['success']) for ep in episodes],
        feedback_enabled=feedback, history_enabled=history, execute_horizon=horizon,
        query_count=count, executed_steps=steps, query_ms_sum=ms, query_ms_mean=ms/count,
        ms_per_executed_action=ms/steps, response=response)


def canary_evidence(rows):
    if set(rows) != {f'{arm}/{task}' for arm in ('zero', 'temporal') for task in TASKS}:
        raise EvaluationIncomplete('canary requires all six complete task/arm records')
    for task in TASKS:
        left, right = rows[f'zero/{task}'], rows[f'temporal/{task}']
        if left['seeds'] != right['seeds'] or len(left['seeds']) != 2:
            raise EvaluationIncomplete('canary seeds are not exactly paired')
    zero = sum(rows[f'zero/{task}']['success_count'] for task in TASKS)
    trained = sum(rows[f'temporal/{task}']['success_count'] for task in TASKS)
    return dict(zero_successes=zero, temporal_successes=trained, episodes_per_arm=6,
        passed=trained >= zero-1, no_success_in_either_arm=zero == trained == 0, rows=rows,
        scope='prespecified catastrophic regression tripwire only; six episodes do not establish noninferiority')


def formal_arms(horizon):
    arms = {f'zero_H{horizon}': dict(feedback=False, history=True, horizon=horizon),
            f'temporal_H{horizon}': dict(feedback=True, history=True, horizon=horizon)}
    if horizon != 4:
        arms['temporal_H4'] = dict(feedback=True, history=True, horizon=4)
    arms[f'no_history_H{horizon}'] = dict(feedback=True, history=False, horizon=horizon)
    return arms


def common_evidence(arms, expected_tasks):
    common = sorted(set(expected_tasks).intersection(*(set(arm.get('tasks', {})) for arm in arms.values()))) if arms else []
    rows, pairs = {}, {}
    for name, arm in arms.items():
        tasks = [arm['tasks'][task] for task in common]
        count = sum(t['reported_episodes'] for t in tasks)
        successes = sum(t['success_count'] for t in tasks)
        steps = sum(t['executed_steps'] for t in tasks)
        rows[name] = dict(tasks=len(tasks), episodes=count, successes=successes,
            success_rate=successes/count if count else None,
            ms_per_executed_action=sum(t['query_ms_sum'] for t in tasks)/steps if steps else None)
    names = list(arms)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            gained = lost = 0
            for task in common:
                a, b = arms[left]['tasks'][task], arms[right]['tasks'][task]
                if a['seeds'] != b['seeds']:
                    raise RuntimeError('formal comparison encountered different accepted seed manifests')
                amap = {ep['seed']: ep['success'] for ep in a['episodes']}
                bmap = {ep['seed']: ep['success'] for ep in b['episodes']}
                if set(amap) != set(bmap) or len(amap) != len(a['seeds']):
                    raise RuntimeError('formal paired episode keys do not match')
                gained += sum(bmap[s] and not amap[s] for s in amap)
                lost += sum(amap[s] and not bmap[s] for s in amap)
            n = rows[left]['episodes']
            pairs[f'{right}_minus_{left}'] = dict(improved=gained, regressed=lost,
                delta_percentage_points=100*(gained-lost)/n if n else None, episodes=n)
    return dict(valid_common_tasks=common, requested_tasks=len(expected_tasks), arms=rows, pairs=pairs,
        scope='intersection of complete task/seed records; missing preparation is excluded, never scored as zero; same seeds do not imply bitwise identical expert motion preparation')


class Pipeline:
    def __init__(self, args):
        import fcntl
        self.args, self.root = args, args.run_root.resolve()
        if ROOT.resolve() != self.root / 'source':
            raise RuntimeError('execute RUN_ROOT/source/scripts/run_temporal_feedback_pipeline.py')
        self.gpus = [s.strip() for s in args.gpus.split(',')]
        if len(self.gpus) != 7 or len(set(self.gpus)) != 7 or not all(s.isdigit() for s in self.gpus):
            raise ValueError('the declared global-batch14 recipe requires seven distinct GPUs')
        self.lock = (self.root / 'pipeline.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('status.json', 'commands.jsonl', 'pipeline_source_manifest.json', 'smoke', 'probe', 'stage2', 'stage3', 'early_canary', 'canary', 'eval'):
            if (self.root / name).exists():
                raise FileExistsError(f'fresh run required: {self.root / name}')
        self.fingerprint = snapshot_identity(self.root)
        write_json(self.root / 'pipeline_source_manifest.json', self.fingerprint)
        self.state = dict(status='running', phase='preflight', started=time.time(), pid=os.getpid(),
            source=str(ROOT), source_sha256=self.fingerprint['sha256'], completed=[], evidence={})
        self.python = str(ROOT / '.venv/bin/python')
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(self.gpus), OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        self.env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + self.env.get('PYTHONPATH', '')
        self.save()

    def save(self):
        self.state['updated'] = time.time()
        write_json(self.root/'status.json', self.state)

    def record(self, name, evidence):
        self.state['evidence'][name] = evidence
        self.state['completed'].append(name)
        self.save()

    def run(self, name, command, verify=lambda: {'passed': True}, *, env=None, gpu=True, allow_nonzero=False):
        if snapshot_identity(self.root) != self.fingerprint:
            raise RuntimeError('immutable source changed during the run')
        if gpu:
            assert_gpus_idle(self.gpus)
        log = self.root/f'{name}.log'
        self.state.update(phase=name, command=command, log=str(log))
        self.save()
        with (self.root/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), phase=name, argv=command, cwd=str(ROOT),
                cuda_visible_devices=(env or self.env).get('CUDA_VISIBLE_DEVICES'))) + '\n')
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env or self.env, stdout=stream, stderr=subprocess.STDOUT)
            self.state['child_pid'] = child.pid
            self.save()
            code = child.wait()
        self.state.pop('child_pid', None)
        self.save()
        if code and not allow_nonzero:
            raise RuntimeError(f'{name} exited {code}; inspect {log}; no automatic retry')
        evidence = verify()
        evidence['command_exit_code'] = code
        evidence['log'] = dict(path=str(log), sha256=sha256(log))
        evidence['source_sha256'] = self.fingerprint['sha256']
        self.record(name, evidence)
        print(f'completed {name}', flush=True)
        return evidence

    def train(self, name, steps, lr, initialization=None):
        directory = self.root/name
        if directory.exists():
            raise FileExistsError(directory)
        checkpoint = directory/'exports'/f'feedback_step_{steps}.pt'
        command = [self.python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=7',
            str(ROOT/'scripts/train_temporal_feedback.py'), '--config', str(self.args.config),
            '--run-dir', str(directory), '--steps', str(steps), '--lr', str(lr), '--batch-size', '1',
            '--gradient-accumulation', '2', '--validate-every', '500', '--validation-batches', '32',
            '--workers', '2', '--history-mode', 'temporal', '--history-dropout', '.15']
        if initialization:
            command.extend(['--init-feedback', str(initialization)])
        self.run(name, command, lambda: checkpoint_evidence(checkpoint, steps=steps, base=self.base,
            training=self.training, initialization=initialization))
        return checkpoint

    def profile_checkpoint(self, name, checkpoint):
        path = self.root/f'{name}.json'
        return self.run(name, [self.deployment_python, str(ROOT/'scripts/profile_temporal_feedback.py'), '--checkpoint',
            str(checkpoint), '--output', str(path), '--samples', '30', '--packet', str(self.args.packet)],
            lambda: latency_evidence(path, self.base['sha256'], checkpoint, self.args.packet, self.deployment_runtime),
            env=dict(self.env, CUDA_VISIBLE_DEVICES=self.gpus[0]))

    def options(self, arm, checkpoint, settings):
        return dict(policy_name='ciwam.adapters.domino.deploy_policy_temporal_feedback', ckpt_setting=arm,
            temporal_feedback_checkpoint=str(checkpoint), temporal_feedback_disabled=not settings['feedback'],
            temporal_history_disabled=not settings['history'], execute_horizon=settings['horizon'],
            eval_video_log=False, plan_graph_enabled=True)

    def canary(self, name, checkpoint, horizon):
        import yaml
        rows = {}
        for arm in ('zero', 'temporal'):
            settings = dict(feedback=arm == 'temporal', history=True, horizon=horizon)
            for task in TASKS:
                directory = self.root/name/arm/task
                worker = self.profile.domino_base_config()
                worker.update(self.options(arm, checkpoint, settings), task_name=task, start_seed=100000,
                    fixed_episode_seeds=self.seeds[task][:2], eval_output_root=str(directory))
                config = self.root/'eval_configs'/f'{name}_{arm}_{task}.yaml'
                config.parent.mkdir(parents=True, exist_ok=True)
                with config.open('x') as stream:
                    yaml.safe_dump(worker, stream, sort_keys=False)
                command = [self.python, str(ROOT/'src/dynamicwam/evaluation/run_domino_parallel_eval.py'),
                    '--python', self.benchmark['python'], '--domino-root', self.benchmark['domino_root'],
                    '--eval-policy', str(self.fixed_policy), '--config', str(config), '--runtime-root',
                    self.benchmark['runtime_root'], '--curobo-root', self.benchmark['curobo_root'],
                    '--run-root', str(directory), '--gpus', self.gpus[0], '--total-episodes', '2',
                    '--start-seed', '100000', '--slot-seed-stride', '100000', '--timeout-seconds',
                    str(self.benchmark['task_timeout_seconds']), '--stagger-seconds', '0']
                rows[f'{arm}/{task}'] = self.run(f'{name}_{arm}_{task}', command,
                    lambda d=directory, t=task, s=settings: task_evidence(d, self.seeds[t][:2], checkpoint,
                        self.base['sha256'], **s))
        gate = canary_evidence(rows)
        write_json(self.root/f'{name}_summary.json', gate)
        self.record(f'{name}_gate', gate)
        if not gate['passed']:
            raise RuntimeError(f'{name} catastrophic tripwire failed (>1 of six paired episodes lost); stop for analysis')

    def collect_arm(self, directory, checkpoint, settings):
        tasks, missing = {}, {}
        for task, seeds in self.seeds.items():
            try:
                tasks[task] = task_evidence(directory/'tasks'/task, seeds, checkpoint, self.base['sha256'], **settings)
            except (OSError, ValueError, KeyError, SyntaxError, RuntimeError) as error:
                missing[task] = dict(error=repr(error), classification='incomplete_or_preparation' if isinstance(error, EvaluationIncomplete) else 'invalid_or_runtime')
        reported = sum(row['reported_episodes'] for row in tasks.values())
        aggregate = directory/'aggregate.json'
        return dict(status='complete' if not missing and reported == 700 else 'incomplete', tasks=tasks,
            missing_tasks=missing, completed_tasks=len(tasks), requested_tasks=35, reported_episodes=reported,
            requested_episodes=700, success_count=sum(row['success_count'] for row in tasks.values()),
            aggregate_sha256=sha256(aggregate) if aggregate.is_file() else None,
            episode_manifest_sha256=sha256(self.manifest))

    def evaluate(self, checkpoint, horizon):
        arms = {}
        for arm, settings in formal_arms(horizon).items():
            directory = self.root/'eval'/arm
            override = self.root/'eval_configs'/f'{arm}.json'
            write_json(override, self.options(arm, checkpoint, settings))
            command = [self.python, str(ROOT/'scripts/eval_domino.py'), '--config', str(self.deploy_config),
                '--eval-policy', str(self.fixed_policy), '--policy-overrides', str(override), '--run-root',
                str(directory), '--gpus', ','.join(self.gpus), '--episodes-per-task', '20', '--start-seed',
                '100000', '--episode-seed-manifest', str(self.manifest)]
            # Nonzero evaluator exits are durable evidence. Collect what is valid,
            # then proceed to the next independently specified arm, without retry.
            arms[arm] = self.run(f'formal_{arm}', command,
                lambda d=directory, s=settings: self.collect_arm(d, checkpoint, s), allow_nonzero=True)
            if arms[arm]['command_exit_code']:
                arms[arm]['status'] = 'incomplete'
            write_json(self.root/'evaluation_summary.json', dict(arms=arms,
                comparison=common_evidence(arms, list(self.seeds)), selected_horizon=horizon))
        self.record('formal_comparison', common_evidence(arms, list(self.seeds)))
        return all(arm['status'] == 'complete' for arm in arms.values())

    def execute(self):
        self.deploy_config = ROOT/'configs/absolute_motion_v2.yaml'
        self.profile = load_profile(self.deploy_config)
        self.benchmark = self.profile.benchmark_config()
        # Preserve the benchmark's interpreter symlink: resolving its target can
        # escape the intended conda environment despite selecting the same binary.
        self.deployment_python = os.path.abspath(self.benchmark['python'])
        if (Path(self.benchmark['project_root']).resolve() != ROOT
                or Path(self.profile.domino_base_config()['dynamicwam_root']).resolve() != ROOT
                or Path(self.benchmark['runtime_root']).resolve() != ROOT/'src/dynamicwam/runtime'):
            raise RuntimeError('deployment profile escaped the immutable source snapshot')
        self.training = load_multirate_profile(self.args.config).training
        base_path = Path(self.training['model']['native_teacher_checkpoint']).resolve()
        self.base = dict(path=str(base_path), sha256=sha256(base_path))
        self.seeds, evidence, raw = load_episode_seed_manifest(self.args.manifest,
            tasks=list(self.benchmark['tasks']), episodes_per_task=20, start_seed=100000)
        if len(self.seeds) != 35 or any(len(v) != 20 for v in self.seeds.values()):
            raise RuntimeError('formal evaluation requires the complete 35-task/700-episode manifest')
        self.manifest = self.root/'episode_seed_manifest.json'
        if self.manifest.exists() and self.manifest.read_bytes() != raw:
            raise RuntimeError('run-root manifest differs from requested accepted seeds')
        if not self.manifest.exists():
            self.manifest.write_bytes(raw)
        self.record('manifest', evidence)
        cpu = dict(self.env, CUDA_VISIBLE_DEVICES='', TEMPORAL_TEST_CUDA='0')
        runtime_code = '\n'.join((
            'import json, os, platform, sys, torch',
            'print(json.dumps(dict(executable=os.path.abspath(sys.executable), python=platform.python_version(),',
            '    torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),',
            '    prefix=sys.prefix, cuda_initialized=torch.cuda.is_initialized())))'))
        self.deployment_runtime = self.run('deployment_runtime', [self.deployment_python, '-c', runtime_code],
            lambda: runtime_identity_evidence(self.root/'deployment_runtime.log', self.deployment_python), env=cpu, gpu=False)
        self.record('latency_preflight', latency_evidence(self.args.latency_report, self.base['sha256'],
            packet=self.args.packet, expected_runtime=self.deployment_runtime))
        for relative in ('tests/test_temporal_feedback_model.py', 'scripts/check_temporal_feedback_runtime.py',
                         'scripts/check_temporal_feedback_pipeline.py',
                         'scripts/check_temporal_feedback_probe.py', 'scripts/check_wan_wrapper_equivalence.py'):
            self.run(Path(relative).stem, [self.python, str(ROOT/relative)], env=cpu, gpu=False)
        data_audit = self.root/'temporal_dataset_audit.json'
        audit = self.run('check_temporal_feedback_data', [self.python, str(ROOT/'scripts/check_temporal_feedback_data.py'),
            '--config', str(self.args.config), '--output', str(data_audit), '--samples', '256'],
            lambda: data_evidence(data_audit, config=self.args.config, base=self.base), env=cpu, gpu=False)
        self.fixed_policy = self.root/'fixed_eval_policy.py'
        self.record('fixed_evaluator', generate(Path(self.benchmark['eval_policy']), self.fixed_policy))
        smoke = self.train('smoke', 4, 2e-4)
        for key in ('dataset_identity', 'pairing_identity'):
            if self.state['evidence']['smoke'][key] != audit[key]:
                raise RuntimeError(f'smoke training escaped the audited dataset: {key}')
        early_latency = self.profile_checkpoint('smoke_latency', smoke)
        if self.args.verified_probe_report:
            probe_report = self.args.verified_probe_report.resolve()
            probe = probe_evidence(probe_report, config=self.args.config, base=self.base, smoke=self.state['evidence']['smoke'])
            self.record('probe', probe)
            self.record('probe_explicit_reuse', dict(path=str(probe_report), sha256=sha256(probe_report),
                policy='explicit verified reuse; no repeated teacher caching, head training, or probe selection'))
        else:
            probe_report = self.root/'probe/probe_report.json'
            probe = self.run('probe', [self.python, str(ROOT/'scripts/probe_temporal_feedback.py'),
                '--config', str(self.args.config), '--run-dir', str(self.root/'probe'), '--train-samples', '512',
                '--validation-samples', '256', '--teacher-batch-size', '1', '--batch-size', '32', '--epochs', '20',
                '--lr', '2e-4', '--seed', '20261006', '--device', 'cuda:0'],
                lambda: probe_evidence(probe_report, config=self.args.config, base=self.base, smoke=self.state['evidence']['smoke']),
                env=dict(self.env, CUDA_VISIBLE_DEVICES=self.gpus[0]))
        probe_checkpoint = Path(probe['deployment_checkpoint']['path'])
        # The same shared runtime loader strictly validates the deployable probe
        # export; this remains an explicitly labeled probe, never a stage2 result.
        probe_latency = self.profile_checkpoint('probe_latency', probe_checkpoint)
        self.canary('early_canary', probe_checkpoint, probe_latency['selected_horizon'])
        stage2 = self.train('stage2', 7000, 2e-4)
        stage3 = self.train('stage3', 2000, 5e-5, initialization=stage2)
        final_latency = self.profile_checkpoint('final_latency', stage3)
        horizon = final_latency['selected_horizon']
        self.record('horizon_selection', dict(early=early_latency['selected_horizon'], final=horizon,
            rule='smallest H in [1,2,4] satisfying unchanged ratio<=.85 and fast p95<15ms; shared trained weights'))
        self.canary('canary', stage3, horizon)
        complete = self.evaluate(stage3, horizon)
        self.state.update(status='complete' if complete else 'incomplete', finished=time.time())
        self.save()
        return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--latency-report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/multirate/native_stage2_7gpu.yaml')
    parser.add_argument('--packet', type=Path, default=PACKET)
    parser.add_argument('--verified-probe-report', type=Path,
        help='Explicit reuse only: verify fixed budget, seeds, exact source/cache/weights/data identities before deployment')
    args = parser.parse_args()
    args.config = args.config.resolve()
    if not args.config.is_relative_to(ROOT):
        parser.error('training config must be in the immutable source snapshot')
    pipeline = Pipeline(args)
    try:
        return pipeline.execute()
    except BaseException as error:
        pipeline.state.update(status='incomplete' if isinstance(error, EvaluationIncomplete) else 'failed',
            error=repr(error), stopped=time.time())
        pipeline.save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
