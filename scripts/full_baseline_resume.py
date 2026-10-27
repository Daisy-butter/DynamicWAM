#!/usr/bin/env python3
"""Audit completed tasks once, resume only whole missing tasks, publish a 700 view.

No GPU imports, process launch, retry, or mutation of either source experiment.
Video decoding uses at most two CPU workers. Unified native videos are symlinks
to immutable hash-bound originals; reports disclose both source generations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import sys

from full_baseline_worker import ROOT, canonical_hash, load_frozen_task, read, save, sha, source_hash

SCHEMA = 'full_baseline_completed_task_resume_v1'
COMPARISON_FIELDS = ('protocol', 'checkpoint_sha256', 'init_checkpoint_sha256', 'profile_sha256',
    'prepared_sha256', 'task_config', 'instruction_mode', 'instruction_type', 'seed',
    'action_dt', 'frame_period', 'max_backlog', 'max_observation_age', 'bootstrap_hold_steps',
    'model_query_timeout_s', 'benchmark_contract', 'gpu_uuids')


def acceptance_smoke_count(config):
    """Legacy receipts retain 8x2; an explicit expanded gate is at least 8x8."""
    if 'acceptance_smoke_episodes_per_task' not in config:
        if config.get('required_acceptance_cases'):
            raise ValueError('required cases need an explicit expanded smoke count')
        return 2
    count = config['acceptance_smoke_episodes_per_task']
    if type(count) is not int or not 8 <= count <= 20:
        raise ValueError('expanded smoke episodes per task must be an integer in 8..20')
    return count


def expanded_acceptance_contract(config):
    """Bind the complete ordered 8xN frozen episode matrix and required failure."""
    count = acceptance_smoke_count(config)
    if count == 2:
        return None
    tasks = config['tasks'][:8]
    if len(tasks) != 8 or len(set(tasks)) != 8:
        raise ValueError('expanded smoke requires eight distinct ordered tasks')
    required = config.get('required_acceptance_cases')
    fields = {'task', 'episode_index', 'seed', 'manifest_sha256', 'motion_sha256'}
    if (not isinstance(required, list) or not required
            or any(not isinstance(case, dict) or set(case) != fields for case in required)
            or not any(case['episode_index'] == 7 and case['seed'] == 100007 for case in required)):
        raise ValueError('expanded smoke requires the exact failed episode 7 / seed 100007 identity')
    root = Path(config['prepared'])
    if sha(root/'frozen_manifest.json') != config['prepared_sha256']:
        raise ValueError('expanded smoke frozen manifest hash changed')
    frozen = read(root/'frozen_manifest.json')
    if frozen.get('complete') is not True:
        raise ValueError('expanded smoke frozen scenes are incomplete')
    index = {}
    for row in frozen['episodes']:
        key = (row['task'], row['episode_index'])
        if key in index:
            raise ValueError('duplicate frozen episode identity')
        index[key] = row
    matrix = []
    for task in tasks:
        for episode in range(count):
            entry = index.get((task, episode))
            if entry is None:
                raise ValueError('expanded smoke episode absent from frozen scenes')
            directory = root/task/f'episode_{episode:04d}'
            if sha(directory/'manifest.json') != entry['manifest_sha256'] or sha(directory/'expert_motion.pkl') != entry['motion_sha256']:
                raise ValueError('expanded smoke scene/motion bytes changed')
            descriptor = read(directory/'manifest.json')
            if (descriptor.get('task') != task or descriptor.get('episode_index') != episode
                    or descriptor.get('seed') != entry['seed']):
                raise ValueError('expanded smoke descriptor identity differs')
            matrix.append({key: entry[key] for key in sorted(fields)})
    required_keys = [canonical_hash(case) for case in required]
    if len(set(required_keys)) != len(required_keys) or not set(required_keys) <= {canonical_hash(row) for row in matrix}:
        raise ValueError('required failed case is not covered by the exact expanded smoke matrix')
    return dict(schema='full_baseline_expanded_acceptance_v1', smoke_tasks=tasks,
                smoke_episodes_per_task=count, complete_smoke_episodes=8*count,
                required_acceptance_cases=required, episode_identities=matrix,
                episode_identity_sha256=canonical_hash(matrix))


def validate_expanded_acceptance(config, receipt):
    expected = expanded_acceptance_contract(config)
    if expected is None:
        return None
    actual = receipt.get('expanded_acceptance')
    if actual != expected or receipt.get('smoke_episodes_per_task') != expected['smoke_episodes_per_task']:
        raise ValueError('expanded smoke acceptance identity/coverage differs')
    return expected


def validate_physics_replay(config, source=None):
    """CPU scheduling changes need paired recorded-control parity, never replace smoke."""
    workers = config.get('physx_cpu_workers', 0)
    if type(workers) is not int or not 0 <= workers <= 4:
        raise ValueError('PhysX CPU worker count must be an integer in 0..4')
    if workers == 0:
        return None
    expanded = expanded_acceptance_contract(config)
    if expanded is None:
        raise ValueError('nonzero PhysX CPU workers require expanded real-policy acceptance')
    path = Path(config['physics_replay_receipt']).resolve(strict=True)
    if sha(path) != config['physics_replay_receipt_sha256']:
        raise ValueError('physics replay comparison receipt changed')
    receipt = read(path)
    if (receipt.get('schema') != 'full_physics_replay_comparison_v1'
            or any(receipt.get(key) is not True for key in ('passed', 'state_parity_passed', 'selected_candidate_passed'))
            or receipt.get('selected_physx_cpu_workers') != workers
            or type(receipt.get('parity_atol')) not in (int, float)
            or not 0 < receipt['parity_atol'] <= 1e-6):
        raise ValueError('selected PhysX candidate lacks bounded recorded-state parity')
    expected_source = source or dict(source_sha256=source_hash(ROOT/'src/dynamicwam'), scripts_sha256=source_hash(ROOT/'scripts'))
    if any(receipt.get(key) != expected_source[key] for key in ('source_sha256', 'scripts_sha256')):
        raise ValueError('physics replay ran a different frozen source')
    fields = ('task', 'episode_index', 'seed', 'manifest_sha256', 'motion_sha256')
    case = {key: receipt.get(key) for key in fields}
    if (case not in expanded['required_acceptance_cases']
            or receipt.get('frozen_manifest_sha256') != config['prepared_sha256']):
        raise ValueError('physics replay is not the required failed frozen scene')
    evidence = receipt.get('evidence', [])
    if not evidence or not receipt.get('source_report_sha256') or not any(
            row.get('sha256') == receipt['source_report_sha256'] for row in evidence):
        raise ValueError('physics replay lacks its original failed trace evidence')
    for row in evidence:
        if sha(row['path']) != row['sha256']:
            raise ValueError('physics replay evidence changed: ' + row['path'])
    return receipt


def _closed_replay_guard(directory, *, expected_source=None):
    """Validate saved diagnostic closure without querying GPUs or live processes."""
    directory = Path(directory).resolve(strict=True)
    names = ('status.json', 'owned_exit.json', 'kernel_after.json', 'source.json',
             'config.frozen.json', 'authorization.json', 'authorization.json.consumed', 'guard_events.json')
    status, owned, kernel, source, config, auth, consumed, events = [read(directory/name) for name in names]
    if (status.get('state') != 'completed' or status.get('guard_closed') is not True or status.get('errors')
            or owned.get('remaining') != [] or owned.get('returncode') != 0
            or kernel.get('unchanged') is not True or kernel.get('same_boot') is not True or kernel.get('added') != []):
        raise ValueError('passive-force replay guard is not cleanly closed')
    if (config.get('run_kind') != 'full_physics_replay' or auth.get('mode') != 'full_physics_replay'
            or auth.get('source') != source or auth.get('identity', {}).get('config_sha256') != canonical_hash(config)
            or consumed.get('receipt_sha256') != sha(directory/'authorization.json')):
        raise ValueError('passive-force replay guard configuration/source/claim differs')
    if expected_source is not None and any(source.get(k) != expected_source[k] for k in ('source_sha256', 'scripts_sha256')):
        raise ValueError('passive-force replay guard used another frozen source')
    opened = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_open']
    closed = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_close']
    if (len(opened) != 8 or len(closed) != 8 or len({r[0] for r in opened}) != 8 or set(opened) != set(closed)
            or len(auth.get('gpu_uuids', [])) != 1 or not set(auth['gpu_uuids']) <= {r[0] for r in opened}):
        raise ValueError('passive-force replay did not close all eight guard descriptors')
    return config, [directory/name for name in names]


def validate_passive_force_replay(config, source=None):
    enabled = config.get('passive_force_dedup', False)
    if type(enabled) is not bool:
        raise ValueError('passive_force_dedup must be boolean')
    if not enabled:
        return None
    if config.get('physx_cpu_workers', 0) != 0 or acceptance_smoke_count(config) < 10:
        raise ValueError('passive-force dedup requires CPU workers0 and at least 8x10 real-policy smoke')
    expanded = expanded_acceptance_contract(config)
    path = Path(config['passive_force_replay_receipt']).resolve(strict=True)
    if sha(path) != config['passive_force_replay_receipt_sha256']:
        raise ValueError('passive-force replay comparison changed')
    proof = read(path)
    flags = ('passed', 'state_parity_passed', 'extension_bitwise_equal', 'prefix_bitwise_equal',
             'selected_passive_force_dedup', 'timing_benefit_recorded', 'control_wall_improved', 'environment_closed')
    if (proof.get('schema') != 'full_passive_force_dedup_replay_v1'
            or any(proof.get(key) is not True for key in flags) or proof.get('physx_cpu_workers') != 0
            or proof.get('state_count') != 3151 or proof.get('prefix_state_count') != 2662
            or proof.get('tail_hold_rows') != 10):
        raise ValueError('passive-force replay lacks exact 3151/2662-state bitwise acceptance')
    current = source or dict(source_sha256=source_hash(ROOT/'src/dynamicwam'), scripts_sha256=source_hash(ROOT/'scripts'))
    if any(proof.get(key) != current[key] for key in ('source_sha256', 'scripts_sha256')):
        raise ValueError('passive-force replay ran another frozen source')
    case = {key: proof.get(key) for key in ('task', 'episode_index', 'seed', 'manifest_sha256', 'motion_sha256')}
    if case not in expanded['required_acceptance_cases'] or proof.get('frozen_manifest_sha256') != config['prepared_sha256']:
        raise ValueError('passive-force replay used another frozen failure scene')
    timing_keys = ('total_control_wall_seconds', 'total_physx_wall_seconds', 'worst_full_row_control_wall_seconds')
    for label in ('timing_before', 'timing_after'):
        values = proof.get(label, {})
        if any(type(values.get(k)) not in (int, float) or not math.isfinite(values[k]) or values[k] <= 0 for k in timing_keys):
            raise ValueError('passive-force replay timing evidence is absent/nonfinite')
    if proof['timing_after']['total_control_wall_seconds'] >= proof['timing_before']['total_control_wall_seconds']:
        raise ValueError('passive-force replay did not record a control CPU wall-time benefit')
    # No per-row <=180ms gate: bounded bursts still need actual 80-episode
    # policy/camera/backlog acceptance. Recorded-control replay alone cannot pass.
    guards = config.get('passive_force_replay_guards', {})
    if set(guards) != {'original', 'baseline', 'optimized'}:
        raise ValueError('three closed original/baseline/optimized replay guards required')
    original, original_files = _closed_replay_guard(guards['original'])
    baseline, baseline_files = _closed_replay_guard(guards['baseline'], expected_source=current)
    optimized, optimized_files = _closed_replay_guard(guards['optimized'], expected_source=current)
    if (path != (Path(optimized['output'])/'comparison.json').resolve(strict=True)
            or canonical_hash(optimized) != proof.get('config_sha256')
            or baseline.get('passive_force_dedup', False) is not False
            or optimized.get('passive_force_dedup') is not True
            or baseline.get('cpu_workers') != [0] or optimized.get('cpu_workers') != [0]
            or baseline.get('tail_hold_rows') != 10 or optimized.get('tail_hold_rows') != 10):
        raise ValueError('passive-force proof is not the exact paired CPU0 continuation')
    references = {'reference_state_trace': Path(original['output'])/'workers_0/state_trace.npz',
                  'reference_extension_trace': Path(baseline['output'])/'workers_0/state_trace.npz',
                  'reference_extension_timings': Path(baseline['output'])/'workers_0/row_timings.json'}
    for key, expected in references.items():
        if Path(optimized[key]).resolve(strict=True) != expected.resolve(strict=True) or sha(expected) != optimized[key+'_sha256']:
            raise ValueError('passive-force replay reference does not belong to its closed guard: ' + key)
    evidence = proof.get('evidence', [])
    if not evidence or not any(row.get('sha256') == proof.get('source_report_sha256') for row in evidence):
        raise ValueError('passive-force comparison lacks original failed trace evidence')
    for row in evidence:
        if sha(row['path']) != row['sha256']:
            raise ValueError('passive-force replay evidence changed: ' + row['path'])
    return dict(proof=proof, guard_evidence=[dict(path=str(p), sha256=sha(p))
                for p in original_files+baseline_files+optimized_files])


def task_partition(full_tasks, retained):
    if (len(full_tasks) != 35 or len(set(full_tasks)) != 35 or len(retained) != 24
            or len(set(retained)) != 24 or not set(retained) <= set(full_tasks)):
        raise ValueError('this continuation requires 35 tasks with exactly 24 whole completed tasks')
    kept = [task for task in full_tasks if task in retained]
    missing = [task for task in full_tasks if task not in retained]
    if len(missing) != 11:
        raise ValueError('remaining task partition differs')
    return kept, missing


def compare_benchmark(old, new):
    for key in COMPARISON_FIELDS:
        if key not in old or old.get(key) != new.get(key):
            raise ValueError('continuation changes benchmark/policy identity: ' + key)
    if old.get('action_limit') is not None or new.get('action_limit') is not None:
        raise ValueError('partial action budgets cannot be continued')
    if old.get('episodes') != 20:
        raise ValueError('original source must have twenty episodes per task')


def verify_failed_release(directory, config):
    directory = Path(directory)
    status, owned, kernel = (read(directory/name) for name in ('status.json', 'owned_exit.json', 'kernel_after.json'))
    if (status.get('state') != 'failed' or status.get('guard_closed') is not True
            or owned.get('remaining') != [] or not isinstance(owned.get('returncode'), int)):
        raise ValueError('failed source workers/guard are not proven closed')
    if kernel.get('unchanged') is not True or kernel.get('same_boot') is not True or kernel.get('added') != []:
        raise ValueError('failed source kernel/boot evidence changed')
    if read(directory/'config.frozen.json') != config:
        raise ValueError('failed source guard/config binding differs')
    auth = read(directory/'authorization.json')
    if (auth.get('identity', {}).get('config_sha256') != canonical_hash(config)
            or auth.get('gpu_uuids') != config['gpu_uuids']):
        raise ValueError('failed source authorization differs')
    events = read(directory/'guard_events.json')
    opened = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_open']
    closed = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_close']
    if len(opened) != 8 or len(closed) != 8 or set(opened) != set(closed) or {r[0] for r in opened} != set(config['gpu_uuids']):
        raise ValueError('failed source did not close all eight owned GPU descriptors')
    return [directory/name for name in ('status.json', 'owned_exit.json', 'kernel_before.json', 'kernel_after.json',
        'source.json', 'config.frozen.json', 'authorization.json', 'guard_events.json')]


def audit_source_task(config, task, *, decode=False):
    from full_baseline_suite import audit_task
    from full_baseline_sequence import audit_task_liveness
    from video_artifacts import probe_head_video
    reports, metrics = audit_task(config, task)
    liveness = audit_task_liveness(reports, config)
    references = load_frozen_task(config, task)
    evidence, decoded = [], []
    base = Path(config['output'])/task
    completed = read(base/'completed.json')
    evidence.append(base/'completed.json')
    for episode, (report, reference) in enumerate(zip(reports, references)):
        directory = base/f'episode_{episode:04d}'
        descriptor, _, motion_bytes = reference
        actual = read(directory/'manifest.json')
        if any(actual.get(k) != v for k, v in descriptor.items()):
            raise ValueError('retained scene descriptor differs from frozen source')
        if (directory/'expert_motion.pkl').read_bytes() != motion_bytes:
            raise ValueError('retained expert motion differs from frozen source')
        if report.get('source_sha256') != read(Path(config['output'])/'suite_started.json')['source']['source_sha256']:
            raise ValueError('retained episode source identity differs')
        evidence.extend([directory/name for name in ('report.json', 'metrics.json', 'manifest.json', 'expert_motion.pkl')])
        evidence.extend((Path(report['video']), Path(report['video_timestamps'])))
        if decode:
            probe = probe_head_video(Path(report['video']))
            if probe != report['video_probe']:
                raise ValueError('independent retained-video full decode differs')
            decoded.append(dict(episode=episode, video=report['video'], probe=probe))
    native = Path(completed['native_result_dir'])
    evidence.extend(native/name for name in ('_result.txt', '_metrics.json', '_metrics_report.txt', '_episodes_detail.json'))
    return dict(task=task, episodes=len(reports), successes=sum(bool(r['success']) for r in reports),
                liveness=liveness, decoded=decoded), evidence, reports, metrics


def prepare(old_config_path, old_guard, destination, decode_workers=2):
    if decode_workers not in (1, 2):
        raise ValueError('retention audit permits one or two CPU decode workers')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('retention receipt must be new')
    old_config_path = Path(old_config_path).resolve(strict=True)
    config = read(old_config_path)
    if config.get('stage') != 'formal' or config.get('episodes') != 20:
        raise ValueError('only a complete-budget formal source is eligible')
    output = Path(config['output'])
    progress, started = read(output/'suite_status.json'), read(output/'suite_started.json')
    if progress.get('state') != 'failed' or started.get('config_sha256') != canonical_hash(config):
        raise ValueError('failed source suite identity missing')
    retained, remaining = task_partition(config['tasks'], progress.get('finished_tasks', []))
    # suite_status.finished_tasks is added only after child exit==0 plus audit;
    # completed.json alone cannot prove a task process exited successfully.
    evidence = verify_failed_release(old_guard, config)
    evidence.extend((old_config_path, output/'suite_status.json', output/'suite_started.json'))
    source = read(Path(old_guard)/'source.json')
    if source != started['source']:
        raise ValueError('source suite/guard generation differs')
    if sha(config['checkpoint']) != config['checkpoint_sha256'] or sha(Path(config['prepared'])/'frozen_manifest.json') != config['prepared_sha256']:
        raise ValueError('checkpoint or scene manifest changed before retention audit')
    audits = []
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for row, files, _, _ in pool.map(lambda task: audit_source_task(config, task, decode=True), retained):
            audits.append(row)
            evidence.extend(files)
    required_smoke = remaining[:8]
    if 'place_bread_basket' not in required_smoke:
        raise ValueError('repaired smoke must include the actual failed breadth cohort')
    receipt = dict(schema=SCHEMA, complete=True, old_config=dict(path=str(old_config_path), sha256=sha(old_config_path)),
        old_guard=str(Path(old_guard).resolve()), old_source=source, checkpoint_sha256=config['checkpoint_sha256'],
        prepared_sha256=config['prepared_sha256'], benchmark_contract=config['benchmark_contract'],
        full_tasks=config['tasks'], retained_tasks=retained, remaining_tasks=remaining,
        retained_episodes=480, remaining_episodes=220, expected_episodes=700,
        retained_successes=sum(a['successes'] for a in audits), required_smoke_tasks=required_smoke,
        native_mp4_full_decode=True, query_liveness_passed=True, task_exit_zero_proven=True,
        retained_tasks_audit=audits, no_retry=True,
        evidence=[dict(path=str(p.resolve()), sha256=sha(p)) for p in sorted(set(evidence))])
    save(destination, receipt)
    return receipt


def validate_resumption(config, *, verify_evidence=True):
    path = Path(config['resumption_receipt']).resolve(strict=True)
    if sha(path) != config['resumption_receipt_sha256']:
        raise ValueError('retention receipt hash differs')
    receipt = read(path)
    if receipt.get('schema') != SCHEMA or receipt.get('complete') is not True:
        raise ValueError('retention audit is not complete')
    for key in ('native_mp4_full_decode', 'query_liveness_passed', 'task_exit_zero_proven'):
        if receipt.get(key) is not True:
            raise ValueError('retention gate missing: ' + key)
    retained, remaining = task_partition(receipt['full_tasks'], receipt['retained_tasks'])
    if (retained != receipt['retained_tasks'] or remaining != receipt['remaining_tasks'] or receipt.get('retained_episodes') != 480
            or receipt.get('remaining_episodes') != 220 or receipt.get('expected_episodes') != 700
            or receipt.get('required_smoke_tasks') != remaining[:8]):
        raise ValueError('retained/missing episode partition differs')
    old = receipt['old_config']
    if sha(old['path']) != old['sha256']:
        raise ValueError('original frozen configuration changed')
    compare_benchmark(read(old['path']), config)
    if config.get('stage') == 'formal':
        wanted, episodes = remaining, 20
    elif config.get('stage') == 'smoke':
        wanted, episodes = remaining[:8], acceptance_smoke_count(config)
    else:
        raise ValueError('continuation must retain ordinary smoke/formal gating')
    if config.get('tasks') != wanted or config.get('episodes') != episodes:
        raise ValueError('continuation task order/count differs from audited partition')
    expanded_acceptance_contract(config)
    if Path(config['output']).resolve() == Path(read(old['path'])['output']).resolve():
        raise ValueError('continuation cannot overwrite the failed source output')
    if not receipt.get('evidence'):
        raise ValueError('retention audit lacks file-bound evidence')
    if verify_evidence:
        for row in receipt['evidence']:
            if sha(row['path']) != row['sha256']:
                raise ValueError('retained evidence changed: ' + row['path'])
    return receipt


def finalize(config_path, guard_dir, destination, stamp):
    from full_baseline_sequence import require_clean_guard
    from video_artifacts import aggregate_native_metrics, validate_video_artifacts, write_domino_results
    config = read(config_path)
    receipt = validate_resumption(config)
    source = read(Path(guard_dir)/'source.json')
    require_clean_guard(guard_dir, config, source)
    completed = read(Path(config['output'])/'suite_completed.json')
    if (completed.get('complete') is not True or completed.get('valid_completed') != 220
            or completed.get('all_workers_exit_zero') is not True
            or completed.get('config_sha256') != canonical_hash(config) or completed.get('source') != source):
        raise ValueError('remaining 220 episodes are not cleanly complete')
    destination = Path(destination).resolve()
    storage = Path(config['storage_root']).resolve(strict=True)
    if destination.exists() or not destination.is_relative_to(storage) or destination == storage:
        raise ValueError('unified output must be a fresh artifact-disk directory')
    if not stamp or '/' in stamp or '\\' in stamp or stamp in ('.', '..'):
        raise ValueError('invalid unified result stamp')
    old_config = read(receipt['old_config']['path'])
    collected = []
    for task in receipt['full_tasks']:
        owner = old_config if task in receipt['retained_tasks'] else config
        audit, _, reports, metrics = audit_source_task(owner, task)
        collected.append((task, owner, audit, reports, metrics))
    # All originals pass before creating the new materialized report view.
    destination.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(config['domino_root'])/'script'))
    all_metrics, origins = [], []
    for task, owner, audit, reports, metrics in collected:
        native = destination/'eval_result'/task/'dynamicwam_full'/config['task_config']/Path(config['checkpoint']).stem/stamp
        native.mkdir(parents=True)
        for episode, report in enumerate(reports):
            origin = Path(owner['output'])/task/f'episode_{episode:04d}'
            directory = destination/task/f'episode_{episode:04d}'
            directory.mkdir(parents=True)
            copied = dict(report, origin_report=str(origin/'report.json'), origin_report_sha256=sha(origin/'report.json'),
                          origin_config_sha256=canonical_hash(owner), aggregate_view=True)
            for field, name in (('video', f'episode{episode}.mp4'), ('video_timestamps', f'episode{episode}_timestamps.json')):
                target = Path(report[field]).resolve(strict=True)
                link = native/name
                link.symlink_to(target)
                copied[field] = str(link)
            if not validate_video_artifacts(copied):
                raise ValueError('unified native video alias failed hash validation')
            save(directory/'report.json', copied)
            for name in ('manifest.json', 'expert_motion.pkl', 'metrics.json'):
                (directory/name).symlink_to((origin/name).resolve(strict=True))
            origins.append(dict(task=task, episode_index=episode, source_sha256=report['source_sha256'],
                                original_report=str(origin/'report.json'), original_sha256=copied['origin_report_sha256']))
        aggregate = aggregate_native_metrics(metrics)
        write_domino_results(native, stamp, config['instruction_type'], aggregate)
        all_metrics.extend(metrics)
        save(destination/task/'completed.json', dict(complete=True, episodes=20, original_output=owner['output'],
                                                   original_source_sha256=reports[0]['source_sha256'], native_result_dir=str(native)))
    aggregate = aggregate_native_metrics(all_metrics)
    summary = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in aggregate.get_summary().items()}
    result = dict(complete=True, valid_completed=700, expected_episodes=700, native_metrics=summary,
        retained_episodes=480, newly_evaluated_episodes=220, successes=sum(bool(r['success']) for r in all_metrics),
        success_rate_percent=100.*sum(bool(r['success']) for r in all_metrics)/700.,
        resumption_receipt=str(Path(config['resumption_receipt']).resolve()), resumption_receipt_sha256=config['resumption_receipt_sha256'],
        old_source=receipt['old_source'], new_source=source, checkpoint_sha256=config['checkpoint_sha256'],
        prepared_sha256=config['prepared_sha256'], benchmark_contract=config['benchmark_contract'],
        physical_cpu_workers_old=int(old_config.get('physx_cpu_workers', 0)),
        physical_cpu_workers_new=int(config.get('physx_cpu_workers', 0)),
        simulator_cpu_workers_change=config.get('simulator_cpu_workers_change'),
        physics_replay_receipt=config.get('physics_replay_receipt'),
        physics_replay_receipt_sha256=config.get('physics_replay_receipt_sha256'),
        passive_force_dedup_old=bool(old_config.get('passive_force_dedup', False)),
        passive_force_dedup_new=bool(config.get('passive_force_dedup', False)),
        passive_force_replay_receipt=config.get('passive_force_replay_receipt'),
        passive_force_replay_receipt_sha256=config.get('passive_force_replay_receipt_sha256'),
        passive_force_replay_guards=config.get('passive_force_replay_guards'),
        video_storage='symlink references to hash-bound originals; merger never writes original bytes', origins=origins)
    save(destination/'suite_completed.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    for name in ('old-config', 'old-guard', 'destination'):
        prepare_parser.add_argument('--'+name, type=Path, required=True)
    prepare_parser.add_argument('--decode-workers', type=int, default=2)
    final_parser = commands.add_parser('finalize')
    for name in ('config', 'guard', 'destination'):
        final_parser.add_argument('--'+name, type=Path, required=True)
    final_parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.old_config, args.old_guard, args.destination, args.decode_workers)
    else:
        finalize(args.config, args.guard, args.destination, args.stamp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
