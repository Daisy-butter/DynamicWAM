#!/usr/bin/env python3
"""Finite two-environment V4 suite: three GPUs each, two guarded GPUs idle."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from dualrate_eval_worker import (ROOT, MODE, EpisodeCohort, canonical_hash, environment_groups,
    checked_episode, load_frozen_task, proc_identity, read, save, sha,
    validate_authorization, validate_config, verify_artifacts)


def worker_command(config_path, authorization, task, gpu_index, cohort):
    return [sys.executable, '-B', str(ROOT / 'scripts/dualrate_eval_worker.py'),
            '--config', str(config_path), '--authorization', str(authorization),
            '--task', task, '--gpu-index', str(gpu_index), '--cohort-dir', str(cohort)]


def worker_environment(config, gpu_index, output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(environment_groups(config)[gpu_index]),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    # Per-worker cache prevents compilation/cache writers colliding. Same process
    # session is retained deliberately for the parent's exact-SID cleanup.
    cache = Path(output) / 'worker_cache' / f'gpu_{gpu_index}'
    cache.mkdir(parents=True, exist_ok=False)
    env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'))
    return env


def audit_task(config, task):
    from dualrate_retention import retained_owner
    owner = retained_owner(config, task)
    if owner is not None:
        return audit_task(owner, task)
    output = Path(config['output'])
    completed = read(output / task / 'completed.json')
    if (completed.get('complete') is not True or completed.get('task') != task
            or completed.get('episodes') != config['episodes']
            or completed.get('protocol') != config['protocol']
            or completed.get('config_sha256') != canonical_hash(config)):
        raise ValueError('worker did not publish a complete matching task: ' + task)
    reports, metrics = [], []
    for episode in range(config['episodes']):
        report, metric = checked_episode(config, task, episode)
        reports.append(report)
        metrics.append(metric)
    if len(list((output / task).glob('episode_*/report.json'))) != config['episodes']:
        raise ValueError('unexpected extra or missing episode reports')
    native = Path(completed['native_result_dir'])
    if len(list(native.glob('episode*.mp4'))) != config['episodes']:
        raise ValueError('native task MP4 count differs from episode count')
    for filename in ('_result.txt', '_metrics.json', '_metrics_report.txt', '_episodes_detail.json'):
        if not (native / filename).is_file():
            raise ValueError('native task result artifact missing: ' + filename)
    return reports, metrics


def consumed_background_versions(report):
    """Prove an accepted version >=2 predates its fast query and later commit."""
    completed, started, used = {}, {}, set()
    for row in report.get('events', []):
        event = row.get('event')
        if event == 'slow_completed' and row.get('accepted') is True:
            completed[row['version']] = row
        elif event == 'fast_started':
            started[row['query_id']] = row
        elif event == 'fast_committed':
            version = row.get('reference_version', -1)
            slow, query = completed.get(version), started.get(row.get('query_id'))
            if (version < 2 or slow is None or query is None
                    or query.get('reference_version') != version or query.get('query_time') != row.get('query_time')):
                continue
            times = [slow.get('source_time'), slow.get('completed_time'), row.get('query_time'), row.get('commit_time')]
            if (all(type(t) in (int, float) and math.isfinite(t) for t in times)
                    and times == sorted(times)):
                used.add(version)
    if not used <= set(report.get('reference_versions_consumed', [])):
        raise ValueError('background reference event proof differs from reported consumption')
    return sorted(used)


def deployment_acceptance(reports):
    graph = bool(reports) and all(r.get('fast_graph_acceptance', {}).get('enabled')
        and r['fast_graph_acceptance'].get('passed') and r['fast_graph_acceptance'].get('sealed')
        and r['fast_graph_acceptance'].get('shape_count') == 2 for r in reports)
    during = sum(r.get('fast_queries_during_slow', 0) for r in reports)
    overlap = sum(r.get('model_overlap_fast_queries', 0) for r in reports)
    camera_steps = sum(r.get('physics_steps_during_camera', 0) for r in reports)
    fast_steps = sum(r.get('physics_steps_during_fast_inflight', 0) for r in reports)
    consumed = [consumed_background_versions(r) for r in reports]
    refreshed = any(consumed)
    return dict(fast_graph_passed=graph, fast_queries_during_slow=during,
        model_overlap_fast_queries=overlap, refreshed_reference_consumed=refreshed,
        physics_steps_during_camera=camera_steps, physics_steps_during_fast_inflight=fast_steps,
        consumed_background_versions=consumed,
        passed=bool(graph and during and overlap and refreshed and camera_steps>0 and fast_steps>0))


def cleanup_children(active):
    """Signal only direct children created by this suite; parent owns escalation."""
    for row in active:
        if row['process'].poll() is None:
            row['process'].terminate()
    deadline = time.monotonic() + 15.
    for row in active:
        try:
            row['process'].wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Never release the all-GPU guard here; parent supervisor must reap
            # the still-owned SID before closing those descriptors.
            pass
        row['log'].close()


def require_healthy_cohort(output, directory, members):
    """Observe failed-runtime signals even when the worker is stuck joining CUDA."""
    if (Path(directory) / 'abort.json').exists():
        raise RuntimeError('cohort abort marker published; terminate owned workers without retry')
    for task in members:
        marker = Path(output) / task / 'runtime_failure.json'
        if marker.exists():
            raise RuntimeError('runtime failed before thread join: ' + task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    config_path, authorization = Path(args.config).resolve(strict=True), Path(args.authorization).resolve(strict=True)
    config = read(config_path)
    output = validate_config(config)
    receipt = validate_authorization(config, authorization)
    verify_artifacts(config)
    # Verify every selected task's entire 20-scene chain before any GPU worker
    # starts, including smoke runs that execute only the first two episodes.
    for task in config['tasks']:
        load_frozen_task(config, task)
    output.mkdir(parents=True, exist_ok=False)
    started_unix, began = time.time(), time.monotonic()
    if started_unix >= receipt['expires_at_unix']:
        raise ValueError('authorization expired during CPU preparation')
    claim_path = authorization.with_suffix('.json.started')
    with claim_path.open('x', encoding='utf-8') as handle:
        import json
        json.dump(dict(**proc_identity(os.getpid()), started_unix=started_unix,
                       receipt_sha256=sha(authorization), config_sha256=canonical_hash(config)), handle)
    save(output / 'suite_started.json', dict(protocol=config['protocol'], stage=config['stage'],
        tasks=config['tasks'], episodes_per_task=config['episodes'], seed=config['seed'],
        config_sha256=canonical_hash(config), source=receipt['source'], checkpoint_sha256=config['checkpoint_sha256'],
        prepared_sha256=config['prepared_sha256'], gpu_uuids=config['gpu_uuids'],
        resources=dict(parallel_environments=2, gpu_per_environment=3, placement='renderer0_fast1_slow2',
                       environment_groups=environment_groups(config), idle_guarded_gpu_uuids=config['gpu_uuids'][6:]),
        started_unix=started_unix, no_retry=True))
    active, finished, all_reports, all_metrics = [], [], [], []
    def interrupted(signum, _frame):
        raise InterruptedError(f'suite received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        from dualrate_retention import initialize_view, provenance as retention_provenance
        retained_tasks, all_reports, all_metrics = initialize_view(config, receipt['source'])
        finished.extend(retained_tasks)
        for cohort_index, start in enumerate(range(0, len(config['tasks']), 2)):
            members = config['tasks'][start:start + 2]
            if set(members) <= set(retained_tasks):
                continue  # Whole original cohort already audited and linked.
            cohort = output / 'cohorts' / f'cohort_{cohort_index:03d}'
            save(cohort / 'manifest.json', dict(tasks=members, episodes=config['episodes'],
                timeout_seconds=float(config.get('cohort_timeout_seconds', 3600.)),
                mode='inactive_setup_and_finalization_barriers_v1', config_sha256=canonical_hash(config)))
            active = []
            for index, task in enumerate(members):
                logfile = output / 'logs' / (task + '.log')
                logfile.parent.mkdir(parents=True, exist_ok=True)
                log = logfile.open('x', encoding='utf-8')
                env = worker_environment(config, index, cohort)
                command = worker_command(config_path, authorization, task, index, cohort)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                except BaseException:
                    log.close()
                    raise
                active.append(dict(task=task, gpu_index=index, process=process, log=log, started=time.monotonic()))
            while active:
                require_healthy_cohort(output, cohort, members)
                save(output / 'suite_status.json', dict(state='running', stage=config['stage'], elapsed_seconds=time.monotonic()-began,
                    finished_tasks=finished, running=[dict(task=r['task'], pid=r['process'].pid,
                    gpu_uuids=environment_groups(config)[r['gpu_index']]) for r in active],
                    pending_tasks=config['tasks'][start+len(members):], no_retry=True))
                for row in list(active):
                    code = row['process'].poll()
                    if code is None:
                        if time.monotonic() - row['started'] > float(config.get('task_timeout_seconds', 21600.)):
                            raise TimeoutError('task wall budget exceeded: ' + row['task'])
                        continue
                    row['log'].close()
                    if code != 0:
                        raise RuntimeError(f"worker {row['task']} exited {code}; retry disabled")
                    reports, metrics = audit_task(config, row['task'])
                    all_reports.extend(reports)
                    all_metrics.extend(metrics)
                    finished.append(row['task'])
                    active.remove(row)
                if active:
                    time.sleep(2.)
        from video_artifacts import aggregate_native_metrics
        sys.path.insert(0, str(Path(config['domino_root']) / 'script'))
        aggregate = aggregate_native_metrics(all_metrics)
        expected = len(config['tasks']) * config['episodes']
        if len(all_reports) != expected:
            raise RuntimeError('suite completed-count mismatch')
        acceptance = deployment_acceptance(all_reports)
        if config['stage'] == 'smoke' and not acceptance['passed']:
            save(output/'deployment_acceptance.json', acceptance)
            raise RuntimeError('complete smoke lacks actual overlap/new reference/Graph acceptance')
        summary = aggregate.get_summary()
        # Native score summaries may intentionally contain NaN for an absent
        # event class. Preserve semantic null instead of invalid JSON.
        import math
        summary = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in summary.items()}
        verify_artifacts(config)
        completed = dict(complete=True, protocol=config['protocol'], stage=config['stage'],
            retention=retention_provenance(config, receipt['source']),
            valid_completed=expected, expected_episodes=expected, tasks=len(config['tasks']),
            episodes_per_task=config['episodes'], successes=sum(bool(r['success']) for r in all_reports),
            success_rate_percent=100.*sum(bool(r['success']) for r in all_reports)/expected,
            native_metrics=summary, every_video_fully_decoded=True, all_workers_exit_zero=True,
            deployment_acceptance=acceptance,
            resources=dict(parallel_environments=2, gpu_per_environment=3, environment_groups=environment_groups(config),
                           idle_guarded_gpu_uuids=config['gpu_uuids'][6:]),
            source=receipt['source'], checkpoint_sha256=config['checkpoint_sha256'], prepared_sha256=config['prepared_sha256'],
            config_sha256=canonical_hash(config), elapsed_seconds=time.monotonic()-began, no_retry=True,
            reports=[dict(path=str(output/r['task']/f"episode_{r['episode_index']:04d}/report.json"),
                sha256=sha(output/r['task']/f"episode_{r['episode_index']:04d}/report.json")) for r in all_reports])
        save(output / 'suite_completed.json', completed)
        save(output / 'suite_status.json', dict(state='completed', valid_completed=expected, no_retry=True))
        save(authorization.with_suffix('.json.consumed'), dict(**proc_identity(os.getpid()),
            receipt_sha256=sha(authorization), suite_completed_sha256=sha(output/'suite_completed.json'), output=str(output)))
        return 0
    except BaseException as exc:
        for row in active:
            EpisodeCohort(cohort, row['task']).abort(exc)
        save(output / 'suite_status.json', dict(state='failed', complete=False, error=repr(exc),
            traceback=traceback.format_exc(), finished_tasks=finished, no_retry=True))
        raise
    finally:
        cleanup_children(active)


if __name__ == '__main__':
    raise SystemExit(main())
