#!/usr/bin/env python3
"""Finite CPU-only smoke -> acceptance -> formal dispatch and initial observation.

No NVML, retry, recovery, or continuing observer. The existing GPU guard owns
all device handles and descendant cleanup throughout both jobs.
"""
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

import guarded_dualrate_run as guard
from full_baseline_worker import ROOT, canonical_hash, proc_identity, read, save, sha

INITIAL_SECONDS = 300.
LIVENESS_CONTRACT = 'full_sync16_fresh_dispatch_within_3rows_v1'
DIFFERING_FIELDS = {'stage', 'tasks', 'episodes', 'output', 'result_stamp',
    'acceptance_receipt', 'acceptance_receipt_sha256', 'sequence_id',
    'task_timeout_seconds', 'cohort_timeout_seconds'}


def validate_pair(smoke, formal):
    """Only cohort size/output and administrative deadlines may differ."""
    if (smoke.get('run_kind') != 'full_baseline_suite' or formal.get('run_kind') != 'full_baseline_suite'
            or smoke.get('stage') != 'smoke' or formal.get('stage') != 'formal'
            or len(smoke.get('tasks', [])) != 8 or smoke.get('episodes') != 2
            or len(formal.get('tasks', [])) != 35 or formal.get('episodes') != 20
            or smoke['tasks'] != formal['tasks'][:8]):
        raise ValueError('sequence requires first-eight-task 8x2 smoke then 35x20 formal')
    if smoke.get('action_limit') is not None or formal.get('action_limit') is not None:
        raise ValueError('both phases require complete native episode budgets')
    for key in set(smoke) | set(formal):
        if key not in DIFFERING_FIELDS and smoke.get(key) != formal.get(key):
            raise ValueError('smoke/formal contract differs: ' + key)
    if not formal.get('benchmark_contract') or len(set(formal['gpu_uuids'])) != 8:
        raise ValueError('explicit shared benchmark contract and eight UUIDs required')
    if Path(smoke['output']).resolve() == Path(formal['output']).resolve():
        raise ValueError('smoke and formal outputs must be distinct')


def guard_command(config_path, run_dir, config, max_runtime, *, detach=False):
    result = [sys.executable, '-B', str(ROOT/'scripts/guarded_dualrate_run.py'),
        '--config', str(config_path), '--run-dir', str(run_dir),
        '--storage-root', config['storage_root'], '--gpus', ','.join(config['gpu_uuids']),
        '--max-runtime', str(max_runtime)]
    return result + (['--detach'] if detach else [])


def require_clean_guard(directory, config, source):
    directory = Path(directory)
    status = read(directory/'status.json')
    if status.get('state') != 'completed' or status.get('guard_closed') is not True or status.get('errors'):
        raise ValueError('guard did not complete with closed handles and no errors')
    owned = read(directory/'owned_exit.json')
    kernel = read(directory/'kernel_after.json')
    if owned.get('remaining') != [] or owned.get('returncode') != 0:
        raise ValueError('owned workers did not exit cleanly')
    if kernel.get('unchanged') is not True or kernel.get('same_boot') is not True or kernel.get('added') != []:
        raise ValueError('kernel/boot changed during evaluation')
    if read(directory/'source.json') != source or read(directory/'config.frozen.json') != config:
        raise ValueError('guard source/config binding differs')
    auth = read(directory/'authorization.json')
    if (auth.get('identity', {}).get('config_sha256') != canonical_hash(config)
            or auth.get('gpu_uuids') != config['gpu_uuids']):
        raise ValueError('guard authorization identity differs')
    events = read(directory/'guard_events.json')
    opened = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_open']
    closed = [(r['uuid'], r['path'], r['fd']) for r in events if r['event'] == 'after_close']
    if (len(opened) != 8 or len(closed) != 8 or set(opened) != set(closed)
            or {r[0] for r in opened} != set(config['gpu_uuids'])):
        raise ValueError('all eight guard descriptors must open and close exactly once')
    return [directory/name for name in ('status.json', 'owned_exit.json', 'kernel_before.json',
        'kernel_after.json', 'source.json', 'config.frozen.json', 'authorization.json', 'guard_events.json')]


def audit_full_liveness(report, config):
    """A usable published observation must cause dispatch, not endless wait holds.

    This is an acceptance bound, not a relaxed runtime deadline: three action
    rows allow more than the .25s maximum physical backlog. Normal Full model
    inference is excluded; only idle time after the previous A16 ends is tested.
    Both physical and wall-equivalent observation ages must be valid.
    """
    dt = float(config['action_dt']); tolerance = 1e-7
    bound = 3*dt; max_age = float(config['max_observation_age'])
    query_timeout = float(config.get('model_query_timeout_s', 30.))
    events = report.get('events', [])
    activation = [e for e in events if e.get('event') == 'activation']
    bootstrap = [e for e in events if e.get('event') == 'full_bootstrap_plan']
    if len(activation) != 1 or len(bootstrap) != 1 or bootstrap[0].get('rows') != 16:
        raise ValueError('liveness requires one activation and complete bootstrap A16')
    origin = float(activation[0]['physical_time'])
    terminal = origin+float(report['physical_duration'])
    bootstrap_end = origin+(int(bootstrap[0]['install_row'])+16)*dt
    block_end = bootstrap_end
    latest = None; pending = None; fresh_since = None
    queries = installs = 0; max_wait = 0.; fresh_opportunities = 0
    last_time = origin
    for event in events:
        if event.get('post_terminal'):
            continue
        kind = event.get('event')
        clock = event.get('clock_pair')
        if not clock or not clock.get('active'):
            continue
        now = float(clock['physical_time']); wall_now = float(clock['expected_physical_time'])
        if not all(math.isfinite(v) for v in (now, wall_now)) or now < last_time-tolerance:
            raise ValueError('liveness event clocks are invalid or reordered')
        last_time = now
        if fresh_since is not None:
            max_wait = max(max_wait, now-fresh_since)
            if now-fresh_since > bound+tolerance:
                raise ValueError('Full query starvation: fresh observation available for more than three rows')
        if kind == 'capture':
            latest = (float(event['exposure']), float(event['available']))
        elif kind == 'full_query_started':
            if pending is not None or now < block_end-tolerance:
                raise ValueError('Full query is overlapping or precedes complete A16 execution')
            if any(not -tolerance <= float(event[key]) <= max_age+tolerance
                   for key in ('observation_physical_age', 'observation_wall_age')):
                raise ValueError('Full query used an observation stale in physical or wall time')
            pending = (event['query_id'], now)
            queries += 1; fresh_since = None
        elif kind == 'full_plan_installed':
            if pending is None or event['query_id'] != pending[0] or event.get('rows') != 16:
                raise ValueError('Full installation has no matching A16 query')
            installs += 1; pending = None; fresh_since = None
            block_end = float(event['execution_end_time'])
            if block_end <= now:
                raise ValueError('Full installed plan has no future execution support')
        if pending is None and now >= block_end-tolerance and latest is not None:
            exposure, available = latest
            physical_age, wall_age = now-exposure, wall_now-exposure
            if (available <= now+tolerance and -tolerance <= physical_age <= max_age+tolerance
                    and -tolerance <= wall_age <= max_age+tolerance and fresh_since is None):
                fresh_since = now; fresh_opportunities += 1
    if fresh_since is not None:
        max_wait = max(max_wait, terminal-fresh_since)
        if terminal-fresh_since > bound+tolerance:
            raise ValueError('Full query starvation continued to episode termination')
    if pending is not None and terminal-pending[1] > query_timeout+tolerance:
        raise ValueError('Full query failed to install within its bounded model timeout')
    if terminal > bootstrap_end+query_timeout+bound and installs == 0:
        raise ValueError('long Full episode never installed a nonbootstrap plan')
    if report.get('inference_calls') != queries+1:
        raise ValueError('Full inference call count differs from recorded query events')
    return dict(contract=LIVENESS_CONTRACT, passed=True, nonbootstrap_queries=queries,
        nonbootstrap_installs=installs, bootstrap_end_time=bootstrap_end, terminal_time=terminal,
        fresh_opportunities=fresh_opportunities, max_fresh_dispatch_wait_seconds=max_wait,
        dispatch_bound_seconds=bound, short_episode=terminal <= bootstrap_end+tolerance)


def audit_task_liveness(reports, config):
    audits = [audit_full_liveness(report, config) for report in reports]
    if (any(not row['short_episode'] for row in audits)
            and not any(row['nonbootstrap_queries'] for row in audits)):
        raise ValueError('smoke task exceeded initial held2+A16 but never issued a subsequent query')
    return audits


def make_acceptance(smoke, smoke_guard, source, destination):
    from full_baseline_suite import audit_task
    from video_artifacts import probe_head_video
    evidence = require_clean_guard(smoke_guard, smoke, source)
    completed_path = Path(smoke['output'])/'suite_completed.json'
    completed = read(completed_path)
    if (completed.get('complete') is not True or completed.get('valid_completed') != 16
            or completed.get('every_video_fully_decoded') is not True
            or completed.get('all_workers_exit_zero') is not True
            or completed.get('config_sha256') != canonical_hash(smoke)
            or completed.get('source') != source):
        raise ValueError('actual eight-GPU sixteen-episode smoke is incomplete')
    evidence.append(completed_path)
    decoded = []; liveness = {}
    for task in smoke['tasks']:
        reports, _ = audit_task(smoke, task)
        liveness[task] = audit_task_liveness(reports, smoke)
        for report in reports:
            # Re-decode outside all active episodes; retain the independent receipt.
            probe = probe_head_video(Path(report['video']))
            if probe != report['video_probe']:
                raise ValueError('independent complete MP4 decode differs')
            path = Path(smoke['output'])/task/f"episode_{report['episode_index']:04d}/report.json"
            evidence.extend((path, Path(report['video']), Path(report['video_timestamps'])))
            decoded.append(dict(report=str(path), video=report['video'], probe=probe))
    if len(decoded) != 16:
        raise ValueError('exactly sixteen full episodes required')
    decode_path = Path(destination).with_name('smoke_video_decode.json')
    save(decode_path, dict(passed=True, count=16, decoded=decoded))
    evidence.append(decode_path)
    liveness_path = Path(destination).with_name('smoke_liveness.json')
    save(liveness_path, dict(passed=True, contract=LIVENESS_CONTRACT, tasks=liveness))
    evidence.append(liveness_path)
    receipt = dict(passed=True, source_contract=guard.source_contract(source),
        benchmark_contract=smoke['benchmark_contract'], checkpoint_sha256=smoke['checkpoint_sha256'],
        world_size=8, complete_smoke_episodes=16, native_mp4_full_decode=True,
        full_query_liveness_passed=True, liveness_contract=LIVENESS_CONTRACT,
        clean_worker_exit=True, guard_closed=True, kernel_unchanged=True,
        evidence=[dict(path=str(p.resolve()), sha256=sha(p)) for p in evidence])
    save(destination, receipt)
    return receipt


def live_identity(expected, *, parent=None):
    actual = proc_identity(expected['pid'])
    if (actual['start_ticks'] != expected['start_ticks'] or actual['state'] in ('Z', 'X')
            or (parent is not None and actual['ppid'] != parent)):
        raise ValueError('owned process exited, changed parent, or PID was reused')
    return actual


def initial_snapshot(config, directory, *, now=None):
    """Read saved status and /proc only. A missing/transitioning cohort is pending."""
    directory = Path(directory)
    status = read(directory/'status.json')
    if status['state'] in ('failed', 'failed_cleaning', 'cleanup_pending', 'completed'):
        raise RuntimeError('formal guard terminated before initial acceptance: ' + status['state'])
    if status['state'] != 'running':
        return None
    supervisor = live_identity(status['supervisor'])
    suite = live_identity(dict(pid=status['worker_pid'], start_ticks=status['worker_start_ticks']), parent=supervisor['pid'])
    auth = read(directory/'authorization.json')
    if (auth['external_guard']['pid'] != supervisor['pid'] or
            auth['external_guard']['start_ticks'] != supervisor['start_ticks']):
        raise ValueError('formal guard authorization owner differs')
    held = {str(p.resolve()) for p in (Path('/proc')/str(supervisor['pid'])/'fd').iterdir()}
    if not set(auth['external_guard']['device_paths']) <= held:
        raise ValueError('formal guard descriptor coverage was lost')
    output = Path(config['output'])
    if not (output/'suite_started.json').exists() or not (output/'suite_status.json').exists():
        return None
    started, progress = read(output/'suite_started.json'), read(output/'suite_status.json')
    if progress['state'] == 'failed':
        raise RuntimeError('formal suite failed: ' + progress.get('error', ''))
    if progress['state'] != 'running' or len(progress.get('running', [])) != 8:
        return None
    workers = []
    for row in progress['running']:
        try:
            identity = proc_identity(row['pid'])
        except (FileNotFoundError, ProcessLookupError):
            return None  # An ordinary cohort transition; never a successful snapshot.
        if identity['state'] in ('Z', 'X'):
            return None
        if identity['ppid'] != suite['pid'] or identity['session'] != suite['session']:
            raise ValueError('formal workers are outside their owned suite')
        workers.append(dict(identity, task=row['task'], gpu_uuid=row['gpu_uuid']))
    if {r['gpu_uuid'] for r in workers} != set(config['gpu_uuids']):
        raise ValueError('formal workers do not cover the eight allocated GPUs')
    now = time.time() if now is None else now
    elapsed = now-started['started_unix']
    if elapsed < INITIAL_SECONDS:
        return None
    paths = [output/task/'episode_0000/report.json' for task in config['tasks'][:8]]
    if not all(path.exists() for path in paths):
        return None
    return dict(elapsed_seconds=elapsed, supervisor=supervisor, suite=suite,
                workers=workers, first_reports=[str(p) for p in paths])


def observe_initial(config, directory, destination, timeout):
    from full_baseline_worker import checked_episode
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if (Path(directory)/'status.json').exists():
            snapshot = initial_snapshot(config, directory)
            if snapshot is not None:
                reports = [checked_episode(config, task, 0)[0] for task in config['tasks'][:8]]
                liveness = {task: audit_task_liveness([report], config)
                    for task, report in zip(config['tasks'][:8], reports)}
                before = read(Path(directory)/'kernel_before.json')
                if (guard.base.kernel_evidence() != before['lines'] or
                        Path('/proc/sys/kernel/random/boot_id').read_text().strip() != before['boot_id']):
                    raise ValueError('kernel changed during formal initial window')
                source = read(Path(directory)/'source.json')
                if guard.frozen_source(ROOT) != source:
                    raise ValueError('formal source changed during initial window')
                # Validate process liveness again after potentially expensive video hashing.
                snapshot = initial_snapshot(config, directory)
                if snapshot is None:
                    continue
                receipt = dict(snapshot, passed=True, initial_only=True, no_retry=True,
                    observer_exits_now=True, formal_complete=False, kernel_unchanged=True,
                    validated_first_episodes=len(reports), source_contract=guard.source_contract(source),
                    full_query_liveness_passed=True, liveness=liveness,
                    first_report_hashes=[dict(path=p, sha256=sha(p)) for p in snapshot['first_reports']])
                save(destination, receipt)
                return receipt
        time.sleep(5.)
    raise TimeoutError('finite formal initial acceptance window expired')


def request_owned_stop(expected):
    """Leave process/FD escalation to that exact guard, including pending cleanup."""
    try:
        live_identity(expected)
        os.kill(expected['pid'], signal.SIGTERM)
        return dict(requested=True, owner=expected, cleanup_owner='existing_guard')
    except (FileNotFoundError, ProcessLookupError):
        return dict(requested=False, already_exited=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-config', type=Path, required=True)
    parser.add_argument('--formal-config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--storage-root', type=Path, default=Path('/SSD_DISK/users/wuruihan'))
    parser.add_argument('--smoke-max-runtime', type=float, default=7200.)
    parser.add_argument('--formal-max-runtime', type=float, default=172800.)
    parser.add_argument('--initial-timeout', type=float, default=7200.)
    args = parser.parse_args()
    if os.name != 'posix' or not INITIAL_SECONDS < args.initial_timeout <= 21600:
        parser.error('Linux and bounded initial timeout >300s and <=21600s required')
    if not all(0 < x <= 30*86400 for x in (args.smoke_max_runtime, args.formal_max_runtime)):
        parser.error('bounded positive guard runtimes <=30 days required')
    smoke, formal = read(args.smoke_config), read(args.formal_config)
    validate_pair(smoke, formal)
    if args.gpus.split(',') != smoke['gpu_uuids'] or Path(smoke['storage_root']).resolve() != args.storage_root.resolve():
        parser.error('CLI GPU/storage allocation differs from frozen configurations')
    out = guard.base.storage_directory(args.run_dir, args.storage_root)
    out.mkdir(parents=True, exist_ok=False)
    source = guard.frozen_source(ROOT)
    formal_owner = None
    def state(value, **extra):
        save(out/'status.json', dict(state=value, controller=proc_identity(os.getpid()), no_retry=True, **extra))
    try:
        state('smoke_running')
        smoke_path = out/'smoke.config.json'
        save(smoke_path, smoke)
        with (out/'smoke_guard.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(guard_command(smoke_path, out/'smoke_guard', smoke, args.smoke_max_runtime),
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError('smoke guard failed; no formal dispatch or automatic retry')
        state('validating_smoke')
        acceptance = out/'acceptance.json'
        make_acceptance(smoke, out/'smoke_guard', source, acceptance)
        formal = dict(formal, acceptance_receipt=str(acceptance), acceptance_receipt_sha256=sha(acceptance))
        formal_path = out/'formal.config.json'
        save(formal_path, formal)
        guard.validate_acceptance(formal, source)
        state('dispatching_formal')
        result = subprocess.run(guard_command(formal_path, out/'formal_guard', formal,
            args.formal_max_runtime, detach=True), cwd=ROOT, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=True, timeout=30)
        import json
        dispatch = json.loads(result.stdout)
        formal_owner = proc_identity(dispatch['supervisor_pid'])
        save(out/'formal_dispatch.json', dict(dispatch, owner=formal_owner, argv=guard_command(
            formal_path, out/'formal_guard', formal, args.formal_max_runtime, detach=True)))
        state('formal_initial_observation', supervisor=formal_owner)
        receipt = observe_initial(formal, out/'formal_guard', out/'initial_acceptance.json', args.initial_timeout)
        state('formal_running_initial_accepted', supervisor=formal_owner,
              initial_elapsed_seconds=receipt['elapsed_seconds'], observer_finished=True, formal_complete=False)
        return 0
    except BaseException as exc:
        stop = None
        if formal_owner is not None:
            try:
                stop = request_owned_stop(formal_owner)
            except BaseException as stop_error:
                stop = dict(requested=False, error=repr(stop_error), cleanup_owner='existing_guard')
        state('failed', error=repr(exc), traceback=traceback.format_exc(), formal_stop=stop,
              observer_finished=True, formal_complete=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
