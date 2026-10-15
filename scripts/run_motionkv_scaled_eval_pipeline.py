#!/usr/bin/env python3
"""One-shot formal 35x20 restart from frozen scenes after fresh guarded 8-GPU smoke.

Install this standalone script on SSD OUTSIDE the frozen repository. Start it
under nohup/setsid; it owns one supervised evaluation stage, not an automation.
It never queries GPUs: only the stage supervisor performs its bounded 300s
startup checks and final guard-close/driver verification. No retry or resume.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SSD = Path('/SSD_DISK/users/wuruihan')
CHECKPOINT = '/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/motionkv_rtc_adapt_8gpu_20261014/exports/multirate_step_12000.pt'
LIFETIME_MODE = 'fds_before_telemetry_v1'
PROTOCOL = 'continuous_world_scaled_wall_motionkv_rtc_h4r4_v1'
SIMULATION_RATE = .25
PREPARED = Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/motionkv_rtc_20261014/formal35x20_prepare')
FROZEN_SCENE_SHA = 'cff71ba1b8bf17c58082e615016e20775077787b455746e87261eaaaf60b6fb2'
DRIVER_KEYS = ('host', 'boot_id', 'anomaly_count', 'driver_anomalies_sha256', 'latest_driver_anomaly_sha256')


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition, message):
    if not condition: raise RuntimeError(message)
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)
def identity(driver): return tuple(driver.get(key) for key in DRIVER_KEYS)
def inside_ssd(path):
    result = Path(path).resolve()
    require(result.is_relative_to(SSD), f'path escapes verified SSD tree: {result}')
    return result


def source_identity(repo):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=repo, check=True)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=repo).decode().split('\0')
    digest = hashlib.sha256()
    for name in sorted({name for name in names if name.startswith(('src/', 'scripts/')) and name.endswith('.py')}):
        digest.update(name.encode() + b'\0' + (repo / name).read_bytes() + b'\0')
    return head, digest.hexdigest()


def verify_supervisor(directory, *, stage, uuids, authorization, authorization_sha, source_sha, returncode=0):
    directory = Path(directory)
    status, manifest = read(directory / 'status.json'), read(directory / 'manifest.json')
    require(returncode == 0 and status.get('final_exit_code') == 0 and status.get('exit_code') == 0,
            f'{stage}: supervisor/worker did not actually exit zero')
    require(status.get('ended_at') and not status.get('stop_reason'), f'{stage}: missing clean terminal status')
    for key in ('driver_exit_clean', 'guard_all_handles_closed', 'guard_workers_exited'):
        require(status.get(key) is True, f'{stage}: {key} did not pass')
    require(authorization.get('device_lifetime_mode') == LIFETIME_MODE
            and manifest.get('device_lifetime_mode') == LIFETIME_MODE, f'{stage}: unguarded telemetry mode')
    require(manifest.get('preflight_only', False) is False and status.get('worker_started') is True,
            f'{stage}: receipt was not an actual worker run')
    require(datetime.fromisoformat(status['started_at']) >= datetime.fromisoformat(authorization['created_at']),
            f'{stage}: smoke/run predates fresh authorization')
    require(manifest.get('stage') == stage and manifest.get('source_sha256') == source_sha,
            f'{stage}: wrong source/stage')
    for key in ('worker_gpu_uuids', 'guard_gpu_uuids'):
        require(len(manifest.get(key, [])) == 8 and set(manifest[key]) == set(uuids), f'{stage}: {key} differs')
    require(manifest.get('authorization_sha256') == authorization_sha, f'{stage}: authorization changed')
    require(manifest.get('device_guard_sha256') == authorization['guard_sha256'], f'{stage}: guard source changed')
    require(identity(manifest['driver']) == identity(authorization['driver']) == identity(read(directory / 'driver_postexit.json')),
            f'{stage}: driver evidence changed')
    events_path = directory / 'device_guard_events.json'
    require(sha(events_path) == status.get('device_guard_events_sha256'), f'{stage}: guard event digest changed')
    events = read(events_path)
    for uuid in uuids:
        opened = [e for e in events if e.get('event') == 'after_open' and e.get('uuid') == uuid]
        closed = [e for e in events if e.get('event') == 'after_close' and e.get('uuid') == uuid]
        require(len(opened) == len(closed) == 1, f'{stage}: missing/duplicate guard lifecycle for {uuid}')
        require(opened[0]['fd'] == closed[0]['fd'] and opened[0]['path'] == closed[0]['path']
                and opened[0]['monotonic'] < closed[0]['monotonic'], f'{stage}: unmatched guard close')
    return status


def verify_suite(output, *, expected, tasks=None, prepare=False, full_formal=False):
    output = Path(output)
    summary, completion, spec = (read(output / name) for name in ('summary.json', 'suite_completed.json', 'suite_started.json'))
    require(summary.get('complete') is True and completion.get('complete') is True
            and completion.get('all_processes_exited') is True, 'suite did not complete with all workers exited')
    require(all(item.get('valid_completed') == expected and item.get('planned_total') == expected
                for item in (summary, completion)), 'suite count differs from required valid episodes')
    require(not any(row.get('exit_code') != 0 for row in completion.get('finished', [])), 'suite has nonzero worker exits')
    require(len(completion.get('finished', [])) == len(spec['tasks']), 'missing task exit receipts')
    require({row.get('task') for row in completion['finished']} == set(spec['tasks']), 'task exit receipts contain duplicates/missing tasks')
    require(sum(row.get('valid_completed', 0) for row in summary['tasks']) == expected
            and all(not row.get('errors') and row.get('completion_marker_valid') for row in summary['tasks']),
            'per-task suite validation failed')
    require(len(summary['tasks']) == len(spec['tasks'])
            and {row.get('task') for row in summary['tasks']} == set(spec['tasks']), 'summary task identities differ')
    if tasks is not None:
        require(set(spec['tasks']) == set(tasks) and len(spec['tasks']) == len(tasks), 'suite task list changed')
    require(spec.get('prepare_only') is prepare, 'suite phase mismatch')
    if full_formal: require(summary.get('formal_35x20') is True and spec.get('probe_action_limit') is None,
                            'formal suite is a truncated probe')
    if not prepare:
        require(spec.get('protocol') == PROTOCOL and summary.get('protocol') == PROTOCOL, 'RTC protocol changed')
        require(spec.get('virtual_delay') is False and spec.get('simulation_rate') == SIMULATION_RATE, 'suite clock rate differs')
    if prepare:
        frozen = read(output / 'frozen_manifest.json')
        episodes = frozen['episodes']
        require(frozen.get('complete') is True and len(episodes) == 700, 'scene manifest is not complete 700')
        require(len({(row['task'], row['episode_index']) for row in episodes}) == 700, 'duplicate frozen episodes')
        require(Counter(row['task'] for row in episodes) == Counter({task: 20 for task in spec['tasks']})
                and len(spec['tasks']) == 35, 'frozen scenes are not exactly 35x20')
        for row in episodes:
            directory = output / row['task'] / f"episode_{row['episode_index']:04d}"
            for key, filename in (('manifest_sha256', 'manifest.json'), ('motion_sha256', 'expert_motion.pkl')):
                require(sha(directory / filename) == row[key], f'frozen scene changed: {directory}/{filename}')
    return spec, summary


def decode_videos(output, destination):
    """CPU decoder only; validate every saved video through its final frame."""
    spec = read(Path(output) / 'suite_started.json')
    records, seen = [], set()
    for task in spec['tasks']:
        for episode in range(spec['episodes_per_task']):
            report = read(Path(output) / task / f'episode_{episode:04d}' / 'report.json')
            path = inside_ssd(report['video'])
            require(path not in seen and sha(path) == report['video_sha256'], 'duplicate/changed video')
            seen.add(path)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames',
                '-of', 'json', str(path)], capture_output=True, text=True, timeout=120, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            require(int(stream['nb_read_frames']) == report['video_frames'] > 0
                    and stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p', 'video frame/codec mismatch')
            numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
            require(denominator > 0 and numerator / denominator == 10., 'video differs from native 10fps')
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-hwaccel', 'none',
                       '-threads', '1', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-']
            decoded = subprocess.run(command, capture_output=True, text=True, timeout=120)
            require(decoded.returncode == 0 and not decoded.stderr.strip(), f'video full decode failed: {path}')
            records.append(dict(path=str(path), sha256=report['video_sha256'], frames=report['video_frames'],
                                ffprobe=stream, full_decode_exit_code=decoded.returncode, command=command))
            save(destination, dict(complete=False, videos=records))
    require(len(records) == len(spec['tasks']) * spec['episodes_per_task'], 'video count differs from complete suite')
    save(destination, dict(complete=True, videos=records))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--wait-for-smoke', action='store_true')
    parser.add_argument('--repo', type=Path, default=Path('/home/wuruihan/DynamicWAM'))
    for name in ('eval-root', 'gpus', 'expected-head', 'expected-source-sha256', 'smoke-guard', 'smoke-output'):
        parser.add_argument('--' + name)
    parser.add_argument('--checkpoint', type=Path, default=Path(CHECKPOINT))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seed', type=int, default=100000)
    args = parser.parse_args()
    if not args.self_test:
        for name in ('eval_root', 'gpus', 'expected_head', 'expected_source_sha256', 'smoke_guard', 'smoke_output'):
            if not getattr(args, name): parser.error('--' + name.replace('_', '-') + ' is required')
    return args


def main(args):
    root, repo = inside_ssd(args.eval_root), args.repo.resolve()
    uuids = args.gpus.split(',')
    require(len(uuids) == len(set(uuids)) == 8 and all(uuid.startswith('GPU-') for uuid in uuids), 'need eight exact distinct GPU UUIDs')
    directory = root / 'pipeline'
    directory.mkdir(exist_ok=False)
    status_path = directory / 'pipeline_status.json'
    status = dict(state='validating', pipeline_pid=os.getpid(), created_at=utc(), phases=[],
                  retry_policy='none', recurring_gpu_polling=False, startup_health_seconds=300)
    active = None
    def persist():
        status['timestamp'] = utc(); save(status_path, status)
    def interrupted(signum, frame): raise RuntimeError(f'pipeline received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    persist()
    try:
        config = args.config.resolve() if args.config else repo / 'configs/motionkv_rtc_eval_20261014.yaml'
        auth_path, audit_path, timing_path = (root / name for name in
            ('health_authorization.json', 'checkpoint_audit.json', 'timing_contract.json'))
        auth, audit = read(auth_path), read(audit_path)
        for path, expected in auth['native_runtime_sha256'].items():
            require(sha(repo / path) == expected, 'native runtime differs from authorization')
        require(auth.get('device_lifetime_mode') == LIFETIME_MODE, 'fresh authorization has wrong lifetime mode')
        require(set(auth['gpu_uuids']) == set(uuids), 'authorization does not cover exactly all eight GPUs')
        require(audit.get('passed') is True and sha(args.checkpoint) == audit['full']['sha256'], 'checkpoint audit mismatch')
        checkpoint_stat = (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
        require(sha(repo / 'scripts/device_lifetime_guard.py') == auth['guard_sha256'], 'guard helper changed')
        frozen_files = {str(path): sha(path) for path in (auth_path, audit_path, timing_path, config)}
        frozen_files.update({str(repo / path): expected for path, expected in auth['native_runtime_sha256'].items()})
        expected_source = (args.expected_head, args.expected_source_sha256)
        def unchanged():
            require(source_identity(repo) == expected_source, 'frozen Git/source identity changed')
            require(all(sha(path) == value for path, value in frozen_files.items()), 'frozen input receipt/config changed')
            require((args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns) == checkpoint_stat, 'checkpoint changed during pipeline')
        unchanged()
        smoke_guard = inside_ssd(args.smoke_guard); smoke_output = inside_ssd(args.smoke_output)
        if args.wait_for_smoke:
            status.update(state='waiting_for_smoke'); persist()
            wait_deadline = time.monotonic() + 7200
            while True:
                if (smoke_guard / 'status.json').exists():
                    smoke_status = read(smoke_guard / 'status.json')
                    if smoke_status.get('ended_at'):
                        break
                require(time.monotonic() < wait_deadline, 'smoke wait deadline expired; no formal launch')
                time.sleep(10)
            unchanged()
        verify_supervisor(smoke_guard, stage='smoke', uuids=uuids, authorization=auth,
            authorization_sha=frozen_files[str(auth_path)], source_sha=args.expected_source_sha256)
        smoke_spec, smoke_summary = verify_suite(smoke_output, expected=16)
        require(smoke_spec.get('max_cohort_size') == 8 and smoke_spec.get('simulation_rate') == SIMULATION_RATE, 'smoke concurrency/rate mismatch')
        require(smoke_spec.get('probe_action_limit') is None, 'scaled smoke must cover full native episode limits')
        require(smoke_spec.get('scheduling_mode') == 'phased_episode_cohorts_v1' and smoke_spec.get('opencv_threads') == 1, 'smoke scheduling/thread contract differs')
        require(len(smoke_spec['tasks']) == 8 and smoke_spec['episodes_per_task'] == 2, 'smoke must cover eight tasks and two episodes each')
        require(set(smoke_spec['gpus']) == set(uuids) and len(smoke_spec['gpus']) == 8, 'smoke did not use all eight GPUs')
        smoke_exits = read(smoke_output / 'suite_completed.json')['finished']
        require({row.get('gpu') for row in smoke_exits} == set(uuids), 'actual smoke worker exits do not cover eight GPUs')
        require(smoke_spec['checkpoint_sha256'] == audit['full']['sha256']
                and smoke_spec['timing_contract_sha256'] == frozen_files[str(timing_path)]
                and smoke_spec['config_sha256'] == frozen_files[str(config)], 'smoke configuration differs')
        for task in smoke_spec['tasks']:
            for episode in range(smoke_spec['episodes_per_task']):
                report = read(smoke_output / task / f'episode_{episode:04d}' / 'report.json')
                require(report.get('probe_action_limit') is None, 'smoke episode was truncated')
                require(report.get('camera_equivalence', {}).get('passed') is True,
                        'same-frame native RGB equivalence did not pass')
        decode_videos(smoke_output, directory / 'smoke_video_decode.json')
        timing_args = []
        for key in ('action_interval_s', 'history_frame_period_s', 'max_backlog_s', 'infer_clock_tolerance_s'):
            timing_args.extend(['--' + key.replace('_', '-'), str(smoke_spec[key])])
        # Formal uses all profile tasks and native episode limits: no --tasks,
        # --resume or --probe-action-limit is forwarded from the smoke.
        base = [str(repo / 'external/robotwin/bin/python'), str(repo / 'scripts/eval_domino_motionkv_rtc_suite.py'),
                '--config', str(config), '--gpus', args.gpus, '--episodes', '20', '--seed', str(args.seed), '--simulation-rate', str(SIMULATION_RATE), '--max-cohort-size', '8', '--task-timeout-s', '57600', '--cohort-timeout-s', '3600', *timing_args]
        prepare, formal = PREPARED, root / 'formal35x20'
        prepare_spec, _ = verify_suite(prepare, expected=700, prepare=True)
        frozen_path = prepare / 'frozen_manifest.json'
        require(sha(frozen_path) == FROZEN_SCENE_SHA, 'saved 700-scene manifest differs from accepted preparation')
        require(prepare_spec['config_sha256'] == frozen_files[str(config)] and prepare_spec['seed'] == args.seed,
                'prepared scenes have different config/seed')
        frozen_files[str(frozen_path)] = FROZEN_SCENE_SHA
        status['reused_prepare'] = dict(path=str(prepare), scenes=700, files_verified=1400,
            manifest_sha256=FROZEN_SCENE_SHA, verified_at=utc()); persist()
        commands = [('formal', root / 'formal_guard', formal,
                     [*base, '--output-dir', str(formal), '--replay-from', str(prepare),
                      '--checkpoint', str(args.checkpoint), '--checkpoint-audit', str(audit_path), '--timing-contract', str(timing_path)])]
        for _, guard, output, _ in commands:
            require(not guard.exists() and not output.exists(), 'pipeline never overwrites or resumes an existing stage')
        save(directory / 'pipeline_manifest.json', dict(pipeline_script_sha256=sha(__file__), git_head=args.expected_head,
             source_sha256=args.expected_source_sha256, inputs=frozen_files, checkpoint=str(args.checkpoint),
             device_lifetime_mode=LIFETIME_MODE, reused_prepare=status['reused_prepare'],
             checkpoint_sha256=audit['full']['sha256'], gpu_uuids=uuids, smoke_guard=str(smoke_guard),
             smoke_output=str(smoke_output), seed=args.seed, source_frozen=True,
             commands=[dict(stage=stage, guard_dir=str(guard), output_dir=str(output), command=command)
                       for stage, guard, output, command in commands]))
        for stage, guard, output, child_command in commands:
            unchanged()
            command = [sys.executable, str(repo / 'scripts/supervise_motionkv_eval.py'), '--stage', stage,
                '--run-dir', str(guard), '--gpus', args.gpus, '--guard-gpus', args.gpus,
                '--authorization', str(auth_path), '--startup-seconds', '300', '--', *child_command]
            phase = dict(stage=stage, guard_dir=str(guard), output_dir=str(output), command=command, started_at=utc())
            status.update(state=stage + '_running', active_stage=stage); status['phases'].append(phase)
            with (directory / (stage + '_supervisor.log')).open('x') as log:
                active = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase['supervisor_pid'] = active.pid; persist()
                while active.poll() is None:
                    phase['elapsed_seconds'] = time.time() - datetime.fromisoformat(phase['started_at']).timestamp()
                    persist(); time.sleep(10)
                phase['returncode'] = active.returncode
                verify_supervisor(guard, stage=stage, uuids=uuids, authorization=auth,
                    authorization_sha=frozen_files[str(auth_path)], source_sha=args.expected_source_sha256, returncode=active.returncode)
            active = None
            spec, summary = verify_suite(output, expected=700, tasks=prepare_spec['tasks'], full_formal=True)
            require(spec['source_digest'] == smoke_spec['source_digest'], 'stage source differs from actual smoke')
            require(spec.get('scheduling_mode') == smoke_spec['scheduling_mode'] and spec.get('opencv_threads') == 1, 'formal scheduling/thread contract differs')
            require(len(spec['tasks']) == 35 and spec['episodes_per_task'] == 20, 'stage is not exactly 35x20')
            require(spec['frozen_manifest_sha256'] == FROZEN_SCENE_SHA, 'formal used a different scene manifest')
            require(spec['config_sha256'] == frozen_files[str(config)]
                    and spec['checkpoint_sha256'] == audit['full']['sha256']
                    and spec['timing_contract_sha256'] == frozen_files[str(timing_path)], 'formal config/checkpoint/timing differs')
            status.update(state='verifying_all_videos'); persist()
            decode_videos(output, directory / 'formal_video_decode.json')
            unchanged(); phase.update(verified=True, valid_completed=700, ended_at=utc()); persist()
        status.update(state='complete', valid_completed=700, ended_at=utc(), exit_code=0); persist(); return 0
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}', exit_code=1)
        if active is not None and active.poll() is None:
            # This exact Popen-owned supervisor performs its own child cleanup
            # and final FD closure. Never bypass it with killpg/driver recovery.
            active.terminate()
            try: active.wait(timeout=120)
            except subprocess.TimeoutExpired: status['supervisor_cleanup_pending_pid'] = active.pid
        status['ended_at'] = utc(); persist(); return 1


def self_test():
    """CPU-only negative gate tests; no subprocess/GPU access."""
    import copy
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); uuids = [f'GPU-test-{i}' for i in range(8)]
        driver = dict(zip(DRIVER_KEYS, ('host', 'boot', 3, 'all', 'last')))
        auth = dict(driver=driver, guard_sha256='guard', device_lifetime_mode=LIFETIME_MODE,
                    created_at='2026-10-14T12:00:00+00:00')
        events = [dict(event=kind, uuid=uuid, fd=i, path=f'/dev/nvidia{i}', monotonic=float(when))
                  for kind, when in (('after_open', 1), ('after_close', 2)) for i, uuid in enumerate(uuids)]
        save(root/'device_guard_events.json', events)
        status = dict(final_exit_code=0, exit_code=0, ended_at='now', driver_exit_clean=True,
                      started_at='2026-10-14T12:01:00+00:00', worker_started=True,
                      guard_all_handles_closed=True, guard_workers_exited=True,
                      device_guard_events_sha256=sha(root/'device_guard_events.json'))
        manifest = dict(stage='smoke', source_sha256='source', worker_gpu_uuids=uuids,
            device_lifetime_mode=LIFETIME_MODE, preflight_only=False,
            guard_gpu_uuids=uuids, authorization_sha256='auth', device_guard_sha256='guard', driver=driver)
        save(root/'manifest.json', manifest); save(root/'driver_postexit.json', driver); save(root/'status.json', status)
        options = dict(stage='smoke', uuids=uuids, authorization=auth, authorization_sha='auth', source_sha='source')
        verify_supervisor(root, **options); checks = 1
        for key, value in (('final_exit_code', 1), ('exit_code', 1), ('driver_exit_clean', False),
                           ('guard_all_handles_closed', False), ('guard_workers_exited', False), ('stop_reason', 'fault')):
            bad = dict(status, **{key: value}); save(root/'status.json', bad)
            try: verify_supervisor(root, **options)
            except RuntimeError: checks += 1
            else: raise AssertionError(f'accepted false guard gate: {key}')
        save(root/'status.json', status)
        for key, value in (('device_lifetime_mode', 'legacy'), ('preflight_only', True)):
            save(root/'manifest.json', dict(manifest, **{key: value}))
            try: verify_supervisor(root, **options)
            except RuntimeError: checks += 1
            else: raise AssertionError(f'accepted stale/telemetry-only smoke: {key}')
        save(root/'manifest.json', manifest)
        save(root/'status.json', dict(status, started_at='2026-10-14T11:59:00+00:00'))
        try: verify_supervisor(root, **options)
        except RuntimeError: checks += 1
        else: raise AssertionError('accepted smoke before fresh authorization')
        save(root/'status.json', status)
        changed = copy.deepcopy(driver); changed['anomaly_count'] += 1; save(root/'driver_postexit.json', changed)
        try: verify_supervisor(root, **options)
        except RuntimeError: checks += 1
        else: raise AssertionError('accepted changed driver')
        scenes = root / 'scenes'; tasks = [f'task{i:02d}' for i in range(35)]; frozen = []
        for task in tasks:
            for episode in range(20):
                where = scenes / task / f'episode_{episode:04d}'
                save(where / 'manifest.json', dict(task=task, episode_index=episode))
                (where / 'expert_motion.pkl').write_bytes(b'CPU test fixture')
                frozen.append(dict(task=task, episode_index=episode,
                    manifest_sha256=sha(where/'manifest.json'), motion_sha256=sha(where/'expert_motion.pkl')))
        save(scenes/'suite_started.json', dict(tasks=tasks, prepare_only=True))
        summary = dict(complete=True, valid_completed=700, planned_total=700,
            tasks=[dict(task=task, valid_completed=20, errors=[], completion_marker_valid=True) for task in tasks])
        save(scenes/'summary.json', summary)
        save(scenes/'suite_completed.json', dict(complete=True, valid_completed=700, planned_total=700,
            all_processes_exited=True, finished=[dict(task=task, exit_code=0) for task in tasks]))
        frozen_manifest = dict(complete=True, episodes=frozen)
        save(scenes/'frozen_manifest.json', frozen_manifest)
        verify_suite(scenes, expected=700, prepare=True); checks += 1
        for broken in (dict(frozen_manifest, complete=False),
                       dict(frozen_manifest, episodes=frozen[:-1]),
                       dict(frozen_manifest, episodes=frozen[:-1]+[frozen[0]])):
            save(scenes/'frozen_manifest.json', broken)
            try: verify_suite(scenes, expected=700, prepare=True)
            except RuntimeError: checks += 1
            else: raise AssertionError('accepted invalid frozen 700 scenes')
        save(scenes/'frozen_manifest.json', frozen_manifest)
        (scenes/tasks[0]/'episode_0000'/'expert_motion.pkl').write_bytes(b'corrupted')
        try: verify_suite(scenes, expected=700, prepare=True)
        except RuntimeError: checks += 1
        else: raise AssertionError('accepted changed frozen scene bytes')
        print(dict(passed=True, checks=checks, scope='CPU pipeline guard/exit/driver negative gates'))
    return 0


if __name__ == '__main__':
    arguments = parse()
    raise SystemExit(self_test() if arguments.self_test else main(arguments))
