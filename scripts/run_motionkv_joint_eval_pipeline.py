#!/usr/bin/env python3
"""One-shot joint-v3 1x RTC: verified local graph -> 8x2 smoke -> 35x20.

This coordinator never opens CUDA/NVML or queries a GPU. Each stage uses the
existing temporary all-adapter guard supervisor, a 300s startup health window
and a final post-close driver check. No retry, resume or recurring inspection.
Use --print-plan to preview unvalidated commands; --self-test is CPU-only.
All large inputs/results must remain in the explicitly verified storage root.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from run_motionkv_scaled_eval_pipeline import (
    DRIVER_KEYS, LIFETIME_MODE, identity, read, require, save, sha,
    source_identity, utc, verify_supervisor, verify_suite,
)

PROTOCOL = 'continuous_world_actual_latency_joint_motionkv_async_v3'
GRAPH_KIND = 'joint_v3_deployment_graph_acceptance'
REQUIRED_SMOKE_TASKS = ('place_object_basket', 'place_container_plate')


def contained(path, storage):
    path = Path(path).expanduser().resolve()
    require(path.is_relative_to(Path(storage).resolve()), f'path escapes storage root: {path}')
    return path


def check_graph_receipt(receipt, *, auth, uuids, hashes, storage):
    require(receipt.get('schema_version') == 1 and receipt.get('kind') == GRAPH_KIND
            and receipt.get('passed') is True, 'missing actual joint-v3 graph acceptance')
    for key in ('host', 'boot_id'):
        require(receipt.get(key) == auth['driver'][key], 'graph evidence belongs to another host/boot')
    for key, value in hashes.items():
        require(receipt.get(key) == value, f'graph evidence has different {key}')
    tested = receipt.get('gpu_uuids_tested', [])
    require(tested and len(set(tested)) == len(tested) and set(tested) <= set(uuids),
            'graph tested GPU identities are missing or outside this launch')
    require(len(receipt.get('guard_gpu_uuids', [])) == 8
            and set(receipt['guard_gpu_uuids']) == set(uuids), 'graph must guard all visible adapters')
    parity = receipt.get('graph_parity', {})
    error = parity.get('max_abs_error')
    require(parity.get('passed') is True and type(parity.get('cases')) is int
            and parity['cases'] > 0 and isinstance(error, (int, float))
            and not isinstance(error, bool) and math.isfinite(error) and error >= 0,
            'graph numerical parity did not pass')
    concurrency = receipt.get('background_concurrency', {})
    require(concurrency.get('passed') is True
            and concurrency.get('fast_consumed_while_slow_running') is True,
            'actual simultaneous slow/fast execution was not accepted')
    references = receipt.get('evidence_files', [])
    require(references, 'graph acceptance must reference actual GPU test artifacts')
    for reference in references:
        path = contained(reference['path'], storage)
        require(path.is_file() and sha(path) == reference['sha256'], 'graph underlying evidence changed')
    return tested


def verify_support_guard(directory, *, preflight, tested, uuids, auth, auth_sha, source_sha):
    """Graph may test one GPU; its lifecycle still guards all eight adapters."""
    directory = Path(directory)
    status, manifest = read(directory/'status.json'), read(directory/'manifest.json')
    require(status.get('final_exit_code') == status.get('exit_code') == 0
            and status.get('ended_at') and not status.get('stop_reason'), 'support supervisor did not exit cleanly')
    for key in ('driver_exit_clean', 'guard_all_handles_closed', 'guard_workers_exited'):
        require(status.get(key) is True, 'support supervisor missing ' + key)
    require(manifest.get('preflight_only', False) is preflight
            and status.get('worker_started') is (not preflight), 'wrong support worker/preflight evidence')
    require(manifest.get('stage') in (('graph', 'smoke') if preflight else ('graph',)), 'wrong support stage')
    require(manifest.get('source_sha256') == source_sha, 'support guard source changed')
    require(manifest.get('device_lifetime_mode') == LIFETIME_MODE
            and manifest.get('authorization_sha256') == auth_sha
            and manifest.get('device_guard_sha256') == auth['guard_sha256'], 'support authorization/guard mismatch')
    require(set(manifest.get('worker_gpu_uuids', [])) == set(tested)
            and len(manifest.get('worker_gpu_uuids', [])) == len(tested), 'support tested GPU mismatch')
    require(set(manifest.get('guard_gpu_uuids', [])) == set(uuids)
            and len(manifest.get('guard_gpu_uuids', [])) == 8, 'support guard coverage mismatch')
    require(datetime.fromisoformat(status['started_at']) >= datetime.fromisoformat(auth['created_at']),
            'support run predates current authorization')
    require(identity(manifest['driver']) == identity(auth['driver'])
            == identity(read(directory/'driver_postexit.json')), 'support driver identity changed')
    events_path = directory/'device_guard_events.json'
    require(sha(events_path) == status.get('device_guard_events_sha256'), 'support guard event hash changed')
    events = read(events_path)
    for uuid in uuids:
        opened = [e for e in events if e.get('event') == 'after_open' and e.get('uuid') == uuid]
        closed = [e for e in events if e.get('event') == 'after_close' and e.get('uuid') == uuid]
        require(len(opened) == len(closed) == 1, 'support guard lifecycle incomplete')
        require(opened[0]['fd'] == closed[0]['fd'] and opened[0]['path'] == closed[0]['path']
                and opened[0]['monotonic'] < closed[0]['monotonic'], 'support guard handle close mismatch')


def smoke_tasks(configured, selection=''):
    require(len(configured) == len(set(configured)) == 35, 'configuration is not the full 35-task benchmark')
    tasks = selection.split(',') if selection else [*REQUIRED_SMOKE_TASKS,
        *[task for task in configured if task not in REQUIRED_SMOKE_TASKS][:6]]
    require(len(tasks) == len(set(tasks)) == 8 and set(tasks) <= set(configured)
            and set(REQUIRED_SMOKE_TASKS) <= set(tasks), 'smoke needs eight tasks including POB and PCP')
    return tasks


def suite_command(args, *, stage, tasks):
    require(stage in ('smoke', 'formal'), 'unknown pipeline stage')
    repo = Path(args.repo).resolve()
    output = Path(args.run_root).resolve()/('smoke8x2' if stage == 'smoke' else 'formal35x20')
    command = [str(repo/'external/robotwin/bin/python'), str(repo/'scripts/eval_domino_motionkv_rtc_suite.py'),
        '--config', str(Path(args.config).resolve()), '--output-dir', str(output), '--gpus', args.gpus,
        '--episodes', '2' if stage == 'smoke' else '20', '--seed', str(args.seed),
        '--checkpoint', str(Path(args.checkpoint).resolve()),
        '--checkpoint-audit', str(Path(args.checkpoint_audit).resolve()),
        '--timing-contract', str(Path(args.timing_contract).resolve()),
        '--replay-from', str(Path(args.frozen_scenes).resolve()), '--joint-v3', '--simulation-rate', '1',
        '--max-cohort-size', '8', '--action-interval-s', '0.16', '--history-frame-period-s', '0.16',
        '--max-backlog-s', '0.25', '--infer-clock-tolerance-s', '0.02',
        '--task-timeout-s', str(args.task_timeout_s), '--cohort-timeout-s', str(args.cohort_timeout_s)]
    if stage == 'smoke': command += ['--tasks', ','.join(tasks)]
    if args.scene_compatibility:
        command += ['--scene-compatibility', str(Path(args.scene_compatibility).resolve())]
    return output, command


def verify_stage(output, *, expected_tasks, episodes, uuids, hashes, frozen_sha, previous_source=None):
    # Recompute per-episode validity, seeds, numerical metrics and MP4 sidecars
    # with the same native checker, not merely a prewritten complete flag.
    from eval_domino_motionkv_rtc_suite import summarize
    output = Path(output)
    spec = read(output/'suite_started.json')
    expected = len(expected_tasks)*episodes
    require(spec.get('protocol') == PROTOCOL and spec.get('simulation_rate') == 1.
            and spec.get('virtual_delay') is False and spec.get('prepare_only') is False,
            'stage is not declared actual-wall joint-v3 RTC')
    require(set(spec.get('tasks', [])) == set(expected_tasks) and len(spec['tasks']) == len(expected_tasks)
            and spec.get('episodes_per_task') == episodes and spec.get('probe_action_limit') is None,
            'stage task/episode/native-action scope changed')
    require(spec.get('max_cohort_size') == 8 and not spec.get('solo_tasks')
            and spec.get('scheduling_mode') == 'phased_episode_cohorts_v1' and spec.get('opencv_threads') == 1,
            'stage concurrency/scheduling changed')
    require(len(spec.get('gpus', [])) == 8 and set(spec['gpus']) == set(uuids), 'stage GPU identities changed')
    for key, expected_value in (('action_interval_s', .16), ('history_frame_period_s', .16),
                               ('max_backlog_s', .25), ('infer_clock_tolerance_s', .02)):
        require(spec.get(key) == expected_value, 'stage timing contract changed: ' + key)
    for key in ('checkpoint_sha256', 'config_sha256', 'timing_contract_sha256'):
        require(spec.get(key) == hashes[key], 'stage provenance changed: ' + key)
    require(spec.get('frozen_manifest_sha256') == frozen_sha, 'stage changed the frozen 700 scenes')
    require(spec.get('scene_compatibility_sha256') == hashes.get('scene_compatibility_sha256'),
            'scene compatibility evidence changed')
    if previous_source is not None:
        require(spec.get('source_digest') == previous_source, 'formal source differs from actual smoke')
    completion, original = read(output/'suite_completed.json'), read(output/'summary.json')
    summary = summarize(output, spec)
    require(summary == original, 'saved aggregate differs from independently re-read episodes')
    require(summary.get('complete') is True and completion.get('complete') is True
            and completion.get('all_processes_exited') is True, 'stage lacks clean complete terminal evidence')
    for item in (summary, completion):
        require(item.get('valid_completed') == item.get('planned_total') == expected, 'stage valid count mismatch')
    exits = completion.get('finished', [])
    require(len(exits) == len(expected_tasks) and {row.get('task') for row in exits} == set(expected_tasks)
            and all(row.get('exit_code') == 0 for row in exits), 'stage task worker exits incomplete')
    require({row.get('gpu') for row in exits} == set(uuids), 'actual worker exits do not cover all eight GPUs')
    for task in expected_tasks:
        for episode in range(episodes):
            report = read(output/task/f'episode_{episode:04d}'/'report.json')
            require(report.get('camera_equivalence', {}).get('passed') is True,
                    'native same-frame RGB equivalence missing')
            require(report.get('post_terminal_policy_drain', {}).get('slow_quiescent') is True,
                    'background producer was not quiescent at episode completion')
            require(report.get('probe_action_limit') is None and report.get('joint_v3') is True,
                    'episode is truncated or not joint-v3')
            require(report.get('scene_compatibility_sha256') == hashes.get('scene_compatibility_sha256'),
                    'episode used another scene compatibility receipt')
    return spec, summary


def decode_stage(output, destination, storage):
    """Decode every native MP4 on CPU; preserve original 10fps output semantics."""
    spec, records, seen = read(Path(output)/'suite_started.json'), [], set()
    for task in spec['tasks']:
        for episode in range(spec['episodes_per_task']):
            report = read(Path(output)/task/f'episode_{episode:04d}'/'report.json')
            path = contained(report['video'], storage)
            require(path not in seen and sha(path) == report['video_sha256'], 'duplicate or changed MP4')
            seen.add(path)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames',
                '-of', 'json', str(path)], capture_output=True, text=True, timeout=120, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            require(int(stream['nb_read_frames']) == report['video_frames'] > 0
                    and stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p', 'native MP4 format differs')
            numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
            require(denominator > 0 and numerator/denominator == 10., 'native MP4 must remain 10fps')
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-hwaccel', 'none',
                       '-threads', '1', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-']
            decoded = subprocess.run(command, capture_output=True, text=True, timeout=120)
            require(decoded.returncode == 0 and not decoded.stderr.strip(), 'full MP4 decode failed: ' + str(path))
            records.append(dict(path=str(path), sha256=report['video_sha256'], frames=report['video_frames'],
                                full_decode_exit_code=0, ffprobe=stream))
            save(destination, dict(complete=False, videos=records))
    require(len(records) == len(spec['tasks'])*spec['episodes_per_task'], 'MP4 count mismatch')
    save(destination, dict(complete=True, videos=records))


def revalidate_graph_receipt(receipt, *, guard_dir, checkpoint, config):
    """Reconstruct acceptance from the guard's actual bound graph test output."""
    from accept_joint_deployment_graph import accept
    candidates = []
    for entry in receipt['evidence_files']:
        evidence = read(entry['path'])
        if isinstance(evidence, dict) and evidence.get('kind') == 'motionkv_rtc_native_graph':
            candidates.append(entry['path'])
    require(len(candidates) == 1, 'need exactly one actual graph evidence file')
    rebuilt = accept(raw_path=candidates[0], guard_dir=guard_dir, checkpoint=checkpoint, config=config)
    without_timestamp = lambda value: {key:item for key,item in value.items() if key != 'created_at'}
    require(without_timestamp(receipt) == without_timestamp(rebuilt), 'graph summary differs from actual guarded evidence')


def close_kernel_broker(status, *, enabled, launched, socket_path, stopper=None):
    """Close only this run's broker after all owned supervisor cleanup finished."""
    if not enabled or not socket_path:
        return
    if not launched:
        status['kernel_broker_stop'] = {'stopped':False, 'reason':'no_pipeline_stage_launched'}
        return
    if status.get('supervisor_cleanup_pending_pid'):
        status['kernel_broker_stop'] = {'stopped':False, 'reason':'supervisor_cleanup_pending'}
        return
    if stopper is None:
        from eval_kernel_log import stop_broker
        stopper = stop_broker
    result = stopper(socket_path)
    status['kernel_broker_stop'] = {'stopped':True, 'socket':socket_path, 'receipt':result}


def main(args):
    repo, storage = Path(args.repo).resolve(), Path(args.storage_root).resolve()
    require(repo == Path(__file__).resolve().parents[1], 'run the pipeline from the selected frozen repo')
    require('wuruihan' in storage.parts and storage.is_dir(), 'storage must be a verified wuruihan directory')
    require(storage.stat().st_dev != Path('/').stat().st_dev, 'storage may not be the system/home filesystem')
    root = contained(args.run_root, storage)
    root.mkdir(parents=True, exist_ok=True)
    directory = root/'pipeline'; directory.mkdir(exist_ok=False)
    status = dict(state='validating', created_at=utc(), pipeline_pid=os.getpid(), phases=[],
                  retry_policy='none', recurring_gpu_polling=False, startup_health_seconds=300, protocol=PROTOCOL)
    status_path, active = directory/'pipeline_status.json', None
    launched = False
    def persist():
        status['timestamp'] = utc(); save(status_path, status)
    def interrupted(signum, frame): raise RuntimeError('pipeline received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    persist()
    try:
        import yaml
        uuids = args.gpus.split(',')
        require(len(uuids) == len(set(uuids)) == 8 and all(u.startswith('GPU-') for u in uuids), 'need eight exact GPU UUIDs')
        paths = {key: contained(getattr(args, key), storage) for key in
                 ('checkpoint', 'checkpoint_audit', 'timing_contract', 'authorization', 'graph_receipt',
                  'preflight_guard', 'graph_guard', 'frozen_scenes')}
        config = Path(args.config).resolve()
        auth, audit, graph = (read(paths[key]) for key in ('authorization', 'checkpoint_audit', 'graph_receipt'))
        require(auth['driver']['host'] == socket.gethostname()
                and auth['driver']['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'authorization does not belong to this local host/boot')
        require(auth.get('device_lifetime_mode') == LIFETIME_MODE and set(auth.get('gpu_uuids', [])) == set(uuids),
                'authorization does not cover this eight-GPU lifetime mode')
        checkpoint_sha = sha(paths['checkpoint'])
        require(audit.get('passed') is True and audit.get('strict_full_reload') is True
                and audit.get('full_fast_equal') is True and audit['full']['sha256'] == checkpoint_sha,
                'checkpoint strict full/fast audit did not pass')
        require(Path(audit['full']['path']).resolve() == paths['checkpoint']
                and audit['full']['bytes'] == paths['checkpoint'].stat().st_size, 'audit refers to another local checkpoint')
        hashes = dict(checkpoint_sha256=checkpoint_sha, config_sha256=sha(config),
                      timing_contract_sha256=sha(paths['timing_contract']), source_sha256=args.expected_source_sha256)
        tested = check_graph_receipt(graph, auth=auth, uuids=uuids, storage=storage,
            hashes={k:hashes[k] for k in ('checkpoint_sha256','config_sha256','source_sha256')})
        auth_sha = sha(paths['authorization'])
        verify_support_guard(paths['preflight_guard'], preflight=True, tested=uuids, uuids=uuids,
                             auth=auth, auth_sha=auth_sha, source_sha=args.expected_source_sha256)
        verify_support_guard(paths['graph_guard'], preflight=False, tested=tested, uuids=uuids,
                             auth=auth, auth_sha=auth_sha, source_sha=args.expected_source_sha256)
        revalidate_graph_receipt(graph, guard_dir=paths['graph_guard'], checkpoint=paths['checkpoint'], config=config)
        prepared, _ = verify_suite(paths['frozen_scenes'], expected=700, prepare=True)
        frozen_path = paths['frozen_scenes']/'frozen_manifest.json'
        require(sha(frozen_path) == args.frozen_scenes_sha256, 'frozen 700-scene manifest changed')
        tasks = yaml.safe_load(config.read_text())['benchmark']['tasks']
        selected = smoke_tasks(tasks, args.smoke_tasks)
        require(set(prepared['tasks']) == set(tasks) and prepared['seed'] == args.seed,
                'prepared 700 scenes have another task/seed set')
        frozen = read(frozen_path)
        require({(row['task'],row['episode_index']) for row in frozen['episodes']}
                == {(task,i) for task in tasks for i in range(20)}, 'frozen episode indices differ')
        frozen_files = {str(path):sha(path) for path in (*[paths[k] for k in
            ('checkpoint_audit','timing_contract','authorization','graph_receipt')], config, frozen_path,
            repo/'harness.md', repo/'scripts/device_lifetime_guard.py')}
        require(sha(repo/'scripts/device_lifetime_guard.py') == auth['guard_sha256'], 'guard helper changed')
        for name, expected in auth.get('native_runtime_sha256', {}).items():
            path = repo/name; require(sha(path) == expected, 'native runtime changed'); frozen_files[str(path)] = expected
        for entry in graph['evidence_files']:
            frozen_files[str(contained(entry['path'], storage))] = entry['sha256']
        for key in ('preflight_guard','graph_guard'):
            for name in ('status.json','manifest.json','device_guard_events.json','driver_postexit.json'):
                path=paths[key]/name; frozen_files[str(path)]=sha(path)
        if args.scene_compatibility:
            from scene_compatibility import validate_receipt
            path=contained(args.scene_compatibility,storage)
            compatibility=validate_receipt(path,repo=repo,config=config,frozen_scenes=paths['frozen_scenes'])
            frozen_files[str(path)]=sha(path)
            frozen_files[compatibility['source_config']]=compatibility['source_config_sha256']
            hashes['scene_compatibility_sha256']=sha(path)
        checkpoint_stat = (paths['checkpoint'].stat().st_size, paths['checkpoint'].stat().st_mtime_ns)
        def unchanged():
            from supervise_motionkv_eval import driver_evidence
            require(identity(driver_evidence()) == identity(auth['driver']), 'current driver identity changed')
            require(source_identity(repo) == (args.expected_head,args.expected_source_sha256), 'frozen Git/source changed')
            require(all(sha(path) == value for path,value in frozen_files.items()), 'frozen config/evidence changed')
            require((paths['checkpoint'].stat().st_size,paths['checkpoint'].stat().st_mtime_ns) == checkpoint_stat,
                    'checkpoint changed while pipeline was running')
            # Scene bytes remain independently checked before each new stage.
            verify_suite(paths['frozen_scenes'], expected=700, prepare=True)
        unchanged()
        commands=[]
        for stage in ('smoke','formal'):
            output, child = suite_command(args, stage=stage, tasks=selected)
            guard=root/(stage+'_guard')
            require(not output.exists() and not guard.exists(), 'never overwrite/resume a previous stage')
            command=[sys.executable,str(repo/'scripts/supervise_motionkv_eval.py'),'--stage',stage,
                '--run-dir',str(guard),'--gpus',args.gpus,'--guard-gpus',args.gpus,
                '--authorization',str(paths['authorization']),'--startup-seconds','300',
                '--max-runtime-seconds',str(21600 if stage=='smoke' else 172800),'--',*child]
            commands.append(dict(stage=stage,guard_dir=str(guard),output_dir=str(output),command=command))
        save(directory/'pipeline_manifest.json', dict(created_at=utc(), protocol=PROTOCOL,
            pipeline_script_sha256=sha(__file__), git_head=args.expected_head, **hashes,
            inputs=frozen_files, gpu_uuids=uuids, smoke_tasks=selected,
            storage_root=str(storage), frozen_scene_manifest_sha256=args.frozen_scenes_sha256,
            commands=commands, no_retry=True, no_resume=True))
        previous_source=None
        for plan in commands:
            unchanged(); stage=plan['stage']; guard=Path(plan['guard_dir']); output=Path(plan['output_dir'])
            phase=dict(plan, started_at=utc()); status['phases'].append(phase)
            status.update(state=stage+'_running', active_stage=stage); persist()
            with (directory/(stage+'_supervisor.log')).open('x') as log:
                active=subprocess.Popen(plan['command'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                launched=True
                phase['supervisor_pid']=active.pid; persist()
                while active.poll() is None:
                    phase['elapsed_seconds']=time.time()-datetime.fromisoformat(phase['started_at']).timestamp()
                    persist(); time.sleep(10)
                phase['returncode']=active.returncode
                verify_supervisor(guard,stage=stage,uuids=uuids,authorization=auth,authorization_sha=auth_sha,
                                  source_sha=args.expected_source_sha256,returncode=active.returncode)
            active=None
            spec,summary=verify_stage(output,expected_tasks=selected if stage=='smoke' else tasks,
                episodes=2 if stage=='smoke' else 20,uuids=uuids,hashes=hashes,
                frozen_sha=args.frozen_scenes_sha256,previous_source=previous_source)
            status.update(state='verifying_'+stage+'_videos');persist()
            decode_stage(output,directory/(stage+'_video_decode.json'),storage)
            unchanged(); previous_source=spec['source_digest']
            phase.update(verified=True,valid_completed=summary['valid_completed'],ended_at=utc());persist()
        status.update(state='complete',valid_completed=700,ended_at=utc(),exit_code=0);persist();return 0
    except BaseException as error:
        status.update(state='failed',error=f'{type(error).__name__}: {error}',exit_code=1)
        if active is not None and active.poll() is None:
            active.terminate()
            try: active.wait(timeout=120)
            except subprocess.TimeoutExpired: status['supervisor_cleanup_pending_pid']=active.pid
        status['ended_at']=utc();persist();return 1
    finally:
        try:
            close_kernel_broker(status, enabled=args.stop_kernel_broker_on_exit, launched=launched,
                                socket_path=os.environ.get('WAM_KERNEL_LOG_SOCKET'))
            persist()
        except BaseException as cleanup_error:
            status.update(state='failed', exit_code=1,
                          kernel_broker_stop_error=f'{type(cleanup_error).__name__}: {cleanup_error}')
            persist()
            return 1


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--stop-kernel-broker-on-exit',action='store_true',
        help='After owned stages fully exit, close this run WAM_KERNEL_LOG_SOCKET; no launch leaves it available')
    parser.add_argument('--print-plan',action='store_true',help='Print commands only; does not validate or launch')
    for name in ('repo','run-root','storage-root','config','checkpoint','checkpoint-audit','timing-contract',
                 'frozen-scenes','authorization','preflight-guard','graph-guard','graph-receipt','scene-compatibility'):
        parser.add_argument('--'+name,type=Path)
    for name in ('gpus','expected-head','expected-source-sha256','frozen-scenes-sha256'):
        parser.add_argument('--'+name)
    parser.add_argument('--seed',type=int,default=100000)
    parser.add_argument('--smoke-tasks',default='')
    parser.add_argument('--task-timeout-s',type=float,default=14400.)
    parser.add_argument('--cohort-timeout-s',type=float,default=3600.)
    args=parser.parse_args(argv)
    if not args.self_test:
        for name in ('repo','run_root','storage_root','config','checkpoint','checkpoint_audit','timing_contract',
                     'frozen_scenes','authorization','preflight_guard','graph_guard','graph_receipt',
                     'gpus','expected_head','expected_source_sha256','frozen_scenes_sha256'):
            if getattr(args,name) is None:parser.error('--'+name.replace('_','-')+' is required')
        for value in (args.task_timeout_s,args.cohort_timeout_s):
            if not math.isfinite(value) or value<=0:parser.error('timeouts must be finite and positive')
    return args


if __name__=='__main__':
    args=parse()
    if args.self_test:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_joint_eval_pipeline.py')
        require(suite.countTestCases() > 0, 'CPU self-test suite is missing')
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    if args.print_plan:
        selected=args.smoke_tasks.split(',') if args.smoke_tasks else ['<eight tasks including POB and PCP>']
        print(json.dumps({'validated':False,'commands':{stage:suite_command(args,stage=stage,tasks=selected)[1]
            for stage in ('smoke','formal')}},indent=2));raise SystemExit(0)
    raise SystemExit(main(args))
