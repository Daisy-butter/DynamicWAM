#!/usr/bin/env python3
"""One UUID-pinned Full baseline task; immutable scenes, no retry or recovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
MODE = 'full_baseline_suite'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                   default=lambda x: x.tolist()) + '\n', encoding='utf-8')
    temporary.replace(path)


def proc_identity(pid, proc_root=Path('/proc')):
    fields = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), session=int(fields[3]), start_ticks=int(fields[19]))


def validate_config(config):
    if config.get('run_kind') != MODE or config.get('stage') not in ('smoke', 'formal'):
        raise ValueError('explicit Full baseline smoke/formal configuration required')
    tasks, uuids = config['tasks'], config['gpu_uuids']
    if (not tasks or len(tasks) != len(set(tasks)) or
            any(not re.fullmatch(r'[a-z][a-z0-9_]*', task) for task in tasks)):
        raise ValueError('distinct safe task names required')
    if len(uuids) != 8 or len(set(uuids)) != 8 or any(not u.startswith('GPU-') for u in uuids):
        raise ValueError('eight distinct UUID-pinned workers required')
    if type(config['episodes']) is not int or not 1 <= config['episodes'] <= 20:
        raise ValueError('episode count must be 1..20')
    if config['stage'] == 'formal' and (len(tasks) != 35 or config['episodes'] != 20 or config.get('action_limit') is not None):
        raise ValueError('formal evaluation requires 35 x 20 complete native episodes')
    if config.get('instruction_mode', 'episode_complete_template') != 'episode_complete_template':
        raise ValueError('paired evaluation uses the common complete episode template')
    for key in ('action_dt', 'frame_period', 'max_backlog', 'max_observation_age'):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError('invalid positive timing field: ' + key)
    # A comparison protocol cannot silently change the accepted V4 timing grid.
    if not math.isclose(float(config['action_dt']), 0.18000000854954124, abs_tol=1e-8):
        raise ValueError('common action grid must be 45 float32 physics steps')
    for name in ('profile', 'checkpoint'):
        if sha(config[name]) != config[name + '_sha256']:
            raise ValueError(name + ' hash mismatch')
    if config.get('init_checkpoint_sha256') != config['checkpoint_sha256']:
        raise ValueError('guard checkpoint identity differs')
    storage = Path(config['storage_root']).resolve(strict=True)
    output = Path(config['output']).resolve()
    if storage.name != 'wuruihan' or not output.is_relative_to(storage):
        raise ValueError('outputs must be inside the verified wuruihan artifact disk')
    return output


def validate_authorization(config, path, *, task=None, gpu_index=None):
    """CPU-only lease check. Later cohorts inherit the suite's timely claim."""
    path = Path(path)
    receipt = read(path)
    if receipt.get('mode') != MODE or receipt.get('authorized') is not True:
        raise ValueError('authorization is not for a Full baseline suite')
    expected = dict(source_sha256=source_hash(ROOT / 'src/dynamicwam'),
                    config_sha256=canonical_hash(config), init_checkpoint_sha256=config['checkpoint_sha256'])
    if any(receipt.get('identity', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('authorization source/config/checkpoint identity differs')
    if receipt.get('source', {}).get('scripts_sha256') != source_hash(ROOT / 'scripts'):
        raise ValueError('suite/worker script source changed')
    uuids = config['gpu_uuids']
    if receipt.get('gpu_uuids') != uuids or receipt.get('cuda_visible_devices') != ','.join(uuids):
        raise ValueError('parent eight-GPU allocation differs')
    visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
    if task is None:
        if visibility != ','.join(uuids) or float(receipt.get('expires_at_unix', 0)) <= time.time():
            raise ValueError('suite authorization expired or visibility differs')
    else:
        if task not in config['tasks'] or not 0 <= gpu_index < 8 or visibility != uuids[gpu_index]:
            raise ValueError('worker task or UUID is outside parent allocation')
        claim = read(path.with_suffix('.json.started'))
        if (claim.get('receipt_sha256') != sha(path) or claim.get('config_sha256') != expected['config_sha256']
                or claim.get('started_unix', math.inf) >= float(receipt['expires_at_unix'])):
            raise ValueError('worker does not inherit a timely suite claim')
        suite = proc_identity(claim['pid'])
        if suite['start_ticks'] != claim['start_ticks'] or suite['state'] in ('Z', 'X'):
            raise ValueError('owning suite exited or PID reused')
        if os.getppid() != claim['pid']:
            raise ValueError('worker must be a direct child of its suite')
    guard = receipt['external_guard']
    owner = proc_identity(guard['pid'])
    if (guard.get('mode') != 'fds_before_telemetry_v1' or owner['state'] in ('Z', 'X')
            or owner['start_ticks'] != int(guard['start_ticks'])):
        raise ValueError('all-adapter guard owner absent or PID reused')
    held = {str(fd.resolve()) for fd in (Path('/proc') / str(owner['pid']) / 'fd').iterdir()}
    required = set(guard['device_paths'])
    adapters = {str(p) for p in Path('/dev').glob('nvidia*') if p.name[6:].isdigit()}
    if not adapters or not adapters <= required or not required <= held:
        raise ValueError('guard no longer holds all adapter file descriptors')
    ancestor = os.getppid()
    while ancestor > 1 and ancestor != owner['pid']:
        ancestor = proc_identity(ancestor)['ppid']
    if ancestor != owner['pid']:
        raise ValueError('all-adapter guard is not an ancestor supervisor')
    return receipt


def load_frozen_task(config, task):
    root = Path(config['prepared']).resolve(strict=True)
    if sha(root / 'frozen_manifest.json') != config['prepared_sha256']:
        raise ValueError('frozen scene manifest hash mismatch')
    frozen, suite, completed = (read(root / f) for f in ('frozen_manifest.json', 'suite_started.json', 'suite_completed.json'))
    if (frozen.get('complete') is not True or completed.get('complete') is not True
            or task not in suite['tasks'] or suite['seed'] != config['seed'] or suite['episodes_per_task'] < 20):
        raise ValueError('prepared suite is not a complete compatible 20-episode source')
    entries = [r for r in frozen['episodes'] if r['task'] == task]
    if len(entries) != 20 or {r['episode_index'] for r in entries} != set(range(20)):
        raise ValueError('frozen task must contain exactly 20 unique episode references')
    entries.sort(key=lambda r: r['episode_index'])
    references, requested = [], config['seed']
    for index, entry in enumerate(entries):
        directory = root / task / f'episode_{index:04d}'
        if sha(directory / 'manifest.json') != entry['manifest_sha256'] or sha(directory / 'expert_motion.pkl') != entry['motion_sha256']:
            raise ValueError('frozen scene/motion bytes changed')
        descriptor = read(directory / 'manifest.json')
        if (descriptor.get('task') != task or descriptor.get('episode_index') != index
                or descriptor.get('requested_seed') != requested or type(descriptor.get('seed')) is not int
                or descriptor['seed'] < requested or entry.get('seed') != descriptor['seed']):
            raise ValueError('frozen scene task/index/accepted seed chain mismatch')
        motion_bytes = (directory / 'expert_motion.pkl').read_bytes()
        motion = pickle.loads(motion_bytes)  # Only hash-bound user-owned local experiment data.
        if motion['seed'] != descriptor['seed'] or 'info' not in (motion.get('episode_info') or {}):
            raise ValueError('frozen motion seed/episode language bindings missing')
        references.append((descriptor, motion, motion_bytes))
        requested = descriptor['seed'] + 1
    return references[:config['episodes']]


class EpisodeCohort:
    """Only inactive-clock setup and encoding barriers; fixed cohort membership."""
    def __init__(self, directory, task):
        self.directory, self.task = Path(directory), task
        self.manifest = read(self.directory / 'manifest.json')
        if task not in self.manifest['tasks']:
            raise ValueError('task is outside cohort')

    def abort(self, error):
        try:
            with (self.directory / 'abort.json').open('x', encoding='utf-8') as handle:
                json.dump(dict(task=self.task, error=str(error), time_unix=time.time()), handle)
        except FileExistsError:
            pass

    def wait(self, episode, phase):
        if phase not in ('ready', 'finished', 'finalized') or not 0 <= episode < self.manifest['episodes']:
            raise ValueError('invalid inactive episode barrier')
        directory = self.directory / f'episode_{episode:04d}' / phase
        save(directory / f'{self.task}.json', dict(task=self.task, episode=episode, phase=phase, pid=os.getpid()))
        deadline = time.monotonic() + self.manifest['timeout_seconds']
        while True:
            if (self.directory / 'abort.json').exists():
                raise RuntimeError('peer failed; cohort aborted')
            if all((directory / f'{task}.json').is_file() for task in self.manifest['tasks']):
                return
            if time.monotonic() >= deadline:
                self.abort('cohort barrier timeout')
                raise TimeoutError('cohort barrier timeout')
            time.sleep(.02)


def publish_runtime_failure(output, cohort, task, episode, exc):
    """Notify CPU supervisor before a failed runtime joins its inference thread.

    A timed-out CUDA call cannot be cancelled safely by freeing Python objects.
    Keep the thread/resources owned; the parent suite sees this atomic marker
    and terminates the cohort, and its GPU guard reaps that process session.
    """
    failure = dict(task=task, episode_index=episode, pid=os.getpid(), complete=False,
        error=f'{type(exc).__name__}: {exc}', time_unix=time.time(),
        runtime_report=getattr(exc, 'dualrate_report', None))
    save(Path(output) / task / 'runtime_failure.json', failure)
    cohort.abort(exc)


def verify_artifacts(config):
    domino = Path(config['domino_root']).resolve(strict=True)
    files = config['deployment_artifact_sha256']
    camera = list((ROOT / 'src/dynamicwam/runtime/domino_rt').glob('_sapien_color_reader*.so'))
    required = [domino / 'envs/_base_task.py', domino / 'script/eval_metrics.py',
                domino / 'task_config/_eval_step_limit.yml', domino / 'task_config' / (config['task_config'] + '.yml'),
                *[domino / 'envs' / (task + '.py') for task in config['tasks']]]
    if len(camera) != 1 or not all(str(p.resolve()) in files for p in [*camera, *required]):
        raise ValueError('native bridge, task/environment/scoring hashes must be bound')
    for name, digest in files.items():
        if not Path(name).is_absolute() or sha(name) != digest:
            raise ValueError('deployment artifact changed: ' + name)


def policy_instruction(prompt, scene_prefix):
    if not scene_prefix or not prompt.startswith(scene_prefix):
        raise ValueError('native text scene prefix differs from common V4 prompt')
    return prompt[len(scene_prefix):]


def checked_episode(config, task, episode):
    from video_artifacts import validate_video_artifacts
    base = Path(config['output']) / task / f'episode_{episode:04d}'
    report, metric, descriptor = (read(base / f) for f in ('report.json', 'metrics.json', 'manifest.json'))
    if (report.get('protocol') != config['protocol'] or report.get('realtime_valid') is not True
            or report.get('continuous_valid') is not True or report.get('complete') is not True
            or report.get('video_required') is not True or not validate_video_artifacts(report)):
        raise ValueError('episode clock/native MP4/terminal verification failed')
    for key, expected in (('task', task), ('episode_index', episode), ('checkpoint_sha256', config['checkpoint_sha256']),
                          ('prepared_sha256', config['prepared_sha256']), ('config_sha256', canonical_hash(config))):
        if report.get(key) != expected:
            raise ValueError('episode identity mismatch: ' + key)
    if report['seed'] != descriptor['seed'] or metric['seed'] != descriptor['seed']:
        raise ValueError('episode/metric/scene seed mismatch')
    if bool(metric['success']) != bool(report['success']):
        raise ValueError('native metric success differs from rollout')
    return report, metric


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'task', 'authorization', 'cohort-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    config = read(args.config)
    output = validate_config(config)
    receipt = validate_authorization(config, args.authorization, task=args.task, gpu_index=args.gpu_index)
    references = load_frozen_task(config, args.task)
    verify_artifacts(config)
    claim = Path(args.authorization).with_name(Path(args.authorization).name + '.' + args.task + '.consumed')
    with claim.open('x', encoding='utf-8') as handle:
        json.dump(dict(**proc_identity(os.getpid()), receipt_sha256=sha(args.authorization), task=args.task), handle)
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(dict(config, run_kind='domino_deployment_smoke'), os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay, activate_dependency_overlay
    dependencies = validate_dependency_overlay(config)
    if receipt['python_dependencies'] != dependencies:
        raise ValueError('dependency overlay differs from parent guard')
    activate_dependency_overlay(dependencies)
    import torch
    import cv2
    import numpy as np
    import yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    uuid = config['gpu_uuids'][args.gpu_index]
    pci = None
    for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID', '').strip() == uuid:
            pci = info.parent.name
    if pci is None:
        raise ValueError('UUID-to-PCI mapping missing')
    os.environ['DYNAMICWAM_GPU_PCI'] = pci
    torch.cuda.set_device(0)
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    renderer = install_gpu_binding()
    from dynamicwam.inference.model_loader import load_deploy_config
    profile = load_deploy_config(config['profile'])
    if Path(profile['checkpoint_path']).resolve() != Path(config['checkpoint']).resolve():
        raise ValueError('native profile points at a different checkpoint')
    from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy
    policy = DynamicWAMPolicy(config['profile'], project_root=str(ROOT))
    policy.setup()
    if policy.runtime.num_inference_steps != 10 or list(policy.runtime.video_refresh_steps) != [0, 1]:
        raise ValueError('official Full policy requires native K10 video refresh [0,1]')
    from dynamicwam.dualrate.producer import episode_prompt
    from dynamicwam.dualrate.full_baseline import run_episode, PROTOCOL
    if config['protocol'] != PROTOCOL:
        raise ValueError('runtime protocol differs from configuration')
    from dynamicwam.runtime.domino_rt.domino_session import DominoLevel1Session, prepare_level1_env
    from dynamicwam.runtime.domino_rt.camera_capture import AsyncCameraCapture, validate_native_rgb_conversion
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from video_artifacts import BufferedHeadVideo, aggregate_native_metrics, write_domino_results
    if _sapien_color_reader.rgb_batch_contract != 'float32_rgba_to_u8_rgb_exact_v1':
        raise ValueError('native camera bridge contract differs')
    save(output / args.task / 'native_rgb_conversion.json', validate_native_rgb_conversion(_sapien_color_reader))
    domino = Path(config['domino_root']).resolve(strict=True)
    sys.path.insert(0, str(domino / 'script'))
    from eval_metrics import EvalMetricsTracker
    limits = yaml.safe_load((domino / 'task_config/_eval_step_limit.yml').read_text())
    task_config = yaml.safe_load((domino / 'task_config' / (config['task_config'] + '.yml')).read_text())
    if not task_config.get('use_dynamic') or task_config.get('dynamic_level') != 1:
        raise ValueError('only DOMINO dynamic Level-1 is permitted')
    cohort = EpisodeCohort(args.cohort_dir, args.task)
    native = output / 'eval_result' / args.task / 'dynamicwam_full' / config['task_config'] / Path(config['checkpoint']).stem / config['result_stamp']
    metrics = []
    for episode, (descriptor, motion, motion_bytes) in enumerate(references):
        env = camera = session = None
        video, report = BufferedHeadVideo(), None
        directory = output / args.task / f'episode_{episode:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with SapienInitLock():
                env, prepared = prepare_level1_env(domino, task_name=args.task, task_config=config['task_config'],
                    start_seed=descriptor['seed'], instruction_type=config.get('instruction_type', 'seen'),
                    replay=dict(seed=descriptor['seed'], instruction=descriptor['instruction'], saved_motion=motion['saved_motion']))
            env._rt_episode_info = motion['episode_info']
            if descriptor['arm_tag']:
                from envs.utils.action import ArmTag
                env.arm_tag = ArmTag(descriptor['arm_tag'])
            actual = dict(seed=int(prepared['seed']), instruction=prepared['instruction'],
                velocity_m_per_s=prepared['velocity'], initial_target_position=np.asarray(env._eval_target_actor.actor.get_pose().p),
                initial_state=env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
                stop_on_contact=prepared['stop_on_contact'], check_z_threshold=prepared['check_z_threshold'],
                arm_tag=str(getattr(env, 'arm_tag', '')), physics_dt_s=float(env.scene.get_timestep()), native_action_limit=int(limits[args.task]))
            for key in ('seed', 'instruction', 'arm_tag', 'stop_on_contact', 'check_z_threshold', 'native_action_limit'):
                if actual[key] != descriptor[key]:
                    raise ValueError('frozen replay field differs: ' + key)
            for key in ('initial_target_position', 'initial_state', 'velocity_m_per_s', 'physics_dt_s'):
                if not np.allclose(actual[key], descriptor[key], atol=1e-8, rtol=0):
                    raise ValueError('frozen replay numeric field differs: ' + key)
            if int(env.step_lim) != int(limits[args.task]):
                raise ValueError('native action budget changed')
            prompt, prompt_source = episode_prompt(args.task, motion['episode_info']['info'], domino / 'description')
            env.set_instruction(instruction=prompt)
            policy.reset()
            policy.set_instruction(policy_instruction(prompt, policy.runtime.scene_prefix))
            torch.manual_seed(descriptor['seed'])
            session = DominoLevel1Session(env, flow=None, observation_config={}, stop_on_contact=prepared['stop_on_contact'],
                check_z_threshold=prepared['check_z_threshold'], lift_actor=prepared['lift_actor'], start_z=prepared['start_z'],
                instruction=prompt, control_interval_s=config['action_dt'])
            session.metrics_tracker = EvalMetricsTracker(env, {'use_dynamic': True})
            session.metrics_tracker.on_episode_start()
            camera = AsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch)
            save(directory / 'manifest.json', dict(descriptor, evaluation_instruction=prompt, prompt_source=prompt_source,
                action_dt=config['action_dt'], prepared_sha256=config['prepared_sha256']))
            (directory / 'expert_motion.pkl').write_bytes(motion_bytes)
            report = run_episode(session, policy, camera, action_dt=float(config['action_dt']),
                action_limit=min(int(limits[args.task]), int(config.get('action_limit') or limits[args.task])),
                frame_period=float(config['frame_period']), max_backlog=float(config['max_backlog']),
                max_observation_age=float(config['max_observation_age']), seed=descriptor['seed'], video=video,
                camera_warmup_captures=int(config.get('camera_warmup_captures', 3)),
                bootstrap_hold_steps=int(config.get('bootstrap_hold_steps', 2)),
                model_query_timeout_s=float(config.get('model_query_timeout_s', 30.)),
                on_ready=lambda: cohort.wait(episode, 'ready'),
                on_failure=lambda exc: publish_runtime_failure(output, cohort, args.task, episode, exc))
            report.update(task=args.task, episode_index=episode, seed=descriptor['seed'],
                requested_seed=descriptor['requested_seed'], checkpoint_sha256=config['checkpoint_sha256'],
                prepared_sha256=config['prepared_sha256'], config_sha256=canonical_hash(config),
                source_sha256=receipt['identity']['source_sha256'], video_required=True,
                instruction_mode='episode_complete_template', instruction=prompt,
                gpu_count=1, gpu_uuid=uuid, renderer=renderer, complete=False)
            if not report.get('realtime_valid') or not report.get('continuous_valid'):
                raise RuntimeError('runtime rejected clock/continuous simulation contract')
            cohort.wait(episode, 'finished')
            camera.close()
            camera = None
            report.update(video.finalize(native / f'episode{episode}.mp4', episode_end_physics_s=session.physics_time()))
            if report['stop_reason'] == 'out_of_bounds':
                session.metrics_tracker.record_out_of_bounds()
            metric = session.metrics_tracker.get_episode_metrics(report['success'],
                None if report['stop_reason'] in ('success', 'action_budget_exhausted') else report['stop_reason'], seed=descriptor['seed'])
            metric = asdict(metric) if is_dataclass(metric) else vars(metric)
            save(directory / 'metrics.json', metric)
            report['complete'] = True
            save(directory / 'report.json', report)
            checked_episode(config, args.task, episode)
            metrics.append(metric)
            write_domino_results(native, config['result_stamp'], config.get('instruction_type', 'seen'), aggregate_native_metrics(metrics))
        except BaseException as exc:
            cohort.abort(exc)
            failure = dict(getattr(exc, 'dualrate_report', {}) or report or {})
            failure.update(complete=False, task=args.task, episode_index=episode, error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc(), events=getattr(exc, 'dualrate_events', failure.get('events', [])))
            save(directory / 'report.json', failure)
            if session is not None and video.frames and not getattr(video, '_finalized', False):
                try:
                    save(directory / 'failure_video_receipt.json', video.finalize(directory / 'failed_episode.mp4', episode_end_physics_s=session.physics_time()))
                except BaseException as artifact_error:
                    save(directory / 'failure_video_receipt.json', dict(error=repr(artifact_error)))
            raise
        finally:
            try:
                try:
                    if camera is not None:
                        camera.close()
                finally:
                    if env is not None:
                        env.close_env()
                        env.release_episode_resources()
            except BaseException as cleanup_error:
                publish_runtime_failure(output, cohort, args.task, episode, cleanup_error)
                raise
        # The first barrier prevents encoding while a peer is still active.
        # This separate barrier prevents next-scene GPU setup before every
        # peer's MP4 validation, native metrics and environment teardown finish.
        cohort.wait(episode, 'finalized')
    verify_artifacts(config)
    save(output / args.task / 'completed.json', dict(complete=True, protocol=config['protocol'], task=args.task,
        episodes=len(metrics), checkpoint_sha256=config['checkpoint_sha256'], config_sha256=canonical_hash(config),
        gpu_uuid=uuid, native_result_dir=str(native), no_retry=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
