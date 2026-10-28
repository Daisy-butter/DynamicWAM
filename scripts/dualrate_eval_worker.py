#!/usr/bin/env python3
"""One three-GPU V4 evaluation task; frozen scenes, no retry or recovery."""
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
MODE = 'dualrate_eval_suite'


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


def environment_groups(config):
    uuids = config['gpu_uuids']
    groups = config['environment_groups']
    if (len(uuids) != 8 or len(set(uuids)) != 8 or any(not u.startswith('GPU-') for u in uuids)
            or len(groups) != 2 or any(len(group) != 3 for group in groups)
            or [u for group in groups for u in group] != uuids[:6]):
        raise ValueError('two disjoint renderer/fast/slow triplets must use the first six of eight guarded UUIDs')
    return groups


def validate_source_compatibility(config, checkpoint_contract=None):
    path = Path(config['source_compatibility_receipt']).resolve(strict=True)
    if sha(path) != config['source_compatibility_sha256']:
        raise ValueError('training/evaluation compatibility receipt changed')
    receipt = read(path)
    current = source_hash(ROOT / 'src/dynamicwam')
    files = receipt.get('same_model_files_sha256', {})
    if receipt.get('evaluation_source_sha256') != current or not files:
        raise ValueError('compatibility receipt is not bound to this evaluation source')
    inference = receipt.get('inference_only_files_sha256')
    if not isinstance(inference, dict) or set(inference) != {'dualrate/online.py'}:
        raise ValueError('exact online inference-only compatibility binding required')
    online = inference['dualrate/online.py']
    if (not isinstance(online, dict) or set(online) != {'training_sha256', 'evaluation_sha256', 'change'}
            or online.get('change') != 'fp32_image_normalization_and_camera_numeric_tolerance'
            or any(not isinstance(online.get(k), str) or re.fullmatch(r'[0-9a-f]{64}', online[k]) is None
                   for k in ('training_sha256', 'evaluation_sha256'))
            or online['training_sha256'] == online['evaluation_sha256']):
        raise ValueError('online inference-only change identity differs')
    if sha(ROOT/'src/dynamicwam/dualrate/online.py') != online['evaluation_sha256']:
        raise ValueError('online inference-only evaluation hash differs')
    camera = receipt.get('camera_executor_change', {})
    if (camera.get('change') != 'owner_frozen_snapshot_background_render_and_readback'
            or camera.get('evaluation_sha256') != sha(ROOT/'src/dynamicwam/runtime/domino_rt/camera_capture.py')
            or re.fullmatch(r'[0-9a-f]{64}', str(camera.get('training_sha256', ''))) is None):
        raise ValueError('exact asynchronous camera executor binding required')
    protected = {p.relative_to(ROOT/'src/dynamicwam').as_posix(): p
                 for p in (ROOT/'src/dynamicwam').rglob('*.py')
                 if p.relative_to(ROOT/'src/dynamicwam').as_posix() not in
                 {'dualrate/deployment.py', 'runtime/domino_rt/domino_session.py',
                  'dualrate/contact_cache.py', 'dualrate/online.py',
                  'runtime/domino_rt/camera_capture.py'}}
    if set(files) != set(protected):
        raise ValueError('compatibility receipt must cover every non-executor Python source')
    for name, digest in files.items():
        if sha(protected[name]) != digest:
            raise ValueError('trained model/data/timing source changed: ' + name)
    if checkpoint_contract is not None and checkpoint_contract.get('source_sha256') != receipt['training_source_sha256']:
        raise ValueError('checkpoint was trained by a different source')
    return receipt


def validate_checkpoint_payload(config, payload, manifest):
    """CPU metadata gate for the exact completed 1k/10k/20k formal sequence."""
    expected = dict(schema='dualrate-distributed-checkpoint-v1', run_kind='formal_distributed_training',
                    phase='cross_time_joint', phase_step=20000, configured_steps=20000, global_step=31000)
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError('final checkpoint metadata differs: ' + name)
    for name in ('formal_training_ready', 'phase_complete', 'optimizer_boundary'):
        if payload.get(name) is not True:
            raise ValueError('final checkpoint gate missing: ' + name)
    if not isinstance(payload.get('model'), dict) or not payload['model']:
        raise ValueError('final checkpoint model state is absent')
    contract = payload.get('contract', {})
    keys = ('sequence_id', 'architecture', 'codec_id', 'init_checkpoint_sha256',
            'action_normalization_sha256', 'motion_normalization_sha256',
            'time_contract', 'action_interval_seconds')
    for name in keys:
        if name not in config or contract.get(name) != config[name]:
            raise ValueError('deployment/training checkpoint contract differs: ' + name)
    if not isinstance(config['sequence_id'], str) or not config['sequence_id'] or contract.get('world_size') != 8:
        raise ValueError('final formal sequence/world size identity missing')
    if (contract.get('manifest_sha256') != sha(config['manifest'])
            or read(config['manifest']) != manifest):
        raise ValueError('final checkpoint training manifest hash/content differs')
    if (config['time_contract'] != 'exposure_anchor_wall_release_v4'
            or manifest.get('time_contract') != config['time_contract']):
        raise ValueError('V4 hard-time contract differs')
    if (manifest.get('action_interval_seconds') != config['action_interval_seconds']
            or config['runtime']['action_dt'] != config['action_interval_seconds']
            or config['action_dt'] != config['action_interval_seconds']):
        raise ValueError('training/runtime action interval differs')
    if manifest.get('measured_runtime') != config['runtime']:
        raise ValueError('deployment runtime differs from protected measured training runtime')
    # measured_01 predates this optional manifest field. OnlineBatchBuilder
    # uses producer.CONDITION_RGB_MAX_AGE_SECONDS = 4 * .06 when it is absent.
    # Keep this pure-CPU gate's literal tied to that protected source default;
    # never rewrite the training manifest or override an explicit value.
    if (not math.isclose(float(config['max_observation_age']), .24, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(float(manifest.get('condition_rgb_max_age_seconds', .24)), .24, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('camera observation age must retain the accepted 0.24 second ceiling')
    return contract


def effective_runtime_settings(config):
    """Increase pre-lease admission reserve within the trained 2/3-row support.

    This is not extra inference time after a lease: both original deadlines
    remain strict. Model inputs use the actual selected prefix and timestamps.
    """
    trained = dict(config['runtime'])
    reserve = float(config.get('deployment_min_query_budget_seconds', trained['min_query_budget_seconds']))
    if (not math.isfinite(reserve) or reserve < trained['min_query_budget_seconds']
            or reserve > .40 or trained['protect_steps'] != 2 or trained['max_protect_steps'] != 3):
        raise ValueError('admission reserve must retain trained two/three-row prefix support')
    return dict(trained, min_query_budget_seconds=reserve)


def validate_config(config):
    if config.get('run_kind') != MODE or config.get('stage') not in ('smoke', 'formal'):
        raise ValueError('explicit DualRate V4 smoke/formal configuration required')
    tasks, uuids = config['tasks'], config['gpu_uuids']
    if (not tasks or len(tasks) != len(set(tasks)) or
            any(not re.fullmatch(r'[a-z][a-z0-9_]*', task) for task in tasks)):
        raise ValueError('distinct safe task names required')
    environment_groups(config)
    effective_runtime_settings(config)
    switch_interval = float(config.get('python_thread_switch_interval_seconds', .005))
    if not math.isfinite(switch_interval) or not .0001 <= switch_interval <= .005:
        raise ValueError('invalid finite per-worker Python thread switching interval')
    if (config.get('executor_contract') != 'immutable_wall_continuous_v1'
            or config.get('camera_async_submit') is not True
            or config.get('deployment_language_cache') != 'exact_per_device_v1'
            or config.get('slow_model_process') != 'same_gpu_strict_parity_v1'
            or float(config.get('wall_scheduling_tolerance', -1)) != 0.
            or float(config.get('physical_scheduling_tolerance', -1)) != 0.
            or float(config.get('backlog_scheduling_tolerance', -1)) != .005):
        raise ValueError('continuous executor requires async rendering, strict deadlines and Full09 backlog tolerance')
    if type(config['episodes']) is not int or not 1 <= config['episodes'] <= 20:
        raise ValueError('episode count must be 1..20')
    if config['stage'] == 'formal' and (len(tasks) != 35 or config['episodes'] != 20 or config.get('action_limit') is not None):
        raise ValueError('formal evaluation requires 35 x 20 complete native episodes')
    if config['stage'] == 'smoke' and (len(tasks) != 2 or config['episodes'] != 2 or config.get('action_limit') is not None):
        raise ValueError('smoke requires two environments x two complete native episodes')
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
    if Path(config['profile']).resolve() != Path(config['native_profile']).resolve():
        raise ValueError('native initialization profile alias differs')
    if sha(config['init_checkpoint']) != config['init_checkpoint_sha256']:
        raise ValueError('official Full initialization checkpoint changed')
    validate_source_compatibility(config)
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
        raise ValueError('authorization is not for a DualRate V4 suite')
    expected = dict(source_sha256=source_hash(ROOT / 'src/dynamicwam'),
                    config_sha256=canonical_hash(config), init_checkpoint_sha256=config['init_checkpoint_sha256'])
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
        if (task not in config['tasks'] or not 0 <= gpu_index < 2
                or visibility != ','.join(environment_groups(config)[gpu_index])):
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
        if phase not in ('ready', 'armed', 'finished', 'finalized') or not 0 <= episode < self.manifest['episodes']:
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
        runtime_report=getattr(exc, 'dualrate_report', None),
        events=getattr(exc, 'dualrate_events', []))
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


def episode_language(config, manifest, prompt, device, directory):
    """Reuse exact cached T5 text or encode that same prompt before activation."""
    from dynamicwam.dualrate.online import load_cached_language
    manifest_path = Path(config['manifest']).resolve(strict=True)
    try:
        return load_cached_language(manifest_path.parent, prompt, index_path=manifest.get('text_index', 'text/index.json'))
    except KeyError:
        import torch
        from dynamicwam.inference.model_loader import _build_t5_encoder, load_deploy_config
        from prepare_dualrate_text import pack_contexts, native_text_inputs, tokenizer_identity
        native = load_deploy_config(config['native_profile'])
        _, encoder_path, tokenizer_path = native_text_inputs(native, {'query': prompt})
        index = Path(manifest.get('text_index', 'text/index.json'))
        if not index.is_absolute():
            index = manifest_path.parent / index
        identity = read(index)['identity']
        if sha(encoder_path) != identity['source_encoder_sha256'] or tokenizer_identity(tokenizer_path) != identity['tokenizer_sha256']:
            raise ValueError('online T5 encoder/tokenizer differ from the training text bank')
        encoder = _build_t5_encoder(native, str(device))
        language, token_hash = pack_contexts(encoder, [prompt], device)[0]
        del encoder
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(language, directory/'context.pt')
        save(directory/'identity.json', dict(text=prompt, text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            token_sha256=token_hash, context_sha256=sha(directory/'context.pt'),
            encoder_sha256=identity['source_encoder_sha256'], tokenizer_sha256=identity['tokenizer_sha256']))
        return language


def checked_episode(config, task, episode):
    from video_artifacts import validate_video_artifacts
    from dualrate_continuous_acceptance import validate_continuous_report
    base = Path(config['output']) / task / f'episode_{episode:04d}'
    report, metric, descriptor = (read(base / f) for f in ('report.json', 'metrics.json', 'manifest.json'))
    validate_continuous_report(report, config)
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
    graph = report.get('fast_graph_acceptance', {})
    if (report.get('executable_action_source') != 'fast_only' or not graph.get('enabled')
            or not graph.get('passed') or not graph.get('sealed') or graph.get('shape_count') != 2):
        raise ValueError('fast-only two-shape CUDA Graph contract not accepted')
    return report, metric


def _run_worker(resources):
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'task', 'authorization', 'cohort-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    config = read(args.config)
    output = validate_config(config)
    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(float(config.get('python_thread_switch_interval_seconds', .005)))
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
    uuids = environment_groups(config)[args.gpu_index]
    uuid = uuids[0]
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
    from dynamicwam.dualrate.factory import build_system, place_inference_devices
    from dynamicwam.dualrate.train import _validate_normalization
    manifest = read(config['manifest'])
    _validate_normalization(config, manifest)
    if manifest.get('time_contract') != 'exposure_anchor_wall_release_v4':
        raise ValueError('V4 evaluation requires measured hard-time training data')
    payload = torch.load(config['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    contract = validate_checkpoint_payload(config, payload, manifest)
    compatibility = validate_source_compatibility(config, contract)
    fast_device, slow_device = torch.device('cuda', 1), torch.device('cuda', 2)
    torch.cuda.set_device(slow_device)
    system = build_system(config, device=slow_device).eval().requires_grad_(False)
    system.load_state_dict(payload['model'], strict=True)
    placement = place_inference_devices(system, fast_device=fast_device, slow_device=slow_device)
    checkpoint_meta = {k: payload.get(k) for k in ('schema', 'phase', 'phase_step', 'global_step', 'run_kind')}
    del payload
    from dualrate_slow_process import SlowProcessSystem
    system = SlowProcessSystem(system, config, slow_device=str(slow_device),
        receipt_path=output/args.task/'slow_process.json')
    resources['slow_process'] = system
    torch.cuda.set_device(0)
    from dynamicwam.dualrate.producer import episode_prompt
    from dualrate_deployment_inputs import build_cached_online_batch_builder
    from dualrate_physics_optimization import passive_force_scope
    from dynamicwam.dualrate.runtime import RuntimeConfig
    from dynamicwam.dualrate.deployment import run_episode, PROTOCOL
    runtime_config = RuntimeConfig(**effective_runtime_settings(config))
    if abs(runtime_config.action_dt - config['action_dt']) > 1e-10 or not runtime_config.fast_cuda_graph:
        raise ValueError('runtime grid/CUDA Graph differs from accepted V4 configuration')
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
    native = output / 'eval_result' / args.task / 'dualrate_dynamicwam' / config['task_config'] / Path(config['checkpoint']).stem / config['result_stamp']
    save(output/args.task/'deployment_identity.json', dict(checkpoint=checkpoint_meta, checkpoint_sha256=config['checkpoint_sha256'],
        training_source_sha256=compatibility['training_source_sha256'], source_compatibility_sha256=config['source_compatibility_sha256'],
        inference_device_map=placement, renderer=renderer, gpu_uuids=uuids, gpu_count=3,
        trained_runtime=config['runtime'], effective_runtime=effective_runtime_settings(config),
        python_thread_switch_interval_seconds=sys.getswitchinterval(),
        original_python_thread_switch_interval_seconds=previous_switch_interval,
        admission_policy='prequery_reserve_within_trained_prefix_support_strict_deadlines'))
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
            language = episode_language(config, manifest, prompt, slow_device,
                output/'episode_language'/args.task/f'episode_{episode:04d}')
            builder = build_cached_online_batch_builder(manifest, action_mean=manifest['action_statistics']['mean'],
                action_scale=manifest['action_statistics']['std'], language=language, device=fast_device, slow_device=slow_device)
            save(directory/'language_cache_receipt.json', builder.deployment_language_receipt)
            if len(builder.entity_roles) != system.fast.config.future_entities:
                raise ValueError('online entity roles differ from trained fast head')
            torch.manual_seed(descriptor['seed'])
            session = DominoLevel1Session(env, flow=None, observation_config={}, stop_on_contact=prepared['stop_on_contact'],
                check_z_threshold=prepared['check_z_threshold'], lift_actor=prepared['lift_actor'], start_z=prepared['start_z'],
                instruction=prompt, control_interval_s=config['action_dt'])
            session.metrics_tracker = EvalMetricsTracker(env, {'use_dynamic': True})
            session.metrics_tracker.on_episode_start()
            camera = AsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch,
                async_submit=config['camera_async_submit'], worker_initializer=lambda: torch.cuda.set_device(0))
            save(directory / 'manifest.json', dict(descriptor, evaluation_instruction=prompt, prompt_source=prompt_source,
                action_dt=config['action_dt'], prepared_sha256=config['prepared_sha256']))
            (directory / 'expert_motion.pkl').write_bytes(motion_bytes)
            report = run_episode(session, system, builder, camera, runtime_config,
                action_limit=min(int(limits[args.task]), int(config.get('action_limit') or limits[args.task])),
                frame_period=float(config['frame_period']), max_backlog=float(config['max_backlog']),
                seed=descriptor['seed'], video=video,
                camera_warmup_captures=int(config.get('camera_warmup_captures', 3)),
                wall_scheduling_tolerance=float(config['wall_scheduling_tolerance']),
                physical_scheduling_tolerance=float(config['physical_scheduling_tolerance']),
                backlog_scheduling_tolerance=float(config['backlog_scheduling_tolerance']),
                on_ready=lambda: cohort.wait(episode, 'ready'),
                on_armed=lambda: cohort.wait(episode, 'armed'),
                control_scope=lambda: passive_force_scope(env.robot, enabled=config.get('passive_force_dedup', False)),
                on_failure=lambda exc: publish_runtime_failure(output, cohort, args.task, episode, exc))
            report.update(task=args.task, episode_index=episode, seed=descriptor['seed'],
                requested_seed=descriptor['requested_seed'], checkpoint_sha256=config['checkpoint_sha256'],
                prepared_sha256=config['prepared_sha256'], config_sha256=canonical_hash(config),
                source_sha256=receipt['identity']['source_sha256'], video_required=True,
                instruction_mode='episode_complete_template', instruction=prompt,
                gpu_count=3, gpu_uuids=uuids, inference_device_map=placement,
                renderer=renderer, complete=False)
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
        gpu_uuids=uuids, native_result_dir=str(native), no_retry=True))
    return 0


def main():
    resources = {}
    try:
        return _run_worker(resources)
    finally:
        if 'slow_process' in resources:
            resources['slow_process'].close()


if __name__ == '__main__':
    raise SystemExit(main())
