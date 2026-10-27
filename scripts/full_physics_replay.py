#!/usr/bin/env python3
"""Bounded, unscored same-control PhysX comparison; never benchmark acceptance.

The CPU controller starts fresh sequential 0/2/4-worker processes under the
existing all-adapter guard. The children reconstruct the frozen failed scene,
replay its recorded endpoints, and save actual state after every real tick.
No model, render submissions, wall pacing, scoring, or automatic retry occurs.
"""
from __future__ import annotations

import argparse
import faulthandler
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full_baseline_worker import (canonical_hash, load_frozen_task, proc_identity,
                                 read, save, sha, source_hash, verify_artifacts)

MODE = 'full_physics_replay'
SCHEMA = 'full_physics_replay_comparison_v1'


def validate_trace(report, *, task, episode_index):
    if (report.get('task') != task or report.get('episode_index') != episode_index
            or report.get('complete') is not False
            or report.get('stop_reason') != 'infrastructure_error'
            or 'wall/physics backlog' not in report.get('error', '')
            or report.get('protocol') != 'continuous_full_sync16_v4_grid180_linear_bootstrap_v2'):
        raise ValueError('exact failed Full backlog trace required')
    dt = float(report['physics_step_seconds']); action_dt = float(report['action_interval_s'])
    steps = report['physics_steps']
    if (not math.isclose(dt, .004000000189989805, abs_tol=1e-12, rel_tol=0)
            or abs(action_dt-45*dt) > 1e-10 or type(steps) is not int or not 0 < steps <= 700*45
            or abs(float(report['physical_duration'])-steps*dt) > 1e-8):
        raise ValueError('trace changed the actual physics/action grid')
    events = [e for e in report['events'] if e.get('event') == 'execute']
    if (len(events) != report['started_targets'] or not (len(events)-1)*45 < steps <= len(events)*45
            or len(events) < 2):
        raise ValueError('recorded control rows do not cover the exact failure tick')
    for row, event in enumerate(events):
        target = event['target']
        if (event['row'] != row or event['physical_tick'] != row*45
                or abs(event['actual_start']-row*action_dt) > 1e-8
                or event.get('controller') != 'endpoint_linear_arm_native_gripper'
                or len(target) != 14 or not all(math.isfinite(float(x)) for x in target)):
            raise ValueError('noncontiguous, mistimed or nonfinite recorded control')
    if any(e['row_kind'] != 'bootstrap_hold' for e in events[:2]) or events[0]['target'] != events[1]['target']:
        raise ValueError('the original two held startup rows must remain intact')
    # This failure happened immediately after the last actual PhysX step.
    if report['owner_stage_timing']['totals']['transient']['calls'] != steps-1:
        raise ValueError('unsupported failure location: last transient stage differs')
    return dict(dt=dt, action_dt=action_dt, steps=steps, targets=[e['target'] for e in events])


def validate_config(config):
    if (config.get('run_kind') != MODE or config.get('cpu_workers') != [0, 2, 4]
            or config.get('task') != 'place_object_stand' or config.get('episode_index') != 7
            or config.get('seed') != 100000 or config.get('episodes') != 20):
        raise ValueError('explicit failed stand episode7, frozen20 and workers[0,2,4] required')
    if (not 0 < float(config['timeout_seconds']) <= 1800
            or not 0 < float(config.get('parity_atol', 1e-6)) <= 1e-6):
        raise ValueError('bounded timeout and strict finite state tolerance required')
    uuids = config['gpu_uuids']
    if len(uuids) != 1 or not uuids[0].startswith('GPU-'):
        raise ValueError('one renderer UUID required; the external guard still holds all adapters')
    storage = Path(config['storage_root']).resolve(strict=True)
    output = Path(config['output']).resolve()
    if storage.name != 'wuruihan' or not output.is_relative_to(storage):
        raise ValueError('diagnostic output must be on the verified wuruihan artifact disk')
    if sha(config['trace_report']) != config['trace_report_sha256']:
        raise ValueError('failed trace changed')
    trace = validate_trace(read(config['trace_report']), task=config['task'], episode_index=config['episode_index'])
    if abs(float(config['action_dt'])-trace['action_dt']) > 1e-10:
        raise ValueError('configuration action grid differs from trace')
    return output, trace


def validate_authorization(config, authorization, *, child=False):
    """Read-only guard check; no NVML, CUDA, model import or ad-hoc guard."""
    path = Path(authorization).resolve(strict=True); receipt = read(path)
    expected = dict(source_sha256=source_hash(ROOT/'src/dynamicwam'),
                    config_sha256=canonical_hash(config), init_checkpoint_sha256=config['init_checkpoint_sha256'])
    if (receipt.get('mode') != MODE or receipt.get('authorized') is not True
            or any(receipt.get('identity', {}).get(k) != v for k, v in expected.items())
            or receipt.get('source', {}).get('scripts_sha256') != source_hash(ROOT/'scripts')
            or receipt.get('gpu_uuids') != config['gpu_uuids'] or receipt.get('world_size') != 1
            or receipt.get('cuda_visible_devices') != config['gpu_uuids'][0]
            or os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuids'][0]):
        raise ValueError('replay authorization source/config/UUID differs')
    claim_path = path.with_suffix('.json.consumed')
    if child:
        claim = read(claim_path); parent = proc_identity(claim['pid'])
        if (claim.get('receipt_sha256') != sha(path) or parent['pid'] != os.getppid()
                or parent['start_ticks'] != claim['start_ticks'] or parent['state'] in ('Z', 'X')
                or claim['started_unix'] >= receipt['expires_at_unix']
                or os.getsid(0) != claim['session']):
            raise ValueError('child lost its timely same-session controller')
    elif float(receipt.get('expires_at_unix', 0)) <= time.time():
        raise ValueError('replay authorization expired')
    guard = receipt['external_guard']; owner = proc_identity(guard['pid'])
    if (guard.get('mode') != 'fds_before_telemetry_v1' or owner['state'] in ('Z', 'X')
            or owner['start_ticks'] != int(guard['start_ticks'])):
        raise ValueError('external guard exited or PID was reused')
    held = {str(fd.resolve()) for fd in (Path('/proc')/str(owner['pid'])/'fd').iterdir()}
    adapters = {str(p) for p in Path('/dev').glob('nvidia*') if p.name[6:].isdigit()}
    if not adapters or not adapters <= set(guard['device_paths']) <= held:
        raise ValueError('external guard lost adapter descriptors')
    ancestor = os.getppid()
    while ancestor > 1 and ancestor != owner['pid']:
        ancestor = proc_identity(ancestor)['ppid']
    if ancestor != owner['pid']:
        raise ValueError('external guard is not the replay ancestor')
    storage = Path(config['storage_root']).resolve(strict=True)
    if not path.is_relative_to(storage) or receipt.get('storage_root') != str(storage):
        raise ValueError('authorization must remain on declared artifact disk')
    return receipt, claim_path


def snapshot_state(env, dynamic_type):
    """Actual articulation state and both task actors; never drive targets."""
    import numpy as np
    fields = {}
    for side in ('left', 'right'):
        entity = getattr(env.robot, side+'_entity')
        fields[side+'_qpos'] = entity.get_qpos()
        fields[side+'_qvel'] = entity.get_qvel()
    for name in ('object', 'displaystand'):
        actor = getattr(env, name).actor; pose = actor.get_pose()
        components = [c for c in actor.get_components() if isinstance(c, dynamic_type)]
        if len(components) != 1:
            raise ValueError('expected one native dynamic body for '+name)
        fields[name+'_position'] = pose.p; fields[name+'_quaternion'] = pose.q
        fields[name+'_linear_velocity'] = components[0].linear_velocity
        fields[name+'_angular_velocity'] = components[0].angular_velocity
    labels, values = [], []
    for name, value in fields.items():
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError('invalid actual state '+name)
        labels.extend(f'{name}[{i}]' for i in range(vector.size)); values.extend(vector)
    return labels, np.asarray(values, dtype=np.float64)


def process_thread_count():
    return len(list(Path('/proc/self/task').iterdir()))


def replay_controls(session, trace, state_reader, *, deadline):
    """Mirror the Full per-tick executor; timing excludes diagnostic snapshots."""
    import numpy as np
    env = session.env; robot = env.robot; dt = trace['dt']; action_dt = trace['action_dt']
    names, initial = state_reader(); states = [initial]; rows = []; physical_times = [session.physics_time()]
    if abs(physical_times[0]) > 1e-12:
        raise ValueError('prepared replay clock is not zero')
    previous = np.asarray(trace['targets'][0], np.float64)
    target = previous.copy(); terminal = None
    for tick in range(trace['steps']):
        if time.monotonic() >= deadline:
            raise TimeoutError('bounded physics replay expired')
        row = tick//45
        if tick % 45 == 0:
            previous = target.copy(); target = np.asarray(trace['targets'][row], np.float64)
            env.take_action_cnt = int(getattr(env, 'take_action_cnt', 0))+1
            rows.append(dict(row=row, steps=0, physx_wall_seconds=0., owner_thread_cpu_seconds=0.,
                             process_cpu_seconds=0., max_step_wall_seconds=0., control_wall_seconds=0.))
        stat = rows[-1]; control_start = time.perf_counter()
        position = previous+(target-previous)*((tick % 45+1)/45); velocity = (target-previous)/action_dt
        position[[6,13]] = target[[6,13]]; velocity[[6,13]] = 0
        robot._entity_qf(robot.left_entity); robot._entity_qf(robot.right_entity)
        robot.set_arm_joints(position[:6], velocity[:6], 'left')
        robot.set_arm_joints(position[7:13], velocity[7:13], 'right')
        robot.set_gripper(position[6], 'left'); robot.set_gripper(position[13], 'right')
        env._update_kinematic_tasks()
        wall0 = time.perf_counter(); thread0 = time.thread_time_ns(); process0 = time.process_time_ns()
        env.scene.step()
        process_used = (time.process_time_ns()-process0)*1e-9
        thread_used = (time.thread_time_ns()-thread0)*1e-9; wall_used = time.perf_counter()-wall0
        stat['steps'] += 1; stat['physx_wall_seconds'] += wall_used
        stat['owner_thread_cpu_seconds'] += thread_used; stat['process_cpu_seconds'] += process_used
        stat['max_step_wall_seconds'] = max(stat['max_step_wall_seconds'], wall_used)
        now = session.physics_time()
        if abs(now-(tick+1)*dt) > 1e-8:
            raise ValueError('native replay skipped or duplicated a real physics step')
        # The recorded failure fires after its last scene.step, before these
        # hooks. All preceding ticks use the original transient/event flow.
        if tick+1 < trace['steps']:
            env._update_transient_checks(); session._consume_events()
            terminal = session.stop_reason()
            if terminal is not None:
                raise ValueError('replay terminated before the recorded failure: '+terminal)
        stat['control_wall_seconds'] += time.perf_counter()-control_start
        labels, values = state_reader()
        if labels != names:
            raise ValueError('actual state layout changed during replay')
        states.append(values); physical_times.append(now)
    return names, np.stack(states), np.asarray(physical_times, np.float64), rows


def compare_state_files(reference, candidate, *, atol):
    import numpy as np
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        if (set(a.files) != {'names','states','physics_times'} or set(a.files) != set(b.files)
                or not np.array_equal(a['names'], b['names'])
                or not np.array_equal(a['physics_times'], b['physics_times'])
                or a['states'].shape != b['states'].shape
                or not np.isfinite(a['states']).all() or not np.isfinite(b['states']).all()):
            raise ValueError('replay state layout, clocks or finite values differ')
        delta = np.abs(a['states']-b['states']); flat = int(delta.argmax())
        step, field = np.unravel_index(flat, delta.shape); maximum = float(delta[step,field])
        return dict(passed=maximum <= atol, max_absolute_difference=maximum, atol=atol,
                    worst_tick=int(step), worst_field=str(a['names'][field]),
                    compared_ticks=int(delta.shape[0]), compared_fields=int(delta.shape[1]))


def run_worker(config, authorization, workers):
    output, trace = validate_config(config); receipt, _ = validate_authorization(config, authorization, child=True)
    if workers not in config['cpu_workers']:
        raise ValueError('unrequested PhysX worker count')
    case = output/f'workers_{workers}'; case.mkdir(parents=True, exist_ok=False)
    descriptor, motion, _ = load_frozen_task(config, config['task'])[config['episode_index']]
    recorded = read(Path(config['trace_report']).parent/'manifest.json')
    if descriptor['seed'] != 100007 or any(recorded.get(k) != v for k,v in descriptor.items()):
        raise ValueError('trace manifest differs from frozen accepted scene')
    verify_artifacts(config)
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(dict(config, run_kind='domino_deployment_smoke'), os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay, activate_dependency_overlay
    dependencies = validate_dependency_overlay(config)
    if receipt['python_dependencies'] != dependencies:
        raise ValueError('dependency overlay differs from guard')
    activate_dependency_overlay(dependencies)
    import numpy as np
    import torch
    import cv2
    import yaml
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    uuid = config['gpu_uuids'][0]; pci = []
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields = dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID','').strip() == uuid: pci.append(path.parent.name)
    if len(pci) != 1: raise ValueError('selected UUID has no unique PCI mapping')
    os.environ['DYNAMICWAM_GPU_PCI'] = pci[0]; torch.cuda.set_device(0)
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    renderer = install_gpu_binding(physx_cpu_workers=workers)
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env, DominoLevel1Session
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    from dynamicwam.dualrate.deployment import bind_gil_releasing_physx_step
    from dynamicwam.dualrate.contact_cache import StepContactCache
    from dynamicwam.dualrate.producer import episode_prompt
    import sapien
    domino = Path(config['domino_root']).resolve(strict=True)
    limits = yaml.safe_load((domino/'task_config/_eval_step_limit.yml').read_text())
    env = cache = None; report = None; deadline = time.monotonic()+float(config['timeout_seconds'])
    thread_counts = dict(before_scene_creation=process_thread_count())
    try:
        with SapienInitLock():
            env, prepared = prepare_level1_env(domino, task_name=config['task'], task_config=config['task_config'],
                start_seed=descriptor['seed'], instruction_type=config.get('instruction_type','seen'),
                replay=dict(seed=descriptor['seed'], instruction=descriptor['instruction'], saved_motion=motion['saved_motion']))
        thread_counts['after_scene_creation'] = process_thread_count()
        binding = getattr(env.scene, '_dynamicwam_physx_binding', None)
        if (not isinstance(binding,dict) or binding.get('cpu_workers') != workers
                or binding.get('native_scene_fields_verified') is not True):
            raise ValueError('scene has no verified exact native PhysX configuration')
        env._rt_episode_info = motion['episode_info']
        if descriptor['arm_tag']:
            from envs.utils.action import ArmTag
            env.arm_tag = ArmTag(descriptor['arm_tag'])
        actual = dict(seed=int(prepared['seed']), instruction=prepared['instruction'], velocity_m_per_s=prepared['velocity'],
            initial_target_position=np.asarray(env._eval_target_actor.actor.get_pose().p),
            initial_state=env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState(),
            stop_on_contact=prepared['stop_on_contact'], check_z_threshold=prepared['check_z_threshold'],
            arm_tag=str(getattr(env,'arm_tag','')), physics_dt_s=float(env.scene.get_timestep()),
            native_action_limit=int(limits[config['task']]))
        for key in ('seed','instruction','arm_tag','stop_on_contact','check_z_threshold','native_action_limit'):
            if actual[key] != descriptor[key]: raise ValueError('frozen replay differs: '+key)
        for key in ('initial_target_position','initial_state','velocity_m_per_s','physics_dt_s'):
            if not np.allclose(actual[key], descriptor[key], atol=1e-8, rtol=0):
                raise ValueError('frozen replay numeric field differs: '+key)
        if int(env.step_lim) != int(limits[config['task']]): raise ValueError('native action limit differs')
        if not np.allclose(actual['initial_state'], trace['targets'][0], atol=1e-8, rtol=0):
            raise ValueError('initial held command differs from failed trace')
        prompt, _ = episode_prompt(config['task'], motion['episode_info']['info'], domino/'description')
        env.set_instruction(instruction=prompt)
        session = DominoLevel1Session(env, flow=None, observation_config={}, stop_on_contact=prepared['stop_on_contact'],
            check_z_threshold=prepared['check_z_threshold'], lift_actor=prepared['lift_actor'], start_z=prepared['start_z'],
            instruction=prompt, control_interval_s=trace['action_dt'])
        cache = StepContactCache(env.scene, session.physics_time).install()
        bind_gil_releasing_physx_step(env, _sapien_color_reader.step_cpu_physx)
        names, states, times, rows = replay_controls(session, trace,
            lambda:snapshot_state(env, sapien.physx.PhysxRigidDynamicComponent), deadline=deadline)
        if cache.error is not None: raise cache.error
        thread_counts['after_replay'] = process_thread_count()
        np.savez_compressed(case/'state_trace.npz', names=np.asarray(names), states=states, physics_times=times)
        save(case/'row_timings.json', rows)
        report = dict(complete=True, diagnostic_only=True, benchmark_acceptance=False, realtime_valid=False,
            cpu_workers=workers, physics_steps=trace['steps'], task=config['task'], episode_index=config['episode_index'],
            seed=descriptor['seed'], renderer=renderer, physx_binding=binding, process_thread_counts=thread_counts,
            states_sha256=sha(case/'state_trace.npz'), row_timings_sha256=sha(case/'row_timings.json'),
            native_control_semantics='recorded_endpoint_linear_arm_native_gripper',
            cameras_submitted=0, model_calls=0, no_wall_pacing=True,
            timing_scope='native_scene_step_wall_owner_thread_and_process_CPU; control_excludes_state_snapshot')
    except BaseException as exc:
        save(case/'failure.json',dict(complete=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if cache is not None: cache.close()
        finally:
            if env is not None:
                env.close_env(); env.release_episode_resources()
    report['environment_closed'] = True
    save(case/'worker_report.json',report)
    return 0


def main(config, authorization):
    config_path = Path(config).resolve(strict=True); config = read(config_path)
    output, trace = validate_config(config); receipt, claim = validate_authorization(config, authorization)
    output.mkdir(parents=True, exist_ok=True)
    with claim.open('x', encoding='utf-8') as handle:
        import json
        json.dump(dict(**proc_identity(os.getpid()),receipt_sha256=sha(authorization),started_unix=time.time()),handle)
    deadline = time.monotonic()+float(config['timeout_seconds']); comparisons = []; candidates = []
    frozen_episode = Path(config['prepared'])/config['task']/'episode_0007'
    evidence = [dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in
                (config['trace_report'],Path(config['prepared'])/'frozen_manifest.json',
                 frozen_episode/'manifest.json',frozen_episode/'expert_motion.pkl')]
    common = dict(schema=SCHEMA, diagnostic_only=True, benchmark_acceptance=False, realtime_valid=False,
        task=config['task'],episode_index=config['episode_index'],seed=100007,
        source_report_sha256=config['trace_report_sha256'],frozen_manifest_sha256=config['prepared_sha256'],
        manifest_sha256=sha(Path(config['prepared'])/config['task']/'episode_0007/manifest.json'),
        motion_sha256=sha(Path(config['prepared'])/config['task']/'episode_0007/expert_motion.pkl'),
        source_sha256=receipt['identity']['source_sha256'], scripts_sha256=receipt['source']['scripts_sha256'],
        config_sha256=canonical_hash(config),gpu_uuids=config['gpu_uuids'],physics_steps=trace['steps'])
    common['parity_atol'] = float(config.get('parity_atol',1e-6))
    try:
        for workers in config['cpu_workers']:
            save(output/'status.json',dict(state='running',cpu_workers=workers,diagnostic_only=True))
            with (output/f'workers_{workers}.log').open('xb') as log:
                process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),
                    '--authorization',str(Path(authorization).resolve()),'--cpu-workers',str(workers)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=max(.01, deadline-time.monotonic()))
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass  # Existing guard owns final bounded SID cleanup.
                    raise
            if code: raise RuntimeError(f'workers_{workers} exited {code}; no retry')
            case = output/f'workers_{workers}'; report = read(case/'worker_report.json')
            if not report.get('complete') or not report.get('environment_closed'):
                raise ValueError('replay child did not close its scene')
            rows = read(case/'row_timings.json')
            worst = max(r['control_wall_seconds'] for r in rows if r['steps']==45)
            candidates.append(dict(cpu_workers=workers,worst_full_row_control_wall_seconds=worst,
                                   all_full_rows_within_action_dt=worst<=trace['action_dt']))
            for name in ('state_trace.npz','row_timings.json','worker_report.json'):
                evidence.append(dict(path=str(case/name),sha256=sha(case/name)))
            if workers:
                result = compare_state_files(output/'workers_0/state_trace.npz',case/'state_trace.npz',
                                             atol=float(config.get('parity_atol',1e-6)))
                comparisons.append(dict(cpu_workers=workers,**result))
                if not result['passed']: raise ValueError(f'workers_{workers} state trajectory differs; no formal eligibility')
        eligible = [c for c in candidates[1:] if c['all_full_rows_within_action_dt'] and
                    c['worst_full_row_control_wall_seconds']<candidates[0]['worst_full_row_control_wall_seconds']]
        selected = min(eligible,key=lambda c:c['worst_full_row_control_wall_seconds']) if eligible else None
        save(output/'comparison.json',dict(common,passed=True,state_parity_passed=True,
            selected_candidate_passed=selected is not None,
            selected_physx_cpu_workers=None if selected is None else selected['cpu_workers'],
            selection_rule='all-step state parity; every full control row <= action_dt; lower worst row than workers0',
            candidates=candidates,comparisons=comparisons,evidence=evidence,
            limitation='Single-scene no-render replay; does not establish 8-environment closed-loop realtime acceptance.'))
        save(output/'status.json',dict(state='completed',diagnostic_only=True))
        return 0
    except BaseException as exc:
        save(output/'comparison.json',dict(common,passed=False,state_parity_passed=False,
            selected_candidate_passed=False,selected_physx_cpu_workers=None,comparisons=comparisons,
            candidates=candidates,evidence=evidence,error=repr(exc),traceback=traceback.format_exc()))
        save(output/'status.json',dict(state='failed',diagnostic_only=True,error=repr(exc)))
        raise


if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--authorization',required=True)
    parser.add_argument('--cpu-workers',type=int,choices=[0,2,4])
    args=parser.parse_args()
    raise SystemExit(main(args.config,args.authorization) if args.cpu_workers is None
                     else run_worker(read(args.config),args.authorization,args.cpu_workers))
