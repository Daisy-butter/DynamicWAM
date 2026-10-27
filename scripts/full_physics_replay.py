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
DIAGNOSIS_MODE = 'failed_trace_cost_v1'
DIAGNOSIS_SCHEMA = 'full_failed_trace_cost_diagnosis_v1'
CAMERA_MODE = 'camera_overlap_v1'


def is_cost_diagnosis(config):
    mode=config.get('diagnosis_mode')
    if mode not in (None,DIAGNOSIS_MODE,CAMERA_MODE):
        raise ValueError('unknown explicitly selected replay diagnosis mode')
    return mode==DIAGNOSIS_MODE


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
    diagnosis=is_cost_diagnosis(config) or config.get('diagnosis_mode')==CAMERA_MODE
    tail = config.get('tail_hold_rows', 0)
    if type(tail) is not int or not 0 <= tail <= 10 or type(config.get('passive_force_dedup', False)) is not bool:
        raise ValueError('tail hold rows must be 0..10 and passive force setting must be boolean')
    if diagnosis:
        if (config.get('run_kind')!=MODE or config.get('cpu_workers')!=[0]
                or config.get('task')!='place_empty_cup' or config.get('episode_index')!=12
                or config.get('expected_episode_seed')!=100014
                or config.get('seed')!=100000 or config.get('episodes')!=20):
            raise ValueError('cost diagnosis requires exact frozen cup12 seed100014 and workers0')
        if any(k.startswith('reference_') for k in config):
            raise ValueError('cost diagnosis cannot consume or certify old state-parity references')
    elif (config.get('run_kind') != MODE or config.get('cpu_workers') != ([0] if tail else [0, 2, 4])
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
    if tail and not diagnosis:
        if sha(config['reference_state_trace']) != config['reference_state_trace_sha256']:
            raise ValueError('original workers0 state reference changed')
        if config.get('reference_extension_trace') is not None:
            if sha(config['reference_extension_trace']) != config['reference_extension_trace_sha256']:
                raise ValueError('extended baseline state reference changed')
    if config.get('passive_force_dedup', False) and not diagnosis:
        if not tail or not config.get('reference_extension_trace'):
            raise ValueError('dedup candidate requires the baseline extended state reference')
        if sha(config['reference_extension_timings']) != config['reference_extension_timings_sha256']:
            raise ValueError('extended baseline timing reference changed')
    recorded=read(config['trace_report'])
    if diagnosis:
        manifest=Path(config['trace_report']).with_name('manifest.json')
        if sha(manifest)!=config['trace_manifest_sha256']:
            raise ValueError('failed cup manifest hash changed')
        bound=read(manifest)
        if (bound.get('task')!=config['task'] or bound.get('episode_index')!=12
                or bound.get('seed')!=config['expected_episode_seed']
                or bound.get('requested_seed')!=config['expected_episode_seed']):
            raise ValueError('failed cup manifest task/episode/seed mismatch')
        if recorded.get('passive_force_dedup',False) is not config.get('passive_force_dedup',False):
            raise ValueError('cost diagnosis must retain the failed controller setting')
        if recorded.get('physx_binding',{}).get('cpu_workers')!=0:
            raise ValueError('cost diagnosis requires the failed native workers0 setting')
    trace = validate_trace(recorded, task=config['task'], episode_index=config['episode_index'])
    if abs(float(config['action_dt'])-trace['action_dt']) > 1e-10:
        raise ValueError('configuration action grid differs from trace')
    if config.get('diagnosis_mode')==CAMERA_MODE:
        from full_camera_overlap import validate_camera_config
        validate_camera_config(config,trace)
    return output, trace


def extended_trace(trace, config):
    """Old comparison semantics stay fixed; new diagnosis adds at most N*45 ticks."""
    tail=config.get('tail_hold_rows',0)
    if not tail: return trace
    steps=trace['steps']+tail*45 if is_cost_diagnosis(config) else len(trace['targets'])*45+tail*45
    return dict(trace,recorded_steps=trace['steps'],steps=steps,
                targets=trace['targets']+[trace['targets'][-1]]*tail)


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


def snapshot_state(env, dynamic_type, *, actor_names=('object','displaystand')):
    """Actual articulation state and both task actors; never drive targets."""
    import numpy as np
    fields = {}
    for side in ('left', 'right'):
        entity = getattr(env.robot, side+'_entity')
        fields[side+'_qpos'] = entity.get_qpos()
        fields[side+'_qvel'] = entity.get_qvel()
    for name in actor_names:
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


def replay_controls(session, trace, state_reader, *, deadline, step_costs=None, passive_force_dedup=False,
                    before_tick=None,after_tick=None):
    """Mirror the Full per-tick executor; timing excludes diagnostic snapshots."""
    import numpy as np
    from contextlib import nullcontext
    if passive_force_dedup:
        from dynamicwam.dualrate.passive_force import passive_force_scope
        force_scope = lambda: passive_force_scope(session.env.robot, enabled=True)
    else:
        force_scope = lambda: nullcontext(None)
    env = session.env; robot = env.robot; dt = trace['dt']; action_dt = trace['action_dt']
    names, initial = state_reader(); states = [initial]; rows = []; physical_times = [session.physics_time()]
    if abs(physical_times[0]) > 1e-12:
        raise ValueError('prepared replay clock is not zero')
    previous = np.asarray(trace['targets'][0], np.float64)
    target = previous.copy(); terminal = None
    recorded_steps = trace.get('recorded_steps', trace['steps'])
    for tick in range(trace['steps']):
        if time.monotonic() >= deadline:
            raise TimeoutError('bounded physics replay expired')
        if before_tick is not None: before_tick(tick)
        row = tick//45
        if tick % 45 == 0:
            previous = target.copy(); target = np.asarray(trace['targets'][row], np.float64)
            env.take_action_cnt = int(getattr(env, 'take_action_cnt', 0))+1
            rows.append(dict(row=row, steps=0, physx_wall_seconds=0., owner_thread_cpu_seconds=0.,
                             process_cpu_seconds=0., max_step_wall_seconds=0., control_wall_seconds=0.,
                             passive_force_native_evaluations=0,passive_force_reused_calls=0))
        stat = rows[-1]; control_start = time.perf_counter()
        position = previous+(target-previous)*((tick % 45+1)/45); velocity = (target-previous)/action_dt
        position[[6,13]] = target[[6,13]]; velocity[[6,13]] = 0
        with force_scope() as force_stats:
            robot._entity_qf(robot.left_entity); robot._entity_qf(robot.right_entity)
            robot.set_arm_joints(position[:6], velocity[:6], 'left')
            robot.set_arm_joints(position[7:13], velocity[7:13], 'right')
            robot.set_gripper(position[6], 'left'); robot.set_gripper(position[13], 'right')
        if force_stats is not None:
            stat['passive_force_native_evaluations'] += force_stats['native_evaluations']
            stat['passive_force_reused_calls'] += force_stats['reused_calls']
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
        if tick+1 != recorded_steps and tick+1 < trace['steps']:
            env._update_transient_checks(); session._consume_events()
            terminal = session.stop_reason()
            if terminal is not None and tick+1 < recorded_steps:
                raise ValueError('replay terminated before the recorded failure: '+terminal)
        control_used = time.perf_counter()-control_start
        labels, values = state_reader()
        if labels != names:
            raise ValueError('actual state layout changed during replay')
        states.append(values); physical_times.append(now)
        if tick+1 == recorded_steps and trace['steps'] > recorded_steps:
            # Save the exact original failure state before performing the
            # transient hooks skipped by its backlog exception. Continuing is
            # unscored fixed-control diagnosis, even after a task stop event.
            hook_start = time.perf_counter()
            env._update_transient_checks(); session._consume_events(); terminal = session.stop_reason()
            control_used += time.perf_counter()-hook_start
        stat['control_wall_seconds'] += control_used
        if terminal is not None and 'first_terminal_tick' not in stat:
            stat.update(first_terminal_tick=tick+1,terminal=terminal)
        if step_costs is not None:
            step_costs.append(dict(tick=tick+1,row=row,physical_time=now,physx_wall_seconds=wall_used,
                owner_thread_cpu_seconds=thread_used,process_cpu_seconds=process_used,
                control_wall_seconds=control_used,terminal=terminal))
        if after_tick is not None: after_tick(tick+1)
    return names, np.stack(states), np.asarray(physical_times, np.float64), rows


def compare_state_files(reference, candidate, *, atol, prefix_ticks=None):
    import numpy as np
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        if (set(a.files) != {'names','states','physics_times'} or set(a.files) != set(b.files)
                or not np.array_equal(a['names'], b['names'])
                or a['states'].dtype != np.float64 or b['states'].dtype != np.float64
                or a['physics_times'].dtype != np.float64 or b['physics_times'].dtype != np.float64
                or not np.isfinite(a['states']).all() or not np.isfinite(b['states']).all()):
            raise ValueError('replay state layout, clocks or finite values differ')
        a_states, b_states = a['states'], b['states']; a_times, b_times = a['physics_times'], b['physics_times']
        if prefix_ticks is not None:
            if len(a_states) != prefix_ticks or len(b_states) < prefix_ticks:
                raise ValueError('reference does not cover exactly the original prefix')
            b_states=b_states[:prefix_ticks]; b_times=b_times[:prefix_ticks]
        if a_states.shape != b_states.shape or a_times.shape != b_times.shape or a_times.tobytes()!=b_times.tobytes():
            raise ValueError('replay state layout, clocks or finite values differ')
        delta = np.abs(a_states-b_states); flat = int(delta.argmax())
        step, field = np.unravel_index(flat, delta.shape); maximum = float(delta[step,field])
        bitwise = a_states.tobytes()==b_states.tobytes()
        return dict(passed=bitwise if atol==0 else maximum <= atol,bitwise_equal=bitwise,
                    max_absolute_difference=maximum, atol=atol,
                    worst_tick=int(step), worst_field=str(a['names'][field]),
                    compared_ticks=int(delta.shape[0]), compared_fields=int(delta.shape[1]))


def camera_free_backlog_projection(recorded_report, step_costs, dt):
    """Optimistic cost projection, not an active-clock test or guaranteed bound."""
    capture = [e for e in recorded_report['events'] if e.get('event')=='capture_submitted'][-1]
    anchor = next(e for e in recorded_report['events'] if e.get('event')=='execute'
                  and e['physical_tick']==capture['exposure_tick'])
    lag = float(anchor['clock_pair']['backlog_seconds']); peak=lag; peak_tick=anchor['physical_tick']
    rows=[]; recovered=[]
    for item in step_costs:
        if item['tick'] <= anchor['physical_tick']: continue
        before=lag;lag=max(0.,lag+item['control_wall_seconds']-dt)
        if lag>peak:peak=lag;peak_tick=item['tick']
        if before>1e-9 and lag<=1e-9:recovered.append(item['tick'])
        if item['tick']%45==0 or item is step_costs[-1]:
            rows.append(dict(tick=item['tick'],physical_time=item['physical_time'],projected_backlog_s=lag))
    return dict(anchor_tick=anchor['physical_tick'],anchor_backlog_s=anchor['clock_pair']['backlog_seconds'],
        peak_projected_backlog_s=peak,peak_tick=peak_tick,final_projected_backlog_s=lag,recovery_ticks=recovered,
        backlog_limit_unchanged_s=recorded_report['max_backlog_limit_s'],row_boundaries=rows,
        actual_benchmark_clock_test=False,excludes='camera, model, pacing, state snapshots and benchmark instrumentation',
        interpretation='Optimistic isolated control-cost projection; not a guarantee of realtime feasibility.')


def run_worker(config, authorization, workers, *, camera_case=None):
    output, trace = validate_config(config); receipt, _ = validate_authorization(config, authorization, child=True)
    camera_diagnosis=config.get('diagnosis_mode')==CAMERA_MODE
    diagnosis=is_cost_diagnosis(config) or camera_diagnosis
    if camera_diagnosis != (camera_case in ('sync','async')):
        raise ValueError('explicit camera child must match camera diagnosis mode')
    recorded_steps = trace['steps']; tail = config.get('tail_hold_rows', 0)
    trace=extended_trace(trace,config)
    if workers not in config['cpu_workers']:
        raise ValueError('unrequested PhysX worker count')
    case = output/(f'camera_{camera_case}' if camera_diagnosis else f'workers_{workers}'); case.mkdir(parents=True, exist_ok=False)
    descriptor, motion, _ = load_frozen_task(config, config['task'])[config['episode_index']]
    recorded = read(Path(config['trace_report']).parent/'manifest.json')
    expected_seed=config['expected_episode_seed'] if diagnosis else 100007
    if descriptor['seed'] != expected_seed or any(recorded.get(k) != v for k,v in descriptor.items()):
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
    env = cache = camera = None; report = None; deadline = time.monotonic()+float(config['timeout_seconds'])
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
        step_costs = []
        actor_names=('cup','coaster') if diagnosis else ('object','displaystand')
        experiment=None;camera_sources=None
        if camera_diagnosis:
            from full_camera_overlap import CameraExperiment,source_evidence
            from dynamicwam.runtime.domino_rt.camera_capture import AsyncCameraCapture
            camera_sources=source_evidence(sapien,_sapien_color_reader,config)
            camera=AsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch,
                async_submit=camera_case=='async',worker_initializer=lambda:torch.cuda.set_device(0))
            experiment=CameraExperiment(session,camera,case,async_submit=camera_case=='async',deadline=deadline)
            experiment.warmup()
        names, states, times, rows = replay_controls(session, trace,
            lambda:snapshot_state(env, sapien.physx.PhysxRigidDynamicComponent,actor_names=actor_names), deadline=deadline,
            step_costs=step_costs,passive_force_dedup=config.get('passive_force_dedup',False),
            before_tick=None if experiment is None else experiment.before_tick,
            after_tick=None if experiment is None else experiment.after_tick)
        if cache.error is not None: raise cache.error
        thread_counts['after_replay'] = process_thread_count()
        np.savez_compressed(case/'state_trace.npz', names=np.asarray(names), states=states, physics_times=times)
        save(case/'row_timings.json', rows)
        save(case/'step_timings.json', step_costs)
        report = dict(complete=True, diagnostic_only=True, benchmark_acceptance=False, realtime_valid=False,
            cpu_workers=workers, physics_steps=trace['steps'], task=config['task'], episode_index=config['episode_index'],
            recorded_physics_steps=recorded_steps,tail_hold_rows=tail,
            passive_force_dedup=config.get('passive_force_dedup',False),
            tail_ignores_task_terminal_for_diagnostic=bool(tail),
            seed=descriptor['seed'], renderer=renderer, physx_binding=binding, process_thread_counts=thread_counts,
            states_sha256=sha(case/'state_trace.npz'), row_timings_sha256=sha(case/'row_timings.json'),
            step_timings_sha256=sha(case/'step_timings.json'),
            native_control_semantics='recorded_endpoint_linear_arm_native_gripper',
            cameras_submitted=0, model_calls=0, no_wall_pacing=True,
            timing_scope='native_scene_step_wall_owner_thread_and_process_CPU; control_excludes_state_snapshot')
        if diagnosis:
            report.update(schema=DIAGNOSIS_SCHEMA,diagnosis_mode=DIAGNOSIS_MODE,
                state_snapshot_scope='actual_robot_articulations_and_task_cup_coaster',
                recorded_state_parity_tested=False,tail_extension_steps=trace['steps']-recorded_steps,
                no_score=True)
        if camera_diagnosis:
            if source_evidence(sapien,_sapien_color_reader,config)!=camera_sources:
                raise ValueError('camera source files changed while the child was active')
            report.update(experiment.report(),camera_source_evidence=camera_sources,
                state_trace_path=str(case/'state_trace.npz'),state_count=len(states),
                diagnosis_mode=CAMERA_MODE,cameras_submitted=6,native_take_picture_calls=24,
                render_update_calls=6)
    except BaseException as exc:
        save(case/'failure.json',dict(complete=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if camera is not None: camera.close()
        finally:
            try:
                if cache is not None: cache.close()
            finally:
                if env is not None:
                    env.close_env(); env.release_episode_resources()
    report['environment_closed'] = True
    if camera_diagnosis: report['camera_closed']=True
    save(case/'worker_report.json',report)
    return 0


def main(config, authorization):
    config_path = Path(config).resolve(strict=True); config = read(config_path)
    if config.get('diagnosis_mode')==CAMERA_MODE:
        from full_camera_overlap import main as camera_main
        return camera_main(config_path,authorization)
    output, trace = validate_config(config); receipt, claim = validate_authorization(config, authorization)
    output.mkdir(parents=True, exist_ok=True)
    with claim.open('x', encoding='utf-8') as handle:
        import json
        json.dump(dict(**proc_identity(os.getpid()),receipt_sha256=sha(authorization),started_unix=time.time()),handle)
    deadline = time.monotonic()+float(config['timeout_seconds']); comparisons = []; candidates = []
    diagnosis=is_cost_diagnosis(config)
    frozen_episode = Path(config['prepared'])/config['task']/f"episode_{config['episode_index']:04d}"
    evidence = [dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in
                (config['trace_report'],Path(config['prepared'])/'frozen_manifest.json',
                 frozen_episode/'manifest.json',frozen_episode/'expert_motion.pkl')]
    common = dict(schema=SCHEMA, diagnostic_only=True, benchmark_acceptance=False, realtime_valid=False,
        task=config['task'],episode_index=config['episode_index'],seed=config['expected_episode_seed'] if diagnosis else 100007,
        source_report_sha256=config['trace_report_sha256'],frozen_manifest_sha256=config['prepared_sha256'],
        manifest_sha256=sha(frozen_episode/'manifest.json'),motion_sha256=sha(frozen_episode/'expert_motion.pkl'),
        source_sha256=receipt['identity']['source_sha256'], scripts_sha256=receipt['source']['scripts_sha256'],
        config_sha256=canonical_hash(config),gpu_uuids=config['gpu_uuids'],physics_steps=trace['steps'])
    common['parity_atol'] = float(config.get('parity_atol',1e-6))
    if diagnosis:
        common.update(schema=DIAGNOSIS_SCHEMA,diagnosis_mode=DIAGNOSIS_MODE,no_score=True,
                      recorded_state_parity_tested=False)
        manifest=Path(config['trace_report']).with_name('manifest.json')
        evidence.append(dict(path=str(manifest),sha256=sha(manifest)))
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
            for name in ('state_trace.npz','row_timings.json','step_timings.json','worker_report.json'):
                evidence.append(dict(path=str(case/name),sha256=sha(case/name)))
            if diagnosis:
                projection=camera_free_backlog_projection(read(config['trace_report']),
                    read(case/'step_timings.json'),trace['dt'])
                save(output/'backlog_projection.json',projection)
                evidence.append(dict(path=str(output/'backlog_projection.json'),sha256=sha(output/'backlog_projection.json')))
                result=dict(common,complete=True,physics_steps=report['physics_steps'],
                    recorded_physics_steps=trace['steps'],tail_hold_rows=config.get('tail_hold_rows',0),
                    tail_extension_steps=report['physics_steps']-trace['steps'],
                    state_count=report['physics_steps']+1,physx_cpu_workers=0,
                    passive_force_dedup=config.get('passive_force_dedup',False),
                    state_snapshot_scope=report['state_snapshot_scope'],environment_closed=True,
                    camera_free_backlog_projection=projection,timing_summary=candidates[0],evidence=evidence,
                    limitation='Unscored fixed-control cost diagnosis; no recorded-state parity or closed-loop acceptance claim.')
                save(output/'diagnosis.json',result)
                save(output/'status.json',dict(state='completed',diagnostic_only=True,no_score=True))
                return 0
            if config.get('tail_hold_rows',0):
                prefix = compare_state_files(config['reference_state_trace'],case/'state_trace.npz',
                    atol=0.,prefix_ticks=trace['steps']+1)
                comparisons.append(dict(kind='original_failure_prefix',**prefix))
                if not prefix['passed']: raise ValueError('continuation changed original failure prefix bits')
                evidence.append(dict(path=config['reference_state_trace'],sha256=sha(config['reference_state_trace'])))
                projection = camera_free_backlog_projection(read(config['trace_report']),
                    read(case/'step_timings.json'),trace['dt'])
                save(output/'backlog_projection.json',projection)
                evidence.append(dict(path=str(output/'backlog_projection.json'),sha256=sha(output/'backlog_projection.json')))
                extra = dict(schema='full_physics_replay_continuation_v1',passed=True,state_parity_passed=True,
                    state_count=report['physics_steps']+1,prefix_state_count=trace['steps']+1,
                    physics_steps=report['physics_steps'],recorded_physics_steps=trace['steps'],
                    prefix_bitwise_equal=True,selected_candidate_passed=False,selected_physx_cpu_workers=None,
                    physx_cpu_workers=0,tail_hold_rows=config['tail_hold_rows'],
                    environment_closed=True,selected_passive_force_dedup=False,
                    camera_free_backlog_projection=projection,comparisons=comparisons,evidence=evidence)
                if config.get('passive_force_dedup',False):
                    extension=compare_state_files(config['reference_extension_trace'],case/'state_trace.npz',atol=0.)
                    comparisons.append(dict(kind='baseline_full_extension',**extension))
                    if not extension['passed']: raise ValueError('dedup changed extended baseline state bits')
                    before_rows=read(config['reference_extension_timings'])
                    if [(r['row'],r['steps']) for r in rows] != [(r['row'],r['steps']) for r in before_rows]:
                        raise ValueError('baseline and dedup timing grids differ')
                    def totals(items):
                        return dict(total_control_wall_seconds=sum(r['control_wall_seconds'] for r in items),
                            total_physx_wall_seconds=sum(r['physx_wall_seconds'] for r in items),
                            worst_full_row_control_wall_seconds=max(r['control_wall_seconds'] for r in items if r['steps']==45))
                    before,after=totals(before_rows),totals(rows)
                    evidence.extend(dict(path=config[k],sha256=sha(config[k])) for k in
                                    ('reference_extension_trace','reference_extension_timings'))
                    extra.update(schema='full_passive_force_dedup_replay_v1',extension_bitwise_equal=True,
                        selected_passive_force_dedup=True,timing_before=before,timing_after=after,
                        timing_benefit_recorded=True,
                        control_wall_improved=after['total_control_wall_seconds']<before['total_control_wall_seconds'])
                save(output/'comparison.json',dict(common,**extra))
                save(output/'status.json',dict(state='completed',diagnostic_only=True))
                return 0
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
        if diagnosis:
            save(output/'diagnosis.json',dict(common,complete=False,evidence=evidence,
                error=repr(exc),traceback=traceback.format_exc()))
            save(output/'status.json',dict(state='failed',diagnostic_only=True,no_score=True,error=repr(exc)))
            raise
        save(output/'comparison.json',dict(common,passed=False,state_parity_passed=False,
            selected_candidate_passed=False,selected_physx_cpu_workers=None,comparisons=comparisons,
            candidates=candidates,evidence=evidence,error=repr(exc),traceback=traceback.format_exc()))
        save(output/'status.json',dict(state='failed',diagnostic_only=True,error=repr(exc)))
        raise


if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--authorization',required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--cpu-workers',type=int,choices=[0,2,4])
    selection.add_argument('--camera-case',choices=['sync','async'])
    args=parser.parse_args()
    raise SystemExit(run_worker(read(args.config),args.authorization,0,camera_case=args.camera_case)
        if args.camera_case is not None else main(args.config,args.authorization) if args.cpu_workers is None
        else run_worker(read(args.config),args.authorization,args.cpu_workers))
