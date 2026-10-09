#!/usr/bin/env python3
"""Real-scene RGB/FK oracle for the separate DOMINO rendering process.

This is a preflight diagnostic, never a policy-success episode. It keeps DOMINO's
render quality and compares cross-process error to measured source repeatability.
No externally chosen pixel-error allowance is added to that empirical envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
VIEWS = ('head_camera', 'left_camera', 'right_camera')


def image_errors(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape or first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError('RGB image shape/dtype mismatch')
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError('RGB image must be HWC with three channels')
    difference = np.abs(first.astype(np.int16)-second.astype(np.int16))
    return dict(mae=float(difference.mean()), p99=float(np.percentile(difference, 99)),
                maximum=int(difference.max()), changed_fraction=float(np.mean(difference != 0)),
                bitwise_equal=bool(np.array_equal(first, second)))


def compare_repetitions(source: list[dict], replica: list[dict]) -> dict:
    if len(source) < 3 or len(source) != len(replica):
        raise ValueError('at least three matched repetitions are required')
    result = {}
    for view in VIEWS:
        within = [image_errors(a[view], b[view]) for a, b in itertools.combinations(source, 2)]
        cross = [image_errors(a[view], b[view]) for a, b in zip(source, replica)]
        envelope = {key: max(row[key] for row in within) for key in ('mae', 'p99', 'maximum')}
        worst = {key: max(row[key] for row in cross) for key in envelope}
        passed = all(worst[key] <= envelope[key] for key in envelope)
        result[view] = dict(passed=passed, source_pair_count=len(within),
            source_repeatability=within, cross_process=cross,
            source_empirical_envelope=envelope, cross_process_worst=worst,
            source_bitwise_repeatable=all(row['bitwise_equal'] for row in within),
            replica_bitwise_identical=all(row['bitwise_equal'] for row in cross))
    return dict(passed=all(row['passed'] for row in result.values()), views=result,
        criterion='cross_process_max_error_le_source_pairwise_max_error_per_view',
        limitation='finite-sample engineering parity gate, not statistical distribution equivalence')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physics_time(env) -> float:
    return float(env._scene_step_clock.snapshot().time_seconds)


def _source_render(env, bridge, snapshot) -> tuple[dict, dict]:
    before = _physics_time(env)
    oracle = bridge.apply(snapshot)
    env.scene.update_render()
    env.cameras.update_picture()
    raw = env.cameras.get_rgb()
    images = {view: np.ascontiguousarray(raw[view]['rgb']).copy() for view in VIEWS}
    if _physics_time(env) != before or oracle['physics_steps'] != 0:
        raise RuntimeError('source parity rendering advanced physics')
    return images, oracle


def _replica_render(worker, snapshot, timeout: float = 60.) -> tuple[dict, dict]:
    packet = dict(source_physics_time_seconds=snapshot['source_sim_time_seconds'],
                  snapshot_sequence=snapshot['sequence'])
    capture_id = worker.submit(snapshot, packet)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is None:
            time.sleep(.001)
            continue
        oracle = result['snapshot_oracle']
        if (result['capture_id'] != capture_id or oracle['sequence'] != snapshot['sequence'] or
                oracle['source_sim_time_seconds'] != snapshot['source_sim_time_seconds'] or
                oracle['physics_steps'] != 0 or not oracle['passed']):
            raise RuntimeError('replica source identity/FK/zero-step oracle mismatch')
        return result['images'], oracle
    raise TimeoutError('replica image deadline exceeded')


def _physical_motion(env, *, ticks: int, joint_delta: float) -> dict:
    robot = env.robot
    q = np.asarray([*robot.get_left_arm_jointState(), *robot.get_right_arm_jointState()], dtype=np.float64)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError('expected 14 commanded joint drive targets')
    robots = (robot.left_entity, robot.right_entity)
    before_q = [np.asarray(a.get_qpos()).copy() for a in robots]
    actor = env._eval_target_actor.actor
    before_object = np.asarray(actor.get_pose().p).copy()
    before_time = _physics_time(env)
    q[0] += joint_delta
    for _ in range(ticks):
        robot._entity_qf(robot.left_entity); robot._entity_qf(robot.right_entity)
        robot.set_arm_joints(q[:6], np.zeros(6), 'left')
        robot.set_arm_joints(q[7:13], np.zeros(6), 'right')
        robot.set_gripper(float(q[6]), 'left'); robot.set_gripper(float(q[13]), 'right')
        env._update_kinematic_tasks()
        env.scene.step()
        env._update_transient_checks()
        env.take_action_cnt = int(_physics_time(env)/.1+1e-8)
    after_q = [np.asarray(a.get_qpos()).copy() for a in robots]
    change_q = max(float(np.max(np.abs(after-before), initial=0.))
                   for before, after in zip(before_q, after_q))
    change_object = float(np.linalg.norm(np.asarray(actor.get_pose().p)-before_object))
    return dict(ticks=ticks, joint_delta=joint_delta,
        physics_time_before=before_time, physics_time_after=_physics_time(env),
        max_actual_robot_qpos_change=change_q, target_translation=change_object,
        robot_motion_observed=change_q > 1e-7, object_motion_observed=change_object > 1e-7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/absolute_motion_v2.yaml')
    parser.add_argument('--domino-root', help='Optional explicit override of the benchmark profile')
    parser.add_argument('--task', required=True)
    parser.add_argument('--task-config', default='demo_clean_dynamic')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--accepted-record', required=True)
    parser.add_argument('--instruction-type', default='seen')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--motion-ticks', type=int, default=25)
    parser.add_argument('--joint-delta', type=float, default=.03)
    args = parser.parse_args()
    if args.repeats < 3 or not 1 <= args.motion_ticks <= 250 or not 0 < abs(args.joint_delta) <= .05:
        parser.error('require repeats>=3, motion ticks 1..250 and joint delta in (0,.05] magnitude')
    output = Path(args.output).expanduser().resolve()
    record = Path(args.accepted_record).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'parity result is immutable: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = output.parent/(output.stem+'_images')
    image_root.mkdir(exist_ok=False)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    import torch
    torch.set_num_threads(1)
    from dynamicwam.config.loader import load_profile
    profile = load_profile(args.config.absolute())
    benchmark = profile.benchmark_config()
    from dynamicwam.evaluation.headless_render import apply_headless_nvidia_egl
    apply_headless_nvidia_egl()
    domino_root = Path(args.domino_root or benchmark['domino_root']).resolve()
    for entry in (benchmark['curobo_root'], str(domino_root/'script')):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from dynamicwam.runtime.domino_rt.preparation import prepare_fixed_scene
    from dynamicwam.runtime.domino_rt.render_worker import ProcessRenderWorker
    from dynamicwam.runtime.domino_rt.scene_snapshot import SceneSnapshotBridge
    config = dict(domino_root=str(domino_root), task=args.task,
        task_config=args.task_config, seed=args.seed, instruction_type=args.instruction_type,
        accepted_record=str(record))
    report = dict(format='dynamicwam_scene_replica_parity', version=1, status='running',
        config=dict(path=str(args.config.absolute()), sha256=_sha(args.config)),
        task=args.task, seed=args.seed, instruction_type=args.instruction_type, task_config=args.task_config,
        accepted_record=dict(path=str(record), sha256=_sha(record)),
        source_sha256={str(p.relative_to(ROOT)): _sha(p) for p in (
            Path(__file__).resolve(), ROOT/'src/dynamicwam/runtime/domino_rt/scene_snapshot.py',
            ROOT/'src/dynamicwam/runtime/domino_rt/render_worker.py',
            ROOT/'src/dynamicwam/runtime/domino_rt/preparation.py')},
        repetitions=args.repeats, phases=[], policy_success_evaluation=False)
    env = worker = None
    try:
        env, _, _ = prepare_fixed_scene(Path(config['domino_root']), task=args.task,
            task_config=args.task_config, seed=args.seed, instruction_type=args.instruction_type,
            record_path=record)
        bridge = SceneSnapshotBridge(env)
        worker = ProcessRenderWorker(config)
        if bridge.identity != worker.identity:
            raise RuntimeError('source and replica scene/asset identity differ')
        report['scene_identity'] = bridge.identity
        for phase in ('initial', 'after_physical_motion'):
            if phase == 'after_physical_motion':
                report['motion'] = _physical_motion(env, ticks=args.motion_ticks, joint_delta=args.joint_delta)
            snapshot = bridge.capture()
            source, replica, source_oracles, replica_oracles = [], [], [], []
            for _ in range(args.repeats):
                images, oracle = _source_render(env, bridge, snapshot)
                source.append(images); source_oracles.append(oracle)
            for _ in range(args.repeats):
                images, oracle = _replica_render(worker, snapshot)
                replica.append(images); replica_oracles.append(oracle)
            evidence = compare_repetitions(source, replica)
            artifact = image_root/(phase+'.npz')
            np.savez_compressed(artifact, **{f'{owner}_{view}': np.stack([row[view] for row in rows])
                for owner, rows in (('source', source), ('replica', replica)) for view in VIEWS})
            evidence.update(phase=phase, source_time_seconds=snapshot['source_sim_time_seconds'],
                snapshot_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                source_oracles=source_oracles, replica_oracles=replica_oracles,
                repeated_images=dict(path=str(artifact), sha256=_sha(artifact),
                                     axes='repetition,height,width,channel'))
            report['phases'].append(evidence)
        motion_passed = report['motion']['robot_motion_observed'] and report['motion']['object_motion_observed']
        report['passed'] = motion_passed and all(phase['passed'] for phase in report['phases'])
        report['status'] = 'complete' if report['passed'] else 'parity_gate_failed'
    except BaseException:
        report.update(status='infrastructure_error', passed=False, error=traceback.format_exc())
    finally:
        if worker is not None:
            worker.close()
        if env is not None:
            try:
                env.close_env(); env.release_episode_resources()
            except Exception:
                report['cleanup_error'] = traceback.format_exc()
        with output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], passed=report.get('passed'), output=str(output))))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
