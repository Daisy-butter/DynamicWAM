#!/usr/bin/env python3
"""One fixed-seed continuous-physics DOMINO Level-1 episode.

Run with the DOMINO benchmark Python (do not resolve its executable symlink).
--prepare-only accepts the requested expert scene once; every policy arm must
replay the identical record. Continuous scores are a NEW controller protocol,
not directly comparable to the published synchronous native-TOPP benchmark.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BUFFERED_POLICY_FACTORIES = frozenset((
    'dynamicwam.runtime.buffered_world_context_worker:create_worker_policy',
    'dynamicwam.runtime.buffered_predictive_feedback_worker:create_worker_policy',
))
for entry in (ROOT/'src', ROOT/'external'/'flash_attn_runtime'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--task', required=True)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--accepted-record', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--task-config')
    p.add_argument('--policy-config', type=Path)
    p.add_argument('--policy-factory', default='dynamicwam.runtime.world_context_worker:create_worker_policy')
    p.add_argument('--mode', choices=['sync', 'async'], default='async')
    p.add_argument('--physics-timeout', type=float, default=30.)
    p.add_argument('--query-slots', type=int, default=2)
    p.add_argument('--control-period', type=float, default=.1)
    p.add_argument('--max-backlog', type=float, default=.1)
    p.add_argument('--max-tick-gap', type=float, default=.1)
    p.add_argument('--max-velocity', type=float, default=2.)
    p.add_argument('--max-acceleration', type=float, default=20.)
    p.add_argument('--warmup-queries', type=int, default=1)
    p.add_argument('--max-prepare-attempts', type=int, default=10)
    p.add_argument('--record-trajectory', action='store_true',
                   help='Bounded in-memory real RGB/command collection; flush only after physics stops.')
    p.add_argument('--synchronous-camera', action='store_true',
                   help='Diagnostic legacy camera path; formal continuous benchmark uses a replica renderer.')
    return p.parse_args()


def plain(value):
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n')
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    args.config=args.config.absolute(); args.output=args.output.absolute()
    args.accepted_record=args.accepted_record.absolute()
    if args.policy_config:
        args.policy_config=args.policy_config.absolute()
    if abs(args.control_period-.1) > 1e-9:
        raise ValueError('v1 protocol is explicitly 100 ms; changing it requires a new protocol')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'episode directory is immutable: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    from dynamicwam.config.loader import load_profile
    profile = load_profile(args.config)
    benchmark=profile.benchmark_config(); inference=profile.inference_config()
    from dynamicwam.evaluation.headless_render import apply_headless_nvidia_egl
    apply_headless_nvidia_egl()
    for entry in (benchmark['curobo_root'], str(Path(benchmark['domino_root'])/'script')):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from dynamicwam.runtime.domino_rt.preparation import prepare_fixed_scene
    from dynamicwam.runtime.domino_rt.continuous_session import ContinuousSession
    from dynamicwam.runtime.domino_rt.process_worker import ProcessPolicyWorker
    from dynamicwam.runtime.domino_rt.continuous_loop import run_continuous_episode
    policy_config={} if args.policy_config is None else json.loads(args.policy_config.read_text())
    policy_config.setdefault('config', str(args.config))
    policy_config.setdefault('project_root', str(ROOT))
    worker=None; renderer=None; env=None
    result=dict(format='domino_continuous_episode', version=1, task=args.task, seed=args.seed,
        status='running', protocol='domino_continuous_wall_v1', policy_factory=args.policy_factory,
        policy_config=policy_config, mode=args.mode,
        runtime=dict(executable=os.path.abspath(sys.executable), python=platform.python_version(), prefix=sys.prefix),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        controller=dict(control_period_s=.1, physics_dt=.004, kind='bounded_hermite_command',
                        max_velocity=args.max_velocity, max_acceleration=args.max_acceleration,
                        query_slots=args.query_slots, physics_timeout_s=args.physics_timeout,
                        max_backlog_s=args.max_backlog, max_tick_gap_s=args.max_tick_gap,
                        warmup_queries=args.warmup_queries,
                        metrics_cadence_s=.1, transient_checks_every_tick=True,
                        camera_pipeline='synchronous_diagnostic' if args.synchronous_camera else 'independent_scene_snapshot_renderer',
                        event_checks_every_tick=True, native_topp_equivalent=False))
    result['controller_protocol_sha256']=policy_config.get('controller_protocol_sha256')
    result['runtime']['thread_environment']={key:os.environ.get(key) for key in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    result['actual_controller_sha256']=hashlib.sha256(json.dumps(result['controller'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    source_paths=[ROOT/'src'/'dynamicwam'/'runtime'/'domino_rt'/name for name in
        ('motion.py','continuous_session.py','continuous_loop.py','timed_control.py','process_worker.py','preparation.py',
         'render_worker.py','scene_snapshot.py','trajectory_recorder.py','proposal_path.py')]
    source_paths += [Path(__file__).resolve()]
    result['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    atomic_json(args.output/'summary.json', result)
    try:
        # Spawn before SAPIEN. CUDA_VISIBLE_DEVICES belongs to the launcher;
        # one physical GPU may be shared, and its contention remains measured.
        if not args.prepare_only:
            if not args.accepted_record.exists():
                raise FileNotFoundError('prepare the fixed accepted record before evaluation')
            worker=ProcessPolicyWorker(factory=args.policy_factory, config=policy_config)
        env, prepared, record=prepare_fixed_scene(Path(benchmark['domino_root']), task=args.task,
            task_config=args.task_config or benchmark['task_config'], seed=args.seed,
            instruction_type=benchmark['instruction_type'], record_path=args.accepted_record,
            prepare_only=args.prepare_only, max_attempts=args.max_prepare_attempts)
        result['accepted_record']=dict(path=str(args.accepted_record), sha256=hashlib.sha256(args.accepted_record.read_bytes()).hexdigest())
        result['instruction']=record['instruction']; result['domino_source_sha256']=record['source_sha256']
        result['task_state_keys']=sorted(record['task_state'])
        if args.prepare_only:
            result['status']='prepared'; result['prepare_attempt']=record['attempt']
            return 0
        session=ContinuousSession(env, prepared, flow_config=inference['head_flow'],
            observation_config=inference['observation'], control_period=.1)
        if not args.synchronous_camera:
            from dynamicwam.runtime.domino_rt.render_worker import ProcessRenderWorker
            renderer=ProcessRenderWorker(dict(domino_root=benchmark['domino_root'],task=args.task,
                task_config=args.task_config or benchmark['task_config'],seed=args.seed,
                instruction_type=benchmark['instruction_type'],accepted_record=str(args.accepted_record)))
            session.attach_renderer(renderer)
            result['renderer_pid']=renderer.pid
            result['scene_identity']=renderer.identity
            result['scene_identity_sha256']=hashlib.sha256(json.dumps(renderer.identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        from eval_metrics import EvalMetricsTracker
        session.metrics_tracker=EvalMetricsTracker(env, {'use_dynamic': True})
        session.metrics_tracker.on_episode_start()
        session.warm_camera()
        warmup=[]
        for index in range(args.warmup_queries):
            packet=session.capture_packet()
            packet.update(sim_time=0., observation_sim_time=0., control_period_s=.1, command_index=0,
                _reset_policy=True, _warmup_only=True, _sampling_seed=args.seed+index,
                task_name=args.task, episode_seed=args.seed)
            started=time.perf_counter(); worker.submit(packet); worker.drain()
            warmup.append(time.perf_counter()-started)
        # Camera/graph warmup must not leak duplicate-time history into episode.
        session.flow.reset(); session._previous_head=None; session._previous_stamp=None
        result['warmup_query_wall_s']=warmup
        result['warmup_physics_time']=session.physics_time()
        if session.physics_time() != 0.:
            raise RuntimeError('warmup unexpectedly advanced physics')
        recorder=None
        if args.record_trajectory:
            if policy_config.get('collection_dir') and args.policy_factory not in BUFFERED_POLICY_FACTORIES:
                raise ValueError('dense recording prohibits worker disk writes during physics')
            from dynamicwam.runtime.domino_rt.trajectory_recorder import TrajectoryRecorder
            recorder=TrajectoryRecorder(max_seconds=args.physics_timeout)
        with (args.output/'events.jsonl').open('x') as handle:
            def sink(event):
                handle.write(json.dumps(event, default=plain, allow_nan=False)+'\n')
                handle.flush()
            report=run_continuous_episode(session, worker, task=args.task, seed=args.seed,
                mode=args.mode, control_period=.1, query_slots=args.query_slots,
                physics_timeout=args.physics_timeout, max_backlog=args.max_backlog,
                max_tick_gap=args.max_tick_gap, max_velocity=args.max_velocity,
                max_acceleration=args.max_acceleration, sink=sink,trajectory_recorder=recorder)
        result['report']=report
        result['status']='complete' if report['realtime_valid'] else 'infrastructure_invalid'
        if recorder is not None:
            if args.policy_config is None:
                raise ValueError('trajectory recording requires an identified policy config file')
            result['trajectory']=recorder.write(args.output/'trajectory',provenance=dict(
                task=args.task,seed=args.seed,accepted_record=result['accepted_record'],
                accepted_record_sha256=result['accepted_record']['sha256'],
                policy_config_sha256=hashlib.sha256(args.policy_config.read_bytes()).hexdigest(),
                controller_protocol_sha256=result['controller_protocol_sha256'],
                source_sha256=result['source_sha256'],policy_factory=args.policy_factory))
        if args.policy_factory in BUFFERED_POLICY_FACTORIES:
            worker.drain()
            result['buffered_queries']=worker.finalize_recordings()
        if report['realtime_valid']:
            tracker=session.metrics_tracker
            tracker.on_step()  # Final pose, including a terminal inference hold.
            if report['stop_reason']=='out_of_bounds':
                tracker.record_out_of_bounds()
            episode=tracker.get_episode_metrics(report['success'],
                None if report['success'] else report['stop_reason'], seed=args.seed)
            result['metrics']=dataclasses.asdict(episode)
        # Do not install or score the in-flight terminal response.
        return 0 if report['realtime_valid'] else 2
    except BaseException:
        result['status']='error'; result['error']=traceback.format_exc()
        print(result['error'], file=sys.stderr, flush=True)
        return 2
    finally:
        if worker is not None:
            worker.close()
        if renderer is not None:
            renderer.close()
        if env is not None:
            try:
                env.close_env(); env.release_episode_resources()
            except Exception:
                result['cleanup_error']=traceback.format_exc()
        atomic_json(args.output/'summary.json', result)


if __name__=='__main__':
    raise SystemExit(main())
