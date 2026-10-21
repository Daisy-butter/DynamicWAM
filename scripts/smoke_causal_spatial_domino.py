#!/usr/bin/env python3
"""One real development DOMINO episode, H2/R2, continuous wall clock at 1x.

Run only inside the parent's bounded all-adapter guard. This diagnostic does
not authorize a success-rate claim or silently substitute a fake simulator.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "external/flash_attn_runtime", ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path: sys.path.insert(0, str(entry))


def verify_parent_guard(owner_pid):
    """Read-only proof that the named live parent retains every adapter FD."""
    if type(owner_pid) is not int or owner_pid < 1:
        raise ValueError("explicit guard owner PID is required")
    expected = {}
    for path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        expected["/dev/nvidia" + fields["Device Minor"].strip()] = fields["GPU UUID"].strip()
    if len(expected) != 8:
        raise RuntimeError("Route1 smoke requires eight visible adapters and all-adapter guard")
    held = set()
    for path in (Path("/proc") / str(owner_pid) / "fd").iterdir():
        try: held.add(os.readlink(path))
        except FileNotFoundError: pass
    if not set(expected).issubset(held):
        raise RuntimeError("guard owner does not hold all eight adapter FDs")
    return dict(owner_pid=owner_pid, devices=expected, verified=True, recovery_operations=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-owner-pid", type=int, default=os.getppid(),
        help="all-adapter guard supervisor PID; defaults to the actual parent process")
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--seed", type=int, default=1610021)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--formal-scene-manifest", type=Path)
    parser.add_argument("--probe-action-limit", type=int)
    args = parser.parse_args()
    def interrupted(number, frame): raise KeyboardInterrupt(f"signal {number}")
    signal.signal(signal.SIGTERM, interrupted)
    from harness_preflight import require_ssd
    from eval_domino_continuous_sync import (save, source_provenance, close_episode_camera,
        cleanup_episode, drain_finished_episode, publish_episode_video_links, publish_native_task_results)
    from eval_domino_motionkv_rtc import preserve_failed_video
    require_ssd(args.output_dir); require_ssd(args.checkpoint)
    guard = verify_parent_guard(args.guard_owner_pid)
    config_path, checkpoint = args.config.resolve(), args.checkpoint.resolve()
    output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError("smoke output must be new; preserve every failed attempt")
    output.mkdir(parents=True)
    save(output / "guard_receipt.json", guard)
    import cv2
    import numpy as np
    import yaml
    cv2.setNumThreads(1)
    from dynamicwam.config.loader import load_profile
    from dynamicwam.runtime.domino_continuous.domino_session import prepare_level1_env
    import sapien
    from dynamicwam.runtime.domino_continuous.gpu_binding import install_gpu_binding
    binding = install_gpu_binding()
    if binding is None: raise RuntimeError("DYNAMICWAM_GPU_PCI and UUID pinning are required")
    profile = load_profile(config_path)
    benchmark, inference = profile.benchmark_config(), profile.inference_config()
    if args.task not in benchmark["tasks"]: raise ValueError("task outside configured DOMINO Level-1")
    domino = (ROOT / benchmark["domino_root"]).resolve()
    sys.path.insert(0, str(domino / "script"))
    task_config = yaml.safe_load((domino / "task_config" / f"{benchmark['task_config']}.yml").read_text())
    if not task_config.get("use_dynamic") or task_config.get("dynamic_level") != 1:
        raise ValueError("only dynamic DOMINO Level-1 is supported")
    native_limit = int(yaml.safe_load((domino / "task_config/_eval_step_limit.yml").read_text())[args.task])
    action_limit = native_limit if args.probe_action_limit is None else min(native_limit, args.probe_action_limit)
    if action_limit < 1: raise ValueError("positive native action budget required")
    formal_scenes = set()
    if args.formal_scene_manifest:
        frozen = json.loads(args.formal_scene_manifest.read_text())
        formal_scenes = {(row["task"], int(row["seed"])) for row in frozen["episodes"]}
        if (args.task, args.seed) in formal_scenes: raise ValueError("development seed belongs to frozen formal evaluation")
    task_output = output / args.task
    episode_dir = task_output / "episode_0000"
    episode_dir.mkdir(parents=True)
    stamp = output.name
    native = output / "eval_result" / args.task / "causal_spatial" / benchmark["task_config"] / checkpoint.stem / stamp
    env = session = worker = None
    events = []
    completed = False
    try:
        from dynamicwam.inference.causal_spatial_policy import CausalSpatialProcessWorker
        worker = CausalSpatialProcessWorker(project_root=str(ROOT), config_path=str(config_path),
            checkpoint=str(checkpoint), allow_diagnostic=args.allow_diagnostic)
        env, prepared = prepare_level1_env(domino, task_name=args.task, task_config=benchmark["task_config"],
            start_seed=args.seed, instruction_type=str(benchmark["instruction_type"]))
        if (args.task, int(prepared["seed"])) in formal_scenes:
            raise ValueError("expert feasibility filtering reached a frozen formal seed")
        descriptor = dict(task=args.task, episode_index=0, requested_seed=args.seed, seed=int(prepared["seed"]),
            instruction=prepared["instruction"], velocity_m_per_s=prepared["velocity"],
            native_action_limit=native_limit, development_scene=True,
            formal_scene_exclusion_verified=args.formal_scene_manifest is not None)
        save(episode_dir / "manifest.json", descriptor)
        (episode_dir / "expert_motion.pkl").write_bytes(pickle.dumps(dict(seed=descriptor["seed"],
            saved_motion=env._saved_dynamic_motion_info, episode_info=getattr(env, "_rt_episode_info", None))))
        instruction = worker.prepare_instruction(prepared["instruction"], timeout_s=600.)
        from dynamicwam.runtime.domino_continuous.camera_stream import PhysicalTimeHeadHistory
        from dynamicwam.runtime.domino_continuous.continuous_session import ContinuousSyncSession
        from dynamicwam.runtime.continuous_causal_spatial import (run_continuous_causal_spatial_episode,
            validate_causal_spatial_report, PROTOCOL)
        session = ContinuousSyncSession(env, flow=PhysicalTimeHeadHistory(inference["head_flow"],
            frame_period_s=.16, endpoint_tolerance_s=.08), action_interval_s=.16, save_video=True,
            timing_mode="nominal_training_timing", observation_config=dict(inference["observation"]),
            stop_on_contact=prepared["stop_on_contact"], check_z_threshold=prepared["check_z_threshold"],
            lift_actor=prepared["lift_actor"], start_z=prepared["start_z"], instruction=prepared["instruction"])
        from eval_metrics import EvalMetricsTracker
        session.metrics_tracker = EvalMetricsTracker(env, {"use_dynamic": True})
        session.metrics_tracker.on_episode_start()
        initial_clock = session.physics_time()
        equivalence = session.validate_camera_reader(env.get_obs())
        if session.physics_time() != initial_clock: raise RuntimeError("camera warmup advanced physics")
        save(episode_dir / "camera_equivalence.json", equivalence)
        provenance = source_provenance(semantics_id=PROTOCOL, config_path=config_path)
        # Include the newly separated controller/preprocessing contracts.
        for relative in ("src/dynamicwam/runtime/continuous_causal_spatial.py", "src/dynamicwam/runtime/causal_spatial_contract.py",
                         "src/dynamicwam/observations/causal_spatial_window.py", "scripts/smoke_causal_spatial_domino.py"):
            provenance["source_sha256"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        provenance["source_digest"] = hashlib.sha256(json.dumps(provenance["source_sha256"], sort_keys=True).encode()).hexdigest()
        provenance["source_generation"] = provenance["source_digest"][:16]
        save(output / "started.json", dict(**descriptor, **provenance, checkpoint=str(checkpoint),
            worker_startup=worker.startup_metadata, instruction_preparation=instruction,
            GPU_binding=binding, native_result_dir=str(native), simulation_rate=1., action_limit=action_limit))
        report = run_continuous_causal_spatial_episode(session, worker, task=args.task,
            episode_seed=descriptor["seed"], action_limit=action_limit, sink=events.append)
        validate_causal_spatial_report(report)
        endpoint = session.physics_time()
        report["post_terminal_camera_drain_wall_s"] = close_episode_camera(session)
        report["post_terminal_inference_drain_wall_s"] = drain_finished_episode(worker)
        report["post_terminal_policy_drain"] = worker.finish_episode()
        report.update(**descriptor, **provenance, checkpoint=str(checkpoint),
            checkpoint_sha256=worker.startup_metadata["checkpoint_sha256"], camera_equivalence=equivalence,
            video_required=True, native_result_dir=str(native), full_native_episode=args.probe_action_limit is None,
            smoke_only=True, performance_claim=False, worker_startup_metadata=worker.startup_metadata)
        video = session.head_video.finalize(native / "episode0.mp4", episode_end_physics_s=endpoint)
        report.update(video); report.update(publish_episode_video_links(episode_dir, video))
        tracker = session.metrics_tracker
        if report["stop_reason"] == "out_of_bounds": tracker.record_out_of_bounds()
        metric = tracker.get_episode_metrics(bool(report["success"]),
            None if report["stop_reason"] in {"success", "action_budget_exhausted"} else report["stop_reason"], seed=descriptor["seed"])
        save(episode_dir / "metrics.json", asdict(metric) if is_dataclass(metric) else vars(metric))
        save(episode_dir / "report.json", report)
        if report.get("realtime_valid") is not True or report.get("continuous_valid") is not True:
            raise RuntimeError("real closed-loop episode did not satisfy the declared 1x clock")
        if report.get("completed_targets", 0) < 1 or report.get("inference_calls", 0) < 2:
            raise RuntimeError("episode ended before an actual fast policy handoff")
        from video_artifacts import validate_video_artifacts
        if not validate_video_artifacts(report): raise RuntimeError("native MP4 validation failed")
        publish_native_task_results(task_output, native, count=1, planned=1, stamp=stamp,
            instruction_type=str(benchmark["instruction_type"]), checkpoint=checkpoint,
            generation=provenance["source_generation"], protocol=PROTOCOL, timing_mode="nominal_training_timing")
        completed = True
        save(output / "completed.json", dict(passed=True, full_native_episode=args.probe_action_limit is None,
            report=str(episode_dir / "report.json"), video=report["video"], success=report["success"],
            realtime_valid=True, performance_claim=False))
        print(json.dumps(dict(passed=True, report=str(episode_dir / "report.json"), video=report["video"])), flush=True)
    except BaseException as exc:
        save(output / "failure.json", dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            diagnostics=getattr(exc, "diagnostics", None), passed=False))
        if session is not None: preserve_failed_video(session, episode_dir, 0, exc)
        raise
    finally:
        # Drain model work before destroying the scene, and leave outer device
        # guards open until this entire process and its child have exited.
        try:
            if worker is not None:
                try:
                    drain_finished_episode(worker)
                    worker.finish_episode()
                finally:
                    worker.close()
        finally:
            cleanup_episode(session, env, episode_dir, events)
            save(output / "cleanup.json", dict(completed=completed, owned_worker_closed=worker is None or worker._closed,
                outer_guard_owner_pid=args.guard_owner_pid, final_driver_health_owned_by_supervisor=True))


if __name__ == "__main__": main()
