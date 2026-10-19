#!/usr/bin/env python3
"""Frozen-scene DOMINO continuous MotionKV RTC evaluation and scene preparation.

The accepted synchronous evaluator is deliberately unchanged. Preparation only
uses the environment/expert feasibility test; no policy weights are loaded and
no policy success can affect the selected scenes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_world_actual_latency_motionkv_rtc_h4r4_v1"
SCALED_PROTOCOL = "continuous_world_scaled_wall_motionkv_rtc_h4r4_v1"
JOINT_V3_PROTOCOL = "continuous_world_actual_latency_joint_motionkv_async_v3"
PREPARE_PROTOCOL = "domino_level1_policy_independent_scene_manifest_v1"
POLICY_LABEL = "motionkv_rtc"
for entry in (ROOT / "external/flash_attn_runtime", ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from eval_domino_continuous_sync import (
    acquire_lock, archive_file, close_episode_camera, cleanup_episode,
    drain_finished_episode, plain, publish_episode_video_links,
    publish_native_task_results, save, source_provenance,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def evaluation_protocol(rate=1., joint_v3=False):
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("simulation rate must be finite in (0,1]")
    if joint_v3:
        if rate != 1.:
            raise ValueError("joint v3 requires 1x continuous evaluation")
        return JOINT_V3_PROTOCOL
    return PROTOCOL if rate == 1. else SCALED_PROTOCOL


def valid_clock_identity(report):
    rate = report.get("simulation_rate", 1.)
    return (report.get("protocol") == evaluation_protocol(rate, report.get("joint_v3", False))
            and (report.get("realtime_valid") is True if rate == 1. else
                 report.get("clock_valid") is True and report.get("continuous_valid") is True
                 and report.get("realtime_valid") is False))


def provenance(config, simulation_rate=1., joint_v3=False):
    value = source_provenance(semantics_id=evaluation_protocol(simulation_rate, joint_v3), config_path=config)
    for name in ("scripts/eval_domino_motionkv_rtc.py", "scripts/eval_domino_motionkv_rtc_suite.py",
                 "scripts/device_lifetime_guard.py",
                 "src/dynamicwam/runtime/domino_continuous/motionkv_policy.py",
                 "src/dynamicwam/runtime/domino_continuous/development_replay.py",
                 "src/dynamicwam/runtime/domino_continuous/worker.py",
                 "scripts/scene_compatibility.py"):
        path = ROOT / name
        value["source_sha256"][name] = sha(path)
    for path in (ROOT / "src/dynamicwam/runtime/domino_continuous").glob("*rtc*.py"):
        value["source_sha256"][str(path.relative_to(ROOT))] = sha(path)
    value["source_digest"] = hashlib.sha256(json.dumps(value["source_sha256"], sort_keys=True).encode()).hexdigest()
    value["source_generation"] = value["source_digest"][:16]
    return value


def preserve_failed_video(session, episode_dir, episode, original_error):
    """Keep available real camera frames without accepting a failed rollout.

    Encoding happens after the rollout stopped. Diagnostics must never replace
    the original policy/clock failure, and failed videos stay outside native
    committed results and the valid-episode denominator.
    """
    receipt = dict(valid_episode=False, original_error=f"{type(original_error).__name__}: {original_error}",
                   video_preserved=False)
    try:
        if session is None or getattr(session, "head_video", None) is None:
            receipt["reason"] = "no_video_session"
        else:
            close_episode_camera(session)
            video = session.head_video
            if getattr(video, "_finalized", False):
                receipt["reason"] = "native_video_already_finalized_before_later_failure"
            elif not video.frames:
                receipt["reason"] = "no_completed_camera_frames"
            else:
                destination = episode_dir.parent / "failed_artifacts" / episode_dir.name / str(time.time_ns())
                receipt.update(video.finalize(destination / f"episode{episode}.mp4",
                                             episode_end_physics_s=session.physics_time()))
                receipt["video_preserved"] = True
    except BaseException as artifact_error:
        receipt["artifact_error"] = f"{type(artifact_error).__name__}: {artifact_error}"
    try:
        save(episode_dir / "failure_video_receipt.json", receipt)
    except BaseException:
        traceback.print_exc()
    return receipt


def validate_descriptor(item, task, episode, requested):
    if (item.get("task") != task or item.get("episode_index") != episode
            or item.get("requested_seed") != requested or type(item.get("seed")) is not int
            or item["seed"] < requested):
        raise ValueError("frozen scene task/index/seed chain mismatch")


def load_frozen_task(run, task, episodes, seed, config_sha, *, scene_compatibility=None, config_path=None):
    """Scene identity is independent of evaluated policy/checkpoint identity."""
    run = Path(run)
    suite = read(run / "suite_started.json")
    completed = read(run / "suite_completed.json")
    frozen = read(run / "frozen_manifest.json")
    compatible_config = suite.get("config_sha256") == config_sha
    if scene_compatibility is not None:
        from scene_compatibility import validate_receipt
        if config_path is None:
            raise ValueError("scene compatibility requires the actual evaluation profile")
        checked = validate_receipt(scene_compatibility, repo=ROOT, config=config_path, frozen_scenes=run)
        compatible_config = checked["evaluation_config_sha256"] == config_sha
    if (suite.get("protocol") != PREPARE_PROTOCOL or completed.get("complete") is not True
            or frozen.get("complete") is not True or task not in suite["tasks"]
            or suite["seed"] != seed or suite["episodes_per_task"] < episodes
            or not compatible_config):
        raise ValueError("reference is not a complete compatible frozen preparation suite")
    entries = [row for row in frozen["episodes"] if row["task"] == task]
    if len(entries) < episodes:
        raise ValueError("frozen task is incomplete")
    references = []
    requested = seed
    for episode, entry in enumerate(entries[:episodes]):
        base = run / task / f"episode_{episode:04d}"
        descriptor_path, motion_path = base / "manifest.json", base / "expert_motion.pkl"
        if sha(descriptor_path) != entry["manifest_sha256"] or sha(motion_path) != entry["motion_sha256"]:
            raise ValueError("frozen scene artifact hash mismatch")
        descriptor = read(descriptor_path)
        validate_descriptor(descriptor, task, episode, requested)
        motion_bytes = motion_path.read_bytes()
        motion = pickle.loads(motion_bytes)
        if motion["seed"] != descriptor["seed"]:
            raise ValueError("frozen expert motion seed mismatch")
        references.append((descriptor, motion, motion_bytes))
        requested = descriptor["seed"] + 1
    return references, sha(run / "frozen_manifest.json")


def checked_episode(path, task, episode, simulation_rate=None, *, expected_protocol=None):
    from video_artifacts import validate_video_artifacts
    report, metric, descriptor = (read(path / name) for name in ("report.json", "metrics.json", "manifest.json"))
    if expected_protocol is not None and report.get("protocol") != expected_protocol:
        raise ValueError("episode protocol differs from run")
    if (not valid_clock_identity(report)
            or report.get("task") != task or report.get("episode_index") != episode
            or report.get("video_required") is not True or not validate_video_artifacts(report)):
        raise ValueError("episode protocol/video validation failed")
    for key in ("task", "episode_index", "seed", "requested_seed"):
        if report[key] != descriptor[key]:
            raise ValueError("episode scene identity mismatch")
    score = metric["manipulation_score"]
    if (not math.isfinite(score) or not 0 <= score <= 100 or metric["seed"] != report["seed"]
            or bool(metric["success"]) != bool(report["success"])):
        raise ValueError("episode metric mismatch")
    if simulation_rate is not None and report.get("simulation_rate", 1.) != simulation_rate:
        raise ValueError("episode simulation rate differs from run")
    validate_rtc_report(report)
    return report, metric


def validate_rtc_report(report):
    """Independently verify saved causal timing, not just a runtime success flag."""
    rate = report.get("simulation_rate", 1.)
    joint_v3 = report.get("joint_v3", False)
    if joint_v3 and report.get("bootstrap_mode") != "prime_then_hold":
        raise ValueError("joint v3 must prime under hold then query a fresh observation")
    if not valid_clock_identity(report):
        raise ValueError("RTC protocol/clock identity mismatch")
    required = {"controller": "committed_clean_prefix_rtc_fixed_grid_h4r4", "clock": "actual_wall_clock" if rate == 1. else "scaled_wall_clock",
        "virtual_delay": False, "startup_delay_excluded": False, "rtc_strategy": "trained_clean_prefix_conditioning",
        "rtc_prefix_max": 4, "execute_horizon": 4, "slow_refresh_actions": 16, "limit_retiming": False,
        "rows_dropped": 0, "guided_requests": 0, "asynchronous_camera_readback": True}
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"RTC report contract mismatch: {key}")
    limits = report["clock_limits"]
    if (not 0 < limits["max_backlog_s"] <= .25 or not 0 < limits["inference_clock_error_s"] <= .02
            or not 0 <= report["max_backlog_s"] <= limits["max_backlog_s"] + 1e-8
            or (rate == 1. and not 0 <= report["max_inference_clock_error_s"] <= limits["inference_clock_error_s"] + 1e-8)):
        raise ValueError("RTC report clock limits failed")
    interval = report["action_interval_s"]
    previous_source = None
    episode_generation = None
    for index, request in enumerate(report.get("requests", [])):
        if rate != 1.:
            if request.get("simulation_rate") != rate:
                raise ValueError("RTC request rate differs from episode")
            dispatch, received = request["dispatch_wall_s"], request["receive_wall_s"]
            if not (request["observation_wall_s"] <= dispatch <= received):
                raise ValueError("RTC dispatch causal order failed")
            if not math.isclose(received - dispatch, request["dispatch_to_receive_wall_s"], abs_tol=1e-8):
                raise ValueError("RTC actual dispatch latency mismatch")
            global_error = rate * received - (request["receive_physics_s"] - report["initial_physics_s"])
            if not math.isclose(global_error, request["global_scaled_clock_error_s"], abs_tol=1e-8):
                raise ValueError("RTC scaled clock receipt mismatch")
            if not request.get("terminal_before_result") and abs(global_error) > limits["max_backlog_s"] + 1e-8:
                raise ValueError("RTC scaled clock global bound failed")
        expected_prefix = 0 if index == 0 else 4
        if request["query_index"] != index or request["prefix_length"] != expected_prefix:
            raise ValueError("RTC query order/prefix length mismatch")
        if not joint_v3 and (request["cache_age_actions"] != request["action_origin"] % 16 or request["cache_age_actions"] not in (0, 4, 8, 12)):
            raise ValueError("RTC cache age escaped training support")
        if joint_v3:
            import numpy as np
            if index == 0 and request.get("installed") and request.get("bootstrap_actions_executed") is not False:
                raise ValueError("joint v3 cannot execute stale bootstrap predictions")
            if request.get("installed"):
                info = request.get("policy_diagnostics", {})
                async_info = info.get("asynchronous_world", {})
                if (info.get("motion_interface_version") != 3 or info.get("source_bridge") != "student_joint"
                        or async_info.get("slow_schedule") != "asynchronous_latest_completed_v1"):
                    raise ValueError("joint v3 lacks the trained asynchronous source contract")
                source_index = async_info.get("source_action_index")
                source_time = async_info.get("source_sim_time")
                if (type(source_index) is not int or source_index < 0
                        or not isinstance(source_time, (int, float)) or not math.isfinite(source_time)):
                    raise ValueError("joint v3 source identity is missing")
                age = request["action_origin"] - source_index
                age_seconds = request["observation_physics_s"] - source_time
                if (not 0 <= age <= 31 or not 0 <= age_seconds <= 4.96 + 1e-8
                        or info.get("cache_age_actions") != age or async_info.get("cache_age_actions") != age
                        or not math.isclose(info.get("cache_age_seconds", -1), age_seconds, abs_tol=1e-8)
                        or not math.isclose(async_info.get("cache_age_seconds", -1), age_seconds, abs_tol=1e-8)):
                    raise ValueError("joint v3 actual source age escaped trained support or causality")
                times = [async_info.get(key) for key in ("source_submitted_monotonic",
                    "source_published_monotonic", "source_consumed_monotonic")]
                if (any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times)
                        or not times[0] <= times[1] <= times[2]):
                    raise ValueError("joint v3 consumed an unpublished source")
                generation = async_info.get("generation")
                if type(generation) is not int or generation < 0:
                    raise ValueError("joint v3 missing episode generation")
                if episode_generation is not None and generation != episode_generation:
                    raise ValueError("joint v3 source crossed episode generations")
                episode_generation = generation
                if previous_source is not None and (source_index < previous_source[0] or source_time < previous_source[1]):
                    raise ValueError("joint v3 source publication moved backwards")
                previous_source = source_index, source_time
                if index == 0 and (info.get("bootstrap_only") is not True or info.get("fast_sampler_ms") != 0.):
                    raise ValueError("joint v3 bootstrap must only prime the source")
            if index > 0:
                offsets = np.asarray(request.get("action_target_offsets"), dtype=np.float64)
                expected = report["action_grid_start_physics_s"] + (
                    request["action_origin"] + np.arange(1, 17)) * interval - request["observation_physics_s"]
                if offsets.shape != (16,) or not np.isfinite(offsets).all() or (offsets <= 0).any() or not np.allclose(offsets, expected, atol=1e-8, rtol=0):
                    raise ValueError("joint v3 action endpoints do not match the queue clock")
        if request["submit_physics_s"] + 1e-8 < request["observation_physics_s"]:
            raise ValueError("RTC request predates observation")
        if request.get("installed"):
            if request["install_physics_s"] + 1e-8 < request["submit_physics_s"]:
                raise ValueError("RTC installation predates submission")
            if expected_prefix:
                deadline = request["observation_physics_s"] + 4 * interval
                if report["initial_physics_s"] + rate * request["install_wall_s"] > deadline + 1e-8:
                    raise ValueError("RTC suffix missed actual declared-clock deadline")
                if (not math.isclose(request["suffix_not_before_physics_s"], deadline, abs_tol=1e-8)
                        or request["install_physics_s"] > deadline + 1e-8
                        or not math.isclose(request["suffix_effective_physics_s"], deadline, abs_tol=1e-8)):
                    raise ValueError("RTC suffix was late or moved off its committed action clock")
        if rate == 1. and request.get("inference_clock_check") == "complete_interval" and abs(request["inference_clock_error_s"]) > limits["inference_clock_error_s"] + 1e-8:
            raise ValueError("RTC inference interval clock failed")
    previous_end = None
    grid0 = report.get("action_grid_start_physics_s")
    physics_dt = report.get("physics_dt_s")
    ticks_per_action = None
    if report.get("blocks"):
        if not isinstance(grid0, (int, float)) or not math.isfinite(grid0) or not isinstance(physics_dt, (int, float)) or not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("RTC block clock identity is missing")
        ticks_per_action = round(interval / physics_dt)
        if ticks_per_action < 1 or abs(ticks_per_action * physics_dt - interval) > 1e-7:
            raise ValueError("RTC action interval is not an integer physics-tick interval")
    expected_origin = 0
    for block in report.get("blocks", []):
        if block["origin"] != expected_origin or type(block["count"]) is not int or not 1 <= block["count"] <= 4:
            raise ValueError("RTC block action indices are not contiguous")
        if previous_end is not None and abs(block["start_physics_s"] - previous_end) > 1e-7:
            raise ValueError("RTC block execution has a gap or overlap")
        # Planned rows use the immutable nominal action grid. Engine time uses
        # the actual (float32-valued) SAPIEN physics dt; rebasing planned rows on
        # each measured start accumulates deterministic dt representation error.
        # Check both clocks independently without widening either tolerance.
        expected_engine_start = grid0 + block["origin"] * ticks_per_action * physics_dt
        if abs(block["start_physics_s"] - expected_engine_start) > 1e-7:
            raise ValueError("RTC block start moved off the engine tick grid")
        expected_times = [grid0 + (block["origin"] + index + 1) * interval for index in range(block["count"])]
        if any(abs(actual - expected) > 1e-7 for actual, expected in zip(block["row_times_physics_s"], expected_times)) or len(block["row_times_physics_s"]) != len(expected_times):
            raise ValueError("RTC block action times changed")
        expected_origin += block["count"]
        if block.get("completed"):
            expected_engine_end = grid0 + expected_origin * ticks_per_action * physics_dt
            if abs(block["end_physics_s"] - expected_engine_end) > 1e-7:
                raise ValueError("RTC block end moved off the engine tick grid")
        previous_end = block.get("end_physics_s")


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/absolute_motion_v2.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-audit", type=Path)
    parser.add_argument("--timing-contract", type=Path)
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument("--scene-compatibility", type=Path)
    parser.add_argument("--result-stamp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cohort-dir", type=Path)
    parser.add_argument("--probe-action-limit", type=int)
    parser.add_argument("--simulation-rate", type=float, default=1.)
    parser.add_argument("--joint-v3", action="store_true",
                        help="require v3 joint checkpoint, background world and prime-then-hold at 1x")
    parser.add_argument("--development-replay-dir", type=Path,
                        help="opt-in raw same-trajectory collection on excluded development scenes only")
    parser.add_argument("--development-scene-allowlist", type=Path)
    parser.add_argument("--formal-scene-manifest", type=Path)
    parser.add_argument("--max-backlog-s", type=float, default=.25)
    parser.add_argument("--infer-clock-tolerance-s", type=float, default=.02)
    parser.add_argument("--action-interval-s", type=float, default=.16)
    parser.add_argument("--history-frame-period-s", type=float, default=.16)
    return parser.parse_args()


def main():
    args = parse()
    protocol = evaluation_protocol(args.simulation_rate, args.joint_v3)
    replay_scope = None
    replay_args = (args.development_replay_dir,args.development_scene_allowlist,args.formal_scene_manifest)
    if any(value is not None for value in replay_args):
        if not all(value is not None for value in replay_args) or not args.joint_v3 or args.prepare_only:
            raise ValueError("development replay requires joint-v3 and all three explicit replay scope paths")
        from dynamicwam.runtime.domino_continuous.development_replay import validate_development_scope,validate_requested_development_scenes
        replay_scope = validate_development_scope(args.development_scene_allowlist,args.formal_scene_manifest)
        validate_requested_development_scenes(replay_scope,args.replay_from,[args.task],args.episodes)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    import yaml
    from harness_preflight import require_ssd
    if replay_scope is not None: require_ssd(args.development_replay_dir)
    from dynamicwam.config.loader import load_profile
    from dynamicwam.runtime.domino_continuous.domino_session import prepare_level1_env
    import sapien  # Native dependency bootstrap before GPU binding.
    from dynamicwam.runtime.domino_continuous.gpu_binding import install_gpu_binding
    if install_gpu_binding() is None:
        raise RuntimeError("a pinned UUID/PCI identity is required")
    if args.episodes < 1 or args.seed < 0 or not (0 < args.max_backlog_s <= .25):
        raise ValueError("invalid episode count/seed/backlog")
    if not (0 < args.infer_clock_tolerance_s <= .02):
        raise ValueError("inference clock tolerance cannot exceed 20ms")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.action_interval_s, args.history_frame_period_s)):
        raise ValueError("invalid action/history grid")
    if args.probe_action_limit is not None and args.probe_action_limit < 1:
        raise ValueError("probe action limit must be positive")
    require_ssd(args.output_dir)
    profile = load_profile(args.config.resolve())
    benchmark, inference = profile.benchmark_config(), profile.inference_config()
    if args.task not in benchmark["tasks"]:
        raise ValueError("unknown Level-1 task")
    domino = (ROOT / benchmark["domino_root"]).resolve()
    sys.path.insert(0, str(domino / "script"))
    task_cfg = yaml.safe_load((domino / "task_config" / f"{benchmark['task_config']}.yml").read_text())
    if not task_cfg.get("use_dynamic") or task_cfg.get("dynamic_level") != 1:
        raise ValueError("only dynamic DOMINO Level-1 is supported")
    native_limit = int(yaml.safe_load((domino / "task_config/_eval_step_limit.yml").read_text())[args.task])
    action_limit = min(native_limit, args.probe_action_limit) if args.probe_action_limit else native_limit
    from rtc_episode_cohort import EpisodeCohort, MODE as COHORT_MODE
    cohort = None
    if not args.prepare_only:
        if args.cohort_dir is None or args.resume:
            raise ValueError("RTC evaluation requires a fresh fixed cohort; resume is not implicit")
        require_ssd(args.cohort_dir)
        cohort = EpisodeCohort(args.cohort_dir, args.task)
        if cohort.manifest["episodes"] != args.episodes:
            raise ValueError("cohort episode count differs")
    output = args.output_dir.resolve() / args.task
    output.mkdir(parents=True, exist_ok=True)
    source = provenance(args.config, args.simulation_rate, args.joint_v3)
    config_sha = sha(args.config)
    references, frozen_sha = (None, None)
    native_dir = None
    checkpoint_sha = None
    timing_sha = None
    if not args.prepare_only:
        if any(value is None for value in (args.checkpoint, args.checkpoint_audit, args.timing_contract,
                                           args.replay_from, args.result_stamp)):
            raise ValueError("evaluation requires checkpoint/audit/timing/frozen scenes/result stamp")
        require_ssd(args.checkpoint)
        audit = read(args.checkpoint_audit)
        checkpoint_sha = sha(args.checkpoint)
        contract = audit.get("contract", {})
        if (audit.get("passed") is not True or audit.get("strict_full_reload") is not True
                or audit.get("full_fast_equal") is not True or audit["full"]["sha256"] != checkpoint_sha
                or audit["full"]["bytes"] != args.checkpoint.stat().st_size
                or Path(audit["full"]["path"]).resolve() != args.checkpoint.resolve()
                or contract.get("rtc", {}).get("max_prefix_actions") != 4
                or contract.get("native_multirate", {}).get("fast_steps") != 10
                or contract.get("native_multirate", {}).get("source_bridge") != ("student_joint" if args.joint_v3 else "immutable_official")
                or contract.get("execute_horizon") != 4 or contract.get("max_age_actions") != (31 if args.joint_v3 else 12)):
            raise ValueError("checkpoint does not match the strict audited MotionKV RTC export")
        if args.joint_v3 and (contract["native_multirate"].get("version") != 3
                or contract["native_multirate"].get("slow_schedule") != "asynchronous_latest_completed_v1"
                or contract.get("max_age_seconds") != 4.96):
            raise ValueError("checkpoint lacks joint v3 asynchronous physical-time contract")
        timing = read(args.timing_contract)
        expected = timing["recommended_continuous_sync_contract"]
        if (not math.isclose(args.action_interval_s, expected["nominal_action_interval_seconds"], abs_tol=1e-12)
                or not math.isclose(args.history_frame_period_s, expected["nominal_history_frame_interval_seconds"], abs_tol=1e-12)):
            raise ValueError("controller timing differs from the frozen audited training clock")
        timing_sha = sha(args.timing_contract)
        references, frozen_sha = load_frozen_task(args.replay_from, args.task, args.episodes, args.seed, config_sha,
            scene_compatibility=args.scene_compatibility, config_path=args.config)
        if any(value in args.result_stamp for value in ("/", "\\")):
            raise ValueError("result stamp must be one directory name")
        native_dir = args.output_dir.resolve() / "eval_result" / args.task / POLICY_LABEL / benchmark["task_config"] / args.checkpoint.stem / args.result_stamp
        native_dir.mkdir(parents=True, exist_ok=True)
    spec = dict(schema_version=1, protocol=PREPARE_PROTOCOL if args.prepare_only else protocol,
                task=args.task, episodes=args.episodes, start_seed=args.seed, prepare_only=args.prepare_only,
                config_sha256=config_sha, source_digest=source["source_digest"], code_root=str(ROOT),
                scene_compatibility_sha256=sha(args.scene_compatibility) if args.scene_compatibility else None,
                checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                checkpoint_sha256=checkpoint_sha, timing_contract_sha256=timing_sha,
                frozen_manifest_sha256=frozen_sha, native_action_limit=native_limit, action_limit=action_limit,
                probe_action_limit=args.probe_action_limit, native_result_dir=str(native_dir) if native_dir else None,
                action_interval_s=args.action_interval_s, history_frame_period_s=args.history_frame_period_s,
                simulation_rate=args.simulation_rate, max_backlog_s=args.max_backlog_s, infer_clock_tolerance_s=args.infer_clock_tolerance_s,
                opencv_threads=cv2.getNumThreads(), scheduling_mode=COHORT_MODE if cohort else None,
                cohort_dir=str(cohort.directory) if cohort else None,
                cohort_manifest_sha256=sha(cohort.directory / "manifest.json") if cohort else None)
    if replay_scope is not None:
        spec["development_replay"] = dict(root=str(args.development_replay_dir.resolve()),
            allowlist_sha256=replay_scope["allowlist_sha256"],formal_manifest_sha256=replay_scope["formal_manifest_sha256"])
    lock = acquire_lock(output / ".evaluation.lock")
    started = output / "task_started.json"
    if started.exists():
        if not args.resume or read(started) != spec:
            raise RuntimeError("existing task differs or resume was not explicit")
    else:
        save(started, spec)
    save(output / "source_manifest.json", source)
    completed, manifests, worker = [], [], None
    current = dict(task=args.task, episode_index=0, requested_seed=args.seed)
    try:
        requested = args.seed
        for episode in range(args.episodes):
            directory = output / f"episode_{episode:04d}"
            current = dict(task=args.task, episode_index=episode, requested_seed=requested)
            if args.resume and directory.exists():
                if args.prepare_only and (directory / "prepared.json").exists():
                    descriptor = read(directory / "manifest.json")
                    validate_descriptor(descriptor, args.task, episode, requested)
                    receipt = read(directory / "prepared.json")
                    if receipt != {"manifest_sha256": sha(directory / "manifest.json"), "motion_sha256": sha(directory / "expert_motion.pkl")}:
                        raise ValueError("prepared scene changed")
                    manifests.append(descriptor); completed.append(descriptor); requested = descriptor["seed"] + 1
                    continue
                if not args.prepare_only and (directory / "report.json").exists():
                    report, metric = checked_episode(directory, args.task, episode, args.simulation_rate, expected_protocol=protocol)
                    if report["source_digest"] != source["source_digest"] or report["checkpoint_sha256"] != checkpoint_sha:
                        raise ValueError("completed episode source/checkpoint changed")
                    descriptor = read(directory / "manifest.json")
                    validate_descriptor(descriptor, args.task, episode, requested)
                    manifests.append(descriptor); completed.append(dict(seed=report["seed"], success=report["success"])); requested = report["seed"] + 1
                    continue
                archive_file(directory, output / "attempts")
                if native_dir is not None:
                    for suffix in (".mp4", "_timestamps.json"):
                        archive_file(native_dir / f"episode{episode}{suffix}",
                                     output / "artifact_attempts" / f"episode_{episode:04d}")
            directory.mkdir(parents=True, exist_ok=False)
            events, env, session, replay_recorder = [], None, None, None
            reference = references[episode] if references is not None else None
            try:
                env, prepared = prepare_level1_env(domino, task_name=args.task, task_config=benchmark["task_config"],
                    start_seed=reference[0]["seed"] if reference else requested,
                    instruction_type=str(benchmark["instruction_type"]),
                    replay=None if reference is None else dict(seed=reference[0]["seed"], instruction=reference[0]["instruction"], saved_motion=reference[1]["saved_motion"]))
                if reference:
                    env._rt_episode_info = reference[1]["episode_info"]
                    if reference[0]["arm_tag"]:
                        from envs.utils.action import ArmTag
                        env.arm_tag = ArmTag(reference[0]["arm_tag"])
                descriptor = {**current, "seed": int(prepared["seed"]), "instruction": prepared["instruction"],
                    "velocity_m_per_s": prepared["velocity"], "initial_target_position": np.asarray(env._eval_target_actor.actor.get_pose().p),
                    "initial_state": env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
                    "stop_on_contact": prepared["stop_on_contact"], "check_z_threshold": prepared["check_z_threshold"],
                    "arm_tag": str(getattr(env, "arm_tag", "")), "physics_dt_s": float(env.scene.get_timestep()), "native_action_limit": native_limit}
                validate_descriptor(descriptor, args.task, episode, requested)
                if not math.isclose(descriptor["physics_dt_s"], .004, abs_tol=1e-8):
                    raise ValueError("unexpected physics timestep")
                if int(env.step_lim) != native_limit:
                    raise ValueError("native action limit changed")
                if reference:
                    for key in ("seed", "instruction", "arm_tag", "stop_on_contact", "check_z_threshold", "native_action_limit"):
                        if descriptor[key] != reference[0][key]:
                            raise ValueError(f"frozen scene field changed: {key}")
                    for key in ("initial_target_position", "initial_state", "velocity_m_per_s", "physics_dt_s"):
                        if not np.allclose(descriptor[key], reference[0][key], atol=1e-8, rtol=0):
                            raise ValueError(f"frozen scene numeric field changed: {key}")
                    motion_bytes = reference[2]
                else:
                    motion_bytes = pickle.dumps(dict(seed=descriptor["seed"], saved_motion=env._saved_dynamic_motion_info,
                                                    episode_info=getattr(env, "_rt_episode_info", None)))
                save(directory / "manifest.json", descriptor)
                (directory / "expert_motion.pkl").write_bytes(motion_bytes)
                manifests.append(descriptor); save(output / "accepted_manifest.json", manifests)
                requested = descriptor["seed"] + 1
                if args.prepare_only:
                    save(directory / "prepared.json", {"manifest_sha256": sha(directory / "manifest.json"), "motion_sha256": sha(directory / "expert_motion.pkl")})
                    completed.append(descriptor)
                else:
                    from dynamicwam.runtime.domino_continuous.camera_stream import PhysicalTimeHeadHistory
                    from dynamicwam.runtime.domino_continuous.continuous_session import ContinuousSyncSession
                    from dynamicwam.runtime.domino_continuous.continuous_rtc import run_continuous_rtc_episode
                    from dynamicwam.runtime.domino_continuous.worker import ProcessInferenceWorker
                    if replay_scope is not None:
                        from dynamicwam.runtime.domino_continuous.development_replay import DevelopmentReplayRecorder,select_development_scene
                        frozen_scene = args.replay_from / args.task / f"episode_{episode:04d}"
                        selected = select_development_scene(replay_scope,task=args.task,seed=descriptor["seed"],
                            manifest_sha256=sha(frozen_scene / "manifest.json"),motion_sha256=sha(frozen_scene / "expert_motion.pkl"))
                        replay_recorder = DevelopmentReplayRecorder(
                            args.development_replay_dir.resolve() / args.task / f"episode_{episode:04d}",
                            scope=replay_scope,scene=selected,
                            trajectory_id=hashlib.sha256(str(directory.resolve()).encode()).hexdigest(),
                            episode_id=f"{args.task}:{descriptor['seed']}",
                            policy_checkpoint_sha256=checkpoint_sha,policy_source_sha256=source["source_digest"])
                    if worker is None:
                        worker = ProcessInferenceWorker(project_root=str(ROOT), config_path=str(args.config.resolve()),
                            checkpoint=str(args.checkpoint.resolve()), infer_index=0, policy_mode="motionkv_rtc", native_cuda_graph=True)
                    instruction_warmup = worker.prepare_instruction(prepared["instruction"])
                    session = ContinuousSyncSession(env, flow=PhysicalTimeHeadHistory(inference["head_flow"],
                        frame_period_s=args.history_frame_period_s, endpoint_tolerance_s=args.history_frame_period_s / 2),
                        action_interval_s=args.action_interval_s, save_video=True, timing_mode="nominal_training_timing",
                        replay_recorder=replay_recorder,
                        observation_config=dict(inference["observation"]), stop_on_contact=prepared["stop_on_contact"],
                        check_z_threshold=prepared["check_z_threshold"], lift_actor=prepared["lift_actor"],
                        start_z=prepared["start_z"], instruction=prepared["instruction"])
                    from eval_metrics import EvalMetricsTracker
                    session.metrics_tracker = EvalMetricsTracker(env, {"use_dynamic": True})
                    session.metrics_tracker.on_episode_start()
                    initial_clock = session.physics_time()
                    camera_reference = env.get_obs()
                    camera_equivalence = session.validate_camera_reader(camera_reference)
                    save(directory / "camera_equivalence.json", camera_equivalence)
                    del camera_reference
                    if session.physics_time() != initial_clock:
                        raise RuntimeError("camera warmup advanced physics")
                    ready_barrier = cohort.wait(episode, "ready")
                    # Desynchronize live episode clocks across cohort GPUs so
                    # age-0 refresh@16 walls do not align. Spread starts evenly
                    # across one world-refresh period (16 * action_interval):
                    #   stagger = index * (16 * dt) / N_members
                    # With N=8 and dt=0.16s => 0,0.32,...,2.24s. Sleep is BEFORE
                    # wall0/phys0; RTC H/K/R/refresh period and physics unchanged.
                    members = list(cohort.members)
                    live_start_stagger_index = members.index(args.task)
                    refresh_period_s = float(args.action_interval_s) * 16 / args.simulation_rate
                    live_start_stagger_s = (
                        live_start_stagger_index * (refresh_period_s / max(1, len(members)))
                    )
                    if live_start_stagger_s > 0:
                        time.sleep(live_start_stagger_s)
                    report = run_continuous_rtc_episode(session, worker, task=args.task, episode_seed=descriptor["seed"],
                        action_limit=action_limit, sink=events.append, max_backlog_s=args.max_backlog_s,
                        infer_clock_tolerance_s=args.infer_clock_tolerance_s, frame_period_s=args.history_frame_period_s,
                        action_interval_s=args.action_interval_s, simulation_rate=args.simulation_rate,
                        joint_v3=args.joint_v3,replay_recorder=replay_recorder)
                    if report.get("protocol") != protocol or not valid_clock_identity(report):
                        raise RuntimeError("RTC rollout failed its protocol/clock gates")
                    report["post_terminal_camera_drain_wall_s"] = close_episode_camera(session)
                    report["post_terminal_inference_drain_wall_s"] = drain_finished_episode(worker)
                    # Fast idle is not slow idle in joint-v3. Quiesce the source
                    # producer before any cohort member may initialize a scene.
                    if args.joint_v3:
                        drain_started = time.perf_counter()
                        drain_ack = worker.finish_episode(timeout_s=120.)
                        report["post_terminal_policy_drain_wall_s"] = time.perf_counter() - drain_started
                        report["post_terminal_policy_drain"] = drain_ack["episode_drain"]
                    finished_barrier = cohort.wait(episode, "finished")
                    report["camera_equivalence"] = camera_equivalence
                    report["cohort_scheduling"] = dict(mode=COHORT_MODE, ready=ready_barrier, finished=finished_barrier,
                        manifest_sha256=spec["cohort_manifest_sha256"], opencv_threads=cv2.getNumThreads(),
                        live_start_stagger_s=live_start_stagger_s, live_start_stagger_index=live_start_stagger_index,
                        live_start_refresh_period_s=refresh_period_s, live_start_cohort_size=len(members),
                        live_start_stagger_policy="cohort_task_index * (16*action_interval_s/simulation_rate)/N_members  # desync refresh@16")
                    report.update({**descriptor, **source, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
                        "frozen_manifest_sha256": frozen_sha, "video_required": True, "probe_action_limit": args.probe_action_limit,
                        "worker_startup_metadata": worker.startup_metadata, "instruction_warmup": instruction_warmup,
                        "timing_contract_sha256": timing_sha,
                        "scene_compatibility_sha256": sha(args.scene_compatibility) if args.scene_compatibility else None,
                        "native_result_dir": str(native_dir), "action_interval_s": args.action_interval_s,
                        "history_frame_period_s": args.history_frame_period_s})
                    video = session.head_video.finalize(native_dir / f"episode{episode}.mp4", episode_end_physics_s=session.physics_time())
                    report.update(video); report.update(publish_episode_video_links(directory, video))
                    tracker = session.metrics_tracker
                    if report["stop_reason"] == "out_of_bounds":
                        tracker.record_out_of_bounds()
                    metric = tracker.get_episode_metrics(bool(report["success"]), None if report["stop_reason"] in {"success", "action_budget_exhausted"} else report["stop_reason"], seed=descriptor["seed"])
                    save(directory / "metrics.json", asdict(metric) if is_dataclass(metric) else vars(metric))
                    save(directory / "report.json", report)
                    checked_episode(directory, args.task, episode, args.simulation_rate, expected_protocol=protocol)
                    if replay_recorder is not None:
                        replay_recorder.finalize(success=report["success"],complete=True)
                        report["development_replay_manifest"] = str(replay_recorder.directory / "manifest.json")
                        save(directory / "report.json",report)
                    completed.append(dict(seed=descriptor["seed"], success=report["success"]))
                    publish_native_task_results(output, native_dir, count=len(completed), planned=args.episodes,
                        stamp=args.result_stamp, instruction_type=str(benchmark["instruction_type"]), checkpoint=args.checkpoint,
                        generation=source["source_generation"], protocol=protocol, timing_mode="nominal_training_timing")
                save(output / "progress.json", dict(planned_episodes=args.episodes, completed_episodes=len(completed), records=completed))
                print("MOTIONKV_EPISODE " + json.dumps(dict(task=args.task, episode=episode, seed=descriptor["seed"], prepare_only=args.prepare_only)), flush=True)
            finally:
                original_error = sys.exc_info()[1]
                if replay_recorder is not None and not replay_recorder.closed:
                    try:
                        replay_recorder.finalize(success=False,complete=False,
                            error=None if original_error is None else f"{type(original_error).__name__}: {original_error}")
                    except BaseException:
                        traceback.print_exc()
                if original_error is not None:
                    if cohort is not None:
                        cohort.abort(f"{args.task} episode {episode}: {type(original_error).__name__}: {original_error}")
                    preserve_failed_video(session, directory, episode, original_error)
                    try:
                        cleanup_episode(session, env, directory, events)
                    except BaseException as cleanup_error:
                        # Preserve the causative failure while recording a
                        # separate cleanup failure for supervisor diagnosis.
                        try:
                            save(directory / "cleanup_error.json", dict(error=f"{type(cleanup_error).__name__}: {cleanup_error}"))
                        except BaseException:
                            traceback.print_exc()
                else:
                    cleanup_episode(session, env, directory, events)
        if native_dir is not None and completed:
            # A crash may have interrupted publication after committing all
            # episodes. Rebuild only derived native aggregate files on resume.
            publish_native_task_results(output, native_dir, count=len(completed), planned=args.episodes,
                stamp=args.result_stamp, instruction_type=str(benchmark["instruction_type"]), checkpoint=args.checkpoint,
                generation=source["source_generation"], protocol=protocol, timing_mode="nominal_training_timing")
        save(output / "completed.json", dict(protocol=spec["protocol"], episodes=len(completed), probe_action_limit=args.probe_action_limit))
        return 0
    except BaseException as exc:
        save(output / "error.json", {**current, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
             "completed_episodes": len(completed), "remaining_not_replaced": True, "diagnostics": getattr(exc, "diagnostics", None)})
        traceback.print_exc()
        return 1
    finally:
        # Ignore repeated termination while releasing only this task's owners.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if worker is not None:
            worker.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
