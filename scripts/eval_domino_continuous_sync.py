#!/usr/bin/env python3
"""DOMINO Level-1: real wall-clock world and full-chunk continuous synchronous policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_world_actual_latency_sync16_continuous_track_v2"
PROTOCOL_NOMINAL_TIMING = "continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1"
TIMING_NATIVE_MPLIB = "native_mplib_limits"
TIMING_NOMINAL_TRAINING = "nominal_training_timing"
PROTOCOL_BY_TIMING = {
    TIMING_NATIVE_MPLIB: PROTOCOL,
    TIMING_NOMINAL_TRAINING: PROTOCOL_NOMINAL_TIMING,
}
for entry in (ROOT / "external/flash_attn_runtime", ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        import base64
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    raise TypeError(type(value).__name__)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False) + "\n")
    temporary.replace(path)


def is_retryable_clock_error(error):
    message = str(error.get("error", ""))
    return message.startswith(("RealtimeViolation: wall/physics backlog ",
                               "RealtimeViolation: inference clock mismatch:"))



def publish_episode_video_links(episode_dir, video_info):
    """Hard-link finalized native MP4/sidecar into the episode folder for discovery.

    Native DOMINO layout keeps videos under eval_result/.../episodeN.mp4. Task
    episode dirs historically only held JSON/pkl, so a shallow search of the run
    root found zero MP4s even when --save-native-results succeeded. Linking the
    committed artifacts next to report.json makes them visible without changing
    the canonical native path used by validation and DOMINO metrics publication.
    """
    if not video_info:
        return {}
    video = Path(video_info["video"])
    sidecar = Path(video_info["video_timestamps"])
    local_video = episode_dir / video.name
    local_sidecar = episode_dir / sidecar.name
    published = {}
    for src, dst in ((video, local_video), (sidecar, local_sidecar)):
        if not src.is_file():
            raise FileNotFoundError(f"native video artifact missing before episode link: {src}")
        if dst.exists():
            if dst.samefile(src):
                published[str(dst)] = "already_linked"
                continue
            raise FileExistsError(f"refusing to replace existing episode video artifact: {dst}")
        try:
            os.link(src, dst)
            published[str(dst)] = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            published[str(dst)] = "copy"
    return {"episode_video": str(local_video), "episode_video_timestamps": str(local_sidecar),
            "episode_video_link": published}


def archive_file(path, archive_root):
    if path.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        path.rename(archive_root / f"{time.time_ns()}_{path.name}")


def source_provenance(generation=None, semantics_id=PROTOCOL, *, config_path=None, timing_contract_path=None):
    relative = ["harness.md", "AGENTS.md", "scripts/harness_preflight.py", "scripts/run_route1_smoke.sh",
                "src/dynamicwam/runtime/domino_continuous/gpu_binding.py",
                "src/dynamicwam/runtime/domino_continuous/sapien_init_lock.py",
                "src/dynamicwam/runtime/domino_continuous/continuous_track_v2.py","src/dynamicwam/runtime/domino_continuous/continuous_sync.py",
                "src/dynamicwam/runtime/domino_continuous/continuous_session.py",
                "src/dynamicwam/runtime/domino_continuous/continuous_track.py",
                "src/dynamicwam/runtime/domino_continuous/camera_stream.py","scripts/eval_domino_continuous_sync.py", "scripts/eval_domino_continuous_sync_suite.py",
                "scripts/video_artifacts.py", "scripts/summarize_domino_continuous_sync.py",
                "src/dynamicwam/runtime/domino_continuous/wall_sync.py", "src/dynamicwam/runtime/domino_continuous/worker.py",
                "src/dynamicwam/runtime/domino_continuous/domino_session.py", "src/dynamicwam/runtime/domino_continuous/segment.py",
                "src/dynamicwam/runtime/domino_continuous/motion.py", "src/dynamicwam/runtime/ciwam/flow.py",
                "src/dynamicwam/runtime/domino_continuous/camera_capture.py",
                "src/dynamicwam/runtime/domino_continuous/_sapien_color_reader.build.json",
                "src/dynamicwam/runtime/ciwam/adapters/domino/composite.py",
                "external/DOMINO/envs/_base_task.py", "external/DOMINO/envs/camera.py",
                "external/DOMINO/envs/camera/camera.py", "external/DOMINO/script/eval_metrics.py"]
    paths = {name: ROOT / name for name in relative if (ROOT / name).is_file()}
    for path in (ROOT / "src/dynamicwam/runtime/domino_continuous").glob("_sapien_color_reader*.so"):
        paths[str(path.relative_to(ROOT))] = path
    for path in (ROOT / "src/dynamicwam/runtime/domino_continuous/native_camera").rglob("*"):
        if path.is_file() and path.suffix in {".cpp", ".py", ".json"}:
            paths[str(path.relative_to(ROOT))] = path
    for subtree in ("src/dynamicwam/runtime/ciwam", "src/dynamicwam/models",
                    "src/dynamicwam/inference", "src/dynamicwam/config", "src/dynamicwam/vendor",
                    "external/DOMINO/envs/robot", "external/DOMINO/envs/camera", "external/DOMINO/envs/utils",
                    "external/DOMINO/description/utils"):
        for path in (ROOT / subtree).rglob("*.py"):
            paths[str(path.relative_to(ROOT)).replace("\\", "/")] = path
    for path in (ROOT / "configs").rglob("*"):
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}:
            paths[str(path.relative_to(ROOT)).replace("\\", "/")] = path
    if config_path is not None:
        import yaml
        selected = Path(config_path).resolve()
        config = yaml.safe_load(selected.read_text())
        paths["selected_profile.yaml"] = selected
        benchmark = config["benchmark"]
        domino = (ROOT / benchmark["domino_root"]).resolve()
        for task in benchmark["tasks"]:
            paths[f"selected_domino/envs/{task}.py"] = domino / "envs" / f"{task}.py"
        for name in (benchmark["task_config"], "_eval_step_limit", "_embodiment_config", "_camera_config"):
            paths[f"selected_domino/task_config/{name}.yml"] = domino / "task_config" / f"{name}.yml"
    if timing_contract_path is not None:
        paths["timing_contract.json"] = Path(timing_contract_path).resolve()
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(paths.items())}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    label = generation or f"code_{digest[:16]}"
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in label):
        raise ValueError("source generation must be a simple nonempty filename label")
    return {"source_generation": label, "semantics_id": semantics_id,
            "source_sha256": hashes, "source_digest": digest, "code_root": str(ROOT)}


def validate_resume_spec(old, new, keys):
    for key in keys:
        if old.get(key) != new.get(key):
            raise ValueError(f"resume setting mismatch for {key}: {old.get(key)!r} != {new.get(key)!r}")
    if old.get("semantics_id", PROTOCOL) != new.get("semantics_id", PROTOCOL):
        raise ValueError("changed semantics require a separate output directory; no mixed aggregate")
    if old.get("source_generation") != new.get("source_generation") or old.get("source_digest") != new.get("source_digest"):
        raise ValueError("resume cannot pool different source generations; use a new run with --replay-from")


def read_valid_episode(directory, task, episode, *, probe=None):
    report_path, metric_path, manifest_path = directory / "report.json", directory / "metrics.json", directory / "manifest.json"
    if not report_path.exists() or not metric_path.exists() or not manifest_path.exists():
        return None
    report, metrics = json.loads(report_path.read_text()), json.loads(metric_path.read_text())
    descriptor = json.loads(manifest_path.read_text())
    if (report.get("protocol") not in PROTOCOL_BY_TIMING.values() or report.get("realtime_valid") is not True
            or report.get("task") != task or report.get("episode_index") != episode
            or report.get("probe_action_limit") != probe):
        return None
    if any(report.get(key) != descriptor.get(key) for key in ("task", "episode_index", "seed", "requested_seed")):
        return None
    score = float(metrics["manipulation_score"])
    seed = int(report["seed"])
    if (not math.isfinite(score) or not 0 <= score <= 100 or metrics.get("seed") != seed
            or bool(metrics["success"]) != bool(report["success"])):
        return None
    if report.get("video_required"):
        from video_artifacts import validate_video_artifacts
        if not validate_video_artifacts(report):
            return None
    return {"task": task, "episode_index": episode, "seed": seed,
            "requested_seed": report.get("requested_seed"), "success": bool(report["success"]),
            "stop_reason": report["stop_reason"], "realtime_valid": True,
            "manipulation_score": score, "source_generation": report.get("source_generation", "initial"),
            "source_digest": report.get("source_digest"),
            "semantics_id": report.get("semantics_id", PROTOCOL)}


def scan_completed_prefix(output, task, count, *, probe=None):
    rows = []
    missing_seen = False
    seeds = set()
    started_path = output / "task_started.json"
    next_seed = int(json.loads(started_path.read_text())["start_seed"]) if started_path.exists() else None
    for episode in range(count):
        row = read_valid_episode(output / f"episode_{episode:04d}", task, episode, probe=probe)
        if row is None:
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError("valid episodes after an incomplete slot; refusing ambiguous resume")
        if row["seed"] in seeds:
            raise RuntimeError("duplicate accepted seed in completed prefix")
        if next_seed is not None and row["requested_seed"] != next_seed:
            raise RuntimeError("completed prefix requested-seed chain changed")
        if row["requested_seed"] is None or row["seed"] < row["requested_seed"]:
            raise RuntimeError("accepted seed precedes requested seed")
        next_seed = row["seed"] + 1
        seeds.add(row["seed"])
        rows.append(row)
    return rows


def validate_accepted_descriptor(descriptor, *, task, episode, next_seed):
    if descriptor.get("task") != task or descriptor.get("episode_index") != episode:
        raise RuntimeError("accepted descriptor task/index mismatch")
    if descriptor.get("requested_seed") != next_seed:
        raise RuntimeError("accepted descriptor requested_seed differs from expected seed chain")
    if int(descriptor["seed"]) < next_seed:
        raise RuntimeError("accepted seed precedes requested seed")


def replay_reference(output, episode, *, task, next_seed):
    directory = output / f"episode_{episode:04d}"
    candidates = [directory]
    archive = output / "attempts" / f"episode_{episode:04d}"
    if archive.exists():
        candidates.extend(sorted(archive.iterdir(), reverse=True))
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        motion_path = candidate / "expert_motion.pkl"
        if manifest_path.exists() and motion_path.exists():
            descriptor = json.loads(manifest_path.read_text())
            raw = motion_path.read_bytes()
            saved = pickle.loads(raw)
            validate_accepted_descriptor(descriptor, task=task, episode=episode, next_seed=next_seed)
            if int(saved["seed"]) != int(descriptor["seed"]) or descriptor["episode_index"] != episode:
                raise RuntimeError("stored replay seed/index mismatch")
            return descriptor, saved, raw
    manifests = output / "accepted_manifest.json"
    if any((candidate / "expert_motion.pkl").exists() or (candidate / "manifest.json").exists()
           for candidate in candidates):
        raise RuntimeError("partial acceptance metadata without a complete replay pair; refusing planner resampling")
    if manifests.exists() and any(int(row["episode_index"]) == episode for row in json.loads(manifests.read_text())):
        raise RuntimeError("accepted episode has no complete saved motion; refusing planner resampling")
    return None


def archive_incomplete_episode(output, episode):
    directory = output / f"episode_{episode:04d}"
    if directory.exists():
        archive = output / "attempts" / directory.name / f"attempt_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def acquire_lock(path):
    import fcntl
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another evaluator owns {path}")
    return handle


def validate_replay_source(run, *, task, episodes, checkpoint, start_seed, config_sha256):
    suite = json.loads((run / "suite_started.json").read_text())
    spec = json.loads((run / task / "task_started.json").read_text())
    accepted_protocols = {PROTOCOL, PROTOCOL_NOMINAL_TIMING, "continuous_world_actual_latency_sync16_continuous_track_v1", "continuous_world_actual_latency_sync16_native_topp_v1"}
    if (suite.get("protocol") not in accepted_protocols or spec.get("protocol") not in accepted_protocols
            or suite.get("protocol") != spec.get("protocol")
            or suite.get("virtual_delay") is not False or spec.get("virtual_delay") is not False
            or task not in suite["tasks"] or spec.get("task") != task
            or int(spec["episodes"]) < episodes or spec.get("probe_action_limit") is not None
            or Path(spec["checkpoint"]).resolve() != checkpoint
            or spec.get("start_seed") != start_seed or suite.get("seed") != start_seed
            or spec.get("config_sha256") != config_sha256
            or ("config_sha256" in suite and suite["config_sha256"] != config_sha256)):
        raise ValueError("replay source is not a compatible accepted Level-1 full-checkpoint run")
    accepted_path = run / task / "accepted_manifest.json"
    if accepted_path.exists():
        next_seed = start_seed
        for index, descriptor in enumerate(json.loads(accepted_path.read_text())):
            validate_accepted_descriptor(descriptor, task=task, episode=index, next_seed=next_seed)
            next_seed = int(descriptor["seed"]) + 1


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/absolute_motion_v2.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-backlog-s", type=float, default=0.25)
    parser.add_argument("--infer-clock-tolerance-s", type=float, default=0.02)
    parser.add_argument("--timing-contract", type=Path, required=True)
    parser.add_argument("--action-interval-s", type=float, default=.16)
    parser.add_argument("--history-frame-period-s", type=float, default=.16)
    parser.add_argument("--history-endpoint-tolerance-s", type=float, default=.08)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-from", type=Path, default=None,
                        help="Read accepted episodes from a previous run into a NEW output generation.")
    parser.add_argument("--source-generation", default=None)
    parser.add_argument("--timing-mode", choices=sorted(PROTOCOL_BY_TIMING),
                        default=TIMING_NATIVE_MPLIB,
                        help="native_mplib_limits keeps MPLIB v/a stretch; "
                             "nominal_training_timing keeps 16*160ms with no RTC.")
    parser.add_argument("--semantics-id", default=None,
                        help="Defaults to the protocol id selected by --timing-mode.")
    parser.add_argument("--save-native-results", action="store_true",
                        help="Require per-episode MP4 and original DOMINO task-level result files.")
    parser.add_argument("--result-stamp", default=None)
    parser.add_argument("--probe-action-limit", type=int, default=None,
                        help="Smoke only: cap native action count; never a formal benchmark result.")
    args = parser.parse_args()
    from harness_preflight import require_ssd
    require_ssd(args.output_dir)
    require_ssd(args.checkpoint)
    args.protocol = PROTOCOL_BY_TIMING[args.timing_mode]
    if args.semantics_id is None:
        args.semantics_id = args.protocol
    return args


def drain_finished_episode(worker, timeout_s=120.0):
    """Discard a terminal episode's pending result outside the timed rollout."""
    started = time.perf_counter()
    while worker.busy:
        worker.poll()
        if not worker.busy:
            break
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError("terminal episode inference did not drain within 120 seconds")
        time.sleep(0.002)
    return time.perf_counter() - started


def close_episode_camera(session):
    """Join terminal readback outside rollout timing, before accepting its result."""
    started = time.perf_counter()
    try:
        # Preserve a snapshot submitted before a terminal event, without pushing
        # it back into policy history or submitting any new render.
        if getattr(session, "head_video", None) is not None and session.camera_reader.busy:
            captured = session.camera_reader.drain()
            session.record_video_capture(captured)
            session.pending_capture = None
    finally:
        session.camera_reader.close()
    return time.perf_counter() - started


def cleanup_episode(session, env, episode_dir, events):
    """Persist diagnostics and release the scene even when readback/cleanup fails.

    AsyncCameraCapture.close joins its executor in its own finally block, so
    scene teardown remains ordered after the native reader on failure too.
    """
    try:
        if session is not None:
            close_episode_camera(session)
    finally:
        try:
            with (episode_dir / "requests.jsonl").open("w") as handle:
                for event in events:
                    handle.write(json.dumps(event, default=plain, allow_nan=False) + "\n")
        finally:
            if env is not None:
                try:
                    env.close_env()
                finally:
                    env.release_episode_resources()


def publish_native_task_results(output, native_dir, *, count, planned, stamp,
                                instruction_type, checkpoint, generation,
                                protocol, timing_mode):
    """Rebuild derived files from committed episodes, including after a crash."""
    if native_dir is None or count == 0:
        return
    from video_artifacts import aggregate_native_metrics, write_domino_results
    all_metrics = [json.loads((output / f"episode_{index:04d}" / "metrics.json").read_text())
                   for index in range(count)]
    write_domino_results(native_dir, stamp, instruction_type, aggregate_native_metrics(all_metrics))
    save(native_dir / "_evaluation_status.json", {
        "protocol": protocol, "timing_mode": timing_mode, "planned_episodes": planned,
        "completed_episodes": count, "complete": count == planned,
        "checkpoint": str(checkpoint), "source_generation": generation,
        "video_sampling": "native_head_observations_10fps; not continuous physical-time recording"})


def main():
    args = parse()
    import numpy as np
    import yaml
    from dynamicwam.config.loader import load_profile
    from dynamicwam.runtime.ciwam.adapters.domino.composite import head_camera_frame
    from dynamicwam.runtime.domino_continuous.camera_stream import PhysicalTimeHeadHistory
    from dynamicwam.runtime.domino_continuous.continuous_session import ContinuousSyncSession
    from dynamicwam.runtime.domino_continuous.domino_session import DominoLevel1Session, prepare_level1_env
    from dynamicwam.runtime.domino_continuous.continuous_sync import run_continuous_sync_episode
    from dynamicwam.runtime.domino_continuous.worker import ProcessInferenceWorker
    from dynamicwam.runtime.domino_continuous.camera_capture import AsyncCameraCapture
    import sapien  # Bootstrap this wheel's transitive native library paths first.
    from dynamicwam.runtime.domino_continuous.gpu_binding import install_gpu_binding
    gpu_binding = install_gpu_binding()
    if gpu_binding is None:
        raise RuntimeError("continuous evaluator requires a pinned GPU PCI identity")
    from dynamicwam.runtime.domino_continuous import _sapien_color_reader

    if args.episodes < 1 or args.seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.max_backlog_s, args.infer_clock_tolerance_s, args.action_interval_s,
            args.history_frame_period_s, args.history_endpoint_tolerance_s)):
        raise ValueError("clock tolerances must be positive and finite")
    if args.probe_action_limit is not None and args.probe_action_limit < 1:
        raise ValueError("probe action limit must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != "DynamicWAM_full.pt" or not checkpoint.is_file():
        raise ValueError("this protocol requires the official DynamicWAM_full.pt checkpoint")
    timing_path = args.timing_contract.expanduser().resolve()
    timing = json.loads(timing_path.read_text())
    recommended = timing["recommended_continuous_sync_contract"]
    for actual, key in ((args.action_interval_s, "nominal_action_interval_seconds"),
                        (args.history_frame_period_s, "nominal_history_frame_interval_seconds")):
        if not math.isclose(actual, recommended[key], abs_tol=1e-12, rel_tol=0):
            raise ValueError("time parameters differ from the frozen checkpoint calibration")
    if not math.isclose(args.history_endpoint_tolerance_s, args.history_frame_period_s/2,
                        abs_tol=1e-12, rel_tol=0):
        raise ValueError("endpoint tolerance must be half the history sample period")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    checkpoint_sha256 = digest.hexdigest()
    if checkpoint_sha256 != timing["checkpoint"]["sha256"]:
        raise ValueError("checkpoint does not match the audited official full artifact")
    timing_metadata = {"timing_contract": str(timing_path),
        "timing_contract_sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        "action_interval_s": args.action_interval_s,
        "history_frame_period_s": args.history_frame_period_s,
        "history_endpoint_tolerance_s": args.history_endpoint_tolerance_s,
        "controller": "continuous_shape_preserving_bounded_joint_track",
        "policy": "dynamicwam_continuous_sync", "checkpoint_sha256": checkpoint_sha256}
    profile = load_profile(args.config.resolve())
    inference = profile.inference_config()
    if int(inference["head_flow"].get("policy_stride", 4)) != 4:
        raise ValueError("timing calibration requires the original four-frame flow stride")
    benchmark = profile.benchmark_config()
    if args.task not in benchmark["tasks"]:
        raise ValueError("task is outside the configured 35-task benchmark")
    domino = (ROOT / benchmark["domino_root"]).resolve()
    if args.save_native_results:
        sys.path.insert(0, str(domino / "script"))
    task_config = yaml.safe_load((domino / "task_config" / f"{benchmark['task_config']}.yml").read_text())
    if not task_config.get("use_dynamic") or int(task_config.get("dynamic_level", 1)) != 1:
        raise ValueError("only DOMINO dynamic Level-1 is supported")
    limits = yaml.safe_load((domino / "task_config/_eval_step_limit.yml").read_text())
    native_limit = int(limits[args.task])
    action_limit = native_limit if args.probe_action_limit is None else min(native_limit, args.probe_action_limit)
    output = args.output_dir.expanduser().resolve() / args.task
    output.mkdir(parents=True, exist_ok=True)
    if args.save_native_results and not args.result_stamp:
        raise ValueError("--save-native-results requires a stable --result-stamp for resume")
    if args.result_stamp and any(c in args.result_stamp for c in "/\\"):
        raise ValueError("result stamp must be a single directory name")
    native_dir = (output.parent / "eval_result" / args.task / "dynamicwam_continuous_sync"
                  / benchmark["task_config"] / "DynamicWAM_full" / args.result_stamp
                  if args.save_native_results else None)
    if native_dir is not None:
        native_dir.mkdir(parents=True, exist_ok=True)
    replay_run = args.replay_from.expanduser().resolve() if args.replay_from is not None else None
    if replay_run is not None:
        if replay_run == output.parent:
            raise ValueError("--replay-from must reference a different, unchanged output directory")
        validate_replay_source(replay_run, task=args.task, episodes=args.episodes, checkpoint=checkpoint,
                               start_seed=args.seed, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    task_lock = acquire_lock(output / ".evaluation.lock")
    provenance = source_provenance(args.source_generation, args.semantics_id, config_path=args.config,
                                   timing_contract_path=timing_path)
    spec = {"schema_version": 1, "protocol": args.protocol, "timing_mode": args.timing_mode, "pid": os.getpid(),
            "task": args.task, "episodes": args.episodes, "start_seed": args.seed,
            "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "code_root": str(ROOT), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "native_action_limit": native_limit, "action_limit": action_limit,
            "probe_action_limit": args.probe_action_limit,
            "max_backlog_s": args.max_backlog_s, "infer_clock_tolerance_s": args.infer_clock_tolerance_s,
            "startup_query_latency_included": True, "virtual_delay": False,
            "guidance": False, "prefetch": False, "drop_prefix": False, "video": args.save_native_results,
            "native_result_dir": str(native_dir) if native_dir is not None else None,
            "result_stamp": args.result_stamp,
            "replay_from": str(replay_run) if replay_run is not None else None, **timing_metadata, **provenance}
    started = output / "task_started.json"
    if not started.exists() and (list(output.glob("episode_*")) or (output / "accepted_manifest.json").exists()):
        raise RuntimeError("episode data exists without task settings; refusing unverifiable resume")
    if args.resume and started.exists():
        validate_resume_spec(json.loads(started.read_text()), spec,
            ("protocol", "task", "episodes", "start_seed", "checkpoint", "checkpoint_bytes", "config_sha256",
             "native_action_limit", "action_limit", "probe_action_limit", "max_backlog_s", "infer_clock_tolerance_s",
             "startup_query_latency_included", "virtual_delay", "guidance", "prefetch", "drop_prefix", "video", "replay_from",
             "native_result_dir", "result_stamp", "action_interval_s", "history_frame_period_s",
             "history_endpoint_tolerance_s", "timing_contract", "timing_contract_sha256", "checkpoint_sha256"))
        old_error = output / "error.json"
        if old_error.exists() and not is_retryable_clock_error(json.loads(old_error.read_text())):
            raise RuntimeError("resume refuses an unknown/non-clock task error; inspect before retrying")
        completed = scan_completed_prefix(output, args.task, args.episodes, probe=args.probe_action_limit)
        if any(row["semantics_id"] != args.semantics_id for row in completed):
            raise RuntimeError("completed results have different semantics; use a new output directory")
        if any(row["source_generation"] != provenance["source_generation"] or row["source_digest"] != provenance["source_digest"]
               for row in completed):
            raise RuntimeError("completed results have a different source generation/digest; use --replay-from")
        archive_file(old_error, output / "error_history")
        save(output / "resumes" / f"{time.time_ns()}.json", spec)
    else:
        with started.open("x") as handle:
            json.dump(spec, handle, indent=2)
        completed = []
    generation_path = output / "source_generations" / f"{provenance['source_generation']}.json"
    if generation_path.exists() and json.loads(generation_path.read_text()) != provenance:
        raise RuntimeError("source generation label reused for different source; choose a new label")
    save(generation_path, provenance)

    def publish_native_results():
        publish_native_task_results(output, native_dir, count=len(completed), planned=args.episodes,
            stamp=args.result_stamp, instruction_type=str(benchmark["instruction_type"]),
            checkpoint=checkpoint, generation=provenance["source_generation"],
            protocol=args.protocol, timing_mode=args.timing_mode)

    worker = None
    next_seed = int(completed[-1]["seed"]) + 1 if completed else args.seed
    manifests_path = output / "accepted_manifest.json"
    manifests = json.loads(manifests_path.read_text()) if manifests_path.exists() else []
    current = {"task": args.task, "episode_index": 0, "requested_seed": args.seed}
    try:
        if len(completed) < args.episodes:
            worker = ProcessInferenceWorker(project_root=str(ROOT), config_path=str(args.config.resolve()),
                                            checkpoint=str(checkpoint), infer_index=0)
        for episode in range(len(completed), args.episodes):
            env = None
            session = None
            current = {"task": args.task, "episode_index": episode, "requested_seed": next_seed}
            reference = replay_reference(output, episode, task=args.task, next_seed=next_seed) if args.resume else None
            reference_origin = "same_run_retry" if reference is not None else None
            if reference is None and replay_run is not None:
                reference = replay_reference(replay_run / args.task, episode, task=args.task, next_seed=next_seed)
                reference_origin = str(replay_run) if reference is not None else None
            if reference is not None:
                validate_accepted_descriptor(reference[0], task=args.task, episode=episode, next_seed=next_seed)
            episode_dir = archive_incomplete_episode(output, episode)
            if native_dir is not None:
                # Only the incomplete slot is archived; valid videos are never overwritten.
                for suffix in (".mp4", "_timestamps.json"):
                    archive_file(native_dir / f"episode{episode}{suffix}",
                                 output / "artifact_attempts" / f"episode_{episode:04d}")
            if reference is not None:
                save(episode_dir / "manifest.json", reference[0])
                (episode_dir / "expert_motion.pkl").write_bytes(reference[2])
            save(episode_dir / "attempt.json", {**current, **provenance,
                                                "replay_accepted_episode": reference is not None,
                                                "reference_origin": reference_origin})
            events = []
            try:
                print("CONTINUOUS_SYNC_PREPARE " + json.dumps(current), flush=True)
                env, prepared = prepare_level1_env(
                    domino, task_name=args.task, task_config=benchmark["task_config"],
                    start_seed=next_seed if reference is None else int(reference[0]["seed"]),
                    instruction_type=str(benchmark["instruction_type"]),
                    replay=None if reference is None else {"seed": int(reference[0]["seed"]),
                        "instruction": reference[0]["instruction"], "saved_motion": reference[1]["saved_motion"]})
                if reference is not None:
                    env._rt_episode_info = reference[1]["episode_info"]
                    if reference[0]["arm_tag"]:
                        from envs.utils.action import ArmTag
                        env.arm_tag = ArmTag(reference[0]["arm_tag"])
                accepted_seed = int(prepared["seed"])
                if accepted_seed < current["requested_seed"]:
                    raise RuntimeError("environment accepted a seed before the requested seed")
                current["seed"] = accepted_seed
                next_seed = accepted_seed + 1
                if args.task in {"put_object_cabinet", "place_object_scale"}:
                    if str(getattr(env, "arm_tag", "")) not in {"left", "right"}:
                        raise RuntimeError("expert-initialized arm_tag missing after policy reset")
                if int(env.step_lim) != native_limit:
                    raise RuntimeError("prepared environment action limit differs from official task limit")
                session = ContinuousSyncSession(
                    env, flow=PhysicalTimeHeadHistory(inference["head_flow"],
                        frame_period_s=args.history_frame_period_s,
                        endpoint_tolerance_s=args.history_endpoint_tolerance_s),
                    action_interval_s=args.action_interval_s, save_video=args.save_native_results,
                    timing_mode=args.timing_mode,
                    observation_config=dict(inference["observation"]),
                    stop_on_contact=prepared["stop_on_contact"],
                    check_z_threshold=prepared["check_z_threshold"],
                    lift_actor=prepared["lift_actor"], start_z=prepared["start_z"],
                    instruction=prepared["instruction"])
                if not np.isclose(session.timestep(), 1 / 250, atol=1e-8, rtol=0):
                    raise RuntimeError(f"unexpected physics timestep: {session.timestep()}")
                descriptor = {**current, "instruction": prepared["instruction"],
                              "velocity_m_per_s": prepared["velocity"],
                              "initial_target_position": session.object_position(),
                              "initial_state": env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
                              "stop_on_contact": prepared["stop_on_contact"],
                              "check_z_threshold": prepared["check_z_threshold"],
                              "arm_tag": str(getattr(env, "arm_tag", "")),
                              "physics_dt_s": session.timestep(), "native_action_limit": native_limit}
                if reference is not None:
                    for key in ("seed", "instruction", "arm_tag", "stop_on_contact", "check_z_threshold", "native_action_limit"):
                        if descriptor[key] != reference[0][key]:
                            raise RuntimeError(f"replayed accepted scene differs: {key}")
                    for key in ("initial_target_position", "initial_state", "velocity_m_per_s", "physics_dt_s"):
                        if not np.allclose(descriptor[key], reference[0][key], atol=1e-8, rtol=0):
                            raise RuntimeError(f"replayed accepted numeric state differs: {key}")
                else:
                    # Save acceptance before policy execution; a retry never reruns its planner.
                    with (episode_dir / "expert_motion.pkl").open("xb") as handle:
                        pickle.dump({"seed": accepted_seed, "saved_motion": env._saved_dynamic_motion_info,
                                     "episode_info": getattr(env, "_rt_episode_info", None)}, handle)
                    save(episode_dir / "manifest.json", descriptor)
                manifests = [row for row in manifests if int(row["episode_index"]) != episode]
                manifests.append(descriptor)
                manifests.sort(key=lambda row: int(row["episode_index"]))
                save(output / "accepted_manifest.json", manifests)
                from eval_metrics import EvalMetricsTracker
                session.metrics_tracker = EvalMetricsTracker(env, {"use_dynamic": True})
                session.metrics_tracker.on_episode_start()
                # Warm scene/camera allocations before time zero. No flow history push.
                initial_clock = session.physics_time()
                env.get_obs()
                if session.physics_time() != initial_clock:
                    raise RuntimeError("camera warmup unexpectedly advanced physics")
                report = run_continuous_sync_episode(
                    session, worker, task=args.task, episode_seed=accepted_seed,
                    action_limit=action_limit, sink=events.append,
                    max_backlog_s=args.max_backlog_s,
                    infer_clock_tolerance_s=args.infer_clock_tolerance_s,
                    frame_period_s=args.history_frame_period_s)
                if report.get("protocol") != args.protocol:
                    raise RuntimeError("runtime protocol does not match the evaluator")
                # A terminal event can precede readback completion. A failed
                # readback must not leave a seemingly valid benchmark result.
                report["post_terminal_camera_drain_wall_s"] = close_episode_camera(session)
                report["post_terminal_inference_drain_wall_s"] = drain_finished_episode(worker)
                report.update({**current, "protocol": args.protocol, "timing_mode": args.timing_mode,
                               "limit_retiming": args.timing_mode == TIMING_NATIVE_MPLIB,
                               "instruction": prepared["instruction"],
                               "checkpoint": str(checkpoint), "probe_action_limit": args.probe_action_limit,
                               "native_action_limit": native_limit, "action_limit": action_limit,
                               "reference_origin": reference_origin, **timing_metadata, **provenance})
                if report.get("realtime_valid") is not True:
                    raise RuntimeError("rollout failed real-time infrastructure checks; not a policy result")
                report["video_required"] = args.save_native_results
                if native_dir is not None:
                    video_info = session.head_video.finalize(native_dir / f"episode{episode}.mp4",
                                  episode_end_physics_s=session.physics_time())
                    report.update(video_info)
                    report["native_result_dir"] = str(native_dir)
                    # Also expose MP4 beside report.json (hardlink/copy); canonical
                    # path remains under eval_result for DOMINO native layout.
                    report.update(publish_episode_video_links(episode_dir, video_info))
                save(episode_dir / "report.json", report)
                tracker = session.metrics_tracker
                if report["stop_reason"] == "out_of_bounds":
                    tracker.record_out_of_bounds()
                metric = tracker.get_episode_metrics(bool(report["success"]),
                    None if report["stop_reason"] in {"success", "action_budget_exhausted"}
                    else report["stop_reason"], seed=accepted_seed)
                metric_dict = asdict(metric) if is_dataclass(metric) else vars(metric)
                save(episode_dir / "metrics.json", metric_dict)
                stretches = [req.get("trajectory", {}).get("stretch") for req in report.get("requests", [])
                             if isinstance(req.get("trajectory"), dict)]
                stretches = [float(s) for s in stretches if isinstance(s, (int, float))]
                row = {**current, "success": bool(report["success"]),
                       "stop_reason": report["stop_reason"], "realtime_valid": True,
                       "manipulation_score": float(metric_dict["manipulation_score"]),
                       "source_generation": provenance["source_generation"], "semantics_id": args.semantics_id,
                       "protocol": args.protocol, "timing_mode": args.timing_mode,
                       "limit_retiming": args.timing_mode == TIMING_NATIVE_MPLIB,
                       "stretch_n": len(stretches),
                       "stretch_mean": (sum(stretches) / len(stretches)) if stretches else None,
                       "stretch_max": max(stretches) if stretches else None}
                completed.append(row)
                publish_native_results()
                save(output / "progress.json", {"planned_episodes": args.episodes,
                                                "completed_episodes": len(completed), "records": completed})
                print("CONTINUOUS_SYNC_RESULT " + json.dumps(row), flush=True)
            finally:
                # Disk I/O is kept outside the timed episode loop, including error paths.
                # A terminal contact/out-of-bounds may occur before readback
                # finishes. Retain every camera/scene until the worker drains.
                cleanup_episode(session, env, episode_dir, events)
        # Repair any interrupted native-file publication, even on a 20/20 resume.
        publish_native_results()
        save(output / "completed.json", {"protocol": args.protocol, "timing_mode": args.timing_mode,
                                          "episodes": len(completed),
                                          "probe_action_limit": args.probe_action_limit,
                                          "source_generations": sorted({row["source_generation"] for row in completed})})
        return 0
    except Exception as exc:
        traceback.print_exc()
        save(output / "error.json", {**current, "error": f"{type(exc).__name__}: {exc}",
                                     "completed_episodes": len(completed), "remaining_not_replaced": True,
                                     "traceback": traceback.format_exc(),
                                     "diagnostics": getattr(exc, "diagnostics", None), **provenance})
        return 1
    finally:
        if worker is not None:
            worker.close()
        task_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
