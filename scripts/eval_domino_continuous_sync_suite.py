#!/usr/bin/env python3
"""Eight-GPU Level-1 actual-clock continuous-track synchronous-policy benchmark."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import yaml
from eval_domino_continuous_sync import (acquire_lock, archive_file, is_retryable_clock_error,
                                  save, scan_completed_prefix, source_provenance, validate_resume_spec)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "continuous_world_actual_latency_sync16_continuous_track_v2"
PROTOCOL_NOMINAL_TIMING = "continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1"
TIMING_NATIVE_MPLIB = "native_mplib_limits"
TIMING_NOMINAL_TRAINING = "nominal_training_timing"
PROTOCOL_BY_TIMING = {
    TIMING_NATIVE_MPLIB: PROTOCOL,
    TIMING_NOMINAL_TRAINING: PROTOCOL_NOMINAL_TIMING,
}
CONTROLLER_BY_TIMING = {
    TIMING_NATIVE_MPLIB: "continuous_shape_preserving_bounded_joint_track",
    TIMING_NOMINAL_TRAINING: "continuous_shape_preserving_nominal_timing_track",
}


def load_retry_budget(path, tasks, limit, *, resume):
    if path.exists():
        state = json.loads(path.read_text())
        if not resume:
            raise RuntimeError("retry budget already exists; use --resume")
        if state.get("tasks") != tasks or state.get("max_infra_retries") != limit:
            raise ValueError("retry budget tasks/limit changed across resume")
        for task in tasks:
            row = state["per_task"][task]
            if (not isinstance(row.get("used"), int) or not 0 <= row["used"] <= limit
                    or not isinstance(row.get("reserved"), bool) or not isinstance(row.get("launches"), int)
                    or row["launches"] < 0):
                raise ValueError("invalid persisted retry budget")
        return state
    if resume:
        raise RuntimeError("resume has no persisted retry budget; refusing to reset its allowance")
    state = {"schema_version": 1, "tasks": tasks, "max_infra_retries": limit,
             "per_task": {task: {"used": 0, "reserved": False, "launches": 0} for task in tasks}, "history": []}
    save(path, state)
    return state


def reserve_retry(path, state, task, error):
    if not is_retryable_clock_error(error):
        return False
    row = state["per_task"][task]
    if row["reserved"]:
        return True  # A previously persisted queue reservation survives restart.
    if row["used"] >= state["max_infra_retries"]:
        return False
    row["used"] += 1
    row["reserved"] = True
    state["history"].append({"task": task, "event": "retry_reserved", "retry_number": row["used"],
                             "error": error.get("error"), "time_ns": time.time_ns()})
    save(path, state)  # Commit the bounded allowance before enqueue/launch.
    return True


def record_launch(path, state, task):
    row = state["per_task"][task]
    row["reserved"] = False
    row["launches"] += 1
    state["history"].append({"task": task, "event": "launch", "retry_number": row["used"],
                             "launch_number": row["launches"], "time_ns": time.time_ns()})
    save(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/absolute_motion_v2.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--tasks", default="", help="Optional task subset for smoke validation.")
    parser.add_argument("--max-backlog-s", type=float, default=0.25)
    parser.add_argument("--infer-clock-tolerance-s", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-from", type=Path, default=None)
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument("--source-generation", default=None)
    parser.add_argument("--timing-mode", choices=sorted(PROTOCOL_BY_TIMING),
                        default=TIMING_NATIVE_MPLIB,
                        help="native_mplib_limits vs nominal_training_timing ablation.")
    parser.add_argument("--semantics-id", default=None,
                        help="Defaults to the protocol id for --timing-mode.")
    parser.add_argument("--save-native-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-contract", type=Path, required=True)
    parser.add_argument("--action-interval-s", type=float, default=0.16)
    parser.add_argument("--history-frame-period-s", type=float, default=0.16)
    parser.add_argument("--history-endpoint-tolerance-s", type=float, default=0.08)
    parser.add_argument("--result-stamp", default=None)
    args = parser.parse_args()
    from harness_preflight import require_ssd
    require_ssd(args.output_dir)
    require_ssd(args.checkpoint)
    args.protocol = PROTOCOL_BY_TIMING[args.timing_mode]
    if args.semantics_id is None:
        args.semantics_id = args.protocol
    timing_path = args.timing_contract.expanduser().resolve()
    timing_bytes = timing_path.read_bytes()
    timing_contract = json.loads(timing_bytes)
    recommendation = timing_contract["recommended_continuous_sync_contract"]
    timing_sha = hashlib.sha256(timing_bytes).hexdigest()
    expected_timing = {
        "action_interval_s": recommendation["nominal_action_interval_seconds"],
        "history_frame_period_s": recommendation["nominal_history_frame_interval_seconds"],
        "history_endpoint_tolerance_s": recommendation["nominal_history_frame_interval_seconds"] / 2,
    }
    for key, expected in expected_timing.items():
        if not math.isfinite(getattr(args, key)) or abs(getattr(args, key) - expected) > 1e-12:
            raise ValueError(f"{key} must match the frozen training timing contract: {expected}")
    if args.max_backlog_s > 0.25 or args.infer_clock_tolerance_s > 0.02:
        raise ValueError("realtime quality gates cannot be relaxed above 250 ms backlog / 20 ms inference clock error")
    config = yaml.safe_load(args.config.read_text())
    configured = config["benchmark"]["tasks"]
    tasks = [v.strip() for v in args.tasks.split(",")] if args.tasks else list(configured)
    domino = (ROOT / config["benchmark"]["domino_root"]).resolve()
    limits = yaml.safe_load((domino / "task_config/_eval_step_limit.yml").read_text())
    if len(configured) != 35 or len(set(configured)) != 35:
        raise ValueError("expected the configured full 35-task benchmark")
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in configured or t not in limits for t in tasks):
        raise ValueError("task subset must be unique members of the configured Level-1 suite")
    if len(tasks) == 35 and not args.save_native_results:
        raise ValueError("the full benchmark requires native video/results artifacts")
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.name != "DynamicWAM_full.pt" or not checkpoint.is_file():
        raise ValueError("this protocol requires official DynamicWAM_full.pt")
    gpus = [v.strip() for v in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(not (v.isdigit() or v.startswith("GPU-")) for v in gpus):
        raise ValueError("GPUs must be distinct indices or UUIDs")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"
    ], text=True)
    gpu_rows = [tuple(part.strip() for part in row.split(",")) for row in inventory.splitlines()]
    gpu_map = {key: row for row in gpu_rows for key in row[:2]}
    if any(gpu not in gpu_map for gpu in gpus):
        raise ValueError("requested GPU is absent from the current device inventory")
    gpus = [gpu_map[gpu][1] for gpu in gpus]
    if len(gpus) != len(set(gpus)):
        raise ValueError("multiple selectors refer to the same physical GPU")
    if args.episodes < 1 or args.seed < 0 or args.max_infra_retries < 0 or any(not math.isfinite(v) or v <= 0 for v in
                                              (args.max_backlog_s, args.infer_clock_tolerance_s)):
        raise ValueError("invalid episode count, seed, or clock tolerances")
    if os.name != "posix" or shutil.which("xvfb-run") is None:
        raise RuntimeError("suite requires the Linux DOMINO environment and xvfb-run")
    if not (ROOT / "external/robotwin/bin/python").is_file():
        raise RuntimeError("DOMINO Python environment is missing")
    output = args.output_dir.expanduser().resolve()
    if args.result_stamp is None:
        previous = output / "suite_started.json"
        args.result_stamp = (json.loads(previous.read_text()).get("result_stamp")
                             if args.resume and previous.exists() else
                             datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"))
    replay_run = args.replay_from.expanduser().resolve() if args.replay_from is not None else None
    if replay_run == output:
        raise ValueError("--replay-from must reference a different output directory")
    (output / "logs").mkdir(parents=True, exist_ok=True)
    suite_lock = acquire_lock(output / ".suite.lock")
    provenance = source_provenance(args.source_generation, args.semantics_id,
                                   config_path=args.config, timing_contract_path=timing_path)
    if provenance["source_sha256"].get("timing_contract.json") != timing_sha:
        raise ValueError("timing contract bytes do not match source provenance")
    timing_copy = output / "timing_contract.json"
    if timing_copy.exists():
        if hashlib.sha256(timing_copy.read_bytes()).hexdigest() != timing_sha:
            raise ValueError("run timing contract differs from supplied contract")
    else:
        timing_copy.write_bytes(timing_bytes)
    spec = {"schema_version": 1, "protocol": args.protocol, "timing_mode": args.timing_mode,
            "limit_retiming": args.timing_mode == TIMING_NATIVE_MPLIB, "pid": os.getpid(),
                   "tasks": tasks, "configured_tasks": configured, "gpus": gpus,
                   "episodes_per_task": args.episodes, "seed": args.seed,
                   "controller": CONTROLLER_BY_TIMING[args.timing_mode],
                   "policy_label": "dynamicwam_continuous_sync",
                   "timing_contract": str(timing_path), "timing_contract_sha256": timing_sha,
                   **{key: getattr(args, key) for key in expected_timing},
                   "checkpoint": str(checkpoint), "config": str(args.config.resolve()),
                   "config_sha256": provenance["source_sha256"]["selected_profile.yaml"],
                   "code_root": str(ROOT), "max_backlog_s": args.max_backlog_s,
                   "infer_clock_tolerance_s": args.infer_clock_tolerance_s,
                   "formal_35x100": len(tasks) == 35 and args.episodes == 100,
                   "formal_35x20": len(tasks) == 35 and args.episodes == 20,
                   "formal_full_suite": len(tasks) == 35,
                   "video": args.save_native_results, "result_stamp": args.result_stamp,
                   "virtual_delay": False, "startup_query_latency_included": True,
                   "probe_action_limit": None, "max_infra_retries": args.max_infra_retries,
                   "replay_from": str(replay_run) if replay_run is not None else None, **provenance}
    started = output / "suite_started.json"
    resuming_existing = args.resume and started.exists()
    if resuming_existing:
        validate_resume_spec(json.loads(started.read_text()), spec,
            ("protocol", "timing_mode", "tasks", "episodes_per_task", "seed", "checkpoint", "max_backlog_s",
             "infer_clock_tolerance_s", "virtual_delay", "startup_query_latency_included", "probe_action_limit", "replay_from",
             "config_sha256", "max_infra_retries", "video", "result_stamp", "controller", "policy_label",
             "timing_contract_sha256", "action_interval_s", "history_frame_period_s", "history_endpoint_tolerance_s"))
        (output / "resumes").mkdir(exist_ok=True)
        (output / "resumes" / f"{time.time_ns()}.json").write_text(json.dumps(spec, indent=2) + "\n")
        archive_file(output / "suite_error.json", output / "supervisor_history")
    else:
        with started.open("x") as handle:
            json.dump(spec, handle, indent=2)
    pending = deque(sorted(tasks, key=lambda task: limits[task], reverse=True))
    free = deque(gpus)
    running = []
    finished = []
    budget_path = output / "retry_budget.json"
    budget = load_retry_budget(budget_path, tasks, args.max_infra_retries, resume=resuming_existing)

    def terminate_suite(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_suite)
    try:
        while pending or running:
            while pending and free:
                task, gpu = pending.popleft(), free.popleft()
                error_path = output / task / "error.json"
                prior_error = json.loads(error_path.read_text()) if error_path.exists() else None
                prior_started = (output / task / "task_started.json").exists()
                actual_completed = len(scan_completed_prefix(output / task, task, args.episodes)) if prior_started else 0
                needs_recovery = prior_started and actual_completed < args.episodes
                if needs_recovery and not reserve_retry(budget_path, budget, task, prior_error or {}):
                    exit_record = {"task": task, "status": 1, "completed_episodes": actual_completed,
                                   "retry_number": budget["per_task"][task]["used"],
                                   "error": "retry allowance exhausted or failure is not a recognized clock error"}
                    save(output / "logs" / f"{task}.exit.json", exit_record)
                    finished.append(exit_record)
                    free.appendleft(gpu)
                    print(f"CONTINUOUS_SYNC_NOT_RETRIED task={task} preserved={actual_completed}", flush=True)
                    continue
                if args.resume or budget["per_task"][task]["used"]:
                    archive_file(output / "logs" / f"{task}.log", output / "logs/history")
                    archive_file(output / "logs" / f"{task}.exit.json", output / "logs/history")
                handle = (output / "logs" / f"{task}.log").open("w")
                child_env = os.environ.copy()
                child_env.update(CUDA_VISIBLE_DEVICES=gpu, DYNAMICWAM_SIM_CUDA="0", DYNAMICWAM_INFER_CUDA="0",
                                 OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                                 TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                child_env["DYNAMICWAM_GPU_PCI"] = gpu_map[gpu][2]
                command = ["xvfb-run", "-a", str(ROOT / "external/robotwin/bin/python"),
                           str(ROOT / "scripts/eval_domino_continuous_sync.py"),
                           "--config", str(args.config.resolve()), "--checkpoint", str(checkpoint),
                           "--output-dir", str(output), "--task", task, "--episodes", str(args.episodes),
                           "--seed", str(args.seed), "--max-backlog-s", str(args.max_backlog_s),
                           "--infer-clock-tolerance-s", str(args.infer_clock_tolerance_s),
                           "--source-generation", provenance["source_generation"],
                           "--semantics-id", args.semantics_id,
                           "--timing-mode", args.timing_mode,
                           "--timing-contract", str(timing_path),
                           "--action-interval-s", str(args.action_interval_s),
                           "--history-frame-period-s", str(args.history_frame_period_s),
                           "--history-endpoint-tolerance-s", str(args.history_endpoint_tolerance_s)]
                if args.resume or budget["per_task"][task]["used"]:
                    command.append("--resume")
                if replay_run is not None:
                    command.extend(["--replay-from", str(replay_run)])
                if args.save_native_results:
                    command.extend(["--save-native-results", "--result-stamp", args.result_stamp])
                try:
                    # The new process group owns only this task's Xvfb, evaluator,
                    # and inference child; cleanup cannot target other GPU jobs.
                    record_launch(budget_path, budget, task)
                    process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    handle.close()
                    raise
                running.append((process, handle, task, gpu))
                print(f"CONTINUOUS_SYNC_STARTED task={task} gpu={gpu} pid={process.pid}", flush=True)
            for item in list(running):
                process, handle, task, gpu = item
                status = process.poll()
                if status is None:
                    continue
                handle.close()
                running.remove(item)
                free.append(gpu)
                completion_path = output / task / "completed.json"
                completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
                actual_completed = len(scan_completed_prefix(output / task, task, args.episodes))
                if status == 0 and (completion.get("episodes") != args.episodes
                                    or completion.get("protocol") != args.protocol
                                    or completion.get("probe_action_limit") is not None
                                    or actual_completed != args.episodes):
                    status = 1
                exit_record = {"task": task, "status": status,
                               "completed_episodes": actual_completed, "retry_number": budget["per_task"][task]["used"],
                               "source_generation": provenance["source_generation"]}
                (output / "logs" / f"{task}.exit.json").write_text(json.dumps(exit_record, indent=2) + "\n")
                error_path = output / task / "error.json"
                error = json.loads(error_path.read_text()) if error_path.exists() else {}
                if status != 0 and reserve_retry(budget_path, budget, task, error):
                    pending.append(task)
                    print(f"CONTINUOUS_SYNC_RETRY task={task} retry={budget['per_task'][task]['used']} preserved={actual_completed}", flush=True)
                else:
                    finished.append(exit_record)
                print(f"CONTINUOUS_SYNC_FINISHED task={task} gpu={gpu} status={status}", flush=True)
            state = {"protocol": args.protocol, "timing_mode": args.timing_mode, "finished": finished,
                     "heartbeat_unix_s": time.time(), "running": [{"task": task, "gpu": gpu, "pid": process.pid,
                                  "completed_episodes": json.loads((output / task / "progress.json").read_text()).get("completed_episodes", 0)
                                  if (output / task / "progress.json").exists() else 0}
                                 for process, handle, task, gpu in running], "pending": list(pending),
                     "retry_history": budget["history"], "retry_budget": budget["per_task"],
                     "source_generation": provenance["source_generation"]}
            temporary = output / "suite_status.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(output / "suite_status.json")
            if running:
                time.sleep(5)
    except BaseException as exc:
        (output / "suite_error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "finished": finished, "pending": list(pending)}, indent=2) + "\n")
        raise
    finally:
        for process, handle, task, gpu in running:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            handle.close()
        suite_lock.close()
    from summarize_domino_continuous_sync import summarize
    summary = summarize(output)
    save(output / "summary.json", summary)
    save(output / "suite_completed.json", {
        "all_processes_exited": True, "complete": summary["complete"],
        "valid_completed": summary["valid_completed"], "planned_total": len(tasks) * args.episodes,
        "finished": finished, "source_generation": provenance["source_generation"]})
    return int(any(item["status"] for item in finished) or not summary["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
