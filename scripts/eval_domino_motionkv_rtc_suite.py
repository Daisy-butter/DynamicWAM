#!/usr/bin/env python3
"""Eight independent GPU workers for frozen-scene MotionKV RTC DOMINO 35x20.

Use --prepare-only first in a separate run directory. All candidate policies
subsequently consume that immutable scene manifest. No automatic retries.
GPU lifetime guard and driver health checks belong to the parent supervisor.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import yaml
from eval_domino_motionkv_rtc import (ROOT, PROTOCOL, PREPARE_PROTOCOL, acquire_lock,
    archive_file, checked_episode, provenance, read, save, sha, validate_descriptor)


def distribution(values):
    values = sorted(float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))
    if not values:
        return {"n": 0}
    def quantile(q):
        position = (len(values) - 1) * q
        lo = int(position); hi = min(lo + 1, len(values) - 1)
        return values[lo] + (position - lo) * (values[hi] - values[lo])
    return dict(n=len(values), mean=sum(values) / len(values), min=values[0], p50=quantile(.5),
                p95=quantile(.95), p99=quantile(.99), max=values[-1])


def summarize(output, spec):
    """Re-read all episode records; neither worker exit nor marker proves success."""
    output = Path(output)
    results, frozen, latencies, reasons = [], [], {}, Counter()
    all_scores, all_successes = [], []
    for task in spec["tasks"]:
        entries, errors, requested = [], [], spec["seed"]
        for episode in range(spec["episodes_per_task"]):
            directory = output / task / f"episode_{episode:04d}"
            try:
                descriptor = read(directory / "manifest.json")
                validate_descriptor(descriptor, task, episode, requested)
                requested = descriptor["seed"] + 1
                if spec["prepare_only"]:
                    receipt = read(directory / "prepared.json")
                    expected = dict(manifest_sha256=sha(directory / "manifest.json"), motion_sha256=sha(directory / "expert_motion.pkl"))
                    if receipt != expected:
                        raise ValueError("prepared scene hashes changed")
                    frozen.append(dict(task=task, episode_index=episode, seed=descriptor["seed"], **expected))
                    entries.append(descriptor)
                else:
                    report, metric = checked_episode(directory, task, episode)
                    if (report["checkpoint_sha256"] != spec["checkpoint_sha256"]
                            or report["source_digest"] != spec["source_digest"]
                            or report["frozen_manifest_sha256"] != spec["frozen_manifest_sha256"]
                            or report.get("probe_action_limit") != spec["probe_action_limit"]):
                        raise ValueError("episode provenance changed")
                    # Executed action slots and observations must stay on the
                    # causal clock; the controller itself rejects late commits.
                    if report.get("virtual_delay") is not False:
                        raise ValueError("RTC result lacks actual-wall-clock evidence")
                    entries.append(dict(seed=report["seed"], success=bool(report["success"]), manipulation_score=float(metric["manipulation_score"])))
                    reasons[report["stop_reason"]] += 1
                    for key in ("physics_time_s", "wall_time_s", "max_backlog_s", "max_tick_gap_wall_s",
                                "max_inference_clock_error_s", "execution_inference_overlap_s", "realtime_factor",
                                "capture_gap_physics_mean_s", "capture_gap_physics_max_s",
                                "post_terminal_camera_drain_wall_s", "post_terminal_inference_drain_wall_s"):
                        latencies.setdefault("episode." + key, []).append(report.get(key))
                    for request in report.get("requests", []):
                        for key in ("capture_wall_s", "inference_wall_elapsed_s", "inference_physics_advance_s",
                                    "inference_clock_error_s", "model_compute_wall_s", "observation_to_suffix_effect_s",
                                    "observation_to_first_command_wall_s"):
                            latencies.setdefault("request." + key, []).append(request.get(key))
                        for key, value in request.get("policy_diagnostics", {}).items():
                            if isinstance(value, (int, float)):
                                latencies.setdefault("policy." + key, []).append(value)
            except Exception as exc:
                errors.append(dict(episode_index=episode, error=f"{type(exc).__name__}: {exc}"))
                # A broken seed chain cannot be silently restarted later.
                break
        completion = read(output / task / "completed.json") if (output / task / "completed.json").exists() else {}
        marker_ok = completion.get("episodes") == spec["episodes_per_task"] and completion.get("protocol") == spec["protocol"]
        row = dict(task=task, planned=spec["episodes_per_task"], valid_completed=len(entries), errors=errors, completion_marker_valid=marker_ok)
        if not spec["prepare_only"]:
            successes = sum(entry["success"] for entry in entries)
            scores = [entry["manipulation_score"] for entry in entries]
            row.update(successes=successes, success_rate=successes / len(entries) if entries else None,
                       mean_manipulation_score=sum(scores) / len(scores) if scores else None)
            all_successes.extend(entry["success"] for entry in entries); all_scores.extend(scores)
        results.append(row)
    complete = all(row["completion_marker_valid"] and not row["errors"] and row["valid_completed"] == spec["episodes_per_task"] for row in results)
    summary = dict(protocol=spec["protocol"], complete=complete, planned_total=len(spec["tasks"]) * spec["episodes_per_task"],
                   valid_completed=sum(row["valid_completed"] for row in results), tasks=results)
    if spec["prepare_only"]:
        save(output / "frozen_manifest.json", dict(protocol=PREPARE_PROTOCOL, complete=complete,
            config_sha256=spec["config_sha256"], source_digest=spec["source_digest"], seed=spec["seed"], episodes=frozen))
    else:
        summary.update(successes=sum(all_successes), success_rate=sum(all_successes) / len(all_successes) if all_successes else None,
            mean_manipulation_score=sum(all_scores) / len(all_scores) if all_scores else None,
            stop_reasons=dict(reasons), performance={key: distribution(values) for key, values in sorted(latencies.items())},
            video_contract="native_head_captures_10fps_h264_yuv420p; physical timestamps in sidecars",
            formal_35x20=len(spec["tasks"]) == 35 and spec["episodes_per_task"] == 20 and spec["probe_action_limit"] is None)
    return summary


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/absolute_motion_v2.yaml")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--tasks", default="")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-audit", type=Path)
    p.add_argument("--timing-contract", type=Path)
    p.add_argument("--replay-from", type=Path)
    p.add_argument("--result-stamp")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--probe-action-limit", type=int)
    p.add_argument("--max-backlog-s", type=float, default=.25)
    p.add_argument("--infer-clock-tolerance-s", type=float, default=.02)
    p.add_argument("--action-interval-s", type=float, default=.16)
    p.add_argument("--history-frame-period-s", type=float, default=.16)
    p.add_argument("--task-timeout-s", type=float, default=14400.)
    return p.parse_args()


def main():
    args = parse()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    from harness_preflight import require_ssd
    require_ssd(args.output_dir)
    shared_tmp = Path(os.environ.get("TMPDIR", ""))
    if not shared_tmp.is_absolute() or os.environ.get("TMP") != str(shared_tmp) or os.environ.get("TEMP") != str(shared_tmp):
        raise ValueError("all workers require the same explicit absolute TMPDIR/TMP/TEMP for SAPIEN locks")
    require_ssd(shared_tmp)
    cfg = yaml.safe_load(args.config.read_text()); configured = cfg["benchmark"]["tasks"]
    tasks = args.tasks.split(",") if args.tasks else list(configured)
    if len(configured) != 35 or len(set(configured)) != 35 or not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= set(configured):
        raise ValueError("task list must be unique members of the full 35-task benchmark")
    if args.episodes < 1 or args.seed < 0 or args.task_timeout_s <= 0:
        raise ValueError("invalid episode/seed/timeout")
    checkpoint_sha = frozen_sha = None
    if not args.prepare_only:
        if any(value is None for value in (args.checkpoint, args.checkpoint_audit, args.timing_contract, args.replay_from)):
            raise ValueError("evaluation requires audited checkpoint, timing, and frozen scenes")
        require_ssd(args.checkpoint); require_ssd(args.replay_from)
        checkpoint_sha = sha(args.checkpoint)
        if read(args.checkpoint_audit).get("full", {}).get("sha256") != checkpoint_sha or read(args.checkpoint_audit).get("passed") is not True:
            raise ValueError("checkpoint audit does not match")
        if read(args.replay_from / "suite_completed.json").get("complete") is not True:
            raise ValueError("scene preparation is incomplete")
        frozen_sha = sha(args.replay_from / "frozen_manifest.json")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], text=True)
    rows = [tuple(part.strip() for part in row.split(",")) for row in inventory.splitlines()]
    by_selector = {key: row for row in rows for key in row[:2]}
    selectors = args.gpus.split(",")
    if any(value not in by_selector for value in selectors):
        raise ValueError("requested GPU is absent")
    gpus = [by_selector[value][1] for value in selectors]
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU selectors alias the same device")
    by_uuid = {row[1]: row for row in rows}
    output = args.output_dir.resolve(); (output / "logs").mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output / ".suite.lock")
    old_spec = read(output / "suite_started.json") if (output / "suite_started.json").exists() else None
    stamp = args.result_stamp or (old_spec["result_stamp"] if old_spec and args.resume else datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"))
    source = provenance(args.config)
    spec = dict(schema_version=1, protocol=PREPARE_PROTOCOL if args.prepare_only else PROTOCOL, prepare_only=args.prepare_only,
        tasks=tasks, episodes_per_task=args.episodes, seed=args.seed, result_stamp=stamp, gpus=gpus,
        gpu_pci={uuid: by_uuid[uuid][2] for uuid in gpus}, config=str(args.config.resolve()), config_sha256=sha(args.config),
        source_digest=source["source_digest"], checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        checkpoint_sha256=checkpoint_sha, frozen_manifest_sha256=frozen_sha, replay_from=str(args.replay_from.resolve()) if args.replay_from else None,
        timing_contract_sha256=sha(args.timing_contract) if args.timing_contract else None,
        action_interval_s=args.action_interval_s, history_frame_period_s=args.history_frame_period_s,
        max_backlog_s=args.max_backlog_s, infer_clock_tolerance_s=args.infer_clock_tolerance_s,
        probe_action_limit=args.probe_action_limit, shared_tmpdir=str(shared_tmp), virtual_delay=False)
    if old_spec:
        if not args.resume or old_spec != spec:
            raise ValueError("resume requires the exact frozen suite configuration/source")
        archive_file(output / "suite_error.json", output / "attempts")
    else:
        save(output / "suite_started.json", spec)
        save(output / "source_manifest.json", source)
    limits = yaml.safe_load(((ROOT / cfg["benchmark"]["domino_root"]).resolve() / "task_config/_eval_step_limit.yml").read_text())
    pending, free = deque(sorted(tasks, key=lambda task: limits[task], reverse=True)), deque(gpus)
    running, finished = [], []
    failed = False
    try:
        while pending or running:
            while pending and free and not failed:
                task, gpu = pending.popleft(), free.popleft()
                if args.resume:
                    archive_file(output / "logs" / f"{task}.log", output / "logs/history")
                handle = (output / "logs" / f"{task}.log").open("x")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, DYNAMICWAM_GPU_PCI=by_uuid[gpu][2],
                    DYNAMICWAM_SIM_CUDA="0", DYNAMICWAM_INFER_CUDA="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                cache = output / "runtime_cache" / gpu
                for key, suffix in (("CUDA_CACHE_PATH", "cuda"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                    ("TRITON_CACHE_DIR", "triton"), ("__GL_SHADER_DISK_CACHE_PATH", "nvidia")):
                    path = cache / suffix; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
                command = ["xvfb-run", "-a", str(ROOT / "external/robotwin/bin/python"),
                    str(ROOT / "scripts/eval_domino_motionkv_rtc.py"), "--config", str(args.config.resolve()),
                    "--output-dir", str(output), "--task", task, "--episodes", str(args.episodes), "--seed", str(args.seed),
                    "--result-stamp", stamp, "--max-backlog-s", str(args.max_backlog_s),
                    "--infer-clock-tolerance-s", str(args.infer_clock_tolerance_s),
                    "--action-interval-s", str(args.action_interval_s), "--history-frame-period-s", str(args.history_frame_period_s)]
                if args.prepare_only:
                    command.append("--prepare-only")
                else:
                    for key in ("checkpoint", "checkpoint_audit", "timing_contract", "replay_from"):
                        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key).resolve())])
                if args.resume: command.append("--resume")
                if args.probe_action_limit: command.extend(["--probe-action-limit", str(args.probe_action_limit)])
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    handle.close(); raise
                running.append(dict(child=child, handle=handle, task=task, gpu=gpu, started=time.monotonic()))
                save(output / "logs" / f"{task}.launch.json", dict(command=command, pid=child.pid, gpu=gpu, pci=by_uuid[gpu][2], shared_tmpdir=str(shared_tmp)))
                print(f"MOTIONKV_STARTED task={task} gpu={gpu} pid={child.pid}", flush=True)
            for row in list(running):
                child = row["child"]; code = child.poll()
                if code is None and time.monotonic() - row["started"] > args.task_timeout_s:
                    raise TimeoutError(f"task exceeded bounded wall timeout: {row['task']}")
                if code is None: continue
                row["handle"].close(); running.remove(row); free.append(row["gpu"])
                completion_path = output / row["task"] / "completed.json"
                marker = read(completion_path) if completion_path.exists() else {}
                if code == 0 and (marker.get("episodes") != args.episodes or marker.get("protocol") != spec["protocol"]): code = 1
                record = dict(task=row["task"], gpu=row["gpu"], exit_code=code, pid=child.pid)
                finished.append(record); save(output / "logs" / f"{row['task']}.exit.json", record)
                if code != 0: failed = True
            save(output / "suite_status.json", dict(protocol=spec["protocol"], pid=os.getpid(), heartbeat_unix_s=time.time(),
                 state="failed_draining" if failed else "running", pending=list(pending), finished=finished,
                 running=[dict(task=r["task"], gpu=r["gpu"], pid=r["child"].pid) for r in running]))
            if failed:
                raise RuntimeError("a worker failed; expansion stopped, no automatic retry")
            if running: time.sleep(5)
        summary = summarize(output, spec); save(output / "summary.json", summary)
        save(output / "suite_completed.json", dict(all_processes_exited=True, complete=summary["complete"],
             valid_completed=summary["valid_completed"], planned_total=len(tasks) * args.episodes, finished=finished))
        return 0 if summary["complete"] else 1
    except BaseException as exc:
        save(output / "suite_error.json", dict(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(), finished=finished, pending=list(pending)))
        traceback.print_exc(); return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for row in running:
            child = row["child"]
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    child.wait(timeout=10)
            row["handle"].close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
