#!/usr/bin/env python3
"""A few DOMINO episodes of the joint multirate checkpoint against sync.

By default both arms replan after four executed actions, and the sync arm loads
the same multirate checkpoint's backbone. Use --sync-horizon 16 for the native
long chunk, or --official-sync to keep the released stage3 weights. --same-cadence
runs one fast sample per 16 actions and prefetches the bridge 4 actions earlier.
--pace-action-interval pads non-query multirate steps to the 100ms action
interval so that prefetch can leave the decision. --serial plus identical GPU
lists removes cross-GPU timing confounds. Query timing excludes simulator
execution, rendering, and optical flow.
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamicwam.config import load_profile  # noqa: E402
from dynamicwam.evaluation.headless_render import apply_headless_nvidia_egl  # noqa: E402
from dynamicwam.evaluation.run_domino_level1_suite import _worker_env  # noqa: E402

CHECKPOINT = Path(
    "/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/"
    "world_action_multirate_joint/exports/multirate_step_1000.pt"
)
RUN_ROOT = Path(
    "/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/multirate_joint_probe"
)
TASKS = ("place_shoe", "place_fan", "adjust_bottle")
EPISODES = 2
SYNC_GPUS = ("1", "2", "3")
MULTIRATE_GPUS = ("4", "5", "6")
TIMEOUT_SECONDS = 3600.0
SERIAL = False


def wait_for_checkpoint(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_size = -1
    stable_since: float | None = None
    while time.monotonic() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 15.0:
                    print(f"checkpoint ready {path} bytes={size}", flush=True)
                    return
            else:
                stable_since = None
                last_size = size
                print(f"checkpoint appearing bytes={size}", flush=True)
        else:
            print(f"waiting for {path}", flush=True)
        time.sleep(10.0)
    raise TimeoutError(f"joint checkpoint was not ready: {path}")


def write_task_config(
    path: Path,
    base: dict,
    *,
    task: str,
    policy_name: str,
    ckpt_setting: str,
    extra: dict | None = None,
) -> None:
    config = dict(base)
    config["task_name"] = task
    config["policy_name"] = policy_name
    config["ckpt_setting"] = ckpt_setting
    config["episode_num"] = EPISODES
    config["eval_video_log"] = False
    config["eval_output_root"] = str(path.parent / "results")
    if extra:
        config.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def launch(
    *,
    python: Path,
    parallel_runner: Path,
    benchmark: dict,
    config_path: Path,
    run_root: Path,
    gpu: str,
    log_path: Path,
    env: dict[str, str],
) -> subprocess.Popen:
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root is not empty: {run_root}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-u",
        str(parallel_runner),
        "--python",
        str(python),
        "--domino-root",
        str(benchmark["domino_root"]),
        "--eval-policy",
        str(benchmark["eval_policy"]),
        "--config",
        str(config_path),
        "--runtime-root",
        str(benchmark["runtime_root"]),
        "--curobo-root",
        str(benchmark["curobo_root"]),
        "--run-root",
        str(run_root),
        "--gpus",
        gpu,
        "--total-episodes",
        str(EPISODES),
        "--start-seed",
        str(int(benchmark["start_seed"])),
        "--slot-seed-stride",
        str(int(benchmark["slot_seed_stride"])),
        "--timeout-seconds",
        str(TIMEOUT_SECONDS),
        "--stagger-seconds",
        "0",
    ]
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(benchmark["domino_root"]),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._log_handle = handle  # type: ignore[attr-defined]
    print(f"started gpu={gpu} pid={process.pid} config={config_path.name}", flush=True)
    if SERIAL:
        process.wait(timeout=TIMEOUT_SECONDS + 120.0)
    return process


def parse_timing(log_path: Path, prefix: str) -> list[dict]:
    rows = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if prefix not in line:
            continue
        try:
            value = ast.literal_eval(line.split(prefix, 1)[1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _bridge_modes(log_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in parse_timing(log_path, "MULTIRATE-QUERY-DETAIL "):
        mode = str(row.get("bridge_mode", "missing"))
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> int:
    global RUN_ROOT, CHECKPOINT, EPISODES, SYNC_GPUS, MULTIRATE_GPUS, SERIAL, TASKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="new output directory; existing results are never overwritten")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task subset for a functional probe")
    parser.add_argument("--sync-horizon", type=int, choices=range(1,17), default=4)
    parser.add_argument("--scheduling", choices=("controlled_ratio","independent_worker"), default="controlled_ratio")
    parser.add_argument("--same-cadence", action="store_true",
                        help="One fast sample per 16 actions, bridge snapshotted 4 actions earlier.")
    parser.add_argument("--pace-action-interval", action="store_true",
                        help="Pad multirate steps that do not sample to action_interval.")
    parser.add_argument("--bridge-lead-actions", type=int, default=4)
    parser.add_argument("--bridge-wait-budget-ms", type=float, default=150.0)
    parser.add_argument("--sync-backbone-checkpoint", type=Path, default=None,
                        help="Sync sampler weights. Defaults to --checkpoint so the horizon match uses the same backbone.")
    parser.add_argument("--official-sync", action="store_true",
                        help="Keep the profile's released sync checkpoint instead of the multirate backbone.")
    parser.add_argument("--sync-gpus", default=",".join(SYNC_GPUS))
    parser.add_argument("--multirate-gpus", default=",".join(MULTIRATE_GPUS))
    parser.add_argument("--serial", action="store_true", help="run jobs sequentially for a same-GPU timing comparison")
    parser.add_argument("--reuse-action-noise", action="store_true", help="align noise across overlapping action chunks; separate continuity ablation")
    parser.add_argument("--bridge-refresh-ratio", type=int, default=4,
                        help="Controlled R: rebuild the world bridge every R fast queries.")
    args = parser.parse_args()
    TASKS = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    if not TASKS or len(set(TASKS)) != len(TASKS): parser.error("tasks must be non-empty and unique")
    RUN_ROOT, CHECKPOINT, EPISODES = args.run_root, args.checkpoint, args.episodes
    if args.sync_backbone_checkpoint is None and not args.official_sync:
        args.sync_backbone_checkpoint = CHECKPOINT
    SYNC_GPUS, MULTIRATE_GPUS = tuple(args.sync_gpus.split(",")), tuple(args.multirate_gpus.split(","))
    SERIAL = args.serial
    if EPISODES < 1 or len(SYNC_GPUS) != len(TASKS) or len(MULTIRATE_GPUS) != len(TASKS):
        parser.error("positive episodes and one GPU entry per task are required")
    if not SERIAL and len(set(SYNC_GPUS + MULTIRATE_GPUS)) != len(SYNC_GPUS + MULTIRATE_GPUS):
        parser.error("sharing a GPU across probe jobs requires --serial")
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
        raise FileExistsError(f"probe output must be a new/empty directory: {RUN_ROOT}")
    wait_for_checkpoint(CHECKPOINT, timeout_s=2400.0)
    profile = load_profile(ROOT / "configs" / "absolute_motion_v2.yaml")
    benchmark = profile.benchmark_config()
    base = profile.domino_base_config()
    unknown = [task for task in TASKS if task not in benchmark["tasks"]]
    if unknown:
        raise ValueError(f"tasks are not in the official profile: {unknown}")
    # Keep the robotwin venv symlink. resolve() would follow it into the
    # conda interpreter and drop the venv's diffusers install.
    python = Path(benchmark["python"]).expanduser().absolute()
    parallel_runner = (
        SRC / "dynamicwam" / "evaluation" / "run_domino_parallel_eval.py"
    ).resolve()
    for required in (
        python,
        parallel_runner,
        Path(benchmark["domino_root"]),
        Path(benchmark["eval_policy"]),
        Path(benchmark["curobo_root"]),
        CHECKPOINT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    env = apply_headless_nvidia_egl(
        _worker_env(
            tuple(Path(value).expanduser().resolve() for value in benchmark["extra_pythonpath"])
        )
    )
    jobs: list[dict] = []
    for task, gpu in zip(TASKS, SYNC_GPUS, strict=True):
        group = RUN_ROOT / "sync" / task
        config_path = group / "config.yml"
        write_task_config(
            config_path,
            base,
            task=task,
            policy_name="ciwam.adapters.domino.deploy_policy_sync_flow",
            ckpt_setting=f"sync-h{args.sync_horizon}-probe-{task}",
            extra={"execute_horizon": args.sync_horizon,
                   **({"multirate_backbone_checkpoint": str(args.sync_backbone_checkpoint)} if args.sync_backbone_checkpoint else {})},
        )
        jobs.append(
            {
                "arm": "sync",
                "task": task,
                "gpu": gpu,
                "prefix": "SYNC-QUERY-TIMING ",
                "process": launch(
                    python=python,
                    parallel_runner=parallel_runner,
                    benchmark=benchmark,
                    config_path=config_path,
                    run_root=group / "run",
                    gpu=gpu,
                    log_path=group / "launcher.log",
                    env=env,
                ),
                "log": group / "run" / "logs" / f"slot0_gpu{gpu}.log",
                "aggregate": group / "run" / "aggregate.json",
            }
        )
    for task, gpu in zip(TASKS, MULTIRATE_GPUS, strict=True):
        group = RUN_ROOT / "multirate" / task
        config_path = group / "config.yml"
        write_task_config(
            config_path,
            base,
            task=task,
            policy_name="ciwam.adapters.domino.deploy_policy_multirate_flow",
            ckpt_setting=f"multirate-joint-step1000-{task}",
            extra={
                "multirate_checkpoint": str(CHECKPOINT),
                "execute_horizon": 16 if args.same_cadence else 4,
                "bridge_refresh_ratio": int(args.bridge_refresh_ratio),
                "multirate_scheduling": args.scheduling,
                "multirate_reuse_action_noise": args.reuse_action_noise,
                **({"bridge_lead_actions": args.bridge_lead_actions,
                    "bridge_wait_budget_ms": args.bridge_wait_budget_ms} if args.same_cadence else {}),
                **({"pace_action_interval": True} if args.pace_action_interval else {}),
            },
        )
        jobs.append(
            {
                "arm": "multirate",
                "task": task,
                "gpu": gpu,
                "prefix": "MULTIRATE-QUERY-TIMING ",
                "process": launch(
                    python=python,
                    parallel_runner=parallel_runner,
                    benchmark=benchmark,
                    config_path=config_path,
                    run_root=group / "run",
                    gpu=gpu,
                    log_path=group / "launcher.log",
                    env=env,
                ),
                "log": group / "run" / "logs" / f"slot0_gpu{gpu}.log",
                "aggregate": group / "run" / "aggregate.json",
            }
        )

    deadline = time.monotonic() + TIMEOUT_SECONDS + 120.0
    pending = list(jobs)
    while pending and time.monotonic() < deadline:
        still = []
        for job in pending:
            code = job["process"].poll()
            if code is None:
                still.append(job)
                continue
            job["return_code"] = code
            job["process"]._log_handle.close()  # type: ignore[attr-defined]
            print(
                f"finished arm={job['arm']} task={job['task']} gpu={job['gpu']} rc={code}",
                flush=True,
            )
        pending = still
        if pending:
            time.sleep(15.0)
    if pending:
        for job in pending:
            job["process"].kill()
            job["return_code"] = None
        raise TimeoutError("probe still running after the deadline")

    report = []
    for job in jobs:
        aggregate = {}
        if job["aggregate"].is_file():
            aggregate = json.loads(job["aggregate"].read_text(encoding="utf-8"))
        timings = parse_timing(job["log"], job["prefix"])
        per_action = [
            float(row["ms_per_executed_action"])
            for row in timings
            if isinstance(row.get("ms_per_executed_action"), (int, float))
        ]
        query_mean = [
            float(row["query_ms_mean"])
            for row in timings
            if isinstance(row.get("query_ms_mean"), (int, float))
        ]
        report.append(
            {
                "arm": job["arm"],
                "task": job["task"],
                "gpu": job["gpu"],
                "return_code": job.get("return_code"),
                "success_count": aggregate.get("success_count"),
                "reported_episodes": aggregate.get("reported_episodes"),
                "episodes": [
                    {
                        "seed": episode.get("seed"),
                        "success": episode.get("success"),
                        "steps": episode.get("steps"),
                    }
                    for episode in aggregate.get("episodes") or []
                ],
                "execute_horizon": args.sync_horizon if job["arm"] == "sync" else (16 if args.same_cadence else 4),
                "bridge_lead_actions": args.bridge_lead_actions if job["arm"] == "multirate" and args.same_cadence else 0,
                "pace_action_interval": bool(args.pace_action_interval and job["arm"] == "multirate"),
                "scheduling": "sync" if job["arm"] == "sync" else args.scheduling,
                "bridge_modes": _bridge_modes(job["log"]) if job["arm"] == "multirate" else {},
                "query_ms_mean_episode_macro": mean(query_mean),
                "ms_per_executed_action_episode_macro": mean(per_action),
                "query_ms_mean": (sum(row["query_ms_sum"] for row in timings) / sum(row["query_count"] for row in timings)) if timings and sum(row["query_count"] for row in timings) else None,
                "ms_per_executed_action": (sum(row["query_ms_sum"] for row in timings) / sum(row["executed_steps"] for row in timings)) if timings and sum(row["executed_steps"] for row in timings) else None,
                "timings": timings,
            }
        )
    for task in TASKS:
        arms = [row for row in report if row["task"] == task]
        matched = len(arms) == 2 and all(len(arm["episodes"]) == EPISODES for arm in arms) and sorted(e["seed"] for e in arms[0]["episodes"]) == sorted(e["seed"] for e in arms[1]["episodes"])
        for row in arms:
            row["paired_episode_seeds_match"] = matched
    out = RUN_ROOT / "probe_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if all(item.get("return_code") == 0 for item in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
