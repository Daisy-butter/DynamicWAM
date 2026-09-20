#!/usr/bin/env python3
"""Small-scale DOMINO eval for the async fast-slow adapter.

Official Level-1 still uses deploy_policy_sync_flow through the canonical
profile. This launcher loads that profile only for weights/paths, then
overrides the DOMINO policy name and points at the sidecar async config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamicwam.config import load_profile, write_config_snapshot  # noqa: E402
from dynamicwam.evaluation.headless_render import (  # noqa: E402
    apply_headless_nvidia_egl,
)
from dynamicwam.evaluation.run_domino_level1_suite import (  # noqa: E402
    _worker_env,
    archive_incomplete_task,
    summarize,
    valid_task_aggregate,
    write_json,
)
from dynamicwam.evaluation.summarize_async_control import (  # noqa: E402
    collect_async_summaries,
    summarize_control,
)
from dynamicwam.runtime.ciwam.wam.async_control import (  # noqa: E402
    ASYNC_POLICY_REGISTRATION_NAME,
    DEFAULT_ASYNC_CONFIG_NAME,
    load_async_control_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "absolute_motion_v2_trained_level2.yaml"),
        help="Official DynamicWAM profile used for weights and DOMINO paths.",
    )
    parser.add_argument(
        "--async-config",
        default=str(ROOT / "configs" / DEFAULT_ASYNC_CONFIG_NAME),
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated DOMINO task names, or 'all' for the official 35-task list.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--gpus")
    parser.add_argument("--run-root")
    parser.add_argument("--python")
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--task-timeout-seconds", type=float)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the overridden launch config without starting DOMINO.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import yaml  # type: ignore[import-untyped]

    apply_headless_nvidia_egl()
    profile = load_profile(args.config)
    async_config_path = Path(args.async_config).expanduser().resolve()
    control = load_async_control_config(async_config_path)
    benchmark = profile.benchmark_config()
    if str(args.tasks).strip() in {"", "all"}:
        tasks = list(benchmark["tasks"])
    else:
        tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    unknown = [task for task in tasks if task not in benchmark["tasks"]]
    if unknown:
        raise ValueError(f"tasks are not in the official profile: {unknown}")
    if args.episodes_per_task <= 0:
        raise ValueError("episodes-per-task must be positive")

    base_config = profile.domino_base_config()
    base_config["policy_name"] = ASYNC_POLICY_REGISTRATION_NAME
    base_config["dynamicwam_async_config"] = str(async_config_path)
    base_config["ckpt_setting"] = f"{base_config['ckpt_setting']}-async-fast-slow"
    if args.start_seed is not None:
        base_config["start_seed"] = int(args.start_seed)

    run_root = Path(
        args.run_root
        or (
            Path(profile.raw["paths"]["evaluation_runs"])
            / "absolute_motion_domino_async_control"
        )
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    launch = {
        "policy_name": ASYNC_POLICY_REGISTRATION_NAME,
        "async_control": {
            "fast_horizon": control.fast_horizon,
            "max_world_age_steps": control.max_world_age_steps,
            "inference_delay": control.inference_delay,
            "rtc_mode": control.rtc_mode,
            "motion_refresh_ratio": control.motion_refresh_ratio,
        },
        "source_profile": str(Path(args.config).resolve()),
        "official_policy_untouched": (
            "ciwam.adapters.domino.deploy_policy_sync_flow"
        ),
        "tasks": tasks,
        "episodes_per_task": int(args.episodes_per_task),
        "open_loop_horizon_sync": int(benchmark["execute_steps"]),
        "open_loop_horizon_async": control.fast_horizon,
    }
    write_json(run_root / "async_launch.json", launch)
    write_config_snapshot(
        run_root / "config_audit",
        profile=profile,
        label="async_control",
        resolved_config={
            "base_config": base_config,
            "launch": launch,
        },
    )
    print(
        "async control launch "
        f"tasks={tasks} episodes={args.episodes_per_task} "
        f"fast_horizon={control.fast_horizon} "
        f"max_world_age={control.max_world_age_steps}",
        flush=True,
    )
    if args.dry_run:
        print(f"dry-run wrote {run_root / 'async_launch.json'}", flush=True)
        return 0

    # Official suite always uses the sync policy name from the profile.
    # Write per-task configs here so DOMINO imports the async adapter.
    python = Path(args.python or benchmark["python"]).expanduser().absolute()
    parallel_runner = Path(
        SRC / "dynamicwam" / "evaluation" / "run_domino_parallel_eval.py"
    ).resolve()
    runtime_root = Path(benchmark["runtime_root"]).resolve()
    for required in (
        python,
        parallel_runner,
        Path(benchmark["domino_root"]),
        Path(benchmark["eval_policy"]),
        runtime_root,
        Path(benchmark["curobo_root"]),
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"DOMINO async eval is not runnable here; missing {required}. "
                "Use --dry-run or run scripts/verify_async_control.py."
            )

    raw_gpus = args.gpus or ",".join(benchmark["gpus"])
    gpus = [item.strip() for item in raw_gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list must be non-empty and unique: {gpus}")
    tasks_root = run_root / "tasks"
    configs_root = run_root / "configs"
    logs_root = run_root / "logs"
    archive_root = run_root / "archive"
    for path in (tasks_root, configs_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)

    start_seed = int(base_config["start_seed"])
    episodes_per_task = int(args.episodes_per_task)
    task_timeout_seconds = float(
        args.task_timeout_seconds
        if args.task_timeout_seconds is not None
        else benchmark["task_timeout_seconds"]
    )
    extra_pythonpath = tuple(
        Path(value).expanduser().resolve() for value in benchmark["extra_pythonpath"]
    )
    worker_env = apply_headless_nvidia_egl(_worker_env(extra_pythonpath))
    pending: deque[str] = deque()
    failures: dict[str, dict[str, Any]] = {}
    for task in tasks:
        aggregate_path = tasks_root / task / "aggregate.json"
        if (
            valid_task_aggregate(
                aggregate_path,
                episodes_per_task,
                expected_start_seed=start_seed,
            )
            is None
        ):
            archive_incomplete_task(tasks_root / task, archive_root)
            pending.append(task)

    running: dict[str, dict[str, Any]] = {}
    free_gpus: deque[str] = deque(gpus)
    status_path = run_root / "STATUS.jsonl"
    summary = summarize(
        run_root=run_root,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        failures=failures,
        start_seed=start_seed,
        bootstrap_replicates=int(benchmark["bootstrap_replicates"]),
        bootstrap_seed=int(benchmark["bootstrap_seed"]),
    )
    while pending or running:
        while pending and free_gpus:
            task = pending.popleft()
            gpu = free_gpus.popleft()
            task_config = dict(base_config)
            task_config["task_name"] = task
            task_config["ckpt_setting"] = f"{base_config['ckpt_setting']}-{task}"
            config_path = configs_root / f"{task}.yml"
            config_path.write_text(
                yaml.safe_dump(task_config, sort_keys=False),
                encoding="utf-8",
            )
            log_path = logs_root / f"{task}.launcher.log"
            handle: TextIO = log_path.open("w", encoding="utf-8")
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
                str(runtime_root),
                "--curobo-root",
                str(benchmark["curobo_root"]),
                "--run-root",
                str(tasks_root / task),
                "--gpus",
                gpu,
                "--total-episodes",
                str(episodes_per_task),
                "--start-seed",
                str(start_seed),
                "--slot-seed-stride",
                str(benchmark["slot_seed_stride"]),
                "--timeout-seconds",
                str(task_timeout_seconds),
                "--stagger-seconds",
                "0",
            ]
            process = subprocess.Popen(
                command,
                cwd=benchmark["domino_root"],
                env=worker_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[task] = {
                "gpu": gpu,
                "process": process,
                "handle": handle,
                "log": str(log_path),
            }
            with status_path.open("a", encoding="utf-8") as status_stream:
                status_stream.write(
                    json.dumps(
                        {
                            "event": "started",
                            "task": task,
                            "gpu": gpu,
                            "pid": process.pid,
                        }
                    )
                    + "\n"
                )
            print(f"started task={task} gpu={gpu} pid={process.pid}", flush=True)

        finished: list[str] = []
        for task, record in running.items():
            return_code = record["process"].poll()
            if return_code is None:
                continue
            record["handle"].close()
            aggregate = valid_task_aggregate(
                tasks_root / task / "aggregate.json",
                episodes_per_task,
                expected_start_seed=start_seed,
            )
            task_status = (
                "complete" if return_code == 0 and aggregate is not None else "failed"
            )
            if task_status != "complete":
                failures[task] = {
                    "return_code": return_code,
                    "gpu": record["gpu"],
                    "log": record["log"],
                }
            print(
                f"finished task={task} gpu={record['gpu']} "
                f"status={task_status} rc={return_code}",
                flush=True,
            )
            free_gpus.append(record["gpu"])
            finished.append(task)
        for task in finished:
            del running[task]

        summary = summarize(
            run_root=run_root,
            tasks=tasks,
            episodes_per_task=episodes_per_task,
            failures=failures,
            start_seed=start_seed,
            bootstrap_replicates=int(benchmark["bootstrap_replicates"]),
            bootstrap_seed=int(benchmark["bootstrap_seed"]),
        )
        write_json(run_root / "aggregate.json", summary)
        print(
            f"progress tasks={summary['completed_tasks']}/{summary['task_count']} "
            f"episodes={summary['reported_episodes']}/{summary['requested_episodes']}",
            flush=True,
        )
        if pending or running:
            time.sleep(5)

    metrics = summarize_control(collect_async_summaries(run_root))
    write_json(run_root / "async_control_metrics.json", metrics)
    print(f"ASYNC-CONTROL-METRICS {metrics}", flush=True)
    print(
        f"suite complete tasks={summary['completed_tasks']}/{summary['task_count']} "
        f"episodes={summary['reported_episodes']}/{summary['requested_episodes']} "
        f"success={summary['success_count']}/{summary['requested_episodes']}",
        flush=True,
    )
    return 0 if summary["completed_tasks"] == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
