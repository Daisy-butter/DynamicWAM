"""Run an explicitly frozen three-phase eight-GPU sequence once, never retry.

The controller has no CUDA/NVML imports. Each phase is a fresh guarded worker;
only a clean completed phase with strict eight-rank restore can seed the next.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import guarded_dualrate_run as runner
from guarded_dualrate_smoke import atomic, file_sha256, storage_directory, proc_identity

PHASES = ("warmup", "small_age_joint", "cross_time_joint")


def phase_completed(output, config):
    result = runner.read_json(output / "result.json")
    reload = runner.read_json(output / "reload.json")
    if (result.get("result_kind") != "completed_phase_not_policy_quality_acceptance"
            or result.get("completed_steps") != config["steps"]
            or result.get("configured_steps") != config["steps"]
            or result.get("phase") != config["phase"] or result.get("run_kind") != config["run_kind"]
            or result.get("world_size") != 8):
        raise RuntimeError("phase result is not the requested complete eight-rank training")
    ranks = reload.get("ranks", [])
    if (reload.get("passed") is not True or reload.get("world_size") != 8
            or len(ranks) != 8 or {row.get("rank") for row in ranks} != set(range(8))
            or any(row.get("passed") is not True or row.get("phase_step") != config["steps"] for row in ranks)):
        raise RuntimeError("strict checkpoint restore did not pass on all eight ranks")
    checkpoint = (output / "last.pt").resolve(strict=True)
    digest = file_sha256(checkpoint)
    if result.get("checkpoint_sha256") != digest:
        raise RuntimeError("completed checkpoint bytes differ from final receipt")
    return checkpoint, digest, result, reload


def load_sequence(path):
    spec = runner.read_json(path)
    phases = spec["phases"]
    if [row["phase"] for row in phases] != list(PHASES):
        raise ValueError("exact warmup -> small_age_joint -> cross_time_joint order required")
    if not isinstance(spec.get("sequence_id"), str) or not spec["sequence_id"]:
        raise ValueError("explicit unique sequence_id required")
    configs = []
    for row in phases:
        config_path = Path(row["config"]).resolve(strict=True)
        if file_sha256(config_path) != row["config_sha256"]:
            raise ValueError("phase template changed from sequence manifest")
        config = runner.read_json(config_path)
        if config["phase"] != row["phase"] or config["world_size"] != 8:
            raise ValueError("sequence/template phase/world size mismatch")
        if config["run_kind"] != "formal_distributed_training":
            raise ValueError("automatic sequence accepts formal templates only")
        if any(key in config for key in ("resume", "stage_init_checkpoint", "stage_init_checkpoint_sha256")):
            raise ValueError("sequence must bind predecessors itself, not take arbitrary smoke/resume weights")
        if not 0 < row["max_runtime_seconds"] <= 30*86400:
            raise ValueError("each phase requires explicit runtime bound <=30 days")
        configs.append(config)
    return spec, configs


def run(args, out, repo):
    import fcntl
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = runner.frozen_source(repo)
    spec, configs = load_sequence(args.sequence)
    errors, completed = [], []
    child, began = None, time.monotonic()

    def status(state, **extra):
        atomic(out / "status.json", dict(state=state, controller=proc_identity(os.getpid()),
            source=source, sequence_id=spec["sequence_id"], completed_phases=list(completed),
            errors=list(errors), elapsed_seconds=time.monotonic()-began, no_retry=True, **extra))

    def stop(signum, _frame):
        raise InterruptedError(f"sequence interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        # Validate every formal profile before starting the first job.
        for config in configs:
            runner.validate_acceptance(config, source)
            storage_directory(Path(config["output"]), Path(config["storage_root"]))
            if Path(config["output"]).exists():
                raise ValueError("all sequence phase outputs must be new")
        atomic(out / "sequence.frozen.json", spec)
        atomic(out / "source.json", source)
        status("preparing")
        previous = previous_sha = None
        for row, template in zip(spec["phases"], configs):
            if runner.frozen_source(repo) != source:
                raise RuntimeError("source changed during sequence; no next phase launched")
            config = dict(template, sequence_id=spec["sequence_id"])
            if previous is not None:
                config.update(stage_init_checkpoint=str(previous), stage_init_checkpoint_sha256=previous_sha)
            config_path = out / (row["phase"] + ".config.json")
            atomic(config_path, config)
            phase_dir = out / row["phase"]
            command = [sys.executable, str(repo / "scripts/guarded_dualrate_run.py"),
                "--config", str(config_path), "--run-dir", str(phase_dir),
                "--gpus", spec["gpu_uuids"], "--max-runtime", str(row["max_runtime_seconds"])]
            with (out / (row["phase"] + ".supervisor.log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child_ticks = proc_identity(child.pid)["start_ticks"]
                status("phase_running", phase=row["phase"], supervisor_pid=child.pid,
                    supervisor_start_ticks=child_ticks, output=config["output"])
                rc = child.wait()
            child = None
            receipt = runner.read_json(phase_dir / "status.json")
            if rc or receipt.get("state") != "completed" or receipt.get("guard_closed") is not True:
                raise RuntimeError(f"phase {row['phase']} failed or exit unclean; no next phase/retry")
            previous, previous_sha, result, reload = phase_completed(Path(config["output"]), config)
            completed.append(dict(phase=row["phase"], checkpoint=str(previous),
                checkpoint_sha256=previous_sha, global_step=result["global_step"]))
            status("phase_completed", phase=row["phase"])
        status("completed")
        return 0
    except BaseException as exc:
        errors.append(repr(exc))
        # Only signal this directly owned guard. It owns all worker cleanup and
        # retains its temporary FDs until every child exits, including D-state.
        if child is not None and child.poll() is None:
            try:
                if proc_identity(child.pid)["start_ticks"] != child_ticks:
                    raise RuntimeError("supervisor PID reused; refusing signal")
                child.send_signal(signal.SIGTERM)
                status("failed_waiting_guard_cleanup", supervisor_pid=child.pid)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, signal.SIG_IGN)
                child.wait()
            except BaseException as cleanup_exc:
                errors.append(f"supervisor cleanup unconfirmed: {cleanup_exc!r}")
        status("failed")
        return 1
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, default=Path("/SSD_DISK/users/wuruihan"))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Linux required")
    repo = Path(__file__).resolve().parents[1]
    args.sequence = args.sequence.resolve(strict=True)
    out = storage_directory(args.run_dir, args.storage_root)
    if not args.controller:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or any(p.name != "controller.log" for p in out.iterdir()):
        parser.error("detached controller requires its fresh output directory")
    if args.detach and not args.controller:
        argv = [v for v in sys.argv[1:] if v != "--detach"] + ["--controller"]
        with (out / "controller.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv], cwd=repo,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(state="dispatched", controller_pid=child.pid, run_dir=str(out))), flush=True)
        return 0
    return run(args, out, repo)


if __name__ == "__main__":
    raise SystemExit(main())
