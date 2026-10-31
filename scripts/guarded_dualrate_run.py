"""One-shot DualRate eight-rank training or bounded DOMINO deployment supervisor.

Reuse the tested all-adapter lifetime/telemetry and owned-process cleanup code.
No automatic retry, device recovery, or persistent GPU monitoring is performed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import guarded_dualrate_smoke as base

TRAIN_MODES = {"distributed_training_smoke", "formal_distributed_training"}
MODES = TRAIN_MODES | {"domino_deployment_smoke", "dualrate_eval_suite", "dualrate_failure_replay"}
OPERATIONAL_FIELDS = {"run_kind", "output", "steps", "sequence_id", "acceptance_receipt",
    "acceptance_receipt_sha256", "stage_init_checkpoint", "stage_init_checkpoint_sha256", "resume"}


def profile_sha256(config):
    """Acceptance binds all numerical/model/data fields, not run locations/budgets."""
    return base.canonical_hash({k: v for k, v in config.items() if k not in OPERATIONAL_FIELDS})


def frozen_source(repo):
    result = base.source_snapshot(repo)
    result["distributed_supervisor_sha256"] = base.file_sha256(Path(__file__))
    result["scripts_sha256"] = base.source_identity(repo / "scripts")
    return result


def source_contract(source):
    return {key: source[key] for key in ("source_sha256", "guard_sha256", "supervisor_sha256",
            "distributed_supervisor_sha256", "scripts_sha256")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_acceptance(config, source):
    if config["run_kind"] == "dualrate_failure_replay":
        from dualrate_failure_replay import validate_config
        validate_config(config)
        return None
    if config['run_kind'] == 'dualrate_eval_suite':
        if config.get('stage') == 'smoke':
            return None
        if config.get('stage') != 'formal':
            raise ValueError('evaluation requires explicit smoke/formal stage')
        path = Path(config['acceptance_receipt']).resolve(strict=True)
        if base.file_sha256(path) != config['acceptance_receipt_sha256']:
            raise ValueError('evaluation acceptance changed')
        receipt = read_json(path)
        deployment = receipt.get('deployment_acceptance', {})
        if (receipt.get('passed') is not True or receipt.get('source_contract') != source_contract(source)
                or receipt.get('checkpoint_sha256') != config['checkpoint_sha256']
                or receipt.get('benchmark_contract') != config['benchmark_contract']
                or receipt.get('world_size') != 8 or receipt.get('complete_smoke_episodes') != 4
                or receipt.get('native_mp4_full_decode') is not True
                or receipt.get('hardtime_acceptance_passed') is not True
                or deployment.get('passed') is not True
                or deployment.get('fast_graph_passed') is not True
                or deployment.get('fast_queries_during_slow', 0) <= 0
                or deployment.get('model_overlap_fast_queries', 0) <= 0
                or deployment.get('refreshed_reference_consumed') is not True):
            raise ValueError('formal eval requires final-checkpoint, same-source four-episode acceptance')
        if not receipt.get('evidence'):
            raise ValueError('evaluation smoke evidence missing')
        for row in receipt['evidence']:
            if base.file_sha256(Path(row['path']).resolve(strict=True)) != row['sha256']:
                raise ValueError('evaluation smoke evidence changed')
        return receipt
    if config["run_kind"] != "formal_distributed_training":
        return None
    path = Path(config["acceptance_receipt"]).resolve(strict=True)
    if base.file_sha256(path) != config["acceptance_receipt_sha256"]:
        raise ValueError("formal acceptance receipt hash changed")
    receipt = read_json(path)
    if receipt.get("passed") is not True or receipt.get("source_sha256") != source["source_sha256"]:
        raise ValueError("formal acceptance is absent or for another model source")
    if receipt.get("supervisor_sha256") != source["distributed_supervisor_sha256"]:
        raise ValueError("formal acceptance is for another supervisor")
    if receipt.get("source_contract") != source_contract(source):
        raise ValueError("formal acceptance guard/deployment/controller script binding changed")
    if receipt.get("profiles", {}).get(config["phase"]) != profile_sha256(config):
        raise ValueError("phase model/data/optimizer configuration differs from accepted smoke")
    # Every referenced receipt is immutable and independently readable at launch.
    for row in receipt.get("evidence", []):
        evidence_path = Path(row["path"]).resolve(strict=True)
        if base.file_sha256(evidence_path) != row["sha256"]:
            raise ValueError("formal acceptance evidence changed")
    if not receipt.get("evidence") or receipt.get("world_size") != 8:
        raise ValueError("formal acceptance lacks actual eight-rank evidence")
    return receipt


def selected_uuids(value, devices, mode):
    selected = value.split(",")
    if not selected or len(set(selected)) != len(selected) or any(
            re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", u) is None or u not in devices for u in selected):
        raise ValueError("distinct exact inventory GPU UUIDs required")
    if mode in TRAIN_MODES | {'dualrate_eval_suite','dualrate_failure_replay'} and (len(selected) != 8 or set(selected) != set(devices)):
        raise ValueError("training requires exactly all eight adapters; no single-rank substitute")
    if mode == "domino_deployment_smoke" and len(selected) not in (1, 2, 3):
        raise ValueError("deployment smoke requires one to three explicitly pinned GPUs")
    return selected


def worker_command(repo, config, frozen, authorization, resume=None):
    mode = config["run_kind"]
    if mode == "dualrate_failure_replay":
        return [sys.executable,str(repo/"scripts/dualrate_failure_replay.py"),"--config",str(frozen),"--authorization",str(authorization)]
    common = ["--config", str(frozen), "--device", "cuda", "--authorization", str(authorization)]
    if mode == 'dualrate_eval_suite':
        return [sys.executable, str(repo/'scripts/dualrate_eval_suite.py'),
                '--config', str(frozen), '--authorization', str(authorization)]
    if mode in TRAIN_MODES:
        if config.get("world_size") != 8:
            raise ValueError("configuration must explicitly require world_size=8")
        # Torch elastic puts each rank in a new session, defeating session-owned
        # cleanup if its parent crashes. The fixed launcher deliberately retains
        # every rank in this supervisor-created worker session.
        return [sys.executable, str(repo / "scripts/dualrate_rank_launcher.py"), *common,
                *(["--resume", str(resume)] if resume is not None else [])]
    if mode == "domino_deployment_smoke":
        script="dualrate_compact_v7_smoke.py" if config.get("native_v7") else "dualrate_native_v6_smoke.py" if config.get("native_v6") else "dualrate_native_v5_smoke.py" if config.get("native_v5") else "dualrate_domino_smoke.py"
        return [sys.executable, str(repo / "scripts" / script), *common]
    raise ValueError("unsupported fixed worker")


def verify_consumed(authorization, mode):
    paths = ([authorization.with_name(authorization.name + f".rank_{rank}.consumed") for rank in range(8)]
             if mode in TRAIN_MODES else [authorization.with_suffix(".json.consumed")])
    expected = base.file_sha256(authorization)
    for path in paths:
        row = read_json(path)
        if row.get("receipt_sha256") != expected:
            raise RuntimeError("worker consumed a different authorization")


def supervise(args, repo, out):
    from device_lifetime_guard import DeviceLifetimeGuard
    import fcntl
    errors, locks = [], []
    guard = proc = before = boot = source = config = identity = binding = selected = None
    closed, outcome, leader_ticks = True, "failed", None
    began = time.monotonic()

    def status(state, **extra):
        base.atomic(out / "status.json", dict(state=state, supervisor=base.proc_identity(os.getpid()),
            errors=list(errors), guard_closed=closed, no_retry=True,
            elapsed_seconds=time.monotonic()-began, **extra))

    def interrupted(signum, _frame):
        raise InterruptedError(f"supervised run interrupted/timed out: {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.max_runtime)
    status("preparing")
    try:
        source = frozen_source(repo)
        config = read_json(args.config)
        mode = config.get("run_kind")
        if mode not in MODES:
            raise ValueError("explicit supported run_kind required")
        storage = Path(config["storage_root"]).resolve(strict=True)
        base.storage_directory(out, storage)
        output = base.storage_directory(Path(config["output"]), storage)
        resume = args.resume.resolve(strict=True) if args.resume is not None else None
        if resume is not None:
            if mode not in TRAIN_MODES or resume != (output / "last.pt").resolve(strict=True):
                raise ValueError("explicit resume must use this exact run's last.pt")
        elif output.exists():
            raise ValueError("run output must be new unless explicit --resume is supplied")
        accepted = validate_acceptance(config, source)
        binding = base.dependency_helpers().validate_dependency_overlay(config)
        base.atomic(out / "dependencies_before.json", binding)
        frozen = out / "config.frozen.json"
        base.atomic(frozen, config)
        base.atomic(out / "source.json", source)
        devices = base.inventory()
        selected = selected_uuids(args.gpus, devices, mode)
        locks_dir = storage / "locks/dualrate_gpu"
        locks_dir.mkdir(parents=True, exist_ok=True)
        for uuid in sorted(selected):
            lock = (locks_dir / (uuid + ".lock")).open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = base.kernel_evidence()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        base.atomic(out / "kernel_before.json", dict(boot_id=boot, lines=before))
        base.atomic(out / "inventory.json", devices)
        guard = DeviceLifetimeGuard(list(devices), lambda events: base.atomic(out / "guard_events.json", events))
        closed = False
        guard.open()
        status("guard_open")
        rows = base.telemetry(devices, out, "preflight")
        for uuid in selected:
            base.require_idle(rows, uuid)
        if base.kernel_evidence() != before:
            raise RuntimeError("new driver evidence during initial telemetry")
        env = base.worker_environment(repo, out, ",".join(selected), binding, config=config)
        if mode in {"domino_deployment_smoke", "dualrate_eval_suite", "dualrate_failure_replay"}:
            render_config = dict(config, run_kind='domino_deployment_smoke')
            env.update(base.renderer_helpers().renderer_environment(render_config, env))
            base.atomic(out / "renderer_runtime_before.json", config["renderer_runtime"])
            base.atomic(out / "renderer_environment.json",
                        base.renderer_helpers().verify_renderer_environment(render_config, env))
        env.pop("WORLD_SIZE", None)
        # Native communication settings: no driver/P2P recovery flags or retries.
        env.update(TORCH_NCCL_ASYNC_ERROR_HANDLING="1", TORCH_DISTRIBUTED_DEBUG="OFF")
        probe_code = ("import json,sys;from dynamicwam.dualrate.dependencies import probe_dependency_imports;"
            "print(json.dumps(probe_dependency_imports(json.load(open(sys.argv[1])),require_flash_attention=True)))")
        probe = json.loads(subprocess.check_output([sys.executable, "-B", "-c", probe_code, str(frozen)],
            cwd=repo, env=dict(env, CUDA_VISIBLE_DEVICES=""), text=True, timeout=90))
        base.atomic(out / "dependency_import_probe.json", probe)
        identity = base.build_identity(config, source, probe["versions"]["torch"])
        if frozen_source(repo) != source or base.canonical_hash(read_json(args.config)) != identity["config_sha256"]:
            raise RuntimeError("source/config changed during preflight")
        receipt = dict(mode=mode, authorized=True, identity=identity, storage_root=str(storage),
            cuda_visible_devices=",".join(selected), gpu_uuids=selected, world_size=8 if mode in TRAIN_MODES | {'dualrate_eval_suite','dualrate_failure_replay'} else 1,
            expires_at_unix=time.time()+600., external_guard=dict(pid=os.getpid(),
                start_ticks=base.proc_identity(os.getpid())["start_ticks"], mode=base.GUARD_MODE,
                device_paths=[v["device_path"] for v in devices.values()]),
            source=source, boot_id=boot, python_dependencies=binding, dependency_import_probe=probe,
            resume_checkpoint=str(resume) if resume is not None else None,
            resume_checkpoint_sha256=base.file_sha256(resume) if resume is not None else None,
            acceptance_receipt_sha256=config.get("acceptance_receipt_sha256"),
            formal_training_ready=accepted is not None)
        if mode == "domino_deployment_smoke" and len(selected) == 3:
            roles = {"renderer": 0, "fast": 1, "slow": 2}
            if config.get("deployment", {}).get("device_roles") != roles:
                raise ValueError("three-device deployment must explicitly bind renderer0/fast1/slow2")
            receipt["device_roles"] = roles
        authorization = out / "authorization.json"
        base.atomic(authorization, receipt)
        command = worker_command(repo, config, frozen, authorization, resume)
        base.atomic(out / "launch.json", dict(argv=command, gpu_uuids=selected, identity=identity))
        with (out / "worker.log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            leader_ticks = base.proc_identity(proc.pid)["start_ticks"]
            status("running", worker_pid=proc.pid, worker_start_ticks=leader_ticks, mode=mode,
                   identity=identity, formal_training_ready=accepted is not None)
            rc = proc.wait(timeout=max(.01, args.max_runtime-(time.monotonic()-began)))
        if rc:
            raise RuntimeError(f"worker exited {rc}; automatic retry disabled")
        verify_consumed(authorization, mode)
        outcome = "completed"
    except BaseException as exc:
        errors.append(repr(exc))
        status("failed_cleaning")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        if proc is not None:
            try:
                remaining = base.cleanup_owned(proc, leader_ticks)
            except BaseException as exc:
                errors.append(f"owned cleanup unconfirmed: {exc!r}")
                remaining = None
            if remaining is None or remaining:
                errors.append("owned process exit unconfirmed; necessary cleanup supervisor retained")
                status("cleanup_pending", owned_pgid=proc.pid, remaining=remaining)
                while True:
                    try:
                        proc.poll()
                        if not base.live_owned_members(proc.pid):
                            break
                    except BaseException:
                        pass
                    time.sleep(1.)
            base.atomic(out / "owned_exit.json", dict(owned_pgid=proc.pid, remaining=[], returncode=proc.poll()))
        if guard is not None:
            if len(guard.handles) == len(devices) and proc is not None:
                try:
                    time.sleep(2.)
                    post = base.telemetry(devices, out, "postflight")
                    for uuid in selected:
                        base.require_idle(post, uuid)
                except BaseException as exc:
                    errors.append(f"postflight failed: {exc!r}")
            try:
                guard.close()
                closed = not guard.handles
            except BaseException as exc:
                errors.append(f"guard closure unconfirmed: {exc!r}")
        if before is not None:
            try:
                after = base.kernel_evidence()
                same_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
                base.atomic(out / "kernel_after.json", dict(lines=after, unchanged=after==before,
                    same_boot=same_boot, added=list((collections.Counter(after)-collections.Counter(before)).elements())))
                if after != before or not same_boot:
                    errors.append("driver evidence/boot changed; clean exit rejected")
            except BaseException as exc:
                errors.append(f"exit driver evidence unavailable: {exc!r}")
        if source is not None:
            try:
                if frozen_source(repo) != source:
                    errors.append("frozen source changed during job")
                if identity is not None and base.build_identity(config, source, identity["torch_version"]) != identity:
                    errors.append("training identity changed during job")
                if config is not None and read_json(args.config) != config:
                    errors.append("source config changed during job")
                if binding is not None:
                    final = base.dependency_helpers().validate_dependency_overlay(config)
                    base.atomic(out / "dependencies_after.json", final)
                    if final != binding:
                        errors.append("dependency overlay changed during job")
                if config is not None and config.get("run_kind") in {"domino_deployment_smoke", "dualrate_eval_suite", "dualrate_failure_replay"}:
                    renderer = base.renderer_helpers().validate_renderer_runtime(config)
                    base.atomic(out / "renderer_runtime_after.json", renderer)
            except BaseException as exc:
                errors.append(f"final binding verification failed: {exc!r}")
        for lock in locks:
            lock.close()
        if errors or not closed:
            outcome = "failed"
        status(outcome, identity=identity)
    return 0 if outcome == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, default=Path("/SSD_DISK/users/wuruihan"))
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--resume", type=Path, help="explicit same-config full-state resume; never automatic")
    parser.add_argument("--max-runtime", type=float, default=14400.)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix" or not 0 < args.max_runtime <= 30*86400:
        parser.error("Linux and explicit bounded runtime <=30 days required")
    repo = Path(__file__).resolve().parents[1]
    args.config = args.config.resolve(strict=True)
    out = base.storage_directory(args.run_dir, args.storage_root)
    if not args.supervisor:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or any(p.name != "supervisor.log" for p in out.iterdir()):
        parser.error("detached supervisor requires its fresh output directory")
    if args.detach and not args.supervisor:
        argv = [v for v in sys.argv[1:] if v != "--detach"] + ["--supervisor"]
        with (out / "supervisor.log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv], cwd=repo,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(state="dispatched", supervisor_pid=proc.pid, run_dir=str(out))), flush=True)
        return 0
    return supervise(args, repo, out)


if __name__ == "__main__":
    raise SystemExit(main())
