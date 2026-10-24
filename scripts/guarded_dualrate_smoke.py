"""One guarded single-GPU engineering job; no retries, repair, or GPU polling.

--inspect only inventories adapters. --detach creates the persistent parent of
the worker. A timed-out unkillable owned worker leaves cleanup_pending: that
necessary supervisor holds its descriptors until worker exit, without telemetry.
This exception is never reported as successful or as a bounded clean exit.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


MODE = "single_process_engineering_smoke"
GUARD_MODE = "fds_before_telemetry_v1"
KERNEL_PATTERN = re.compile(r"NVRM:.*Xid|IOVAS|pIOVAS|NVLink.*(?:RPC|lock|error)", re.I)


def atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity(root: Path) -> str:
    """Keep byte-identical to train.source_identity; do not import torch here."""
    value = hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    if not paths:
        raise ValueError("empty source tree")
    for path in paths:
        value.update(path.relative_to(root).as_posix().encode() + b"\0")
        value.update(bytes.fromhex(file_sha256(path)))
    return value.hexdigest()


def source_snapshot(repo: Path) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True, timeout=15).strip():
        raise RuntimeError("freeze/commit source before launch; worktree is dirty")
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=15).strip(),
            "source_sha256": source_identity(repo / "src/dynamicwam"),
            "guard_sha256": file_sha256(repo / "scripts/device_lifetime_guard.py"),
            "supervisor_sha256": file_sha256(repo / "scripts/guarded_dualrate_smoke.py")}


def proc_identity(pid: int, proc_root: Path = Path("/proc")) -> dict:
    # stat never exposes command-line arguments or environment credentials.
    fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
            "pgrp": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19])}


def live_owned_members(pgid: int, proc_root: Path = Path("/proc")) -> list[dict]:
    members = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = proc_identity(int(path.name), proc_root)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (row["pgrp"] == pgid or row["session"] == pgid) and row["state"] not in ("Z", "X"):
            members.append(row)
    return members


def cleanup_owned(proc, leader_ticks: int, *, term_seconds=30., kill_seconds=30.,
                  observe=live_owned_members, identify=proc_identity, send=os.kill,
                  clock=time.monotonic, sleep=time.sleep) -> list[dict]:
    """Signal only freshly re-identified members of the new owned session.

    Individual signals also cover a descendant that changes its process group.
    Workers must not daemonize into new sessions; these fixed entrypoints do not.
    """
    for sig, duration in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
        rows = observe(proc.pid)
        for row in rows:
            if row["pid"] == proc.pid and row["start_ticks"] != leader_ticks:
                raise RuntimeError("owned leader PID reused; refusing cleanup signal")
            try:
                current = identify(row["pid"])
                if current["start_ticks"] != row["start_ticks"]:
                    continue
                if current["session"] != proc.pid and current["pgrp"] != proc.pid:
                    continue
                send(row["pid"], sig)
            except ProcessLookupError:
                pass
        deadline = clock() + duration
        while True:
            proc.poll()
            rows = observe(proc.pid)
            if not rows or clock() >= deadline:
                break
            sleep(min(.2, max(0., deadline - clock())))
        if not rows:
            return []
    return rows


def kernel_evidence() -> list[str]:
    raw = subprocess.check_output(["dmesg", "--color=never"], text=True, timeout=15)
    return [line for line in raw.splitlines() if KERNEL_PATTERN.search(line)]


def pci_identity(value: str) -> tuple[int, int, int, int]:
    found = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if found is None:
        raise ValueError("invalid PCI address")
    return tuple(int(v, 16) for v in found.groups())


def inventory(root: Path = Path("/proc/driver/nvidia/gpus"), expected_count: int = 8) -> dict:
    devices = {}
    for info in sorted(root.glob("*/information")):
        fields = dict(line.split(":", 1) for line in info.read_text().splitlines() if ":" in line)
        uuid, minor = fields["GPU UUID"].strip(), fields["Device Minor"].strip()
        if not uuid.startswith("GPU-") or uuid in devices or not minor.isdecimal():
            raise RuntimeError("ambiguous proc UUID/minor mapping")
        pci_identity(info.parent.name)
        devices[uuid] = {"pci": info.parent.name, "device_path": f"/dev/nvidia{minor}", "minor": int(minor)}
    if len(devices) != expected_count or len({d["minor"] for d in devices.values()}) != expected_count:
        raise RuntimeError("exact eight-adapter proc inventory required")
    return devices


def quantity(value: str, unit: str) -> int:
    if not re.fullmatch(r"\d+ " + re.escape(unit), value):
        raise RuntimeError(f"unavailable or invalid telemetry quantity {value!r}")
    return int(value.split()[0])


def parse_telemetry(raw: str, expected: dict) -> list[dict]:
    gpus = ET.fromstring(raw).findall("gpu")
    if len(gpus) != len(expected) or {g.findtext("uuid") for g in gpus} != set(expected):
        raise RuntimeError("telemetry/proc UUID inventory mismatch")
    rows = []
    for g in gpus:
        uuid = g.findtext("uuid")
        pci = g.findtext("pci/pci_bus_id", "")
        if pci_identity(pci) != pci_identity(expected[uuid]["pci"]):
            raise RuntimeError("telemetry/proc PCI mismatch")
        if g.find("processes") is None:
            raise RuntimeError("GPU process inventory unavailable")
        recovery = g.findtext("gpu_recovery_action", "unavailable")
        if recovery not in ("None", "N/A", "unavailable"):
            raise RuntimeError("driver requests recovery; stop without recovery")
        rows.append({"uuid": uuid, "pci": pci, "minor": expected[uuid]["minor"], "name": g.findtext("product_name"),
                     "used_mib": quantity(g.findtext("fb_memory_usage/used", ""), "MiB"),
                     "free_mib": quantity(g.findtext("fb_memory_usage/free", ""), "MiB"),
                     "gpu_util_percent": quantity(g.findtext("utilization/gpu_util", ""), "%"),
                     "recovery_action": recovery,
                     "processes": [{"pid": int(p.findtext("pid")), "type": p.findtext("type"),
                                    "used_memory": p.findtext("used_memory")} for p in g.findall("processes/process_info")]})
    return rows


def telemetry(expected: dict, out: Path, phase: str) -> list[dict]:
    # One serialized all-adapter query, always inside all-adapter guard lifetime.
    raw = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=30)
    (out / f"{phase}.xml").write_text(raw, encoding="utf-8")
    rows = parse_telemetry(raw, expected)
    atomic(out / f"{phase}.json", rows)
    return rows


def require_idle(rows: list[dict], uuid: str) -> None:
    selected = next((row for row in rows if row["uuid"] == uuid), None)
    if selected is None or selected["processes"] or selected["used_mib"] > 512 or selected["gpu_util_percent"] != 0:
        raise RuntimeError("selected GPU is occupied or has unexplained activity; no worker started")


def storage_directory(path: Path, root: Path) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError("storage paths must be explicit absolute paths")
    root = root.resolve(strict=True)
    path = path.resolve()
    if not path.is_absolute() or root.name != "wuruihan" or path == root or not path.is_relative_to(root):
        raise ValueError("run artifacts must be below verified storage wuruihan root")
    if os.name != "posix" or root.stat().st_dev == Path.home().stat().st_dev:
        raise ValueError("storage must be a separately mounted Linux SSD/HDD volume")
    return path


def worker_environment(repo: Path, out: Path, gpu: str) -> dict:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, WORLD_SIZE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR",
                 "TRITON_CACHE_DIR", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        path = out / "cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def build_identity(config: dict, source: dict, torch_version: str) -> dict:
    return {"source_sha256": source["source_sha256"], "config_sha256": canonical_hash(config),
            "manifest_sha256": file_sha256(Path(config["manifest"]).resolve(strict=True)),
            "init_checkpoint_sha256": config.get("init_checkpoint_sha256"),
            "device_type": "cuda", "torch_version": torch_version}


def fixed_worker(repo: Path, script: Path | None, args_path: Path | None) -> tuple[str, list[str], dict]:
    if script is None:
        if args_path is not None:
            raise ValueError("worker args require the fixed preparation entrypoint")
        return MODE, [sys.executable, "-m", "dynamicwam.dualrate.train"], {}
    path = (repo / script).resolve(strict=True)
    if path != (repo / "scripts/prepare_dualrate_text.py").resolve(strict=True):
        raise ValueError("only the audited prepare_dualrate_text.py worker is allowed")
    if args_path is None:
        raise ValueError("preparation worker requires audited JSON argv")
    arguments = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(arguments, list) or not all(isinstance(v, str) and "\0" not in v for v in arguments):
        raise ValueError("worker args must be a JSON string list")
    forbidden = ("--authorization", "--device", "--config")
    if any(arg == key or arg.startswith(key + "=") for arg in arguments for key in forbidden):
        raise ValueError("runner owns worker config/device/authorization arguments")
    return "data_preparation", [sys.executable, str(path), *arguments], {
        "path": str(path), "worker_sha256": file_sha256(path), "worker_args_sha256": file_sha256(args_path),
        "worker_args_path": str(args_path.resolve(strict=True)), "worker_args": arguments}


def supervise(args, repo: Path, out: Path) -> int:
    from device_lifetime_guard import DeviceLifetimeGuard
    errors, guard, proc, leader_ticks, before, devices = [], None, None, None, None, None
    closed, outcome, lock = True, "failed", None
    began = time.monotonic()
    def persist(name, value):
        try:
            atomic(out / name, value)
        except BaseException as exc:
            errors.append(f"receipt write failed {name}: {exc!r}")
            print(errors[-1], file=sys.stderr, flush=True)
    def status(state, **extra):
        persist("status.json", {"state": state, "supervisor": proc_identity(os.getpid()), "errors": list(errors),
                                "guard_closed": closed, "formal_training_ready": False, "no_retry": True,
                                "elapsed_seconds": time.monotonic() - began, **extra})
    def interrupt(signum, _frame):
        raise InterruptedError(f"job interrupted or timed out by signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupt)
    signal.setitimer(signal.ITIMER_REAL, args.max_runtime)
    status("preparing")
    source = config = frozen = identity = boot = worker = None
    try:
        if not args.inspect:
            source = source_snapshot(repo)
            config = json.loads(args.config.read_text(encoding="utf-8"))
            required = {"manifest", "init_checkpoint_sha256", "output", "storage_root", "run_kind"}
            if required - config.keys():
                raise ValueError(f"missing guarded worker config fields: {sorted(required - config.keys())}")
            if not Path(config["manifest"]).is_absolute():
                raise ValueError("manifest must be absolute")
            storage = Path(config["storage_root"]).resolve(strict=True)
            storage_directory(out, storage)
            storage_directory(Path(config["output"]), storage)
            mode, command, worker = fixed_worker(repo, args.worker_script, args.worker_args_json)
            if config["run_kind"] != mode:
                raise ValueError("config run_kind disagrees with fixed engineering worker")
            frozen = out / "config.frozen.json"
            frozen.write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
            persist("source.json", source)
            persist("worker.json", {"mode": mode, **worker})
            if args.gpu is None or re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu) is None:
                raise ValueError("exact single GPU UUID required")
            import fcntl
            locks = storage / "locks/dualrate_gpu"
            locks.mkdir(parents=True, exist_ok=True)
            lock = (locks / (args.gpu + ".lock")).open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = inventory()
        before = kernel_evidence()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        persist("kernel_before.json", {"boot_id": boot, "lines": before})
        persist("inventory.json", devices)
        guard = DeviceLifetimeGuard(list(devices), lambda events: atomic(out / "guard_events.json", events))
        closed = False
        guard.open()
        status("guard_open")
        rows = telemetry(devices, out, "preflight")
        if kernel_evidence() != before:
            raise RuntimeError("new driver evidence during guarded telemetry")
        if args.inspect:
            outcome = "inspected"
        else:
            require_idle(rows, args.gpu)
            env = worker_environment(repo, out, args.gpu)
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
            version = subprocess.check_output([sys.executable, "-c", "import torch; print(str(torch.__version__))"],
                                              cwd=repo, env=cpu_env, text=True, timeout=60).strip()
            identity = build_identity(config, source, version)
            if source_snapshot(repo) != source or canonical_hash(json.loads(args.config.read_text(encoding="utf-8"))) != identity["config_sha256"]:
                raise RuntimeError("source/config changed during preflight")
            if worker:
                _mode, _command, current_worker = fixed_worker(repo, args.worker_script, args.worker_args_json)
                if current_worker != worker:
                    raise RuntimeError("preparation entrypoint/argv changed during preflight")
            authorization = out / "authorization.json"
            self_proc = proc_identity(os.getpid())
            receipt = {"mode": mode, "authorized": True, "identity": identity, "storage_root": str(storage),
                       "cuda_visible_devices": args.gpu, "expires_at_unix": time.time() + min(600., args.max_runtime),
                       "external_guard": {"pid": os.getpid(), "start_ticks": self_proc["start_ticks"],
                                          "mode": GUARD_MODE, "device_paths": [v["device_path"] for v in devices.values()]},
                       "source": source, "selected_gpu": next(r for r in rows if r["uuid"] == args.gpu),
                       "boot_id": boot, "worker": worker,
                       "expected_worker": ({"path": worker["path"], "sha256": worker["worker_sha256"],
                                            "args_sha256": worker["worker_args_sha256"]} if worker else None)}
            atomic(authorization, receipt)
            if errors:
                raise RuntimeError("required receipt write failed before launch")
            command.extend(["--config", str(frozen), "--device", "cuda:0", "--authorization", str(authorization)])
            with (out / "worker.log").open("x", encoding="utf-8") as log:
                proc = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                leader_ticks = proc_identity(proc.pid)["start_ticks"]
                status("running", worker_pid=proc.pid, worker_start_ticks=leader_ticks, mode=mode, identity=identity)
                rc = proc.wait(timeout=max(.01, args.max_runtime - (time.monotonic() - began)))
            if rc:
                raise RuntimeError(f"worker exited {rc}; no retry")
            if not authorization.with_suffix(".json.consumed").exists():
                raise RuntimeError("worker never consumed guarded authorization")
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
                remaining = cleanup_owned(proc, leader_ticks)
            except BaseException as exc:
                errors.append(f"cleanup probe/signal failed: {exc!r}")
                remaining = None
            if remaining is None or remaining:
                outcome = "failed"
                errors.append("owned worker exit unconfirmed; only required cleanup supervisor remains")
                status("cleanup_pending", owned_pgid=proc.pid, remaining=remaining, needs_manual_attention=True)
                # No GPU query, restart, model work, or recovery in this exception.
                while True:
                    try:
                        proc.poll()
                        if not live_owned_members(proc.pid):
                            break
                    except BaseException as exc:
                        print(f"cleanup exit unconfirmed: {exc!r}", file=sys.stderr, flush=True)
                    time.sleep(1)
            persist("owned_exit.json", {"owned_pgid": proc.pid, "remaining": [], "returncode": proc.poll()})
        if guard is not None:
            if devices is not None and len(guard.handles) == len(devices) and proc is not None:
                try:
                    post = telemetry(devices, out, "postflight")
                    require_idle(post, args.gpu)
                except BaseException as exc:
                    errors.append(f"post-worker telemetry failed: {exc!r}")
            try:
                guard.close()
                closed = not guard.handles
            except BaseException as exc:
                errors.append(f"guard closure unconfirmed: {exc!r}")
        if before is not None:
            try:
                after = kernel_evidence()
                same_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
                persist("kernel_after.json", {"lines": after, "unchanged": after == before, "same_boot": same_boot,
                                              "added": list((collections.Counter(after) - collections.Counter(before)).elements())})
                if after != before or not same_boot:
                    errors.append("kernel evidence or boot changed; clean exit not accepted")
            except BaseException as exc:
                errors.append(f"post-closure driver evidence unavailable: {exc!r}")
        if source is not None:
            try:
                if source_snapshot(repo) != source:
                    errors.append("source/config/data identity changed during worker lifetime")
                if identity is not None and (build_identity(config, source, identity["torch_version"]) != identity
                        or canonical_hash(json.loads(args.config.read_text(encoding="utf-8"))) != identity["config_sha256"]
                        or canonical_hash(json.loads(frozen.read_text(encoding="utf-8"))) != identity["config_sha256"]):
                    errors.append("config/data identity changed during worker lifetime")
                if worker and fixed_worker(repo, args.worker_script, args.worker_args_json)[2] != worker:
                    errors.append("preparation worker/argv changed during worker lifetime")
            except BaseException as exc:
                errors.append(f"final source binding unavailable: {exc!r}")
        if lock is not None:
            lock.close()
        if errors or not closed:
            outcome = "failed"
        status(outcome, identity=identity)
    return 0 if outcome in ("completed", "inspected") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, default=Path("/SSD_DISK/users/wuruihan"))
    parser.add_argument("--gpu")
    parser.add_argument("--max-runtime", type=float, default=3600., help="seconds, excluding necessary cleanup_pending exception")
    parser.add_argument("--worker-script", type=Path)
    parser.add_argument("--worker-args-json", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--supervisor", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix" or not 0 < args.max_runtime <= 86400:
        parser.error("Linux and max-runtime in (0,86400] seconds required")
    if args.inspect and (args.config or args.worker_script or args.worker_args_json):
        parser.error("inspect cannot launch a configured worker")
    if not args.inspect and args.config is None:
        parser.error("config required for engineering worker")
    repo = Path(__file__).resolve().parents[1]
    if args.config is not None:
        args.config = args.config.resolve(strict=True)
    if args.worker_args_json is not None:
        args.worker_args_json = args.worker_args_json.resolve(strict=True)
    out = storage_directory(args.run_dir, args.storage_root)
    if not args.supervisor:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or any(p.name != "supervisor.log" for p in out.iterdir()):
        parser.error("internal detached supervisor requires its fresh launch directory")
    if args.detach and not args.supervisor:
        argv = [arg for arg in sys.argv[1:] if arg != "--detach"] + ["--supervisor"]
        with (out / "supervisor.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv], cwd=Path.cwd(),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"state": "dispatched", "supervisor_pid": child.pid, "run_dir": str(out)}), flush=True)
        return 0
    return supervise(args, repo, out)


if __name__ == "__main__":
    raise SystemExit(main())
