#!/usr/bin/env python3
"""One-shot SSD training supervisor with a bounded startup health window.

Use --preflight-only to inspect gates, or --detach to leave a durable supervisor.
Formal runs require an independently verified smoke receipt. Known historical
assertions require explicit, evidence-bound authorization; authorization is not a
claim that a driver was repaired. Never creates clearance, repairs, or retries.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

# Do not leave import caches in the home-directory source checkout.
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def referenced_receipt(reference):
    if not isinstance(reference, dict) or not reference.get("path"):
        raise ValueError("receipt requires path and sha256")
    path = Path(reference["path"]).resolve(strict=True)
    if file_digest(path) != reference.get("sha256"):
        raise ValueError(f"receipt content hash mismatch: {path}")
    return json_file(path)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(argv, *, cwd=None, timeout=30):
    result = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"read-only command failed: {argv[0]}: {result.stderr.strip()}")
    return result.stdout


def inside_ssd(value, ssd, *, must_exist=False):
    path = Path(value).expanduser().resolve()
    if path != ssd and ssd not in path.parents:
        raise ValueError(f"path escapes verified SSD root: {path}")
    if must_exist and not path.exists():
        raise ValueError(f"required SSD path does not exist: {path}")
    return path


def verify_ssd(root, minimum_free_gib):
    root = root.expanduser().resolve(strict=True)
    if "wuruihan" not in root.parts or not root.is_dir():
        raise ValueError("SSD root must be an existing wuruihan directory")
    if root.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("SSD root resolves to the system filesystem")
    mount = json.loads(command(["findmnt", "--json", "--target", str(root)]))["filesystems"][0]
    source = mount["source"].split("[")[0]
    rota = command(["lsblk", "-dn", "-o", "ROTA", source]).strip().splitlines()
    filesystem_uuid = command(["lsblk", "-dn", "-o", "UUID", source]).strip()
    # .50 exposes the existing project SSD through a hardware RAID logical
    # volume whose ROTA flag is 1. ROTA cannot identify its member media. Permit
    # only this previously designated mount identity, never an arbitrary ROTA=1
    # disk or a fallback path, and preserve the physical-media uncertainty.
    known_project_raid = (mount.get("target") == "/SSD_DISK" and
        filesystem_uuid == "efa6ee12-2220-4348-800a-1747091b1672" and
        root == Path("/SSD_DISK/users/wuruihan/DynamicWAM"))
    if not rota or (any(value.strip() != "0" for value in rota) and not known_project_raid):
        raise ValueError("storage is neither non-rotational nor the established project RAID mount")
    usage, fs = shutil.disk_usage(root), os.statvfs(root)
    if usage.free < minimum_free_gib * 1024**3 or fs.f_favail < 10000:
        raise ValueError("insufficient SSD free bytes or inodes")
    if not os.access(root, os.W_OK):
        raise PermissionError("SSD wuruihan root is not writable")
    return root, dict(root=str(root), mount=mount, filesystem_uuid=filesystem_uuid,
        free_bytes=usage.free, free_inodes=fs.f_favail, rota=rota,
        physical_media="not independently observable behind hardware RAID" if known_project_raid else "ROTA=0",
        basis="existing accepted Route1 artifact storage" if known_project_raid else "block-device flag")


def driver_evidence():
    kernel = command(["dmesg", "--color=never"])
    pattern = re.compile(r"NVRM.*(?:assert|error|Xid|RmInitAdapter|IOVAS|iovaspace)|Xid\s*\(|fallen off the bus|GPU.*(?:fault|error)|AER.*\bFatal\b|PCIe Bus Error:.*\bFatal\b", re.I)
    anomalies = [line.strip() for line in kernel.splitlines() if pattern.search(line)]
    latest = anomalies[-1] if anomalies else None
    return dict(host=socket.gethostname(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        latest_driver_anomaly=latest, latest_driver_anomaly_sha256=digest(latest.encode()) if latest else None,
        anomaly_count=len(anomalies), driver_anomalies_sha256=digest("\n".join(anomalies).encode()),
        dmesg_tail=kernel.splitlines()[-200:])


def driver_identity(evidence):
    return tuple(evidence.get(k) for k in (
        "host", "boot_id", "anomaly_count", "driver_anomalies_sha256",
        "latest_driver_anomaly_sha256"))


def validate_clearance(evidence, clearance_path, selected_uuids):
    """Read ONLY; no mechanism in this launcher produces/updates clearance."""
    if evidence["latest_driver_anomaly"] is None:
        return None
    if clearance_path is None:
        raise RuntimeError("driver anomaly blocks launch; latest SHA256=" + evidence["latest_driver_anomaly_sha256"])
    clearance = json_file(clearance_path)
    for key in ("host", "boot_id", "driver_anomalies_sha256", "latest_driver_anomaly_sha256"):
        if clearance.get(key) != evidence[key]:
            raise RuntimeError(f"health clearance does not match current {key}; new/changed anomaly blocks launch")
    if sorted(clearance.get("selected_gpu_uuids", [])) != sorted(selected_uuids):
        raise RuntimeError("authorization must cover exactly the selected GPU UUIDs")
    if not isinstance(clearance.get("reason"), str) or not clearance["reason"].strip():
        raise RuntimeError("authorization requires a reason")
    if not isinstance(clearance.get("approved_by"), str) or not clearance["approved_by"].strip():
        raise RuntimeError("authorization must identify its human source")
    kind = clearance.get("authorization_kind")
    if kind == "admin_health_clearance":
        if clearance.get("healthy_to_start") is not True:
            raise RuntimeError("administrator health clearance requires healthy_to_start=true")
    elif kind == "user_authorized_controlled_start":
        if clearance.get("driver_resolved") is not False or clearance.get("authorized_to_start") is not True:
            raise RuntimeError("controlled start must acknowledge unresolved history and explicit authorization")
        diagnostic = referenced_receipt(clearance.get("diagnostic_receipt"))
        if diagnostic.get("passed") is not True:
            raise RuntimeError("controlled start requires an actually passed diagnostic")
        for key in ("boot_id", "driver_anomalies_sha256", "latest_driver_anomaly_sha256"):
            if diagnostic.get(key) != evidence[key]:
                raise RuntimeError(f"diagnostic does not cover current {key}")
        tested = set(diagnostic.get("gpu_uuids", []))
        if not tested or tested != set(selected_uuids):
            raise RuntimeError("diagnostic UUIDs must exactly cover authorized devices")
        # Every selected device must have an actual diagnostic result.
        clearance["diagnostically_tested_gpu_uuids"] = sorted(tested)
    else:
        raise RuntimeError("explicit user_authorized_controlled_start or admin_health_clearance required")
    return dict(path=str(clearance_path.resolve()), sha256=digest(clearance_path.read_bytes()), value=clearance)


def process_identity(pid):
    """Read Linux process identity without creating processes or signalling it."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm can contain spaces and parentheses; fields after the final ')' are
    # state(3), ppid(4), pgrp(5), ... starttime(22).
    fields = text.rsplit(")", 1)[1].split()
    return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
        pgrp=int(fields[2]), starttime=int(fields[19]))


def training_descendant(pid, root_pid, *, reader=process_identity):
    """True for the exact live torchrun tree, False for outsiders, None if gone.

    Elastic workers may create separate process groups. Membership is therefore
    proved through /proc ancestry, never through a shared username or command.
    The supervisor owns/reaps root_pid, preventing its reuse while Popen is live.
    """
    original = reader(pid)
    root = reader(root_pid)
    if original is None or original["state"] == "Z" or root is None:
        return None
    current, seen = original, set()
    for _ in range(256):
        if current["pid"] in seen:
            break
        seen.add(current["pid"])
        if current["pid"] == root_pid:
            again, root_again = reader(pid), reader(root_pid)
            if again is None or root_again is None or again["state"] == "Z":
                return None
            return (again["starttime"] == original["starttime"] and
                root_again["starttime"] == root["starttime"])
        if current["ppid"] <= 1:
            break
        current = reader(current["ppid"])
        if current is None:
            # CUDA/NVML can still report an already exiting process. Only
            # excuse disappearance of that exact PID, not an unknown ancestor.
            return None if reader(pid) is None else False
    return None if reader(pid) is None else False


def gpu_preflight(uuids, expected_count, max_temp_c, max_used_mib, *, own_pgid=None):
    columns = "uuid,pci.bus_id,name,temperature.gpu,memory.used,memory.total,power.draw,power.limit"
    rows = list(csv.reader(io.StringIO(command(["nvidia-smi", "--query-gpu=" + columns, "--format=csv,noheader,nounits"]))))
    names = columns.split(",")
    devices = [{k: v.strip() for k, v in zip(names, row)} for row in rows]
    if len(devices) != expected_count:
        raise RuntimeError(f"visible GPU count {len(devices)} differs from expected {expected_count}")
    by_uuid = {row["uuid"]: row for row in devices}
    if any(uuid not in by_uuid for uuid in uuids):
        raise RuntimeError("requested UUID missing from nvidia-smi")
    process_rows = csv.reader(io.StringIO(command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])))
    processes = []
    for row in process_rows:
        if len(row) != 4:
            continue
        item = dict(zip(("gpu_uuid", "pid", "process_name", "used_gpu_memory"), [v.strip() for v in row]))
        identity = process_identity(int(item["pid"]))
        if identity is None or identity["state"] == "Z":
            continue
        # /proc can disappear between the NVML snapshot and this read.
        try:
            import pwd
            item["owner"] = pwd.getpwuid(Path(f"/proc/{item['pid']}").stat().st_uid).pw_name
        except (FileNotFoundError, ProcessLookupError):
            continue
        processes.append(item)
    for uuid in uuids:
        gpu = by_uuid[uuid]
        if float(gpu["temperature.gpu"]) >= max_temp_c or (max_used_mib is not None and float(gpu["memory.used"]) > max_used_mib):
            raise RuntimeError(f"GPU {uuid} exceeds explicit temperature/memory launch limit")
        for item in processes:
            if item["gpu_uuid"] == uuid:
                relation = training_descendant(int(item["pid"]), own_pgid) if own_pgid is not None else False
                item["training_ancestry"] = "owned" if relation else "exited" if relation is None else "external"
                if relation is False:
                    raise RuntimeError(f"GPU {uuid} has a compute process outside this run; no unrelated process is stopped")
    return dict(devices=devices, processes=processes, selected_uuids=uuids,
        temperature_limit_c=max_temp_c, used_memory_limit_mib=max_used_mib)


def source_files_and_hash(repo):
    names = command(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo).split("\0")
    files = sorted({name for name in names if name and (name.startswith("src/") or name.startswith("scripts/")) and name.endswith(".py")})
    sha = hashlib.sha256()
    for name in files:
        sha.update(name.encode() + b"\0" + (repo / name).read_bytes() + b"\0")
    return files, sha.hexdigest()


def validate_smoke(path, config_sha256, source_sha256, ssd):
    if path is None:
        raise ValueError("formal training requires --smoke-receipt")
    data = json_file(path)
    for key in ("passed", "gpu_smoke", "checkpoint_reload"):
        if data.get(key) is not True:
            raise ValueError(f"smoke receipt missing verified {key}=true")
    for key, expected in (("formal_config_sha256", config_sha256), ("source_sha256", source_sha256)):
        if data.get(key) != expected:
            raise ValueError(f"smoke receipt does not cover current {key}")
    smoke_run = inside_ssd(data["smoke_run_dir"], ssd, must_exist=True)
    manifest_path, status_path = smoke_run / "manifest.json", smoke_run / "status.json"
    for file, key in ((manifest_path, "smoke_manifest_sha256"), (status_path, "smoke_status_sha256")):
        if file_digest(file) != data.get(key):
            raise ValueError(f"smoke receipt does not match source {key}")
    manifest, status = json_file(manifest_path), json_file(status_path)
    if manifest.get("mode") != "smoke" or manifest.get("source_sha256") != source_sha256:
        raise ValueError("receipt source is not a matching smoke run")
    if status.get("state") != "process_exited" or status.get("exit_code") != 0 or status.get("stop_reason"):
        raise ValueError("smoke source did not exit cleanly without supervisor termination")
    checkpoint = inside_ssd(data["checkpoint_path"], ssd, must_exist=True)
    output = Path(manifest["paths"]["training_output"]).resolve()
    if output not in checkpoint.parents or file_digest(checkpoint) != data.get("checkpoint_sha256"):
        raise ValueError("smoke checkpoint identity/location mismatch")
    reload = referenced_receipt(data.get("checkpoint_reload_receipt"))
    if reload.get("passed") is not True or reload.get("checkpoint_sha256") != data["checkpoint_sha256"]:
        raise ValueError("checkpoint reload evidence does not validate this checkpoint")
    if reload.get("source_sha256") != source_sha256:
        raise ValueError("checkpoint reload evidence used different code")
    return dict(path=str(path.resolve()), sha256=digest(path.read_bytes()))


def startup_health(args, baseline, pgid):
    driver = driver_evidence()
    if driver_identity(driver) != driver_identity(baseline):
        error = RuntimeError("driver evidence changed since authorized baseline (new anomaly or log change)")
        error.driver_evidence = driver
        raise error
    gpu = gpu_preflight(args.uuids, args.expected_gpus, args.max_temp_c, None, own_pgid=pgid)
    return dict(timestamp=utc(), driver_identity=driver_identity(driver), gpu=gpu)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--ssd-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpu-uuids", required=True, help="comma-separated exact GPU UUIDs")
    p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--max-temp-c", type=float, required=True, help="explicit hardware/platform launch temperature limit")
    p.add_argument("--max-used-mib", type=float, default=100)
    p.add_argument("--expected-gpus", type=int, default=8)
    p.add_argument("--min-free-gib", type=float, default=20)
    p.add_argument("--health-clearance", type=Path)
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--startup-health-seconds", type=float, default=600,
        help="bounded initial read-only GPU/driver checking window; none afterwards")
    p.add_argument("--health-check-interval", type=float, default=30)
    p.add_argument("--max-runtime-seconds", type=float, default=0,
        help="positive value sends TERM to this run's process group at the deadline")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--detach", action="store_true")
    a = p.parse_args()
    a.uuids = [v.strip() for v in a.gpu_uuids.split(",") if v.strip()]
    if not a.uuids or len(set(a.uuids)) != len(a.uuids) or any(not v.startswith("GPU-") for v in a.uuids):
        p.error("unique GPU UUIDs required")
    if not 0 < a.max_temp_c < 100 or a.max_used_mib < 0 or a.expected_gpus < len(a.uuids):
        p.error("invalid explicit GPU launch limits")
    if a.detach and a.preflight_only:
        p.error("--detach and --preflight-only are mutually exclusive")
    if a.startup_health_seconds < 0 or a.health_check_interval <= 0 or a.max_runtime_seconds < 0:
        p.error("invalid monitoring/deadline durations")
    return a


def main():
    args = parse_args()
    if os.name != "posix":
        raise RuntimeError("supervisor requires the Linux training host")
    ssd, storage = verify_ssd(args.ssd_root, args.min_free_gib)
    run = inside_ssd(args.run_dir, ssd)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "manifest.json").exists() or (run / "status.json").exists():
        raise RuntimeError("run directory already used; select a new run_id")
    if args.detach:
        argv = [sys.executable, str(Path(__file__).resolve()), *[v for v in sys.argv[1:] if v != "--detach"]]
        with (run / "supervisor.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid, run_dir=str(run), state="dispatched_not_preflight_verified")))
        return 0
    status = dict(state="preflight", timestamp=utc(), supervisor_pid=os.getpid(), mode=args.mode)
    atomic_json(run / "status.json", status)
    process = None
    stop = {"signal": None}
    def request_stop(signum, _frame):
        stop["signal"] = signum
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        repo, config = args.repo.resolve(strict=True), args.config.resolve(strict=True)
        harness = repo / "harness.md"
        if not harness.is_file():
            raise RuntimeError("repository harness.md is required")
        sys.path.insert(0, str(repo / "src"))
        from dynamicwam.config.multirate import load_multirate_profile
        import yaml
        profile = load_multirate_profile(config)
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        if not Path(raw["paths"]["project_root"]).is_absolute():
            raise ValueError("use absolute paths.project_root so the frozen config has unchanged semantics")
        if Path(raw["paths"]["project_root"]).resolve() != repo:
            raise ValueError("config project root and supervisor repo differ")
        training = profile.training
        paths = {"packed_dataset": training["dataset"]["root"], "source_dataset": training["dataset"]["source_root"],
            "wan_root": training["model"]["compact_wan"]["checkpoint_path"],
            "init_checkpoint": training["model"]["initial_checkpoint"],
            "teacher_checkpoint": training["model"]["native_teacher_checkpoint"]}
        if training.get("resume_from"):
            paths["resume_from"] = training["resume_from"]
        resolved = {key: str(inside_ssd(value, ssd, must_exist=True)) for key, value in paths.items() if value}
        output = inside_ssd(Path(training["checkpoint_dir"]) / str(training["name"]), ssd)
        if output.exists() and not training.get("resume_from"):
            raise RuntimeError("training output already exists; choose a fresh training.name")
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved["training_output"] = str(output)
        cache, temporary = ssd / "caches/motionkv", ssd / "tmp"
        env = os.environ.copy()
        env_paths = {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface/hub",
            "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub", "TRANSFORMERS_CACHE": cache / "huggingface/transformers",
            "TORCH_HOME": cache / "torch", "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
            "TORCH_EXTENSIONS_DIR": cache / "torch_extensions", "HF_DATASETS_CACHE": cache / "huggingface/datasets",
            "TRITON_CACHE_DIR": cache / "triton", "CUDA_CACHE_PATH": cache / "cuda",
            "XDG_CACHE_HOME": cache / "xdg", "WANDB_DIR": run / "wandb", "WANDB_CACHE_DIR": cache / "wandb",
            "TMPDIR": temporary, "TMP": temporary, "TEMP": temporary}
        for key, path in env_paths.items():
            actual = inside_ssd(path, ssd)
            actual.mkdir(parents=True, exist_ok=True)
            env[key] = str(actual)
        env.update(CUDA_VISIBLE_DEVICES=",".join(args.uuids), PYTHONUNBUFFERED="1", PYTHONPATH=str(repo / "src"),
            WANDB_MODE="disabled", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
            PYTHONDONTWRITEBYTECODE="1", PYTORCH_NVML_BASED_CUDA_CHECK="1",
            ACCELERATE_USE_DEEPSPEED="true", ACCELERATE_MIXED_PRECISION="bf16")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        config_hash = digest(config.read_bytes())
        source_files, source_hash = source_files_and_hash(repo)
        smoke = validate_smoke(args.smoke_receipt, config_hash, source_hash, ssd) if args.mode == "formal" else None
        driver = driver_evidence()
        atomic_json(run / "driver_preflight.json", driver)
        clearance = validate_clearance(driver, args.health_clearance, args.uuids)
        if clearance and clearance["value"]["authorization_kind"] == "user_authorized_controlled_start" and args.startup_health_seconds <= 0:
            raise ValueError("controlled start requires a nonzero bounded startup health window")
        gpu = gpu_preflight(args.uuids, args.expected_gpus, args.max_temp_c, args.max_used_mib)
        frozen = run / "training_config.yaml"
        shutil.copyfile(config, frozen)
        argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={len(args.uuids)}", str(repo / "scripts/train_multirate.py"), "--config", str(frozen)]
        manifest = dict(created_at=utc(), host=socket.gethostname(), repo=str(repo), mode=args.mode,
            git_head=command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
            git_branch=command(["git", "branch", "--show-current"], cwd=repo).strip(),
            harness_sha256=digest(harness.read_bytes()), route="Route1",
            git_status=command(["git", "status", "--porcelain"], cwd=repo), source_sha256=source_hash,
            config_path=str(config), config_sha256=config_hash, frozen_config=str(frozen), command=argv,
            supervisor_pid=os.getpid(), paths=resolved, storage=storage, gpu_preflight=gpu,
            driver_preflight_file=str(run / "driver_preflight.json"), health_clearance=clearance,
            smoke_receipt=smoke, environment={k: env[k] for k in (*env_paths, "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "WANDB_MODE", "PYTORCH_NVML_BASED_CUDA_CHECK", "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION", "PATH")},
            retry_policy="none", startup_health_seconds=args.startup_health_seconds,
            health_check_interval=args.health_check_interval, max_runtime_seconds=args.max_runtime_seconds,
            postwindow_gpu_polling=False, termination_policy="TERM own registered process group only")
        with tarfile.open(run / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in source_files:
                archive.add(repo / name, arcname=name, recursive=False)
        (run / "source.patch").write_text(command(["git", "diff", "--binary", "HEAD"], cwd=repo), encoding="utf-8")
        atomic_json(run / "manifest.json", manifest)
        if args.preflight_only:
            status.update(state="preflight_passed_not_started", timestamp=utc())
            atomic_json(run / "status.json", status)
            print(json.dumps(status))
            return 0
        # One final driver comparison prevents accepting a clearance for a newer
        # anomaly appearing during configuration/source checks.
        now = driver_evidence()
        if driver_identity(now) != driver_identity(driver):
            atomic_json(run / "driver_changed.json", now)
            raise RuntimeError("new driver anomaly since startup preflight; training not started")
        if stop["signal"]:
            raise RuntimeError("launch cancelled before training process creation")
        import fcntl
        with ExitStack() as stack:
            for uuid in sorted(args.uuids):
                lease = stack.enter_context((temporary / f"motionkv-training-{uuid}.lock").open("a+"))
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (run / "train.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pgid = os.getpgid(process.pid)
                manifest.update(training_pid=process.pid, training_pgid=pgid, started_at=utc())
                atomic_json(run / "manifest.json", manifest)
                status.update(state="running", training_pid=process.pid, training_pgid=pgid)
                started = time.monotonic()
                last_heartbeat, stop_sent = 0., None
                next_health = started + min(args.health_check_interval, args.startup_health_seconds)
                status["startup_health_window_seconds"] = args.startup_health_seconds
                status["stop_reason"] = None
                while process.poll() is None:
                    elapsed = time.monotonic()
                    runtime = elapsed - started
                    if args.max_runtime_seconds > 0 and runtime >= args.max_runtime_seconds and not stop["signal"]:
                        stop["signal"] = signal.SIGTERM
                        status["stop_reason"] = "max_runtime_seconds_exceeded"
                    if not stop["signal"] and args.startup_health_seconds > 0 and runtime <= args.startup_health_seconds and elapsed >= next_health:
                        try:
                            health = startup_health(args, driver, pgid)
                            atomic_json(run / "startup_health_latest.json", health)
                            status["last_startup_health_at"] = health["timestamp"]
                        except Exception as health_error:
                            status["stop_reason"] = "startup_health_failed"
                            status["health_error"] = f"{type(health_error).__name__}: {health_error}"
                            atomic_json(run / "startup_health_failure.json", dict(timestamp=utc(), error=status["health_error"],
                                driver_evidence=getattr(health_error, "driver_evidence", None)))
                            stop["signal"] = signal.SIGTERM
                        next_health = time.monotonic() + args.health_check_interval
                    status["startup_health_window_closed"] = runtime > args.startup_health_seconds
                    if stop["signal"] and stop_sent is None:
                        if os.getpgid(process.pid) != pgid or pgid != process.pid:
                            raise RuntimeError("refusing to signal a process group with changed ownership")
                        os.killpg(pgid, signal.SIGTERM)
                        stop_sent = elapsed
                        status["state"] = "stopping_own_training_group"
                        status["stop_reason"] = status["stop_reason"] or "supervisor_signal"
                    if elapsed - last_heartbeat >= 30:
                        status.update(timestamp=utc(), log_bytes=(run / "train.log").stat().st_size,
                            stop_signal=stop["signal"], runtime_seconds=runtime)
                        atomic_json(run / "status.json", status)
                        last_heartbeat = elapsed
                    time.sleep(1)
                code = process.wait()
        status.update(state="process_exited", timestamp=utc(), exit_code=code,
            training_success_not_inferred_from_exit_alone=True)
        atomic_json(run / "status.json", status)
        print(json.dumps(status))
        if status.get("stop_reason") == "max_runtime_seconds_exceeded":
            return 124
        if status.get("stop_reason"):
            return 2
        return code if code >= 0 else 128 - code
    except Exception as exc:
        if process is not None and process.poll() is None:
            # Only the exact process group created and registered above.
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        status["own_group_termination_pending"] = True
            except ProcessLookupError:
                pass
        status.update(state="blocked_before_training" if process is None else "supervisor_error",
            timestamp=utc(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(run / "status.json", status)
        print(json.dumps(status), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
