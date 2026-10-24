"""One-shot eight-rank stdlib launcher; all ranks retain the guarded session.

Unlike torchrun's elastic subprocess handler, this launcher never calls setsid
for individual ranks. The external guard therefore owns every rank through its
existing session cleanup. There are no restarts or arbitrary worker commands.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

WORLD_SIZE = 8


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], session=int(fields[3]), pgrp=int(fields[2]))


def rendezvous_port():
    # The CPU-only launcher cannot keep this socket bound while rank zero's
    # TCPStore binds it. A rare intervening collision fails the whole group;
    # neither this launcher nor the guard retries with another port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fixed_command(config, authorization, resume=None):
    argv = [sys.executable, "-B", "-m", "dynamicwam.dualrate.train", "--config", str(config),
            "--device", "cuda", "--authorization", str(authorization)]
    if resume is not None:
        argv += ["--resume", str(resume)]
    return argv


def _signal_owned(process, identity, sig, owned_session):
    if process.poll() is not None:
        return
    try:
        current = process_identity(process.pid)
    except FileNotFoundError:
        process.poll()
        return
    if current != identity or current["session"] != owned_session:
        raise RuntimeError("rank identity/session changed; refusing to signal a different process")
    process.send_signal(sig)


def cleanup_ranks(children, owned_session, *, grace_seconds=20., poll_seconds=.05):
    """Only signal exact children; retain the launcher until all children exit."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process, identity in children:
            _signal_owned(process, identity, sig, owned_session)
        deadline = time.monotonic()+grace_seconds
        while any(process.poll() is None for process, _ in children):
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_seconds)
        if all(process.poll() is not None for process, _ in children):
            return
    # A D-state child cannot be repaired by sending more signals or closing the
    # GPU guard. Necessary cleanup remains alive; the outer guard owns this SID.
    while any(process.poll() is None for process, _ in children):
        time.sleep(poll_seconds)


def run_rank_group(config, authorization, resume=None, *, popen_factory=subprocess.Popen,
                   environment=None, emit=None, poll_seconds=.1, grace_seconds=20.):
    if os.name != "posix":
        raise RuntimeError("Linux process/session ownership required")
    emit = emit or (lambda value: print(json.dumps(value), flush=True))
    env = dict(os.environ if environment is None else environment)
    for key in list(env):
        if key.startswith("TORCHELASTIC_"):
            del env[key]
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(rendezvous_port()),
               WORLD_SIZE=str(WORLD_SIZE), LOCAL_WORLD_SIZE=str(WORLD_SIZE), GROUP_RANK="0")
    children, identities = [], []
    owned_session = os.getsid(0)
    previous_handlers = {}
    error = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"eight-rank launcher interrupted: {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        emit(dict(event="rank_group_start", pid=os.getpid(), session=owned_session, world_size=WORLD_SIZE,
                  master_addr=env["MASTER_ADDR"], master_port=int(env["MASTER_PORT"]), automatic_retry=False))
        for rank in range(WORLD_SIZE):
            rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), ROLE_RANK=str(rank))
            process = popen_factory(fixed_command(config, authorization, resume), env=rank_env,
                                    stdin=subprocess.DEVNULL, start_new_session=False)
            # Capture the PID/start identity before any subsequent fork. Even a
            # child that exits immediately remains waitable as a zombie here.
            identity = process_identity(process.pid)
            children.append((process, identity))
            identities.append(dict(rank=rank, **identity))
            if identity["session"] != owned_session:
                raise RuntimeError("rank escaped the external guard's owned session")
            emit(dict(event="rank_started", **identities[-1]))
        while True:
            codes = [process.poll() for process, _ in children]
            failed = [(rank, code) for rank,code in enumerate(codes) if code not in (None, 0)]
            if failed:
                raise RuntimeError(f"rank failure {failed}; stopping owned siblings without retry")
            if all(code == 0 for code in codes):
                break
            time.sleep(poll_seconds)
    except BaseException as exc:
        error = exc
    finally:
        # Repeated external termination cannot interrupt required child cleanup.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup_ranks(children, owned_session, grace_seconds=grace_seconds, poll_seconds=poll_seconds)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    result = dict(event="rank_group_completed" if error is None else "rank_group_failed",
                  passed=error is None, world_size=WORLD_SIZE, session=owned_session,
                  ranks=[dict(**identity, returncode=process.returncode)
                         for identity,(process,_) in zip(identities, children)],
                  automatic_retry=False)
    if error is not None:
        result["error"] = repr(error)
    emit(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    authorization = args.authorization.resolve(strict=True)
    settings = json.loads(config.read_text())
    if settings.get("world_size") != WORLD_SIZE or settings.get("run_kind") not in {
            "distributed_training_smoke", "formal_distributed_training"}:
        parser.error("fixed launcher supports only explicit eight-rank training")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != WORLD_SIZE or len(set(visible)) != WORLD_SIZE or any(not v.startswith("GPU-") for v in visible):
        parser.error("exact eight UUID-pinned adapters required")
    resume = args.resume.resolve(strict=True) if args.resume is not None else None
    return 0 if run_rank_group(config, authorization, resume)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
