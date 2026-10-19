#!/usr/bin/env python3
"""Read-only kernel-log broker for evaluation; never changes drivers or devices."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

MAX_REQUEST = 4096
MAX_RESPONSE = 16 * 1024 * 1024
ANOMALY_PATTERN = re.compile(
    r"NVRM.*(?:assert|error|Xid|RmInitAdapter|IOVAS|iovaspace)|Xid\s*\(|fallen off the bus|GPU.*(?:fault|error)|AER.*\bFatal\b|PCIe Bus Error:.*\bFatal\b", re.I)
DMESG_ARGV = ["/usr/bin/dmesg", "--color=never"]


def _kernel_text(timeout=30):
    result = subprocess.run(DMESG_ARGV, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout, check=False,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"})
    if result.returncode:
        raise RuntimeError("read-only command failed: dmesg: " + result.stderr.strip())
    return result.stdout


def _socket_path(value, uid=None):
    raw = os.fspath(value)
    match = re.fullmatch(r"/run/route1-kernel-([0-9]+)-([A-Za-z0-9][A-Za-z0-9_-]{0,48})\.sock", raw)
    if not match or (uid is not None and int(match.group(1)) != uid):
        raise ValueError("socket must be /run/route1-kernel-<uid>-<runid>.sock")
    return Path(raw)


def _peer_uid(conn):
    return struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET,
        socket.SO_PEERCRED, struct.calcsize("3i")))[1]


def _read_line(conn, limit):
    data = bytearray()
    budget = conn.gettimeout()
    deadline = time.monotonic() + budget if budget is not None else None
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("broker message deadline exceeded")
            conn.settimeout(remaining)
        block = conn.recv(min(65536, limit + 1 - len(data)))
        if not block:
            raise ValueError("incomplete broker message")
        data.extend(block)
        if len(data) > limit:
            raise ValueError("broker message too large")
        if b"\n" in block:
            line, remainder = bytes(data).split(b"\n", 1)
            if remainder:
                raise ValueError("multiple broker messages are forbidden")
            return json.loads(line)


def _send(conn, value):
    encoded = (json.dumps(value, ensure_ascii=True) + "\n").encode()
    if len(encoded) > MAX_RESPONSE:
        raise ValueError("kernel log exceeds broker response limit")
    conn.sendall(encoded)


def _handle_connection(conn, allowed_uid, reader=_kernel_text, *, timeout=35):
    """Only read/stop opcodes, with Linux kernel supplied peer credentials."""
    conn.settimeout(timeout)
    try:
        if _peer_uid(conn) not in (0, allowed_uid):
            raise PermissionError("peer uid not authorized")
        request = _read_line(conn, MAX_REQUEST)
        if not isinstance(request, dict) or set(request) != {"op"}:
            raise ValueError("request must contain only an op")
        if request["op"] == "read":
            _send(conn, {"ok": True, "kernel": reader()})
            return False
        if request["op"] == "stop":
            _send(conn, {"ok": True, "stopping": True})
            return True
        raise ValueError("unsupported broker opcode")
    except (ValueError, PermissionError, RuntimeError, subprocess.TimeoutExpired,
            OSError, UnicodeError) as error:
        try:
            _send(conn, {"ok": False, "error": str(error)})
        except (OSError, ValueError):
            pass
        return False


def _request(value, op, timeout=40, *, expected_server_uid=0):
    path = _socket_path(value)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 120:
        raise ValueError("client timeout must be in (0,120]")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(str(path))
        if _peer_uid(conn) != expected_server_uid:
            raise PermissionError("kernel broker must be owned by root")
        _send(conn, {"op": op})
        result = _read_line(conn, MAX_RESPONSE)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("kernel broker rejected request: " +
                           str(result.get("error") if isinstance(result, dict) else "invalid response"))
    return result


def read_kernel_log(timeout=40):
    path = os.environ.get("WAM_KERNEL_LOG_SOCKET")
    if not path:
        return _kernel_text()
    result = _request(path, "read", timeout)
    if set(result) != {"ok", "kernel"} or not isinstance(result["kernel"], str):
        raise RuntimeError("invalid kernel log response")
    return result["kernel"]


def stop_broker(path, timeout=40):
    result = _request(path, "stop", timeout)
    if result != {"ok": True, "stopping": True}:
        raise RuntimeError("invalid stop response")
    return result


def driver_evidence():
    # Keep the training supervisor's schema, regex, ordering and hash bytes.
    kernel = read_kernel_log()
    anomalies = [line.strip() for line in kernel.splitlines() if ANOMALY_PATTERN.search(line)]
    latest = anomalies[-1] if anomalies else None
    digest = lambda value: hashlib.sha256(value).hexdigest()
    return dict(host=socket.gethostname(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        latest_driver_anomaly=latest, latest_driver_anomaly_sha256=digest(latest.encode()) if latest else None,
        anomaly_count=len(anomalies), driver_anomalies_sha256=digest("\n".join(anomalies).encode()),
        dmesg_tail=kernel.splitlines()[-200:])


def _unlink_owned(path, identity):
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == identity:
        path.unlink()


def _bind_socket(path, uid):
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0 or parent.st_mode & 0o022:
        raise PermissionError("/run must be a root-owned non-writable real directory")
    if os.path.lexists(path):
        raise FileExistsError("refusing to replace an existing socket/path")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    identity = None
    old_umask = os.umask(0o077)
    try:
        server.bind(str(path))
        info = path.lstat()
        identity = (info.st_dev, info.st_ino)
        os.chown(path, uid, -1)
        os.chmod(path, 0o600)
        server.listen(8)
        server.settimeout(0.5)
        return server, identity
    except BaseException:
        server.close()
        if identity is not None:
            _unlink_owned(path, identity)
        raise
    finally:
        os.umask(old_umask)


def _serve_loop(server, path, identity, uid, lifetime, reader=_kernel_text, *, install_signals=True):
    stop = False
    def interrupted(signum, frame):
        nonlocal stop
        stop = True
    previous = {}
    if install_signals:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
    deadline = time.monotonic() + lifetime
    try:
        while not stop and time.monotonic() < deadline:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                def bounded_reader():
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("broker lifetime expired")
                    return _kernel_text(timeout=min(30, left)) if reader is _kernel_text else reader()
                stop = _handle_connection(conn, uid, bounded_reader,
                    timeout=min(35, remaining)) or stop
    finally:
        server.close()
        _unlink_owned(path, identity)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _daemonize(server, path, identity, uid, lifetime):
    read_fd, write_fd = os.pipe()
    try:
        child = os.fork()
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        server.close()
        _unlink_owned(path, identity)
        raise
    if child:
        os.close(write_fd)
        server.close()
        try:
            ready, _, _ = select.select([read_fd], [], [], 15)
            if not ready:
                raise RuntimeError("broker daemon readiness timed out")
            message = os.read(read_fd, 4096)
            result = json.loads(message)
            if result.get("ready") is not True:
                raise RuntimeError("broker daemon startup failed")
            print(json.dumps(result), flush=True)
        finally:
            os.close(read_fd)
        return
    os.close(read_fd)
    try:
        os.setsid()
        os.chdir("/")
        null_fd = os.open("/dev/null", os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null_fd, fd)
        keep = {0, 1, 2, server.fileno(), write_fd}
        for entry in os.listdir("/proc/self/fd"):
            fd = int(entry)
            if fd not in keep:
                try:
                    os.close(fd)
                except OSError:
                    pass
        ready = dict(ready=True, pid=os.getpid(), socket=str(path), uid=uid,
                     lifetime_seconds=lifetime, read_only=True)
        os.write(write_fd, json.dumps(ready).encode())
        os.close(write_fd)
        _serve_loop(server, path, identity, uid, lifetime)
    except BaseException:
        _unlink_owned(path, identity)
        os._exit(1)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--read", action="store_true")
    mode.add_argument("--stop", action="store_true")
    mode.add_argument("--evidence", action="store_true")
    parser.add_argument("--socket", default=os.environ.get("WAM_KERNEL_LOG_SOCKET"))
    parser.add_argument("--uid", type=int)
    parser.add_argument("--lifetime-seconds", type=float, default=172800)
    parser.add_argument("--timeout-seconds", type=float, default=40)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.serve:
        if os.geteuid() != 0:
            raise PermissionError("broker --serve requires root; client operations do not")
        if args.uid is None or args.uid < 1:
            raise ValueError("--serve requires a positive allowed --uid")
        if not math.isfinite(args.lifetime_seconds) or not 0 < args.lifetime_seconds <= 604800:
            raise ValueError("lifetime must be in (0,604800]")
        path = _socket_path(args.socket or "", args.uid)
        server, identity = _bind_socket(path, args.uid)
        if args.daemon:
            _daemonize(server, path, identity, args.uid, args.lifetime_seconds)
        else:
            print(json.dumps(dict(ready=True, pid=os.getpid(), socket=str(path),
                uid=args.uid, lifetime_seconds=args.lifetime_seconds, read_only=True)), flush=True)
            _serve_loop(server, path, identity, args.uid, args.lifetime_seconds)
        return
    if args.daemon or args.uid is not None:
        raise ValueError("--daemon/--uid are server-only")
    if args.socket:
        _socket_path(args.socket)
        os.environ["WAM_KERNEL_LOG_SOCKET"] = args.socket
    if args.stop:
        if not args.socket:
            raise ValueError("--stop requires a broker socket")
        print(json.dumps(stop_broker(args.socket, args.timeout_seconds)))
    elif args.evidence:
        print(json.dumps(driver_evidence()))
    else:
        sys.stdout.write(read_kernel_log(args.timeout_seconds))


if __name__ == "__main__":
    main()
