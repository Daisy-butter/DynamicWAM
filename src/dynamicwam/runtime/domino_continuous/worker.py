"""One in-flight model call, outside the simulation loop.

The thread worker is for interface tests. Evaluation uses a separate process
so the simulator's Python loop cannot stall the forward pass, and so the
policy can sit on a different GPU from the renderer.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np


class InferenceWorker:
    """Run ``factory()`` on a private thread and accept one request at a time."""

    def __init__(self, factory: Callable[[], Callable[[dict[str, Any]], np.ndarray]]) -> None:
        self._requests: queue.Queue[tuple[int, dict[str, Any]] | None] = queue.Queue()
        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._busy = False
        self._next_id = 0
        self.calls = 0
        self.intervals: list[tuple[float, float]] = []
        self._thread = threading.Thread(
            target=self._run,
            args=(factory,),
            name="domino-rt-inference",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("inference worker failed to start") from self._startup_error

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, packet: dict[str, Any]) -> int:
        if self._busy:
            raise RuntimeError("an inference request is already outstanding")
        request_id = self._next_id
        self._next_id += 1
        self._busy = True
        self._requests.put((request_id, _copy_packet(packet)))
        return request_id

    def poll(self) -> dict[str, Any] | None:
        try:
            item = self._results.get_nowait()
        except queue.Empty:
            return None
        self._busy = False
        self.calls += 1
        self.intervals.append((item["started"], item["finished"]))
        error = item["error"]
        if error is not None:
            raise RuntimeError("inference request failed") from error
        return item

    def close(self) -> None:
        if self._thread.is_alive():
            self._requests.put(None)
            self._thread.join(timeout=5.0)

    def _run(
        self,
        factory: Callable[[], Callable[[dict[str, Any]], np.ndarray]],
    ) -> None:
        try:
            infer = factory()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._requests.get()
            if item is None:
                return
            request_id, packet = item
            started = time.perf_counter()
            try:
                actions = np.asarray(infer(packet), dtype=np.float64).copy()
                error: BaseException | None = None
            except BaseException as exc:
                actions = None
                error = exc
            self._results.put(
                {
                    "request_id": request_id,
                    "actions": actions,
                    "error": error,
                    "started": started,
                    "finished": time.perf_counter(),
                }
            )


class ProcessInferenceWorker:
    """Run the policy in a spawned process with at most one request in flight."""

    def __init__(
        self,
        *,
        project_root: str,
        config_path: str,
        checkpoint: str,
        infer_index: int,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._requests: multiprocessing.Queue = context.Queue()
        self._results: multiprocessing.Queue = context.Queue()
        self._ready: multiprocessing.Queue = context.Queue()
        self._process = context.Process(
            target=_serve_policy,
            args=(
                project_root,
                config_path,
                checkpoint,
                int(infer_index),
                self._requests,
                self._results,
                self._ready,
            ),
            name="domino-rt-inference",
            daemon=True,
        )
        self._busy = False
        self._next_id = 0
        self.calls = 0
        self.intervals: list[tuple[float, float]] = []
        self._process.start()
        status = self._wait_until_ready(900.0)
        if status:
            self._process.join(timeout=5.0)
            raise RuntimeError(status)

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, packet: dict[str, Any]) -> int:
        if self._busy:
            raise RuntimeError("an inference request is already outstanding")
        if not self._process.is_alive():
            raise RuntimeError("inference worker exited before the request was submitted")
        request_id = self._next_id
        self._next_id += 1
        self._busy = True
        self._requests.put((request_id, _copy_packet(packet)))
        return request_id

    def _wait_until_ready(self, timeout_s: float) -> str | None:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            try:
                status = self._ready.get(timeout=1.0)
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError("inference worker exited before becoming ready")
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"inference worker did not become ready within {timeout_s:.0f}s"
                    )
                continue
            if status is None:
                return None
            return str(status)

    def poll(self) -> dict[str, Any] | None:
        try:
            item = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._process.is_alive():
                self._busy = False
                raise RuntimeError("inference worker exited while a request was in flight")
            return None
        self._busy = False
        self.calls += 1
        self.intervals.append((item["started"], item["finished"]))
        error = item["error"]
        if error:
            raise RuntimeError(error)
        return item

    def close(self) -> None:
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5.0)


def _die_when_parent_dies() -> None:
    """Free this GPU if the evaluator is killed or exits without running cleanup.

    ``os._exit`` and ``SIGKILL`` skip ``finally`` and leave the spawned policy
    process holding about 17 GiB. The next task on the same card then fails
    to load.
    """
    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL) != 0:
        return
    if os.getppid() == 1:
        raise SystemExit(1)


def _serve_policy(
    project_root: str,
    config_path: str,
    checkpoint: str,
    infer_index: int,
    requests: multiprocessing.Queue,
    results: multiprocessing.Queue,
    ready: multiprocessing.Queue,
) -> None:
    """Load DynamicWAM on ``infer_index`` and answer packets until shutdown."""
    _die_when_parent_dies()
    try:
        import sys
        from pathlib import Path

        root = Path(project_root)
        for entry in (root / "external" / "flash_attn_runtime", root / "src"):
            if entry.is_dir() and str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        import torch

        torch.cuda.set_device(infer_index)
        torch.cuda.init()
        print(f"inference device cuda:{torch.cuda.current_device()}", flush=True)
        from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy

        policy = DynamicWAMPolicy(config_path, project_root=str(root))
        # Route1 keeps the existing policy setup and sampler unchanged. Its
        # profile already selects the full checkpoint; reject an inconsistent
        # CLI artifact instead of monkeypatching the model loader.
        from dynamicwam.inference.model_loader import load_deploy_config
        configured = load_deploy_config(config_path)
        if Path(configured["checkpoint_path"]).resolve() != Path(checkpoint).resolve():
            raise ValueError("profile checkpoint differs from the verified evaluation checkpoint")
        policy.setup()

        def infer(packet: dict[str, Any]) -> np.ndarray:
            # Opt-in evaluation randomness; existing RTC callers keep their behavior.
            if packet.get("_reset_policy", packet.get("_a2_reset_policy", False)):
                policy.reset()
            seed = packet.get("_sampling_seed", packet.get("_a2_sampling_seed"))
            if seed is not None:
                torch.manual_seed(int(seed))
            policy.set_instruction(str(packet["instruction"]))
            return policy.sample(packet)

    except BaseException as exc:
        traceback.print_exc()
        ready.put(f"{type(exc).__name__}: {exc}")
        return
    ready.put(None)
    while True:
        item = requests.get()
        if item is None:
            return
        request_id, packet = item
        started = time.perf_counter()
        try:
            actions = np.asarray(infer(packet), dtype=np.float64).copy()
            error = ""
        except BaseException as exc:
            traceback.print_exc()
            actions = None
            error = f"{type(exc).__name__}: {exc}"
        results.put(
            {
                "request_id": request_id,
                "actions": actions,
                "error": error,
                "started": started,
                "finished": time.perf_counter(),
            }
        )


def _copy_packet(packet: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in packet.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        else:
            copied[key] = value
    return copied
