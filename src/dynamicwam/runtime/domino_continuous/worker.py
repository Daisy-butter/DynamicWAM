"""One in-flight model call, outside the simulation loop.

The thread worker is for interface tests. Evaluation uses a separate process
so the simulator's Python loop cannot stall the forward pass, and so the
policy can sit on a different GPU from the renderer.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import struct
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


# The owner allows one outstanding request. Each direction therefore needs one
# bounded shared snapshot and a tiny notification, not a Queue feeder thread.
# The buffer is anonymous RawArray storage: no named-shm unlink/ownership race.
_REQUEST_CAPACITY = 8 * 1024 * 1024
_RESULT_CAPACITY = 1024 * 1024
_IPC_TRANSPORT = "single_inflight_rawarray_pipe_v1"
_NOTICE = struct.Struct("<QQQdddd")


class _SharedMailbox:
    """Single-producer/single-consumer snapshot slot with a 56-byte notice.

    The caller must consume the previous value before reusing this slot. The
    worker enforces this via its existing one-inflight request/result protocol.
    A stop notice does not touch the slot, so shutdown cannot corrupt a pending
    request. Serialization finishes on the caller before it resumes physics.
    """

    def __init__(self, buffer, connection, capacity, *, sending):
        self._buffer, self._connection = buffer, connection
        self.capacity, self.sending = int(capacity), bool(sending)
        self._sequence, self._closed = 0, False
        self.last_send_timing, self.last_receive_timing = {}, {}

    @classmethod
    def pair(cls, context, capacity):
        if type(capacity) is not int or not 1024 <= capacity <= 64 * 1024 * 1024:
            raise ValueError("mailbox capacity must be 1 KiB..64 MiB")
        buffer = context.RawArray("B", capacity)
        reader, writer = context.Pipe(duplex=False)
        return (cls(buffer, writer, capacity, sending=True),
                cls(buffer, reader, capacity, sending=False))

    def put(self, value):
        if self._closed or not self.sending:
            raise RuntimeError("mailbox is not an open sender")
        started = time.perf_counter()
        payload = None if value is None else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        serialized = time.perf_counter()
        size = 0 if payload is None else len(payload)
        if size > self.capacity:
            raise ValueError(f"IPC snapshot {size} bytes exceeds mailbox capacity {self.capacity}")
        if payload is not None:
            with memoryview(self._buffer).cast("B") as target:
                target[:size] = payload
        copied = time.perf_counter()
        sequence = self._sequence + 1
        notifying = time.perf_counter()
        self._connection.send_bytes(_NOTICE.pack(sequence, int(payload is None), size,
                                                started, serialized, copied, notifying))
        notified = time.perf_counter()
        self._sequence = sequence
        self.last_send_timing = dict(transport=_IPC_TRANSPORT, clock="time.perf_counter_absolute_same_host", sequence=sequence,
            payload_bytes=size, capacity_bytes=self.capacity,
            serialize_started_perf_s=started, serialize_finished_perf_s=serialized,
            snapshot_written_perf_s=copied, notify_started_perf_s=notifying,
            notify_finished_perf_s=notified, serialize_s=serialized-started,
            snapshot_copy_s=copied-serialized, notify_s=notified-notifying)
        return self.last_send_timing

    def get(self, timeout=None):
        if self._closed or self.sending:
            raise RuntimeError("mailbox is not an open receiver")
        if timeout is not None and not self._connection.poll(max(0., float(timeout))):
            raise queue.Empty
        notice = self._connection.recv_bytes(_NOTICE.size)
        received = time.perf_counter()
        if len(notice) != _NOTICE.size:
            raise RuntimeError("invalid IPC notification size")
        sequence, stop, size, started, serialized, copied, notifying = _NOTICE.unpack(notice)
        if sequence != self._sequence + 1 or stop not in (0, 1) or size > self.capacity or (stop and size):
            raise RuntimeError("invalid IPC sequence or snapshot size")
        self._sequence = sequence
        if stop:
            value = None
        else:
            with memoryview(self._buffer).cast("B") as source:
                # Own the bytes before decoding: no returned array can alias a
                # mailbox slot reused after this request's result is consumed.
                payload = bytes(source[:size])
            value = pickle.loads(payload)
        decoded = time.perf_counter()
        self.last_receive_timing = dict(transport=_IPC_TRANSPORT, clock="time.perf_counter_absolute_same_host", sequence=sequence,
            payload_bytes=size, capacity_bytes=self.capacity,
            sender_serialize_started_perf_s=started, sender_serialize_finished_perf_s=serialized,
            sender_snapshot_written_perf_s=copied, sender_notify_started_perf_s=notifying,
            notification_received_perf_s=received, deserialized_perf_s=decoded,
            sender_serialize_s=serialized-started,
            sender_snapshot_copy_s=copied-serialized,
            notification_wait_s=received-notifying,
            receive_copy_deserialize_s=decoded-received)
        return value

    def get_nowait(self):
        return self.get(timeout=0.)

    def close(self):
        if not self._closed:
            self._connection.close()
            self._buffer = None
            self._closed = True


class ProcessInferenceWorker:
    """Run the policy in a spawned process with at most one request in flight."""

    def __init__(
        self,
        *,
        project_root: str,
        config_path: str,
        checkpoint: str,
        infer_index: int,
        policy_mode: str = "native",
        native_cuda_graph: bool = True,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._requests, child_requests = _SharedMailbox.pair(context, _REQUEST_CAPACITY)
        child_results, self._results = _SharedMailbox.pair(context, _RESULT_CAPACITY)
        # Startup is the first response; it is consumed before any request.
        self._ready = self._results
        self._closed = False
        self._pending_ipc = {}
        self._active_request_id = None
        self._process = context.Process(
            target=_serve_policy,
            args=(
                project_root,
                config_path,
                checkpoint,
                int(infer_index),
                policy_mode,
                native_cuda_graph,
                child_requests,
                child_results,
                child_results,
            ),
            name="domino-rt-inference",
            daemon=True,
        )
        self._busy = False
        self._next_id = 0
        self.calls = 0
        self.intervals: list[tuple[float, float]] = []
        self.startup_metadata = {}
        try:
            try:
                self._process.start()
            finally:
                # Spawn owns separate duplicated handles; the parent must not
                # retain the child's pipe ends (including on startup failure).
                child_requests.close()
                child_results.close()
            status = self._wait_until_ready(900.0)
            if status:
                raise RuntimeError(status)
        except BaseException:
            # A constructor exception otherwise loses the only process owner:
            # evaluator assignment has not completed, so its finally sees None.
            # Close only this spawned child before propagating startup failure.
            self.close()
            raise

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, packet: dict[str, Any]) -> int:
        if self._busy:
            raise RuntimeError("an inference request is already outstanding")
        if getattr(self, '_closed', False):
            raise RuntimeError("inference worker is closed")
        if not self._process.is_alive():
            raise RuntimeError("inference worker exited before the request was submitted")
        request_id = self._next_id
        started = time.perf_counter()
        # pickle owns an immutable snapshot before returning; no feeder can be
        # starved by the simulator's subsequent tight catch-up loop.
        timing = self._requests.put((request_id, packet))
        self._pending_ipc = dict(submit_started_perf_s=started,
            submit_returned_perf_s=time.perf_counter(), request_publish=dict(timing))
        self._next_id += 1
        self._active_request_id = request_id
        self._busy = True
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
            if isinstance(status, dict) and status.get('ready') is True:
                self.startup_metadata = status.get('metadata', {})
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
        except (EOFError, OSError) as exc:
            self._busy = False
            raise RuntimeError("inference IPC closed before a complete response") from exc
        expected = getattr(self, '_active_request_id', None)
        if expected is not None and item.get('request_id') != expected:
            raise RuntimeError("inference response does not match the outstanding request")
        self._busy = False
        self._active_request_id = None
        ipc = dict(item.get('ipc_timing', {}))
        ipc.update(getattr(self, '_pending_ipc', {}))
        ipc['parent_result_receive'] = dict(getattr(self._results, 'last_receive_timing', {}))
        ipc['parent_poll_perf_s'] = time.perf_counter()
        item['ipc_timing'] = ipc
        item['policy_timing'] = dict(item.get('policy_timing', {}), worker_ipc=ipc)
        if not item.get('prepared_instruction') and not item.get('finished_episode'):
            self.calls += 1
            self.intervals.append((item["started"], item["finished"]))
        error = item["error"]
        if error:
            raise RuntimeError(error)
        return item

    def prepare_instruction(self, instruction: str, timeout_s: float = 120.0):
        """Encode language before the episode's physical clock starts."""
        self.submit({'_prepare_instruction': str(instruction)})
        deadline = time.monotonic() + timeout_s
        while True:
            item = self.poll()
            if item is not None:
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError('instruction preparation timed out')
            time.sleep(.01)

    def finish_episode(self, timeout_s: float = 120.0):
        """Wait for all policy work before the cohort releases the next setup.

        The caller first drains the ordinary fast request. A fast result alone
        does not imply that its asynchronous Video producer has finished.
        """
        import math
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("episode drain timeout must be finite and positive")
        if self.busy:
            raise RuntimeError("drain the fast request before finishing its episode")
        request_id = self.submit({'_finish_episode': True,
                                  '_episode_drain_timeout_s': float(timeout_s)})
        deadline = time.monotonic() + timeout_s
        while True:
            item = self.poll()
            if item is not None:
                if item.get('request_id') != request_id or item.get('finished_episode') is not True:
                    raise RuntimeError("invalid episode drain acknowledgement")
                return item
            if time.monotonic() >= deadline:
                raise TimeoutError('terminal episode slow producer did not drain')
            time.sleep(.002)

    def close(self) -> None:
        if getattr(self, '_closed', False):
            return
        process = self._process
        exited = False
        try:
            if process.is_alive():
                try:
                    # Stop is a notice only: never overwrite an unreceived
                    # shared request while the model is busy or starting up.
                    if not getattr(self._requests, '_closed', False):
                        self._requests.put(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            if process.is_alive():
                raise RuntimeError("owned inference process did not exit after termination")
            exited = True
            process.close()
        finally:
            for channel in (getattr(self, '_requests', None), getattr(self, '_results', None)):
                if channel is not None:
                    channel.close()
            self._busy = False
            self._closed = exited


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


def _serve_policy(*args):
    requests, results, ready = args[-3:]
    try:
        _serve_policy_impl(*args)
    finally:
        requests.close()
        results.close()


def _configure_cpu_threads():
    import torch
    import cv2
    # Set before model initialization / any parallel CPU work in this spawn.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    return dict(torch_intra_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                opencv_threads=cv2.getNumThreads(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                environment={key: os.environ.get(key) for key in
                    ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})


def _serve_policy_impl(
    project_root: str,
    config_path: str,
    checkpoint: str,
    infer_index: int,
    policy_mode: str,
    native_cuda_graph: bool,
    requests: _SharedMailbox,
    results: _SharedMailbox,
    ready: _SharedMailbox,
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
        cpu_runtime = _configure_cpu_threads()

        torch.cuda.set_device(infer_index)
        torch.cuda.init()
        print(f"inference device cuda:{torch.cuda.current_device()}", flush=True)
        from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy

        if policy_mode == 'motionkv_rtc':
            from .motionkv_policy import MotionKVPolicy
            policy = MotionKVPolicy(config_path, checkpoint, project_root=str(root),
                native_cuda_graph=native_cuda_graph)
        elif policy_mode == 'native':
            policy = DynamicWAMPolicy(config_path, project_root=str(root))
        else:
            raise ValueError(f'unknown evaluation policy mode: {policy_mode}')
        # Route1 keeps the existing policy setup and sampler unchanged. Its
        # profile already selects the full checkpoint; reject an inconsistent
        # CLI artifact instead of monkeypatching the model loader.
        from dynamicwam.inference.model_loader import load_deploy_config
        configured = load_deploy_config(config_path)
        if policy_mode == 'native' and Path(configured["checkpoint_path"]).resolve() != Path(checkpoint).resolve():
            raise ValueError("profile checkpoint differs from the verified evaluation checkpoint")
        policy.setup()

        def infer(packet: dict[str, Any]) -> np.ndarray:
            if '_prepare_instruction' in packet:
                if policy_mode != 'motionkv_rtc':
                    raise RuntimeError('instruction preparation requires MotionKV RTC policy')
                policy.prepare_instruction(packet['_prepare_instruction'])
                torch.cuda.synchronize()
                return np.empty((0, 14), np.float32)
            if policy_mode == 'motionkv_rtc':
                return policy.sample(packet)
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
    ready.put(dict(ready=True, metadata=dict(getattr(policy, 'warmup_timing', {}),
        cpu_runtime=cpu_runtime, ipc_transport=_IPC_TRANSPORT,
        request_capacity_bytes=requests.capacity, result_capacity_bytes=results.capacity)))
    try:
        _run_policy_requests(policy, requests, results, policy_mode, infer)
    finally:
        if hasattr(policy, 'close'):
            policy.close()


def _run_policy_requests(policy, requests, results, policy_mode, infer):
    """Shared CPU-testable service loop; transport never changes model inputs."""
    while True:
        item = requests.get()
        if item is None:
            return
        request_id, packet = item
        received = dict(requests.last_receive_timing)
        started = time.perf_counter()
        episode_drain = None
        try:
            if packet.get('_finish_episode') is True:
                if policy_mode != 'motionkv_rtc':
                    raise RuntimeError('episode drain requires MotionKV RTC policy')
                episode_drain = _drain_policy_episode(policy, packet['_episode_drain_timeout_s'])
                actions = np.empty((0, 14), np.float64)
            else:
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
                "prepared_instruction": '_prepare_instruction' in packet,
                "finished_episode": packet.get('_finish_episode') is True,
                "episode_drain": episode_drain,
                "policy_timing": dict(getattr(policy, 'last_policy_timing', {})),
                "ipc_timing": dict(transport=_IPC_TRANSPORT, clock="time.perf_counter_absolute_same_host", child_request_receive=received,
                    model_started_perf_s=started, result_publish_started_perf_s=time.perf_counter()),
            }
        )


def _drain_policy_episode(policy, timeout_s: float):
    """An explicit producer acknowledgement; never infer idleness from fast IO."""
    import math
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("episode drain timeout must be finite and positive")
    drain = getattr(policy, 'drain_episode', None)
    if drain is None:
        raise RuntimeError('MotionKV policy does not implement episode draining')
    return drain(timeout=float(timeout_s))


def _copy_packet(packet: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in packet.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        else:
            copied[key] = value
    return copied
