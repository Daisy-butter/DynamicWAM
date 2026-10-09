"""One independent policy process. No CUDA/SAPIEN import in the owner module."""
from __future__ import annotations
import copy
import importlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from typing import Any
import numpy as np


def _serve(factory: str, config: dict, requests: Any, results: Any, ready: Any) -> None:
    try:
        # A failed/killed simulator cannot leave an orphan 20 GB model process.
        if os.name == 'posix':
            import ctypes
            import signal
            ctypes.CDLL(None).prctl(1, signal.SIGKILL)
            if os.getppid() == 1:
                raise RuntimeError('physics owner exited before policy startup')
        module, name = factory.split(':', 1)
        policy = getattr(importlib.import_module(module), name)(config)
        ready.put(dict(ready=True, pid=os.getpid()))
        while True:
            request = requests.get()
            if request is None:
                return
            if request.get('operation') == 'finalize':
                try:
                    value=policy.finalize()
                    results.put(dict(operation='finalize',value=value))
                except BaseException:
                    results.put(dict(operation='finalize',error=traceback.format_exc()))
                continue
            started = time.perf_counter()
            try:
                packet = request['packet']
                # The factory owns reset/noise semantics, including graph warmup.
                output = policy.sample(packet)
                if isinstance(output, dict):
                    actions = output['actions']; metadata = output.get('metadata', {})
                else:
                    actions = output; metadata = {}
                if hasattr(actions, 'detach'):
                    actions = actions.detach().cpu().numpy()
                actions = np.asarray(actions, dtype=np.float64).copy()
                if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                    raise RuntimeError('policy output must be finite [H,14] absolute qpos')
                results.put(dict(request_id=request['request_id'], actions=actions,
                    metadata=metadata, observation_sim_time=packet['observation_sim_time'],
                    submitted_wall=request['submitted_wall'], worker_started_wall=started,
                    worker_finished_wall=time.perf_counter()))
            except BaseException:
                results.put(dict(request_id=request['request_id'], error=traceback.format_exc()))
    except BaseException:
        ready.put(dict(ready=False, error=traceback.format_exc()))


class ProcessPolicyWorker:
    def __init__(self, *, factory: str, config: dict, startup_timeout: float = 900.) -> None:
        context = mp.get_context('spawn')
        self.requests = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        ready = context.Queue(maxsize=1)
        self.process = context.Process(target=_serve, args=(factory, copy.deepcopy(config),
            self.requests, self.results, ready), name='continuous-policy', daemon=True)
        self.busy = False; self.counter = 0
        self.process.start()
        try:
            message = ready.get(timeout=startup_timeout)
            if not message['ready']:
                raise RuntimeError(message['error'])
        except BaseException:
            self.close(); raise
        finally:
            ready.close()

    def submit(self, packet: dict) -> int:
        if self.busy:
            raise RuntimeError('only one inference request may be in flight')
        if not self.process.is_alive():
            raise RuntimeError('policy process exited')
        self.counter += 1
        request = dict(request_id=self.counter, packet=copy.deepcopy(packet), submitted_wall=time.perf_counter())
        self.requests.put_nowait(request)
        self.busy = True
        return self.counter

    def poll(self) -> dict | None:
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError(f'policy process exited with {self.process.exitcode}')
            return None
        self.busy = False
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def drain(self, timeout: float = 600.) -> None:
        """Only legal outside a timed episode; no result is installed."""
        end = time.monotonic()+timeout
        while self.busy:
            self.poll()
            if time.monotonic() > end:
                raise TimeoutError('policy drain timed out after episode')
            time.sleep(.005)

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
            self.process.join(timeout=2.)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=5.)
        if self.process.is_alive():
            self.process.kill(); self.process.join(timeout=5.)
        self.requests.cancel_join_thread(); self.results.cancel_join_thread()
        self.requests.close(); self.results.close()

    def finalize_recordings(self, timeout: float = 120.) -> dict:
        """Explicit post-episode operation; never valid while physics is timed."""
        if self.busy: raise RuntimeError('drain terminal inference before finalizing')
        if not self.process.is_alive(): raise RuntimeError('worker exited before finalization')
        self.requests.put(dict(operation='finalize'))
        result=self.results.get(timeout=timeout)
        if result.get('operation')!='finalize': raise RuntimeError('unexpected finalization response')
        if 'error' in result: raise RuntimeError(result['error'])
        return result['value']
