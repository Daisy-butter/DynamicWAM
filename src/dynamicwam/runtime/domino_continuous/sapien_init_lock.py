"""Cross-process lock for Sapien/Vulkan scene + independent renderer startup.

Multiple continuous evals starting prepare_fixed_scene / ProcessRenderWorker at
once can wedge on kernel rwsem (render child stuck, policy child pipe_wait,
episode summary status=running with no events.jsonl). Serialize only the init
window; the timed continuous loop runs without the lock.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class SapienInitLock:
    def __init__(self, path: str | os.PathLike | None = None, *, timeout_s: float = 1800.0,
                 device_context: bool = False):
        directory = Path(os.environ.get("TMPDIR") or os.environ.get("TMP") or "/tmp")
        if device_context:
            # Only the short native device/context constructors serialize across
            # GPUs. Expert play_once and independent scene replay must not hold
            # a global lock for minutes and consume other workers' prep timeout.
            self.path = directory / 'domino_sapien_device_context.lock'
        else:
            pci = os.environ.get('DYNAMICWAM_GPU_PCI', '')
            if pci and re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci) is None:
                raise ValueError('invalid GPU identity for scene initialization lock')
            suffix = '_' + pci.lower().replace(':', '_').replace('.', '_') if pci else ''
            default = directory / ('domino_sapien_init' + suffix + '.lock')
            self.path = Path(os.environ.get("DOMINO_SAPIEN_INIT_LOCK") or path or default)
        self.timeout_s = float(timeout_s)
        self._fh = None

    def __enter__(self):
        if fcntl is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh.seek(0)
                self._fh.truncate()
                self._fh.write(f"pid={os.getpid()} t={time.time():.3f}\n")
                self._fh.flush()
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"SapienInitLock timeout after {self.timeout_s}s path={self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        if self._fh is None:
            return False
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None
        return False
