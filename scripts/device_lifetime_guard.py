"""Bounded application-owned GPU descriptor lifetime contrast; never a daemon.

No ioctl, driver settings or recovery. All descriptors are closed after workers.
Closing the final guard is part of the diagnostic's exit-health requirement.
"""
import os
from pathlib import Path
import re
import stat
import time


class DeviceLifetimeGuard:
    def __init__(self, uuids, record):
        self.uuids = uuids
        self.record = record
        self.handles = []
        self.events = []

    def event(self, name, **values):
        self.events.append(dict(event=name, monotonic=time.monotonic(), **values))
        self.record(self.events)

    def open(self):
        devices = {}
        for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
            fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
            uuid = fields.get('GPU UUID', '').strip()
            minor = fields.get('Device Minor', '').strip()
            if uuid in self.uuids:
                if uuid in devices or not re.fullmatch(r'\d+', minor):
                    raise RuntimeError('ambiguous UUID/minor mapping')
                path = Path('/dev') / ('nvidia' + minor)
                value = path.stat()
                if not stat.S_ISCHR(value.st_mode) or os.major(value.st_rdev) != 195 or os.minor(value.st_rdev) != int(minor):
                    raise RuntimeError('unexpected NVIDIA character device')
                devices[uuid] = path
        if set(devices) != set(self.uuids):
            raise RuntimeError('missing requested UUID in proc inventory')
        try:
            for uuid in self.uuids:
                path = str(devices[uuid])
                self.event('before_open', uuid=uuid, path=path)
                fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
                self.handles.append((fd, uuid, path))
                self.event('after_open', uuid=uuid, path=path, fd=fd)
        except BaseException:
            self.close()
            raise

    def close(self):
        first_error = None
        while self.handles:
            # Reverse order, exactly once per owned fd; never retry an ambiguous close.
            fd, uuid, path = self.handles.pop()
            try:
                self.event('before_close', uuid=uuid, path=path, fd=fd)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            try:
                os.close(fd)
                self.event('after_close', uuid=uuid, path=path, fd=fd)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error:
            raise first_error
