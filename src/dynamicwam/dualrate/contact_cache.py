"""Episode-local reuse of the read-only contacts from one real physics step.

DOMINO's audited contact consumers only read contacts. A fresh list is returned
per call; the native contact objects stay unchanged until the next scene step.
The physical clock, not an action row or wall time, invalidates this cache.
"""
from __future__ import annotations

import math
import threading


_MISSING = object()


class StepContactCache:
    def __init__(self, scene, physical_clock):
        self.scene = scene
        self.clock = physical_clock
        self.owner = threading.get_ident()
        self.original = scene.get_contacts
        self.original_instance = vars(scene).get('get_contacts', _MISSING)
        self.original_marker = vars(scene).get('_full_step_contact_cache', _MISSING)
        self.wrapper = lambda: self.read()
        self.installed = False
        self.key = None
        self.contacts = None
        self.error = None
        self.native_reads = self.cache_hits = self.failed_reads = 0

    def install(self):
        self._require_owner()
        if self.installed or getattr(self.scene, '_full_step_contact_cache', None) is not None:
            raise RuntimeError('contact cache already installed on scene')
        try:
            self.scene.get_contacts = self.wrapper
            self.scene._full_step_contact_cache = self
        except BaseException:
            self._restore_attribute('get_contacts', self.original_instance)
            self._restore_attribute('_full_step_contact_cache', self.original_marker)
            raise
        self.installed = True
        return self

    def _require_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('contact reads require the active scene owner')

    def _restore_attribute(self, name, original):
        if original is _MISSING:
            if name in vars(self.scene):
                delattr(self.scene, name)
        else:
            setattr(self.scene, name, original)

    def read(self):
        if not self.installed or threading.get_ident() != self.owner:
            raise RuntimeError('contact reads require the active scene owner')
        # Do not silently retry a native failure on a later tick or allow an
        # outer best-effort metrics reader to revive a failed episode.
        if self.error is not None:
            raise self.error
        now = float(self.clock())
        if not math.isfinite(now) or (self.key is not None and now < self.key):
            raise RuntimeError('contact cache requires a monotonic real physics clock')
        if now != self.key:
            self.key, self.contacts = now, None
        if self.contacts is None:
            self.native_reads += 1
            try:
                value = self.original()
                if not isinstance(value, list):
                    raise TypeError('native scene contacts must be a list')
                if float(self.clock()) != now:
                    raise RuntimeError('scene advanced during native contact read')
                self.contacts = tuple(value)
            except BaseException as exc:
                self.error = exc
                self.failed_reads += 1
                raise
        else:
            self.cache_hits += 1
        return list(self.contacts)

    def statistics(self):
        return dict(native_reads=self.native_reads, cache_hits=self.cache_hits,
                    failed_reads=self.failed_reads, key='actual_physics_time',
                    consumers='audited_read_only', independent_return_lists=True)

    def close(self):
        self._require_owner()
        if not self.installed:
            return
        if self.scene.get_contacts is not self.wrapper:
            raise RuntimeError('scene contact reader changed while cache was installed')
        self._restore_attribute('get_contacts', self.original_instance)
        self._restore_attribute('_full_step_contact_cache', self.original_marker)
        self.installed = False
        self.contacts = self.error = self.key = None

