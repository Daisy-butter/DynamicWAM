"""Fail-closed ownership of real simulator steps, independent of query load.

Install after SceneStepClock and the GIL bridge. Neither inference nor capture
may call scene.step(), including on the owner thread outside advance().
"""
import math
import threading
from .timing import TimingError


class ContinuousStepOwner:
    def __init__(self, scene, clock, *, tolerance=1e-8):
        self.scene, self.clock = scene, clock
        self.dt = float(scene.get_timestep())
        self.origin = float(clock())
        self.tolerance = tolerance
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.origin):
            raise TimingError('finite origin and positive real timestep required')
        self.owner = threading.get_ident()
        self.native_owner = threading.get_native_id()
        self.steps = 0
        self.entered = False
        self.installed = False
        self.original = scene.step
        # Retain this exact callable for identity checks and restoration.
        self.wrapper = self._guarded_step

    def install(self):
        if self.installed or threading.get_ident() != self.owner:
            raise TimingError('step owner already installed or wrong thread')
        self.scene.step = self.wrapper
        self.installed = True
        return self

    def _guarded_step(self):
        if threading.get_ident() != self.owner or not self.entered:
            raise TimingError('physics may advance only through its sole owner')
        self.original()

    def advance(self):
        if (not self.installed or self.scene.step is not self.wrapper or self.entered
                or threading.get_ident() != self.owner):
            raise TimingError('lost/reentrant/foreign physics ownership')
        expected = self.origin + self.steps * self.dt
        if abs(float(self.clock()) - expected) > self.tolerance:
            raise TimingError('physics advanced outside owner before step')
        self.entered = True
        try:
            self.wrapper()
        finally:
            self.entered = False
        if abs(float(self.clock()) - (expected + self.dt)) > self.tolerance:
            raise TimingError('real step failed to advance exactly one physics quantum')
        self.steps += 1

    def receipt(self):
        return dict(schema='single_physics_owner_v1',owner_native_thread=self.native_owner,
                    steps=self.steps,origin=self.origin,physics_dt=self.dt,
                    physical_time=float(self.clock()),installed=self.installed)

    def close(self):
        if not self.installed:
            return
        if threading.get_ident() != self.owner or self.scene.step is not self.wrapper:
            raise TimingError('step ownership changed before close')
        self.scene.step = self.original
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
