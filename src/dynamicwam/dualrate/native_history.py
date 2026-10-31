"""Physical-time RGB history for synchronous native TOPP deployment only.

Collect real captures after the executor's existing render synchronization.
Never step physics, call get_obs/update_render again, or infer a timestamp from
command count. Query anchors are ephemeral so short commands cannot evict the
periodic history. This is not an asynchronous/real-time camera implementation.
"""
from contextlib import contextmanager
import math
import time
import numpy as np

from .online import ObservationHistory, history_endpoints
from .producer import CAMERAS, RAW_FRAME_SECONDS, NUMERIC_TIME_EPSILON
from .timing import TimingError


def capture_rendered_rgb(env):
    """Same camera pixels and robot drive targets as native get_obs, RGB only.

    Caller has already synchronized render poses. In particular do not repeat
    _update_render: that can advance randomized lighting/RNG in some tasks.
    """
    env.cameras.update_picture()
    rgb = env.cameras.get_rgb()
    vector = list(env.robot.get_left_arm_jointState()) + list(env.robot.get_right_arm_jointState())
    return dict(observation={c: dict(rgb=rgb[c]['rgb']) for c in CAMERAS},
                joint_action=dict(vector=np.asarray(vector, dtype=np.float32)))


class NativePhysicalHistory:
    def __init__(self, *, capture=capture_rendered_rgb):
        self.history = ObservationHistory(max_frames=128)
        self.capture = capture
        self.origin = None
        self.next_sample = 1
        self.clock = None
        self.active = False
        self.capture_events = []

    def _now(self, env):
        clock = env._scene_step_clock
        if self.clock is not clock:
            raise TimingError('new episode/clock requires a fresh physical history')
        now = float(clock.snapshot().time_seconds)
        if not math.isfinite(now) or now < self.history.frames[-1].exposure:
            raise TimingError('native history clock reversed or reset')
        return now

    def query_frames(self, env, observation):
        if self.origin is None:
            self.clock = env._scene_step_clock
            now = float(self.clock.snapshot().time_seconds)
            self.history.append(observation, exposure=now, available=now)
            self.origin = now
            return self.history.snapshot(now)
        now = self._now(env)
        # A fresh query at an already captured instant is an anchor replacement,
        # not a second elapsed frame. Keep periodic storage independent of queries.
        periodic = tuple(x for x in self.history.frames if x.exposure < now)
        anchor_history = ObservationHistory()
        anchor_history.next_id = self.history.next_id
        frame = anchor_history.append(observation, exposure=now, available=now)
        self.history.next_id += 1
        return periodic + (frame,)

    @contextmanager
    def during_action(self, env):
        if self.active or self.origin is None:
            raise TimingError('history must be initialized, with one executor owner')
        original = env._update_render
        # Diagnostic forwarding wrappers expose the actual env's bound method.
        owner = original.__self__
        if owner._scene_step_clock is not self.clock:
            raise TimingError('render owner and physical clock differ')
        dt = float(owner.scene.get_timestep())
        if not math.isfinite(dt) or not 0 < dt <= RAW_FRAME_SECONDS:
            raise TimingError('physics tick must resolve the raw camera interval')
        had_override = '_update_render' in vars(owner)
        old_override = vars(owner).get('_update_render')
        self.active = True

        def render_and_capture(*args, **kwargs):
            result = original(*args, **kwargs)
            now = self._now(owner)
            due = self.origin + self.next_sample * RAW_FRAME_SECONDS
            if now + NUMERIC_TIME_EPSILON >= due:
                if now - due > dt + NUMERIC_TIME_EPSILON:
                    raise TimingError('executor missed a physical camera sampling deadline')
                start = time.monotonic()
                observation = self.capture(owner)
                if self._now(owner) != now:
                    raise TimingError('physics advanced during native camera capture')
                frame = self.history.append(observation, exposure=now, available=now)
                self.capture_events.append(dict(frame_id=frame.frame_id, exposure=now,
                    requested_exposure=due, physics_step_seconds=dt,
                    capture_wall_seconds=time.monotonic()-start))
                self.next_sample += 1
            return result

        owner._update_render = render_and_capture
        try:
            yield
            # A changed executor must not silently remove per-tick sampling.
            if self._now(owner) + NUMERIC_TIME_EPSILON >= self.origin + self.next_sample*RAW_FRAME_SECONDS:
                raise TimingError('native action ended without its due history capture')
        finally:
            if had_override:
                owner._update_render = old_override
            else:
                del owner._update_render
            self.active = False

    def describe(self, frames, *, action_dt):
        selected, matched = history_endpoints(frames, action_dt=action_dt)
        anchor = frames[-1].exposure
        targets = [anchor - i*4*action_dt for i in (5, 4, 3, 2, 1, 0)]
        return dict(history_sampling='physical_time', history_interval_seconds=4*action_dt,
            history_target_times=targets, history_exposure_times=[f.exposure for f in selected],
            history_endpoint_valid=list(matched), history_frame_ids=[f.frame_id for f in selected],
            history_endpoint_errors=[f.exposure-t if ok else None
                                     for f, t, ok in zip(selected, targets, matched)])
