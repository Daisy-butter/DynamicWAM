"""CPU ownership/lifetime contract for frozen-render asynchronous submission."""
from concurrent.futures import TimeoutError
from types import ModuleType, SimpleNamespace
import sys
import threading

import numpy as np
import pytest

from dynamicwam.runtime.domino_rt.camera_capture import AsyncCameraCapture
from dynamicwam.runtime import domino_rt


def fixture(monkeypatch, *, blocked=False, failure=None, async_submit=True):
    owner = threading.get_ident()
    names = ('left_camera', 'right_camera', 'front_camera', 'head_camera')
    live = {'pose': 1.}
    entered, release = threading.Event(), threading.Event()
    if not blocked:
        release.set()
    calls, initialized = [], []
    cameras = tuple((name, SimpleNamespace(name=name, frozen=None, submitted=None)) for name in names)
    def owner_only(value):
        assert threading.get_ident() == owner
        return value
    def update():
        owner_only(None)
        for _, camera in cameras:
            camera.frozen = np.full((2, 3, 4), live['pose'] / 10, np.float32)
    def take(camera):
        ident = threading.get_ident()
        assert (ident != owner) == async_submit
        if async_submit:
            assert initialized == [ident]
        calls.append((camera.name, ident))
        entered.set()
        if not release.wait(2):
            raise RuntimeError('test render was not released')
        if failure:
            raise RuntimeError(failure)
        camera.submitted = camera.frozen.copy()
    def read(camera):
        assert threading.get_ident() != owner
        return camera.submitted.copy()
    native = ModuleType('dynamicwam.runtime.domino_rt._sapien_color_reader')
    native.take_picture = take
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(domino_rt, '_sapien_color_reader', native, raising=False)
    manager = SimpleNamespace(get_config=lambda:owner_only({name:{'pose':live['pose']} for name in names}))
    robot = SimpleNamespace(get_left_arm_jointState=lambda:owner_only([live['pose']] * 7),
                            get_right_arm_jointState=lambda:owner_only([live['pose']] * 7))
    env = SimpleNamespace(_update_render=update, data_type={}, cameras=manager, robot=robot)
    capture = AsyncCameraCapture(reader=read, async_submit=async_submit,
        worker_initializer=lambda:initialized.append(threading.get_ident()))
    monkeypatch.setattr(capture, '_camera_handles', lambda env:cameras)
    return capture, env, live, entered, release, calls, initialized, names


def test_async_submission_freezes_before_owner_physics_resumes(monkeypatch):
    capture, env, live, entered, release, calls, initialized, names = fixture(monkeypatch, blocked=True)
    try:
        pending = capture.begin(env, 2.)
        assert entered.wait(1)
        assert capture.async_submit and pending.async_submit
        assert pending.render_submitted_wall_perf_s is None and pending.camera_submit_seconds == {}
        assert capture.poll(pending) is None
        live['pose'] = 9.  # CPU physics mutates ECS, not renderer's frozen transforms.
        with pytest.raises(RuntimeError, match='pending'):
            capture.begin(env, 2.004)
        with pytest.raises(TimeoutError):
            capture.finish(pending, timeout=.001)
        assert capture.busy
        release.set()
        result = capture.finish(pending, timeout=1)
        assert result.async_submit and result.physics_time_s == 2.
        assert result.observation['joint_action']['vector'].tolist() == [1.] * 14
        for name in names:
            assert result.observation['observation'][name]['pose'] == 1.
            np.testing.assert_array_equal(result.observation['observation'][name]['rgb'],
                                          np.full((2, 3, 3), 25, np.uint8))
        assert [name for name, _ in calls] == list(names)
        assert len({thread for _, thread in calls}) == 1 and len(initialized) == 1
        assert list(result.camera_submit_seconds) == list(names)
        assert all(duration >= 0 for duration in result.camera_submit_seconds.values())
        assert (result.capture_wall_perf_s <= result.render_submitted_wall_perf_s
                <= result.readback_started_wall_perf_s <= result.readback_finished_wall_perf_s
                <= result.finished_wall_perf_s)
        # A second capture copies the new ECS pose and reuses the exclusive worker.
        second = capture.finish(capture.begin(env, 2.004), timeout=1)
        assert second.observation['joint_action']['vector'][0] == 9.
        assert len(initialized) == 1
    finally:
        release.set()
        capture.close()


def test_default_keeps_four_native_submissions_on_owner(monkeypatch):
    capture, env, _, _, _, calls, _, names = fixture(monkeypatch, async_submit=False)
    with capture:
        pending = capture.begin(env, 0.)
        assert not pending.async_submit and pending.render_submitted_wall_perf_s is not None
        assert len(calls) == 4
        result = capture.finish(pending, timeout=1)
        assert not result.async_submit and list(result.camera_submit_seconds) == list(names)


def test_async_native_failure_propagates_and_close_joins(monkeypatch):
    capture, env, _, entered, release, calls, _, _ = fixture(monkeypatch, blocked=True, failure='native failure')
    capture.begin(env, 0.)
    assert entered.wait(1)
    timer = threading.Timer(.03, release.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match='native failure'):
            capture.close()
        assert release.is_set() and len(calls) == 1
        assert all(not thread.is_alive() for thread in capture._executor._threads)
        assert not capture.busy
        capture.close()
    finally:
        release.set()
        timer.join()


def test_owner_metadata_failure_never_dispatches_renderer(monkeypatch):
    capture, env, _, _, _, calls, _, _ = fixture(monkeypatch)
    def bad_config():
        raise RuntimeError('metadata failure')
    env.cameras.get_config = bad_config
    with capture:
        with pytest.raises(RuntimeError, match='metadata failure'):
            capture.begin(env, 0.)
        assert not calls and not capture.busy


def test_initializer_failure_never_touches_native_camera(monkeypatch):
    capture, env, _, _, _, calls, _, _ = fixture(monkeypatch)
    capture._executor.shutdown(wait=True)
    from concurrent.futures import ThreadPoolExecutor
    def fail_init():
        raise RuntimeError('device binding failed')
    capture._executor = ThreadPoolExecutor(max_workers=1, initializer=fail_init)
    try:
        capture.begin(env, 0.)
        with pytest.raises(RuntimeError, match='initializer'):
            capture.finish(timeout=1)
        assert calls == []
    finally:
        capture.close()
