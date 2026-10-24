"""Regression for causal camera selection and physical flow/RGB parity.

Run against the repaired checkout; no cameras, CUDA or source-data files needed.
"""
import numpy as np
import pytest
import torch

import dynamicwam.absolute_motion as motion_impl
from dynamicwam.dualrate import producer
from dynamicwam.dualrate.online import OnlineFrame, history_features


def test_condition_cannot_choose_later_side_of_delay_target():
    # Both rows are equally close; only the earlier one has completed readback.
    times = np.array([0., .225, .255])
    frame = producer.available_condition_frame(times, .36)
    assert frame == 1
    assert times[frame] + producer.CAMERA_AVAILABILITY_SECONDS <= .36 + 1e-6
    # A future-side row alone cannot rescue missing past camera support.
    with pytest.raises(ValueError):
        producer.available_condition_frame(np.array([0., .255]), .36)
    with pytest.raises(ValueError):
        producer.available_condition_frame(np.array([0., .18, .26]), .36)
    assert producer.available_condition_frame(np.array([0., .23, .26]), .36) == 1
    grid = (np.arange(8, dtype=np.float32) * np.float32(.06)).astype(np.float64)
    event = float(np.float32(.06) * np.float32(6))
    assert producer.available_condition_frame(grid, event) == 4


@pytest.mark.parametrize('times,expected_pairs,missing', [
    # Same physical pairs despite irregular source intervals.
    ([0., .70, 1.46, 2.14, 2.90, 3.60], [(1, 2), (2, 3), (3, 4), (4, 5)], []),
    # Missing 1.44 s must invalidate BOTH neighboring pairs, never .72 -> 3.60.
    ([0., .72, 2.16, 2.88, 3.60], [None, None, (2, 3), (3, 4)], [0, 1]),
    # Regular input still requires recomputation when an old cache disagrees.
    ([0., .72, 1.44, 2.16, 2.88, 3.60], [(1, 2), (2, 3), (3, 4), (4, 5)], []),
])
def test_numeric_and_rgb_use_same_physical_pairs(monkeypatch, times, expected_pairs, missing):
    times = np.asarray(times, np.float64)
    images = [np.full((64, 64, 3), i, np.uint8) for i in range(len(times))]
    calls = []

    def fake_flow(a, b, **kwargs):
        first, last = int(a[0, 0, 0]), int(b[0, 0, 0])
        calls.append((first, last))
        rgb = np.empty((*kwargs['compute_size'], 3), np.uint8)
        rgb[:] = (first, last, 17)
        # Keep real build_motion_features; only optical estimation is replaced.
        stats = np.array([last - first, (last - first) * .5, .25, .75], np.float32)
        return rgb, stats, None, True

    monkeypatch.setattr(producer, '_rgb', lambda handle, index, camera: images[int(index)])
    monkeypatch.setattr(motion_impl, 'compute_flow_observation', fake_flow)
    monkeypatch.setattr(motion_impl, 'compute_dense_flow',
                        lambda a, b, **kw: np.zeros((64, 64, 2), np.float32))
    monkeypatch.setattr(motion_impl, 'filter_flow_by_forward_backward_consistency',
                        lambda forward, backward, **kw:
                        (forward, np.zeros((64, 64), bool), None, False))
    contract = dict(compute_size=[64, 64], normalization_percentile=99.,
                    farneback={}, quality={})
    count = len(times)
    # Every interval is deliberately stale. Neither its numeric features nor
    # its RGB may be reused merely because the end frame has the same index.
    cache = dict(
        interval_start_time_seconds=times - .731,
        interval_end_time_seconds=times + .011,
        interval_valid=np.ones(count, bool),
        acceleration_valid=np.ones(count, bool),
        motion_features=np.full((count, 12), 999., np.float32),
        flow_rgb=np.full((count, 64, 64, 3), 250, np.uint8),
    )
    result = producer._flow_features(None, times, count - 1, cache, contract,
                                     return_flow_rgb=True)
    numeric, valid, acceleration, intervals, spatial, spatial_valid, spatial_times, rgb = result
    assert calls, 'stale physical pairs must be recomputed'
    frames = [OnlineFrame((image,) * 3, np.zeros(14, np.float32), float(t), float(t), i)
              for i, (t, image) in enumerate(zip(times, images))]
    online = history_features(frames, contract)
    for value, key in ((numeric, 'motion'), (valid, 'motion_valid'),
                       (acceleration, 'motion_acceleration_valid'), (intervals, 'motion_times'),
                       (spatial, 'spatial_flow'), (spatial_valid, 'spatial_flow_valid'),
                       (spatial_times, 'spatial_flow_times')):
        torch.testing.assert_close(value, online[key], rtol=0, atol=0)
    np.testing.assert_array_equal(rgb, online['flow_rgb'])
    assert rgb.shape == (4, 64, 64, 3)
    for slot, pair in enumerate(expected_pairs):
        if pair is None:
            assert not bool(valid[slot])
            assert not np.any(rgb[slot])
            assert not bool(numeric[slot].any())
            np.testing.assert_array_equal(intervals[slot].numpy(), [times[-1], times[-1]])
        else:
            first, last = pair
            assert bool(valid[slot])
            np.testing.assert_array_equal(rgb[slot, 0, 0], [first, last, 17])
            np.testing.assert_array_equal(intervals[slot].numpy(), [times[first], times[last]])
    assert valid.tolist() == [i not in missing for i in range(4)]


@pytest.mark.parametrize('times,endpoints', [
    ([0., .72, 1.44, 2.16, 2.88, 3.60], [0, 1, 2, 3, 4, 5]),
    ([0., .72, 2.16, 2.88, 3.60], [0, 1, None, 2, 3, 4]),
])
def test_matching_cache_preserves_online_rgb_without_recomputation(monkeypatch, times, endpoints):
    times = np.asarray(times, np.float64)
    images = [np.full((64, 64, 3), i, np.uint8) for i in range(len(times))]

    def fake_flow(a, b, **kwargs):
        first, last = int(a[0, 0, 0]), int(b[0, 0, 0])
        rgb = np.empty((*kwargs['compute_size'], 3), np.uint8)
        rgb[:] = (first, last, 17)
        return rgb, np.array([last - first, .5, .25, .75], np.float32), None, True

    monkeypatch.setattr(producer, '_rgb', lambda handle, index, camera: images[int(index)])
    monkeypatch.setattr(motion_impl, 'compute_flow_observation', fake_flow)
    monkeypatch.setattr(motion_impl, 'compute_dense_flow',
                        lambda a, b, **kw: np.zeros((64, 64, 2), np.float32))
    monkeypatch.setattr(motion_impl, 'filter_flow_by_forward_backward_consistency',
                        lambda forward, backward, **kw:
                        (forward, np.zeros((64, 64), bool), None, False))
    contract = dict(compute_size=[64, 64], normalization_percentile=99., farneback={}, quality={})
    frames = [OnlineFrame((image,) * 3, np.zeros(14, np.float32), float(t), float(t), i)
              for i, (t, image) in enumerate(zip(times, images))]
    expected = history_features(frames, contract)
    count = len(times)
    cache = dict(
        interval_start_time_seconds=np.full(count, np.nan),
        interval_end_time_seconds=np.full(count, np.nan),
        interval_valid=np.ones(count, bool),
        acceleration_valid=np.ones(count, bool),
        motion_features=np.full((count, 12), 999., np.float32),
        flow_rgb=np.full((count, 64, 64, 3), 250, np.uint8),
    )
    for first, last in zip(endpoints, endpoints[1:]):
        if first is not None and last is not None:
            cache['interval_start_time_seconds'][last] = times[first]
            cache['interval_end_time_seconds'][last] = times[last]
    # Fill only real returned pairs. A missing endpoint aliases the anchor
    # placeholder; its nonzero valid cache image must still be masked to zero.
    for slot, (first, last) in enumerate(zip(endpoints[1:], endpoints[2:])):
        if first is not None and last is not None:
            cache['motion_features'][last] = expected['motion'][slot].numpy()
            cache['interval_valid'][last] = bool(expected['motion_valid'][slot])
            cache['acceleration_valid'][last] = bool(expected['motion_acceleration_valid'][slot])
            cache['flow_rgb'][last] = expected['flow_rgb'][slot]

    def forbidden_recompute(*args, **kwargs):
        pytest.fail('matching physical cache must not recompute compute_flow_observation')

    monkeypatch.setattr(motion_impl, 'compute_flow_observation', forbidden_recompute)
    result = producer._flow_features(None, times, count - 1, cache, contract,
                                     return_flow_rgb=True)
    for value, key in zip(result[:7], ('motion', 'motion_valid', 'motion_acceleration_valid',
                                      'motion_times', 'spatial_flow', 'spatial_flow_valid',
                                      'spatial_flow_times')):
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    np.testing.assert_array_equal(result[7], expected['flow_rgb'])
    if None in endpoints:
        assert not np.any(result[7][:2])
        assert not bool(result[1][:2].any())
