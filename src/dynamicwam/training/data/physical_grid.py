"""Physical-time labels and causal asynchronous scheduling (CPU only).

Arm samples are recorded DRIVE TARGETS, not measured qpos. Linear interpolation
reconstructs that command trajectory; grippers always use previous-value hold.
Actual image and history timestamps are never relabelled as grid timestamps.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 1
DT = .16
GRIPPERS = (6, 13)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_clock(frames, times, commands):
    frames, times, commands = np.asarray(frames), np.asarray(times, dtype=np.float64), np.asarray(commands)
    if frames.ndim != 1 or times.shape != frames.shape or len(times) < 2:
        raise ValueError('clock shape is invalid')
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError('frame index must match the actual video frame order')
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('simulator clock must be finite and nondecreasing')
    if commands.shape != (len(frames), 14) or not np.isfinite(commands).all():
        raise ValueError('drive target shape/nonfinite data')
    if np.any(commands[:, GRIPPERS] < -1e-5) or np.any(commands[:, GRIPPERS] > 1.00001):
        raise ValueError('gripper command outside recorded normalized bounds')

def target_grid(observation_time, episode_start, *, dt=DT, count=16):
    phase = (float(observation_time) - float(episode_start)) / dt
    first = int(np.floor(phase + 1e-8)) + 1
    targets = float(episode_start) + dt * np.arange(first, first + count, dtype=np.float64)
    if targets[0] <= observation_time + 1e-9:
        raise ValueError('target endpoint must be strictly after the actual observation')
    return targets

def action_brackets(times, targets):
    times, targets = np.asarray(times, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if np.any(targets < times[0]) or np.any(targets > times[-1] + 1e-9):
        raise ValueError('action target outside recorded trajectory; extrapolation forbidden')
    lower = np.searchsorted(times, targets + 1e-10, side='right') - 1
    upper = np.searchsorted(times, targets - 1e-10, side='left')
    upper = np.maximum(upper, lower)
    upper = np.minimum(upper, len(times) - 1)
    gap = times[upper] - times[lower]
    alpha = np.divide(targets - times[lower], gap, out=np.zeros_like(targets), where=gap > 0)
    return lower, upper, np.clip(alpha, 0, 1), gap

def resample_commands(times, commands, targets):
    lower, upper, alpha, _ = action_brackets(times, targets)
    result = np.asarray(commands)[lower] * (1-alpha[..., None]) + np.asarray(commands)[upper] * alpha[..., None]
    result[..., GRIPPERS] = np.asarray(commands)[lower][..., GRIPPERS]
    return result.astype(np.float32)

def video_endpoints(times, targets):
    # No frame from after the requested endpoint may supervise that endpoint.
    indices = np.searchsorted(times, np.asarray(targets) + 1e-10, side='right') - 1
    if np.any(indices < 0) or np.any(np.asarray(targets) > times[-1] + 1e-9):
        raise ValueError('video endpoints outside capture history')
    actual = np.asarray(times)[indices]
    return indices, actual, np.asarray(targets) - actual

def stable_rng(*identity):
    return np.random.default_rng(int(json_hash(identity)[:16], 16))

def sample_async_timing(source_time, current_time, action_times, *, seed, epoch,
                        source_id, current_id, slow_budget_seconds=2.56, rtc_prefix_max=4, source_action_times=None):
    """Sample publication/compute timing conditioned on one real trajectory.

    Prefix is the queued endpoints irrevocable before a computed plan can be
    installed. The current fractional interval is already committed. Values are
    demonstration commands unless a separately validated real replay is used.
    """
    age = float(current_time-source_time)
    if age < -1e-8: raise ValueError('cache source is in the future')
    rng = stable_rng(seed, epoch, source_id, current_id)
    publication_delay = float(rng.uniform(0, min(max(age, 0), slow_budget_seconds)))
    endpoints = np.asarray(action_times, dtype=np.float64)
    phase = float(current_time - (endpoints[0] - DT))
    phase = float(np.clip(phase, 0, DT))
    # Include d=0 only at exact boundaries with zero latency (bootstrap).
    if phase < 1e-8 and rng.random() < .2:
        delay, length = 0., 0
    else:
        max_delay = max(0., float(endpoints[rtc_prefix_max-1]-current_time)-1e-8)
        delay = float(rng.uniform(0, max_delay))
        ready = current_time + delay
        length = min(rtc_prefix_max, int(np.searchsorted(endpoints, ready, side='right')) + 1)
    source_length = None
    if source_action_times is not None:
        source_endpoints=np.asarray(source_action_times,dtype=np.float64)
        source_phase=max(0.,float(source_time-(source_endpoints[0]-DT)))
        if source_phase<1e-8 and rng.random()<.2:
            source_length=0
        else:
            source_delay=float(rng.uniform(0,max(0.,float(source_endpoints[rtc_prefix_max-1]-source_time)-1e-8)))
            source_length=min(rtc_prefix_max,int(np.searchsorted(source_endpoints,source_time+source_delay,side='right'))+1)
    result = dict(source_capture_time=float(source_time), current_capture_time=float(current_time),
        cache_publication_time=float(source_time+publication_delay),
        slow_latency_seconds=publication_delay, query_phase_seconds=phase,
        query_time=float(current_time), fast_latency_seconds=delay,
        ready_time=float(current_time+delay), rtc_prefix_lengths=int(length))
    if source_length is not None:result['source_rtc_prefix_lengths']=int(source_length)
    return result
