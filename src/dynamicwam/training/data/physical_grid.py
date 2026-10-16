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

def rtc_prefix_sampling_contract(mode='delay_minimum_v1', source_prefix_condition_dropout_probability=.2):
    """Offline conditioning distribution, never evidence of executed policy data."""
    if mode not in ('delay_minimum_v1', 'fixed_h4_v1'):
        raise ValueError('unsupported RTC prefix sampling mode')
    probability = float(source_prefix_condition_dropout_probability)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('source prefix condition dropout probability must be in [0,1]')
    if mode == 'delay_minimum_v1':
        if probability != .2:
            raise ValueError('source prefix dropout override requires fixed_h4_v1')
        return dict(mode=mode)
    return dict(mode=mode, current_prefix_actions=4, source_prefix_actions=[0,4],
        source_zero_role='bootstrap_condition_dropout_only',
        source_prefix_condition_dropout_probability=probability,
        latency_distribution='unchanged_delay_minimum_v1',
        commit_horizon='fourth_actual_target_endpoint_independent_of_ready_latency',
        timing_source='sampled_offline_schedule_on_recorded_capture_and_target_times',
        prefix_source='expert_demonstration', on_policy_ready=False, hold_coverage=False)

def sample_async_timing(source_time, current_time, action_times, *, seed, epoch,
                        source_id, current_id, slow_budget_seconds=2.56, rtc_prefix_max=4, source_action_times=None,
                        rtc_prefix_sampling='delay_minimum_v1', source_prefix_condition_dropout_probability=.2):
    """Sample publication/compute timing conditioned on one real trajectory.

    Prefix is the queued endpoints irrevocable before a computed plan can be
    installed. The current fractional interval is already committed. Values are
    demonstration commands unless a separately validated real replay is used.
    """
    contract = rtc_prefix_sampling_contract(rtc_prefix_sampling, source_prefix_condition_dropout_probability)
    fixed_h4 = contract['mode'] == 'fixed_h4_v1'
    if fixed_h4 and rtc_prefix_max != 4:
        raise ValueError('fixed_h4_v1 requires rtc_prefix_max=4')
    age = float(current_time-source_time)
    if age < -1e-8: raise ValueError('cache source is in the future')
    rng = stable_rng(seed, epoch, source_id, current_id)
    publication_delay = float(rng.uniform(0, min(max(age, 0), slow_budget_seconds)))
    endpoints = np.asarray(action_times, dtype=np.float64)
    if fixed_h4 and (not np.isfinite([source_time,current_time,slow_budget_seconds]).all()
            or slow_budget_seconds < 0 or endpoints.ndim != 1 or len(endpoints) < 4
            or not np.isfinite(endpoints).all() or np.any(np.diff(endpoints) <= 0)
            or endpoints[0] <= current_time or age < 0):
        raise ValueError('fixed_h4_v1 requires finite causal capture times and future target endpoints')
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
    # Preserve the legacy ready-delay draws exactly; a fast response does not
    # shorten the controller's already committed H4 action horizon.
    if fixed_h4:
        length = 4
    source_length = None
    if source_action_times is not None:
        source_endpoints=np.asarray(source_action_times,dtype=np.float64)
        if fixed_h4:
            if (source_endpoints.ndim != 1 or len(source_endpoints) < 4
                    or not np.isfinite(source_endpoints).all() or np.any(np.diff(source_endpoints) <= 0)
                    or source_endpoints[0] <= source_time):
                raise ValueError('fixed_h4_v1 source targets must be finite future endpoints')
            source_length = 0 if rng.random() < contract['source_prefix_condition_dropout_probability'] else 4
        elif max(0.,float(source_time-(source_endpoints[0]-DT)))<1e-8 and rng.random()<.2:
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
    if fixed_h4:
        result.update(commit_horizon_time=float(endpoints[3]),
            commit_horizon_seconds=float(endpoints[3]-current_time))
    return result
