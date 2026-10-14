#!/usr/bin/env python3
"""CPU checks for the explicit timestamped RTC serving boundary."""
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from dynamicwam.inference.fast_runtime import require_rtc_deployment_mode, RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.runtime.domino_continuous.motionkv_policy import validate_packet_contract, pad_rtc_text, slow_refresh_reason

rtc = dict(version=1, mode='training_time_action_prefix', clean_timestep=0.,
    loss='per_sample_suffix_mean', includes_zero_prefix=True,
    queue_clock='absolute_action_effect_times', overflow='reject_and_refresh',
    max_prefix_actions=4, execute_suffix_actions=4)
payload = dict(modules=dict(native_multirate=True, architecture='native',
    action_expert=dict(chunk_size=16), runtime_contract=dict(rtc=rtc)))
assert require_rtc_deployment_mode(payload, rtc_controller=RTC_TIMESTAMPED_CONTROLLER) == rtc
for options in ({}, {'rtc_controller': 'unknown'},
                {'rtc_controller': RTC_TIMESTAMPED_CONTROLLER, 'allow_zero_prefix_ablation': True}):
    try:
        require_rtc_deployment_mode(payload, **options)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError(f'accepted invalid serving mode {options}')
checks = 4
text = torch.randn(13, 8)
padded, mask = pad_rtc_text([text])
assert tuple(padded.shape) == (1, 512, 8) and int(mask.sum()) == 13
assert torch.equal(padded[0, :13], text) and not bool(padded[0, 13:].any())
checks += 1
for args, expected in (((0, 0, 0., 0, None), 'initial'),
                       ((0, 0, 0., 0, 0.), 'reuse'),
                       ((0, 0, .8, 0, 0.), 'new_zero_age_observation'),
                       ((4, 4, 1.44, 0, .8), 'reuse'),
                       ((16, 0, 3.36, 0, .8), 'scheduled_action_age')):
    assert slow_refresh_reason(*args) == expected
    checks += 1
for origin in (0, 4, 8, 12, 16, 20):
    for length in (0, 1, 4):
        packet = dict(_rtc_prefix_actions=np.ones((length, 14), np.float32),
            _rtc_prefix_length=length, _rtc_observation_action_index=origin,
            _rtc_cache_age_actions=origin % 16, sim_time=origin*.16, _rtc_action_interval_s=.16)
        assert validate_packet_contract(packet)[1:] == (length, origin, origin % 16)
        checks += 1
base = dict(_rtc_prefix_actions=np.ones((4, 14), np.float32), _rtc_prefix_length=4,
    _rtc_observation_action_index=4, _rtc_cache_age_actions=4, sim_time=.64, _rtc_action_interval_s=.16)
for changes in (dict(_rtc_prefix_length=5), dict(_rtc_observation_action_index=3),
                dict(_rtc_cache_age_actions=8), dict(sim_time=float('nan')),
                dict(_rtc_prefix_actions=np.full((4, 14), np.nan)),
                dict(_rtc_prefix_length=4.0)):
    packet = dict(copy.deepcopy(base), **changes)
    try:
        validate_packet_contract(packet)
    except ValueError:
        checks += 1
    else:
        raise AssertionError(f'accepted invalid packet {changes}')
print(dict(passed=True, checks=checks, scope='CPU serving mode and RTC packet validation'))
