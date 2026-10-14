"""Trained MotionKV adapter for the timestamped continuous DOMINO controller.

All model work occurs in the inference process; the simulator keeps stepping.
The controller owns absolute action times and supplies its exact committed raw
targets. This adapter normalizes them once, enforces clean-prefix sampling and
returns the full 16-action plan without retiming or trimming any prefix.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from dynamicwam.inference.fast_runtime import RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation, pad_text_embeddings
from dynamicwam.models.rtc_prefix import PrefixCondition
from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, packet_noise_generator


def pad_rtc_text(text_rows):
    """Semantically masked fixed-size T5 memory for one stable CUDA capture."""
    text, mask = pad_text_embeddings(text_rows)
    if text.shape[0] != 1 or text.shape[1] > 512:
        raise ValueError('RTC serving requires one T5 row with at most 512 tokens')
    padded = text.new_zeros((1, 512, text.shape[-1]))
    padded[:, :text.shape[1]] = text
    padded_mask = torch.zeros((1, 512), dtype=torch.bool, device=text.device)
    padded_mask[:, :text.shape[1]] = mask
    return padded, padded_mask


def validate_packet_contract(packet, *, max_prefix=4):
    """Value checks independent of CUDA, tested with real controller packets."""
    prefix = np.asarray(packet['_rtc_prefix_actions'], dtype=np.float32)
    length = packet['_rtc_prefix_length']
    origin = packet['_rtc_observation_action_index']
    age = packet['_rtc_cache_age_actions']
    if any(type(value) is not int for value in (length, origin, age)):
        raise ValueError('RTC lengths and action indices must be integer metadata')
    if not 0 <= length <= max_prefix or prefix.shape != (length, 14):
        raise ValueError('RTC prefix must be [d,14] within the trained 0..4 range')
    if not np.isfinite(prefix).all():
        raise ValueError('RTC committed prefix has nonfinite actions')
    if origin < 0 or origin % 4 or age != origin % 16:
        raise ValueError('RTC query must follow trained H4/R4 observation cadence')
    if not np.isfinite(float(packet['sim_time'])):
        raise ValueError('RTC observation time must be finite')
    period = float(packet['_rtc_action_interval_s'])
    if not np.isfinite(period) or period <= 0:
        raise ValueError('RTC physical action interval must be positive and finite')
    return prefix, length, origin, age


def slow_refresh_reason(origin, expected_age, observation_time, source_index, source_time):
    if source_time is None:
        return 'initial'
    if origin-source_index >= 16:
        return 'scheduled_action_age'
    if expected_age == 0 and abs(observation_time-source_time) > 1e-8:
        return 'new_zero_age_observation'
    return 'reuse'


class MotionKVPolicy(MultiratePolicy):
    def __init__(self, deploy_config_path, checkpoint, *, project_root,
                 native_cuda_graph=True):
        self._instruction_cache = {}
        self._queries = 0
        self.last_policy_timing = {}
        self.warmup_timing = {}
        super().__init__(deploy_config_path, checkpoint, project_root=project_root,
            scheduling='controlled_ratio', native_cuda_graph=native_cuda_graph,
            rtc_controller=RTC_TIMESTAMPED_CONTROLLER)

    def setup(self):
        super().setup()
        rtc = self.contract.get('rtc', {})
        native = self.contract.get('native_multirate', {})
        if (rtc.get('max_prefix_actions') != 4 or rtc.get('execute_suffix_actions') != 4
                or native.get('fast_steps') != 10 or native.get('world_refresh_actions') != 16
                or not getattr(self.fast_model.fast_conditioner, 'motion_kv', False)):
            raise RuntimeError('MotionKV RTC deployment requires the trained d<=4/H4/R4/K10 contract')
        # Dataset clock_time_seconds, not nominal container FPS, determines
        # physical action/history cadence. The audited deploy profile owns it.
        if not np.isfinite(self.runtime.action_interval_seconds) or self.runtime.action_interval_seconds <= 0:
            raise RuntimeError('MotionKV requires a positive audited physical action interval')
        self._warmup_graph()

    def reset(self):
        super().reset()
        self._queries = 0
        self.last_policy_timing = {}

    def prepare_instruction(self, instruction):
        """Called before episode time begins; cache immutable T5 text features."""
        text = str(instruction)
        self.set_instruction(text)
        runner = self.base.runner
        if text not in self._instruction_cache:
            self._instruction_cache[text] = runner._get_text_embeddings()
        runner.cached_text_embeddings = self._instruction_cache[text]

    def _warmup_graph(self):
        flow = self.runtime.head_flow_config
        count = int(flow['count'])
        started = time.perf_counter()
        packet = dict(instruction='warmup',
            frame=np.zeros((*self.runtime.composite_frame_size, 3), np.uint8),
            fast_frame=np.zeros((*self.runtime.composite_frame_size, 3), np.uint8),
            state=np.zeros(14, np.float32),
            flow_frames=np.zeros((count, *flow['compute_size'], 3), np.uint8),
            motion_features=np.zeros((count, 12), np.float32),
            motion_interval_valid_mask=np.zeros(count, np.bool_),
            motion_acceleration_valid_mask=np.zeros(count, np.bool_), sim_time=0.,
            _sampling_seed=0, _rtc_prefix_actions=np.zeros((0, 14), np.float32),
            _rtc_prefix_length=0, _rtc_observation_action_index=0, _rtc_cache_age_actions=0,
            _rtc_action_interval_s=float(self.runtime.action_interval_seconds))
        self.sample(packet)
        # Same graph must accept a different, nonzero committed prefix.
        packet['_rtc_prefix_actions'] = np.zeros((4, 14), np.float32)
        packet['_rtc_prefix_length'] = 4
        self.sample(packet)
        device = torch.device(self.base.runner.device)
        properties = torch.cuda.get_device_properties(device)
        self.warmup_timing = dict(total_ms=(time.perf_counter()-started)*1000,
            graph=dict(self.fast_graph.stats), latest_query=dict(self.last_policy_timing),
            cuda_device=str(device), cuda_device_name=properties.name,
            cuda_total_memory_bytes=int(properties.total_memory),
            cuda_memory_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            cuda_memory_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            cuda_peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        self.reset()

    @torch.inference_mode()
    def sample(self, packet):
        started = time.perf_counter()
        raw_prefix, length, origin, expected_age = validate_packet_contract(packet)
        if abs(float(packet['_rtc_action_interval_s'])-float(self.runtime.action_interval_seconds)) > 1e-8:
            raise RuntimeError('controller action clock disagrees with audited runtime profile')
        if packet.get('_reset_policy', False):
            self.reset()
        self.prepare_instruction(packet['instruction'])
        if packet.get('_sampling_seed') is not None:
            self._episode_seed = int(packet['_sampling_seed'])
        packet = dict(packet, action_index=origin, _query_index=self._queries,
            _episode_seed=self._episode_seed, _text=self.base.runner._get_text_embeddings())
        self._queries += 1
        text_ms = (time.perf_counter()-started)*1000
        slow_started = time.perf_counter()
        # Bootstrap holds the robot while physics advances. Its first d=4
        # request can therefore have action origin zero but a NEW observation.
        # Training age zero means source=current; never relabel that stale
        # startup cache as fresh merely because no action has completed yet.
        refresh_reason = slow_refresh_reason(origin, expected_age, float(packet['sim_time']),
            self.bridge_action_index, self.bridge_sim_time)
        refreshed = refresh_reason != 'reuse'
        if refreshed:
            if expected_age != 0:
                raise RuntimeError('missing slow cache at a non-refresh action boundary')
            self._install_bridge(self._build_bridge_completed(packet))
        age = origin-self.bridge_action_index
        if age != expected_age or age not in (0, 4, 8, 12):
            raise RuntimeError('cache action age disagrees with trained/controller cadence')
        # Bridge build already records+synchronizes its producer stream. Do not
        # insert extra device-wide syncs around the refresh gate: they only
        # inflated wall time on the H4 critical path.
        slow_ms = (time.perf_counter()-slow_started)*1000
        actions, detail = self._sample_rtc_fast(packet, age, raw_prefix, length)
        if not np.isfinite(actions).all():
            raise RuntimeError('MotionKV emitted nonfinite actions')
        if length and not np.allclose(actions[:length], raw_prefix, rtol=0, atol=2e-5):
            raise RuntimeError('normalized RTC sampler altered a committed raw action')
        self.last_policy_timing = dict(detail, text_ms=text_ms, slow_refresh_ms=slow_ms,
            refreshed=refreshed, refresh_reason=refresh_reason, cache_age_actions=age,
            cache_age_seconds=float(packet['sim_time'])-self.bridge_sim_time,
            source_action_index=self.bridge_action_index, observation_action_index=origin,
            prefix_length=length, fast_steps=10, source_bridge='immutable_official',
            rtc_controller=RTC_TIMESTAMPED_CONTROLLER, total_policy_ms=(time.perf_counter()-started)*1000,
            action_interval_seconds=float(self.runtime.action_interval_seconds),
            graph={key: value for key, value in self.fast_graph.stats.items() if key != 'last_signature'},
            cuda_memory_allocated_bytes=int(torch.cuda.memory_allocated()),
            cuda_memory_reserved_bytes=int(torch.cuda.memory_reserved()),
            cuda_peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            cuda_peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved()))
        return actions

    def _sample_rtc_fast(self, packet, age_actions, raw_prefix, length):
        runner = self.base.runner
        started = time.perf_counter()
        views, masks = fast_views_from_composite(packet['fast_frame'])
        device = torch.device(runner.device)
        state = runner._normalize_state(self.base._state_tensor(packet['state']))
        motion = runner._motion_batch(packet['motion_features'], packet['motion_interval_valid_mask'],
            packet['motion_acceleration_valid_mask'])
        text_rows = packet['_text']
        padded, padded_mask = pad_rtc_text(text_rows)
        current = CurrentObservation(rgb=torch.from_numpy(views).unsqueeze(0).to(device),
            rgb_valid_mask=torch.from_numpy(masks).unsqueeze(0).to(device), state=state,
            motion=motion['absolute_motion_features'],
            motion_interval_valid_mask=motion['absolute_motion_interval_valid_mask'],
            motion_acceleration_valid_mask=motion['absolute_motion_acceleration_valid_mask'],
            text_embeddings=text_rows, age=BridgeAge(float(packet['sim_time'])-self.bridge_sim_time,
                age_actions, True, (state-self.bridge_state).reshape(-1)))
        noise = runner._fresh_noisy_actions(generator=packet_noise_generator(packet, device, branch='fast'))
        prefix_actions = torch.zeros_like(noise)
        if length:
            raw = torch.from_numpy(raw_prefix).to(device=device, dtype=noise.dtype).unsqueeze(0)
            prefix_actions[:, :length] = self.runtime.action_normalizer.normalize(raw)
        prefix = PrefixCondition(prefix_actions, torch.tensor([length], dtype=torch.long, device=device))
        torch.cuda.synchronize(device)
        preprocess_ms = (time.perf_counter()-started)*1000
        sampled = time.perf_counter()
        normalized = self.fast_graph.sample(current, self.bridge, padded, padded_mask, noise, prefix=prefix)
        torch.cuda.synchronize(device)
        fast_ms = (time.perf_counter()-sampled)*1000
        copied = time.perf_counter()
        actions = self.runtime.action_normalizer.denormalize(normalized).squeeze(0).float().cpu().numpy()
        return actions, dict(preprocess_ms=preprocess_ms, fast_sampler_ms=fast_ms,
            output_copy_ms=(time.perf_counter()-copied)*1000)
