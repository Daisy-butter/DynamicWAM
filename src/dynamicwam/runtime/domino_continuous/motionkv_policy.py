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
from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, packet_noise_generator, capture_observation_packet
from dynamicwam.runtime.ciwam.wam.async_world_schedule import AsyncWorldScheduler, ASYNC_WORLD_SCHEDULE


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


def validate_packet_contract(packet, *, max_prefix=4, asynchronous=False):
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
    if origin < 0 or origin % 4 or (not asynchronous and age != origin % 16):
        raise ValueError('RTC query must follow trained H4/R4 observation cadence')
    if not np.isfinite(float(packet['sim_time'])):
        raise ValueError('RTC observation time must be finite')
    period = float(packet['_rtc_action_interval_s'])
    if not np.isfinite(period) or period <= 0:
        raise ValueError('RTC physical action interval must be positive and finite')
    return prefix, length, origin, age


def validate_v3_packet_context(packet):
    offsets = np.asarray(packet['_rtc_action_target_offsets'],dtype=np.float64)
    previous = np.asarray(packet['_rtc_previous_actions'],dtype=np.float32)
    valid = np.asarray(packet['_rtc_previous_valid'],dtype=np.bool_)
    if offsets.shape != (16,) or not np.isfinite(offsets).all() or (offsets <= 0).any():
        raise ValueError('v3 target offsets must be sixteen finite future physical times')
    if not np.allclose(np.diff(offsets), float(packet['_rtc_action_interval_s']), rtol=0, atol=1e-6):
        raise ValueError('v3 endpoint grid disagrees with controller action interval')
    if previous.shape != (16,14) or valid.shape != (16,) or not np.isfinite(previous[valid]).all():
        raise ValueError('v3 previous proposal needs finite valid [16,14] rows and [16] mask')
    return offsets, previous, valid


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
        self._async_world = None
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
        if native.get('version') == 3:
            if native.get('slow_schedule') != ASYNC_WORLD_SCHEDULE:
                raise RuntimeError('v3 requires explicit asynchronous source publication')
            self._async_world = AsyncWorldScheduler(self._build_bridge_completed,
                snapshot=capture_observation_packet, refresh_actions=16,
                max_age_actions=int(self.contract['max_age_actions']),
                max_age_seconds=float(self.contract['max_age_seconds']),
                action_interval_seconds=float(self.runtime.action_interval_seconds))
        self._warmup_graph()
        if self._async_world is not None:
            # CUDA global capture cannot race arbitrary launches by the slow
            # producer. Synthetic warmup established the one fixed signature.
            self.fast_graph.allow_recapture = False

    def _complete_bridge_value(self, value):
        if self._async_world is None:
            return value
        bridge, state, timestamp, index = value
        bridge = self.fast_model.fast_conditioner.prepare_source_cache(bridge)
        return bridge, state, timestamp, index

    def _install_bridge(self, value):
        super()._install_bridge(value)
        conditioner = self.fast_model.fast_conditioner
        if getattr(conditioner, 'interface_version', 1) >= 2 and self.bridge.source_visual_tokens is None:
            # Read-only inference cache is rebuilt with each source, cleared by
            # reset, and never reused across weight or episode versions.
            self.bridge = conditioner.prepare_source_cache(self.bridge)

    def reset(self):
        if self._async_world is not None:
            self._async_world.reset()
        super().reset()
        self._queries = 0
        self.last_policy_timing = {}

    @property
    def actual_slow_builds(self):
        if self._async_world is not None:
            return self._async_world.completed_count()
        return super().actual_slow_builds

    def close(self):
        if self._async_world is not None and not self._async_world.close(timeout=120.):
            raise TimeoutError('slow producer still owns its CUDA work; do not unload models')
        super().close()

    def _memory_stats(self):
        device = torch.device(self.base.runner.device)
        if device.type != 'cuda':
            return {}
        return dict(cuda_memory_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            cuda_memory_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            cuda_peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))

    @staticmethod
    def _sync_current_stream(device):
        if device.type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            event.synchronize()  # never waits for unrelated slow-stream work

    def prepare_instruction(self, instruction):
        """Called before episode time begins; cache immutable T5 text features."""
        text = str(instruction)
        self.set_instruction(text)
        runner = self.base.runner
        if text not in self._instruction_cache:
            self._instruction_cache[text] = runner._get_text_embeddings()
        runner.cached_text_embeddings = self._instruction_cache[text]

    def _warmup_graph(self):
        if self._async_world is not None:
            with self._async_world.worker.condition:
                if self._async_world.worker.active is not None or self._async_world.worker.pending is not None:
                    raise RuntimeError('CUDA graph warmup cannot overlap a slow producer')
            self.fast_graph.allow_recapture = True
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
            _rtc_action_interval_s=float(self.runtime.action_interval_seconds),
            _rtc_previous_actions=np.zeros((16,14),np.float32),
            _rtc_previous_valid=np.zeros(16,np.bool_),
            _rtc_action_target_offsets=np.arange(1,17,dtype=np.float64)*float(self.runtime.action_interval_seconds))
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
        raw_prefix, length, origin, expected_age = validate_packet_contract(packet,
            asynchronous=self._async_world is not None)
        if self._async_world is not None:
            validate_v3_packet_context(packet)
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
        asynchronous = {}
        bootstrapped = False
        if self._async_world is not None:
            if self._async_world.current is None:
                if origin != 0 or length != 0:
                    raise RuntimeError('asynchronous bootstrap requires the initial d=0 observation')
                # This wait is part of real first-query latency while physics
                # continues. Synthetic warmup must never substitute a scene.
                published = self._async_world.bootstrap(packet)
                self._install_bridge(published.value)
                bootstrapped = True
            published, asynchronous = self._async_world.poll(packet)
            if self.bridge is not published.value[0]:
                self._install_bridge(published.value)
            age = origin - published.source_index
            refreshed = bootstrapped or asynchronous['published_new']
            refresh_reason = 'initial_continuous_bootstrap' if bootstrapped else (
                'async_published' if asynchronous['published_new'] else (
                'async_submitted' if asynchronous['refresh_submitted'] else 'async_reuse'))
            asynchronous['bootstrap_wait_included'] = bootstrapped
        else:
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
        bootstrap_only = bool(packet.get('_rtc_bootstrap_only',False))
        if bootstrap_only:
            if not bootstrapped or origin != 0 or length != 0:
                raise RuntimeError('bootstrap-only may run once on the initial v3 d=0 query')
            state = np.asarray(packet['state'],dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError('bootstrap hold state must be finite raw [14]')
            actions = np.repeat(state[None],16,axis=0)
            detail = dict(bootstrap_only=True,fast_sampler_ms=0.,preprocess_ms=0.,output_copy_ms=0.)
        else:
            actions, detail = self._sample_rtc_fast(packet, age, raw_prefix, length)
        if not np.isfinite(actions).all():
            raise RuntimeError('MotionKV emitted nonfinite actions')
        if length and not np.allclose(actions[:length], raw_prefix, rtol=0, atol=2e-5):
            raise RuntimeError('normalized RTC sampler altered a committed raw action')
        self.last_policy_timing = dict(detail, text_ms=text_ms, slow_refresh_ms=slow_ms,
            refreshed=refreshed, refresh_reason=refresh_reason, cache_age_actions=age,
            cache_age_seconds=float(packet['sim_time'])-self.bridge_sim_time,
            source_action_index=self.bridge_action_index, observation_action_index=origin,
            prefix_length=length, fast_steps=10,
            source_bridge=self.contract['native_multirate']['source_bridge'],
            motion_interface_version=getattr(self.fast_model.fast_conditioner, 'interface_version', 1),
            rtc_controller=RTC_TIMESTAMPED_CONTROLLER, total_policy_ms=(time.perf_counter()-started)*1000,
            action_interval_seconds=float(self.runtime.action_interval_seconds),
            graph={key: value for key, value in self.fast_graph.stats.items() if key != 'last_signature'},
            asynchronous_world=asynchronous, **self._memory_stats())
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
        current.committed_prefix = prefix.actions
        current.committed_prefix_lengths = prefix.lengths
        # Both train and deploy expose physical endpoint offsets. Training
        # uses recorded simulation timestamps; deployment uses this controller's
        # audited fixed action interval. Neither substitutes model wall latency.
        current.action_target_offsets = (torch.arange(1, noise.shape[1] + 1,
            device=device, dtype=noise.dtype)[None] * float(packet['_rtc_action_interval_s']))
        current.observation_delay_seconds = noise.new_tensor([length * float(packet['_rtc_action_interval_s'])])
        if getattr(self.fast_model.fast_conditioner, 'interface_version', 1) == 3:
            offsets, previous, previous_valid = validate_v3_packet_context(packet)
            current.action_target_offsets = torch.from_numpy(offsets.copy()).to(device=device,dtype=torch.float64)[None]
            current.age.delta_sim_seconds = torch.tensor(float(packet['sim_time'])-self.bridge_sim_time,
                device=device,dtype=torch.float64)
            current.observation_delay_seconds = current.action_target_offsets[:,length-1] if length else current.action_target_offsets.new_zeros(1)
            safe = np.where(previous_valid[:,None],previous,0.)
            previous_tensor = torch.from_numpy(safe).to(device=device,dtype=noise.dtype)[None]
            current.previous_plan = self.runtime.action_normalizer.normalize(previous_tensor)
            current.previous_plan_valid = torch.from_numpy(previous_valid.copy()).to(device)[None]
        self._sync_current_stream(device)
        preprocess_ms = (time.perf_counter()-started)*1000
        sampled = time.perf_counter()
        normalized = self.fast_graph.sample(current, self.bridge, padded, padded_mask, noise, prefix=prefix)
        self._sync_current_stream(device)
        fast_ms = (time.perf_counter()-sampled)*1000
        copied = time.perf_counter()
        actions = self.runtime.action_normalizer.denormalize(normalized).squeeze(0).float().cpu().numpy()
        return actions, dict(preprocess_ms=preprocess_ms, fast_sampler_ms=fast_ms,
            output_copy_ms=(time.perf_counter()-copied)*1000)
