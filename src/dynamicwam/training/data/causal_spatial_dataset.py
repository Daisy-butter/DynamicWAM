"""Causal spatial observations over the verified physical-grid data.

Labels are real expert futures. A finite RGB/flow window is recomputed from
recorded observations; neither an expert future nor a fabricated policy rollout
is used to initialize the belief. ``executed_actions`` is an API compatibility
name: these are capture-time drive targets, NOT dense executed-control history.
The same window/flow helpers are suitable for online observations.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "causal_spatial_v1"
COMMAND_HISTORY_SEMANTICS = "recorded_capture_drive_targets_not_dense_executed_control"


from dynamicwam.observations.causal_spatial_window import (
    causal_window_indices, future_support_end_offsets, farneback_pair, composite_observation, flow_pair_validity,
    causal_plan_position,
)


class CausalSpatialDataset:
    """Wrap ``PhysicalGridPairedWorldActionDataset`` without rebuilding latents."""
    def __init__(self, base, *, window_size=4, window_stride_seconds=.32,
                 prefix_length=2, force_fresh_memory=False, flow_backend=None,
                 observation_backend=None, max_cached_observations=32, plan_stride_seconds=.32):
        if int(window_size) < 1 or not np.isfinite(window_stride_seconds) or window_stride_seconds <= 0:
            raise ValueError("invalid finite observation window")
        if not 0 <= int(prefix_length) < 16:
            raise ValueError("RTC prefix must leave a trainable suffix")
        self.base = base
        self.window_size, self.window_stride_seconds = int(window_size), float(window_stride_seconds)
        self.prefix_length, self.force_fresh_memory = int(prefix_length), bool(force_fresh_memory)
        self.plan_stride_seconds = float(plan_stride_seconds)
        if not np.isfinite(self.plan_stride_seconds) or self.plan_stride_seconds <= 0:
            raise ValueError("previous fast-query interval must be positive")
        self.flow_backend = flow_backend or farneback_pair
        self.observation_backend = observation_backend or composite_observation
        self.max_cached_observations = int(max_cached_observations)
        if self.max_cached_observations < 0:
            raise ValueError("negative observation cache size")
        self._observations, self._flows = OrderedDict(), OrderedDict()
        self.contract = dict(schema=SCHEMA, window_size=self.window_size,
            window_stride_seconds=self.window_stride_seconds, rgb="native_three_view_imagenet_160",
            flow="farneback_current_grid_displacement_fb_consistency_v1",
            actual_capture_dt=True, command_history_semantics=COMMAND_HISTORY_SEMANTICS,
            prefix_source="expert_demonstration", prefix_length=self.prefix_length,
            previous_plan_source="absent_unless_explicit_teacher_proposal_or_real_replay",
            teacher_plan_observation="last_eligible_capture_at_or_before_current_minus_fast_query_interval",
            teacher_plan_query_interval_seconds=self.plan_stride_seconds,
            future_time_semantics="requested_grid_group_support_end_action_endpoints_8_16",
            future_label_time_semantics="earlier_actual_capture_support_end_max_lag_0.16_seconds",
            on_policy_ready=False, counterfactual_ready=False)
        self.pairing_identity = hashlib.sha256(json.dumps(dict(
            base=getattr(base, "pairing_identity", None), contract=self.contract),
            sort_keys=True).encode()).hexdigest()

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def set_force_fresh_memory(self, enabled):
        # The existing record index has age buckets. Mutating base.max_age to
        # zero would leave stale bucket bounds and can corrupt random sampling.
        self.force_fresh_memory = bool(enabled)

    def _cached(self, cache, key, create):
        if key not in cache:
            value = create()
            if not self.max_cached_observations:
                return value
            cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_cached_observations:
            cache.popitem(last=False)
        return cache[key]

    def __getitem__(self, index):
        import torch
        record = self.base._make_record(int(index))
        sample = self.base[int(index)]
        episode = self.base._episode(record.episode_index)
        frames, times, dt, transition_valid = causal_window_indices(episode["times"], record.fast_frame,
            window_size=self.window_size, stride_seconds=self.window_stride_seconds)
        rgb, raw, valid = [], [], []
        for frame in frames:
            key = (int(record.episode_index), int(frame))
            views = self._cached(self._observations, key, lambda frame=int(frame):
                self.observation_backend(self.base._read_frame(replace(record, fast_frame=frame))))
            rgb.append(views[0]); raw.append(views[1]); valid.append(views[2])
        rgb, raw, valid = np.asarray(rgb), np.asarray(raw), np.asarray(valid, dtype=bool)
        t, v, _, h, w = rgb.shape
        flow = np.zeros((t, v, 2, h, w), dtype=np.float32)
        flow_valid = np.zeros((t, v, 1, h, w), dtype=bool)
        for step in range(1, t):
            if not transition_valid[step]:
                continue
            for view in range(v):
                key = (int(record.episode_index), int(frames[step - 1]), int(frames[step]), view)
                delta, mask = self._cached(self._flows, key, lambda step=step, view=view:
                    self.flow_backend(raw[step - 1, view], raw[step, view]))
                mask = flow_pair_validity(delta, mask, valid[step - 1, view], valid[step, view])
                flow[step, view] = np.where(mask[..., None], delta, 0).transpose(2, 0, 1)
                flow_valid[step, view, 0] = mask
        # Target at the start of each observed transition, with honest validity.
        previous_frames = np.concatenate((frames[:1], frames[:-1]))
        commands = self.base.normalizer.normalize(torch.from_numpy(
            np.asarray(episode["commands"][previous_frames], dtype=np.float32).copy()))
        commands[~torch.from_numpy(transition_valid)] = 0
        targets = sample["joint_targets"]
        current_time = float(episode["times"][record.fast_frame])
        source_time = float(episode["times"][record.source_frame])
        source_id = int(record.source_sample_id)
        fresh = self.force_fresh_memory
        if fresh:
            source_time, source_id = current_time, int(record.fast_sample_id)
        action_times = targets["fast_action_target_times"].to(torch.float64)
        source_action_times = targets["fast_action_target_times" if fresh else "source_action_target_times"].to(torch.float64)
        actions = sample["fast_action_sequence"].to(torch.float32)
        prefix = torch.zeros_like(actions)
        prefix[:self.prefix_length] = actions[:self.prefix_length]
        source_prefix = "fast" if fresh else "source"
        future_actual = torch.from_numpy(future_support_end_offsets(
            targets["fast_video_actual_times"].cpu().numpy(), current_time))
        future_requested = action_times[[7, 15]] - current_time
        future_lag = future_requested - future_actual
        if torch.any(future_lag < -1e-9) or torch.any(future_lag > .16 + 1e-9):
            raise ValueError("future cached label capture outside the declared requested-time lag budget")
        result = dict(
            clean_actions=actions, clean_future=sample["fast_future_latent"].to(torch.float32),
            clean_current_condition=sample["fast_condition_latent"],
            source_condition_latent=sample["fast_condition_latent" if fresh else "source_condition_latent"],
            source_text_embeddings=sample["text_embedding"], text_embeddings=sample["text_embedding"],
            source_time=torch.tensor(source_time, dtype=torch.float64),
            current_time=torch.tensor(current_time, dtype=torch.float64),
            source_id=source_id, current_id=int(record.fast_sample_id),
            observation_age=torch.tensor(current_time - source_time, dtype=torch.float64),
            source_state=sample[source_prefix + "_state"],
            source_motion=sample[source_prefix + "_motion"],
            source_motion_interval_valid_mask=sample[source_prefix + "_motion_interval_valid_mask"],
            source_motion_acceleration_valid_mask=sample[source_prefix + "_motion_acceleration_valid_mask"],
            source_action_target_offsets=source_action_times - source_time,
            source_action_target_times=source_action_times,
            initial_state=sample["fast_state"], absolute_motion_features=sample["fast_motion"],
            absolute_motion_interval_valid_mask=sample["fast_motion_interval_valid_mask"],
            absolute_motion_acceleration_valid_mask=sample["fast_motion_acceleration_valid_mask"],
            action_target_offsets=action_times - current_time, action_target_times=action_times,
            future_target_offsets=future_requested,
            # Loss/audit metadata ONLY. Future realized capture clocks are not
            # available to deployment, so these must never condition the model.
            future_actual_support_offsets=future_actual, future_support_lag=future_lag,
            spatial_rgb=torch.from_numpy(rgb.astype(np.float32)), spatial_flow=torch.from_numpy(flow),
            spatial_valid=torch.from_numpy(valid[:, :, None]), spatial_flow_valid=torch.from_numpy(flow_valid),
            spatial_dt=torch.from_numpy(dt), spatial_times=torch.from_numpy(times),
            spatial_frame_indices=torch.from_numpy(frames),
            executed_actions=commands, executed_valid=torch.from_numpy(transition_valid),
            rtc_prefix=prefix, rtc_prefix_lengths=self.prefix_length,
            previous_plan=torch.zeros_like(actions), previous_plan_valid=torch.zeros(16, dtype=torch.bool),
            command_history_semantics=COMMAND_HISTORY_SEMANTICS, prefix_source="expert_demonstration",
            previous_plan_source="absent", target_source="expert_demonstration", schema=SCHEMA,
        )
        if float(result["observation_age"]) < -1e-9 or torch.any(action_times <= current_time):
            raise ValueError("noncausal memory/action target clock")
        plan_position = causal_plan_position(episode["times"], episode["eligible"], record.fast_frame,
            stride_seconds=self.plan_stride_seconds)
        # Native teacher sampling may use ONLY this prior observation bundle,
        # pure noise and its target timestamps. No expert plan/future is exposed.
        result.update(plan_condition_latent=torch.zeros_like(sample["fast_condition_latent"]),
            plan_state=torch.zeros_like(sample["fast_state"]), plan_motion=torch.zeros_like(sample["fast_motion"]),
            plan_motion_interval_valid_mask=torch.zeros_like(sample["fast_motion_interval_valid_mask"]),
            plan_motion_acceleration_valid_mask=torch.zeros_like(sample["fast_motion_acceleration_valid_mask"]),
            plan_text_embeddings=sample["text_embedding"], plan_time=torch.tensor(0., dtype=torch.float64),
            plan_id=-1, plan_context_valid=False, plan_action_target_times=torch.zeros_like(action_times),
            plan_action_target_offsets=torch.zeros_like(action_times))
        if plan_position is not None:
            plan_frame, plan_id = map(int, episode["eligible"][plan_position])
            plan_observation = self.base.packed[plan_id]
            if int(plan_observation["episode_index"]) != int(record.episode_index):
                raise ValueError("teacher plan observation left the current episode")
            plan_time = float(episode["times"][plan_frame])
            plan_times = torch.from_numpy(np.asarray(episode["action_times"][plan_position], dtype=np.float64).copy())
            result.update(plan_condition_latent=plan_observation["condition_latent"],
                plan_state=plan_observation["initial_state"], plan_motion=plan_observation["absolute_motion_features"],
                plan_motion_interval_valid_mask=plan_observation["absolute_motion_interval_valid_mask"],
                plan_motion_acceleration_valid_mask=plan_observation["absolute_motion_acceleration_valid_mask"],
                plan_time=torch.tensor(plan_time, dtype=torch.float64), plan_id=plan_id, plan_context_valid=True,
                plan_action_target_times=plan_times, plan_action_target_offsets=plan_times - plan_time)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_observations"], state["_flows"] = OrderedDict(), OrderedDict()
        return state


def collate_causal_spatial(batch):
    """Stack numeric fields, preserve variable-length language as a list."""
    import torch
    if not batch or any(sample.get("schema") != SCHEMA for sample in batch):
        raise ValueError("empty or mixed causal-spatial batch")
    if any(set(sample) != set(batch[0]) for sample in batch):
        raise ValueError("causal-spatial field presence differs within batch")
    result = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in ("source_text_embeddings", "text_embeddings", "plan_text_embeddings"):
            result[key] = values
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, bool)):
            result[key] = torch.as_tensor(values)
        else:
            if any(value != values[0] for value in values):
                raise ValueError("mixed provenance field: " + key)
            result[key] = values[0]
    return result


def audit_causal_spatial_coverage(physical_grid_root, future_latent_root, *, verify_hashes=False, require_full=True):
    """CPU preflight: every episode's clocks, labels, cache IDs and geometry.

    Optional byte hashing verifies the existing full-cache provenance. This
    audit does not claim every latent value is finite; each selected sample is
    checked by the existing loader. It does not collect on-policy replay.
    """
    from safetensors import safe_open
    from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
    from dynamicwam.training.data.physical_grid import validate_clock
    root, latent = Path(physical_grid_root), Path(future_latent_root)
    readiness = validate_physical_grid_readiness(root, latent, verify_hashes=verify_hashes, require_full=require_full)
    episodes = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines() if line]
    counts = dict(episodes=0, samples=0, train_samples=0, validation_samples=0, duplicate_capture_intervals=0)
    eligible_seen = []
    max_capture_lag = 0.0
    for meta in episodes:
        with np.load(root / meta["index_path"], allow_pickle=False) as ep:
            validate_clock(ep["frames"], ep["times"], ep["commands"])
            counts["duplicate_capture_intervals"] += int(np.sum(np.diff(ep["times"]) == 0))
            eligible = ep["eligible"]
            n = len(eligible)
            if eligible.shape != (n, 2) or n != meta["eligible_samples"]:
                raise ValueError("eligible sample geometry/count changed")
            if not n:
                counts["episodes"] += 1
                continue
            if np.any(np.diff(eligible[:, 1]) <= 0) or np.any(np.diff(eligible[:, 0]) <= 0) or np.any(eligible[:, 0] < 0) or np.any(eligible[:, 0] >= len(ep["times"])):
                raise ValueError("invalid eligible sample IDs/frame indices")
            current = ep["times"][eligible[:, 0]]
            at, vt = ep["action_times"], ep["video_actual_times"]
            if at.shape != (n, 16) or vt.shape != (n, 8) or ep["video_frames"].shape != (n, 8):
                raise ValueError("physical target geometry changed")
            if not np.isfinite(at).all() or not np.isfinite(vt).all() or np.any(at <= current[:, None]):
                raise ValueError("nonfinite/nonfuture action targets")
            if np.any(np.diff(at, axis=1) <= 0) or np.any(np.diff(vt, axis=1) < 0) or np.any(vt[:, [3, 7]] <= current[:, None]):
                raise ValueError("invalid future temporal support")
            if not np.allclose(np.diff(at, axis=1), .16, atol=1e-9, rtol=0):
                raise ValueError("action grid period differs from trained physical time")
            video_frames = ep["video_frames"]
            if np.any(video_frames < 0) or np.any(video_frames >= len(ep["times"])) or not np.allclose(vt, ep["times"][video_frames], atol=1e-10, rtol=0):
                raise ValueError("future frame/clock identity mismatch")
            lags = ep["video_lags"]
            if lags.shape != (n, 8) or not np.isfinite(lags).all() or np.any(lags < -1e-9) or np.any(lags > .16 + 1e-9):
                raise ValueError("future capture uses an image after its target endpoint")
            if not np.allclose(vt + lags, at[:, 1::2], atol=1e-9, rtol=0):
                raise ValueError("future label capture/lag differs from requested video grid")
            max_capture_lag = max(max_capture_lag, float(np.max(lags)))
            path = latent / "episodes" / f"{int(meta['episode_index']):06d}.safetensors"
            with safe_open(str(path), framework="np") as handle:
                if set(handle.keys()) != {"sample_ids", "future_latents"}:
                    raise ValueError("unexpected latent shard tensor schema")
                if handle.get_slice("future_latents").get_shape() != [n, 48, 2, 12, 10]:
                    raise ValueError("future latent geometry/count changed")
                if not np.array_equal(handle.get_tensor("sample_ids"), eligible[:, 1]):
                    raise ValueError("cache sample IDs do not match physical-grid index")
            eligible_seen.append(eligible[:, 1].copy())
            counts["samples"] += n
            counts[meta["subset"] + "_samples"] += n
            counts["episodes"] += 1
    all_ids = np.concatenate(eligible_seen) if eligible_seen else np.empty(0, dtype=np.int64)
    manifest_ids = np.load(root / "eligible_sample_ids.npy", allow_pickle=False)
    if len(np.unique(all_ids)) != len(all_ids) or not np.array_equal(np.sort(all_ids), np.sort(manifest_ids)):
        raise ValueError("full sample coverage is duplicated or differs from the verified index")
    if counts["samples"] != readiness["samples"]:
        raise ValueError("audited sample total differs from readiness manifest")
    return dict(schema=SCHEMA, readiness=readiness, counts=counts, max_capture_lag_seconds=max_capture_lag,
        command_history_semantics=COMMAND_HISTORY_SEMANTICS,
        future_time_semantics="requested_grid_group_support_end_action_endpoints_8_16",
        future_label_time_semantics="earlier_actual_capture_support_end_max_lag_0.16_seconds",
        full_episode_clock_scan=True, full_cache_header_and_id_scan=True,
        all_latent_values_finite_scanned=False, dense_flow_precomputed=False,
        on_policy_ready=False, counterfactual_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-grid-root", required=True)
    parser.add_argument("--future-latent-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--allow-smoke-subset", action="store_true")
    args = parser.parse_args()
    report = audit_causal_spatial_coverage(args.physical_grid_root, args.future_latent_root,
        verify_hashes=args.verify_hashes, require_full=not args.allow_smoke_subset)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
