"""Compact all-age plan pairs with strictly causal, timestamped head history.

The numeric index uses 20 bytes per pair, not sixteen copies of Python
dataclasses. Age groups are contiguous ranges, so the sampler stores no
per-pair Python integer lists. Original paired-data/runtime behavior is intact.
"""
from __future__ import annotations

from array import array
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.multirate_sampling import partition_episodes, pairing_identity
from dynamicwam.training.data.paired_world_action_dataset import (
    PairedWorldActionDataset, PairRecord, clock_time_seconds, episode_lang_id, paired_collate_fn,
)
from dynamicwam.training.data.paired_index import pair_fits_episode

AGES = tuple(range(16))
HISTORY_RAW_FRAME_LAGS = (1, 2, 3, 6, 9, 12, 18, 24, 36)


def causal_history_frame(record, clock, *, seed):
    """Choose an actual past image deterministically, never relabel its time.

    A missing timestamp inside the available clock range is corrupt required
    data, not permission to silently select a different, easier history lag.
    Early episode frames may have fewer supported lags or no history at all.
    """
    frames = clock["frame_index"].to(dtype=torch.long).reshape(-1)
    first_frame = int(frames.min())
    current_time = clock_time_seconds(clock, record.fast_frame)
    if not np.isfinite(current_time):
        raise ValueError("required current history timestamp must be finite")
    candidates = []
    for lag in HISTORY_RAW_FRAME_LAGS:
        previous = record.fast_frame - lag
        if previous < first_frame:
            continue
        previous_time = clock_time_seconds(clock, previous)
        dt = current_time - previous_time
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("required history timestamps must be finite and causal")
        # Multiple rendered frames can share one simulator tick. They are real
        # images but supply no elapsed motion interval; do not invent epsilon
        # time or count them as valid history. Missing clock entries still raise.
        if dt == 0:
            continue
        candidates.append((previous, lag, dt))
    if not candidates:
        return -1, 0, 0.
    identity = [int(seed), record.episode_index, record.source_sample_id,
                record.fast_sample_id, record.source_frame, record.fast_frame]
    hashed = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
    return candidates[int.from_bytes(hashed[:8], "big") % len(candidates)]


class CompactTemporalRecords:
    """A virtual PairRecord sequence backed by five int32 columns per row."""
    def __init__(self, values, episodes, groups, rate):
        if values.itemsize != 4:
            raise RuntimeError("compact temporal index requires 32-bit array integers")
        self.values = values
        self.rows = np.frombuffer(values, dtype=np.int32).reshape(-1, 5)
        self.episodes, self.groups, self.rate = episodes, groups, int(rate)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        episode, source_id, fast_id, source_frame, age = map(int, self.rows[index])
        meta = self.episodes[episode]
        return PairRecord(episode, source_id, fast_id, source_frame, source_frame + age * self.rate,
            meta["lang_group_id"], meta["lang_id"], age * self.rate, "steady_state",
            meta["split"], meta["task_name"], meta["episode_name"])

    @property
    def age_counts(self):
        return {age: sum(stop - start for groups in self.groups.values()
                        for age_key, (start, stop) in groups.items() if age_key == age) for age in AGES}


def build_compact_records(frames, episodes, assignments, *, subset, splits, rate, totals):
    values, groups, drops = array("i"), {}, Counter()
    allowed = set(splits)
    for episode in sorted(frames):
        meta = episodes[episode]
        if assignments[episode] != subset or meta["split"] not in allowed:
            continue
        episode_frames = frames[episode]
        source_frames = sorted(episode_frames)
        age_groups = {}
        for age in AGES:
            start = len(values) // 5
            for source_frame in source_frames:
                fast_frame = source_frame + age * rate
                if fast_frame not in episode_frames:
                    drops["missing_fast_frame"] += 1
                    continue
                if totals and not pair_fits_episode(source_frame, fast_frame,
                        total_frames=totals[episode], chunk_size=16, global_downsample_rate=rate):
                    drops["action_window_past_end"] += 1
                    continue
                row = (episode, episode_frames[source_frame], episode_frames[fast_frame], source_frame, age)
                if any(value < 0 or value > 2**31 - 1 for value in row):
                    raise OverflowError("temporal index exceeds nonnegative int32")
                values.extend(row)
            stop = len(values) // 5
            if stop > start:
                age_groups[age] = (start, stop)
        if age_groups:
            groups[episode] = age_groups
    drops["kept"] = len(values) // 5
    return CompactTemporalRecords(values, episodes, groups, rate), dict(drops)


class TemporalFeedbackDataset(PairedWorldActionDataset):
    def __init__(self, packed_root, source_root, *, splits, global_downsample_rate=3,
                 seconds_per_frame=1/30, max_open_shards=4, subset="train", partition_seed=0,
                 validation_fraction=.1, clock_mode="required", max_open_clocks=4,
                 allow_empty=False, frame_reader=None, clock_reader=None, packed=None,
                 episode_partition=None):
        # Do not call the parent constructor: its 16-age Python record list
        # creates a large transient and duplicates sampler index objects.
        if subset not in {"train", "validation"} or clock_mode != "required":
            raise ValueError("temporal feedback requires a held-out subset and real simulator clocks")
        if global_downsample_rate != 3 or seconds_per_frame <= 0:
            raise ValueError("current temporal contract requires the original three-frame action mapping")
        self.packed = packed if packed is not None else PackedAbsoluteMotionDataset(
            packed_root, max_open_shards=max_open_shards)
        self.source_root, self.subset, self.clock_mode = Path(source_root), subset, clock_mode
        self.chunk_size, self.global_downsample_rate = 16, int(global_downsample_rate)
        self.seconds_per_frame, self.max_open_clocks = float(seconds_per_frame), max(1, int(max_open_clocks))
        self.frame_reader, self.clock_reader = frame_reader, clock_reader
        self.history_seed = int(partition_seed)
        self.history_raw_frame_lags = HISTORY_RAW_FRAME_LAGS
        self.allow_empty = bool(allow_empty)
        self._clocks, self._captures = OrderedDict(), OrderedDict()
        frames, episodes = defaultdict(dict), {}
        with (self.packed.root / "samples.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                episode, frame = int(row["episode_index"]), int(row["condition_frame_idx"])
                meta = dict(episode_index=episode, split=str(row["split"]), task_name=str(row["task_name"]),
                    episode_name=str(row["episode_name"]), lang_group_id=int(row["lang_group_id"]))
                if episode in episodes and episodes[episode] != meta:
                    raise ValueError("inconsistent identity/language within an episode")
                if frame in frames[episode]:
                    raise ValueError("duplicate condition frame in temporal dataset")
                episodes[episode] = meta
                frames[episode][frame] = int(row["sample_id"])
        metadata = list(episodes.values())
        self.episode_partition = episode_partition if episode_partition is not None else partition_episodes(
            metadata, seed=partition_seed, validation_fraction=validation_fraction)
        if set(self.episode_partition) != set(episodes):
            raise ValueError("episode partition does not match packed metadata")
        self.pairing_identity = pairing_identity(metadata, self.episode_partition, seed=partition_seed,
            validation_fraction=validation_fraction, clock_mode=clock_mode,
            offsets=[age * self.global_downsample_rate for age in AGES], stress_offsets=())
        selected = [row for row in metadata if self.episode_partition[row["episode_index"]] == subset]
        totals = self._total_frames(selected, splits)
        for meta in episodes.values():
            meta["lang_id"] = episode_lang_id(self.packed.lang_groups[meta["lang_group_id"]])
        self.records, self.drop_report = build_compact_records(frames, episodes, self.episode_partition,
            subset=subset, splits=splits, rate=self.global_downsample_rate, totals=totals)
        self.index_bytes = self.records.rows.nbytes
        if not len(self.records) and not allow_empty:
            raise RuntimeError(f"no temporal pairs: {self.drop_report}")
        if len(self.records) and any(n <= 0 for n in self.records.age_counts.values()):
            raise RuntimeError("temporal dataset must cover every age 0..15")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        record = self.records[int(index)]
        clock = self._clock_for(record)
        if clock is None:
            raise FileNotFoundError("temporal history requires the original simulator clock")
        previous_frame, frame_offset, dt = causal_history_frame(record, clock, seed=self.history_seed)
        valid = previous_frame >= 0
        previous_head = sample["fast_rgb"][0].clone()
        previous_mask = torch.zeros_like(sample["fast_rgb_valid_mask"][0], dtype=torch.bool)
        if valid:
            views, masks = fast_views_from_composite(self._read_frame(replace(record, fast_frame=previous_frame)))
            previous_head, previous_mask = torch.from_numpy(views[0]), torch.from_numpy(masks[0])
        sample.update(previous_head=previous_head, previous_head_mask=previous_mask,
                      history_dt=float(dt), history_valid=valid,
                      history_frame_offset=int(frame_offset), history_frame_index=int(previous_frame))
        return sample


def temporal_collate_fn(batch):
    result = paired_collate_fn(batch)
    for key in ("previous_head", "previous_head_mask"):
        result[key] = torch.stack([sample[key] for sample in batch])
    result["history_dt"] = torch.tensor([s["history_dt"] for s in batch], dtype=torch.float32)
    result["history_valid"] = torch.tensor([s["history_valid"] for s in batch], dtype=torch.bool)
    result["history_frame_offset"] = torch.tensor([s["history_frame_offset"] for s in batch], dtype=torch.long)
    result["history_frame_index"] = torch.tensor([s["history_frame_index"] for s in batch], dtype=torch.long)
    return result


class TemporalPairSampler(Sampler):
    """Episode-balanced sampling from ranges; validation starts with all ages."""
    def __init__(self, records, *, samples_per_episode=10, seed=0, validation=False):
        if samples_per_episode < 1:
            raise ValueError("positive episode sample quota required")
        self.groups = records.groups
        self.episodes = sorted(self.groups, key=lambda e: (records.episodes[e]["task_name"], e))
        self.samples_per_episode, self.seed = int(samples_per_episode), int(seed)
        self.validation, self.epoch = bool(validation), 0
        self.age_episodes = {age: [ep for ep in self.episodes if age in self.groups[ep]] for age in AGES}
        if any(not episodes for episodes in self.age_episodes.values()):
            raise ValueError("sampler cannot cover every plan age")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.episodes) * self.samples_per_episode

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for episode in self.episodes:
            ages = sorted(self.groups[episode])
            for _ in range(self.samples_per_episode):
                start, stop = self.groups[episode][rng.choice(ages)]
                indices.append(rng.randrange(start, stop))
        rng.shuffle(indices)
        if self.validation:
            if len(indices) < 16:
                raise RuntimeError("validation needs at least one pair per age")
            # All ranks slice this same sequence, so its first complete rounds
            # jointly cover all ages even when validation is deliberately short.
            for age in AGES:
                ep = rng.choice(self.age_episodes[age])
                start, stop = self.groups[ep][age]
                indices[age] = rng.randrange(start, stop)
        return iter(indices)
