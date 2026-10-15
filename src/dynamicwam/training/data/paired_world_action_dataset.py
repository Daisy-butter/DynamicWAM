"""Pairs an earlier packed sample with a later fast sample inside one episode.

The bridge side keeps the source condition latent, state, and motion. Expert
actions and the clean future latent stay on the fast regularizer. One language
row is shared by the source, the fast path, and that regularizer.
"""

from __future__ import annotations

import json
import hashlib
from array import array
from collections import Counter, defaultdict, OrderedDict
from dynamicwam.training.data.multirate_sampling import partition_episodes, pairing_identity
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.paired_index import (
    pair_fits_episode,
    steady_state_offsets,
    causal_cache_offsets,
    action_target_times,
    jittered_cache_age,
)

CLOCK_KEYS = ("frame_index", "sim_step_index", "sim_time_seconds")


@dataclass(frozen=True, slots=True)
class PairRecord:
    episode_index: int
    source_sample_id: int
    fast_sample_id: int
    source_frame: int
    fast_frame: int
    lang_group_id: int
    lang_id: int
    offset: int
    offset_label: str
    split: str
    task_name: str
    episode_name: str


def episode_lang_id(lang_ids: Iterable[int]) -> int:
    """Pin one language row for every sample in an episode.

    The packed loader indexes language by ``sample_id % len(lang_ids)``, so two
    frames from the same episode can carry different embeddings. Pairs always
    use the first id in the group.
    """

    ids = [int(value) for value in lang_ids]
    if not ids:
        raise ValueError("a language group has no embedding ids")
    return ids[0]


def build_pair_records(
    samples: Iterable[dict[str, Any]],
    *,
    offsets: Iterable[int],
    stress_offsets: Iterable[int] = (),
    splits: Iterable[str],
    chunk_size: int,
    global_downsample_rate: int,
    lang_ids_by_group: dict[int, list[int]],
    total_frames_by_episode: dict[int, int] | None = None,
) -> tuple[list[PairRecord], dict[str, int]]:
    """Build pairs after the episode split. Frames past the episode are dropped."""

    allowed_splits = {str(split) for split in splits}
    if not allowed_splits:
        raise ValueError("paired data requires at least one episode split")
    steady = [int(offset) for offset in offsets]
    stress = [int(offset) for offset in stress_offsets]
    labeled: list[tuple[int, str]] = [(offset, "steady_state") for offset in steady]
    labeled.extend(
        (offset, "stress") for offset in stress if offset not in set(steady)
    )
    if not labeled:
        raise ValueError("paired data requires at least one frame offset")
    for offset, _label in labeled:
        if offset < 0 or offset % int(global_downsample_rate) != 0:
            raise ValueError(
                "frame offsets must be non-negative multiples of "
                f"global_downsample_rate, got {offset}"
            )

    by_episode: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in samples:
        episode_index = int(row["episode_index"])
        frame = int(row["condition_frame_idx"])
        if frame in by_episode[episode_index]:
            raise ValueError(
                f"episode {episode_index} repeats condition frame {frame}"
            )
        by_episode[episode_index][frame] = row

    records: list[PairRecord] = []
    drops: Counter[str] = Counter()
    for episode_index, frames in by_episode.items():
        rows = list(frames.values())
        episode_splits = {str(row.get("split", "")) for row in rows}
        if len(episode_splits) != 1:
            raise ValueError(
                f"episode {episode_index} mixes splits: {sorted(episode_splits)}"
            )
        split = next(iter(episode_splits))
        if split not in allowed_splits:
            drops["split_filtered"] += len(rows) * len(labeled)
            continue
        lang_groups = {int(row["lang_group_id"]) for row in rows}
        if len(lang_groups) != 1:
            raise ValueError(
                f"episode {episode_index} mixes language groups: {sorted(lang_groups)}"
            )
        lang_group_id = next(iter(lang_groups))
        lang_id = episode_lang_id(lang_ids_by_group[lang_group_id])
        total_frames = None
        if total_frames_by_episode is not None:
            total_frames = total_frames_by_episode.get(episode_index)
        for source_frame, row in frames.items():
            for offset, label in labeled:
                fast_frame = int(source_frame) + int(offset)
                if fast_frame not in frames:
                    drops["missing_fast_frame"] += 1
                    continue
                if total_frames is not None and not pair_fits_episode(
                    int(source_frame),
                    fast_frame,
                    total_frames=int(total_frames),
                    chunk_size=int(chunk_size),
                    global_downsample_rate=int(global_downsample_rate),
                ):
                    drops["action_window_past_end"] += 1
                    continue
                fast_row = frames[fast_frame]
                records.append(
                    PairRecord(
                        episode_index=episode_index,
                        source_sample_id=int(row["sample_id"]),
                        fast_sample_id=int(fast_row["sample_id"]),
                        source_frame=int(source_frame),
                        fast_frame=fast_frame,
                        lang_group_id=lang_group_id,
                        lang_id=lang_id,
                        offset=int(offset),
                        offset_label=label,
                        split=split,
                        task_name=str(row["task_name"]),
                        episode_name=str(row["episode_name"]),
                    )
                )
    drops["kept"] = len(records)
    return records, dict(drops)


def episode_media_paths(
    source_root: str | Path,
    *,
    split: str,
    task_name: str,
    episode_name: str,
) -> tuple[Path, Path]:
    root = Path(source_root) / split / task_name
    return (
        root / "videos" / f"{episode_name}.mp4",
        root / "interception" / f"{episode_name}.pt",
    )


def load_episode_clock(path: str | Path) -> dict[str, torch.Tensor]:
    """Read simulator timing and drop every other interception field."""

    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise TypeError(f"interception clock is not a mapping: {path}")
    missing = [key for key in CLOCK_KEYS if key not in blob]
    if missing:
        raise KeyError(f"interception clock {path} is missing {missing}")
    clock = {key: torch.as_tensor(blob[key]).detach().cpu() for key in CLOCK_KEYS}
    return clock


def clock_time_seconds(clock: dict[str, torch.Tensor], frame: int) -> float:
    frame_index = clock["frame_index"].to(dtype=torch.long).reshape(-1)
    matches = (frame_index == int(frame)).nonzero(as_tuple=False)
    if int(matches.numel()) != 1:
        raise KeyError(f"frame {frame} is not a unique interception timestamp")
    position = int(matches[0].item())
    return float(clock["sim_time_seconds"].reshape(-1)[position])


def write_pair_manifest(path: str | Path, records: Iterable[PairRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "episode_index": record.episode_index,
                            "source_sample_id": record.source_sample_id,
                            "fast_sample_id": record.fast_sample_id,
                            "source_frame": record.source_frame,
                            "fast_frame": record.fast_frame,
                            "lang_group_id": record.lang_group_id,
                            "lang_id": record.lang_id,
                            "offset": record.offset,
                            "offset_label": record.offset_label,
                            "split": record.split,
                            "task_name": record.task_name,
                            "episode_name": record.episode_name,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def paired_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([sample[key] for sample in batch], dim=0)

    collated = {
        "source_condition_latent": stack("source_condition_latent"),
        "source_state": stack("source_state"),
        "source_motion": stack("source_motion"),
        "source_motion_interval_valid_mask": stack(
            "source_motion_interval_valid_mask"
        ),
        "source_motion_acceleration_valid_mask": stack(
            "source_motion_acceleration_valid_mask"
        ),
        "bridge_noise_shape_latent": stack("bridge_noise_shape_latent"),
        "fast_rgb": stack("fast_rgb"),
        "fast_rgb_valid_mask": stack("fast_rgb_valid_mask"),
        "fast_state": stack("fast_state"),
        "fast_motion": stack("fast_motion"),
        "fast_motion_interval_valid_mask": stack("fast_motion_interval_valid_mask"),
        "fast_motion_acceleration_valid_mask": stack(
            "fast_motion_acceleration_valid_mask"
        ),
        "fast_action_sequence": stack("fast_action_sequence"),
        "fast_future_latent": stack("fast_future_latent"),
        "fast_condition_latent": stack("fast_condition_latent"),
        "text_embeddings": [sample["text_embedding"] for sample in batch],
        "delta_sim_seconds": torch.tensor(
            [float(sample["delta_sim_seconds"]) for sample in batch],
            dtype=torch.float32,
        ),
        "delta_actions": torch.tensor(
            [int(sample["delta_actions"]) for sample in batch],
            dtype=torch.float32,
        ),
        "state_delta": stack("state_delta"),
        "offset_label": [str(sample["offset_label"]) for sample in batch],
    }
    if "source_id" in batch[0]:
        collated["source_id"] = torch.tensor(
            [int(sample["source_id"]) for sample in batch],
            dtype=torch.long,
        )
        collated["source_sim_time"] = torch.tensor(
            [float(sample["source_sim_time"]) for sample in batch],
            dtype=torch.float32,
        )
    if "joint_targets" in batch[0]:
        collated["source_rgb"] = stack("source_rgb")
        collated["source_rgb_valid_mask"] = stack("source_rgb_valid_mask")
        collated["joint_targets"] = {
            key: torch.stack([sample["joint_targets"][key] for sample in batch])
            for key in batch[0]["joint_targets"]
        }
        for key in ("fast_sim_time",):
            collated[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.float64)
        for key in ("source_sample_id", "fast_sample_id", "cache_age_actions"):
            collated[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.long)
    return collated


class PairedWorldActionDataset(Dataset):
    def __init__(
        self,
        packed_root: str | Path,
        source_root: str | Path,
        *,
        splits: Iterable[str],
        execute_horizon: int,
        every_fast_queries: int,
        global_downsample_rate: int,
        chunk_size: int,
        seconds_per_frame: float,
        stress_offsets: Iterable[int] = (),
        max_open_shards: int = 16,
        subset: str = "train",
        partition_seed: int = 0,
        validation_fraction: float = .1,
        clock_mode: str = "required",
        max_open_clocks: int = 16,
        allow_empty: bool = False,
        frame_reader: Callable[[PairRecord], np.ndarray] | None = None,
        clock_reader: Callable[[PairRecord], dict[str, torch.Tensor] | None]
        | None = None,
        packed: PackedAbsoluteMotionDataset | None = None,
        episode_partition: dict[int, str] | None = None,
        cache_age_actions: Iterable[int] | None = None,
        joint_supervision: bool = False,
        cache_age_jitter: bool = False,
        cache_age_seed: int = 0,
    ) -> None:
        self.packed = packed if packed is not None else PackedAbsoluteMotionDataset(
            packed_root,
            max_open_shards=max_open_shards,
        )
        self.source_root = Path(source_root)
        self.joint_supervision = bool(joint_supervision)
        self.cache_age_jitter = bool(cache_age_jitter)
        self.cache_age_seed = int(cache_age_seed)
        self.epoch = 0
        self.cache_slot_width = int(execute_horizon)
        if self.joint_supervision and clock_mode != "required":
            raise ValueError("joint supervision requires exact simulator timestamps")
        if subset not in {"train", "validation"} or clock_mode not in {"required", "action_only"}:
            raise ValueError("invalid subset or clock_mode")
        self.subset, self.clock_mode = subset, clock_mode
        self.max_open_clocks = max(1, int(max_open_clocks))
        self.chunk_size = int(chunk_size)
        self.global_downsample_rate = int(global_downsample_rate)
        self.seconds_per_frame = float(seconds_per_frame)
        if self.seconds_per_frame <= 0.0:
            raise ValueError("seconds_per_frame must be positive")
        self.frame_reader = frame_reader
        self.clock_reader = clock_reader
        self.allow_empty = bool(allow_empty)
        self._clocks = OrderedDict()
        self._captures: OrderedDict[str, Any] = OrderedDict()
        samples_path = self.packed.root / "samples.jsonl"
        samples = []
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        assignments = episode_partition if episode_partition is not None else partition_episodes(
            samples, seed=partition_seed, validation_fraction=validation_fraction
        )
        self.episode_partition = assignments
        # Same holdout identity even when a new recipe expands causal cache ages.
        self.episode_partition_identity = pairing_identity(samples, assignments,
            seed=partition_seed, validation_fraction=validation_fraction,
            clock_mode=clock_mode, offsets=(), stress_offsets=())
        all_samples = samples
        samples = [row for row in samples if assignments[int(row["episode_index"])] == subset]
        total_frames_by_episode = self._total_frames(samples, splits)
        offsets = steady_state_offsets(
            execute_horizon=int(execute_horizon),
            every_fast_queries=int(every_fast_queries),
            global_downsample_rate=self.global_downsample_rate,
        )
        self.legacy_pairing_identity = pairing_identity(all_samples, assignments, seed=partition_seed,
            validation_fraction=validation_fraction, clock_mode=clock_mode, offsets=offsets,
            stress_offsets=stress_offsets)
        if self.cache_age_jitter:
            support = list(range(int(execute_horizon) * int(every_fast_queries)))
            if cache_age_actions is not None and sorted(cache_age_actions) != support:
                raise ValueError("epoch-jitter age support must cover every action in the refresh cycle")
            # Only one record per slot/source, not one record per possible age.
            self._sample_ids_by_episode_frame = {}
            for row in samples:
                episode, frame = int(row["episode_index"]), int(row["condition_frame_idx"])
                ids = self._sample_ids_by_episode_frame.setdefault(episode, array("q"))
                if frame >= len(ids):
                    ids.extend([-1] * (frame + 1 - len(ids)))
                if ids[frame] != -1:
                    raise ValueError("duplicate packed episode/frame in jitter index")
                ids[frame] = int(row["sample_id"])
        elif cache_age_actions is not None:
            offsets = causal_cache_offsets(cache_age_actions=list(cache_age_actions),
                global_downsample_rate=self.global_downsample_rate)
        self.pairing_identity = pairing_identity(all_samples, assignments, seed=partition_seed,
            validation_fraction=validation_fraction, clock_mode=clock_mode, offsets=offsets,
            stress_offsets=stress_offsets)
        if self.cache_age_jitter:
            self.pairing_identity = hashlib.sha256(json.dumps(dict(
                base_pairing_identity=self.pairing_identity, strategy="causal_slot_epoch_jitter_v1",
                slot_width=self.cache_slot_width, support=support, seed=self.cache_age_seed,
            ), sort_keys=True).encode()).hexdigest()
        self.max_supported_cache_age_actions = (int(execute_horizon) * int(every_fast_queries) - 1
            if self.cache_age_jitter else max(offsets) // self.global_downsample_rate)
        self.records, self.drop_report = build_pair_records(
            samples,
            offsets=offsets,
            stress_offsets=stress_offsets,
            splits=splits,
            chunk_size=self.chunk_size,
            global_downsample_rate=self.global_downsample_rate,
            lang_ids_by_group=self.packed.lang_groups,
            total_frames_by_episode=total_frames_by_episode,
        )
        if not self.records and not self.allow_empty:
            raise RuntimeError(
                f"paired dataset kept no samples under {packed_root}: {self.drop_report}"
            )

    def _total_frames(
        self,
        samples: list[dict[str, Any]],
        splits: Iterable[str],
    ) -> dict[int, int] | None:
        if self.clock_reader is not None:
            return None
        allowed = {str(split) for split in splits}
        totals: dict[int, int] = {}
        seen: set[int] = set()
        for row in samples:
            episode_index = int(row["episode_index"])
            if episode_index in seen:
                continue
            seen.add(episode_index)
            if str(row.get("split", "")) not in allowed:
                continue
            _video_path, clock_path = episode_media_paths(
                self.source_root,
                split=str(row.get("split", "")),
                task_name=str(row["task_name"]),
                episode_name=str(row["episode_name"]),
            )
            if not clock_path.is_file():
                if self.clock_mode == "required":
                    raise FileNotFoundError(f"simulator clock required: {clock_path}")
                continue
            clock = load_episode_clock(clock_path)
            totals[episode_index] = int(clock["frame_index"].max()) + 1
        return totals or None

    def __len__(self) -> int:
        return len(self.records)

    def _stable_language(self, lang_group_id: int) -> torch.Tensor:
        return self.packed._load_language_embedding(int(lang_group_id), 0)

    def __getstate__(self):
        # Never pickle a live decoder handle into a spawned DataLoader worker.
        state = dict(self.__dict__)
        state["_captures"] = OrderedDict()
        state["_clocks"] = OrderedDict()
        return state

    def close(self) -> None:
        while self._captures:
            _path, capture = self._captures.popitem(last=False)
            capture.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def _capture_for(self, video_path: Path):
        import cv2

        key = str(video_path)
        capture = self._captures.get(key)
        if capture is not None:
            self._captures.move_to_end(key)
            return capture
        capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open paired RGB video: {video_path}")
        self._captures[key] = capture
        while len(self._captures) > self.max_open_clocks:
            _old_path, old = self._captures.popitem(last=False)
            old.release()
        return capture

    def _read_frame(self, record: PairRecord) -> np.ndarray:
        import cv2

        if self.frame_reader is not None:
            frame = self.frame_reader(record)
        else:
            video_path, _clock_path = episode_media_paths(
                self.source_root,
                split=record.split,
                task_name=record.task_name,
                episode_name=record.episode_name,
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"paired RGB video is missing: {video_path}")
            capture = self._capture_for(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(record.fast_frame))
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                self._captures.pop(str(video_path), None)
                capture = self._capture_for(video_path)
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(record.fast_frame))
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed to read frame {record.fast_frame} from {video_path}"
                )
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        if frame.dtype != np.uint8:
            raise ValueError(
                f"fast RGB must stay uint8 until the view letterbox, got {frame.dtype}"
            )
        return frame

    def _clock_for(self, record: PairRecord) -> dict[str, torch.Tensor] | None:
        cached = self._clocks.get(record.episode_index)
        if record.episode_index in self._clocks:
            self._clocks.move_to_end(record.episode_index)
            return cached
        if self.clock_reader is not None:
            clock = self.clock_reader(record)
        else:
            _video_path, clock_path = episode_media_paths(
                self.source_root,
                split=record.split,
                task_name=record.task_name,
                episode_name=record.episode_name,
            )
            clock = load_episode_clock(clock_path) if clock_path.is_file() else None
        self._clocks[record.episode_index] = clock
        while len(self._clocks) > self.max_open_clocks:
            self._clocks.popitem(last=False)
        return clock

    def set_epoch(self, epoch: int) -> None:
        # Call before iterating each epoch. Persistent DataLoader workers must
        # be disabled so their copy receives the new epoch (num_workers=0 also works).
        if int(epoch) < 0:
            raise ValueError("dataset epoch must be nonnegative")
        self.epoch = int(epoch)

    def _record_for_epoch(self, index: int) -> PairRecord:
        record = self.records[int(index)]
        if not getattr(self, "cache_age_jitter", False) or record.offset_label != "steady_state":
            return record
        ids = self._sample_ids_by_episode_frame[record.episode_index]
        base_age = record.offset // self.global_downsample_rate
        valid = [age for age in range(base_age, base_age + self.cache_slot_width)
            if record.source_frame + age * self.global_downsample_rate < len(ids)
            and ids[record.source_frame + age * self.global_downsample_rate] >= 0]
        age = jittered_cache_age(source_sample_id=record.source_sample_id,
            base_age=base_age, slot_width=self.cache_slot_width, valid_ages=valid,
            seed=self.cache_age_seed, epoch=self.epoch)
        frame = record.source_frame + age * self.global_downsample_rate
        return replace(record, fast_frame=frame, fast_sample_id=int(ids[frame]),
                       offset=age * self.global_downsample_rate)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record_for_epoch(int(index))
        source = self.packed[record.source_sample_id]
        fast = self.packed[record.fast_sample_id]
        if int(source["episode_index"]) != record.episode_index:
            raise RuntimeError(
                f"source sample {record.source_sample_id} left episode {record.episode_index}"
            )
        if int(fast["episode_index"]) != record.episode_index:
            raise RuntimeError(
                f"fast sample {record.fast_sample_id} left episode {record.episode_index}"
            )
        language = self._stable_language(record.lang_group_id)
        views, masks = fast_views_from_composite(self._read_frame(record))
        clock = self._clock_for(record)
        if self.clock_mode == "action_only":
            delta_seconds = 0.0
            source_sim_time = 0.0
        elif clock is None:
            raise FileNotFoundError(f"simulator clock required for episode {record.episode_index}")
        else:
            source_sim_time = clock_time_seconds(clock, record.source_frame)
            delta_seconds = clock_time_seconds(
                clock, record.fast_frame
            ) - source_sim_time
            if delta_seconds < 0.0:
                raise RuntimeError(
                    "fast frame is earlier in simulator time than the source: "
                    f"episode {record.episode_index} "
                    f"{record.source_frame} -> {record.fast_frame}"
                )
        state_delta = fast["initial_state"] - source["initial_state"]
        sample = {
            "source_condition_latent": source["condition_latent"],
            "source_state": source["initial_state"],
            "source_motion": source["absolute_motion_features"],
            "source_motion_interval_valid_mask": source[
                "absolute_motion_interval_valid_mask"
            ],
            "source_motion_acceleration_valid_mask": source[
                "absolute_motion_acceleration_valid_mask"
            ],
            "bridge_noise_shape_latent": source["future_latent"],
            "fast_rgb": torch.from_numpy(views),
            "fast_rgb_valid_mask": torch.from_numpy(masks),
            "fast_state": fast["initial_state"],
            "fast_motion": fast["absolute_motion_features"],
            "fast_motion_interval_valid_mask": fast[
                "absolute_motion_interval_valid_mask"
            ],
            "fast_motion_acceleration_valid_mask": fast[
                "absolute_motion_acceleration_valid_mask"
            ],
            "fast_action_sequence": fast["action_sequence"],
            "fast_future_latent": fast["future_latent"],
            "fast_condition_latent": fast["condition_latent"],
            "text_embedding": language,
            "delta_sim_seconds": float(delta_seconds),
            "delta_actions": int(record.offset // self.global_downsample_rate),
            "source_id": int(record.source_sample_id),
            "source_sim_time": float(source_sim_time),
            "state_delta": state_delta,
            "offset_label": record.offset_label,
        }
        if getattr(self, "joint_supervision", False):
            source_views, source_masks = (views, masks) if record.offset == 0 else fast_views_from_composite(
                self._read_frame(replace(record, fast_frame=record.source_frame)))
            sample["source_rgb"] = torch.from_numpy(source_views)
            sample["source_rgb_valid_mask"] = torch.from_numpy(source_masks)
            # Label-only namespace: never unpack this mapping into bridge or
            # runtime conditioning. No expert future enters the causal cache.
            frame_indices = clock["frame_index"].reshape(-1).tolist()
            sim_times = clock["sim_time_seconds"].reshape(-1).tolist()
            if len(frame_indices) != len(sim_times) or len(set(frame_indices)) != len(frame_indices):
                raise ValueError("joint targets require unique aligned simulator frame timestamps")
            frame_times = dict(zip(map(int, frame_indices), map(float, sim_times)))
            def targets(frame):
                return torch.tensor(action_target_times(frame_times, frame,
                    chunk_size=self.chunk_size,
                    global_downsample_rate=self.global_downsample_rate), dtype=torch.float64)
            sample["joint_targets"] = {
                "source_future_latent": source["future_latent"],
                "source_action_sequence": source["action_sequence"],
                "source_action_target_times": targets(record.source_frame),
                "fast_action_target_times": targets(record.fast_frame),
            }
            sample.update(fast_sim_time=source_sim_time + delta_seconds,
                source_sample_id=record.source_sample_id, fast_sample_id=record.fast_sample_id,
                cache_age_actions=record.offset // self.global_downsample_rate)
        forbidden = {
            "future_latent",
            "action_sequence",
            "clean_future_latent",
            "expert_actions",
            "video_frames",
        }
        leaked = sorted(forbidden.intersection(sample))
        if leaked:
            raise RuntimeError(f"paired sample exposed source labels: {leaked}")
        return sample
