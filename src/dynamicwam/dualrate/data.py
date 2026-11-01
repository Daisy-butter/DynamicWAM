"""Auditable offline labels for DualRate; no simulator truth enters policy inputs.

The producer consumes an already decoded *single* episode. Feature extraction is
deliberately outside this module: features must come from the named frozen encoder
on real future images, with instance identity and visibility verified upstream.
Tokens are entity-major, time-minor: object/support/left/right x .1/.2/.4/.8 s.
Fields 0:128 are fixed visual features; 128:131 are world xyz; 131:134 are
entity minus the declared relevant gripper xyz. Gripper relative fields are masked
(they would duplicate/collapse their absolute position). Contact is [token, hand].
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

FUTURE_OFFSETS = (0.1, 0.2, 0.4, 0.8)
ENTITY_ROLES = ("object", "support", "left_gripper", "right_gripper")
VISUAL_DIM = 128
FUTURE_DIM = 134
SCHEMA_VERSION = "dualrate-labels-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EpisodeRecords:
    episode_id: str
    scene_id: str
    split: str
    visual_encoder_id: str
    entity_ids: tuple[str | None, str | None, str | None, str | None]
    timestamps: Tensor                      # [N], seconds, simulation clock
    entity_world_positions: Tensor          # [N,4,3]
    gripper_positions: Tensor               # [N,2,3], same world frame
    frozen_visual_features: Tensor          # [N,4,128], real image crops
    entity_valid: Tensor                    # [N,4], bool
    gripper_valid: Tensor                   # [N,2], bool
    visibility: Tensor                     # [N,4], bool
    projectability: Tensor                 # [N,4], bool
    track_ids: Tensor                       # [N,4], int; -1 means unavailable
    contacts: Tensor                        # [N,4,2] or task-target-only [N,2]
    contact_valid: Tensor                   # same shape as contacts, bool
    relevant_gripper: tuple[int, int] = (0, 0)  # object/support, fixed per episode
    visual_context_valid: Tensor | None = None  # explicit projected-context semantics; not visibility


@dataclass(frozen=True)
class FutureLabels:
    future: Tensor                         # [16,134], invalid fields = 0
    future_loss_valid: Tensor               # [16,134], label-only field mask
    contact: Tensor                        # [16,2], exact/endpoint-agreement labels
    contact_valid: Tensor                  # [16,2]
    future_times: Tensor                   # [16], absolute physical seconds
    query_time: float
    episode_id: str
    scene_id: str
    split: str
    visual_encoder_id: str

    def tensors(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).clone() for name in
                ("future", "future_loss_valid", "contact", "contact_valid", "future_times")}


def _episode_indices(records: EpisodeRecords, duplicate_policy: str) -> Tensor:
    t = records.timestamps
    _require(t.device.type == "cpu" and t.ndim == 1 and len(t) > 0,
             "timestamps must be a nonempty CPU vector")
    _require(bool(torch.isfinite(t).all()), "nonfinite timestamps")
    delta = t[1:] - t[:-1]
    _require(bool((delta >= 0).all()), "timestamps are not ordered")
    _require(duplicate_policy in ("error", "last"), "unknown duplicate policy")
    if bool((delta == 0).any()) and duplicate_policy == "error":
        raise ValueError("duplicate timestamps; choose explicit last-value policy")
    # Last *whole record* at each timestamp, never mix fields from different rows.
    return torch.nonzero(torch.cat((delta != 0, torch.ones(1, dtype=torch.bool))),
                         as_tuple=False).flatten()


def _validate_records(r: EpisodeRecords) -> None:
    _require(all(isinstance(x, str) and x for x in
                 (r.episode_id, r.scene_id, r.split, r.visual_encoder_id)),
             "episode/scene/split/encoder identity required")
    _require(r.split in ("train", "validation", "test"), "unknown split")
    _require(len(r.entity_ids) == 4 and len({x for x in r.entity_ids if x}) ==
             len([x for x in r.entity_ids if x]), "entity identities must be distinct")
    _require(all(x in (0, 1) for x in r.relevant_gripper), "unknown relevant gripper")
    n = len(r.timestamps)
    shapes = {"entity_world_positions": (n,4,3), "gripper_positions": (n,2,3),
              "frozen_visual_features": (n,4,128), "entity_valid": (n,4),
              "gripper_valid": (n,2), "visibility": (n,4), "projectability": (n,4),
              "track_ids": (n,4)}
    _require(tuple(r.contacts.shape) in ((n,2), (n,4,2)), "contact shape must be [N,2] or [N,4,2]")
    shapes["contacts"] = tuple(r.contacts.shape)
    shapes["contact_valid"] = tuple(r.contacts.shape)
    for name, shape in shapes.items():
        x = getattr(r, name)
        _require(tuple(x.shape) == shape and x.device.type == "cpu", f"{name}: expected CPU {shape}")
        _require(not x.requires_grad, f"{name} must be a fixed label, not a learned target")
    for name in ("entity_valid", "gripper_valid", "visibility", "projectability", "contact_valid"):
        _require(getattr(r, name).dtype == torch.bool, f"{name} must be boolean")
    _require(not r.track_ids.is_floating_point(), "track IDs must be integer")
    for e, identity in enumerate(r.entity_ids):
        if identity is None:
            _require(not bool(r.entity_valid[:, e].any()), "missing identity marked valid")
    _require(bool(((r.contacts == 0) | (r.contacts == 1) | ~r.contact_valid).all()),
             "valid contacts must be binary")
    for hand in (0, 1):
        valid = r.entity_valid[:, hand + 2] & r.gripper_valid[:, hand]
        if bool(valid.any()):
            _require(torch.allclose(r.entity_world_positions[valid, hand+2],
                                    r.gripper_positions[valid, hand], atol=1e-5, rtol=0),
                     "gripper entity and gripper state use inconsistent coordinates")


def produce_future_labels(records: EpisodeRecords, query_time: float, *,
                          max_interpolation_gap: float,
                          max_contact_age: float,
                          max_visual_time_error: float = 0.030001,
                          duplicate_policy: str = "error", target_offsets=None) -> FutureLabels:
    """Interpolate continuous targets only inside valid, identity-stable brackets.

    Contact uses exact observations or agreeing valid bracket endpoints, with an
    explicit age bound. Unknown transition times are masked, never interpolated.
    Nothing is extrapolated beyond the
    episode's last timestamp. Missing data produce false masks, not fake targets.
    """
    _validate_records(records)
    _require(math.isfinite(query_time), "query_time must be finite")
    _require(math.isfinite(max_interpolation_gap) and max_interpolation_gap > 0,
             "max_interpolation_gap must be positive")
    _require(math.isfinite(max_contact_age) and max_contact_age >= 0,
             "max_contact_age must be nonnegative")
    indices = _episode_indices(records, duplicate_policy)
    times = records.timestamps[indices].double()
    _require(float(times[0]) <= query_time <= float(times[-1]), "query outside episode")
    values = torch.zeros(4, 4, FUTURE_DIM)
    masks = torch.zeros_like(values, dtype=torch.bool)
    contacts = torch.zeros(4, 4, 2)
    contacts_valid = torch.zeros(4, 4, 2, dtype=torch.bool)
    offsets=FUTURE_OFFSETS if target_offsets is None else tuple(target_offsets)
    _require(len(offsets)==4 and all(math.isfinite(v) and v>0 for v in offsets),"four causal target offsets required")
    target_times = torch.tensor([query_time + h for h in offsets], dtype=torch.float64)

    for h, target in enumerate(target_times.tolist()):
        if target < float(times[0]) or target > float(times[-1]) + 1e-10:
            continue
        hi = int(torch.searchsorted(times, torch.tensor(target, dtype=times.dtype)))
        exact = [i for i in (hi,hi-1) if 0 <= i < len(times) and abs(float(times[i])-target) <= 1e-9]
        if exact:
            lo = hi = min(exact,key=lambda i: abs(float(times[i])-target))
        else:
            if hi == 0 or hi == len(times):
                continue
            lo = hi - 1
        a, b = int(indices[lo]), int(indices[hi])
        gap = float(times[hi] - times[lo])
        alpha = 0.0 if lo == hi else (target - float(times[lo])) / gap
        # Contact remains well defined even if continuous interpolation is rejected.
        age = target - float(times[lo])
        if age <= max_contact_age + 1e-9:
            cv = records.contact_valid[a] & torch.isfinite(records.contacts[a])
            if a != b:
                cv = cv & records.contact_valid[b] & torch.isfinite(records.contacts[b])
                cv = cv & (records.contacts[a] == records.contacts[b])
            track_ok = ((records.track_ids[a] >= 0) & (records.track_ids[a] == records.track_ids[b]) &
                        records.entity_valid[a] & records.entity_valid[b])
            if records.contacts.ndim == 2:
                # Only the task target has supervision. No duplicated fake labels.
                cv = cv & track_ok[0]
                contacts[0,h] = torch.where(cv, records.contacts[a].float(), 0)
                contacts_valid[0,h] = cv
            else:
                cv = cv & track_ok[:,None]
                contacts[:,h] = torch.where(cv, records.contacts[a].float(), 0)
                contacts_valid[:,h] = cv
        if gap > max_interpolation_gap + 1e-9:
            continue

        def blend(x: Tensor) -> Tensor:
            return x[a].float() * (1 - alpha) + x[b].float() * alpha

        xyz, hands = blend(records.entity_world_positions), blend(records.gripper_positions)
        visual_index = a if target - float(times[lo]) <= float(times[hi]) - target else b
        visual_error = abs(float(records.timestamps[visual_index]) - target)
        for entity in range(4):
            same = (int(records.track_ids[a, entity]) >= 0 and
                    int(records.track_ids[a, entity]) == int(records.track_ids[b, entity]))
            present = same and bool(records.entity_valid[a, entity] & records.entity_valid[b, entity])
            if not present:
                continue
            pos_ok = bool(torch.isfinite(records.entity_world_positions[[a,b], entity]).all())
            if pos_ok:
                values[entity, h, 128:131] = xyz[entity]
                masks[entity, h, 128:131] = True
            visual_ok = visual_error <= max_visual_time_error
            region_valid = (records.visibility if records.visual_context_valid is None
                            else records.visual_context_valid)
            visual_ok &= bool(region_valid[visual_index, entity] & records.projectability[visual_index, entity])
            visual_ok &= bool(torch.isfinite(records.frozen_visual_features[visual_index, entity]).all())
            if visual_ok:
                # This is an actual frame encoding, never a blend of image features.
                values[entity, h, :128] = records.frozen_visual_features[visual_index, entity]
                masks[entity, h, :128] = True
            if entity < 2:
                hand = records.relevant_gripper[entity]
                hand_ok = bool(records.gripper_valid[[a,b], hand].all())
                hand_ok &= bool(torch.isfinite(records.gripper_positions[[a,b], hand]).all())
                # Verify reference gripper identity too; no identity-switch interpolation.
                hand_ok &= (int(records.track_ids[a,hand+2]) >= 0 and
                            int(records.track_ids[a,hand+2]) == int(records.track_ids[b,hand+2]))
                if pos_ok and hand_ok:
                    values[entity, h, 131:134] = xyz[entity] - hands[hand]
                    masks[entity, h, 131:134] = True
    return FutureLabels(values.reshape(16,134), masks.reshape(16,134), contacts.reshape(16,2),
                        contacts_valid.reshape(16,2), target_times.repeat(4), query_time,
                        records.episode_id, records.scene_id, records.split,
                        records.visual_encoder_id)


def fit_train_statistics(samples: Sequence[FutureLabels], *, train_episode_ids: Sequence[str],
                         split_manifest_sha256: str) -> dict:
    """Fit per-field statistics over training labels only; mask constant/missing fields."""
    _require(bool(samples), "cannot fit empty statistics")
    _require(len(split_manifest_sha256) == 64, "split manifest SHA256 required")
    permitted = set(train_episode_ids)
    _require(bool(permitted), "training episode list is empty")
    encoder = samples[0].visual_encoder_id
    count = torch.zeros(FUTURE_DIM, dtype=torch.float64)
    total = torch.zeros_like(count)
    total2 = torch.zeros_like(count)
    used: set[str] = set()
    scenes: set[str] = set()
    for item in samples:
        _require(item.split == "train" and item.episode_id in permitted,
                 "statistics may only use declared training episodes")
        _require(item.visual_encoder_id == encoder, "mixed visual encoder identities")
        valid = item.future_loss_valid
        _require(bool(torch.isfinite(item.future[valid]).all()), "nonfinite valid label")
        x = torch.where(valid, item.future.double(), 0)
        count += valid.sum(0)
        total += x.sum(0)
        total2 += (x*x).sum(0)
        used.add(item.episode_id)
        scenes.add(item.scene_id)
    mean = total / count.clamp_min(1)
    variance = (total2 / count.clamp_min(1) - mean.square()).clamp_min(0)
    std = torch.where(variance > 1e-12, variance.sqrt(), torch.ones_like(variance))
    return {"schema_version": SCHEMA_VERSION, "fit_split": "train",
            "visual_encoder_id": encoder, "split_manifest_sha256": split_manifest_sha256,
            "train_episode_ids": sorted(permitted), "fitted_episode_ids": sorted(used),
            "fitted_scene_ids": sorted(scenes),
            "mean": mean.tolist(), "std": std.tolist(), "count": count.tolist()}


def normalize_future(future: Tensor, valid: Tensor, statistics: Mapping) -> Tensor:
    _require(statistics.get("fit_split") == "train", "untrusted normalization split")
    mean = torch.as_tensor(statistics["mean"], device=future.device, dtype=torch.float32)
    std = torch.as_tensor(statistics["std"], device=future.device, dtype=torch.float32)
    _require(mean.shape == std.shape == (134,), "normalization must have 134 fields")
    _require(bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0).all()),
             "invalid normalization values")
    # Normalize FP32 before any downstream BF16 conversion.
    return torch.where(valid, (future.float() - mean) / std, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DualRateShardDataset(Dataset):
    """JSON-indexed, weights_only tensor shards; one shard is one training row.

    Manifest keys: schema_version, split, visual_encoder_id, split_manifest_sha256,
    native_checkpoint_sha256, action_normalization_sha256, motion_normalization_sha256,
    motion_representation='raw_absolute_motion_v2', action_representation='checkpoint_mean_std',
    statistics (fit_train_statistics result), rows=[{path,sha256,episode_id,scene_id}].
    Each relative .pt path contains a flat string->Tensor dictionary. Required label
    keys match FutureLabels.tensors(). Other tensors have an explicit allowlist;
    the consumer must explicitly select conditions. Raw oracle fields are
    rejected. This is a strict storage contract, not an existing DOMINO decoder.
    """
    LABEL_KEYS = {"future", "future_loss_valid", "contact", "contact_valid", "future_times"}
    # SOURCE_FUTURE_LATENT and action fields are supervision, never pass the full
    # dictionary wholesale as policy conditions. Explicit model adapters select keys.
    EXTRA_KEYS = {
        "source_condition_latent", "source_initial_state", "source_motion",
        "source_motion_valid", "source_motion_acceleration_valid", "source_motion_times", "source_text", "reference_available",
        "source_action", "fast_action", "source_future_latent",
        "source_future_intervals", "source_action_times", "source_known_actions",
        "source_known_mask", "fast_known_actions", "fast_known_mask", "rgb", "rgb_times",
        "motion", "motion_times", "motion_valid", "motion_acceleration_valid", "state", "text", "text_valid",
        "source_time", "completed_time", "query_time", "action_times", "query_entity_valid",
        "source_temporal_support", "source_view_overlap", "source_token_valid", "source_input_type",
        "future_view_overlap", "future_token_valid", "future_input_type", "source_action_intervals",
        "rgb_pixel_valid", "state_time", "spatial_flow", "spatial_flow_valid", "spatial_flow_times",
        "executed_actions", "executed_action_intervals", "executed_action_valid",
        "source_action_valid", "fast_action_valid", "source_condition_frames", "source_future_frames",
        "source_rgb_actual_times", "future_rgb_actual_times", "future_rgb_time_errors",
        "future_visibility_known", "future_visual_actual_times",
        "source_cache_id",
        "action_intervals",
        "source_request_time", "source_available_time", "query_available_time",
        "source_prefix_end", "fast_prefix_end", "query_rgb_template_time", "query_rgb_sampling_error",
    }

    def __init__(self, manifest_path: str | Path, *, verify_hash: bool = True):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        _require(m.get("schema_version") == SCHEMA_VERSION, "unsupported label manifest")
        _require(m.get("split") in ("train", "validation", "test"), "unknown manifest split")
        for key in ("native_checkpoint_sha256", "action_normalization_sha256", "motion_normalization_sha256"):
            _require(isinstance(m.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", m[key]) is not None,
                     f"missing or invalid {key}")
        _require(m.get("motion_representation") == "raw_absolute_motion_v2",
                 "motion inputs must remain raw for native normalization")
        _require(m.get("action_representation") == "checkpoint_mean_std",
                 "actions must use checkpoint mean/std")
        stats = m.get("statistics", {})
        _require(stats.get("fit_split") == "train", "statistics must be train-only")
        _require(stats.get("visual_encoder_id") == m.get("visual_encoder_id") and
                 bool(m.get("visual_encoder_id")), "encoder identity mismatch")
        _require(stats.get("split_manifest_sha256") == m.get("split_manifest_sha256") and
                 len(m.get("split_manifest_sha256", "")) == 64, "split provenance mismatch")
        self.rows = m.get("rows", [])
        _require(isinstance(self.rows, list) and bool(self.rows), "empty label manifest")
        train_ids = set(stats.get("train_episode_ids", []))
        for row in self.rows:
            _require(bool(row.get("episode_id")) and bool(row.get("scene_id")), "row identity absent")
            if m["split"] == "train":
                _require(row["episode_id"] in train_ids, "undeclared training episode")
            else:
                _require(row["episode_id"] not in train_ids, "held-out episode leaks into statistics")
                _require(row["scene_id"] not in stats.get("fitted_scene_ids", []),
                         "held-out scene leaks into statistics")
        self.verify_hash = verify_hash

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        path = (self.path.parent / row["path"]).resolve()
        _require(path.is_relative_to(self.path.parent) and path.suffix == ".pt", "shard path escapes manifest")
        if self.verify_hash:
            _require(sha256_file(path) == row.get("sha256"), "shard hash mismatch")
        item = torch.load(path, map_location="cpu", weights_only=True)
        _require(isinstance(item, dict) and self.LABEL_KEYS <= item.keys(), "invalid shard keys")
        for key, tensor in item.items():
            _require(isinstance(key, str) and isinstance(tensor, Tensor), "shards contain tensors only")
            _require(key in self.LABEL_KEYS or key in self.EXTRA_KEYS,
                     f"unclassified shard field: {key}")
        shapes = {"future": (16,134), "future_loss_valid": (16,134), "contact": (16,2),
                  "contact_valid": (16,2), "future_times": (16,)}
        for name, shape in shapes.items():
            _require(tuple(item[name].shape) == shape, f"bad {name} shape")
        _require(item["future_loss_valid"].dtype == item["contact_valid"].dtype == torch.bool,
                 "label masks must be bool")
        _require(bool(torch.isfinite(item["future"][item["future_loss_valid"]]).all()), "nonfinite future")
        cv = item["contact_valid"]
        _require(bool(((item["contact"][cv] == 0) | (item["contact"][cv] == 1)).all()), "nonbinary contact")
        _require(bool(torch.isfinite(item["future_times"]).all()), "invalid future timestamps")
        if "query_time" in item:
            query = item["query_time"]
            _require(query.numel() == 1 and bool(torch.isfinite(query).all()), "invalid query time")
            expected = (query.double().reshape(()) + torch.tensor(FUTURE_OFFSETS, dtype=torch.float64)).repeat(4)
            _require(torch.allclose(item["future_times"].double(), expected, atol=1e-6, rtol=0),
                     "future labels are not aligned to query physical time")
            for name in ("rgb_times", "motion_times"):
                if name in item:
                    _require(bool(torch.isfinite(item[name]).all() & (item[name] <= query).all()),
                             f"{name} contains unavailable observations")
            if "source_time" in item or "completed_time" in item:
                _require("source_time" in item and "completed_time" in item,
                         "source and completion clocks must be paired")
                source, complete = item["source_time"], item["completed_time"]
                _require(source.numel() == complete.numel() == 1 and
                         bool(torch.isfinite(source).all() & torch.isfinite(complete).all() &
                              (source <= complete).all() & (complete <= query).all()),
                         "expected source <= completed <= query")
        item["future"] = normalize_future(item["future"], item["future_loss_valid"], self.manifest["statistics"])
        return item


def open_training_dataset(manifest_path: str | Path, **kwargs) -> Dataset:
    """Open either audited legacy tensor shards or fixed-time real DOMINO episodes."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("native_v9_dataset"):
        from .native_v9_data import NativeReplayDataset
        return NativeReplayDataset(manifest_path,**kwargs)
    if manifest.get("schema_version") == "dualrate-domino-lazy-v2":
        from .producer import DominoPhysicalDataset
        if manifest.get('native_v6'):
            from .native_v6_data import EventDataset
            return EventDataset(manifest_path,**kwargs)
        return DominoPhysicalDataset(manifest_path, **kwargs)
    return DualRateShardDataset(manifest_path, **kwargs)
