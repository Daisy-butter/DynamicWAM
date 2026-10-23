"""Synthetic CPU contract tests, not real DOMINO annotation acceptance."""
from dataclasses import replace
import json

import pytest
import torch

from dynamicwam.dualrate.data import (
    EpisodeRecords, FUTURE_OFFSETS, SCHEMA_VERSION, DualRateShardDataset,
    fit_train_statistics, normalize_future, produce_future_labels, sha256_file,
)


def episode(times=(0., .1, .2, .4, .8, 1.0)):
    t = torch.tensor(times, dtype=torch.float64)
    n = len(t)
    xyz = torch.zeros(n, 4, 3)
    xyz[:,:,0] = t[:,None]
    xyz[:,:,1] = torch.arange(4.)
    features = t.float()[:,None,None].expand(n,4,128).clone()
    return EpisodeRecords(
        episode_id="train-episode-1", scene_id="train-scene-1", split="train",
        visual_encoder_id="fixed-encoder:weights-sha256", entity_ids=("cup","basket","left","right"),
        timestamps=t, entity_world_positions=xyz,
        gripper_positions=xyz[:,2:].clone(), frozen_visual_features=features,
        entity_valid=torch.ones(n,4,dtype=torch.bool), gripper_valid=torch.ones(n,2,dtype=torch.bool),
        visibility=torch.ones(n,4,dtype=torch.bool), projectability=torch.ones(n,4,dtype=torch.bool),
        track_ids=torch.arange(4).repeat(n,1), contacts=torch.zeros(n,2),
        contact_valid=torch.ones(n,2,dtype=torch.bool), relevant_gripper=(0,1),
    )


def labels(r=None, **kwargs):
    return produce_future_labels(episode() if r is None else r, 0.0,
                                 max_interpolation_gap=1., max_contact_age=.3, **kwargs)


def stats(item):
    return fit_train_statistics([item], train_episode_ids=[item.episode_id],
                                 split_manifest_sha256="f"*64)


def write_dataset(tmp_path, extra=None, item=None):
    item = labels() if item is None else item
    tensor_dict = item.tensors()
    tensor_dict.update(extra or {})
    shard = tmp_path / "row.pt"
    torch.save(tensor_dict, shard)
    manifest = {"schema_version": SCHEMA_VERSION, "split": "train",
                "visual_encoder_id": item.visual_encoder_id,
                "native_checkpoint_sha256": "1"*64,
                "action_normalization_sha256": "2"*64,
                "motion_normalization_sha256": "3"*64,
                "motion_representation": "raw_absolute_motion_v2",
                "action_representation": "checkpoint_mean_std",
                "split_manifest_sha256": "f"*64, "statistics": stats(item),
                "rows": [{"path": "row.pt", "sha256": sha256_file(shard),
                          "episode_id": item.episode_id, "scene_id": item.scene_id}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_exact_timestamps_field_layout_and_nonredundant_gripper_mask():
    x = labels()
    assert x.future.shape == (16,134)
    assert x.contact.shape == (16,2)
    assert torch.allclose(x.future[:4,128], torch.tensor(FUTURE_OFFSETS))
    assert torch.equal(x.future[:4,132], torch.full((4,), -2.))
    assert x.future_loss_valid[:8].all()
    assert not x.future_loss_valid[8:,131:].any()
    assert x.future_loss_valid[8:,:131].all()
    assert x.contact_valid[:4].all()
    assert not x.contact_valid[4:].any()  # target-only contact is not hallucinated


def test_nonuniform_continuous_interpolation_and_contact_endpoint_agreement():
    r = episode((0., .15, .35, .9, 1.))
    r.contacts[1:,0] = 1
    x = labels(r)
    assert torch.allclose(x.future[:4,128], torch.tensor(FUTURE_OFFSETS), atol=1e-7)
    assert not x.contact_valid[0,0]  # change occurred somewhere in (0,.15), time unknown
    assert x.contact[1,0] == 1
    assert set(x.contact.unique().tolist()) == {0., 1.}
    assert not x.contact_valid[3].any()  # .8 - .35 exceeds contact age bound


def test_contact_exact_transition_is_valid_but_unknown_interval_or_endpoint_is_not():
    r = episode((0.,.15,.35,.9,1.))
    r.contacts[1:,0] = 1
    # .05 + .1 is microscopically greater than the stored .15 float64 value.
    # It must still select the exact physical observation, not the next interval.
    exact = produce_future_labels(r,.05,max_interpolation_gap=1.,max_contact_age=.3)
    assert exact.contact_valid[0,0] and exact.contact[0,0] == 1
    r.contact_valid[2,0] = False
    x = labels(r)
    assert not x.contact_valid[1,0]  # .2 needs both .15 and .35 endpoint evidence


def test_large_gap_masks_continuous_fields():
    r = episode((0., .3, 1.))
    x = produce_future_labels(r, 0., max_interpolation_gap=.2, max_contact_age=.1)
    assert not x.future_loss_valid.any()
    assert x.contact_valid[0].all()


def test_duplicate_default_fails_and_explicit_last_uses_whole_row():
    r = episode((0., .1, .1, .4, .8, 1.))
    r.frozen_visual_features[1].fill_(11)
    r.frozen_visual_features[2].fill_(22)
    with pytest.raises(ValueError, match="duplicate"):
        labels(r)
    x = labels(r, duplicate_policy="last")
    assert x.future[0,0] == 22


def test_unsorted_times_fail():
    with pytest.raises(ValueError, match="ordered"):
        labels(episode((0., .2, .1, 1.)))


def test_instance_switch_cannot_interpolate_visual_geometry_or_contact():
    r = episode((0., .15, .35, .9, 1.))
    r.track_ids[1:,0] = 8
    x = labels(r)
    assert not x.future_loss_valid[0].any()
    assert not x.contact_valid[0].any()
    assert x.future_loss_valid[1].all()


def test_visibility_only_masks_visual_not_world_position():
    r = episode()
    r.visibility[1,0] = False
    x = labels(r)
    assert not x.future_loss_valid[0,:128].any()
    assert x.future_loss_valid[0,128:].all()


def test_projection_and_nan_mask_do_not_emit_nan():
    r = episode()
    r.projectability[1,0] = False
    r.frozen_visual_features[1,0] = torch.nan
    x = labels(r)
    assert not x.future_loss_valid[0,:128].any()
    assert torch.isfinite(x.future).all()


def test_missing_entity_mask_and_episode_boundary():
    r = episode((0., .1, .2))
    r.entity_valid[:,1] = False
    r = replace(r, entity_ids=("cup",None,"left","right"))
    x = labels(r)
    assert not x.future_loss_valid[4:8].any()
    assert not x.future_loss_valid[2:4].any()
    assert not x.contact_valid[2:4].any()
    with pytest.raises(ValueError, match="outside episode"):
        produce_future_labels(r, -.1, max_interpolation_gap=1., max_contact_age=1.)


def test_gripper_coordinate_mismatch_rejected():
    r = episode()
    r.gripper_positions[0,0,0] = 9
    with pytest.raises(ValueError, match="inconsistent coordinates"):
        labels(r)


def test_per_entity_contacts_keep_instance_layout():
    r = episode()
    contacts = torch.zeros(len(r.timestamps),4,2)
    contacts[:,1,1] = 1
    r = replace(r, contacts=contacts, contact_valid=torch.ones_like(contacts,dtype=torch.bool))
    x = labels(r)
    assert x.contact_valid.all()
    assert torch.equal(x.contact[4:8,1], torch.ones(4))
    assert not x.contact[:4].any()


def test_train_only_statistics_and_encoder_identity():
    x = labels()
    result = stats(x)
    assert result["fit_split"] == "train"
    assert result["fitted_scene_ids"] == ["train-scene-1"]
    normalized = normalize_future(x.future, x.future_loss_valid, result)
    assert torch.isfinite(normalized).all()
    assert not normalized[~x.future_loss_valid].any()
    with pytest.raises(ValueError, match="training episodes"):
        stats(replace(x, split="validation"))
    with pytest.raises(ValueError, match="encoder"):
        fit_train_statistics([x,replace(x,visual_encoder_id="other")],
                             train_episode_ids=[x.episode_id], split_manifest_sha256="a"*64)


def test_weights_only_shard_roundtrip_and_hash(tmp_path):
    path, _ = write_dataset(tmp_path)
    dataset = DualRateShardDataset(path)
    assert len(dataset) == 1
    assert dataset[0]["future"].shape == (16,134)
    with (tmp_path / "row.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        dataset[0]


def test_oracle_condition_rejected(tmp_path):
    path, _ = write_dataset(tmp_path, {"condition_entity_world_positions": torch.zeros(4,3)})
    with pytest.raises(ValueError, match="unclassified"):
        DualRateShardDataset(path)[0]


def test_shard_escape_rejected(tmp_path):
    path, manifest = write_dataset(tmp_path)
    manifest["rows"][0]["path"] = "../outside.pt"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        DualRateShardDataset(path)[0]


@pytest.mark.parametrize("key", ["native_checkpoint_sha256", "action_normalization_sha256",
                                 "motion_normalization_sha256", "motion_representation"])
def test_checkpoint_normalization_identity_is_mandatory(tmp_path, key):
    path, manifest = write_dataset(tmp_path)
    del manifest[key]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        DualRateShardDataset(path)


def test_heldout_scene_cannot_enter_train_statistics(tmp_path):
    path, manifest = write_dataset(tmp_path)
    manifest["split"] = "validation"
    manifest["rows"][0]["episode_id"] = "another-style-same-scene"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="scene leaks"):
        DualRateShardDataset(path)


@pytest.mark.parametrize("extra,match", [
    ({"query_time": torch.tensor(0.), "rgb_times": torch.tensor([.01])}, "unavailable"),
    ({"query_time": torch.tensor(0.), "source_time": torch.tensor(0.),
      "completed_time": torch.tensor(.1)}, "source <= completed"),
    ({"query_time": torch.tensor(.1)}, "not aligned"),
])
def test_training_time_availability_is_checked(tmp_path, extra, match):
    path, _ = write_dataset(tmp_path, extra)
    with pytest.raises(ValueError, match=match):
        DualRateShardDataset(path)[0]
