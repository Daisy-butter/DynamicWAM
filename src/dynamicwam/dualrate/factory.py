"""Strict full native initialization; never migrate Stage1 or smoke weights."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .data import sha256_file
from .fast import CurrentObservationEncoder, FastConfig, FastJointWAM
from .slow import SlowReferenceAdapter, StructuredWorldReadout
from .system import DualRateSystem, LossConfig


def metadata_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checkpoint_identity(payload):
    return {
        "action_normalization_sha256": metadata_sha256(payload["config"]["action_normalization"]),
        "motion_normalization_sha256": metadata_sha256(payload["config"]["model"]["absolute_motion"]),
    }


def build_system(config, device="cpu"):
    from dynamicwam.inference.model_loader import _build_model, _load_checkpoint_payload, load_deploy_config
    path = Path(config["init_checkpoint"]).resolve(strict=True)
    actual = sha256_file(path)
    if actual != config["init_checkpoint_sha256"]:
        raise ValueError("full native initialization checkpoint hash mismatch")
    payload = _load_checkpoint_payload(path)
    identities = checkpoint_identity(payload)
    for key, value in identities.items():
        if config.get(key) != value:
            raise ValueError(f"{key} differs from the actual native checkpoint")
    runtime = load_deploy_config(config["native_profile"])
    native = _build_model(runtime, payload, device=device)  # exact strict state load
    native.config = replace(native.config, wan_frozen=False)
    native.requires_grad_(True)
    # Wan's VAE wrapper is not necessarily an nn.Module, and is never optimized.
    vae = native.compact_wan.video_model.vae
    for value in (vae, getattr(vae, "model", None)):
        if isinstance(value, torch.nn.Module):
            value.requires_grad_(False).eval()
    arch = FastConfig(**config.get("architecture", {}))
    if (arch.action_dim, arch.chunk_size) != (native.config.action_dim, native.config.chunk_size):
        raise ValueError("fast action dimensions must match the full native checkpoint")
    latent_shape = tuple(config["future_latent_shape"])
    if len(latent_shape) != 4 or latent_shape[0] != 48:
        raise ValueError("native Wan2.2 latent layout must have 48 channels")
    fm = runtime["flow_matching"]
    slow = SlowReferenceAdapter(native, gradient_checkpointing=config.get("gradient_checkpointing", True),
        shift=fm["shift"], sigma_min=fm["sigma_min"], extra_one_step=fm["extra_one_step"])
    model = DualRateSystem(slow,
        StructuredWorldReadout(latent_channels=48, action_dim=arch.action_dim, world_dim=arch.world_dim,
            grid_size=config.get("readout_grid", 4), action_time_tolerance=config.get("action_time_tolerance", 0.02)),
        CurrentObservationEncoder(dim=arch.observation_dim, state_dim=native.config.state_dim),
        FastJointWAM(arch), future_latent_shape=latent_shape,
        loss_config=LossConfig(**config.get("loss", {})),
        motion_mean=native.config.motion_feature_mean, motion_scale=native.config.motion_feature_scale)
    # Preserve native BF16 VideoDiT + FP32 Action/motion and new modules. No global
    # model.bfloat16(): time arithmetic and normalization stay explicitly FP32/64.
    model = model.to(device)
    model.initialization_identity = {"native_checkpoint_sha256": actual, **identities}
    return model
