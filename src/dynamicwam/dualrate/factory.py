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


def restore_native_time_precision(native):
    """Keep native FP32 time arithmetic compatible with the BF16 video loader.

    WanVideoModel prepares sinusoidal features in FP32 with autocast disabled;
    its two time MLPs therefore need FP32 weights too. The native inference
    loader casts all VideoDiT weights to BF16, so restore only these MLPs after
    strict loading. No state-dict key or loaded numerical value is changed.
    """
    video = native.compact_wan.video_model.wan_model
    for name in ("time_embedding", "time_projection"):
        module = getattr(video, name)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"native {name} must be a module")
        module.float()


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
    restore_native_time_precision(native)
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
            native_hidden_dim=2048),
        CurrentObservationEncoder(dim=arch.observation_dim, state_dim=native.config.state_dim,
            vision_weights_path=config["vision_weights"],vision_weights_sha256=config["vision_weights_sha256"],
            flow_mean=[native.config.motion_feature_mean[i] for i in (0,1,5,6)],
            flow_scale=[native.config.motion_feature_scale[i] for i in (0,1,5,6)]),
        FastJointWAM(arch), future_latent_shape=latent_shape,
        loss_config=LossConfig(**config.get("loss", {})),
        motion_mean=native.config.motion_feature_mean, motion_scale=native.config.motion_feature_scale,
        codec_id=config["codec_id"],future_field_support=config["future_field_support"],
        contact_field_support=config["contact_field_support"])
    # Preserve BF16 VideoDiT (FP32 time MLPs) + FP32 Action/motion/new modules. No global
    # model.bfloat16(): time arithmetic and normalization stay explicitly FP32/64.
    model = model.to(device)
    model.initialization_identity = {"native_checkpoint_sha256": actual, **identities}
    return model


def place_inference_devices(system, *, fast_device, slow_device):
    """Place an already strictly loaded eval model; training construction is unchanged.

    Slow source encoding, native prediction and readout stay on the slow device.
    The current observation encoder, A/Z sampler and its system-level constants
    stay on the fast device. Runtime owns completed-reference transfer/publication.
    """
    fast_device,slow_device=torch.device(fast_device),torch.device(slow_device)
    if system.training or any(p.requires_grad for p in system.parameters()):
        raise ValueError('inference placement requires an eval, frozen model')
    if fast_device.type=='cuda' and fast_device==slow_device:
        raise ValueError('hard-time deployment requires distinct fast and slow GPUs')
    system.slow.to(slow_device)
    system.world_readout.to(slow_device)
    system.observation.to(fast_device)
    system.fast.to(fast_device)
    for name,value in list(system.named_buffers(recurse=False)):
        setattr(system,name,value.to(fast_device))
    vae=system.slow.native.compact_wan.video_model.vae
    if isinstance(vae,torch.nn.Module):
        vae.to(slow_device)
    model=getattr(vae,'model',None)
    if isinstance(model,torch.nn.Module):
        model.to(slow_device)
    if hasattr(vae,'device'):
        vae.device=slow_device
    if hasattr(vae,'scale'):
        vae.scale=[x.to(slow_device) if isinstance(x,torch.Tensor) else x for x in vae.scale]
    for name,expected in (('slow',slow_device),('world_readout',slow_device),
                          ('observation',fast_device),('fast',fast_device)):
        module=getattr(system,name)
        if any(value.device!=expected for value in (*module.parameters(),*module.buffers())):
            raise RuntimeError(f'{name} contains tensors on the wrong inference device')
    system.inference_device_map=dict(fast=str(fast_device),slow=str(slow_device),
                                     readout=str(slow_device),vae=str(slow_device))
    return system.inference_device_map
