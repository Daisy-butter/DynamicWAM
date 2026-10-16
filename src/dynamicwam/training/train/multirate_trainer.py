"""Accelerate loop for cross-time world/action training.

One wrapped ``forward(mode="multirate_train")`` owns the graph. The warmup
phase freezes the video expert and builds the bridge under ``no_grad``. The
joint phase starts from a warmup export with a new optimizer.
"""

from __future__ import annotations

import json
import time
import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dynamicwam.absolute_motion import validate_checkpoint_motion_metadata
from dynamicwam.inference.multirate_model_loader import (
    MULTIRATE_CHECKPOINT_FORMAT,
    MULTIRATE_CHECKPOINT_VERSION,
    export_fast_only_payload,
    install_full_checkpoint,
    load_checkpoint_payload,
)
from dynamicwam.models.fast_conditioner import FastConditioner
from dynamicwam.models.compact_fast_policy import CompactFastPolicy, CompactFastConfig
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.training.data.multirate_sampling import (
    BalancedPairSampler,
    ProcessGroupAlignedSampler,
)
from dynamicwam.training.data.paired_index import deployment_age_weights
from dynamicwam.models.multirate_types import (
    BridgeAge,
    CurrentObservation,
    SourceNoise,
    SourceObservation,
    pad_text_embeddings,
    reject_leaky_source_fields,
)
from dynamicwam.models.multirate_wam import MultirateWAM, canonical_phase
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.training.data.paired_world_action_dataset import (
    PairedWorldActionDataset,
    paired_collate_fn,
    write_pair_manifest,
)
from dynamicwam.training.data.training_dataset import validate_dataset_identity
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel
from dynamicwam.training.train.common import (
    LogTracker,
    add_speed_metrics,
    build_training_flow_scheduler,
    get_dataloader_config,
    get_effective_global_batch_size,
    get_epoch_progress_metrics,
    get_export_dir,
    get_learning_rate_metrics,
    get_run_dir,
    init_experiment_trackers,
    logger,
    reduce_metrics_for_log,
)
from dynamicwam.training.train.multirate_objective import MultirateLossObjective
from dynamicwam.training.utils.scheduler import create_scheduler


_JOINT_NEW_STATE_PREFIXES = (
    "backbone.action_expert.source_time_encoder.",
    "fast_conditioner.visual_innovation.",
    "fast_conditioner.motion_innovation.",
    "fast_conditioner.plan_innovation.",
    "fast_conditioner.innovation_kind",
    "fast_conditioner.innovation_gate",
    "fast_conditioner.motion_forecast.",
    "fast_conditioner.proposal_encoder.",
    "fast_conditioner.proposal_gate",
    "fast_conditioner.completed_plan_encoder.",
    "fast_conditioner.completed_plan_gate",
)


def rtc_sampling_export_metadata(config, dataset):
    """Recorded replay is never relabelled as the offline H4 sampler."""
    if config["dataset"].get("kind") == "real_rollout":
        identity = getattr(dataset, "replay_identity", None)
        if not isinstance(identity, str) or len(identity) != 64:
            raise RuntimeError("real replay RTC export lacks validated replay identity")
        return dict(mode="recorded_real_prefix_v1", prefix_source="on_policy_executed",
            sampling="none", max_prefix_actions=4, zero_fast_prefix_coverage="not_asserted",
            bootstrap="recorded_prime_then_hold", replay_identity=identity)
    return getattr(dataset, "rtc_prefix_sampling_contract", None)


def migrate_joint_interface_initialization(model, payload):
    """One explicit V1 -> V2 initialization; never a permissive resume/load."""
    if payload.get("format") != MULTIRATE_CHECKPOINT_FORMAT:
        raise ValueError("joint interface migration requires a full multirate checkpoint")
    previous = payload.get("config", {}).get("model", {}).get("fast_conditioner", {})
    if int(previous.get("interface_version", 1)) != 1 or not previous.get("motion_kv"):
        raise ValueError("joint interface migration requires MotionKV V1 weights")
    if getattr(model.fast_conditioner, "interface_version", 1) not in (2, 3):
        raise ValueError("joint interface migration target must be V2 or V3")
    current, incoming = model.state_dict(), payload["model"]
    missing, extra = set(current) - set(incoming), set(incoming) - set(current)
    if extra or not missing or any(not key.startswith(_JOINT_NEW_STATE_PREFIXES) for key in missing):
        raise RuntimeError(f"unsafe joint initialization: missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}")
    for key, value in incoming.items():
        if value.shape != current[key].shape:
            raise RuntimeError(f"joint initialization shape mismatch: {key}")
    return dict(payload, model={**current, **incoming}), sorted(missing)


def configure_multirate_trainable(model: MultirateWAM, phase: str) -> str:
    phase = canonical_phase(phase)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if getattr(model, "native_multirate", False):
        teacher_free_v3 = (getattr(model.fast_conditioner,"interface_version",1) >= 3
            and model.native_source_bridge == "student_joint" and model.native_teacher_weight == 0)
        if not model.external_native_teacher and not teacher_free_v3:
            raise ValueError("native action adaptation requires an independent fixed teacher")
        for module in (model.backbone.action_expert, model.backbone.absolute_motion_tokens,
                       model.fast_conditioner):
            module.requires_grad_(True)
        model.native_training_phase = phase
        if phase == "joint":
            if model.native_source_bridge != "student_joint":
                raise ValueError("joint native training requires the differentiable student bridge")
            model.backbone.compact_wan.video_model.wan_model.requires_grad_(True)
        return phase
    if getattr(model, "feedback_phase", ""):
        if model.feedback_phase == 'dynamics':
            model.fast_policy.feedback_dynamics.requires_grad_(True)
        else:
            # Freeze observation/bridge representation and dynamics while the
            # endpoint actor learns. Its latent target cannot drift into a code.
            for name,p in model.fast_policy.named_parameters():
                if name.startswith(('action_encoder.','action_position','time_encoder.','blocks.','out_norm.','out.','feedback_projection.','feedback_position')):
                    p.requires_grad_(True)
        return phase
    if getattr(model, 'native_teacher_weight', 0.0) > 0:
        if phase == 'warmup':
            for parameter in model.fast_policy.parameters():
                parameter.requires_grad_(True)
            return phase
        if not model.external_native_teacher:
            raise ValueError('joint distillation requires a separate fixed teacher')
    for module in (
        model.backbone.action_expert,
        model.backbone.absolute_motion_tokens,
        getattr(model, "fast_policy", None) or model.fast_conditioner,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if phase == "joint":
        video = model.backbone.compact_wan.video_model.wan_model
        for parameter in video.parameters():
            parameter.requires_grad_(True)
    return phase


def multirate_optimizer_groups(
    model: MultirateWAM,
    config: dict[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    phase = canonical_phase(phase)

    def trainable(module: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    action_params = trainable(model.backbone.action_expert) + trainable(
        model.backbone.absolute_motion_tokens
    )
    fast_params = trainable(getattr(model, "fast_policy", None) or model.fast_conditioner)
    preserve = bool(getattr(model, "feedback_phase", "")) or getattr(model, 'native_teacher_weight', 0.0) > 0
    if (not action_params and not preserve) or not fast_params:
        raise RuntimeError("multirate training expected action and fast parameters")
    weight_decay = float(config["weight_decay"])
    groups = [
        {
            "name": "action_core",
            "params": action_params,
            "lr": float(config["action_learning_rate"]),
            "weight_decay": weight_decay,
        },
        {
            "name": "fast_conditioner",
            "params": fast_params,
            "lr": float(config["fast_learning_rate"]),
            "weight_decay": weight_decay,
        },
    ]
    groups = [group for group in groups if group['params']]
    if phase == "joint":
        video_params = trainable(model.backbone.compact_wan.video_model.wan_model)
        if not video_params:
            raise RuntimeError("joint phase expected trainable video expert parameters")
        groups.append(
            {
                "name": "video_expert",
                "params": video_params,
                "lr": float(config["video_learning_rate"]),
                "weight_decay": weight_decay,
            }
        )
    _assert_optimizer_groups(model, groups)
    return groups


def _assert_optimizer_groups(model: torch.nn.Module, groups: list[dict[str, Any]]) -> None:
    seen: list[torch.nn.Parameter] = []
    for group in groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                raise RuntimeError("a frozen parameter is in the optimizer")
            seen.append(parameter)
    seen_ids = [id(parameter) for parameter in seen]
    if len(seen_ids) != len(set(seen_ids)):
        raise RuntimeError("a trainable parameter is in more than one optimizer group")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if set(seen_ids) != trainable_ids:
        raise RuntimeError("optimizer groups do not match requires_grad parameters")


def _flow_pair(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(clean.shape[0])
    index = torch.randint(
        0,
        int(timesteps.shape[0]),
        (batch,),
        device=clean.device,
        generator=generator,
    )
    sigma = sigmas[index].to(dtype=clean.dtype).view(batch, *([1] * (clean.ndim - 1)))
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noisy = clean * (1.0 - sigma) + noise * sigma
    target = noise - clean
    return noisy, timesteps[index].to(dtype=clean.dtype), target, sigma


def _batch_identity(value, *, device, dtype: torch.dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return value


def prepare_multirate_batch(
    sample: dict[str, Any],
    *,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    phase: str,
    null_bridge_probability: float,
    sim_scale: float,
    action_scale: float,
    null_bridge: bool | None = None,
    generator: torch.Generator | None = None,
    native_teacher: bool = False,
    native_multirate: bool = False,
    joint_interface_version: int = 2,
) -> dict[str, Any]:
    """Build one training step. The bridge noise is fresh, not a GT mixture."""

    phase = canonical_phase(phase)
    source_state = sample["source_state"].to(device=device, dtype=dtype)
    source_motion = sample["source_motion"].to(device=device, dtype=torch.float32)
    text = [
        embedding.to(device=device, dtype=dtype) for embedding in sample["text_embeddings"]
    ]
    source = SourceObservation(
        condition_latent=sample["source_condition_latent"].to(device=device, dtype=dtype),
        state=source_state,
        motion=source_motion,
        motion_interval_valid_mask=sample["source_motion_interval_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        motion_acceleration_valid_mask=sample["source_motion_acceleration_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        text_embeddings=text,
        source_id=_batch_identity(sample.get("source_id", 0), device=device, dtype=torch.long),
        sim_time=_batch_identity(sample.get("source_sim_time", 0.0), device=device,
            dtype=torch.float64 if joint_interface_version >= 3 else torch.float32),
    )
    if "source_rgb" in sample:
        source = replace(source,
            rgb=sample["source_rgb"].to(device=device, dtype=torch.float32),
            rgb_valid_mask=sample["source_rgb_valid_mask"].to(device=device, dtype=torch.bool))
    if joint_interface_version >= 3:
        if "joint_targets" not in sample or "async_timing" not in sample or int(sample.get("physical_grid_version",0)) != 1:
            raise ValueError("V3 requires physically aligned cached video/action samples")
        source_offsets = sample["joint_targets"]["source_action_target_times"].to(device=device, dtype=torch.float64) - sample["source_sim_time"].to(device=device, dtype=torch.float64).reshape(-1,1)
        source = replace(source, action_target_offsets=source_offsets)
        if "source_rtc_prefix_actions" not in sample or "source_rtc_prefix_lengths" not in sample:
            raise ValueError("V3 requires source-query committed prefix metadata, including explicit zero bootstrap")
        from dynamicwam.models.rtc_prefix import PrefixCondition, validate_prefix_condition, apply_prefix
        source_prefix = PrefixCondition(sample["source_rtc_prefix_actions"].to(device=device,dtype=dtype),
            sample["source_rtc_prefix_lengths"].to(device=device,dtype=torch.long))
        validate_prefix_condition(source_prefix,max_prefix=4,
            reference=sample["fast_action_sequence"].to(device=device,dtype=dtype))
        source = replace(source,committed_prefix=source_prefix.actions,committed_prefix_lengths=source_prefix.lengths)

    reject_leaky_source_fields(source.as_model_batch())
    noise_shape = tuple(sample["bridge_noise_shape_latent"].shape)
    action_shape = tuple(sample["fast_action_sequence"].shape)
    source_noise = SourceNoise(
        future_latent=torch.randn(
            noise_shape, device=device, dtype=dtype, generator=generator
        ),
        actions=torch.randn(action_shape, device=device, dtype=dtype, generator=generator),
    )
    fast_actions = sample["fast_action_sequence"].to(device=device, dtype=dtype)
    fast_noisy, fast_t, fast_target, fast_sigma = _flow_pair(
        fast_actions, timesteps, sigmas, generator=generator
    )
    future = sample["fast_future_latent"].to(device=device, dtype=dtype)
    future_noisy, video_t, video_target, _ = _flow_pair(
        future, timesteps, sigmas, generator=generator
    )
    fast_state = sample["fast_state"].to(device=device, dtype=dtype)
    state_delta = sample["state_delta"].to(device=device, dtype=dtype)
    batch_size = int(source_state.shape[0])
    if null_bridge is None:
        draw = torch.rand((batch_size,), device=device, generator=generator)
        null_mask = draw < float(null_bridge_probability)
    elif torch.is_tensor(null_bridge):
        null_mask = null_bridge.to(device=device, dtype=torch.bool).reshape(-1)
        if int(null_mask.numel()) != batch_size:
            raise ValueError("null_bridge mask does not match the batch")
    else:
        null_mask = torch.full(
            (batch_size,),
            bool(null_bridge),
            dtype=torch.bool,
            device=device,
        )
    age = BridgeAge(
        delta_sim_seconds=sample["delta_sim_seconds"].to(device=device,
            dtype=torch.float64 if joint_interface_version >= 3 else dtype),
        delta_actions=sample["delta_actions"].to(device=device, dtype=dtype),
        bridge_valid=~null_mask,
        state_delta=state_delta,
    )
    current = CurrentObservation(
        rgb=sample["fast_rgb"].to(device=device, dtype=torch.float32),
        state=fast_state,
        motion=sample["fast_motion"].to(device=device, dtype=torch.float32),
        motion_interval_valid_mask=sample["fast_motion_interval_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        motion_acceleration_valid_mask=sample["fast_motion_acceleration_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        text_embeddings=text,
        age=age,
        rgb_valid_mask=sample["fast_rgb_valid_mask"].to(device=device, dtype=torch.bool) if "fast_rgb_valid_mask" in sample else None,
    )
    fast_text, fast_text_mask = pad_text_embeddings(text)
    action_noisy, action_t, action_target, _ = _flow_pair(
        fast_actions, timesteps, sigmas, generator=generator
    )
    aligned = {
        "condition_latent": sample["fast_condition_latent"].to(device=device, dtype=dtype),
        "future_latent": future_noisy,
        "initial_state": fast_state,
        "noisy_actions": action_noisy,
        "action_t": action_t,
        "video_t": video_t,
        "absolute_motion_features": current.motion,
        "absolute_motion_interval_valid_mask": current.motion_interval_valid_mask,
        "absolute_motion_acceleration_valid_mask": current.motion_acceleration_valid_mask,
        "text_embeddings": text,
    }
    prepared = {
        "clean_actions": fast_actions,
        "clean_future": future,
        "source": source,
        "source_noise": source_noise,
        "current": current,
        "fast_text": fast_text,
        "fast_text_mask": fast_text_mask,
        "fast_noisy_actions": fast_noisy,
        "fast_action_t": fast_t,
        "fast_target": fast_target,
        "aligned": aligned,
        "action_target": action_target,
        "null_bridge": null_mask,
        "sim_scale": float(sim_scale),
        "action_scale": float(action_scale),
    }
    if "joint_targets" in sample:
        # Timing is observable queue metadata, not future visual/action content.
        prepared["action_target_times"] = sample["joint_targets"]["fast_action_target_times"].to(device=device, dtype=torch.float64)
        prepared["observation_sim_time"] = sample["fast_sim_time"].to(device=device, dtype=torch.float64)
        offsets = prepared["action_target_times"] - prepared["observation_sim_time"].reshape(-1, 1)
        if not bool(torch.isfinite(offsets).all()) or bool((offsets <= 0).any()) or bool((offsets[:, 1:] <= offsets[:, :-1]).any()):
            raise ValueError("joint action target times must be finite, future and increasing")
        prepared["current"] = replace(prepared["current"], action_target_offsets=offsets)
    if joint_interface_version >= 3:
        source_labels = sample["joint_targets"]
        source_future = source_labels["source_future_latent"].to(device=device, dtype=dtype)
        source_actions = source_labels["source_action_sequence"].to(device=device, dtype=dtype)
        source_future_noisy, source_video_t, source_video_target, _ = _flow_pair(source_future,timesteps,sigmas,generator=generator)
        source_action_noisy, source_action_t, source_action_target, _ = _flow_pair(source_actions,timesteps,sigmas,generator=generator)
        prepared["source_aligned"] = dict(source.as_model_batch(), future_latent=source_future_noisy,
            noisy_actions=apply_prefix(source_action_noisy,source_prefix), action_t=source_action_t, video_t=source_video_t)
        prepared["source_video_target"] = source_video_target
        prepared["source_action_target"] = source_action_target
        prepared["source_rtc_prefix"] = source_prefix
        prepared["async_timing"] = {key: value.to(device=device) if torch.is_tensor(value) else value
            for key, value in sample["async_timing"].items()}
        prepared["rtc_sample_prefix_actions"] = sample["rtc_prefix_actions"].to(device=device,dtype=dtype)
        if "previous_plan" in sample or "previous_plan_valid_mask" in sample:
            if "previous_plan" not in sample or "previous_plan_valid_mask" not in sample:
                raise ValueError("real previous proposal requires both normalized actions and validity")
            previous = sample["previous_plan"].to(device=device,dtype=dtype)
            previous_valid = sample["previous_plan_valid_mask"].to(device=device,dtype=torch.bool)
            if previous.shape != fast_actions.shape or previous_valid.shape != fast_actions.shape[:2]:
                raise ValueError("real previous proposal must share the current physical endpoint grid")
            if not bool(torch.isfinite(torch.where(previous_valid[...,None],previous,torch.zeros_like(previous))).all()):
                raise ValueError("real previous proposal contains nonfinite valid actions")
            prepared["current"] = replace(prepared["current"], previous_plan=previous,
                previous_plan_valid=previous_valid)
        if "replay_identity" in sample:
            prepared["replay_identity"] = sample["replay_identity"]
    if native_teacher:
        prepared['fast_sigma'] = fast_sigma
        # Independent of the teacher's initial sampling noise; otherwise the
        # student would not see a valid flow-matching corruption distribution.
        prepared['teacher_flow_noise'] = torch.randn(action_shape, device=device, dtype=dtype, generator=generator)
    if native_multirate:
        # Deployment draws the fast trajectory independently from the older
        # source bridge. Share this fresh seed only with the current teacher.
        prepared['native_fast_noise'] = torch.randn(action_shape, device=device, dtype=dtype, generator=generator)
    if phase == "joint":
        prepared["video_target"] = video_target
    return prepared


def fast_module_spec(
    config: dict[str, Any],
    *,
    history_count: int,
    motion_mean: tuple[float, ...],
    motion_scale: tuple[float, ...],
) -> dict[str, Any]:
    action = config["model"]["action_expert"]
    wan = config["model"]["compact_wan"]
    fast = config["model"]["fast_conditioner"]
    return {
        "action_expert": {
            "dim": int(action["dim"]),
            "ffn_dim": int(action["ffn_dim"]),
            "num_layers": int(action["num_layers"]),
            "state_dim": int(action["state_dim"]),
            "action_dim": int(action["action_dim"]),
            "chunk_size": int(action["chunk_size"]),
            "source_physical_time_conditioning": bool(action.get("source_physical_time_conditioning", False)),
        },
        "wan_attention": {
            "dim": int(wan["dim"]),
            "num_heads": int(wan["num_heads"]),
            "head_dim": int(wan["head_dim"]),
        },
        "motion": {
            "history_count": int(history_count),
            "mean": [float(value) for value in motion_mean],
            "scale": [float(value) for value in motion_scale],
        },
        "fast_conditioner": {
            **fast,
            "dim": int(fast["dim"]),
            "layers": int(fast["layers"]),
            "heads": int(fast["heads"]),
            "head_dim": int(fast["head_dim"]),
            "text_dim": int(fast["text_dim"]),
            "state_dim": int(fast["state_dim"]),
            "sim_scale": float(fast["sim_scale"]),
            "action_scale": float(fast["action_scale"]),
        },
    }


def _pair_age_weights(dataset_cfg: dict[str, Any]) -> dict[int, float] | None:
    mode = str(dataset_cfg.get("age_weighting", "schedule_mixture"))
    if mode == "uniform":
        return None
    if mode != "schedule_mixture":
        raise ValueError(f"unsupported age weighting: {mode}")
    return deployment_age_weights(
        execute_horizon=int(dataset_cfg["execute_horizon"]),
        every_fast_queries=int(dataset_cfg["every_fast_queries"]),
        global_downsample_rate=int(dataset_cfg["global_downsample_rate"]),
    )


class MultirateTrainer:
    def __init__(self, config: dict[str, Any], accelerator) -> None:
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        self.phase = canonical_phase(str(config["phase"]))
        loss_cfg = config["loss"]
        if self.phase == "warmup" and float(loss_cfg["video_weight"]) != 0.0:
            raise ValueError("fast_warmup does not train the video regularizer")
        self.objective = MultirateLossObjective(
            action_weight=float(loss_cfg["action_weight"]),
            video_weight=float(loss_cfg["video_weight"]),
            fast_weight=float(loss_cfg["fast_weight"]),
            native_teacher_weight=float(loss_cfg.get('native_teacher_weight', 0.0)),
        )
        dataset_cfg = config["dataset"]
        if dataset_cfg.get("smoke_subset", False):
            if (not 0 < int(config.get("max_steps", 0)) <= 4 or config.get("resume_from")
                    or self.phase not in {"warmup", "joint"}
                    or dataset_cfg.get("kind", "physical_grid") == "real_rollout"):
                raise ValueError("physical subset is only for fresh <=4-step warmup/joint smoke")
        dataset_class, collate_function = PairedWorldActionDataset, paired_collate_fn
        physical_kwargs = {}
        if int(config["model"]["fast_conditioner"].get("interface_version",1)) >= 3:
            from dynamicwam.training.data.physical_grid_dataset import PhysicalGridPairedWorldActionDataset, physical_grid_collate_fn
            dataset_class, collate_function = PhysicalGridPairedWorldActionDataset, physical_grid_collate_fn
            physical_kwargs = dict(physical_grid_root=dataset_cfg["physical_grid_root"],
                future_latent_root=dataset_cfg["future_latent_root"],
                max_cache_age_actions=31, async_seed=int(dataset_cfg.get("async_seed",20261015)),
                slow_budget_seconds=2.56, rtc_prefix_max=int(config["rtc_prefix_max"]),
                strict_future_latents=True, smoke_subset=bool(dataset_cfg.get("smoke_subset", False)))
            if dataset_cfg.get("kind", "physical_grid") == "physical_grid":
                physical_kwargs.update(rtc_prefix_sampling=dataset_cfg.get("rtc_prefix_sampling", "delay_minimum_v1"),
                    source_prefix_condition_dropout_probability=float(dataset_cfg.get("source_prefix_condition_dropout_probability", .2)))
        if dataset_cfg.get("kind", "physical_grid") == "real_rollout":
            if int(config["model"]["fast_conditioner"].get("interface_version",1)) != 3:
                raise ValueError("real rollout refinement requires the complete V3 interface")
            from dynamicwam.training.data.rollout_replay import RealRolloutReplayDataset
            def dataset_class(packed_root, source_root, **kwargs):
                return RealRolloutReplayDataset(dataset_cfg["replay_root"], packed_root=packed_root,
                    source_root=source_root, strict_provenance=True, **kwargs)
        self.dataset = dataset_class(
            dataset_cfg["root"],
            dataset_cfg["source_root"],
            **physical_kwargs,
            splits=dataset_cfg["splits"],
            execute_horizon=int(dataset_cfg["execute_horizon"]),
            every_fast_queries=int(dataset_cfg["every_fast_queries"]),
            global_downsample_rate=int(dataset_cfg["global_downsample_rate"]),
            chunk_size=int(dataset_cfg["chunk_size"]),
            seconds_per_frame=float(dataset_cfg["seconds_per_frame"]),
            stress_offsets=dataset_cfg["stress_offsets"],
            max_open_shards=int(dataset_cfg["max_open_shards"]),
            partition_seed=int(dataset_cfg["partition_seed"]),
            validation_fraction=float(dataset_cfg["validation_fraction"]),
            clock_mode=dataset_cfg["clock_mode"],
            cache_age_actions=dataset_cfg.get("cache_age_actions"),
            joint_supervision=bool(dataset_cfg.get("joint_supervision", False)),
            cache_age_jitter=bool(dataset_cfg.get("cache_age_jitter", False)),
            cache_age_seed=int(dataset_cfg.get("cache_age_seed", 0)),
        )
        self.dataset_identity = validate_dataset_identity(
            self.dataset.packed.dataset_identity
        )
        self.motion_checkpoint_metadata = dict(
            self.dataset.packed.motion_checkpoint_metadata
        )
        self.motion_statistics = dict(self.dataset.packed.motion_statistics)
        loader_kwargs = get_dataloader_config(config)
        # Accelerate owns batching/sharding. Every rank must supply the SAME
        # global sequence; pre-sharding here would shard it a second time.
        # Drop incomplete process rounds so even_batches=False never desyncs
        # ranks and Accelerate does not recycle the first batches of the epoch.
        batch_size = max(1, int(config["per_device_batch_size"]))
        age_weights = _pair_age_weights(dataset_cfg)
        self.pair_sampler = ProcessGroupAlignedSampler(
            BalancedPairSampler(
                self.dataset.records,
                samples_per_episode=int(dataset_cfg["samples_per_episode"]),
                stress_probability=float(dataset_cfg["stress_probability"]),
                seed=int(dataset_cfg.get("sampler_seed", dataset_cfg["partition_seed"])),
                rank=0,
                world_size=1,
                age_weights=age_weights,
            ),
            batch_size=batch_size,
            num_processes=max(1, int(self.accelerator.num_processes)),
        )
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.pair_sampler,
            collate_fn=collate_function,
            generator=torch.Generator().manual_seed(int(dataset_cfg["partition_seed"])),
            **loader_kwargs,
        )
        self.val_dataset = dataset_class(
            dataset_cfg["root"],
            dataset_cfg["source_root"],
            **physical_kwargs,
            splits=dataset_cfg["splits"],
            execute_horizon=int(dataset_cfg["execute_horizon"]),
            every_fast_queries=int(dataset_cfg["every_fast_queries"]),
            global_downsample_rate=int(dataset_cfg["global_downsample_rate"]),
            chunk_size=int(dataset_cfg["chunk_size"]),
            seconds_per_frame=float(dataset_cfg["seconds_per_frame"]),
            stress_offsets=dataset_cfg["stress_offsets"],
            max_open_shards=int(dataset_cfg["max_open_shards"]),
            subset="validation",
            partition_seed=int(dataset_cfg["partition_seed"]),
            validation_fraction=float(dataset_cfg["validation_fraction"]),
            clock_mode=dataset_cfg["clock_mode"],
            cache_age_actions=dataset_cfg.get("cache_age_actions"),
            joint_supervision=bool(dataset_cfg.get("joint_supervision", False)),
            cache_age_jitter=bool(dataset_cfg.get("cache_age_jitter", False)),
            cache_age_seed=int(dataset_cfg.get("cache_age_seed", 0)),
            allow_empty=True,
            packed=self.dataset.packed,
            episode_partition=self.dataset.episode_partition,
        )
        self.val_loader = None
        if self.val_dataset.records:
            self.val_sampler = ProcessGroupAlignedSampler(
                BalancedPairSampler(
                    self.val_dataset.records,
                    samples_per_episode=int(dataset_cfg["samples_per_episode"]),
                    stress_probability=0.0,
                    seed=int(dataset_cfg["partition_seed"]),
                    rank=0,
                    world_size=1,
                    age_weights=age_weights,
                ),
                batch_size=batch_size,
                num_processes=max(1, int(self.accelerator.num_processes)),
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                sampler=self.val_sampler,
                collate_fn=collate_function,
                generator=torch.Generator().manual_seed(int(dataset_cfg["partition_seed"]) + 1),
                **loader_kwargs,
            )
        # Official transfer preserves the teacher's normalization buffers.
        # The training dataset identity remains the self-collected dataset.
        self.initialization_provenance = None
        if config.get("allow_pretrained_dataset_transfer", False):
            self._apply_pretrained_dataset_transfer(config["model"]["initial_checkpoint"])
        self.model = self._build_model()
        self.model.feedback_phase = config.get("feedback_phase", "")
        self.model.feedback_expert_weight = float(loss_cfg["fast_weight"])
        self.model.native_phase = config.get("native_phase", "")
        self.model.native_expert_weight = float(config.get("native_expert_weight", 1.0))
        self.model.native_fm_anchor_weight = float(config.get("native_fm_anchor_weight", 0.0))
        self.model.rtc_prefix_max = int(config.get("rtc_prefix_max", 0))
        self.model.native_source_bridge = str(config.get("native_source_bridge", "immutable_official"))
        self.model.gradient_checkpointing = bool(config.get("gradient_checkpointing", False))
        self.model.joint_video_weight = float(loss_cfg["video_weight"])
        self.model.joint_action_weight = float(loss_cfg["action_weight"])
        self.model.motion_prediction_weight = float(config.get("motion_prediction_weight", 0.0))
        self._load_initialization()
        self._build_native_teacher()
        self.model.max_bridge_age_seconds = config.get("max_bridge_age_seconds")
        configure_multirate_trainable(self.model, self.phase)
        groups = multirate_optimizer_groups(self.model, config, self.phase)
        self.optimizer = AdamW(groups, lr=float(config["action_learning_rate"]))
        scheduler = build_training_flow_scheduler(config)
        weight_dtype = next(self.model.backbone.action_expert.parameters()).dtype
        self._timesteps = scheduler.timesteps.to(device=self.device, dtype=weight_dtype)
        self._sigmas = scheduler.sigmas.to(device=self.device, dtype=weight_dtype)
        self.global_step = 0
        self.data_epoch = 0
        self.batches_in_epoch = 0
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
        )

        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare_data_loader(self.val_loader)
        # ClampedCosineScheduler is not a torch LRScheduler. Accelerate leaves
        # it unwrapped/unregistered; own its optimizer-step clock explicitly.
        self.scheduler = create_scheduler(
            self.optimizer,
            total_steps=int(config["max_steps"]),
            warmup_steps=int(config["warmup_steps"]),
            min_lr_ratio=float(config["min_lr_ratio"]),
        )
        self.accelerator.register_for_checkpointing(self.scheduler)

    def _apply_pretrained_dataset_transfer(self, checkpoint_path: str) -> None:
        """Keep official motion scales when initializing onto self-collected pairs."""

        initial = load_checkpoint_payload(checkpoint_path)
        motion = validate_checkpoint_motion_metadata(
            initial["config"]["model"]["absolute_motion"]
        )
        self.motion_checkpoint_metadata = dict(motion)
        self.motion_statistics = dict(
            self.motion_statistics,
            mean=self.motion_checkpoint_metadata["feature_mean"],
            scale=self.motion_checkpoint_metadata["feature_scale"],
        )
        parent = initial["config"].get("initialization_provenance")
        if initial.get("format") == "dynamicwam_absolute_motion_checkpoint":
            self._check_transfer_compatibility(initial)
            self.initialization_provenance = {
                "path": checkpoint_path,
                "source_dataset_identity": initial["config"]["dataset_identity"],
                "target_dataset_identity": self.dataset_identity,
                "motion_statistics_sha256": motion["statistics_sha256"],
            }
        else:
            if not isinstance(parent, dict):
                raise RuntimeError(
                    "pretrained transfer continuation requires the parent "
                    "multirate export to record initialization_provenance"
                )
            self.initialization_provenance = {
                "path": checkpoint_path,
                "parent_provenance": parent,
                "source_dataset_identity": parent.get("source_dataset_identity"),
                "target_dataset_identity": self.dataset_identity,
                "motion_statistics_sha256": motion["statistics_sha256"],
            }

    def _build_model(self) -> MultirateWAM:
        model_cfg = self.config["model"]
        wan_cfg = model_cfg["compact_wan"]
        future = wan_cfg["future_video_size"]
        compact = CompactWANModel.from_config(
            CompactWANConfig(
                checkpoint_path=wan_cfg["checkpoint_path"],
                config_path=wan_cfg["config_path"],
                precision=wan_cfg["precision"],
                dim=int(wan_cfg["dim"]),
                ffn_dim=int(wan_cfg["ffn_dim"]),
                num_heads=int(wan_cfg["num_heads"]),
                num_layers=int(wan_cfg["num_layers"]),
                head_dim=int(wan_cfg["head_dim"]),
                future_video_size=(int(future[0]), int(future[1])),
                hidden_anchor_layers=list(wan_cfg["hidden_anchor_layers"]),
                motion_anchor_layers=list(wan_cfg["motion_anchor_layers"]),
                teacher_layer_mapping=list(wan_cfg["teacher_layer_mapping"]),
            ),
            device=str(self.device),
        )
        action_cfg = model_cfg["action_expert"]
        backbone = SmallWAMActionModel(
            config=SmallWAMActionConfig(
                compact_wan=compact.config,
                action_dim=int(action_cfg["action_dim"]),
                state_dim=int(action_cfg["state_dim"]),
                chunk_size=int(action_cfg["chunk_size"]),
                ae_dim=int(action_cfg["dim"]),
                ae_ffn_dim=int(action_cfg["ffn_dim"]),
                ae_num_layers=int(action_cfg["num_layers"]),
                source_physical_time_conditioning=bool(action_cfg.get("source_physical_time_conditioning", False)),
                wan_frozen=True,
                motion_history_count=int(self.dataset.packed.motion_history_count),
                motion_feature_mean=tuple(
                    float(value) for value in self.motion_statistics["mean"]
                ),
                motion_feature_scale=tuple(
                    float(value) for value in self.motion_statistics["scale"]
                ),
            ),
            compact_wan=compact,
        )
        schedule = BridgeSchedule.from_training_config(self.config)
        if model_cfg.get("compact_fast"):
            compact_policy = CompactFastPolicy(CompactFastConfig(**model_cfg["compact_fast"]),
                motion_history_count=int(self.dataset.packed.motion_history_count),
                motion_mean=self.motion_statistics["mean"], motion_scale=self.motion_statistics["scale"], schedule=schedule)
            if model_cfg.get("encoder_weights"):
                compact_policy.encoder.load_pretrained(model_cfg["encoder_weights"])
            return MultirateWAM(backbone, None, schedule, compact_policy=compact_policy,
                native_teacher_weight=float(self.config.get('loss', {}).get('native_teacher_weight', 0.0))).to(self.device)
        fast_cfg = model_cfg["fast_conditioner"]
        if model_cfg.get("native_multirate", False):
            from dynamicwam.inference.multirate_model_loader import build_fast_conditioner
            kwargs = dict(fast_cfg)
            if kwargs.get("motion_kv", False):
                kwargs.update(motion_mean=self.motion_statistics["mean"], motion_scale=self.motion_statistics["scale"])
            conditioner = build_fast_conditioner(kwargs)
            return MultirateWAM(backbone, conditioner, schedule,
                native_multirate=True, fast_steps=int(self.config["native_fast_steps"]),
                native_teacher_weight=float(self.config["loss"]["native_teacher_weight"])).to(self.device)
        conditioner = FastConditioner(
            dim=int(fast_cfg["dim"]),
            layers=int(fast_cfg["layers"]),
            heads=int(fast_cfg["heads"]),
            head_dim=int(fast_cfg["head_dim"]),
            text_dim=int(fast_cfg["text_dim"]),
            state_dim=int(fast_cfg["state_dim"]),
            sim_scale=float(fast_cfg["sim_scale"]),
            action_scale=float(fast_cfg["action_scale"]),
        )
        return MultirateWAM(backbone, conditioner, schedule).to(self.device)

    def _load_initialization(self) -> None:
        checkpoint_path = self.config["model"]["initial_checkpoint"]
        if not checkpoint_path:
            raise ValueError("multirate training requires init_checkpoint")
        payload = load_checkpoint_payload(checkpoint_path)
        self._check_checkpoint_identity(payload, checkpoint_path)
        if self.config.get("native_phase") == "distill":
            native_contract = payload.get("config", {}).get("runtime_contract", {}).get("native_multirate", {})
            if native_contract.get("phase") != "adapt" or not native_contract.get("world_aware"):
                raise ValueError("native distillation must initialize a trained native adaptation export")
        self.action_normalization = payload["config"].get("action_normalization")
        if not self.action_normalization or "stats" not in self.action_normalization:
            raise RuntimeError("initial checkpoint lacks explicit action normalization statistics")
        if self.config.get('feedback_initialize',False):
            current = self.model.state_dict()
            incoming = payload['model']
            missing = set(current)-set(incoming)
            if not missing or any(not k.startswith('fast_policy.feedback_') for k in missing) or set(incoming)-set(current):
                raise ValueError('feedback migration may initialize only the new feedback modules')
            payload = dict(payload,model={**current,**incoming})
        if self.config.get("initialize_joint_interface", False):
            payload, new_keys = migrate_joint_interface_initialization(self.model, payload)
            self.joint_initialization = {"source": checkpoint_path,
                "new_parameter_keys": new_keys, "kind": "explicit_v1_to_v2_initialization"}
        kind = install_full_checkpoint(
            self.model,
            payload,
            allow_v2_init=True,
        )
        logger.info("Initialized multirate weights from %s (%s)", checkpoint_path, kind)

    def _build_native_teacher(self) -> None:
        self.native_teacher = None
        self.native_teacher_identity = None
        if self.objective.native_teacher_weight <= 0:
            return
        path = self.config["model"].get("native_teacher_checkpoint")
        if self.phase == "warmup" and not self.model.feedback_phase and not self.model.native_multirate:
            path = self.config["model"]["initial_checkpoint"]
        if not path:
            raise ValueError("joint training requires an immutable native teacher checkpoint")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.native_teacher_identity = {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}
        if self.phase == "warmup" and not self.model.feedback_phase and not self.model.native_multirate:
            return  # The frozen backbone itself is the teacher in this phase.
        payload = load_checkpoint_payload(path)
        self._check_checkpoint_identity(payload, path)
        if payload.get("format") != "dynamicwam_absolute_motion_checkpoint":
            raise ValueError("fixed teacher must be the original absolute-motion WAM export")
        if payload["config"].get("action_normalization") != self.action_normalization:
            raise ValueError("fixed teacher uses different action normalization")
        # Unregistered on the trainable wrapper: no DDP/ZeRO ownership and no
        # duplicated teacher tensors in each checkpoint. Identity is persisted.
        teacher = copy.deepcopy(self.model.backbone)
        teacher.load_state_dict(payload["model"], strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        self.native_teacher = teacher
        if self.model.native_multirate:
            from dynamicwam.models.causal_world_bridge import CausalWorldBridge
            self.native_source_bridge_builder = CausalWorldBridge(BridgeSchedule.from_training_config(self.config))
        self.model.external_native_teacher = True
        logger.info("Fixed native teacher: %s", self.native_teacher_identity)

    def _attach_rtc_prefix(self, prepared, generator=None):
        maximum = int(self.config.get("rtc_prefix_max", 0))
        if maximum:
            from dynamicwam.models.rtc_prefix import PrefixCondition
            clean = prepared["clean_actions"]
            if "async_timing" in prepared:
                lengths = prepared["async_timing"]["rtc_prefix_lengths"].to(device=clean.device,dtype=torch.long)
                actions = prepared["rtc_sample_prefix_actions"].detach()
                from dynamicwam.models.rtc_prefix import validate_prefix_condition
                validate_prefix_condition(PrefixCondition(actions,lengths),max_prefix=maximum,reference=clean)
            else:
                lengths = torch.randint(0, maximum + 1, (clean.shape[0],), device=clean.device, generator=generator)
                actions = clean.detach()
            prepared["rtc_prefix"] = PrefixCondition(actions, lengths)
            if "action_target_times" in prepared:
                offsets = prepared["current"].action_target_offsets
                if offsets is None:
                    raise ValueError("joint prefix conditioning requires action target offsets")
                endpoint_offset = offsets.gather(1, (lengths - 1).clamp_min(0)[:, None]).squeeze(1)
                delay = torch.where(lengths > 0, endpoint_offset, torch.zeros_like(endpoint_offset))
                if not bool(torch.isfinite(delay).all()) or bool((delay < 0).any()):
                    raise ValueError("RTC prefix endpoint times precede current observation")
                prepared["current"] = replace(prepared["current"], observation_delay_seconds=delay)
            elif self.config.get("native_source_bridge") == "student_joint":
                raise ValueError("joint RTC training requires exact prefix endpoint times")

    @torch.no_grad()
    def _attach_native_teacher(self, prepared: dict) -> None:
        teacher = getattr(self, "native_teacher", None)
        if teacher is None:
            return
        from dynamicwam.models.native_wam_teacher import sample_native_wam
        current = prepared["current"]
        source = replace(prepared["source"],
            condition_latent=prepared["aligned"]["condition_latent"], state=current.state,
            motion=current.motion, motion_interval_valid_mask=current.motion_interval_valid_mask,
            motion_acceleration_valid_mask=current.motion_acceleration_valid_mask)
        teacher.eval()
        teacher_noise = prepared["source_noise"]
        if self.config.get("native_phase"):
            teacher_noise = replace(teacher_noise, actions=prepared["native_fast_noise"])
        prepared["native_teacher_actions"] = sample_native_wam(
            teacher, source, teacher_noise, BridgeSchedule.from_training_config(self.config))
        if self.config.get("native_phase") and self.config.get("native_source_bridge", "immutable_official") == "immutable_official":
            prepared["native_source_bridge"] = self.native_source_bridge_builder(
                teacher, prepared["source"], prepared["source_noise"], detach_cache=True)

    def _check_transfer_compatibility(self, payload: dict) -> None:
        cfg = payload["config"]
        identity = validate_dataset_identity(cfg["dataset_identity"])
        if identity["action_stats_sha256"] != self.dataset_identity["action_stats_sha256"]:
            raise RuntimeError("pretrained transfer requires identical action normalization")
        motion = validate_checkpoint_motion_metadata(cfg["model"]["absolute_motion"])
        ignored = {"feature_mean", "feature_scale", "statistics_sha256"}
        if {k:v for k,v in motion.items() if k not in ignored} != {
            k:v for k,v in self.motion_checkpoint_metadata.items() if k not in ignored}:
            raise RuntimeError("pretrained motion feature semantics differ from packed data")

    def _check_checkpoint_identity(self, payload: dict[str, Any], path: str) -> None:
        config = payload.get("config")
        if not isinstance(config, dict) or "dataset_identity" not in config:
            raise RuntimeError(f"checkpoint is missing dataset_identity: {path}")
        identity = validate_dataset_identity(config["dataset_identity"])
        transfer = bool(self.config.get("allow_pretrained_dataset_transfer", False))
        if identity != self.dataset_identity:
            if not transfer or payload.get("format") != "dynamicwam_absolute_motion_checkpoint":
                raise RuntimeError(f"checkpoint dataset identity differs from paired data: {path}")
            self._check_transfer_compatibility(payload)
        saved_runtime = config.get("runtime_contract", {})
        if saved_runtime.get("training_scope") == "smoke_subset" and (
                not 0 < int(self.config.get("max_steps", 0)) <= 4 or self.config.get("resume_from")):
            raise RuntimeError("a smoke-subset checkpoint cannot initialize or resume formal training")
        replay_transition = False
        if self.config["dataset"].get("kind") == "real_rollout":
            replay_transition = (
                int(config.get("model",{}).get("fast_conditioner",{}).get("interface_version",1)) == 3
                and saved_runtime.get("physical_grid") == self.dataset.physical_grid_identity
                and saved_runtime.get("episode_partition_identity") == self.dataset.parent_episode_partition_identity
                and bool(getattr(self.dataset,"replay_identity",None)))
            if saved_runtime.get("pairing_identity") != self.dataset.pairing_identity and not replay_transition:
                raise RuntimeError("replay refinement initialization lacks matching V3 physical-grid and parent holdout provenance")
        saved_pairing = config.get("runtime_contract", {}).get("pairing_identity")
        if saved_pairing is not None and saved_pairing != self.dataset.pairing_identity:
            # Old exports lack the age-independent partition hash. Recompute
            # their exact H/R pairing on the current data+holdout before allowing
            # the explicit new age support; a changed holdout still fails.
            legacy = getattr(self.dataset, "legacy_pairing_identity", None)
            old_fast = config.get("model", {}).get("fast_conditioner", {})
            migration = (self.config.get("initialize_joint_interface", False)
                and int(old_fast.get("interface_version", 1)) == 1
                and old_fast.get("motion_kv") and saved_pairing == legacy)
            if not migration and not replay_transition:
                raise RuntimeError("initial multirate checkpoint used a different episode holdout/pairing")
            self.pairing_migration = {"previous": saved_pairing,
                "current": self.dataset.pairing_identity,
                "episode_partition_identity": self.dataset.episode_partition_identity,
                "reason": "explicit verified real-rollout refinement on new dev episodes" if replay_transition else "explicit causal cache age enrichment, unchanged episodes"}
        saved_partition = config.get("runtime_contract", {}).get("episode_partition_identity")
        if saved_partition is not None and saved_partition != self.dataset.episode_partition_identity and not replay_transition:
            raise RuntimeError("initial multirate checkpoint episode holdout changed")
        model_cfg = config.get("model")
        if not isinstance(model_cfg, dict) or "absolute_motion" not in model_cfg:
            raise RuntimeError(f"checkpoint is missing motion metadata: {path}")
        motion = validate_checkpoint_motion_metadata(model_cfg["absolute_motion"])
        if motion != self.motion_checkpoint_metadata:
            raise RuntimeError(
                f"checkpoint motion statistics differ from the paired data: {path}"
            )

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def train_step(self, batch: dict[str, Any]) -> dict[str, float] | None:
        self.model.train()
        fast_cfg = self.config["model"]["fast_conditioner"]
        grad_norm = None
        distributed_type = getattr(self.accelerator, "distributed_type", None)
        is_deepspeed = getattr(distributed_type, "value", distributed_type) == "DEEPSPEED"
        with self.accelerator.accumulate(self.model):
            prepared = prepare_multirate_batch(
                batch,
                timesteps=self._timesteps,
                sigmas=self._sigmas,
                device=self.device,
                dtype=self._timesteps.dtype,
                phase=self.phase,
                native_teacher=float(self.config.get('loss', {}).get('native_teacher_weight', 0.0)) > 0,
                native_multirate=bool(self.config.get("native_phase")),
                joint_interface_version=int(fast_cfg.get("interface_version", 2)),
                null_bridge_probability=float(self.config["null_bridge_probability"]),
                sim_scale=float(fast_cfg["sim_scale"]),
                action_scale=float(fast_cfg["action_scale"]),
            )
            self._attach_rtc_prefix(prepared)
            self._attach_native_teacher(prepared)
            outputs = self.model(prepared, mode="multirate_train", phase=self.phase)
            total, metrics = self.objective(
                outputs,
                prepared["fast_target"],
                prepared["action_target"],
                prepared.get("video_target"),
                executed_horizon=int(self.config["dataset"]["execute_horizon"])
                if self.config.get("fast_supervision", "executed_prefix") != "full_chunk"
                and "dataset" in self.config and "execute_horizon" in self.config["dataset"]
                else None,
            )
            if not torch.isfinite(total.detach()):
                raise FloatingPointError("nonfinite multirate loss; aborting before optimizer update")
            self.accelerator.backward(total)
            if self.accelerator.sync_gradients and not is_deepspeed:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self._trainable_parameters(),
                    max_norm=float(self.config["grad_clip_norm"]),
                )
                if grad_norm is None:
                    raise RuntimeError("gradient clipping did not return a gradient norm")
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("nonfinite gradient norm; aborting before optimizer update")
            self.optimizer.step()
            if self.accelerator.sync_gradients and is_deepspeed:
                # Accelerate delegates clipping and the optimizer update to the
                # DeepSpeed engine (some versions do so inside backward()). Read
                # the engine's stored global norm only after that update boundary.
                engine = self.model
                if not callable(getattr(engine, "get_global_grad_norm", None)):
                    engine = getattr(getattr(self.accelerator, "deepspeed_engine_wrapped", None), "engine", None)
                if not callable(getattr(engine, "get_global_grad_norm", None)):
                    raise RuntimeError("DeepSpeed engine does not expose get_global_grad_norm")
                grad_norm = engine.get_global_grad_norm()
                if grad_norm is None:
                    raise RuntimeError("DeepSpeed global gradient norm is unavailable after a synchronized update")
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("nonfinite DeepSpeed global gradient norm after optimizer update; stopping training")
            if self.accelerator.sync_gradients and not getattr(self.accelerator, "optimizer_step_was_skipped", False):
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.sync_gradients:
            return None
        if grad_norm is None:
            raise RuntimeError("synchronized training step has no measured gradient norm")
        metrics["grad/norm"] = float(
            grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
        )
        metrics.update(get_learning_rate_metrics(self.optimizer))
        return metrics

    def _resume_contract(self) -> dict:
        # The cursor is meaningful only under the same sampling/optimizer clock.
        keys = ("per_device_batch_size", "gradient_accumulation_steps", "max_steps",
                "warmup_steps", "min_lr_ratio", "action_learning_rate", "fast_learning_rate",
                "video_learning_rate", "weight_decay", "grad_clip_norm", "null_bridge_probability", "loss")
        dataset_keys = ("samples_per_episode", "stress_probability", "partition_seed",
                        "global_downsample_rate", "chunk_size", "execute_horizon", "every_fast_queries",
                        "age_weighting")
        return {"training": {**{k: self.config[k] for k in keys}, "rtc_prefix_max": int(self.config.get("rtc_prefix_max", 0)), "fast_supervision": self.config.get("fast_supervision", "executed_prefix") + "_v1",
                                     "max_bridge_age_seconds": self.config.get("max_bridge_age_seconds"),
                                     **({"native_source_bridge": "student_joint",
                                         "gradient_checkpointing": bool(self.config.get("gradient_checkpointing", False)),
                                         "motion_prediction_weight": float(self.config.get("motion_prediction_weight",0.0))}
                                        if self.config.get("native_source_bridge") == "student_joint" else {}),
                                     **({"feedback_phase": self.config["feedback_phase"],
                                        "feedback_world_refresh_actions": self.config.get("feedback_world_refresh_actions",16)} if self.config.get("feedback_phase") else {}),
                                     **({k: self.config[k] for k in ("native_phase", "native_fast_steps", "native_expert_weight", "native_fm_anchor_weight")}
                                        if self.config.get("native_phase") else {}),
                                     **({"native_teacher": self.native_teacher_identity} if getattr(self, "native_teacher_identity", None) else {})},
                "sampling": {**{k: self.config["dataset"][k] for k in dataset_keys},
                             "sampler_seed": int(self.config["dataset"].get("sampler_seed", self.config["dataset"]["partition_seed"])),
                             **({"cache_age_actions": self.config["dataset"].get("cache_age_actions"),
                                 "joint_supervision": True,
                                 "cache_age_jitter": bool(self.config["dataset"].get("cache_age_jitter", False)),
                                 "cache_age_seed": int(self.config["dataset"].get("cache_age_seed", 0)),
                                 **({"physical_grid_identity": self.dataset.physical_grid_identity,
                                     "async_seed": int(self.config["dataset"].get("async_seed",20261015)),
                                     **({"rtc_prefix_sampling": self.dataset.rtc_prefix_sampling_contract}
                                        if getattr(self.dataset,"rtc_prefix_sampling_contract",{}).get("mode","delay_minimum_v1") != "delay_minimum_v1" else {}),
                                      "dataset_kind": self.config["dataset"].get("kind","physical_grid"),
                                      "replay_identity": getattr(self.dataset,"replay_identity",None),
                                      "replay_target_policy": self.config["dataset"].get("replay_target_policy")}
                                    if hasattr(self.dataset,"physical_grid_identity") else {})}
                                if self.config["dataset"].get("joint_supervision") else {})},
                "batches_per_epoch": len(self.train_loader), "pair_count": len(self.dataset)}

    def _resume_if_needed(self) -> None:
        resume_from = self.config.get("resume_from")
        if not resume_from:
            return
        progress_path = Path(resume_from) / "multirate_progress.json"
        if not progress_path.is_file():
            raise RuntimeError("resume requires multirate_progress.json; legacy checkpoints may initialize a new run, not claim exact resume")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["pairing_identity"] != self.dataset.pairing_identity:
            raise RuntimeError("resume episode partition/pairing differs from checkpoint")
        if progress["phase"] != self.phase or progress["world_size"] != self.accelerator.num_processes:
            raise RuntimeError("exact resume requires the same phase and world size")
        if progress["schedule"] != BridgeSchedule.from_training_config(self.config).to_config():
            raise RuntimeError("resume schedule differs from checkpoint")
        saved_contract = progress.get("resume_contract", {})
        if "sampling" in saved_contract and "sampler_seed" not in saved_contract["sampling"]:
            saved_contract["sampling"]["sampler_seed"] = saved_contract["sampling"].get("partition_seed")
        if saved_contract != self._resume_contract():
            raise RuntimeError("resume sampling/optimization configuration differs from checkpoint")
        self.accelerator.load_state(str(resume_from))
        self.global_step = int(progress["global_step"])
        self.data_epoch = int(progress["data_epoch"])
        self.batches_in_epoch = int(progress["batches_in_epoch"])
        logger.info("Resumed multirate training from %s at step %d", resume_from, self.global_step)

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "phase": self.config["phase"],
            "training_recipe": self._resume_contract(),
            "initialization_provenance": getattr(self, "initialization_provenance", None),
            "joint_initialization": getattr(self, "joint_initialization", None),
            "pairing_migration": getattr(self, "pairing_migration", None),
            "dataset_identity": dict(self.dataset_identity),
            "action_normalization": self.action_normalization,
            "runtime_contract": {
                "version": 1,
                **({"rtc": {"version": 1, "mode": "training_time_action_prefix", "max_prefix_actions": int(self.config["rtc_prefix_max"]),
                    "clean_timestep": 0.0, "loss": "per_sample_suffix_mean", "includes_zero_prefix": (rtc_sampling_export_metadata(self.config,self.dataset) or {}).get("mode") not in ("fixed_h4_v1", "recorded_real_prefix_v1"),
                    "queue_clock": "absolute_action_effect_times", "overflow": "reject_and_refresh", "execute_suffix_actions": 4}}
                    if self.config.get("rtc_prefix_max", 0) else {}),
                **({"motion_kv": {"version": int(self.config["model"]["fast_conditioner"].get("interface_version", 1)),
                    "interface_version": int(self.config["model"]["fast_conditioner"].get("interface_version", 1)), "rank": self.config["model"]["fast_conditioner"]["kv_rank"],
                    "max_relative_update": self.config["model"]["fast_conditioner"]["max_relative_update"],
                    "scope": "all_layer_native_joint_attention_kv", "motion": "causal_native_optical_flow_statistics_with_masks",
                    "localization": "current_visual_spatial_tokens; no ground-truth object mask"}}
                    if self.config["model"]["fast_conditioner"].get("motion_kv") else {}),
                **({"native_multirate": {"version": int(self.config["model"]["fast_conditioner"].get("interface_version", 2)) if self.config.get("native_source_bridge") == "student_joint" else 1, "phase": self.config["native_phase"],
                    "fast_steps": int(self.config["native_fast_steps"]), "world_aware": True,
                    "source_bridge": self.config.get("native_source_bridge", "immutable_official"),
                    "world_refresh_actions": int(self.config["dataset"]["execute_horizon"]) * int(self.config["dataset"]["every_fast_queries"]),
                    **({"slow_schedule": "asynchronous_latest_completed_v1"}
                       if int(self.config["model"]["fast_conditioner"].get("interface_version",1)) >= 3 else {}),
                    "teacher": self.native_teacher_identity}}
                    if self.config.get("native_phase") else {}),
                **({'feedback': {'ready':self.config.get('feedback_phase')=='policy',
                    'version':1,'actor_evaluations':2,'world_refresh_actions':int(self.config.get('feedback_world_refresh_actions',16)),
                    'forecast':'action_conditioned_vae_patches_2x4x4'}} if self.config.get('feedback_phase') else {}),
                "schedule": BridgeSchedule.from_training_config(self.config).to_config(),
                "preprocess": "three_view_160_imagenet_mask_v1",
                "clock_mode": self.config["dataset"]["clock_mode"],
                "max_age_actions": self.dataset.max_supported_cache_age_actions,
                **({"max_age_seconds": 4.96,
                    "physical_grid": getattr(self.dataset,"physical_grid_identity",None),
                    "physical_grid_contract": {"schema_version": 1, "action_interval_seconds": .16,
                        "action_horizon": 16, "source_plan_alignment": "physical_time_linear_arm_previous_hold_gripper",
                        "latent_manifest_sha256": self.dataset.readiness["latent_manifest_sha256"]},
                    **({"rtc_prefix_sampling": rtc_sampling_export_metadata(self.config,self.dataset)}
                       if rtc_sampling_export_metadata(self.config,self.dataset) is not None else {}),
                    "source_retention": "fixed_grid_source_video_action_fm",
                    "slow_source_prefix": "clean_committed_v1",
                    "motion_forecast": "source_kv_native_motion_elapsed_to_current_stats",
                    "proposal_feedback": "sigma_conditioned_physical_plan_v1" if self.config["model"]["fast_conditioner"].get("proposal_feedback", False) else "disabled",
                    "previous_plan": "soft_context_only; hard_prefix_is_irrevocable",
                    "source_physical_time_conditioning": "native_action_tokens_target_offset_phase_v1" if self.config["model"]["action_expert"].get("source_physical_time_conditioning", False) else "disabled",
                    "source_plan_completion": "same_noise_native_K10_tail_stopgrad_v1" if self.config["model"]["fast_conditioner"].get("complete_source_plan", False) else "disabled",
                    "training_data_kind": self.config["dataset"].get("kind","physical_grid"),
                    "training_scope": "smoke_subset" if self.config["dataset"].get("smoke_subset", False) else "full",
                    "replay_identity": getattr(self.dataset,"replay_identity",None),
                    "prefix_training_source": "on_policy_executed" if self.config["dataset"].get("kind") == "real_rollout" else "expert_demonstration"}
                   if int(self.config["model"]["fast_conditioner"].get("interface_version",1)) >= 3 else {}),
                "pairing_identity": self.dataset.pairing_identity,
                "episode_partition_identity": self.dataset.episode_partition_identity,
                "execute_horizon": int(self.config["dataset"]["execute_horizon"]),
                "supervised_action_horizon": int(self.config["model"]["action_expert"]["chunk_size"])
                    if self.config.get("fast_supervision") == "full_chunk" else int(self.config["dataset"]["execute_horizon"]),
                **({"max_age_seconds": float(self.config["max_bridge_age_seconds"])}
                   if self.config.get("max_bridge_age_seconds") is not None else {}),
            },
            "model": {
                "absolute_motion": self.motion_checkpoint_metadata,
                "fast_conditioner": dict(self.config["model"]["fast_conditioner"]),
                "action_expert": dict(self.config["model"]["action_expert"]),
                "compact_fast": self.config["model"].get("compact_fast"),
                **({"native_multirate": True, "fast_steps": int(self.config["native_fast_steps"])}
                   if self.config.get("native_phase") else {}),
            },
        }

    def save_checkpoint(self) -> None:
        ckpt_dir = get_run_dir(self.config) / f"step_{self.global_step}"
        self.accelerator.save_state(str(ckpt_dir))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            progress = dict(global_step=self.global_step, data_epoch=self.data_epoch,
                batches_in_epoch=self.batches_in_epoch, pairing_identity=self.dataset.pairing_identity,
                phase=self.phase, world_size=self.accelerator.num_processes,
                schedule=BridgeSchedule.from_training_config(self.config).to_config(),
                resume_contract=self._resume_contract())
            path = ckpt_dir / "multirate_progress.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(progress, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        # State gathering may be collective (e.g. ZeRO-3). Every rank must enter.
        self.export_multirate()
        self.accelerator.wait_for_everyone()

    def export_multirate(self) -> None:
        gathered_state = self.accelerator.get_state_dict(self.model)
        if not self.accelerator.is_main_process:
            return
        export_dir = get_export_dir(self.config)
        export_dir.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in gathered_state.items()}
        config = self._checkpoint_config()
        full_path = export_dir / f"multirate_step_{self.global_step}.pt"
        fast_path = export_dir / f"multirate_fast_step_{self.global_step}.pt"
        spec = fast_module_spec(
            self.config,
            history_count=int(self.motion_checkpoint_metadata["history_count"]),
            motion_mean=tuple(float(value) for value in self.motion_statistics["mean"]),
            motion_scale=tuple(float(value) for value in self.motion_statistics["scale"]),
        )
        spec["architecture"] = "native" if self.config.get("native_phase") else "compact" if self.config["model"].get("compact_fast") else "reference"
        if self.config.get("native_phase"):
            spec["native_multirate"] = True
            spec["fast_steps"] = int(self.config["native_fast_steps"])
            spec["fast_schedule"] = BridgeSchedule.from_training_config(self.config).with_action_steps(
                int(self.config["native_fast_steps"])).to_config()
        spec["compact_fast"] = self.config["model"].get("compact_fast")
        spec["schedule"] = BridgeSchedule.from_training_config(self.config).to_config()
        spec["wan_attention"]["precision"] = self.config["model"]["compact_wan"]["precision"]
        spec["wan_attention"]["window_sizes"] = [list(w) for w in self.accelerator.unwrap_model(self.model).backbone.action_attention_spec.window_sizes]
        spec["runtime_contract"] = config["runtime_contract"]
        spec["action_normalization"] = self.action_normalization
        payload = {
            "format": MULTIRATE_CHECKPOINT_FORMAT,
            "version": MULTIRATE_CHECKPOINT_VERSION,
            "model": state,
            "global_step": self.global_step,
            "config": config,
            "modules": spec,
        }
        fast_payload = export_fast_only_payload(
            state,
            module_spec=spec,
            global_step=self.global_step,
        )
        _atomic_torch_save(payload, full_path)
        _atomic_torch_save(fast_payload, fast_path)
        logger.info("Exported multirate checkpoint to %s", full_path)
        logger.info("Exported fast-only weights to %s", fast_path)

    def _close_data_loaders(self) -> None:
        """Stop owned data workers/pinning threads before CUDA/NCCL teardown."""
        errors = []
        iterator = getattr(self, "_active_data_iterator", None)
        self._active_data_iterator = None
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
        loaders = list(getattr(self, "_run_loaders", None) or ())
        loaders.extend((getattr(self, "train_loader", None), getattr(self, "val_loader", None)))
        seen = set()
        for loader in loaders:
            if loader is None:
                continue
            base = getattr(loader, "base_dataloader", loader)
            if id(base) in seen:
                continue
            seen.add(id(base))
            worker_iterator = getattr(base, "_iterator", None)
            shutdown = getattr(worker_iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    # PyTorch has no public close for persistent DataLoaders.
                    # This joins the pin-memory thread before owned workers.
                    shutdown()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    base._iterator = None
        for dataset in (getattr(self, "dataset", None), getattr(self, "val_dataset", None)):
            close = getattr(dataset, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
        self._run_loaders = []

    def release_training_resources(self) -> None:
        """Release our references; Accelerator then owns engine destruction."""
        try:
            self._close_data_loaders()
        finally:
            # free_memory does not clear registered custom checkpoint objects;
            # our scheduler retains its optimizer. Remove only this scheduler.
            scheduler = getattr(self, "scheduler", None)
            registered = getattr(self.accelerator, "_custom_objects", None)
            if isinstance(registered, list):
                registered[:] = [obj for obj in registered if obj is not scheduler]
            for name in ("model", "optimizer", "scheduler", "native_teacher",
                         "native_source_bridge_builder", "_timesteps", "_sigmas",
                         "train_loader", "val_loader", "dataset", "val_dataset",
                         "_active_data_iterator", "_run_loaders"):
                if hasattr(self, name):
                    setattr(self, name, None)

    def run(self) -> None:
        self._run_loaders = []
        self._active_data_iterator = None
        try:
            self._run_training()
        except BaseException:
            try:
                self._close_data_loaders()
            except BaseException:
                logger.exception("DataLoader cleanup failed while preserving training error")
            raise
        else:
            self._close_data_loaders()

    def _run_training(self) -> None:
        if self.accelerator.is_main_process:
            partition = get_run_dir(self.config) / "episode_partition.json"
            partition.parent.mkdir(parents=True, exist_ok=True)
            partition.write_text(json.dumps({"identity": self.dataset.pairing_identity, "assignments": self.dataset.episode_partition}, sort_keys=True), encoding="utf-8")
            manifest = get_run_dir(self.config) / "pair_manifest.jsonl"
            write_pair_manifest(manifest, self.dataset.records)
            logger.info("Paired samples: %s", json.dumps(self.dataset.drop_report, sort_keys=True))
            logger.info("Wrote pair manifest to %s", manifest)
            logger.info("Multirate phase: %s", self.config["phase"])
            logger.info("Run directory: %s", get_run_dir(self.config))
        self.log_tracker = LogTracker(
            str(self.config["phase"]),
            int(self.config["max_steps"]),
            loss_precision=4 if self.phase == "joint" else 3,
        )
        init_experiment_trackers(self.accelerator, self.config)
        self._resume_if_needed()
        self._last_log_step = self.global_step
        self._last_log_time = time.monotonic()
        samples_per_step = get_effective_global_batch_size(
            self.config, self.accelerator.num_processes
        )
        log_interval = int(self.config["log_interval"])
        checkpoint_interval = int(self.config["checkpoint_interval"])
        self.dataset.set_epoch(self.data_epoch)
        self.pair_sampler.set_epoch(self.data_epoch)
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(self.data_epoch)
        if self.batches_in_epoch > len(self.train_loader):
            raise RuntimeError("resume data cursor exceeds epoch length")
        # A zero-cursor skip still clones DataLoaderShard in Accelerate 1.14.
        # Reuse the owned loader unless an actual resume cursor requires a clone.
        loader = (self.accelerator.skip_first_batches(self.train_loader, self.batches_in_epoch)
                  if self.batches_in_epoch else self.train_loader)
        self._run_loaders.append(loader)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(self.data_epoch)
        data_iter = iter(loader)
        self._active_data_iterator = data_iter
        while self.global_step < int(self.config["max_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                self.data_epoch += 1
                self.batches_in_epoch = 0
                self.dataset.set_epoch(self.data_epoch)
                self.pair_sampler.set_epoch(self.data_epoch)
                if hasattr(self.train_loader, "set_epoch"):
                    self.train_loader.set_epoch(self.data_epoch)
                data_iter = iter(self.train_loader)
                self._active_data_iterator = data_iter
                batch = next(data_iter)
            self.batches_in_epoch += 1
            metrics = self.train_step(batch)
            if metrics is None:
                continue
            self.global_step += 1
            metrics.update(
                get_epoch_progress_metrics(
                    self.global_step,
                    self.train_loader,
                    gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
                    world_size=self.accelerator.num_processes,
                )
            )
            if self.global_step % log_interval == 0:
                self._last_log_step, self._last_log_time = add_speed_metrics(
                    metrics,
                    self.global_step,
                    self._last_log_step,
                    self._last_log_time,
                    samples_per_step=samples_per_step,
                )
                log_metrics = reduce_metrics_for_log(self.accelerator, metrics)
                self.accelerator.log(log_metrics, step=self.global_step)
                if self.accelerator.is_main_process:
                    for line in self.log_tracker.log_step(
                        self.global_step,
                        log_metrics,
                        loss_keys=[
                            ("total", "loss/total"),
                            ("fast", "loss/fast"),
                            ("action", "loss/action"),
                            ("video", "loss/video"),
                            ("native_teacher", "loss/native_teacher"),
                            ("future", "loss/future"),
                        ],
                        lr_keys=[
                            ("action", "lr/action_core"),
                            ("fast", "lr/fast_conditioner"),
                            ("video", "lr/video_expert"),
                        ],
                    ):
                        logger.info("%s", line)
            if self.global_step % checkpoint_interval == 0:
                self._run_validation()
                self.save_checkpoint()
        if self.global_step == 0 or self.global_step % checkpoint_interval != 0:
            self._run_validation()
            self.save_checkpoint()

    def _run_validation(self) -> None:
        if self.config.get("native_phase"):
            return self._run_native_validation()
        if self.config.get('feedback_phase'):
            return self._run_feedback_validation()
        if self.val_loader is None:
            return
        was_training = self.model.training
        self.model.eval()
        fast_cfg = self.config["model"]["fast_conditioner"]
        losses = []
        generator = torch.Generator(device=self.device).manual_seed(int(self.config["dataset"]["partition_seed"]) + 7919)
        self.val_dataset.set_epoch(0)
        self.val_sampler.set_epoch(0)
        if hasattr(self.val_loader, "set_epoch"):
            self.val_loader.set_epoch(0)
        try:
            for batch_index, batch in enumerate(self.val_loader):
                limit = int(self.config.get("validation_batches", -1))
                if limit >= 0 and batch_index >= limit:
                    break
                with torch.no_grad():
                    prepared = prepare_multirate_batch(
                        batch,
                        timesteps=self._timesteps,
                        sigmas=self._sigmas,
                        device=self.device,
                        dtype=self._timesteps.dtype,
                        phase=self.phase,
                        native_teacher=float(self.config.get('loss', {}).get('native_teacher_weight', 0.0)) > 0,
                        null_bridge_probability=0.0,
                        sim_scale=float(fast_cfg["sim_scale"]),
                        action_scale=float(fast_cfg["action_scale"]),
                        null_bridge=False,
                        generator=generator,
                    )
                    self._attach_rtc_prefix(prepared, generator)
                    self._attach_native_teacher(prepared)
                    outputs = self.model(prepared, mode="multirate_train", phase=self.phase)
                    # Same targets, bridge removed, so the gap is the historical bridge.
                    null_prepared = dict(prepared)
                    null_prepared["null_bridge"] = True
                    null_prepared["skip_native_teacher"] = True
                    null_outputs = self.model(null_prepared, mode="multirate_train", phase=self.phase)
                fast_error = (outputs["fast_pred"].float() - prepared["fast_target"].float()).square()
                fast_loss = fast_error.flatten(1).mean(1)
                prefix_loss = fast_error[:, :int(self.config["dataset"]["execute_horizon"])].flatten(1).mean(1)
                null_error = (null_outputs["fast_pred"].float() - prepared["fast_target"].float()).square()
                null_loss = null_error.flatten(1).mean(1)
                null_prefix = null_error[:, :int(self.config["dataset"]["execute_horizon"])].flatten(1).mean(1)
                action_loss = (outputs["action_pred"].float() - prepared["action_target"].float()).square().flatten(1).mean(1)
                video_loss = torch.zeros_like(fast_loss)
                if self.objective.video_weight > 0:
                    video_loss = (outputs["video_pred"].float() - prepared["video_target"].float()).square().flatten(1).mean(1)
                supervised = fast_loss if self.config.get("fast_supervision") == "full_chunk" else prefix_loss
                total = supervised*self.objective.fast_weight + action_loss*self.objective.action_weight + video_loss*self.objective.video_weight
                teacher_loss = torch.zeros_like(fast_loss)
                teacher_weight = float(self.config.get('loss', {}).get('native_teacher_weight', 0.0))
                if teacher_weight > 0:
                    teacher_loss = (outputs['teacher_pred'].float() - outputs['teacher_target'].float()).square().flatten(1).mean(1)
                    total = total + teacher_weight * teacher_loss
                values = torch.stack((total, fast_loss, action_loss, video_loss, prefix_loss,
                                      batch["delta_actions"].to(device=total.device, dtype=total.dtype),
                                      null_loss, null_prefix, teacher_loss), dim=1)
                # ProcessGroupAlignedSampler + even_batches=False has NO padded
                # examples. gather_for_metrics incorrectly truncates real final
                # rows using len(dataset), which differs from len(sampler).
                losses.append(self.accelerator.gather(values).detach().cpu())
        finally:
            self.model.train(was_training)
        if losses and self.accelerator.is_main_process:
            rows = torch.cat(losses)
            metrics = {"val/"+name: float(rows[:, index].mean()) for index, name in enumerate(
                ("loss", "fast", "action", "video", "fast_executed_prefix"))}
            metrics["val/fast_null"] = float(rows[:, 6].mean())
            metrics["val/fast_executed_prefix_null"] = float(rows[:, 7].mean())
            metrics["val/bridge_prefix_gain"] = float((rows[:, 7] - rows[:, 4]).mean())
            metrics['val/native_teacher'] = float(rows[:, 8].mean())
            metrics["val/samples"] = int(rows.shape[0])
            for age in rows[:, 5].unique().tolist():
                subset = rows[rows[:, 5] == age]
                metrics[f"val/age_{int(age)}/fast"] = float(subset[:, 1].mean())
                metrics[f"val/age_{int(age)}/fast_executed_prefix"] = float(subset[:, 4].mean())
                metrics[f"val/age_{int(age)}/fast_executed_prefix_null"] = float(subset[:, 7].mean())
                metrics[f"val/age_{int(age)}/bridge_prefix_gain"] = float((subset[:, 7] - subset[:, 4]).mean())
                metrics[f"val/age_{int(age)}/samples"] = int(subset.shape[0])
            self.accelerator.log(metrics, step=self.global_step)


    def _run_native_validation(self):
        """Paired fresh/stale/null evaluation, stratified by every deployed age."""
        if self.val_loader is None:
            return
        was_training = self.model.training
        self.model.eval()
        self.val_dataset.set_epoch(0)
        self.val_sampler.set_epoch(0)
        if hasattr(self.val_loader, "set_epoch"):
            self.val_loader.set_epoch(0)
        generator = torch.Generator(device=self.device).manual_seed(20261004)
        rows = []
        horizon = int(self.config["dataset"]["execute_horizon"])
        limit = int(self.config.get("validation_batches", 32))
        try:
            for index, batch in enumerate(self.val_loader):
                if limit >= 0 and index >= limit:
                    break
                with torch.no_grad():
                    prepared = prepare_multirate_batch(batch, timesteps=self._timesteps,
                        sigmas=self._sigmas, device=self.device, dtype=self._timesteps.dtype,
                        phase=self.phase, native_teacher=True, native_multirate=True,
                        joint_interface_version=int(self.config["model"]["fast_conditioner"].get("interface_version", 2)), null_bridge_probability=0.,
                        null_bridge=False, sim_scale=1., action_scale=16., generator=generator)
                    self._attach_rtc_prefix(prepared, generator)
                    self._attach_native_teacher(prepared)
                    outputs = self.model(prepared, mode="multirate_train", phase=self.phase)
                    null_prepared = dict(prepared, null_bridge=True)
                    current = prepared["current"]
                    null_prepared["current"] = replace(current,
                        age=replace(current.age, bridge_valid=torch.zeros_like(
                            torch.as_tensor(current.age.delta_actions), dtype=torch.bool)))
                    null_outputs = self.model(null_prepared, mode="multirate_train", phase=self.phase)
                    if self.config["native_phase"] == "distill":
                        prediction, dropped = outputs["native_actions"], null_outputs["native_actions"]
                        target, teacher = prepared["clean_actions"], prepared["native_teacher_actions"]
                        teacher_error = (prediction.float() - teacher.float()).square().flatten(1).mean(1)
                    else:
                        prediction, dropped = outputs["fast_pred"], null_outputs["fast_pred"]
                        target = prepared["fast_target"]
                        teacher_error = ((outputs["teacher_pred"].float() - outputs["teacher_target"].float()).square().flatten(1).mean(1)
                            if "teacher_pred" in outputs else torch.zeros(prediction.shape[0],device=prediction.device))
                    errors = (prediction.float() - target.float()).square()
                    null_errors = (dropped.float() - target.float()).square()
                    if prepared.get("rtc_prefix") is not None:
                        lengths = prepared["rtc_prefix"].lengths
                        positions = torch.arange(errors.shape[1], device=errors.device)[None]
                        suffix = positions >= lengths[:, None]
                        executable = suffix & (positions < lengths[:, None] + horizon)
                        full = (errors.mean(-1) * suffix).sum(-1) / suffix.sum(-1)
                        prefix = (errors.mean(-1) * executable).sum(-1) / executable.sum(-1)
                        null_prefix = (null_errors.mean(-1) * executable).sum(-1) / executable.sum(-1)
                        if "teacher_pred" in outputs:
                            teacher_errors = (outputs["teacher_pred"].float() - outputs["teacher_target"].float()).square().mean(-1)
                            teacher_error = (teacher_errors * suffix).sum(-1) / suffix.sum(-1)
                    else:
                        full = errors.flatten(1).mean(1)
                        prefix = errors[:, :horizon].flatten(1).mean(1)
                        null_prefix = null_errors[:, :horizon].flatten(1).mean(1)
                    age = batch["delta_actions"].to(full)
                    total = outputs["native_loss"].detach().expand_as(full)
                    values = torch.stack((total, full, prefix, teacher_error, null_prefix, age), dim=1)
                    rows.append(self.accelerator.gather(values).cpu())
            if rows and self.accelerator.is_main_process:
                values = torch.cat(rows)
                names = ("loss", "expert", "executed_prefix", "native_teacher", "null_prefix")
                metrics = {"val/" + name: float(values[:, i].mean()) for i, name in enumerate(names)}
                metrics["val/bridge_prefix_gain"] = float((values[:, 4] - values[:, 2]).mean())
                metrics["val/samples"] = int(values.shape[0])
                for age in values[:, 5].unique().tolist():
                    subset = values[values[:, 5] == age]
                    for i, name in enumerate(names):
                        metrics[f"val/age_{int(age)}/{name}"] = float(subset[:, i].mean())
                    metrics[f"val/age_{int(age)}/bridge_prefix_gain"] = float((subset[:, 4] - subset[:, 2]).mean())
                    metrics[f"val/age_{int(age)}/samples"] = int(subset.shape[0])
                metrics["val/native_teacher_enabled"] = float(self.config["loss"].get("native_teacher_weight",0)>0)
                self.accelerator.log(metrics, step=self.global_step)
                logger.info("Native multirate validation: %s", json.dumps(metrics, sort_keys=True))
        finally:
            self.model.train(was_training)

    def _run_feedback_validation(self):
        if self.val_loader is None: return
        was_training = self.model.training
        self.model.eval()
        self.val_dataset.set_epoch(0)
        self.val_sampler.set_epoch(0)
        if hasattr(self.val_loader,'set_epoch'): self.val_loader.set_epoch(0)
        rows = []
        generator = torch.Generator(device=self.device).manual_seed(20261003)
        try:
            for i,batch in enumerate(self.val_loader):
                if i >= max(1,int(self.config.get('validation_batches',32))): break
                with torch.no_grad():
                    prepared = prepare_multirate_batch(batch,timesteps=self._timesteps,sigmas=self._sigmas,
                        device=self.device,dtype=self._timesteps.dtype,phase=self.phase,
                        null_bridge_probability=.75,sim_scale=1.,action_scale=16.,generator=generator)
                    self._attach_native_teacher(prepared)
                    outputs = self.model(prepared,mode='multirate_train',phase=self.phase)
                    values = dict(outputs['feedback_metrics'], **{'loss/total':outputs['feedback_loss']})
                    keys = sorted(values)
                    rows.append(self.accelerator.gather(torch.stack([values[k] for k in keys]).reshape(1,-1)).cpu())
            if rows:
                values = torch.cat(rows).mean(0)
                metrics = {k.replace('loss/','val/'):float(v) for k,v in zip(keys,values)}
                self.accelerator.log(metrics,step=self.global_step)
                if self.accelerator.is_main_process: logger.info('Feedback validation: %s',json.dumps(metrics))
        finally:
            self.model.train(was_training)



def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


