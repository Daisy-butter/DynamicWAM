"""Schema for world-action multirate training.

This loader does not accept an absolute-motion v2 profile, and the v2 loader
does not accept this schema. The synchronous baseline file stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamicwam.config.schema import require_exact_keys

MULTIRATE_SCHEMA_VERSION = 3
MULTIRATE_BASELINE = "world_action_multirate_v1"
MULTIRATE_BASELINES = {MULTIRATE_BASELINE, "world_action_multirate_v2", "world_action_multirate_native_v1"}
MULTIRATE_PHASES = {"fast_warmup", "cross_time_joint"}
# Episode holdout stays on partition_seed. Joint must not replay warmup's shuffle.
JOINT_SAMPLER_SEED_SALT = 1_000_003


def _resolve(value: Any, *, base: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Loading the multirate profile requires PyYAML") from exc
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"multirate profile must be a mapping: {path}")
    return raw


@dataclass(frozen=True)
class MultirateProfile:
    path: Path
    training: dict[str, Any]
    deepspeed: dict[str, Any]


def load_multirate_profile(path: str | Path) -> MultirateProfile:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"multirate profiles must be YAML files: {resolved}")
    raw = _load_yaml(resolved)
    if raw.get("schema_version") == 2 or raw.get("baseline") == "absolute_motion_v2":
        raise ValueError(
            "absolute-motion v2 profiles stay on scripts/train.py; "
            f"{resolved.name} is not a multirate profile"
        )
    top = require_exact_keys(
        raw,
        {
            "schema_version",
            "baseline",
            "phase",
            "paths",
            "pairing",
            "training",
            "loss",
            "flow_matching",
            "model",
            "distributed",
        },
        "multirate profile",
    )
    if top["schema_version"] != MULTIRATE_SCHEMA_VERSION:
        raise ValueError(
            f"multirate schema_version must be {MULTIRATE_SCHEMA_VERSION}"
        )
    if top["baseline"] not in MULTIRATE_BASELINES:
        raise ValueError(f"multirate baseline must be {MULTIRATE_BASELINE}")
    phase = str(top["phase"])
    if phase not in MULTIRATE_PHASES:
        raise ValueError(f"multirate phase must be one of {sorted(MULTIRATE_PHASES)}")

    paths_raw = dict(top["paths"])
    teacher_checkpoint = paths_raw.pop("native_teacher_checkpoint", "")
    paths = require_exact_keys(
        paths_raw,
        {
            "project_root",
            "packed_dataset",
            "source_dataset",
            "wan_root",
            "training_runs",
            "init_checkpoint",
        },
        "multirate paths",
    )
    project_root = Path(
        _resolve(paths["project_root"], base=resolved.parent, label="paths.project_root")
    )
    resolved_paths = {
        key: _resolve(value, base=project_root, label=f"paths.{key}")
        for key, value in paths.items()
        if key != "project_root"
    }
    resolved_paths["project_root"] = str(project_root)

    optional_pair = {"partition_seed": 0, "validation_fraction": .1, "clock_mode": "required",
                     "samples_per_episode": 10, "stress_probability": 0.0, "sampler_seed": None,
                     "age_weighting": "schedule_mixture", "cache_age_actions": None,
                     "joint_supervision": False, "cache_age_jitter": False, "cache_age_seed": 0}
    pair_raw = dict(top["pairing"])
    pair_options = {key: pair_raw.pop(key, default) for key, default in optional_pair.items()}
    if pair_options["sampler_seed"] is None:
        pair_options["sampler_seed"] = int(pair_options["partition_seed"])
        if phase == "cross_time_joint":
            pair_options["sampler_seed"] += JOINT_SAMPLER_SEED_SALT
    else:
        pair_options["sampler_seed"] = int(pair_options["sampler_seed"])
    if pair_options["clock_mode"] not in {"required", "action_only"}:
        raise ValueError("clock_mode must be required or action_only")
    if not 0 < float(pair_options["validation_fraction"]) < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if not 0 <= float(pair_options["stress_probability"]) < 1:
        raise ValueError("stress_probability must be in [0,1)")
    if pair_options["age_weighting"] not in {"schedule_mixture", "uniform"}:
        raise ValueError("age_weighting must be schedule_mixture or uniform")
    pairing = require_exact_keys(
        pair_raw,
        {
            "splits",
            "execute_horizon",
            "every_fast_queries",
            "stress_offsets",
            "seconds_per_frame",
        },
        "multirate pairing",
    )
    training_raw = dict(top["training"])
    extra_training = {k: training_raw.pop(k, v) for k, v in {
        "allow_pretrained_dataset_transfer": False, "fast_supervision": "executed_prefix",
        "max_bridge_age_seconds": None, "initialization_seed": 20260930,
        "feedback_phase": "", "feedback_initialize": False, "feedback_world_refresh_actions": 16,
        "native_phase": "", "native_fast_steps": 10,
        "native_expert_weight": 1.0, "native_fm_anchor_weight": 0.0,
        "rtc_prefix_max": 0,
        "native_source_bridge": "immutable_official", "gradient_checkpointing": False, "initialize_joint_interface": False,
    }.items()}
    if extra_training["fast_supervision"] not in {"executed_prefix", "full_chunk"}:
        raise ValueError("invalid fast_supervision")
    ttl = extra_training["max_bridge_age_seconds"]
    if ttl is not None and (not 0 < float(ttl) < float("inf")):
        raise ValueError("max_bridge_age_seconds must be finite and positive")
    validation_batches = int(training_raw.pop("validation_batches", -1))
    training = require_exact_keys(
        training_raw,
        {
            "name",
            "device",
            "log_level",
            "max_steps",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "action_learning_rate",
            "fast_learning_rate",
            "video_learning_rate",
            "weight_decay",
            "min_lr_ratio",
            "warmup_steps",
            "grad_clip_norm",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "log_interval",
            "checkpoint_interval",
            "checkpoint_total_limit",
            "resume_from",
            "null_bridge_probability",
            "max_open_shards",
        },
        "multirate training",
    )
    if pair_options["cache_age_jitter"] and int(training["num_workers"]) > 0 and bool(training["persistent_workers"]):
        raise ValueError("epoch cache-age jitter requires num_workers=0 or nonpersistent workers")
    training["validation_batches"] = validation_batches
    training.update(extra_training)
    native = top["baseline"] == "world_action_multirate_native_v1"
    if native != bool(training["native_phase"]):
        raise ValueError("native baseline requires an explicit native_phase")
    if not native and int(training["rtc_prefix_max"]) != 0:
        raise ValueError("RTC prefix conditioning currently requires the native Action DiT path")
    if native:
        if not 0 <= int(training["rtc_prefix_max"]) <= int(top["model"]["action_expert"]["chunk_size"]) - int(pairing["execute_horizon"]):
            raise ValueError("RTC prefix must leave a complete executable suffix")
        if int(training["rtc_prefix_max"]) and training["native_phase"] != "adapt":
            raise ValueError("RTC endpoint distillation requires a prefix-conditioned teacher; use adaptation first")
        if training["native_phase"] not in {"adapt", "distill"}:
            raise ValueError("native_phase must be adapt or distill")
        if training["native_source_bridge"] not in {"immutable_official", "student_joint"}:
            raise ValueError("unsupported native_source_bridge")
        if training["feedback_phase"]:
            raise ValueError("native phases cannot mix with compact feedback")
        if phase != "fast_warmup" and training["native_source_bridge"] != "student_joint":
            raise ValueError("joint native training requires student_joint bridge")
        if training["initialize_joint_interface"] and (training["native_source_bridge"] != "student_joint" or phase != "fast_warmup"):
            raise ValueError("V1 to V2 initialization belongs only to student warmup")
        if training["native_source_bridge"] == "student_joint":
            if training["native_phase"] != "adapt":
                raise ValueError("joint bridge adaptation keeps the native ten-step solver")
            if not pair_options["joint_supervision"]:
                raise ValueError("student_joint requires causal source RGB and explicit joint_supervision")
        if int(training["native_fast_steps"]) not in {1, 2, 4, 10}:
            raise ValueError("native fast sampler supports 1, 2, 4 or 10 steps")
        if training["native_phase"] == "adapt" and int(training["native_fast_steps"]) != 10:
            raise ValueError("native adaptation retains the 10-step quality reference")
        if training["max_bridge_age_seconds"] is not None:
            raise ValueError("native age coverage is the complete refresh cycle; do not silently mask by TTL")
        if (int(pairing["execute_horizon"]), int(pairing["every_fast_queries"])) != (4, 4):
            raise ValueError("native controlled-ratio training requires H=4 and R=4")
        if any(float(training[k]) < 0 for k in ("native_expert_weight", "native_fm_anchor_weight")):
            raise ValueError("native objective weights must be nonnegative")
    if training['feedback_phase'] not in {'','dynamics','policy'}:
        raise ValueError('invalid feedback training phase')
    if int(training['feedback_world_refresh_actions']) < 16:
        raise ValueError('heavy world refresh budget must cover at least 16 actions')
    loss_raw = dict(top['loss'])
    native_teacher_weight = float(loss_raw.pop('native_teacher_weight', 0.0))
    if native_teacher_weight < 0:
        raise ValueError('native_teacher_weight must be nonnegative')
    loss = require_exact_keys(
        loss_raw,
        {"fast_weight", "action_weight", "video_weight"},
        "multirate loss",
    )
    if training['feedback_phase'] and (str(top['phase']) != 'fast_warmup' or float(loss['action_weight']) != 0 or float(loss['video_weight']) != 0):
        raise ValueError('feedback phases freeze the original backbone and use separate objectives')
    flow = require_exact_keys(
        top["flow_matching"],
        {"shift", "sigma_min", "extra_one_step", "training_timesteps"},
        "multirate flow_matching",
    )
    model_raw = dict(top["model"])
    compact_fast = model_raw.pop("compact_fast", None)
    encoder_weights = model_raw.pop("encoder_weights", "")
    if (top["baseline"] == "world_action_multirate_v2") != (compact_fast is not None):
        raise ValueError("v2 profile requires compact_fast; v1 must not contain it")
    if compact_fast is not None:
        from dynamicwam.models.compact_fast_policy import CompactFastConfig
        compact_fast = dict(compact_fast)
        compact_fast["world_layers"] = tuple(compact_fast["world_layers"])
        CompactFastConfig(**compact_fast)
        if compact_fast.get("encoder") == "resnet18" and not encoder_weights:
            raise ValueError("resnet18 training requires explicit offline encoder_weights")
    model = require_exact_keys(
        model_raw,
        {
            "future_video_size",
            "global_downsample_rate",
            "compact_wan",
            "action_expert",
            "fast_conditioner",
        },
        "multirate model",
    )
    compact = require_exact_keys(
        model["compact_wan"],
        {
            "precision",
            "dim",
            "ffn_dim",
            "num_heads",
            "num_layers",
            "head_dim",
            "hidden_anchor_layers",
            "motion_anchor_layers",
            "teacher_layer_mapping",
        },
        "multirate compact_wan",
    )
    action = require_exact_keys(
        model["action_expert"],
        {"dim", "ffn_dim", "num_layers", "chunk_size", "state_dim", "action_dim"},
        "multirate action_expert",
    )
    fast_raw = dict(model["fast_conditioner"])
    world_defaults = {"world_aware": False, "world_layers": [3, 7, 11],
                      "world_heads": 16, "world_head_dim": 128, "world_queries": 16,
                      "world_dim": 192, "encoder": "compact_gn"}
    world_options = {k: fast_raw.pop(k, v) for k, v in world_defaults.items()}
    motion_options = {k: fast_raw.pop(k, v) for k, v in {
        "motion_kv": False, "kv_rank": 8, "max_relative_update": .25,
        "interface_version": 1, "action_dim": 14, "correction_layers": None,
        "require_source_rgb": True,
    }.items()}
    fast = require_exact_keys(
        fast_raw,
        {
            "dim",
            "layers",
            "heads",
            "head_dim",
            "text_dim",
            "state_dim",
            "sim_scale",
            "action_scale",
        },
        "multirate fast_conditioner",
    )
    if native:
        if not world_options["world_aware"]:
            raise ValueError("native multirate requires the world-aware observation router")
        if not world_options["world_layers"] or min(world_options["world_layers"]) < 0 or max(world_options["world_layers"]) >= int(compact["num_layers"]):
            raise ValueError("native router selects unavailable video layers")
        if (int(world_options["world_heads"]), int(world_options["world_head_dim"])) != (int(compact["num_heads"]), int(compact["head_dim"])):
            raise ValueError("native router world head layout differs from video expert")
        fast.update(world_options)
        if motion_options["motion_kv"]:
            if int(motion_options["kv_rank"]) < 1 or not 0 < float(motion_options["max_relative_update"]) <= 1:
                raise ValueError("invalid motion K/V correction bounds")
            if int(motion_options["interface_version"]) == 1:
                for name in ("interface_version", "action_dim", "correction_layers", "require_source_rgb"):
                    motion_options.pop(name)
            elif int(motion_options["interface_version"]) != 2:
                raise ValueError("unsupported motion K/V interface version")
            elif int(motion_options["action_dim"]) != int(action["action_dim"]):
                raise ValueError("joint interface plan action dimension differs from actor")
            fast.update(motion_options)
    elif world_options["world_aware"] or motion_options["motion_kv"]:
        raise ValueError("world-aware conditioner requires the native baseline")
    distributed = require_exact_keys(
        top["distributed"],
        {"zero_stage", "bf16", "steps_per_print"},
        "multirate distributed",
    )
    if int(action["num_layers"]) != int(compact["num_layers"]):
        raise ValueError("action expert layers must match compact WAN layers")
    if int(fast["layers"]) != int(action["num_layers"]):
        raise ValueError("fast conditioner layers must match the action expert")
    if int(fast["heads"]) * int(fast["head_dim"]) != int(fast["dim"]):
        raise ValueError("fast conditioner heads do not match its width")
    if not 1 <= int(pairing["execute_horizon"]) <= int(action["chunk_size"]):
        raise ValueError("execute horizon must be in [1, action chunk]")
    if int(pairing["every_fast_queries"]) < 1 or int(pair_options["samples_per_episode"]) < 1:
        raise ValueError("refresh ratio and per-episode sampling quota must be positive")
    if int(fast["dim"]) != int(action["dim"]) or int(fast["state_dim"]) != int(action["state_dim"]):
        raise ValueError("reference conditioner width/state must match the action expert")
    if float(fast["sim_scale"]) <= 0 or float(fast["action_scale"]) <= 0:
        raise ValueError("age scales must be positive")
    if compact_fast is not None:
        for key in ("state_dim", "action_dim", "chunk_size"):
            if int(compact_fast[key]) != int(action[key]):
                raise ValueError(f"compact_fast {key} differs from action expert")
        if max(compact_fast["world_layers"]) >= int(compact["num_layers"]):
            raise ValueError("compact_fast selects an unavailable world layer")
        if (int(compact_fast["world_heads"]), int(compact_fast["world_head_dim"])) != (int(compact["num_heads"]), int(compact["head_dim"])):
            raise ValueError("compact_fast world head layout differs from WAN")
    if int(distributed["zero_stage"]) not in {0, 1, 2, 3}:
        raise ValueError("unsupported ZeRO stage")
    if phase == "fast_warmup" and float(loss["video_weight"]) != 0.0:
        raise ValueError("fast_warmup does not train the video regularizer")
    if float(training["null_bridge_probability"]) < 0.0 or float(
        training["null_bridge_probability"]
    ) >= 1.0:
        raise ValueError("null_bridge_probability must be in [0, 1)")

    if native_teacher_weight > 0:
        if compact_fast is None and not native:
            raise ValueError('native teacher loss requires an independent compact student')
        if phase == 'fast_warmup' and (float(loss['action_weight']) != 0 or float(loss['video_weight']) != 0):
            raise ValueError('frozen native teacher uses only expert fast loss + teacher distillation')
        if phase == 'cross_time_joint' and not teacher_checkpoint:
            raise ValueError('joint distillation requires a separate fixed native_teacher_checkpoint')
        if phase == 'cross_time_joint' and (float(loss['action_weight']) <= 0 or float(loss['video_weight']) <= 0):
            raise ValueError('joint distillation requires aligned action and video retention losses')
        loss = dict(loss, native_teacher_weight=native_teacher_weight)
    if native:
        if not teacher_checkpoint or native_teacher_weight <= 0:
            raise ValueError("native training requires an independent fixed teacher")
        if phase == "fast_warmup" and (float(loss["action_weight"]) != 0 or float(loss["video_weight"]) != 0):
            raise ValueError("native warmup freezes video and uses only fast objectives")
        if phase == "cross_time_joint" and (float(loss["action_weight"]) <= 0 or float(loss["video_weight"]) <= 0):
            raise ValueError("joint native training preserves aligned video and action objectives")
        if training["native_source_bridge"] == "student_joint" and int(fast.get("interface_version", 1)) != 2:
            raise ValueError("student_joint requires the complete V2 innovation interface")
    if training['feedback_phase'] and (not compact_fast or not compact_fast.get('feedback_enabled')):
        raise ValueError('feedback training requires a feedback-enabled compact model')
    if compact_fast and compact_fast.get('feedback_enabled') and not training['feedback_phase']:
        raise ValueError('feedback model needs an explicit training phase')
    if training['feedback_phase']=='policy' and (not teacher_checkpoint or native_teacher_weight <= 0):
        raise ValueError('endpoint policy distillation requires an explicit fixed teacher')
    flat_model = {
        "initial_checkpoint": resolved_paths["init_checkpoint"],
        "native_teacher_checkpoint": _resolve(teacher_checkpoint, base=project_root, label="native_teacher_checkpoint") if teacher_checkpoint else "",
        "compact_wan": {
            "checkpoint_path": resolved_paths["wan_root"],
            "config_path": resolved_paths["wan_root"],
            "future_video_size": [int(value) for value in model["future_video_size"]],
            **compact,
        },
        "action_expert": action,
        "fast_conditioner": fast,
        "compact_fast": compact_fast,
        "native_multirate": native,
        "encoder_weights": _resolve(encoder_weights, base=project_root, label="encoder_weights") if encoder_weights else "",
    }
    flat = {
        **training,
        "phase": phase,
        "checkpoint_dir": resolved_paths["training_runs"],
        "flow_matching": flow,
        "loss": loss,
        "model": flat_model,
        "dataset": {
            **pair_options,
            "root": resolved_paths["packed_dataset"],
            "source_root": resolved_paths["source_dataset"],
            "max_open_shards": int(training["max_open_shards"]),
            "splits": [str(value) for value in pairing["splits"]],
            "execute_horizon": int(pairing["execute_horizon"]),
            "every_fast_queries": int(pairing["every_fast_queries"]),
            "stress_offsets": [int(value) for value in pairing["stress_offsets"]],
            "seconds_per_frame": float(pairing["seconds_per_frame"]),
            "global_downsample_rate": int(model["global_downsample_rate"]),
            "chunk_size": int(action["chunk_size"]),
        },
    }
    deepspeed = {
        "zero_optimization": {"stage": int(distributed["zero_stage"]),
                              "stage3_gather_16bit_weights_on_model_save": True},
        "bf16": {"enabled": bool(distributed["bf16"])},
        "gradient_clipping": float(training["grad_clip_norm"]),
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": int(distributed["steps_per_print"]),
    }
    return MultirateProfile(path=resolved, training=flat, deepspeed=deepspeed)
