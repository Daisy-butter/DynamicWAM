"""Load and export multirate checkpoints without relaxing the v2 loader.

A v2 checkpoint can initialize the shared backbone. New fast-conditioner keys
are filled from the constructed module, then the full state is loaded strictly.
Deployment of a finished run requires the multirate format. The fast-only view
contains the action expert, motion tokens, and fast conditioner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import torch
import torch.nn as nn

from dynamicwam.absolute_motion import ABSOLUTE_MOTION_CHECKPOINT_VERSION
from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
from dynamicwam.models.action_expert import ActionExpert, ActionExpertConfig
from dynamicwam.models.fast_conditioner import FastConditioner


def build_fast_conditioner(spec):
    if spec.get('motion_kv', False):
        from dynamicwam.models.motion_kv_conditioner import MotionKVConditioner
        return MotionKVConditioner(**spec)
    if spec.get('world_aware', False):
        from dynamicwam.models.world_aware_conditioner import WorldAwareConditioner
        return WorldAwareConditioner(**spec)
    return FastConditioner(**spec)


def rtc_prefix_contract(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Read RTC metadata without loading weights or granting deployment access.

    Full and fast-only exports both carry the contract in ``modules``. A full
    export additionally carries the identical contract in ``config``; reject a
    missing/mismatched RTC entry instead of treating it as a legacy checkpoint.
    """
    modules = payload.get("modules", {})
    if not isinstance(modules, dict):
        raise RuntimeError("checkpoint modules must be a mapping")
    module_contract = modules.get("runtime_contract")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint config must be a mapping")
    config_contract = config.get("runtime_contract")
    for contract in (module_contract, config_contract):
        if contract is not None and not isinstance(contract, dict):
            raise RuntimeError("runtime contract must be a mapping")
    module_rtc = (module_contract or {}).get("rtc")
    config_rtc = (config_contract or {}).get("rtc")
    if config_contract is not None and config_rtc != module_rtc:
        raise RuntimeError("module and full-checkpoint RTC contracts disagree")
    recipe = config.get("training_recipe", {}).get("training", {})
    if int(recipe.get("rtc_prefix_max", 0)) > 0 and module_rtc is None:
        raise RuntimeError("RTC training recipe is missing its deployment contract")
    if module_rtc is None:
        return None
    if not isinstance(module_rtc, dict):
        raise RuntimeError("RTC contract must be a mapping")
    required = {
        "version": 1,
        "mode": "training_time_action_prefix",
        "clean_timestep": 0.0,
        "loss": "per_sample_suffix_mean",
        "includes_zero_prefix": True,
        "queue_clock": "absolute_action_effect_times",
        "overflow": "reject_and_refresh",
    }
    if any(module_rtc.get(key) != value for key, value in required.items()):
        raise RuntimeError("unsupported RTC prefix training/deployment semantics")
    if not modules.get("native_multirate") or modules.get("architecture") != "native":
        raise RuntimeError("RTC prefix requires the native action architecture")
    maximum = module_rtc.get("max_prefix_actions")
    horizon = modules.get("action_expert", {}).get("chunk_size")
    execute = module_rtc.get("execute_suffix_actions")
    if any(type(value) is not int for value in (maximum, horizon, execute)):
        raise RuntimeError("RTC action horizons must be integer metadata")
    if not 1 <= execute < horizon or not 1 <= maximum <= horizon - execute:
        raise RuntimeError("RTC prefix must leave the trained executable suffix")
    if int(recipe.get("rtc_prefix_max", maximum)) != maximum:
        raise RuntimeError("RTC training recipe and exported prefix limit disagree")
    return dict(module_rtc)


def install_immutable_source_backbone(backbone, contract, normalization):
    """The trained executor must not silently replace the slow source expert.

    Native training builds every source bridge with this immutable teacher.
    Loading the student action expert for that forward changes video context
    despite frozen video weights, so deployment validates and restores it.
    """
    native = contract['native_multirate']
    if native.get('source_bridge') != 'immutable_official':
        raise RuntimeError('unknown native bridge provenance')
    identity = native['teacher']
    path = Path(identity['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != identity['sha256']:
        raise RuntimeError('immutable slow model checkpoint hash changed')
    payload = load_checkpoint_payload(path)
    if payload.get('format') != V2_CHECKPOINT_FORMAT:
        raise RuntimeError('native source bridge requires an official WAM export')
    if payload.get('config', {}).get('action_normalization') != normalization:
        raise RuntimeError('immutable slow and trained fast normalization differs')
    backbone.load_state_dict(payload['model'], strict=True)
    backbone.eval().requires_grad_(False)

MULTIRATE_CHECKPOINT_FORMAT = "dynamicwam_multirate_checkpoint"
MULTIRATE_CHECKPOINT_VERSION = 2
FAST_ONLY_FORMAT = "dynamicwam_multirate_fast_only"
V2_CHECKPOINT_FORMAT = "dynamicwam_absolute_motion_checkpoint"
FAST_STATE_PREFIXES = (
    "backbone.action_expert.",
    "backbone.absolute_motion_tokens.",
    "fast_conditioner.",
    "fast_policy.",
)
_WORLD_MARKERS = ("compact_wan", "vae", "text_encoder", ".t5")


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is not a mapping: {path}")
    return payload


def _shapes_match(current: torch.Tensor, incoming: torch.Tensor, key: str) -> None:
    if tuple(current.shape) != tuple(incoming.shape):
        raise RuntimeError(
            f"checkpoint tensor {key} has shape {tuple(incoming.shape)}, "
            f"model has {tuple(current.shape)}"
        )


def install_full_checkpoint(
    model: nn.Module,
    payload: dict[str, Any],
    *,
    allow_v2_init: bool,
) -> str:
    """Load weights strictly. Returns ``v2`` or ``multirate``."""

    fmt = payload.get("format")
    current = model.state_dict()
    if fmt == V2_CHECKPOINT_FORMAT:
        if not allow_v2_init:
            raise RuntimeError(
                "a multirate deployment checkpoint is required; "
                "v2 weights do not include the fast conditioner"
            )
        if payload.get("version") != ABSOLUTE_MOTION_CHECKPOINT_VERSION:
            raise RuntimeError(
                "v2 initialization version mismatch: "
                f"{payload.get('version')!r}"
            )
        incoming = payload.get("model")
        if not isinstance(incoming, dict):
            raise TypeError("v2 checkpoint is missing its model state")
        migrated: dict[str, torch.Tensor] = {}
        for key, value in incoming.items():
            target = f"backbone.{key}"
            if target not in current:
                raise RuntimeError(f"v2 key is not in the multirate model: {key}")
            _shapes_match(current[target], value, target)
            migrated[target] = value
        filled = dict(current)
        filled.update(migrated)
        new_keys = [key for key in filled if key not in migrated]
        unexpected = [
            key for key in new_keys if not key.startswith(("fast_conditioner.", "fast_policy."))
        ]
        if unexpected:
            raise RuntimeError(
                "v2 migration would leave non-fast weights uninitialized: "
                + ", ".join(sorted(unexpected))
            )
        model.load_state_dict(filled, strict=True)
        return "v2"
    if fmt == MULTIRATE_CHECKPOINT_FORMAT:
        if payload.get("version") != MULTIRATE_CHECKPOINT_VERSION:
            raise RuntimeError(
                "multirate checkpoint version mismatch: "
                f"{payload.get('version')!r}"
            )
        incoming = payload.get("model")
        if not isinstance(incoming, dict):
            raise TypeError("multirate checkpoint is missing its model state")
        if set(incoming) != set(current):
            missing = sorted(set(current) - set(incoming))
            extra = sorted(set(incoming) - set(current))
            raise RuntimeError(
                "multirate checkpoint does not match the model: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        for key, value in incoming.items():
            _shapes_match(current[key], value, key)
        model.load_state_dict(incoming, strict=True)
        return "multirate"
    raise RuntimeError(f"unsupported checkpoint format: {fmt!r}")


def fast_only_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("fast_policy.",) if any(k.startswith("fast_policy.") for k in state) else FAST_STATE_PREFIXES
    selected = {
        key: value.detach().cpu()
        for key, value in state.items()
        if key.startswith(prefixes)
    }
    if not any(key.startswith(("fast_conditioner.", "fast_policy.")) for key in selected):
        raise RuntimeError("fast-only export is missing the fast conditioner")
    if not any(key.startswith(("backbone.action_expert.", "fast_policy.")) for key in selected):
        raise RuntimeError("fast-only export is missing the action expert")
    for key in selected:
        lowered = key.lower()
        if any(marker in lowered for marker in _WORLD_MARKERS):
            raise RuntimeError(f"fast-only export contains a world weight: {key}")
    expected = {key for key in state if key.startswith(prefixes)}
    if set(selected) != expected:
        raise RuntimeError("fast-only export dropped a fast weight")
    return selected


def fast_only_matches(
    full_state: dict[str, torch.Tensor],
    fast_payload: dict[str, Any],
) -> bool:
    exported = fast_only_state(full_state)
    incoming = fast_payload.get("model")
    if set(exported) != set(incoming):
        return False
    for key, value in exported.items():
        if not torch.equal(value, incoming[key].detach().cpu()):
            return False
    return True


def export_fast_only_payload(
    state: dict[str, torch.Tensor],
    *,
    module_spec: dict[str, Any],
    global_step: int,
) -> dict[str, Any]:
    return {
        "format": FAST_ONLY_FORMAT,
        "version": MULTIRATE_CHECKPOINT_VERSION,
        "model": fast_only_state(state),
        "modules": module_spec,
        "global_step": int(global_step),
    }


def assert_fast_only_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if payload.get("format") != FAST_ONLY_FORMAT:
        raise RuntimeError(
            f"expected a fast-only checkpoint, got {payload.get('format')!r}"
        )
    if payload.get("version") != MULTIRATE_CHECKPOINT_VERSION:
        raise RuntimeError("fast-only checkpoint version mismatch")
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        raise RuntimeError("fast-only checkpoint has no tensors")
    for key in model:
        if not key.startswith(FAST_STATE_PREFIXES):
            raise RuntimeError(f"fast-only checkpoint has an unexpected key: {key}")
        lowered = key.lower()
        if any(marker in lowered for marker in _WORLD_MARKERS):
            raise RuntimeError(f"fast-only checkpoint contains a world weight: {key}")
    return model


def build_fast_only_modules(spec: dict[str, Any]) -> nn.ModuleDict:
    """Construct the fast worker modules. This does not build WAN, VAE, or T5."""

    action_spec = spec["action_expert"]
    wan_spec = spec["wan_attention"]
    motion_spec = spec["motion"]
    fast_spec = spec["fast_conditioner"]
    modules = nn.ModuleDict(
        {
            "action_expert": ActionExpert(
                ActionExpertConfig(
                    dim=int(action_spec["dim"]),
                    ffn_dim=int(action_spec["ffn_dim"]),
                    num_layers=int(action_spec["num_layers"]),
                    state_dim=int(action_spec["state_dim"]),
                    action_dim=int(action_spec["action_dim"]),
                    chunk_size=int(action_spec["chunk_size"]),
                ),
                {
                    "dim": int(wan_spec["dim"]),
                    "num_heads": int(wan_spec["num_heads"]),
                    "head_dim": int(wan_spec["head_dim"]),
                },
            ),
            "absolute_motion_tokens": AbsoluteMotionTokenModule(
                dim=int(action_spec["dim"]),
                history_count=int(motion_spec["history_count"]),
                feature_mean=tuple(float(value) for value in motion_spec["mean"]),
                feature_scale=tuple(float(value) for value in motion_spec["scale"]),
            ),
            "fast_conditioner": build_fast_conditioner(fast_spec),
        }
    )
    return modules


def load_fast_only_modules(payload: dict[str, Any]) -> nn.ModuleDict:
    state = assert_fast_only_payload(payload)
    modules = build_fast_only_modules(payload["modules"])
    modules.action_expert.load_state_dict(
        _strip_prefix(state, "backbone.action_expert."),
        strict=True,
    )
    modules.absolute_motion_tokens.load_state_dict(
        _strip_prefix(state, "backbone.absolute_motion_tokens."),
        strict=True,
    )
    modules.fast_conditioner.load_state_dict(
        _strip_prefix(state, "fast_conditioner."),
        strict=True,
    )
    return modules


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise RuntimeError(f"fast-only checkpoint has no tensors under {prefix}")
    return selected


def extract_multirate_backbone(payload: dict) -> dict:
    """Strict full-export to original backbone mapping for a sampler ablation."""
    if payload.get("format") != MULTIRATE_CHECKPOINT_FORMAT or payload.get("version") != MULTIRATE_CHECKPOINT_VERSION:
        raise RuntimeError("backbone ablation requires a full multirate checkpoint")
    state = payload["model"]
    unexpected = [k for k in state if not k.startswith(("backbone.", "fast_conditioner.", "fast_policy."))]
    if unexpected:
        raise RuntimeError(f"unexpected multirate weights: {unexpected[:5]}")
    backbone = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    if not backbone:
        raise RuntimeError("full multirate checkpoint has no backbone")
    return backbone
