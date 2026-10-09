"""v2 migration, strict multirate load, and fast-only export."""

from __future__ import annotations

import sys
import tempfile

import torch
import torch.nn as nn

from dynamicwam.absolute_motion import ABSOLUTE_MOTION_CHECKPOINT_VERSION
from dynamicwam.config import load_profile
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import (
    FAST_ONLY_FORMAT,
    MULTIRATE_CHECKPOINT_FORMAT,
    MULTIRATE_CHECKPOINT_VERSION,
    V2_CHECKPOINT_FORMAT,
    build_fast_only_modules,
    export_fast_only_payload,
    fast_only_matches,
    install_full_checkpoint,
    load_fast_only_modules,
)


def _tiny_spec() -> dict:
    mean = [0.1] * 12
    scale = [1.0] * 12
    return {
        "action_expert": {
            "dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "state_dim": 4,
            "action_dim": 4,
            "chunk_size": 2,
        },
        "wan_attention": {"dim": 16, "num_heads": 2, "head_dim": 8},
        "motion": {"history_count": 1, "mean": mean, "scale": scale},
        "fast_conditioner": {
            "dim": 16,
            "layers": 1,
            "heads": 4,
            "head_dim": 4,
            "text_dim": 8,
            "state_dim": 4,
            "sim_scale": 1.0,
            "action_scale": 16.0,
        },
    }


class _V2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert = nn.Linear(2, 2)
        self.absolute_motion_tokens = nn.Linear(2, 2)
        self.compact_wan = nn.Linear(2, 2)


class _Multi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.action_expert = nn.Linear(2, 2)
        self.backbone.absolute_motion_tokens = nn.Linear(2, 2)
        self.backbone.compact_wan = nn.Linear(2, 2)
        self.fast_conditioner = nn.Linear(2, 2)


def main() -> None:
    repo = __file__
    from pathlib import Path

    root = Path(repo).resolve().parents[1]
    multirate_path = root / "configs" / "world_action_multirate_v1.yaml"
    v2_path = root / "configs" / "absolute_motion_v2.yaml"
    try:
        load_profile(multirate_path)
    except ValueError:
        pass
    else:
        raise SystemExit("the v2 loader accepted a multirate profile")
    try:
        load_multirate_profile(v2_path)
    except ValueError:
        pass
    else:
        raise SystemExit("the multirate loader accepted the v2 profile")
    # The live run profile may move between phases. Verify both explicitly,
    # instead of forcing the user's current experiment back to warmup.
    import yaml
    raw = yaml.safe_load(multirate_path.read_text())
    with tempfile.TemporaryDirectory() as directory:
        for phase in ('fast_warmup', 'cross_time_joint'):
            raw['phase'] = phase
            raw['loss']['video_weight'] = 0.0 if phase == 'fast_warmup' else 0.001
            path = Path(directory) / 'profile.yaml'
            path.write_text(yaml.safe_dump(raw))
            profile = load_multirate_profile(path)
            if profile.training['phase'] != phase:
                raise SystemExit(f'multirate profile did not resolve {phase}')
            if profile.deepspeed['zero_optimization']['stage'] != 0:
                raise SystemExit('multirate DeepSpeed stage was not 0')
    load_profile(v2_path)

    v2 = _V2()
    with torch.no_grad():
        v2.action_expert.weight.fill_(0.25)
        v2.compact_wan.weight.fill_(0.5)
    multi = _Multi()
    fast_before = multi.fast_conditioner.weight.detach().clone()
    kind = install_full_checkpoint(
        multi,
        {
            "format": V2_CHECKPOINT_FORMAT,
            "version": ABSOLUTE_MOTION_CHECKPOINT_VERSION,
            "model": v2.state_dict(),
        },
        allow_v2_init=True,
    )
    if kind != "v2":
        raise SystemExit(f"migration reported {kind}")
    if not torch.equal(multi.backbone.action_expert.weight, v2.action_expert.weight):
        raise SystemExit("v2 action weights were not copied onto the backbone")
    if not torch.equal(multi.backbone.compact_wan.weight, v2.compact_wan.weight):
        raise SystemExit("v2 world weights were not copied")
    if not torch.equal(multi.fast_conditioner.weight, fast_before):
        raise SystemExit("v2 migration replaced the fresh fast conditioner")
    try:
        install_full_checkpoint(
            _Multi(),
            {
                "format": V2_CHECKPOINT_FORMAT,
                "version": ABSOLUTE_MOTION_CHECKPOINT_VERSION,
                "model": v2.state_dict(),
            },
            allow_v2_init=False,
        )
    except RuntimeError:
        pass
    else:
        raise SystemExit("deployment accepted a v2 checkpoint")

    full_state = {key: value.detach().cpu() for key, value in multi.state_dict().items()}
    reloaded = _Multi()
    install_full_checkpoint(
        reloaded,
        {
            "format": MULTIRATE_CHECKPOINT_FORMAT,
            "version": MULTIRATE_CHECKPOINT_VERSION,
            "model": full_state,
        },
        allow_v2_init=False,
    )
    if not torch.equal(reloaded.backbone.compact_wan.weight, multi.backbone.compact_wan.weight):
        raise SystemExit("strict multirate reload changed a backbone tensor")

    spec = _tiny_spec()
    modules = build_fast_only_modules(spec)
    fast_state = {}
    for prefix, module in (
        ("backbone.action_expert.", modules.action_expert),
        ("backbone.absolute_motion_tokens.", modules.absolute_motion_tokens),
        ("fast_conditioner.", modules.fast_conditioner),
    ):
        for key, value in module.state_dict().items():
            fast_state[prefix + key] = value.detach().cpu()
    fast_state["backbone.compact_wan.weight"] = torch.zeros(2, 2)
    payload = export_fast_only_payload(fast_state, module_spec=spec, global_step=3)
    if payload["format"] != FAST_ONLY_FORMAT:
        raise SystemExit("fast-only format was not written")
    if any("compact_wan" in key for key in payload["model"]):
        raise SystemExit("fast-only export kept a world tensor")
    if not fast_only_matches(fast_state, payload):
        raise SystemExit("fast-only view does not match the full fast tensors")
    loaded = load_fast_only_modules(payload)
    left = modules.fast_conditioner.state_dict()
    right = loaded.fast_conditioner.state_dict()
    if not torch.equal(left["blocks.0.out_proj.weight"], right["blocks.0.out_proj.weight"]):
        raise SystemExit("fast-only reload changed the output projection")
    print("verified multirate checkpoint")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_multirate_checkpoint failed: {exc}", file=sys.stderr)
        raise
