"""CPU audit of actual MotionKV RTC smoke exports and their training log.

Video DiT is constructed on meta solely for the real full-model state schema;
strict assign reload installs the saved CPU tensors. No VAE, teacher weights,
CUDA allocations, forward inference, or claim of controller support is made.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import (
    MULTIRATE_CHECKPOINT_FORMAT, MULTIRATE_CHECKPOINT_VERSION, FAST_ONLY_FORMAT,
    build_fast_conditioner, fast_only_matches,
    load_fast_only_modules, rtc_prefix_contract,
)
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_identity(path):
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def finite_state(state, label):
    require(isinstance(state, dict) and bool(state), f"{label}: empty/malformed state")
    count = 0
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor), f"{label}:{name}: non-tensor state")
        require(tensor.device.type == "cpu", f"{label}:{name}: expected CPU checkpoint tensor")
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), f"{label}:{name}: nonfinite weight/buffer")
        count += tensor.numel()
    return dict(tensors=len(state), elements=count, all_finite=True)


def log_evidence(path, expected_steps):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    rows = {}
    number = r"([+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf|nan))"
    for line in text.splitlines():
        step = re.search(r"\b(\d{1,8})/(\d{1,8})\b", line)
        loss = re.search(r"\bloss\s+" + number, line, re.I)
        grad = re.search(r"\bgrad\s+" + number, line, re.I)
        if step is None or loss is None or grad is None:
            continue
        current, total = map(int, step.groups())
        if total != expected_steps or not 1 <= current <= expected_steps:
            continue
        values = dict(loss=float(loss.group(1)), grad_norm=float(grad.group(1)))
        require(all(math.isfinite(value) for value in values.values()),
                f"nonfinite smoke log metric at optimizer step {current}: {values}")
        require(values["grad_norm"] >= 0, f"negative gradient norm at step {current}")
        rows.setdefault(current, []).append(values)
    require(set(rows) == set(range(1, expected_steps + 1)),
            f"smoke log must contain loss and grad at every step 1..{expected_steps}; found {sorted(rows)}")
    require(any(item["grad_norm"] > 0 for items in rows.values() for item in items),
            "all logged gradient norms are zero; a missing DeepSpeed norm may have been replaced by zero")
    return dict(path=str(path.resolve()), optimizer_steps=sorted(rows), metrics=rows,
        all_logged_loss_and_grad_finite=True,
        precision_note="Human log contains rounded metrics. This checks recorded norms, not every gradient tensor; all-zero norms are rejected because legacy logging may substitute zero for an unavailable norm.")


def build_full_cpu_schema(payload, profile):
    model_config = profile.training["model"]
    wan = model_config["compact_wan"]
    compact_args = {field.name: wan[field.name] for field in fields(CompactWANConfig)}
    compact_args["future_video_size"] = tuple(compact_args["future_video_size"])
    compact_config = CompactWANConfig(**compact_args)
    # No allocations on CUDA; meta avoids allocating another large video model.
    with torch.device("meta"):
        compact = CompactWANModel.from_config(compact_config, device="meta")
    action, motion = payload["modules"]["action_expert"], payload["modules"]["motion"]
    backbone = SmallWAMActionModel(SmallWAMActionConfig(
        compact_wan=compact_config, action_dim=action["action_dim"], state_dim=action["state_dim"],
        chunk_size=action["chunk_size"], ae_dim=action["dim"], ae_ffn_dim=action["ffn_dim"],
        ae_num_layers=action["num_layers"], wan_frozen=True,
        motion_history_count=motion["history_count"], motion_feature_mean=tuple(motion["mean"]),
        motion_feature_scale=tuple(motion["scale"])), compact)
    conditioner = build_fast_conditioner(payload["modules"]["fast_conditioner"])
    schedule = BridgeSchedule.from_flow_match(**payload["modules"]["schedule"])
    return MultirateWAM(backbone, conditioner, schedule, native_multirate=True,
                        fast_steps=payload["modules"]["fast_steps"])


def audit(args, result):
    torch.set_num_threads(2)
    profile = load_multirate_profile(args.config)
    require(int(profile.training["max_steps"]) == args.expected_steps, "smoke profile step count differs")
    result["training_log"] = log_evidence(args.log, args.expected_steps)
    result["log_finite"] = True
    full = torch.load(args.full, map_location="cpu", weights_only=True, mmap=True)
    fast = torch.load(args.fast, map_location="cpu", weights_only=True, mmap=True)
    require(full.get("format") == MULTIRATE_CHECKPOINT_FORMAT, "full export format mismatch")
    require(fast.get("format") == FAST_ONLY_FORMAT, "fast export format mismatch")
    require(full.get("version") == fast.get("version") == MULTIRATE_CHECKPOINT_VERSION,
            "checkpoint format version mismatch")
    require(full.get("global_step") == fast.get("global_step") == args.expected_steps,
            "exports are not the expected completed smoke step")
    require(full.get("modules") == fast.get("modules"), "full/fast module specs differ")
    spec, contract = full["modules"], full["config"]["runtime_contract"]
    require(spec.get("runtime_contract") == contract, "full/config runtime contracts differ")
    rtc = rtc_prefix_contract(full)
    require(rtc is not None and rtc == rtc_prefix_contract(fast), "RTC contract missing/mismatched")
    expected = profile.training["model"]["fast_conditioner"]
    conditioner = spec["fast_conditioner"]
    require(conditioner.get("motion_kv") is True, "smoke export is not MotionKV")
    for field in ("kv_rank", "max_relative_update", "dim", "layers", "world_heads", "world_head_dim", "world_dim"):
        require(conditioner.get(field) == expected.get(field), f"MotionKV {field} differs from smoke profile")
    require(rtc["max_prefix_actions"] == int(profile.training["rtc_prefix_max"]), "RTC prefix limit differs")
    require(spec["fast_steps"] == int(profile.training["native_fast_steps"]), "fast solver steps differ")
    require(contract["motion_kv"]["rank"] == conditioner["kv_rank"], "KV rank contract mismatch")
    require(contract["motion_kv"]["max_relative_update"] == conditioner["max_relative_update"], "KV bound contract mismatch")
    require(contract["native_multirate"]["phase"] == profile.training["native_phase"] == "adapt",
            "smoke must be the trained adaptation recipe")
    require(spec["fast_schedule"] == BridgeSchedule.from_flow_match(**spec["schedule"]).with_action_steps(spec["fast_steps"]).to_config(),
            "serialized fast schedule differs")
    require(full["config"]["training_recipe"]["training"]["max_steps"] == args.expected_steps,
            "training recipe was not the smoke budget")
    result["hyperparameters"] = dict(global_step=full["global_step"], native_fast_steps=spec["fast_steps"],
        rtc_prefix_max=rtc["max_prefix_actions"], kv_rank=conditioner["kv_rank"],
        max_relative_update=conditioner["max_relative_update"], profile=str(args.config.resolve()))
    result["contracts_verified"] = True
    result["full_weights"] = finite_state(full["model"], "full")
    result["fast_weights"] = finite_state(fast["model"], "fast")
    result["weights_finite"] = True
    require(fast_only_matches(full["model"], fast), "fast export tensors differ from full export")
    result["full_fast_tensors_equal"] = True
    delta_keys = [key for key in full["model"] if key.startswith(("fast_conditioner.delta_k.", "fast_conditioner.delta_v."))]
    require(bool(delta_keys) and any(bool(torch.count_nonzero(full["model"][key])) for key in delta_keys),
            "KV correction output projections remained entirely zero after smoke updates")
    result["learned_kv_delta_nonzero"] = True

    full_model = build_full_cpu_schema(full, profile)
    loaded = full_model.load_state_dict(full["model"], strict=True, assign=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, "full strict reload reported incompatible keys")
    require(all(p.device.type == "cpu" for p in full_model.parameters()), "full reload left non-CPU parameters")
    full_state = full_model.state_dict()
    require(set(full_state) == set(full["model"]), "full schema lost state keys")
    require(all(torch.equal(value, full["model"][key]) for key, value in full_state.items()),
            "full strict reload changed stored tensors")
    result["full_strict_reload"] = dict(passed=True, state_keys=len(full_state),
        schema="real MultirateWAM/SmallWAMActionModel/CompactWANModel; Video DiT meta construction then strict CPU assign")
    del full_state, full_model
    gc.collect()
    fast_modules = load_fast_only_modules(fast)
    for prefix, module in (("backbone.action_expert.", fast_modules.action_expert),
                           ("backbone.absolute_motion_tokens.", fast_modules.absolute_motion_tokens),
                           ("fast_conditioner.", fast_modules.fast_conditioner)):
        require(all(torch.equal(value, fast["model"][prefix + key].to(value.dtype))
                    for key, value in module.state_dict().items()), "fast strict reload changed stored values")
    result["fast_strict_reload"] = dict(passed=True, modules=["action_expert", "absolute_motion_tokens", "fast_conditioner"])
    # BF16 training can round buffers as well as parameters; compare metadata
    # through the actual saved storage dtype, and separately require the native
    # and new modules to restore exactly the same normalization buffers.
    expected_mean = torch.tensor(spec["motion"]["mean"],
        dtype=fast["model"]["fast_conditioner.motion_mean"].dtype).to(fast_modules.fast_conditioner.motion_mean.dtype)
    expected_scale = torch.tensor(spec["motion"]["scale"],
        dtype=fast["model"]["fast_conditioner.motion_scale"].dtype).to(fast_modules.fast_conditioner.motion_scale.dtype)
    require(torch.equal(fast_modules.fast_conditioner.motion_mean, expected_mean), "MotionKV motion mean differs from native metadata")
    require(torch.equal(fast_modules.fast_conditioner.motion_scale, expected_scale), "MotionKV motion scale differs from native metadata")
    require(torch.equal(fast_modules.fast_conditioner.motion_mean, fast_modules.absolute_motion_tokens.feature_mean),
            "MotionKV and native action motion mean buffers disagree")
    require(torch.equal(fast_modules.fast_conditioner.motion_scale, fast_modules.absolute_motion_tokens.feature_scale),
            "MotionKV and native action motion scale buffers disagree")
    result["motion_statistics_match_native"] = True
    result["checkpoint_reload"] = True
    result["artifacts"] = dict(full=file_identity(args.full), fast=file_identity(args.fast), log=file_identity(args.log))
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multirate/motionkv_rtc_smoke_1gpu.yaml")
    parser.add_argument("--expected-steps", type=int, default=4)
    args = parser.parse_args()
    result = dict(passed=False, checkpoint_reload=False, full_fast_tensors_equal=False,
        contracts_verified=False, weights_finite=False, log_finite=False,
        device="cpu", scope="smoke export and recorded optimizer-step validation; no RTC controller/quality claim")
    try:
        require(args.expected_steps > 0, "expected steps must be positive")
        audit(args, result)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("passed", "checkpoint_reload", "contracts_verified", "weights_finite", "log_finite")}
                     | {"output": str(args.output.resolve()), "error": result.get("error")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
