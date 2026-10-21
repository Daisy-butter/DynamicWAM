#!/usr/bin/env python3
"""CPU completion gate for actual causal-spatial full/fast training exports.

Uses the production loaders and exact tensor equality. No inference, CUDA,
teacher sampling, driver query, or optimizer execution occurs. State hashes and
serialized categories establish saved-state completeness, not a tested resume.
The lifecycle supervisor separately verifies all workers and device FDs exit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def same(a, b, label):
    require(canonical(a) == canonical(b), label + " differs")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve(strict=True)
    require(p.is_file() and p.stat().st_size > 0, "missing/empty artifact: " + str(p))
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))


def finite_metrics(values, label):
    require(isinstance(values, dict) and bool(values), label + ": no metrics")
    for key, value in values.items():
        require(isinstance(value, (int, float)) and math.isfinite(value), f"{label}: nonfinite/non-numeric {key}")


def audit_ranks(run, recipe):
    expected = {f"rank_{rank}_resources.json" for rank in range(8)}
    found = {p.name for p in Path(run).glob("rank_*_resources.json")}
    require(found == expected, f"require exactly ranks 0..7, found {sorted(found)}")
    result = []
    for rank in range(8):
        path = Path(run) / f"rank_{rank}_resources.json"
        r = read_json(path)
        require(r.get("rank") == rank and r.get("world_size") == 8, "rank/world identity mismatch")
        require(r.get("completed") is True and r.get("error_type") is None, f"rank {rank} did not complete successfully")
        require(r.get("step") == r.get("optimizer_updates") == recipe["steps"], f"rank {rank} optimizer budget incomplete")
        for key in ("nonfinite_loss_count", "nonfinite_grad_count", "oom_count"):
            require(r.get(key) == 0, f"rank {rank}: {key} is nonzero or missing")
        require(r.get("finite_loss_checks", -1) >= recipe["steps"] * recipe["accumulation"], f"rank {rank}: loss check coverage incomplete")
        require(r.get("finite_grad_checks") == recipe["steps"], f"rank {rank}: gradient norm check coverage incomplete")
        norm = r.get("last_grad_norm")
        require(isinstance(norm, (int, float)) and math.isfinite(norm) and norm >= 0, f"rank {rank}: gradient norm unavailable")
        allocated, reserved = r.get("peak_allocated"), r.get("peak_reserved")
        require(isinstance(allocated, int) and isinstance(reserved, int) and 0 < allocated <= reserved,
                f"rank {rank}: invalid memory receipt")
        finite_metrics(r.get("last_metrics"), f"rank {rank}")
        finite_metrics(r.get("gradient_evidence"), f"rank {rank} gradient evidence")
        required_groups = {"video", "observer", "future_and_interfaces"}
        if recipe["phase"] != "interface_adapt":
            required_groups |= {"action", "motion"}
        for group in required_groups:
            require(r["gradient_evidence"].get(group, 0) > 0, f"rank {rank}: no measured gradient in {group}")
        result.append(dict(**r, receipt=identity(path)))
    return result


def audit_logs(run, recipe):
    metrics = [json.loads(line) for line in (Path(run) / "metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = [row["step"] for row in metrics]
    require(steps and steps[0] == 1 and steps[-1] == recipe["steps"], "training log lacks first/final optimizer update")
    require(steps == sorted(set(steps)), "training log steps duplicated or out of order")
    for row in metrics:
        require(row["phase"] == recipe["phase"], "training log phase mismatch")
        finite_metrics({k: v for k, v in row.items() if k != "phase"}, "training log")
    if recipe["diagnostic"]:
        require(steps == [1, 2, 3, 4], "four-update smoke must record every update")
    validation = [json.loads(line) for line in (Path(run) / "validation.jsonl").read_text().splitlines() if line.strip()]
    require(validation and validation[-1]["step"] == recipe["steps"], "completed-step validation is absent")
    for row in validation:
        finite_metrics(row["metrics"], "validation log")
    return dict(training_steps=steps, validation_steps=[v["step"] for v in validation], all_recorded_metrics_finite=True)


def compare_tensor_states(actual, expected, label):
    import torch
    require(set(actual) == set(expected), label + ": state keys differ")
    elements = 0
    for key, value in expected.items():
        other = actual[key]
        require(torch.is_tensor(value) and torch.is_tensor(other), label + ": non-tensor state " + key)
        require(value.device.type == other.device.type == "cpu", label + ": tensor is not on CPU " + key)
        require(value.dtype == other.dtype and value.shape == other.shape, label + ": dtype/shape differs " + key)
        require(torch.equal(value, other), label + ": values differ " + key)
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), label + ": nonfinite weight " + key)
        elements += value.numel()
    return dict(passed=True, tensor_count=len(expected), elements=elements, exact_values=True, all_finite=True)


def audit_data(recipe, payload):
    """Rebind small manifests/stats to actual roots without rehashing all VAE shards."""
    from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
    from dynamicwam.training.data.physical_grid import json_hash
    from dynamicwam.training.data.causal_spatial_dataset import CausalSpatialDataset
    from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
    d = recipe["data"]
    readiness = validate_physical_grid_readiness(d["physical_grid_root"], d["future_latent_root"],
                                                verify_hashes=False, require_full=True)
    grid = read_json(Path(d["physical_grid_root"]) / "manifest.json")
    for key in ("packed_root", "source_root"):
        require(Path(d[key]).resolve() == Path(grid[key]).resolve(), "recipe data root differs: " + key)
    packed = PackedAbsoluteMotionDataset(d["packed_root"], max_open_shards=1)
    saved = payload["data_identity"]
    for key, value in packed.dataset_identity.items():
        same(saved[key], value, "packed identity " + key)
    for key in ("physical_grid_identity", "episode_partition_identity"):
        same(saved[key], readiness[key], key)
    # This is the exact constructor identity formula, without loading samples.
    base_pairing = json_hash(dict(grid=readiness["physical_grid_identity"],
        async_schema="joint_source_current_prefix_v2", max_age=31, async_seed=20261021,
        slow_budget_seconds=2.56, rtc_prefix_max=2))
    class BaseIdentity:
        pairing_identity = base_pairing
    spatial = CausalSpatialDataset(BaseIdentity(), window_size=recipe["model"]["history_length"],
        window_stride_seconds=.32, prefix_length=2, force_fresh_memory=recipe["phase"] == "interface_adapt")
    same(saved["spatial_contract"], spatial.contract, "spatial observation contract")
    same(saved["spatial_pairing_identity"], spatial.pairing_identity, "spatial pairing identity")
    norm = payload["action_normalization"]
    same(norm, payload["native_config"]["action_normalization"], "native action normalization")
    require(norm["stats_sha256"] == packed.dataset_identity["action_stats_sha256"], "action statistics SHA differs")
    stats = read_json(Path(d["packed_root"]) / "action_stats.json")
    stats = stats.get("robotwin_qpos", stats)
    exported = norm["stats"].get("robotwin_qpos", norm["stats"])
    for key in ("mean", "std"):
        same(exported[key], stats[key], "action statistics " + key)
    require(float(norm["epsilon"]) == 1e-6, "normalization epsilon differs from actual dataset loader")
    native_motion = payload["native_config"]["model"]["absolute_motion"]
    for key, native_key in (("mean", "feature_mean"), ("scale", "feature_scale")):
        same(packed.motion_statistics[key], native_motion[native_key], "motion statistics " + key)
    del packed
    return dict(**readiness, normalization_equal=True, spatial_pairing_equal=True,
                scope="manifests/stats hashed and all cache shards stat-checked; full shard hashes belong to launch data audit")


def audit_saved_state(run, recipe, data_identity, full_state):
    import torch
    state_dir = (Path(run) / "state" / f"step_{recipe['steps']}").resolve(strict=True)
    manifest = read_json(state_dir / "state_manifest.json")
    require(manifest["step"] == recipe["steps"], "saved-state manifest step differs")
    files = {}
    for item in manifest["files"]:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe state manifest path")
        path = (state_dir / relative).resolve(strict=True)
        require(path.is_relative_to(state_dir) and path.is_file(), "state file escaped its directory")
        key = relative.as_posix()
        require(key not in files, "duplicate saved-state manifest entry")
        require(path.stat().st_size == item["bytes"] and item["bytes"] > 0, "saved-state file missing/truncated: " + key)
        require(digest(path) == item["sha256"], "saved-state file SHA differs: " + key)
        files[key] = path
    actual = {p.relative_to(state_dir).as_posix() for p in state_dir.rglob("*") if p.is_file() and p.name != "state_manifest.json"}
    require(set(files) == actual, "saved-state manifest omits/adds files")
    progress = read_json(state_dir / "trainer_state.json")
    require(progress["step"] == recipe["steps"], "saved trainer progress step differs")
    for key in ("epoch", "batch_offset"):
        require(type(progress[key]) is int and progress[key] >= 0, "invalid trainer cursor " + key)
    for key, value in (("recipe_sha256", recipe), ("data_identity_sha256", data_identity)):
        expected = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        require(progress[key] == expected, "saved trainer progress identity differs: " + key)
    rng = {int(match.group(1)) for name in files if (match := re.fullmatch(r"random_states_(\d+)\.pkl", name))}
    require(rng == set(range(8)), "saved state lacks exact RNG files for all eight ranks")
    scheduler = [name for name in files if re.fullmatch(r"custom_checkpoint_\d+\.pkl", name) or Path(name).name.startswith("scheduler")]
    require(scheduler, "saved custom LR scheduler state is missing")
    optimizer = [name for name in files if "optim_states" in Path(name).name or Path(name).name.startswith("optimizer")]
    model_files = [name for name in files if name.endswith("model_states.pt")]
    model_equal = []
    embedded_optimizer = False
    for name in model_files:
        value = torch.load(files[name], map_location="cpu", weights_only=True, mmap=True)
        require(isinstance(value, dict) and isinstance(value.get("module"), dict), "unrecognized DeepSpeed model state")
        state = value["module"]
        expected = {"student." + key: tensor for key, tensor in full_state.items()}
        model_equal.append(compare_tensor_states(state, expected, "saved DeepSpeed model " + name))
        embedded_optimizer |= isinstance(value.get("optimizer"), dict) and bool(value["optimizer"])
        require(value.get("global_steps", recipe["steps"]) == recipe["steps"], "DeepSpeed saved step differs")
        del value, state, expected
        gc.collect()
    require(model_files, "actual DeepSpeed model state file is missing")
    require(optimizer or embedded_optimizer, "saved optimizer state is missing")
    return dict(directory=str(state_dir), manifest=identity(state_dir / "state_manifest.json"),
        files=len(files), bytes=sum(path.stat().st_size for path in files.values()),
        rng_ranks=sorted(rng), scheduler_files=scheduler, optimizer_files=optimizer,
        embedded_optimizer=embedded_optimizer, model_equals_export=model_equal,
        progress=progress, resume_execution_tested=False)


def audit(config, result):
    import torch
    from train_causal_spatial import load_recipe
    from dynamicwam.inference.causal_spatial_loader import load_full, load_fast, INIT_SHA, FORMAT, VERSION
    torch.set_num_threads(2)
    recipe = load_recipe(str(config))
    run = Path(recipe["run_dir"]).resolve(strict=True)
    result.update(config=identity(config), run_dir=str(run), phase=recipe["phase"], diagnostic=recipe["diagnostic"])
    same(read_json(run / "recipe.json"), recipe, "saved recipe")
    result["ranks"] = audit_ranks(run, recipe)
    result["logs"] = audit_logs(run, recipe)
    latest = read_json(run / "latest_export.json")
    require(latest.get("final") is True and latest.get("step") == recipe["steps"], "latest export is not final requested step")
    full_path = run / "exports" / f"causal_spatial_step_{recipe['steps']}.pt"
    fast_path = run / "exports" / f"causal_spatial_fast_step_{recipe['steps']}.pt"
    require(Path(latest["path"]).resolve() == full_path and Path(latest["fast_path"]).resolve() == fast_path,
            "latest export paths are not the canonical final phase artifacts")
    result["full_checkpoint"], result["fast_checkpoint"] = identity(full_path), identity(fast_path)
    require(result["full_checkpoint"]["sha256"] == latest["sha256"] and result["full_checkpoint"]["bytes"] == latest["bytes"], "full export SHA/size differs")
    require(result["fast_checkpoint"]["sha256"] == latest["fast_sha256"], "fast export SHA differs")
    model, full = load_full(full_path, "cpu", allow_diagnostic=recipe["diagnostic"])
    require(full["format"] == FORMAT and full["version"] == VERSION, "full format/version differs")
    require(full["global_step"] == recipe["steps"] and full["diagnostic_only"] == recipe["diagnostic"], "export phase step or diagnostic status differs")
    for field, value in (("recipe", recipe), ("model_config", recipe["model"]), ("runtime_contract", recipe["runtime_contract"]),
                         ("data_identity", read_json(run / "data_identity.json"))):
        same(full[field], value, field)
    result["full_strict_reload"] = compare_tensor_states(model.state_dict(), full["model"], "production full loader")
    require(hasattr(model.backbone, "compact_wan"), "full production loader lacks native Video backbone")
    del model
    gc.collect()
    fast_model, fast = load_fast(fast_path, "cpu", allow_diagnostic=recipe["diagnostic"])
    require(not hasattr(fast_model.backbone, "compact_wan"), "fast construction allocated a Video backbone")
    require(fast["source_full_state_format"] == FORMAT and fast["version"] == VERSION, "fast source/version differs")
    for key in full.keys() - {"model", "format"}:
        same(fast[key], full[key], "full/fast metadata " + key)
    expected = {key: value for key, value in full["model"].items() if not key.startswith("backbone.compact_wan.")}
    require(any(key.startswith("backbone.compact_wan.") for key in full["model"]), "full export lacks Video weights")
    result["full_fast_equal"] = compare_tensor_states(fast["model"], expected, "full/fast stored state")
    result["fast_strict_reload"] = compare_tensor_states(fast_model.state_dict(), expected, "production fast loader")
    del expected, fast_model, fast
    gc.collect()
    init = full["initialization"]
    require(init["kind"] == "stage1_video_wo_motion_action_fresh_motion" and init["sha256"] == INIT_SHA,
            "initialization lineage is not the audited Stage1/wo-motion/fresh-motion source")
    require(Path(init["path"]).resolve() == Path(recipe["initial_checkpoint"]).resolve(), "initial checkpoint path differs")
    result["initialization"] = identity(init["path"])
    require(result["initialization"]["sha256"] == INIT_SHA, "actual initialization checkpoint changed")
    initial = torch.load(init["path"], map_location="cpu", weights_only=True, mmap=True)
    same(initial["config"], full["native_config"], "native architecture/normalization lineage")
    del initial
    if recipe.get("continue_from"):
        source = torch.load(recipe["continue_from"], map_location="cpu", weights_only=True, mmap=True)
        require(source.get("format") == FORMAT and source.get("diagnostic_only") is False, "phase continuation uses a diagnostic or wrong-format checkpoint")
        require(source["global_step"] == source["recipe"]["steps"], "source phase is incomplete")
        require({"policy_pretrain": "interface_adapt", "joint": "policy_pretrain"}.get(recipe["phase"]) == source["recipe"]["phase"], "phase order changed")
        for key in ("initialization", "native_config", "data_identity", "model_config"):
            same(source[key], full[key], "continued phase " + key)
        result["continue_from"] = identity(recipe["continue_from"])
        del source
    result["data"] = audit_data(recipe, full)
    result["saved_state"] = audit_saved_state(run, recipe, full["data_identity"], full["model"])
    result["cpu_only"] = True
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = dict(format="causal_spatial_export_audit_v1", passed=False,
                  lifecycle_exit_verified_here=False, resume_execution_tested=False)
    try:
        audit(args.config, result)
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(dict(passed=result["passed"], output=str(args.output.resolve()))), flush=True)


if __name__ == "__main__":
    main()
