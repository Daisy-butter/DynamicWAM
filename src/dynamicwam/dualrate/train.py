"""Single-process engineering training entry; formal/DDP training is gated off.

Example: python -m dynamicwam.dualrate.train --config run.json --device cpu
GPU execution additionally requires an externally supervised, expiring, one-use
authorization receipt. This module never launches a GPU guard or a watcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid
from typing import Mapping

import torch
from torch import nn
from torch.utils.data import default_collate

CHECKPOINT_SCHEMA = "dualrate-engineering-checkpoint-v1"
TRAINING_FIELDS = {
    "source_condition_latent", "source_initial_state", "source_motion",
    "source_motion_valid", "source_motion_acceleration_valid", "source_text",
    "source_action", "fast_action", "source_future_latent", "source_future_intervals",
    "source_action_times", "source_known_actions", "source_known_mask",
    "fast_known_actions", "fast_known_mask", "rgb", "rgb_times", "motion",
    "motion_times", "motion_valid", "motion_acceleration_valid", "state", "text", "text_valid", "source_time",
    "completed_time", "query_time", "action_times", "future", "future_loss_valid",
    "contact", "contact_valid", "future_times",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity(source_root: str | Path | None = None) -> str:
    """Hash native and new Python implementation, including uncommitted files."""
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError("empty source tree")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


class EpochCursor:
    """Reconstruct epoch permutation without consuming model/global RNG."""
    def __init__(self, length: int, batch_size: int, seed: int, epoch: int = 0, position: int = 0):
        if length <= 0 or batch_size <= 0 or epoch < 0 or not 0 <= position <= length:
            raise ValueError("invalid sampler cursor")
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.epoch, self.position = epoch, position
        self._order_epoch, self._order = None, None

    def next_indices(self) -> list[int]:
        if self.position == self.length:
            self.epoch += 1
            self.position = 0
        if self._order_epoch != self.epoch:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
            self._order = torch.randperm(self.length, generator=generator)
            self._order_epoch = self.epoch
        result = self._order[self.position:self.position + self.batch_size].tolist()
        self.position += len(result)
        return result

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in ("length", "batch_size", "seed", "epoch", "position")}

    def load_state_dict(self, state: Mapping) -> None:
        for key in ("length", "batch_size", "seed"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"sampler {key} changed on resume")
        restored = EpochCursor(**dict(state))
        self.epoch, self.position = restored.epoch, restored.position
        self._order_epoch, self._order = None, None


def capture_rng(generator: torch.Generator | None = None) -> dict:
    result = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
              "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
              "generator": generator.get_state() if generator is not None else None}
    try:
        import numpy as np
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        result["numpy"] = (kind, keys.tolist(), int(pos), int(has_gauss), float(cached))
    except ImportError:
        result["numpy"] = None
    return result


def restore_rng(state: Mapping, generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA checkpoint cannot silently resume on CPU")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda"]])
    if (state["generator"] is None) != (generator is None):
        raise ValueError("dedicated noise-generator identity changed")
    if generator is not None:
        generator.set_state(state["generator"].cpu())
    if state.get("numpy") is not None:
        import numpy as np
        kind, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


def atomic_checkpoint(path: str | Path, payload: Mapping, *, min_free_bytes: int = 0) -> None:
    """Write a new file then atomic replace; preserve prior checkpoint on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reserve = max(1 << 20, int(_tensor_bytes(payload) * 1.15))
    if shutil.disk_usage(path.parent).free < min_free_bytes + reserve:
        raise OSError("storage floor would be crossed by atomic checkpoint")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(path: str | Path, system: nn.Module, optimizer: torch.optim.Optimizer,
                             step: int, cursor: EpochCursor, identity: Mapping, *,
                             generator: torch.Generator | None = None, min_free_bytes: int = 0) -> None:
    atomic_checkpoint(path, {
        "schema": CHECKPOINT_SCHEMA, "formal_training_ready": False,
        "step": int(step), "identity": dict(identity), "model": system.state_dict(),
        "optimizer": optimizer.state_dict(), "cursor": cursor.state_dict(),
        "rng": capture_rng(generator), "model_training": system.training,
    }, min_free_bytes=min_free_bytes)


def load_training_checkpoint(path: str | Path, system: nn.Module, optimizer: torch.optim.Optimizer,
                             cursor: EpochCursor, identity: Mapping, *,
                             generator: torch.Generator | None = None) -> int:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("identity") != dict(identity):
        raise ValueError("checkpoint schema/source/config/data/runtime identity mismatch")
    if not isinstance(value.get("step"), int) or value["step"] < 0:
        raise ValueError("invalid completed-step count")
    # Check data identity before changing any model or optimizer state.
    checked = EpochCursor(cursor.length, cursor.batch_size, cursor.seed)
    checked.load_state_dict(value["cursor"])
    system.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    cursor.load_state_dict(value["cursor"])
    system.train(bool(value["model_training"]))
    restore_rng(value["rng"], generator)
    return value["step"]


def _validate_storage(config: Mapping) -> Path:
    root_arg, output_arg = Path(config["storage_root"]), Path(config["output"])
    if not root_arg.is_absolute() or not output_arg.is_absolute():
        raise ValueError("storage_root and output must be absolute")
    root, output = root_arg.resolve(strict=True), output_arg.resolve()
    if root.name != "wuruihan" or not output.is_relative_to(root) or output == root:
        raise ValueError("output must be below verified disk's wuruihan directory")
    if os.name != "posix" or root.stat().st_dev == Path.home().stat().st_dev:
        raise ValueError("training artifacts require a verified separate SSD/HDD filesystem")
    return output


def _validate_normalization(config: Mapping, manifest: Mapping) -> None:
    expected = {"native_checkpoint_sha256": config["init_checkpoint_sha256"],
                "action_normalization_sha256": config["action_normalization_sha256"],
                "motion_normalization_sha256": config["motion_normalization_sha256"],
                "action_representation": "checkpoint_mean_std",
                "motion_representation": "raw_absolute_motion_v2"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} does not match native checkpoint contract")
    if manifest.get("split") != "train":
        raise ValueError("training entry only accepts the training split")


def _gpu_authorization(path: Path, identity: Mapping, output: Path) -> None:
    """Validate external supervisor receipt without CUDA/NVML/driver telemetry."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("mode") != "single_process_engineering_smoke" or receipt.get("authorized") is not True:
        raise ValueError("GPU authorization is not for this engineering smoke")
    if receipt.get("identity") != dict(identity) or float(receipt.get("expires_at_unix", 0)) <= time.time():
        raise ValueError("stale or differently bound GPU authorization")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visibility != receipt.get("cuda_visible_devices") or not visibility.startswith("GPU-") or "," in visibility:
        raise ValueError("single UUID-pinned compute GPU required")
    guard = receipt.get("external_guard", {})
    pid = int(guard.get("pid", -1))
    proc = Path(f"/proc/{pid}")
    if os.name != "posix" or pid <= 0 or not proc.exists() or guard.get("mode") != "fds_before_telemetry_v1":
        raise ValueError("external temporary all-adapter guard absent")
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    if str(guard.get("start_ticks")) != fields[19]:
        raise ValueError("guard PID was reused")
    held = {str(fd.resolve()) for fd in (proc / "fd").iterdir()}
    required = set(guard.get("device_paths", []))
    device_nodes = {str(p) for p in Path("/dev").glob("nvidia*") if p.name[6:].isdigit()}
    if not device_nodes or not device_nodes <= required or not required <= held:
        raise ValueError("guard does not hold its declared device handles")
    # Require the guard to supervise this worker, not an unrelated keepalive.
    ancestor = os.getppid()
    while ancestor > 1 and ancestor != pid:
        ancestor_fields = Path(f"/proc/{ancestor}/stat").read_text().rsplit(")", 1)[1].split()
        ancestor = int(ancestor_fields[1])
    if ancestor != pid:
        raise ValueError("device guard must be an ancestor supervisor")
    claim = path.with_suffix(path.suffix + ".consumed")
    storage = Path(receipt.get("storage_root", "/nonexistent")).resolve(strict=True)
    if storage.name != "wuruihan" or not output.is_relative_to(storage) or not claim.resolve().is_relative_to(storage):
        raise ValueError("authorization receipt must live on the declared artifact disk")
    with claim.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "output": str(output), "receipt_sha256": file_sha256(path)}, handle)


def _device_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _device_batch(v, device) for k, v in value.items()}
    raise TypeError("training batches must contain tensors only")


def run(config: dict, *, device: str = "cpu", resume: str | None = None,
        authorization: str | None = None) -> dict:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("DDP/eight-rank support is not accepted; refusing launch")
    if config.get("run_kind") != "single_process_engineering_smoke":
        raise ValueError("formal training is not ready; set explicit engineering-smoke run_kind")
    required = {"architecture", "init_checkpoint", "init_checkpoint_sha256", "action_normalization_sha256",
                "motion_normalization_sha256", "slow_lr", "fast_lr", "manifest", "steps", "batch_size",
                "seed", "output", "storage_root"}
    if required - config.keys():
        raise ValueError(f"missing training config keys: {sorted(required - config.keys())}")
    steps, warmup = int(config["steps"]), int(config.get("warmup_steps", 0))
    if steps <= 0 or not 0 <= warmup < steps or min(float(config["slow_lr"]), float(config["fast_lr"])) <= 0:
        raise ValueError("invalid step/warmup/learning-rate schedule")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("only explicit CPU or CUDA devices supported")
    output = _validate_storage(config)
    manifest_path = Path(config["manifest"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_normalization(config, manifest)
    init_path = Path(config["init_checkpoint"])
    if not init_path.is_absolute() or file_sha256(init_path) != config["init_checkpoint_sha256"]:
        raise ValueError("native initialization checkpoint hash mismatch")
    identity = {"source_sha256": source_identity(), "config_sha256": canonical_hash(config),
                "manifest_sha256": file_sha256(manifest_path), "init_checkpoint_sha256": config["init_checkpoint_sha256"],
                "device_type": target.type, "torch_version": str(torch.__version__)}
    if target.type == "cuda":
        if authorization is None:
            raise ValueError("CUDA requires a fresh external guarded authorization receipt")
        _gpu_authorization(Path(authorization), identity, output)
    from .data import DualRateShardDataset
    from .factory import build_system
    dataset = DualRateShardDataset(manifest_path, verify_hash=True)
    # Fail before model allocation on label-only preparation outputs. Other rows
    # are hash/field checked as consumed, avoiding a multi-terabyte startup scan.
    missing = TRAINING_FIELDS - dataset[0].keys()
    if missing:
        raise ValueError(f"first row is not trainable; missing {sorted(missing)}")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh output directory is not empty; use explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    floor = int(config.get("min_free_bytes", 10 * (1 << 30)))
    save_every = int(config.get("save_every", 1))
    if floor < 0 or save_every <= 0:
        raise ValueError("invalid storage floor/checkpoint interval")
    if shutil.disk_usage(output).free < floor:
        raise OSError("artifact storage is already below configured floor")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    system = build_system(config, target)
    system.train()
    slow_parameters = [p for p in system.slow.parameters() if p.requires_grad]
    slow_ids = {id(p) for p in slow_parameters}
    fast_parameters = [p for p in system.parameters() if p.requires_grad and id(p) not in slow_ids]
    if not slow_parameters or not fast_parameters:
        raise ValueError("joint optimization requires trainable slow and new modules")
    optimizer = torch.optim.AdamW([
        {"params": slow_parameters, "lr": float(config["slow_lr"]), "name": "slow"},
        {"params": fast_parameters, "lr": float(config["fast_lr"]), "name": "new"},
    ], weight_decay=float(config.get("weight_decay", 0.01)))
    cursor = EpochCursor(len(dataset), int(config["batch_size"]), seed)
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    start = load_training_checkpoint(resume, system, optimizer, cursor, identity, generator=generator) if resume else 0
    if start > steps:
        raise ValueError("checkpoint step exceeds configured steps")
    log_path = output / "training.jsonl"
    checkpoint_path = output / "last.pt"
    for step in range(start, steps):
        for parameter in slow_parameters:
            parameter.requires_grad_(step >= warmup)
        rows = [dataset[i] for i in cursor.next_indices()]
        for row in rows:
            if TRAINING_FIELDS - row.keys():
                raise ValueError(f"consumed row missing required fields: {sorted(TRAINING_FIELDS - row.keys())}")
        batch = _device_batch(default_collate(rows), target)
        optimizer.zero_grad(set_to_none=True)
        result = system.training_loss(batch, generator=generator)
        loss = result["loss"]
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite/non-scalar training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(slow_parameters + fast_parameters, float(config.get("max_grad_norm", 1.0)), error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm),
                  "phase": "warmup" if step < warmup else "joint", "formal_training_ready": False}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            save_training_checkpoint(checkpoint_path, system, optimizer, step + 1, cursor, identity,
                                     generator=generator, min_free_bytes=floor)
    return {"completed_steps": steps, "checkpoint": str(checkpoint_path), "identity": identity,
            "formal_training_ready": False, "run_kind": "single_process_engineering_smoke"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--resume")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(config, device=args.device, resume=args.resume, authorization=args.authorization), indent=2))


if __name__ == "__main__":
    main()
