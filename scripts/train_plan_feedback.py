#!/usr/bin/env python3
"""Train bounded H4 feedback around the unchanged official 16-action plan.

Run with torchrun. Stage 2 learns the feedback head; stage 3 continues the
same head with --init-feedback and a lower learning rate. Neither stage
updates the original WAM nor distills its sampler to a different step count.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from dynamicwam.absolute_motion import validate_checkpoint_motion_metadata
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import load_checkpoint_payload
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import SourceNoise
from dynamicwam.models.native_wam_teacher import sample_native_wam
from dynamicwam.models.plan_feedback import PlanFeedbackHead, plan_window
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.training.data.multirate_sampling import BalancedPairSampler, ProcessGroupAlignedSampler
from dynamicwam.training.data.paired_world_action_dataset import PairedWorldActionDataset, paired_collate_fn
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel
from dynamicwam.training.train.common import build_training_flow_scheduler
from dynamicwam.training.train.multirate_trainer import prepare_multirate_batch

FORMAT = "dynamicwam_plan_feedback_checkpoint"
VERSION = 1
AGES = (0, 4, 8, 12)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RankSampler(Sampler):
    """Shard an explicitly truncated global draw, with equal rank lengths."""

    def __init__(self, base, rank, world_size):
        self.base, self.rank, self.world_size = base, rank, world_size
        if len(base) % world_size:
            raise ValueError("global sampler must contain complete process rounds")

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base) // self.world_size

    def __iter__(self):
        return itertools.islice(iter(self.base), self.rank, None, self.world_size)


def make_dataset(config, *, validation=False, train=None):
    cfg = config["dataset"]
    kwargs = dict(splits=cfg["splits"], execute_horizon=4, every_fast_queries=4,
                  global_downsample_rate=cfg["global_downsample_rate"], chunk_size=16,
                  seconds_per_frame=cfg["seconds_per_frame"], stress_offsets=(),
                  max_open_shards=cfg["max_open_shards"],
                  subset="validation" if validation else "train",
                  partition_seed=cfg["partition_seed"], validation_fraction=cfg["validation_fraction"],
                  clock_mode=cfg["clock_mode"])
    if train is not None:
        kwargs.update(packed=train.packed, episode_partition=train.episode_partition)
    return PairedWorldActionDataset(cfg["root"], cfg["source_root"], **kwargs)


def make_loader(dataset, config, args, rank, world_size, *, validation=False):
    cfg = config["dataset"]
    base = BalancedPairSampler(dataset.records, samples_per_episode=cfg["samples_per_episode"],
        stress_probability=0., seed=cfg["partition_seed"] if validation else args.seed,
        rank=0, world_size=1, age_weights=None)
    base = ProcessGroupAlignedSampler(base, batch_size=args.batch_size, num_processes=world_size)
    sampler = RankSampler(base, rank, world_size)
    if len(sampler) < args.batch_size:
        raise RuntimeError("dataset cannot supply one complete distributed batch")
    kwargs = dict(batch_size=args.batch_size, sampler=sampler, collate_fn=paired_collate_fn,
                  num_workers=args.workers, pin_memory=True, drop_last=True,
                  generator=torch.Generator().manual_seed(args.seed + rank + 100 * validation))
    if args.workers:
        # Match the existing Linux CPU loader. Forcing spawn serializes millions
        # of immutable PairRecords into each worker and multiplies host memory.
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs), sampler


def build_teacher(config, payload, dataset, device):
    """Original inference precision: WAN BF16, action/motion/solver FP32."""
    if payload.get("format") != "dynamicwam_absolute_motion_checkpoint":
        raise ValueError("feedback requires the immutable official absolute-motion export")
    model_cfg = config["model"]
    wan = model_cfg["compact_wan"]
    motion = validate_checkpoint_motion_metadata(payload["config"]["model"]["absolute_motion"])
    if dataset.packed.action_stats_sha256 != payload["config"]["action_normalization"]["stats_sha256"]:
        raise ValueError("packed actions and official teacher use different normalization")
    if int(motion["history_count"]) != int(dataset.packed.motion_history_count):
        raise ValueError("motion history count differs from official teacher")
    # Physical motion values are normalized by the teacher's original buffers.
    packed_motion = dataset.packed.motion_checkpoint_metadata
    statistics_only = {"feature_mean", "feature_scale", "statistics_sha256"}
    if ({key: value for key, value in packed_motion.items() if key not in statistics_only}
            != {key: value for key, value in motion.items() if key not in statistics_only}):
        raise ValueError("packed and official physical motion semantics differ")
    compact = CompactWANModel.from_config(CompactWANConfig(
        checkpoint_path="", config_path=wan["config_path"], precision="bfloat16",
        dim=wan["dim"], ffn_dim=wan["ffn_dim"], num_heads=wan["num_heads"],
        num_layers=wan["num_layers"], head_dim=wan["head_dim"],
        future_video_size=tuple(wan["future_video_size"]),
        hidden_anchor_layers=list(wan["hidden_anchor_layers"]),
        motion_anchor_layers=list(wan["motion_anchor_layers"]),
        teacher_layer_mapping=list(wan["teacher_layer_mapping"])), device=str(device))
    ae = model_cfg["action_expert"]
    teacher = SmallWAMActionModel(SmallWAMActionConfig(compact_wan=compact.config,
        action_dim=ae["action_dim"], state_dim=ae["state_dim"], chunk_size=16,
        ae_dim=ae["dim"], ae_ffn_dim=ae["ffn_dim"], ae_num_layers=ae["num_layers"],
        wan_frozen=True, motion_history_count=motion["history_count"],
        motion_feature_mean=tuple(motion["feature_mean"]),
        motion_feature_scale=tuple(motion["feature_scale"])), compact)
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.to(device=device).eval().requires_grad_(False)
    if next(teacher.action_expert.parameters()).dtype != torch.float32:
        raise RuntimeError("frozen official action expert must retain FP32 parameters")
    return teacher


def shifted_action_noise(source_noise, ages, *, generator=None):
    """Reuse overlapping noise coordinates; independently sample only the tail.

    Common random numbers reduce a source/current target's sampling variance.
    This is not a guarantee that independently conditioned teacher plans pick
    the same behavioral mode.
    """
    batch, length, dim = source_noise.shape
    if (length, dim) != (16, 14):
        raise ValueError("source noise must be B x 16 x 14")
    ages = ages.to(device=source_noise.device, dtype=torch.long).reshape(batch)
    indices = ages[:, None] + torch.arange(length, device=source_noise.device)[None]
    overlap = source_noise.gather(1, indices.clamp_max(length - 1)[..., None].expand(-1, -1, dim))
    tail = torch.randn(source_noise.shape, device=source_noise.device,
                       dtype=source_noise.dtype, generator=generator)
    return torch.where((indices < length)[..., None], overlap, tail)


@torch.no_grad()
def prepare_targets(batch, teacher, schedule, timesteps, sigmas, device, *, generator=None):
    teacher.eval()
    prepared = prepare_multirate_batch(batch, timesteps=timesteps, sigmas=sigmas,
        device=device, dtype=torch.float32, phase="warmup", null_bridge_probability=0.,
        sim_scale=1., action_scale=16., null_bridge=False, generator=generator)
    ages = batch["delta_actions"].to(device=device, dtype=torch.long)
    # The native deploy VAE/world sampler owns BF16 latents, while the action
    # solver and state stay FP32. Do not use one dtype for both noise branches.
    video_dtype = teacher.compact_wan.video_model.precision
    source = replace(prepared["source"],
        condition_latent=prepared["source"].condition_latent.to(dtype=video_dtype))
    noise = replace(prepared["source_noise"],
        future_latent=prepared["source_noise"].future_latent.to(dtype=video_dtype))
    source_plan = sample_native_wam(teacher, source, noise, schedule).float()
    current = prepared["current"]
    if bool((ages == 0).all()):
        current_plan = source_plan
    else:
        aligned_source = replace(source,
            condition_latent=prepared["aligned"]["condition_latent"].to(dtype=video_dtype),
            state=current.state, motion=current.motion,
            motion_interval_valid_mask=current.motion_interval_valid_mask,
            motion_acceleration_valid_mask=current.motion_acceleration_valid_mask)
        aligned_noise = SourceNoise(future_latent=noise.future_latent,
            actions=shifted_action_noise(noise.actions, ages, generator=generator))
        current_plan = sample_native_wam(teacher, aligned_source, aligned_noise, schedule).float()
    inputs = dict(rgb=current.rgb, rgb_valid_mask=current.rgb_valid_mask,
        state=current.state, source_state=source.state, plan=source_plan,
        age_actions=ages, text=prepared["fast_text"], text_mask=prepared["fast_text_mask"])
    return inputs, plan_window(source_plan, ages), current_plan[:, :4], prepared["clean_actions"][:, :4]


def feedback_objective(residual, base, teacher, expert, ages, *, limit, expert_weight, l2_weight):
    arm = torch.ones(14, device=residual.device, dtype=torch.bool)
    arm[[6, 13]] = False
    active = (ages > 0)[:, None, None] & arm[None, None, :]
    mask = active.expand_as(residual).float()
    denominator = mask.sum().clamp_min(1.)
    teacher_delta, expert_delta = teacher.float() - base.float(), expert.float() - base.float()
    target = teacher_delta.clamp(-limit, limit)
    expert_target = expert_delta.clamp(-limit, limit)
    teacher_loss = ((residual.float() - target).square() * mask).sum() / denominator
    expert_loss = ((residual.float() - expert_target).square() * mask).sum() / denominator
    penalty = (residual.float().square() * mask).sum() / denominator
    loss = teacher_loss + expert_weight * expert_loss + l2_weight * penalty
    corrected = base.float() + residual.float()
    metrics = {
        "loss": loss.detach(), "loss_teacher": teacher_loss.detach(),
        "loss_expert": expert_loss.detach(), "residual_l2": penalty.detach(),
        "teacher_clip_fraction": ((teacher_delta.abs() > limit).float() * mask).sum() / denominator,
        "expert_clip_fraction": ((expert_delta.abs() > limit).float() * mask).sum() / denominator,
        "zero_teacher_mse": (base.float() - teacher.float()).square().mean(),
        "feedback_teacher_mse": (corrected - teacher.float()).square().mean(),
        "zero_expert_mse": (base.float() - expert.float()).square().mean(),
        "feedback_expert_mse": (corrected - expert.float()).square().mean(),
        "current_teacher_expert_mse": (teacher.float() - expert.float()).square().mean(),
        "residual_abs_mean": residual.float().abs().mean(),
        "target_sign_conflict_fraction": (((teacher_delta * expert_delta) < 0).float() * mask).sum() / denominator,
    }
    return loss, metrics


def runtime_contract(schedule, limit):
    return dict(source_plan="immutable_official_full_16", execute_horizon=4,
        refresh_actions=16, ages=list(AGES), age0_correction=False, gripper_correction=False,
        residual_units="normalized_qpos", residual_limit=limit,
        preprocess="three_view_160_imagenet_mask_v1", schedule=schedule.to_config(),
        precision={"video": "bfloat16", "action": "float32", "solver": "float32",
                   "feedback_parameters": "float32", "feedback_autocast": "bfloat16"})


def validate_initialization(payload, *, base_identity, dataset_identity, pairing_identity, contract, spec):
    expected = {"format": FORMAT, "version": VERSION, "base_checkpoint": base_identity,
                "dataset_identity": dataset_identity, "pairing_identity": pairing_identity,
                "runtime_contract": contract, "spec": spec}
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"feedback initialization contract mismatch: {name}")


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(head, loader, sampler, teacher, schedule, timesteps, sigmas, args, device, rank, world_size):
    was_training = head.training
    head.eval()
    sampler.set_epoch(0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 100000 + rank)
    totals, names = None, None
    try:
        for index, batch in enumerate(loader):
            if index >= args.validation_batches:
                break
            inputs, base, target, expert = prepare_targets(batch, teacher, schedule,
                timesteps, sigmas, device, generator=generator)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = head(**inputs)
            if not torch.isfinite(residual).all():
                raise FloatingPointError("nonfinite validation residual")
            if residual.abs().max() > head.residual_limit + 1e-6 or residual[..., [6, 13]].count_nonzero():
                raise RuntimeError("feedback violated the bounded arm-only contract")
            if residual[inputs["age_actions"] == 0].count_nonzero():
                raise RuntimeError("age-zero feedback changed the native first window")
            for row, age in enumerate(inputs["age_actions"].tolist()):
                _, metrics = feedback_objective(residual[row:row+1], base[row:row+1], target[row:row+1],
                    expert[row:row+1], inputs["age_actions"][row:row+1], limit=head.residual_limit,
                    expert_weight=args.expert_weight, l2_weight=args.residual_l2)
                if names is None:
                    names = sorted(metrics)
                    totals = torch.zeros((4, len(names) + 1), device=device, dtype=torch.float64)
                vector = torch.stack([metrics[key].double() for key in names])
                totals[AGES.index(age), :-1] += vector
                totals[AGES.index(age), -1] += 1
        if totals is None:
            raise RuntimeError("feedback validation received no complete batch")
        if world_size > 1:
            dist.all_reduce(totals)
        if any(int(row[-1]) == 0 for row in totals):
            raise RuntimeError("validation did not cover all deployed plan ages")
        all_count = float(totals[:, -1].sum())
        output = {key: float(totals[:, i].sum() / all_count) for i, key in enumerate(names)}
        output["samples"] = int(all_count)
        for age, row in zip(AGES, totals):
            output[f"age_{age}/samples"] = int(row[-1])
            for i, key in enumerate(names):
                output[f"age_{age}/{key}"] = float(row[i] / row[-1])
        output["teacher_gain"] = output["zero_teacher_mse"] - output["feedback_teacher_mse"]
        output["expert_gain"] = output["zero_expert_mse"] - output["feedback_expert_mse"]
        return output
    finally:
        head.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--init-feedback", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--expert-weight", type=float, default=0.25)
    parser.add_argument("--residual-l2", type=float, default=0.01)
    args = parser.parse_args()
    if args.seed is None:
        # Refinement keeps the episode holdout but gets a different pair order.
        args.seed = 20261006 if args.init_feedback else 20261005
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.validation_batches,
           args.validate_every, args.log_every) < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("training budgets and learning rate must be positive")
    if args.expert_weight < 0 or args.residual_l2 < 0 or args.warmup_steps < 0:
        parser.error("loss weights and warmup must be nonnegative")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("official teacher training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    seed_all(args.seed)
    config = load_multirate_profile(args.config).training
    if (config["dataset"]["execute_horizon"], config["dataset"]["every_fast_queries"],
            config["model"]["action_expert"]["chunk_size"]) != (4, 4, 16):
        raise ValueError("feedback training requires the H4/R4/16-action data contract")
    schedule = BridgeSchedule.from_training_config(config)
    if schedule.to_config()["action_steps"] != 10 or schedule.to_config()["video_steps"] != 2:
        raise ValueError("official slow sampler must retain its two-video/ten-action schedule")
    base_path = Path(config["model"]["native_teacher_checkpoint"]).resolve()
    identity = [{"path": str(base_path), "sha256": sha256(base_path)} if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(identity, src=0)
    base_identity = identity[0]
    run_dir = args.run_dir.resolve()
    if rank == 0:
        if (run_dir / "training_manifest.json").exists() or (run_dir / "exports").exists():
            raise FileExistsError("use a fresh run directory; --init-feedback starts a new optimizer")
        run_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    log_handle = (run_dir / "train.log").open("a", encoding="utf-8") if rank == 0 else None
    metric_handle = (run_dir / "metrics.jsonl").open("a", encoding="utf-8") if rank == 0 else None

    def log(kind, step=0, **fields):
        if rank == 0:
            record = dict(kind=kind, step=step, wall_time=time.time(), **fields)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            print(line, flush=True)
            log_handle.write(line + "\n"); log_handle.flush()
            metric_handle.write(line + "\n"); metric_handle.flush()

    train_data = make_dataset(config)
    val_data = make_dataset(config, validation=True, train=train_data)
    train_loader, train_sampler = make_loader(train_data, config, args, rank, world_size)
    val_loader, val_sampler = make_loader(val_data, config, args, rank, world_size, validation=True)
    payload = load_checkpoint_payload(base_path)
    normalization = payload["config"]["action_normalization"]
    teacher = build_teacher(config, payload, train_data, device)
    del payload
    head = PlanFeedbackHead().to(device=device, dtype=torch.float32)
    contract = runtime_contract(schedule, head.residual_limit)
    if args.init_feedback:
        initial = torch.load(args.init_feedback, map_location="cpu", weights_only=True, mmap=True)
        validate_initialization(initial, base_identity=base_identity,
            dataset_identity=train_data.packed.dataset_identity, pairing_identity=train_data.pairing_identity,
            contract=contract, spec=head.constructor_spec)
        if initial["action_normalization"] != normalization:
            raise ValueError("feedback initialization action normalizer differs")
        head.load_state_dict(initial["model"], strict=True)
        del initial
    head.train()
    if any(parameter.dtype != torch.float32 for parameter in head.parameters()):
        raise RuntimeError("feedback parameters and Adam master weights must stay FP32")
    wrapped = DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else head
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.001)
    warmup = min(args.warmup_steps, max(1, args.steps // 10))

    def lr_multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1. + math.cos(math.pi * min(1., max(0., progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    fm = build_training_flow_scheduler(config)
    timesteps, sigmas = fm.timesteps.to(device=device, dtype=torch.float32), fm.sigmas.to(device=device, dtype=torch.float32)
    manifest = dict(format=FORMAT, version=VERSION, base_checkpoint=base_identity,
        dataset_identity=train_data.packed.dataset_identity, pairing_identity=train_data.pairing_identity,
        action_normalization=normalization, runtime_contract=contract, spec=head.constructor_spec,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        phase="feedback_refine" if args.init_feedback else "feedback_adapt", world_size=world_size,
        global_batch=args.batch_size * args.gradient_accumulation * world_size,
        data=dict(train_pairs=len(train_data), validation_pairs=len(val_data),
                  train_ages=dict(Counter(record.offset // config["dataset"]["global_downsample_rate"]
                                          for record in train_data.records)),
                  validation_ages=dict(Counter(record.offset // config["dataset"]["global_downsample_rate"]
                                               for record in val_data.records))),
        source_hashes={name: sha256(ROOT / name) for name in (
            "scripts/train_plan_feedback.py", "src/dynamicwam/models/plan_feedback.py",
            "src/dynamicwam/models/flow_schedule.py", "src/dynamicwam/models/native_wam_teacher.py",
            "src/dynamicwam/models/causal_world_bridge.py")},
        initialization={"path": str(args.init_feedback.resolve()), "sha256": sha256(args.init_feedback)}
            if args.init_feedback else None)
    if rank == 0:
        (run_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(__file__, run_dir / "source_train_plan_feedback.py")
        shutil.copyfile(ROOT / "src/dynamicwam/models/plan_feedback.py", run_dir / "source_plan_feedback.py")
        shutil.copyfile(args.config, run_dir / "base_profile.yaml")
    log("initialized", trainable_parameters=sum(p.numel() for p in head.parameters()),
        teacher_trainable_parameters=sum(p.numel() for p in teacher.parameters() if p.requires_grad),
        global_batch=manifest["global_batch"], teacher_precision=contract["precision"])
    seed_all(args.seed + rank)
    train_sampler.set_epoch(0)
    iterator, epoch = iter(train_loader), 0
    started = time.perf_counter()
    last_validation = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = None
        for micro in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            inputs, base, target, expert = prepare_targets(batch, teacher, schedule, timesteps, sigmas, device)
            sync = wrapped.no_sync() if world_size > 1 and micro + 1 < args.gradient_accumulation else contextlib.nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    residual = wrapped(**inputs)
                loss, metrics = feedback_objective(residual, base, target, expert, inputs["age_actions"],
                    limit=head.residual_limit, expert_weight=args.expert_weight, l2_weight=args.residual_l2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite feedback loss")
                (loss / args.gradient_accumulation).backward()
            if accumulated is None:
                accumulated = {key: value.detach().float() / args.gradient_accumulation for key, value in metrics.items()}
            else:
                for key, value in metrics.items():
                    accumulated[key] += value.detach().float() / args.gradient_accumulation
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        grad_tensors = sum(p.grad is not None for p in head.parameters())
        if grad_tensors != sum(1 for _ in head.parameters()):
            raise RuntimeError("a feedback parameter did not participate in the training graph")
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            keys = sorted(accumulated)
            vector = torch.stack([accumulated[key] for key in keys])
            if world_size > 1:
                dist.all_reduce(vector)
                vector /= world_size
            log("train", step, **dict(zip(keys, vector.cpu().tolist())), grad_norm=float(norm),
                grad_tensors=grad_tensors, lr=optimizer.param_groups[0]["lr"], data_epoch=epoch,
                samples=step * manifest["global_batch"], steps_per_second=step / (time.perf_counter() - started))
        if step % args.validate_every == 0 or step == args.steps:
            last_validation = validate(head, val_loader, val_sampler, teacher, schedule, timesteps, sigmas,
                args, device, rank, world_size)
            log("validation", step, **last_validation)
            if rank == 0:
                exported = dict(manifest, model={key: value.detach().cpu() for key, value in head.state_dict().items()},
                                global_step=step, validation=last_validation)
                atomic_save(exported, run_dir / "exports" / f"feedback_step_{step}.pt")
                atomic_save(exported, run_dir / "exports" / "feedback_latest.pt")
                atomic_save(dict(global_step=step, data_epoch=epoch, optimizer=optimizer.state_dict(),
                                 scheduler=scheduler.state_dict()), run_dir / "optimizer_latest.pt")
            if world_size > 1:
                dist.barrier()
    log("complete", args.steps, export=str(run_dir / "exports" / f"feedback_step_{args.steps}.pt"))
    train_data.close()
    val_data.close()
    if log_handle:
        log_handle.close(); metric_handle.close()
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
