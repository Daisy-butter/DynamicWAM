"""Optimizer groups, one Accelerate-style step, and the controlled age schedule."""

from __future__ import annotations

import contextlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dynamicwam.models.fast_conditioner import FastConditioner
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.runtime.ciwam.adapters.domino.deploy_policy_multirate_flow import (
    fast_composite_from_observation,
)
from dynamicwam.runtime.ciwam.wam.multirate_control import ControlledMultirateController
from dynamicwam.training.train.common import build_training_flow_scheduler
from dynamicwam.training.train.multirate_objective import MultirateLossObjective
from dynamicwam.training.train.multirate_trainer import (
    MultirateTrainer,
    configure_multirate_trainable,
    multirate_optimizer_groups,
)
from dynamicwam.training.utils.scheduler import create_scheduler


class _Accel:
    sync_gradients = True

    def accumulate(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(list(params), max_norm)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert = nn.Linear(4, 4)
        self.absolute_motion_tokens = nn.Linear(4, 4)
        self.compact_wan = nn.Module()
        self.compact_wan.video_model = nn.Module()
        self.compact_wan.video_model.wan_model = nn.Linear(4, 4)

    def _forward_video_action(
        self,
        batch,
        *,
        include_video_prediction: bool,
        build_video_cache: bool,
        detach_cache: bool = True,
    ):
        del detach_cache
        prediction = self.action_expert(batch["noisy_actions"])
        outputs = {"action_pred": prediction}
        if include_video_prediction:
            scalar = self.compact_wan.video_model.wan_model.weight.reshape(-1)[0]
            outputs["video_pred"] = torch.zeros_like(batch["future_latent"]) + scalar
        if build_video_cache:
            token = prediction.new_zeros(prediction.shape[0], 1, 1, 1)
            outputs["video_cache"] = {
                "video_k": [token],
                "video_v": [token],
                "seq_lens": torch.ones(prediction.shape[0], dtype=torch.long),
                "grid_sizes": {"condition_seq_len": 1},
            }
        return outputs

    def forward_action_with_video_cache(self, batch, fast_residual=None, null_bridge=False):
        del null_bridge
        prediction = self.action_expert(batch["noisy_actions"])
        if fast_residual is not None:
            hidden = prediction.new_zeros(prediction.shape[0], prediction.shape[1], 32)
            hidden = hidden + prediction.mean(dim=-1, keepdim=True)
            residual = fast_residual(0, hidden)
            prediction = prediction + residual.mean(dim=-1, keepdim=True)
        return {"action_pred": prediction}


def _config() -> dict:
    return {
        "action_learning_rate": 1e-3,
        "fast_learning_rate": 2e-3,
        "video_learning_rate": 1e-4,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "null_bridge_probability": 0.0,
        "max_steps": 4,
        "warmup_steps": 0,
        "min_lr_ratio": 0.1,
        "model": {
            "fast_conditioner": {"sim_scale": 1.0, "action_scale": 16.0},
        },
        "flow_matching": {
            "shift": 5.0,
            "sigma_min": 0.0,
            "extra_one_step": True,
            "training_timesteps": 8,
        },
    }


def _batch() -> dict:
    return {
        "source_condition_latent": torch.zeros(2, 2, 1, 2, 2),
        "source_state": torch.zeros(2, 4),
        "source_motion": torch.zeros(2, 4, 12),
        "source_motion_interval_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "source_motion_acceleration_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "bridge_noise_shape_latent": torch.zeros(2, 2, 1, 2, 2),
        "fast_rgb": torch.zeros(2, 3, 3, 32, 32),
        "fast_state": torch.zeros(2, 4),
        "fast_motion": torch.zeros(2, 4, 12),
        "fast_motion_interval_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "fast_motion_acceleration_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "fast_action_sequence": torch.zeros(2, 4, 4),
        "fast_future_latent": torch.zeros(2, 2, 1, 2, 2),
        "fast_condition_latent": torch.zeros(2, 2, 1, 2, 2),
        "text_embeddings": [torch.zeros(3, 8), torch.zeros(3, 8)],
        "delta_sim_seconds": torch.zeros(2),
        "delta_actions": torch.tensor([0.0, 4.0]),
        "state_delta": torch.zeros(2, 4),
    }


def _ages() -> list[int]:
    controller = ControlledMultirateController(
        execute_horizon=4,
        every_fast_queries=4,
        chunk_size=16,
    )
    ages = []
    for _ in range(16):
        if controller.needs_query():
            ages.append(controller.bridge_age_actions())
            controller.commit(np.zeros((16, 14), dtype=np.float32))
        controller.take_action()
        controller.mark_executed()
    return ages


def main() -> None:
    config = _config()

    class _Shell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.fast_conditioner = nn.Linear(2, 2)

    model = _Shell()
    configure_multirate_trainable(model, "fast_warmup")
    if any(parameter.requires_grad for parameter in model.backbone.compact_wan.parameters()):
        raise SystemExit("warmup left the video expert trainable")
    groups = multirate_optimizer_groups(model, config, "warmup")
    if [group["name"] for group in groups] != ["action_core", "fast_conditioner"]:
        raise SystemExit(f"warmup groups were {[group['name'] for group in groups]}")
    configure_multirate_trainable(model, "cross_time_joint")
    joint_groups = multirate_optimizer_groups(model, config, "joint")
    if [group["name"] for group in joint_groups] != [
        "action_core",
        "fast_conditioner",
        "video_expert",
    ]:
        raise SystemExit("joint optimizer is missing the video expert")

    backbone = _Backbone()
    conditioner = FastConditioner(
        dim=32,
        layers=1,
        heads=4,
        head_dim=8,
        text_dim=8,
        state_dim=4,
    )
    system = MultirateWAM(backbone, conditioner)
    configure_multirate_trainable(system, "fast_warmup")
    optimizer = AdamW(
        multirate_optimizer_groups(system, config, "warmup"),
        lr=1e-3,
    )
    scheduler = build_training_flow_scheduler(config)
    trainer = MultirateTrainer.__new__(MultirateTrainer)
    trainer.model = system
    trainer.accelerator = _Accel()
    trainer.optimizer = optimizer
    trainer.scheduler = create_scheduler(
        optimizer,
        total_steps=4,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    trainer.objective = MultirateLossObjective(video_weight=0.0)
    trainer.phase = "warmup"
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer._timesteps = scheduler.timesteps
    trainer._sigmas = scheduler.sigmas
    video_before = system.backbone.compact_wan.video_model.wan_model.weight.detach().clone()
    action_before = system.backbone.action_expert.weight.detach().clone()
    out_before = system.fast_conditioner.blocks[0].out_proj.weight.detach().clone()
    metrics = trainer.train_step(_batch())
    if metrics is None or "loss/fast" not in metrics:
        raise SystemExit("warmup step did not return a fast loss")
    if not torch.equal(
        system.backbone.compact_wan.video_model.wan_model.weight, video_before
    ):
        raise SystemExit("warmup updated the frozen video expert")
    if torch.equal(system.backbone.action_expert.weight, action_before):
        raise SystemExit("warmup action expert did not receive a gradient")
    if torch.equal(system.fast_conditioner.blocks[0].out_proj.weight, out_before):
        raise SystemExit("fast output projection did not receive a gradient")

    configure_multirate_trainable(system, "cross_time_joint")
    trainer.phase = "joint"
    trainer.objective = MultirateLossObjective(video_weight=1.0)
    trainer.optimizer = AdamW(
        multirate_optimizer_groups(system, config, "joint"),
        lr=1e-3,
    )
    trainer.scheduler = create_scheduler(
        trainer.optimizer,
        total_steps=4,
        warmup_steps=0,
        min_lr_ratio=0.1,
    )
    video_before = system.backbone.compact_wan.video_model.wan_model.weight.detach().clone()
    trainer.train_step(_batch())
    if torch.equal(
        system.backbone.compact_wan.video_model.wan_model.weight, video_before
    ):
        raise SystemExit("joint phase did not train the video expert")

    if _ages() != [0, 4, 8, 12]:
        raise SystemExit(f"controlled ages were {_ages()}")
    observation = {
        "observation": {
            "head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
        }
    }
    composite = fast_composite_from_observation(
        observation,
        {"head_size": [240, 320], "wrist_size": [120, 160]},
    )
    if composite.shape != (360, 320, 3) or composite.dtype != np.uint8:
        raise SystemExit(f"fast composite shape was {composite.shape} {composite.dtype}")
    print("verified multirate training step")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_multirate_training failed: {exc}", file=sys.stderr)
        raise
