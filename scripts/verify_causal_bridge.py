"""Check that the causal bridge runs two joint steps and ignores future labels."""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_types import (
    SourceNoise,
    SourceObservation,
    reject_leaky_source_fields,
)


class _FakeJoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[float, float, bool]] = []

    def _forward_video_action(
        self,
        batch: dict,
        *,
        include_video_prediction: bool,
        build_video_cache: bool,
        detach_cache: bool = True,
    ) -> dict:
        if not include_video_prediction:
            raise RuntimeError("bridge step 0 needs video_pred even when video weights are frozen")
        self.calls.append(
            (
                float(batch["video_t"][0]),
                float(batch["action_t"][0]),
                build_video_cache,
            )
        )
        future = batch["future_latent"]
        actions = batch["noisy_actions"]
        outputs = {
            "video_pred": torch.zeros_like(future),
            "action_pred": torch.zeros_like(actions),
        }
        if build_video_cache:
            token = actions[:, :1, :1].reshape(actions.shape[0], 1, 1, 1)
            outputs["video_cache"] = {
                "video_k": [token],
                "video_v": [token],
                "seq_lens": torch.ones(actions.shape[0], dtype=torch.long),
                "grid_sizes": {"condition_seq_len": 1},
            }
        return outputs


def main() -> None:
    try:
        reject_leaky_source_fields({"future_latent": torch.zeros(1)})
    except ValueError:
        pass
    else:
        raise SystemExit("future labels were accepted as a causal source")

    schedule = BridgeSchedule.from_flow_match()
    source = SourceObservation(
        condition_latent=torch.zeros(1, 4, 1, 2, 2),
        state=torch.zeros(1, 14),
        motion=torch.zeros(1, 4, 12),
        motion_interval_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        motion_acceleration_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        text_embeddings=[torch.zeros(1, 8)],
        source_id=3,
        sim_time=1.5,
    )
    noise = SourceNoise(
        future_latent=torch.randn(1, 4, 2, 2, 2),
        actions=torch.randn(1, 16, 14),
    )
    model = _FakeJoint()
    bundle = CausalWorldBridge(schedule)(model, source, noise, detach_cache=False)
    if len(model.calls) != 2 or model.calls[0][2] or not model.calls[1][2]:
        raise SystemExit(f"bridge calls were {model.calls}, expected cache only on step 1")
    video_t = float(schedule.video_timesteps[0])
    action_t = float(schedule.action_timesteps[0])
    if abs(model.calls[0][0] - video_t) > 1e-4 or abs(model.calls[0][1] - action_t) > 1e-4:
        raise SystemExit("step 0 did not use the inference schedule prefix")
    if bundle.source_id != 3 or len(bundle.video_k) != 1:
        raise SystemExit("bridge metadata was not preserved")
    print("verified causal bridge prefix")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_causal_bridge failed: {exc}", file=sys.stderr)
        raise
