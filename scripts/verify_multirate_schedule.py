"""Check that the bridge uses the 10-step action prefix and the same Euler step."""

from __future__ import annotations

import sys

import torch

from dynamicwam.models.flow_schedule import (
    BridgeSchedule,
    euler_update,
    scheduler_step_reference,
)
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler


def main() -> None:
    schedule = BridgeSchedule.from_flow_match()
    video_sigma, video_next, _video_t = schedule.video_pair(0)
    action_sigma, action_next, _action_t = schedule.action_pair(0)
    if not torch.allclose(video_sigma, torch.tensor(1.0)):
        raise SystemExit(f"video sigma 0 is {float(video_sigma)}, expected 1")
    if not torch.allclose(video_next, torch.tensor(5.0 / 6.0), atol=1e-5):
        raise SystemExit(f"video sigma 1 is {float(video_next)}, expected 5/6")
    if not torch.allclose(action_sigma, torch.tensor(1.0)):
        raise SystemExit(f"action sigma 0 is {float(action_sigma)}, expected 1")
    if not torch.allclose(action_next, torch.tensor(45.0 / 46.0), atol=1e-5):
        raise SystemExit(f"action sigma 1 is {float(action_next)}, expected 45/46")
    if int(schedule.action_sigmas.shape[0]) != 10:
        raise SystemExit("source actions must use the 10-step schedule, not a new 2-step one")

    reference = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
    )
    reference.set_timesteps(num_inference_steps=10, training=False)
    sample = torch.randn(2, 4, 3)
    update = torch.randn_like(sample)
    timestep = reference.timesteps[0]
    vendor = scheduler_step_reference(reference, sample, update, timestep)
    local = euler_update(sample, update, reference.sigmas[0], reference.sigmas[1])
    error = (vendor - local).abs().max().item()
    if error > 1e-5:
        raise SystemExit(f"device Euler differs from FlowMatchScheduler.step by {error}")
    print("verified multirate schedule")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_multirate_schedule failed: {exc}", file=sys.stderr)
        raise
