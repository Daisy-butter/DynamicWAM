"""Inference-schedule coefficients and a device-side Euler update.

The coefficients come from ``FlowMatchScheduler.set_timesteps``. The update
matches ``FlowMatchScheduler.step`` without moving the sample timestep to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler


@dataclass(frozen=True)
class BridgeSchedule:
    """Action uses the 10-step inference prefix. Video uses its own 2-step schedule."""

    action_sigmas: torch.Tensor
    action_timesteps: torch.Tensor
    video_sigmas: torch.Tensor
    video_timesteps: torch.Tensor
    shift: float
    sigma_min: float
    extra_one_step: bool
    num_train_timesteps: int

    def __post_init__(self) -> None:
        if int(self.action_sigmas.shape[0]) < 1:
            raise ValueError("action schedule must contain at least one step")
        if int(self.video_sigmas.shape[0]) != 2:
            raise ValueError(
                f"video bridge schedule must have 2 sigmas, got {tuple(self.video_sigmas.shape)}"
            )
        if int(self.action_timesteps.shape[0]) != int(self.action_sigmas.shape[0]):
            raise ValueError("action timesteps and sigmas differ in length")
        if int(self.video_timesteps.shape[0]) != 2:
            raise ValueError("video timesteps must have length 2")

    @classmethod
    def from_flow_match(
        cls,
        *,
        shift: float = 5.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
        num_train_timesteps: int = 1000,
        action_steps: int = 10,
        video_steps: int = 2,
    ) -> "BridgeSchedule":
        if action_steps < 1:
            raise ValueError(f"action_steps must be >= 1, got {action_steps}")
        if video_steps != 2:
            raise ValueError(
                "the causal bridge reads video steps 0 and 1 of a 2-step schedule, "
                f"got video_steps={video_steps}"
            )
        action = _scheduler(
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=action_steps,
        )
        video = _scheduler(
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=video_steps,
        )
        return cls(
            action_sigmas=action.sigmas.detach().float().cpu(),
            action_timesteps=action.timesteps.detach().float().cpu(),
            video_sigmas=video.sigmas.detach().float().cpu(),
            video_timesteps=video.timesteps.detach().float().cpu(),
            shift=float(shift),
            sigma_min=float(sigma_min),
            extra_one_step=bool(extra_one_step),
            num_train_timesteps=int(num_train_timesteps),
        )

    def to_config(self) -> dict:
        return dict(shift=self.shift, sigma_min=self.sigma_min,
                    extra_one_step=self.extra_one_step, num_train_timesteps=self.num_train_timesteps,
                    action_steps=len(self.action_sigmas), video_steps=len(self.video_sigmas))

    def with_action_steps(self, action_steps: int) -> "BridgeSchedule":
        """Build an independent fast solver without altering the slow bridge."""
        config = self.to_config()
        config["action_steps"] = int(action_steps)
        return type(self).from_flow_match(**config)

    @classmethod
    def from_training_config(cls, config: dict):
        flow = config["flow_matching"]
        return cls.from_flow_match(shift=float(flow["shift"]), sigma_min=float(flow["sigma_min"]),
             extra_one_step=bool(flow["extra_one_step"]), num_train_timesteps=int(flow["training_timesteps"]))

    def action_pair(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _pair(self.action_sigmas, self.action_timesteps, index)

    def action_step(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sigma pair for one inference step. The last step integrates to zero."""
        if index < 0 or index >= int(self.action_sigmas.shape[0]):
            raise IndexError(f"action step {index} is outside the 10-step schedule")
        sigma = self.action_sigmas[index]
        if index + 1 >= int(self.action_sigmas.shape[0]):
            sigma_next = torch.zeros((), dtype=sigma.dtype)
        else:
            sigma_next = self.action_sigmas[index + 1]
        return sigma, sigma_next, self.action_timesteps[index]

    def video_pair(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _pair(self.video_sigmas, self.video_timesteps, index)


def euler_update(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Match ``sample + model_output * (sigma_next - sigma)`` on the sample device."""
    delta = (sigma_next - sigma).to(device=sample.device)
    return sample + scheduler_delta_product(model_output, delta)


def scheduler_delta_product(model_output: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Match the vendor's CPU-scalar product, including low-precision rounding.

    On CUDA, a CPU FP32 scalar retains its value for the opmath multiplication,
    while a CUDA FP32 zero-dimensional tensor is narrowed to the BF16/FP16
    tensor operand's dtype. Compute that product in FP32 explicitly and round
    BEFORE adding the sample, exactly as FlowMatchScheduler.step does. For an
    FP32 action output this is the original multiply without casts or copies.
    """
    if model_output.dtype in (torch.bfloat16, torch.float16):
        return (model_output.float() * delta).to(model_output.dtype)
    return model_output * delta


def scheduler_step_reference(
    scheduler: FlowMatchScheduler,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Call the vendor solver. Used only to check the device-side update."""
    return scheduler.step(model_output, timestep, sample)


def _scheduler(
    *,
    shift: float,
    sigma_min: float,
    extra_one_step: bool,
    num_train_timesteps: int,
    num_inference_steps: int,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        num_inference_steps=num_inference_steps,
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        sigma_min=sigma_min,
        extra_one_step=extra_one_step,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
    return scheduler


def _pair(
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if index < 0 or index + 1 >= int(sigmas.shape[0]):
        raise IndexError(
            f"schedule index {index} is outside a prefix of length {int(sigmas.shape[0])}"
        )
    return sigmas[index], sigmas[index + 1], timesteps[index]
