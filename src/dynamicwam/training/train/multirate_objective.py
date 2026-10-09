"""Loss for one multirate training step.

The fast target is the flow-matching vector at time t. The aligned regularizer
uses the original video and action targets and never reads the causal bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MultirateLossObjective:
    action_weight: float = 1.0
    video_weight: float = 0.0
    fast_weight: float = 1.0
    native_teacher_weight: float = 0.0

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        fast_target: torch.Tensor,
        action_target: torch.Tensor,
        video_target: torch.Tensor | None = None,
        *,
        executed_horizon: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if 'native_loss' in outputs:
            total = outputs['native_loss']
            metrics = {k: float(v.detach()) for k, v in outputs['native_metrics'].items()}
            return total, dict(metrics, **{'loss/total': float(total.detach())})
        if 'feedback_loss' in outputs:
            total = outputs['feedback_loss']
            metrics = {k:float(v.detach()) for k,v in outputs['feedback_metrics'].items()}
            return total, dict(metrics, **{'loss/total':float(total.detach())})
        fast_pred = outputs["fast_pred"].float()
        fast_target = fast_target.float()
        chunk_loss = F.mse_loss(fast_pred, fast_target)
        if executed_horizon is None:
            fast_loss = chunk_loss
        else:
            horizon = int(executed_horizon)
            if horizon < 1 or horizon > int(fast_pred.shape[1]):
                raise ValueError(
                    f"executed horizon {horizon} does not fit fast chunk {tuple(fast_pred.shape)}"
                )
            # Deploy keeps actions[:H] and discards the rest of the chunk.
            fast_loss = F.mse_loss(fast_pred[:, :horizon], fast_target[:, :horizon])
        action_loss = F.mse_loss(outputs["action_pred"].float(), action_target.float())
        total = self.fast_weight * fast_loss + self.action_weight * action_loss
        metrics = {
            "loss/fast": float(fast_loss.detach()),
            "loss/fast_chunk": float(chunk_loss.detach()),
            "loss/action": float(action_loss.detach()),
        }
        if self.video_weight > 0.0:
            if video_target is None or "video_pred" not in outputs:
                raise RuntimeError("joint phase requires a video prediction and target")
            video_loss = F.mse_loss(outputs["video_pred"].float(), video_target.float())
            total = total + self.video_weight * video_loss
            metrics["loss/video"] = float(video_loss.detach())
        if self.native_teacher_weight > 0:
            teacher_loss = F.mse_loss(outputs['teacher_pred'].float(), outputs['teacher_target'].float())
            total = total + self.native_teacher_weight * teacher_loss
            metrics['loss/native_teacher'] = float(teacher_loss.detach())
        metrics["loss/total"] = float(total.detach())
        return total, metrics
