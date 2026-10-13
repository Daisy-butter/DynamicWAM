"""Shared-weight multirate wrapper around the existing world-action model.

The action expert and motion tokens stay on the backbone. Fast visual
parameters live only under ``fast_conditioner``. On the reference path the fast
loss and the aligned regularizer both update that shared expert: the regularizer
sees the current frame, and the fast loss sees the historical bridge.
"""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import replace

import torch
import torch.nn as nn

from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.fast_conditioner import FastConditionMemory, FastConditioner
from dynamicwam.models.flow_schedule import BridgeSchedule, euler_update
from dynamicwam.models.multirate_types import (
    BridgeTensorBundle,
    CurrentObservation,
    SourceNoise,
    SourceObservation,
)


_PHASES = {
    "warmup": "warmup",
    "fast_warmup": "warmup",
    "joint": "joint",
    "cross_time_joint": "joint",
}


def canonical_phase(phase: str) -> str:
    try:
        return _PHASES[str(phase)]
    except KeyError as exc:
        raise ValueError(f"unsupported multirate phase: {phase}") from exc


class MultirateWAM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        conditioner: FastConditioner | None,
        schedule: BridgeSchedule | None = None,
        compact_policy: nn.Module | None = None,
        native_teacher_weight: float = 0.0,
        native_multirate: bool = False,
        fast_steps: int = 10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fast_conditioner = conditioner
        self.fast_policy = compact_policy
        self.native_teacher_weight = float(native_teacher_weight)
        self.native_multirate = bool(native_multirate)
        self.native_phase = ""
        self.native_expert_weight = 1.0
        self.native_fm_anchor_weight = 0.0
        # Joint training gets immutable teacher trajectories from the trainer.
        # The teacher stays outside this module and its optimizer/checkpoints.
        self.external_native_teacher = False
        self.feedback_phase = ""
        self.feedback_expert_weight = .1
        self.max_bridge_age_seconds = None
        if self.native_teacher_weight > 0 and compact_policy is None and not self.native_multirate:
            raise ValueError('native teacher distillation requires an independent fast student')
        if (conditioner is None) == (compact_policy is None):
            raise ValueError("select exactly one fast architecture")
        self.schedule = schedule or BridgeSchedule.from_flow_match()
        if self.native_multirate:
            if compact_policy is not None:
                raise ValueError("native multirate must retain the original action expert")
            if len(self.schedule.action_sigmas) != 10:
                raise ValueError("native slow bridge must retain the official 10-action-step schedule")
            if int(fast_steps) not in (1, 2, 4, 10):
                raise ValueError("native fast_steps must be 1, 2, 4 or 10")
        elif int(fast_steps) != 10:
            raise ValueError("independent fast_steps requires native_multirate")
        self.fast_schedule = self.schedule.with_action_steps(fast_steps) if self.native_multirate else self.schedule
        self.bridge = CausalWorldBridge(self.schedule)

    def prepare_fast_memory(self, current, bridge, text, text_mask=None):
        if self.fast_policy is not None:
            return self.fast_policy.prepare_memory(current, bridge, text, text_mask)
        if getattr(self.fast_conditioner, "world_aware", False):
            return self.fast_conditioner.prepare_memory(current, text, text_mask, bridge=bridge)
        return self.fast_conditioner.prepare_memory(current, text, text_mask)

    def predict_fast_velocity(
        self,
        noisy_actions: torch.Tensor,
        action_t: torch.Tensor,
        current: CurrentObservation,
        bridge: BridgeTensorBundle,
        memory: FastConditionMemory,
        video_cache: dict | None = None,
        prefix=None,
    ) -> torch.Tensor:
        if self.fast_policy is not None:
            return self.fast_policy.predict_velocity(noisy_actions, action_t, memory)
        if prefix is not None:
            from dynamicwam.models.rtc_prefix import apply_prefix
            noisy_actions = apply_prefix(noisy_actions, prefix)
        batch = {
            "video_cache": video_cache if video_cache is not None else bridge.as_video_cache(),
            "noisy_actions": noisy_actions,
            "action_t": action_t,
            "initial_state": current.state,
            "rtc_prefix": prefix,
            "absolute_motion_features": current.motion,
            "absolute_motion_interval_valid_mask": current.motion_interval_valid_mask,
            "absolute_motion_acceleration_valid_mask": (
                current.motion_acceleration_valid_mask
            ),
        }

        def residual(layer_idx: int, tokens: torch.Tensor) -> torch.Tensor:
            return self.fast_conditioner.apply_layer(layer_idx, tokens, memory)

        adapter_kwargs = {}
        if getattr(self.fast_conditioner, "motion_kv", False):
            adapter_kwargs["video_kv_adapter"] = lambda i, tokens, k, v: self.fast_conditioner.adapt_video_kv(i, tokens, memory, k, v)
        outputs = self.backbone.forward_action_with_video_cache(
            batch,
            fast_residual=residual,
            null_bridge=bridge.null_bridge,
            **adapter_kwargs,
        )
        return outputs["action_pred"]

    def sample_fast_actions(
        self,
        current: CurrentObservation,
        bridge: BridgeTensorBundle,
        memory: FastConditionMemory,
        noisy_actions: torch.Tensor,
        *,
        denormalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prefix=None,
    ) -> torch.Tensor:
        """Integrate a fresh action noise through the published action schedule.

        This is a new trajectory conditioned on the bridge and the current
        observation. It does not continue the slow sampler after the two joint
        prefix steps, including when the bridge age is zero.
        """
        if self.fast_policy is not None and self.fast_policy.config.feedback_enabled:
            actions, _, _ = self.fast_policy.sample_feedback(noisy_actions,memory)
            return denormalize(actions) if denormalize is not None else actions
        actions = noisy_actions
        solver = self.fast_schedule
        steps = int(solver.action_sigmas.shape[0])
        video_cache = None
        if self.fast_policy is None:
            from dynamicwam.models.action_core_ops import materialize_attention_cache

            video_cache = materialize_attention_cache(
                bridge.as_video_cache(),
                device=actions.device,
                dtype=self.backbone.action_attention_spec.precision,
            )
        for index in range(steps):
            sigma, sigma_next, timestep = solver.action_step(index)
            action_t = timestep.to(device=actions.device, dtype=actions.dtype)
            action_t = action_t.reshape(1).expand(actions.shape[0])
            velocity = self.predict_fast_velocity(
                actions,
                action_t,
                current,
                bridge,
                memory,
                video_cache=video_cache,
                prefix=prefix,
            )
            actions = euler_update(actions, velocity, sigma, sigma_next)
            if prefix is not None:
                from dynamicwam.models.rtc_prefix import apply_prefix
                actions = apply_prefix(actions, prefix)
        if denormalize is not None:
            return denormalize(actions)
        return actions

    def forward(
        self,
        batch: dict[str, Any],
        *,
        mode: str = "multirate_train",
        phase: str = "warmup",
    ) -> dict[str, torch.Tensor]:
        if mode != "multirate_train":
            raise ValueError(f"unsupported multirate mode: {mode}")
        phase = canonical_phase(phase)
        if self.native_multirate:
            return self._native_forward(batch)
        if self.feedback_phase: self.backbone.eval()
        if self.native_teacher_weight > 0:
            if phase == 'warmup':
                if any(p.requires_grad for p in self.backbone.parameters()):
                    raise RuntimeError('native teacher must remain frozen in student warmup')
                self.backbone.eval()
            elif not self.external_native_teacher:
                raise RuntimeError('joint distillation requires a separate frozen teacher')
        source: SourceObservation = batch["source"]
        noise: SourceNoise = batch["source_noise"]
        current: CurrentObservation = batch["current"]
        detach_cache = phase == "warmup"
        if detach_cache:
            with torch.no_grad():
                bridge = self.bridge(
                    self.backbone,
                    source,
                    noise,
                    detach_cache=True,
                )
        else:
            bridge = self.bridge(
                self.backbone,
                source,
                noise,
                detach_cache=False,
            )
        flag = batch.get("null_bridge", False)
        if self.max_bridge_age_seconds is not None:
            elapsed = torch.as_tensor(current.age.delta_sim_seconds, device=current.state.device)
            expired = elapsed > self.max_bridge_age_seconds
            flag = torch.as_tensor(flag, device=expired.device, dtype=torch.bool) | expired
            current = replace(current, age=replace(current.age, bridge_valid=~flag))
        if torch.is_tensor(flag):
            bridge.null_bridge = flag.to(dtype=torch.bool)
        elif flag:
            bridge.null_bridge = True
        memory = self.prepare_fast_memory(
            current,
            bridge,
            batch["fast_text"],
            batch.get("fast_text_mask"),
        )
        if self.feedback_phase:
            return self._feedback_forward(batch, memory, source, noise, current)
        fast_pred = self.predict_fast_velocity(
            batch["fast_noisy_actions"],
            batch["fast_action_t"],
            current,
            bridge,
            memory,
        )
        regularizer = self.backbone._forward_video_action(
            batch["aligned"],
            include_video_prediction=phase == "joint",
            build_video_cache=False,
            detach_cache=True,
        )
        outputs = {
            "fast_pred": fast_pred,
            "action_pred": regularizer["action_pred"],
        }
        if "video_pred" in regularizer:
            outputs["video_pred"] = regularizer["video_pred"]
        if self.native_teacher_weight > 0 and not batch.get('skip_native_teacher', False):
            from dynamicwam.models.native_wam_teacher import sample_native_wam
            # Only current causal observations, never aligned GT future/noisy
            # expert actions, are passed to the native inference teacher.
            teacher_source = replace(source,
                condition_latent=batch['aligned']['condition_latent'], state=current.state,
                motion=current.motion, motion_interval_valid_mask=current.motion_interval_valid_mask,
                motion_acceleration_valid_mask=current.motion_acceleration_valid_mask)
            if self.external_native_teacher:
                if 'native_teacher_actions' not in batch:
                    raise RuntimeError('missing fixed teacher trajectory for joint distillation')
                teacher_actions = batch['native_teacher_actions'].detach()
            else:
                teacher_actions = sample_native_wam(self.backbone, teacher_source, noise, self.schedule)
            flow_noise, sigma = batch['teacher_flow_noise'], batch['fast_sigma']
            student_input = (1.0 - sigma) * teacher_actions + sigma * flow_noise
            outputs['teacher_pred'] = self.predict_fast_velocity(
                student_input, batch['fast_action_t'], current, bridge, memory)
            outputs['teacher_target'] = flow_noise - teacher_actions
        return outputs

    def _native_forward(self, batch):
        """Train the deployed aged-prior path while preserving the video model.

        Stage 2 learns expert and immutable-teacher vector fields. Stage 3
        unrolls the actual short solver, rather than changing the scheduler at
        deployment without training. The one-step mode is endpoint distillation.
        """
        import torch.nn.functional as F

        if self.native_phase not in {"adapt", "distill"}:
            raise RuntimeError("native training needs native_phase=adapt or distill")
        if not self.external_native_teacher:
            raise RuntimeError("native training requires an immutable external teacher")
        teacher = batch.get("native_teacher_actions")
        skip_teacher = bool(batch.get("skip_native_teacher", False))
        if teacher is None and not skip_teacher:
            raise RuntimeError("native training batch lacks fixed teacher actions")
        if teacher is not None:
            teacher = teacher.detach()
        # Source context comes from the immutable official model. Rebuilding it
        # with the student would change its joint-action prefix during training
        # even if every video parameter remained frozen.
        self.backbone.eval()
        source_bridge = batch.get("native_source_bridge")
        if source_bridge is None:
            raise RuntimeError("native training requires the immutable official source bridge")
        if any(t.requires_grad for t in (*source_bridge.video_k, *source_bridge.video_v)):
            raise RuntimeError("official source bridge must be detached from teacher gradients")
        flag = batch.get("null_bridge", False)
        bridge = replace(source_bridge,
            null_bridge=flag.to(dtype=torch.bool) if torch.is_tensor(flag) else bool(flag))
        current = batch["current"]
        memory = self.prepare_fast_memory(current, bridge, batch["fast_text"], batch.get("fast_text_mask"))
        metrics = {}
        outputs = {}
        prefix = batch.get("rtc_prefix")
        if prefix is not None:
            from dynamicwam.models.rtc_prefix import PrefixCondition, masked_suffix_mse
            loss_fn = lambda pred, target: masked_suffix_mse(pred, target, prefix)
            metrics["rtc/prefix_mean"] = prefix.lengths.float().mean().detach()
            metrics["rtc/suffix_mean"] = (batch["clean_actions"].shape[1] - prefix.lengths.float()).mean().detach()
        else:
            loss_fn = lambda pred, target: F.mse_loss(pred.float(), target.float())
        if self.native_phase == "adapt":
            prediction = self.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                current, bridge, memory, prefix=prefix)
            expert_loss = loss_fn(prediction, batch["fast_target"])
            loss = self.native_expert_weight * expert_loss
            outputs["fast_pred"] = prediction
            metrics["loss/fast"] = expert_loss.detach()
            if not skip_teacher:
                flow_noise, sigma = batch["teacher_flow_noise"], batch["fast_sigma"]
                noisy = (1 - sigma) * teacher + sigma * flow_noise
                teacher_prefix = PrefixCondition(teacher, prefix.lengths) if prefix is not None else None
                teacher_prediction = self.predict_fast_velocity(noisy, batch["fast_action_t"], current, bridge, memory, prefix=teacher_prefix)
                teacher_target = flow_noise - teacher
                teacher_loss = loss_fn(teacher_prediction, teacher_target)
                loss = loss + self.native_teacher_weight * teacher_loss
                outputs.update(teacher_pred=teacher_prediction, teacher_target=teacher_target)
                metrics["loss/native_teacher"] = teacher_loss.detach()
        else:
            if prefix is not None:
                raise ValueError("RTC endpoint distillation needs its own prefix-conditioned teacher recipe")
            actions = self.sample_fast_actions(current, bridge, memory, batch["native_fast_noise"])
            expert_loss = F.mse_loss(actions.float(), batch["clean_actions"].float())
            loss = self.native_expert_weight * expert_loss
            outputs["native_actions"] = actions
            metrics["loss/fast"] = expert_loss.detach()
            if not skip_teacher:
                teacher_loss = F.mse_loss(actions.float(), teacher.float())
                loss = loss + self.native_teacher_weight * teacher_loss
                metrics["loss/native_teacher"] = teacher_loss.detach()
            if self.native_fm_anchor_weight > 0:
                prediction = self.predict_fast_velocity(batch["fast_noisy_actions"], batch["fast_action_t"],
                    current, bridge, memory)
                anchor_loss = F.mse_loss(prediction.float(), batch["fast_target"].float())
                loss = loss + self.native_fm_anchor_weight * anchor_loss
                metrics["loss/fm_anchor"] = anchor_loss.detach()
        metrics["loss/total"] = loss.detach()
        outputs.update(native_loss=loss, native_metrics=metrics)
        return outputs

    def _feedback_forward(self, batch, memory, source, noise, current):
        from dynamicwam.models.action_feedback import future_patch_target, dynamics_loss
        import torch.nn.functional as F
        policy = self.fast_policy
        if self.feedback_phase == 'dynamics':
            # This is supervised dynamics, so action/future labels must match.
            prediction = policy.feedback_dynamics(memory.tokens,batch['clean_actions'])
            target = future_patch_target(batch['clean_future'],policy.config.feedback_channels)
            loss = dynamics_loss(prediction,target)
            metrics = {'loss/future':loss.detach()}
            if not self.training:
                altered = torch.roll(batch['clean_actions'], shifts=8, dims=1)
                other = policy.feedback_dynamics(memory.tokens,altered)
                metrics['loss/future_wrong_order'] = dynamics_loss(other,target)
                metrics['loss/future_action_response'] = (other-prediction).float().square().mean()
            return {'feedback_loss':loss,'feedback_metrics':metrics}
        teacher_actions = batch.get('native_teacher_actions')
        if teacher_actions is None: raise RuntimeError('feedback policy needs the fixed native teacher')
        # Same initial noise as the teacher is essential for endpoint distillation.
        actions, proposal, _ = policy.sample_feedback(noise.actions,memory)
        teacher_loss = F.mse_loss(actions.float(),teacher_actions.float()) + .5 * F.mse_loss(proposal.float(),teacher_actions.float())
        expert_loss = F.mse_loss(actions.float(),batch['clean_actions'].float())
        loss = self.native_teacher_weight*teacher_loss + self.feedback_expert_weight*expert_loss
        return {'feedback_loss':loss,'feedback_metrics':{'loss/native_teacher':teacher_loss.detach(),'loss/fast':expert_loss.detach()}}
