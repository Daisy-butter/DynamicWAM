"""Two joint forwards that publish the video K/V used by later fast queries.

Step 0 updates the future latent and the source action proposal. Step 1 is the
forward whose pre-attention video K/V becomes the bridge. Later action-only
denoising steps do not change that cache and are not run here.
"""

from __future__ import annotations

from typing import Any

import torch

from dynamicwam.models.flow_schedule import BridgeSchedule, euler_update
from dynamicwam.models.rtc_prefix import PrefixCondition, apply_prefix, validate_prefix_condition
from dynamicwam.models.native_wam_teacher import native_sampler_dtypes
from dynamicwam.models.multirate_types import (
    BridgeTensorBundle,
    SourceNoise,
    SourceObservation,
    reject_leaky_source_fields,
)


class CausalWorldBridge(torch.nn.Module):
    def __init__(self, schedule: BridgeSchedule | None = None) -> None:
        super().__init__()
        self.schedule = schedule or BridgeSchedule.from_flow_match()

    def forward(
        self,
        model: torch.nn.Module,
        source: SourceObservation,
        noise: SourceNoise,
        *,
        detach_cache: bool,
    ) -> BridgeTensorBundle:
        reject_leaky_source_fields(getattr(source, "__dict__", {}))
        batch = source.as_model_batch()
        future = noise.future_latent
        actions = noise.actions
        prefix = batch.get("rtc_prefix")
        if prefix is not None:
            prefix = PrefixCondition(prefix.actions.to(actions), prefix.lengths.to(device=actions.device))
            validate_prefix_condition(prefix, max_prefix=4, reference=actions)
            batch["rtc_prefix"] = prefix
            actions = apply_prefix(actions, prefix)
        action_dtype, video_dtype = native_sampler_dtypes(model, noise)
        if future.shape[0] != actions.shape[0]:
            raise ValueError("source noise batch sizes differ")
        future, actions, _step0 = self._joint_step(
            model,
            batch,
            future,
            actions,
            index=0,
            build_video_cache=False,
            detach_cache=detach_cache,
            action_dtype=action_dtype, video_dtype=video_dtype,
        )
        _future, _actions, outputs = self._joint_step(
            model,
            batch,
            future,
            actions,
            index=1,
            build_video_cache=True,
            detach_cache=detach_cache,
            action_dtype=action_dtype, video_dtype=video_dtype,
        )
        cache = outputs["video_cache"]
        # In flow matching v = noise - clean. Compare future committed clean
        # actions with this clean estimate, never with the noisy conditioning.
        action_sigma, _next_sigma, _time = self.schedule.action_step(1)
        plan = actions - action_sigma.to(actions) * outputs["action_pred"]
        if prefix is not None:
            plan = apply_prefix(plan, prefix)
        if detach_cache:
            plan = plan.detach()
        own = lambda value: value.detach() if detach_cache and torch.is_tensor(value) else value
        return BridgeTensorBundle(
            video_k=tuple(cache["video_k"]),
            video_v=tuple(cache["video_v"]),
            seq_lens=cache["seq_lens"],
            grid_sizes=dict(cache["grid_sizes"]),
            source_id=source.source_id
            if torch.is_tensor(source.source_id)
            else int(source.source_id),
            sim_time=source.sim_time
            if torch.is_tensor(source.sim_time)
            else float(source.sim_time),
            null_bridge=False,
            all_video_tokens_valid=cache.get("all_video_tokens_valid", False),
            interface_version=3 if source.action_target_offsets is not None else 2,
            source_action_plan=plan,
            source_action_sigma=action_sigma.to(actions).detach(),
            source_action_target_offsets=own(source.action_target_offsets),
            source_motion=own(source.motion),
            source_motion_interval_valid_mask=source.motion_interval_valid_mask,
            source_motion_acceleration_valid_mask=source.motion_acceleration_valid_mask,
            source_rgb=own(source.rgb), source_rgb_valid_mask=source.rgb_valid_mask,
        )

    def _joint_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        future: torch.Tensor,
        actions: torch.Tensor,
        *,
        index: int,
        build_video_cache: bool,
        detach_cache: bool,
        action_dtype: torch.dtype,
        video_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        video_t = self.schedule.video_timesteps[index]
        action_sigma, action_sigma_next, action_t = self.schedule.action_step(index)
        step_batch = dict(batch)
        step_batch["future_latent"] = future
        prefix = batch.get("rtc_prefix")
        if prefix is not None:
            actions = apply_prefix(actions, prefix)
        step_batch["noisy_actions"] = actions
        step_batch["video_t"] = _expand_time(video_t, future, video_dtype)
        step_batch["action_t"] = _expand_time(action_t, actions, action_dtype)
        outputs = model._forward_video_action(
            step_batch,
            include_video_prediction=True,
            build_video_cache=build_video_cache,
            detach_cache=detach_cache,
        )
        if not build_video_cache:
            video_sigma, video_sigma_next, _paired_t = self.schedule.video_pair(index)
            next_future = euler_update(
                future, outputs["video_pred"], video_sigma, video_sigma_next
            )
            next_actions = euler_update(
                actions, outputs["action_pred"], action_sigma, action_sigma_next
            )
            if prefix is not None:
                next_actions = apply_prefix(next_actions, prefix)
            return next_future, next_actions, outputs
        return future, actions, outputs


def _expand_time(timestep: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    value = timestep.to(device=reference.device, dtype=dtype)
    return value.reshape(1).expand(reference.shape[0])
