"""Frozen native WAM sampling for causal, current-observation distillation."""
from __future__ import annotations

import torch

from dynamicwam.models.flow_schedule import euler_update


def native_sampler_dtypes(model, noise):
    """Time embeddings follow native branch precision, never evolving states."""
    action_expert = getattr(model, "action_expert", None)
    action_parameter = next(action_expert.parameters(), None) if action_expert is not None else None
    video_model = getattr(getattr(model, "compact_wan", None), "video_model", None)
    # Initial-noise fallback supports minimal contract-test models. The real
    # WAM exposes both branch precisions, matching HeadFlowRunner exactly.
    return (getattr(action_parameter, "dtype", noise.actions.dtype),
            getattr(video_model, "precision", noise.future_latent.dtype))


@torch.no_grad()
def sample_native_wam(model, source, noise, schedule):
    """Keep the native two joint + eight cached-action schedule.

    Unlike the fast sampler, the action noise updated in the two joint steps
    continues through the remaining eight steps. No expert future is an input.
    Returns normalized actions; no VAE decode, RTC, or new fast modules.
    """
    observation = source.as_model_batch()
    future, actions, cache = noise.future_latent, noise.actions, None
    action_dtype, video_dtype = native_sampler_dtypes(model, noise)
    for index in range(len(schedule.action_sigmas)):
        sigma, next_sigma, timestep = schedule.action_step(index)
        batch = dict(observation, noisy_actions=actions,
                     action_t=timestep.to(device=actions.device, dtype=action_dtype).reshape(1).expand(actions.shape[0]))
        if index < 2:
            batch.update(future_latent=future,
                video_t=schedule.video_timesteps[index].to(device=future.device,
                    dtype=video_dtype).reshape(1).expand(actions.shape[0]))
            outputs = model._forward_video_action(batch, include_video_prediction=True,
                                                   build_video_cache=True, detach_cache=True)
            cache = outputs['video_cache']
            if index == 0:
                vs, vn, _ = schedule.video_pair(index)
                future = euler_update(future, outputs['video_pred'], vs, vn)
            # The final video update is dead: the published cache comes from
            # this forward and no later action step consumes the video latent.
        else:
            outputs = model.forward_action_with_video_cache(dict(batch, video_cache=cache))
        actions = euler_update(actions, outputs['action_pred'], sigma, next_sigma)
    return actions
