from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dynamicwam.inference.runner import DynamicWAMRunner

from .rtc import apply_hard_prefix


def flow_frames_to_pixels(
    flow_frames: np.ndarray,
    *,
    count: int,
    height: int,
    width: int,
    device,
    dtype,
) -> torch.Tensor:
    """Convert four uint8 RGB flow frames to ``[1,3,4,H,W]`` in [-1, 1]."""
    import cv2

    if (
        flow_frames.dtype != np.uint8
        or flow_frames.ndim != 4
        or flow_frames.shape[0] != count
        or flow_frames.shape[-1] != 3
    ):
        raise ValueError(
            f"DynamicWAM requires flow_frames with shape [{count},H,W,3] and dtype uint8, "
            f"got {flow_frames.shape} {flow_frames.dtype}"
        )
    resized = np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in flow_frames
        ],
        axis=0,
    )
    tensor = (
        torch.from_numpy(resized)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    return tensor.div_(127.5).sub_(1.0)


class HeadFlowRunner(DynamicWAMRunner):
    """DynamicWAM full-chunk sampler: head-flow condition, WAM, 10 steps, refresh 0/1."""

    def _initialize_video_latent(
        self,
        first_frame: torch.Tensor,
        flow_frames: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_dtype = self.model.compact_wan.video_model.precision
        first_frame = first_frame.to(self.device, dtype=video_dtype)
        frame_pixels = first_frame.mul(2.0).sub(1.0).unsqueeze(2)
        flow_pixels = flow_frames_to_pixels(
            flow_frames,
            count=int(self.runtime.head_flow_config["count"]),
            height=first_frame.shape[-2],
            width=first_frame.shape[-1],
            device=self.device,
            dtype=video_dtype,
        )
        condition_pixels = torch.cat([flow_pixels, frame_pixels], dim=2)
        with torch.inference_mode():
            condition_latent = self.model.compact_wan.encode_video(condition_pixels)
        future_size = self.model.compact_wan.config.future_video_size
        latent_height, latent_width = self._latent_spatial_size(future_size)
        future_latent = torch.randn(
            (
                1,
                int(condition_latent.shape[1]),
                self.runtime.num_video_frames // 4,
                latent_height,
                latent_width,
            ),
            device=self.device,
            dtype=video_dtype,
        )
        return future_latent, condition_latent

    def _action_dtype(self) -> torch.dtype:
        return next(self.model.action_expert.parameters()).dtype

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.runtime.action_normalizer.normalize(
            state.to(self.device, dtype=self._action_dtype())
        )

    def _motion_batch(
        self,
        motion_features: np.ndarray,
        motion_interval_valid_mask: np.ndarray,
        motion_acceleration_valid_mask: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        return {
            "absolute_motion_features": torch.as_tensor(
                motion_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "absolute_motion_interval_valid_mask": torch.as_tensor(
                motion_interval_valid_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
            "absolute_motion_acceleration_valid_mask": torch.as_tensor(
                motion_acceleration_valid_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
        }

    def _fresh_noisy_actions(self) -> torch.Tensor:
        return torch.randn(
            (1, self.runtime.chunk_size, self.model.config.action_dim),
            device=self.device,
            dtype=self._action_dtype(),
        )

    def _action_timesteps(self) -> torch.Tensor:
        return self.action_scheduler.timesteps.to(
            device=self.device,
            dtype=self._action_dtype(),
        )

    def _prepare_hard_prefix(
        self,
        leftover_qpos: np.ndarray | None,
        inference_delay: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if leftover_qpos is None or inference_delay <= 0:
            return None, None, 0
        leftover = torch.as_tensor(
            leftover_qpos,
            dtype=self._action_dtype(),
            device=self.device,
        )
        if leftover.ndim == 2:
            leftover = leftover.unsqueeze(0)
        if leftover.ndim != 3:
            raise ValueError(
                f"leftover_qpos must be [P, A] or [1, P, A], got {tuple(leftover.shape)}"
            )
        prefix_len = min(
            int(leftover.shape[1]),
            int(inference_delay),
            int(self.runtime.chunk_size),
        )
        if prefix_len <= 0:
            return None, None, 0
        prefix = self.runtime.action_normalizer.normalize(leftover[:, :prefix_len])
        prefix_noise = torch.randn_like(prefix)
        return prefix, prefix_noise, prefix_len

    def _seed_noisy_prefix(
        self,
        noisy_actions: torch.Tensor,
        leftover_qpos: np.ndarray | None,
        inference_delay: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        prefix, prefix_noise, prefix_len = self._prepare_hard_prefix(
            leftover_qpos,
            inference_delay,
        )
        if prefix is None or prefix_noise is None:
            return None, None, 0
        noisy_actions[:, :prefix_len] = self.action_scheduler.add_noise(
            prefix,
            prefix_noise,
            self._action_timesteps()[0],
        )
        return prefix, prefix_noise, prefix_len

    def _step_actions(
        self,
        noisy_actions: torch.Tensor,
        action_pred: torch.Tensor,
        action_t: torch.Tensor,
        prefix: torch.Tensor | None,
        prefix_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        noisy_actions = self.action_scheduler.step(
            action_pred,
            action_t,
            noisy_actions,
        )
        if prefix is not None and prefix_noise is not None:
            apply_hard_prefix(
                noisy_actions,
                prefix,
                prefix_noise,
                self.action_scheduler,
                action_t[0],
            )
        return noisy_actions

    def _freeze_output_prefix(
        self,
        actions: torch.Tensor,
        leftover_qpos: np.ndarray | None,
        prefix_len: int,
    ) -> torch.Tensor:
        if leftover_qpos is None or prefix_len <= 0:
            return actions
        leftover = torch.as_tensor(
            leftover_qpos,
            dtype=actions.dtype,
            device=actions.device,
        )
        if leftover.ndim == 2:
            leftover = leftover.unsqueeze(0)
        actions[:, :prefix_len] = leftover[:, :prefix_len]
        return actions

    @torch.inference_mode()
    def sample_chunk(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        flow_frames: np.ndarray,
        motion_features: np.ndarray,
        motion_interval_valid_mask: np.ndarray,
        motion_acceleration_valid_mask: np.ndarray,
    ) -> torch.Tensor:
        actions, _cache = self.sample_chunk_slow(
            first_frame=first_frame,
            state=state,
            flow_frames=flow_frames,
            motion_features=motion_features,
            motion_interval_valid_mask=motion_interval_valid_mask,
            motion_acceleration_valid_mask=motion_acceleration_valid_mask,
        )
        return actions

    @torch.inference_mode()
    def sample_chunk_slow(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        flow_frames: np.ndarray,
        motion_features: np.ndarray,
        motion_interval_valid_mask: np.ndarray,
        motion_acceleration_valid_mask: np.ndarray,
        leftover_qpos: np.ndarray | None = None,
        inference_delay: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Full WAM refresh: VAE + video steps 0/1, then action denoise."""
        video_dtype = self.model.compact_wan.video_model.precision
        normalized_state = self._normalize_state(state)
        text_embeddings = self._get_text_embeddings()
        video_latent, condition_latent = self._initialize_video_latent(
            first_frame,
            flow_frames,
        )
        motion_batch = self._motion_batch(
            motion_features,
            motion_interval_valid_mask,
            motion_acceleration_valid_mask,
        )
        noisy_actions = self._fresh_noisy_actions()
        action_timesteps = self._action_timesteps()
        video_timesteps = self.video_scheduler.timesteps.to(
            device=self.device,
            dtype=video_dtype,
        )
        refresh_schedule = self._video_refresh_schedule()
        video_cache = None
        prefix, prefix_noise, prefix_len = self._seed_noisy_prefix(
            noisy_actions,
            leftover_qpos,
            inference_delay,
        )

        for action_step in range(self.runtime.num_inference_steps):
            action_t = action_timesteps[action_step].expand(1)
            video_step = refresh_schedule.get(action_step)
            if video_step is not None:
                video_t = video_timesteps[video_step].expand(1)
                outputs = self.model.forward_with_video_cache(
                    {
                        "video_t": video_t,
                        "initial_state": normalized_state,
                        "noisy_actions": noisy_actions,
                        "action_t": action_t,
                        "text_embeddings": text_embeddings,
                        "condition_latent": condition_latent,
                        "future_latent": video_latent,
                        **motion_batch,
                    }
                )
                video_cache = outputs["video_cache"]
                video_latent = self.video_scheduler.step(
                    outputs["video_pred"],
                    video_t,
                    video_latent,
                )
            else:
                if video_cache is None:
                    raise RuntimeError(
                        "video_refresh_steps must build a cache at step 0"
                    )
                outputs = self.model.forward_action_with_video_cache(
                    {
                        "video_cache": video_cache,
                        "initial_state": normalized_state,
                        "noisy_actions": noisy_actions,
                        "action_t": action_t,
                        **motion_batch,
                    }
                )
            noisy_actions = self._step_actions(
                noisy_actions,
                outputs["action_pred"],
                action_t,
                prefix,
                prefix_noise,
            )
        if video_cache is None:
            raise RuntimeError("slow path finished without a reusable video cache")
        actions = self.runtime.action_normalizer.denormalize(noisy_actions)
        return self._freeze_output_prefix(actions, leftover_qpos, prefix_len), video_cache

    @torch.inference_mode()
    def sample_chunk_fast(
        self,
        *,
        state: torch.Tensor,
        motion_features: np.ndarray,
        motion_interval_valid_mask: np.ndarray,
        motion_acceleration_valid_mask: np.ndarray,
        video_cache: dict[str, Any],
        leftover_qpos: np.ndarray | None = None,
        inference_delay: int = 0,
    ) -> torch.Tensor:
        """Action-only denoise attending to a previously written video cache."""
        if not video_cache:
            raise RuntimeError("fast path requires a WorldBridge video cache")
        normalized_state = self._normalize_state(state)
        motion_batch = self._motion_batch(
            motion_features,
            motion_interval_valid_mask,
            motion_acceleration_valid_mask,
        )
        noisy_actions = self._fresh_noisy_actions()
        action_timesteps = self._action_timesteps()
        prefix, prefix_noise, prefix_len = self._seed_noisy_prefix(
            noisy_actions,
            leftover_qpos,
            inference_delay,
        )

        for action_step in range(self.runtime.num_inference_steps):
            action_t = action_timesteps[action_step].expand(1)
            outputs = self.model.forward_action_with_video_cache(
                {
                    "video_cache": video_cache,
                    "initial_state": normalized_state,
                    "noisy_actions": noisy_actions,
                    "action_t": action_t,
                    **motion_batch,
                }
            )
            noisy_actions = self._step_actions(
                noisy_actions,
                outputs["action_pred"],
                action_t,
                prefix,
                prefix_noise,
            )
        actions = self.runtime.action_normalizer.denormalize(noisy_actions)
        return self._freeze_output_prefix(actions, leftover_qpos, prefix_len)
