"""Shared action-only operations. No world model is owned by this mixin."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from dynamicwam.vendor.wan.modules.attention import flash_attention
from dynamicwam.vendor.wan.modules import attention as wan_attention
from dynamicwam.vendor.wan.modules.model import sinusoidal_embedding_1d


def action_sinusoidal_embedding_1d(dim, position):
    """Vendor-equivalent FP64 sinusoid with no CPU-to-GPU arange copy."""
    if dim % 2:
        raise ValueError("sinusoidal embedding width must be even")
    half = dim // 2
    position = position.to(dtype=torch.float64)
    frequency = torch.pow(10000, -torch.arange(half, device=position.device, dtype=position.dtype).div(half))
    sinusoid = torch.outer(position, frequency)
    return torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=1)

@dataclass(frozen=True)
class ActionAttentionSpec:
    precision: torch.dtype
    window_sizes: tuple[tuple[int, int], ...]


def dense_action_attention(q, k, v, *, window_size=(-1, -1)):
    """Exact dense FlashAttention without CUDA length scalars in Python slices.

    Published fixed-shape video caches explicitly certify every token valid.
    The generic varlen wrapper slices once per row using CUDA lengths, which
    synchronizes the host on every layer. Never use this path for padded rows.
    """
    if wan_attention.FLASH_ATTN_2_AVAILABLE:
        return wan_attention.flash_attn.flash_attn_func(
            q, k, v, dropout_p=0., causal=False, window_size=window_size)
    return flash_attention(q=q, k=k, v=v, window_size=window_size)

def _null_rows(null_bridge, batch: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(null_bridge):
        mask = null_bridge.to(device=device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) == 1:
            mask = mask.expand(batch)
        if int(mask.numel()) != batch:
            raise ValueError(f"null bridge mask has {int(mask.numel())} rows for a batch of {batch}")
        return mask
    value = bool(null_bridge)
    return torch.full((batch,), value, dtype=torch.bool, device=device)


class ActionCoreOps:
    @staticmethod
    def _expand_time_for_tokens(t: torch.Tensor, seq_len: int) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        return t


    def _build_action_time_embeddings(
        self,
        t: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._expand_time_for_tokens(t, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32, enabled=t.is_cuda):
            batch = t.size(0)
            flat_t = t.flatten()
            emb = self.action_expert.time_embedding(
                action_sinusoidal_embedding_1d(self.action_expert.freq_dim, flat_t)
                .unflatten(0, (batch, seq_len))
                .to(device=t.device, dtype=next(self.action_expert.time_embedding.parameters()).dtype)
            )
            adaln = self.action_expert.time_projection(emb).unflatten(
                2, (6, self.config.ae_dim)
            )
        return emb, adaln


    @staticmethod
    def _block_modulation(block, adaln_params: torch.Tensor):
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=adaln_params.is_cuda
        ):
            return (block.modulation.unsqueeze(0) + adaln_params).chunk(6, dim=2)


    @staticmethod
    def _action_qkv_for_wan(
        block, norm_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = block.wan_action_qkv.to(
            device=norm_action.device, dtype=norm_action.dtype
        )
        action_qkv = (
            torch.einsum("btd,hdf->bthf", norm_action, qkv[0]),
            torch.einsum("btd,hdf->bthf", norm_action, qkv[1]),
            torch.einsum("btd,hdf->bthf", norm_action, qkv[2]),
        )
        action_q_h, action_k_h, action_v_h = action_qkv
        batch, seq_len, heads, head_dim = action_q_h.shape
        action_q = block.wan_action_norm_q(action_q_h.flatten(2)).view(
            batch, seq_len, heads, head_dim
        )
        action_k = block.wan_action_norm_k(action_k_h.flatten(2)).view(
            batch, seq_len, heads, head_dim
        )
        action_v = action_v_h.view(batch, seq_len, heads, head_dim)
        return action_q, action_k, action_v


    def _action_attention_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_modulation: tuple[torch.Tensor, ...],
        layer_idx: int,
        video_cache: Dict[str, object],
        *,
        null_bridge: bool = False,
        video_kv_adapter=None,
    ) -> torch.Tensor:
        spec = self.action_attention_spec
        action_block = self.action_expert.blocks[layer_idx]
        action_mod = action_modulation

        norm_action = action_block.norm1(action_tokens).float() * (
            1 + action_mod[1].squeeze(2)
        ) + action_mod[0].squeeze(2)
        action_q, action_k, action_v = self._action_qkv_for_wan(
            action_block, norm_action
        )

        video_seq_lens = video_cache["seq_lens"].to(device=action_q.device)
        attention_dtype = spec.precision

        def attend(joint_k, joint_v, joint_seq_lens, *, dense=False):
            with torch.autocast("cuda", dtype=attention_dtype, enabled=action_q.is_cuda):
                if dense:
                    return dense_action_attention(
                        action_q.to(dtype=attention_dtype),
                        joint_k.to(dtype=attention_dtype),
                        joint_v.to(dtype=attention_dtype),
                        window_size=spec.window_sizes[layer_idx])
                return flash_attention(
                    q=action_q.to(dtype=attention_dtype),
                    k=joint_k.to(dtype=attention_dtype),
                    v=joint_v.to(dtype=attention_dtype),
                    k_lens=joint_seq_lens,
                    window_size=spec.window_sizes[layer_idx],
                )

        # Concatenation/gather does no arithmetic: casting before these copies
        # preserves the attention inputs and avoids an FP32 copy of the world
        # cache in every layer of every denoising step.
        action_k = action_k.to(dtype=attention_dtype)
        action_v = action_v.to(dtype=attention_dtype)
        action_only_lens = torch.full_like(video_seq_lens, action_tokens.shape[1])
        if isinstance(null_bridge, bool) and null_bridge:
            action_out_h = attend(action_k, action_v, action_only_lens, dense=True)
        else:
            vk = video_cache["video_k"][layer_idx]
            vv = video_cache["video_v"][layer_idx]
            if not video_cache.get("attention_ready", False):
                vk = vk.to(device=action_q.device, dtype=attention_dtype)
                vv = vv.to(device=action_v.device, dtype=attention_dtype)
            if video_kv_adapter is not None:
                vk, vv = video_kv_adapter(layer_idx, action_tokens, vk, vv)
            dense = bool(video_cache.get("all_video_tokens_valid", False))
            if isinstance(null_bridge, bool):
                # No Tensor -> Python decisions in the 12-layer/10-step loop.
                if dense:
                    jk, jv = torch.cat((vk, action_k), dim=1), torch.cat((vv, action_v), dim=1)
                    lengths = video_seq_lens + action_tokens.shape[1]
                else:
                    jk, jv, lengths = pack_world_action_kv(vk, vv, action_k, action_v, video_seq_lens)
                action_out_h = attend(jk, jv, lengths, dense=dense)
            else:
                null_mask = _null_rows(null_bridge, int(action_tokens.shape[0]), action_q.device)
                full_window = tuple(spec.window_sizes[layer_idx]) == (-1, -1)
                if dense or not full_window:
                    # Unpadded video is already a dense prefix, so cat replaces gather.
                    # A per-row null mask cannot share that prefix length; those rows
                    # use the action-only result. Padded windows still gather.
                    if dense:
                        jk, jv = torch.cat((vk, action_k), dim=1), torch.cat((vv, action_v), dim=1)
                        lengths = video_seq_lens + action_tokens.shape[1]
                    else:
                        jk, jv, lengths = pack_world_action_kv(
                            vk, vv, action_k, action_v, video_seq_lens
                        )
                    full = attend(jk, jv, lengths, dense=dense)
                    dropped = attend(action_k, action_v, action_only_lens, dense=True)
                    action_out_h = torch.where(null_mask.view(-1, 1, 1, 1), dropped, full)
                else:
                    kept_lengths = torch.where(
                        null_mask, torch.zeros_like(video_seq_lens), video_seq_lens
                    )
                    jk, jv, lengths = pack_world_action_kv(
                        vk, vv, action_k, action_v, kept_lengths
                    )
                    action_out_h = attend(jk, jv, lengths)

        action_out_h = action_out_h.flatten(2).to(
            device=action_block.wan_action_o.weight.device,
            dtype=action_block.wan_action_o.weight.dtype,
        )
        action_out = action_block.wan_action_o(action_out_h)
        with torch.amp.autocast(
            "cuda", dtype=torch.float32, enabled=action_tokens.is_cuda
        ):
            return action_tokens + action_out * action_mod[2].squeeze(2)


    def _build_action_tokens(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_actions: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        required = {
            "absolute_motion_features",
            "absolute_motion_interval_valid_mask",
            "absolute_motion_acceleration_valid_mask",
        }
        missing = sorted(required - batch.keys())
        if missing:
            raise KeyError(f"absolute motion inputs are missing: {missing}")
        registers = self.action_expert.registers.expand(
            noisy_actions.shape[0],
            -1,
            -1,
        )
        state_tokens = initial_state.unsqueeze(1)
        base_tokens = self.action_expert.input_encoder(
            state_tokens,
            noisy_actions,
            registers,
        )
        core_length = 1 + int(noisy_actions.shape[1])
        register_count = int(self.action_expert.config.num_registers)
        if int(base_tokens.shape[1]) != core_length + register_count:
            raise RuntimeError("unexpected state/action/register token layout")
        motion_tokens = self.absolute_motion_tokens(
            batch["absolute_motion_features"],
            batch["absolute_motion_interval_valid_mask"],
            batch["absolute_motion_acceleration_valid_mask"],
            dtype=base_tokens.dtype,
            device=base_tokens.device,
            validate_inputs=not bool(getattr(self, "_motion_inputs_prevalidated", False)),
        )
        return (
            torch.cat(
                (
                    base_tokens[:, :core_length],
                    motion_tokens,
                    base_tokens[:, core_length:],
                ),
                dim=1,
            ),
            core_length,
        )


    def _decode_action_tokens(
        self,
        action_tokens: torch.Tensor,
        action_head_time_emb: torch.Tensor,
        core_length: int,
    ) -> torch.Tensor:
        if core_length != 1 + int(self.config.chunk_size):
            raise RuntimeError(
                f"action core length {core_length} differs from configured "
                f"chunk size {self.config.chunk_size}"
            )
        action_pred_full = self.action_expert.decoder(
            action_tokens[:, :core_length],
            action_head_time_emb[:, :core_length],
        )
        return action_pred_full[:, 1:, :]


    def forward_action_with_video_cache(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        fast_residual=None,
        null_bridge: bool = False,
        video_kv_adapter=None,
    ) -> Dict[str, torch.Tensor]:
        video_cache = batch["video_cache"]
        noisy_actions = batch["noisy_actions"]
        action_t = batch["action_t"]
        initial_state = batch["initial_state"]

        action_tokens, action_core_length = self._build_action_tokens(
            batch,
            noisy_actions,
            initial_state,
        )
        if batch.get("rtc_prefix") is not None:
            from dynamicwam.models.rtc_prefix import action_token_times
            action_t = action_token_times(action_t, batch["rtc_prefix"], action_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self._build_action_time_embeddings(
            action_t,
            action_tokens.shape[1],
        )
        for layer_idx in range(self.config.ae_num_layers):
            action_block = self.action_expert.blocks[layer_idx]
            action_modulation = self._block_modulation(
                action_block,
                action_adaln_params,
            )
            action_tokens = self._action_attention_with_video_cache(
                action_tokens,
                action_modulation,
                layer_idx,
                video_cache,
                null_bridge=null_bridge,
                video_kv_adapter=video_kv_adapter,
            )
            if fast_residual is not None:
                action_tokens = fast_residual(layer_idx, action_tokens)
            action_ffn_in = action_block.norm2(action_tokens).float() * (
                1 + action_modulation[4].squeeze(2)
            ) + action_modulation[3].squeeze(2)
            action_ffn_weight = action_block.ffn[0].weight
            action_ffn = action_block.ffn(
                action_ffn_in.to(
                    device=action_ffn_weight.device,
                    dtype=action_ffn_weight.dtype,
                )
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float32,
                enabled=action_tokens.is_cuda,
            ):
                action_tokens = action_tokens + action_ffn * action_modulation[
                    5
                ].squeeze(2)

        action_pred = self._decode_action_tokens(
            action_tokens,
            action_head_time_emb,
            action_core_length,
        )
        return {"action_pred": action_pred}



def materialize_attention_cache(video_cache: Dict[str, object], *, device, dtype) -> Dict[str, object]:
    """Move the published video K/V once per fast trajectory, not once per layer per step."""

    if video_cache.get("attention_ready", False):
        return video_cache
    ready = dict(video_cache)
    ready["video_k"] = [tensor.to(device=device, dtype=dtype) for tensor in video_cache["video_k"]]
    ready["video_v"] = [tensor.to(device=device, dtype=dtype) for tensor in video_cache["video_v"]]
    ready["seq_lens"] = video_cache["seq_lens"].to(device=device)
    ready["attention_ready"] = True
    return ready


def pack_world_action_kv(video_k, video_v, action_k, action_v, valid_video_lengths):
    """Keep every action after the valid video prefix, before padded video.

    Merely concatenating [padded_video, actions] and setting k_lens to Nv+Na
    incorrectly drops action tokens when a video row is shorter than Nmax.
    This gather is the identity for an unpadded row; no host synchronization.
    """
    nv, na = video_k.shape[1], action_k.shape[1]
    position = torch.arange(nv + na, device=video_k.device)[None]
    lengths = valid_video_lengths[:, None]
    index = torch.where(position < lengths, position,
                        torch.where(position < lengths + na, nv + position - lengths, position - na))
    index = index[:, :, None, None].expand(-1, -1, video_k.shape[2], video_k.shape[3])
    key = torch.cat((video_k, action_k), dim=1).gather(1, index)
    value = torch.cat((video_v, action_v), dim=1).gather(1, index)
    return key, value, valid_video_lengths + na
