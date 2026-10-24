"""Actual native time-embedding modules in production mixed precision, CPU only."""
from types import SimpleNamespace

import pytest
import torch

from dynamicwam.dualrate.factory import restore_native_time_precision


def test_actual_native_bf16_video_requires_fp32_time_mlps(monkeypatch):
    import dynamicwam.inference.models.wan_model as module

    class UnusedVAE:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(module, "Wan2_2_VAE", UnusedVAE)
    video = module.WanVideoModel(dict(model_type="t2v", patch_size=(1, 2, 2),
        text_len=3, in_dim=48, out_dim=48, dim=24, ffn_dim=48, num_heads=2,
        num_layers=1, freq_dim=16, text_dim=8), vae_path="unused", device="cpu", precision="bfloat16")
    native = SimpleNamespace(compact_wan=SimpleNamespace(video_model=video))
    original = {name: value.detach().clone() for name, value in video.state_dict().items()}
    assert video.wan_model.time_embedding[0].weight.dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="same dtype"):
        video.prepare_time_embeddings(torch.tensor([500.]), 3)
    restore_native_time_precision(native)
    assert video.wan_model.patch_embedding.weight.dtype == torch.bfloat16
    assert video.wan_model.blocks[0].self_attn.q.weight.dtype == torch.bfloat16
    for name, value in video.state_dict().items():
        torch.testing.assert_close(value.float(), original[name].float(), atol=0, rtol=0)
    head, modulation = video.prepare_time_embeddings(torch.tensor([500.]), 3)
    assert head.dtype == modulation.dtype == torch.float32
    (head.square().mean() + modulation.square().mean()).backward()
    for block in (video.wan_model.time_embedding, video.wan_model.time_projection):
        assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
                   for p in block.parameters())
    assert not torch.cuda.is_initialized()
