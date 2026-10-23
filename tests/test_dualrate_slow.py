"""CPU contract fixtures, not a full Wan forward or pretrained acceptance claim."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dynamicwam.dualrate.slow import (
    DifferentiableVideoCacheMixin, SlowReferenceAdapter, SourceCondition,
    StructuredWorldReadout, adopt_native_for_training,
)
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler


class AttentionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Linear(2, 2)
        self.v = nn.Linear(2, 2)
        self.norm_k = nn.Identity()
        self.num_heads, self.head_dim = 1, 2


class CompactFixture(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.norm1, layer.self_attn = nn.Identity(), AttentionFixture()
        self.video_model = nn.Module()
        self.video_model.precision = torch.float32
        self.video_model.wan_model = nn.Module()
        self.video_model.wan_model.blocks = nn.ModuleList([layer])

    @staticmethod
    def apply_multiscale_rope(key, grids, freqs):
        return key


class NativeContractFixture(nn.Module):
    """Only emulates the native callable contract; its network is intentionally tiny."""
    def __init__(self):
        super().__init__()
        self.video_weight = nn.Parameter(torch.tensor(.12))
        self.action_weight = nn.Parameter(torch.tensor(.15))
        self.motion_weight = nn.Parameter(torch.tensor(.04))
        self.compact_wan = CompactFixture()
        self.action_expert = nn.Module()
        self.action_expert.config = SimpleNamespace(num_registers=4)
        self.config = SimpleNamespace(chunk_size=4, action_dim=2, motion_history_count=4)
        self.calls = []

    def _video_attention_kv_for_cache(self, *args):
        key, value = DifferentiableVideoCacheMixin._video_attention_kv_for_cache(self, *args)
        return key.detach(), value.detach()

    def _forward_video_action(self, batch, *, include_video_prediction, build_video_cache):
        self.calls.append(("joint", batch["noisy_actions"].detach().clone(),
                           batch["action_t"].detach().clone()))
        future, actions = batch["future_latent"], batch["noisy_actions"]
        video_pred = (future * self.video_weight + actions.mean() * .03
                      + batch["absolute_motion_features"].mean() * self.motion_weight)
        result = {"video_pred": video_pred,
                  "action_pred": actions * self.action_weight + future.mean() * .03}
        if build_video_cache:
            tokens = future.permute(0, 2, 3, 4, 1).flatten(1, 3) * self.video_weight
            zeros = torch.zeros(tokens.shape[0], tokens.shape[1], 1, 2)
            key, value = self._video_attention_kv_for_cache(tokens, (zeros,) * 6, 0, {}, None)
            result["video_cache"] = {"video_k": [key], "video_v": [value]}
        return result

    def forward_action_with_video_cache(self, batch):
        self.calls.append(("tail", batch["noisy_actions"].detach().clone(),
                           batch["action_t"].detach().clone()))
        context = batch["video_cache"]["video_k"][0].mean() + batch["video_cache"]["video_v"][0].mean()
        return {"action_pred": batch["noisy_actions"] * self.action_weight + context * .03}


def fixture():
    torch.manual_seed(77)
    source = dict(condition_latent=torch.randn(1, 2, 1, 4, 4),
                  text_embeddings=torch.randn(1, 3, 4), initial_state=torch.randn(1, 2),
                  absolute_motion_features=torch.randn(1, 4, 3),
                  absolute_motion_interval_valid_mask=torch.ones(1, 4, dtype=torch.bool),
                  absolute_motion_acceleration_valid_mask=torch.ones(1, 4, dtype=torch.bool))
    args = dict(source=source, video_noise=torch.randn(1, 2, 2, 4, 4),
                action_noise=torch.randn(1, 4, 2), known_actions=torch.full((1, 4, 2), 3.),
                known_mask=torch.tensor([[True, False, False, False]]),
                action_times=torch.tensor([[2.1, 2.2, 2.3, 2.4]], dtype=torch.float64),
                future_intervals=torch.tensor([[[2.1, 2.3], [2.4, 2.6]]], dtype=torch.float64),
                source_time=2., completed_time=2.05)
    return args


def test_two_joint_eight_tail_same_noise_and_clean_prefix():
    adapter = SlowReferenceAdapter(NativeContractFixture())
    args = fixture()
    result = adapter.sample_reference(**args)
    calls = adapter.native.calls
    assert [call[0] for call in calls] == ["joint"] * 2 + ["tail"] * 8
    assert torch.equal(calls[0][1][:, 1:], args["action_noise"][:, 1:])
    for _, actions, time in calls:
        assert torch.equal(actions[:, :1], args["known_actions"][:, :1])
        assert time.shape == (1, 13)
        assert time[0, 1] == 0
        assert (time[0, torch.tensor([0, 2, 3, 4, 5, 12])] > 0).all()
    assert torch.equal(result["reference_actions"][:, :1], args["known_actions"][:, :1])
    assert result["video_plan_consistency_guaranteed"] is False
    assert result["generation_stage"] == "video_joint_step_2_final_action_step_10"


def test_readout_loss_reaches_native_video_action_motion_and_internal_cache():
    adapter = SlowReferenceAdapter(NativeContractFixture())
    readout = StructuredWorldReadout(latent_channels=2, action_dim=2, world_dim=12)
    reference = adapter.sample_reference(**fixture())
    world = readout(reference)
    world["world_tokens"].square().mean().backward()
    assert world["world_tokens"].shape == (1, 52, 12)
    for parameter in (adapter.native.video_weight, adapter.native.action_weight,
                      adapter.native.motion_weight,
                      adapter.native.compact_wan.video_model.wan_model.blocks[0].self_attn.k.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert reference["predicted_video_latent"].grad_fn is not None
    assert reference["reference_actions"].grad_fn is not None


def test_final_video_step_is_applied_and_matches_vendor():
    adapter = SlowReferenceAdapter(NativeContractFixture())
    args = fixture()
    expected_video = args["video_noise"]
    action = torch.where(args["known_mask"][..., None], args["known_actions"], args["action_noise"])
    video_scheduler = FlowMatchScheduler(num_inference_steps=2, shift=5., sigma_min=0., extra_one_step=True)
    action_scheduler = FlowMatchScheduler(num_inference_steps=10, shift=5., sigma_min=0., extra_one_step=True)
    for step in range(2):
        batch = dict(args["source"], future_latent=expected_video, noisy_actions=action,
                     video_t=video_scheduler.timesteps[step:step+1],
                     action_t=action_scheduler.timesteps[step:step+1])
        output = adapter.native._forward_video_action(batch, include_video_prediction=True,
                                                     build_video_cache=False)
        if step == 1:
            before_final = expected_video.clone()
        expected_video = video_scheduler.step(output["video_pred"], video_scheduler.timesteps[step], expected_video)
        action = action_scheduler.step(output["action_pred"], action_scheduler.timesteps[step], action)
        action = torch.where(args["known_mask"][..., None], args["known_actions"], action)
    actual = adapter.sample_reference(**args)["predicted_video_latent"]
    torch.testing.assert_close(actual, expected_video, atol=0, rtol=0)
    assert not torch.equal(actual, before_final)


def test_checkpointed_sampling_matches_forward_and_parameter_gradients():
    first = SlowReferenceAdapter(NativeContractFixture(), gradient_checkpointing=False)
    second = SlowReferenceAdapter(NativeContractFixture(), gradient_checkpointing=True)
    second.load_state_dict(first.state_dict(), strict=True)
    args = fixture()
    one, two = first.sample_reference(**args), second.sample_reference(**args)
    for key in ("predicted_video_latent", "reference_actions"):
        torch.testing.assert_close(one[key], two[key])
    (one["predicted_video_latent"].square().mean() + one["reference_actions"].square().mean()).backward()
    (two["predicted_video_latent"].square().mean() + two["reference_actions"].square().mean()).backward()
    for (_, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_native_time_dtype_stays_initial_even_when_solver_promotes_sample():
    class PromotingNative(NativeContractFixture):
        def _forward_video_action(self, *args, **kwargs):
            result = super()._forward_video_action(*args, **kwargs)
            result["action_pred"] = result["action_pred"].float()
            return result

        def forward_action_with_video_cache(self, *args, **kwargs):
            result = super().forward_action_with_video_cache(*args, **kwargs)
            result["action_pred"] = result["action_pred"].float()
            return result

    args = fixture()
    args["action_noise"] = args["action_noise"].bfloat16()
    args["known_actions"] = args["known_actions"].bfloat16()
    slow = SlowReferenceAdapter(PromotingNative())
    result = slow.sample_reference(**args)
    assert result["reference_actions"].dtype == torch.float32
    assert all(time.dtype == torch.bfloat16 for _, _, time in slow.native.calls)


def test_original_native_class_is_not_patched_and_checkpoint_keys_stay_native():
    native = NativeContractFixture()
    wrapped = adopt_native_for_training(native)
    assert type(native) is NativeContractFixture
    assert isinstance(wrapped, DifferentiableVideoCacheMixin)
    assert wrapped.video_weight is native.video_weight
    assert set(wrapped.state_dict()) == set(native.state_dict())
    tokens = torch.randn(1, 2, 2, requires_grad=True)
    zeros = torch.zeros(1, 2, 1, 2)
    assert not native._video_attention_kv_for_cache(tokens, (zeros,) * 6, 0, {}, None)[0].requires_grad
    assert wrapped._video_attention_kv_for_cache(tokens, (zeros,) * 6, 0, {}, None)[0].requires_grad


@pytest.mark.parametrize("illegal", ["future_latent", "clean_actions", "future_labels", "video_cache"])
def test_source_whitelist_rejects_future_leakage(illegal):
    args = fixture()
    args["source"][illegal] = torch.zeros(1)
    with pytest.raises(ValueError, match="whitelist"):
        SlowReferenceAdapter(NativeContractFixture()).sample_reference(**args)


def test_supervision_is_separate_and_returns_no_reference_or_cache():
    adapter = SlowReferenceAdapter(NativeContractFixture())
    args = fixture()
    result = adapter.supervised_forward(args["source"], args["video_noise"],
        args["action_noise"], torch.tensor([.3]), torch.tensor([.5]),
        args["known_actions"], args["known_mask"])
    assert set(result) == {"action_pred", "video_pred"}
    result["video_pred"].mean().backward()
    assert adapter.native.video_weight.grad is not None
    with pytest.raises(ValueError, match="predictive"):
        StructuredWorldReadout()(result)


def test_codec_intervals_are_preserved_and_not_invented_from_latent_count():
    args = fixture()
    adapter = SlowReferenceAdapter(NativeContractFixture())
    ref = adapter.sample_reference(**args)
    world = StructuredWorldReadout(2, 2, 12)(ref)
    assert torch.isneginf(world["world_intervals"][:, :16, 0]).all()
    assert torch.isposinf(world["world_intervals"][:, :16, 1]).all()
    torch.testing.assert_close(world["world_intervals"][:, 16:48],
                               args["future_intervals"].repeat_interleave(16, dim=1))
    assert torch.isfinite(world["world_tokens"]).all()
    args["future_intervals"] = torch.tensor([[[2.1, 2.6]]])
    with pytest.raises(ValueError, match="explicitly"):
        adapter.sample_reference(**args)


@pytest.mark.parametrize("case", ["late_source", "reverse_interval", "reversed_times", "prefix_hole"])
def test_invalid_temporal_and_prefix_contracts_fail_closed(case):
    args = fixture()
    if case == "late_source":
        args["completed_time"] = 1.
    elif case == "reverse_interval":
        args["future_intervals"][0, 0] = torch.tensor([2.3, 2.1])
    elif case == "reversed_times":
        args["action_times"] = args["action_times"].flip(1)
    else:
        args["known_mask"][0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError):
        SlowReferenceAdapter(NativeContractFixture()).sample_reference(**args)


def test_actual_native_tiny_wan_action_motion_contract_cpu(monkeypatch):
    """Real native modules, tiny random sizes, CPU SDPA kernel instead of CUDA FA.

    This checks native token layout, cache/RoPE, both video updates, eight action
    tails, checkpoint replay and gradients. It does NOT load full weights or VAE,
    use production precision/CUDA FlashAttention, or establish GPU acceptance.
    """
    import dynamicwam.models.small_wam as native_module
    import dynamicwam.vendor.wan.modules.model as wan_module
    import dynamicwam.inference.models.wan_model as native_video_module
    from dynamicwam.inference.models.wan_model import WanVideoModel
    from dynamicwam.inference.models.compact_wan import CompactWANConfig, CompactWANModel

    def cpu_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        assert q.device.type == k.device.type == v.device.type == "cpu"
        assert q_lens is None
        key_mask = None
        if k_lens is not None:
            key_mask = torch.arange(k.shape[1])[None, :] < k_lens[:, None]
            key_mask = key_mask[:, None, None, :]
        result = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(),
            v.transpose(1, 2).float(), attn_mask=key_mask)
        return result.transpose(1, 2).to(q.dtype)

    monkeypatch.setattr(native_module, "flash_attention", cpu_attention)
    monkeypatch.setattr(wan_module, "flash_attention", cpu_attention)
    class UnusedVAE:
        def __init__(self, *, vae_pth, device):
            assert vae_pth == "diagnostic-no-vae"

        def encode(self, pixels):
            raise AssertionError("the latent-input native test must never invoke a fake VAE")

    monkeypatch.setattr(native_video_module, "Wan2_2_VAE", UnusedVAE)
    torch.manual_seed(91)
    compact_config = CompactWANConfig(checkpoint_path="diagnostic-no-checkpoint",
        config_path="diagnostic-no-config", vae_path="diagnostic-no-vae",
        future_video_size=(64, 64), precision="bfloat16",
        dim=24, ffn_dim=48, num_heads=2, num_layers=1, head_dim=12)
    video = WanVideoModel(dict(model_type="t2v", patch_size=(1, 2, 2), text_len=3,
        in_dim=48, out_dim=48, dim=24, ffn_dim=48, num_heads=2, num_layers=1,
        freq_dim=16, text_dim=8), vae_path="diagnostic-no-vae", device="cpu", precision="bfloat16")
    video.float()
    video.precision = torch.float32
    compact = CompactWANModel(compact_config, video)
    native_config = native_module.SmallWAMActionConfig(compact_wan=compact_config,
        action_dim=14, state_dim=14, chunk_size=4, ae_dim=24, ae_ffn_dim=48,
        ae_num_layers=1, wan_frozen=False, motion_history_count=4,
        motion_feature_mean=(0.,) * 12, motion_feature_scale=(1.,) * 12)
    native = native_module.SmallWAMActionModel(native_config, compact)
    # Native new-model heads are zero-initialized. Nonzero diagnostic heads make
    # the gradient reach earlier random layers, as a trained checkpoint would.
    nn.init.normal_(native.action_expert.decoder.action_head[0].weight, std=.02)
    nn.init.normal_(video.wan_model.head.head.weight, std=.02)
    slow = SlowReferenceAdapter(native, gradient_checkpointing=True)
    source = SourceCondition(condition_latent=torch.randn(1, 48, 1, 4, 4),
        text_embeddings=[torch.randn(3, 8)], initial_state=torch.randn(1, 14),
        absolute_motion_features=torch.randn(1, 4, 12),
        absolute_motion_interval_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        absolute_motion_acceleration_valid_mask=torch.ones(1, 4, dtype=torch.bool))
    noise = torch.randn(1, 4, 14)
    video_noise = torch.randn(1, 48, 2, 4, 4)
    prefix = torch.randn_like(noise)
    mask = torch.tensor([[True, False, False, False]])
    reference = slow.sample_reference(source, video_noise, noise, prefix, mask,
        torch.tensor([[.1, .2, .3, .4]]), torch.tensor([[[.1, .3], [.4, .6]]]), 0., .02)
    assert reference["predicted_video_latent"].shape == video_noise.shape
    assert reference["reference_actions"].shape == noise.shape
    torch.testing.assert_close(reference["reference_actions"][:, :1], prefix[:, :1], atol=0, rtol=0)
    readout = StructuredWorldReadout(world_dim=24)
    world = readout(reference)
    world["world_tokens"].square().mean().backward()
    for parameter in (slow.native.compact_wan.video_model.wan_model.patch_embedding.weight,
                      slow.native.compact_wan.video_model.wan_model.blocks[0].self_attn.k.weight,
                      slow.native.action_expert.blocks[0].wan_action_qkv,
                      slow.native.absolute_motion_tokens.encoder[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    slow.zero_grad(set_to_none=True)
    supervised = slow.supervised_forward(source, video_noise, noise, .5, .4, prefix, mask)
    assert supervised["video_pred"].shape == video_noise.shape
    assert supervised["action_pred"].shape == noise.shape
    (supervised["video_pred"].square().mean() +
     supervised["action_pred"][:, 1:].square().mean()).backward()
    assert torch.isfinite(slow.native.compact_wan.video_model.wan_model.head.head.weight.grad).all()
