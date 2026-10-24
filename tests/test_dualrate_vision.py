"""Explicit synthetic weights only: these tests do not establish pretrained quality."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from dynamicwam.dualrate.fast import CurrentObservationEncoder
from dynamicwam.dualrate.vision import ResNet18Stage3, spatial_flow_grid


class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_requires_explicit_provenance_and_no_random_fallback(self):
        with self.assertRaises(ValueError):
            ResNet18Stage3()
        with self.assertRaises(ValueError):
            ResNet18Stage3(test_fixture=True, weights_path="a")
        with self.assertRaises(ValueError):
            CurrentObservationEncoder()

    def test_strict_offline_loading_and_hash_failure(self):
        fixture = ResNet18Stage3(test_fixture=True)
        state = copy.deepcopy(fixture.state_dict())
        # Tiny disk fixture; mock only deserialization to avoid writing a large
        # fake pretrained file. Real hash validation and strict load execute.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "explicit-test-fixture.bin"
            path.write_bytes(b"synthetic test only")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch("dynamicwam.dualrate.vision.torch.load", return_value=state) as loader:
                loaded = ResNet18Stage3(path, digest)
                self.assertTrue(loaded.pretrained)
                self.assertEqual(loaded.weight_identity, digest)
                self.assertTrue(loader.call_args.kwargs["weights_only"])
            sample = torch.randn(1, 3, 32, 48)
            torch.testing.assert_close(fixture(sample), loaded(sample), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                ResNet18Stage3(path, "0" * 64)
            with mock.patch("dynamicwam.dualrate.vision.torch.load", return_value={k: v for k, v in state.items() if k != "conv1.weight"}):
                with self.assertRaises(RuntimeError):
                    ResNet18Stage3(path, digest)
            with mock.patch("dynamicwam.dualrate.vision.torch.load", return_value=dict(state, unrelated=torch.ones(1))):
                with self.assertRaises(ValueError):
                    ResNet18Stage3(path, digest)

    def test_trunk_stride_bn_freeze_and_conv_gradient(self):
        trunk = ResNet18Stage3(test_fixture=True).train()
        before = trunk.bn1.running_mean.clone()
        self.assertEqual(sum(p.numel() for p in trunk.parameters()), 2782784)
        self.assertTrue(all(not m.training for m in trunk.modules() if isinstance(m, nn.BatchNorm2d)))
        x = torch.randn(1, 3, 32, 48, requires_grad=True)
        out = trunk(x)
        self.assertEqual(out.shape, (1, 256, 2, 3))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(trunk.conv1.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(before, trunk.bn1.running_mean, rtol=0, atol=0)

    def fixture_observation(self, **config):
        model = CurrentObservationEncoder(dim=12, views=1, text_dim=8, test_fixture=True, **config)
        x = [torch.rand(1, 1, 3, 17, 33), torch.randn(1, 4, 12), torch.randn(1, 14), torch.randn(1, 2, 8).bfloat16()]
        kwargs = dict(query_time=torch.tensor([10.], dtype=torch.float64),
            rgb_times=torch.tensor([[9.9]], dtype=torch.float64),
            motion_times=torch.tensor([[[9., 9.2], [9.2, 9.4], [9.4, 9.6], [9.6, 9.8]]], dtype=torch.float64),
            motion_valid=torch.ones(1, 4, dtype=torch.bool), text_valid=torch.ones(1, 2, dtype=torch.bool),
            rgb_pixel_valid=torch.ones(1, 1, 17, 33, dtype=torch.bool),
            state_time=torch.tensor([9.99], dtype=torch.float64),
            spatial_flow=torch.ones(1, 16, 5), spatial_flow_valid=torch.ones(1, 16, dtype=torch.bool),
            spatial_flow_times=torch.tensor([[9.8, 9.9]], dtype=torch.float64))
        return model, x, kwargs

    def test_partial_edge_pixels_and_masked_padding_are_preserved(self):
        model, x, kwargs = self.fixture_observation()
        x[0].requires_grad_()
        out, valid = model(*x, **kwargs)
        self.assertEqual(out.shape, (1, 29, 12))  # ceil(17/16)*ceil(33/16) = six
        self.assertTrue(valid[:, :6].all())
        # All external padding is masked before ImageNet normalization/convs.
        kwargs["rgb_pixel_valid"][..., 20:] = False
        before, valid = model(*x, **kwargs)
        bad = x[0].detach().clone()
        bad[..., 20:] = float("nan")
        after, _ = model(bad, *x[1:], **kwargs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(valid[:, :6].sum().item(), 4)
        before[..., 0].sum().backward()
        self.assertEqual(x[0].grad[..., 20:].abs().sum().item(), 0)
        self.assertGreater(x[0].grad[..., :20].abs().sum().item(), 0)

    def test_local_flow_normalized_once_and_invalid_intervals_masked(self):
        model, x, kwargs = self.fixture_observation(flow_mean=[1, 2, 3, 4], flow_scale=[2, 4, 8, 16])
        captured = []
        hook = model.spatial_flow.register_forward_pre_hook(lambda _m, values: captured.append(values[0].detach().clone()))
        model(*x, **kwargs)
        hook.remove()
        expected = torch.tensor([0, -.25, -.25, -.1875, 1.])
        torch.testing.assert_close(captured[0][0, 0], expected, rtol=0, atol=0)
        kwargs["spatial_flow_times"] = torch.tensor([[9.8, 10.1]], dtype=torch.float64)
        before, valid = model(*x, **kwargs)
        self.assertFalse(valid[:, -16:].any())
        kwargs["spatial_flow"].fill_(float("nan"))
        after, _ = model(*x, **kwargs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_future_camera_state_and_clock_origin(self):
        model, x, kwargs = self.fixture_observation()
        before, valid = model(*x, **kwargs)
        shifted = {k: (v + 1e8 if k.endswith("time") or k.endswith("times") else v) for k, v in kwargs.items()}
        after, _ = model(*x, **shifted)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=3e-6)
        kwargs["rgb_times"] += 1
        kwargs["state_time"] += 1
        _, valid = model(*x, **kwargs)
        self.assertFalse(valid[:, :6].any())
        self.assertFalse(valid[:, 10].item())

    def test_full_encoder_strict_reload(self):
        model, x, kwargs = self.fixture_observation()
        other, _, _ = self.fixture_observation()
        other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        torch.testing.assert_close(model(*x, **kwargs)[0], other(*x, **kwargs)[0], rtol=0, atol=0)

    def test_spatial_flow_grid_uses_real_dt_and_retains_locality(self):
        flow = torch.zeros(1, 2, 8, 8)
        flow[:, 0, :2, :2] = 2
        reliable = torch.ones(1, 8, 8, dtype=torch.bool)
        values, valid = spatial_flow_grid(flow, reliable, torch.tensor([[1e8, 1e8 + .25]], dtype=torch.float64))
        torch.testing.assert_close(values[0, 0], torch.tensor([2., 0., 8., 0., 1.]))
        self.assertEqual(values[0, 1, 0].item(), 0)
        self.assertTrue(valid.all())
        reliable[:, :2, :2] = False
        values, valid = spatial_flow_grid(flow, reliable, torch.tensor([[1., 2.]], dtype=torch.float64))
        self.assertFalse(valid[0, 0])
        self.assertEqual(values[0, 0].abs().sum().item(), 0)
        values, valid = spatial_flow_grid(flow, reliable, torch.tensor([[2., 1.]], dtype=torch.float64))
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
