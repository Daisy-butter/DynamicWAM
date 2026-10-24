"""Small spatial vision trunk with explicit, offline pretrained provenance.

The module/key layout matches torchvision ResNet18 through layer3. It does not
import torchvision, download weights, crop images, or substitute random weights
when a checkpoint is unavailable. ``test_fixture=True`` is for CPU tests only.
"""

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = None
        self.stride = stride
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)


class ResNet18Stage3(nn.Module):
    """256-channel stride-16 map; convolution/affine trainable, BN stats frozen."""

    stride = 16
    out_channels = 256

    def __init__(self, weights_path=None, weights_sha256=None, *, test_fixture=False):
        super().__init__()
        self.test_fixture = bool(test_fixture)
        if self.test_fixture and (weights_path is not None or weights_sha256 is not None):
            raise ValueError("test_fixture cannot masquerade as pretrained initialization")
        if not self.test_fixture and (weights_path is None or weights_sha256 is None):
            raise ValueError("pretrained ResNet18 requires an explicit local path and SHA256")
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, 2), _BasicBlock(128, 128))
        self.layer3 = nn.Sequential(_BasicBlock(128, 256, 2), _BasicBlock(256, 256))
        if self.test_fixture:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            self.weight_identity = "explicit_random_test_fixture"
        else:
            path = Path(weights_path).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ValueError("vision checkpoint must be an existing absolute file path")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != str(weights_sha256).lower():
                raise ValueError("vision checkpoint SHA256 mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
                raise ValueError("vision checkpoint must be a bare torchvision ResNet18 state_dict")
            required = set(self.state_dict())
            unexpected = set(state) - required
            # A full official ResNet18 contains an unused layer4 and classifier.
            if any(not (key.startswith("layer4.") or key in ("fc.weight", "fc.bias")) for key in unexpected):
                raise ValueError("unexpected vision state_dict keys")
            self.load_state_dict({key: state[key] for key in required if key in state}, strict=True)
            self.weight_identity = actual
        self.pretrained = not self.test_fixture
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # Batch size / camera availability must not change inference statistics.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        return self.layer3(self.layer2(self.layer1(x)))


def spatial_flow_grid(flow_xy: Tensor, reliable: Tensor, interval: Tensor, *, grid_size=4):
    """Pool an already-computed causal optical flow; never estimates new flow.

    Input flow is [B,2,H,W] pixel displacement, reliability is [B,H,W], and
    interval is float64 [B,2] physical seconds. Output is [B,grid_size**2,5]
    (dx,dy,vx,vy,reliable_fraction), plus a task-time availability mask. Invalid
    pixels have no contribution; partial edge bins are retained by adaptive
    pooling. The caller additionally checks the interval against query time.
    """
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2 or reliable.shape != (flow_xy.shape[0], *flow_xy.shape[-2:]):
        raise ValueError("flow/reliability must be [B,2,H,W] and [B,H,W]")
    if interval.dtype != torch.float64 or interval.shape != (flow_xy.shape[0], 2):
        raise ValueError("flow interval must be float64 [B,2]")
    dt = interval[:, 1] - interval[:, 0]
    time_valid = torch.isfinite(interval).all(-1) & (dt > 0)
    pixel_valid = reliable.bool() & torch.isfinite(flow_xy).all(1) & time_valid[:, None, None]
    coverage = F.adaptive_avg_pool2d(pixel_valid[:, None].float(), grid_size)
    clean = torch.where(pixel_valid[:, None], flow_xy.float(), torch.zeros_like(flow_xy, dtype=torch.float32))
    displacement = F.adaptive_avg_pool2d(clean, grid_size) / coverage.clamp_min(1e-8)
    velocity = displacement / torch.where(time_valid, dt, torch.ones_like(dt)).float()[:, None, None, None]
    values = torch.cat((displacement, velocity, coverage), dim=1).flatten(2).transpose(1, 2)
    valid = coverage.flatten(1) > 0
    return torch.where(valid[..., None], values, torch.zeros_like(values)), valid
