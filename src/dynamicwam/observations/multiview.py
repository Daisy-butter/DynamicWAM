"""Three-camera crops shared by the fast path in training and deployment.

The stored composite is the head frame stacked above the two wrists. Wrists are
half the head width and half the head height. The fast encoder never crops the
VAE canvas, which may have been letterboxed to a different size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamicwam.image import resize_with_padding

HEAD_SIZE = (240, 320)
WRIST_SIZE = (120, 160)
FAST_CANVAS = (160, 160)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class CompositeLayout:
    head_size: tuple[int, int] = HEAD_SIZE
    wrist_size: tuple[int, int] = WRIST_SIZE

    def __post_init__(self) -> None:
        head_h, head_w = self.head_size
        wrist_h, wrist_w = self.wrist_size
        if head_h <= 0 or head_w <= 0 or wrist_h <= 0 or wrist_w <= 0:
            raise ValueError(f"invalid composite layout: {self}")
        if wrist_h * 2 != head_h or wrist_w * 2 != head_w:
            raise ValueError(
                "wrist frames must be half the head height and width, got "
                f"head={self.head_size} wrist={self.wrist_size}"
            )

    @property
    def composite_size(self) -> tuple[int, int]:
        head_h, head_w = self.head_size
        wrist_h, _wrist_w = self.wrist_size
        return head_h + wrist_h, head_w


def split_composite(
    frame: np.ndarray,
    layout: CompositeLayout | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return head, left wrist, right wrist as HWC uint8 RGB."""
    spec = layout or CompositeLayout()
    if frame.ndim != 3 or int(frame.shape[2]) != 3:
        raise ValueError(f"composite frame must be HWC RGB, got {frame.shape}")
    expected_h, expected_w = spec.composite_size
    if tuple(frame.shape[:2]) != (expected_h, expected_w):
        raise ValueError(
            f"composite shape {frame.shape[:2]} does not match layout "
            f"{(expected_h, expected_w)}"
        )
    head_h, head_w = spec.head_size
    wrist_h, wrist_w = spec.wrist_size
    image = frame
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    head = image[:head_h, :head_w]
    left = image[head_h : head_h + wrist_h, :wrist_w]
    right = image[head_h : head_h + wrist_h, wrist_w : wrist_w * 2]
    return head, left, right


def letterbox_view(
    frame: np.ndarray,
    canvas: tuple[int, int] = FAST_CANVAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Letterbox one HWC uint8 view. Returns RGB float [0, 1] and a valid mask."""
    if frame.dtype != np.uint8:
        raise ValueError(f"fast views must stay uint8 until letterbox, got {frame.dtype}")
    padded = resize_with_padding(frame, canvas)
    valid = np.zeros(canvas, dtype=np.bool_)
    scale = min(canvas[0] / frame.shape[0], canvas[1] / frame.shape[1])
    new_h = max(1, int(frame.shape[0] * scale))
    new_w = max(1, int(frame.shape[1] * scale))
    top = (canvas[0] - new_h) // 2
    left = (canvas[1] - new_w) // 2
    valid[top : top + new_h, left : left + new_w] = True
    rgb = padded.astype(np.float32) / 255.0
    return rgb, valid


def imagenet_normalize(rgb01: np.ndarray) -> np.ndarray:
    """Normalize an HWC RGB image already scaled to [0, 1]."""
    return (rgb01 - IMAGENET_MEAN) / IMAGENET_STD


def fast_views_from_composite(
    frame: np.ndarray,
    layout: CompositeLayout | None = None,
    canvas: tuple[int, int] = FAST_CANVAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[3,3,H,W]`` ImageNet-normalized views and ``[3,H,W]`` masks."""
    views = []
    masks = []
    for part in split_composite(frame, layout):
        rgb, mask = letterbox_view(part, canvas)
        views.append(imagenet_normalize(rgb).transpose(2, 0, 1))
        masks.append(mask)
    return np.stack(views, axis=0).astype(np.float32), np.stack(masks, axis=0)
