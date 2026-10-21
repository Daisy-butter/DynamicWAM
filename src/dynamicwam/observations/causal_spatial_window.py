"""Shared causal capture window and image-flow contract (NumPy; lazy OpenCV)."""
from __future__ import annotations
import numpy as np


def causal_window_indices(times, current_frame, *, window_size=4, stride_seconds=.32):
    """Last available captures on a backwards physical-time grid, ending now.

    Frame order also bounds selection, so another capture with an equal clock
    but a later index is never admitted. Early episodes repeat their first
    capture; zero-time transitions must not supervise dynamics.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("capture times must be finite and nondecreasing")
    if not 0 <= int(current_frame) < len(times) or int(window_size) < 1:
        raise ValueError("invalid current frame/window size")
    if not np.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError("window stride must be positive")
    current_frame, window_size = int(current_frame), int(window_size)
    available = times[:current_frame + 1]
    targets = available[-1] - np.arange(window_size - 1, -1, -1) * stride_seconds
    selected = np.maximum(np.searchsorted(available, targets + 1e-10, side="right") - 1, 0)
    selected[-1] = current_frame
    actual = times[selected]
    dt = np.concatenate((np.zeros(1), np.diff(actual)))
    valid = np.concatenate((np.zeros(1, dtype=bool), (np.diff(selected) > 0) & (dt[1:] > 1e-9)))
    return selected.astype(np.int64), actual, dt, valid


def future_support_end_offsets(video_actual_times, observation_time):
    """Wan causal 9->3 encoding, drop condition: group ends are frames 4/8.

    These are receptive-field support ends, not invented latent-time centers.
    Earlier frames can influence both latents through the causal VAE state.
    """
    times = np.asarray(video_actual_times, dtype=np.float64)
    if times.shape != (8,) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("future VAE labels require eight ordered capture times")
    offsets = times[[3, 7]] - float(observation_time)
    if np.any(offsets <= 0):
        raise ValueError("future latent support must end after the current observation")
    return offsets


def causal_plan_position(times, eligible, current_frame, *, stride_seconds=.32):
    """Latest eligible prior fast-query observation, independent of slow memory.

    Return None at bootstrap instead of inventing a prior plan from the first
    frame or the current observation. No target/trajectory values are consumed.
    """
    times, eligible = np.asarray(times, dtype=np.float64), np.asarray(eligible)
    if eligible.ndim != 2 or eligible.shape[1] != 2 or stride_seconds <= 0:
        raise ValueError("invalid plan observation selection")
    frames = eligible[:, 0]
    candidates = np.flatnonzero((frames < int(current_frame)) &
        (times[frames] <= times[int(current_frame)] - stride_seconds + 1e-10))
    return int(candidates[-1]) if len(candidates) else None


def farneback_pair(previous_rgb, current_rgb):
    """Observed displacement on the CURRENT image grid, with F/B validity.

    This is an image measurement, not an object segmentation or a calibrated
    metric velocity. Padding and time validity are applied by the caller.
    """
    import cv2
    previous_rgb, current_rgb = np.asarray(previous_rgb), np.asarray(current_rgb)
    if previous_rgb.shape != current_rgb.shape or previous_rgb.ndim != 3 or previous_rgb.shape[-1] != 3:
        raise ValueError("flow pair must have matching HWC RGB shapes")
    if previous_rgb.dtype != np.uint8 or current_rgb.dtype != np.uint8:
        raise ValueError("flow input must be uint8 RGB")
    a = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    parameters = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, **parameters)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, **parameters)
    h, w = a.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # Backward flow locates each current pixel in the previous image. Negating
    # it gives observed motion with the same sign as forward flow, now aligned
    # with the current RGB features used by the spatial correction module.
    map_x, map_y = xx + backward[..., 0], yy + backward[..., 1]
    paired = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    residual = np.sum((backward + paired) ** 2, axis=-1)
    tolerance = .01 * np.sum(backward ** 2 + paired ** 2, axis=-1) + .5
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    valid &= np.isfinite(backward).all(axis=-1) & (residual <= tolerance)
    return np.where(valid[..., None], -backward, 0).astype(np.float32), valid


def composite_observation(frame):
    from dynamicwam.observations.multiview import split_composite, letterbox_view, imagenet_normalize
    rgb, raw, masks = [], [], []
    for view in split_composite(frame):
        image, valid = letterbox_view(view)
        rgb.append(imagenet_normalize(image).transpose(2, 0, 1))
        raw.append(np.rint(image * 255).astype(np.uint8))
        masks.append(valid)
    return np.asarray(rgb, dtype=np.float32), np.asarray(raw), np.asarray(masks)


def flow_pair_validity(displacement, confidence, previous_valid, current_valid):
    """Require current RGB and the corresponding previous bilinear footprint."""
    displacement = np.asarray(displacement)
    previous_valid, current_valid = np.asarray(previous_valid, bool), np.asarray(current_valid, bool)
    confidence = np.asarray(confidence, bool)
    h, w = current_valid.shape
    if displacement.shape != (h, w, 2) or previous_valid.shape != (h, w) or confidence.shape != (h, w):
        raise ValueError("dense flow backend geometry mismatch")
    if not np.isfinite(displacement).all():
        raise ValueError("dense flow contains nonfinite values")
    yy, xx = np.mgrid[:h, :w]
    # Current-grid displacement is -backward flow.
    px, py = xx - displacement[..., 0], yy - displacement[..., 1]
    x0, x1 = np.floor(px).astype(np.int64), np.ceil(px).astype(np.int64)
    y0, y1 = np.floor(py).astype(np.int64), np.ceil(py).astype(np.int64)
    inside = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    x0, x1 = x0.clip(0, w - 1), x1.clip(0, w - 1)
    y0, y1 = y0.clip(0, h - 1), y1.clip(0, h - 1)
    return confidence & current_valid & inside & previous_valid[y0, x0] & previous_valid[y0, x1] & previous_valid[y1, x0] & previous_valid[y1, x1]


