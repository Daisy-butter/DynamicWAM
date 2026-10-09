#!/usr/bin/env python3
"""Two-GPU service time: slow bridge on cuda:0, fast query on cuda:1.

The fast conditioner is still randomly initialized. This measures placement
and the bridge copy, not task success.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation, pad_text_embeddings
from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.runtime.ciwam.wam.multirate_policy import packet_noise_generator
from profile_pretrain_latency import load_pretrain_policy


def _stats(rows: list[float]) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _sync_both(slow: torch.device, fast: torch.device) -> None:
    torch.cuda.synchronize(slow)
    torch.cuda.synchronize(fast)


def _measure(function, slow: torch.device, fast: torch.device, *, warmup: int, samples: int):
    for _ in range(warmup):
        function()
    _sync_both(slow, fast)
    rows = []
    for _ in range(samples):
        _sync_both(slow, fast)
        start = time.perf_counter()
        function()
        _sync_both(slow, fast)
        rows.append((time.perf_counter() - start) * 1000.0)
    return _stats(rows)


def _move(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _bridge_bytes(bridge) -> int:
    total = 0
    for layer in (*bridge.video_k, *bridge.video_v):
        total += int(layer.numel()) * int(layer.element_size())
    total += int(bridge.seq_lens.numel()) * int(bridge.seq_lens.element_size())
    return total


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two-GPU timing needs two visible CUDA devices")
    slow = torch.device("cuda:0")
    fast = torch.device("cuda:1")
    policy, runtime, _config, packet, loaded = load_pretrain_policy(slow)
    runner = policy.base.runner
    policy.fast_model.to(fast)
    policy._install_bridge(policy._build_bridge_completed(packet))
    resident = dataclasses.replace(
        policy.bridge,
        video_k=_move(policy.bridge.video_k, fast),
        video_v=_move(policy.bridge.video_v, fast),
        seq_lens=_move(policy.bridge.seq_lens, fast),
        grid_sizes=_move(policy.bridge.grid_sizes, fast),
    )
    bridge_state = policy.bridge_state.to(fast)
    text = [_move(item, fast) for item in runner._get_text_embeddings()]
    shapes = [tuple(layer.shape) for layer in policy.bridge.video_k]

    def publish_copy():
        copied = dataclasses.replace(
            policy.bridge,
            video_k=_move(policy.bridge.video_k, fast),
            video_v=_move(policy.bridge.video_v, fast),
            seq_lens=_move(policy.bridge.seq_lens, fast),
            grid_sizes=_move(policy.bridge.grid_sizes, fast),
        )
        torch.cuda.synchronize(fast)
        return copied

    def fast_once():
        torch.cuda.set_device(fast)
        views, masks = fast_views_from_composite(packet["fast_frame"])
        rgb = torch.from_numpy(views).unsqueeze(0).to(fast, non_blocking=False)
        state = runtime.action_normalizer.normalize(
            torch.as_tensor(packet["state"], dtype=torch.float32, device=fast).reshape(1, -1)
        )
        motion = torch.as_tensor(packet["motion_features"], dtype=torch.float32, device=fast).unsqueeze(0)
        interval_valid = torch.as_tensor(
            packet["motion_interval_valid_mask"], dtype=torch.bool, device=fast
        ).unsqueeze(0)
        acceleration_valid = torch.as_tensor(
            packet["motion_acceleration_valid_mask"], dtype=torch.bool, device=fast
        ).unsqueeze(0)
        text_tensor, text_mask = pad_text_embeddings(text)
        current = CurrentObservation(
            rgb=rgb,
            state=state,
            motion=motion,
            motion_interval_valid_mask=interval_valid,
            motion_acceleration_valid_mask=acceleration_valid,
            text_embeddings=text,
            age=BridgeAge(
                delta_sim_seconds=0.0,
                delta_actions=0,
                bridge_valid=True,
                state_delta=(state - bridge_state).reshape(-1),
            ),
            rgb_valid_mask=torch.from_numpy(masks).unsqueeze(0).to(fast, non_blocking=False),
        )
        noise = torch.randn(
            (1, int(runtime.chunk_size), int(runtime.model.config.action_dim)),
            device=fast,
            dtype=next(policy.fast_model.parameters()).dtype,
            generator=packet_noise_generator(packet, fast, branch="fast"),
        )
        with torch.inference_mode():
            memory = policy.fast_model.prepare_fast_memory(current, resident, text_tensor, text_mask)
            actions = policy.fast_model.sample_fast_actions(
                current,
                resident,
                memory,
                noise,
                denormalize=runtime.action_normalizer.denormalize,
            )
        if int(actions.shape[1]) != int(runtime.chunk_size):
            raise RuntimeError(f"unexpected fast chunk {tuple(actions.shape)}")
        return actions

    def bridge_once():
        torch.cuda.set_device(slow)
        policy._build_bridge(packet)

    overlap_device_ms = {"slow": [], "fast": []}

    def overlap_once():
        errors = []
        slow_start, slow_end = torch.cuda.Event(True), torch.cuda.Event(True)
        fast_start, fast_end = torch.cuda.Event(True), torch.cuda.Event(True)

        def slow_call():
            try:
                torch.cuda.set_device(slow)
                slow_start.record()
                policy._build_bridge(packet)
                slow_end.record()
                torch.cuda.synchronize(slow)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def fast_call():
            try:
                torch.cuda.set_device(fast)
                fast_start.record()
                fast_once()
                fast_end.record()
                torch.cuda.synchronize(fast)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=slow_call), threading.Thread(target=fast_call)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        overlap_device_ms["slow"].append(float(slow_start.elapsed_time(slow_end)))
        overlap_device_ms["fast"].append(float(fast_start.elapsed_time(fast_end)))

    with torch.inference_mode():
        fast_once()
        bridge_ms = _measure(bridge_once, slow, fast, warmup=2, samples=5)
        fast_ms = _measure(fast_once, slow, fast, warmup=2, samples=5)
        copy_ms = _measure(publish_copy, slow, fast, warmup=2, samples=5)
        overlap_ms = _measure(overlap_once, slow, fast, warmup=1, samples=4)

    report = {
        "kind": "two_gpu_service_time",
        "slow_gpu": torch.cuda.get_device_name(slow),
        "fast_gpu": torch.cuda.get_device_name(fast),
        "peer_access_fast_reads_slow": bool(torch.cuda.can_device_access_peer(1, 0)),
        "weight_source": loaded,
        "bridge_layer_shapes": shapes,
        "bridge_kv_bytes": _bridge_bytes(policy.bridge),
        "slow_bridge_ms": bridge_ms,
        "fast_query_ms": fast_ms,
        "bridge_copy_ms": copy_ms,
        "overlap_bridge_and_fast_ms": overlap_ms,
        "overlap_device_ms": {
            "slow": _stats(overlap_device_ms["slow"][1:]),
            "fast": _stats(overlap_device_ms["fast"][1:]),
        },
        "max_of_separate_ms": max(bridge_ms["mean_ms"], fast_ms["mean_ms"]),
        "slow_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(slow)),
        "fast_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(fast)),
        "limits": [
            "fast query uses a bridge already resident on the fast GPU",
            "copy is timed separately and is not inside the overlap window",
            "untrained fast conditioner; synthetic frames",
        ],
    }
    out = ROOT / "outputs" / "profile_two_gpu_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
