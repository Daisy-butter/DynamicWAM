#!/usr/bin/env python3
"""Service-time check before multirate training.

Loads the released absolute-motion checkpoint, attaches an untrained fast
conditioner of the deployment shape, and times three CUDA paths on synthetic
frames: the synchronous 10-step sampler, the two-step causal bridge, and the
isolated 10-step fast query. Action values are not meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
import torch

from dynamicwam.inference.fast_runtime import finalize_inference, isolated_fast_runtime
from dynamicwam.inference.model_loader import build_runtime_from_config, load_deploy_config
from dynamicwam.inference.multirate_model_loader import install_full_checkpoint, load_checkpoint_payload
from dynamicwam.models.fast_conditioner import FastConditioner
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.runtime.ciwam.wam.flow_runner import HeadFlowRunner
from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, _default_fast_config
from dynamicwam.runtime.ciwam.wam.multirate_control import forward_budget


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "configs" / "absolute_motion_v2.yaml"
CHECKPOINT = ROOT / "external" / "checkpoints" / "DynamicWAM_full.pt"


def _stats(rows: list[float]) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _measure(device: torch.device, function, *, warmup: int, samples: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    rows = []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        rows.append((time.perf_counter() - start) * 1000.0)
    return _stats(rows)


def load_pretrain_policy(device: torch.device):
    """Load the released checkpoint and an untrained fast conditioner."""

    if device.type != "cuda":
        raise RuntimeError("this timing check needs a CUDA device")
    config = load_deploy_config(DEPLOY)
    runtime = build_runtime_from_config(config)
    runner = HeadFlowRunner(runtime=runtime)
    runner.set_instruction("pick up the object")
    payload = load_checkpoint_payload(CHECKPOINT)
    schedule = BridgeSchedule.from_flow_match(
        shift=float(runtime.flow_match_shift),
        sigma_min=float(runtime.flow_match_sigma_min),
        extra_one_step=bool(runtime.flow_match_extra_one_step),
        action_steps=int(runtime.num_inference_steps),
        video_steps=2,
    )
    fast_cfg = _default_fast_config()
    precision = runner.model.compact_wan.video_model.precision
    wrapper = MultirateWAM(
        runner.model,
        FastConditioner(**fast_cfg).to(device),
        schedule,
    ).to(device)
    loaded = install_full_checkpoint(wrapper, payload, allow_v2_init=True)
    finalize_inference(wrapper.fast_conditioner, device=device, dtype=precision)
    wrapper.eval().requires_grad_(False)
    expert = runner.model.action_expert.config
    qkv = runner.model.action_expert.blocks[0].wan_action_qkv
    wan_heads = int(qkv.shape[1])
    wan_head_dim = int(qkv.shape[3])
    motion = runner.model.absolute_motion_tokens
    windows = [
        [int(item) for item in pair]
        for pair in runner.model.action_attention_spec.window_sizes
    ]
    modules = {
        "schedule": schedule.to_config(),
        "architecture": "reference",
        "action_expert": {
            "dim": int(expert.dim),
            "ffn_dim": int(expert.ffn_dim),
            "num_layers": int(expert.num_layers),
            "state_dim": int(expert.state_dim),
            "action_dim": int(expert.action_dim),
            "chunk_size": int(expert.chunk_size),
        },
        "wan_attention": {
            "dim": wan_heads * wan_head_dim,
            "num_heads": wan_heads,
            "head_dim": wan_head_dim,
            "precision": "bfloat16",
            "window_sizes": windows,
        },
        "motion": {
            "history_count": int(motion.history_count),
            "mean": [float(value) for value in motion.feature_mean.detach().cpu().tolist()],
            "scale": [float(value) for value in motion.feature_scale.detach().cpu().tolist()],
        },
        "fast_conditioner": fast_cfg,
    }
    fast_model = isolated_fast_runtime(
        {"model": {key: value.detach().cpu() for key, value in wrapper.state_dict().items()},
         "modules": modules, "global_step": int(payload.get("global_step", 0))},
        device=device,
        dtype=precision,
    )
    policy = MultiratePolicy(str(DEPLOY), str(CHECKPOINT), project_root=str(ROOT))
    policy.base.runner = runner
    policy.wrapper = wrapper
    policy.fast_model = fast_model
    policy.contract = {"clock_mode": "action_only", "execute_horizon": 4, "max_age_actions": 12}

    height, width = runtime.composite_frame_size
    flow = runtime.head_flow_config
    flow_h, flow_w = (int(value) for value in flow["compute_size"])
    text = runner._get_text_embeddings()
    packet = {
        "frame": np.zeros((height, width, 3), dtype=np.uint8),
        "fast_frame": np.zeros((height, width, 3), dtype=np.uint8),
        "state": np.zeros(int(runtime.model.config.state_dim), dtype=np.float32),
        "flow_frames": np.zeros((int(flow["count"]), flow_h, flow_w, 3), dtype=np.uint8),
        "motion_features": np.zeros((int(flow["count"]), 12), dtype=np.float32),
        "motion_interval_valid_mask": np.ones(int(flow["count"]), dtype=np.bool_),
        "motion_acceleration_valid_mask": np.ones(int(flow["count"]), dtype=np.bool_),
        "sim_time": 0.0,
        "action_index": 0,
        "_query_index": 0,
        "_episode_seed": 1,
        "_text": text,
    }
    return policy, runtime, config, packet, loaded


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this timing check needs a CUDA device")
    device = torch.device("cuda")
    policy, runtime, config, packet, loaded = load_pretrain_policy(device)
    runner = policy.base.runner
    frame = policy.base._frame_tensor(packet["frame"])

    def sync_once():
        runner.sample_chunk(
            first_frame=frame,
            state=policy.base._state_tensor(packet["state"]),
            flow_frames=packet["flow_frames"],
            motion_features=packet["motion_features"],
            motion_interval_valid_mask=packet["motion_interval_valid_mask"],
            motion_acceleration_valid_mask=packet["motion_acceleration_valid_mask"],
        )

    def bridge_once():
        policy._install_bridge(policy._build_bridge_completed(packet))

    def fast_once():
        policy._sample_fast(packet, 0)

    slow_stream = torch.cuda.Stream(device=device)
    fast_stream = torch.cuda.Stream(device=device)

    def overlap_once():
        errors = []

        def slow():
            try:
                with torch.cuda.stream(slow_stream):
                    policy._build_bridge(packet)
                slow_stream.synchronize()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def fast():
            try:
                with torch.cuda.stream(fast_stream):
                    policy._sample_fast(packet, 0)
                fast_stream.synchronize()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=slow), threading.Thread(target=fast)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    with torch.inference_mode():
        bridge_once()
        sync_ms = _measure(device, sync_once, warmup=2, samples=5)
        bridge_ms = _measure(device, bridge_once, warmup=2, samples=5)
        fast_ms = _measure(device, fast_once, warmup=2, samples=5)
        overlap_ms = _measure(device, overlap_once, warmup=1, samples=4)

    horizon = 4
    ratio = 4
    chunk = int(runtime.chunk_size)
    interval_ms = float(config["inference"]["action_interval_ms"])
    budget = forward_budget(execute_horizon=horizon, refresh_queries=ratio, action_steps=int(runtime.num_inference_steps), joint_steps=2)
    report = {
        "kind": "pretrain_service_time",
        "gpu": torch.cuda.get_device_name(device),
        "checkpoint": str(CHECKPOINT),
        "weight_source": loaded,
        "fast_conditioner": "deployment shape, randomly initialized",
        "sync_sampler_ms": sync_ms,
        "slow_bridge_ms": bridge_ms,
        "fast_query_ms": fast_ms,
        "overlap_bridge_and_fast_ms": overlap_ms,
        "serial_sum_ms": bridge_ms["mean_ms"] + fast_ms["mean_ms"],
        "amortized_ms_per_executed_action": {
            "multirate_if_bridge_fully_hidden": fast_ms["mean_ms"] / horizon,
            "multirate_serial_controlled": (bridge_ms["mean_ms"] / ratio + fast_ms["mean_ms"]) / horizon,
            "sync_replanning_every_horizon": sync_ms["mean_ms"] / horizon,
            "sync_one_long_chunk": sync_ms["mean_ms"] / chunk,
        },
        "configured_execute_window_ms": horizon * interval_ms,
        "forward_counts_per_16_actions": budget,
        "limits": [
            "synthetic frames and motion; no simulator, rendering, or optical flow",
            "untrained fast conditioner; timings are service time, not task latency",
            "one GPU; overlap is wall clock of the two forwards together",
        ],
    }
    out = ROOT / "outputs" / "profile_pretrain_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
