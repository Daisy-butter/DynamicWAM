#!/usr/bin/env python3
"""Check exact native planner outputs, input refresh, and cold/warm latency.

One explicitly reserved CUDA GPU is required. This does not run DOMINO or
measure success. Both engines use the same original model, precision and 10/2
schedule; only the eight-step action suffix is CUDA-graph captured.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("samples must be >= 2")
    import numpy as np
    import torch
    from dynamicwam.inference.native_plan_graph import NativePlanSampler
    from dynamicwam.models.native_wam_teacher import sample_native_wam
    from dynamicwam.models.causal_world_bridge import CausalWorldBridge
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from dynamicwam.models.multirate_types import SourceObservation, SourceNoise
    from dynamicwam.runtime.ciwam.wam.policy import DynamicWAMPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("select one idle CUDA GPU before running this test")
    torch.manual_seed(args.seed)
    report = dict(kind="native_plan_graph_parity", passed=False, cases=[], seed=args.seed,
                  atol=args.atol, rtol=args.rtol,
                  limitation="compute-only; synthetic packet is not a policy-success or physical reaction test")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        policy = DynamicWAMPolicy(args.deploy_config, project_root=args.project_root)
        policy.setup()
        policy.set_instruction(args.instruction)
        runner = policy.runner
        model = runner.model
        model.eval().requires_grad_(False)
        device = torch.device(runner.device)
        if device.type != "cuda":
            raise RuntimeError("deploy config must select CUDA")
        schedule = BridgeSchedule.from_flow_match(shift=runner.runtime.flow_match_shift,
            sigma_min=runner.runtime.flow_match_sigma_min,
            extra_one_step=runner.runtime.flow_match_extra_one_step,
            action_steps=runner.runtime.num_inference_steps,
            video_steps=runner.runtime.num_video_inference_steps)
        with np.load(args.packet, allow_pickle=False) as saved:
            packet = {key: np.array(saved[key], copy=True) for key in saved.files}
        frame = policy._frame_tensor(packet["frame"])
        state = runner._normalize_state(policy._state_tensor(packet["state"]))
        motion = runner._motion_batch(packet["motion_features"], packet["motion_interval_valid_mask"],
                                      packet["motion_acceleration_valid_mask"])
        text = runner._get_text_embeddings()

        def make_inputs():
            future, condition = runner._initialize_video_latent(frame, packet["flow_frames"])
            source = SourceObservation(condition, state, motion["absolute_motion_features"],
                motion["absolute_motion_interval_valid_mask"], motion["absolute_motion_acceleration_valid_mask"],
                text, source_id=0, sim_time=0.)
            return source, SourceNoise(future, runner._fresh_noisy_actions())

        def runner_reference(source, noise):
            # Authoritative HeadFlowRunner sampler, with only its input
            # preparation pinned to this supplied observation/noise. Scheduler
            # steps and forward dispatch are the untouched production path.
            overrides = dict(
                _initialize_video_latent=lambda *a, **k: (noise.future_latent.clone(), source.condition_latent),
                _fresh_noisy_actions=lambda *a, **k: noise.actions.clone(),
                _normalize_state=lambda *a, **k: source.state,
                _get_text_embeddings=lambda *a, **k: source.text_embeddings,
                _motion_batch=lambda *a, **k: dict(
                    absolute_motion_features=source.motion,
                    absolute_motion_interval_valid_mask=source.motion_interval_valid_mask,
                    absolute_motion_acceleration_valid_mask=source.motion_acceleration_valid_mask))
            original = {name: getattr(runner, name) for name in overrides}
            try:
                for name, value in overrides.items():
                    setattr(runner, name, value)
                return policy.sample(packet)
            finally:
                for name, value in original.items():
                    setattr(runner, name, value)

        def fingerprint():
            digest = hashlib.sha256()
            for name, value in model.named_parameters():
                digest.update(name.encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            return digest.hexdigest()

        def measure(function):
            for _ in range(3):
                function()
            torch.cuda.synchronize(device)
            rows = []
            for _ in range(args.samples):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                result = function()
                torch.cuda.synchronize(device)
                rows.append((time.perf_counter() - start) * 1000)
                del result
            return dict(mean_ms=float(np.mean(rows)), p50_ms=float(np.percentile(rows, 50)),
                        p95_ms=float(np.percentile(rows, 95)), samples_ms=rows)

        with torch.inference_mode():
            source, noise = make_inputs()
            before = fingerprint()
            sampler = NativePlanSampler(model, schedule)
            expected = sample_native_wam(model, source, noise, schedule).clone()
            authoritative = runner_reference(source, noise)
            normalized_reference_output = runner.runtime.action_normalizer.denormalize(expected).squeeze(0).float().cpu().numpy()
            np.testing.assert_array_equal(normalized_reference_output, authoritative)
            report["actual_headflowrunner_reference_exact"] = True
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            first, original_cache = sampler.sample(source, noise)
            torch.cuda.synchronize(device)
            report["cold_first_query_ms"] = (time.perf_counter() - start) * 1000
            snapshot = first.clone()
            torch.testing.assert_close(first, expected, atol=args.atol, rtol=args.rtol)
            causal = CausalWorldBridge(schedule)(model, source, noise, detach_cache=True)
            for generated, authoritative in zip((*causal.video_k, *causal.video_v),
                    (*original_cache["video_k"], *original_cache["video_v"]), strict=True):
                torch.testing.assert_close(generated, authoritative, atol=0, rtol=0)
            report["causal_bridge_matches_native_plan_cache_exact"] = True
            cases = [
                ("repeat", source, noise),
                ("new_action_noise", source, dataclasses.replace(noise, actions=torch.randn_like(noise.actions))),
                ("new_video_noise", source, dataclasses.replace(noise, future_latent=torch.randn_like(noise.future_latent))),
                ("current_state", dataclasses.replace(source, state=source.state + .025), noise),
                ("current_rgb_latent", dataclasses.replace(source, condition_latent=source.condition_latent + .1), noise),
                ("motion", dataclasses.replace(source, motion=source.motion + .02), noise),
                ("motion_masks", dataclasses.replace(source,
                    motion_interval_valid_mask=torch.zeros_like(source.motion_interval_valid_mask),
                    motion_acceleration_valid_mask=torch.zeros_like(source.motion_acceleration_valid_mask)), noise),
                ("language", dataclasses.replace(source, text_embeddings=[value * .95 for value in text]), noise),
                ("restore", source, noise),
            ]
            cache_snapshot = original_cache["video_v"][0].clone()
            for name, current, initial in cases:
                eager = sample_native_wam(model, current, initial, schedule)
                actual, _cache = sampler.sample(current, initial)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(actual, eager, atol=args.atol, rtol=args.rtol)
                np.testing.assert_array_equal(
                    runner.runtime.action_normalizer.denormalize(actual).squeeze(0).float().cpu().numpy(),
                    runner_reference(current, initial))
                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                for field, tensor in dict(initial_state=current.state,
                    absolute_motion_features=current.motion,
                    absolute_motion_interval_valid_mask=current.motion_interval_valid_mask,
                    absolute_motion_acceleration_valid_mask=current.motion_acceleration_valid_mask).items():
                    torch.testing.assert_close(sampler.static_batch[field], tensor, atol=0, rtol=0)
                torch.testing.assert_close(original_cache["video_v"][0], cache_snapshot, atol=0, rtol=0)
                if not torch.isfinite(actual).all():
                    raise AssertionError("nonfinite plan")
                report["cases"].append(dict(name=name,
                    max_abs_error=float((actual - eager).abs().max()),
                    rmse=float((actual - eager).square().mean().sqrt()),
                    output_change=float((actual - expected).abs().max())))
            if sampler.stats["captures"] != 1:
                raise AssertionError("same-shape input changes must update buffers, not recapture")
            disabled = NativePlanSampler(model, schedule, enabled=False)
            torch.testing.assert_close(disabled.sample(source, noise)[0], expected, atol=0, rtol=0)
            if fingerprint() != before:
                raise AssertionError("planner execution changed model weights or dtype")
            report["native_latent_core"] = measure(lambda: sample_native_wam(model, source, noise, schedule))
            report["graph_latent_core"] = measure(lambda: sampler.sample(source, noise))

            def policy_query(graph):
                current, initial = make_inputs()
                normalized = sampler.sample(current, initial)[0] if graph else sample_native_wam(model, current, initial, schedule)
                return runner.runtime.action_normalizer.denormalize(normalized).float().cpu().numpy()

            report["native_with_vae_output_copy"] = measure(lambda: policy_query(False))
            report["graph_with_vae_output_copy"] = measure(lambda: policy_query(True))
            report["timing_scope"] = {
                "latent_core": "same pre-encoded SourceObservation/noise, no VAE/T5/normalization/output copy",
                "with_vae_output_copy": "VAE encode existing frame/flow + noise + native 10/2 + denormalize + CPU result; text cached",
                "excluded": "camera acquisition, external preprocessing/Farneback, T5 cold load, DOMINO execution"}
            report.update(passed=True, stats=sampler.stats, weights_sha256=before,
                action_parameter_dtypes=sorted({str(p.dtype) for p in model.action_expert.parameters()}),
                motion_parameter_dtypes=sorted({str(p.dtype) for p in model.absolute_motion_tokens.parameters()}),
                video_time_embedding_dtype=str(next(model.compact_wan.video_model.wan_model.time_embedding.parameters()).dtype),
                video_precision=str(model.compact_wan.video_model.precision),
                input_noise_dtype=str(noise.actions.dtype), output_dtype=str(first.dtype),
                gpu=torch.cuda.get_device_name(device))
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
