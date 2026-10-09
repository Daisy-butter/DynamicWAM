#!/usr/bin/env python3
"""Check CUDA-graph correctness on a real native checkpoint and observation.

This is an inference-engine check, not a policy-quality evaluation. Changing
--fast-steps-override only tests engine behavior; it does not create trained
few-step weights. A real bridge is built once from the supplied observation.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--fast-steps-override", type=int, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0 or args.samples < 2:
        parser.error("nonnegative tolerances and samples >= 2 required")
    return args


def tensor_fingerprint(named_tensors):
    """Full byte-level fingerprint outside all timing measurements."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from dynamicwam.inference.native_fast_graph import NativeFastGraph
    from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation, pad_text_embeddings
    from dynamicwam.observations.multiview import fast_views_from_composite
    from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, packet_noise_generator

    if not torch.cuda.is_available():
        raise RuntimeError("this check requires one explicitly available CUDA GPU")
    torch.manual_seed(args.seed)
    with np.load(args.packet, allow_pickle=False) as saved:
        required = ("frame", "fast_frame", "state", "flow_frames", "motion_features",
                    "motion_interval_valid_mask", "motion_acceleration_valid_mask", "sim_time")
        packet = {key: np.array(saved[key], copy=True) for key in required}
    packet["sim_time"] = float(packet["sim_time"].item())
    policy = MultiratePolicy(args.deploy_config, args.checkpoint, project_root=args.project_root,
                            scheduling="controlled_ratio")
    report = dict(kind="native_cuda_graph_correctness", checkpoint=str(Path(args.checkpoint).resolve()),
                  packet=str(Path(args.packet).resolve()), seed=args.seed,
                  atol=args.atol, rtol=args.rtol, cases=[], passed=False)
    graph = None
    try:
        policy.setup()
        policy.set_instruction(args.instruction)
        if not policy.contract.get("native_multirate"):
            raise RuntimeError("graph check requires a native multirate checkpoint")
        runner, model = policy.base.runner, policy.fast_model
        device = torch.device(runner.device)
        if device.type != "cuda":
            raise RuntimeError("deploy profile must use CUDA")
        trained_steps = len(model.fast_schedule.action_sigmas)
        if args.fast_steps_override is not None:
            model.fast_schedule = model.schedule.with_action_steps(args.fast_steps_override)
        steps = len(model.fast_schedule.action_sigmas)
        if steps not in (1, 2):
            raise RuntimeError("use native K=1/2 weights or an explicit engine-only fast-step override")
        if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise RuntimeError("the actual inference model must be frozen and in eval mode")
        report.update(gpu=torch.cuda.get_device_name(device), trained_fast_steps=trained_steps,
                      tested_fast_steps=steps, engine_only_override=args.fast_steps_override is not None)
        text_rows = runner._get_text_embeddings()
        packet.update(action_index=0, _query_index=0, _episode_seed=policy._episode_seed, _text=text_rows)
        with torch.inference_mode():
            policy._install_bridge(policy._build_bridge_completed(packet))
            bridge = policy.bridge
            if not isinstance(bridge.null_bridge, bool) or bridge.null_bridge:
                raise RuntimeError("this check exercises the native non-null source bridge")
            views, masks = fast_views_from_composite(packet["fast_frame"])
            state = runner._normalize_state(policy.base._state_tensor(packet["state"]))
            motion = runner._motion_batch(packet["motion_features"], packet["motion_interval_valid_mask"],
                                          packet["motion_acceleration_valid_mask"])
            current = CurrentObservation(
                rgb=torch.from_numpy(views).unsqueeze(0).to(device), state=state,
                motion=motion["absolute_motion_features"],
                motion_interval_valid_mask=motion["absolute_motion_interval_valid_mask"],
                motion_acceleration_valid_mask=motion["absolute_motion_acceleration_valid_mask"],
                text_embeddings=text_rows, rgb_valid_mask=torch.from_numpy(masks).unsqueeze(0).to(device),
                age=BridgeAge(0., 0, True, (state - policy.bridge_state).reshape(-1)))
            text, text_mask = pad_text_embeddings(text_rows)
            noise = runner._fresh_noisy_actions(generator=packet_noise_generator(packet, device, branch="fast"))
            base = (current, bridge, text, text_mask, noise)
            graph = NativeFastGraph(model, enabled=True, warmup=3)
            weight_hash = tensor_fingerprint(model.named_parameters())
            input_tensors = [("rgb", current.rgb), ("state", current.state), ("motion", current.motion),
                ("rgb_mask", current.rgb_valid_mask), ("text", text), ("text_mask", text_mask), ("noise", noise)]
            input_tensors += [(f"k{i}", value) for i, value in enumerate(bridge.video_k)]
            input_tensors += [(f"v{i}", value) for i, value in enumerate(bridge.video_v)]
            input_hash = tensor_fingerprint(input_tensors)

            def eager(inputs):
                obs, world, words, words_mask, initial = inputs
                memory = model.prepare_fast_memory(obs, world, words, words_mask)
                return model.sample_fast_actions(obs, world, memory, initial)

            def replay(inputs):
                return graph.sample(*inputs)

            def assert_copied_inputs(inputs):
                # Output equality alone cannot detect an ignored input when a
                # smoke checkpoint has an almost-zero conditioning residual.
                obs, world, words, words_mask, initial = inputs
                for name in ("rgb", "state", "motion", "motion_interval_valid_mask",
                             "motion_acceleration_valid_mask", "rgb_valid_mask"):
                    expected, actual = getattr(obs, name), getattr(graph.current, name)
                    if expected is None:
                        if actual is not None:
                            raise AssertionError(f"graph unexpectedly materialized {name}")
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in ("delta_actions", "delta_sim_seconds", "bridge_valid", "state_delta"):
                    actual = getattr(graph.current.age, name)
                    expected = torch.as_tensor(getattr(obs.age, name), device=actual.device, dtype=actual.dtype)
                    if name != "state_delta":
                        expected = expected.reshape(-1).expand_as(actual)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(graph.text, words, rtol=0, atol=0)
                torch.testing.assert_close(graph.text_mask, words_mask, rtol=0, atol=0)
                torch.testing.assert_close(graph.noise, initial, rtol=0, atol=0)
                torch.testing.assert_close(graph.bridge.seq_lens, world.seq_lens, rtol=0, atol=0)
                for actual, expected in zip((*graph.bridge.video_k, *graph.bridge.video_v),
                                            (*world.video_k, *world.video_v), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            baseline = eager(base).detach().clone()
            caller_owned_first = replay(base)
            base_captured = caller_owned_first.detach().clone()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(base_captured, baseline, atol=args.atol, rtol=args.rtol)
            if not graph.stats.get("enabled") or graph.stats.get("captures", 0) < 1:
                raise AssertionError("check accidentally used an eager fallback instead of CUDA graph replay")
            capture_count = graph.stats["captures"]

            changed_rgb = dataclasses.replace(current, rgb=current.rgb.flip(-1) + .1)
            changed_state = current.state + torch.linspace(-.15, .15, current.state.shape[-1], device=device)
            changed_robot = dataclasses.replace(current, state=changed_state,
                age=dataclasses.replace(current.age, state_delta=(changed_state - policy.bridge_state).reshape(-1)))
            changed_age = dataclasses.replace(current,
                age=dataclasses.replace(current.age, delta_actions=12, delta_sim_seconds=1.2))
            changed_rgb_mask = current.rgb_valid_mask.clone()
            changed_rgb_mask[:, 1] = False
            changed_masks = dataclasses.replace(current, rgb_valid_mask=changed_rgb_mask,
                motion_interval_valid_mask=torch.zeros_like(current.motion_interval_valid_mask),
                motion_acceleration_valid_mask=torch.zeros_like(current.motion_acceleration_valid_mask))
            changed_motion = dataclasses.replace(current, motion=current.motion + .25)
            changed_text = text * .9 + .1
            changed_text_mask = text_mask.clone()
            if changed_text_mask.shape[1] > 1:
                changed_text_mask[:, 1::2] = False
            changed_world = dataclasses.replace(bridge,
                video_k=tuple(k + .05 * torch.randn_like(k) for k in bridge.video_k),
                video_v=tuple(v + .10 * torch.randn_like(v) for v in bridge.video_v), source_id=16, sim_time=1.2)
            changed_noise = torch.randn_like(noise)
            cases = [
                ("same_input_repeat", base),
                ("fresh_rgb", (changed_rgb, bridge, text, text_mask, noise)),
                ("current_state", (changed_robot, bridge, text, text_mask, noise)),
                ("bridge_age", (changed_age, bridge, text, text_mask, noise)),
                ("rgb_and_motion_masks", (changed_masks, bridge, text, text_mask, noise)),
                ("motion_values", (changed_motion, bridge, text, text_mask, noise)),
                ("text_and_text_mask", (current, bridge, changed_text, changed_text_mask, noise)),
                ("replacement_bridge", (current, changed_world, text, text_mask, noise)),
                ("fresh_sampling_noise", (current, bridge, text, text_mask, changed_noise)),
                ("combined_updates", (changed_robot, changed_world, changed_text, changed_text_mask, changed_noise)),
                ("restore_original_inputs", base),
            ]
            for name, inputs in cases:
                expected = eager(inputs).detach().clone()
                actual = replay(inputs).detach().clone()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                assert_copied_inputs(inputs)
                torch.testing.assert_close(caller_owned_first, base_captured, atol=0, rtol=0)
                if not torch.isfinite(actual).all():
                    raise AssertionError(f"non-finite graph output in {name}")
                max_error = float((expected.float() - actual.float()).abs().max())
                eager_delta = float((expected.float() - baseline.float()).abs().max())
                captured_delta = float((actual.float() - base_captured.float()).abs().max())
                report["cases"].append(dict(name=name, max_abs_error=max_error,
                    eager_delta_from_original=eager_delta, graph_delta_from_original=captured_delta,
                    eager_input_response=eager_delta > 1e-6))
                if name in {"same_input_repeat", "restore_original_inputs"}:
                    torch.testing.assert_close(actual, base_captured, atol=0, rtol=0)
                if name in {"fresh_sampling_noise", "current_state", "replacement_bridge"} and eager_delta <= 1e-6:
                    raise AssertionError(f"{name} did not affect eager output; stale-input protection was not exercised")
            # Inference-mode tensors have no version counter. Exercise buffer
            # reuse too, so identity-based bridge caching cannot hide updates.
            changed_world.video_v[0].add_(.2 * torch.randn_like(changed_world.video_v[0]))
            reused_bridge_inputs = (current, changed_world, text, text_mask, noise)
            expected = eager(reused_bridge_inputs).detach().clone()
            actual = replay(reused_bridge_inputs).detach().clone()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            assert_copied_inputs(reused_bridge_inputs)
            torch.testing.assert_close(caller_owned_first, base_captured, atol=0, rtol=0)
            report["cases"].append(dict(name="in_place_bridge_buffer_update",
                max_abs_error=float((expected.float() - actual.float()).abs().max())))
            if graph.stats["captures"] != capture_count:
                raise AssertionError("same-shape content changes rebuilt the graph instead of updating captured inputs")
            if tensor_fingerprint(input_tensors) != input_hash:
                raise AssertionError("graph execution mutated caller-owned observation or source cache tensors")
            if tensor_fingerprint(model.named_parameters()) != weight_hash:
                raise AssertionError("graph capture/replay changed trained model parameters")

            def measure(function):
                for _ in range(3):
                    function(base)
                rows = []
                for _ in range(args.samples):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    function(base)
                    torch.cuda.synchronize(device)
                    rows.append((time.perf_counter() - started) * 1000.)
                return dict(mean_ms=float(np.mean(rows)), p50_ms=float(np.percentile(rows, 50)),
                            p95_ms=float(np.percentile(rows, 95)), samples_ms=rows)

            report.update(eager_timing=measure(eager), captured_timing=measure(replay),
                          graph_stats=dict(graph.stats), weights_unchanged_sha256=weight_hash,
                          caller_inputs_unchanged=True, passed=True,
                          limitations=["normalized action sampler only; excludes preprocessing, denormalization and slow refresh",
                                       "non-null native source bridges only", "no policy-success conclusion"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if graph is not None:
            graph.invalidate()
        policy.close()
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
