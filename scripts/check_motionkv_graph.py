#!/usr/bin/env python3
"""Real-checkpoint MotionKV/RTC eager-vs-CUDA-Graph check, never a CPU surrogate.

Uses the native graph check's recorded observation fixture and byte fingerprint.
Run only after GPU preflight succeeds. This script performs no device recovery.
The output JSON belongs on the host's verified SSD wuruihan directory. A passing
result validates inference plumbing, not policy quality or asynchronous control.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from check_native_fast_graph import tensor_fingerprint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("deploy-config", "checkpoint", "packet", "instruction", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--fast-steps-override", type=int, choices=(1, 2, 4, 10))
    p.add_argument("--prefix-length", type=int, default=4)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--atol", type=float, default=2e-3)
    p.add_argument("--rtol", type=float, default=2e-3)
    a = p.parse_args()
    if a.samples < 2 or a.prefix_length < 1 or a.atol < 0 or a.rtol < 0:
        p.error("samples >= 2, prefix-length >= 1 and nonnegative tolerances required")
    return a


def main():
    args = parse_args()
    import numpy as np
    import torch
    from dynamicwam.inference.native_fast_graph import NativeFastGraph
    from dynamicwam.models.multirate_types import BridgeAge, CurrentObservation, pad_text_embeddings
    from dynamicwam.models.rtc_prefix import PrefixCondition, prefix_mask
    from dynamicwam.observations.multiview import fast_views_from_composite
    from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy, packet_noise_generator
    from dynamicwam.runtime.domino_continuous.motionkv_policy import pad_rtc_text

    if not torch.cuda.is_available():
        raise RuntimeError("requires one preflight-approved CUDA GPU; CPU tests cannot establish capture parity")
    torch.manual_seed(args.seed)
    with np.load(args.packet, allow_pickle=False) as saved:
        keys = ("frame", "fast_frame", "state", "flow_frames", "motion_features",
                "motion_interval_valid_mask", "motion_acceleration_valid_mask", "sim_time")
        packet = {k: np.array(saved[k], copy=True) for k in keys}
    packet["sim_time"] = float(packet["sim_time"].item())
    policy = MultiratePolicy(args.deploy_config, args.checkpoint, project_root=args.project_root,
        scheduling="controlled_ratio", native_cuda_graph=False,
        allow_zero_prefix_ablation=True)
    report = dict(kind="motionkv_rtc_native_graph", checkpoint=str(Path(args.checkpoint).resolve()),
        packet=str(Path(args.packet).resolve()), seed=args.seed, cases=[], passed=False,
        limitations=["sampler only: excludes preprocessing, slow refresh and physical execution",
                     "queue adapter must separately establish asynchronous train/deploy timing parity",
                     "step override is engine-only and does not create a trained distilled model"])
    graph = None
    try:
        policy.setup()
        policy.set_instruction(args.instruction)
        runner, model = policy.base.runner, policy.fast_model
        device = torch.device(runner.device)
        if device.type != "cuda" or not getattr(model.fast_conditioner, "motion_kv", False):
            raise RuntimeError("requires actual MotionKV checkpoint on CUDA")
        trained_prefix_max = int(policy.contract.get("rtc", {}).get("max_prefix_actions", 0))
        if trained_prefix_max != int(getattr(model, "rtc_prefix_max", 0)):
            raise RuntimeError("inference loader failed to restore RTC prefix maximum from checkpoint contract")
        if not 1 <= args.prefix_length <= trained_prefix_max:
            raise RuntimeError("requested prefix case exceeds checkpoint's trained RTC range")
        trained_steps = len(model.fast_schedule.action_sigmas)
        if args.fast_steps_override is not None:
            model.fast_schedule = model.schedule.with_action_steps(args.fast_steps_override)
        report.update(gpu=torch.cuda.get_device_name(device), trained_fast_steps=trained_steps,
            tested_fast_steps=len(model.fast_schedule.action_sigmas), trained_prefix_max=trained_prefix_max,
            tested_prefix=args.prefix_length, engine_only_override=args.fast_steps_override is not None)
        text_rows = runner._get_text_embeddings()
        packet.update(action_index=0, _query_index=0, _episode_seed=policy._episode_seed, _text=text_rows)
        with torch.inference_mode():
            policy._install_bridge(policy._build_bridge_completed(packet))
            world = policy.bridge
            if world.null_bridge is not False or not world.all_video_tokens_valid:
                raise RuntimeError("fixture must produce a non-null dense real world cache")
            views, masks = fast_views_from_composite(packet["fast_frame"])
            state = runner._normalize_state(policy.base._state_tensor(packet["state"]))
            motion = runner._motion_batch(packet["motion_features"], packet["motion_interval_valid_mask"], packet["motion_acceleration_valid_mask"])
            current = CurrentObservation(rgb=torch.from_numpy(views).unsqueeze(0).to(device), state=state,
                motion=motion["absolute_motion_features"],
                motion_interval_valid_mask=motion["absolute_motion_interval_valid_mask"],
                motion_acceleration_valid_mask=motion["absolute_motion_acceleration_valid_mask"],
                text_embeddings=text_rows, rgb_valid_mask=torch.from_numpy(masks).unsqueeze(0).to(device),
                age=BridgeAge(0., 0, True, (state - policy.bridge_state).reshape(-1)))
            # Match the real serving wrapper's shape-stable masked text layout.
            text, text_mask = pad_rtc_text(text_rows)
            report['serving_text_shape'] = list(text.shape)
            noise = runner._fresh_noisy_actions(generator=packet_noise_generator(packet, device, branch="fast"))
            batch = noise.shape[0]
            if args.prefix_length >= noise.shape[1]:
                raise RuntimeError("fixture has no supervised suffix at requested delay")
            d0 = PrefixCondition(torch.zeros_like(noise), torch.zeros(batch, device=device, dtype=torch.long))
            prefix = PrefixCondition(noise.clone() * .1,
                torch.full((batch,), args.prefix_length, device=device, dtype=torch.long))
            changed_prefix = PrefixCondition(prefix.actions + .125, prefix.lengths.clone())
            changed_state = state + torch.linspace(-.1, .1, state.shape[-1], device=device)
            robot = dataclasses.replace(current, state=changed_state,
                age=dataclasses.replace(current.age, state_delta=(changed_state - policy.bridge_state).reshape(-1)))
            moved = dataclasses.replace(current, motion=current.motion + .25)
            refreshed = dataclasses.replace(world,
                video_k=tuple(k + .05 * torch.randn_like(k) for k in world.video_k),
                video_v=tuple(v + .1 * torch.randn_like(v) for v in world.video_v))
            graph = NativeFastGraph(model, enabled=True, warmup=3)
            parameters_hash = tensor_fingerprint(model.named_parameters())
            source_inputs = [("state", state), ("motion", current.motion), ("noise", noise),
                ("prefix_actions", prefix.actions), ("prefix_lengths", prefix.lengths)]
            source_inputs += [(f"k{i}", x) for i, x in enumerate(world.video_k)]
            source_inputs += [(f"v{i}", x) for i, x in enumerate(world.video_v)]
            source_hash = tensor_fingerprint(source_inputs)

            def eager(obs, cache, cond):
                memory = model.prepare_fast_memory(obs, cache, text, text_mask)
                return model.sample_fast_actions(obs, cache, memory, noise, prefix=cond)

            def replay(obs, cache, cond):
                return graph.sample(obs, cache, text, text_mask, noise, prefix=cond)

            first = replay(current, world, None)
            first_snapshot = first.clone()
            torch.testing.assert_close(first, eager(current, world, None), atol=args.atol, rtol=args.rtol)
            captures = graph.stats["captures"]
            cases = [("none_to_d0", current, world, d0),
                ("committed_prefix", current, world, prefix),
                ("prefix_values_changed", current, world, changed_prefix),
                ("current_state_changed", robot, world, prefix),
                ("current_motion_changed", moved, world, prefix),
                ("world_kv_replaced", current, refreshed, prefix),
                ("combined_values_changed", robot, refreshed, changed_prefix),
                ("return_to_none_clears_previous_prefix", current, world, None),
                ("return_to_d0", current, world, d0)]
            for name, obs, cache, cond in cases:
                expected = eager(obs, cache, cond)
                actual = replay(obs, cache, cond)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                torch.testing.assert_close(first, first_snapshot, atol=0, rtol=0)
                effective = d0 if cond is None else cond
                torch.testing.assert_close(graph.prefix.actions, effective.actions, atol=0, rtol=0)
                torch.testing.assert_close(graph.prefix.lengths, effective.lengths, atol=0, rtol=0)
                torch.testing.assert_close(graph.current.state, obs.state, atol=0, rtol=0)
                torch.testing.assert_close(graph.current.motion, obs.motion, atol=0, rtol=0)
                for src, dst in zip((*cache.video_k, *cache.video_v), (*graph.bridge.video_k, *graph.bridge.video_v), strict=True):
                    torch.testing.assert_close(src, dst, atol=0, rtol=0)
                mask = prefix_mask(effective).unsqueeze(-1).expand_as(actual)
                torch.testing.assert_close(actual[mask], effective.actions[mask], atol=0, rtol=0)
                if not bool(torch.isfinite(actual).all()):
                    raise AssertionError("graph produced nonfinite actions")
                report["cases"].append(dict(name=name, max_abs_error=float((actual.float() - expected.float()).abs().max())))
            # Reusing caller storage must still copy fresh values into capture.
            changed_prefix.actions.add_(.1)
            actual = replay(current, world, changed_prefix)
            expected = eager(current, world, changed_prefix)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            torch.testing.assert_close(graph.prefix.actions, changed_prefix.actions, atol=0, rtol=0)
            report["cases"].append(dict(name="in_place_prefix_buffer_update", max_abs_error=float((actual.float() - expected.float()).abs().max())))
            if graph.stats["captures"] != captures or captures != 1 or not graph.stats["enabled"]:
                raise AssertionError("prefix/content changes must use one actual CUDA Graph capture")
            bad = PrefixCondition(prefix.actions, torch.full_like(prefix.lengths, trained_prefix_max + 1))
            try:
                replay(current, world, bad)
            except ValueError:
                report["rejects_prefix_beyond_training_range"] = True
            else:
                raise AssertionError("graph silently accepted an untrained prefix length")
            if tensor_fingerprint(source_inputs) != source_hash or tensor_fingerprint(model.named_parameters()) != parameters_hash:
                raise AssertionError("graph mutated caller inputs or model parameters")

            def measure(fn, cond):
                for _ in range(3):
                    fn(current, world, cond)
                samples = []
                for _ in range(args.samples):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    fn(current, world, cond)
                    torch.cuda.synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000)
                return dict(p50_ms=float(np.percentile(samples, 50)), p95_ms=float(np.percentile(samples, 95)), samples_ms=samples)

            report.update(eager_d0=measure(eager, d0), graph_d0=measure(replay, d0),
                eager_prefix=measure(eager, prefix), graph_prefix=measure(replay, prefix),
                timing_boundary="synchronized wall latency includes public sample prefix validation and input copies",
                graph_stats=dict(graph.stats), weights_unchanged_sha256=parameters_hash, caller_inputs_unchanged=True, passed=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if graph is not None:
            graph.invalidate()
        policy.close()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
