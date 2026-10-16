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
    p.add_argument("--concurrent-slow", action="store_true",
                   help="one actual full slow refresh while five graph calls consume the old cache")
    a = p.parse_args()
    if a.samples < 2 or a.prefix_length < 1 or a.atol < 0 or a.rtol < 0:
        p.error("samples >= 2, prefix-length >= 1 and nonnegative tolerances required")
    return a


def concurrent_slow_probe(*, policy, packet, model, graph, current, world,
                          prefix, replay, eager, atol, rtol):
    """Bounded real worker/stream probe, with honest unchanged observation time.

    Interval overlap proves fast consumption before slow completion; it cannot
    prove simultaneous kernel execution or camera/controller real-time capacity.
    """
    import threading
    import numpy as np
    import torch
    from dynamicwam.inference.native_fast_graph import _BRIDGE_CONTEXT_TENSORS
    from dynamicwam.runtime.ciwam.wam.fast_slow_worker import LatestWorldWorker
    from dynamicwam.runtime.ciwam.wam.multirate_policy import capture_observation_packet

    if policy.worker is not None:
        raise RuntimeError("concurrency probe requires ownership of its one temporary worker")
    device = current.state.device
    expected = eager(current, world, prefix).clone()
    anchor = torch.cuda.Event(enable_timing=True)
    anchor.record(torch.cuda.current_stream(device)); anchor.synchronize()
    producer = torch.cuda.Stream(device=device)
    policy._producer_stream = producer
    slow_start = torch.cuda.Event(enable_timing=True)
    slow_end = torch.cuda.Event(enable_timing=True)
    started = threading.Event()
    trace = {}
    captures = graph.stats['captures']
    frozen = [(f'k{i}', x) for i, x in enumerate(world.video_k)]
    frozen += [(f'v{i}', x) for i, x in enumerate(world.video_v)]
    frozen += [(name, getattr(world, name)) for name in _BRIDGE_CONTEXT_TENSORS
               if torch.is_tensor(getattr(world, name))]
    frozen_hash = tensor_fingerprint(frozen)
    snapshot = capture_observation_packet(packet)

    def build(observation):
        with torch.cuda.device(device), torch.inference_mode(), torch.cuda.stream(producer):
            trace['slow_started_monotonic'] = time.monotonic()
            slow_start.record(producer)
            started.set()
            try:
                value = policy._build_bridge_completed(observation)
                # The serving subclass prepares derived source context before
                # publication. Include that actual work on the same owned stream.
                cache = model.fast_conditioner.prepare_source_cache(value[0])
                value = (cache, *value[1:])
            finally:
                slow_end.record(producer)
                slow_end.synchronize()
                trace['slow_completed_monotonic'] = time.monotonic()
        trace['producer_completion_event_ready'] = slow_end.query()
        return value

    worker = LatestWorldWorker(build)
    policy.worker = worker  # Actual builder selects its dedicated producer stream.
    calls, outputs = [], []
    try:
        submitted = time.monotonic()
        worker.request(1, snapshot)
        if not started.wait(timeout=30.):
            raise TimeoutError('slow worker did not begin within its bounded startup interval')
        for index in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin = time.monotonic()
            pending = worker.latest() is None
            start.record(torch.cuda.current_stream(device))
            result = replay(current, world, prefix)
            end.record(torch.cuda.current_stream(device))
            # Wait only for this owned fast stream; no device-wide fence.
            end.synchronize()
            finish = time.monotonic()
            outputs.append(result.clone())
            calls.append(dict(index=index, slow_unpublished_at_start=pending,
                started_monotonic=begin, completed_monotonic=finish,
                wall_ms=(finish-begin)*1000, device_ms=start.elapsed_time(end),
                timeline_start_ms=anchor.elapsed_time(start), timeline_end_ms=anchor.elapsed_time(end)))
        ready = worker.wait_for(1, timeout=120.)
        if ready.source_index != 1 or ready.value[0] is world:
            raise AssertionError('actual refresh did not publish a distinct completed cache')
        if not slow_end.query() or not trace.get('producer_completion_event_ready'):
            raise AssertionError('slow cache published before its owned CUDA event completed')
        for result in outputs:
            torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
        refreshed_expected = eager(current, ready.value[0], prefix)
        refreshed_actual = replay(current, ready.value[0], prefix)
        refreshed_done = torch.cuda.Event()
        refreshed_done.record(torch.cuda.current_stream(device)); refreshed_done.synchronize()
        torch.testing.assert_close(refreshed_actual, refreshed_expected, atol=atol, rtol=rtol)
        if tensor_fingerprint(frozen) != frozen_hash:
            raise AssertionError('concurrent producer modified the cache consumed by the fast path')
        if graph.stats['captures'] != captures:
            raise AssertionError('concurrent slow refresh triggered fast graph recapture')
        before = sum(call['slow_unpublished_at_start'] for call in calls)
        if before == 0:
            raise AssertionError('no fast call began before the actual slow refresh completed')
        result = dict(passed=True, calls=calls, fast_calls_before_slow_publication=before,
            submitted_monotonic=submitted, **trace,
            published_monotonic=ready.ready_monotonic, published_generation=ready.generation,
            completed_cache_consumed=True,
            slow_wall_ms=(trace['slow_completed_monotonic']-trace['slow_started_monotonic'])*1000,
            slow_device_interval_ms=slow_start.elapsed_time(slow_end),
            slow_timeline_start_ms=anchor.elapsed_time(slow_start), slow_timeline_end_ms=anchor.elapsed_time(slow_end),
            fast_p50_ms=float(np.percentile([x['wall_ms'] for x in calls],50)),
            fast_p95_ms=float(np.percentile([x['wall_ms'] for x in calls],95)),
            old_cache_unchanged_sha256=frozen_hash, fast_captures_before=captures,
            fast_captures_after=graph.stats['captures'], producer_stream_distinct=producer.cuda_stream != torch.cuda.current_stream(device).cuda_stream,
            source_observation_time=float(packet['sim_time']), source_action_index=int(packet['action_index']),
            source_time_semantics='same real recorded observation reprocessed; no fabricated physical advance',
            timing_boundary='real full slow Video/Action/VAE plus derived context; graph fast on last complete immutable cache',
            proves_kernel_concurrency=False, proves_continuous_1x=False)
    finally:
        if not worker.close(timeout=120.):
            raise TimeoutError('concurrency probe producer did not drain; retain model until outer process cleanup')
        policy.worker = None
    result['worker_closed'] = True
    return result


def main():
    args = parse_args()
    import numpy as np
    import torch
    from dynamicwam.inference.native_fast_graph import NativeFastGraph, _BRIDGE_CONTEXT_TENSORS
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
        interface_version = getattr(model.fast_conditioner, "interface_version", 1)
        if interface_version >= 3:
            packet['_rtc_action_target_offsets'] = np.arange(1, 17, dtype=np.float64) * float(policy.runtime.action_interval_seconds)
            packet['_rtc_prefix_actions'] = np.zeros((0,14),np.float32)
            packet['_rtc_prefix_length'] = 0
        text_rows = runner._get_text_embeddings()
        packet.update(action_index=0, _query_index=0, _episode_seed=policy._episode_seed, _text=text_rows)
        with torch.inference_mode():
            policy._install_bridge(policy._build_bridge_completed(packet))
            world = policy.bridge
            joint_v2 = interface_version >= 2
            joint_v3 = interface_version >= 3
            if joint_v2:
                # Match MotionKVPolicy._install_bridge source-cache lifecycle.
                world = model.fast_conditioner.prepare_source_cache(world)
            report["conditioner_interface_version"] = getattr(model.fast_conditioner, "interface_version", 1)
            completed_plan = getattr(world, "source_completed_plan", None)
            if completed_plan is not None:
                delta = completed_plan.float() - world.source_action_plan.float()
                report["source_plan_refinement"] = dict(
                    mean_abs_normalized=float(delta.abs().mean()),
                    max_abs_normalized=float(delta.abs().max()),
                    arm_mean_abs_normalized=float(delta[..., [i for i in range(14) if i not in (6, 13)]].abs().mean()),
                    gripper_mean_abs_normalized=float(delta[..., [6, 13]].abs().mean()),
                    semantics="same-model partial clean estimate versus completed K10 proposal; not an official-checkpoint quality comparison")
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
            if joint_v2:
                current.action_target_offsets = torch.arange(1, policy.runtime.chunk_size + 1,
                    device=device,dtype=torch.float64 if joint_v3 else torch.float32)[None] * float(policy.runtime.action_interval_seconds)
            if joint_v3:
                current.age.delta_sim_seconds = torch.tensor(0.,device=device,dtype=torch.float64)
                current.previous_plan = torch.zeros((1, 16, 14),device=device,dtype=state.dtype)
                current.previous_plan_valid = torch.zeros((1, 16),device=device,dtype=torch.bool)
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
            if joint_v2:
                source_inputs.append(("action_target_offsets",current.action_target_offsets))
            if joint_v3:
                source_inputs += [("previous_plan",current.previous_plan),("previous_plan_valid",current.previous_plan_valid)]
            source_inputs += [(name, getattr(world, name)) for name in _BRIDGE_CONTEXT_TENSORS
                              if torch.is_tensor(getattr(world, name))]
            source_hash = tensor_fingerprint(source_inputs)

            def eager(obs, cache, cond):
                effective = d0 if cond is None else cond
                obs, cache = graph._canonical_context(obs, cache, noise, effective)
                memory = model.prepare_fast_memory(obs, cache, text, text_mask)
                return model.sample_fast_actions(obs, cache, memory, noise, prefix=effective)

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
            if joint_v2:
                new_source_rgb = dataclasses.replace(world, source_rgb=world.source_rgb + .05)
                new_source_rgb = model.fast_conditioner.prepare_source_cache(new_source_rgb)
                cases += [
                    ("source_clean_plan_changed", current, dataclasses.replace(world,
                        source_action_plan=world.source_action_plan + .125), prefix),
                    ("source_motion_changed", current, dataclasses.replace(world,
                        source_motion=world.source_motion + .25), prefix),
                    ("source_rgb_cache_refreshed", current, new_source_rgb, prefix),
                    ("source_plan_sigma_changed", current, dataclasses.replace(world,
                        source_action_sigma=torch.as_tensor(.4, device=device)), prefix),
                    ("observation_delay_changed", dataclasses.replace(current,
                        observation_delay_seconds=.125), world, prefix),
                    ("physical_action_target_offsets_changed", dataclasses.replace(current,
                        action_target_offsets=current.action_target_offsets * 1.25), world, prefix),
                ]
            if joint_v3:
                cases += [
                    ("source_physical_offsets_changed", current, dataclasses.replace(world,
                        source_action_target_offsets=world.source_action_target_offsets + .03), prefix),
                    ("previous_proposal_changed", dataclasses.replace(current,
                        previous_plan=current.previous_plan + .1,
                        previous_plan_valid=torch.ones_like(current.previous_plan_valid)), world, prefix),
                    ("source_publication_age_changed", dataclasses.replace(current,
                        age=dataclasses.replace(current.age,delta_sim_seconds=torch.tensor(2.8,device=device,dtype=torch.float64),delta_actions=20)), world, prefix),
                ]
                if getattr(world, 'source_completed_plan', None) is not None:
                    cases.append(("source_completed_K10_plan_changed", current,
                        dataclasses.replace(world, source_completed_plan=world.source_completed_plan + .125), prefix))
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
                if joint_v2:
                    for field in _BRIDGE_CONTEXT_TENSORS:
                        source_value = getattr(cache, field)
                        if source_value is not None:
                            torch.testing.assert_close(getattr(graph.bridge, field),
                                torch.as_tensor(source_value, device=device), atol=0, rtol=0)
                    torch.testing.assert_close(graph.current.committed_prefix, effective.actions, atol=0, rtol=0)
                    torch.testing.assert_close(graph.current.committed_prefix_lengths, effective.lengths, atol=0, rtol=0)
                    torch.testing.assert_close(graph.current.action_target_offsets, obs.action_target_offsets, atol=0, rtol=0)
                if joint_v3:
                    torch.testing.assert_close(graph.current.previous_plan,obs.previous_plan,atol=0,rtol=0)
                    torch.testing.assert_close(graph.current.previous_plan_valid,obs.previous_plan_valid,atol=0,rtol=0)
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
            if args.concurrent_slow:
                if not joint_v3:
                    raise RuntimeError('concurrent slow acceptance requires the complete v3 source context')
                report['concurrent_slow'] = concurrent_slow_probe(policy=policy, packet=packet,
                    model=model, graph=graph, current=current, world=world, prefix=prefix,
                    replay=replay, eager=eager, atol=args.atol, rtol=args.rtol)
                if tensor_fingerprint(model.named_parameters()) != parameters_hash:
                    raise AssertionError('concurrent producer mutated model parameters')
    except Exception as exc:
        report['passed'] = False
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
