#!/usr/bin/env python3
"""Real model, complete adapter-compute budget; no render/TOPP/RTC claim."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dynamicwam.runtime.ciwam.wam.temporal_feedback_policy import TemporalFeedbackPolicy
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256
from dynamicwam.runtime.ciwam.adapters.domino.deploy_policy_multirate_flow import fast_composite_from_observation
from dynamicwam.runtime.ciwam.adapters.domino.composite import composite_robotwin_frame
from dynamicwam.runtime.ciwam.flow import HeadFlowBuffer
from dynamicwam.models.onpolicy_feedback_contract import FORMAT as ONPOLICY_FORMAT

FILES = ('src/dynamicwam/models/temporal_plan_feedback.py',
    'src/dynamicwam/models/onpolicy_feedback_contract.py',
    'src/dynamicwam/models/temporal_feedback_contract.py', 'src/dynamicwam/inference/feedback_graph.py',
    'src/dynamicwam/inference/native_plan_graph.py', 'src/dynamicwam/models/flow_schedule.py',
    'src/dynamicwam/models/native_wam_teacher.py', 'src/dynamicwam/models/causal_world_bridge.py',
    'src/dynamicwam/runtime/ciwam/wam/temporal_feedback_policy.py',
    'src/dynamicwam/runtime/ciwam/adapters/domino/deploy_policy_temporal_feedback.py',
    'scripts/profile_temporal_feedback.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--packet', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 20:
        parser.error('at least 20 samples required')
    if args.output.exists():
        parser.error('profile output already exists; use a fresh path')
    checkpoint_format = (torch.load(args.checkpoint, map_location='cpu', weights_only=True).get('format')
                         if args.checkpoint else None)
    initial_horizon = 2 if checkpoint_format == ONPOLICY_FORMAT else 1
    torch.set_num_threads(1)
    policy = TemporalFeedbackPolicy(str(ROOT/'configs/absolute_motion_v2.yaml'), project_root=str(ROOT),
        checkpoint=args.checkpoint, feedback_enabled=bool(args.checkpoint), execute_horizon=initial_horizon)
    policy.setup()
    policy.set_instruction('pick up the object')
    with np.load(args.packet, allow_pickle=False) as archive:
        original = {k: np.array(archive[k], copy=True) for k in archive.files}
    # Reconstitute an observation fixture, then run the SAME CPU operations as
    # the adapter. Camera acquisition and TOPP are intentionally not included.
    frame = original['frame']
    if frame.shape != (360, 320, 3):
        raise ValueError('profile packet must contain unletterboxed 360x320 RGB')
    import cv2
    obs = {'observation': {'head_camera': {'rgb': frame[:240].copy()},
        'left_camera': {'rgb': cv2.resize(frame[240:, :160], (320, 240))},
        'right_camera': {'rgb': cv2.resize(frame[240:, 160:], (320, 240))}}}
    flow = HeadFlowBuffer(policy.runtime.head_flow_config)
    clock = 0.
    previous = frame[:240].copy()
    for index in range(4):
        flow.push(previous, simulator_time_seconds=index * .2)
    clock = .8

    def packet(refresh):
        nonlocal clock, previous
        clock += .2
        head = obs['observation']['head_camera']['rgb']
        flow.push(head, simulator_time_seconds=clock)
        fast_frame = fast_composite_from_observation(obs, policy.runtime.observation_config)
        value = dict(original, frame=composite_robotwin_frame(obs, policy.runtime.observation_config) if refresh else fast_frame,
            fast_frame=fast_frame, sim_time=clock, action_index=0, previous_head=previous,
            history_dt=.2, history_valid=True)
        if refresh:
            motion = flow.observation()
            value.update(flow_frames=motion.flow_rgb, motion_features=motion.motion_features,
                motion_interval_valid_mask=motion.interval_valid_mask,
                motion_acceleration_valid_mask=motion.acceleration_valid_mask)
        previous = head.copy()
        return value

    def measure(fn):
        for _ in range(3):
            fn()
        values = []
        for _ in range(args.samples):
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter()-started)*1000)
        return dict(samples_ms=values, mean_ms=float(np.mean(values)),
            **{f'p{q}_ms':float(np.percentile(values, q)) for q in (50,95,99)})

    def refresh():
        policy.plan = None
        return policy.query(packet(True))

    def fast():
        policy.age = 4
        return policy.query(packet(False))

    with torch.inference_mode():
        policy.feedback_enabled = False
        anchors = []
        for h in (1,2,4):
            policy.execute_horizon = h
            for variation in range(3):
                fixed = packet(True)
                fixed['frame'] = np.roll(fixed['frame'], variation, axis=1).copy()
                fixed['state'] = fixed['state'] + variation * .0001
                rng = torch.cuda.get_rng_state()
                expected = policy.base.sample(fixed)
                torch.cuda.set_rng_state(rng)
                policy.plan = None
                start = time.perf_counter()
                pieces = [policy.query(fixed) for _ in range(16//h)]
                elapsed = (time.perf_counter()-start)*1000
                actual = np.concatenate(pieces)
                np.testing.assert_array_equal(actual, expected)
                anchors.append(dict(horizon=h,variation=variation,exact=True,cycle_ms=elapsed))
        policy.execute_horizon = initial_horizon
        original_eager = measure(lambda:policy.base.sample(packet(True)))
        planner = measure(refresh)
        policy.feedback_enabled = True
        started = time.perf_counter()
        fast()
        torch.cuda.synchronize()
        cold_feedback_ms = (time.perf_counter()-started)*1000
        feedback = measure(fast)
        # Changing H changes only the number of consumed output rows, but
        # measure each explicitly rather than assuming equal copy/return costs.
        by_h = {}
        for h in policy.supported_horizons:
            policy.execute_horizon = h
            fast_h = feedback if h == initial_horizon else measure(fast)
            cycle_ms = planner['mean_ms'] + (16//h-1)*fast_h['mean_ms']
            ratio = cycle_ms/original_eager['mean_ms']
            by_h[str(h)] = dict(fast=fast_h,ms_per_action=cycle_ms/16,ratio_to_official=ratio,
                ratio_to_graph_zero=cycle_ms/planner['mean_ms'],
                passed=ratio<=.85 and fast_h['p95_ms']<15.)
        selected = next((h for h in policy.supported_horizons if by_h[str(h)]['passed']), None)
    runtime = dict(executable=os.path.abspath(sys.executable), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        prefix=sys.prefix, gpu_name=torch.cuda.get_device_name(),
        gpu_capability=list(torch.cuda.get_device_capability()))
    result = dict(runtime=runtime, official_eager=original_eager,native_graph_planner=planner,feedback_query=feedback,
        by_horizon=by_h,selected_horizon=selected,passed=selected is not None,
        cold_feedback_query_ms=cold_feedback_ms,planner_graph=dict(policy.sampler.stats),
        feedback_graph=dict(policy.feedback_graph.stats),native_zero_anchors=anchors,
        zero_feedback_native_full_plan_exact=True,base_checkpoint_sha256=policy.base_sha256,
        checkpoint=args.checkpoint,checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
        checkpoint_format=checkpoint_format,supported_horizons=list(policy.supported_horizons),
        history_enabled=policy.history_enabled,
        packet_sha256=file_sha256(args.packet),source_hashes={name:file_sha256(ROOT/name) for name in FILES},
        source_tree_hashes={path.relative_to(ROOT).as_posix():file_sha256(path)
            for path in sorted((ROOT/'src').rglob('*.py'))},
        head_parameters=sum(p.numel() for p in policy.head.parameters()),
        scope='real weights; observation-fixture adapter preprocessing, history copy, Farneback on refresh, VAE, native 10/2, head graph, normalization and CPU result; excludes render, TOPP, stdout logging and text cold start',
        limitations='same cached RGB fixture and physical delta; includes query callbacks but excludes flow.push/history copy on nonquery callbacks, render, TOPP, stdout and per-episode text encoding; this is a service-budget estimate, not deployed reaction speed; Graph speedup is engineering, feedback adds cost to equally optimized H16')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected_horizon','passed','cold_feedback_query_ms','head_parameters')},indent=2))
    if not result['passed']:
        raise RuntimeError('no horizon met the predeclared total-compute latency gate')


if __name__ == '__main__':
    main()
