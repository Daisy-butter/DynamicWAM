#!/usr/bin/env python3
"""Serial GPU service-time profiler using a captured real observation packet.

Does not run DOMINO, estimate success, or demonstrate asynchronous overlap.
--packet is an NPZ (allow_pickle=False) with frame, fast_frame, state, flow_frames,
motion_features, motion_interval_valid_mask, motion_acceleration_valid_mask,
sim_time. Copy these fields from the multirate adapter immediately before query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def amortized_budget(bridge_ms, fast_ms, sync_ms, *, horizon, ratio, chunk):
    if min(horizon,ratio,chunk) < 1:
        raise ValueError('positive horizon, ratio, and chunk required')
    return dict(multirate_ms_per_action=(bridge_ms/ratio+fast_ms)/horizon,
                sync_matched_horizon_ms_per_action=sync_ms/horizon,
                sync_long_chunk_ms_per_action=sync_ms/chunk)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy-config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--packet',required=True)
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--project-root',default=str(ROOT))
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--ratio',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--no-native-cuda-graph', action='store_true',
                        help='Run the identical native network eagerly for an execution-only ablation')
    parser.add_argument('--fast-steps-override',type=int,choices=(1,2,4,10),
                        help='COMPUTE PROBE ONLY: override the native solver without retraining; not a quality evaluation')
    parser.add_argument('--baseline-fast-precision',action='store_true',
                        help='Also measure official sync after matching its action/motion weights to fast BF16 precision')
    args=parser.parse_args()
    if args.samples < 2 or args.warmup < 1 or args.ratio < 1:
        parser.error('samples >= 2, warmup >= 1, ratio >= 1 required')
    import numpy as np
    import torch
    from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy
    if not torch.cuda.is_available():
        raise RuntimeError('GPU profiler must run explicitly on an available CUDA device')
    torch.manual_seed(args.seed)
    with np.load(args.packet,allow_pickle=False) as saved:
        required=('frame','fast_frame','state','flow_frames','motion_features',
                  'motion_interval_valid_mask','motion_acceleration_valid_mask','sim_time')
        packet={key:np.array(saved[key],copy=True) for key in required}
    packet['sim_time']=float(packet['sim_time'].item())
    policy=MultiratePolicy(args.deploy_config,args.checkpoint,project_root=args.project_root,
                          native_cuda_graph=not args.no_native_cuda_graph)
    policy.setup();policy.set_instruction(args.instruction)
    runner=policy.base.runner
    native = policy.contract.get('native_multirate')
    trained_fast_steps = native.get('fast_steps') if native else None
    if args.fast_steps_override is not None:
        if not native:
            raise ValueError('fast step override is restricted to native COMPUTE probes')
        policy.fast_model.fast_schedule = policy.fast_model.schedule.with_action_steps(args.fast_steps_override)
    measured_fast_steps = args.fast_steps_override if args.fast_steps_override is not None else trained_fast_steps
    def parameter_dtypes(module):
        return sorted({str(p.dtype) for p in module.parameters()})
    precision_before = dict(slow_action=parameter_dtypes(runner.model.action_expert),
        slow_motion=parameter_dtypes(runner.model.absolute_motion_tokens),
        fast_action=parameter_dtypes(policy.fast_model.backbone.action_expert) if native else None,
        fast_conditioner=parameter_dtypes(policy.fast_model.fast_conditioner) if native else None)
    packet.update(action_index=0,_query_index=0,_episode_seed=policy._episode_seed,
                  _text=runner._get_text_embeddings())
    device=torch.device(runner.device)
    if device.type != 'cuda':
        raise RuntimeError('deploy profile must select CUDA')

    def measure(function):
        for _ in range(args.warmup):
            function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        rows=[]
        for _ in range(args.samples):
            torch.cuda.synchronize(device)
            start=time.perf_counter()
            function()
            torch.cuda.synchronize(device)
            rows.append((time.perf_counter()-start)*1000.)
        return dict(mean_ms=float(np.mean(rows)),p50_ms=float(np.percentile(rows,50)),
                    p95_ms=float(np.percentile(rows,95)),samples_ms=rows,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))

    def bridge():
        policy._install_bridge(policy._build_bridge_completed(packet))

    def fast():
        return policy._sample_fast(packet,0)

    def sync():
        output=runner.sample_chunk(first_frame=policy.base._frame_tensor(packet['frame']),
            state=policy.base._state_tensor(packet['state']),flow_frames=packet['flow_frames'],
            motion_features=packet['motion_features'],
            motion_interval_valid_mask=packet['motion_interval_valid_mask'],
            motion_acceleration_valid_mask=packet['motion_acceleration_valid_mask'])
        return output.float().cpu().numpy()

    try:
        with torch.inference_mode():
            b=measure(bridge);f=measure(fast);s=measure(sync)
            matched = None
            if args.baseline_fast_precision:
                if not native:
                    raise ValueError('matched native precision baseline requires a native checkpoint')
                fast_dtype = next(policy.fast_model.backbone.action_expert.parameters()).dtype
                if fast_dtype != torch.bfloat16:
                    raise ValueError('baseline-fast-precision currently requires a BF16 native fast expert')
                # The baseline run uses the official weights, now BF16. This
                # also changes its two joint action-prefix forwards. The slow
                # bridge/fast timings above retain their original precision.
                runner.model.action_expert.to(dtype=fast_dtype)
                runner.model.absolute_motion_tokens.to(dtype=fast_dtype)
                matched = measure(sync)
        precision_after = dict(slow_action=parameter_dtypes(runner.model.action_expert),
            slow_motion=parameter_dtypes(runner.model.absolute_motion_tokens))
        horizon=int(policy.contract['execute_horizon'])
        max_age=(args.ratio-1)*horizon
        if not policy.contract.get('feedback') and max_age > int(policy.contract['max_age_actions']):
            raise ValueError('requested ratio exceeds trained age support; amortization would be invalid')
        if policy.contract.get('feedback') and args.ratio*horizon != policy.contract['feedback']['world_refresh_actions']:
            raise ValueError('feedback amortization must use the actual heavy refresh budget')
        report=dict(kind='serial_service_time_only',gpu=torch.cuda.get_device_name(device),
            checkpoint=str(Path(args.checkpoint).resolve()),packet=str(Path(args.packet).resolve()),
            backbone=('immutable official source backbone; separate trained native fast action expert'
                      if policy.contract.get('native_multirate') else
                      'multirate checkpoint backbone for all three timings; not an original-weight quality baseline'),
            slow_bridge=b,fast_query=f,sync_sampler=s,
            sync_sampler_matched_precision=matched,
            amortized=amortized_budget(b['mean_ms'],f['mean_ms'],s['mean_ms'],horizon=horizon,
                                      ratio=args.ratio,chunk=int(policy.runtime.chunk_size)),
            execute_horizon=horizon,ratio=args.ratio,seed=args.seed,
            native_multirate=policy.contract.get('native_multirate'),
            native_cuda_graph=(dict(policy.fast_graph.stats) if policy.fast_graph is not None else None),
            solver=dict(trained_fast_steps=trained_fast_steps,measured_fast_steps=measured_fast_steps,
                overridden=args.fast_steps_override is not None,
                quality_evaluation_allowed=args.fast_steps_override is None,
                provenance='UNTRAINED SOLVER OVERRIDE: COMPUTE LOWER-BOUND PROBE ONLY'
                    if args.fast_steps_override is not None else 'checkpoint-trained solver'),
            precision=dict(before=precision_before,after=precision_after,
                published_sync='official deployed weight dtypes and shared optimized attention kernels',
                matched_sync='official weights with BF16 action/motion, including joint prefix; quality is not validated'
                    if matched is not None else None),
            limitations=['one replayed observation; age zero fast query','excludes rendering, optical flow construction, execution and IPC',
                         'serial measurements cannot establish concurrent p95 or DOMINO success',
                         'a solver override is untrained and cannot be used as policy quality evidence',
                         'BF16 baseline changes arithmetic and needs its own rollout quality validation'])
        if matched is not None:
            report['amortized_matched_precision'] = amortized_budget(
                b['mean_ms'],f['mean_ms'],matched['mean_ms'],horizon=horizon,
                ratio=args.ratio,chunk=int(policy.runtime.chunk_size))
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2,ensure_ascii=False))
    finally:
        policy.close()


if __name__=='__main__':
    main()
