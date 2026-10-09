#!/usr/bin/env python3
"""Full model-query budget before training; no environment/control-Hz claim."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import PlanFeedbackPolicy, file_sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--checkpoint')
    p.add_argument('--packet',type=Path,default=Path('/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/feedback_preflight_20261003/synthetic_packet.npz'))
    args=p.parse_args()
    policy=PlanFeedbackPolicy(str(ROOT/'configs/absolute_motion_v2.yaml'),project_root=str(ROOT),
        checkpoint=args.checkpoint,feedback_enabled=bool(args.checkpoint))
    policy.setup()
    policy.set_instruction('pick up the object')
    with np.load(args.packet,allow_pickle=False) as z:
        packet={k:np.array(z[k],copy=True) for k in z.files}
    packet['fast_frame']=packet['frame']
    packet.update(sim_time=0.,action_index=0)
    def measure(fn):
        for _ in range(3): fn()
        times=[]
        for _ in range(args.samples):
            torch.cuda.synchronize(); t=time.perf_counter(); fn(); torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
        return dict(mean_ms=float(np.mean(times)),p50_ms=float(np.percentile(times,50)),
            p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),samples_ms=times)
    def refresh():
        policy.plan=None
        return policy.query(packet)
    def feedback():
        policy.age=4
        return policy.query(packet)
    with torch.inference_mode():
        enabled = policy.feedback_enabled
        policy.feedback_enabled = False
        rng = torch.cuda.get_rng_state()
        expected = policy.base.sample(packet)
        torch.cuda.set_rng_state(rng)
        policy.plan = None
        torch.cuda.synchronize(); t=time.perf_counter()
        first = policy.query(packet)
        torch.cuda.synchronize()
        cold_ms=(time.perf_counter()-t)*1000
        pieces = [first] + [policy.query(packet) for _ in range(3)]
        actual = np.concatenate(pieces, axis=0)
        np.testing.assert_array_equal(actual, expected)
        policy.feedback_enabled = enabled
        baseline=measure(lambda:policy.base.sample(packet))
        native=measure(refresh)
        # Even an untrained head executes all operations; its zero output is
        # a quality anchor, never a basis for claiming learned performance.
        policy.feedback_enabled=True
        fast=measure(feedback)
    amortized=(native['mean_ms']+3*fast['mean_ms'])/16
    original=baseline['mean_ms']/16
    result=dict(official_eager=baseline,native_graph_planner=native,feedback_query=fast,
        cold_planner_query_ms=cold_ms,graph_stats=policy.sampler.stats,
        ms_per_action=amortized,official_ms_per_action=original,
        ratio_to_official=amortized/original,
        ratio_to_equally_optimized_h16=amortized/(native['mean_ms']/16),
        base_checkpoint_sha256=policy.base_sha256,checkpoint=args.checkpoint,
        checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
        source_hashes={name:file_sha256(ROOT/name) for name in (
            'src/dynamicwam/models/plan_feedback.py', 'src/dynamicwam/models/flow_schedule.py',
            'src/dynamicwam/models/native_wam_teacher.py', 'src/dynamicwam/models/causal_world_bridge.py',
            'src/dynamicwam/inference/native_plan_graph.py',
            'src/dynamicwam/runtime/ciwam/wam/plan_feedback_policy.py',
            'scripts/profile_plan_feedback.py')},
        head_parameters=sum(p.numel() for p in policy.head.parameters()),
        zero_feedback_native_full_plan_exact=True,
        passed=amortized <= .9*original and fast['p95_ms']<30,
        scope='cached text; VAE, current RGB conversion, full native 10/2, denormalize and CPU action copy; excludes camera acquisition/Farneback/environment execution',
        caveat='speed versus the original eager baseline includes CUDA graph engineering; feedback adds work versus equally optimized H16; no closed-loop quality or RTC claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('official_eager','native_graph_planner','feedback_query')},indent=2))
    if not result['passed']:raise RuntimeError('pretraining latency gate failed')


if __name__=='__main__':main()
