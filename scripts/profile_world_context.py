#!/usr/bin/env python3
"""Loaded policy compute profile; physical clock evidence is a separate gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'external/flash_attn_runtime'))
from dynamicwam.runtime.world_context_worker import WorldContextWorker
from dynamicwam.inference.native_plan_graph import NativePlanSampler
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-config',type=Path,required=True)
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=30)
    a=p.parse_args()
    if a.samples<20 or a.output.exists(): p.error('need >=20 samples and a fresh output')
    torch.set_num_threads(1)
    config=json.loads(a.policy_config.read_text());config.pop('collection_dir',None)
    config['mode']='graph_zero'
    policy=WorldContextWorker(config)
    if policy.head is None: raise ValueError('profile requires explicit v1 initialization or trained head')
    with np.load(a.packet,allow_pickle=False) as store:
        raw={k:np.array(store[k],copy=True) for k in store.files}
    raw.update(instruction='pick up the object',control_period_s=.1,command_index=0,_sampling_seed=20261008,
               sim_time=0.,history_dt=.2,previous_head_rgb=raw['frame'][:240].copy())
    def measure(fn):
        values=[]
        for i in range(a.samples+3):
            torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
            if i>=3: values.append((time.perf_counter()-start)*1000)
        return values
    start=time.perf_counter();policy.sample(raw);torch.cuda.synchronize()
    cold=(time.perf_counter()-start)*1000
    graph_sampler=policy.sampler
    def fresh():
        policy.plan=None
        return policy.sample(raw)
    graph=measure(fresh)
    eager_sampler=NativePlanSampler(policy.base.runner.model,policy.schedule,enabled=False)
    policy.sampler=eager_sampler
    saved_head=policy.head
    policy.head=None  # The eager control has no unused small-head preprocessing.
    eager=measure(fresh)
    policy.sampler=graph_sampler
    graph_zero=measure(fresh)
    policy.head=saved_head
    # Check identical random source/noise with the actual loaded native model.
    anchors=[]
    with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for scale in (.9,1.,1.1):
            for seed in (20261008,20261009,20261010):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                fixture=dict(raw,state=np.asarray(raw['state'])*scale)
                source,noise,_=policy._source(fixture,sampling_seed=seed)
                expected,_=eager_sampler.sample(source,noise)
                observed,_=graph_sampler.sample(source,noise)
                anchors.append(dict(state_scale=scale,seed=seed,
                    max_abs_error=float((expected-observed).abs().max().item())))
    difference=max(row['max_abs_error'] for row in anchors)
    if difference>1e-5: raise RuntimeError(f'native graph action parity failed {difference}')
    policy.sampler=graph_sampler;policy.mode='world_context';fresh()
    packet=dict(raw,sim_time=.4,command_index=4)
    fast=measure(lambda:policy.sample(packet))
    # A new 16-row plan is produced every 12 slots (1.2s), with fast requests
    # at slots2,4,6,8,10. Count slow and fast costs over those 12 executed slots.
    amortized=(float(np.mean(graph))+5*float(np.mean(fast)))/12
    baseline=float(np.mean(eager))/12
    passed=bool(np.percentile(fast,95)<100 and amortized<baseline*.85)
    summary=lambda xs:dict(mean_ms=float(np.mean(xs)),**{f'p{q}_ms':float(np.percentile(xs,q)) for q in (50,95,99)})
    result=dict(status='complete',passed=passed,scope='loaded_compute_only_not_RTC_or_physics_validity',
        fast_samples_ms=fast,graph_samples_ms=graph,eager_samples_ms=eager,graph_zero_samples_ms=graph_zero,cold_start_ms=cold,
        measurements=dict(fast=summary(fast),graph_refresh=summary(graph),eager_refresh=summary(eager),graph_zero_refresh=summary(graph_zero)),
        amortized_ms_per_slot=amortized,eager_ms_per_slot=baseline,
        saving_fraction=1-amortized/baseline,native_action_max_abs_error=difference,
        native_parity_anchors=anchors,
        feedback_over_graph_zero_fraction=amortized/(float(np.mean(graph_zero))/12)-1,
        base_checkpoint_sha256=policy.base_sha,head_checkpoint_sha256=policy.checkpoint_sha,
        controller_protocol_sha256=policy.protocol_sha,
        packet_sha256=file_sha256(a.packet),policy_config_sha256=file_sha256(a.policy_config),
        source_files={str(q.relative_to(ROOT)):file_sha256(q) for q in sorted((ROOT/'src').rglob('*.py'))},
        note='Legacy initialization is a timing fixture only; learned checkpoints must be measured again.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if not passed: raise SystemExit(2)


if __name__=='__main__': main()
