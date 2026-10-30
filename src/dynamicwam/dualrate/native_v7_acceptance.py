"""Real GPU connectivity, shifted-sampler Graph parity and matched latency gate."""
from dataclasses import replace
import time
import numpy as np
import torch
from .native_v7 import CompactEventFast,aligned_reference
from .native_v6 import validate_events
from .fast_graph import FastGraphSampler


@torch.no_grad()
def verify_compact(system,batch):
    validate_events(batch)
    fast=system.fast
    assert not hasattr(fast,'motor') and fast.config.dim==448 and fast.config.layers==6
    old=system.predict_reference(batch);world=system.world_readout(old)
    world['native_available']=torch.ones_like(batch['native_query_index'],dtype=torch.bool)
    cond=system.conditions(batch,world)
    a=torch.randn_like(batch['fast_action']);z=torch.randn_like(batch['future']);sigma=a.new_full((len(a),),.5)
    baseline=fast(a,z,sigma,**cond)['action_velocity']
    deltas={}
    for name,changed in [('observation_tokens',dict(cond,observation_tokens=cond['observation_tokens']+.2)),
                         ('world_tokens',dict(cond,world_tokens=cond['world_tokens']*1.2+.1))]:
        deltas[name]=float((fast(a,z,sigma,**changed)['action_velocity']-baseline).abs().max())
    deltas['Z']=float((fast(a,z+.2,sigma,**cond)['action_velocity']-baseline).abs().max())
    if min(deltas.values())<=0:raise AssertionError(('disconnected compact input',deltas))
    center,ref,valid,w=aligned_reference(batch,world)
    if not valid.any() or center.abs().max()<=0:raise AssertionError('aligned A10 prior disconnected')
    if not torch.equal(center[:,:,[6,13]],torch.zeros_like(center[:,:,[6,13]])):raise AssertionError('gripper prior')
    # Independent comparison against production eager sample (not just a helper
    # comparing its own graph with itself) catches shifted/uniform schedule drift.
    state=torch.cuda.get_rng_state();g=torch.Generator(device=a.device).manual_seed(71)
    eager=system.sample_fast(batch,world,generator=g)
    engine=system.enable_fast_graph()
    torch.cuda.set_rng_state(state);g.manual_seed(71)
    actual=system.sample_fast(batch,world,generator=g)
    error=0.
    for key in eager:
        torch.testing.assert_close(actual[key],eager[key],atol=2e-5,rtol=2e-5)
        if actual[key].dtype!=torch.bool:error=max(error,float((actual[key]-eager[key]).abs().max()))
    receipt=engine.seal(expected_shapes=1)
    # Matched 384/6 control: identical event conditioning, encoder, hardware,
    # K10 shifted schedule and Graph mode. This is timing, not an SR comparison.
    timings={}
    small=CompactEventFast(replace(fast.config,dim=384,heads=6)).to(a.device).eval().requires_grad_(False)
    small.register_buffer('sampling_sigmas',fast.sampling_sigmas.clone(),persistent=False)
    for name,model in [('384x6_control',small),('448x6_candidate',fast)]:
        system.fast=model
        system._fast_graph_sampler=engine if model is fast else FastGraphSampler(model)
        for _ in range(4):system.sample_fast(batch,world)
        if model is small:system._fast_graph_sampler.seal(expected_shapes=1)
        times=[]
        for _ in range(24):
            torch.cuda.synchronize(a.device);t=time.monotonic();system.sample_fast(batch,world)
            torch.cuda.synchronize(a.device);times.append(time.monotonic()-t)
        timings[name]=dict(samples=len(times),p50_seconds=float(np.median(times)),
            p95_seconds=float(np.percentile(times,95)),maximum_seconds=max(times),
            parameters=sum(p.numel() for p in model.parameters()))
    system.fast=fast;system._fast_graph_sampler=engine
    ratios={k:timings['448x6_candidate'][k]/timings['384x6_control'][k] for k in ('p50_seconds','p95_seconds')}
    passed=ratios['p50_seconds']<=1.20 and ratios['p95_seconds']<=1.25
    report=dict(passed=passed,single_action_decoder=True,no_native_fast_motor=True,
        sensitivity=deltas,shifted_eager_graph_max_error=error,graph=receipt,timings=timings,ratios=ratios,
        latency_scope='full sample_fast including latest observation encoder and condition preparation; warmed, same GPU, no simulator contention',
        prior_alignment='integer command join, zero expired suffix, no gripper bias',
        teacher_alignment='current query physical observation and command origin',
        fast_total_parameters=sum(p.numel() for p in fast.parameters()),performance_accepted=False)
    if not passed:raise RuntimeError(('compact latency budget failed',report))
    return report
