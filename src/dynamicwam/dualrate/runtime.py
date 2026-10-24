"""Two independent in-flight slots around the actual DualRateSystem.

The simulator owns the physical clock and calls poll/action_at while model
workers run. This module never pauses, advances or rescales the environment.
All callbacks are inside the measured query lease; preprocessing is not free.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Callable
import torch
from .timing import (CompletedReferenceStore, ExecutionQueue, TimedPlan,
                     TimingError, DeadlineMiss, prefix_digest)


@dataclass(frozen=True)
class RuntimeConfig:
    action_dt: float
    fast_period_steps: int = 2
    slow_period_steps: int = 4
    protect_steps: int = 2
    submit_margin: float = .01
    reference_max_age: float = 1.
    sample_steps: int = 10

    def __post_init__(self):
        if (not math.isfinite(self.action_dt) or self.action_dt <= 0 or
                min(self.fast_period_steps,self.slow_period_steps,self.protect_steps,self.sample_steps) < 1 or
                not 0 <= self.submit_margin < self.protect_steps*self.action_dt or
                not math.isfinite(self.reference_max_age) or self.reference_max_age <= 0):
            raise ValueError("invalid physical runtime configuration")


def detached_batch(batch):
    if not all(isinstance(value,torch.Tensor) for value in batch.values()):
        raise TypeError("online batches contain tensors only")
    return {key:value.detach().clone() for key,value in batch.items()}


class DualRateRuntime:
    """Explicitly created by a guarded deployment worker, never at import.

    batch_factory(lease, reference) returns observations available at lease.query_time.
    The initial queue/hold is caller-owned, logged and common to baseline policies.
    Call close before releasing the external device-lifetime guard.
    """
    def __init__(self, system, queue: ExecutionQueue, clock: Callable[[],float],
                 config: RuntimeConfig, *, action_mean, action_scale, seed=0, record=None):
        if abs(queue.dt-config.action_dt) > 1e-10:
            raise ValueError("runtime/queue action grid differs")
        if system.training:
            raise ValueError("runtime requires eval mode; no optimizer shares deployment weights")
        self.system,self.queue,self.clock,self.config = system,queue,clock,config
        self.record = record or (lambda event: None)
        self.store = CompletedReferenceStore(episode_id=queue.episode_id,clock_id=queue.clock_id,
                                            max_age=config.reference_max_age)
        self.mean = torch.as_tensor(action_mean,dtype=torch.float32)
        self.scale = torch.as_tensor(action_scale,dtype=torch.float32)
        if (self.mean.shape != (system.fast.config.action_dim,) or self.scale.shape != self.mean.shape or
                not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale<=0).any()):
            raise ValueError("exact checkpoint action normalization required")
        self.pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix="dualrate")
        device = next(system.fast.parameters()).device
        self.fast_rng = torch.Generator(device=device).manual_seed(seed)
        self.slow_rng = torch.Generator(device=device).manual_seed(seed+1)
        self.fast_job = self.slow_job = None
        self.slow_version = 0
        self.closed = False
        self.last_fast_tick = self.last_slow_tick = -10**12

    def due(self, now):
        tick = math.floor((now-self.queue.origin)/self.queue.dt + 1e-9)
        return (self.fast_job is None and tick-self.last_fast_tick>=self.config.fast_period_steps,
                self.slow_job is None and tick-self.last_slow_tick>=self.config.slow_period_steps)

    def _prefix(self,batch,prefix,kind):
        times = batch["source_action_times" if kind=="source" else "action_times"]
        if times.shape[0] != 1:
            raise ValueError("one environment per runtime")
        known = torch.zeros((1,times.shape[1],self.mean.numel()),device=times.device)
        mask = torch.zeros(times.shape,dtype=torch.bool,device=times.device)
        for i,row in enumerate(prefix.actions):
            when = prefix.start_time+i*prefix.dt
            slots = (times[0]-when).abs()<1e-8
            if slots.sum()!=1:
                raise TimingError("requested action grid omits an immutable prefix row")
            known[0,slots] = ((torch.tensor(row)-self.mean)/self.scale).to(known.device)
            mask[0,slots] = True
        batch[kind+"_known_actions"],batch[kind+"_known_mask"] = known,mask

    def start_slow(self, batch, source_prefix):
        if self.closed or self.slow_job is not None:
            raise TimingError("slow slot unavailable")
        batch=detached_batch(batch)
        source=float(batch["source_time"].item())
        now=float(self.clock())
        if source>now or source_prefix.start_time>source+1e-9:
            raise TimingError("source/commitment snapshot is in the future")
        self._prefix(batch,source_prefix,"source")
        # Placeholder only while computing; real completion is installed below.
        batch["completed_time"]=batch["source_time"].clone()
        batch["query_time"]=batch["source_time"].clone()
        self.slow_version += 1
        version=self.slow_version
        self.last_slow_tick=math.floor((now-self.queue.origin)/self.queue.dt+1e-9)
        def work():
            with torch.no_grad():
                reference=self.system.predict_reference(batch,generator=self.slow_rng)
                world=self.system.world_readout(reference)
                # CPU transfer waits for GPU writes before taking completed_time.
                world={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in world.items()}
                completed=float(self.clock())
                world["completed_time"]=torch.tensor([completed],dtype=torch.float64)
                return world,completed
        self.slow_job=(self.pool.submit(work),source,source_prefix,version)

    def start_fast(self,batch_factory,*,observation_time):
        if self.closed or self.fast_job is not None:
            raise TimingError("fast slot unavailable")
        query=float(self.clock())
        reference=self.store.snapshot(query)
        lease=self.queue.begin_query(query_time=query,observation_time=observation_time,
            protect_steps=self.config.protect_steps,submit_margin=self.config.submit_margin,
            reference_version=reference.version if reference else -1)
        self.last_fast_tick=math.floor((query-self.queue.origin)/self.queue.dt+1e-9)
        def work():
            with torch.no_grad():
                batch=detached_batch(batch_factory(lease,reference))
                if float(batch["query_time"].item()) != query:
                    raise TimingError("batch query differs from queue lease")
                self._prefix(batch,lease.prefix,"fast")
                device=batch["state"].device
                if reference:
                    world={k:v.to(device) for k,v in reference.payload.items()}
                else:
                    world=self.system.empty_world(batch)
                output=self.system.sample_fast(batch,world,steps=self.config.sample_steps,generator=self.fast_rng)
                raw=output["actions"][0].float().cpu()*self.scale+self.mean
                start=float(batch["action_times"][0,0])
                return TimedPlan(start,self.config.action_dt,tuple(map(tuple,raw.tolist())),
                                 lease.base_queue_version+1,query,lease),output
        self.fast_job=(self.pool.submit(work),lease)

    def poll(self):
        """Caller invokes alongside continuous physics; never waits on unfinished work."""
        if self.slow_job is not None and self.slow_job[0].done():
            job,source,prefix,version=self.slow_job;self.slow_job=None
            world,completed=job.result()
            payload={k:v for k,v in world.items() if isinstance(v,torch.Tensor)}
            intervals=payload["world_intervals"][0][payload["world_types"][0]==1]
            times=tuple(sorted(set(intervals.flatten().tolist())))
            accepted=self.store.publish(source_time=source,completed_time=completed,version=version,
                generation_stage=str(world["generation_stage"]),future_times=times,
                future_valid=(True,)*len(times),payload=payload,now=float(self.clock()),
                episode_id=self.queue.episode_id,clock_id=self.queue.clock_id,
                source_queue_version=prefix.queue_version,source_prefix_hash=prefix_digest(prefix))
            self.record(dict(event="slow_completed",source_time=source,completed_time=completed,
                             version=version,accepted=accepted))
        if self.fast_job is not None and self.fast_job[0].done():
            job,lease=self.fast_job;self.fast_job=None
            try:
                plan,output=job.result()
                discarded=self.queue.replace(plan,now=float(self.clock()))
            except BaseException as exc:
                self.queue.cancel_query(lease)
                self.record(dict(event="fast_rejected",query_id=lease.query_id,error=type(exc).__name__))
                raise
            self.record(dict(event="fast_committed",query_id=lease.query_id,discarded_rows=discarded,
                             protected_until=lease.protected_until))
            return output
        return None

    def close(self):
        self.closed=True
        self.pool.shutdown(wait=True,cancel_futures=False)
        errors=[]
        for pending in (self.slow_job,self.fast_job):
            if pending is not None:
                try:
                    pending[0].result()
                except BaseException as exc:
                    errors.append(exc)
        if self.fast_job is not None:
            self.queue.cancel_query(self.fast_job[1]);self.fast_job=None
        self.slow_job=None
        if errors:
            raise errors[0]
