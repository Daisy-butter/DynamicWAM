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
from contextlib import nullcontext
import time
import torch
from .timing import (CompletedReferenceStore, ExecutionQueue, TimedPlan,
                     TimingError, DeadlineMiss, StalePlan, prefix_digest)


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
                 config: RuntimeConfig, *, action_mean, action_scale, seed=0, record=None, action_projection=None):
        if abs(queue.dt-config.action_dt) > 1e-10:
            raise ValueError("runtime/queue action grid differs")
        if system.training:
            raise ValueError("runtime requires eval mode; no optimizer shares deployment weights")
        self.system,self.queue,self.clock,self.config = system,queue,clock,config
        self.record = record or (lambda event: None)
        self.action_projection=action_projection
        self.store = CompletedReferenceStore(episode_id=queue.episode_id,clock_id=queue.clock_id,
                                            max_age=config.reference_max_age)
        self.mean = torch.as_tensor(action_mean,dtype=torch.float32)
        self.scale = torch.as_tensor(action_scale,dtype=torch.float32)
        if (self.mean.shape != (system.fast.config.action_dim,) or self.scale.shape != self.mean.shape or
                not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale<=0).any()):
            raise ValueError("exact checkpoint action normalization required")
        self.pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix="dualrate")
        device = next(system.fast.parameters()).device
        self.device = device
        self.fast_stream = torch.cuda.Stream(device=device,priority=-1) if device.type=='cuda' else None
        self.slow_stream = torch.cuda.Stream(device=device) if device.type=='cuda' else None
        self.fast_rng = torch.Generator(device=device).manual_seed(seed)
        self.slow_rng = torch.Generator(device=device).manual_seed(seed+1)
        self.fast_job = self.slow_job = None
        self.slow_version = 0
        self.closed = False
        self.last_fast_tick = self.last_slow_tick = -10**12
        # Fast and slow never share the device. A running slow job holds the
        # next fast query; that query then runs alone on an action boundary
        # before the following slow job is admitted. Concurrent kernels made
        # the resumed fast miss the 0.35 s protected-prefix budget.
        self.fast_align_boundary = False

    def _row_offset(self, now):
        dt=self.queue.dt
        elapsed=float(now)-self.queue.origin
        row=math.floor(elapsed/dt+1e-9)
        return elapsed-row*dt

    def due(self, now):
        tick = math.floor((now-self.queue.origin)/self.queue.dt + 1e-9)
        period=self.fast_job is None and tick-self.last_fast_tick>=self.config.fast_period_steps
        if self.slow_job is not None:
            if period and not self.fast_align_boundary:
                self.fast_align_boundary=True
                version=self.slow_job[3]
                self.record(dict(event='fast_held',reason='slow_job_owns_the_device',
                                 physics_time=float(now),slow_version=version))
            return False,False
        if self.fast_job is not None or self.fast_align_boundary:
            # Finish the fast query that waited out the previous slow job
            # before admitting another slow job.
            fast=self.fast_align_boundary and period and self._row_offset(now)<=.04
            return fast,False
        slow=tick-self.last_slow_tick>=self.config.slow_period_steps
        if slow:
            return False,True
        return period,False

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

    def start_slow(self, batch, source_prefix, *, source_time=None):
        if self.closed or self.slow_job is not None:
            raise TimingError("slow slot unavailable")
        if (source_prefix.episode_id,source_prefix.clock_id)!=(self.queue.episode_id,self.queue.clock_id):
            raise StalePlan("slow commitment snapshot belongs to another episode/clock")
        source=float(source_time if callable(batch) else batch["source_time"].item())
        now=float(self.clock())
        if source>now or source_prefix.start_time>source+1e-9:
            raise TimingError("source/commitment snapshot is in the future")
        self.slow_version += 1
        version=self.slow_version
        self.record(dict(event='slow_started',source_time=source,version=version,
                         source_queue_version=source_prefix.queue_version,
                         source_prefix_hash=prefix_digest(source_prefix),
                         prefix_start=source_prefix.start_time,prefix_end=source_prefix.end_time))
        self.last_slow_tick=math.floor((now-self.queue.origin)/self.queue.dt+1e-9)
        def work():
            started=time.perf_counter()
            context=torch.cuda.stream(self.slow_stream) if self.slow_stream is not None else nullcontext()
            with context, torch.no_grad():
                local=detached_batch(batch() if callable(batch) else batch)
                if float(local['source_time'].item())!=source:
                    raise TimingError('source preprocessing changed the captured source time')
                self._prefix(local,source_prefix,"source")
                local["completed_time"]=local["source_time"].clone()
                local["query_time"]=local["source_time"].clone()
                reference=self.system.predict_reference(local,generator=self.slow_rng)
                world=self.system.world_readout(reference)
                # CPU transfer waits for GPU writes before taking completed_time.
                world={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in world.items()}
                completed=float(self.clock())
                world["completed_time"]=torch.tensor([completed],dtype=torch.float64)
                return world,completed,time.perf_counter()-started
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
        self.record(dict(event='fast_started',query_id=lease.query_id,query_time=query,
            observation_time=observation_time,reference_version=lease.reference_version,
            source_time=reference.source_time if reference else None,
            reference_age=query-reference.source_time if reference else None,
            protected_until=lease.protected_until,prefix_hash=lease.prefix_hash,
            remaining_prefix_budget_seconds=lease.protected_until-lease.submit_margin-query))
        def work():
            started=time.perf_counter()
            with (torch.cuda.stream(self.fast_stream) if self.fast_stream is not None else nullcontext()), torch.no_grad():
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
                known_cpu=batch['fast_known_mask'][0].cpu()
                # Reuse authoritative queue values, avoiding normalization round-trip
                # changes even in the returned executable-action diagnostic.
                if lease.prefix.actions:
                    raw[known_cpu]=torch.tensor(lease.prefix.actions,dtype=raw.dtype)
                if self.action_projection is not None:
                    raw,projection=self.action_projection(raw,known_cpu)
                    if projection['forecast_invalidated']:
                        output=dict(output)
                        for key in ('future_valid','future_field_valid','contact_valid'):
                            output[key]=torch.zeros_like(output[key])
                        for key in ('future','contact_probability'):
                            output[key]=torch.zeros_like(output[key])
                    output['executable_actions_raw']=raw.clone()
                    self.record(dict(event='actuator_projection',query_id=lease.query_id,**projection))
                start=float(batch["action_times"][0,0])
                return TimedPlan(start,self.config.action_dt,tuple(map(tuple,raw.tolist())),
                                 lease.base_queue_version+1,query,lease),output,time.perf_counter()-started
        try:
            self.fast_job=(self.pool.submit(work),lease)
        except BaseException:
            self.queue.cancel_query(lease)
            raise
        self.fast_align_boundary=False

    def poll(self):
        """Caller invokes alongside continuous physics; never waits on unfinished work."""
        if self.slow_job is not None and self.slow_job[0].done():
            job,source,prefix,version=self.slow_job;self.slow_job=None
            world,completed,duration=job.result()
            payload={k:v for k,v in world.items() if isinstance(v,torch.Tensor)}
            intervals=payload["world_intervals"][0][payload["world_types"][0]==1]
            times=tuple(sorted(set(intervals.flatten().tolist())))
            accepted=self.store.publish(source_time=source,completed_time=completed,version=version,
                generation_stage=str(world["generation_stage"]),future_times=times,
                future_valid=(True,)*len(times),payload=payload,now=float(self.clock()),
                episode_id=self.queue.episode_id,clock_id=self.queue.clock_id,
                source_queue_version=prefix.queue_version,source_prefix_hash=prefix_digest(prefix))
            self.record(dict(event="slow_completed",source_time=source,completed_time=completed,
                             version=version,accepted=accepted,wall_seconds=duration))
        if self.fast_job is not None and self.fast_job[0].done():
            job,lease=self.fast_job;self.fast_job=None
            try:
                plan,output,duration=job.result()
                discarded=self.queue.replace(plan,now=float(self.clock()))
            except BaseException as exc:
                self.queue.cancel_query(lease)
                self.record(dict(event="fast_rejected",query_id=lease.query_id,error=type(exc).__name__))
                raise
            self.record(dict(event="fast_committed",query_id=lease.query_id,discarded_rows=discarded,
                             protected_until=lease.protected_until,query_time=lease.query_time,
                             commit_time=float(self.clock()),observation_time=lease.observation_time,
                             reference_version=lease.reference_version,wall_seconds=duration))
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
