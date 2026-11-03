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
from contextlib import contextmanager
import time
import torch
from .timing import (CompletedReferenceStore, ExecutionQueue, TimedPlan,
                     TimingError, DeadlineMiss, StalePlan, prefix_digest)
from .physical_clock import WallPhysicsBridge
from .submission import FastMailbox, SubmissionCancelled


@contextmanager
def device_stream(device, stream):
    if device.type == 'cuda':
        with torch.cuda.device(device), torch.cuda.stream(stream):
            yield
    else:
        yield


@dataclass(frozen=True)
class RuntimeConfig:
    action_dt: float
    fast_period_steps: int = 2
    slow_period_steps: int = 4
    protect_steps: int = 2
    submit_margin: float = .01
    reference_max_age: float = 1.
    sample_steps: int = 10
    fast_cuda_graph: bool = False
    min_query_budget_seconds: float = 0.
    max_protect_steps: int | None = None
    submission_contract: str = 'owner_v4'
    fast_requires_new_observation: bool = False

    def __post_init__(self):
        if self.submission_contract not in ('owner_v4','worker_receipt_v8'):
            raise ValueError('unknown fast submission contract')
        if (type(self.fast_cuda_graph) is not bool or type(self.fast_requires_new_observation) is not bool or
                not math.isfinite(self.action_dt) or self.action_dt <= 0 or
                min(self.fast_period_steps,self.slow_period_steps,self.protect_steps,self.sample_steps) < 1 or
                not 0 <= self.submit_margin < self.protect_steps*self.action_dt or
                not math.isfinite(self.reference_max_age) or self.reference_max_age <= 0):
            raise ValueError("invalid physical runtime configuration")
        if (not math.isfinite(self.min_query_budget_seconds) or self.min_query_budget_seconds<0 or
                (self.max_protect_steps is not None and
                 (type(self.max_protect_steps) is not int or self.max_protect_steps<1)) or
                (self.min_query_budget_seconds>0 and
                 (self.max_protect_steps is None or self.max_protect_steps<self.protect_steps))):
            raise ValueError('bounded query budget requires a finite budget and explicit protection cap')


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
    native_mailbox_receipt_refresh=True
    def __init__(self, system, queue: ExecutionQueue, clock: Callable[[],float],
                 config: RuntimeConfig, *, action_mean, action_scale, seed=0, record=None,
                 action_projection=None, clock_bridge: WallPhysicsBridge | None = None):
        if abs(queue.dt-config.action_dt) > 1e-10:
            raise ValueError("runtime/queue action grid differs")
        if system.training:
            raise ValueError("runtime requires eval mode; no optimizer shares deployment weights")
        self.system,self.queue,self.clock,self.config = system,queue,clock,config
        self.clock_bridge = clock_bridge
        self.wall_clock = clock_bridge.wall_now if clock_bridge is not None else time.perf_counter
        self.record = record or (lambda event: None)
        self.action_projection=action_projection
        self.store = CompletedReferenceStore(episode_id=queue.episode_id,clock_id=queue.clock_id,
                                            max_age=config.reference_max_age)
        self.mean = torch.as_tensor(action_mean,dtype=torch.float32)
        self.scale = torch.as_tensor(action_scale,dtype=torch.float32)
        if (self.mean.shape != (system.fast.config.action_dim,) or self.scale.shape != self.mean.shape or
                not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale<=0).any()):
            raise ValueError("exact checkpoint action normalization required")
        device = next(system.fast.parameters()).device
        self.device = device
        self.slow_device = next(system.slow.parameters()).device if hasattr(system, 'slow') else device
        if device.type == 'cuda' and clock_bridge is None:
            raise ValueError('production CUDA runtime requires an explicit wall/physics bridge')
        if device.type == 'cuda' and (self.slow_device.type != 'cuda' or self.slow_device == device):
            raise ValueError('production hard-time fast and slow require distinct CUDA devices')
        if config.fast_cuda_graph and (device.type!='cuda' or clock_bridge is None or clock_bridge.active):
            raise ValueError('fast CUDA Graph requires CUDA and an inactive bootstrap clock')
        self.fast_graph=system.enable_fast_graph() if config.fast_cuda_graph else None
        self._graph_prepared=False
        self.pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix="dualrate")
        self.fast_stream = torch.cuda.Stream(device=device,priority=-1) if device.type=='cuda' else None
        self.slow_stream = torch.cuda.Stream(device=self.slow_device) if self.slow_device.type=='cuda' else None
        self.transfer_stream = torch.cuda.Stream(device=device) if device.type=='cuda' else None
        self.fast_rng = torch.Generator(device=device).manual_seed(seed)
        self.slow_rng = torch.Generator(device=self.slow_device).manual_seed(seed+1)
        self.fast_job = self.slow_job = None
        self.slow_version = 0
        self.closed = False
        self.last_fast_tick = self.last_slow_tick = -10**12
        self.last_slow_source = -math.inf
        self.last_fast_observation = -math.inf
        self._failure = None
        self._cancelled_queries = set()
        self._discarded_jobs = []
        self._accepted_jobs = []
        # CPU store owns defensive copies. This separate ready device payload
        # avoids owner-thread GPU copies; each fast worker clones it on its stream.
        self._published_device_world = None
        self._previous_feedback_plan = None
        self.feedback_seed=int(seed)+1
        self.feedback_recorder=None
        self._feedback_sources={}
        self._reference_lease = None
        if hasattr(system,'continuity_v11'):
            from .reference_lease import ReferenceLease
            self._reference_lease=ReferenceLease(system.continuity_v11['reference_hold_steps'],config.action_dt)
        self._mailbox = FastMailbox(self.wall_clock,clock_bridge) if config.submission_contract=='worker_receipt_v8' else None

    def fast_graph_receipt(self):
        if self.fast_graph is None:
            return dict(enabled=False,passed=False,sealed=False,shape_count=0)
        return dict(self.fast_graph.receipt(),enabled=True)

    def _prepare_fast_graph(self,batch,world):
        """Bootstrap both reference layouts without advancing the query RNG."""
        if self.fast_graph is None or self._graph_prepared:
            return
        if self.clock_bridge is None or self.clock_bridge.active:
            raise TimingError('CUDA Graph capture is forbidden after episode activation')
        state=self.fast_rng.get_state()
        expected_shapes=1 if getattr(self.system,'residual_v10',None) else 2
        if expected_shapes==1:
            from .residual_v10 import verify_projection_parity
            self.system._projection_parity=verify_projection_parity(self.system,batch,world)
            self.record(dict(event='projector_cpu_cuda_parity',**self.system._projection_parity))
        try:
            try:
                self.system.sample_fast(batch,world,steps=self.config.sample_steps,generator=self.fast_rng)
                if expected_shapes==2:
                    self.system.sample_fast(batch,self.system.empty_world(batch),
                                            steps=self.config.sample_steps,generator=self.fast_rng)
            finally:
                self.fast_rng.set_state(state)
            self.fast_graph.seal(expected_shapes=expected_shapes)
            receipt=self.fast_graph_receipt()
            if not receipt.get('passed') or not receipt.get('sealed') or receipt.get('shape_count')!=expected_shapes:
                raise TimingError('fast CUDA Graph parity/shape acceptance did not pass')
        except BaseException as exc:
            self.record(dict(event='fast_graph_preparation_failed',acceptance=self.fast_graph_receipt(),
                             error=type(exc).__name__,error_message=str(exc),rng_preserved=True))
            raise
        self._graph_prepared=True
        self.record(dict(event='fast_graph_prepared',acceptance=receipt,
                         preparation_wall_time=self.wall_clock(),rng_preserved=True))

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

    def start_slow(self, batch, source_prefix, *, source_time=None, request_time=None,
                   source_available_time=None):
        if self.closed or self.slow_job is not None:
            raise TimingError("slow slot unavailable")
        if (source_prefix.episode_id,source_prefix.clock_id)!=(self.queue.episode_id,self.queue.clock_id):
            raise StalePlan("slow commitment snapshot belongs to another episode/clock")
        source=float(source_time if callable(batch) else batch["source_time"].item())
        now=float(self.clock())
        request = now if request_time is None else float(request_time)
        tolerance = self.clock_bridge.numeric_tolerance if self.clock_bridge else 1e-9
        if source>request+tolerance or abs(request-now)>tolerance or source_prefix.start_time>source+tolerance:
            raise TimingError("source/commitment snapshot is in the future")
        available = None if source_available_time is None else float(source_available_time)
        if available is not None and (not math.isfinite(available) or available < source-tolerance or available > request+tolerance):
            raise TimingError('source observation availability does not precede request')
        if self.device.type == 'cuda' and available is None:
            raise TimingError('production source request requires observed availability time')
        if self.clock_bridge is not None:
            if abs(source_prefix.start_time-source)>tolerance:
                raise TimingError('hard-time source prefix must start at the real exposure')
            if source <= self.last_slow_source:
                raise StalePlan('slow source exposure was already requested')
            if (len(source_prefix.actions) >= self.system.fast.config.chunk_size or
                    request >= source+self.system.fast.config.chunk_size*self.queue.dt-tolerance):
                raise StalePlan('source request exhausts native horizon')
        self.slow_version += 1
        version=self.slow_version
        self.last_slow_source = source
        requested_wall = self.wall_clock()
        self.record(dict(event='slow_started',source_time=source,request_time=request,
                         source_available_time=available,request_wall_time=requested_wall,version=version,
                         source_queue_version=source_prefix.queue_version,
                         source_prefix_hash=prefix_digest(source_prefix),
                         prefix_start=source_prefix.start_time,prefix_end=source_prefix.end_time))
        self.last_slow_tick=math.floor((now-self.queue.origin)/self.queue.dt+1e-9)
        def work():
            started=self.wall_clock()
            with device_stream(self.slow_device, self.slow_stream), torch.no_grad():
                local=detached_batch(batch() if callable(batch) else batch)
                local={k:v.to(self.slow_device) for k,v in local.items()}
                if float(local['source_time'].item())!=source:
                    raise TimingError('source preprocessing changed the captured source time')
                for key, expected in (('source_request_time',request),('source_available_time',available)):
                    if key in local and expected is not None and abs(float(local[key].item())-expected)>tolerance:
                        raise TimingError('source preprocessing changed request/availability provenance')
                self._prefix(local,source_prefix,"source")
                # Request time is a causal placeholder for the source-only pass;
                # true completion is assigned by the owner after physical catch-up.
                local["completed_time"]=local["source_time"].new_full((1,),request)
                local["query_time"]=local["source_time"].new_full((1,),request)
                if self._reference_lease is not None:
                    local['feedback_noise_seed']=torch.tensor([self.feedback_seed],device=self.slow_device,dtype=torch.long)
                    local['action_grid_origin']=local['source_time'].new_tensor([self.queue.origin])
                if self.feedback_recorder is not None:
                    from .feedback_replay import cpu_batch
                    self._feedback_sources[version]=(cpu_batch(local),self.slow_rng.get_state().cpu().clone())
                    # Retain active + newest sources; no unbounded per-episode GPU/host cache.
                    active=self._reference_lease.active.version if self._reference_lease.active else -1
                    self._feedback_sources={k:v for k,v in self._feedback_sources.items() if k in (active,version,version-1)}
                model_started_wall=self.wall_clock()
                reference=self.system.predict_reference(local,generator=self.slow_rng)
                world=self.system.world_readout(reference)
                if self.slow_stream is not None:
                    self.slow_stream.synchronize()
                gpu_done_wall=self.wall_clock()
                # Host staging avoids an implicit cross-device/P2P dependency.
                world={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in world.items()}
            with device_stream(self.device, self.transfer_stream):
                device_world={k:v.to(self.device) for k,v in world.items() if isinstance(v,torch.Tensor)}
                if self.transfer_stream is not None:
                    self.transfer_stream.synchronize()
            ready_wall=self.wall_clock()
            # No worker reads the advancing physics clock. Publication is an
            # owner transition after actual steps reach this wall-ready bound.
            return world,device_world,ready_wall,dict(gpu_done_wall_time=gpu_done_wall,
                transfer_ready_wall_time=ready_wall,worker_started_wall_time=started,
                model_started_wall_time=model_started_wall,
                request_time=request,wall_seconds=ready_wall-requested_wall)
        self.slow_job=(self.pool.submit(work),source,source_prefix,version)

    def start_fast(self,batch_factory,*,observation_time):
        if self.closed or self.fast_job is not None:
            raise TimingError("fast slot unavailable")
        if self.config.fast_requires_new_observation and observation_time<=self.last_fast_observation:
            raise StalePlan('fast query requires a newly available real exposure')
        requested_wall=self.wall_clock()
        query=float(self.clock())
        protection,budget=self._fast_query_protection(query,requested_wall)
        reference=self.store.snapshot(query)
        device_reference = self._published_device_world
        if self._reference_lease is not None:
            first=self.queue.origin+(math.floor((query-self.queue.origin)/self.queue.dt+1e-9)+protection)*self.queue.dt
            reference,device_reference,selection=self._reference_lease.select(reference,device_reference,now=query,first=first)
            self.record(selection)
        previous_feedback=self._previous_feedback_plan
        if reference is not None and (device_reference is None or device_reference[0] != reference.version):
            raise TimingError('published CPU/device reference version differs')
        lease=self.queue.begin_query(query_time=query,observation_time=observation_time,
            protect_steps=protection,submit_margin=self.config.submit_margin,
            reference_version=reference.version if reference else -1)
        self.last_fast_tick=math.floor((query-self.queue.origin)/self.queue.dt+1e-9)
        self.last_fast_observation=float(observation_time)
        if self._mailbox is not None:self._mailbox.register(lease)
        budget['effective_protect_steps']=len(lease.prefix.actions)
        self.record(dict(event='fast_started',query_id=lease.query_id,query_time=query,
            request_wall_time=requested_wall,
            observation_time=observation_time,reference_version=lease.reference_version,
            source_time=reference.source_time if reference else None,
            reference_age=query-reference.source_time if reference else None,
            protected_until=lease.protected_until,submit_margin=lease.submit_margin,prefix_hash=lease.prefix_hash,
            remaining_prefix_budget_seconds=lease.protected_until-lease.submit_margin-query,**budget))
        def work():
            worker_started_wall=self.wall_clock()
            with device_stream(self.device, self.fast_stream), torch.no_grad():
                if getattr(self.system,'residual_v10',None):
                    self.system._residual_trace=lambda stage:self.record(dict(event='fast_stage',query_id=lease.query_id,
                        stage=stage,wall_time=self.wall_clock(),clock='CPU enqueue/return; no added CUDA synchronization'))
                    self.system._residual_trace('batch_start')
                batch=detached_batch(batch_factory(lease,reference))
                batch={k:v.to(self.device) for k,v in batch.items()}
                if getattr(self.system,'residual_v10',None):self.system._residual_trace('batch_done')
                if float(batch["query_time"].item()) != query:
                    raise TimingError("batch query differs from queue lease")
                self._prefix(batch,lease.prefix,"fast")
                if self._reference_lease is not None and previous_feedback is not None:
                    from .continuity_v11 import align
                    previous,intervals,valid,previous_version=previous_feedback[:4]
                    batch['previous_actions'],batch['previous_valid']=align(previous.to(self.device),
                        intervals.to(self.device),batch['action_intervals'],valid.to(self.device))
                    batch['previous_reference_same']=torch.tensor([previous_version==lease.reference_version],device=self.device)
                    if hasattr(self.system,'handoff_v12'):
                        batch['previous_query_time']=batch['query_time'].new_tensor([previous_feedback[4]])
                        batch['previous_handoff']=torch.tensor([previous_feedback[5]],device=self.device)
                if getattr(self.system,'residual_v10',None):
                    from .residual_v10 import anchor_future
                    batch['future_times']=anchor_future(batch,self.system.fast.config.future_entities)
                if reference:
                    world={k:v.clone() for k,v in device_reference[1].items()}
                    # Publication time is assigned on the owner, after transfer.
                    world['completed_time']=reference.payload['completed_time'].to(self.device)
                else:
                    world=self.system.empty_world(batch)
                self._prepare_fast_graph(batch,world)
                model_started_wall=self.wall_clock()
                output=self.system.sample_fast(batch,world,steps=self.config.sample_steps,generator=self.fast_rng)
                raw=output["actions"][0].float().cpu()*self.scale+self.mean
                if 'baseline_actions' in output:
                    base=output['baseline_actions'][0].float().cpu()*self.scale+self.mean
                    self.record(dict(event='residual_action_audit',query_id=lease.query_id,
                        action_intervals=batch['action_intervals'][0].cpu().tolist(),
                        baseline_raw=base.tolist(),proposed_raw=raw.tolist(),
                        correction_raw=(raw-base).tolist(),draft_valid=output['draft_valid'][0].cpu().tolist(),
                        correction_accepted=bool(output['correction_accepted'][0]),
                        observation_times=batch['rgb_times'][0].cpu().tolist(),
                        reference_version=lease.reference_version))
                    if hasattr(self.system,'handoff_v12'):
                        original=output['raw_baseline_actions'][0].float().cpu()*self.scale+self.mean
                        learned=output['learned_correction_actions'][0].float().cpu()*self.scale
                        self.record(dict(event='whole_plan_handoff_audit',query_id=lease.query_id,
                            reference_version=lease.reference_version,baseline_kind='explicit_connector_not_raw_Full',
                            raw_full_baseline=original.tolist(),handoff_offset=(base-original).tolist(),learned_correction=learned.tolist(),
                            active=bool(output['handoff_active'][0]),enabled=bool(output['handoff_enabled'][0]),
                            alpha=float(output['handoff_alpha'][0]),budget_rad=float(output['handoff_budget_rad'][0]),
                            raw_revision_rad=float(output['handoff_raw_change_rad'][0]),revision_rad=float(output['handoff_revision_rad'][0])))
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
                if self.fast_stream is not None:
                    self.fast_stream.synchronize()
                gpu_done_wall=self.wall_clock()
                plan=TimedPlan(start,self.config.action_dt,tuple(map(tuple,raw.tolist())),
                               lease.base_queue_version+1,query,lease)
                # Include host plan construction in readiness; it is not free.
                ready_wall=self.wall_clock()
                result=plan,output,ready_wall,dict(wall_seconds=ready_wall-requested_wall,
                    worker_started_wall_time=worker_started_wall,model_started_wall_time=model_started_wall,
                    gpu_done_wall_time=gpu_done_wall)
                if self._mailbox is not None:
                    try:receipt=self._mailbox.submit(lease,result)
                    except SubmissionCancelled:
                        self.record(dict(event='fast_cancelled_at_terminal',query_id=lease.query_id,
                                         ready_wall_time=ready_wall,discarded=True))
                        return result
                    self.record(dict(event='fast_submitted',query_id=lease.query_id,
                        submitted_wall_time=receipt.submitted_wall,ready_wall_time=ready_wall,
                        earliest_physical_time=receipt.earliest_physical_time,
                        protected_until=lease.protected_until,prefix_hash=lease.prefix_hash,
                        submission_contract=self.config.submission_contract))
                if self.feedback_recorder is not None and self.clock_bridge.active:
                    self.feedback_recorder.capture(lease.query_id,self._feedback_sources[lease.reference_version],batch,
                        output,reference)
                return result
        try:
            self.fast_job=(self.pool.submit(work),lease)
        except BaseException:
            self.queue.cancel_query(lease)
            if self._mailbox is not None:self._mailbox.finish(lease)
            raise

    def _fast_query_protection(self,query,requested_wall):
        """Choose a bounded prefix once, before acquiring the immutable lease."""
        cfg=self.config;dt=self.queue.dt
        index=math.floor((query-self.queue.origin)/dt+1e-9)
        phase=max(0.,query-(self.queue.origin+index*dt))
        bridge=self.clock_bridge
        active=bridge is not None and bridge.active
        lag=max(0.,bridge.expected_physics(requested_wall)-query) if active else 0.
        enabled=cfg.min_query_budget_seconds>0 and active
        count=cfg.protect_steps
        details=dict(requested_query_budget_seconds=cfg.min_query_budget_seconds,
            effective_protect_steps=count,query_phase_seconds=phase,
            wall_physics_lag_seconds=lag,budget_decision_wall_time=requested_wall,
            bounded_query_budget_enabled=enabled,max_protect_steps=cfg.max_protect_steps)
        if cfg.min_query_budget_seconds>0 and bridge is None:
            raise TimingError('bounded query budget requires a wall/physics bridge')
        if enabled:
            count=max(count,math.ceil((phase+lag+cfg.min_query_budget_seconds+cfg.submit_margin)/dt))
            # Existing commitments cannot be shortened to satisfy the cap.
            existing=self.queue.snapshot_committed(query)
            count=max(count,len(existing.actions))
            details['effective_protect_steps']=count
            if count>cfg.max_protect_steps:
                self.record(dict(event='fast_budget_rejected',query_time=query,
                    reason='required_prefix_exceeds_configured_cap',**details))
                raise DeadlineMiss('required query budget exceeds bounded protected prefix')
        return count,details

    def _cancel_fast(self, lease):
        if lease.query_id not in self._cancelled_queries:
            self.queue.cancel_query(lease)
            if self._mailbox is not None:self._mailbox.finish(lease)
            self._cancelled_queries.add(lease.query_id)

    def poll(self):
        """Caller invokes alongside continuous physics; never waits on unfinished work."""
        if self._failure is not None:
            raise TimingError('runtime has failed; close owned workers before continuing') from self._failure
        receipt = None
        for accepted in tuple(self._accepted_jobs):
            if accepted.done():
                try:accepted.result()
                except BaseException as exc:
                    self._failure=exc
                    raise
                self._accepted_jobs.remove(accepted)
        # A deadline is a limit on execution, not merely a check when a worker
        # eventually returns. Fail before issuing the old uncommitted tail.
        if self.fast_job is not None and (self.clock_bridge is None or self.clock_bridge.active):
            _,lease=self.fast_job
            try:
                if self._mailbox is not None:
                    receipt=self._mailbox.check_pending(lease)
                if float(self.clock()) >= lease.protected_until-lease.submit_margin:
                    raise DeadlineMiss('in-flight query exceeded physical submission deadline')
                if self.clock_bridge is not None and self._mailbox is None:
                    self.clock_bridge.check_deadline(lease.protected_until,lease.submit_margin)
            except BaseException as exc:
                self._cancel_fast(lease)
                self._failure=exc
                self.record(dict(event='fast_rejected',query_id=lease.query_id,error=type(exc).__name__,
                                 reason='inflight_deadline',worker_done=self.fast_job[0].done()))
                raise
        elif self.fast_job is not None and self._mailbox is not None:
            receipt=self._mailbox.peek(self.fast_job[1])
        if self.slow_job is not None and self.slow_job[0].done():
            job,source,prefix,version=self.slow_job
            world,device_world,ready_wall,metrics=job.result()
            completed=float(self.clock())
            if self.clock_bridge is None or self.clock_bridge.ready(ready_wall,completed):
                self.slow_job=None
                payload={k:v for k,v in world.items() if isinstance(v,torch.Tensor)}
                payload['completed_time']=torch.tensor([completed],dtype=torch.float64)
                intervals=payload["world_intervals"][0][payload["world_types"][0]==1]
                times=tuple(sorted(set(intervals.flatten().tolist())))
                accepted=self.store.publish(source_time=source,completed_time=completed,version=version,
                    generation_stage=str(world["generation_stage"]),future_times=times,
                    future_valid=(True,)*len(times),payload=payload,now=completed,
                    episode_id=self.queue.episode_id,clock_id=self.queue.clock_id,
                    source_queue_version=prefix.queue_version,source_prefix_hash=prefix_digest(prefix))
                if accepted:
                    self._published_device_world=(version,device_world)
                self.record(dict(event="slow_completed",source_time=source,completed_time=completed,
                    published_time=completed,publication_wall_time=self.wall_clock(),
                    version=version,accepted=accepted,**metrics))
        if self.fast_job is not None and (receipt is not None or self.fast_job[0].done()):
            job,lease=self.fast_job
            submitted=False
            try:
                plan,output,ready_wall,metrics=receipt.result if receipt is not None else job.result()
                if self._mailbox is not None and receipt is None:
                    # A fast worker can submit while slow publication runs above.
                    # Once result() returns, submit MUST already have happened.
                    # Re-read the atomic mailbox instead of delaying ready slow
                    # references until the fast slot happens to be idle.
                    receipt=self._mailbox.peek(lease)
                    if receipt is None:
                        raise TimingError('completed fast job lacks its on-time submission receipt')
                    plan,output,ready_wall,metrics=receipt.result
                if not isinstance(metrics,dict):
                    metrics=dict(wall_seconds=metrics)  # legacy CPU fixture
                if self.clock_bridge is not None:
                    if receipt is None:
                        self.clock_bridge.check_deadline(lease.protected_until,lease.submit_margin,ready_wall=ready_wall)
                    release_wall=ready_wall if receipt is None else receipt.submitted_wall
                    if not self.clock_bridge.ready(release_wall,float(self.clock())):
                        return None
                discarded=self.queue.replace(plan,now=float(self.clock()))
                if self._reference_lease is not None:
                    p=torch.tensor(plan.actions,dtype=torch.float32)[None]
                    starts=torch.arange(p.shape[1],dtype=torch.float64)*plan.dt+plan.start_time
                    intervals=torch.stack((starts,starts+plan.dt),-1)[None]
                    valid=output['draft_valid'].detach().cpu().clone()
                    self._previous_feedback_plan=((p-self.mean)/self.scale,intervals,valid,lease.reference_version)
                    if hasattr(self.system,'handoff_v12'):
                        self._previous_feedback_plan+=(lease.query_time,bool(output['handoff_active'].item()))
                submitted=True
                if self.clock_bridge is not None and receipt is None:
                    self.clock_bridge.check_deadline(lease.protected_until,lease.submit_margin,ready_wall=ready_wall)
                if self._mailbox is not None:self._mailbox.finish(lease)
                if receipt is not None:self._accepted_jobs.append(job)
                self.fast_job=None
            except BaseException as exc:
                self.fast_job=None
                self._discarded_jobs.append(('fast',(job,lease),dict(
                    reason='failed_submission',queue_updated_before_failure=submitted)))
                if not submitted:
                    self._cancel_fast(lease)
                self._failure=exc
                self.record(dict(event="fast_rejected",query_id=lease.query_id,error=type(exc).__name__,
                                 queue_updated_before_failure=submitted))
                raise
            self.record(dict(event="fast_committed",query_id=lease.query_id,discarded_rows=discarded,
                             protected_until=lease.protected_until,submit_margin=lease.submit_margin,query_time=lease.query_time,
                             commit_time=float(self.clock()),observation_time=lease.observation_time,
                             commit_wall_time=self.wall_clock(),ready_wall_time=ready_wall,
                             submitted_wall_time=ready_wall if receipt is None else receipt.submitted_wall,
                             submission_contract=self.config.submission_contract,
                             reference_version=lease.reference_version,**metrics))
            return output
        return None

    def close(self):
        if self.closed:
            return
        self.closed=True
        if self._mailbox is not None:self._mailbox.close()
        self.pool.shutdown(wait=True,cancel_futures=False)
        errors=[]
        for job in self._accepted_jobs:
            try:job.result()
            except BaseException as exc:errors.append(exc)
        self._accepted_jobs.clear()
        remaining=list(self._discarded_jobs)
        if self.slow_job is not None:
            remaining.append(('slow',self.slow_job,dict(reason='episode_closed_before_publication')))
        if self.fast_job is not None:
            remaining.append(('fast',self.fast_job,dict(reason='episode_closed_before_commit')))
        seen=set()
        for route,pending,detail in remaining:
            if pending is not None:
                job=pending[0]
                if id(job) in seen:
                    continue
                seen.add(id(job))
                event=dict(event='worker_finished_discarded',route=route,
                    discarded=True,commit_performed_during_close=False,
                    owner_observed_wall_time=self.wall_clock(),**detail)
                if route=='fast':
                    lease=pending[1]
                    event.update(query_id=lease.query_id,query_time=lease.query_time,
                                 protected_until=lease.protected_until)
                else:
                    event.update(source_time=pending[1],version=pending[3])
                try:
                    result=job.result()
                    ready_wall,metrics=result[-2:]
                    if not isinstance(metrics,dict):
                        metrics=dict(wall_seconds=metrics)
                    event.update(worker_outcome='completed',ready_wall_time=ready_wall,**metrics)
                except BaseException as exc:
                    event.update(worker_outcome='failed',error=type(exc).__name__,error_message=str(exc))
                    errors.append(exc)
                self.record(event)
        if self.fast_job is not None:
            self._cancel_fast(self.fast_job[1]);self.fast_job=None
        self.slow_job=None
        self._discarded_jobs.clear()
        if self._mailbox is not None:self._mailbox.close()
        if self.fast_graph is not None and getattr(self.system,'_fast_graph_sampler',None) is self.fast_graph:
            self.system._fast_graph_sampler=None
        if errors:
            raise errors[0]
