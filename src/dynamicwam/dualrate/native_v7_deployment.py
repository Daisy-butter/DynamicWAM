"""Native DOMINO policy: synchronous physics, TOPP and ordinal action commands.

Inference consumes wall time only. A's order is known; TOPP end times are not.
Z uses command events; camera/history timestamps use the scene physical clock.
"""
from collections import deque
import time
import numpy as np
import torch
from .online import ObservationHistory
from .actuators import ActuatorProjection
from .native_v6 import align_plan,Z_EVENTS

PROTOCOL="native_domino_sync_topp_compact_v7"

class NativeDualRatePolicy:
    def __init__(self,system,builder,*,replan_actions=2,refresh_actions=4,max_reference_age=1.44,seed=0):
        if not getattr(system,'native_v7',None) or builder.manifest.get('time_contract')!='native_event_single_v6':
            raise ValueError("native model and native timing data required")
        if replan_actions!=2 or refresh_actions!=4 or max_reference_age!=1.44:
            raise ValueError("untrained native refresh contract")
        if builder.device!=builder.slow_device:
            raise ValueError("this native synchronous implementation uses one policy device")
        self.system=system.eval();self.builder=builder
        self.replan_actions=replan_actions;self.refresh_actions=refresh_actions;self.ttl=max_reference_age
        self.generator=torch.Generator(device=builder.device).manual_seed(seed)
        self.noise_seed=seed
        self.history=ObservationHistory(allow_equal_exposure=True);self.pending=deque()
        self.command_index=0;self.source_index=None;self.world=None;self.records=[]
        self.env=None;self.bounds=None;self.previous_plan=None;self.previous_origin=None

    def _time(self,env):
        clock=getattr(env,'_scene_step_clock',None)
        if clock is None:raise ValueError("actual native scene-step clock required")
        return float(clock.snapshot().time_seconds)

    @torch.no_grad()
    def step(self,env,observation):
        if self.env is None:
            self.env=env;self.bounds=ActuatorProjection.from_robot(env.robot)
        if self.env is not env:raise ValueError("new episode requires a fresh policy/history")
        if self.command_index!=env.take_action_cnt:raise RuntimeError('native command ledger diverged from executor')
        now=self._time(env)
        self.history.append(observation,exposure=now,available=now)
        frames=self.history.snapshot(now)
        if not self.pending:
            begin=time.monotonic();slow_seconds=0.
            if self.world is None or self.command_index-self.source_index>=self.refresh_actions or now-float(self.world['source_time'][0])>self.ttl:
                t=time.monotonic()
                source=self.builder.source(frames,source_time=now,action_start=now,
                    vae=self.system.slow.native.compact_wan.video_model.vae)
                source['source_known_actions']=source['source_initial_state'].new_zeros((1,16,14))
                source['source_known_mask']=torch.zeros((1,16),device=self.builder.slow_device,dtype=torch.bool)
                source['native_source_index']=torch.tensor([self.command_index],device=self.builder.device)
                reference=self.system.predict_reference(source,generator=self.generator)
                self.world=self.system.world_readout(reference)
                self.source_index=self.command_index
                self.world['native_source_index']=torch.tensor([self.source_index],device=self.builder.device)
                self.world['native_available']=torch.ones(1,device=self.builder.device,dtype=torch.bool)
                if self.builder.device.type=='cuda':torch.cuda.synchronize(self.builder.device)
                slow_seconds=time.monotonic()-t
            batch=self.builder.current(frames,query_time=now,action_start=now)
            batch['fast_known_actions']=batch['state'].new_zeros((1,16,14))
            batch['fast_known_mask']=torch.zeros((1,16),device=self.builder.device,dtype=torch.bool)
            batch['native_query_index']=torch.tensor([self.command_index],device=self.builder.device)
            batch['native_source_index']=self.world['native_source_index']
            batch['future_times']=torch.full((1,12),float('nan'),dtype=torch.float64,device=self.builder.device)
            batch['future_events']=batch['native_query_index'][:,None]+torch.tensor(list(Z_EVENTS)*3,device=self.builder.device)[None]
            had_previous=self.previous_plan is not None
            if had_previous:
                batch['previous_actions'],batch['previous_valid']=align_plan(self.previous_plan,self.previous_origin,batch['native_query_index'])
            t=time.monotonic()
            noise=ordinal_action_noise(self.command_index,16,14,self.noise_seed,batch['state'])
            sampled=self.system.sample_fast(batch,self.world,steps=10,generator=self.generator,initial_action_noise=noise)
            # Actual native units and bounds. A changed actuator target invalidates Z.
            raw=sampled['actions'][0].cpu()*self.builder.scale+self.builder.mean
            raw_before=raw.clone()
            raw,projection=self.bounds(raw,torch.zeros(16,dtype=torch.bool))
            if self._time(env)!=now:raise RuntimeError("native physics advanced during inference")
            self.previous_plan=((raw-self.builder.mean)/self.builder.scale).to(self.builder.device)[None]
            self.previous_origin=batch['native_query_index'].clone()
            self.pending.extend(raw[:self.replan_actions].numpy().copy())
            self.records.append(dict(query_physical_time=now,command_index=self.command_index,
                source_physical_time=float(self.world['source_time'][0]),source_command_index=self.source_index,
                slow_seconds=slow_seconds,fast_seconds=time.monotonic()-t,query_seconds=time.monotonic()-begin,
                reference_age=now-float(self.world['source_time'][0]),prefix_rows=0,
                forecast_invalidated=projection['forecast_invalidated'],protocol=PROTOCOL,
                normalized_action_rms=float(sampled['actions'].float().square().mean().sqrt()),
                first_action_raw=raw_before[0].tolist(),first_action_executed_target=raw[0].tolist(),
                projection_changed_coordinates=projection['changed_coordinates'],
                future_command_events=batch['future_events'][0].tolist(),future_physical_end_times_known=False,
                previous_plan_aligned=had_previous,action_prior='event-aligned bounded A10 Gaussian prior; no output blend',
                prior_source_event=self.source_index,prior_row_offset=self.command_index-self.source_index,
                prior_overlap_rows=max(0,16-(self.command_index-self.source_index)),
                prior_maximum=self.system.native_v7['prior_maximum'],graph_ready=self.system._fast_graph_sampler.receipt()['passed']))
        before_count=env.take_action_cnt
        env.take_action(self.pending.popleft(),action_type='qpos')
        if env.take_action_cnt!=before_count+1:raise RuntimeError("native action budget counter changed")
        self.command_index+=1
        after=self._time(env)
        if after<now:raise RuntimeError("native scene clock reversed")
        self.records.append(dict(command_index=self.command_index-1,physical_start=now,physical_end=after,
                                 actual_duration=after-now,kind='native_topp_execute'))

def ordinal_action_noise(start,horizon,dim,seed,like):
    """Overlap has identical base noise; each ordinal still has N(0,I) marginal."""
    rows=[]
    for i in range(start,start+horizon):
        gen=torch.Generator(device='cpu').manual_seed((int(seed)+1000003*i)%((1<<63)-1))
        rows.append(torch.randn(dim,generator=gen))
    return torch.stack(rows)[None].to(like)

