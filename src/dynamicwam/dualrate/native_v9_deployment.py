"""Independent background slow updates in native synchronous DOMINO.

Only TOPP steps physics. Slow snapshots may include the currently committed
waypoint; fast queries occur at command boundaries and have zero hard prefix.
"""
from collections import deque
import time
import numpy as np
import torch
from .native_history import NativePhysicalHistory
from .native_v7_deployment import ordinal_action_noise
from .native_v9 import draft
from .actuators import ActuatorProjection

PROTOCOL='native_topp_async_full_residual_v9'


def device_tree(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:device_tree(v,device) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(device_tree(v,device) for v in value)
    return value


class NativeResidualPolicy:
    def __init__(self,system,builder,service,*,seed=0,record_query=None):
        if not getattr(system,'native_v9',None):raise ValueError('V9 model required')
        if builder.manifest.get('motion_sampling')!='raw_frames_v1':raise ValueError('raw motion parity required')
        self.system=system;self.builder=builder;self.service=service;self.seed=seed
        self.generator=torch.Generator(device=builder.device).manual_seed(seed)
        self.sensors=NativePhysicalHistory(on_capture=self._capture)
        self.pending=deque();self.command_index=0;self.world=None;self.source=None
        self.records=[];self.env=None;self.current_target=None;self.last_request=-1e30;self.last_request_index=-100
        self.record_query=record_query;self.consumed=set();self.version=0;self.completed=None;self.bootstrap_checked=False

    def _time(self,env):return float(env._scene_step_clock.snapshot().time_seconds)

    def _publish(self,wait=False):
        result=self.service.poll(wait)
        if result is not None:self.completed=result
        result=self.completed;self.completed=None
        if result is None:return
        now=self._time(self.env);r=result['reference'];source=result['source']
        s=float(r['source_time'][0]);complete=result['complete_physical']
        if not s<=complete<=now+1e-8:raise RuntimeError('noncausal native publication')
        index=int(r['native_source_index'][0])
        if self.world is not None and index<int(self.world['native_source_index'][0]):raise RuntimeError('out-of-order publication')
        r['completed_time'].fill_(complete);source['completed_time'].fill_(complete)
        if not torch.equal(r['reference_actions'][source['source_known_mask']],source['source_known_actions'][source['source_known_mask']]):
            raise RuntimeError('slow prediction changed its committed waypoint')
        if not self.bootstrap_checked:
            check=device_tree(source,self.builder.device);check['query_time']=check['completed_time'].clone()
            expected=self.system.predict_reference(check)
            for key in ('reference_actions','condition_latent','predicted_video_latent','future_hidden'):
                if not torch.equal(expected[key].cpu(),r[key]):raise RuntimeError('independent Full source parity failed: '+key)
            self.bootstrap_checked=True
            self.records.append(dict(kind='bootstrap_full_parity',passed=True,exact=True))
        reference=device_tree(r,self.builder.device)
        self.world=self.system.world_readout(reference)
        self.world['native_available']=torch.ones(1,device=self.builder.device,dtype=torch.bool)
        self.source=source;self.version=result['request_id']
        self.records.append(dict(kind='slow_publish',version=self.version,source_index=index,source_time=s,
            complete_physical=complete,consumed_at_physical=now,complete_wall=result['complete_wall'],
            compute_wall_seconds=result['compute_wall_seconds'],model_start_wall=result.get('model_start_wall'),model_end_wall=result.get('model_end_wall'),source_inflight=bool(source['source_inflight'][0])))

    def _request(self,frames,*,force=False):
        now=self._time(self.env)
        if self.service.future is not None or (not force and self.command_index-self.last_request_index<self.system.native_v9['slow_refresh_commands']):return False
        if not frames or abs(frames[-1].exposure-now)>1e-8:raise RuntimeError('source snapshot is not current owner exposure')
        committed=None if self.current_target is None else ((self.current_target-self.builder.mean.numpy())/self.builder.scale.numpy()).copy()
        from .online import history_endpoints
        selected,_=history_endpoints(frames,action_dt=self.builder.action_dt)
        selected={f.exposure:f for f in selected}
        frames=tuple(selected[k] for k in sorted(selected))
        self.service.submit(dict(frames=frames,source_time=now,index=self.command_index,committed=committed,
            seed=(self.seed+1000003*self.command_index+round(now*1e6))%((1<<63)-1),
            manifest=self.builder.manifest,language=self.builder.language),self.env._scene_step_clock)
        self.last_request=now;self.last_request_index=self.command_index
        self.records.append(dict(kind='slow_request',version=self.service.counter,source_index=self.command_index,
            source_time=now,request_wall=time.monotonic(),committed_rows=int(committed is not None)))
        return True

    def _capture(self,frames):
        # Called by the sole simulator owner after its normal render/capture.
        # No model forward, IPC wait, extra scene step or extra render here.
        completed=self.service.poll()
        if completed is not None:self.completed=completed
        self._request(frames)

    def _usable(self,now):
        if self.world is None:return False
        offset=self.command_index-int(self.world['native_source_index'][0])
        return 0<=offset<16

    @torch.no_grad()
    def step(self,env,observation):
        if self.env is None:self.env=env;self.bounds=ActuatorProjection.from_robot(env.robot)
        if self.env is not env or self.command_index!=env.take_action_cnt:raise RuntimeError('episode/command ledger mismatch')
        now=self._time(env);frames=self.sensors.query_frames(env,observation)
        if not self.pending:
            begin=time.monotonic();self._publish()
            wait_reason=None
            if not self._usable(now):
                wait_reason='bootstrap' if self.world is None else 'no_valid_reference'
                # Exceptional missing/expired draft: freeze native physics and
                # get a fresh boundary snapshot. Never execute a masked suffix.
                self._publish(wait=True)
                if not self._usable(now):
                    self._request(frames,force=True);self._publish(wait=True)
                if not self._usable(now):raise RuntimeError('fresh source did not supply executable draft')
            batch=self.builder.current(frames,query_time=now,action_start=now)
            batch.update(native_query_index=torch.tensor([self.command_index],device=self.builder.device),
                native_source_index=self.world['native_source_index'],
                fast_known_actions=batch['state'].new_zeros(1,16,14),
                fast_known_mask=torch.zeros(1,16,dtype=torch.bool,device=self.builder.device),
                future_times=torch.full((1,12),float('nan'),dtype=torch.float64,device=self.builder.device))
            a10,valid=draft(self.world,batch,self.system.native_v9['reference_ttl'])
            if not bool(valid[:,:1].all()):raise RuntimeError('uncovered execution prefix')
            t=time.monotonic();overlap=self.service.future is not None
            sample=self.system.sample_fast(batch,self.world,steps=10,generator=self.generator,
                initial_action_noise=ordinal_action_noise(self.command_index,16,14,self.seed,batch['state']))
            raw=sample['actions'][0].cpu()*self.builder.scale+self.builder.mean
            raw,project=self.bounds(raw,torch.zeros(16,dtype=torch.bool))
            if self._time(env)!=now:raise RuntimeError('native physics advanced during fast query')
            self.pending.extend(raw[:1].numpy().copy())
            self.records.append(dict(kind='fast_query',command_index=self.command_index,query_physical_time=now,
                source_command_index=int(self.world['native_source_index'][0]),source_physical_time=float(self.world['source_time'][0]),
                completed_physical_time=float(self.world['completed_time'][0]),version=self.version,
                reference_age=now-float(self.world['source_time'][0]),wait_reason=wait_reason,
                world_context_fresh=now-float(self.world['source_time'][0])<=self.system.native_v9['reference_ttl']+1e-8,
                prior_row_offset=self.command_index-int(self.world['native_source_index'][0]),
                slow_inflight_at_fast_start=overlap,fast_start_wall=t,fast_end_wall=time.monotonic(),fast_seconds=time.monotonic()-t,query_seconds=time.monotonic()-begin,
                prefix_rows=0,reference_reused=self.version in self.consumed,
                raw_correction_max=float(((sample['actions']-a10)[0,:1].cpu()*self.builder.scale).abs().max()),
                forecast_invalidated=project['forecast_invalidated']))
            self.consumed.add(self.version)
            if self.record_query is not None:self.record_query(batch,self.source,frames,self.records[-1])
        self.current_target=self.pending.popleft()
        before=len(self.sensors.capture_events)
        action_wall=time.monotonic()
        with self.sensors.during_action(env):env.take_action(self.current_target,action_type='qpos')
        self.current_target=None;self.command_index+=1
        if self.command_index!=env.take_action_cnt:raise RuntimeError('TOPP command count mismatch')
        self.records.extend(dict(e,kind='native_history_capture') for e in self.sensors.capture_events[before:])
        self.records.append(dict(kind='native_topp_execute',command_index=self.command_index-1,
            physical_start=now,physical_end=self._time(env),actual_duration=self._time(env)-now,
            execute_start_wall=action_wall,execute_end_wall=time.monotonic()))

    def finish(self):
        self._publish(wait=True)
