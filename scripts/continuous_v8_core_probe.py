"""Guarded replay-only GPU probe; synthetic inputs, no trained policy/SR claim.

Real 448x6 joint A/Z core and CUDA Graph, on the fast GPU independently of the
simulator GPU. Live probes never change the recorded commands or physics clock.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import torch
from dynamicwam.dualrate.fast import FastConfig
from dynamicwam.dualrate.continuous_v8 import ContinuousResidualFast
from dynamicwam.dualrate.fast_graph import FastGraphSampler


class CoreProbe:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='v8-core-probe')
        self.job=None;self.records=[];self.error=None;self.receipt=None
        try:
            self.pool.submit(self._setup).result(timeout=180)
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def _setup(self):
        torch.cuda.set_device(1)
        torch.set_num_threads(1)
        torch.manual_seed(18031)
        c=FastConfig(dim=448,layers=6,heads=7,observation_dim=448,world_dim=448,
                     future_tokens=12,future_entities=3)
        self.model=ContinuousResidualFast(c).to('cuda:1').eval().requires_grad_(False)
        d=torch.device('cuda:1');rand=lambda *shape:torch.randn(*shape,device=d)
        full=lambda shape,value,dtype:torch.full(shape,value,dtype=dtype,device=d)
        obs=256;world=528
        types=full((1,world),1,torch.long);types[:,:256]=0;types[:,-16:]=2
        intervals=full((1,world,2),0.,torch.float64)
        intervals[:,256:,0]=.1;intervals[:,256:,1]=3.
        target=torch.cat((torch.arange(1,17,device=d)*.18,torch.tensor([.1,.2,.4,.8]*3,device=d)))[None].double()+.18
        conditions=dict(observation_tokens=rand(1,obs,448),observation_valid=full((1,obs),True,torch.bool),
            world_tokens=rand(1,world,448),world_valid=full((1,world),True,torch.bool),
            world_intervals=intervals,world_types=types,world_action_delta=rand(1,world,14),
            world_action_delta_valid=full((1,world),False,torch.bool),target_times=target,
            query_time=full((1,),.18,torch.float64),source_time=full((1,),0.,torch.float64),
            completed_time=full((1,),.1,torch.float64),action_draft=rand(1,16,14),
            draft_valid=full((1,16),True,torch.bool),future_allowed=full((1,),True,torch.bool),
            future_valid=full((1,12),True,torch.bool))
        self.context=self.model.prepare_context(**conditions)
        self.args=(rand(1,16,14),rand(1,12,134),rand(1,16,14),
                   torch.arange(16,device=d)[None]<2,self.context,
                   full((1,12,134),True,torch.bool),full((1,12,2),True,torch.bool))
        self.engine=FastGraphSampler(self.model)
        self.engine.sample(*self.args,steps=10)
        self.engine.seal(expected_shapes=1)
        self.receipt=dict(architecture=vars(c),parameters=sum(p.numel() for p in self.model.parameters()),
            observation_tokens=obs,world_tokens=world,graph=self.engine.receipt(),
            scope='synthetic_context_real_fast_core_only; excludes encoders/slow/IPC/closed_loop')

    @torch.no_grad()
    def _query(self,tick):
        torch.cuda.set_device(1)
        start=time.perf_counter()
        self.engine.sample(*self.args,steps=10)
        torch.cuda.synchronize(1)
        return dict(request_tick=tick,started_wall=start,ready_wall=time.perf_counter(),
                    duration=time.perf_counter()-start)

    def poll(self,tick):
        if self.job is not None and self.job.done():
            self.records.append(self.job.result());self.job=None
        if tick%90==0 and self.job is None:
            self.job=self.pool.submit(self._query,tick)

    def close(self):
        try:
            if self.job is not None:
                self.records.append(self.job.result(timeout=60));self.job=None
        finally:
            self.pool.shutdown(wait=True,cancel_futures=True)

    def report(self):
        return dict(bootstrap=self.receipt,queries=self.records,scored=False,synthetic_conditions=True)
