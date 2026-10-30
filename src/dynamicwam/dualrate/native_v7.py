"""Compact single A/Z expert with event-aligned, bounded Full action prior.

Native V6 observation/data/executor semantics are preserved. Seconds and command
ordinals never share an arithmetic origin. No native motor is deployed in fast.
"""
from types import MethodType
import torch
from torch import nn
from .fast import FastJointWAM, QueryContext, _version
from .native_v6 import align_plan, conditions as event_conditions, configure as configure_v6
from .system import DualRateSystem


class CompactEventFast(FastJointWAM):
    def __init__(self, config):
        super().__init__(config)
        d=config.dim
        self.event_embedding=nn.Linear(3,d)
        self.world_event_embedding=nn.Linear(2,d)
        self.previous_in=nn.Linear(config.action_dim,d)

    def prepare_context(self, *, observation_tokens, observation_valid, world_tokens,
            world_valid, world_intervals, world_types, query_time, source_time, completed_time,
            native_source_index, native_query_index, native_available, future_valid=None,
            previous_actions=None, previous_valid=None, **unused):
        c=self.config;b=len(query_time);dev=query_time.device;n=c.chunk_size+c.future_tokens
        DualRateSystem._validate_clocks(dict(source_time=source_time,completed_time=completed_time,query_time=query_time))
        if native_source_index.dtype!=torch.long or native_query_index.dtype!=torch.long or (native_query_index<native_source_index).any():
            raise ValueError('invalid native command origins')
        if world_intervals.dtype!=torch.float64 or world_intervals.shape!=(*world_valid.shape,2):
            raise ValueError('explicit FP64 world support required')
        if ((world_types!=0)&(world_types!=1)).any():
            raise ValueError('A10 is a direct soft prior, never an attention token')
        if not torch.isfinite(world_intervals).all() or (world_intervals[...,1]<world_intervals[...,0]).any():
            raise ValueError('invalid world support')
        if future_valid is None:future_valid=torch.ones((b,c.future_tokens),device=dev,dtype=torch.bool)
        if previous_actions is None:
            previous_actions=observation_tokens.new_zeros((b,c.chunk_size,c.action_dim))
            previous_valid=torch.zeros((b,c.chunk_size),device=dev,dtype=torch.bool)
        if previous_valid is None or previous_actions.shape!=(b,c.chunk_size,c.action_dim) or previous_valid.shape!=(b,c.chunk_size):
            raise ValueError('previous plan must already be joined by command ordinal')
        inputs=dict(observation_tokens=observation_tokens,observation_valid=observation_valid,
            world_tokens=world_tokens,world_valid=world_valid,world_intervals=world_intervals,world_types=world_types,
            query_time=query_time,source_time=source_time,completed_time=completed_time,
            native_source_index=native_source_index,native_query_index=native_query_index,native_available=native_available,
            future_valid=future_valid,previous_actions=previous_actions,previous_valid=previous_valid)
        if any(_version(v) is None for v in inputs.values()):raise ValueError('versioned query tensors required')
        age=query_time-source_time;lag=completed_time-source_time
        offsets=torch.cat((torch.arange(1,c.chunk_size+1,device=dev),
            torch.tensor([1,2,4,8],device=dev).repeat(c.future_entities)))
        if len(offsets)!=n:raise ValueError('four future event offsets per entity required')
        targets=native_query_index[:,None]+offsets[None]
        seconds=torch.stack((age,lag,torch.zeros_like(age),torch.zeros_like(age)),-1)
        seconds=self.query_time_embedding(seconds)[:,None].expand(-1,n,-1)
        ordinal=torch.stack((offsets[None].expand(b,-1),targets-native_source_index[:,None],
            (native_query_index-native_source_index)[:,None].expand(-1,n)),-1).float()/16.
        times=seconds+self.event_embedding(ordinal)
        current=self.observation_in(torch.where(observation_valid[...,None],observation_tokens,0.))+self.type_embedding.weight[2]
        previous=self.previous_in(torch.where(previous_valid[...,None],previous_actions,0.))+self.action_position.weight[None]
        current=torch.cat((current,previous),1)
        cv=torch.cat((observation_valid,previous_valid),1)
        hist=world_types==0;pred=world_types==1
        start,end=world_intervals.unbind(-1)
        eligible=world_valid & native_available[:,None] & (age[:,None]<=1.44+1e-8)
        allowed=eligible[:,None] & ((hist[:,None] & (end[:,None]<=query_time[:,None,None]+1e-8)) |
            (pred[:,None] & (targets[...,None]>=start[:,None]) & (targets[...,None]<=end[:,None])))
        readable=allowed.any(1)
        slow=self.world_in(torch.where(readable[...,None],world_tokens,0.))+self.type_embedding.weight[3]+self.world_type_embedding(world_types)
        physical=torch.where(hist[...,None],world_intervals-query_time[:,None,None],0.)
        event=torch.where(pred[...,None],world_intervals-native_query_index[:,None,None],0.)
        slow=slow+torch.where(hist[...,None],self.world_time_embedding(physical),self.world_event_embedding(event.float()/16.))
        ca=cv[:,None].expand(-1,n,-1)
        ca=torch.cat((~ca.any(-1,keepdim=True),ca),-1)
        allowed=torch.cat((~allowed.any(-1,keepdim=True),allowed),-1)
        kv=tuple(block.prepare_context(current,slow) for block in self.blocks)
        return QueryContext(id(self),inputs,{k:_version(v) for k,v in inputs.items()},
            tuple((id(p),_version(p)) for p in self.parameters()),kv,ca,allowed,times,
            times.new_zeros(b),future_valid)


def aligned_reference(batch,world,*,maximum=.15):
    """Only exact ordinal overlap; expired/missing suffix has zero influence."""
    DualRateSystem._validate_clocks(dict(source_time=world['source_time'],completed_time=world['completed_time'],query_time=batch['query_time']))
    plan,valid=align_plan(world['reference_actions'],world['native_source_index'],batch['native_query_index'])
    age=batch['query_time']-world['source_time']
    available=world.get('native_available',torch.ones_like(age,dtype=torch.bool))
    valid=valid & available[:,None] & (age[:,None]<=1.44+1e-8) & ~batch['fast_known_mask']
    if not torch.isfinite(plan).all():raise ValueError('nonfinite slow action reference')
    weight=maximum*torch.exp(-age/.72).to(plan.dtype)[:,None]*valid
    # It is a bounded noise-distribution prior, not an average of final actions.
    center=plan.clamp(-3.,3.)*weight[...,None]
    center=center.clone();center[:,:,[6,13]]=0. # preserve independent gripper decisions
    return center,plan,valid,weight


def overlapping_noise(noise,current,previous,generator=None):
    """Reuse Gaussian draws for the same command, as deployment does."""
    b,h,d=noise.shape
    indices=previous[:,None]-current[:,None]+torch.arange(h,device=noise.device)[None]
    valid=(indices>=0)&(indices<h)
    drawn=torch.randn(noise.shape,device=noise.device,dtype=noise.dtype,generator=generator)
    joined=noise.gather(1,indices.clamp(0,h-1)[...,None].expand(-1,-1,d))
    return torch.where(valid[...,None],joined,drawn)


def conditions(self,batch,world):
    r=event_conditions(self,batch,world)
    r.pop('native_batch');r.pop('native_video_cache')
    return r


def configure(model,settings,supervision):
    if set(settings)!={'prior_maximum','reference_weight'} or not 0<settings['prior_maximum']<=.2 or not 0<=settings['reference_weight']<=.05:
        raise ValueError('bounded compact slow-action constraint required')
    configure_v6(model,supervision)
    model.native_v7=dict(settings)
    model.conditions=MethodType(conditions,model)
    model.action_center=MethodType(lambda self,b,w:aligned_reference(b,w,maximum=self.native_v7['prior_maximum'])[0],model)
    model.training_loss=MethodType(training_loss,model)
    # Keep the already-trained native schedule semantics in eager AND CUDA Graph.
    from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler
    schedule=dict(model.slow.schedule_config);schedule.pop('action_steps');schedule.pop('video_steps')
    scheduler=FlowMatchScheduler(num_inference_steps=10,**schedule)
    scheduler.set_timesteps(num_inference_steps=10,training=False)
    model.fast.register_buffer('sampling_sigmas',scheduler.sigmas.float(),persistent=False)


# training_loss is appended from the preserved V6 objectives below; only the
# compact prior and shared-overlap noise are added. No changed labels or clocks.

from torch.nn import functional as F
from .native_v6 import validate_events,current_source
from .system import masked_mean,fm_input

def training_loss(self,batch,generator=None,action_conditioned=None):
    validate_events(batch);self.slow.eval()
    a=batch['fast_action'];z=batch['future'];b=len(a);dev=a.device;c=self.native_v6
    base_noise=torch.randn(a.shape,device=dev,dtype=a.dtype,generator=generator)
    old=self.predict_reference(batch,generator=generator)
    world=self.world_readout(old)
    keep=(torch.rand(b,device=dev,generator=generator)>=self.loss_config.cache_dropout)
    world['native_available']=keep
    world['world_valid']=world['world_valid'] & keep[:,None]
    teacher_batch=current_source(batch)
    teacher=old if torch.equal(batch['native_source_index'],batch['native_query_index']) else self.predict_reference(teacher_batch,generator=generator)
    if not torch.equal(teacher['source_time'],batch['query_time']) or not torch.equal(teacher['native_source_index'],batch['native_query_index']):
        raise ValueError('A10 teacher does not start at this query/event')
    target=teacher['reference_actions'].detach()
    if target.shape!=a.shape:raise ValueError('teacher horizon mismatch')
    # Paired queries share an event identity; previous plans are soft proposals,
    # never represented as executed history or ground-truth future.
    previous={k[9:]:v for k,v in batch.items() if k.startswith('previous_') and k not in ('previous_actions','previous_valid')}
    batch=dict(batch)
    prior=a.new_zeros(a.shape);pv=torch.zeros(a.shape[:2],dtype=torch.bool,device=dev)
    if previous and bool(batch['pair_valid'].any()):
        with torch.no_grad():
            previous['native_query_index']=batch['previous_native_query_index']
            previous_noise=overlapping_noise(base_noise,batch['native_query_index'],previous['native_query_index'],generator)
            sampled=self.sample_fast(previous,world,steps=10,generator=generator,initial_action_noise=previous_noise)['actions']
            prior,pv=align_plan(sampled,previous['native_query_index'],batch['native_query_index'])
            pv=pv & batch['pair_valid'][:,None]
    batch.update(previous_actions=prior,previous_valid=pv)
    sigma=torch.rand(b,device=dev,generator=generator)
    center,reference,reference_valid,reference_weight=aligned_reference(batch,world,maximum=self.native_v7['prior_maximum'])
    na=base_noise+center
    nz=torch.randn(z.shape,device=dev,dtype=z.dtype,generator=generator)
    av=batch['fast_action_valid'];zv=batch['future_loss_valid']
    x=torch.where(av[...,None],fm_input(a,sigma,na),na)
    zz=torch.where(zv,fm_input(z,sigma,nz),nz)
    zz=torch.where(self.future_field_support[None],zz,0.)
    result=self.fast(x,zz,sigma,**self.conditions(batch,world))
    clean=x-sigma[:,None,None]*result['action_velocity']
    weight=a.new_ones((1,16,1));weight[:,:2]=c['execute_weight']
    main=masked_mean((result['action_velocity']-(na-a)).float().square()*weight,av[...,None])
    # Teacher clean endpoint is a soft auxiliary target; data remains authoritative.
    # Agreement gating avoids forcing demonstrably poor or incompatible teacher actions.
    agreement=torch.exp(-((target-a).square().mean(-1).detach())/c['teacher_scale'])
    teacher_loss=masked_mean(F.smooth_l1_loss(clean,target,reduction='none')*agreement[...,None],av[...,None])
    reference_agreement=torch.exp(-((reference-a).square().mean(-1).detach())/c['teacher_scale'])
    reference_fields=(av & reference_valid)[...,None].expand_as(a).clone()
    reference_fields[:,:,[6,13]]=False
    reference_loss=masked_mean(F.smooth_l1_loss(clean,reference,reduction='none')*reference_agreement[...,None]*reference_weight[...,None],reference_fields)
    arm=torch.tensor([0,1,2,3,4,5,7,8,9,10,11,12],device=dev)
    d=(clean[:,1:]-clean[:,:-1])-(a[:,1:]-a[:,:-1])
    differential=masked_mean(d.index_select(-1,arm).square(),(av[:,1:]&av[:,:-1])[...,None])
    overlap=pv & av
    confidence=torch.exp(-((prior-a).square().mean(-1).detach())/c['continuity_scale'])
    continuation=masked_mean(F.smooth_l1_loss(clean.index_select(-1,arm),prior.index_select(-1,arm),reduction='none')*confidence[...,None],overlap[...,None])
    ze=(result['future_velocity']-(nz-z)).float().square();vd=self.loss_config.visual_dim
    visual=masked_mean(ze[...,:vd],zv[...,:vd]);geometry=masked_mean(ze[...,vd:],zv[...,vd:])
    ct=torch.where(batch['contact_valid'],batch['contact'],0.)
    contact=masked_mean(F.binary_cross_entropy_with_logits(result['contact_logits'],ct,reduction='none'),batch['contact_valid'])
    total=(main+c['teacher_weight']*teacher_loss+c['boundary_weight']*differential+c['continuity_weight']*continuation+
        self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact+
        self.native_v7['reference_weight']*reference_loss)
    return dict(loss=total,slow_reference_constraint=reference_loss,fast_action=main,teacher_endpoint=teacher_loss,boundary=differential,continuation=continuation,
        future_visual=visual,future_geometry=geometry,contact=contact)
