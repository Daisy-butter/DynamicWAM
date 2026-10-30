"""Single pretrained action backbone with joint Z and event-aligned supervision.

Only one decoder emits A. Full is a frozen source-only world model and teacher.
Prediction coordinates are command events; historical clocks remain seconds.
"""
from dataclasses import replace
from types import MethodType, SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .native_v5 import NativeMotor
from .fast import _TimeFeatures
from .slow import StructuredWorldReadout, latent_patch_grid, latent_patches
from .system import DualRateSystem, masked_mean, fm_input

CONTRACT='native_event_single_v6'
Z_EVENTS=(1,2,4,8)

def validate_events(batch):
    DualRateSystem._validate_clocks(batch)
    s,q=batch['native_source_index'],batch['native_query_index']
    if s.dtype!=torch.long or q.dtype!=torch.long or s.shape!=q.shape or (q<s).any():
        raise ValueError('invalid source/query command identity')
    if batch['fast_known_mask'].any() or batch['source_known_mask'].any():
        raise ValueError('native synchronous inference has zero committed prefix')
    if not torch.allclose(batch['rgb_times'],batch['query_time'][:,None].expand_as(batch['rgb_times']),atol=1e-7,rtol=0):
        raise ValueError('query must be anchored to the actual captured observation')
    if not torch.allclose(batch['state_time'],batch['query_time'],atol=1e-7,rtol=0):
        raise ValueError('state and RGB do not describe the same native event')

def align_plan(actions, origin, current, horizon=16):
    """Exact ordinal join; elapsed seconds can never select an action row."""
    if origin.dtype!=torch.long or current.dtype!=torch.long or (current<origin).any():
        raise ValueError('noncausal action-plan identity')
    i=current[:,None]-origin[:,None]+torch.arange(horizon,device=actions.device)[None]
    valid=(i>=0)&(i<actions.shape[1])
    rows=actions.gather(1,i.clamp(0,actions.shape[1]-1)[...,None].expand(-1,-1,actions.shape[-1]))
    return torch.where(valid[...,None],rows,0.),valid

class EventWorldReadout(StructuredWorldReadout):
    def forward(self,r):
        if r.get('provenance')!='source_only_independent_noise_prediction':
            raise ValueError('world requires source-only prediction')
        s=r['source_time'];idx=r['native_source_index'];dtype=self.source_projection.weight.dtype
        observed,predicted=r['condition_latent'],r['predicted_video_latent']
        og,fg=latent_patch_grid(observed),latent_patch_grid(predicted)
        o=self.source_projection(self.source_norm(r['source_tokens'].to(dtype)))
        o,oi,op=self._metadata_features(o,r['source_metadata'],og,s,0)
        f=self.future_projection(torch.cat((self.future_norm(r['future_hidden'].to(dtype)),latent_patches(predicted).to(dtype)),-1))
        b,n,_=f.shape
        # Two native future groups predict command events 1..8 and 9..16.
        # This is separate from their causal receptive-field support.
        intervals=idx[:,None,None]+idx.new_tensor([[1,8],[9,16]])[None]
        fm=dict(r['future_metadata'],temporal_support=intervals.double())
        f,fi,fp=self._metadata_features(f,fm,fg,idx.double(),1)
        f=f+self.stage_projection(r['prediction_sigmas'].to(dtype))[None,None]
        return dict(world_tokens=torch.cat((o,f),1),world_valid=torch.cat((r['source_metadata']['valid'],fm['valid']),1),
            world_types=torch.cat((torch.zeros_like(r['source_metadata']['valid'],dtype=torch.long),torch.ones_like(fm['valid'],dtype=torch.long)),1),
            world_intervals=torch.cat((oi,fi),1),world_domains=('observed_seconds','predicted_command_events'),
            source_time=s,completed_time=r['completed_time'],native_source_index=idx,
            native_video_cache=r['native_video_cache'],reference_actions=r['reference_actions'])

class SingleNativeFast(nn.Module):
    """One 12-layer pretrained motor; O/M/Z adapters have no action decoder."""
    def __init__(self,config,native):
        super().__init__();self.config=config;self.motor=NativeMotor(native)
        d=native.config.ae_dim;self.dim=d;self.heads=12
        self.future_in=nn.Linear(config.future_dim,d)
        self.future_position=nn.Embedding(config.future_tokens,d)
        self.observation_in=nn.Linear(config.observation_dim,d)
        self.world_in=nn.Linear(config.world_dim,d)
        self.plan_in=nn.Linear(config.action_dim,d)
        self.plan_position=nn.Embedding(config.chunk_size,d)
        self.time_in=_TimeFeatures(2,d)
        self.event_in=nn.Linear(2,d)
        self.context_norm=nn.LayerNorm(d)
        self.cross=nn.ModuleDict({str(i):nn.MultiheadAttention(d,self.heads,batch_first=True,dropout=0.) for i in (0,3,6,9)})
        self.action_gates=nn.Parameter(torch.zeros(4))
        self.future_norm=nn.LayerNorm(d)
        self.future_out=nn.Linear(d,config.future_dim)
        self.contact_out=nn.Linear(d,2)

    def prepare_context(self,*,native_batch,native_video_cache,native_available,observation_tokens,
            observation_valid,world_tokens,world_valid,world_intervals,world_types,query_time,source_time,
            native_source_index,native_query_index,future_valid=None,previous_actions=None,previous_valid=None,**unused):
        b=len(query_time);dev=query_time.device
        if native_query_index.dtype!=torch.long or native_source_index.dtype!=torch.long or (native_query_index<native_source_index).any():
            raise ValueError('world command origin is ahead of the current query')
        if (query_time<source_time).any():raise ValueError('noncausal physical observation')
        if future_valid is None:future_valid=torch.ones((b,self.config.future_tokens),device=dev,dtype=torch.bool)
        hist=world_types==0;pred=world_types==1
        valid=world_valid & native_available[:,None] & ((hist & (world_intervals[...,1]<=query_time[:,None]+1e-8))|
            (pred & (world_intervals[...,1]>native_query_index[:,None])))
        # Each coordinate has its own origin and unit; never subtract seconds from events.
        relative=torch.where(hist[...,None],world_intervals-query_time[:,None,None],
            world_intervals-native_query_index[:,None,None])
        w=self.world_in(world_tokens)+torch.where(hist[...,None],self.time_in(relative),self.event_in(relative.float()/16.))
        age=torch.stack((query_time-source_time,(native_query_index-native_source_index).double()),-1)
        o=self.observation_in(observation_tokens)+self.time_in(torch.stack((age[:,0],torch.zeros_like(age[:,0])),-1))[:,None]
        contexts=[o,w];masks=[observation_valid,valid]
        if previous_actions is None:
            previous_actions=observation_tokens.new_zeros((b,16,14));previous_valid=torch.zeros((b,16),dtype=torch.bool,device=dev)
        p=self.plan_in(previous_actions)+self.plan_position.weight[None]
        contexts.append(p);masks.append(previous_valid)
        context=torch.cat(contexts,1);mask=torch.cat(masks,1)
        context=torch.where(mask[...,None],context,0.)
        # A sentinel handles missing M and any fully absent context without NaNs.
        context=F.pad(context,(0,0,1,0));mask=F.pad(mask,(1,0),value=True)
        return SimpleNamespace(native_batch=native_batch,video=native_video_cache,available=native_available,
            context=context,mask=mask,future_valid=future_valid,origin=native_query_index)

    def forward(self,a,z,sigma,*,context_cache=None,action_known_mask=None,known_actions=None,**conditions):
        return self.forward_cached(a,z,sigma,context_cache or self.prepare_context(**conditions),
            action_known_mask=action_known_mask,known_actions=known_actions)

    def forward_cached(self,a,z,sigma,cache,*,action_known_mask=None,known_actions=None,**unused):
        known=torch.zeros(a.shape[:2],dtype=torch.bool,device=a.device) if action_known_mask is None else action_known_mask
        if known_actions is not None:a=torch.where(known[...,None],known_actions,a)
        x,core=self.motor._build_action_tokens(cache.native_batch,a,cache.native_batch['initial_state'])
        t=sigma[:,None].expand(-1,x.shape[1]).clone()*1000
        t[:,1:1+a.shape[1]]=torch.where(known,0.,t[:,1:1+a.shape[1]])
        te,mod=self.motor._build_action_time_embeddings(t,x.shape[1])
        zt=sigma[:,None].expand(-1,z.shape[1])*1000
        ze,zm=self.motor._build_action_time_embeddings(zt,z.shape[1])
        z=self.future_in(z)+self.future_position.weight[None]
        z=torch.where(cache.future_valid[...,None],z,0.)
        for i,block in enumerate(self.motor.action_expert.blocks):
            am=self.motor._block_modulation(block,mod);zz=self.motor._block_modulation(block,zm)
            if bool(cache.available.all()):
                x=self.motor._action_attention_with_video_cache(x,am,i,cache.video)
            else:
                parts=[]
                for j,keep in enumerate(cache.available.tolist()):
                    v=dict(cache.video,video_k=[v[j:j+1] if keep else v[j:j+1,:0] for v in cache.video['video_k']],
                        video_v=[v[j:j+1] if keep else v[j:j+1,:0] for v in cache.video['video_v']],
                        seq_lens=cache.video['seq_lens'][j:j+1] if keep else cache.video['seq_lens'][j:j+1]*0)
                    parts.append(self.motor._action_attention_with_video_cache(x[j:j+1],tuple(v[j:j+1] for v in am),i,v))
                x=torch.cat(parts)
            if str(i) in self.cross:
                # A and Z interact inside the same native blocks; only A's native decoder emits actions.
                cat=torch.cat((x,z),1);keys=torch.cat((cache.context,cat),1)
                readable=torch.cat((cache.mask,torch.ones(x.shape[:2],dtype=torch.bool,device=x.device),cache.future_valid),1)
                update=self.cross[str(i)](self.context_norm(cat),self.context_norm(keys),keys,key_padding_mask=~readable,need_weights=False)[0]
                x=x+self.action_gates[i//3].tanh()*update[:,:x.shape[1]]
                z=z+update[:,x.shape[1]:]
            for which,h,m in (('x',x,am),('z',z,zz)):
                ff=block.ffn((block.norm2(h).float()*(1+m[4].squeeze(2))+m[3].squeeze(2)).to(block.ffn[0].weight.dtype))
                h=h+ff*m[5].squeeze(2)
                if which=='x':x=h
                else:z=torch.where(cache.future_valid[...,None],h,0.)
        action=self.motor._decode_action_tokens(x,te,core)
        return dict(action_velocity=torch.where(known[...,None],0.,action),
            future_velocity=self.future_out(self.future_norm(z)),contact_logits=self.contact_out(self.future_norm(z)))

def conditions(self,batch,world):
    r=DualRateSystem.conditions(self,batch,world)
    r.update(native_batch=dict(initial_state=batch['state'],absolute_motion_features=batch['motion'],
        absolute_motion_interval_valid_mask=batch['motion_valid'],absolute_motion_acceleration_valid_mask=batch['motion_acceleration_valid']),
        native_video_cache=world['native_video_cache'],native_available=world.get('native_available',torch.ones_like(batch['native_query_index'],dtype=torch.bool)),
        native_source_index=world['native_source_index'],native_query_index=batch['native_query_index'])
    r.update({k:batch[k] for k in ('previous_actions','previous_valid') if k in batch})
    r.pop('target_times')  # Unknown TOPP endpoint seconds are never network inputs.
    return r

def current_source(batch):
    """Teacher sees the exact current observation, never target frames/actions."""
    r=dict(batch)
    for k,v in batch.items():
        if k.startswith('teacher_'):r['source_'+k[len('teacher_'):]]=v
    r.update(source_time=batch['query_time'],completed_time=batch['query_time'],native_source_index=batch['native_query_index'])
    return r

def reference(self,batch,generator=None):
    with torch.no_grad():r=DualRateSystem.predict_reference(self,batch,generator=generator)
    r['native_source_index']=batch['native_source_index']
    return r

def training_loss(self,batch,generator=None,action_conditioned=None):
    validate_events(batch);self.slow.eval()
    a=batch['fast_action'];z=batch['future'];b=len(a);dev=a.device;c=self.native_v6
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
            sampled=self.sample_fast(previous,world,steps=10,generator=generator)['actions']
            prior,pv=align_plan(sampled,previous['native_query_index'],batch['native_query_index'])
            pv=pv & batch['pair_valid'][:,None]
    batch.update(previous_actions=prior,previous_valid=pv)
    sigma=torch.rand(b,device=dev,generator=generator)
    na=torch.randn(a.shape,device=dev,dtype=a.dtype,generator=generator)
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
        self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact)
    return dict(loss=total,fast_action=main,teacher_endpoint=teacher_loss,boundary=differential,continuation=continuation,
        future_visual=visual,future_geometry=geometry,contact=contact)

def configure(model,settings):
    keys={'teacher_weight','teacher_scale','boundary_weight','continuity_weight','continuity_scale','execute_weight'}
    if set(settings)!=keys or any(not isinstance(v,(int,float)) or not 0<v<10 for v in settings.values()):
        raise ValueError('explicit bounded V6 supervision settings required')
    model.native_v6=dict(settings)
    # Reuse the verified native shifted sampler, with no V5 action-noise prior.
    model.native_v5=dict(prior_maximum=0.,boundary_weight=0.)
    model.slow.requires_grad_(False).eval()
    model._action_deviation=MethodType(lambda self,b,w:(torch.zeros_like(w['world_tokens'][...,:14]),torch.zeros_like(w['world_valid'])),model)
    model.conditions=MethodType(conditions,model)
    model.predict_reference=MethodType(reference,model)
    model.training_loss=MethodType(training_loss,model)

@torch.no_grad()
def verify_single_motor(system,batch):
    validate_events(batch)
    native=system.slow.native;fast=system.fast
    for name,p in fast.motor.action_expert.named_parameters():
        original=dict(native.action_expert.named_parameters())[name]
        if p.data_ptr()==original.data_ptr() or not torch.equal(p,original):raise AssertionError('motor weight identity')
    ref=system.predict_reference(batch);world=system.world_readout(ref)
    world['native_available']=torch.ones_like(batch['native_query_index'],dtype=torch.bool)
    cond=system.conditions(batch,world)
    a=torch.randn_like(batch['fast_action']);z=torch.randn_like(batch['future']);sigma=a.new_full((len(a),),.5)
    expected=native.forward_action_with_video_cache(dict(cond['native_batch'],noisy_actions=a,action_t=sigma*1000,video_cache=ref['native_video_cache']))['action_pred']
    actual=fast(a,z,sigma,**cond)['action_velocity']
    error=float((actual-expected).abs().max())
    if error!=0:raise AssertionError(('single motor initialization mismatch',error))
    fast.action_gates.fill_(.05)
    baseline=fast(a,z,sigma,**cond)['action_velocity']
    changed=dict(cond,observation_tokens=cond['observation_tokens']+.2)
    obs_delta=float((fast(a,z,sigma,**changed)['action_velocity']-baseline).abs().max())
    z_delta=float((fast(a,z+.3,sigma,**cond)['action_velocity']-baseline).abs().max())
    fast.action_gates.zero_()
    if min(obs_delta,z_delta)<=0:raise AssertionError('current/Z feedback disconnected')
    import time
    if a.is_cuda:torch.cuda.synchronize(a.device)
    start=time.monotonic();sample=system.sample_fast(batch,world,steps=10)
    if a.is_cuda:torch.cuda.synchronize(a.device)
    seconds=time.monotonic()-start
    if not all(torch.isfinite(sample[k]).all() for k in ('actions','future','contact_probability')):raise AssertionError('nonfinite joint sample')
    return dict(passed=True,exact_motor_max_error=error,independent_weights=True,single_action_decoder=True,
        observation_sensitivity=obs_delta,z_to_action_sensitivity=z_delta,joint_sampler_seconds=seconds,
        native_parameters=sum(p.numel() for p in fast.motor.parameters()),fast_total_parameters=sum(p.numel() for p in fast.parameters()),
        teacher_alignment='same physical observation and command-origin',performance_accepted=False)
