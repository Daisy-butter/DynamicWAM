"""Native pretrained motor plus reactive joint A/Z residual.
Slow actions shift a bounded diffusion prior; they never become hard commands.
"""
import copy
from types import SimpleNamespace, MethodType
import torch
from torch import nn
from .system import DualRateSystem
from dynamicwam.models.small_wam import SmallWAMActionModel

class NativeMotor(nn.Module):
    _expand_time_for_tokens=staticmethod(SmallWAMActionModel._expand_time_for_tokens)
    _block_modulation=staticmethod(SmallWAMActionModel._block_modulation)
    _action_qkv_for_wan=staticmethod(SmallWAMActionModel._action_qkv_for_wan)
    _build_action_time_embeddings=SmallWAMActionModel._build_action_time_embeddings
    _build_action_tokens=SmallWAMActionModel._build_action_tokens
    _decode_action_tokens=SmallWAMActionModel._decode_action_tokens
    _action_attention_with_video_cache=SmallWAMActionModel._action_attention_with_video_cache
    forward=SmallWAMActionModel.forward_action_with_video_cache
    def __init__(self,native):
        super().__init__()
        self.config=copy.deepcopy(native.config)
        self.action_expert=copy.deepcopy(native.action_expert)
        self.absolute_motion_tokens=copy.deepcopy(native.absolute_motion_tokens)
        video=native.compact_wan.video_model
        self.compact_wan=SimpleNamespace(video_model=SimpleNamespace(precision=video.precision,
            wan_model=SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace(window_size=b.self_attn.window_size)) for b in video.wan_model.blocks])))

class NativeFast(nn.Module):
    def __init__(self,adapter,native):
        super().__init__();self.config=adapter.config
        self.adapter=adapter;self.adapter.native_ordinal_actions=True
        self.motor=NativeMotor(native)
        self.reactive_gate=nn.Parameter(torch.tensor(0.))
    def prepare_context(self,*,native_batch,native_video_cache,native_available,**conditions):
        return SimpleNamespace(adapter=self.adapter.prepare_context(**conditions),native_batch=native_batch,
            native_video_cache=native_video_cache,native_available=native_available,
            future_valid=conditions.get('future_valid'),inputs=conditions)
    def forward(self,a,z,sigma,*,context_cache=None,action_known_mask=None,known_actions=None,**conditions):
        cache=context_cache or self.prepare_context(**conditions)
        return self.forward_cached(a,z,sigma,cache,action_known_mask=action_known_mask,known_actions=known_actions)
    def forward_cached(self,a,z,sigma,cache,*,action_known_mask=None,known_actions=None,**kwargs):
        b,h,_=a.shape
        known=torch.zeros((b,h),dtype=torch.bool,device=a.device) if action_known_mask is None else action_known_mask
        if known_actions is not None:a=torch.where(known[...,None],known_actions,a)
        result=self.adapter.forward_cached(a,z,sigma,cache.adapter,action_known_mask=known,known_actions=known_actions)
        n=1+h+4+self.motor.action_expert.config.num_registers
        t=sigma[:,None].expand(b,n).clone()*1000
        t[:,1:1+h]=torch.where(known,0.,t[:,1:1+h])
        video=cache.native_video_cache
        if video is None:raise ValueError('native motor cache must be explicit')
        available=cache.native_available
        if not bool(available.all()):
            outputs=[]
            for i in range(b):
                keep=bool(available[i])
                v={k:([x[i:i+1] if keep else x[i:i+1,:0] for x in value] if k in ('video_k','video_v') else value) for k,value in video.items()}
                v['seq_lens']=video['seq_lens'][i:i+1] if keep else video['seq_lens'][i:i+1]*0
                nb={k:x[i:i+1] for k,x in cache.native_batch.items()}
                outputs.append(self.motor(dict(nb,noisy_actions=a[i:i+1],action_t=t[i:i+1],video_cache=v))['action_pred'])
            velocity=torch.cat(outputs)
        else:velocity=self.motor(dict(cache.native_batch,noisy_actions=a,action_t=t,video_cache=video))['action_pred']
        result['action_velocity']=torch.where(known[...,None],0.,velocity+self.reactive_gate.tanh()*result['action_velocity'])
        return result

def aligned_center(batch,world,maximum=.15):
    """Ordinal alignment only. Full residual freedom, no GT or output averaging."""
    ref=world['reference_actions'];b,h,_=batch['fast_known_actions'].shape
    center=torch.zeros_like(batch['fast_known_actions'])
    if not ref.shape[1]:return center
    source=world.get('native_source_index',batch.get('native_source_index'));query=batch.get('native_query_index')
    if source is None or query is None:raise ValueError('ordinal reference identity required')
    indices=query[:,None]-source[:,None]+torch.arange(h,device=ref.device)[None]
    valid=(indices>=0)&(indices<ref.shape[1])
    value=ref.gather(1,indices.clamp(0,ref.shape[1]-1)[...,None].expand(-1,-1,ref.shape[-1]))
    age=(batch['query_time']-world['source_time']).clamp_min(0)
    weight=maximum*torch.exp(-age/.72).to(value.dtype)
    known=batch['fast_known_mask'] & valid
    deviation=((value-batch['fast_known_actions']).abs()*known[...,None]).sum((1,2))/(known.sum(1)*ref.shape[-1]).clamp_min(1)
    weight=weight*torch.exp(-deviation)
    available=world.get('native_available',torch.ones(b,dtype=torch.bool,device=ref.device))
    center=torch.where((valid & available[:,None] & ~batch['fast_known_mask'])[...,None],value.clamp(-3.,3.)*weight[:,None,None],center)
    center=center.clone();center[:,:,[6,13]]=0
    return center

def v5_conditions(self,batch,world):
    result=DualRateSystem.conditions(self,batch,world)
    result['world_valid']=result['world_valid'] & (result['world_types']!=2)
    # Ordinal waypoints have no known TOPP end timestamp. Token position carries
    # command order; only Z and world features have physical prediction times.
    result['target_times']=result['target_times'].clone()
    result['target_times'][:,:self.fast.config.chunk_size]=batch['query_time'][:,None]
    result['native_batch']=dict(initial_state=batch['state'],absolute_motion_features=batch['motion'],
        absolute_motion_interval_valid_mask=batch['motion_valid'],absolute_motion_acceleration_valid_mask=batch['motion_acceleration_valid'])
    result['native_video_cache']=world['native_video_cache']
    result['native_available']=world.get('native_available',torch.ones(batch['state'].shape[0],dtype=torch.bool,device=batch['state'].device))
    return result

def configure_v5(model,settings):
    if set(settings)!={'prior_maximum','boundary_weight'} or not 0<=settings['prior_maximum']<=.2 or not 0<=settings['boundary_weight']<=.2:
        raise ValueError('invalid v5 bounded prior/boundary contract')
    model.native_v5=dict(settings)
    model._action_deviation=MethodType(lambda self,batch,world:(
        torch.zeros_like(world['world_tokens'][...,:self.fast.config.action_dim]),
        torch.zeros_like(world['world_valid'])),model)
    model.conditions=MethodType(v5_conditions,model)
    model.action_center=MethodType(lambda self,batch,world:aligned_center(batch,world,self.native_v5['prior_maximum']),model)


@torch.no_grad()
def verify_native_motor(system,batch):
    """Full real-weight/cache parity before any optimizer step."""
    import time
    source=system.slow.native;motor=system.fast.motor
    source_params=dict(source.action_expert.named_parameters())
    for name,value in motor.action_expert.named_parameters():
        original=source_params[name]
        if value.data_ptr()==original.data_ptr() or not torch.equal(value,original):
            raise AssertionError("native motor initialization/independence failed: "+name)
    reference=system.predict_reference(batch)
    world=system.world_readout(reference)
    world['native_source_index']=batch['native_source_index']
    world['native_available']=torch.ones_like(batch['native_source_index'],dtype=torch.bool)
    nb=system.conditions(batch,world)['native_batch']
    actions=torch.randn_like(batch['fast_action'])
    modelbatch=dict(nb,noisy_actions=actions,action_t=actions.new_full((len(actions),),500.),
                    video_cache=reference['native_video_cache'])
    expected=source.forward_action_with_video_cache(modelbatch)['action_pred']
    actual=motor(modelbatch)['action_pred']
    error=float((expected-actual).abs().max())
    if error!=0:raise AssertionError("native full motor parity failed")
    if actions.is_cuda:torch.cuda.synchronize(actions.device)
    t=time.monotonic();sample=system.sample_fast(batch,world,steps=10)
    if actions.is_cuda:torch.cuda.synchronize(actions.device)
    latency=time.monotonic()-t
    if not all(torch.isfinite(sample[k]).all() for k in ('actions','future','contact_probability')):
        raise AssertionError("nonfinite joint sampler")
    # Same batch with missing M must remain executable; no fake future provided.
    world['native_available'].zero_()
    world['world_valid'].zero_()
    missing=system.sample_fast(batch,world,steps=2)
    if not torch.isfinite(missing['actions']).all():raise AssertionError("missing M path failed")
    return dict(passed=True,exact_motor_max_error=error,independent_weights=True,
                joint_sampler_seconds=latency,joint_sampler_steps=10,missing_M_finite=True,
                native_parameters=sum(p.numel() for p in motor.parameters()),
                reactive_parameters=sum(p.numel() for p in system.fast.adapter.parameters()),
                performance_accepted=False)

