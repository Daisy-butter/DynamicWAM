"""Deterministic residual A/Z controller with an exact zero-correction baseline.

This is not an action flow sampler. One shared compact transformer predicts a
bounded clean correction and compact future. Baseline means the aligned Full
plan under the SAME causal fixed-duration C2 feasibility projection, not an SR
guarantee relative to a different executor or to a fresh Full query.
"""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .continuous_v8 import ContinuousResidualFast,aligned_draft
from .system import masked_mean,DualRateSystem
from .c2_executor import project_chunk,displacement_interval,ARMS


class ResidualFast(ContinuousResidualFast):
    clean_residual=True
    def __init__(self,config):
        super().__init__(config)
        nn.init.zeros_(self.action_out.weight);nn.init.zeros_(self.action_out.bias)


def baseline(self,batch,world):
    draft,valid=aligned_draft(batch,world,max_age=self.continuous_v8['draft_max_age'])
    # V10 includes the actual Full gripper reference, unlike V8's zero center.
    match=(batch['action_intervals'][:,:,None]-world['reference_action_intervals'][:,None]).abs().amax(-1)<=1e-7
    if world['reference_actions'].shape[1]:
        idx=match.long().argmax(-1)
        ref=world['reference_actions'].gather(1,idx[...,None].expand(-1,-1,14))
        draft=draft.clone();draft[:,:,[6,13]]=ref[:,:,[6,13]]
    known=batch['fast_known_mask'];coverage=valid | known
    # Missing tail is explicitly masked. No extrapolated slow plan is called valid.
    # A missing *first executable* row is a scheduling failure, never random action.
    first=known.long().sum(-1)
    if not bool(valid.gather(1,first[:,None]).all()):
        raise ValueError('no aligned slow coverage for the first executable residual action')
    seed=torch.where(valid[...,None],draft,batch['fast_known_actions'])
    rows=[]
    for i in range(seed.shape[1]):
        value=seed[:,i] if i==0 else torch.where(coverage[:,i,None],seed[:,i],rows[-1])
        rows.append(value)
    seed=torch.stack(rows,1)
    return project(self,seed,batch)[0],valid


def project(self,action,batch,center=None):
    s=self.residual_v10
    return project_chunk(action,batch['fast_known_actions'],batch['fast_known_mask'],mean=self.action_mean_v10,
        scale=self.action_scale_v10,dt=s['action_dt'],bounds=s['bounds'],limits=s['limits'],
        center=center,radius=self.radius_raw_v10 if center is not None else None)


def predict(self,batch,world,*,expert_actions=None):
    base,valid=baseline(self,batch,world)
    # Z is anchored AFTER the protected action prefix, identically in data/runtime.
    first=batch['fast_known_mask'].long().sum(-1)
    origin=batch['action_times'].gather(1,first[:,None])
    offsets=origin.new_tensor((.1,.2,.4,.8)).repeat(self.fast.config.future_entities)
    expected=origin+offsets[None]
    if not torch.allclose(batch['future_times'],expected,atol=1e-7,rtol=0):
        raise ValueError('V10 future labels must use the first mutable physical interval')
    cond=self.conditions(batch,world)
    cond.setdefault('future_valid',(self.future_field_support.any(-1)|self.contact_field_support.any(-1))[None].expand(len(base),-1))
    cond['action_draft']=base;cond['draft_valid']=valid
    known=batch['fast_known_mask'];a=torch.where(known[...,None],batch['fast_known_actions'],base)
    if expert_actions is not None:
        # Future supervision is conditioned on the recorded trajectory, without
        # lying to the model that its entire future is already committed.
        a=torch.where(known[...,None],batch['fast_known_actions'],expert_actions)
    z=a.new_zeros(a.shape[0],self.fast.config.future_tokens,self.fast.config.future_dim)
    graph=getattr(self,'_fast_graph_sampler',None)
    if graph is not None and expert_actions is None:
        context=self.fast.prepare_context(**cond)
        fields=cond['future_valid'][...,None]&self.future_field_support[None]
        contacts=cond['future_valid'][...,None]&self.contact_field_support[None]
        core=graph.sample(a,z,a,known,context,fields,contacts,steps=1)
        raw=dict(action_velocity=core['actions'],future_velocity=core['future'],contact_logits=core['contact_logits'])
    else:
        raw=self.fast(a,z,a.new_zeros(a.shape[0]),action_known_mask=known,known_actions=a,**cond)
    correction=raw['action_velocity'].tanh()*self.radius_raw_v10/self.action_scale_v10
    raw['normalized_correction']=raw['action_velocity'].tanh()
    proposed=base+torch.where(valid[...,None],correction,0.)
    final,accepted=project(self,proposed,batch,center=base)
    final=torch.where(batch['fast_known_mask'][...,None],batch['fast_known_actions'],final)
    return final,base,valid,accepted,raw,cond


def anchor_future(batch,entities):
    """Only runtime can set requested times; training labels must already match."""
    first=batch['fast_known_mask'].long().sum(-1)
    if bool((first>=batch['action_times'].shape[1]).any()):
        raise ValueError('query has no mutable action interval')
    origin=batch['action_times'].gather(1,first[:,None])
    return origin+origin.new_tensor((.1,.2,.4,.8)).repeat(entities)[None]


def sample(self,batch,world,*,steps=1,generator=None,initial_action_noise=None):
    if steps!=1 or initial_action_noise is not None:raise ValueError('V10 is a deterministic one-pass residual controller')
    action,base,valid,accepted,raw,cond=predict(self,batch,world)
    fields=cond['future_valid'][...,None]&self.future_field_support[None]
    contacts=cond['future_valid'][...,None]&self.contact_field_support[None]
    return dict(actions=action,future=torch.where(fields,raw['future_velocity'],0.),
        future_valid=cond['future_valid'],future_field_valid=fields,contact_valid=contacts,
        contact_probability=torch.where(contacts,raw['contact_logits'].sigmoid(),0.),
        baseline_actions=base,correction_actions=action-base,draft_valid=valid,correction_accepted=accepted)


def current_teacher_batch(batch):
    """Fresh observation, identical physical action intervals and immutable queue."""
    result={k:v for k,v in batch.items() if k.startswith('future_')}
    for key,value in batch.items():
        if key.startswith('teacher_'):
            result['source_'+key[len('teacher_'):]]=value
    result.update(source_time=batch['state_time'],completed_time=batch['query_time'],query_time=batch['query_time'],
        source_known_actions=batch['teacher_known_actions'],source_known_mask=batch['teacher_known_mask'],
        source_action_times=batch['teacher_action_times'],source_action_intervals=batch['teacher_action_intervals'])
    match=(result['source_action_intervals'][:,:,None]-batch['action_intervals'][:,None]).abs().amax(-1)<1e-7
    match=match & batch['fast_known_mask'][:,None,:]
    index=match.long().argmax(-1)
    values=batch['fast_known_actions'].gather(1,index[...,None].expand(-1,-1,14))
    result['source_known_actions']=torch.where(match.any(-1)[...,None],values,result['source_known_actions'])
    return result


def future_queue_augmentation(self,batch,generator):
    """Causal counterfactual FUTURE queue, never alter past observations/actions.

    A newly protected second row may differ from the old slow proposal. Change
    it only if it has NOT started and was NOT already committed at the slow
    source. The old-world prediction is unchanged. Fresh Full supplies labels;
    expert future/action labels are invalid for this counterfactual branch.
    This is synthetic queue robustness, explicitly not on-policy data.
    """
    known=batch['fast_known_actions'];mask=batch['fast_known_mask'];b=len(known)
    target_interval=batch['action_intervals'][:,1]
    locked=((batch['source_action_intervals']-target_interval[:,None]).abs().amax(-1)<1e-7)&batch['source_known_mask']
    eligible=mask[:,1] & ~locked.any(-1) & (target_interval[:,0]>batch['query_time']+1e-8)
    prior=((batch['executed_action_intervals'][...,1]-batch['action_times'][:,0,None]).abs()<1e-7)&batch['executed_action_valid']
    eligible=eligible&prior.any(-1)
    chosen=eligible&(torch.rand(b,device=known.device,generator=generator)<self.residual_v10['queue_perturb_probability'])
    mean,scale=self.action_mean_v10,self.action_scale_v10
    previous=batch['executed_actions'].gather(1,prior.long().argmax(-1)[:,None,None].expand(-1,1,14))[:,0]*scale+mean
    first=known[:,0]*scale+mean;second=known[:,1]*scale+mean
    idx=torch.tensor(ARMS,device=known.device);dt=self.residual_v10['action_dt']
    bounds=known.new_tensor(self.residual_v10['bounds'])
    lo,hi,feasible=displacement_interval(first[:,idx],(first-previous)[:,idx]/dt,dt,bounds[idx,0],bounds[idx,1],self.residual_v10['limits'])
    noise=(torch.rand(b,len(ARMS),device=known.device,generator=generator)*2-1)*self.residual_v10['queue_perturb_radius']
    proposal=second[:,idx]+noise
    proposal=torch.maximum(torch.minimum(proposal,first[:,idx]+hi),first[:,idx]+lo)
    chosen=chosen&feasible.all(-1)
    changed=second.scatter(1,idx[None].expand(b,-1),proposal)
    values=known.clone();values[:,1]=torch.where(chosen[:,None],(changed-mean)/scale,known[:,1])
    result=dict(batch,fast_known_actions=values)
    return result,chosen


def reference_pair(self,batch,generator):
    """Coupled Full samples isolate observation changes from sampling noise.

    Overlap noise is matched by BOTH physical interval endpoints. This is
    training supervision only; the deployed fast route never waits for a fresh
    Full teacher and never receives privileged future observations.
    """
    b=len(batch['query_time']);like=batch['source_initial_state']
    vn=self._noise((b,*self.future_latent_shape),batch['source_condition_latent'],generator)
    old_noise=self._noise(batch['source_known_actions'].shape,like,generator)
    new_noise=self._noise(batch['teacher_known_actions'].shape,like,generator)
    match=(batch['teacher_action_intervals'][:,:,None]-batch['source_action_intervals'][:,None]).abs().amax(-1)<1e-7
    index=match.long().argmax(-1)
    new_noise=torch.where(match.any(-1)[...,None],old_noise.gather(1,index[...,None].expand_as(new_noise)),new_noise)
    def run(data,noise):
        return self.slow.sample_reference(self.source(data),vn,noise,
            data['source_known_actions'],data['source_known_mask'],data['source_action_times'],data['source_future_intervals'],
            data['source_time'],data['completed_time'],action_intervals=data['source_action_intervals'],
            source_metadata=self._spatial_metadata(data,'source'),future_metadata=self._spatial_metadata(data,'future'))
    with torch.no_grad():
        old=run(batch,old_noise);teacher=run(current_teacher_batch(batch),new_noise)
    match=(batch['action_intervals'][:,:,None]-teacher['action_intervals'][:,None]).abs().amax(-1)<1e-7
    if bool((match.sum(-1)>1).any()):raise ValueError('ambiguous teacher physical action interval')
    valid=match.any(-1);index=match.long().argmax(-1)
    target=teacher['reference_actions'].gather(1,index[...,None].expand(-1,-1,14)).detach()
    return old,target,valid


def loss(self,batch,generator=None,action_conditioned=None):
    DualRateSystem._validate_clocks(batch);DualRateSystem._validate_prefix_pair(batch)
    self.slow.eval()
    batch,augmented=future_queue_augmentation(self,batch,generator)
    DualRateSystem._validate_prefix_pair(batch)
    old,teacher,teacher_valid=reference_pair(self,batch,generator)
    world=self.world_readout(old)
    action,base,valid,accepted,raw,cond=predict(self,batch,world)
    teacher=torch.where(teacher_valid[...,None],teacher,base.detach())
    expert=torch.where(augmented[:,None,None],teacher,batch['fast_action'])
    feasible,_=project(self,expert,batch)
    desired=base+(feasible-base).clamp(-self.radius_raw_v10/self.action_scale_v10,self.radius_raw_v10/self.action_scale_v10)
    desired,_=project(self,desired,batch,center=base)
    teacher_feasible,_=project(self,teacher,batch)
    teacher_desired=base+(teacher_feasible-base).clamp(-self.radius_raw_v10/self.action_scale_v10,self.radius_raw_v10/self.action_scale_v10)
    teacher_desired,_=project(self,teacher_desired,batch,center=base)
    target_valid=torch.where(augmented[:,None],teacher_valid,batch['fast_action_valid'])
    mask=valid & target_valid & ~batch['fast_known_mask']
    radius=self.radius_raw_v10/self.action_scale_v10
    correction=(action-base)/radius;target=(desired-base)/radius
    weights=torch.ones_like(mask,dtype=action.dtype)
    first=batch['fast_known_mask'].long().sum(-1)
    weights=weights.scatter(1,first[:,None],self.residual_v10['execute_weight'])
    # Supervise the CLEAN head too: a hard feasibility clamp must not erase the
    # gradient needed to bring an initially saturated proposal back into range.
    head_error=F.smooth_l1_loss(raw['normalized_correction'],target.detach(),reduction='none')
    executed_error=F.smooth_l1_loss(correction,target.detach(),reduction='none')
    residual=masked_mean((head_error+.25*executed_error)*weights[...,None],mask[...,None])
    teacher_target=((teacher_desired-base)/radius).detach()
    teacher_error=F.smooth_l1_loss(raw['normalized_correction'],teacher_target,reduction='none')
    teacher_loss=masked_mean(teacher_error*weights[...,None],(mask&teacher_valid)[...,None])
    # Prefer no change where the current-state target agrees with the slow draft.
    reliable=((feasible-base).abs()*self.action_scale_v10).amax(-1)<self.residual_v10['preserve_tolerance']
    preserve=masked_mean(correction.square(),(mask&reliable)[...,None])
    edge=mask[:,1:] & mask[:,:-1]
    change=(correction[:,1:]-correction[:,:-1])-(target[:,1:]-target[:,:-1])
    continuity=masked_mean(change[:,:,list(ARMS)].square(),edge[...,None])
    # True expert futures only supervise the explicit expert-conditioned branch.
    # Predicted student actions do not borrow counterfactual expert labels.
    _,_,_,_,aux,_=predict(self,batch,world,expert_actions=batch['fast_action'])
    zv=batch['future_loss_valid']&~augmented[:,None,None];cv=batch['contact_valid']&~augmented[:,None,None];d=self.loss_config.visual_dim
    ze=(aux['future_velocity']-batch['future']).float().square()
    visual=masked_mean(ze[...,:d],zv[...,:d]);geometry=masked_mean(ze[...,d:],zv[...,d:])
    contact=masked_mean(F.binary_cross_entropy_with_logits(aux['contact_logits'],torch.where(cv,batch['contact'],0.),reduction='none'),cv)
    total=residual+self.residual_v10['teacher_weight']*teacher_loss+.1*preserve+.1*continuity+self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact
    return dict(loss=total,fast_action=residual,teacher_endpoint=teacher_loss,baseline_preservation=preserve,boundary=continuity,
        future_visual=visual,future_geometry=geometry,contact=contact,
        correction_rejected=(~accepted).float().mean(),queue_augmented=augmented.float().mean(),
        residual_target_saturation=((feasible-base).abs()>radius).float().mean())


def configure(model,settings,mean,scale):
    required={'action_dt','bounds','limits','arm_radius','gripper_radius','execute_weight','preserve_tolerance','future_anchor','deterministic','teacher_weight','queue_perturb_probability','queue_perturb_radius'}
    if set(settings)!=required or settings['future_anchor']!='first_mutable_start' or settings['deterministic'] is not True:
        raise ValueError('explicit V10 residual/executor/clock contract required')
    if not 0<settings['arm_radius']<=.15 or not 0<settings['gripper_radius']<=.1:raise ValueError('bounded raw correction radius required')
    if not 0<=settings['queue_perturb_probability']<=1 or not 0<settings['queue_perturb_radius']<=settings['arm_radius']:
        raise ValueError('future queue augmentation must fit the trained residual range')
    model.residual_v10=dict(settings)
    if model.continuous_v8['draft_dropout']!=0 or model.continuous_v8['action_mode']!='residual':
        raise ValueError('V10 has no random/full-action fallback or dropped action baseline')
    model.register_buffer('action_mean_v10',torch.as_tensor(mean,dtype=torch.float32))
    model.register_buffer('action_scale_v10',torch.as_tensor(scale,dtype=torch.float32))
    radius=torch.full((14,),settings['arm_radius']);radius[[6,13]]=settings['gripper_radius']
    model.register_buffer('radius_raw_v10',radius)
    model.training_loss=MethodType(loss,model);model.sample_fast=MethodType(sample,model)
