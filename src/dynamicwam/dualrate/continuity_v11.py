"""Continuous feedback contracts; no changes to legacy V10 checkpoints.

The baseline is immutable within a reference lease. Corrections are supervised
by a fresh Full teacher, with explicit measured-state and previous-plan inputs.
Neither commanded state nor an expert's counterfactual future is measured q.
"""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .system import masked_mean

ARMS = (0,1,2,3,4,5,7,8,9,10,11,12)
CONTRACT = 'physical_feedback_continuity_v11'


def align(values, source, target, valid=None):
    if source.dtype != torch.float64 or target.dtype != torch.float64:
        raise ValueError('alignment requires actual FP64 interval endpoints')
    match = (target[:,:,None]-source[:,None]).abs().amax(-1) <= 1e-7
    if bool((match.sum(-1)>1).any()):
        raise ValueError('ambiguous prior physical interval')
    index = match.long().argmax(-1)
    keep = match.any(-1)
    if valid is not None: keep = keep & valid.gather(1,index)
    out = values.gather(1,index[...,None].expand(-1,-1,values.shape[-1]))
    return torch.where(keep[...,None],out,0.),keep


def defaults(batch):
    """Missing real sensors are MASKED, never filled with drive targets."""
    b=dict(batch); a=b['fast_known_actions'];n=len(a)
    b.setdefault('measured_state',a.new_zeros(n,37))
    b.setdefault('measured_state_valid',torch.zeros(n,dtype=torch.bool,device=a.device))
    b.setdefault('previous_actions',torch.zeros_like(a))
    b.setdefault('previous_valid',torch.zeros(a.shape[:2],dtype=torch.bool,device=a.device))
    b.setdefault('previous_reference_same',torch.zeros(n,dtype=torch.bool,device=a.device))
    return b


def augment_context(system,batch,cond):
    b=defaults(batch);state=b['measured_state'];valid=b['measured_state_valid']
    if state.shape!=(len(state),37) or valid.shape!=(len(state),):
        raise ValueError('measured q/error/qdot/age feature shape differs')
    if not torch.isfinite(state[valid]).all():raise ValueError('nonfinite measured state')
    sensor=system.fast.sensor_projection(torch.where(valid[:,None],state,0.))
    previous=system.fast.previous_projection(torch.where(b['previous_valid'][...,None],b['previous_actions'],0.))
    cond=dict(cond)
    cond['observation_tokens']=torch.cat((cond['observation_tokens'],sensor[:,None],previous),1)
    cond['observation_valid']=torch.cat((cond['observation_valid'],valid[:,None],b['previous_valid']),1)
    return cond


def shape_correction(raw,valid):
    """Local, bounded curve parameters; zero head -> exact zero correction.

    This convex temporal basis is inside both training and inference, not an
    evaluation-only EMA. Invalid tail tokens cannot affect valid corrections.
    Gripper semantics stay with Full, as in the historical bounded controller.
    """
    x=torch.where(valid[...,None],raw,0.)
    left=torch.cat((x[:,:1],x[:,:-1]),1)
    right=torch.cat((x[:,1:],x[:,-1:]),1)
    lv=torch.cat((valid[:,:1],valid[:,:-1]),1)
    rv=torch.cat((valid[:,1:],valid[:,-1:]),1)
    x=(2*x+left+right)/(2+lv.to(x.dtype)+rv.to(x.dtype))[...,None]
    mask=x.new_ones(x.shape[-1]);mask[[6,13]]=0
    return x*mask


def residual_bounds(system,batch,base):
    b=defaults(batch);radius=system.radius_raw_v10/system.action_scale_v10
    same=b['previous_valid'] & b['previous_reference_same'][:,None]
    anchor=torch.where(same[...,None],(b['previous_actions']-base).clamp(-radius,radius),0.)
    step=radius*system.continuity_v11['revision_radius_fraction']
    lo=torch.where(same[...,None],torch.maximum(-radius,anchor-step),-radius)
    hi=torch.where(same[...,None],torch.minimum(radius,anchor+step),radius)
    return anchor,lo,hi


def bounded_residual(system,batch,base,clean):
    anchor,lo,hi=residual_bounds(system,batch,base)
    # Zero innovation continues the previously accepted SAME-reference
    # correction. A new reference starts a new bounded correction cycle.
    result=anchor+torch.where(clean>=0,hi-anchor,anchor-lo)*clean
    arms=result.new_ones(14);arms[[6,13]]=0
    return result*arms


def teacher_reference(system,batch,generator,shared=None):
    from .residual_v10 import current_teacher_batch
    from .continuity_noise import prediction_noise
    if shared is None:
        vn,old_noise=prediction_noise(system,batch,batch['source_action_intervals'])
    else:vn,old_noise,old=shared
    _,new_noise=prediction_noise(system,batch,batch['teacher_action_intervals'])
    def run(data,noise):
        return system.slow.sample_reference(system.source(data),vn,noise,data['source_known_actions'],data['source_known_mask'],
            data['source_action_times'],data['source_future_intervals'],data['source_time'],data['completed_time'],
            action_intervals=data['source_action_intervals'],source_metadata=system._spatial_metadata(data,'source'),
            future_metadata=system._spatial_metadata(data,'future'))
    with torch.no_grad():
        if shared is None:old=run(batch,old_noise)
        teacher=run(current_teacher_batch(batch),new_noise)
        if shared is None and 'recorded_replay' in batch:
            recorded=batch['recorded_replay']
            if bool(recorded.any()) and not torch.allclose(old['reference_actions'][recorded],
                    batch['recorded_reference_actions'][recorded],atol=2e-5,rtol=2e-5):
                raise ValueError('recorded source/RNG no longer regenerates the actual Full baseline')
    target,valid=align(teacher['reference_actions'],teacher['action_intervals'],batch['action_intervals'])
    return old,target,valid,(vn,old_noise,old)


def loss(self,batch,generator=None,action_conditioned=None):
    from .system import DualRateSystem
    from .residual_v10 import predict,project
    b=defaults(batch);DualRateSystem._validate_clocks(b);DualRateSystem._validate_prefix_pair(b)
    self.slow.eval()
    old,target,tvalid,shared=teacher_reference(self,b,generator)
    world=self.world_readout(old)
    prior={k[9:]:v for k,v in b.items() if k.startswith('previous_') and k not in ('previous_actions','previous_valid')}
    paired=b.get('pair_valid',torch.zeros(len(target),dtype=torch.bool,device=target.device))
    prior_target=None
    if prior and bool(paired.any()):
        prior=defaults(prior)
        if not torch.equal(prior['source_action_intervals'],b['source_action_intervals']):
            raise ValueError('paired queries must share the same physical slow source')
        with torch.no_grad():
            p=predict(self,prior,world)[0]
            p,pv=align(p,prior['action_intervals'],b['action_intervals'])
            _,pt,ptv,_=teacher_reference(self,prior,generator,shared)
            prior_target,ptv=align(pt,prior['action_intervals'],b['action_intervals'],ptv)
        b['previous_actions']=p.detach();b['previous_valid']=pv&paired[:,None]
        b['previous_reference_same']=paired
    action,base,valid,accepted,raw,cond=predict(self,b,world)
    radius=self.radius_raw_v10/self.action_scale_v10
    teacher_delta=(target-base).detach()
    desired=base+teacher_delta.clamp(-radius,radius)
    _,lo,hi=residual_bounds(self,b,base)
    desired=base+torch.maximum(torch.minimum(desired-base,hi),lo)
    desired,_=project(self,desired,b,center=base)
    mask=valid&tvalid&~b['fast_known_mask']
    arm=torch.tensor(ARMS,device=action.device)
    target_r=(desired-base).detach()/radius.clamp_min(1e-8)
    actual_r=(action-base)/radius.clamp_min(1e-8)
    # Teacher is primary. Large deviations are out-of-local-recovery examples;
    # do not teach persistent saturation with equal weight to valid feedback.
    recoverable=teacher_delta[:,:,arm].abs().amax(-1)<=radius[arm].max()*1.5
    weights=torch.where(recoverable,1.,self.continuity_v11['out_of_radius_weight'])
    first=b['fast_known_mask'].long().sum(-1)
    weights=weights.scatter(1,first[:,None],weights.gather(1,first[:,None])*self.residual_v10['execute_weight'])
    teacher_error=F.smooth_l1_loss(actual_r[:,:,arm],target_r[:,:,arm],reduction='none')
    primary=masked_mean(teacher_error*weights[...,None],mask[...,None])
    # Healthy same-state teacher agrees: explicitly train no unnecessary change.
    keep=teacher_delta[:,:,arm].abs().amax(-1)<self.residual_v10['preserve_tolerance']
    preserve=masked_mean(actual_r[:,:,arm].square(),(mask&keep)[...,None])
    # Supervise FULL command increments, including known->new seam. Prefix is
    # a detached known condition; never edited to reduce a continuity loss.
    teacher_action=torch.where(b['fast_known_mask'][...,None],b['fast_known_actions'],desired.detach())
    edge=(mask[:,1:] & (mask[:,:-1]|b['fast_known_mask'][:,:-1]))
    d=(action[:,1:]-action[:,:-1])-(teacher_action[:,1:]-teacher_action[:,:-1])
    seam=masked_mean((d[:,:,arm]/radius[arm]).square(),edge[...,None])
    revision=action.sum()*0
    if prior_target is not None:
        pv=b['previous_valid']&mask&ptv
        requested=(target-prior_target).detach()
        changed=(action-b['previous_actions'])-requested
        revision=masked_mean(F.smooth_l1_loss(changed[:,:,arm]/radius[arm],torch.zeros_like(changed[:,:,arm]),reduction='none'),pv[...,None])
    # Real expert futures remain conditional on expert actions, never relabelled
    # as outcomes of the different corrected student trajectory.
    _,_,_,_,aux,_=predict(self,b,world,expert_actions=b['fast_action'])
    zv=b['future_loss_valid'];cv=b['contact_valid'];dim=self.loss_config.visual_dim
    ze=(aux['future_velocity']-b['future']).float().square()
    visual=masked_mean(ze[...,:dim],zv[...,:dim]);geometry=masked_mean(ze[...,dim:],zv[...,dim:])
    contact=masked_mean(F.binary_cross_entropy_with_logits(aux['contact_logits'],torch.where(cv,b['contact'],0.),reduction='none'),cv)
    total=primary+.1*preserve+self.continuity_v11['seam_weight']*seam+self.continuity_v11['revision_weight']*revision
    total=total+self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact
    return dict(loss=total,fast_action=primary,teacher_endpoint=primary,baseline_preservation=preserve,boundary=seam,
        cross_query_revision=revision,future_visual=visual,future_geometry=geometry,contact=contact,
        residual_target_saturation=masked_mean((~recoverable).float(),mask),
        measured_state_fraction=b['measured_state_valid'].float().mean(),paired_fraction=paired.float().mean())


def configure(system,settings):
    required={'contract','reference_hold_steps','seam_weight','revision_weight','out_of_radius_weight','revision_radius_fraction'}
    if set(settings)!=required or settings['contract']!=CONTRACT or settings['reference_hold_steps']<1:
        raise ValueError('explicit V11 continuity contract required')
    if not all(0<=float(settings[k])<=1 for k in ('seam_weight','revision_weight','out_of_radius_weight')):
        raise ValueError('invalid continuity loss weights')
    if not 0<float(settings['revision_radius_fraction'])<=1:raise ValueError('bounded per-query innovation required')
    system.continuity_v11=dict(settings)
    dim=system.fast.config.observation_dim
    system.fast.sensor_projection=nn.Sequential(nn.Linear(37,dim),nn.SiLU(),nn.Linear(dim,dim))
    system.fast.previous_projection=nn.Linear(14,dim)
    system.training_loss=MethodType(loss,system)
