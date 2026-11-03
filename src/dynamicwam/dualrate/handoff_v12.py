"""Full-plan handoff, separate from the learned local residual.

Raw Full references are immutable. One scalar blends the whole arm trajectory
towards a new proposal, so independent joint clipping cannot change coordination.
The same operation produces training targets and deployed commands. It is not a
claim of globally optimal trajectories or a success-rate lower bound.
"""
from types import MethodType
import torch
from torch.nn import functional as F
from .continuity_v11 import ARMS,defaults,align,teacher_reference
from .system import DualRateSystem,masked_mean

CONTRACT='whole_plan_handoff_v12'

def local_mask(system,b,valid):
    first=b['fast_known_mask'].long().sum(-1)
    index=torch.arange(valid.shape[1],device=valid.device)[None]
    return valid & (index>=first[:,None]) & (index<first[:,None]+system.handoff_v12['local_rows'])

def reconcile(system,b,base,proposal,valid):
    """Bound complete overlapping-plan revisions in raw radians, not delta alone.

    Uncovered tail uses the new Full increments with a common boundary offset;
    this is an explicit generated connector, never a future observation/label.
    The immutable RTC prefix always wins. Grippers retain an existing Full event
    during a large handoff; they are never averaged or learned as corrections.
    """
    b=defaults(b);known=b['fast_known_mask'];free=valid & ~known
    first=known.long().sum(-1);n=base.shape[1];idx=torch.arange(n,device=base.device)[None]
    take=lambda x,i:x.gather(1,i[:,None,None].expand(-1,1,x.shape[-1]))[:,0]
    overlap=b['previous_valid'] & free
    # A missing first overlap connects from the actual last committed command.
    has_first=overlap.gather(1,first[:,None]).squeeze(1)
    has=overlap.any(-1)&has_first
    last=torch.where(overlap,idx,-1).amax(-1).clamp_min(0)
    old_last=take(b['previous_actions'],last);base_last=take(base,last)
    anchor=base+(old_last-base_last)[:,None]
    anchor=torch.where(overlap[...,None],b['previous_actions'],anchor)
    committed=take(b['fast_known_actions'],(first-1).clamp_min(0))
    fallback=base+(committed-take(base,first))[:,None]
    anchor=torch.where(has[:,None,None],anchor,fallback)
    # No known prefix is a supported CPU-only/startup case: preserve raw Full.
    enabled=has | (first>0)
    prior_active=b.get('previous_handoff',torch.zeros_like(enabled))
    cross=(~b['previous_reference_same'])|prior_active|~has
    previous_time=b.get('previous_query_time',b['query_time']-system.residual_v10['action_dt'])
    elapsed=((b['query_time']-previous_time)/system.residual_v10['action_dt']).to(base.dtype).clamp(.5,2.)
    elapsed=torch.where(has,elapsed,1.)
    budget=torch.where(cross,base.new_tensor(system.handoff_v12['handoff_step_rad']),
                       base.new_tensor(system.handoff_v12['same_step_rad']))*elapsed
    change=(proposal-anchor)*system.action_scale_v10
    arm=torch.as_tensor(ARMS,device=base.device)
    magnitude=torch.where(free[...,None],change.index_select(-1,arm).abs(),0.).amax((1,2))
    alpha=torch.minimum(torch.ones_like(magnitude),budget/magnitude.clamp_min(1e-8))
    alpha=torch.where(enabled,alpha,1.)
    effective=anchor+alpha[:,None,None]*(base-anchor)
    result=effective+alpha[:,None,None]*(proposal-base)
    modified=enabled & (alpha<1.-1e-6)
    active=cross & modified
    # Hold only previous Full gripper events, never a fractional average.
    grip_anchor=anchor.clone()
    grip_anchor[:,:,[6,13]]=torch.where(overlap[...,None],b['previous_actions'],committed[:,None])[:,:,[6,13]]
    for out in (effective,result):
        out[:,:,[6,13]]=torch.where(active[:,None,None],grip_anchor[:,:,[6,13]],base[:,:,[6,13]])
    effective=torch.where(known[...,None],b['fast_known_actions'],effective)
    result=torch.where(known[...,None],b['fast_known_actions'],result)
    limits=result.new_tensor(system.residual_v10['bounds']);physical=result*system.action_scale_v10+system.action_mean_v10
    if not bool((torch.isfinite(physical)&(physical>=limits[:,0]-1e-5)&(physical<=limits[:,1]+1e-5)).all()):
        raise ValueError('whole-plan connector exceeds native joint limits')
    achieved=torch.where(free[...,None],((result-anchor)*system.action_scale_v10).index_select(-1,arm).abs(),0.).amax((1,2))
    return result,effective,dict(alpha=alpha,active=active,modified=modified,budget_rad=budget,
        raw_change_rad=magnitude,anchor=anchor,overlap=overlap,cross=cross,enabled=enabled,achieved_change_rad=achieved)

def diagnostics(system,b,target,base,valid,handoff,action=None):
    free=valid&~b['fast_known_mask'];near=local_mask(system,b,valid)
    first=b['fast_known_mask'].long().sum(-1)
    first_mask=torch.zeros_like(free).scatter(1,first[:,None],True)&free
    raw=((target-base)*system.action_scale_v10).index_select(-1,torch.as_tensor(ARMS,device=base.device)).abs().amax(-1)
    command_error=None if action is None else ((target-action)*system.action_scale_v10).index_select(-1,torch.as_tensor(ARMS,device=base.device)).abs().amax(-1)
    replay=b.get('recorded_replay',torch.zeros(len(base),dtype=torch.bool,device=base.device))
    metrics={}
    for domain,domain_mask in (('all',torch.ones_like(replay)),('replay',replay),('expert',~replay)):
        for band,mask in (('first',first_mask),('near',near),('tail',free&~near)):
            mask=mask&domain_mask[:,None]
            metrics[f'teacher_{domain}_{band}_over_radius']=masked_mean((raw>system.residual_v10['arm_radius']).float(),mask)
            metrics[f'teacher_{domain}_{band}_error_rad']=masked_mean(raw,mask)
            metrics[f'teacher_{domain}_{band}_rows']=mask.float().sum()
            # Numerators survive gradient-accumulation averaging even when a
            # microbatch has no rows in one domain. Divide summed numerators by
            # summed counts when reporting conditional domain statistics.
            metrics[f'teacher_{domain}_{band}_error_sum']=torch.where(mask,raw,0.).sum()
            metrics[f'teacher_{domain}_{band}_over_count']=((raw>system.residual_v10['arm_radius'])&mask).float().sum()
            if command_error is not None:metrics[f'teacher_{domain}_{band}_command_error_sum']=torch.where(mask,command_error,0.).sum()
            for threshold in (.06,.15,.24):
                metrics[f'teacher_{domain}_{band}_over_{int(threshold*100):02d}_count']=((raw>threshold)&mask).float().sum()
    metrics.update(handoff_fraction=handoff['active'].float().mean(),handoff_alpha=handoff['alpha'].mean(),
                   cross_reference_fraction=(handoff['cross']&handoff['overlap'].any(-1)).float().mean())
    return metrics

def loss(self,batch,generator=None,action_conditioned=None):
    from .residual_v10 import predict,project
    b=defaults(batch);DualRateSystem._validate_clocks(b);DualRateSystem._validate_prefix_pair(b)
    self.slow.eval();old,target,tvalid,_=teacher_reference(self,b,generator);world=self.world_readout(old)
    paired=b.get('pair_valid',torch.zeros(len(target),dtype=torch.bool,device=target.device))
    if bool(paired.any()):
        prior={k[9:]:v for k,v in b.items() if k.startswith('previous_') and k not in
               ('previous_actions','previous_valid','previous_reference_same','previous_handoff')}
        # Non-paired replay has a real prior-query timestamp, possibly before
        # the CURRENT reference was published. It is not a prior-label example.
        # Use valid current placeholders only for this unused paired forward;
        # retain the actual previous plan/time in b for the real controller.
        for key,value in list(prior.items()):
            if key in b and isinstance(value,torch.Tensor) and value.shape==b[key].shape:
                selector=paired.reshape((len(paired),)+(1,)*(value.ndim-1))
                prior[key]=torch.where(selector,value,b[key])
        prior=defaults(prior)
        DualRateSystem._validate_clocks(prior);DualRateSystem._validate_prefix_pair(prior)
        for key in ('source_time','source_request_time','source_available_time','completed_time','source_known_mask','source_known_actions'):
            if key in b and not torch.equal(prior[key][paired],b[key][paired]):raise ValueError('expert pair crosses reference identity')
        with torch.no_grad():
            previous,_,_,_,previous_raw,_=predict(self,prior,world)
            previous,pv=align(previous,prior['action_intervals'],b['action_intervals'])
        # Keep real recorded cross-reference inputs on replay rows in mixed batches.
        b['previous_actions']=torch.where(paired[:,None,None],previous,b['previous_actions'])
        b['previous_valid']=torch.where(paired[:,None],pv,b['previous_valid'])
        b['previous_reference_same']=torch.where(paired,True,b['previous_reference_same'])
        b['previous_handoff']=torch.where(paired,previous_raw['handoff']['active'],b.get('previous_handoff',torch.zeros_like(paired)))
    action,effective,valid,accepted,raw,cond=predict(self,b,world)
    base=raw['raw_baseline'];radius=self.radius_raw_v10/self.action_scale_v10
    mask=local_mask(self,b,valid&tvalid);arm=torch.as_tensor(ARMS,device=action.device)
    desired=base+(target-base).detach().clamp(-radius,radius)*mask[...,None]
    desired[:,:,[6,13]]=base[:,:,[6,13]]
    desired,_=project(self,desired,b,center=base)
    with torch.no_grad():desired_action,_,_=reconcile(self,b,base,desired,valid)
    raw_error=((raw['raw_proposal']-desired)/radius.clamp_min(1e-8)).index_select(-1,arm)
    weights=torch.ones_like(mask,dtype=action.dtype)
    first=b['fast_known_mask'].long().sum(-1)
    weights.scatter_(1,first[:,None],self.residual_v10['execute_weight'])
    # Local teacher target remains primary, even while the handoff restricts output.
    primary=masked_mean(F.smooth_l1_loss(raw_error,torch.zeros_like(raw_error),reduction='none')*weights[...,None],mask[...,None])
    control_error=((action-desired_action)/radius.clamp_min(1e-8)).index_select(-1,arm)
    control=masked_mean(control_error.square(),mask[...,None])
    edge=mask[:,1:]&(mask[:,:-1]|b['fast_known_mask'][:,:-1])
    d=((action[:,1:]-action[:,:-1])-(desired_action[:,1:]-desired_action[:,:-1])).index_select(-1,arm)/radius[arm]
    seam=masked_mean(d.square(),edge[...,None])
    keep=(((target-base)*self.action_scale_v10).index_select(-1,arm).abs().amax(-1)<self.residual_v10['preserve_tolerance'])&mask
    preserve=masked_mean(((raw['raw_proposal']-base)/radius.clamp_min(1e-8)).index_select(-1,arm).square(),keep[...,None])
    _,_,_,_,aux,_=predict(self,b,world,expert_actions=b['fast_action'])
    zv=b['future_loss_valid'];cv=b['contact_valid'];dim=self.loss_config.visual_dim
    ze=(aux['future_velocity']-b['future']).float().square()
    visual=masked_mean(ze[...,:dim],zv[...,:dim]);geometry=masked_mean(ze[...,dim:],zv[...,dim:])
    contact=masked_mean(F.binary_cross_entropy_with_logits(aux['contact_logits'],torch.where(cv,b['contact'],0.),reduction='none'),cv)
    total=primary+.1*preserve+.1*control+self.continuity_v11['seam_weight']*seam
    total=total+self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact
    result=dict(loss=total,fast_action=primary,teacher_endpoint=primary,baseline_preservation=preserve,boundary=seam,
        cross_query_revision=control,future_visual=visual,future_geometry=geometry,contact=contact,
        measured_state_fraction=b['measured_state_valid'].float().mean(),paired_fraction=paired.float().mean())
    result.update(diagnostics(self,b,target,base,valid&tvalid,raw['handoff'],action))
    return result

def configure(system,settings):
    if set(settings)!={'contract','local_rows','handoff_step_rad','same_step_rad'} or settings['contract']!=CONTRACT:
        raise ValueError('explicit V12 handoff contract required')
    if settings['local_rows']!=4 or not 0<settings['same_step_rad']<=settings['handoff_step_rad']<=.06:
        raise ValueError('local four-row feedback and bounded complete-plan revision required')
    system.handoff_v12=dict(settings);system.training_loss=MethodType(loss,system)
