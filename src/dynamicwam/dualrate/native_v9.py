"""Native command-event residual policy. Seconds never choose action rows.

Full supplies the action baseline, a single compact A/Z network supplies bounded
corrections. The clean-action parameterization is used inside every flow step,
not as a post-sampling blend. Zero action-head weights reproduce aligned A10.
"""
from dataclasses import replace
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .native_v7 import CompactEventFast, conditions as base_conditions
from .native_v6 import align_plan, configure as configure_v6, current_source, validate_events
from .fast import _JointBlock, _version
from .system import masked_mean, fm_input


class SplitWorldBlock(_JointBlock):
    """History and predicted future have independent attention normalization/gates."""
    def __init__(self, config):
        super().__init__(config)
        self.history_gate=nn.Parameter(torch.tensor(0.))

    def forward(self,x,valid,kv,current_allowed,slow_allowed,condition,deviation):
        sa,sc,ca,cc,fa,fc=self.modulation(condition).chunk(6,-1)
        q,k,v=self.qkv(self.norm_self(x)*(1+sc)+sa).chunk(3,-1)
        mixed=F.scaled_dot_product_attention(self.q_norm(self._heads(q)),self.k_norm(self._heads(k)),
            self._heads(v),attn_mask=valid[:,None,None,:])
        x=x+self.self_output(self._merge(mixed))
        latest=self._cross(self.norm_current(x)*(1+cc)+ca,kv[0],kv[1],current_allowed)
        x=x+latest
        query=self.norm_slow(x)*(1+cc)+ca
        history=self._cross(query,kv[2],kv[3],slow_allowed[:,0])
        future=self._cross(query,kv[2],kv[3],slow_allowed[:,1])
        gate=self.slow_gate(torch.cat((condition,self.norm_current(latest),
            deviation[:,None,None].expand(-1,x.shape[1],1)),-1)).sigmoid()
        x=x+self.history_gate.sigmoid()*history+gate*future
        return torch.where(valid[...,None],x+self.ff(self.norm_ff(x)*(1+fc)+fa),0.)


class ResidualEventFast(CompactEventFast):
    def __init__(self,config,radius):
        super().__init__(config)
        self.blocks=nn.ModuleList([SplitWorldBlock(config) for _ in range(config.layers)])
        self.draft_in=nn.Linear(config.action_dim+1,config.dim)
        self.register_buffer('correction_radius',torch.as_tensor(radius,dtype=torch.float32))
        nn.init.zeros_(self.action_out.weight);nn.init.zeros_(self.action_out.bias)

    def prepare_context(self,*,draft_actions,draft_valid,executed_deviation=None,**kw):
        cache=super().prepare_context(**kw)
        if draft_actions.shape!=(len(draft_valid),self.config.chunk_size,self.config.action_dim):
            raise ValueError('draft dimensions differ')
        if draft_valid.dtype!=torch.bool or draft_valid.shape!=draft_actions.shape[:2]:
            raise ValueError('draft validity shape/dtype')
        if not torch.isfinite(draft_actions).all():raise ValueError('nonfinite A10')
        inputs=dict(cache.inputs,draft_actions=draft_actions,draft_valid=draft_valid)
        times=cache.time_features.clone()
        times[:,:self.config.chunk_size]=times[:,:self.config.chunk_size]+self.draft_in(
            torch.cat((torch.where(draft_valid[...,None],draft_actions,0.),draft_valid[...,None].to(draft_actions)),-1))
        masks=[]
        for kind in (0,1):
            real=cache.slow_allowed[...,1:] & (kw['world_types'][:,None]==kind)
            masks.append(torch.cat((~real.any(-1,keepdim=True),real),-1))
        deviation=cache.action_deviation if executed_deviation is None else executed_deviation
        if executed_deviation is not None:inputs['executed_deviation']=executed_deviation
        return replace(cache,inputs=inputs,versions={k:_version(v) for k,v in inputs.items()},
            time_features=times,slow_allowed=torch.stack(masks,1),action_deviation=deviation,
            draft_actions=draft_actions,draft_valid=draft_valid)

    def forward_cached(self,actions,future,sigma,cache,**kw):
        result=super().forward_cached(actions,future,sigma,cache,**kw)
        delta=result['action_velocity'].tanh()*self.correction_radius
        # Missing suffix is masked, never exposed as a valid fallback action.
        clean=cache.draft_actions+torch.where(cache.draft_valid[...,None],delta,0.)
        result['action_clean']=clean
        velocity=(actions-clean)/sigma[:,None,None].clamp_min(1e-5)
        known=kw.get('action_known_mask')
        result['action_velocity']=velocity if known is None else torch.where(known[...,None],0.,velocity)
        return result


def draft(world,batch,ttl):
    actions,valid=align_plan(world['reference_actions'],world['native_source_index'],batch['native_query_index'])
    age=batch['query_time']-world['source_time']
    if (age<0).any():raise ValueError('future source used as action draft')
    valid=valid & world.get('native_available',torch.ones_like(age,dtype=torch.bool))[:,None]
    return actions,valid


def conditions(self,batch,world):
    r=base_conditions(self,batch,world)
    r['draft_actions'],r['draft_valid']=draft(world,batch,self.native_v9['reference_ttl'])
    if 'executed_deviation' in batch:r['executed_deviation']=batch['executed_deviation']
    return r


def training_loss(self,batch,generator=None,action_conditioned=None):
    from .system import DualRateSystem
    DualRateSystem._validate_clocks(batch)
    if batch['fast_known_mask'].any():raise ValueError('native boundary queries have zero hard prefix')
    allowed=torch.zeros_like(batch['source_known_mask']);allowed[:,0]=batch.get('source_has_committed_waypoint',torch.zeros(len(allowed),device=allowed.device,dtype=torch.bool))
    if not torch.equal(allowed,batch['source_known_mask']):raise ValueError('slow commitment must be the actually committed waypoint')
    if not torch.allclose(batch['rgb_times'],batch['query_time'][:,None].expand_as(batch['rgb_times']),atol=1e-7,rtol=0):raise ValueError('native query exposure mismatch')
    self.slow.eval()
    # Actual native-replay samples and offline demonstration samples remain
    # distinguishable. No replay trajectory is assigned another trajectory's Z.
    old=self.predict_reference(batch,generator=generator)
    world=self.world_readout(old)
    world['native_available']=torch.ones_like(batch['native_query_index'],dtype=torch.bool)
    cond=self.conditions(batch,world)
    prior,pv=cond['draft_actions'],cond['draft_valid']
    if not bool(pv[:,:1].all()):raise ValueError('training row has no executable aligned draft')
    teacher_batch=current_source(batch)
    if 'teacher_seed' in batch:teacher_batch['frozen_source_seed']=batch['teacher_seed']
    teacher=self.predict_reference(teacher_batch,generator=generator)['reference_actions'].detach()
    a=batch['fast_action'];av=batch['fast_action_valid'] & pv;z=batch['future'];zv=batch['future_loss_valid']
    replay=batch.get('native_replay',torch.zeros(len(a),device=a.device,dtype=torch.bool))
    # Demonstrations supervise offline rows. Actual replay uses a fresh Full
    # teacher at the SAME state; it does not borrow the old expert suffix.
    desired=torch.where(replay[:,None,None],teacher,a)
    radius=self.fast.correction_radius
    reachable=prior+(desired-prior).clamp(-radius,radius)
    # A clipped expert action target no longer describes the expert future.
    # Mask those Z/contact labels rather than train contradictory A/Z pairs.
    changed=(reachable-desired).abs().amax(-1)>1e-5
    prefix_changed=changed.long().cumsum(1)>0
    compatible=(~prefix_changed[:,torch.tensor([0,1,3,7],device=a.device)]).repeat(1,self.fast.config.future_entities)
    zv=zv & compatible[...,None]
    cv=batch['contact_valid'] & compatible[...,None]
    sigma=torch.rand(len(a),device=a.device,generator=generator)*.98+.01
    na=torch.randn(a.shape,device=a.device,dtype=a.dtype,generator=generator)
    nz=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=generator)
    x=fm_input(reachable,sigma,na)
    zz=torch.where(self.future_field_support[None],torch.where(zv,fm_input(z,sigma,nz),nz),0.)
    out=self.fast(x,zz,sigma,**cond);clean=out['action_clean']
    weight=a.new_ones((1,16,1));weight[:,:1]=self.native_v6['execute_weight']
    main=masked_mean(((clean-reachable)/radius).square()*weight,av[...,None])
    agreement=torch.exp(-((teacher-a).square().mean(-1).detach())/self.native_v6['teacher_scale'])
    agreement=torch.where(replay[:,None],torch.ones_like(agreement),agreement)
    endpoint=masked_mean(F.smooth_l1_loss(clean,teacher,reduction='none')*agreement[...,None],av[...,None])
    # Preserve baseline's joint trajectory; smooth changes to it, not gripper events.
    correction=(clean-prior)/radius
    arm=torch.tensor([0,1,2,3,4,5,7,8,9,10,11,12],device=a.device)
    boundary=masked_mean((correction[:,1:]-correction[:,:-1]).index_select(-1,arm).square(),
        (av[:,1:]&av[:,:-1])[...,None])
    ze=(out['future_velocity']-(nz-z)).float().square();vd=self.loss_config.visual_dim
    visual=masked_mean(ze[...,:vd],zv[...,:vd]);geometry=masked_mean(ze[...,vd:],zv[...,vd:])
    contact=masked_mean(F.binary_cross_entropy_with_logits(out['contact_logits'],torch.where(cv,batch['contact'],0.),reduction='none'),cv)
    auxiliary=a.new_zeros(())
    # Preserve long-horizon Z supervision without attaching an expert future to
    # a different clipped action. This explicit TRAIN-ONLY conditional-world
    # branch supplies the actual expert actions to the SAME A/Z network.
    if not bool(replay.any()) and bool(torch.rand((),device=a.device,generator=generator)<self.loss_config.action_conditioned_probability):
        labels=batch['future_loss_valid']
        aux_z=torch.where(self.future_field_support[None],torch.where(labels,fm_input(z,sigma,nz),nz),0.)
        aux=self.fast(a,aux_z,sigma,action_known_mask=torch.ones_like(av),known_actions=a,**cond)
        ae=(aux['future_velocity']-(nz-z)).float().square()
        auxiliary=self.loss_config.future_visual*masked_mean(ae[...,:vd],labels[...,:vd])+self.loss_config.future_geometry*masked_mean(ae[...,vd:],labels[...,vd:])
        auxiliary=auxiliary+self.loss_config.contact*masked_mean(F.binary_cross_entropy_with_logits(aux['contact_logits'],torch.where(batch['contact_valid'],batch['contact'],0.),reduction='none'),batch['contact_valid'])
    loss=main+.1*endpoint+.05*boundary+self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact+auxiliary
    return dict(loss=loss,fast_action=main,teacher_endpoint=endpoint,boundary=boundary,
        future_visual=visual,future_geometry=geometry,contact=contact,conditional_expert_future=auxiliary,
        target_clipped_fraction=((desired-prior).abs()>radius).float().mean().detach(),
        z_action_compatible_fraction=compatible.float().mean().detach())


def configure(model,settings,supervision):
    expected={'reference_ttl','arm_radius_radians','gripper_radius','motion_sampling','native_replay_required','fast_replan_commands','slow_refresh_commands'}
    if set(settings)!=expected or settings['motion_sampling']!='raw_frames_v1' or settings['native_replay_required'] is not True:
        raise ValueError('explicit V9 timing/motion/replay contract required')
    if settings['reference_ttl']!=4.0 or settings['fast_replan_commands']!=1 or settings['slow_refresh_commands']!=4 or not 0<settings['arm_radius_radians']<=.15 or not 0<settings['gripper_radius']<=.1:
        raise ValueError('bounded physical correction and explicit V9 cadence/TTL required')
    configure_v6(model,supervision)
    model.native_v9=dict(settings)
    model.fast.reference_ttl=settings['reference_ttl']
    model.predict_reference=MethodType(reference,model)
    model.conditions=MethodType(conditions,model)
    model.training_loss=MethodType(training_loss,model)
    from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler
    schedule=dict(model.slow.schedule_config);schedule.pop('action_steps');schedule.pop('video_steps')
    scheduler=FlowMatchScheduler(num_inference_steps=10,**schedule);scheduler.set_timesteps(num_inference_steps=10,training=False)
    model.fast.register_buffer('sampling_sigmas',scheduler.sigmas.float(),persistent=False)


def reference(self,batch,generator=None):
    from .native_v6 import reference as original
    if 'frozen_source_seed' not in batch:return original(self,batch,generator)
    if len(batch['frozen_source_seed'])!=1:raise ValueError('native replay uses per-rank microbatch one')
    g=torch.Generator(device=batch['source_initial_state'].device).manual_seed(int(batch['frozen_source_seed'][0]))
    return original(self,batch,g)
