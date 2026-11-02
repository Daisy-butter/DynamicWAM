"""One-pass, baseline-anchored native TOPP feedback (train/deploy shared).

No sampled action/future tokens, residual accumulation, or gripper correction.
The future decoder is conditioned on actions AFTER the sole action decoder.
Privileged future targets are losses only; they cannot enter action inference.
"""
from dataclasses import replace
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .native_v9 import ResidualEventFast, conditions as v9_conditions, draft
from .native_v6 import current_source, align_plan
from .system import DualRateSystem, masked_mean
from .fast import _version

ARM = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
CONTRACT = 'native_bounded_feedback_onepass_v10'


class InteractionEncoder(nn.Module):
    """Eight completed command/achieved tuples -> four causal context tokens."""
    def __init__(self, dim):
        super().__init__()
        self.input = nn.Linear(64, dim)
        self.position = nn.Embedding(8, dim)
        self.query = nn.Parameter(torch.randn(1, 4, dim) * .02)
        self.blocks = nn.ModuleList(nn.TransformerDecoderLayer(dim, 4,
            dim_feedforward=2*dim, dropout=0., batch_first=True, norm_first=True)
            for _ in range(2))

    def forward(self, history, valid):
        x = self.input(torch.where(valid[..., None], history, 0.))
        x = x + self.position.weight[None]
        # A null key avoids all-masked softmax. Empty histories emit masked zeros.
        x = F.pad(x, (0, 0, 1, 0))
        mask = torch.cat((~valid.any(-1, keepdim=True), valid), -1)
        q = self.query.expand(len(history), -1, -1)
        for block in self.blocks:
            q = block(q, x, memory_key_padding_mask=~mask)
        keep = valid.any(-1, keepdim=True).expand(-1, 4)
        return torch.where(keep[..., None], q, 0.), keep


class BoundedFeedback(ResidualEventFast):
    def __init__(self, config, radius):
        super().__init__(config, radius)
        self.interaction = InteractionEncoder(config.observation_dim)
        self.plan_tokens = nn.Parameter(torch.randn(1, config.future_tokens, config.dim)*.02)
        self.forecast_action = nn.Linear(config.action_dim, config.dim)
        self.forecast = nn.TransformerDecoderLayer(config.dim, config.heads,
            dim_feedforward=config.dim*2, dropout=0., batch_first=True, norm_first=True)
        arm = torch.ones(config.action_dim); arm[[6, 13]] = 0
        self.register_buffer('arm_mask', arm)

    def prepare_context(self, *, interaction_history=None, interaction_valid=None,
                        observation_tokens, observation_valid, **kw):
        b = len(observation_tokens)
        if interaction_history is None:
            interaction_history = observation_tokens.new_zeros(b, 8, 64)
            interaction_valid = torch.zeros(b, 8, dtype=torch.bool, device=observation_tokens.device)
        if (interaction_history.shape != (b, 8, 64) or interaction_valid is None
                or interaction_valid.shape != (b, 8) or interaction_valid.dtype != torch.bool):
            raise ValueError('eight completed interaction tuples and boolean mask required')
        if not torch.isfinite(interaction_history[interaction_valid]).all():
            raise ValueError('nonfinite achieved interaction')
        tokens, valid = self.interaction(interaction_history, interaction_valid)
        cache = super().prepare_context(observation_tokens=torch.cat((observation_tokens, tokens), 1),
            observation_valid=torch.cat((observation_valid, valid), 1), **kw)
        fresh = kw['native_query_index'] == kw['native_source_index']
        correction_valid = cache.draft_valid & ~fresh[:, None]
        inputs = dict(cache.inputs, interaction_history=interaction_history, interaction_valid=interaction_valid,
                      original_observation=observation_tokens, original_observation_valid=observation_valid)
        return replace(cache, inputs=inputs, versions={k:_version(v) for k,v in inputs.items()},
                       correction_valid=correction_valid)

    def hidden(self, cache):
        self._check_cache(cache)
        c = self.config
        a = self.action_in(torch.where(cache.draft_valid[...,None],cache.draft_actions,0.))
        a = a + self.action_position.weight[None] + self.type_embedding.weight[0]
        zi = torch.arange(c.future_tokens, device=a.device)
        p = self.plan_tokens + self.entity_embedding(zi // 4)[None] + self.future_position(zi % 4)[None]
        p = p.expand(len(a),-1,-1) + self.type_embedding.weight[1]
        x = torch.cat((a,p),1) + cache.time_features
        valid = torch.cat((cache.draft_valid, cache.future_valid),1)
        # At least the current draft row is required by the public entry point.
        for block, kv in zip(self.blocks, cache.kv):
            x = block(x, valid, kv, cache.current_allowed, cache.slow_allowed,
                      cache.time_features, cache.action_deviation)
        return self.norm(x)

    def forecast_for(self, hidden, actions, cache):
        c = self.config
        memory = self.forecast_action(torch.where(cache.draft_valid[...,None], actions, 0.))
        memory = memory + self.action_position.weight[None]
        # Invalid/terminal suffix has no influence on valid future predictions.
        p = self.forecast(hidden[:,c.chunk_size:], memory, memory_key_padding_mask=~cache.draft_valid)
        return self.future_out(p), self.contact_out(p)

    def predict(self, cache):
        hidden = self.hidden(cache)
        delta = self.action_out(hidden[:,:self.config.chunk_size]).float().tanh()
        delta = delta * self.correction_radius * self.arm_mask
        clean = cache.draft_actions + torch.where(cache.correction_valid[...,None], delta, 0.)
        future, contact = self.forecast_for(hidden, clean, cache)
        return dict(actions=clean, future=future, contact_logits=contact, hidden=hidden)


def conditions(self, batch, world):
    if 'interaction_history' in batch:
        from .native_interaction import validate_interaction
        validate_interaction(batch)
    result = v9_conditions(self,batch,world)
    for key in ('interaction_history','interaction_valid'):
        if key in batch: result[key] = batch[key]
    return result


def outputs(fast, context, fields, contacts):
    result = fast.predict(context)
    return dict(actions=result['actions'], future=torch.where(fields,result['future'],0.),
        contact_probability=torch.where(contacts,result['contact_logits'].sigmoid(),0.),
        future_valid=context.future_valid,future_field_valid=fields,contact_valid=contacts)


@torch.no_grad()
def sample_fast(self,batch,world,*,steps=1,generator=None,initial_action_noise=None):
    if steps != 1: raise ValueError('one-pass feedback has no diffusion step count')
    if batch['fast_known_mask'].any():raise ValueError('native feedback requires zero hard prefix')
    cond = self.conditions(batch,world)
    if not cond['draft_valid'][:,0].all():raise ValueError('no current baseline command')
    cache=self.fast.prepare_context(**cond)
    fields=cache.future_valid[...,None] & self.future_field_support[None]
    contacts=cache.future_valid[...,None] & self.contact_field_support[None]
    engine=getattr(self,'_fast_graph_sampler',None)
    if engine is None:return outputs(self.fast,cache,fields,contacts)
    a=cache.draft_actions;z=a.new_zeros(len(a),self.fast.config.future_tokens,self.fast.config.future_dim)
    return engine.sample(a,z,torch.zeros_like(a),torch.zeros(a.shape[:2],device=a.device,dtype=torch.bool),
                         cache,fields,contacts,steps=1)


def enable_fast_graph(self):
    from .fast_graph import FastGraphSampler
    if self.training or getattr(self,'_fast_graph_sampler',None) is not None:
        raise ValueError('one fresh graph engine per frozen eval model required')
    class OnePassGraph(FastGraphSampler):
        def _eager(engine,values,context,steps):
            if steps!=1:raise ValueError('one-pass graph only')
            out=outputs(engine.fast,context,values['fields'],values['contacts'])
            # Internal mutation parity fixtures test masks; production rejects prefixes.
            out['actions']=torch.where(values['mask'][...,None],values['known'],out['actions'])
            return out
    self._fast_graph_sampler=OnePassGraph(self.fast)
    return self._fast_graph_sampler


def training_loss(self,batch,generator=None,action_conditioned=None):
    DualRateSystem._validate_clocks(batch)
    if batch['fast_known_mask'].any():raise ValueError('native boundary has no hard prefix')
    allowed=torch.zeros_like(batch['source_known_mask'])
    allowed[:,0]=batch.get('source_has_committed_waypoint',torch.zeros(len(allowed),device=allowed.device,dtype=torch.bool))
    if not torch.equal(allowed,batch['source_known_mask']):raise ValueError('only irrevocably selected source waypoint may be clamped')
    if not torch.allclose(batch['rgb_times'],batch['query_time'][:,None].expand_as(batch['rgb_times']),rtol=0,atol=1e-7):
        raise ValueError('current RGB must use actual native query exposure')
    self.slow.eval()
    old=self.predict_reference(batch,generator=generator)
    world=self.world_readout(old)
    world['native_available']=torch.ones_like(batch['native_query_index'],dtype=torch.bool)
    b=dict(batch)
    previous={k[9:]:v for k,v in batch.items() if k.startswith('previous_') and k not in ('previous_actions','previous_valid')}
    # Recompute the prior proposal with this model, joined by absolute command id.
    if previous and bool(batch['pair_valid'].any()):
        with torch.no_grad():
            p=self.sample_fast(previous,world,steps=1)['actions']
            p,pv=align_plan(p,previous['native_query_index'],batch['native_query_index'])
        b.update(previous_actions=p,previous_valid=pv & batch['pair_valid'][:,None])
    cond=self.conditions(b,world);cache=self.fast.prepare_context(**cond)
    out=self.fast.predict(cache);clean=out['actions'];prior=cache.draft_actions
    expert=batch['fast_action'];valid=batch['fast_action_valid'] & cache.draft_valid
    replay=batch.get('native_replay',torch.zeros(len(expert),device=expert.device,dtype=torch.bool))
    tb=current_source(batch)
    # The historical positive experiment learned the CURRENT Full correction,
    # with an auxiliary expert term, not expert replacement of the Full mode.
    # The same absolute-command draws reduce arbitrary teacher mode variance.
    tb['feedback_noise_seed']=batch['feedback_noise_seed']
    target=(old if torch.equal(batch['native_source_index'],batch['native_query_index']) and not batch['source_known_mask'].any()
            else self.predict_reference(tb,generator=generator))['reference_actions'].detach()
    arm=torch.tensor(ARM,device=expert.device)
    radius=self.fast.correction_radius.index_select(0,arm)
    difference=(target-prior).index_select(-1,arm)
    reachable=(difference.abs()<=radius+1e-7).all(-1)
    grip_ok=(target[:,:,[6,13]]-prior[:,:,[6,13]]).abs().amax(-1)<.1
    learn=valid & cache.correction_valid & grip_ok
    weight=clean.new_ones(1,16,1);weight[:,:1]=self.native_v6['execute_weight']
    error=(clean-target).index_select(-1,arm)/radius
    # Huber against the UNCLIPPED endpoint still provides a corrective direction
    # outside the trust region. Dropping every unreachable row starves feedback.
    # This is a bounded local update, never a claim that the endpoint was reached.
    trust=(.25+.75*torch.exp(-difference.abs().mean(-1)/radius.mean())).detach()
    action=masked_mean(F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')*weight*trust[...,None],learn[...,None])
    expert_delta=(expert-prior).index_select(-1,arm)
    same_direction=(expert_delta*difference).sum(-1)>=0
    expert_mask=learn & ~replay[:,None] & same_direction
    expert_error=(clean-expert).index_select(-1,arm)/radius
    expert_loss=masked_mean(F.smooth_l1_loss(expert_error,torch.zeros_like(expert_error),reduction='none')*weight,expert_mask[...,None])
    correction=(clean-prior).index_select(-1,arm)/radius
    retention=masked_mean(correction.square(),(valid & ~learn)[...,None])
    # Match target changes, not low-pass filtering the commanded motion itself.
    edge=learn[:,1:] & learn[:,:-1]
    derivative=(clean[:,1:]-clean[:,:-1]-(target[:,1:]-target[:,:-1])).index_select(-1,arm)/radius
    boundary=masked_mean(F.smooth_l1_loss(derivative,torch.zeros_like(derivative),reduction='none'),edge[...,None])
    continuation=action.new_zeros(())
    if 'previous_actions' in b:
        pa=b['previous_actions'];pv=b['previous_valid'] & learn
        agreement=torch.exp(-((pa-target).index_select(-1,arm)/radius).square().mean(-1).detach())
        continuation=masked_mean(F.smooth_l1_loss(clean.index_select(-1,arm),pa.index_select(-1,arm),reduction='none')/radius*agreement[...,None],pv[...,None])
    # Future supervision uses the actual demonstrated action sequence explicitly.
    # These targets never enter the action trunk, even on the training path.
    future,contact=self.fast.forecast_for(out['hidden'],expert,cache)
    fv=batch['future_loss_valid'] & ~replay[:,None,None]
    cv=batch['contact_valid'] & ~replay[:,None,None]
    # Labels must have an actual fully recorded prefix through each event.
    supported=valid.long().cumprod(1)[:,torch.tensor([0,1,3,7],device=valid.device)].bool().repeat(1,3)
    fv=fv & supported[...,None];cv=cv & supported[...,None]
    vd=self.loss_config.visual_dim
    ze=F.smooth_l1_loss(future,torch.where(fv,batch['future'],0.),reduction='none')
    visual=masked_mean(ze[...,:vd],fv[...,:vd]);geometry=masked_mean(ze[...,vd:],fv[...,vd:])
    contact_loss=masked_mean(F.binary_cross_entropy_with_logits(contact,torch.where(cv,batch['contact'],0.),reduction='none'),cv)
    total=action+.1*expert_loss+.01*retention+.05*boundary+.05*continuation+self.loss_config.future_visual*visual+self.loss_config.future_geometry*geometry+self.loss_config.contact*contact_loss
    return dict(loss=total,fast_action=action,expert_auxiliary=expert_loss,retention=retention,boundary=boundary,continuation=continuation,
        future_visual=visual,future_geometry=geometry,contact=contact_loss,
        reachable_fraction=masked_mean(reachable.float(),valid).detach(),
        supervised_action_fraction=masked_mean(learn.float(),valid).detach())


def reference(self,batch,generator=None):
    """Frozen Full 2+8; source/current teacher share per-command action noise."""
    from .native_v7_deployment import ordinal_action_noise
    self._validate_clocks(batch);source=self.source(batch);b=len(source.initial_state)
    seeds=batch.get('feedback_noise_seed',batch.get('frozen_source_seed'))
    if seeds is None:
        # Expert bootstrap has no stored draw; one draw is copied to its teacher.
        seeds=torch.randint(0,2**31,(b,),device=source.initial_state.device,generator=generator)
        batch['feedback_noise_seed']=seeds
    video=[];action=[]
    for i in range(b):
        seed=int(seeds[i]);g=torch.Generator(device=source.initial_state.device).manual_seed(seed)
        video.append(self._noise((1,*self.future_latent_shape),source.condition_latent,g))
        action.append(ordinal_action_noise(int(batch['native_source_index'][i]),16,14,seed,source.initial_state[i:i+1]))
    with torch.no_grad():
        r=self.slow.sample_reference(source,torch.cat(video),torch.cat(action),batch['source_known_actions'],batch['source_known_mask'],
            batch['source_action_times'],batch['source_future_intervals'],batch['source_time'],batch['completed_time'],
            action_intervals=batch['source_action_intervals'],source_metadata=self._spatial_metadata(batch,'source'),future_metadata=self._spatial_metadata(batch,'future'))
    r['native_source_index']=batch['native_source_index'];return r


def configure(model,settings):
    if settings != dict(contract=CONTRACT,action_steps=1,gripper_correction=False,fresh_correction=False,
                        interaction_history=8,interaction_tokens=4,forecast='action_conditioned_targets_only'):
        raise ValueError('unsupported bounded feedback train/deploy contract')
    model.bounded_feedback=dict(settings)
    for name,fn in dict(conditions=conditions,sample_fast=sample_fast,enable_fast_graph=enable_fast_graph,
                        training_loss=training_loss,predict_reference=reference).items():
        setattr(model,name,MethodType(fn,model))
