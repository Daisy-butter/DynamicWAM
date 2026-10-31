"""Physical-time single A/Z model: revocable numerical draft + residual.

One decoder generates final arms and absolute grippers. The noisy trajectory is
draft + FM(target-draft, noise), not a bounded bias toward a teacher. No native
event ordinals or learned clock alignment are used. Old future and draft have
independent validity; missing draft is the identical model's full-action mode.
"""
from dataclasses import replace
from types import MethodType
import math
import torch
from torch import nn
from torch.nn import functional as F
from .fast import FastJointWAM, _JointBlock, _version
from .system import DualRateSystem


def aligned_draft(batch, world, *, max_age):
    target = batch['action_intervals']
    source = world['reference_action_intervals']
    actions = world['reference_actions']
    b, h = target.shape[:2]
    d = actions.shape[-1]
    if (target.dtype != torch.float64 or source.dtype != torch.float64
            or target.shape != (b,h,2) or source.shape != (*actions.shape[:2],2)
            or not torch.isfinite(target).all() or not torch.isfinite(source).all()
            or (target[...,1] <= target[...,0]).any() or (source[...,1] <= source[...,0]).any()
            or not torch.isfinite(actions).all()):
        raise ValueError('draft requires finite actions and ordered FP64 physical intervals')
    if source.shape[1] == 0:
        return actions.new_zeros(b,h,d), torch.zeros(b,h,dtype=torch.bool,device=actions.device)
    # StructuredWorldReadout explicitly appends the action rows. Validate this
    # contract instead of guessing that any valid world token means a valid draft.
    n = actions.shape[1]
    if (world['world_types'].shape[1] < n or not (world['world_types'][:,-n:] == 2).all()
            or not torch.equal(world['world_intervals'][:,-n:],source)):
        raise ValueError('draft validity must be bound to its own action intervals')
    match = (target[:,:,None]-source[:,None]).abs().amax(-1) <= 1e-7
    if (match.sum(-1)>1).any():
        raise ValueError('ambiguous duplicate draft interval')
    index = match.long().argmax(-1)
    age = batch['query_time']-world['source_time']
    valid = (match.any(-1) & world['world_valid'][:,-n:].gather(1,index)
             & (age[:,None]>=0) & (age[:,None]<=max_age+1e-8)
             & (world['completed_time'][:,None]<=batch['query_time'][:,None])
             & (target[...,1]>batch['query_time'][:,None]))
    revoke = batch.get('draft_allowed',torch.ones(b,h,dtype=torch.bool,device=actions.device))
    if revoke.shape != valid.shape or revoke.dtype != torch.bool:
        raise ValueError('draft_allowed must be boolean [B,H], independent of future validity')
    valid = valid & revoke
    draft = actions.gather(1,index[...,None].expand(-1,-1,d))
    draft = torch.where(valid[...,None],draft,torch.zeros_like(draft))
    # Native gripper positions remain absolute decisions, not signed residuals.
    if d == 14:
        arms = draft.new_ones(d); arms[[6,13]] = 0
        draft = draft*arms
    return draft,valid


class IndependentWorldBlock(_JointBlock):
    def __init__(self, config):
        super().__init__(config)
        self.future_gate = nn.Linear(2*config.dim+1,1)
        nn.init.zeros_(self.future_gate.weight)
        nn.init.constant_(self.future_gate.bias,6*config.slow_gate_init-3)
        # The old gate would redundantly couple history and prediction.
        del self.slow_gate

    def forward(self,x,valid,kv,current_allowed,slow_allowed,condition,deviation):
        history_allowed,future_allowed = slow_allowed.unbind(1)
        sa,sc,ca,cc,fa,fc = self.modulation(condition).chunk(6,-1)
        q,k,v = self.qkv(self.norm_self(x)*(1+sc)+sa).chunk(3,-1)
        mixed = F.scaled_dot_product_attention(self.q_norm(self._heads(q)),
            self.k_norm(self._heads(k)),self._heads(v),attn_mask=valid[:,None,None,:])
        x = x+self.self_output(self._merge(mixed))
        latest = self._cross(self.norm_current(x)*(1+cc)+ca,kv[0],kv[1],current_allowed)
        x = x+latest
        history = self._cross(self.norm_slow(x)*(1+cc)+ca,kv[2],kv[3],history_allowed)
        x = x+history
        future = self._cross(self.norm_slow(x)*(1+cc)+ca,kv[2],kv[3],future_allowed)
        gate = F.hardsigmoid(self.future_gate(torch.cat((condition,self.norm_current(latest),
            deviation[:,None,None].expand(-1,x.shape[1],1)),-1)))
        x = x+gate*future
        x = x+self.ff(self.norm_ff(x)*(1+fc)+fa)
        return torch.where(valid[...,None],x,torch.zeros_like(x))


class ContinuousResidualFast(FastJointWAM):
    def __init__(self,config,*,independent_future_gate=True):
        super().__init__(config)
        self.independent_future_gate = independent_future_gate
        if independent_future_gate:
            self.blocks = nn.ModuleList([IndependentWorldBlock(config) for _ in range(config.layers)])
        self.draft_embedding = nn.Linear(config.action_dim+1,config.dim)

    def prepare_context(self,*,action_draft,draft_valid,future_allowed,**conditions):
        b = conditions['query_time'].shape[0]; c = self.config
        if (action_draft.shape != (b,c.chunk_size,c.action_dim)
                or draft_valid.shape != (b,c.chunk_size) or draft_valid.dtype != torch.bool
                or future_allowed.shape != (b,) or future_allowed.dtype != torch.bool
                or not torch.isfinite(action_draft).all()):
            raise ValueError('invalid draft/future revocation interface')
        context = super().prepare_context(**conditions)
        types = conditions['world_types']
        def allowed(kind):
            real = context.slow_allowed[...,1:] & (types[:,None]==kind)
            if kind == 1:
                real = real & future_allowed[:,None,None]
            return torch.cat((~real.any(-1,keepdim=True),real),-1)
        # Type2 is not reread as generic memory: its numerical values explicitly
        # condition every A token and center its FM path in train AND inference.
        draft = torch.where(draft_valid[...,None],action_draft,torch.zeros_like(action_draft))
        embed = self.draft_embedding(torch.cat((draft,draft_valid[...,None].to(draft.dtype)),-1))
        features = context.time_features+F.pad(embed,(0,0,0,c.future_tokens))
        inputs = dict(conditions,action_draft=action_draft,draft_valid=draft_valid,future_allowed=future_allowed)
        if any(_version(v) is None for v in inputs.values() if isinstance(v,torch.Tensor)):
            raise ValueError('query context requires versioned tensors')
        history,future = allowed(0),allowed(1)
        separate = torch.stack((history,future),1)
        combined = history[...,1:] | future[...,1:]
        combined = torch.cat((~combined.any(-1,keepdim=True),combined),-1)
        return replace(context,inputs=inputs,
            versions={k:_version(v) for k,v in inputs.items() if isinstance(v,torch.Tensor)},
            slow_allowed=separate if self.independent_future_gate else combined,time_features=features)


def conditions(self,batch,world):
    result = DualRateSystem.conditions(self,batch,world)
    draft,valid = aligned_draft(batch,world,max_age=self.continuous_v8['draft_max_age'])
    future_allowed = batch.get('world_future_allowed',torch.ones_like(batch['query_time'],dtype=torch.bool))
    return dict(result,action_draft=draft,draft_valid=valid,future_allowed=future_allowed)


def configure(model,settings):
    if (set(settings) != {'draft_max_age','draft_dropout','future_dropout','action_mode','independent_future_gate'}
            or not math.isfinite(settings['draft_max_age'])
            or not 0 < settings['draft_max_age'] <= 2.88
            or settings['action_mode'] not in ('full','residual')
            or type(settings['independent_future_gate']) is not bool
            or model.fast.independent_future_gate != settings['independent_future_gate']
            or any(not 0 <= settings[k] <= 1 for k in ('draft_dropout','future_dropout'))):
        raise ValueError('bounded physical-time draft age required')
    if any(getattr(model,k,None) for k in ('native_v5','native_v6','native_v7')):
        raise ValueError('continuous physical time cannot use native event-coordinate policy')
    model.continuous_v8 = dict(settings)
    model.conditions = MethodType(conditions,model)
    model.action_center = MethodType(action_center,model)
    model.training_loss = MethodType(training_loss,model)
    model.slow.requires_grad_(False).eval()


def action_center(self,batch,world):
    draft,_=aligned_draft(batch,world,max_age=self.continuous_v8['draft_max_age'])
    return draft if self.continuous_v8['action_mode']=='residual' else torch.zeros_like(draft)


def training_loss(self,batch,generator=None,action_conditioned=None):
    self.slow.eval()
    batch = dict(batch)
    b,h = batch['fast_action'].shape[:2]
    device = batch['fast_action'].device
    # Explicit independent ablations/full-action recovery are seen in training.
    # Keep the real sampled slow predictions; never replace drafts with expert GT.
    for key,setting,shape in (('draft_allowed','draft_dropout',(b,1)),
                              ('world_future_allowed','future_dropout',(b,))):
        keep = torch.rand(shape,device=device,generator=generator)>=self.continuous_v8[setting]
        if key=='draft_allowed':
            keep = keep.expand(b,h)
        batch[key] = keep & batch.get(key,torch.ones_like(keep))
    return DualRateSystem.training_loss(self,batch,generator,action_conditioned=action_conditioned)
