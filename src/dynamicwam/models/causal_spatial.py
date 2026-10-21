"""Observation-only slow memory and fresh joint future/action flow matching.

The backbone's Action residual stream and original native motion/state/register
tokens are retained. A new lightweight future stream participates bidirectionally
at every native Action layer. Only observation-only M may persist across cycles.
The original backbone forward is available separately for a native reference;
an immutable teacher must be a separately loaded instance owned by the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dynamicwam.models.causal_spatial_modules import CausalSpatialConfig, FutureBlock, SpatialBelief
from dynamicwam.models.rtc_prefix import apply_prefix, action_token_times
from dynamicwam.models.action_core_ops import ActionCoreOps
from dynamicwam.vendor.wan.modules.model import sinusoidal_embedding_1d


@dataclass(frozen=True)
class ObservationMemory:
    keys: tuple[torch.Tensor,...]
    values: tuple[torch.Tensor,...]
    source_time: torch.Tensor
    source_id: torch.Tensor
    model_version: str
    text_context: torch.Tensor
    version: int = 1

    def detached(self):
        return ObservationMemory(tuple(x.detach() for x in self.keys),
            tuple(x.detach() for x in self.values),self.source_time.detach(),
            self.source_id.detach(),self.model_version,self.text_context.detach(),self.version)


class CausalSpatialActionCore(ActionCoreOps,nn.Module):
    """Construct fast-only state without allocating WAN/VAE/T5 parameters."""
    def __init__(self,action_expert,absolute_motion_tokens,native_config):
        super().__init__()
        self.action_expert = action_expert
        self.absolute_motion_tokens = absolute_motion_tokens
        self.config = native_config


def _attention(q,k,v,valid=None):
    """Common head layout [B,N,H,D], with no data-dependent host decisions."""
    if valid is not None:
        empty = ~valid.any(-1)
        sentinel = torch.arange(k.shape[1],device=k.device)[None] == 0
        valid = valid | (empty[:,None] & sentinel)
        k = torch.where(empty[:,None,None,None],torch.zeros_like(k),k)
        v = torch.where(empty[:,None,None,None],torch.zeros_like(v),v)
        mask = valid[:,None,None,:]
    else:
        empty,mask = None,None
    out = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                        attn_mask=mask,dropout_p=0.).transpose(1,2)
    return out if empty is None else torch.where(empty[:,None,None,None],torch.zeros_like(out),out)


def _native_layer_norm_fp32(module,x):
    """Native WAN normalization math without mixed BF16 affine/FP32 inputs."""
    weight = getattr(module,"weight",None)
    bias = getattr(module,"bias",None)
    if not hasattr(module,"normalized_shape"):
        return module(x).float()
    return F.layer_norm(x.float(),module.normalized_shape,
                        None if weight is None else weight.float(),
                        None if bias is None else bias.float(),module.eps)


class CausalSpatialModel(nn.Module):
    def __init__(self,backbone,config: CausalSpatialConfig | None=None,*,patch_size=None):
        super().__init__()
        self.backbone = backbone
        self.config = config or CausalSpatialConfig()
        cfg, native = self.config, backbone.config
        self.world_dim = int(native.compact_wan.dim)
        self.heads = int(native.compact_wan.num_heads)
        self.head_dim = int(native.compact_wan.head_dim)
        self.layers = int(native.ae_num_layers)
        if self.world_dim != self.heads*self.head_dim:
            raise ValueError("native world/head geometry differs")
        if patch_size is None:
            if not hasattr(backbone,"compact_wan"):
                raise ValueError("fast-only constructor requires the serialized patch_size")
            patch_size = backbone.compact_wan.video_model.wan_model.patch_size
        self.patch_size = tuple(patch_size)
        self._rope_grids = {}
        self.future_patch = nn.Conv3d(cfg.latent_channels,cfg.z_dim,self.patch_size,stride=self.patch_size)
        self.future_head = nn.Linear(cfg.z_dim,cfg.latent_channels*math.prod(self.patch_size))
        self.future_norm = nn.LayerNorm(cfg.z_dim)
        self.future_time = nn.Sequential(nn.Linear(4,cfg.z_dim),nn.SiLU(),nn.Linear(cfg.z_dim,6*cfg.z_dim))
        self.future_physical = nn.Sequential(nn.Linear(4,cfg.z_dim),nn.SiLU(),nn.Linear(cfg.z_dim,cfg.z_dim))
        self.future_blocks = nn.ModuleList(FutureBlock(cfg.z_dim,self.world_dim,self.heads,cfg.z_ffn_ratio) for _ in range(self.layers))
        self.belief = SpatialBelief(cfg,int(native.action_dim))
        self.belief_kv = nn.ModuleList(nn.Linear(cfg.belief_dim,2*self.world_dim) for _ in range(self.layers))
        self.belief_position = nn.Linear(4,cfg.belief_dim)
        self.plan_encoder = nn.Sequential(nn.Linear(native.action_dim+3,cfg.belief_dim),nn.SiLU(),nn.Linear(cfg.belief_dim,cfg.belief_dim))
        self.action_plan = nn.Linear(cfg.belief_dim,native.ae_dim)
        self.future_plan = nn.Linear(cfg.belief_dim,cfg.z_dim)
        self.plan_gate = nn.Parameter(torch.zeros(()))
        self.action_time = nn.Sequential(nn.Linear(4,native.ae_dim),nn.SiLU(),nn.Linear(native.ae_dim,native.ae_dim))
        nn.init.zeros_(self.action_time[-1].weight)
        nn.init.zeros_(self.action_time[-1].bias)

    @property
    def gradient_checkpointing(self):
        return self.config.gradient_checkpointing and self.training and torch.is_grad_enabled()

    def _checkpoint(self,fn,*args):
        return checkpoint(fn,*args,use_reentrant=False) if self.gradient_checkpointing else fn(*args)

    def _rope(self,x,shape):
        # Same native analytic RoPE, independent of any world-model parameters.
        key = (x.device.type,x.device.index,tuple(shape),self.head_dim)
        grid = self._rope_grids.get(key)
        if grid is None:
            complex_dim = self.head_dim//2
            counts = (complex_dim-2*(complex_dim//3),complex_dim//3,complex_dim//3)
            pieces = []
            for length,count in zip(shape,counts):
                position = torch.arange(length,device=x.device,dtype=torch.float64)
                frequency = 1./torch.pow(10000.,torch.arange(count,device=x.device,dtype=torch.float64)/count)
                phase = position[:,None]*frequency[None]
                pieces.append(torch.polar(torch.ones_like(phase),phase))
            f,h,w = shape
            grid = torch.cat((pieces[0][:,None,None].expand(f,h,w,-1),
                              pieces[1][None,:,None].expand(f,h,w,-1),
                              pieces[2][None,None,:].expand(f,h,w,-1)),-1).reshape(f*h*w,1,complex_dim)
            self._rope_grids[key] = grid
        value = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1],-1,2).contiguous())
        return torch.view_as_real(value*grid).flatten(-2).float()

    def _observe_layer(self,tokens,time,text,layer,shape):
        wan = self.backbone.compact_wan.video_model.wan_model.blocks[layer]
        mod = self.backbone._block_modulation(wan,time)
        shift,scale,gate,fshift,fscale,fgate = (x.squeeze(2) for x in mod)
        normalized = _native_layer_norm_fp32(wan.norm1,tokens)*(1+scale)+shift
        attn = wan.self_attn
        cast = normalized.to(attn.q.weight.dtype)
        b,n,_ = cast.shape
        q = attn.norm_q(attn.q(cast)).reshape(b,n,self.heads,self.head_dim)
        k = attn.norm_k(attn.k(cast)).reshape(b,n,self.heads,self.head_dim)
        v = attn.v(cast).reshape(b,n,self.heads,self.head_dim)
        q,k = self._rope(q,shape),self._rope(k,shape)
        dtype = attn.q.weight.dtype
        attention_dtype = getattr(torch,self.config.attention_precision) if q.is_cuda else torch.float32
        attended = _attention(q.to(attention_dtype),k.to(attention_dtype),v.to(attention_dtype))
        updated = tokens.float()+attn.o(attended.flatten(2).to(attn.o.weight.dtype)).float()*gate
        updated = updated+wan.cross_attn(_native_layer_norm_fp32(wan.norm3,updated).to(dtype),text.to(dtype),None).float()
        feed = _native_layer_norm_fp32(wan.norm2,updated)*(1+fscale)+fshift
        updated = updated+wan.ffn(feed.to(wan.ffn[0].weight.dtype)).float()*fgate
        return updated,k,v

    def encode_memory(self,condition_latent,text_embeddings,source_time=None,source_id=None):
        """The signature deliberately cannot accept actions or future latents."""
        if not hasattr(self.backbone,"compact_wan"):
            raise RuntimeError("fast-only model requires externally published observation memory")
        wrapper = self.backbone.compact_wan.video_model
        tokens,grid = wrapper._patch_video_latent(condition_latent)
        patch = self.patch_size
        shape = tuple(int(condition_latent.shape[2+i])//patch[i] for i in range(3))
        b = tokens.shape[0]
        text = self.backbone.compact_wan.prepare_text_context(text_embeddings)
        zero = torch.zeros(b,device=tokens.device,dtype=torch.float32)
        wan = wrapper.wan_model
        if hasattr(wan,"time_embedding"):
            # Real native WAN weights are BF16 before an accelerator is attached.
            # Do not assume FP32 time weights as a reduced CPU fixture might.
            positions = zero[:,None].expand(-1,tokens.shape[1]).reshape(-1)
            te = wan.time_embedding(sinusoidal_embedding_1d(wan.freq_dim,positions)
                .reshape(b,tokens.shape[1],-1).to(next(wan.time_embedding.parameters()).dtype))
            time = wan.time_projection(te.to(next(wan.time_projection.parameters()).dtype)).reshape(
                b,tokens.shape[1],6,self.world_dim)
        else:
            _,time = self.backbone._build_video_time_embeddings(zero,tokens.shape[1])
        keys,values = [],[]
        for layer in range(self.layers):
            def run(x,tm,tx,index=layer):
                return self._observe_layer(x,tm,tx,index,shape)
            tokens,k,v = self._checkpoint(run,tokens,time,text)
            keys.append(k);values.append(v)
        def column(value,dtype):
            if value is None:
                return torch.zeros(b,device=tokens.device,dtype=dtype)
            return torch.as_tensor(value,device=tokens.device,dtype=dtype).reshape(-1).expand(b)
        return ObservationMemory(tuple(keys),tuple(values),column(source_time,torch.float64),
                                 column(source_id,torch.long),self.config.model_version,text)

    def prepare_belief(self,batch):
        commands = batch.get("recorded_capture_commands",batch.get("executed_actions"))
        command_valid = batch.get("command_valid",batch.get("executed_valid"))
        if commands is None or command_valid is None:
            raise ValueError("recorded capture commands and explicit validity are required")
        out = self.belief(batch["spatial_rgb"],batch["spatial_flow"],batch["spatial_valid"],
                          batch["spatial_flow_valid"],batch["spatial_dt"],commands,command_valid)
        tokens = out["tokens"]
        b = tokens.shape[0]
        age = torch.as_tensor(batch["observation_age"],device=tokens.device,dtype=torch.float32).reshape(b,1)
        time = torch.cat((age,age.square(),torch.log1p(age.clamp_min(0)),torch.ones_like(age)),-1)
        tokens = tokens+self.belief_position(time.to(self.belief_position.weight.dtype))[:,None]
        out["tokens"] = tokens
        out["kv"] = tuple(proj(tokens.to(proj.weight.dtype)).reshape(
            b,tokens.shape[1],2,self.heads,self.head_dim).unbind(2) for proj in self.belief_kv)
        return out

    def prepare_plan(self,batch):
        """Editable proposal context for A/Z only; never a current-state input.

        The H2 committed prefix stays in the native prefix interface. Only the
        two subsequent editable rows may supply soft context. This fixed input
        is prepared once per sample trajectory and cannot change M or b.
        """
        previous = batch.get("previous_plan")
        if previous is None:
            previous = torch.zeros_like(batch["noisy_actions"])
            valid = torch.zeros(previous.shape[:2],device=previous.device,dtype=torch.bool)
        else:
            valid = batch["previous_plan_valid"].bool()
        rows = torch.arange(previous.shape[1],device=previous.device)
        valid = valid & ((rows>=2)&(rows<4))[None]
        age = torch.as_tensor(batch["observation_age"],device=previous.device,dtype=torch.float32).reshape(previous.shape[0],1)
        offsets = batch["action_target_offsets"].to(device=previous.device,dtype=torch.float32)
        safe = torch.where(valid[...,None],previous,torch.zeros_like(previous))
        feature = torch.cat((safe,offsets[...,None],valid[...,None].to(safe),
                              age[:,None].expand(-1,safe.shape[1],-1).to(safe)),-1)
        plan = self.plan_encoder(feature.to(self.plan_encoder[0].weight.dtype))
        gate = self.plan_gate.tanh()
        action = self.action_plan(plan.to(self.action_plan.weight.dtype))*valid[...,None]
        future = self.future_plan(plan.to(self.future_plan.weight.dtype))*valid[...,None]
        future = future.sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)[...,None]
        # Additive conditioning adds no keys that could change the denominator
        # of attention when absent. Invalid rows and gate zero are exact no-ops.
        return dict(action=gate*action,future=gate*future)

    def _joint_layer(self,z,actions,ztime,atime,mk,mv,bk,bv,bvalid,layer,shape):
        block = self.future_blocks[layer]
        action_block = self.backbone.action_expert.blocks[layer]
        amod = self.backbone._block_modulation(action_block,atime)
        norm_action = action_block.norm1(actions).float()*(1+amod[1].squeeze(2))+amod[0].squeeze(2)
        aq,ak,av = self.backbone._action_qkv_for_wan(action_block,norm_action)
        zmod = block.modulation_for(ztime)
        zq,zk,zv = block.qkv_for(z,zmod)
        zq,zk = self._rope(zq,shape),self._rope(zk,shape)
        dtype = getattr(torch,self.config.attention_precision) if aq.is_cuda else torch.float32
        q = torch.cat((zq,aq),1).to(dtype)
        k = torch.cat((mk,bk,zk,ak),1).to(dtype)
        v = torch.cat((mv,bv,zv,av),1).to(dtype)
        valid = torch.cat((torch.ones(mk.shape[:2],device=k.device,dtype=torch.bool),
            bvalid,torch.ones(zk.shape[:2],device=k.device,dtype=torch.bool),
            torch.ones(ak.shape[:2],device=k.device,dtype=torch.bool)),1)
        output = _attention(q,k,v,valid)
        zout,aout = output.split((z.shape[1],actions.shape[1]),1)
        z = block.finish(z,zout,zmod)
        actions = actions.float()+action_block.wan_action_o(aout.flatten(2).to(action_block.wan_action_o.weight.dtype)).float()*amod[2].squeeze(2)
        feed = action_block.norm2(actions).float()*(1+amod[4].squeeze(2))+amod[3].squeeze(2)
        actions = actions+action_block.ffn(feed.to(action_block.ffn[0].weight.dtype)).float()*amod[5].squeeze(2)
        return z,actions

    def forward(self,batch,memory=None,belief=None,plan=None):
        if memory is None:
            memory = self.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"],
                                        batch.get("source_time"),batch.get("source_id"))
        if memory.version != 1 or memory.model_version != self.config.model_version:
            raise ValueError("observation memory version differs from model export")
        if len(memory.keys) != self.layers or len(memory.values) != self.layers:
            raise ValueError("observation memory layer count differs")
        belief = self.prepare_belief(batch) if belief is None else belief
        plan = self.prepare_plan(batch) if plan is None else plan
        noisy = batch["noisy_actions"]
        prefix = batch.get("rtc_prefix")
        if prefix is not None:
            noisy = apply_prefix(noisy,prefix)
        actions,core_length = self.backbone._build_action_tokens(batch,noisy,batch["initial_state"])
        offsets = batch["action_target_offsets"].to(device=actions.device,dtype=torch.float32)
        physical = torch.stack((offsets,offsets.square(),torch.sin(offsets),torch.cos(offsets)),-1)
        bias = self.action_time(physical.to(self.action_time[0].weight.dtype))
        actions = torch.cat((actions[:,:1],actions[:,1:1+noisy.shape[1]]+bias+plan["action"],actions[:,1+noisy.shape[1]:]),1)
        action_t = batch["action_t"]
        if prefix is not None:
            action_t = action_token_times(action_t,prefix,actions.shape[1])
        ahead,atime = self.backbone._build_action_time_embeddings(action_t,actions.shape[1])
        future = batch["future_latent"]
        zgrid = self.future_patch(future.to(self.future_patch.weight.dtype))
        shape = tuple(zgrid.shape[2:])
        z = zgrid.flatten(2).transpose(1,2)
        foffsets = batch["future_target_offsets"].to(device=z.device,dtype=torch.float32)
        if foffsets.shape != (z.shape[0],shape[0]):
            raise ValueError("future target offsets must cover actual latent temporal slices")
        ftime = foffsets[:,:,None,None].expand(-1,-1,shape[1],shape[2]).reshape(z.shape[0],-1)
        ffeatures = torch.stack((ftime,ftime.square(),torch.sin(ftime),torch.cos(ftime)),-1)
        z = z+self.future_physical(ffeatures.to(self.future_physical[0].weight.dtype))+plan["future"]
        sigma = batch["video_t"].float().reshape(-1,1)/1000.
        embed = torch.cat((sigma,sigma.square(),torch.sin(math.pi*sigma),torch.cos(math.pi*sigma)),-1)
        ztime = self.future_time(embed.to(self.future_time[0].weight.dtype)).reshape(z.shape[0],1,6,self.config.z_dim)
        for layer in range(self.layers):
            bk,bv = belief["kv"][layer]
            def run(zz,aa,zt,at,mk,mv,kk,vv,valid,index=layer):
                return self._joint_layer(zz,aa,zt,at,mk,mv,kk,vv,valid,index,shape)
            z,actions = self._checkpoint(run,z,actions,ztime,atime,memory.keys[layer],memory.values[layer],bk,bv,belief["valid"])
        action_pred = self.backbone._decode_action_tokens(actions,ahead,core_length)
        patches = self.future_head(self.future_norm(z.to(self.future_norm.weight.dtype)).to(self.future_head.weight.dtype))
        b,f,h,w = z.shape[0],*shape
        pf,ph,pw = self.patch_size
        pixels = patches.reshape(b,f,h,w,pf,ph,pw,self.config.latent_channels)
        pixels = pixels.permute(0,7,1,4,2,5,3,6).reshape(b,self.config.latent_channels,f*pf,h*ph,w*pw)
        return dict(action_pred=action_pred,video_pred=pixels.float(),
            belief_state=belief["tokens"],belief_prediction=belief["prediction"],
            belief_target=belief["target"],belief_logvar=belief["logvar"],belief_valid=belief["prediction_valid"])

    def validate_sampling_inputs(self,batch,*,action_noise,future_noise,sigmas,memory=None):
        """Value checks outside CUDA capture; training forward keeps its checks.

        The native motion validator synchronizes to reject nonfinite features
        and invalid acceleration masks. Do this once per request, never inside
        the captured native Action layers or their ten denoising iterations.
        """
        from dynamicwam.models.rtc_prefix import validate_prefix_condition
        if action_noise.ndim != 3 or tuple(action_noise.shape[1:]) != (16,self.backbone.config.action_dim):
            raise ValueError("sampling action noise must be [B,16,action_dim]")
        if future_noise.ndim != 5 or future_noise.shape[0] != action_noise.shape[0] or future_noise.shape[1] != self.config.latent_channels:
            raise ValueError("sampling future noise geometry differs")
        if future_noise.device != action_noise.device or sigmas.device != action_noise.device:
            raise ValueError("sampling noise/schedule must share one device before invocation")
        if sigmas.ndim != 1 or len(sigmas) != 11 or not bool(torch.isfinite(sigmas).all()):
            raise ValueError("K10 requires eleven finite schedule endpoints")
        if not bool((sigmas[:-1] > sigmas[1:]).all()) or not bool((sigmas[0] == 1) & (sigmas[-1] == 0)):
            raise ValueError("sampling schedule must decrease strictly from noise 1 to clean 0")
        for name,value in (("action",action_noise),("future",future_noise)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(name+" sampling noise is nonfinite")
        prefix = batch.get("rtc_prefix")
        if prefix is not None:
            validate_prefix_condition(prefix,max_prefix=2,reference=action_noise)
        motion = self.backbone.absolute_motion_tokens
        features = batch["absolute_motion_features"]
        interval = batch["absolute_motion_interval_valid_mask"]
        acceleration = batch["absolute_motion_acceleration_valid_mask"]
        if features.device != action_noise.device or interval.device != action_noise.device or acceleration.device != action_noise.device:
            raise ValueError("native motion inputs must be on the sampling device")
        if interval.dtype != torch.bool or acceleration.dtype != torch.bool:
            raise TypeError("native motion validity masks must be boolean")
        motion.validate_inputs(features,interval,acceleration)
        if memory is not None and (memory.version != 1 or memory.model_version != self.config.model_version):
            raise ValueError("observation memory version differs from model export")

    def sample(self,batch,*,action_noise,future_noise,sigmas,memory=None,validate_input=True):
        """Fresh joint trajectory; only a prevalidated eval Graph may skip checks."""
        if type(validate_input) is not bool:
            raise TypeError("validate_input must be a Python bool")
        if not validate_input and self.training:
            raise ValueError("unchecked sampling is only permitted for an eval Graph")
        if validate_input:
            sigmas = torch.as_tensor(sigmas,device=action_noise.device,dtype=torch.float32)
            self.validate_sampling_inputs(batch,action_noise=action_noise,future_noise=future_noise,
                                          sigmas=sigmas,memory=memory)
        # Follow the existing native Graph convention. A finally clause keeps
        # regular training/forward validation enabled even when sampling fails.
        previous = getattr(self.backbone,"_motion_inputs_prevalidated",False)
        self.backbone._motion_inputs_prevalidated = True
        try:
            return self._sample_prevalidated(batch,action_noise=action_noise,future_noise=future_noise,
                                             sigmas=sigmas,memory=memory)
        finally:
            self.backbone._motion_inputs_prevalidated = previous

    def _sample_prevalidated(self,batch,*,action_noise,future_noise,sigmas,memory=None):
        if len(sigmas) != 11:
            raise ValueError("V4 deployment requires K10 with eleven sigma endpoints")
        if memory is None:
            memory = self.encode_memory(batch["source_condition_latent"],batch["source_text_embeddings"],
                                        batch.get("source_time"),batch.get("source_id"))
        batch = dict(batch,noisy_actions=action_noise)
        belief = self.prepare_belief(batch)
        plan = self.prepare_plan(batch)
        actions,future = action_noise.clone(),future_noise.clone()
        prefix = batch.get("rtc_prefix")
        for i in range(10):
            sigma = torch.as_tensor(sigmas[i],device=actions.device,dtype=torch.float32)
            following = torch.as_tensor(sigmas[i+1],device=actions.device,dtype=torch.float32)
            if prefix is not None:
                actions = apply_prefix(actions,prefix)
            timestep = sigma.expand(actions.shape[0])*1000.
            output = self.forward(dict(batch,noisy_actions=actions,future_latent=future,
                                       action_t=timestep,video_t=timestep),memory,belief,plan)
            actions = actions+(following-sigma).to(actions)*output["action_pred"].to(actions)
            future = future+(following-sigma).to(future)*output["video_pred"].to(future)
            if prefix is not None:
                actions = apply_prefix(actions,prefix)
        return dict(actions=actions,future_latent=future,belief_state=belief["tokens"])

    def forward_native(self,batch):
        """Structural native reference; immutable-teacher weights are caller-owned."""
        return self.backbone._forward_video_action(batch,include_video_prediction=True,
                                                   build_video_cache=False,detach_cache=False)

    def fast_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith("backbone.compact_wan.")}

    def make_fast_model(self):
        core = CausalSpatialActionCore(copy.deepcopy(self.backbone.action_expert),
            copy.deepcopy(self.backbone.absolute_motion_tokens),copy.deepcopy(self.backbone.config))
        fast = CausalSpatialModel(core,self.config,patch_size=self.patch_size)
        first = next(self.parameters())
        fast.to(device=first.device,dtype=first.dtype)
        fast.load_state_dict(self.fast_state_dict(),strict=True)
        return fast
