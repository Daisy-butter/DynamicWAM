"""Finite-window spatial state and lightweight future blocks for causal WAM.

Command row i is recorded at the LEFT capture of the interval ending at image i.
It is NOT a measured joint position or the actuator history within that interval.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalSpatialConfig:
    z_dim: int = 768
    z_ffn_ratio: int = 2
    belief_dim: int = 192
    belief_grid: tuple[int, int] = (8, 8)
    views: int = 3
    latent_channels: int = 48
    history_length: int = 4
    gradient_checkpointing: bool = True
    attention_precision: str = "bfloat16"
    model_version: str = "causal_spatial_v4"
    command_history_kind: str = "recorded_capture_commands"

    def __post_init__(self):
        if min(self.z_dim, self.belief_dim, self.history_length, *self.belief_grid) <= 0:
            raise ValueError("causal spatial dimensions must be positive")
        if self.views != 3 or self.command_history_kind != "recorded_capture_commands":
            raise ValueError("V4 requires three views and explicit capture-command semantics")
        if self.attention_precision not in ("bfloat16","float16","float32"):
            raise ValueError("unsupported attention precision")


class SpatialBelief(nn.Module):
    """Predict before observing, then correct from a finite causal image window.

Supervision targets are pooled *observed RGB and flow velocity*, detached from
the model, not self-generated state. No current image enters its own prediction.
The finite window is replayed identically in training and deployment.
"""
    def __init__(self, config: CausalSpatialConfig, action_dim: int):
        super().__init__()
        self.config = config
        d = config.belief_dim
        self.encoder = nn.Sequential(nn.Conv2d(5, 64, 5, 2, 2), nn.GroupNorm(8, 64),
                                     nn.SiLU(), nn.Conv2d(64, d, 3, 2, 1), nn.SiLU())
        self.position = nn.Linear(3, d)
        self.command = nn.Sequential(nn.Linear(action_dim + 4, d), nn.SiLU(), nn.Linear(d, d))
        self.predict = nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, d))
        self.correct = nn.GRUCell(d, d)
        self.observation_head = nn.Linear(d, 10)
        self.output_norm = nn.LayerNorm(d)
        self.uncertainty = nn.Linear(5,d)
        gh, gw = config.belief_grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, gh), torch.linspace(-1, 1, gw), indexing="ij")
        coords = []
        for v in range(config.views):
            coords.append(torch.stack((x, y, torch.full_like(x, float(v)/(config.views-1))), -1))
        self.register_buffer("coordinates", torch.stack(coords).reshape(-1, 3), persistent=False)

    def forward(self, rgb, flow, valid, flow_valid, dt, commands, command_valid):
        if rgb.ndim != 6 or rgb.shape[2:4] != (self.config.views, 3):
            raise ValueError("spatial_rgb must be [B,T,3,3,H,W]")
        b, t, v, _, h, w = rgb.shape
        if t != self.config.history_length:
            raise ValueError("history length must match the exported finite-window contract")
        if flow.shape != (b,t,v,2,h,w) or valid.shape != (b,t,v,1,h,w) or flow_valid.shape != valid.shape:
            raise ValueError("spatial flow/valid geometry differs from RGB")
        if dt.shape != (b,t) or commands.shape[:2] != (b,t) or command_valid.shape != (b,t):
            raise ValueError("spatial times/recorded capture commands must align with the history")
        gh, gw = self.config.belief_grid
        n, d = v*gh*gw, self.config.belief_dim
        dtype = self.encoder[0].weight.dtype
        position = self.position(self.coordinates.to(dtype)).unsqueeze(0)
        state = torch.zeros(b,n,d,device=rgb.device,dtype=dtype)
        seen = torch.zeros(b,n,device=rgb.device,dtype=torch.bool)
        predictions, targets, logvars, masks = [], [], [], []
        for i in range(t):
            seconds = dt[:,i].float()
            temporal_valid = seconds > 0
            safe_dt = seconds.clamp_min(1e-5)
            previous_command = commands[:,i]
            previous_command_valid = command_valid[:,i].bool()
            features = torch.stack((safe_dt, torch.log1p(safe_dt),
                                    previous_command_valid.float(), temporal_valid.float()),-1)
            safe_command = torch.where(previous_command_valid[:,None], previous_command,
                                       torch.zeros_like(previous_command))
            drive = self.command(torch.cat((safe_command.float(),features),-1).to(dtype))[:,None].expand(-1,n,-1)
            predicted = state + self.predict(torch.cat((state,drive),-1))
            output = self.observation_head(predicted)
            predictions.append(output[...,:5])
            logvars.append(output[...,5:].clamp(-6,4))
            pixel_valid = valid[:,i].bool()
            # Flow is displacement in input-image pixels over dt, not a color map.
            velocity = flow[:,i].float() / safe_dt[:,None,None,None,None]
            velocity = velocity / velocity.new_tensor([w,h]).view(1,1,2,1,1)
            motion_valid = flow_valid[:,i].bool() & pixel_valid & temporal_valid[:,None,None,None,None]
            velocity = torch.where(motion_valid, velocity, 0.)
            observed = torch.cat((rgb[:,i].float(),velocity),2)
            observed = torch.where(pixel_valid, observed, torch.zeros_like(observed))
            flat = observed.reshape(b*v,5,h,w)
            mass = F.adaptive_avg_pool2d(pixel_valid.float().reshape(b*v,1,h,w),(gh,gw))
            flow_mass = F.adaptive_avg_pool2d(motion_valid.float().reshape(b*v,1,h,w),(gh,gw))
            component_mass = torch.cat((mass.expand(-1,3,-1,-1),flow_mass.expand(-1,2,-1,-1)),1)
            target = F.adaptive_avg_pool2d(flat,(gh,gw)) / component_mass.clamp_min(1e-6)
            targets.append(target.reshape(b,v,5,gh,gw).permute(0,1,3,4,2).reshape(b,n,5).detach())
            observed_valid = mass.reshape(b,n) > 0
            component_valid = (component_mass > 0).reshape(b,v,5,gh,gw).permute(0,1,3,4,2).reshape(b,n,5)
            masks.append(component_valid & temporal_valid[:,None,None] & (i > 0))
            encoded = self.encoder(flat.to(dtype))
            encoded = F.adaptive_avg_pool2d(encoded,(gh,gw)).reshape(b,v,d,gh,gw)
            encoded = encoded.permute(0,1,3,4,2).reshape(b,n,d) + position
            corrected = self.correct(encoded.reshape(b*n,d),predicted.reshape(b*n,d)).reshape(b,n,d)
            state = torch.where(observed_valid[...,None],corrected,predicted)
            seen = seen | observed_valid
        tokens = self.output_norm(state+self.uncertainty(logvars[-1].to(self.uncertainty.weight.dtype)))
        return dict(tokens=tokens, valid=seen,
                    prediction=torch.stack(predictions,1),target=torch.stack(targets,1),
                    logvar=torch.stack(logvars,1),prediction_valid=torch.stack(masks,1))


class FutureBlock(nn.Module):
    """768-dimensional residual stream; native 2048-wide joint attention ABI."""
    def __init__(self, dim, world_dim, heads, ratio):
        super().__init__()
        self.heads, self.head_dim = heads, world_dim//heads
        self.norm1 = nn.LayerNorm(dim,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim,elementwise_affine=False)
        self.qkv = nn.Linear(dim,3*world_dim)
        self.qnorm = nn.LayerNorm(self.head_dim)
        self.knorm = nn.LayerNorm(self.head_dim)
        self.output = nn.Linear(world_dim,dim)
        self.ffn = nn.Sequential(nn.Linear(dim,ratio*dim),nn.GELU(approximate="tanh"),nn.Linear(ratio*dim,dim))
        self.modulation = nn.Parameter(torch.zeros(1,6,dim))

    def modulation_for(self, time):
        return (self.modulation.unsqueeze(0)+time).unbind(2)

    def qkv_for(self, tokens, modulation):
        shift,scale,_,_,_,_ = modulation
        normalized = self.norm1(tokens.float())*(1+scale)+shift
        q,k,v = self.qkv(normalized.to(self.qkv.weight.dtype)).reshape(
            tokens.shape[0],tokens.shape[1],3,self.heads,self.head_dim).unbind(2)
        return self.qnorm(q),self.knorm(k),v

    def finish(self, tokens, attended, modulation):
        _,_,gate,shift,scale,ffn_gate = modulation
        tokens = tokens.float()+self.output(attended.flatten(2).to(self.output.weight.dtype)).float()*(1+gate)
        normalized = self.norm2(tokens)*(1+scale)+shift
        return tokens+self.ffn(normalized.to(self.ffn[0].weight.dtype)).float()*(1+ffn_gate)
