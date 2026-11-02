"""Endpoint/time preserving interpolation; no independent joint retiming/clipping.

All arm coordinates share the same monotone phase. This preserves each original
joint-space line segment, with zero velocity/acceleration at its boundaries.
It provides C2 continuity, NOT arbitrary global velocity/acceleration/jerk limits.
The simulator retains native collision, torque/drive and joint-limit behavior.
"""
import numpy as np
import torch


def phase(u):
    return u**3*(10+u*(-15+6*u))


def evaluate(q0,q1,dt,elapsed):
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive physical interval required')
    u=float(elapsed)/dt
    if not -1e-9<=u<=1+1e-9:raise ValueError('outside the committed physical interval')
    u=min(1.,max(0.,u));q0=np.asarray(q0,np.float64);d=np.asarray(q1,np.float64)-q0
    return q0+phase(u)*d,30*u*u*(1-u)**2*d/dt,60*u*(1-u)*(1-2*u)*d/dt**2


def bound_endpoints(action,known,known_mask,*,mean,scale,bounds,center=None,radius=None):
    """Only native joint limits and optional residual trust bounds; exact prefix."""
    if (center is None)!=(radius is None):raise ValueError('trust center/radius must be paired')
    raw=action*scale+mean;limits=raw.new_tensor(bounds)
    raw=torch.maximum(torch.minimum(raw,limits[:,1]),limits[:,0])
    if center is not None:
        base=center*scale+mean
        raw=torch.maximum(torch.minimum(raw,base+radius),base-radius)
    result=(raw-mean)/scale
    return torch.where(known_mask[...,None],known,result),torch.ones(len(action),dtype=torch.bool,device=action.device)


def measured_reference_discrepancy(batch,world):
    """Expected old drive command vs measured q, evaluated at the real exposure.

This is commanded-state discrepancy, not a claim that Full predicts actual q.
No privileged state, contact or object pose is consumed. Unsupported historical
intervals are explicitly masked, rather than assigning a fake zero error.
"""
    times=world['reference_action_intervals'];ref=world['reference_actions']
    now=batch['state_time'];actual=batch['state'];b=len(actual)
    if ref.shape[1]<2:
        return actual.new_zeros(b,29),torch.zeros(b,dtype=torch.bool,device=actual.device)
    matches=(now[:,None]>=times[...,0]-1e-8)&(now[:,None]<=times[...,1]+1e-8)
    # At a boundary use the completed left interval, matching post-step exposure.
    index=matches.long().argmax(-1);left=(index-1).clamp_min(0)
    take=lambda x,i:x.gather(1,i[:,None,None].expand(-1,1,x.shape[-1]))[:,0]
    span=take(times,index);previous=take(times,left)
    valid=matches.any(-1)&(index>0)&((previous[:,1]-span[:,0]).abs()<1e-7)
    q0=take(ref,left);q1=take(ref,index)
    u=((now-span[:,0])/(span[:,1]-span[:,0])).clamp(0,1).to(actual.dtype)
    expected=q0+phase(u[:,None])*(q1-q0)
    expected=expected.clone();expected[:,[6,13]]=q1[:,[6,13]]
    values=torch.cat((expected,actual-expected,valid[:,None].to(actual.dtype)),-1)
    return torch.where(valid[:,None],values,0.),valid
