"""Fixed-duration C2 endpoint trajectories with continuous derivative bounds.

The endpoint velocity is the interval secant; endpoint acceleration is zero.
Bernstein derivative control points certify the entire curve, not sampled ticks.
The same causal projection is used by training and deployment. No retiming.
"""
import math
from functools import lru_cache
import numpy as np
import torch

ARMS=(0,1,2,3,4,5,7,8,9,10,11,12)
_A=np.array([0.,0.,0.,.6,.8,1.])
_B=np.array([0.,.2,.4,0.,0.,0.])


@lru_cache(maxsize=32)
def _coefficients(dt, velocity, acceleration, jerk):
    coefficients=[]
    for order,bound in enumerate((None,velocity,acceleration,jerk)):
        factor=math.factorial(5)/math.factorial(5-order)/dt**order
        for a,b in zip(np.diff(_A,n=order)*factor,np.diff(_B,n=order)*factor):
            coefficients.append((a,b,order==0,0. if bound is None else bound))
    return np.asarray(coefficients)


def displacement_interval(q, v, dt, lower, upper, limits):
    """Torch-batched feasible displacement interval for a1=a0=0."""
    if not math.isfinite(dt) or dt<=0 or any(not math.isfinite(float(limits[k])) or limits[k]<=0 for k in ('velocity','acceleration','jerk')):
        raise ValueError('positive finite duration and kinematic limits required')
    coeff=q.new_tensor(_coefficients(dt,limits['velocity'],limits['acceleration'],limits['jerk']))
    a,b,pos,bound=(x for x in coeff.T)
    offset=v[...,None]*dt*b+q[...,None]*pos
    low=torch.where(pos.bool(),lower[...,None],-bound)
    high=torch.where(pos.bool(),upper[...,None],bound)
    nonzero=a.abs()>1e-10
    divisor=torch.where(nonzero,a,1.)
    x=(low-offset)/divisor;y=(high-offset)/divisor
    lo=torch.where(nonzero,torch.minimum(x,y),-float('inf')).amax(-1)
    hi=torch.where(nonzero,torch.maximum(x,y),float('inf')).amin(-1)
    valid=(nonzero | ((offset>=low-1e-6)&(offset<=high+1e-6))).all(-1)
    return lo,hi,valid & (lo<=hi+1e-7)


def project_chunk(actions, known, known_mask, *, mean, scale, dt, bounds, limits, center=None, radius=None):
    """Project a normalized chunk, preserving >=2 immutable prefix endpoints.

    An optional trust box intersects feasibility around an already feasible slow
    plan. If any suffix row has an empty intersection, reject the correction for
    that sample and return the entire feasible slow suffix, never edit its prefix.
    """
    if actions.ndim!=3 or actions.shape[-1]!=14 or known_mask.shape!=actions.shape[:2]:
        raise ValueError('expected [B,H,14] and [B,H]')
    if not bool(known_mask[:,:2].all()) or bool((known_mask[:,1:] & ~known_mask[:,:-1]).any()):
        raise ValueError('C2 continuation requires two contiguous committed endpoints')
    if (center is None)!=(radius is None):raise ValueError('trust center/radius must be paired')
    raw=actions*scale+mean;fixed=known*scale+mean
    raw=torch.where(known_mask[...,None],fixed,raw)
    base=None if center is None else center*scale+mean
    arm=torch.tensor(ARMS,device=raw.device);lo=raw.new_tensor(bounds)[:,0];hi=raw.new_tensor(bounds)[:,1]
    rows=[raw[:,0],raw[:,1]];valid=torch.ones(raw.shape[0],dtype=torch.bool,device=raw.device)
    for i in range(2,raw.shape[1]):
        prev=rows[-1];velocity=(rows[-1]-rows[-2])/dt
        low,high,ok=displacement_interval(prev[:,arm],velocity[:,arm],dt,lo[arm],hi[arm],limits)
        low,high=prev[:,arm]+low,prev[:,arm]+high
        if base is not None:
            low=torch.maximum(low,base[:,i,arm]-radius[arm]);high=torch.minimum(high,base[:,i,arm]+radius[arm])
            ok=ok & (low<=high+1e-7)
        value=torch.maximum(torch.minimum(raw[:,i],hi),lo)
        target=torch.maximum(torch.minimum(value[:,arm],high),low)
        value=value.scatter(1,arm[None].expand(len(raw),-1),target)
        active=~known_mask[:,i]
        valid=valid & (~active | ok.all(-1))
        rows.append(torch.where(active[:,None],value,fixed[:,i]))
    result=(torch.stack(rows,1)-mean)/scale
    result=torch.where(known_mask[...,None],known,result)
    if center is not None:
        result=torch.where(valid[:,None,None],result,torch.where(known_mask[...,None],known,center))
    elif not bool(valid.all()):
        raise ValueError('committed C2 state has no feasible fixed-time continuation')
    return result,valid


def evaluate(q0, q1, v0, dt, elapsed):
    """Evaluate one certified arm segment in physical seconds; q/v/a are raw."""
    u=float(elapsed)/dt
    if not -1e-9<=u<=1+1e-9:raise ValueError('trajectory queried outside committed interval')
    u=min(1.,max(0.,u));q0=np.asarray(q0,np.float64);d=np.asarray(q1,np.float64)-q0;v0=np.asarray(v0,np.float64)
    control=q0[None]+_A[:,None]*d+_B[:,None]*v0*dt
    def at(points):
        points=points.copy()
        while len(points)>1:points=(1-u)*points[:-1]+u*points[1:]
        return points[0]
    return at(control),at(np.diff(control,axis=0)*5/dt),at(np.diff(control,n=2,axis=0)*20/dt**2)
