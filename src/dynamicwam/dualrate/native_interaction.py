"""Measured native command/achieved history; no drive-target-as-qpos alias."""
from collections import deque
import numpy as np
import torch

ARM=(0,1,2,3,4,5,7,8,9,10,11,12)


def read_arm_state(robot):
    q=[];v=[]
    for side in ('left','right'):
        entity=getattr(robot,side+'_entity')
        joints=list(getattr(robot,side+'_arm_joints'))
        names=list(getattr(robot,side+'_arm_joints_name'))
        active=list(entity.get_active_joints())
        if len(joints)!=6 or [j.get_name() for j in joints]!=names:
            raise ValueError('unverified native arm joint order')
        pos=np.asarray(entity.get_qpos(),dtype=np.float64)
        vel=np.asarray(entity.get_qvel(),dtype=np.float64)
        ids=[active.index(j) for j in joints]
        q.extend(pos[ids]);v.extend(vel[ids])
    q=np.asarray(q);v=np.asarray(v)
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('nonfinite actual arm state')
    return q,v


class InteractionHistory:
    def __init__(self,mean,scale):
        self.mean=np.asarray(mean,dtype=np.float64);self.scale=np.asarray(scale,dtype=np.float64)
        if self.mean.shape!=(14,) or self.scale.shape!=(14,) or not (self.scale>0).all():raise ValueError('normalizer')
        self.rows=deque(maxlen=8);self.last_index=-1;self.end=0.

    def append(self,*,index,start,end,command,before,after,velocity,source_time):
        if index!=self.last_index+1 or start<self.end-1e-7 or end<=start or source_time>start:
            raise ValueError('interaction must be a completed consecutive native command')
        arm=np.asarray(ARM);mu=self.mean[arm];scale=self.scale[arm]
        command=np.asarray(command);before=np.asarray(before);after=np.asarray(after);velocity=np.asarray(velocity)
        if command.shape!=(14,) or any(x.shape!=(12,) for x in (before,after,velocity)):
            raise ValueError('actual twelve-arm state and fourteen-command required')
        features=np.concatenate(((command-self.mean)/self.scale,(before-mu)/scale,(after-mu)/scale,
            velocity/scale,(command[arm]-after)/scale,[np.log1p(end-start),np.log1p(start-source_time)]))
        if features.shape!=(64,) or not np.isfinite(features).all():raise ValueError('invalid measured tuple')
        self.rows.append((index,float(end),features.astype(np.float32)))
        self.last_index=index;self.end=end

    def tensors(self,query_time,query_index,device):
        if query_index!=self.last_index+1 or self.end>query_time+1e-7:
            raise ValueError('interaction query includes an unfinished/future command')
        x=np.zeros((8,64),np.float32);valid=np.zeros(8,bool)
        if self.rows:
            x[-len(self.rows):]=np.stack([r[2] for r in self.rows]);valid[-len(self.rows):]=True
        return dict(interaction_history=torch.from_numpy(x)[None].to(device),
                    interaction_valid=torch.from_numpy(valid)[None].to(device))
