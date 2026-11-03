"""Shared raw-unit sensor encoding. Missing measurement is explicitly masked."""
import math
import numpy as np
import torch
ARMS=(0,1,2,3,4,5,7,8,9,10,11,12)


def measured_features(q,v,command,mean,scale,age):
    if q is None and v is None:return torch.zeros(37),torch.tensor(False)
    if q is None or v is None or not math.isfinite(float(age)) or age<0:
        raise ValueError('paired measured q/qdot and causal physical age required')
    q=torch.as_tensor(np.array(q,copy=True),dtype=torch.float32)
    v=torch.as_tensor(np.array(v,copy=True),dtype=torch.float32)
    command=torch.as_tensor(np.array(command,copy=True),dtype=torch.float32)
    mean=torch.as_tensor(mean,dtype=torch.float32)[list(ARMS)]
    scale=torch.as_tensor(scale,dtype=torch.float32)[list(ARMS)]
    if q.shape!=(12,) or v.shape!=(12,) or command.shape!=(14,) or not (scale>0).all():
        raise ValueError('explicit native joint ordering required')
    result=torch.cat(((q-mean)/scale,(q-command[list(ARMS)])/scale,v/scale,torch.tensor([age],dtype=torch.float32)))
    if not torch.isfinite(result).all():raise ValueError('nonfinite measured feedback')
    return result,torch.tensor(True)
