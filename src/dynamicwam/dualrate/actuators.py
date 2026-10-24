"""Explicit native actuator bounds; immutable queue prefixes are never edited."""
from __future__ import annotations
import numpy as np
import torch


class ActuatorProjection:
    def __init__(self, bounds, *, provenance):
        self.bounds=torch.as_tensor(bounds,dtype=torch.float32).clone()
        if self.bounds.shape!=(14,2) or torch.isnan(self.bounds).any() or (self.bounds[:,1]<=self.bounds[:,0]).any():
            raise ValueError('ordered actual fourteen-actuator bounds required; native infinities allowed')
        if not torch.equal(self.bounds[[6,13]],torch.tensor([[0.,1.],[0.,1.]])):
            raise ValueError('native gripper commands are normalized to [0,1]')
        self.provenance=provenance

    @property
    def serialized_bounds(self):
        # JSON has no infinity. Null denotes an unbounded native side, not a
        # guessed angle wrap or planner constraint.
        return [[float(x) if np.isfinite(x) else None for x in row] for row in self.bounds.tolist()]

    def audit_training_range(self, minimum, maximum):
        lo=torch.as_tensor(minimum,dtype=torch.float32);hi=torch.as_tensor(maximum,dtype=torch.float32)
        if lo.shape!=(14,) or hi.shape!=(14,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (lo>hi).any():
            raise ValueError('finite complete raw expert action extrema required')
        violations=((lo<self.bounds[:,0]-1e-6)|(hi>self.bounds[:,1]+1e-6)).nonzero().flatten().tolist()
        return dict(all_training_extrema_within_native_limits=not violations,violating_coordinates=violations,
                    native_bounds=self.serialized_bounds,training_min=lo.tolist(),training_max=hi.tolist())

    @classmethod
    def from_robot(cls, robot):
        bounds=[];sources=[]
        for side in ('left','right'):
            joints=list(getattr(robot,f'{side}_arm_joints'))
            expected=list(getattr(robot,f'{side}_arm_joints_name'))
            names=[joint.get_name() for joint in joints]
            if len(joints)!=6 or len(set(names))!=6 or names!=expected:
                raise ValueError(f'{side} configured actuator/articulation joint mapping unverified')
            side_limits=[];semantics=[]
            for joint in joints:
                actual=np.asarray(joint.get_limits(),dtype=np.float64)
                if actual.shape!=(1,2) or np.isnan(actual).any() or actual[0,1]<=actual[0,0]:
                    raise ValueError('one-DoF native articulation limits required')
                side_limits.append(actual[0].tolist())
                semantics.append('unbounded_native_joint_no_angle_wrap' if np.isinf(actual).all()
                                 else 'native_articulation_joint_limits_no_angle_wrap')
            bounds.extend(side_limits);bounds.append([0.,1.])
            sources.append(dict(side=side,joint_names=names,
                limits=[[float(x) if np.isfinite(x) else None for x in row] for row in side_limits],
                semantics=semantics,source='actual_SAPIEN_joint_get_limits_in_set_arm_joints_order'))
        return cls(bounds,provenance=sources)

    def __call__(self, raw, known_mask):
        if raw.device.type!='cpu' or raw.ndim!=2 or raw.shape[1]!=14 or not torch.isfinite(raw).all():
            raise ValueError('finite CPU raw actuator targets required')
        if known_mask.shape!=raw.shape[:1] or known_mask.dtype!=torch.bool:
            raise ValueError('one hard-prefix mask per action row required')
        clipped=torch.maximum(torch.minimum(raw,self.bounds[:,1]),self.bounds[:,0])
        result=torch.where(known_mask[:,None],raw,clipped)
        delta=result-raw;changed=delta!=0
        rows=changed.any(-1).nonzero().flatten()
        receipt=dict(changed_coordinates=int(changed.sum()),changed_rows=rows.tolist(),
            max_absolute_change=float(delta.abs().max()) if delta.numel() else 0.,
            hard_prefix_preserved=torch.equal(result[known_mask],raw[known_mask]),
            before=raw[rows].tolist(),after=result[rows].tolist(),
            forecast_invalidated=bool(changed.any()),reason='native_actuator_projection' if changed.any() else None)
        return result,receipt
