"""Read actual articulation q/qdot, distinct from commanded drive targets."""
import numpy as np


class TrackingAudit:
    def __init__(self,robot):
        self.parts=[]
        for side in ('left','right'):
            articulation=getattr(robot,side+'_entity')
            names=[j.get_name() for j in articulation.get_active_joints()]
            configured=list(getattr(robot,side+'_arm_joints_name'))
            if len(configured)!=6 or any(names.count(n)!=1 for n in configured):
                raise ValueError('actual articulation tracking joint mapping is ambiguous')
            self.parts.append((articulation,[names.index(n) for n in configured]))
        self.rows=[]

    def sample(self,when,command,command_velocity):
        q=np.concatenate([np.asarray(a.get_qpos(),np.float64)[i] for a,i in self.parts])
        v=np.concatenate([np.asarray(a.get_qvel(),np.float64)[i] for a,i in self.parts])
        idx=[0,1,2,3,4,5,7,8,9,10,11,12]
        cmd=np.asarray(command)[idx];cv=np.asarray(command_velocity)[idx]
        if not all(np.isfinite(x).all() for x in (q,v,cmd,cv)):raise ValueError('nonfinite articulation tracking sample')
        self.rows.append(dict(physical_time=float(when),actual_q=q.tolist(),actual_v=v.tolist(),
                             commanded_q=cmd.tolist(),commanded_v=cv.tolist()))

    def receipt(self):
        if not self.rows:return dict(samples=0,rows=[],actual_physical_tracking=True)
        q=np.array([r['actual_q'] for r in self.rows]);cmd=np.array([r['commanded_q'] for r in self.rows])
        v=np.array([r['actual_v'] for r in self.rows]);t=np.array([r['physical_time'] for r in self.rows])
        if len(t)>1 and (np.diff(t)<=0).any():raise ValueError('tracking clock must advance monotonically')
        return dict(samples=len(t),rows=self.rows,actual_physical_tracking=True,
            max_joint_position_error=float(np.abs(q-cmd).max()),rms_joint_position_error=float(np.square(q-cmd).mean()**.5),
            max_actual_joint_velocity=float(abs(v).max()),
            max_sampled_joint_acceleration=float(abs(np.diff(v,axis=0)/np.diff(t)[:,None]).max()) if len(t)>1 else 0.,
            limitation='sampled actual dynamics; analytic trajectory bounds apply to commands, not guaranteed tracking')
