"""Separate a causally published candidate from an executable reference lease."""
import math
import torch


class ReferenceLease:
    def __init__(self,hold_steps,dt):
        if type(hold_steps) is not int or hold_steps<1 or not dt>0:
            raise ValueError('positive reference dwell on the physical grid required')
        self.duration=hold_steps*dt;self.dt=dt
        self.active=None;self.device=None;self.activated=None

    def usable(self,reference,now,first):
        if reference is None or not reference.completed_time<=now<reference.expires_at:return False
        p=reference.payload;n=p['reference_actions'].shape[1]
        intervals=p['reference_action_intervals'][0]
        valid=p['world_valid'][0,-n:]
        # Two genuinely predicted editable rows, never extrapolated repeated tail.
        targets=intervals.new_tensor([[first,first+self.dt],[first+self.dt,first+2*self.dt]])
        match=(targets[:,None]-intervals[None]).abs().amax(-1)<=1e-7
        return bool((match & valid[None]).any(-1).all())

    def select(self,latest,device,*,now,first):
        previous=self.active
        old_ok=self.usable(previous,now,first)
        new_ok=self.usable(latest,now,first)
        held=old_ok and now < self.activated+self.duration-1e-9
        if new_ok and (not held or previous is None):
            if previous is None or latest.version!=previous.version:
                self.active,self.device,self.activated=latest,device,float(now)
                reason='candidate_activated'
            else:reason='same_candidate'
        elif old_ok:reason='minimum_dwell' if held else 'candidate_unavailable'
        else:
            raise ValueError('no causal reference with two editable physical rows')
        return self.active,self.device,dict(event='reference_selected',query_time=float(now),
            candidate_version=latest.version if latest else -1,reference_version=self.active.version,
            activated_time=self.activated,hold_seconds=self.duration,first_editable_time=float(first),
            reason=reason,required_editable_rows=2)
