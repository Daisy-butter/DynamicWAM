#!/usr/bin/env python3
"""Adapter contract test: previous action history, clock, copy and horizon."""
from __future__ import annotations
import sys
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.runtime.ciwam.adapters.domino import deploy_policy_temporal_feedback as adapter


def check(horizon):
    class Clock:
        time = 0.
        def snapshot(self):
            return SimpleNamespace(time_seconds=self.time)
    clock = Clock()
    durations = [.17,.33,.05,.61,.22,.41,.18,.27]*2
    executed, packets = [], []
    class Policy:
        plan = None
        age = 0
        runtime = SimpleNamespace(observation_config={'head_size':[240,320],'wrist_size':[120,160]})
        last_trace = {}
        def query(self, packet):
            packets.append({key:value.copy() if isinstance(value,np.ndarray) else value for key,value in packet.items()})
            if self.plan is None or self.age == 16:
                self.plan=np.arange(16*14,dtype=np.float32).reshape(16,14)
                self.age=0
            result=self.plan[self.age:self.age+horizon]
            self.age+=horizon
            return result
    class Stepper:
        def execute(self, action):
            clock.time += durations[len(executed)]
            executed.append(action.copy())
    class Flow:
        def push(self, *args, **kwargs):
            pass
        def observation(self):
            return SimpleNamespace(flow_rgb=np.zeros((4,32,32,3),np.uint8),motion_features=np.zeros((3,4)),
                interval_valid_mask=np.ones(3,bool),acceleration_valid_mask=np.ones(3,bool))
    model=SimpleNamespace(policy=Policy(),stepper=Stepper(),flow=Flow(),pending=deque(),index=0,query_ms=[],
        previous_wall=None,previous_sim=None,previous_head=None,previous_head_time=None)
    env=SimpleNamespace(_scene_step_clock=clock)
    head=np.zeros((240,320,3),np.uint8)
    obs={'observation':{'head_camera':{'rgb':head},'left_camera':{'rgb':head},'right_camera':{'rgb':head}},
         'joint_action':{'vector':np.zeros(14,np.float32)}}
    for action in range(16):
        head.fill(action)
        adapter.eval(env,model,obs)
    assert len(packets)==16//horizon
    assert not packets[0]['history_valid'] and packets[0]['history_dt']==0
    for query,packet in enumerate(packets[1:],1):
        row=query*horizon
        assert packet['history_valid'] and np.all(packet['previous_head']==row-1)
        assert abs(packet['history_dt']-durations[row-1])<1e-10
    np.testing.assert_array_equal(np.stack(executed),model.policy.plan)
    head.fill(255)
    assert np.all(model.previous_head==15), 'history aliases the mutable render buffer'


if __name__=='__main__':
    for horizon in (1,2,4):
        check(horizon)
    print('PASS previous-command history, physical clock, H1/H2/H4 and copy isolation')
