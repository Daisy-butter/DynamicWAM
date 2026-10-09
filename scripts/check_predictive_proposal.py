#!/usr/bin/env python3
"""CPU numerical checks of causal path construction against the real controller."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.runtime.domino_rt.timed_control import LimitedHermiteController,TimedPlanQueue
from dynamicwam.runtime.domino_rt.trajectory_recorder import controller_snapshot
from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path,aggregate_command_path,forecast_horizons


class Tests(unittest.TestCase):
    def setUp(self):
        self.mean=np.linspace(-.2,.2,14);self.scale=np.linspace(.5,1.5,14)
        self.plan=np.random.default_rng(42).normal(0,.2,(16,14)).astype(np.float32)
        self.controller=LimitedHermiteController(np.zeros(14))
        self.controller.begin(np.linspace(0,.1,14),np.linspace(0,.2,14))
        self.snapshot=controller_snapshot(self.controller,source_physics_time=.4)

    def call(self,plan=None):
        return proposal_action_path(controller=self.snapshot,plan=self.plan if plan is None else plan,
            age=4,observation_time=.4,mean=self.mean,scale=self.scale)

    def test_exact_real_queue_and_controller(self):
        path,dt,meta=self.call()
        control=copy.deepcopy(self.controller);queue=TimedPlanQueue(.1)
        physical=self.plan.astype(np.float64)*self.scale+self.mean
        queue.stage(dict(actions=physical[4:],observation_sim_time=.4,request_id=3,
            metadata={'action_origin_sim_time':.4}),now=.401)
        start=[];end=[];q=[];v=[]
        for index in range(100):
            now=.4+index*.004
            if index and index%25==0:
                slot=round(now/.1);queue.publish(slot);item=queue.pop(slot+1)
                control.begin(None if item is None else item[0],queue.peek(slot+2))
            qi,vi=control.step(.004)
            start.append(now);end.append(now+.004);q.append(qi);v.append(vi)
        for i,horizon in enumerate((.2,.4)):
            expected,ok=aggregate_command_path(start,end,q,v,.4,.4+horizon,self.mean,self.scale)
            self.assertTrue(ok);np.testing.assert_array_equal(path[i],expected)
        self.assertFalse(meta['actual_future_commands_used'])

    def test_committed_prefix_does_not_depend_on_proposal(self):
        a=self.call()[0];changed=self.plan.copy();changed+=.5;b=self.call(changed)[0]
        np.testing.assert_array_equal(a[0,:2],b[0,:2])
        self.assertGreater(np.abs(a-b).max(),0)

    def test_snapshot_and_plan_not_mutated(self):
        old=json.dumps(self.snapshot,sort_keys=True);p=self.plan.copy();self.call()
        self.assertEqual(old,json.dumps(self.snapshot,sort_keys=True));np.testing.assert_array_equal(p,self.plan)

    def test_random_limited_controller_equivalence(self):
        rng=np.random.default_rng(713)
        for trial in range(24):
            self.controller=LimitedHermiteController(rng.normal(0,.3,14))
            self.controller.v=rng.uniform(-.7,.7,14)
            self.controller.begin(rng.normal(0,.7,14),rng.normal(0,.7,14))
            self.snapshot=controller_snapshot(self.controller,source_physics_time=.4)
            self.plan=rng.normal(0,.8,(16,14)).astype(np.float32)
            self.test_exact_real_queue_and_controller()

    def test_unaligned_committed_segment_rejected(self):
        self.snapshot['elapsed']=.004
        with self.assertRaises(ValueError):self.call()

    def test_delayed_camera_prefix_and_future_queue_equivalence(self):
        for lag in (.044,.164,.196,.284):
            source=.4;age=4;dispatch=source+lag
            control=copy.deepcopy(self.controller);queue=TimedPlanQueue(.1)
            for i in range(5,16):queue.knots[i+1]=(self.plan[i]*self.scale+self.mean,{})
            source_snapshot=controller_snapshot(control,source_physics_time=source,
                future_knots=[dict(end_slot=k,q=v[0]) for k,v in queue.knots.items()])
            starts=[];ends=[];commands=[];velocities=[]
            def tick(index):
                now=source+index*.004
                if index and index%25==0:
                    slot=round(now/.1);queue.publish(slot);item=queue.pop(slot+1)
                    control.begin(None if item is None else item[0],queue.peek(slot+2))
                q,v=control.step(.004)
                starts.append(now);ends.append(now+.004);commands.append(q);velocities.append(v)
            prefix_ticks=round(lag/.004)
            for i in range(prefix_ticks):tick(i)
            source_snapshot['dispatch_controller']=controller_snapshot(control,source_physics_time=dispatch,
                future_knots=[dict(end_slot=k,q=v[0]) for k,v in queue.knots.items()])
            source_snapshot['executed_prefix']=dict(start_sim_time=list(starts),end_sim_time=list(ends),
                q_command=np.asarray(commands).copy(),velocity_command=np.asarray(velocities).copy())
            candidate=self.plan+.13
            path,dt,meta=proposal_action_path(controller=source_snapshot,plan=candidate,age=age,
                observation_time=source,mean=self.mean,scale=self.scale)
            queue.stage(dict(actions=(candidate.astype(np.float64)*self.scale+self.mean)[age:],
                observation_sim_time=source,request_id=4,metadata={'action_origin_sim_time':source}),now=dispatch+.02)
            horizons=forecast_horizons(source,dispatch)
            for i in range(prefix_ticks,round(horizons[-1]/.004)):tick(i)
            for h,d in enumerate(horizons):
                expected,ok=aggregate_command_path(starts,ends,commands,velocities,source,source+d,self.mean,self.scale)
                self.assertTrue(ok);np.testing.assert_array_equal(path[h],expected)
            self.assertGreater(source+float(dt[0]),dispatch+.02)
            self.assertEqual(meta['known_executed_prefix_ticks'],prefix_ticks)
            invalid=copy.deepcopy(source_snapshot)
            invalid['executed_prefix']['end_sim_time'][-1]+=.004
            with self.assertRaises(ValueError):
                proposal_action_path(controller=invalid,plan=candidate,age=age,
                    observation_time=source,mean=self.mean,scale=self.scale)

    def test_physical_clock_required(self):
        self.snapshot['source_physics_time']=.5
        with self.assertRaises(ValueError):self.call()

    def test_gap_invalid_not_padded(self):
        q=np.zeros((2,14));x,ok=aggregate_command_path([0,.008],[.004,.012],q,q,0,.012,self.mean,self.scale)
        self.assertFalse(ok)

    def test_velocity_is_time_weighted_and_q_is_endpoint(self):
        q=np.tile(np.arange(8)[:,None],(1,14));v=q*2
        x,ok=aggregate_command_path(np.arange(8)*.004,(np.arange(8)+1)*.004,q,v,0,.032,np.zeros(14),np.ones(14))
        self.assertTrue(ok)
        np.testing.assert_array_equal(x[:,:14],q[[1,3,5,7]])
        np.testing.assert_array_equal(x[:,14:],np.tile(np.array([1,5,9,13])[:,None],(1,14)))

if __name__=='__main__':unittest.main()
