#!/usr/bin/env python3
"""CPU checks for the delayed-observation profiler fixture; no CUDA imports."""
from __future__ import annotations
import ast
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.runtime.domino_rt.timed_control import LimitedHermiteController,TimedPlanQueue
from dynamicwam.runtime.domino_rt.trajectory_recorder import controller_snapshot
from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path

SOURCE=ROOT/'scripts/profile_predictive_feedback.py'
tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='delayed_controller_fixture')
namespace=dict(np=np,LimitedHermiteController=LimitedHermiteController,TimedPlanQueue=TimedPlanQueue,
               controller_snapshot=controller_snapshot)
exec(compile(ast.Module(body=[definition],type_ignores=[]),str(SOURCE),'exec'),namespace)
fixture=namespace['delayed_controller_fixture']


class DelayedFixtureTests(unittest.TestCase):
    def setUp(self):
        self.initial=np.zeros(14)
        self.plan=np.arange(1,17)[:,None]*np.full((1,14),.02)
        self.snapshot=fixture(self.initial,self.plan)

    def test_observation_clock_and_dispatch_clock_are_not_relabelled(self):
        source=self.snapshot;dispatch=source['dispatch_controller']
        self.assertEqual(source['source_physics_time'],.4)
        self.assertAlmostEqual(dispatch['source_physics_time'],.564,places=14)
        self.assertEqual(source['ticks'],100);self.assertEqual(dispatch['ticks'],141)
        self.assertAlmostEqual(source['elapsed'],0.,places=14)
        self.assertAlmostEqual(dispatch['elapsed'],.064,places=14)
        self.assertFalse(np.array_equal(source['q'],dispatch['q']))

    def test_only_41_already_executed_commands_are_attached(self):
        prefix=self.snapshot['executed_prefix']
        self.assertEqual(set(prefix),{'start_sim_time','end_sim_time','q_command','velocity_command'})
        self.assertEqual(np.asarray(prefix['q_command']).shape,(41,14))
        np.testing.assert_allclose(prefix['start_sim_time'],np.arange(100,141)*.004,atol=0,rtol=0)
        np.testing.assert_allclose(prefix['end_sim_time'],np.arange(101,142)*.004,atol=0,rtol=0)
        self.assertLessEqual(max(prefix['end_sim_time']),self.snapshot['dispatch_controller']['source_physics_time'])
        np.testing.assert_array_equal(prefix['q_command'][-1],self.snapshot['dispatch_controller']['q'])
        np.testing.assert_array_equal(prefix['velocity_command'][-1],self.snapshot['dispatch_controller']['v'])

    def test_exact_original_controller_trace_with_boundary_transition(self):
        controller=LimitedHermiteController(self.initial,period=.1)
        commands=[];velocities=[]
        for tick in range(141):
            if tick%25==0:
                index=tick//25;controller.begin(self.plan[index],self.plan[index+1])
            q,v=controller.step(.004)
            if tick>=100:commands.append(q);velocities.append(v)
        np.testing.assert_array_equal(commands,self.snapshot['executed_prefix']['q_command'])
        np.testing.assert_array_equal(velocities,self.snapshot['executed_prefix']['velocity_command'])
        self.assertEqual(self.snapshot['future_knots'][0]['end_slot'],6)
        self.assertEqual(self.snapshot['dispatch_controller']['future_knots'][0]['end_slot'],7)

    def test_synthetic_fixture_does_not_mutate_plan_or_state(self):
        initial=self.initial.copy();plan=self.plan.copy()
        fixture(initial,plan)
        np.testing.assert_array_equal(initial,self.initial)
        np.testing.assert_array_equal(plan,self.plan)

    def test_real_helper_uses_dynamic_horizons_and_keeps_past_fixed(self):
        kwargs=dict(controller=self.snapshot,age=4,observation_time=.4,mean=np.zeros(14),scale=np.ones(14))
        path,dt,provenance=proposal_action_path(plan=self.plan,**kwargs)
        changed=self.plan.copy();changed[4:8]+=.03
        changed_path,changed_dt,_=proposal_action_path(plan=changed,**kwargs)
        self.assertEqual(path.shape,(2,4,28))
        np.testing.assert_allclose(dt,[.3,.5],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(dt,changed_dt)
        self.assertEqual(provenance['known_executed_prefix_ticks'],41)
        self.assertAlmostEqual(provenance['assumed_install_sim_time'],.6,places=14)
        self.assertFalse(provenance['actual_future_commands_used'])
        # The first .075/.125 s bins precede dispatch. Only the hypothetical
        # not-yet-executed suffix may change when a candidate is corrected.
        np.testing.assert_array_equal(path[:,0],changed_path[:,0])
        self.assertFalse(np.array_equal(path,changed_path))


if __name__=='__main__':unittest.main()
