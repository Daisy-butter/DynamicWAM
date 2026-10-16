"""CPU regression: physical alignment, unknown tails, hard prefixes, actual execution."""
from copy import deepcopy
import json
import unittest
import numpy as np
from dynamicwam.runtime.domino_continuous.plan_switch_diagnostics import diagnose_query,summarize_plan_switches


def fixture():
    offsets=np.arange(1,17)*.16
    row=dict(query_index=2,capture_time=3.,prefix_length=4,ready_physics_s=3.5,
             previous_endpoint_times=(3.+offsets).tolist(),result_received=True,installed=True)
    a=dict(returned_actions=np.zeros((16,14)),_rtc_action_target_offsets=offsets,
           _rtc_previous_actions=np.zeros((16,14)),_rtc_previous_valid=np.ones(16,bool),
           _rtc_prefix_actions=np.zeros((4,14)))
    return row,a


class PlanSwitchTests(unittest.TestCase):
    def test_unknown_tail_and_committed_prefix_not_switches(self):
        row,a=fixture();a['_rtc_previous_valid'][8:]=False;a['_rtc_previous_actions'][8:]=np.nan
        a['returned_actions'][4:8,6]=1.;a['returned_actions'][8:,6]=99.
        before=deepcopy(a);report=diagnose_query(row,a)
        self.assertEqual(report['future_overlap_rows'],4)
        self.assertEqual(report['gripper_state_disagreements'],[4,0])
        self.assertEqual(report['gripper_delta']['max_abs'],1.)
        for key in a:np.testing.assert_array_equal(a[key],before[key])
        json.dumps(report,allow_nan=False)

    def test_physical_endpoints_not_array_indices(self):
        row,a=fixture();row['prefix_length']=0;row['ready_physics_s']=3.
        row['previous_endpoint_times']=(3.+a['_rtc_action_target_offsets']+.16).tolist()
        a['_rtc_previous_actions'][:,0]=np.arange(16)
        a['returned_actions'][1:,0]=np.arange(15)
        report=diagnose_query(row,a)
        self.assertEqual(report['future_overlap_rows'],15)
        self.assertEqual(report['arm_delta']['max_abs'],0.)
        self.assertAlmostEqual(report['compared_endpoint_times'][0],3.32)

    def test_no_interpolation_or_stale_deadline_comparison(self):
        row,a=fixture();row['previous_endpoint_times']=(np.array(row['previous_endpoint_times'])+.01).tolist()
        self.assertEqual(diagnose_query(row,a)['future_overlap_rows'],0)
        row,a=fixture();row['ready_physics_s']=3.+.16*7
        self.assertEqual(diagnose_query(row,a)['future_overlap_rows'],9)

    def test_prefix_violation_measured_separately(self):
        row,a=fixture();a['returned_actions'][0,13]=.7
        report=diagnose_query(row,a)
        self.assertEqual(report['prefix_delta']['max_abs'],.7)
        self.assertEqual(report['gripper_state_disagreements'],[0,0])

    def test_no_transition_across_invalid_gap(self):
        row,a=fixture();a['_rtc_previous_valid'][5]=False
        a['returned_actions'][6:,6]=1.
        report=diagnose_query(row,a)
        self.assertEqual(report['proposal_gripper_transitions'],[0,0])
        a['_rtc_previous_valid'][5]=True
        self.assertEqual(diagnose_query(row,a)['proposal_gripper_transitions'],[1,0])

    def test_actual_execution_is_separate_from_proposal(self):
        row,a=fixture();a['returned_actions'][7:,6]=1.
        q=np.zeros(14);p=q.copy();p[13]=1.
        report=summarize_plan_switches([row],[a],[(0,1.,q),(1,1.16,p),(3,1.48,q)])
        self.assertEqual(report['queries'][0]['proposal_gripper_transitions'],[1,0])
        self.assertEqual(report['actual_completed_gripper_transitions'],[0,1])
        self.assertEqual(report['actual_adjacent_target_pairs'],1)
        self.assertTrue(report['measurement_only'])


if __name__=='__main__':unittest.main()
