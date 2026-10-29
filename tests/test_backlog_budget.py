import copy
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1];WORK=HERE.parent
sys.path[:0]=[str(HERE/'scripts'),str(WORK/'dualrate_continuous_repair_20261028/scripts'),
              str(WORK/'cohort_repair_20261029/scripts')]
from dualrate_eval_protocol import validate_backlog_contract,BASE_PROTOCOL,BACKLOG500_PROTOCOL
from dualrate_continuous_acceptance import validate_continuous_report,validate_reference_leases
from dualrate_response_acceptance import DEFAULT_CRITERIA

def clock_report(lag):
    evidence=dict(passed=True,checks={'immutable':True},activation_wall_time=100.,
        activation_physics_time=0.,final_wall_time=101.+lag,final_physics_time=1.)
    return dict(raw_clock_acceptance=evidence,physics_step_seconds=.004,physics_steps=250,
        active_wall_seconds=1.+lag,physical_duration=1.,events=[
            dict(event='activation',wall_time=100.,physics_time=0.),dict(event='terminal_clock',**evidence)])

class BudgetTest(unittest.TestCase):
    def config(self):
        return dict(max_backlog=.5,protocol=BACKLOG500_PROTOCOL,max_observation_age=.24,
            action_dt=.18000000854954124,backlog_scheduling_tolerance=.005,
            wall_scheduling_tolerance=0.,physical_scheduling_tolerance=0.,
            benchmark_contract=dict(max_backlog=.5,max_observation_age=.24,simulation_rate=1.,
                                    evaluation_setting='continuous1x_backlog500ms_v1'))
    def test_budget_protocol_metadata_and_retention_binding(self):
        c=self.config();self.assertEqual(validate_backlog_contract(c),BACKLOG500_PROTOCOL)
        for changed in (dict(c,protocol=BASE_PROTOCOL),dict(c,retained_task_receipt='old80'),
                        dict(c,max_observation_age=.5),dict(c,wall_scheduling_tolerance=.25)):
            with self.assertRaises(ValueError):validate_backlog_contract(changed)
        changed=copy.deepcopy(c);changed['benchmark_contract']['max_backlog']=.25
        with self.assertRaisesRegex(ValueError,'metadata'):validate_backlog_contract(changed)
    def test_nominal_budget_keeps_original5ms_tolerance(self):
        old=dict(max_backlog=.25,backlog_scheduling_tolerance=.005)
        new=dict(max_backlog=.5,backlog_scheduling_tolerance=.005)
        with self.assertRaises(ValueError):validate_continuous_report(clock_report(.3),old)
        self.assertTrue(validate_continuous_report(clock_report(.3),new)['passed'])
        with self.assertRaises(ValueError):validate_continuous_report(clock_report(.506),new)
    def test_late_wall_or_physical_commit_still_rejected(self):
        activation=dict(event='activation',wall_time=100.,physics_time=0.)
        start=dict(event='fast_started',query_id=1,reference_version=1,query_time=.36,source_time=0.,
            reference_age=.36,observation_time=.18,request_wall_time=100.36,protected_until=.72,submit_margin=.01)
        commit=dict(start,event='fast_committed',ready_wall_time=100.46,commit_wall_time=100.48,commit_time=.48)
        events=[dict(event='slow_completed',accepted=True,version=1,source_time=0.,completed_time=0.),activation,start,commit]
        self.assertEqual(validate_reference_leases(events,self.config(),activation)['active_commits'],1)
        for key,value in [('commit_wall_time',100.73),('commit_time',.73)]:
            altered=copy.deepcopy(events);altered[-1][key]=value
            with self.assertRaisesRegex(ValueError,'deadline'):validate_reference_leases(altered,self.config(),activation)
    def test_response_gate_not_relaxed(self):
        self.assertEqual(DEFAULT_CRITERIA,dict(min_two_prefix_fraction=.90,
            max_episode_observation_response_median_s=.60,max_episode_observation_response_p95_s=.65))

if __name__=='__main__':unittest.main()
