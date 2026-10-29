import copy
import json
from pathlib import Path
import tempfile
import unittest
from dualrate_response_acceptance import evaluate_reports, episode_response, sha, validate_for_formal


def fixture(task='task_a', episode=0, prefix=2):
    dt=.18;aw=100.
    bootstrap=dict(event='fast_committed',query_id=0,protected_until=.36,
                   commit_wall_time=99.,query_time=0.,reference_version=1)
    activation=dict(event='activation',wall_time=aw,physics_time=0.,held_prefix_seconds=.36,
                    immutable_wall_origin=True,inflight_catchup_hold=False,soft_horizon_reserve_hold=False)
    events=[bootstrap,activation]
    qid=0
    for i in range(10):
        q=i*dt
        if i in (2,4,6):
            qid+=1;end=(i+prefix)*dt
            event=dict(event='fast_started',query_id=qid,query_time=q,observation_time=q-dt,
                       request_wall_time=aw+q,observation_exposure_wall_time=aw+q-dt,
                       effective_protect_steps=prefix,protected_until=end,submit_margin=.01,
                       reference_version=1)
            events.append(event)
        events.append(dict(event='execute',row=i,start_time=q,end_time=(i+1)*dt,
                           actual_start=q,execution_wall_time=aw+q,target=[float(i)]*14))
        if i in (2,4,6):
            events.append(dict(event='fast_committed',query_id=qid,query_time=q,observation_time=q-dt,
                               protected_until=end,submit_margin=.01,reference_version=1,
                               commit_time=q+.10,commit_wall_time=aw+q+.10))
    return dict(task=task,episode_index=episode,events=events,action_interval_s=dt,
                complete=True,error=None,executable_action_source='fast_only',no_skipped_physics_steps=True,
                raw_clock_acceptance=dict(passed=True),started_targets=10,completed_targets=10,
                stop_reason='action_budget_exhausted',action_limit=10)


def reports(prefix=2):
    return [fixture(task,ep,prefix) for task in ('task_a','task_b') for ep in range(2)]


class ResponseGateTest(unittest.TestCase):
    def test_real_executed_response_recomputed(self):
        result=evaluate_reports(reports())
        self.assertTrue(result['passed'])
        self.assertEqual(result['two_prefix_fraction'],1.)
        for row in result['episodes']:
            self.assertAlmostEqual(row['observation_to_execution_wall_s']['median'],.54)
            self.assertGreater(row['wall_commit_deadline_margin_s']['min'],0.)

    def test_old_three_prefix_schedule_fails(self):
        result=evaluate_reports(reports(3))
        self.assertFalse(result['passed'])
        self.assertIn('two_prefix_fraction',result['failures'])

    def test_reported_fake_response_summary_is_not_trusted(self):
        data=reports(3)
        for r in data: r['action_response_events']=[dict(observation_to_execution_wall_seconds=.01)]
        self.assertFalse(evaluate_reports(data)['passed'])

    def test_missing_reference_rejected(self):
        r=fixture();next(e for e in r['events'] if e['event']=='fast_started')['reference_version']=-1
        with self.assertRaisesRegex(ValueError,'reference'): episode_response(r)

    def test_prefix_outside_training_rejected(self):
        r=fixture();next(e for e in r['events'] if e['event']=='fast_started')['effective_protect_steps']=4
        with self.assertRaisesRegex(ValueError,'prefix'): episode_response(r)

    def test_deadline_violation_rejected(self):
        r=fixture();e=[e for e in r['events'] if e['event']=='fast_committed'][1]
        e['commit_time']=e['protected_until']
        with self.assertRaisesRegex(ValueError,'deadline'):episode_response(r)

    def test_hidden_hold_and_skip_rejected(self):
        r=fixture();r['events'][1]['inflight_catchup_hold']=True
        with self.assertRaisesRegex(ValueError,'hold'):episode_response(r)
        r=fixture();next(e for e in r['events'] if e['event']=='execute')['row']=1
        with self.assertRaisesRegex(ValueError,'row'):episode_response(r)

    def test_policy_selected_identical_targets_are_not_falsely_rejected(self):
        r=fixture()
        for e in r['events']:
            if e['event']=='execute':e['target']=[0.]*14
        self.assertEqual(episode_response(r)['executed_rows'],10)

    def test_actual_four_reports_required(self):
        with self.assertRaisesRegex(ValueError,'four'):evaluate_reports(reports()[:3])
        data=reports();data[-1]=copy.deepcopy(data[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):evaluate_reports(data)

    def test_formal_binding_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);engineering=root/'engineering.json';engineering.write_text('{}')
            ev=[]
            for i,r in enumerate(reports()):
                path=root/f'{i}.json';path.write_text(json.dumps(r));ev.append(dict(path=str(path),sha256=sha(path)))
            response=evaluate_reports(reports());response.update(report_evidence=ev,
                engineering_receipt=str(engineering),engineering_receipt_sha256=sha(engineering))
            receipt=root/'accepted.json';source=dict(source_sha256='frozen')
            r=dict(passed=True,slow_process_exact_parity_and_clean_exit=True,evidence=[],source_contract=source,
                   response_acceptance=response,response_contract=dict(checkpoint_sha256='final',reserve=.3))
            receipt.write_text(json.dumps(r))
            config=dict(checkpoint_sha256='final',reserve=.3,stage='formal',tasks=['all'],episodes=20,
                        acceptance_receipt=str(receipt),acceptance_receipt_sha256=sha(receipt))
            self.assertTrue(validate_for_formal(config,source,receipt))
            with self.assertRaisesRegex(ValueError,'configuration'):
                validate_for_formal(dict(config,reserve=.4),source,receipt)
            Path(ev[0]['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'report changed'):validate_for_formal(config,source,receipt)

if __name__=='__main__':unittest.main()
