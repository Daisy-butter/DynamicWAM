#!/usr/bin/env python3
"""Small CPU contracts for read-only shadow analysis; no real-run analysis."""
from pathlib import Path
import copy
import json
import pickle
import tempfile
import unittest
import analyze_feedback_shadow as a

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


def metric_row(episode, value, *, age=4, task='task'):
    return dict(record_id=str((episode,value)),episode=[task,episode],task=task,age=age,count=48,
        metrics={'zero':dict(sse=48.,raw_sse=96.,residual_abs_sum=0.),
                 'state_rule':dict(sse=48.*value,raw_sse=48.,residual_abs_sum=2.)})


class PureTests(unittest.TestCase):
    def test_missing_report_is_partial_without_failure_count(self):
        with tempfile.TemporaryDirectory() as directory:
            result=a.aggregate_status(directory,[1,3])
            self.assertEqual(result['status'],'partial')
            self.assertIsNone(result['success_count'])

    def test_full_report_and_seed_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'report.json'
            raw=[dict(seed=1,success=True),dict(seed=3,success=False)]
            report.write_text(json.dumps(raw))
            slot=dict(status='complete',return_code=0,report=str(report),requested_episodes=2,episodes=raw)
            payload=dict(requested_episodes=2,reported_episodes=2,episodes=raw,seed_sequence=[1,3],
                seeds_reported=True,episode_seed_manifest_matches=True,slots=[slot])
            path=root/'aggregate.json';path.write_text(json.dumps(payload))
            self.assertEqual(a.aggregate_status(root,[1,3])['success_count'],1)
            payload['seed_sequence']=[1,4];path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):a.aggregate_status(root,[1,3])
            payload['seed_sequence']=[1,3];path.write_text(json.dumps(payload))
            raw[0]['success']=False;report.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):a.aggregate_status(root,[1,3])

    def test_episode_cluster_not_query_independence(self):
        rows=[metric_row(0,.5)]*20+[metric_row(1,.8)]*3
        result=a.episode_ci(rows,'zero','state_rule',seed=4,repetitions=100)
        self.assertEqual(result['episodes'],2)
        self.assertAlmostEqual(result['difference'],(20*(-.5)+3*(-.2))/23)
        self.assertLess(result['ci95'][1],0)

    def test_common_task_age_support_is_explicit(self):
        online=[metric_row(0,.5,age=2),metric_row(1,.8,age=4)]
        offline=[metric_row(8,.4,age=4),metric_row(9,.1,age=8)]
        result=a.common_cells(online,offline)
        self.assertEqual([r['age'] for r in result['arms']['zero']['cells']],[4])
        self.assertIn('not paired or causal',result['interpretation'])


@unittest.skipIf(torch is None,'PyTorch unavailable')
class TensorTests(unittest.TestCase):
    def test_trusted_numpy_string_remains_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.pt'
            torch.save(dict(instruction=np.str_('rotate bottle'),state=torch.zeros(1,14)),path)
            value=a.trusted_shadow_load(path,a.sha256(path))
            self.assertEqual(str(value['instruction']),'rotate bottle')
            with self.assertRaises(ValueError):a.trusted_shadow_load(path,'0'*64)
            torch.save(dict(unapproved=Path('/not/allowed')),path)
            with self.assertRaises(pickle.UnpicklingError):a.trusted_shadow_load(path,a.sha256(path))

    def test_teacher_only_target_arm_tail_and_rule(self):
        ages=torch.tensor([0,4,14,15])
        plan=torch.zeros(4,16,14)
        state=torch.ones(4,14)*.04
        target,mask,delta,error,rule=a.target_and_rule(dict(plan=plan,state=state,age_actions=ages),
            torch.ones(4,4,14)*.3,.1)
        self.assertEqual(mask.flatten(1).sum(1).tolist(),[0,48,24,12])
        self.assertTrue(torch.all(target[mask]==.1))
        self.assertTrue(torch.all(rule[mask]==.04))
        self.assertTrue(torch.all(rule[~mask]==0))
        self.assertTrue(torch.all(delta==.3))
        # No expert field is supplied; the target cannot silently blend it.
        self.assertEqual(error.shape,(4,14))

    def test_donors_share_task_and_cross_episode(self):
        rows=[dict(task='a',episode=('a',i),inputs=dict(history_valid=torch.tensor(True))) for i in range(3)]
        rows.append(dict(task='b',episode=('b',0),inputs=dict(history_valid=torch.tensor(True))))
        donors=a.history_donors(rows,4)
        self.assertNotIn(3,donors)
        self.assertEqual(donors,a.history_donors(rows,4))
        for index,donor in donors.items():
            self.assertNotEqual(rows[index]['episode'],rows[donor]['episode'])
            self.assertEqual(rows[index]['task'],rows[donor]['task'])

    def test_record_manifest_clock_and_shape_guards(self):
        row=dict(format='dynamicwam_on_policy_shadow',version=1,task='a',seed=4,episode_index=0,
            command_index=4,age=4,base_checkpoint_sha256='base',schedule={},sim_time=2.,
            source_sim_time=0.,history_dt=.2,history_valid=True,plan=torch.zeros(1,16,14),
            teacher=torch.zeros(1,16,14),state=torch.zeros(1,14),source_state=torch.zeros(1,14),
            text=torch.ones(1,2,4096),text_mask=torch.ones(1,2,dtype=torch.bool))
        index=dict(episode=0,seed=4,command_index=4,age=4)
        args=dict(task='a',seeds=[4,8],base_sha='base',schedule={})
        a.validate_record(row,index,**args)
        for field,value in [('seed',8),('sim_time',-1),('history_dt',0),('teacher',torch.ones(1,4,14))]:
            altered=copy.copy(row);altered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):a.validate_record(altered,index,**args)


if __name__=='__main__':
    if torch is not None:torch.set_num_threads(1)
    unittest.main(verbosity=2)
