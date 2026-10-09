#!/usr/bin/env python3
"""CPU contracts for fixed targets, paired probe metrics and causal data."""
from __future__ import annotations
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('feedback_probe',ROOT/'scripts/probe_temporal_feedback.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
try:
    import torch
except ImportError:
    torch=None


def record(episode,identity=None):
    return dict(record=dict(split='train',task_name='task',episode_name=str(episode),
                            history_frame_offset=3,history_frame_index=0),
                record_id=str(episode if identity is None else identity))


class StatisticsContracts(unittest.TestCase):
    def test_episode_holdout_ignores_distinct_pair_ids(self):
        with self.assertRaisesRegex(ValueError,'leakage'):
            probe.assert_disjoint([record('same','frame1')],[record('same','frame99')])
        probe.assert_disjoint([record('a')],[record('b')])

    def test_episode_cluster_weighting_and_pair_identity(self):
        left=[dict(record_id='a',episode=['s','t','e1'],sse=1.,count=1),
              dict(record_id='b',episode=['s','t','e2'],sse=3.,count=3)]
        right=[dict(row,sse=row['sse']+2*row['count']) for row in left]
        result=probe.paired_episode_ci(left,right,repetitions=100)
        self.assertEqual(result['difference'],2.)
        self.assertEqual(result['ci95'],[2.,2.])
        right[1]['record_id']='other'
        with self.assertRaisesRegex(ValueError,'same records'):
            probe.paired_episode_ci(left,right)

    def test_gate_requires_gain_shuffle_zero_and_initial_not_significance(self):
        self.assertTrue(probe.gate_result(1.,.98,.99,1.,1.)['passed'])
        self.assertFalse(probe.gate_result(1.,.995,1.,1.,1.)['passed'])
        self.assertFalse(probe.gate_result(1.,.98,.97,1.,1.)['passed'])
        self.assertFalse(probe.gate_result(1.,.98,1.,.97,1.)['passed'])
        self.assertFalse(probe.gate_result(1.,.98,1.,1.,.97)['passed'])
        self.assertFalse(probe.gate_result(0.,0.,0.,0.,0.)['passed'])
        self.assertFalse(probe.gate_result(1.,float('nan'),1.,1.,1.)['passed'])

    def test_fixed_development_budget_and_checkpoint(self):
        args=SimpleNamespace(train_samples=512,validation_samples=256,teacher_batch_size=1,
            batch_size=32,epochs=20,lr=2e-5,seed=20261006,expert_weight=.25,residual_l2=.01,history_dropout=.15)
        probe.validate_fixed_budget(args)
        for key,value in [('lr',2e-4),('epochs',21),('seed',1),('train_samples',1024)]:
            altered=SimpleNamespace(**vars(args));setattr(altered,key,value)
            with self.assertRaisesRegex(ValueError,'budget mismatch'):
                probe.validate_fixed_budget(altered)
        with patch.object(probe,'sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'preregistered trained'):
                probe.initialization_file_identity('/not_read.pt')
        with patch.object(probe,'sha256',return_value=probe.EXPECTED_INITIALIZATION_SHA256):
            self.assertEqual(probe.initialization_file_identity('/not_read.pt')['sha256'],
                             probe.EXPECTED_INITIALIZATION_SHA256)

    def test_shuffle_is_heldout_valid_and_different_episode(self):
        rows=[record(i) for i in range(8)]
        valid=[True]*7+[False]
        indices,info=probe.history_permutation(rows,valid,seed=11)
        self.assertEqual(indices[7],7)
        self.assertEqual(sorted(indices[:7]),list(range(7)))
        self.assertEqual(info['changed_episode_fraction'],1.)
        self.assertTrue(all(i!=indices[i] for i in range(7)))
        self.assertEqual(indices,probe.history_permutation(rows,valid,seed=11)[0])
        with self.assertRaises(ValueError):probe.history_permutation([record(1)],[True],seed=1)

    def test_empty_metrics_and_dt_boundaries(self):
        self.assertIsNone(probe.summarize_metrics([])['target_mse'])
        self.assertEqual(probe.bin_name(.08,probe.DT_EDGES),'[0.08,0.16)')
        self.assertEqual(probe.bin_name(9.,probe.DT_EDGES),'[8,inf)')

    def test_previous_cache_cannot_become_a_v3_gate(self):
        contract=dict(history_training_raw_lags=list(probe.EXPECTED_HISTORY_LAGS),
                      history_sampling='pair_identity_sha256_v1',plan_age_seconds='telemetry_only',
                      architecture='pretrained_static_head_additive_zero_temporal_v3',
                      age_initialization='legacy_0_4_8_12_exact_linear_between_hold_after12')
        identity=probe.deployment_contract_identity(contract,checkpoint_format='temporal',checkpoint_version=3)
        self.assertEqual(identity['version'],3)
        with self.assertRaises(ValueError):
            probe.deployment_contract_identity(contract,checkpoint_format='temporal',checkpoint_version=1)
        with self.assertRaises(ValueError):
            probe.deployment_contract_identity(contract,checkpoint_format='temporal',checkpoint_version=2)
        with self.assertRaises(ValueError):
            probe.deployment_contract_identity(dict(contract,history_training_raw_lags=[3]),
                                                checkpoint_format='temporal',checkpoint_version=3)
        with self.assertRaises(ValueError):
            probe.deployment_contract_identity(dict(contract,plan_age_seconds=True),
                                                checkpoint_format='temporal',checkpoint_version=3)
        with self.assertRaises(ValueError):
            probe.deployment_contract_identity(dict(contract,architecture='old'),
                                                checkpoint_format='temporal',checkpoint_version=3)

    def test_multilag_coverage_reports_empty_cells_and_valid_dt(self):
        rows=[record('a'),record('b')]
        rows[0]['record'].update(history_frame_offset=0,history_frame_index=-1)
        rows[1]['record'].update(history_frame_offset=9,history_frame_index=21)
        rows[0]['inputs']=dict(age_actions=0,history_valid=False,history_dt=0.,plan_age_seconds=0.)
        rows[1]['inputs']=dict(age_actions=10,history_valid=True,history_dt=.54,plan_age_seconds=1.8)
        coverage=probe.cache_coverage(rows)
        self.assertEqual(coverage['history_raw_frame_lags']['0'],1)
        self.assertEqual(coverage['history_raw_frame_lags']['9'],1)
        self.assertEqual(coverage['history_raw_frame_lags']['36'],0)
        self.assertEqual(coverage['history_lag_by_age']['10']['9'],1)
        self.assertEqual(coverage['history_lag_by_task']['task']['9'],1)
        self.assertEqual(coverage['history_dt_valid_only']['min'],.54)


@unittest.skipIf(torch is None,'PyTorch is required for CPU tensor tests')
class TensorContracts(unittest.TestCase):
    def test_untrained_baseline_tensor_export_and_legacy_age_group(self):
        class Head(torch.nn.Module):
            residual_limit=.1
            def forward(self, **inputs):
                return inputs['plan'][:,:4].float()
        rows=[]
        for i,age in enumerate([0,4,5,12,15]):
            inputs=dict(rgb=torch.zeros(3,3,2,2),previous_head=torch.zeros(3,2,2),
                history_valid=torch.tensor(True),history_dt=torch.tensor(.18),
                plan_age_seconds=torch.tensor(age*.18),age_actions=torch.tensor(age),
                plan=torch.full((16,14),.01))
            rows.append(dict(record(i),inputs=inputs,text_id='text',
                             **{key:torch.zeros(4,14) for key in ('base','teacher','expert')}))
        cache=dict(validation=rows,texts={'text':dict(text=torch.zeros(1,8),text_mask=torch.ones(1,dtype=torch.bool))},
                   metadata=dict(initialization_checkpoint=dict(path='v1.pt',sha256='pinned')))
        args=SimpleNamespace(batch_size=2,expert_weight=.25)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'initial.pt'
            baseline=probe.evaluate(Head(),cache,args,torch.device('cpu'),history='fresh',prediction_path=path)
            zero=probe.evaluate(Head(),cache,args,torch.device('cpu'),history='fresh',zero=True)
            payload=torch.load(path,weights_only=True)
        self.assertEqual(payload['residual'].shape,(5,4,14))
        self.assertEqual(payload['ages'].tolist(),[0,4,5,12,15])
        self.assertTrue(torch.allclose(payload['residual'],torch.full((5,4,14),.01)))
        self.assertEqual(payload['record_ids'],[str(i) for i in range(5)])
        self.assertEqual(probe.summarize_metrics(zero)['target_mse'],0.)
        legacy=probe.stratified_metrics(baseline)['legacy_ages0_4_8_12']
        self.assertEqual(legacy['rows'],3)
        self.assertEqual(legacy['active_coordinates'],96.)
        self.assertAlmostEqual(legacy['target_mse'],.0001,places=8)

    def test_reachable_target_excludes_gripper_age0_and_tail(self):
        ages=torch.tensor([0,12,13,15])
        targets=dict(base=torch.zeros(4,4,14),teacher=torch.full((4,4,14),.3),
                     expert=torch.full((4,4,14),-.3))
        target,mask,_,_=probe.reachable_target(targets,ages,limit=.1,expert_weight=.25)
        self.assertTrue(torch.allclose(target,torch.full_like(target,.06)))
        self.assertEqual(mask.sum((1,2)).tolist(),[0,48,36,12])
        self.assertFalse(bool(mask[:,:,[6,13]].any()))
        residual=torch.zeros_like(target,requires_grad=True)
        ((residual-target).square()*mask).sum().backward()
        self.assertEqual(int(torch.count_nonzero(residual.grad)),96)
        self.assertEqual(float(residual.grad[0].abs().sum()),0.)
        self.assertEqual(float(residual.grad[3,1:].abs().sum()),0.)

    def test_cached_images_same_rounding_and_static_no_cache_mutation(self):
        inputs=dict(rgb=torch.randn(2,3,3,2,2),previous_head=torch.randn(2,3,2,2),
                    history_valid=torch.tensor([True,True]),history_dt=torch.tensor([.1,.2]),
                    text=torch.randn(2,3,8),text_mask=torch.tensor([[True,True,False],[True,True,True]]),
                    previous_head_mask=torch.ones(2,2,2,dtype=torch.bool),
                    plan=torch.randn(2,16,14),age_actions=torch.tensor([4,15]))
        table={};rows=[]
        for i in range(2):
            cached,text_id=probe.cache_inputs(inputs,i,table)
            rows.append(dict(record(i),inputs=cached,text_id=text_id,
                             **{key:torch.zeros(4,14) for key in ('base','teacher','expert')}))
        fresh,_=probe.collate_cache(rows,table,torch.device('cpu'))
        static,_=probe.collate_cache(rows,table,torch.device('cpu'),history='static')
        shuffled,_=probe.collate_cache(rows,table,torch.device('cpu'),history='shuffled',donors=rows[::-1])
        self.assertTrue(torch.equal(fresh['rgb'],static['rgb']))
        self.assertTrue(torch.equal(fresh['rgb'],inputs['rgb'].half().float()))
        self.assertTrue(torch.equal(fresh['plan'],inputs['plan']))
        self.assertTrue(bool(fresh['history_valid'].all()))
        self.assertFalse(bool(static['history_valid'].any()))
        self.assertTrue(bool(rows[0]['inputs']['history_valid']))
        self.assertTrue(torch.equal(shuffled['previous_head'][0],fresh['previous_head'][1]))
        self.assertTrue(torch.equal(shuffled['history_dt'],fresh['history_dt']))
        self.assertTrue(torch.equal(fresh['text_mask'],inputs['text_mask']))

    def test_active_coordinate_metrics_not_per_sample_mean(self):
        rows=[record(1),record(2)]
        for row in rows:row['inputs']={'history_valid':torch.tensor(True)}
        ages=torch.tensor([12,15])
        inputs=dict(age_actions=ages,history_valid=torch.tensor([False,False]),history_dt=torch.tensor([.1,.2]),
                    plan_age_seconds=torch.tensor([2.,3.]))
        targets={key:torch.zeros(2,4,14) for key in ('base','teacher','expert')}
        residual=torch.stack((torch.ones(4,14),torch.full((4,14),2.)))
        metrics=probe.metric_rows(residual,targets,inputs,rows,limit=.1,expert_weight=.25)
        summary=probe.summarize_metrics(metrics)
        self.assertEqual(summary['active_coordinates'],60.)
        self.assertEqual(summary['target_mse'],(48+4*12)/60)
        self.assertTrue(all(row['history_valid'] for row in metrics))

    def test_compact_all_ages_and_episode_partition(self):
        from dynamicwam.training.data.temporal_feedback_dataset import build_compact_records,TemporalPairSampler
        frames={e:{i:e*1000+i for i in range(100)} for e in (1,2,3,4)}
        episodes={e:dict(split='train',task_name='task',episode_name=f'e{e}',lang_group_id=1,lang_id=0)
                  for e in frames}
        assignment={1:'train',2:'train',3:'validation',4:'validation'}
        args=dict(splits=['train'],rate=3,totals={e:200 for e in frames})
        train,_=build_compact_records(frames,episodes,assignment,subset='train',**args)
        val,_=build_compact_records(frames,episodes,assignment,subset='validation',**args)
        self.assertEqual(train.rows.nbytes,len(train)*20)
        self.assertEqual(set(train.groups),{1,2});self.assertEqual(set(val.groups),{3,4})
        sampler=TemporalPairSampler(val,samples_per_episode=10,seed=3,validation=True)
        indices=list(sampler)
        self.assertEqual([val[i].offset//3 for i in indices[:16]],list(range(16)))
        self.assertEqual(indices,list(sampler))

    def test_previous_frame_and_real_clock_are_strictly_causal(self):
        import numpy as np
        from dynamicwam.training.data import temporal_feedback_dataset as module
        from dynamicwam.training.data.paired_world_action_dataset import PairRecord
        dataset=module.TemporalFeedbackDataset.__new__(module.TemporalFeedbackDataset)
        dataset.global_downsample_rate=3
        dataset.history_seed=20261006
        dataset.history_raw_frame_lags=module.HISTORY_RAW_FRAME_LAGS
        dataset.records=[PairRecord(1,0,3,0,9,1,0,9,'steady_state','train','task','e1')]
        clock={'frame_index':torch.arange(20),'sim_time_seconds':torch.arange(20).float().square()*.003}
        dataset._clock_for=lambda record:clock
        read=[]
        dataset._read_frame=lambda record:read.append((record.episode_index,record.fast_frame)) or np.zeros((2,2,3),np.uint8)
        base=dict(fast_rgb=torch.zeros(3,3,160,160),fast_rgb_valid_mask=torch.ones(3,160,160,dtype=torch.bool))
        views=np.ones((3,3,160,160),np.float32);masks=np.ones((3,160,160),bool)
        with patch.object(module.PairedWorldActionDataset,'__getitem__',return_value=base.copy()), \
             patch.object(module,'fast_views_from_composite',return_value=(views,masks)):
            sample=dataset[0]
        lag=sample['history_frame_offset'];previous=sample['history_frame_index']
        self.assertIn(lag,module.HISTORY_RAW_FRAME_LAGS)
        self.assertEqual(read,[(1,previous)])
        self.assertEqual(previous,9-lag)
        self.assertGreaterEqual(previous,0)
        self.assertLess(previous,9)
        self.assertTrue(sample['history_valid'])
        self.assertAlmostEqual(sample['history_dt'],float(clock['sim_time_seconds'][9]-clock['sim_time_seconds'][previous]),places=6)
        self.assertEqual(tuple(sample['previous_head'].shape),(3,160,160))
        dataset.records=[PairRecord(1,0,0,0,0,1,0,0,'steady_state','train','task','e1')]
        with patch.object(module.PairedWorldActionDataset,'__getitem__',return_value=base.copy()):sample=dataset[0]
        self.assertFalse(sample['history_valid']);self.assertEqual(sample['history_dt'],0.)
        self.assertFalse(bool(sample['previous_head_mask'].any()))
        self.assertEqual(sample['history_frame_index'],-1)
        self.assertEqual(sample['history_frame_offset'],0)

    def test_multilag_hash_repeats_and_never_crosses_episode_clock(self):
        from dynamicwam.training.data.temporal_feedback_dataset import causal_history_frame,HISTORY_RAW_FRAME_LAGS
        from dynamicwam.training.data.paired_world_action_dataset import PairRecord
        clock={'frame_index':torch.arange(100),'sim_time_seconds':torch.arange(100).double().square()*.003}
        records=[PairRecord(i+1,i,i+100,0,45,1,0,45,'steady_state','train','task',f'e{i}') for i in range(256)]
        first=[causal_history_frame(row,clock,seed=20261006) for row in records]
        second=[causal_history_frame(row,clock,seed=20261006) for row in records]
        changed=[causal_history_frame(row,clock,seed=20261007) for row in records]
        self.assertEqual(first,second)
        self.assertNotEqual(first,changed)
        self.assertEqual({lag for _,lag,_ in first},set(HISTORY_RAW_FRAME_LAGS))
        for previous,lag,dt in first:
            self.assertEqual(previous+lag,45)
            self.assertGreaterEqual(previous,0)
            self.assertAlmostEqual(dt,float(clock['sim_time_seconds'][45]-clock['sim_time_seconds'][previous]))
        cropped={'frame_index':torch.arange(44,60),'sim_time_seconds':clock['sim_time_seconds'][44:60]}
        self.assertEqual(causal_history_frame(records[0],cropped,seed=1)[0:2],(44,1))
        missing={'frame_index':clock['frame_index'][clock['frame_index']!=44],
                 'sim_time_seconds':clock['sim_time_seconds'][clock['frame_index']!=44]}
        with self.assertRaises(KeyError):causal_history_frame(records[0],missing,seed=1)

    def test_duplicate_sim_ticks_exclude_zero_dt_without_invented_time(self):
        from dynamicwam.training.data.temporal_feedback_dataset import causal_history_frame
        from dynamicwam.training.data.paired_world_action_dataset import PairRecord
        row=PairRecord(1,0,15,0,45,1,0,45,'steady_state','train','task','e1')
        times=torch.arange(100).double()*.1
        times[36:46]=times[36].clone()
        clock=dict(frame_index=torch.arange(100),sim_time_seconds=times)
        for seed in range(32):
            previous,lag,dt=causal_history_frame(row,clock,seed=seed)
            self.assertIn(lag,(12,18,24,36))
            self.assertGreater(dt,0.)
            self.assertAlmostEqual(dt,float(times[45]-times[previous]))
        flat=dict(frame_index=torch.arange(100),sim_time_seconds=torch.zeros(100))
        self.assertEqual(causal_history_frame(row,flat,seed=1),(-1,0,0.))
        corrupt=dict(frame_index=torch.arange(100),sim_time_seconds=times.clone())
        corrupt['sim_time_seconds'][44]=times[45]+.1
        with self.assertRaises(ValueError):causal_history_frame(row,corrupt,seed=1)


if __name__=='__main__':
    unittest.main(verbosity=2)
