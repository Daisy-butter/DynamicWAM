#!/usr/bin/env python3
"""Numerical CPU tests of real time pairing, immutable labels and loss masks."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def extract(path,names,namespace):
    tree=ast.parse(path.read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


pure=extract(ROOT/'src/dynamicwam/runtime/domino_rt/proposal_path.py',{'aggregate_command_path','forecast_horizons'},
    dict(np=np,math=math,PATH_BINS=4,TIME_TOLERANCE=1e-8,COMPUTE_RESERVE_SECONDS=.02))
selection=extract(ROOT/'src/dynamicwam/training/data/predictive_feedback.py',{'select_future_frames'},
    dict(np=np,TIME_TOLERANCE=1e-8,MINIMUM_NOMINAL_HORIZONS=(.2,.4),WINDOW_OFFSETS=((-.1,.1),(-.1,.2))))
aggregate=pure['aggregate_command_path'];select=selection['select_future_frames']
forecast=pure['forecast_horizons']
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from dynamicwam.training.data.predictive_feedback import PredictiveFeedbackDataset,predictive_collate_fn
    from train_predictive_feedback import (teacher_objective,latent_objective,ExactEpochSampler,BalancedBehaviorSampler,
        dynamic_quartile_mask,configure_phase,predict,valid_tokens)
    from dynamicwam.models.predictive_feedback import PredictiveFeedbackHead


class AccumulationScopeTests(unittest.TestCase):
    def test_ddp_forward_backward_share_no_sync_but_not_amp(self):
        tree=ast.parse((ROOT/'scripts/train_predictive_feedback.py').read_text())
        scopes=[node for node in ast.walk(tree) if isinstance(node,ast.With) and
            any(isinstance(item.context_expr,ast.Name) and item.context_expr.id=='sync' for item in node.items)]
        self.assertEqual(len(scopes),1)
        sync=scopes[0]
        def calls(node,name):
            return [item for item in ast.walk(node) if isinstance(item,ast.Call) and
                ((isinstance(item.func,ast.Name) and item.func.id==name) or
                 (isinstance(item.func,ast.Attribute) and item.func.attr==name))]
        self.assertEqual(len(calls(sync,'wrapped')),1)
        self.assertEqual(len(calls(sync,'backward')),1)
        amps=[node for node in ast.walk(sync) if isinstance(node,ast.With) and
            any(isinstance(item.context_expr,ast.Call) and isinstance(item.context_expr.func,ast.Attribute)
                and item.context_expr.func.attr=='autocast' for item in node.items)]
        self.assertEqual(len(amps),1)
        self.assertEqual(len(calls(amps[0],'wrapped')),1)
        self.assertFalse(calls(amps[0],'backward'))


class PathTests(unittest.TestCase):
    def path(self):
        start=np.arange(100)*.004;end=start+.004
        q=np.repeat(np.arange(100)[:,None],14,1).astype(float)
        velocity=q*2
        return start,end,q,velocity
    def test_time_weighted_and_normalized_bins(self):
        s,e,q,v=self.path();out,valid=aggregate(s,e,q,v,0,.4,np.ones(14)*2,np.ones(14)*2)
        self.assertTrue(valid)
        np.testing.assert_allclose(out[:,0],(np.array([24,49,74,99])-2)/2)
        np.testing.assert_allclose(out[:,14],[12,37,62,87])
    def test_crossing_ticks_split_by_duration(self):
        s=np.array([0.,.003,.009]);e=np.array([.003,.009,.012])
        q=np.repeat(np.array([1.,2.,3.])[:,None],14,1);v=q.copy()
        out,valid=aggregate(s,e,q,v,0,.012,np.zeros(14),np.ones(14))
        self.assertTrue(valid);np.testing.assert_allclose(out[:,0],[1,2,2,3])
        np.testing.assert_allclose(out[:,14],[1,2,2,3])
    def test_real_gap_not_padded(self):
        s,e,q,v=self.path();keep=np.arange(100)!=50
        out,valid=aggregate(s[keep],e[keep],q[keep],v[keep],0,.4,np.zeros(14),np.ones(14))
        self.assertFalse(valid);self.assertFalse(out.any())
    def test_incomplete_future_not_clamped(self):
        s,e,q,v=self.path();_,valid=aggregate(s,e,q,v,0,.404,np.zeros(14),np.ones(14))
        self.assertFalse(valid)
    def test_overlap_and_nonfinite_fail(self):
        s,e,q,v=self.path();e[0]=.006
        with self.assertRaises(ValueError):aggregate(s,e,q,v,0,.4,np.zeros(14),np.ones(14))
        s,e,q,v=self.path();q[0,0]=np.nan
        with self.assertRaises(ValueError):aggregate(s,e,q,v,0,.4,np.zeros(14),np.ones(14))
    def test_actual_future_window_and_distinct_indices(self):
        np.testing.assert_array_equal(select([0,.1,.2,.3,.4,.6],0),[2,4])
        np.testing.assert_array_equal(select([0,.3,.6],0),[1,2])
        np.testing.assert_array_equal(select([0,.1,.2],2),[-1,-1])
    def test_duplicate_or_backward_clock_rejected(self):
        for clock in ([0,.2,.2],[0,.2,.1],[0,float('nan')]):
            with self.assertRaises(ValueError):select(clock,0)
    def test_real_camera_lag_moves_targets_after_install(self):
        desired=forecast(0.,.16)
        np.testing.assert_allclose(desired,[.3,.5])
        # The .2 frame was captured at installation and cannot label the new action.
        np.testing.assert_array_equal(select([0,.2,.4,.6],0,desired),[2,3])
        np.testing.assert_array_equal(select([0,.2],0,desired),[-1,-1])
        np.testing.assert_allclose(forecast(1.2,1.364),[.3,.5])
    def test_actual_future_time_is_not_desired_or_dispatch(self):
        times=np.array([0.,.16,.32,.48,.64]);desired=forecast(0.,.16)
        indices=select(times,0,desired)
        np.testing.assert_array_equal(indices,[2,3])
        np.testing.assert_allclose(times[indices],[.32,.48])
        self.assertFalse(np.array_equal(times[indices],desired))
    def test_invalid_dispatch_and_overlapping_horizons_refused(self):
        for source,dispatch in [(1.,.9),(0.,2.01),(0.,float('nan'))]:
            with self.assertRaises(ValueError):forecast(source,dispatch)
        for desired in [(.1,.3),(.3,.4),(.3,float('inf'))]:
            with self.assertRaises(ValueError):select([0,.2,.4,.6],0,desired)


@unittest.skipUnless(HAS_TORCH,'PyTorch numerical dataset/training tests require installed torch')
class TensorTests(unittest.TestCase):
    def test_teacher_mask_age_tail_and_gripper(self):
        inputs=dict(plan=torch.zeros(3,16,14),age_actions=torch.tensor([0,14,15]),proposal_dt=torch.tensor([[.2,.4]]*3))
        residual=torch.zeros(3,4,14,requires_grad=True);teacher=torch.ones_like(residual)
        loss,m=teacher_objective(residual,inputs,teacher,l2_weight=0)
        self.assertEqual(int(m['count']),12);loss.backward()
        self.assertFalse(bool(residual.grad[0].any()))
        self.assertFalse(bool(residual.grad[:,:,[6,13]].any()))
        self.assertFalse(bool(residual.grad[1,2:].any()))
        self.assertFalse(bool(residual.grad[2].any()))
        self.assertFalse(bool(residual.grad[:,0].any()))
        self.assertAlmostEqual(float(loss),.01,places=6)
    def test_policy_expired_prefix_has_no_gradient(self):
        inputs=dict(plan=torch.zeros(1,16,14),age_actions=torch.tensor([4]),proposal_dt=torch.tensor([[.3,.5]]))
        residual=torch.zeros(1,4,14,requires_grad=True)
        loss,m=teacher_objective(residual,inputs,torch.ones_like(residual),l2_weight=0)
        loss.backward();self.assertEqual(int(m['count']),24)
        self.assertEqual(int(m['expired_prefix_rows']),2)
        self.assertFalse(bool(residual.grad[:,:2].any()))
        self.assertTrue(bool(residual.grad[:,2:,:6].abs().min()>0))
    def test_latent_missing_horizons_have_no_gradient(self):
        prediction=torch.zeros(1,2,48,3,requires_grad=True);target=torch.ones_like(prediction)
        mask=torch.zeros(1,2,48,dtype=torch.bool);mask[:,0,:16]=True
        loss,m=latent_objective(prediction,target,mask);loss.backward()
        self.assertEqual(int(m['count']),48);self.assertFalse(bool(prediction.grad[:,1].any()))
    def test_dynamic_mask_keeps_each_view_and_excludes_zero(self):
        current=torch.zeros(1,48,3);target=torch.arange(48).float()[None,None,:,None].expand(1,2,48,3)
        mask=dynamic_quartile_mask(dict(target=target,current_tokens=current),torch.ones(1,2,48,dtype=torch.bool))
        self.assertEqual(mask.reshape(1,2,3,16).sum(-1).tolist(),[[[4,4,4],[4,4,4]]])
        zero=dynamic_quartile_mask(dict(target=torch.zeros_like(target),current_tokens=current),torch.ones_like(mask))
        self.assertFalse(bool(zero.any()))
    def test_rank_sampler_no_padding_duplicates(self):
        samplers=[ExactEpochSampler(67,rank,4,2,77) for rank in range(4)]
        first=[list(s) for s in samplers]
        self.assertEqual([len(s) for s in first],[16]*4)
        self.assertEqual(len(set(sum(first,[]))),64)
        for s in samplers:s.set_epoch(1)
        self.assertNotEqual(first,[list(s) for s in samplers])
    def test_global_coordinate_normalization(self):
        # Simulate DDP averaging over two unequal-coordinate microbatch ranks.
        w=torch.tensor(.3,requires_grad=True)
        nums=[(w-1).square()*3,(w+2).square()*7]
        loss=sum(2*n/10 for n in nums)/2;loss.backward();gradient=w.grad.clone()
        w2=torch.tensor(.3,requires_grad=True)
        reference=((w2-1).square()*3+(w2+2).square()*7)/10;reference.backward()
        torch.testing.assert_close(gradient,w2.grad)
    def test_refinement_batches_are_half_each_behavior(self):
        rows=[dict(behavior='graph_zero')]*9+[dict(behavior='predictive')]*4
        for rank in range(2):
            sampler=BalancedBehaviorSampler(rows,rank,2,4,19);values=list(sampler)
            for start in range(0,len(values),4):
                kinds=[rows[i]['behavior'] for i in values[start:start+4]]
                self.assertEqual(kinds.count('graph_zero'),2);self.assertEqual(kinds.count('predictive'),2)
    def test_world_warmup_releases_zero_output_gradient(self):
        torch.manual_seed(19);head=PredictiveFeedbackHead(dim=12,heads=3,layers=1)
        configure_phase(head,'world')
        encoded=dict(current_tokens=torch.randn(2,48,12),previous_head_tokens=torch.randn(2,16,12),
            previous_valid=torch.ones(2,16,dtype=torch.bool),history_dt=torch.full((2,),.2),
            history_valid=torch.ones(2,dtype=torch.bool),token_valid=torch.ones(2,48,dtype=torch.bool),
            actual_action_path=torch.randn(2,2,4,28),future_dt=torch.tensor([[.2,.4]]*2),
            future_valid=torch.ones(2,2,dtype=torch.bool),target=torch.randn(2,2,48,12),
            target_valid=torch.ones(2,2,48,dtype=torch.bool))
        opt=torch.optim.AdamW(head.latent_predictor.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad();loss,_=latent_objective(predict(head,encoded),encoded['target'],valid_tokens(encoded));loss.backward();opt.step()
        self.assertGreater(float(head.latent_predictor.action[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in head.vision.parameters()))
        self.assertTrue(all(not p.requires_grad for n,p in head.named_parameters() if not n.startswith('latent_predictor.')))


def identity(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@unittest.skipUnless(HAS_TORCH,'PyTorch dataset tests require installed torch')
class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.reference=dict(base_checkpoint=dict(sha256='a'*64),action_normalization=dict(
            stats=dict(mean=[0.]*14,std=[1.]*14),epsilon=1e-6))
        official=self.root/'official.json';official.write_text(json.dumps(dict(source_arm='official_sync_h16',
            tasks={f'task{i}':list(range(100000,100020)) for i in range(35)})))
        self.manifest=dict(format='dynamicwam_predictive_feedback_dataset',version=1,
            **self.reference,controller_protocol_sha256='b'*64,official_seed_manifest=identity(official),episodes=[])
        for seed,partition in [(1,'train'),(2,'validation')]:
            root=self.root/str(seed);root.mkdir();f=4;t=150;times=np.arange(f)*.2
            observations=dict(rgb=np.zeros((f,3,160,160,3),np.uint8),rgb_valid_mask=np.ones((f,3,160,160),bool),
                observation_sim_time=times,source_physics_time=times,capture_roundoff_s=np.zeros(f),received_sim_time=times+.1,
                dispatch_sim_time=times.copy(),policy_submitted=np.zeros(f,dtype=bool),
                drive_targets=np.zeros((f,14)),policy_state=np.zeros((f,14),np.float32),capture_id=np.arange(f),snapshot_sequence=np.arange(f))
            commands=dict(q_command=np.zeros((t,14)),velocity_command=np.ones((t,14)),
                start_sim_time=np.arange(t)*.004,end_sim_time=(np.arange(t)+1)*.004,
                engine_start_time=np.arange(t)*.004,engine_end_time=(np.arange(t)+1)*.004,tick_index=np.arange(t))
            np.savez(root/'observations.npz',**observations);np.savez(root/'commands.npz',**commands)
            (root/'events.jsonl').write_text('')
            dense=dict(format='dynamicwam_continuous_trajectory',version=1,status='complete',frames=f,ticks=t,
                physics_dt=.004,control_period=.1,view_order=['head','left_wrist','right_wrist'],
                metadata=dict(task='task0',seed=seed,realtime_valid=True,origin_sim_time=0.,physical_end_time=.6),
                provenance=dict(controller_protocol_sha256='b'*64,source_sha256={'collector':'c'*64}),
                artifacts={k:identity(root/v) for k,v in [('observations','observations.npz'),('commands','commands.npz'),('events','events.jsonl')]})
            (root/'manifest.json').write_text(json.dumps(dense))
            (root/'completion.json').write_text(json.dumps(dict(format='dynamicwam_continuous_trajectory_completion',
                version=1,status='complete',manifest=identity(root/'manifest.json'))))
            self.manifest['episodes'].append(dict(task='task0',seed=seed,partition=partition,trajectory_completion=identity(root/'completion.json')))
        self.path=self.root/'data.json';self.write()
    def tearDown(self):self.tmp.cleanup()
    def write(self):self.path.write_text(json.dumps(self.manifest))
    def dataset(self):return PredictiveFeedbackDataset(self.path,partition='train',reference_checkpoint=self.reference)
    def change_dispatch(self,*,capture=1,dispatch=.3,submitted=True):
        root=self.root/'1'
        with np.load(root/'observations.npz') as archive:obs={k:archive[k].copy() for k in archive.files}
        obs['dispatch_sim_time'][capture]=dispatch;obs['policy_submitted'][capture]=submitted
        np.savez(root/'observations.npz',**obs)
        dense=json.loads((root/'manifest.json').read_text());dense['artifacts']['observations']=identity(root/'observations.npz')
        (root/'manifest.json').write_text(json.dumps(dense))
        complete=json.loads((root/'completion.json').read_text());complete['manifest']=identity(root/'manifest.json')
        (root/'completion.json').write_text(json.dumps(complete))
        self.manifest['episodes'][0]['trajectory_completion']=identity(root/'completion.json');self.write()
    def test_exact_future_coverage_no_tail_padding(self):
        data=self.dataset();self.assertEqual(len(data),3)
        last=data[2];self.assertEqual(last['future_valid'].tolist(),[True,False])
        self.assertEqual(last['future_dt'][1].item(),0.)
        self.assertFalse(bool(last['future_rgb_valid_mask'][1].any()))
        batch=predictive_collate_fn([data[0],data[1]])
        self.assertEqual(tuple(batch['actual_action_path'].shape),(2,2,4,28))
        self.assertNotIn('policy_inputs',batch)
    def test_sha_change_refused(self):
        with (self.root/'1/commands.npz').open('ab') as f:f.write(b'altered')
        with self.assertRaisesRegex(ValueError,'changed immutable artifact'):self.dataset()
    def test_episode_leak_refused(self):
        self.manifest['episodes'][1]['seed']=1;self.write()
        with self.assertRaises(ValueError):self.dataset()
    def test_official_seed_refused(self):
        self.manifest['episodes'][0]['seed']=100000;self.write()
        with self.assertRaises(ValueError):self.dataset()
    def test_policy_without_labels_is_not_fake_training(self):
        with self.assertRaises(ValueError):
            PredictiveFeedbackDataset(self.path,partition='train',reference_checkpoint=self.reference,mode='policy')
    def test_unsubmitted_dispatch_cannot_pretend_real_latency(self):
        self.change_dispatch(submitted=False)
        with self.assertRaisesRegex(ValueError,'dispatch clock'):self.dataset()
    def test_submission_cannot_predate_received_camera(self):
        self.change_dispatch(dispatch=.25)
        with self.assertRaisesRegex(ValueError,'dispatch clock'):self.dataset()
    def test_dynamic_dataset_time_and_valid_masks(self):
        self.change_dispatch();data=self.dataset();row=next(r for r in data.records if r['current']==1)
        np.testing.assert_allclose(row['desired_future_dt'],[.3,.5])
        np.testing.assert_array_equal(row['future_valid'],[True,False])
        np.testing.assert_allclose(row['future_dt'],[.4,0.])
        self.assertTrue(row['policy_submitted'])
        self.assertFalse(data.records[0]['policy_submitted'])
    def add_policy(self,extra_input=False):
        from dynamicwam.training.data.predictive_feedback import _rgb_tensor
        self.change_dispatch()
        root=self.root/'1';current=_rgb_tensor(np.zeros((3,160,160,3),np.uint8))
        inputs=dict(rgb=current,rgb_valid_mask=torch.ones(3,160,160,dtype=torch.bool),
            previous_head=current[0],previous_head_mask=torch.ones(160,160,dtype=torch.bool),
            history_dt=torch.tensor(.2),history_valid=torch.tensor(True),state=torch.zeros(14),source_state=torch.zeros(14),
            plan=torch.zeros(16,14),age_actions=torch.tensor(2),plan_age_seconds=torch.tensor(.2),
            proposal_path=torch.ones(2,4,28),proposal_dt=torch.tensor([.3,.5]),
            text=torch.zeros(2,4096),text_mask=torch.ones(2,dtype=torch.bool))
        if extra_input:inputs['actual_future_path']=torch.ones(2,4,28)
        label=dict(format='dynamicwam_predictive_feedback_record',version=1,task='task0',seed=1,capture_id=1,
            observation_sim_time=.2,dispatch_sim_time=.3,base_checkpoint_sha256='a'*64,controller_protocol_sha256='b'*64,
            trajectory_manifest_sha256=identity(root/'manifest.json')['sha256'],inputs=inputs,teacher=torch.ones(16,14),
            proposal_provenance=dict(actual_future_commands_used=False),label_provenance=dict(
                teacher_only=True,current_observation=True,generation='offline_after_rollout',source_sha256={'a':'d'*64}))
        torch.save(label,root/'label.pt')
        (root/'index.jsonl').write_text(json.dumps(dict(**identity(root/'label.pt'),capture_id=1))+'\n')
        self.manifest['episodes'][0]['policy_index']=identity(root/'index.jsonl');self.write()
    def test_policy_targets_and_future_paths_stay_separate(self):
        self.add_policy();data=PredictiveFeedbackDataset(self.path,partition='train',reference_checkpoint=self.reference,mode='policy')
        row=data[0];self.assertIn('actual_action_path',row);self.assertNotIn('actual_action_path',row['policy_inputs'])
        self.assertEqual(tuple(row['teacher'].shape),(4,14))
        self.assertFalse(torch.equal(row['actual_action_path'],row['policy_inputs']['proposal_path']))
    def test_teacherforcing_extra_policy_input_refused(self):
        self.add_policy(extra_input=True)
        with self.assertRaisesRegex(ValueError,'deployable tensor contract'):self.dataset()
    def test_policy_uses_same_dispatch_horizons_as_world(self):
        self.add_policy();root=self.root/'1';label=torch.load(root/'label.pt',weights_only=True)
        label['inputs']['proposal_dt']=torch.tensor([.2,.4]);torch.save(label,root/'label.pt')
        (root/'index.jsonl').write_text(json.dumps(dict(**identity(root/'label.pt'),capture_id=1))+'\n')
        self.manifest['episodes'][0]['policy_index']=identity(root/'index.jsonl');self.write()
        with self.assertRaisesRegex(ValueError,'proposal horizons'):self.dataset()
    def test_all_expired_policy_record_does_not_remove_world_pair(self):
        self.add_policy();root=self.root/'1';label=torch.load(root/'label.pt',weights_only=True)
        label['inputs']['age_actions']=torch.tensor(15);torch.save(label,root/'label.pt')
        (root/'index.jsonl').write_text(json.dumps(dict(**identity(root/'label.pt'),capture_id=1))+'\n')
        self.manifest['episodes'][0]['policy_index']=identity(root/'index.jsonl');self.write()
        data=self.dataset();self.assertEqual(data.provenance['coverage']['train']['policy_no_live_rows'],1)
        self.assertTrue(any(r['current']==1 for r in data.records))
        with self.assertRaisesRegex(ValueError,'nonempty'):
            PredictiveFeedbackDataset(self.path,partition='train',reference_checkpoint=self.reference,mode='policy')


if __name__=='__main__':unittest.main(verbosity=2)
