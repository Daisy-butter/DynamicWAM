#!/usr/bin/env python3
"""CPU tests of the trainer's actual pure functions, without loading WAN.

AST extraction compiles unmodified function bodies from train_plan_feedback.py;
it only avoids that executable's optional CUDA-heavy imports.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import itertools
import os
from pathlib import Path
import unittest

import torch
from torch.utils.data import Sampler

trainer=Path(os.environ.get('PLAN_FEEDBACK_TRAINER',Path(__file__).with_name('train_plan_feedback.py')))
names={'RankSampler','shifted_action_noise','feedback_objective','runtime_contract','validate_initialization'}
tree=ast.parse(trainer.read_text())
scope=dict(torch=torch,itertools=itertools,Sampler=Sampler,
           FORMAT='dynamicwam_plan_feedback_checkpoint',VERSION=1,AGES=(0,4,8,12))
selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert {n.name for n in selected}==names
exec(compile(ast.Module(body=selected,type_ignores=[]),str(trainer),'exec'),scope)
globals().update({name:scope[name] for name in names})
torch.set_num_threads(1)

class BaseSampler:
    def __init__(self,size):self.size=size;self.epoch=0
    def __len__(self):return self.size
    def __iter__(self):return iter(range(self.epoch*self.size,(self.epoch+1)*self.size))
    def set_epoch(self,epoch):self.epoch=epoch

class TrainingChecks(unittest.TestCase):
    def test_shifted_noise_preserves_every_overlapping_coordinate(self):
        noise=torch.arange(4*16*14).float().reshape(4,16,14)
        before=noise.clone();ages=torch.tensor([0,4,8,12])
        result=shifted_action_noise(noise,ages,generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(noise,before))
        for b,age in enumerate(ages.tolist()):
            self.assertTrue(torch.equal(result[b,:16-age],noise[b,age:]))
            if age:
                self.assertTrue(torch.isfinite(result[b,16-age:]).all())
                self.assertFalse(torch.equal(result[b,16-age:],noise[b,16-age:]))

    def test_noise_tail_is_seeded_and_zero_age_is_exact_identity(self):
        source=torch.randn(2,16,14);ages=torch.tensor([8,12])
        a=shifted_action_noise(source,ages,generator=torch.Generator().manual_seed(11))
        b=shifted_action_noise(source,ages,generator=torch.Generator().manual_seed(11))
        c=shifted_action_noise(source,ages,generator=torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(a[:,8:],c[:,8:]))
        zero=shifted_action_noise(source,torch.zeros(2,dtype=torch.long))
        self.assertTrue(torch.equal(zero,source))

    def test_objective_excludes_age_zero_and_gripper_targets(self):
        residual=torch.zeros(2,4,14,requires_grad=True);base=torch.zeros_like(residual)
        teacher=torch.ones_like(residual)*100;expert=-teacher
        loss,metrics=feedback_objective(residual,base,teacher,expert,torch.tensor([0,4]),
            limit=.1,expert_weight=.25,l2_weight=0)
        self.assertAlmostEqual(float(loss),.0125,places=6)
        loss.backward()
        self.assertTrue(torch.equal(residual.grad[0],torch.zeros_like(residual.grad[0])))
        self.assertTrue(torch.equal(residual.grad[..., [6,13]],torch.zeros_like(residual.grad[..., [6,13]])))
        self.assertGreater(float(residual.grad[1].abs().sum()),0)
        self.assertEqual(float(metrics['teacher_clip_fraction']),1.)
        self.assertEqual(float(metrics['expert_clip_fraction']),1.)
        self.assertEqual(float(metrics['target_sign_conflict_fraction']),1.)

    def test_objective_clipped_optimum_and_regularization(self):
        residual=torch.full((1,4,14),.1,requires_grad=True);zero=torch.zeros_like(residual)
        teacher=torch.full_like(residual,5.)
        loss,metrics=feedback_objective(residual,zero,teacher,teacher,torch.tensor([12]),
            limit=.1,expert_weight=.25,l2_weight=0)
        self.assertEqual(float(loss),0.)
        self.assertLess(float(metrics['feedback_teacher_mse']),float(metrics['zero_teacher_mse']))
        regularized,_=feedback_objective(residual,zero,teacher,teacher,torch.tensor([12]),
            limit=.1,expert_weight=.25,l2_weight=.01)
        self.assertAlmostEqual(float(regularized),.0001,places=7)

    def test_age_zero_batch_has_finite_zero_loss_and_gradient(self):
        residual=torch.randn(2,4,14,requires_grad=True);base=torch.randn_like(residual)
        loss,_=feedback_objective(residual,base,base+100,base-100,torch.zeros(2,dtype=torch.long),
            limit=.1,expert_weight=.25,l2_weight=.01)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(loss),0.)
        loss.backward();self.assertTrue(torch.equal(residual.grad,torch.zeros_like(residual)))

    def test_rank_sampler_equal_length_disjoint_reconstructs_global_stream(self):
        world=7;size=world*4*3
        streams=[]
        for rank in range(world):
            sampler=RankSampler(BaseSampler(size),rank,world)
            self.assertEqual(len(sampler),12);streams.append(list(sampler))
        merged=[streams[rank][step] for step in range(12) for rank in range(world)]
        self.assertEqual(merged,list(range(size)))
        self.assertEqual(len(set(merged)),size)
        sampler.set_epoch(2)
        self.assertEqual(list(sampler),list(range(size*2+6,size*3,world)))
        with self.assertRaises(ValueError):RankSampler(BaseSampler(size-1),0,world)

    def test_export_contract_is_exactly_the_bounded_head_runtime(self):
        class Schedule:
            def to_config(self):return {'action_steps':10,'video_steps':2}
        c=runtime_contract(Schedule(),.1)
        self.assertEqual(c['source_plan'],'immutable_official_full_16')
        self.assertEqual((c['execute_horizon'],c['refresh_actions'],c['ages']),(4,16,[0,4,8,12]))
        self.assertFalse(c['age0_correction']);self.assertFalse(c['gripper_correction'])
        self.assertEqual(c['residual_units'],'normalized_qpos')
        self.assertEqual(c['precision']['action'],'float32')
        args=dict(base_identity={'path':'frozen.pt','sha256':'abc'},dataset_identity={'sha':'data'},
            pairing_identity={'sha':'pairs'},contract=c,spec={'dim':192})
        payload=dict(format=scope['FORMAT'],version=scope['VERSION'],base_checkpoint=args['base_identity'],
            dataset_identity=args['dataset_identity'],pairing_identity=args['pairing_identity'],
            runtime_contract=c,spec=args['spec'])
        validate_initialization(payload,**args)
        for key in payload:
            bad=copy.deepcopy(payload);bad[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):validate_initialization(bad,**args)

if __name__=='__main__':
    print('TRAINER_SHA256',hashlib.sha256(trainer.read_bytes()).hexdigest(),flush=True)
    unittest.main(verbosity=2)
