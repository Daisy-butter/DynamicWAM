#!/usr/bin/env python3
"""CPU regression: shadow labels do not alter native actions, age, or RNG."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
import io
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dynamicwam.runtime.ciwam.adapters.domino.deploy_policy_feedback_shadow import (
    ShadowFeedbackDeployModel, shifted_noise, cpu_copy)


class ShadowTests(unittest.TestCase):
    def test_shift_reuses_overlap_and_never_mutates_source(self):
        noise = torch.arange(224).reshape(1,16,14).float()
        original = noise.clone()
        for age in (1,2,4,8,14,15):
            result = shifted_noise(noise,age)
            self.assertTrue(torch.equal(result[:,:16-age],noise[:,age:]))
            self.assertEqual(result.shape,noise.shape)
        self.assertTrue(torch.equal(noise,original))
        with self.assertRaises(ValueError): shifted_noise(noise,16)

    def test_cpu_copy_owns_pixels_and_plan(self):
        values = {'pixels':np.zeros((3,4,3),np.uint8),'plan':torch.zeros(1,16,14)}
        copied = cpu_copy(values)
        values['pixels'].fill(1);values['plan'].add_(1)
        self.assertEqual(int(copied['pixels'].sum()),0)
        self.assertEqual(float(copied['plan'].sum()),0.)

    def test_numpy_metadata_becomes_native_weights_only_primitives(self):
        value=cpu_copy(dict(instruction=np.str_('pick object'),nested=[np.int64(4),np.float32(.5)]))
        self.assertIs(type(value['instruction']),str)
        stream=io.BytesIO();torch.save(value,stream);stream.seek(0)
        self.assertEqual(torch.load(stream,weights_only=True),dict(instruction='pick object',nested=[4,.5]))

    def fixture(self, directory, mutate=False):
        collector=ShadowFeedbackDeployModel.__new__(ShadowFeedbackDeployModel)
        runner=NS(device=torch.device('cpu'))
        runner._initialize_video_latent=lambda *args:(torch.randn(1,2,1,1,1),torch.zeros(1,2,1,1,1))
        runner._fresh_noisy_actions=lambda:torch.randn(1,16,14)
        runner._normalize_state=lambda value:value.float()
        runner._get_text_embeddings=lambda:[torch.ones(5,4096)]
        runner._motion_batch=lambda *args:dict(absolute_motion_features=torch.zeros(1,3,4),
            absolute_motion_interval_valid_mask=torch.ones(1,3,dtype=torch.bool),
            absolute_motion_acceleration_valid_mask=torch.ones(1,3,dtype=torch.bool))
        base=NS(runner=runner,_frame_tensor=lambda value:value,
            _state_tensor=lambda value:torch.tensor(value).reshape(1,14))
        cfg={'head_size':[240,320],'wrist_size':[120,160]}
        normal=NS(denormalize=lambda value:value)
        policy=NS(base=base,plan=None,age=0,source_state=torch.zeros(1,14),source_sim_time=0.,
            runtime=NS(observation_config=cfg,action_normalizer=normal),base_sha256='a'*64,
            text=torch.ones(1,5,4096),text_mask=torch.ones(1,5,dtype=torch.bool))
        def sample(source,noise):
            collector.teacher_noise=noise
            if mutate:policy.plan.add_(1)
            return noise.actions.clone(),{}
        policy.sampler=NS(sample=sample,schedule=NS(to_config=lambda:dict(action_steps=10,video_steps=2)))
        def query(packet):
            if policy.plan is None:
                runner._initialize_video_latent(None,None)
                runner._fresh_noisy_actions()
                policy.plan=torch.arange(224).float().reshape(1,16,14)
            result=policy.plan[:,policy.age:policy.age+2].squeeze(0).numpy().copy()
            policy.age+=2
            return result
        collector.policy=policy;collector.original_query=query
        collector.flow=NS(observation=lambda:NS(flow_rgb=np.zeros((4,32,32,3),np.uint8),
            motion_features=np.zeros((3,4)),interval_valid_mask=np.ones(3,bool),
            acceleration_valid_mask=np.ones(3,bool)))
        collector.environment=NS(ep_num=0,get_instruction=lambda:'test instruction',
            _scene_step_clock=NS(snapshot=lambda:NS(time_seconds=1.)))
        camera={'rgb':np.zeros((240,320,3),np.uint8)}
        collector.observation={'observation':{name:camera for name in ('head_camera','left_camera','right_camera')}}
        collector.fixed_seeds=[700000];collector.task='test';collector.index=0
        collector.max_samples=1;collector.output_root=Path(directory)
        collector.samples_by_episode={};collector.source_noise=None
        packet=dict(state=np.full(14,.3,np.float32),sim_time=1.,history_dt=.2,history_valid=True,
            fast_frame=np.zeros((360,320,3),np.uint8),previous_head=np.zeros((240,320,3),np.uint8))
        return collector,packet

    def test_query_preserves_reference_rng_and_actual_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            model,packet=self.fixture(directory)
            model.shadow_query(packet)
            saved=model.policy.plan.clone();source_noise=model.source_noise.actions.clone()
            model.index=2
            rng=torch.get_rng_state().clone()
            result=model.shadow_query(packet)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(saved,model.policy.plan))
            np.testing.assert_array_equal(result,saved[0,2:4].numpy())
            self.assertTrue(torch.equal(model.teacher_noise.actions[:,:14],source_noise[:,2:]))
            record=torch.load(Path(directory)/'episode000_query0000.pt',weights_only=True)
            self.assertEqual(record['age'],2);self.assertEqual(record['seed'],700000)
            self.assertEqual(record['command_index'],2)
            self.assertTrue(torch.equal(record['state'],torch.full((1,14),.3)))
            model.index=4;model.shadow_query(packet)
            self.assertEqual(len(list(Path(directory).glob('*.pt'))),1)

    def test_teacher_plan_mutation_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            model,packet=self.fixture(directory,mutate=True)
            model.shadow_query(packet)
            with self.assertRaisesRegex(RuntimeError,'mutated'):
                model.shadow_query(packet)
            self.assertEqual(len(list(Path(directory).glob('*.pt'))),0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
