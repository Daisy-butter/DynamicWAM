"""CPU regression of actual deployment input normalization, without model/GPU."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.runtime.ciwam.wam.flow_runner import HeadFlowRunner
from dynamicwam.runtime.ciwam.wam.multirate_policy import normalize_source_rtc_prefix
from dynamicwam.runtime.domino_continuous.motionkv_policy import MotionKVPolicy

class DeploymentNormalizationParity(unittest.TestCase):
    def setUp(self):
        self.normalizer=RoboTwinQposNormalizer(
            mean=torch.linspace(-.234567,1.067891,14),
            std=torch.linspace(.291123,1.004987,14),epsilon=1e-6)
        self.raw=torch.linspace(-.987654,2.123456,16*14).reshape(1,16,14)
        self.runner=object.__new__(HeadFlowRunner)
        model=SimpleNamespace(action_expert=torch.nn.Linear(14,14).to(torch.bfloat16),config=SimpleNamespace(action_dim=14))
        self.runner.runtime=SimpleNamespace(action_normalizer=self.normalizer,chunk_size=16,model=model,device='cpu')

    def expected(self,raw):
        return self.normalizer.normalize(raw.float()).to(torch.bfloat16)

    def test_real_state_and_native_hard_prefix_match_training_order(self):
        state=self.raw[:,0]
        torch.testing.assert_close(self.runner._normalize_state(state),self.expected(state),rtol=0,atol=0)
        prefix,noise,length=self.runner._prepare_hard_prefix(self.raw[0].numpy(),4)
        self.assertEqual(length,4);self.assertEqual(noise.dtype,torch.bfloat16)
        torch.testing.assert_close(prefix,self.expected(self.raw[:,:4]),rtol=0,atol=0)
        # This fixture reproduces the old raw-cast-before-normalize defect.
        self.assertGreater(float((self.normalizer.normalize(state.bfloat16()).float()-self.expected(state).float()).abs().max()),0)

    def test_source_prefix_and_bootstrap_are_immutable(self):
        raw=self.raw[0,:4].numpy().copy();original=raw.copy()
        reference=torch.zeros_like(self.raw,dtype=torch.bfloat16)
        prefix,lengths=normalize_source_rtc_prefix({'_rtc_prefix_length':4,'_rtc_prefix_actions':raw},self.normalizer,reference)
        torch.testing.assert_close(prefix[:,:4],self.expected(self.raw[:,:4]),rtol=0,atol=0)
        self.assertTrue(torch.equal(prefix[:,4:],reference[:,4:]));np.testing.assert_array_equal(raw,original)
        empty,d=normalize_source_rtc_prefix({'_rtc_prefix_length':0,'_rtc_prefix_actions':np.empty((0,14),np.float32)},self.normalizer,reference)
        self.assertEqual(int(d[0]),0);self.assertEqual(int(torch.count_nonzero(empty)),0)

    def test_actual_fast_packet_normalizes_state_prefix_and_queue_in_fp32(self):
        policy=object.__new__(MotionKVPolicy)
        policy.base=SimpleNamespace(runtime=self.runner.runtime,runner=self.runner,_state_tensor=lambda x:torch.as_tensor(x,dtype=torch.float32).reshape(1,-1))
        policy.bridge_sim_time=1.;policy.bridge_state=self.expected(self.raw[:,0]);policy.bridge=object()
        policy.fast_model=SimpleNamespace(fast_conditioner=SimpleNamespace(interface_version=3))
        captured={}
        def sample(current,bridge,text,mask,noise,*,prefix):
            captured.update(current=current,prefix=prefix)
            return noise.new_zeros(noise.shape)
        policy.fast_graph=SimpleNamespace(sample=sample)
        previous=self.raw[0].numpy().copy();valid=np.ones(16,dtype=bool);valid[-1]=False;previous[-1]=np.nan
        packet={'fast_frame':None,'state':self.raw[0,0].numpy(),'motion_features':np.linspace(.123456,.654321,48,dtype=np.float32).reshape(4,12),
            'motion_interval_valid_mask':np.ones(4,dtype=bool),'motion_acceleration_valid_mask':np.ones(4,dtype=bool),
            '_text':[torch.ones(2,8)],'sim_time':1.64,'_rtc_action_interval_s':.16,
            '_rtc_action_target_offsets':np.arange(1,17,dtype=np.float64)*.16,
            '_rtc_previous_actions':previous,'_rtc_previous_valid':valid}
        with patch('dynamicwam.runtime.domino_continuous.motionkv_policy.fast_views_from_composite',return_value=(np.zeros((3,3,2,2),np.float32),np.ones(3,bool))),patch('dynamicwam.runtime.domino_continuous.motionkv_policy.packet_noise_generator',return_value=None):
            policy._sample_rtc_fast(packet,4,self.raw[0,:4].numpy(),4)
        current=captured['current'];prefix=captured['prefix']
        torch.testing.assert_close(current.state,self.expected(self.raw[:,0]),rtol=0,atol=0)
        torch.testing.assert_close(prefix.actions[:,:4],self.expected(self.raw[:,:4]),rtol=0,atol=0)
        torch.testing.assert_close(current.previous_plan[:,:15],self.expected(self.raw[:,:15]),rtol=0,atol=0)
        self.assertFalse(bool(current.previous_plan_valid[0,-1]));self.assertTrue(bool(torch.isfinite(current.previous_plan).all()))
        # Numeric motion already enters in raw FP32; do not renormalize twice.
        self.assertEqual(current.motion.dtype,torch.float32)
        torch.testing.assert_close(current.motion,torch.from_numpy(packet['motion_features'])[None],rtol=0,atol=0)
        np.testing.assert_array_equal(previous[:15],self.raw[0,:15].numpy())

if __name__=='__main__':unittest.main()
