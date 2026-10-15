import unittest
import tempfile,json
from pathlib import Path
import torch
from dynamicwam.training.data.rollout_replay import load_real_rollout_replay
from dynamicwam.training.data.physical_grid import sha256_file
import numpy as np
from dynamicwam.training.data.physical_grid import (target_grid,resample_commands,
    video_endpoints,validate_clock,sample_async_timing)
from dynamicwam.training.data.multirate_sampling import BalancedPairSampler

class PhysicalGridTest(unittest.TestCase):
    def test_fractional_phase_keeps_real_observation(self):
        endpoints=target_grid(10.07,10.)
        self.assertAlmostEqual(endpoints[0],10.16)
        np.testing.assert_allclose(np.diff(endpoints),.16)
        self.assertEqual(len(endpoints),16)
    def test_no_current_or_future_frame_relabelling(self):
        times=np.asarray([10.,10.06,10.12,10.18])
        idx,actual,lag=video_endpoints(times,np.array([10.16]))
        self.assertEqual(idx.item(),2);self.assertEqual(actual.item(),10.12)
        self.assertAlmostEqual(lag.item(),.04)
        with self.assertRaises(ValueError):video_endpoints(times,np.array([10.32]))
    def test_command_linear_gripper_previous_hold(self):
        commands=np.stack([np.zeros(14),np.ones(14)])
        output=resample_commands(np.array([0.,.2]),commands,np.array([.1,.2]))
        np.testing.assert_allclose(output[0,[0,1,7,8]],.5)
        np.testing.assert_array_equal(output[0,[6,13]],0.)
        np.testing.assert_array_equal(output[1],1.)
        with self.assertRaises(ValueError):resample_commands(np.array([0.,.2]),commands,np.array([.3]))
    def test_duplicate_clock_uses_last_command_at_endpoint(self):
        commands=np.stack([np.zeros(14),np.ones(14),np.ones(14)*2]);commands[:,[6,13]]=0
        out=resample_commands(np.array([0.,0.,.2]),commands,np.array([0.,.1]))
        self.assertEqual(out[0,0],1.);self.assertEqual(out[1,0],1.5)
    def test_invalid_clock_rejected(self):
        with self.assertRaises(ValueError):validate_clock([0,2],[0.,.1],np.zeros((2,14)))
        with self.assertRaises(ValueError):validate_clock([0,1],[.1,0.],np.zeros((2,14)))
    def test_publication_precedes_query_prefix_covers_inflight(self):
        endpoints=target_grid(12.07,10.)
        lengths=set()
        for epoch in range(100):
            t=sample_async_timing(10.,12.07,endpoints,seed=7,epoch=epoch,source_id=1,current_id=2)
            self.assertLessEqual(t['cache_publication_time'],t['query_time'])
            self.assertLessEqual(t['slow_latency_seconds'],2.56)
            length=t['rtc_prefix_lengths'];lengths.add(length)
            self.assertGreaterEqual(length,1);self.assertLessEqual(length,4)
            self.assertLess(t['ready_time'],endpoints[length-1])
            if length>1:self.assertGreaterEqual(t['ready_time'],endpoints[length-2])
        self.assertEqual(lengths,{1,2,3,4})
    def test_deterministic_epoch_timing(self):
        kw=dict(seed=1,epoch=3,source_id=5,current_id=7)
        self.assertEqual(sample_async_timing(0.,1.,target_grid(1.,0.),**kw),sample_async_timing(0.,1.,target_grid(1.,0.),**kw))
        with self.assertRaises(ValueError):sample_async_timing(1.,0.,target_grid(0.,0.),**kw)
    def test_source_prefix_is_sampled_on_its_own_grid(self):
        source=target_grid(0.,0.);current=target_grid(1.03,0.)
        seen=set()
        for epoch in range(200):
            timing=sample_async_timing(0.,1.03,current,source_action_times=source,
                seed=2,epoch=epoch,source_id=1,current_id=2)
            seen.add(timing['source_rtc_prefix_lengths'])
            self.assertGreaterEqual(timing['rtc_prefix_lengths'],1)
        self.assertEqual(seen,{0,1,2,3,4})

    def test_sampler_uses_compact_ranges_without_record_enumeration(self):
        class Records:
            sampling_groups={'task':{0:{('steady_state',0):range(0,20),('steady_state',8):range(20,40)}}}
            def __iter__(self):raise AssertionError('must not enumerate compact records')
        sampler=BalancedPairSampler(Records(),samples_per_episode=10,seed=4)
        result=list(sampler);self.assertEqual(len(result),10);self.assertTrue(all(0<=i<40 for i in result))

class RealReplayTest(unittest.TestCase):
    def fixture(self,root):
        identity=dict(trajectory_id='policy-rollout-1',episode_id='episode-1')
        obs=dict(**identity,origin='policy_execution_capture',policy_checkpoint_sha256='a'*64,
            capture_time=.04,rgb=torch.zeros(3,3,8,8),state=torch.zeros(14),native_motion=torch.zeros(4,12),
            motion_interval_valid_mask=torch.ones(4),motion_acceleration_valid_mask=torch.ones(4))
        artifacts={'source_observation':dict(obs,capture_time=0.),'current_observation':obs,
            'executed_timeline':dict(**identity,policy_checkpoint_sha256='a'*64,
                semantics='executed_drive_target_queue',endpoint_times=torch.tensor([.16,.32],dtype=torch.float64),
                actions=torch.zeros(2,14),committed_before_query=torch.tensor([True,False])),
            'suffix_labels':dict(**identity,origin='expert_relabel_of_policy_trajectory',
                action_target_times=torch.arange(1,17,dtype=torch.float64)*.16,action_sequence=torch.zeros(16,14))}
        m=dict(**identity,policy_checkpoint_sha256='a'*64,policy_source_sha256='b'*64,
            physical_grid_identity='c'*64,observation_source='policy_executed_same_trajectory')
        for key,value in artifacts.items():
            if key=='suffix_labels':value['observation_sha256']=m['current_observation']['sha256']
            path=root/(key+'.pt');torch.save(value,path);m[key]=dict(path=path.name,sha256=sha256_file(path))
        path=root/'replay.json';path.write_text(json.dumps(m));return path,m
    def test_valid_real_trace_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path,m=self.fixture(Path(d));sample=load_real_rollout_replay(path)
            self.assertEqual(sample['rtc_prefix_length'],1)
            self.assertEqual(sample['prefix_source'],'real_same_trajectory_policy')
    def test_counterfactual_expert_observation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path,m=self.fixture(Path(d));m['observation_source']='expert_observation_with_model_prefix';path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'counterfactual'):load_real_rollout_replay(path)
    def test_trajectory_substitution_rejected_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as d:
            path,m=self.fixture(Path(d));p=Path(d)/m['current_observation']['path']
            value=torch.load(p,weights_only=True);value['trajectory_id']='other-trajectory';torch.save(value,p)
            m['current_observation']['sha256']=sha256_file(p);path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'same actual trajectory'):load_real_rollout_replay(path)

if __name__=='__main__':unittest.main()
