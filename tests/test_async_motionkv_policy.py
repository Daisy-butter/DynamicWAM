"""CPU real-policy plumbing with controlled slow producer, no CUDA or simulator."""
from __future__ import annotations
import copy
import threading
import time
import unittest
from types import SimpleNamespace
import numpy as np
from dynamicwam.runtime.ciwam.wam.async_world_schedule import AsyncWorldScheduler
from dynamicwam.runtime.domino_continuous.motionkv_policy import MotionKVPolicy, validate_v3_packet_context


def packet(index, timestamp=None, prefix=4, **extra):
    result=dict(action_index=index,sim_time=index*.16 if timestamp is None else timestamp,
        instruction='test',state=np.arange(14,dtype=np.float32),
        _rtc_prefix_actions=np.zeros((prefix,14),np.float32),_rtc_prefix_length=prefix,
        _rtc_observation_action_index=index,_rtc_cache_age_actions=index%16,
        _rtc_action_interval_s=.16,_rtc_action_target_offsets=np.arange(1,17)*.16,
        _rtc_previous_actions=np.zeros((16,14),np.float32),_rtc_previous_valid=np.zeros(16,np.bool_))
    result.update(extra)
    return result


def value(p):
    return SimpleNamespace(source_visual_tokens=True),np.array(p['state'],copy=True),float(p['sim_time']),int(p['action_index'])


class FakePolicy(MotionKVPolicy):
    def __init__(self, build):
        self._queries=0;self._episode_seed=1;self._episode_index=0
        self.worker=None;self.bridge=None;self.bridge_action_index=0;self.bridge_sim_time=None
        self.base=SimpleNamespace(runtime=SimpleNamespace(action_interval_seconds=.16),
            runner=SimpleNamespace(device='cpu',_get_text_embeddings=lambda:[]),reset=lambda:None)
        self.fast_model=SimpleNamespace(fast_conditioner=SimpleNamespace(interface_version=3,prepare_source_cache=lambda b:b))
        self.fast_graph=SimpleNamespace(stats={})
        self.contract={'native_multirate':{'version':3,'source_bridge':'student_joint'}}
        self.last_policy_timing={};self.fast_calls=[]
        self._async_world=AsyncWorldScheduler(build,snapshot=copy.deepcopy)
    def prepare_instruction(self,instruction): pass
    def _sample_rtc_fast(self,p,age,raw,length):
        self.fast_calls.append((p['action_index'],self.bridge_action_index,age))
        actions=np.zeros((16,14),np.float32);actions[:length]=raw
        return actions,{'fast_sampler_ms':0.}


class AsyncPolicyTests(unittest.TestCase):
    def test_real_sample_progresses_while_slow_pending(self):
        started,release=threading.Event(),threading.Event()
        def build(p):
            if p['action_index']==16:
                started.set()
                if not release.wait(2):raise TimeoutError('test release')
            return value(p)
        policy=FakePolicy(build)
        try:
            policy.sample(packet(0,prefix=0))
            begin=time.monotonic()
            policy.sample(packet(16))
            self.assertLess(time.monotonic()-begin,.5)
            self.assertTrue(started.wait(1))
            for index in (20,24,28):policy.sample(packet(index))
            self.assertEqual(policy.fast_calls[-3:],[(20,0,20),(24,0,24),(28,0,28)])
            self.assertEqual(policy.last_policy_timing['cache_age_actions'],28)
            release.set();policy._async_world.worker.wait_for(policy._async_world.sequence,timeout=1)
            policy.sample(packet(28,timestamp=4.5))
            self.assertEqual(policy.fast_calls[-1],(28,16,12))
            self.assertTrue(policy.last_policy_timing['asynchronous_world']['published_new'])
        finally:release.set();policy.close()

    def test_prime_only_does_not_execute_fast_and_new_observation_uses_cache(self):
        policy=FakePolicy(value)
        try:
            hold=policy.sample(packet(0,prefix=0,_rtc_bootstrap_only=True))
            np.testing.assert_equal(hold,np.repeat(np.arange(14)[None],16,axis=0))
            self.assertFalse(policy.fast_calls)
            self.assertTrue(policy.last_policy_timing['bootstrap_only'])
            self.assertTrue(policy.last_policy_timing['asynchronous_world']['bootstrap_wait_included'])
            policy.sample(packet(0,timestamp=.4,prefix=4))
            self.assertEqual(policy.fast_calls,[(0,0,0)])
            self.assertAlmostEqual(policy.last_policy_timing['cache_age_seconds'],.4)
            with self.assertRaisesRegex(RuntimeError,'once'):
                policy.sample(packet(0,timestamp=.5,prefix=0,_rtc_bootstrap_only=True))
        finally:policy.close()

    def test_future_publication_and_expired_cache_are_rejected(self):
        schedule=AsyncWorldScheduler(value,snapshot=copy.deepcopy)
        try:
            schedule.bootstrap(packet(0,prefix=0))
            sequence=schedule.request(packet(20))
            schedule.worker.wait_for(sequence,timeout=1)
            chosen,meta=schedule.poll(packet(16))
            self.assertEqual(chosen.source_index,0)
            self.assertFalse(meta['published_new'])
            chosen,meta=schedule.poll(packet(20))
            self.assertEqual(chosen.source_index,20)
            with self.assertRaisesRegex(RuntimeError,'expired'):
                schedule.poll(packet(52))
            with self.assertRaisesRegex(RuntimeError,'expired'):
                schedule.poll(packet(24,timestamp=8.3))
        finally:self.assertTrue(schedule.close())

    def test_latest_pending_request_replaces_obsolete_and_snapshot_owned(self):
        started,release=threading.Event(),threading.Event();built=[]
        def build(p):
            built.append((p['action_index'],float(p['state'][0])))
            if p['action_index']==16:
                started.set();release.wait(2)
            return value(p)
        schedule=AsyncWorldScheduler(build,snapshot=copy.deepcopy)
        try:
            schedule.bootstrap(packet(0,prefix=0));schedule.request(packet(16));self.assertTrue(started.wait(1))
            schedule.request(packet(20));latest=packet(24);sequence=schedule.request(latest)
            latest['state'][0]=1000
            release.set();schedule.worker.wait_for(sequence,timeout=1)
            self.assertEqual([i for i,v in built],[0,16,24]);self.assertEqual(built[-1][1],0.)
        finally:release.set();self.assertTrue(schedule.close())

    def test_episode_reset_discards_inflight_and_shutdown_wait_is_bounded(self):
        started,release=threading.Event(),threading.Event()
        def build(p):
            if p['action_index']==16:started.set();release.wait(2)
            return value(p)
        schedule=AsyncWorldScheduler(build,snapshot=copy.deepcopy)
        try:
            schedule.bootstrap(packet(0,prefix=0));schedule.request(packet(16));self.assertTrue(started.wait(1))
            schedule.reset();release.set()
            fresh=schedule.bootstrap(packet(0,timestamp=100.,prefix=0))
            self.assertEqual(fresh.source_time,100.)
            self.assertEqual(schedule.worker.generation,1)
        finally:release.set();self.assertTrue(schedule.close())
        release.clear();started.clear()
        schedule=AsyncWorldScheduler(build,snapshot=copy.deepcopy)
        schedule.request(packet(16));self.assertTrue(started.wait(1))
        self.assertFalse(schedule.close(timeout=.01));release.set();self.assertTrue(schedule.close(timeout=1))

    def test_errors_propagate_to_fast_consumer(self):
        def build(p):
            if p['action_index']==16:raise ValueError('intentional producer error')
            return value(p)
        schedule=AsyncWorldScheduler(build,snapshot=copy.deepcopy)
        try:
            schedule.bootstrap(packet(0,prefix=0));sequence=schedule.request(packet(16))
            with self.assertRaisesRegex(RuntimeError,'producer failed'):
                schedule.worker.wait_for(sequence,timeout=1)
            with self.assertRaisesRegex(RuntimeError,'producer failed'):
                schedule.poll(packet(20))
        finally:self.assertTrue(schedule.close())

    def test_v3_exact_clock_and_previous_plan_inputs(self):
        p=packet(0,prefix=0);p['_rtc_action_target_offsets']-=.03
        offsets,previous,valid=validate_v3_packet_context(p)
        self.assertAlmostEqual(offsets[0],.13)
        p['_rtc_action_target_offsets'][0]=0
        with self.assertRaises(ValueError):validate_v3_packet_context(p)
        p=packet(0,prefix=0);p['_rtc_previous_actions'][0]=float('nan')
        validate_v3_packet_context(p) # invalid rows are masked before normalization
        p['_rtc_previous_valid'][0]=True
        with self.assertRaises(ValueError):validate_v3_packet_context(p)

if __name__=='__main__':unittest.main()
