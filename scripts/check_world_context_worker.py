#!/usr/bin/env python3
"""CPU test the real packet-to-noise method without importing CUDA deployment."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'src/dynamicwam/runtime/world_context_worker.py'


def source_method():
    tree=ast.parse(WORKER.read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='WorldContextWorker')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_source')
    namespace=dict(torch=torch,SourceObservation=SimpleNamespace,SourceNoise=SimpleNamespace)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(WORKER),'exec'),namespace)
    return namespace['_source']


class Runner:
    device='cpu'
    def _normalize_state(self,x):return x
    def _get_text_embeddings(self):return torch.zeros(1,2,3)
    def _initialize_video_latent(self,frame,flow_frames,*,generator):
        return torch.randn(1,16,generator=generator),frame
    def _fresh_noisy_actions(self,*,generator):return torch.randn(1,16,14,generator=generator)
    def _motion_batch(self,features,interval,acceleration):
        return dict(absolute_motion_features=features,absolute_motion_interval_valid_mask=interval,
                    absolute_motion_acceleration_valid_mask=acceleration)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.worker=SimpleNamespace(base=SimpleNamespace(runner=Runner(),_state_tensor=lambda x:x,_frame_tensor=lambda x:x))
        self.packet=dict(_sampling_seed=42,state=torch.zeros(14),frame=torch.zeros(3,4,4),flow_frames=[],
            motion_features=None,motion_interval_valid_mask=None,motion_acceleration_valid_mask=None,
            command_index=4,sim_time=.4)
        self.method=source_method()
    def sample(self,**kwargs):return self.method(self.worker,self.packet,**kwargs)[1]
    def test_packet_seed_survives_global_draws_and_warmup(self):
        before=torch.random.get_rng_state().clone();a=self.sample()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        torch.randn(1000);self.sample(sampling_seed=777);b=self.sample()
        torch.testing.assert_close(a.future_latent,b.future_latent,rtol=0,atol=0)
        torch.testing.assert_close(a.actions,b.actions,rtol=0,atol=0)
    def test_distinct_seed_changes_both_streams(self):
        a=self.sample();b=self.sample(sampling_seed=43)
        self.assertFalse(torch.equal(a.future_latent,b.future_latent))
        self.assertFalse(torch.equal(a.actions,b.actions))
    def test_offline_override_controls_the_noise(self):
        a=self.sample(sampling_seed=123);self.packet['_sampling_seed']=999
        b=self.sample(sampling_seed=123)
        torch.testing.assert_close(a.actions,b.actions,rtol=0,atol=0)
        torch.testing.assert_close(a.future_latent,b.future_latent,rtol=0,atol=0)
    def test_missing_invalid_or_boolean_seed_fails(self):
        for value in (None,True,-1,2**63,42.0,'42'):
            self.packet['_sampling_seed']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.sample()


if __name__=='__main__':unittest.main()
