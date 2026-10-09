#!/usr/bin/env python3
"""CPU contracts for learned-policy collection, without loading the WAM model.

The real worker methods are compiled without their heavyweight import graph;
their dependencies are CPU fixtures. Numerical tests require PyTorch and never
construct a CUDA model. Torch-free hosts run the explicit source/API checks.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'src/dynamicwam/runtime'
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import numpy as np
    import torch


def definitions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes}!=set(names):raise AssertionError('source API changed')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


class SourceContracts(unittest.TestCase):
    def test_explicit_factories_finalize_after_physics(self):
        tree=ast.parse((ROOT/'scripts/eval_domino_continuous.py').read_text())
        constant=next(node for node in tree.body if isinstance(node,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='BUFFERED_POLICY_FACTORIES' for t in node.targets))
        factories=ast.literal_eval(constant.value.args[0])
        self.assertEqual(set(factories),{
            'dynamicwam.runtime.buffered_world_context_worker:create_worker_policy',
            'dynamicwam.runtime.buffered_predictive_feedback_worker:create_worker_policy'})
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        calls=list(ast.walk(main))
        physics=next(n for n in calls if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                     and n.func.id=='run_continuous_episode')
        finalize=next(n for n in calls if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                      and n.func.attr=='finalize_recordings')
        self.assertGreater(finalize.lineno,physics.end_lineno)

    def test_new_collector_reuses_the_legacy_writer(self):
        tree=ast.parse((RUNTIME/'buffered_predictive_feedback_worker.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        for name in ('_record','finalize'):
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            calls=[n for n in ast.walk(method) if isinstance(n,ast.Call)]
            self.assertTrue(any(isinstance(c.func,ast.Attribute) and c.func.attr==name
                and isinstance(c.func.value,ast.Name) and c.func.value.id=='BufferedWorldContextWorker'
                for c in calls))


@unittest.skipUnless(HAS_TORCH,'PyTorch unavailable; run this script in the DOMINO Python environment')
class CollectionTests(unittest.TestCase):
    def setUp(self):
        class FakeBase: pass
        self.ns=dict(torch=torch,np=np,Path=Path,json=json,math=math,time=time,uuid=uuid,
            WorldContextWorker=FakeBase,file_sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            WORLD_CONTEXT_SPEC={'fixture':'same writer'},
            fixed_feedback_text=lambda text:(text,torch.ones(text.shape[:2],dtype=torch.bool)),
            pack_world_context=lambda cache:{k:v.detach().clone() for k,v in cache.items()},
            proposal_action_path=lambda **kw:(np.zeros((2,4,28),np.float32),np.array([.2,.4],np.float32),{'fixture':True}))
        definitions(RUNTIME/'world_context_worker.py',{'cpu_copy','exclusive_tensor'},self.ns)
        definitions(RUNTIME/'buffered_world_context_worker.py',{'tensor_bytes','BufferedWorldContextWorker'},self.ns)
        definitions(RUNTIME/'predictive_feedback_worker.py',{'PredictiveFeedbackWorker'},self.ns)
        definitions(RUNTIME/'buffered_predictive_feedback_worker.py',{'BufferedPredictiveFeedbackWorker'},self.ns)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def worker(self,collect=True):
        cls=self.ns['BufferedPredictiveFeedbackWorker' if collect else 'PredictiveFeedbackWorker']
        w=object.__new__(cls)
        w.config=dict(mode='predictive',task='fixture',seed=19,samples_per_episode=64)
        if collect:w.config['collection_dir']=str(Path(self.temp.name)/'records')
        w.mode='predictive';w.ablation='none';w.checkpoint_sha='c'*64;w.base_sha='b'*64;w.protocol_sha='d'*64
        w.pending_records=[];w.pending_bundles={};w.buffer_bytes=0;w.finalized=False;w.record_count=0
        w.plan=None;w.plan_numpy=None;w.plan_id=None;w.source_time=None;w.instruction='fixture';w.period=.1;w.refresh_age=12
        w.mean=np.zeros(14,np.float32);w.scale=np.ones(14,np.float32);w.context=None;w.source_noise=None
        self.noise=SimpleNamespace(actions=torch.arange(224).reshape(1,16,14).float(),future_latent=torch.ones(1,2,2))
        w._source=lambda packet:(SimpleNamespace(state=torch.zeros(1,14)),self.noise,torch.zeros(1,4,8))
        self.cache=dict(world_values=torch.ones(1,3,8),world_mask=torch.ones(1,3,dtype=torch.bool),world_coordinates=torch.zeros(1,3,4))
        w.sampler=SimpleNamespace(sample=lambda source,noise:(torch.zeros(1,16,14),self.cache),stats={})
        w.fast=SimpleNamespace(stats={},run_head=lambda inputs:{'residual':torch.full((1,4,14),.01)},
                               run_forecast=lambda *a,**kw:torch.zeros(1,2,48,8))
        w._inputs=lambda packet,age:dict(rgb=torch.ones(1,3,3,8,8),state=torch.zeros(1,14),
            source_state=w.source_state,plan=w.plan,text=w.text,text_mask=w.text_mask,
            age_actions=torch.tensor([age]),history_dt=torch.tensor([.2]),history_valid=torch.tensor([True]),
            previous_head=torch.ones(1,3,8,8),previous_head_mask=torch.ones(1,8,8,dtype=torch.bool))
        return w

    def packet(self,stamp=0.,**kwargs):
        return dict(instruction='fixture',sim_time=stamp,control_period_s=.1,command_index=round(stamp*10),
                    _sampling_seed=73,capture_id=round(stamp*100),history_dt=.2,dispatch_sim_time=stamp,
                    controller_state={'source_physics_time':stamp,
                        'dispatch_controller':{'source_physics_time':stamp},
                        'executed_prefix':{'start_sim_time':[],'end_sim_time':[],
                                           'q_command':[],'velocity_command':[]}},**kwargs)

    def test_refresh_owns_exact_noise_and_context_without_rng_draw(self):
        w=self.worker();before=torch.random.get_rng_state().clone()
        result=w.sample(self.packet())
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(w.source_noise['actions'],self.noise.actions))
        self.noise.actions.zero_();self.cache['world_values'].zero_()
        self.assertGreater(w.source_noise['actions'].sum().item(),0)
        self.assertEqual(w.context['world_values'].min().item(),1)
        self.assertTrue(result['metadata']['collection_enabled'])
        self.assertGreaterEqual(result['metadata']['collection_source_wall_s'],0)
        self.assertEqual(w.pending_records,[])

    def test_causal_rows_no_io_until_finalize_and_identified_learned_behavior(self):
        w=self.worker();w.sample(self.packet())
        with patch.object(Path,'open',side_effect=AssertionError('disk IO during physics')):
            before=torch.random.get_rng_state().clone();result=w.sample(self.packet(.4))
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        row=w.pending_records[0]
        self.assertEqual(row['behavior'],'predictive');self.assertEqual(row['behavior_checkpoint_sha256'],'c'*64)
        self.assertEqual(row['packet']['capture_id'],40)
        self.assertNotIn('proposal_path',row['inputs']);self.assertNotIn('predicted_latents',row['inputs'])
        self.assertGreaterEqual(result['metadata']['recording_wall_s'],0)
        self.assertFalse(Path(w.config['collection_dir']).exists())
        saved=w.finalize();self.assertEqual(saved['writing'],'after_physics_stop');self.assertEqual(saved['records'],1)
        index=json.loads((Path(w.config['collection_dir'])/'raw_index.jsonl').read_text())
        payload=torch.load(index['path'],weights_only=True)
        self.assertEqual(index['sha256'],hashlib.sha256(Path(index['path']).read_bytes()).hexdigest())
        bundle=torch.load(payload['world_context_ref']['path'],weights_only=True)
        self.assertEqual(bundle['format'],'dynamicwam_world_context_bundle')
        self.assertTrue({'plan','source_state','text','text_mask','source_noise','world_values'}<=bundle.keys())
        with self.assertRaises(RuntimeError):w.finalize()

    def test_pure_deployment_and_warmup_do_not_retain_collection_tensors(self):
        w=self.worker(False)
        with patch.dict(self.ns,pack_world_context=lambda *a:(_ for _ in ()).throw(AssertionError('extra copy'))):
            result=w.sample(self.packet());w.sample(self.packet(.4))
        self.assertIsNone(w.context);self.assertIsNone(w.source_noise);self.assertEqual(w.pending_records,[])
        self.assertFalse(result['metadata']['collection_enabled'])
        self.assertFalse(self.worker()._collect_this_query(self.packet(_warmup_only=True)))

    def test_collection_preserves_deployed_actions_and_rng(self):
        deployed=self.worker(False);collected=self.worker()
        for stamp in (0.,.4,.8,1.2,1.6):
            packet=self.packet(stamp);before=torch.random.get_rng_state().clone()
            expected=deployed.sample(packet);actual=collected.sample(packet)
            np.testing.assert_array_equal(expected['actions'],actual['actions'])
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(len(collected.pending_records),3)
        self.assertEqual(len(collected.pending_bundles),2)

    def test_capacity_is_fail_closed_and_sampling_limit_is_explicit(self):
        w=self.worker();w.sample(self.packet());w.buffer_bytes=512*1024**2
        with self.assertRaises(MemoryError):w.sample(self.packet(.4))
        self.assertFalse(Path(w.config['collection_dir']).exists())
        w=self.worker();w.record_count=64
        w.sample(self.packet());self.assertIsNone(w.context)
        w.sample(self.packet(.4));self.assertEqual(w.pending_records,[])

    def test_forbidden_future_inputs_and_unbuffered_collection_rejected(self):
        with self.assertRaises(ValueError):
            self.ns['PredictiveFeedbackWorker']({'mode':'predictive','collection_dir':'unused'})
        w=self.worker();w.sample(self.packet())
        for key in ('proposal_path','proposal_dt','actual_action_path','predicted_latents','future_target'):
            with self.assertRaises(ValueError):w._record(self.packet(.4),{key:torch.zeros(1)},4)


if __name__=='__main__':unittest.main()
