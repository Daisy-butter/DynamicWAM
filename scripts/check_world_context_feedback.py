#!/usr/bin/env python3
"""CPU contracts for the generated-world feedback head and offline labels."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from dynamicwam.models.plan_feedback import PlanFeedbackHead
from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
from dynamicwam.models.world_context_feedback import WorldContextFeedbackHead,pack_world_context,WORLD_CONTEXT_SPEC
from dynamicwam.training.data.world_context_feedback_dataset import WorldContextFeedbackDataset,world_context_collate_fn,sha256

spec=importlib.util.spec_from_file_location('world_feedback_training',ROOT/'scripts/train_world_context_feedback.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def inputs(ages=(4,12),size=32):
    b=len(ages)
    return dict(rgb=torch.randn(b,3,3,size,size),rgb_valid_mask=torch.ones(b,3,size,size,dtype=torch.bool),
        state=torch.randn(b,14),source_state=torch.randn(b,14),plan=torch.randn(b,16,14),
        age_actions=torch.tensor(ages),text=torch.randn(b,3,4096),text_mask=torch.ones(b,3,dtype=torch.bool),
        previous_head=torch.randn(b,3,size,size),previous_head_mask=torch.ones(b,size,size,dtype=torch.bool),
        history_dt=torch.full((b,),.2),history_valid=torch.ones(b,dtype=torch.bool),plan_age_seconds=torch.tensor(ages).float()*.1)


def world(b=2):
    return dict(world_values=torch.randn(b,5,2048),world_coordinates=torch.randn(b,5,4),
        world_mask=torch.ones(b,5,dtype=torch.bool),context_valid=torch.ones(b,dtype=torch.bool))


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(4)

    def test_pool_preserves_bin_order_and_owns_storage(self):
        v=torch.cat([torch.full((1,4,16,128),float(i)) for i in range(3)],1).bfloat16()
        cache=dict(video_v=[v]*12,all_video_tokens_valid=True,grid_sizes=dict(
            condition_seq_len=4,future_seq_len=8,condition_grid_shape=(1,2,2),future_grid_shape=(2,2,2)))
        packed=pack_world_context(cache)
        self.assertEqual(tuple(packed['world_values'].shape),(1,48,2048))
        for i in range(3):self.assertTrue(torch.equal(packed['world_values'][:,i*16:(i+1)*16],torch.full((1,16,2048),float(i)).bfloat16()))
        v.fill_(10)
        self.assertEqual(float(packed['world_values'][0,16,0]),1.)
        self.assertFalse(packed['world_values'].requires_grad)
        self.assertTrue(torch.equal(packed['world_coordinates'][0,:16,0],torch.zeros(16)))
        self.assertTrue(torch.equal(packed['world_coordinates'][0,16:,0],torch.ones(32)))

    def test_legacy_migration_and_invalid_context_are_function_preserving(self):
        legacy=PlanFeedbackHead(dim=24,heads=3,layers=1).eval()
        torch.nn.init.normal_(legacy.output.weight,std=.03)
        base=TemporalPlanFeedbackHead(**legacy.constructor_spec).eval();base.initialize_from_plan_feedback(legacy.state_dict())
        head=WorldContextFeedbackHead(**legacy.constructor_spec).eval();head.initialize_from_plan_feedback(legacy.state_dict())
        x=inputs();w=world()
        with torch.no_grad():
            expected=base(**x);actual=head(**x,**w)
            torch.testing.assert_close(actual,expected,rtol=0,atol=2e-7)
            head.world_output.weight.normal_(std=.5);w['context_valid'].zero_()
            torch.testing.assert_close(head(**x,**w),expected,rtol=0,atol=2e-7)

    def test_world_readout_receives_real_gradients(self):
        legacy=PlanFeedbackHead(dim=24,heads=3,layers=1)
        torch.nn.init.normal_(legacy.output.weight,std=.03)
        head=WorldContextFeedbackHead(**legacy.constructor_spec);head.initialize_from_plan_feedback(legacy.state_dict())
        x=inputs((3,11));w=world();target=torch.randn(2,4,14)
        pred=head(**x,**w);loss,_=trainer.teacher_only_objective(pred,x,target);loss.backward()
        self.assertGreater(float(head.world_output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
        opt=torch.optim.AdamW(head.parameters(),lr=2e-5);opt.step();opt.zero_grad()
        loss,_=trainer.teacher_only_objective(head(**x,**w),x,target);loss.backward()
        self.assertGreater(float(head.world_projection[1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.dtype==torch.float32 for p in head.parameters()))

    def test_masks_and_additive_coordinate_normalization(self):
        x=inputs((0,15));prediction=torch.randn(2,4,14,requires_grad=True)
        target=torch.randn(2,4,14)
        loss,m=trainer.teacher_only_objective(prediction,x,target);loss.backward()
        self.assertEqual(int(m['count']),12)
        self.assertEqual(float(prediction.grad[0].abs().sum()),0.)
        self.assertEqual(float(prediction.grad[1,1:].abs().sum()),0.)
        self.assertEqual(float(prediction.grad[:,:, [6,13]].abs().sum()),0.)
        pieces=[trainer.teacher_only_objective(prediction[i:i+1],{k:v[i:i+1] for k,v in x.items()},target[i:i+1])[1] for i in range(2)]
        torch.testing.assert_close(loss,sum(p['loss_numerator'] for p in pieces)/sum(p['count'] for p in pieces))

    def test_balanced_sampler_is_exact_at_each_rank_microbatch(self):
        rows=[dict(age=3,behavior='graph_zero') for _ in range(31)]+[dict(age=4,behavior='world_context') for _ in range(27)]
        for rank in range(6):
            sampler=trainer.BalancedBehaviorSampler(rows,rank=rank,world_size=6,batch_size=2,seed=44)
            selected=list(sampler)
            for begin in range(0,len(selected),2):
                self.assertEqual([rows[j]['behavior'] for j in selected[begin:begin+2]],['graph_zero','world_context'])
            self.assertEqual(selected,list(sampler));sampler.set_epoch(1);self.assertNotEqual(selected,list(sampler))

    def test_static_world_disabled_preserves_inputs_without_alias_mutation(self):
        x=inputs();x.update(world())
        batch=dict(inputs=x,teacher=torch.zeros(2,4,14))
        actual,_=trainer.prepare_batch(batch,torch.device('cpu'),history_mode='static',world_mode='disabled')
        self.assertFalse(bool(actual['history_valid'].any()));self.assertFalse(bool(actual['context_valid'].any()))
        self.assertTrue(bool(x['history_valid'].all()));self.assertTrue(bool(x['context_valid'].all()))


class DataTests(unittest.TestCase):
    def fixture(self,root):
        def save(name,value,tensor=False):
            path=root/name
            if tensor:torch.save(value,path)
            else:path.write_text(json.dumps(value))
            return dict(path=str(path),sha256=sha256(path))
        base='a'*64;protocol=save('protocol.json',dict(control_period_s=.1))
        official=save('official.json',dict(source_arm='official_sync_h16',tasks={f't{i}':list(range(100000,100020)) for i in range(35)}))
        source=save('collector.json',dict(source='immutable test collector'))
        episodes=[]
        for seed,partition in ((1,'train'),(2,'validation')):
            x=inputs((3,),size=160);x['plan_age_seconds']=torch.tensor([.3])
            values=torch.randn(1,12,16,128).bfloat16()
            packed=pack_world_context(dict(video_v=[values]*12,all_video_tokens_valid=True,grid_sizes=dict(
                condition_seq_len=4,future_seq_len=8,condition_grid_shape=(1,2,2),future_grid_shape=(2,2,2))))
            bundle=dict(format='dynamicwam_world_context_bundle',version=1,plan_id=f'p{seed}',
                source_action_index=0,source_sim_time=0.,published_sim_time=None,
                base_checkpoint_sha256=base,controller_protocol_sha256=protocol['sha256'],world_context_spec=WORLD_CONTEXT_SPEC,
                **{k:x.pop(k) for k in ('plan','source_state','text','text_mask')},**packed)
            ref=save(f'bundle{seed}.pt',bundle,True)
            row=dict(format='dynamicwam_world_context_record',version=1,task='t0',seed=seed,plan_id=f'p{seed}',
                command_index=3,source_action_index=0,age=3,source_sim_time=0.,current_sim_time=.3,inputs=x,
                world_context_ref=ref,behavior='graph_zero',behavior_checkpoint_sha256=None,
                base_checkpoint_sha256=base,controller_protocol_sha256=protocol['sha256'])
            raw=save(f'raw{seed}.pt',row,True)
            row=dict(row,teacher=torch.randn(1,16,14),label_provenance=dict(teacher_only=True,current_observation=True,
                teacher_base_checkpoint_sha256=base,raw_record=raw))
            record=save(f'label{seed}.pt',row,True)
            index=root/f'index{seed}.jsonl';index.write_text(json.dumps(dict(record,plan_id=f'p{seed}',age=3,command_index=3))+'\n')
            report=save(f'report{seed}.json',dict(status='complete',task='t0',seed=seed,controller_protocol_sha256=protocol['sha256']))
            episodes.append(dict(partition=partition,task='t0',seed=seed,behavior='graph_zero',behavior_checkpoint_sha256=None,
                complete=True,index=dict(path=str(index),sha256=sha256(index)),report=report))
        manifest=dict(format='dynamicwam_world_context_feedback_data',version=1,base_checkpoint=dict(sha256=base),
            action_normalization={},world_context_spec=WORLD_CONTEXT_SPEC,controller_protocol=protocol,
            official_seed_manifest=official,source_artifacts=[source],episodes=episodes)
        path=root/'manifest.json';path.write_text(json.dumps(manifest))
        return path,dict(base_checkpoint=dict(sha256=base),action_normalization={})

    def test_bundle_shared_inputs_sha_identity_and_episode_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,reference=self.fixture(root)
            train=WorldContextFeedbackDataset(path,partition='train',reference_checkpoint=reference)
            val=WorldContextFeedbackDataset(path,partition='validation',reference_checkpoint=reference)
            self.assertEqual(train.identity,val.identity);self.assertEqual(train.heldout_episodes,{('t0',2)})
            batch=world_context_collate_fn([train[0],val[0]])
            self.assertEqual(tuple(batch['inputs']['plan'].shape),(2,16,14))
            self.assertEqual(tuple(batch['teacher'].shape),(2,4,14))
            raw=root/'bundle1.pt';raw.write_bytes(raw.read_bytes()+b'x')
            fresh=WorldContextFeedbackDataset(path,partition='train',reference_checkpoint=reference,max_cached=0)
            with self.assertRaisesRegex(ValueError,'changed immutable'):fresh[0]

    def test_duplicate_or_evaluation_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,reference=self.fixture(root);manifest=json.loads(path.read_text())
            manifest['episodes'][1]['seed']=1;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'duplicate'):WorldContextFeedbackDataset(path,partition='train',reference_checkpoint=reference)

    def test_learned_behavior_may_revisit_training_seed_without_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,reference=self.fixture(root);manifest=json.loads(path.read_text())
            raw=torch.load(root/'raw1.pt',weights_only=True);raw=dict(raw,behavior='world_context',behavior_checkpoint_sha256='b'*64)
            torch.save(raw,root/'new_raw.pt')
            label=torch.load(root/'label1.pt',weights_only=True)
            label=dict(label,behavior='world_context',behavior_checkpoint_sha256='b'*64,
                label_provenance=dict(label['label_provenance'],raw_record=dict(path=str(root/'new_raw.pt'),sha256=sha256(root/'new_raw.pt'))))
            torch.save(label,root/'new_label.pt')
            index=root/'new_index.jsonl';index.write_text(json.dumps(dict(path=str(root/'new_label.pt'),sha256=sha256(root/'new_label.pt'),
                plan_id='p1',age=3,command_index=3))+'\n')
            manifest['episodes'].append(dict(manifest['episodes'][0],behavior='world_context',behavior_checkpoint_sha256='b'*64,
                index=dict(path=str(index),sha256=sha256(index))))
            path.write_text(json.dumps(manifest))
            data=WorldContextFeedbackDataset(path,partition='train',reference_checkpoint=reference)
            self.assertEqual(len(data),2);self.assertEqual(data[1]['behavior'],'world_context')
            self.assertEqual(data.heldout_episodes,{('t0',2)})


if __name__=='__main__':unittest.main(verbosity=2)
