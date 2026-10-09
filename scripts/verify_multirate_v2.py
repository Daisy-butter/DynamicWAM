"""CPU regression suite for compact fusion, executable exports and lifecycle.

No checkpoint download, simulator or CUDA context. Reference action comparison
uses the real ActionExpert operations with a CPU SDPA attention backend.
"""
from __future__ import annotations
import dataclasses
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dynamicwam.models.compact_fast_policy import CompactFastConfig, CompactFastPolicy
from dynamicwam.models.fast_conditioner import FastConditioner, FastResidualBlock
from dynamicwam.models.multirate_types import BridgeAge, BridgeTensorBundle, CurrentObservation
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.inference.fast_runtime import build_fast_runtime, finalize_inference
from dynamicwam.inference.multirate_model_loader import export_fast_only_payload
from dynamicwam.runtime.ciwam.wam.bridge_store import BridgeStore, SlotState
from dynamicwam.runtime.ciwam.wam.fast_slow_worker import LatestWorldWorker
from dynamicwam.runtime.ciwam.wam.multirate_control import forward_budget
from dynamicwam.runtime.ciwam.wam.multirate_policy import capture_observation_packet, plan_worker_bridge
from dynamicwam.training.data.multirate_sampling import partition_episodes, BalancedPairSampler
from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.fast_observation_encoder import FastObservationEncoder
from dynamicwam.models.multirate_types import SourceNoise, SourceObservation


def sample():
    age = BridgeAge(torch.tensor([.1, .2]), torch.tensor([4, 8]), True, torch.zeros(2, 4))
    current = CurrentObservation(torch.randn(2, 3, 3, 32, 32), torch.randn(2, 4),
        torch.randn(2, 2, 12), torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool), [torch.randn(5, 8)]*2, age,
        torch.ones(2, 3, 32, 32, dtype=torch.bool))
    keys = tuple(torch.randn(2, 5, 2, 8, requires_grad=True) for _ in range(2))
    values = tuple(torch.randn(2, 5, 2, 8, requires_grad=True) for _ in range(2))
    bridge = BridgeTensorBundle(keys, values, torch.tensor([5, 4]), {}, 0, 0.)
    return current, bridge, torch.randn(2, 5, 8)


def compact():
    c = CompactFastConfig(dim=32, heads=4, layers=2, memory_tokens=4,
        world_layers=(0, 1), world_heads=2, world_head_dim=8,
        text_dim=8, state_dim=4, action_dim=4, chunk_size=2)
    return CompactFastPolicy(c, motion_history_count=2, motion_mean=[0.]*12, motion_scale=[1.]*12)


class MultirateRegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_inference_mode_and_bfloat_residual(self):
        module = FastConditioner(dim=16, layers=1, heads=2, head_dim=8, text_dim=8, state_dim=4)
        finalize_inference(module, device='cpu')
        before = {k:v.clone() for k,v in module.named_buffers()}
        with torch.inference_mode():
            module.encoder(torch.randn(2,3,3,32,32))
        self.assertFalse(any(m.training for m in module.modules()))
        for k,v in module.named_buffers():
            self.assertTrue(torch.equal(v, before[k]))
        residual = FastResidualBlock(16, 2, 8).bfloat16()
        x = torch.randn(2,3,16,requires_grad=True)
        y = residual(x, torch.randn(2,4,16).bfloat16(), torch.randn(2,4,16).bfloat16())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_single_fusion_ten_step_sampler_and_bridge_gradients(self):
        model = compact()
        current, bridge, text = sample()
        calls = [0]
        hook = model.encoder.register_forward_hook(lambda *args: calls.__setitem__(0, calls[0]+1))
        memory = model.prepare_memory(current, bridge, text)
        result = model.sample_actions(torch.randn(2,2,4), memory)
        hook.remove()
        self.assertEqual(calls[0], 1)
        self.assertEqual(len(model.schedule.action_sigmas), 10)
        self.assertEqual(result.shape, (2,2,4))
        result.square().mean().backward()
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum() > 0 for x in bridge.video_k+bridge.video_v))
        self.assertTrue(model.encoder.projection.weight.grad.abs().sum() > 0)

    def test_null_bridge_independence_and_stable_parameter_graph(self):
        model = compact()
        current, bridge, text = sample()
        bridge.null_bridge = True
        memory = model.prepare_memory(current, bridge, text)
        x,t = torch.randn(2,2,4),torch.ones(2)
        first = model.predict_velocity(x,t,memory)
        other = dataclasses.replace(bridge, video_k=tuple(v*3 for v in bridge.video_k), video_v=tuple(v+4 for v in bridge.video_v))
        second = model.predict_velocity(x,t,model.prepare_memory(current,other,text))
        torch.testing.assert_close(first,second)
        first.sum().backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)

    def test_compact_fast_export_reloads_executable_policy(self):
        model = compact().eval()
        spec = dict(architecture='compact', compact_fast=dataclasses.asdict(model.config),
                    schedule=model.schedule.to_config(), motion=dict(history_count=2,mean=[0.]*12,scale=[1.]*12))
        state = {'fast_policy.'+k:v for k,v in model.state_dict().items()}
        state['backbone.compact_wan.fake'] = torch.zeros(1)
        payload = export_fast_only_payload(state, module_spec=spec, global_step=3)
        self.assertTrue(all(k.startswith('fast_policy.') for k in payload['model']))
        loaded = build_fast_runtime(payload)
        self.assertFalse(any('compact_wan' in k for k in loaded.state_dict()))
        current,bridge,text = sample()
        noise=torch.randn(2,2,4)
        with torch.inference_mode():
            a=model.sample_actions(noise,model.prepare_memory(current,bridge,text))
            b=loaded.sample_actions(noise,loaded.prepare_memory(current,bridge,text))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_bfloat_compact_path_and_padding(self):
        model=compact().bfloat16()
        current,bridge,text=sample()
        current.rgb_valid_mask[:,:,0:8]=False
        with torch.no_grad():
            original=model.sample_actions(torch.zeros(2,2,4), model.prepare_memory(current,bridge,text))
            current.rgb[:,:, :,0:8]=1000
            changed=model.sample_actions(torch.zeros(2,2,4), model.prepare_memory(current,bridge,text))
        self.assertTrue(torch.isfinite(original).all())
        torch.testing.assert_close(original,changed,rtol=0,atol=0)

    def test_real_action_expert_standalone_parity(self):
        cfg=SimpleNamespace(dim=16,num_heads=2,head_dim=8,num_layers=2)
        wan=nn.Module()
        wan.video_model=nn.Module()
        wan.video_model.precision=torch.bfloat16
        wan.video_model.wan_model=nn.Module()
        wan.video_model.wan_model.blocks=nn.ModuleList([nn.Module(),nn.Module()])
        for block in wan.video_model.wan_model.blocks:
            block.self_attn=SimpleNamespace(window_size=(-1,-1))
        model=SmallWAMActionModel(SmallWAMActionConfig(cfg,4,4,2,16,32,2,True,2,tuple([0.]*12),tuple([1.]*12)),wan)
        conditioner=FastConditioner(dim=16,layers=2,heads=2,head_dim=8,text_dim=8,state_dim=4)
        full=MultirateWAM(model,conditioner).eval()
        spec=dict(architecture='reference',schedule=full.schedule.to_config(),
             action_expert=dict(dim=16,ffn_dim=32,num_layers=2,state_dim=4,action_dim=4,chunk_size=2),
             wan_attention=dict(dim=16,num_heads=2,head_dim=8,precision='bfloat16',window_sizes=[[-1,-1]]*2),
             motion=dict(history_count=2,mean=[0.]*12,scale=[1.]*12),
             fast_conditioner=dict(dim=16,layers=2,heads=2,head_dim=8,text_dim=8,state_dim=4,sim_scale=1.,action_scale=16.))
        standalone=build_fast_runtime(export_fast_only_payload(full.state_dict(),module_spec=spec,global_step=0))
        current,bridge,text=sample()
        def cpu_attention(q,k,v,k_lens,window_size):
            mask=torch.arange(k.shape[1])[None]<k_lens[:,None]
            return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask[:,None,None]).transpose(1,2)
        with patch('dynamicwam.models.action_core_ops.flash_attention',cpu_attention),torch.inference_mode():
            noise=torch.randn(2,2,4)
            a=full.sample_fast_actions(current,bridge,full.prepare_fast_memory(current,bridge,text),noise)
            b=standalone.sample_fast_actions(current,bridge,standalone.prepare_fast_memory(current,bridge,text),noise)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_cache_pool_reclamation_and_generation(self):
        store=BridgeStore(2)
        def publish(source,generation):
            slot=store.claim_free()
            store.mark_pending(slot,source,source_id=source,generation=generation)
            store.mark_producer_complete(slot)
            return store.publish(slot)
        for n in range(8):
            self.assertTrue(publish(n,0))
        store.reset(1)
        self.assertTrue(publish(0,1))
        self.assertFalse(publish(99,0))
        self.assertEqual(store.acquire(generation=1).payload,0)

    def test_worker_nonblocking_and_reset(self):
        entered,release=threading.Event(),threading.Event()
        def build(packet):
            if packet=='blocked':
                entered.set()
                if not release.wait(3):raise TimeoutError('test producer gate')
            return packet
        worker=LatestWorldWorker(build)
        try:
            worker.request(0,'initial')
            self.assertEqual(worker.wait_for(0,3).value,'initial')
            worker.request(4,'blocked')
            self.assertTrue(entered.wait(3))
            self.assertEqual(worker.latest().value,'initial')
            worker.reset()
            worker.request(0,'new_episode')
            release.set()
            self.assertEqual(worker.wait_for(0,3).value,'new_episode')
            self.assertGreaterEqual(worker.dropped,1)
        finally:
            release.set();worker.close()

    def test_episode_holdout_and_stress_sampling(self):
        rows=[dict(episode_index=i,split='clean',task_name='task',episode_name=f'ep{i}') for i in range(20)]
        assignments=partition_episodes(rows,seed=42,validation_fraction=.2)
        self.assertEqual(assignments,partition_episodes(list(reversed(rows)),seed=42,validation_fraction=.2))
        self.assertEqual(sum(v=='validation' for v in assignments.values()),4)
        records=[SimpleNamespace(task_name='task',episode_index=e,offset_label=label,offset=offset)
                 for e in range(3) for label,offset in [('steady_state',0),('steady_state',12),('stress',48)]]
        sampler=BalancedPairSampler(records,samples_per_episode=30,stress_probability=0,seed=42)
        drawn=list(sampler)
        self.assertTrue(all(records[i].offset_label=='steady_state' for i in drawn))
        counts={}
        for index in drawn:
            episode=records[index].episode_index
            counts[episode]=counts.get(episode,0)+1
        self.assertEqual(counts,{0:30,1:30,2:30})
        first=drawn
        sampler.set_epoch(1)
        self.assertNotEqual(first,list(sampler))
        rank0=list(BalancedPairSampler(records,samples_per_episode=30,stress_probability=0,seed=42,rank=0,world_size=2))
        rank1=list(BalancedPairSampler(records,samples_per_episode=30,stress_probability=0,seed=42,rank=1,world_size=2))
        self.assertEqual(len(rank0)+len(rank1),90)
        self.assertNotEqual(rank0,rank1)

    def test_worker_prefetches_before_the_refresh_and_copies_tensors(self):
        self.assertEqual(plan_worker_bridge(action_index=0,bridge_index=0,bridge_present=False,max_age=12,horizon=4,submitted_source=None),('blocking',0))
        self.assertEqual(plan_worker_bridge(action_index=8,bridge_index=0,bridge_present=True,max_age=12,horizon=4,submitted_source=None),('use',None))
        self.assertEqual(plan_worker_bridge(action_index=12,bridge_index=0,bridge_present=True,max_age=12,horizon=4,submitted_source=None),('prefetch',12))
        self.assertEqual(plan_worker_bridge(action_index=16,bridge_index=0,bridge_present=True,max_age=12,horizon=4,submitted_source=12),('stale',12))
        self.assertEqual(plan_worker_bridge(action_index=16,bridge_index=0,bridge_present=True,max_age=12,horizon=4,submitted_source=None),('stale',16))
        self.assertEqual(plan_worker_bridge(action_index=0,bridge_index=0,bridge_present=False,max_age=12,horizon=4,submitted_source=4),('blocking',4))
        self.assertEqual(plan_worker_bridge(action_index=16,bridge_index=12,bridge_present=True,max_age=12,horizon=4,submitted_source=None),('use',None))
        text=torch.ones(2,4)
        frame=np.zeros((2,2,3),dtype=np.uint8)
        copied=capture_observation_packet({'frame':frame,'_text':[text]})
        text.zero_()
        frame[...] = 7
        self.assertTrue(torch.equal(copied['_text'][0],torch.ones(2,4)))
        self.assertEqual(int(copied['frame'][0,0,0]),0)

    def test_default_budget_and_fresh_fast_trajectory(self):
        budget=forward_budget()
        self.assertEqual(budget['executed_actions'],16)
        self.assertEqual(budget['multirate_world_joint_forwards'],budget['sync_long_chunk_joint_forwards'])
        self.assertGreater(budget['multirate_fast_action_forwards'],budget['sync_long_chunk_action_forwards'])
        self.assertLess(budget['multirate_world_joint_forwards'],budget['sync_matched_horizon_joint_forwards'])
        model=compact()
        self.assertEqual(int(model.schedule.action_sigmas.shape[0]),10)

    def test_letterbox_padding_and_batchnorm_train_eval(self):
        encoder=FastObservationEncoder(dim=16)
        encoder.train()
        running=encoder.stem[1].running_mean.clone()
        encoder(torch.randn(2,3,3,32,32))
        self.assertFalse(torch.equal(encoder.stem[1].running_mean,running))
        encoder.eval()
        frozen=encoder.stem[1].running_mean.clone()
        with torch.inference_mode():
            encoder(torch.randn(2,3,3,32,32))
        self.assertTrue(torch.equal(encoder.stem[1].running_mean,frozen))
        rgb=torch.randn(1,3,3,32,32)
        mask=torch.ones(1,3,32,32,dtype=torch.bool)
        mask[:,:,:8]=False
        with torch.no_grad():
            original,_valid=encoder(rgb,mask)
            rgb[:,:,:,:8]=1000
            changed,_valid=encoder(rgb,mask)
        torch.testing.assert_close(original,changed,rtol=0,atol=0)

    def test_joint_prefix_keeps_the_first_euler_step(self):
        class Twin(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale0=nn.Parameter(torch.tensor(1.))
                self.scale1=nn.Parameter(torch.tensor(1.))
                self.calls=0
            def _forward_video_action(self,batch,include_video_prediction,build_video_cache,detach_cache):
                self.calls+=1
                scale=self.scale0 if self.calls==1 else self.scale1
                video=batch['future_latent']*scale
                action=batch['noisy_actions']*scale
                out={'video_pred':video,'action_pred':action}
                if build_video_cache:
                    token=video.reshape(video.shape[0],-1)[:,:1].reshape(video.shape[0],1,1,1)
                    if detach_cache:
                        token=token.detach()
                    out['video_cache']={'video_k':[token],'video_v':[token],'seq_lens':torch.ones(video.shape[0],dtype=torch.long),'grid_sizes':{}}
                return out
        model=Twin()
        bridge=CausalWorldBridge(BridgeSchedule.from_flow_match())
        source=SourceObservation(torch.randn(2,1,2,2,2),torch.randn(2,4),torch.randn(2,2,12),
            torch.ones(2,2,dtype=torch.bool),torch.ones(2,2,dtype=torch.bool),[torch.randn(3,8)]*2,
            torch.tensor([3,9]),torch.tensor([1.5,2.5]))
        bundle=bridge(model,source,SourceNoise(torch.randn(2,1,2,2,2),torch.randn(2,4,4)),detach_cache=False)
        bundle.video_k[0].sum().backward()
        self.assertIsNotNone(model.scale0.grad)
        self.assertGreater(float(model.scale0.grad.abs()),0)
        self.assertGreater(float(model.scale1.grad.abs()),0)

    def test_per_sample_null_bridge(self):
        model=compact()
        current,bridge,text=sample()
        bridge.null_bridge=torch.tensor([True,False])
        other=dataclasses.replace(bridge,video_k=tuple(v*3 for v in bridge.video_k),video_v=tuple(v+4 for v in bridge.video_v),null_bridge=torch.tensor([True,False]))
        x,t=torch.randn(2,2,4),torch.ones(2)
        first=model.predict_velocity(x,t,model.prepare_memory(current,bridge,text))
        second=model.predict_velocity(x,t,model.prepare_memory(current,other,text))
        torch.testing.assert_close(first[0],second[0])
        self.assertFalse(torch.allclose(first[1],second[1]))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
