"""CPU tests for temporal world memory and frozen native WAM distillation."""
from __future__ import annotations
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from dynamicwam.models.compact_fast_policy import CompactFastConfig, CompactFastPolicy
from dynamicwam.models.multirate_types import BridgeTensorBundle, SourceObservation, SourceNoise
from dynamicwam.models.temporal_world_memory import temporal_world_layout
from dynamicwam.models.native_wam_teacher import sample_native_wam
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.runtime.ciwam.wam.multirate_policy import align_replanning_noise
from verify_multirate_v2 import sample


def temporal_sample():
    current, _, text = sample()
    keys = tuple(torch.randn(2,6,2,8,requires_grad=True) for _ in range(2))
    values = tuple(torch.randn(2,6,2,8,requires_grad=True) for _ in range(2))
    current.age.delta_actions = torch.tensor([4.,12.])
    bridge = BridgeTensorBundle(keys, values, torch.tensor([6,6]),
        dict(condition_seq_len=2,future_seq_len=4,future_grid_shape=(2,1,2)),0,0., all_video_tokens_valid=True)
    cfg = CompactFastConfig(dim=32,heads=4,layers=1,memory_tokens=6,
        world_layers=(0,1),world_heads=2,world_head_dim=8,text_dim=8,state_dim=4,
        action_dim=4,chunk_size=16,temporal_world_memory=True)
    model = CompactFastPolicy(cfg,motion_history_count=2,motion_mean=[0.]*12,motion_scale=[1.]*12)
    return model,current,bridge,text


class TemporalRegressions(unittest.TestCase):
    def setUp(self): torch.manual_seed(42)

    def test_forecast_rebasing_expiry_and_no_host_boolean(self):
        _,current,bridge,_ = temporal_sample()
        with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host sync')):
            feature, mask, keep = temporal_world_layout(bridge, current.age.delta_actions,
                queries=6,horizon_actions=16,device='cpu',dtype=torch.float32)
        self.assertEqual(mask.shape,(2,1,6,6))
        self.assertEqual(keep[0,:,0].tolist(),[1.]*6)
        self.assertEqual(keep[1,:,0].tolist(),[1.,1.,0.,0.,1.,1.])
        self.assertEqual(feature[1,4,2], .25)  # source t+16 is now only 4 steps away
        self.assertTrue(mask.any(-1).all())

    def test_expired_future_values_do_not_change_memory_but_live_future_does(self):
        model,current,bridge,text=temporal_sample()
        before=model.prepare_memory(current,bridge,text).layer_kv[0][0]
        values=tuple(v.detach().clone() for v in bridge.video_v)
        for v in values: v[1,2:4] += 1000  # first future bin expired at age12
        after=model.prepare_memory(current,dataclasses.replace(bridge,video_v=values),text).layer_kv[0][0]
        torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)
        values=tuple(v.detach().clone() for v in bridge.video_v)
        for v in values: v[1,4:] += 10  # last future bin still covers execution prefix
        changed=model.prepare_memory(current,dataclasses.replace(bridge,video_v=values),text).layer_kv[0][0]
        self.assertFalse(torch.allclose(before[1],changed[1]))
        before.square().sum().backward()
        self.assertEqual(bridge.video_v[0].grad[1,2:4].abs().sum(),0)
        self.assertGreater(bridge.video_v[0].grad[1,4:].abs().sum(),0)
        self.assertIsNotNone(model.world_time_encoder[0].weight.grad)

    def test_temporal_fast_export_is_executable_and_exact_on_cpu(self):
        from dynamicwam.inference.multirate_model_loader import export_fast_only_payload
        from dynamicwam.inference.fast_runtime import build_fast_runtime
        model,current,bridge,text=temporal_sample(); model.eval()
        spec=dict(architecture='compact',compact_fast=dataclasses.asdict(model.config),
                  schedule=model.schedule.to_config(),motion=dict(history_count=2,mean=[0.]*12,scale=[1.]*12))
        payload=export_fast_only_payload({'fast_policy.'+k:v for k,v in model.state_dict().items()},module_spec=spec,global_step=0)
        loaded=build_fast_runtime(payload)
        noise=torch.randn(2,16,4)
        with torch.no_grad():
            a=model.sample_actions(noise,model.prepare_memory(current,bridge,text))
            b=loaded.sample_actions(noise,loaded.prepare_memory(current,bridge,text))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_native_teacher_keeps_joint_action_trajectory(self):
        schedule=BridgeSchedule.from_flow_match()
        calls=[]
        class Teacher:
            def _forward_video_action(self,batch,**kw):
                calls.append(('joint',batch['noisy_actions'].clone()))
                self.assert_causal(batch)
                return dict(action_pred=batch['noisy_actions']*.1+batch['initial_state'][:,None],
                    video_pred=torch.ones_like(batch['future_latent']),video_cache={'state':batch['initial_state']})
            def assert_causal(self,batch):
                assert 'action_sequence' not in batch
            def forward_action_with_video_cache(self,batch):
                calls.append(('action',batch['noisy_actions'].clone()))
                return dict(action_pred=batch['noisy_actions']*.1+batch['video_cache']['state'][:,None])
        source=SourceObservation(torch.zeros(1,1,1,1,1),torch.ones(1,2),torch.zeros(1,1,12),
            torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),[],0,0.)
        noise=SourceNoise(torch.randn(1,1,1,1,1),torch.randn(1,16,2))
        result=sample_native_wam(Teacher(),source,noise,schedule)
        from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler
        vendor=FlowMatchScheduler(shift=5.,sigma_min=0.,extra_one_step=True)
        vendor.set_timesteps(10,training=False)
        expected=noise.actions.clone()
        for t in vendor.timesteps:
            expected=vendor.step(expected*.1+source.state[:,None],t,expected)
        torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-5)
        self.assertEqual([kind for kind,_ in calls],['joint']*2+['action']*8)
        self.assertFalse(torch.equal(calls[2][1],noise.actions))
        self.assertFalse(result.requires_grad)

    def test_distillation_step_updates_student_and_preserves_all_teacher_weights(self):
        from verify_multirate_training import _Backbone,_config,_batch,_Accel
        from dynamicwam.models.multirate_wam import MultirateWAM
        from dynamicwam.training.train.multirate_trainer import MultirateTrainer,configure_multirate_trainable,multirate_optimizer_groups
        from dynamicwam.training.train.multirate_objective import MultirateLossObjective
        from dynamicwam.training.train.common import build_training_flow_scheduler
        from dynamicwam.training.utils.scheduler import create_scheduler
        cfg=CompactFastConfig(dim=32,heads=4,layers=1,memory_tokens=4,world_layers=(0,),
            world_heads=1,world_head_dim=1,text_dim=8,state_dim=4,action_dim=4,chunk_size=4)
        student=CompactFastPolicy(cfg,motion_history_count=4,motion_mean=[0.]*12,motion_scale=[1.]*12)
        model=MultirateWAM(_Backbone(),None,compact_policy=student,native_teacher_weight=.5)
        configure_multirate_trainable(model,'warmup')
        trainer=MultirateTrainer.__new__(MultirateTrainer)
        trainer.config=_config(); trainer.config['loss']={'native_teacher_weight':.5}
        trainer.device='cpu'; trainer.phase='warmup'; trainer.model=model; trainer.accelerator=_Accel()
        groups=multirate_optimizer_groups(model,trainer.config,'warmup')
        self.assertEqual([g['name'] for g in groups],['fast_conditioner'])
        trainer.optimizer=torch.optim.AdamW(groups)
        trainer.scheduler=create_scheduler(trainer.optimizer,total_steps=4,warmup_steps=0,min_lr_ratio=.1)
        flow=build_training_flow_scheduler(trainer.config);trainer._timesteps=flow.timesteps;trainer._sigmas=flow.sigmas
        trainer.objective=MultirateLossObjective(action_weight=0.,native_teacher_weight=.5)
        before={k:v.clone() for k,v in model.backbone.state_dict().items()}
        old=student.out.weight.detach().clone()
        metrics=trainer.train_step(_batch())
        self.assertIn('loss/native_teacher',metrics)
        self.assertFalse(torch.equal(old,student.out.weight))
        for k,v in model.backbone.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_noise_overlap_is_shifted_without_reusing_predicted_actions(self):
        fresh,previous=torch.randn(1,16,14),torch.randn(1,16,14)
        mixed=align_replanning_noise(fresh,previous,12,8)
        torch.testing.assert_close(mixed[:,:12],previous[:,4:],rtol=0,atol=0)
        torch.testing.assert_close(mixed[:,12:],fresh[:,-4:],rtol=0,atol=0)
        self.assertIs(align_replanning_noise(fresh,previous,24,8),fresh)
        self.assertIs(align_replanning_noise(fresh,None,0,None),fresh)

    def test_temporal_profile_uses_existing_parent_and_explicit_teacher_loss(self):
        from dynamicwam.config.multirate import load_multirate_profile
        cfg=load_multirate_profile(Path(__file__).resolve().parents[1]/'configs/world_action_multirate_temporal.yaml').training
        self.assertEqual(cfg['loss']['native_teacher_weight'],.5)
        self.assertEqual(cfg['phase'],'fast_warmup')
        self.assertTrue(cfg['model']['compact_fast']['temporal_world_memory'])
        self.assertEqual(cfg['model']['compact_fast']['memory_tokens'],48)

    def test_h16_prefetch_cannot_bypass_new_training_or_forecast_contract(self):
        from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy
        for temporal in (False, True):
            policy=MultiratePolicy.__new__(MultiratePolicy)
            policy.contract=dict(execute_horizon=4,max_age_actions=12,supervised_action_horizon=4)
            policy._episode_seed=3
            policy.base=SimpleNamespace(runner=SimpleNamespace(_get_text_embeddings=lambda: []))
            policy.fast_model=SimpleNamespace(fast_policy=SimpleNamespace(config=SimpleNamespace(
                temporal_world_memory=temporal,chunk_size=16)))
            controller=SimpleNamespace(execute_horizon=16,bridge_lead_actions=4,executed_total=16,query_id=1)
            with self.assertRaisesRegex(RuntimeError, 'trained execution horizon|supervised fast prefix'):
                policy.query({},controller)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
