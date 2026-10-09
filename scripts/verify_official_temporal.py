"""CPU contract regressions for official-anchored temporal student."""
import dataclasses
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from verify_multirate_temporal import temporal_sample
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.training.train.multirate_objective import MultirateLossObjective
from dynamicwam.models.multirate_wam import MultirateWAM
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable

class Contracts(unittest.TestCase):
    def test_full_chunk_loss_supervises_tail(self):
        p=torch.ones(1,16,2,requires_grad=True)
        obj=MultirateLossObjective(action_weight=0)
        loss,_=obj(dict(fast_pred=p,action_pred=torch.zeros_like(p)),torch.zeros_like(p),torch.zeros_like(p))
        loss.backward()
        self.assertGreater(p.grad[:,4:].abs().sum(),0)

    def test_seconds_drive_temporal_mask_even_with_same_action_age(self):
        model,current,bridge,text=temporal_sample()
        model.config=dataclasses.replace(model.config,temporal_seconds_per_action=.15)
        current.age.delta_actions=torch.tensor([4.,4.])
        current.age.delta_sim_seconds=torch.tensor([.6,2.])
        from dynamicwam.models.temporal_world_memory import temporal_world_layout
        with patch('dynamicwam.models.temporal_world_memory.temporal_world_layout', wraps=temporal_world_layout) as f:
            memory=model.prepare_memory(current,bridge,text)
            torch.testing.assert_close(f.call_args.args[1],torch.tensor([4.,2/.15]))
        self.assertTrue(torch.isfinite(memory.layer_kv[0][0]).all())

    def test_joint_requires_external_teacher_and_keeps_it_out_of_model(self):
        from verify_multirate_training import _Backbone
        student,*_=temporal_sample()
        m=MultirateWAM(_Backbone(),None,compact_policy=student,native_teacher_weight=.5)
        with self.assertRaisesRegex(ValueError,'fixed teacher'):
            configure_multirate_trainable(m,'joint')
        m.external_native_teacher=True
        configure_multirate_trainable(m,'joint')
        self.assertTrue(any(p.requires_grad for p in m.backbone.parameters()))
        self.assertFalse(any('teacher' in k for k in m.state_dict()))

    def test_configs_pin_full_supervision_and_physical_ttl(self):
        root=Path(__file__).resolve().parents[1]
        for phase in ['stage2','stage3']:
            c=load_multirate_profile(root/'configs'/f'official_temporal_{phase}_7gpu.yaml').training
            self.assertEqual(c['fast_supervision'],'full_chunk')
            self.assertEqual(c['max_bridge_age_seconds'],.72)
            self.assertEqual(c['loss']['native_teacher_weight'],.5)
            self.assertTrue(c['model']['native_teacher_checkpoint'].endswith('DynamicWAM_full.pt'))

    def test_transfer_provenance_keeps_official_motion_across_stages(self):
        from dynamicwam.training.train.multirate_trainer import MultirateTrainer
        packed = {
            'format': 'dynamicwam_absolute_motion_dataset', 'version': 2,
            'dataset_fingerprint': 'target', 'action_stats_sha256': 'act',
            'motion_statistics_sha256': 'packed-motion',
        }
        official_motion = {
            'feature_mean': [0.0]*12, 'feature_scale': [1.0]*12,
            'statistics_sha256': 'official-motion', 'history_count': 4,
            'feature_names': [f'f{i}' for i in range(12)],
            'spatial_unit': 'pixels', 'timestamp_source': 'sim',
            'temporal_contract': 'exact', 'flow_contract': {},
            'statistics_file': 'motion_stats.json',
        }
        parent = {
            'path': '/ckpt/full.pt',
            'source_dataset_identity': {'motion_statistics_sha256': 'official-motion'},
            'target_dataset_identity': packed,
            'motion_statistics_sha256': 'official-motion',
        }
        payload = {
            'format': 'dynamicwam_multirate_checkpoint',
            'config': {
                'dataset_identity': packed,
                'initialization_provenance': parent,
                'model': {'absolute_motion': official_motion},
            },
        }
        trainer = MultirateTrainer.__new__(MultirateTrainer)
        trainer.dataset_identity = packed
        trainer.motion_checkpoint_metadata = {'statistics_sha256': 'packed-motion'}
        trainer.motion_statistics = {'mean': [1.0]*12, 'scale': [1.0]*12}
        with patch('dynamicwam.training.train.multirate_trainer.load_checkpoint_payload', return_value=payload), \
             patch('dynamicwam.training.train.multirate_trainer.validate_checkpoint_motion_metadata', return_value=official_motion):
            trainer._apply_pretrained_dataset_transfer('/ckpt/stage2.pt')
        self.assertEqual(trainer.motion_checkpoint_metadata['statistics_sha256'], 'official-motion')
        self.assertEqual(
            trainer.initialization_provenance['source_dataset_identity']['motion_statistics_sha256'],
            'official-motion',
        )
        self.assertIs(trainer.initialization_provenance['parent_provenance'], parent)

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
