"""CPU regressions for exact source-time backbone loading; no VAE/CUDA."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from torch import nn
from dynamicwam.inference.model_loader import _source_time_conditioning, _build_model, _validate_model_config
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel


class TinyVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(dim=16, num_heads=2, head_dim=8, num_layers=1)


class SourceTimeDeploymentLoader(unittest.TestCase):
    def fixture(self, enabled):
        action = dict(dim=16, ffn_dim=32, num_layers=1, state_dim=14, action_dim=14, chunk_size=16)
        motion = dict(history_count=4, feature_mean=[0.]*12, feature_scale=[1.]*12)
        config = dict(model=dict(action_expert=dict(action, source_physical_time_conditioning=enabled), absolute_motion=motion))
        model = SmallWAMActionModel(SmallWAMActionConfig(compact_wan=TinyVideo().config,
            ae_dim=16, ae_ffn_dim=32, ae_num_layers=1, action_dim=14, state_dim=14, chunk_size=16,
            wan_frozen=True, motion_history_count=4, motion_feature_mean=(0.,)*12,
            motion_feature_scale=(1.,)*12, source_physical_time_conditioning=enabled), TinyVideo())
        if enabled:
            with torch.no_grad():
                for p in model.action_expert.source_time_encoder.parameters():p.fill_(.123)
        return dict(model=dict(action_expert=action)), dict(config=config, model=model.state_dict())

    def build(self, runtime, payload):
        with patch('dynamicwam.inference.model_loader._validate_model_config'), \
             patch('dynamicwam.inference.model_loader._compact_config', return_value=None), \
             patch('dynamicwam.inference.model_loader.CompactWANModel.from_config', side_effect=lambda *a,**k:TinyVideo()):
            return _build_model(runtime,payload,device='cpu')

    def test_new_encoder_strictly_restored_and_old_backbone_still_loads(self):
        for enabled in (False,True):
            runtime,payload=self.fixture(enabled)
            if not enabled:payload['config']['model']['action_expert'].pop('source_physical_time_conditioning')
            model=self.build(runtime,payload)
            self.assertEqual(hasattr(model.action_expert,'source_time_encoder'),enabled)
            self.assertEqual(set(model.state_dict()),set(payload['model']))
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,payload['model'][key],rtol=0,atol=0)

    def test_missing_or_undeclared_source_encoder_is_rejected(self):
        runtime,payload=self.fixture(True)
        broken=copy.deepcopy(payload);broken['model'].pop('action_expert.source_time_encoder.0.weight')
        with self.assertRaises(RuntimeError):self.build(runtime,broken)
        broken=copy.deepcopy(payload);broken['config']['model']['action_expert']['source_physical_time_conditioning']=False
        with self.assertRaises(RuntimeError):self.build(runtime,broken)

    def test_source_time_flag_is_strict_boolean(self):
        self.assertFalse(_source_time_conditioning({}))
        self.assertTrue(_source_time_conditioning({'source_physical_time_conditioning':True}))
        for value in (1,0,'true',None):
            with self.assertRaises(TypeError):_source_time_conditioning({'source_physical_time_conditioning':value})

    def test_profile_ignores_only_the_declared_source_encoder_not_native_dimensions(self):
        action=dict(dim=16,ffn_dim=32,num_layers=1,state_dim=14,action_dim=14,chunk_size=16)
        wan=dict(precision='bfloat16',dim=16,ffn_dim=32,num_heads=2,num_layers=1,head_dim=8)
        motion=dict(history_count=4,temporal_contract='test',timestamp_source='test',spatial_unit='test',flow_contract={})
        runtime=dict(model=dict(action_expert=action,compact_wan=wan,absolute_motion=motion))
        checkpoint=dict(model=dict(initial_checkpoint='source.pt',action_expert=dict(action,source_physical_time_conditioning=True),compact_wan=wan,absolute_motion=motion))
        with patch('dynamicwam.inference.model_loader.validate_checkpoint_motion_metadata',side_effect=lambda x:x):
            _validate_model_config(runtime,checkpoint)
            checkpoint['model']['action_expert']['dim']=32
            with self.assertRaises(RuntimeError):_validate_model_config(runtime,checkpoint)

if __name__=='__main__':unittest.main()
