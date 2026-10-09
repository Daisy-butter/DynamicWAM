#!/usr/bin/env python3
"""CPU-only v1-to-v3 initialization identity, function and gradient checks."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
tree = ast.parse((ROOT/'scripts/train_temporal_feedback.py').read_text(encoding='utf-8'))
scope = {}
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='validate_plan_initialization')
exec(compile(ast.Module(body=[function], type_ignores=[]), '<trainer initialization validator>', 'exec'), scope)
validate_plan_initialization = scope['validate_plan_initialization']


def fixture():
    spec = dict(dim=24, heads=6, layers=1, residual_limit=.1)
    schedule = dict(action_steps=10, video_steps=2)
    contract = dict(source_plan='immutable_official_full_16', execute_horizon=4,
        refresh_actions=16, ages=[0,4,8,12], age0_correction=False, gripper_correction=False,
        residual_units='normalized_qpos', residual_limit=.1,
        preprocess='three_view_160_imagenet_mask_v1', schedule=schedule,
        precision=dict(video='bfloat16',action='float32',solver='float32',
            feedback_parameters='float32',feedback_autocast='bfloat16'))
    payload = dict(format='dynamicwam_plan_feedback_checkpoint',version=1,
        dataset_identity='dataset',pairing_identity='old_partition_hash',
        action_normalization=dict(stats_sha256='normalizer'),runtime_contract=contract,spec=spec,
        base_checkpoint=dict(path='/old/mount/full.pt',sha256='a'*64),
        global_step=2000,model={'placeholder':1})
    expected = dict(base_identity=dict(path='/current/mount/full.pt',sha256='a'*64),
        dataset_identity='dataset',legacy_pairing_identity='old_partition_hash',
        normalization=payload['action_normalization'],schedule=SimpleNamespace(to_config=lambda:schedule),spec=spec)
    return payload,expected


class ExportIdentityTests(unittest.TestCase):
    def test_same_content_different_official_mount_is_allowed(self):
        payload,expected=fixture()
        validate_plan_initialization(payload,**expected)

    def test_invalid_identity_rejected(self):
        payload,expected=fixture()
        changes=dict(format='other',version=2,dataset_identity='other',pairing_identity='other',
            action_normalization={},runtime_contract={},spec={},global_step=0,model={})
        for key,value in changes.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                altered=copy.deepcopy(payload);altered[key]=value
                validate_plan_initialization(altered,**expected)
        with self.assertRaises(ValueError):
            payload['base_checkpoint']['sha256']='b'*64
            validate_plan_initialization(payload,**expected)

    def test_init_flags_are_mutually_exclusive(self):
        source=(ROOT/'scripts/train_temporal_feedback.py').read_text(encoding='utf-8')
        self.assertIn('initialization = parser.add_mutually_exclusive_group()',source)
        self.assertIn('initialization.add_argument("--init-feedback"',source)
        self.assertIn('initialization.add_argument("--init-plan-feedback"',source)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable in this CPU interpreter')
class TensorMigrationTests(unittest.TestCase):
    def setUp(self):
        from dynamicwam.models.plan_feedback import PlanFeedbackHead
        from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead
        torch.set_num_threads(2);torch.manual_seed(918)
        self.old=PlanFeedbackHead(dim=24,heads=6,layers=1).eval()
        # Nonzero original output is essential: an untrained zero head would
        # make a broken copying/attention test pass trivially.
        torch.nn.init.normal_(self.old.output.weight,std=.02)
        torch.nn.init.normal_(self.old.output.bias,std=.01)
        self.new=TemporalPlanFeedbackHead(dim=24,heads=6,layers=1).eval()
        self.mapping=self.new.initialize_from_plan_feedback(self.old.state_dict())
        b=4
        self.inputs=dict(rgb=torch.randn(b,3,3,32,32),rgb_valid_mask=torch.ones(b,3,32,32,dtype=torch.bool),
            state=torch.randn(b,14),source_state=torch.randn(b,14),plan=torch.randn(b,16,14),
            age_actions=torch.tensor([0,4,8,12]),text=torch.randn(b,4,4096),
            text_mask=torch.ones(b,4,dtype=torch.bool),previous_head=torch.randn(b,3,32,32),
            previous_head_mask=torch.ones(b,32,32,dtype=torch.bool),history_dt=torch.tensor([.06,.18,.7,1.8]),
            history_valid=torch.ones(b,dtype=torch.bool),plan_age_seconds=torch.arange(b).float())

    def test_old_function_preserved_with_real_history_present(self):
        oldkeys=('rgb','rgb_valid_mask','state','source_state','plan','age_actions','text','text_mask')
        with torch.no_grad():
            expected=self.old(**{k:self.inputs[k] for k in oldkeys})
            actual=self.new(**self.inputs)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
            self.inputs['history_valid'].fill_(False)
            torch.testing.assert_close(self.new(**self.inputs),expected,rtol=1e-5,atol=1e-7)
        json.dumps(self.mapping,allow_nan=False)

    def test_age_anchors_and_tail_extension(self):
        for oldrow,age in enumerate((0,4,8,12)):
            torch.testing.assert_close(self.new.age.weight[age],self.old.age.weight[oldrow],rtol=0,atol=0)
        for age in (13,14,15):
            torch.testing.assert_close(self.new.age.weight[age],self.old.age.weight[3],rtol=0,atol=0)
        torch.testing.assert_close(self.new.age.weight[2],(self.old.age.weight[0]+self.old.age.weight[1])*.5)

    def test_new_zero_output_branches_have_trainable_gradients(self):
        self.new.train()
        actual=self.new(**self.inputs)
        loss=(actual-.025).square().mean()
        loss.backward()
        for branch in (self.new.temporal,self.new.time):
            self.assertIsNotNone(branch[-1].weight.grad)
            self.assertTrue(torch.isfinite(branch[-1].weight.grad).all())
            self.assertGreater(float(branch[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.dtype==torch.float32 for p in self.new.parameters()))
        self.assertTrue(all(p.grad is not None for p in self.new.parameters()))

    def test_state_structure_dtype_and_finiteness_are_strict(self):
        base=self.old.state_dict()
        for modification in ('missing','unexpected','age_shape','dtype','nan'):
            bad={k:v.clone() for k,v in base.items()}
            if modification=='missing':bad.pop('plan.bias')
            elif modification=='unexpected':bad['unexpected']=torch.ones(1)
            elif modification=='age_shape':bad['age.weight']=torch.zeros(16,24)
            elif modification=='dtype':bad['plan.weight']=bad['plan.weight'].double()
            else:bad['plan.weight'].view(-1)[0]=float('nan')
            with self.subTest(change=modification),self.assertRaises((ValueError,RuntimeError,TypeError)):
                self.new.initialize_from_plan_feedback(bad)


if __name__=='__main__':
    unittest.main(verbosity=2)
