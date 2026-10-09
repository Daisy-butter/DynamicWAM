#!/usr/bin/env python3
"""CPU behavioral checks for bounded feedback around an immutable 16-row plan.

No WAN construction, checkpoint download, simulator or CUDA initialization.
Set PLAN_FEEDBACK_MODULE to test an isolated head file against existing source.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if os.environ.get('PLAN_FEEDBACK_MODULE'):
    spec = importlib.util.spec_from_file_location('isolated_plan_feedback', os.environ['PLAN_FEEDBACK_MODULE'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PlanFeedbackHead, plan_window = module.PlanFeedbackHead, module.plan_window
else:
    from dynamicwam.models.plan_feedback import PlanFeedbackHead, plan_window

torch.set_num_threads(1)


class PlanFeedbackChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)
        self.model = PlanFeedbackHead(dim=48, heads=4, layers=1, residual_limit=.1)
        self.args = dict(rgb=torch.randn(4, 3, 3, 32, 32),
            rgb_valid_mask=torch.ones(4, 3, 32, 32, dtype=torch.bool),
            state=torch.randn(4, 14), source_state=torch.randn(4, 14),
            plan=torch.randn(4, 16, 14), age_actions=torch.tensor([0, 4, 8, 12]),
            text=torch.randn(4, 5, 4096),
            text_mask=torch.tensor([[True, True, True, False, False]]).expand(4, -1).clone())

    def activate_output(self, scale=.1):
        with torch.no_grad():
            self.model.output.weight.normal_(0, scale)
            self.model.output.bias.normal_(0, scale)

    def test_zero_feedback_recovers_each_original_plan_window(self):
        residual = self.model(**self.args)
        self.assertEqual(tuple(residual.shape), (4, 4, 14))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        window = plan_window(self.args['plan'], self.args['age_actions'])
        for row, age in enumerate([0, 4, 8, 12]):
            self.assertTrue(torch.equal(window[row] + residual[row], self.args['plan'][row, age:age+4]))
        # All four deployment windows reconstruct one immutable original plan.
        plan = self.args['plan'][:1]
        restored = torch.cat([plan_window(plan, age) for age in (0, 4, 8, 12)], dim=1)
        self.assertTrue(torch.equal(restored, plan))

    def test_plan_window_scalar_broadcast_and_distinct_batch_ages(self):
        plan = torch.arange(4 * 16 * 14).reshape(4, 16, 14)
        self.assertTrue(torch.equal(plan_window(plan, 12), plan[:, 12:16]))
        result = plan_window(plan, torch.tensor([0, 4, 8, 12]))
        for b, age in enumerate([0, 4, 8, 12]):
            self.assertTrue(torch.equal(result[b], plan[b, age:age+4]))

    def test_invalid_age_or_plan_never_wraps_or_pads(self):
        for age in (-4, 1, 2, 13, 16, 4.5, float('nan'), float('inf'), [0, 4], [0, 4, 8, -4]):
            with self.subTest(age=age), self.assertRaises(ValueError):
                plan_window(self.args['plan'], age)
        for shape in ((4, 15, 14), (4, 16, 13), (16, 14)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                plan_window(torch.zeros(shape), 0)
        with self.assertRaises(ValueError):
            plan_window(self.args['plan'], 0, horizon=8)

    def test_age_zero_grippers_and_residual_cap_remain_hard_constraints(self):
        self.activate_output(scale=20)
        residual = self.model(**self.args)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.equal(residual[0], torch.zeros_like(residual[0])))
        self.assertTrue(torch.equal(residual[..., [6, 13]], torch.zeros_like(residual[..., [6, 13]])))
        self.assertLessEqual(float(residual.abs().max()), .100001)
        self.assertGreater(float(residual[1:].abs().max()), .09)
        action = plan_window(self.args['plan'], self.args['age_actions']) + residual
        self.assertTrue(torch.equal(action[..., [6, 13]], plan_window(self.args['plan'], self.args['age_actions'])[..., [6, 13]]))

    def test_output_then_visual_gradients_follow_actual_feedback_loss(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.5)
        target = torch.randn(4, 4, 14) * .03
        target[0] = 0; target[..., [6, 13]] = 0
        prediction = self.model(**self.args)
        loss = (prediction - target).square().mean()
        loss.backward()
        self.assertGreater(float(self.model.output.weight.grad.abs().sum()), 0)
        first_visual = sum(float(p.grad.abs().sum()) for p in self.model.vision.parameters() if p.grad is not None)
        self.assertEqual(first_visual, 0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        loss_after = (self.model(**self.args) - target).square().mean()
        loss_after.backward()
        visual = sum(float(p.grad.abs().sum()) for p in self.model.vision.parameters() if p.grad is not None)
        self.assertGreater(visual, 0)
        self.assertGreater(float(self.model.plan.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.state[0].weight.grad.abs().sum()), 0)
        self.assertLess(float(loss_after.detach()), float(loss.detach()))

    def test_current_rgb_state_plan_and_language_change_feedback(self):
        self.activate_output(); self.model.eval()
        baseline = self.model(**self.args)
        changes = {
            'rgb': lambda v: v.flip(-1) + .7,
            'state': lambda v: v + .4,
            'source_state': lambda v: v - .4,
            'plan': lambda v: v.roll(2, dims=1) + .3,
            'text': lambda v: v + .3,
        }
        for field, change in changes.items():
            args = dict(self.args);args[field] = change(args[field])
            delta = float((self.model(**args)[1:] - baseline[1:]).abs().max())
            with self.subTest(field=field):
                self.assertGreater(delta, 1e-7)

    def test_masked_pixels_and_text_padding_do_not_change_feedback(self):
        self.activate_output(); self.model.eval()
        args=copy.deepcopy(self.args)
        args['rgb_valid_mask'][:, :, :, :12] = False
        baseline=self.model(**args)
        changed=copy.deepcopy(args)
        changed['rgb'][:, :, :, :, :12] = 1000
        changed['text'][:, 3:] = 1000
        self.assertTrue(torch.equal(self.model(**changed), baseline))
        # Entirely missing camera views cannot contaminate outputs or produce NaN.
        args['rgb_valid_mask'].fill_(False)
        self.assertTrue(torch.isfinite(self.model(**args)).all())

    def test_forward_does_not_mutate_source_plan_or_observations(self):
        self.activate_output()
        original={k:v.clone() for k,v in self.args.items()}
        self.model(**self.args)
        for k,v in self.args.items():
            with self.subTest(field=k):
                self.assertTrue(torch.equal(v, original[k]))

    def test_checkpoint_strict_roundtrip_and_missing_key_rejection(self):
        self.activate_output();self.model.eval()
        buffer=io.BytesIO()
        torch.save(dict(spec=self.model.constructor_spec,model=self.model.state_dict()),buffer)
        buffer.seek(0);payload=torch.load(buffer,map_location='cpu',weights_only=True)
        restored=PlanFeedbackHead(**payload['spec']).eval()
        restored.load_state_dict(payload['model'],strict=True)
        self.assertTrue(torch.equal(restored(**self.args),self.model(**self.args)))
        damaged=dict(payload['model']);damaged.pop('output.weight')
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(damaged,strict=True)
        extra=dict(payload['model']);extra['unexpected']=torch.zeros(1)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(extra,strict=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
