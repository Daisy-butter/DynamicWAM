"""CPU optimizer regressions; no dataset, model checkpoint, CUDA or DS launch.

Load the production helpers directly from their AST to avoid importing WAN/data
dependencies. Optimizer tests use real Torch AdamW and its foreach kernel; only
the Accelerate/DeepSpeed boundary is simulated for ordering assertions.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import yaml

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/dynamicwam/training/train/causal_spatial_trainer.py'
NAMES = {'normalize_recipe_numbers', 'build_causal_spatial_optimizer', 'optimizer_step_with_norm'}
nodes = [node for node in ast.parse(SOURCE.read_text()).body
         if isinstance(node, ast.FunctionDef) and node.name in NAMES]
assert {node.name for node in nodes} == NAMES
helpers = dict(copy=copy, math=math, torch=torch)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), helpers)
normalize_recipe_numbers = helpers['normalize_recipe_numbers']
build_optimizer = helpers['build_causal_spatial_optimizer']
optimizer_step_with_norm = helpers['optimizer_step_with_norm']
spec = importlib.util.spec_from_file_location('causal_test_scheduler',
    ROOT / 'src/dynamicwam/training/utils/scheduler.py')
scheduler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler_module)


def recipes():
    for phase in ('interface_adapt', 'policy_pretrain', 'joint'):
        for suffix in ('', '_smoke'):
            path = ROOT / 'configs/causal_spatial' / (phase + suffix + '.yaml')
            yield path, yaml.safe_load(path.read_text())


class NumericRecipeTests(unittest.TestCase):
    def test_all_six_yaml_recipes_preserve_numeric_values(self):
        for path, recipe in recipes():
            with self.subTest(recipe=path.name):
                original = copy.deepcopy(recipe)
                normalized = normalize_recipe_numbers(recipe)
                expected = json.loads(path.read_text())
                self.assertEqual(normalized, expected)
                self.assertEqual(recipe, original)
                self.assertTrue(all(type(v) is float for v in normalized['lr'].values()))
                self.assertEqual(normalize_recipe_numbers(normalized), normalized)

    def test_quoted_numbers_are_normalized_and_metadata_untouched(self):
        _, recipe = next(recipes())
        recipe['steps'] = '2000'
        recipe['microbatch'] = '4'
        recipe['runtime_contract']['dt'] = '1.6e-1'
        recipe['model']['belief_grid'] = ['8', '8']
        recipe['loss']['future'] = '1e-1'
        fixed = normalize_recipe_numbers(recipe)
        self.assertEqual(fixed['steps'], 2000)
        self.assertEqual(fixed['runtime_contract']['dt'], .16)
        self.assertEqual(fixed['model']['belief_grid'], [8, 8])
        self.assertEqual(fixed['loss']['future'], .1)
        self.assertEqual(fixed['initial_checkpoint'], recipe['initial_checkpoint'])
        self.assertIs(fixed['diagnostic'], False)

    def test_invalid_numbers_fail_before_optimizer_construction(self):
        _, recipe = next(recipes())
        for value in ('nan', 'inf', '-1e-6', True, [], 'broken'):
            with self.subTest(value=value):
                bad = copy.deepcopy(recipe)
                bad['lr']['video'] = value
                with self.assertRaisesRegex(ValueError, 'lr.video'):
                    normalize_recipe_numbers(bad)
        for field, value in (('steps', 0), ('microbatch', 1.5), ('workers', -1)):
            bad = copy.deepcopy(recipe)
            bad[field] = value
            with self.assertRaisesRegex(ValueError, field):
                normalize_recipe_numbers(bad)


@unittest.skipUnless(torch is not None, 'Torch is required for real CPU optimizer steps')
class ActualOptimizerTests(unittest.TestCase):
    def tiny_model(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        for name in ('compact_wan', 'action_expert', 'absolute_motion_tokens'):
            setattr(model.backbone, name, torch.nn.Linear(2, 2, bias=False, dtype=torch.float64))
        model.future = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.5)
        return model

    def test_all_phases_real_foreach_initialization_and_training_update(self):
        for path, raw in recipes():
            with self.subTest(recipe=path.name):
                model = self.tiny_model()
                optimizer = build_optimizer(model, raw)
                normalized = normalize_recipe_numbers(raw)
                expected_groups = ['video', 'new_spatial_future'] if raw['phase'] == 'interface_adapt' else [
                    'video', 'action', 'motion', 'new_spatial_future']
                self.assertEqual([g['name'] for g in optimizer.param_groups], expected_groups)
                frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
                # DeepSpeed initializes optimizer state by stepping BEFORE the
                # training loop/scheduler. Force the failing foreach CPU path.
                for group in optimizer.param_groups:
                    self.assertIs(type(group['lr']), float)
                    group['foreach'] = True
                    for parameter in group['params']:
                        parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
                optimizer.zero_grad()
                schedule = scheduler_module.create_scheduler(optimizer, total_steps=normalized['steps'],
                    warmup_steps=normalized['warmup_steps'], min_lr_ratio=.01)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                for group, base_lr in zip(optimizer.param_groups, schedule.base_lrs, strict=True):
                    group['lr'] = schedule._lr_at(1, base_lr)
                    if normalized['warmup_steps']:
                        self.assertEqual(group['lr'], base_lr / normalized['warmup_steps'])
                sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
                accelerator = SimpleNamespace(distributed_type='NO', sync_gradients=True,
                    optimizer_step_was_skipped=False, clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
                norm = optimizer_step_with_norm(accelerator, model, optimizer)
                schedule.step()
                self.assertTrue(math.isfinite(norm) and norm > 0)
                self.assertEqual(schedule.step_count, 1)
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all())
                    if parameter.requires_grad:
                        self.assertFalse(torch.equal(parameter, before[name]), name)
                    else:
                        self.assertTrue(torch.equal(parameter, frozen[name]), name)

    def boundary(self, *, deepspeed, norm=2., skipped=False, sync=True, fallback=False):
        events = []
        model = self.tiny_model()
        accelerator = SimpleNamespace(distributed_type='DEEPSPEED' if deepspeed else 'NO',
            sync_gradients=sync, optimizer_step_was_skipped=skipped)
        def clip(*args):
            events.append('clip')
            return norm
        def read_norm():
            self.assertIn('step', events)
            events.append('engine_norm')
            return norm
        accelerator.clip_grad_norm_ = clip
        if fallback:
            accelerator.deepspeed_engine_wrapped = SimpleNamespace(engine=SimpleNamespace(get_global_grad_norm=read_norm))
        else:
            model.get_global_grad_norm = read_norm
        optimizer = SimpleNamespace(step=lambda: events.append('step'), zero_grad=lambda: events.append('zero'))
        return accelerator, model, optimizer, events

    def test_deepspeed_norm_only_after_step_with_engine_fallback(self):
        for fallback in (False, True):
            accelerator, model, optimizer, events = self.boundary(deepspeed=True, fallback=fallback)
            self.assertEqual(optimizer_step_with_norm(accelerator, model, optimizer), 2.)
            self.assertEqual(events, ['step', 'engine_norm', 'zero'])

    def test_non_deepspeed_clips_before_update(self):
        accelerator, model, optimizer, events = self.boundary(deepspeed=False)
        optimizer_step_with_norm(accelerator, model, optimizer)
        self.assertEqual(events, ['clip', 'step', 'zero'])

    def test_accumulation_does_not_query_unavailable_norm(self):
        accelerator, model, optimizer, events = self.boundary(deepspeed=True, norm=None, sync=False)
        self.assertIsNone(optimizer_step_with_norm(accelerator, model, optimizer))
        self.assertEqual(events, ['step', 'zero'])

    def test_missing_nonfinite_or_skipped_update_rejected(self):
        for norm, skipped, exception in ((None, False, RuntimeError), (float('nan'), False, FloatingPointError),
                                         (1., True, RuntimeError)):
            with self.subTest(norm=norm, skipped=skipped):
                accelerator, model, optimizer, events = self.boundary(deepspeed=True, norm=norm, skipped=skipped)
                with self.assertRaises(exception):
                    optimizer_step_with_norm(accelerator, model, optimizer)
                self.assertNotIn('zero', events)


if __name__ == '__main__':
    unittest.main()
