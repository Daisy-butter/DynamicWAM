"""CPU contracts; optional CUDA parity checks run only with an explicit flag.

python tests/test_temporal_feedback_model.py
TEMPORAL_TEST_CUDA=1 python tests/test_temporal_feedback_model.py
Isolated CPU validation can set TEMPORAL_MODEL_FILE and FEEDBACK_GRAPH_FILE.
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
from dynamicwam.models.plan_feedback import PlanFeedbackHead


def isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if os.environ.get('TEMPORAL_MODEL_FILE'):
    mod = isolated('isolated_temporal_feedback', os.environ['TEMPORAL_MODEL_FILE'])
    TemporalPlanFeedbackHead, remaining_window = mod.TemporalPlanFeedbackHead, mod.remaining_window
else:
    from dynamicwam.models.temporal_plan_feedback import TemporalPlanFeedbackHead, remaining_window
if os.environ.get('FEEDBACK_GRAPH_FILE'):
    FeedbackGraph = isolated('isolated_feedback_graph', os.environ['FEEDBACK_GRAPH_FILE']).FeedbackGraph
else:
    from dynamicwam.inference.feedback_graph import FeedbackGraph

torch.set_num_threads(1)


def inputs(batch=3, side=32):
    return dict(rgb=torch.randn(batch, 3, 3, side, side),
        rgb_valid_mask=torch.ones(batch, 3, side, side, dtype=torch.bool),
        state=torch.randn(batch, 14), source_state=torch.randn(batch, 14),
        plan=torch.randn(batch, 16, 14), age_actions=torch.tensor([1, 13, 15][:batch]) if batch <= 3 else torch.arange(batch) % 16,
        text=torch.randn(batch, 5, 4096), text_mask=torch.ones(batch, 5, dtype=torch.bool),
        previous_head=torch.randn(batch, 3, side, side),
        previous_head_mask=torch.ones(batch, side, side, dtype=torch.bool),
        history_dt=torch.full((batch,), .15), history_valid=torch.ones(batch, dtype=torch.bool),
        plan_age_seconds=torch.linspace(.15, 2., batch))


class TemporalFeedbackChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261006)
        self.head = TemporalPlanFeedbackHead(dim=48, heads=4, layers=1)
        self.args = inputs()

    def activate(self):
        with torch.no_grad():
            self.head.output.weight.normal_(0, .15)
            self.head.output.bias.normal_(0, .03)
            for branch in (self.head.temporal, self.head.time):
                branch[-1].weight.normal_(0, .02)
                branch[-1].bias.normal_(0, .01)

    def test_remaining_window_every_age_and_all_horizon_reconstruction(self):
        plan = torch.arange(16 * 14).float().reshape(1, 16, 14)
        for age in range(16):
            window, valid = remaining_window(plan, age)
            count = min(4, 16 - age)
            self.assertEqual(int(valid.sum()), count)
            self.assertTrue(torch.equal(window[0, :count], plan[0, age:age + count]))
        for horizon in (1, 2, 4):
            joined = torch.cat([remaining_window(plan, age)[0][:, :horizon]
                                for age in range(0, 16, horizon)], dim=1)
            self.assertTrue(torch.equal(joined, plan))
        for age in (-1, 16, 1.2, float('nan'), float('inf'), [1, 2]):
            with self.subTest(age=age), self.assertRaises(ValueError):
                remaining_window(plan, age)

    def test_zero_output_and_forbidden_channels(self):
        self.assertEqual(torch.count_nonzero(self.head(**self.args)), 0)
        self.activate()
        args = inputs(batch=16)
        out = self.head(**args)
        _, valid = remaining_window(args['plan'], args['age_actions'])
        self.assertEqual(tuple(out.shape), (16, 4, 14))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(torch.count_nonzero(out[0]), 0)
        self.assertEqual(torch.count_nonzero(out[..., [6, 13]]), 0)
        self.assertEqual(torch.count_nonzero(out[~valid]), 0)
        self.assertLessEqual(float(out.abs().max()), .100001)
        self.assertGreater(float(out[1:].abs().max()), 0.)

    def test_invalid_history_is_content_and_duration_independent(self):
        self.activate()
        self.args['history_valid'].zero_()
        before = self.head(**self.args)
        changed = copy.deepcopy(self.args)
        changed['previous_head'].normal_(10, 50)
        changed['previous_head_mask'].zero_()
        changed['history_dt'].fill_(99.)
        self.assertTrue(torch.equal(before, self.head(**changed)))

    def test_current_and_history_order_and_actual_time_are_used(self):
        self.activate()
        before = self.head(**self.args)
        for name in ('rgb', 'state', 'previous_head', 'history_dt', 'plan', 'text'):
            changed = copy.deepcopy(self.args)
            changed[name] = changed[name] + .4
            with self.subTest(name=name):
                self.assertGreater(float((before - self.head(**changed)).abs().max()), 1e-7)

    def test_unidentifiable_physical_plan_age_is_telemetry_only(self):
        self.activate()
        self.assertEqual(self.head.ARCHITECTURE_VERSION, 3)
        before = self.head(**self.args)
        changed = copy.deepcopy(self.args)
        changed['plan_age_seconds'].mul_(100).add_(30.)
        self.assertTrue(torch.equal(before, self.head(**changed)))

    def test_masked_fixed_text_padding_preserves_eager_semantics(self):
        self.activate()
        for length in (5, 17):
            args = copy.deepcopy(self.args)
            args['text'] = torch.randn(3, length, 4096)
            args['text_mask'] = torch.ones(3, length, dtype=torch.bool)
            expected = self.head(**args)
            padded = dict(args,
                text=torch.nn.functional.pad(args['text'], (0, 0, 0, 512-length)),
                text_mask=torch.nn.functional.pad(args['text_mask'], (0, 512-length), value=False))
            torch.testing.assert_close(self.head(**padded), expected, atol=2e-6, rtol=2e-5)

    def test_shared_visual_body_has_one_four_view_forward(self):
        shapes = []
        hook = self.head.vision.body.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
        self.head(**self.args)
        hook.remove()
        self.assertEqual(shapes, [(12, 3, 32, 32)])

    def test_temporal_and_vision_gradients_after_zero_output_opens(self):
        optim = torch.optim.SGD(self.head.parameters(), lr=1.)
        target = torch.randn(3, 4, 14) * .03
        out = self.head(**self.args)
        loss = (out - target).square().mean()
        loss.backward()
        self.assertGreater(float(self.head.output.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.temporal[0].weight.grad.abs().sum()), 0.)
        optim.step(); optim.zero_grad(set_to_none=True)
        (self.head(**self.args) - target).square().mean().backward()
        for name, parameter in [('temporal_last', self.head.temporal[-1].weight), ('time_last', self.head.time[-1].weight),
                                ('vision', self.head.vision.body[0].weight)]:
            with self.subTest(branch=name):
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.temporal[0].weight.grad.abs().sum()), 0.)
        optim.step(); optim.zero_grad(set_to_none=True)
        (self.head(**self.args) - target).square().mean().backward()
        self.assertGreater(float(self.head.temporal[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.head.time[0].weight.grad.abs().sum()), 0.)

    def legacy(self):
        old = PlanFeedbackHead(**self.head.constructor_spec)
        with torch.no_grad():
            old.output.weight.normal_(0, .15)
            old.output.bias.normal_(0, .03)
        return old

    def test_legacy_migration_preserves_nonzero_function_and_memory(self):
        old = self.legacy().eval()
        mapping = self.head.initialize_from_plan_feedback(old.state_dict())
        self.head.eval()
        args = inputs(batch=4)
        args['age_actions'] = torch.tensor([0, 4, 8, 12])
        args['rgb_valid_mask'][:, 1, :8] = False
        legacy_args = {key: value for key, value in args.items() if key not in
            ('previous_head', 'previous_head_mask', 'history_dt', 'history_valid', 'plan_age_seconds')}
        sizes = []
        hook = self.head.blocks[0].register_forward_pre_hook(lambda _, values: sizes.append(values[1].shape[1]))
        with torch.no_grad():
            expected = old(**legacy_args)
            self.assertGreater(float(expected.abs().max()), 1e-3)
            for enabled in (True, False):
                args['history_valid'].fill_(enabled)
                actual = self.head(**args)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.)
        hook.remove()
        self.assertEqual(sizes, [69, 69])
        self.assertEqual(mapping['preserved_ages'], [0, 4, 8, 12])
        self.assertFalse(mapping['age_extension_function_guarantee'])
        for key, value in old.state_dict().items():
            if key != 'age.weight':
                self.assertTrue(torch.equal(value, self.head.state_dict()[key]), key)
        for index, age in enumerate((0, 4, 8, 12)):
            self.assertTrue(torch.equal(self.head.age.weight[age], old.age.weight[index]))
        torch.testing.assert_close(self.head.age.weight[2], torch.lerp(old.age.weight[0], old.age.weight[1], .5))
        self.assertTrue(torch.equal(self.head.age.weight[15], old.age.weight[3]))

    def test_legacy_migration_rejects_partial_wrong_shape_or_dtype_before_copy(self):
        old = self.legacy().state_dict()
        bad = []
        state = dict(old);state.pop('position');bad.append(state)
        state = dict(old);state['unexpected'] = torch.zeros(1);bad.append(state)
        state = dict(old);state['age.weight'] = torch.zeros(16, 48);bad.append(state)
        state = dict(old);state['position'] = state['position'].double();bad.append(state)
        state = dict(old);state['position'] = torch.full_like(state['position'], float('nan'));bad.append(state)
        before = {key: value.clone() for key, value in self.head.state_dict().items()}
        for state in bad:
            with self.assertRaises(ValueError):
                self.head.initialize_from_plan_feedback(state)
            for key, value in self.head.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)

    def test_migrated_temporal_branch_opens_in_two_gradient_steps(self):
        self.head.initialize_from_plan_feedback(self.legacy().state_dict())
        optimizer = torch.optim.SGD(self.head.parameters(), lr=.1)
        target = torch.randn(3, 4, 14)*.03
        (self.head(**self.args)-target).square().mean().backward()
        self.assertGreater(float(self.head.temporal[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.head.time[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.temporal[0].weight.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.time[0].weight.grad.abs().sum()), 0.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        (self.head(**self.args)-target).square().mean().backward()
        self.assertGreater(float(self.head.temporal[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.head.time[0].weight.grad.abs().sum()), 0.)

    def test_validated_tensor_path_and_serialization(self):
        self.activate()
        expected = self.head(**self.args)
        self.assertTrue(torch.equal(expected, self.head(**self.args, validate_inputs=False)))
        stream = io.BytesIO()
        torch.save(dict(spec=self.head.constructor_spec, state=self.head.state_dict()), stream)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        other = TemporalPlanFeedbackHead(**payload['spec'])
        other.load_state_dict(payload['state'], strict=True)
        self.assertTrue(torch.equal(expected, other(**self.args)))

    def test_validation_rejects_nonfinite_or_impossible_history(self):
        cases = [('history_dt', torch.zeros(3)), ('history_dt', torch.full((3,), -1.)),
                 ('plan_age_seconds', torch.full((3,), float('nan'))),
                 ('age_actions', torch.tensor([0, 1, 16])),
                 ('previous_head_mask', torch.ones(3, 1, 32, 32, dtype=torch.bool)),
                 ('text_mask', torch.zeros(3, 5, dtype=torch.bool))]
        for key, value in cases:
            changed = dict(self.args, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.head(**changed)

    def test_cpu_wrapper_is_eager_and_updates_inputs_without_mutation(self):
        self.activate()
        self.head.eval().requires_grad_(False)
        graph = FeedbackGraph(self.head, enabled=True)
        plan = self.args['plan'].clone()
        first = graph(**self.args)
        saved = first.clone()
        changed = copy.deepcopy(self.args)
        changed['previous_head'].add_(1)
        second = graph(**changed)
        self.assertTrue(torch.equal(first, saved))
        self.assertTrue(torch.equal(first, self.head(**self.args)))
        self.assertGreater(float((first-second).abs().max()), 0.)
        self.assertTrue(torch.equal(self.args['plan'], plan))
        self.assertEqual(graph.stats['captures'], 0)
        self.assertEqual(graph.stats['eager_calls'], 2)
        self.assertFalse(graph.stats['graph_enabled'])

    @unittest.skipUnless(os.environ.get('TEMPORAL_TEST_CUDA') == '1', 'explicit GPU test opt-in required')
    def test_cuda_static_replay_matches_eager_after_all_inputs_change(self):
        self.activate()
        self.head.cuda().eval().requires_grad_(False)
        args = {key: value.cuda() for key, value in self.args.items()}
        graph = FeedbackGraph(self.head)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            first = graph(**args)
            saved = first.clone()
            for key in ('rgb', 'state', 'plan', 'text', 'previous_head', 'history_dt', 'plan_age_seconds'):
                changed = {k: v.clone() for k, v in args.items()}
                changed[key].add_(.25)
                expected = self.head(**changed)
                torch.testing.assert_close(graph(**changed), expected, atol=2e-5, rtol=2e-5)
            changed = {k: v.clone() for k, v in args.items()}
            changed['age_actions'].fill_(14)
            changed['history_valid'].zero_()
            changed['rgb_valid_mask'][:, 1].zero_()
            changed['previous_head_mask'].zero_()
            changed['text_mask'][:, -2:].zero_()
            torch.testing.assert_close(graph(**changed), self.head(**changed), atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(graph(**args), saved, atol=2e-5, rtol=2e-5)
            self.assertTrue(torch.equal(first, saved))
            self.assertEqual(graph.stats['captures'], 1)

    @unittest.skipUnless(os.environ.get('TEMPORAL_TEST_CUDA') == '1', 'explicit GPU test opt-in required')
    def test_cuda_legacy_migration_retains_supported_age_function(self):
        old = self.legacy().eval()
        self.head.initialize_from_plan_feedback(old.state_dict())
        old.cuda().requires_grad_(False)
        self.head.cuda().eval().requires_grad_(False)
        args = {key:value.cuda() for key,value in inputs(batch=4).items()}
        args['age_actions'] = torch.tensor([0,4,8,12],device='cuda')
        legacy_args = {key:value for key,value in args.items() if key not in
            ('previous_head','previous_head_mask','history_dt','history_valid','plan_age_seconds')}
        errors=[]
        with torch.inference_mode():
            expected=old(**legacy_args)
            for valid in (True,False):
                args['history_valid'].fill_(valid)
                actual=self.head(**args)
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0.)
            # Mathematical preservation is tested above in FP32. Different
            # BF16 kernel shapes can round differently; quantify separately,
            # never turn this diagnostic into a bitwise-equivalence claim.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                expected=old(**legacy_args)
                for valid in (True,False):
                    args['history_valid'].fill_(valid)
                    actual=self.head(**args)
                    self.assertTrue(torch.isfinite(actual).all())
                    errors.append(float((actual-expected).abs().max()))
        print('legacy_migration_bf16_max_error',max(errors))

    @unittest.skipUnless(os.environ.get('TEMPORAL_TEST_CUDA') == '1', 'explicit GPU test opt-in required')
    def test_cuda_variable_instructions_pad_to_one_graph_signature(self):
        self.activate()
        self.head.cuda().eval().requires_grad_(False)
        args = {key: value.cuda() for key, value in self.args.items()}
        longer = dict(args, text=torch.randn(3, 17, 4096, device='cuda'),
            text_mask=torch.ones(3, 17, device='cuda', dtype=torch.bool))
        graph = FeedbackGraph(self.head)
        outputs, errors = [], []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for current in (args, longer, args):
                length = current['text'].shape[1]
                padded = dict(current,
                    text=torch.nn.functional.pad(current['text'], (0, 0, 0, 512-length)),
                    text_mask=torch.nn.functional.pad(current['text_mask'], (0, 512-length), value=False))
                unpadded_eager, padded_eager = self.head(**current), self.head(**padded)
                errors.append(float((unpadded_eager-padded_eager).abs().max()))
                # CPU FP32 separately proves masked-padding semantics. BF16
                # shape-dependent rounding is a diagnostic, not the graph gate.
                self.assertTrue(torch.isfinite(padded_eager).all())
                actual = graph(**padded)
                torch.testing.assert_close(actual, padded_eager, atol=2e-5, rtol=2e-5)
                outputs.append(actual.clone())
            torch.testing.assert_close(outputs[0], outputs[2], atol=2e-5, rtol=2e-5)
            self.assertGreater(float((outputs[0]-outputs[1]).abs().max()), 1e-7)
            self.assertEqual(graph.stats['captures'], 1)
            self.assertEqual(graph.stats['replays'], 3)
        print('fixed_text_padding_bf16_max_error', max(errors))


if __name__ == '__main__':
    unittest.main(verbosity=2)
