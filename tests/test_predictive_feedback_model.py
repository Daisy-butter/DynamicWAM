"""Predictive feedback CPU contracts; no CUDA work or downloads.

Run: python tests/test_predictive_feedback_model.py
Torch-free hosts still run the source/API tests; numerical skips are explicit.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import io
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/dynamicwam/models/predictive_feedback.py'
sys.path.insert(0, str(ROOT / 'src'))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from dynamicwam.models.plan_feedback import PlanFeedbackHead
    from dynamicwam.models.predictive_feedback import PredictiveFeedbackHead, LatentPredictor
    from dynamicwam.models.temporal_plan_feedback import remaining_window
    torch.set_num_threads(1)


def inputs(batch=3):
    return dict(rgb=torch.randn(batch, 3, 3, 32, 32),
        rgb_valid_mask=torch.ones(batch, 3, 32, 32, dtype=torch.bool),
        previous_head=torch.randn(batch, 3, 32, 32),
        previous_head_mask=torch.ones(batch, 32, 32, dtype=torch.bool),
        state=torch.randn(batch, 14), source_state=torch.randn(batch, 14),
        plan=torch.randn(batch, 16, 14), age_actions=torch.arange(batch) % 16,
        text=torch.randn(batch, 5, 4096), text_mask=torch.ones(batch, 5, dtype=torch.bool),
        history_dt=torch.full((batch,), .2), history_valid=torch.ones(batch, dtype=torch.bool),
        plan_age_seconds=torch.full((batch,), .5), proposal_path=torch.randn(batch, 2, 4, 28),
        proposal_dt=torch.tensor([.2, .4]).expand(batch, -1).clone())


class SourceContracts(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        self.head = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == 'PredictiveFeedbackHead')

    def test_policy_accepts_proposals_never_future_action_labels(self):
        for method in ('forward', 'forward_encoded'):
            forward = next(n for n in self.head.body if isinstance(n, ast.FunctionDef) and n.name == method)
            names = {a.arg for a in forward.args.args + forward.args.kwonlyargs}
            self.assertTrue({'proposal_path', 'proposal_dt'} <= names)
            self.assertTrue(names.isdisjoint({'action_path', 'future_dt', 'teacher_actions', 'target_actions'}))
            self.assertIsNone(forward.args.kwarg)

    def test_target_and_observation_encoders_are_no_grad(self):
        for name in ('encode_targets', 'encode_observations'):
            method = next(n for n in self.head.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn('torch.no_grad()', [ast.unparse(n) for n in method.decorator_list])

    def test_no_ema_or_target_network_updates(self):
        text = SOURCE.read_text(encoding='utf-8')
        self.assertNotIn('moving_average', text)
        self.assertNotIn('lerp_', text)


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable: numerical CPU checks not executed')
class NumericalContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261009)
        self.head = PredictiveFeedbackHead(dim=48, heads=4, layers=1)
        self.args = inputs()
        self.args['age_actions'] = torch.tensor([4, 8, 12])

    def legacy(self):
        old = PlanFeedbackHead(**self.head.constructor_spec)
        with torch.no_grad():
            old.output.weight.normal_(0, .15)
            old.output.bias.normal_(0, .03)
        return old

    def encoded(self):
        keys = ('rgb', 'rgb_valid_mask', 'previous_head', 'previous_head_mask', 'history_valid')
        return self.head.encode_observations(**{k:self.args[k] for k in keys})

    def predict(self, encoded=None, **overrides):
        arguments = dict(self.encoded() if encoded is None else encoded,
            history_dt=self.args['history_dt'], history_valid=self.args['history_valid'],
            action_path=self.args['proposal_path'], future_dt=self.args['proposal_dt'])
        arguments.update(overrides)
        return self.head.predict_latents(**arguments)

    def activate(self):
        with torch.no_grad():
            self.head.output.weight.normal_(0, .1)
            self.head.latent_predictor.output.weight.normal_(0, .03)
            self.head.future_readout.weight.normal_(0, .03)

    def test_v1_nonzero_migration_exact_and_fixed_memory(self):
        old = self.legacy().eval()
        report = self.head.initialize_from_plan_feedback(old.state_dict())
        args = inputs(4)
        args['age_actions'] = torch.tensor([0, 4, 8, 12])
        args['rgb_valid_mask'][:, 1, :8] = False
        allowed = inspect.signature(old.forward).parameters
        legacy_args = {k:v for k,v in args.items() if k in allowed}
        sizes = []
        hook = self.head.blocks[0].register_forward_pre_hook(lambda _, x: sizes.append(x[1].shape[1]))
        expected = old(**legacy_args)
        self.assertGreater(float(expected.abs().max()), .001)
        for history in (True, False):
            args['history_valid'].fill_(history)
            torch.testing.assert_close(self.head(**args), expected, rtol=0, atol=1e-6)
        hook.remove()
        self.assertEqual(sizes, [69, 69])
        self.assertEqual(report['preserved_ages'], [0, 4, 8, 12])
        self.assertFalse(report['age_extension_function_guarantee'])
        for key, value in old.state_dict().items():
            if key != 'age.weight':
                self.assertTrue(torch.equal(value, self.head.state_dict()[key]), key)
        self.head.train()
        self.assertFalse(self.head.vision.training)
        self.assertTrue(all(not p.requires_grad for p in self.head.vision.parameters()))

    def test_migration_failure_does_not_partially_modify_state(self):
        state = self.legacy().state_dict()
        state['position'] = state['position'].double()
        before = {k:v.clone() for k,v in self.head.state_dict().items()}
        with self.assertRaises(ValueError):
            self.head.initialize_from_plan_feedback(state)
        for key, value in self.head.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_one_four_view_cnn_and_targets_same_fixed_space(self):
        calls = []
        hook = self.head.vision.body.register_forward_pre_hook(lambda _, x:calls.append(tuple(x[0].shape)))
        aux = self.head(**self.args, return_aux=True)
        self.assertEqual(calls, [(12, 3, 32, 32)])
        hook.remove()
        target, valid = self.head.encode_targets(self.args['rgb'], self.args['rgb_valid_mask'])
        torch.testing.assert_close(target, aux['current_tokens'], rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(valid, aux['token_valid']))
        self.assertFalse(target.requires_grad)
        self.assertFalse(aux['current_tokens'].requires_grad)
        self.assertTrue(torch.equal(aux['predicted_latents'], target[:, None].expand(-1, 2, -1, -1)))

    def test_world_supervision_opens_predictor_without_encoder_gradient(self):
        encoded = self.encoded()
        target = encoded['current_tokens'][:, None] + torch.randn(3, 2, 48, 48) * .1
        optimizer = torch.optim.SGD(self.head.latent_predictor.parameters(), lr=1.)
        (self.predict(encoded)-target).square().mean().backward()
        self.assertGreater(float(self.head.latent_predictor.output.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.latent_predictor.action[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.head.vision.parameters()))
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        (self.predict(encoded)-target).square().mean().backward()
        for parameter in (self.head.latent_predictor.action[0].weight,
                          self.head.latent_predictor.history[0].weight,
                          self.head.latent_predictor.horizon[0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_preencoded_decode_and_final_reforecast_never_repeat_cnn(self):
        self.activate()
        args = dict(self.args)
        calls = []
        hook = self.head.vision.body.register_forward_pre_hook(lambda _, x:calls.append(x[0].shape[0]))
        aux = self.head(**args, return_aux=True)
        encoded = {k:aux[k] for k in ('current_tokens', 'previous_head_tokens', 'token_valid', 'previous_valid')}
        policy_args = {k:v for k,v in args.items() if k not in
                       ('rgb', 'rgb_valid_mask', 'previous_head', 'previous_head_mask')}
        torch.testing.assert_close(self.head.forward_encoded(**encoded, **policy_args), aux['residual'], rtol=0, atol=0)
        corrected = args['proposal_path'] + .1
        after = self.predict(encoded, action_path=corrected)
        self.assertGreater(float((after-aux['predicted_latents']).abs().max()), 1e-6)
        hook.remove()
        self.assertEqual(calls, [12])
        for key in encoded:
            if encoded[key].is_floating_point():
                self.assertFalse(encoded[key].requires_grad)

    def test_fixed_encoder_is_not_changed_by_outer_cpu_amp(self):
        expected = self.encoded()
        target, _ = self.head.encode_targets(self.args['rgb'], self.args['rgb_valid_mask'])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = self.encoded()
            target_amp, _ = self.head.encode_targets(self.args['rgb'], self.args['rgb_valid_mask'])
        self.assertTrue(torch.equal(target, target_amp))
        for key in ('current_tokens', 'previous_head_tokens'):
            self.assertEqual(actual[key].dtype, torch.float32)
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_two_stage_wrapper_cpu_parity_input_updates_and_ablations(self):
        from dynamicwam.inference.predictive_feedback_graph import PredictiveFeedbackGraph
        self.activate()
        self.head.eval().requires_grad_(False)
        engine = PredictiveFeedbackGraph(self.head, enabled=True)
        aux = engine.run_head(self.args)
        saved = aux['residual'].clone()
        expected = self.head(**self.args, return_aux=True)
        self.assertTrue(torch.equal(aux['residual'], expected['residual']))
        forecast_kwargs = dict(history_dt=self.args['history_dt'], history_valid=self.args['history_valid'],
            action_path=self.args['proposal_path']+.1, future_dt=self.args['proposal_dt'])
        future = engine.run_forecast(aux, **forecast_kwargs)
        direct = self.head.predict_latents(**{k:aux[k] for k in
            ('current_tokens','previous_head_tokens','token_valid','previous_valid')}, **forecast_kwargs)
        self.assertTrue(torch.equal(future, direct))
        changed = dict(self.args, rgb=self.args['rgb']+.3)
        updated = engine.run_head(changed)
        self.assertTrue(torch.equal(saved, aux['residual']))
        self.assertGreater(float((updated['residual']-saved).abs().max()), 0.)
        self.assertFalse(engine.stats['head']['graph_enabled'])
        self.assertEqual(engine.stats['forecast']['eager_calls'], 1)
        for ablation in ('nohistory','noaction','current_only','shuffle_future'):
            wrapper = PredictiveFeedbackGraph(self.head, enabled=False, ablation=ablation)
            observed = wrapper.run_head(self.args)
            expected = self.head(**self.args, ablation=ablation)
            self.assertTrue(torch.equal(observed['residual'], expected))

    def test_policy_opens_readout_then_reaches_pretrained_predictor(self):
        self.head.initialize_from_plan_feedback(self.legacy().state_dict())
        with torch.no_grad():
            self.head.latent_predictor.output.weight.normal_(0, .03)
        optimizer = torch.optim.SGD((p for p in self.head.parameters() if p.requires_grad), lr=1.)
        target = torch.randn(3, 4, 14)*.03
        (self.head(**self.args)-target).square().mean().backward()
        self.assertGreater(float(self.head.future_readout.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(self.head.latent_predictor.output.weight.grad.abs().sum()), 0.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        (self.head(**self.args)-target).square().mean().backward()
        self.assertGreater(float(self.head.latent_predictor.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.head.vision.parameters()))

    def test_ablation_independence_and_active_condition_sensitivity(self):
        self.activate()
        original = self.predict()
        path = self.args['proposal_path'] + 10
        self.assertGreater(float((original-self.predict(action_path=path)).abs().max()), 1e-5)
        self.assertTrue(torch.equal(self.predict(ablation='noaction'), self.predict(action_path=path, ablation='noaction')))
        encoded = self.encoded()
        alternate = dict(encoded, previous_head_tokens=encoded['previous_head_tokens'] + 10)
        self.assertTrue(torch.equal(self.predict(encoded, ablation='nohistory'), self.predict(alternate, ablation='nohistory')))
        invalid = torch.zeros(3, dtype=torch.bool)
        self.assertTrue(torch.equal(self.predict(encoded, history_valid=invalid), self.predict(alternate, history_valid=invalid)))
        changed = dict(self.args, proposal_path=path, proposal_dt=self.args['proposal_dt'] * 3)
        self.assertTrue(torch.equal(self.head(**self.args, ablation='current_only'), self.head(**changed, ablation='current_only')))
        self.assertGreater(float((original-self.predict(future_dt=self.args['proposal_dt']*3)).abs().max()), 1e-5)
        self.assertGreater(float((original-self.predict(ablation='shuffle_future')).abs().max()), 1e-5)

    def test_masks_and_output_safety_all_ages(self):
        self.activate()
        args = inputs(16)
        args['rgb_valid_mask'][2].zero_()
        out = self.head(**args)
        _, valid = remaining_window(args['plan'], args['age_actions'])
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(int(torch.count_nonzero(out[0])), 0)
        self.assertEqual(int(torch.count_nonzero(out[..., [6, 13]])), 0)
        self.assertEqual(int(torch.count_nonzero(out[~valid])), 0)
        self.assertLessEqual(float(out.abs().max()), .100001)
        encoded = self.encoded()
        encoded['token_valid'][:, 16:] = False
        a = self.predict(encoded)
        changed = dict(encoded, current_tokens=encoded['current_tokens'].clone())
        changed['current_tokens'][:, 16:] += 100
        b = self.predict(changed)
        self.assertTrue(torch.equal(a[:, :, :16], b[:, :, :16]))

    def test_validation_and_serialization(self):
        self.activate()
        for value in (torch.tensor([[0., .4]]).expand(3, -1),
                      torch.tensor([[.4, .2]]).expand(3, -1),
                      torch.full((3, 2), float('nan'))):
            with self.assertRaises(ValueError):
                self.predict(future_dt=value)
        with self.assertRaises(ValueError):
            self.predict(action_path=torch.zeros(3, 2, 4, 14))
        with self.assertRaises(ValueError):
            self.predict(history_dt=torch.zeros(3))
        with self.assertRaises(TypeError):
            self.head(**self.args, action_path=self.args['proposal_path'])
        self.head.validate_inputs(**self.args)
        expected = self.head(**self.args)
        self.assertTrue(torch.equal(expected, self.head(**self.args, validate_inputs=False)))
        stream = io.BytesIO()
        torch.save(dict(spec=self.head.constructor_spec, model=self.head.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        other = PredictiveFeedbackHead(**saved['spec'])
        other.load_state_dict(saved['model'], strict=True)
        self.assertTrue(torch.equal(expected, other(**self.args)))

    def test_missing_future_requires_explicit_mask_and_remains_unsupervised(self):
        self.activate()
        dt = self.args['proposal_dt'].clone()
        valid = torch.ones(3, 2, dtype=torch.bool)
        valid[0, 1] = False
        valid[1] = False
        dt[~valid] = 0
        with self.assertRaises(ValueError):
            self.predict(future_dt=dt)
        encoded = self.encoded()
        result = self.predict(encoded, future_dt=dt, future_valid=valid)
        current = encoded['current_tokens'][:, None].expand_as(result)
        self.assertTrue(torch.equal(result[~valid], current[~valid]))
        with self.assertRaises(ValueError):
            self.predict(future_dt=self.args['proposal_dt'], future_valid=valid)
        with self.assertRaises(ValueError):
            self.head(**dict(self.args, proposal_dt=dt))

    @unittest.skipUnless(os.environ.get('PREDICTIVE_TEST_CUDA') == '1', 'explicit CUDA opt-in required')
    def test_cuda_nonzero_graph_replay_and_fixed_encoder_space(self):
        from dynamicwam.inference.predictive_feedback_graph import PredictiveFeedbackGraph
        self.activate()
        self.head.cuda().eval().requires_grad_(False)
        args = {k:v.cuda() for k,v in self.args.items()}
        graph = PredictiveFeedbackGraph(self.head)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            first = graph.run_head(args)
            baseline = first['residual'].clone()
            forecast_kwargs = dict(history_dt=args['history_dt'],history_valid=args['history_valid'],
                action_path=args['proposal_path']+.1,future_dt=args['proposal_dt'])
            forecast = graph.run_forecast(first, **forecast_kwargs)
            direct = self.head.predict_latents(**{k:first[k] for k in
                ('current_tokens','previous_head_tokens','token_valid','previous_valid')}, **forecast_kwargs)
            torch.testing.assert_close(forecast, direct, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(baseline, self.head(**args), rtol=2e-5, atol=2e-5)
            for key in ('rgb', 'previous_head', 'state', 'proposal_path', 'proposal_dt'):
                changed = {k:v.clone() for k,v in args.items()}
                changed[key].add_(.1)
                observed = graph.run_head(changed)
                torch.testing.assert_close(observed['residual'], self.head(**changed), rtol=2e-5, atol=2e-5)
                altered_forecast = dict(forecast_kwargs, action_path=changed['proposal_path'],future_dt=changed['proposal_dt'])
                final = graph.run_forecast(observed, **altered_forecast)
                direct = self.head.predict_latents(**{k:observed[k] for k in
                    ('current_tokens','previous_head_tokens','token_valid','previous_valid')}, **altered_forecast)
                torch.testing.assert_close(final, direct, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(graph.run_head(args)['residual'], baseline, rtol=2e-5, atol=2e-5)
            self.assertEqual(graph.stats['head']['captures'], 1)
            self.assertEqual(graph.stats['forecast']['captures'], 1)
            aux = self.head(**args, return_aux=True)
            self.assertEqual(aux['current_tokens'].dtype, torch.float32)
        with torch.inference_mode():
            target, _ = self.head.encode_targets(args['rgb'], args['rgb_valid_mask'])
            torch.testing.assert_close(aux['current_tokens'], target, rtol=0, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
