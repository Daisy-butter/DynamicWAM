import copy
import unittest

import torch

from dynamicwam.dualrate.fast import CurrentObservationEncoder, FastConfig, FastJointWAM


class FastJointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.cfg = FastConfig(dim=24, layers=2, heads=3, chunk_size=4, future_tokens=4, future_dim=10, observation_dim=12, world_dim=12, test_fixture=True)
        self.model = FastJointWAM(self.cfg)
        self.a = torch.randn(2, 4, 14)
        self.z = torch.randn(2, 4, 10)
        self.sigma = torch.tensor([0.2, 0.8])
        self.kw = dict(
            observation_tokens=torch.randn(2, 3, 12), observation_valid=torch.ones(2, 3, dtype=torch.bool),
            world_tokens=torch.randn(2, 2, 12), world_valid=torch.ones(2, 2, dtype=torch.bool),
            world_intervals=torch.tensor([[[10.0, 11.0], [11.0, 12.0]]] * 2, dtype=torch.float64),
            target_times=torch.tensor([[10.1, 10.2, 10.3, 10.4, 10.1, 10.2, 10.4, 10.8]] * 2, dtype=torch.float64),
            query_time=torch.full((2,), 10.0, dtype=torch.float64),
            source_time=torch.full((2,), 9.5, dtype=torch.float64),
            completed_time=torch.full((2,), 9.8, dtype=torch.float64),
        )

    def run_model(self, **overrides):
        return self.model(self.a, self.z, self.sigma, **dict(self.kw, **overrides))

    def test_output_and_joint_gradients(self):
        self.z.requires_grad_()
        self.kw["observation_tokens"].requires_grad_()
        self.kw["world_tokens"].requires_grad_()
        output = self.run_model()
        self.assertEqual(output["contact_logits"].shape, (2, 4, 2))
        output["action_velocity"].square().sum().backward()
        for value in (self.z, self.kw["observation_tokens"], self.kw["world_tokens"]):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        # Uncovered world token cannot influence any target, including indirectly.
        self.assertEqual(self.kw["world_tokens"].grad[:, 1].abs().sum().item(), 0)

    def test_actions_depend_on_future_and_latest_observations(self):
        base = self.run_model()["action_velocity"]
        altered_z = self.model(self.a, self.z + 1, self.sigma, **self.kw)["action_velocity"]
        altered_obs = self.run_model(observation_tokens=self.kw["observation_tokens"] + 1)["action_velocity"]
        self.assertFalse(torch.allclose(base, altered_z))
        self.assertFalse(torch.allclose(base, altered_obs))

    def test_expired_and_unavailable_world_has_no_influence(self):
        for kwargs in (
            dict(world_intervals=self.kw["world_intervals"] - 10),
            dict(completed_time=self.kw["query_time"] + 1),
            dict(world_valid=torch.zeros(2, 2, dtype=torch.bool)),
        ):
            before = self.run_model(**kwargs)
            after = self.run_model(**dict(kwargs, world_tokens=torch.full((2, 2, 12), float("nan"))))
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    def test_empty_and_fully_masked_context_finite(self):
        for size in (0, 2):
            output = self.run_model(
                observation_tokens=torch.empty(2, 0, 12), observation_valid=torch.empty(2, 0, dtype=torch.bool),
                world_tokens=torch.full((2, size, 12), float("nan")), world_valid=torch.zeros(2, size, dtype=torch.bool),
                world_intervals=torch.zeros(2, size, 2, dtype=torch.float64),
            )
            self.assertTrue(all(torch.isfinite(x).all() for x in output.values()))

    def test_prefix_is_conditioned_and_zero_velocity(self):
        mask = torch.tensor([[1, 1, 0, 0]] * 2, dtype=torch.bool)
        known = torch.randn_like(self.a)
        first = self.run_model(action_known_mask=mask, known_actions=known)
        second = self.model(self.a + mask[..., None] * 50, self.z, self.sigma, **self.kw, action_known_mask=mask, known_actions=known)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertEqual(first["action_velocity"][:, :2].abs().sum().item(), 0)
        changed = self.run_model(action_known_mask=mask, known_actions=known + 3)
        self.assertFalse(torch.allclose(first["future_velocity"], changed["future_velocity"]))

    def test_known_missing_slots_do_not_contaminate_actions(self):
        mask = torch.tensor([[1, 0, 1, 0]] * 2, dtype=torch.bool)
        first = self.run_model(future_valid=mask)
        z = torch.where(mask[..., None], self.z, torch.full_like(self.z, float("nan")))
        second = self.model(self.a, z, self.sigma, **self.kw, future_valid=mask)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_clock_origin_invariance_and_float64_enforced(self):
        baseline = self.run_model()
        shift = 100000000.0
        shifted = {k: v + shift for k, v in self.kw.items() if k in ("query_time", "source_time", "completed_time", "target_times", "world_intervals")}
        result = self.run_model(**shifted)
        for name in baseline:
            torch.testing.assert_close(baseline[name], result[name], rtol=1e-5, atol=2e-6)
        with self.assertRaises(TypeError):
            self.run_model(query_time=self.kw["query_time"].float())

    def test_strict_state_reload(self):
        other = FastJointWAM(self.cfg)
        other.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=True)
        actual = other(self.a, self.z, self.sigma, **self.kw)
        expected = self.run_model()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

    def test_context_cache_equivalence_and_full_gradients(self):
        other = copy.deepcopy(self.model)
        one = {k: v.detach().clone() for k, v in self.kw.items()}
        two = {k: v.detach().clone() for k, v in self.kw.items()}
        for args in (one, two):
            args["observation_tokens"].requires_grad_()
            args["world_tokens"].requires_grad_()
        cached = other.prepare_context(**two)
        # Two forwards share one differentiable KV computation and one backward.
        lhs = [self.model(self.a, self.z, s, **one) for s in (self.sigma, self.sigma / 2)]
        rhs = [other.forward_cached(self.a, self.z, s, cached) for s in (self.sigma, self.sigma / 2)]
        for actual, expected in zip(lhs, rhs):
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        sum(v["action_velocity"].square().sum() for v in lhs).backward()
        sum(v["action_velocity"].square().sum() for v in rhs).backward()
        for key in ("observation_tokens", "world_tokens"):
            torch.testing.assert_close(one[key].grad, two[key].grad, rtol=2e-5, atol=1e-6)
        for a, b in zip(self.model.parameters(), other.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6)

    def test_cache_rejects_mutated_conditions_parameters_and_other_model(self):
        cache = self.model.prepare_context(**self.kw)
        with self.assertRaises(ValueError):
            copy.deepcopy(self.model).forward_cached(self.a, self.z, self.sigma, cache)
        with self.assertRaises(ValueError):
            self.model(self.a, self.z, self.sigma, context_cache=cache,
                **dict(self.kw, query_time=self.kw["query_time"].clone()))
        self.kw["world_valid"][0, 0] = False
        with self.assertRaises(ValueError):
            self.model.forward_cached(self.a, self.z, self.sigma, cache)
        cache = self.model.prepare_context(**self.kw)
        with torch.no_grad():
            self.model.action_in.weight.add_(.1)
        with self.assertRaises(ValueError):
            self.model.forward_cached(self.a, self.z, self.sigma, cache)

    def test_inference_tensor_cache_fails_closed(self):
        with torch.inference_mode():
            kw = {k: v.clone() for k, v in self.kw.items()}
            with self.assertRaises(ValueError):
                self.model.prepare_context(**kw)

    def test_world_types_allow_history_but_reject_expired_future(self):
        intervals = self.kw["world_intervals"] - 1
        history = torch.zeros(2, 2, dtype=torch.long)
        cache = self.model.prepare_context(**dict(self.kw, world_types=history, world_intervals=intervals))
        self.assertTrue(cache.slow_allowed[..., 1].all())
        # type=1 expired first interval cannot be read; type=2 historical action
        # remains causally relevant to future targets from this source plan.
        types = torch.tensor([[1, 2]] * 2)
        cache = self.model.prepare_context(**dict(self.kw, world_types=types,
            world_intervals=torch.tensor([[[8., 9.], [9.5, 9.7]]] * 2, dtype=torch.float64)))
        self.assertFalse(cache.slow_allowed[..., 1].any())
        self.assertTrue(cache.slow_allowed[..., 2].all())

    def test_action_deviation_only_affects_valid_soft_actions(self):
        types = torch.tensor([[2, 1]] * 2)
        delta = torch.randn(2, 2, 14, requires_grad=True)
        valid = torch.tensor([[1, 0]] * 2, dtype=torch.bool)
        base = self.run_model(world_types=types, world_action_delta=delta, world_action_delta_valid=valid)
        base["action_velocity"].square().sum().backward()
        self.assertGreater(delta.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(delta.grad[:, 1].abs().sum().item(), 0)
        changed = delta.detach().clone()
        changed[:, 1] = float("nan")
        actual = self.run_model(world_types=types, world_action_delta=changed, world_action_delta_valid=valid)
        torch.testing.assert_close(actual["action_velocity"], base["action_velocity"], rtol=0, atol=0)
        self.assertGreater(self.model.blocks[0].slow_gate.bias.grad.abs().sum().item(), 0)

    def test_production_requires_explicit_typed_context(self):
        model = FastJointWAM(FastConfig(**dict(vars(self.cfg), test_fixture=False)))
        with self.assertRaises(ValueError):
            model(self.a, self.z, self.sigma, **self.kw)


class CurrentEncoderTests(unittest.TestCase):
    def test_native_bfloat_language_cast_and_acceleration_mask(self):
        torch.manual_seed(13)
        model = CurrentObservationEncoder(dim=24, text_dim=16, test_fixture=True)
        rgb, motion, state = torch.rand(1, 3, 3, 32, 32), torch.randn(1, 4, 12), torch.randn(1, 14)
        text = torch.randn(1, 2, 16).bfloat16()
        args = dict(query_time=torch.tensor([5.0], dtype=torch.float64),
            rgb_times=torch.full((1, 3), 5.0, dtype=torch.float64),
            motion_times=torch.tensor([[[4.0, 4.1], [4.1, 4.2], [4.2, 4.3], [4.3, 4.4]]], dtype=torch.float64),
            motion_valid=torch.ones(1, 4, dtype=torch.bool), text_valid=torch.ones(1, 2, dtype=torch.bool),
            motion_acceleration_valid=torch.tensor([[0, 0, 1, 1]], dtype=torch.bool))
        expected, _ = model(rgb, motion, state, text.float(), **args)
        actual, _ = model(rgb, motion, state, text, **args)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        motion[:, :2, 9:12] = float("nan")
        masked, _ = model(rgb, motion, state, text, **args)
        torch.testing.assert_close(masked, actual, rtol=0, atol=0)

    def test_spatial_tokens_timestamps_masks_and_gradients(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        model = CurrentObservationEncoder(dim=24, text_dim=16, test_fixture=True)
        rgb = torch.rand(1, 3, 3, 64, 64, requires_grad=True)
        motion = torch.randn(1, 4, 12, requires_grad=True)
        args = dict(
            query_time=torch.tensor([5.0], dtype=torch.float64),
            rgb_times=torch.tensor([[5.0, 4.9, 5.1]], dtype=torch.float64),
            motion_times=torch.tensor([[[4.0, 4.1], [4.1, 4.2], [4.8, 5.1], [4.7, 4.8]]], dtype=torch.float64),
            motion_valid=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
            text_valid=torch.tensor([[1, 0]], dtype=torch.bool),
        )
        tokens, valid = model(rgb, motion, torch.randn(1, 14), torch.randn(1, 2, 16), **args)
        self.assertFalse(model.pretrained)
        self.assertEqual(tokens.shape, (1, 71, 24))  # 3 x 4x4 RGB + 4 motion + state + 2 text + 16 local flow
        self.assertEqual(valid.sum().item(), 36)
        self.assertEqual(tokens[~valid].abs().sum().item(), 0)
        self.assertFalse(torch.allclose(tokens[:, 0], tokens[:, 1]))
        tokens[..., 0].sum().backward()
        self.assertEqual(rgb.grad[:, 2].abs().sum().item(), 0)
        self.assertGreater(rgb.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(motion.grad[:, 2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
