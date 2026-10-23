import copy
import unittest

import torch

from dynamicwam.dualrate.fast import CurrentObservationEncoder, FastConfig, FastJointWAM


class FastJointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.cfg = FastConfig(dim=24, layers=2, heads=3, chunk_size=4, future_tokens=4, future_dim=10, observation_dim=12, world_dim=12)
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


class CurrentEncoderTests(unittest.TestCase):
    def test_native_bfloat_language_cast_and_acceleration_mask(self):
        torch.manual_seed(13)
        model = CurrentObservationEncoder(dim=24, text_dim=16)
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
        model = CurrentObservationEncoder(dim=24, text_dim=16)
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
        self.assertEqual(tokens.shape, (1, 19, 24))  # 3 views x 2x2 patches + 4 motion + state + 2 language
        self.assertEqual(valid.sum().item(), 12)
        self.assertEqual(tokens[~valid].abs().sum().item(), 0)
        self.assertFalse(torch.allclose(tokens[:, 0], tokens[:, 1]))
        tokens[..., 0].sum().backward()
        self.assertEqual(rgb.grad[:, 2].abs().sum().item(), 0)
        self.assertGreater(rgb.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(motion.grad[:, 2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
