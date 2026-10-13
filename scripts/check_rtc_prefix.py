"""CPU-only checks for RTC conditioning and actual-time queue reservations."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

_path = Path(__file__).resolve().parents[1] / "src/dynamicwam/models/rtc_prefix.py"
_spec = importlib.util.spec_from_file_location("rtc_prefix_under_test", _path)
rtc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rtc
_spec.loader.exec_module(rtc)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.clean = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        self.condition = rtc.PrefixCondition(self.clean, torch.tensor([0, 2]))
        rtc.validate_prefix_condition(self.condition, max_prefix=2)

    def test_zero_delay_matches_native(self):
        c = rtc.PrefixCondition(self.clean, torch.zeros(2, dtype=torch.long))
        x = torch.randn_like(self.clean)
        self.assertTrue(torch.equal(rtc.apply_prefix(x, c), x))
        t = torch.tensor([150., 850.])
        self.assertTrue(torch.equal(rtc.action_token_times(t, c, 12), t[:, None].expand(2, 12)))
        torch.testing.assert_close(rtc.masked_suffix_mse(x, self.clean, c), (x - self.clean).square().mean())

    def test_mixed_lengths_and_dynamics_token_times(self):
        x = torch.full_like(self.clean, -99.)
        y = rtc.apply_prefix(x, self.condition)
        self.assertTrue(torch.equal(y[0], x[0]))
        self.assertTrue(torch.equal(y[1, :2], self.clean[1, :2]))
        self.assertTrue(torch.equal(y[1, 2:], x[1, 2:]))
        times = rtc.action_token_times(torch.tensor([400., 700.]), self.condition, 12)
        self.assertEqual(times[1].tolist(), [700., 0., 0., 700., 700., 700., 700., 700., 700., 700., 700., 700.])

    def test_loss_per_sample_and_no_prefix_gradient(self):
        pred = torch.cat((torch.ones(1, 4, 3), torch.full((1, 4, 3), 2.))).requires_grad_()
        loss = rtc.masked_suffix_mse(pred, torch.zeros_like(pred), self.condition)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[1, :2], torch.zeros(2, 3)))
        self.assertTrue(bool((pred.grad[1, 2:] != 0).all()))

    def test_prefix_protected_after_every_euler_step(self):
        x = rtc.apply_prefix(torch.randn_like(self.clean), self.condition)
        for _ in range(5):
            x = rtc.apply_prefix(x + torch.randn_like(x) * .2, self.condition)
            self.assertTrue(torch.equal(x[1, :2], self.clean[1, :2]))

    def test_reject_delay_out_of_range_without_clipping(self):
        for lengths in ([-1, 1], [0, 3], [0, 4], [0, 5]):
            with self.assertRaises(ValueError):
                rtc.validate_prefix_condition(rtc.PrefixCondition(self.clean, torch.tensor(lengths)), max_prefix=2)
        with self.assertRaises(ValueError):
            rtc.validate_prefix_condition(rtc.PrefixCondition(self.clean, torch.tensor([0, 4])))

    def test_prefix_condition_exact_normalized_clean_endpoint(self):
        sigma = torch.tensor([.25, .75])
        noise = torch.randn_like(self.clean)
        x = (1 - sigma[:, None, None]) * self.clean + sigma[:, None, None] * noise
        x = rtc.apply_prefix(x, self.condition)
        times = rtc.action_token_times(sigma * 1000, self.condition, 9)
        self.assertTrue(torch.equal(x[1, :2], self.clean[1, :2]))
        self.assertTrue(torch.equal(times[1, 1:3], torch.zeros(2)))


class QueueTests(unittest.TestCase):
    def reservation(self, **kw):
        args = dict(query_time=10.05, estimated_ready_time=10.30, queue_end_time=11., horizon=5, max_prefix=4)
        args.update(kw)
        return rtc.committed_prefix_from_queue(torch.arange(8, dtype=torch.float32).reshape(4, 2), torch.tensor([10., 10.1, 10.35, 10.8], dtype=torch.float64), **args)

    def test_nonuniform_action_timestamps(self):
        r = self.reservation()
        self.assertEqual(r.condition.lengths.tolist(), [2])
        self.assertEqual(r.prefix_effective_times.tolist(), [10., 10.1])
        self.assertEqual(r.suffix_not_before, 10.35)
        self.assertEqual(r.anchor_time, 10.)
        rtc.validate_suffix_handoff(r, actual_ready_time=10.31, suffix_effective_time=10.35)

    def test_exact_future_boundary_can_be_replaced(self):
        r = self.reservation(estimated_ready_time=10.35)
        self.assertEqual(r.condition.lengths.tolist(), [2])
        self.assertEqual(r.suffix_not_before, 10.35)

    def test_current_action_remains_frozen_for_zero_delay(self):
        r = self.reservation(query_time=10.1, estimated_ready_time=10.1)
        self.assertEqual(r.condition.lengths.tolist(), [1])
        self.assertEqual(r.prefix_effective_times.tolist(), [10.1])
        self.assertEqual(r.suffix_not_before, 10.35)

    def test_no_queue_truncation_or_silent_underrun(self):
        for kw in (dict(max_prefix=1), dict(estimated_ready_time=11.01), dict(estimated_ready_time=10.04), dict(query_time=9.9), dict(horizon=2, max_prefix=1)):
            with self.assertRaises(ValueError):
                self.reservation(**kw)

    def test_suffix_cannot_overwrite_running_action_or_backdate(self):
        r = self.reservation()
        for ready, effective in ((10.30, 10.31), (10.36, 10.4)):
            with self.assertRaises(ValueError):
                rtc.validate_suffix_handoff(r, actual_ready_time=ready, suffix_effective_time=effective)

    def test_explicit_last_interval(self):
        r = self.reservation(estimated_ready_time=11.)
        self.assertEqual(r.condition.lengths.tolist(), [4])
        self.assertEqual(r.suffix_not_before, 11.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
