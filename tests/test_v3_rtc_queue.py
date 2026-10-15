"""Actual controller integration: proposals never become commitments implicitly."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_motionkv_rtc_controller import Clock, Session, Worker
from dynamicwam.runtime.domino_continuous.continuous_rtc import (
    CommittedActionQueue, run_continuous_rtc_episode,
)


class ProposalTests(unittest.TestCase):
    def test_empty_bootstrap_has_no_previous_future(self):
        q = CommittedActionQueue()
        actions, valid = q.proposal(0)
        self.assertFalse(valid.any())
        self.assertFalse(actions.any())

    def test_only_covered_rows_available_and_snapshot_owned(self):
        q = CommittedActionQueue()
        original = np.arange(224).reshape(16, 14) / 1000
        q.bootstrap(original)
        actions, valid = q.proposal(4)
        np.testing.assert_array_equal(actions[:12], original[4:])
        self.assertEqual(valid.sum(), 12)
        self.assertFalse(actions[12:].any())
        actions[:] = 9
        np.testing.assert_array_equal(q.take(0, 16), original)

    def test_old_result_rejected_even_when_prefix_unchanged(self):
        q = CommittedActionQueue()
        initial = np.zeros((16, 14))
        q.bootstrap(initial)
        old = q.reserve(0, 0)
        newer = initial.copy()
        newer[4:] = 1
        q.install(newer, old, ready_physics_s=.1)
        with self.assertRaisesRegex(RuntimeError, 'generation'):
            q.install(initial, old, ready_physics_s=.2)
        np.testing.assert_array_equal(q.take(4, 12), newer[4:])

    def test_controller_wires_prior_queue_and_preserves_live_execution(self):
        clock = Clock()
        session = Session(clock, interval_s=.16)
        worker = Worker(clock, durations=(.05, .1))
        report = run_continuous_rtc_episode(session, worker, task='v3', episode_seed=1,
            action_limit=20, action_interval_s=.16, frame_period_s=.16,
            clock=clock, sleep=clock.sleep)
        self.assertEqual(report['completed_targets'], 20)
        self.assertFalse(worker.calls[0]['_rtc_previous_valid'].any())
        for p in worker.calls[1:]:
            self.assertEqual(p['_rtc_previous_actions'].shape, (16, 14))
            d = p['_rtc_prefix_length']
            self.assertTrue(p['_rtc_previous_valid'][:d].all())
            np.testing.assert_array_equal(p['_rtc_previous_actions'][:d], p['_rtc_prefix_actions'])
        self.assertTrue(report['execution_inference_overlap'])
        self.assertTrue(all(r['rows_dropped'] == 0 for r in report['requests']))
        self.assertEqual([p['_rtc_queue_generation'] for p in worker.calls], [0, 0, 1, 2, 3])
        # A new result arrives while old committed rows still execute, then
        # replaces the tail; it does not wait for the old 16-action plan to end.
        nonbootstrap = report['requests'][1:]
        self.assertTrue(all(r['install_physics_s'] < r['suffix_not_before_physics_s']
                            for r in nonbootstrap))
        self.assertTrue(all(r['replacement_semantics'] ==
            'immutable_prefix_then_atomic_unexecuted_tail' for r in nonbootstrap))

    def test_v3_bootstrap_primes_then_uses_fresh_observation_on_real_grid(self):
        clock = Clock()
        session = Session(clock, interval_s=.16)
        worker = Worker(clock, durations=(1.1, .05))
        report = run_continuous_rtc_episode(session, worker, task='v3', episode_seed=2,
            action_limit=12, action_interval_s=.16, frame_period_s=.16,
            joint_v3=True, clock=clock, sleep=clock.sleep)
        self.assertEqual(report['bootstrap_mode'], 'prime_then_hold')
        self.assertFalse(report['requests'][0]['bootstrap_actions_executed'])
        self.assertGreater(report['requests'][0]['inference_physics_advance_s'], 1.)
        self.assertTrue(worker.calls[0]['_rtc_bootstrap_only'])
        self.assertFalse(worker.calls[1]['_rtc_bootstrap_only'])
        self.assertGreater(worker.calls[1]['sim_time'], worker.calls[0]['sim_time'])
        np.testing.assert_array_equal(worker.calls[1]['_rtc_prefix_actions'], np.zeros((4, 14)))
        for packet in worker.calls[1:]:
            targets = packet['sim_time'] + packet['_rtc_action_target_offsets']
            expected = report['action_grid_start_physics_s'] + (
                packet['_rtc_observation_action_index'] + np.arange(1, 17)) * .16
            np.testing.assert_allclose(targets, expected, rtol=0, atol=1e-8)
        self.assertEqual(report['completed_targets'], 12)

    def test_v3_cannot_be_silently_scaled(self):
        clock = Clock()
        with self.assertRaisesRegex(ValueError, '1x'):
            run_continuous_rtc_episode(Session(clock), Worker(clock), task='v3', episode_seed=3,
                action_limit=8, joint_v3=True, simulation_rate=.25)


if __name__ == '__main__':
    unittest.main()
