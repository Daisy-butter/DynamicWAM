"""CPU-only ownership and cross-episode background producer fences."""
import queue
import unittest
from unittest.mock import Mock, patch

from dynamicwam.runtime.domino_continuous.worker import (
    ProcessInferenceWorker, _drain_policy_episode, _SharedMailbox,
)


class StartupOwnershipTests(unittest.TestCase):
    def test_failed_constructor_closes_its_child_before_propagating(self):
        for result in (RuntimeError("startup timeout"), "model initialization failed"):
            with self.subTest(result=result):
                process = Mock()
                context = Mock()
                context.Process.return_value = process
                wait = ({"side_effect": result} if isinstance(result, BaseException)
                        else {"return_value": result})
                with patch.object(_SharedMailbox, "pair", side_effect=[(Mock(), Mock()), (Mock(), Mock())]), \
                        patch("dynamicwam.runtime.domino_continuous.worker.multiprocessing.get_context", return_value=context), \
                        patch.object(ProcessInferenceWorker, "_wait_until_ready", **wait), \
                        patch.object(ProcessInferenceWorker, "close") as close:
                    with self.assertRaises(RuntimeError):
                        ProcessInferenceWorker(project_root="/unused", config_path="/unused",
                                               checkpoint="/unused", infer_index=0)
                    process.start.assert_called_once()
                    close.assert_called_once()

    def test_successful_constructor_keeps_its_owner_alive(self):
        context = Mock()
        with patch.object(_SharedMailbox, "pair", side_effect=[(Mock(), Mock()), (Mock(), Mock())]), \
                        patch("dynamicwam.runtime.domino_continuous.worker.multiprocessing.get_context", return_value=context), \
                patch.object(ProcessInferenceWorker, "_wait_until_ready", return_value=None), \
                patch.object(ProcessInferenceWorker, "close") as close:
            ProcessInferenceWorker(project_root="/unused", config_path="/unused",
                                   checkpoint="/unused", infer_index=0)
            close.assert_not_called()

    def test_close_cannot_claim_exit_when_owned_child_is_still_alive(self):
        worker = object.__new__(ProcessInferenceWorker)
        worker._process = Mock()
        worker._process.is_alive.return_value = True
        worker._requests = Mock()
        with self.assertRaisesRegex(RuntimeError, "did not exit"):
            worker.close()
        worker._process.terminate.assert_called_once()
        worker._process.kill.assert_called_once()
        self.assertFalse(worker._closed)


class EpisodeDrainTests(unittest.TestCase):
    def worker(self):
        worker = object.__new__(ProcessInferenceWorker)
        worker._busy = False
        worker.submit = Mock(return_value=7)
        worker.poll = Mock(return_value={"request_id": 7, "finished_episode": True,
                                       "episode_drain": {"active_completed": True}})
        return worker

    def test_explicit_control_roundtrip_and_receipt(self):
        worker = self.worker()
        receipt = worker.finish_episode(timeout_s=3.)
        worker.submit.assert_called_once_with({"_finish_episode": True, "_episode_drain_timeout_s": 3.})
        self.assertTrue(receipt["episode_drain"]["active_completed"])

    def test_cannot_skip_pending_fast_work(self):
        worker = self.worker()
        worker._busy = True
        with self.assertRaisesRegex(RuntimeError, "fast request"):
            worker.finish_episode()
        worker.submit.assert_not_called()

    def test_ack_must_match_control_request_and_kind(self):
        for ack in ({"request_id": 6, "finished_episode": True},
                    {"request_id": 7, "finished_episode": False}):
            worker = self.worker()
            worker.poll.return_value = ack
            with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                worker.finish_episode()

    def test_timeout_does_not_claim_background_completion(self):
        worker = self.worker()
        worker.poll.return_value = None
        with patch("dynamicwam.runtime.domino_continuous.worker.time.monotonic", side_effect=[10., 12.]):
            with self.assertRaisesRegex(TimeoutError, "slow producer"):
                worker.finish_episode(timeout_s=1.)

    def test_control_ack_does_not_inflate_policy_sample_count(self):
        worker = object.__new__(ProcessInferenceWorker)
        worker._busy = True
        worker.calls, worker.intervals = 5, []
        worker._results = queue.Queue()
        worker._results.put({"request_id": 7, "finished_episode": True, "error": ""})
        worker.poll()
        self.assertFalse(worker.busy)
        self.assertEqual(worker.calls, 5)
        self.assertEqual(worker.intervals, [])

    def test_actual_policy_hook_is_required_and_failure_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "does not implement"):
            _drain_policy_episode(object(), 2.)
        policy = Mock()
        policy.drain_episode.return_value = {"pending_cancelled": True}
        self.assertEqual(_drain_policy_episode(policy, 2.), {"pending_cancelled": True})
        policy.drain_episode.assert_called_once_with(timeout=2.)
        policy.drain_episode.side_effect = TimeoutError("active slow still running")
        with self.assertRaisesRegex(TimeoutError, "slow still running"):
            _drain_policy_episode(policy, 2.)

    def test_invalid_timeout_is_rejected_before_dispatch(self):
        for value in (0, -1, float("nan"), float("inf")):
            worker = self.worker()
            with self.assertRaises(ValueError):
                worker.finish_episode(timeout_s=value)
            worker.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
