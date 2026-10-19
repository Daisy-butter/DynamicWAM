"""Real spawned CPU workers exercise shared-slot IPC without CUDA or model loads."""
import multiprocessing
import os
import queue
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

from dynamicwam.runtime.domino_continuous import worker as module
from dynamicwam.runtime.domino_continuous.worker import ProcessInferenceWorker, _SharedMailbox


class EchoPolicy:
    last_policy_timing = {"cpu_test": True}

    def drain_episode(self, timeout):
        return {"slow_quiescent": True, "timeout": timeout}

    def infer(self, packet):
        if packet.get("exit_now"):
            os._exit(7)
        if packet.get("raise_error"):
            raise ValueError("intentional CPU policy error")
        if "_prepare_instruction" in packet:
            return np.empty((0, 14))
        return np.broadcast_to(packet["state"], (16, 14)).copy()


def cpu_policy_server(*args):
    requests, results, ready = args[-3:]
    try:
        if args[2] == "startup-error":
            ready.put("intentional CPU startup failure")
            return
        metadata = {"cpu_only": True}
        if args[2] == "cpu-threads":
            metadata["cpu_runtime"] = module._configure_cpu_threads()
            import torch
            metadata["cuda_initialized"] = torch.cuda.is_initialized()
        ready.put(dict(ready=True, metadata=metadata))
        policy = EchoPolicy()
        module._run_policy_requests(policy, requests, results, "motionkv_rtc", policy.infer)
    finally:
        requests.close()
        results.close()


def read_pending_then_stop(requests, results):
    try:
        first = requests.get(timeout=5)
        second = requests.get(timeout=5)
        results.put((first, second))
    finally:
        requests.close()
        results.close()


class SharedMailboxTests(unittest.TestCase):
    def setUp(self):
        self.ctx = multiprocessing.get_context("spawn")
        self.sender, self.receiver = _SharedMailbox.pair(self.ctx, 8192)

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def test_snapshot_is_owned_and_slot_reuse_cannot_mutate_old_arrays(self):
        source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        original = source.copy()
        nested = {"array": source, "nested": {"values": [1, 2]}}
        self.sender.put(nested)
        source[:] = -3
        nested["nested"]["values"][0] = 99
        result = self.receiver.get(timeout=1)
        np.testing.assert_array_equal(result["array"], original)
        self.assertEqual(result["nested"]["values"], [1, 2])
        self.sender.put({"array": np.ones((4, 3))})
        self.receiver.get(timeout=1)
        np.testing.assert_array_equal(result["array"], original)
        self.assertFalse(np.shares_memory(result["array"], source))

    def test_overflow_rejected_without_consuming_sequence_or_publication(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.sender.put(np.zeros(20000, np.uint8))
        with self.assertRaises(queue.Empty):
            self.receiver.get_nowait()
        self.sender.put("valid")
        self.assertEqual(self.receiver.get(timeout=1), "valid")
        self.assertEqual(self.receiver.last_receive_timing["sequence"], 1)

    def test_stop_notice_does_not_overwrite_unread_snapshot(self):
        result_sender, result_receiver = _SharedMailbox.pair(self.ctx, 8192)
        process = self.ctx.Process(target=read_pending_then_stop, args=(self.receiver, result_sender))
        source = np.arange(80)
        try:
            process.start()
            self.receiver.close()
            result_sender.close()
            self.sender.put(source)
            self.sender.put(None)
            first, second = result_receiver.get(timeout=10)
            np.testing.assert_array_equal(first, source)
            self.assertIsNone(second)
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate(); process.join(5)
            process.close(); result_receiver.close(); result_sender.close()

    def test_corrupt_notice_and_closed_channel_rejected(self):
        self.sender._connection.send_bytes(b"bad")
        with self.assertRaisesRegex(RuntimeError, "notification size"):
            self.receiver.get(timeout=1)
        self.sender.close()
        with self.assertRaisesRegex(RuntimeError, "open sender"):
            self.sender.put("bad")


class SpawnedWorkerTests(unittest.TestCase):
    def start(self, checkpoint="cpu"):
        with patch.object(module, "_serve_policy", cpu_policy_server):
            return ProcessInferenceWorker(project_root="/unused", config_path="/unused",
                                          checkpoint=checkpoint, infer_index=0,
                                          policy_mode="motionkv_rtc")

    def await_result(self, worker):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            item = worker.poll()
            if item is not None:
                return item
            time.sleep(.001)
        self.fail("CPU worker timed out")

    def test_real_spawn_roundtrip_large_snapshot_reuse_timing_and_no_feeder(self):
        worker = self.start()
        try:
            state = np.arange(14, dtype=np.float32)
            packet = dict(state=state, frame=np.zeros((360, 320, 3), np.uint8),
                fast_frame=np.zeros((360, 320, 3), np.uint8),
                flow_frames=np.zeros((4, 384, 320, 3), np.uint8))
            request_id = worker.submit(packet)
            state[:] = -1
            with self.assertRaisesRegex(RuntimeError, "outstanding"):
                worker.submit(packet)
            first = self.await_result(worker)
            self.assertEqual(first["request_id"], request_id)
            np.testing.assert_array_equal(first["actions"][0], np.arange(14))
            ipc = first["ipc_timing"]
            self.assertGreater(ipc["request_publish"]["payload_bytes"], 2_000_000)
            self.assertLessEqual(ipc["request_publish"]["notify_finished_perf_s"],
                                 ipc["submit_returned_perf_s"])
            self.assertLessEqual(ipc["child_request_receive"]["deserialized_perf_s"], first["started"])
            self.assertLessEqual(first["finished"], ipc["parent_poll_perf_s"])
            self.assertEqual(first["policy_timing"]["worker_ipc"], ipc)
            worker.submit(packet)
            second = self.await_result(worker)
            np.testing.assert_array_equal(second["actions"][0], -np.ones(14))
            np.testing.assert_array_equal(first["actions"][0], np.arange(14))
            self.assertFalse(any(t.name == "QueueFeederThread" for t in threading.enumerate()))
        finally:
            worker.close()
        self.assertTrue(worker._requests._closed and worker._results._closed)
        worker.close()

    def test_capacity_failure_leaves_worker_usable_and_not_busy(self):
        worker = self.start()
        try:
            worker._requests.capacity = 1024
            with self.assertRaisesRegex(ValueError, "capacity"):
                worker.submit({"state": np.zeros(14), "large": np.zeros(2048)})
            self.assertFalse(worker.busy)
            self.assertEqual(worker._next_id, 0)
            worker.submit({"state": np.ones(14)})
            self.await_result(worker)
        finally:
            worker.close()

    def test_control_messages_not_counted_and_model_errors_propagate(self):
        worker = self.start()
        try:
            worker.prepare_instruction("CPU test")
            receipt = worker.finish_episode(timeout_s=2)
            self.assertTrue(receipt["episode_drain"]["slow_quiescent"])
            self.assertEqual(worker.calls, 0)
            worker.submit({"raise_error": True})
            with self.assertRaisesRegex(RuntimeError, "intentional CPU policy error"):
                self.await_result(worker)
            self.assertFalse(worker.busy)
        finally:
            worker.close()

    def test_startup_failure_leaves_no_owned_live_child(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "intentional CPU startup failure"):
            self.start(checkpoint="startup-error")
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_close_pending_request_closes_ipc_and_child_without_overwrite(self):
        worker = self.start()
        pid = worker._process.pid
        worker.submit({"state": np.ones(14), "large": np.zeros(2_000_000, np.uint8)})
        worker.close()
        self.assertFalse(worker.busy)
        self.assertTrue(worker._requests._closed and worker._results._closed)
        self.assertNotIn(pid, {p.pid for p in multiprocessing.active_children()})

    def test_child_crash_is_detected_and_all_owner_endpoints_close(self):
        worker = self.start()
        pid = worker._process.pid
        try:
            worker.submit({"exit_now": True})
            with self.assertRaisesRegex(RuntimeError, "IPC closed|exited"):
                self.await_result(worker)
        finally:
            worker.close()
        self.assertNotIn(pid, {p.pid for p in multiprocessing.active_children()})
        self.assertTrue(worker._requests._closed and worker._results._closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.submit({"state": np.ones(14)})

    def test_explicit_cpu_threads_and_no_cuda_initialization(self):
        worker = self.start(checkpoint="cpu-threads")
        try:
            runtime = worker.startup_metadata["cpu_runtime"]
            self.assertEqual(runtime["torch_intra_threads"], 1)
            self.assertEqual(runtime["torch_interop_threads"], 1)
            self.assertEqual(runtime["opencv_threads"], 1)
            self.assertFalse(worker.startup_metadata["cuda_initialized"])
            self.assertTrue(runtime["affinity"])
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
