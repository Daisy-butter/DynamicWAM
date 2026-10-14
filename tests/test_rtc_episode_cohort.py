"""No CUDA: prove cohort barriers exclude setup/cleanup from live peers."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rtc_episode_cohort import EpisodeCohort, MODE, atomic_json


class CohortTests(unittest.TestCase):
    def make(self, root, tasks=("a", "b"), episodes=2, timeout=2):
        atomic_json(Path(root) / "manifest.json", dict(mode=MODE, tasks=list(tasks),
            episodes=episodes, barrier_timeout_s=timeout))
        return [EpisodeCohort(root, task) for task in tasks]

    def test_live_excludes_peer_setup_and_cleanup_across_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            cohorts = self.make(root)
            events, errors = [], []
            lock = threading.Lock()
            def event(name, episode, task):
                with lock:
                    events.append((name, episode, task, time.monotonic()))
            def worker(index):
                cohort = cohorts[index]
                try:
                    for episode in range(2):
                        event("setup_start", episode, index)
                        time.sleep(.02 if index else .06)
                        event("setup_end", episode, index)
                        cohort.wait(episode, "ready")
                        event("live_start", episode, index)
                        time.sleep(.07 if index else .02)
                        event("live_end", episode, index)
                        cohort.wait(episode, "finished")
                        event("cleanup", episode, index)
                except BaseException as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(timeout=3)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            for episode in range(2):
                times = lambda name: [row[3] for row in events if row[:2] == (name, episode)]
                self.assertLessEqual(max(times("setup_end")), min(times("live_start")))
                self.assertLessEqual(max(times("live_end")), min(times("cleanup")))
            self.assertLessEqual(max(row[3] for row in events if row[:2] == ("live_end", 0)),
                min(row[3] for row in events if row[:2] == ("setup_start", 1)))

    def test_abort_releases_waiter(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = self.make(root)
            errors = []
            def waiter():
                try: a.wait(0, "ready")
                except BaseException as exc: errors.append(exc)
            thread = threading.Thread(target=waiter)
            thread.start()
            time.sleep(.05)
            b.abort("peer failure")
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertEqual(json.loads((Path(root) / "abort.json").read_text())["reason"], "peer failure")

    def test_timeout_aborts_cohort(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = self.make(root, timeout=.04)
            with self.assertRaises(TimeoutError): a.wait(0, "ready")
            with self.assertRaises(RuntimeError): b.wait(0, "ready")

    def test_single_member_and_invalid_member(self):
        with tempfile.TemporaryDirectory() as root:
            a, = self.make(root, tasks=("a",))
            self.assertEqual(a.wait(0, "ready")["phase"], "ready")
            with self.assertRaises(ValueError): EpisodeCohort(root, "not_member")
            with self.assertRaises(ValueError): a.wait(2, "ready")

if __name__ == "__main__":
    unittest.main()
