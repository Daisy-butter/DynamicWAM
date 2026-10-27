"""CPU-only checks of scene identity, no retries, prefix and cohort contracts."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import full_baseline_worker as worker
import full_baseline_suite as suite


class FullBaselineSuiteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def prepared(self):
        entries = []
        for index in range(20):
            base = self.root / 'task_a' / f'episode_{index:04d}'
            base.mkdir(parents=True)
            worker.save(base / 'manifest.json', dict(task='task_a', episode_index=index,
                requested_seed=100+index, seed=100+index))
            (base / 'expert_motion.pkl').write_bytes(pickle.dumps(dict(seed=100+index, episode_info={'info': {}})))
            entries.append(dict(task='task_a', episode_index=index, seed=100+index,
                manifest_sha256=worker.sha(base/'manifest.json'), motion_sha256=worker.sha(base/'expert_motion.pkl')))
        worker.save(self.root/'frozen_manifest.json', dict(complete=True, episodes=entries))
        worker.save(self.root/'suite_started.json', dict(tasks=['task_a'], seed=100, episodes_per_task=20))
        worker.save(self.root/'suite_completed.json', dict(complete=True))
        return dict(prepared=str(self.root), prepared_sha256=worker.sha(self.root/'frozen_manifest.json'), seed=100, episodes=2)

    def test_smoke_validates_all_twenty_references(self):
        config = self.prepared()
        self.assertEqual(len(worker.load_frozen_task(config, 'task_a')), 2)
        (self.root/'task_a/episode_0019/expert_motion.pkl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            worker.load_frozen_task(config, 'task_a')

    def test_duplicate_frozen_index_rejected(self):
        config = self.prepared()
        frozen = worker.read(self.root/'frozen_manifest.json')
        frozen['episodes'][-1]['episode_index'] = 0
        worker.save(self.root/'frozen_manifest.json', frozen)
        config['prepared_sha256'] = worker.sha(self.root/'frozen_manifest.json')
        with self.assertRaisesRegex(ValueError, 'unique episode'):
            worker.load_frozen_task(config, 'task_a')

    def test_seed_chain_rejected_even_after_manifest_rehash(self):
        config = self.prepared()
        path = self.root/'task_a/episode_0001/manifest.json'
        descriptor = worker.read(path)
        descriptor['requested_seed'] = 105
        worker.save(path, descriptor)
        frozen = worker.read(self.root/'frozen_manifest.json')
        frozen['episodes'][1]['manifest_sha256'] = worker.sha(path)
        worker.save(self.root/'frozen_manifest.json', frozen)
        config['prepared_sha256'] = worker.sha(self.root/'frozen_manifest.json')
        with self.assertRaisesRegex(ValueError, 'seed chain'):
            worker.load_frozen_task(config, 'task_a')

    def test_prompt_prefix_added_exactly_once_by_native_runner(self):
        prompt, prefix = 'scene: complete episode task', 'scene: '
        raw = worker.policy_instruction(prompt, prefix)
        self.assertEqual(prefix + raw, prompt)
        with self.assertRaises(ValueError):
            worker.policy_instruction(prompt, 'different')

    def test_cohort_waits_only_inactive_phases_and_aborts(self):
        worker.save(self.root/'manifest.json', dict(tasks=['task_a'], episodes=2, timeout_seconds=.01))
        cohort = worker.EpisodeCohort(self.root, 'task_a')
        cohort.wait(0, 'ready')
        cohort.wait(0, 'finished')
        cohort.wait(0, 'finalized')
        with self.assertRaises(ValueError):
            cohort.wait(0, 'active')
        cohort.abort('peer failed')
        with self.assertRaisesRegex(RuntimeError, 'cohort aborted'):
            cohort.wait(1, 'ready')

    def test_finalized_barrier_waits_for_slowest_teardown(self):
        worker.save(self.root/'manifest.json', dict(tasks=['task_a', 'task_b'], episodes=1, timeout_seconds=1.))
        first = worker.EpisodeCohort(self.root, 'task_a')
        second = worker.EpisodeCohort(self.root, 'task_b')
        released = threading.Event()
        thread = threading.Thread(target=lambda: (first.wait(0, 'finalized'), released.set()))
        thread.start()
        try:
            self.assertFalse(released.wait(.05))
            second.wait(0, 'finalized')
            self.assertTrue(released.wait(.5))
        finally:
            thread.join(timeout=1.)

    def test_timeout_marker_reaches_suite_without_waiting_for_worker_exit(self):
        directory = self.root/'cohort'
        worker.save(directory/'manifest.json', dict(tasks=['task_a'], episodes=1, timeout_seconds=1.))
        cohort = worker.EpisodeCohort(directory, 'task_a')
        error = TimeoutError('inference did not return')
        error.dualrate_report = {'realtime_valid': False}
        worker.publish_runtime_failure(self.root, cohort, 'task_a', 0, error)
        marker = worker.read(self.root/'task_a/runtime_failure.json')
        self.assertEqual(marker['runtime_report'], {'realtime_valid': False})
        with self.assertRaisesRegex(RuntimeError, 'abort marker'):
            suite.require_healthy_cohort(self.root, directory, ['task_a'])

    def test_worker_command_has_no_shell_and_exact_task_identity(self):
        command = suite.worker_command('/config', '/authorization', 'task_a', 3, '/cohort')
        self.assertEqual(command[command.index('--gpu-index')+1], '3')
        self.assertEqual(command[command.index('--task')+1], 'task_a')
        self.assertNotIn('bash', command)
        self.assertNotIn('--resume', command)

    def test_uuid_isolation_and_fresh_cache(self):
        config = {'gpu_uuids': ['GPU-'+str(i) for i in range(8)]}
        env = suite.worker_environment(config, 4, self.root)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-4')
        with self.assertRaises(FileExistsError):
            suite.worker_environment(config, 4, self.root)


if __name__ == '__main__':
    unittest.main()
