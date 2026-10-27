import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import full_baseline_resume as resume
from full_baseline_worker import save, sha


class WholeTaskResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = [f'task_{i}' for i in range(35)]
        self.tasks[24] = 'place_bread_basket'

    def tearDown(self):
        self.temp.cleanup()

    def configs(self):
        old = {k: f'value_{k}' for k in resume.COMPARISON_FIELDS}
        old.update(tasks=self.tasks, stage='formal', episodes=20, output=str(self.root/'old'),
                   action_limit=None, benchmark_contract={'grid': .18}, gpu_uuids=[f'GPU-{i}' for i in range(8)])
        path = self.root/'old_config.json'
        save(path, old)
        evidence = self.root/'retained_evidence.json'
        save(evidence, {'passed': True})
        receipt = dict(schema=resume.SCHEMA, complete=True, native_mp4_full_decode=True,
            query_liveness_passed=True, task_exit_zero_proven=True, full_tasks=self.tasks,
            retained_tasks=self.tasks[:24], remaining_tasks=self.tasks[24:],
            retained_episodes=480, remaining_episodes=220, expected_episodes=700,
            required_smoke_tasks=self.tasks[24:32], old_config={'path': str(path), 'sha256': sha(path)},
            evidence=[{'path': str(evidence), 'sha256': sha(evidence)}])
        receipt_path = self.root/'receipt.json'
        save(receipt_path, receipt)
        new = dict(old, tasks=self.tasks[24:], output=str(self.root/'new'),
                   resumption_receipt=str(receipt_path), resumption_receipt_sha256=sha(receipt_path))
        return old, new, receipt

    def test_partition_preserves_original_order_despite_completion_exit_order(self):
        kept, remaining = resume.task_partition(self.tasks, self.tasks[:24][::-1])
        self.assertEqual(kept, self.tasks[:24])
        self.assertEqual(remaining, self.tasks[24:])
        with self.assertRaises(ValueError):
            resume.task_partition(self.tasks, self.tasks[:23])
        with self.assertRaises(ValueError):
            resume.task_partition(self.tasks, [self.tasks[0]]*24)

    def test_remaining_formal_and_failed_cohort_smoke_are_accepted(self):
        _, new, _ = self.configs()
        resume.validate_resumption(new)
        resume.validate_resumption(dict(new, stage='smoke', episodes=2, tasks=self.tasks[24:32]))

    def test_changed_checkpoint_benchmark_tasks_and_budget_rejected(self):
        _, new, _ = self.configs()
        mutations = [dict(checkpoint_sha256='other'), dict(benchmark_contract={'grid': .2}),
                     dict(tasks=self.tasks[24:][::-1]), dict(episodes=19), dict(action_limit=10)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resume.validate_resumption(dict(new, **mutation))

    def test_asynchronous_camera_cannot_reuse_synchronous_completed_tasks(self):
        old, new, _ = self.configs()
        with self.assertRaisesRegex(ValueError, 'camera submission scheduling'):
            resume.validate_resumption(dict(new, camera_async_submit=True))
        resume.compare_benchmark(old, dict(new, camera_async_submit=False))

    def test_smoke_cannot_drop_failed_task_or_shortcut_full_budgets(self):
        _, new, _ = self.configs()
        with self.assertRaises(ValueError):
            resume.validate_resumption(dict(new, stage='smoke', episodes=2, tasks=self.tasks[:8]))
        with self.assertRaises(ValueError):
            resume.validate_resumption(dict(new, stage='smoke', episodes=1, tasks=self.tasks[24:32]))

    def test_mutated_retained_evidence_and_partition_rejected(self):
        _, new, receipt = self.configs()
        save(receipt['evidence'][0]['path'], {'passed': False})
        with self.assertRaisesRegex(ValueError, 'retained evidence changed'):
            resume.validate_resumption(new)
        resume.validate_resumption(new, verify_evidence=False)
        broken = copy.deepcopy(receipt)
        broken['remaining_episodes'] = 219
        save(new['resumption_receipt'], broken)
        new['resumption_receipt_sha256'] = sha(new['resumption_receipt'])
        with self.assertRaisesRegex(ValueError, 'partition differs'):
            resume.validate_resumption(new, verify_evidence=False)

    def test_original_output_cannot_be_reused(self):
        old, new, _ = self.configs()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            resume.validate_resumption(dict(new, output=old['output']))


if __name__ == '__main__':
    unittest.main()
