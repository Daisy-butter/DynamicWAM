import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import full_baseline_resume as resume
import full_baseline_sequence as sequence
import guarded_dualrate_run as guard
from full_baseline_worker import save, sha


class ExpandedAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixtures(self):
        tasks = [f'task_{i}' for i in range(35)]
        entries = []
        for task in tasks[:8]:
            for episode in range(10):
                directory = self.root/'prepared'/task/f'episode_{episode:04d}'
                descriptor = dict(task=task, episode_index=episode, seed=100000+episode)
                save(directory/'manifest.json', descriptor)
                (directory/'expert_motion.pkl').write_bytes(f'{task}-{episode}'.encode())
                entries.append(dict(descriptor, manifest_sha256=sha(directory/'manifest.json'),
                                    motion_sha256=sha(directory/'expert_motion.pkl')))
        frozen = self.root/'prepared/frozen_manifest.json'
        save(frozen, dict(complete=True, episodes=entries))
        common = dict(run_kind='full_baseline_suite', action_limit=None, gpu_uuids=[f'GPU-{i}' for i in range(8)],
            checkpoint_sha256='ckpt', init_checkpoint_sha256='ckpt', benchmark_contract={'same': True},
            prepared=str(self.root/'prepared'), prepared_sha256=sha(frozen),
            acceptance_smoke_episodes_per_task=10, required_acceptance_cases=[entries[7]])
        smoke = dict(common, stage='smoke', tasks=tasks[:8], episodes=10, output=str(self.root/'smoke'))
        formal = dict(common, stage='formal', tasks=tasks, episodes=20, output=str(self.root/'formal'))
        source = {key: 'source' for key in ('source_sha256', 'guard_sha256', 'supervisor_sha256',
                                          'distributed_supervisor_sha256', 'scripts_sha256')}
        proof = self.root/'proof.json'
        save(proof, {'valid': True})
        receipt = dict(passed=True, source_contract=guard.source_contract(source), benchmark_contract=common['benchmark_contract'],
            checkpoint_sha256='ckpt', full_query_liveness_passed=True, liveness_contract=sequence.LIVENESS_CONTRACT,
            native_mp4_full_decode=True, clean_worker_exit=True, guard_closed=True, kernel_unchanged=True,
            world_size=8, complete_smoke_episodes=80, smoke_episodes_per_task=10,
            expanded_acceptance=resume.expanded_acceptance_contract(smoke), evidence=[dict(path=str(proof), sha256=sha(proof))])
        return smoke, formal, source, receipt

    def authorize(self, formal, receipt):
        path = self.root/'acceptance.json'
        save(path, receipt)
        return dict(formal, acceptance_receipt=str(path), acceptance_receipt_sha256=sha(path))

    def test_legacy_default_and_strict_expanded_range(self):
        self.assertEqual(resume.acceptance_smoke_count({}), 2)
        for bad in (2, 7, 21, True, 10.):
            with self.subTest(count=bad), self.assertRaises(ValueError):
                resume.acceptance_smoke_count({'acceptance_smoke_episodes_per_task': bad})

    def test_eighty_case_gate_and_formal_acceptance(self):
        smoke, formal, source, receipt = self.fixtures()
        sequence.validate_pair(smoke, formal)
        guard.validate_acceptance(self.authorize(formal, receipt), source)
        self.assertEqual(len(receipt['expanded_acceptance']['episode_identities']), 80)

    def test_old_sixteen_receipt_and_forged_coverage_are_rejected(self):
        _, formal, source, receipt = self.fixtures()
        old = dict(receipt, complete_smoke_episodes=16)
        with self.assertRaises(ValueError):
            guard.validate_acceptance(self.authorize(formal, old), source)
        broken = copy.deepcopy(receipt)
        broken['expanded_acceptance']['episode_identities'][7]['seed'] = 100008
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            guard.validate_acceptance(self.authorize(formal, broken), source)
        broken = copy.deepcopy(receipt)
        broken['expanded_acceptance']['episode_identity_sha256'] = 'forged'
        with self.assertRaises(ValueError):
            guard.validate_acceptance(self.authorize(formal, broken), source)

    def test_failure_seed_hash_and_actual_motion_bytes_are_bound(self):
        smoke, _, _, _ = self.fixtures()
        broken = copy.deepcopy(smoke)
        broken['required_acceptance_cases'][0]['motion_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'not covered'):
            resume.expanded_acceptance_contract(broken)
        (self.root/'prepared/task_0/episode_0007/expert_motion.pkl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            resume.expanded_acceptance_contract(smoke)

    def test_smoke_and_formal_cannot_have_different_expanded_gates(self):
        smoke, formal, _, _ = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            sequence.validate_pair(smoke, dict(formal, acceptance_smoke_episodes_per_task=8))

    def test_replay_mode_is_single_gpu_and_separate_worker(self):
        one = 'GPU-00000000-0000-0000-0000-000000000000'
        two = 'GPU-00000000-0000-0000-0000-000000000001'
        devices = {one: {}, two: {}}
        self.assertEqual(guard.selected_uuids(one, devices, 'full_physics_replay'), [one])
        with self.assertRaises(ValueError):
            guard.selected_uuids(one+','+two, devices, 'full_physics_replay')
        command = guard.worker_command(Path('/repo'), {'run_kind': 'full_physics_replay'}, '/cfg', '/auth')
        self.assertTrue(command[1].endswith('full_physics_replay.py'))
        self.assertNotIn('--device', command)
        self.assertIsNone(guard.validate_acceptance({'run_kind': 'full_physics_replay'}, {}))

    def test_nonzero_cpu_candidate_requires_matching_trace_parity_and_hashes(self):
        smoke, _, source, _ = self.fixtures()
        failure = self.root/'failed_episode_report.json'
        save(failure, {'seed': 100007, 'episode_index': 7})
        proof = dict(schema='full_physics_replay_comparison_v1', passed=True,
            state_parity_passed=True, selected_candidate_passed=True, selected_physx_cpu_workers=2,
            parity_atol=1e-6, source_sha256=source['source_sha256'], scripts_sha256=source['scripts_sha256'],
            frozen_manifest_sha256=smoke['prepared_sha256'], source_report_sha256=sha(failure),
            evidence=[dict(path=str(failure), sha256=sha(failure))], **smoke['required_acceptance_cases'][0])
        path = self.root/'physics_comparison.json'
        def config_for(value):
            save(path, value)
            return dict(smoke, physx_cpu_workers=2, physics_replay_receipt=str(path), physics_replay_receipt_sha256=sha(path))
        resume.validate_physics_replay(config_for(proof), source)
        for mutation in (dict(selected_physx_cpu_workers=4), dict(state_parity_passed=False),
                         dict(parity_atol=1e-3), dict(seed=100006), dict(evidence=[]), dict(scripts_sha256='other')):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resume.validate_physics_replay(config_for(dict(proof, **mutation)), source)
        config = config_for(proof)
        save(failure, {'changed': True})
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            resume.validate_physics_replay(config, source)


if __name__ == '__main__':
    unittest.main()
