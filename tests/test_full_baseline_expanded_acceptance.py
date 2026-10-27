import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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


class PassiveForceReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        source = dict(source_sha256='current-source', scripts_sha256='current-scripts')
        outputs = {name: self.root/name for name in ('original', 'baseline', 'optimized')}
        for output in outputs.values():
            (output/'workers_0').mkdir(parents=True)
            (output/'workers_0/state_trace.npz').write_bytes(b'bound-state-fixture')
            save(output/'workers_0/row_timings.json', [])
        configs = {name: dict(run_kind='full_physics_replay', output=str(output), cpu_workers=[0],
                             tail_hold_rows=10, passive_force_dedup=name=='optimized') for name, output in outputs.items()}
        for key, path in {'reference_state_trace': outputs['original']/'workers_0/state_trace.npz',
                          'reference_extension_trace': outputs['baseline']/'workers_0/state_trace.npz',
                          'reference_extension_timings': outputs['baseline']/'workers_0/row_timings.json'}.items():
            configs['optimized'].update({key: str(path), key+'_sha256': sha(path)})
        guards = {name: self.root/(name+'_guard') for name in outputs}
        for name, directory in guards.items():
            save(directory/'status.json', dict(state='completed', guard_closed=True, errors=[]))
            save(directory/'owned_exit.json', dict(remaining=[], returncode=0))
            save(directory/'kernel_after.json', dict(unchanged=True, same_boot=True, added=[]))
            save(directory/'source.json', source)
            save(directory/'config.frozen.json', configs[name])
            save(directory/'authorization.json', dict(mode='full_physics_replay', source=source,
                identity={'config_sha256': resume.canonical_hash(configs[name])}, gpu_uuids=['GPU-0']))
            save(directory/'authorization.json.consumed', {'receipt_sha256': sha(directory/'authorization.json')})
            save(directory/'guard_events.json', [dict(event=e,uuid=f'GPU-{i}',path=f'/dev/nvidia{i}',fd=i+10)
                for e in ('after_open','after_close') for i in range(8)])
        case = dict(task='place_object_stand', episode_index=7, seed=100007, manifest_sha256='manifest', motion_sha256='motion')
        failure = self.root/'failure.json'
        save(failure, case)
        proof = dict(schema='full_passive_force_dedup_replay_v1', passed=True, state_parity_passed=True,
            extension_bitwise_equal=True, prefix_bitwise_equal=True, selected_passive_force_dedup=True,
            timing_benefit_recorded=True, control_wall_improved=True, environment_closed=True,
            state_count=3151, prefix_state_count=2662, physx_cpu_workers=0, tail_hold_rows=10,
            frozen_manifest_sha256='frozen', config_sha256=resume.canonical_hash(configs['optimized']),
            source_report_sha256=sha(failure), evidence=[dict(path=str(failure),sha256=sha(failure))],
            timing_before=dict(total_control_wall_seconds=4.,total_physx_wall_seconds=3.,worst_full_row_control_wall_seconds=.3),
            timing_after=dict(total_control_wall_seconds=2.,total_physx_wall_seconds=1.,worst_full_row_control_wall_seconds=.22),
            **source, **case)
        path = outputs['optimized']/'comparison.json'
        save(path, proof)
        config = dict(passive_force_dedup=True, physx_cpu_workers=0, acceptance_smoke_episodes_per_task=10,
            passive_force_replay_receipt=str(path), passive_force_replay_receipt_sha256=sha(path), prepared_sha256='frozen',
            passive_force_replay_guards={k:str(v) for k,v in guards.items()})
        return config, proof, source, case

    def validate(self, config, source, case):
        with patch.object(resume, 'expanded_acceptance_contract', return_value={'required_acceptance_cases':[case]}):
            return resume.validate_passive_force_replay(config, source)

    def test_true_bitwise_pair_with_closed_guards_and_absorbable_burst(self):
        config, _, source, case = self.fixture()
        result = self.validate(config, source, case)
        self.assertEqual(len(result['guard_evidence']), 24)
        self.assertIsNone(resume.validate_passive_force_replay({}))

    def test_short_wrong_nonbitwise_and_nonbeneficial_pairs_fail(self):
        config, proof, source, case = self.fixture()
        for mutation in ({'state_count':3150}, {'prefix_state_count':2661}, {'extension_bitwise_equal':False},
                         {'prefix_bitwise_equal':False}, {'control_wall_improved':False}, {'scripts_sha256':'old'}):
            with self.subTest(mutation=mutation):
                save(config['passive_force_replay_receipt'], dict(proof, **mutation))
                config['passive_force_replay_receipt_sha256'] = sha(config['passive_force_replay_receipt'])
                with self.assertRaises(ValueError):
                    self.validate(config, source, case)

    def test_kernel_failure_or_missing_guard_closure_rejected(self):
        config, _, source, case = self.fixture()
        directory = Path(config['passive_force_replay_guards']['baseline'])
        save(directory/'kernel_after.json', dict(unchanged=False,same_boot=True,added=['error']))
        with self.assertRaisesRegex(ValueError, 'cleanly closed'):
            self.validate(config, source, case)
        save(directory/'kernel_after.json', dict(unchanged=True,same_boot=True,added=[]))
        events = resume.read(directory/'guard_events.json')
        save(directory/'guard_events.json', events[:-1])
        with self.assertRaisesRegex(ValueError, 'eight guard'):
            self.validate(config, source, case)


if __name__ == '__main__':
    unittest.main()
