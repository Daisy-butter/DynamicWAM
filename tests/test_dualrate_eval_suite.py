"""CPU-only checks of scene identity, no retries, prefix and cohort contracts."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import dualrate_eval_worker as worker
import dualrate_eval_suite as suite


class DualRateEvalSuiteTest(unittest.TestCase):
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
        config['environment_groups'] = [config['gpu_uuids'][:3], config['gpu_uuids'][3:6]]
        env = suite.worker_environment(config, 1, self.root)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-3,GPU-4,GPU-5')
        with self.assertRaises(FileExistsError):
            suite.worker_environment(config, 1, self.root)

    def test_overlapping_triplets_and_nonidle_last_two_rejected(self):
        config = {'gpu_uuids': ['GPU-'+str(i) for i in range(8)]}
        config['environment_groups'] = [config['gpu_uuids'][:3], config['gpu_uuids'][2:5]]
        with self.assertRaises(ValueError):
            worker.environment_groups(config)
        config['environment_groups'] = [config['gpu_uuids'][:3], config['gpu_uuids'][5:8]]
        with self.assertRaises(ValueError):
            worker.environment_groups(config)

    def test_smoke_acceptance_requires_graph_actual_overlap_and_new_reference(self):
        report = dict(fast_graph_acceptance=dict(enabled=True, passed=True, sealed=True, shape_count=2),
                      fast_queries_during_slow=1, model_overlap_fast_queries=1, reference_versions_consumed=[1, 2],
                      events=[dict(event='slow_completed', accepted=True, version=2, source_time=.1, completed_time=.3),
                              dict(event='fast_started', query_id=4, reference_version=2, query_time=.4),
                              dict(event='fast_committed', query_id=4, reference_version=2, query_time=.4, commit_time=.5)])
        self.assertTrue(suite.deployment_acceptance([report])['passed'])
        self.assertFalse(suite.deployment_acceptance([dict(report, model_overlap_fast_queries=0)])['passed'])
        self.assertFalse(suite.deployment_acceptance([dict(report, reference_versions_consumed=[0, 1], events=[])])['passed'])
        with self.assertRaisesRegex(ValueError, 'event proof'):
            suite.deployment_acceptance([dict(report, reference_versions_consumed=[0])])
        self.assertFalse(suite.deployment_acceptance([])['passed'])

    def test_background_reference_must_complete_before_the_consuming_query(self):
        report = dict(reference_versions_consumed=[2], events=[
            dict(event='slow_completed', accepted=True, version=2, source_time=.1, completed_time=.6),
            dict(event='fast_started', query_id=4, reference_version=2, query_time=.4),
            dict(event='fast_committed', query_id=4, reference_version=2, query_time=.4, commit_time=.7)])
        self.assertEqual(suite.consumed_background_versions(report), [])
        report['events'][0]['completed_time'] = .3
        report['events'][0]['accepted'] = False
        self.assertEqual(suite.consumed_background_versions(report), [])

    def final_payload(self):
        config = dict(sequence_id='formal-sequence', architecture={'dim': 384}, codec_id='native',
            init_checkpoint_sha256='full-sha', action_normalization_sha256='action-sha',
            motion_normalization_sha256='motion-sha', time_contract='exposure_anchor_wall_release_v4',
            action_interval_seconds=.18000000854954124, action_dt=.18000000854954124,
            runtime={'action_dt': .18000000854954124, 'fast_period_steps': 2, 'slow_period_steps': 4},
            max_observation_age=.24, manifest=str(self.root/'manifest.json'))
        manifest = dict(time_contract=config['time_contract'], action_interval_seconds=config['action_interval_seconds'],
                        measured_runtime=copy.deepcopy(config['runtime']), condition_rgb_max_age_seconds=.24)
        worker.save(config['manifest'], manifest)
        keys = ('sequence_id', 'architecture', 'codec_id', 'init_checkpoint_sha256',
                'action_normalization_sha256', 'motion_normalization_sha256', 'time_contract', 'action_interval_seconds')
        contract = {k: config[k] for k in keys}
        contract.update(world_size=8, manifest_sha256=worker.sha(config['manifest']))
        payload = dict(schema='dualrate-distributed-checkpoint-v1', run_kind='formal_distributed_training',
            phase='cross_time_joint', phase_step=20000, configured_steps=20000, global_step=31000,
            formal_training_ready=True, phase_complete=True, optimizer_boundary=True, model={'weight': 'CPU fixture'},
            contract=contract)
        return config, payload, manifest

    def test_exact_completed_final_checkpoint_contract(self):
        config, payload, manifest = self.final_payload()
        worker.validate_checkpoint_payload(config, payload, manifest)
        for key in ('schema', 'run_kind', 'phase', 'phase_step', 'configured_steps', 'global_step',
                    'formal_training_ready', 'phase_complete', 'optimizer_boundary'):
            with self.subTest(field=key):
                broken = dict(payload, **{key: None})
                with self.assertRaises(ValueError):
                    worker.validate_checkpoint_payload(config, broken, manifest)
        for key in ('sequence_id', 'time_contract', 'action_interval_seconds', 'manifest_sha256'):
            with self.subTest(contract=key):
                broken = copy.deepcopy(payload)
                broken['contract'][key] = 'wrong'
                with self.assertRaises(ValueError):
                    worker.validate_checkpoint_payload(config, broken, manifest)

    def test_checkpoint_rejects_changed_manifest_runtime_and_camera_age(self):
        config, payload, manifest = self.final_payload()
        altered = copy.deepcopy(config)
        altered['runtime']['slow_period_steps'] = 8
        with self.assertRaisesRegex(ValueError, 'measured training runtime'):
            worker.validate_checkpoint_payload(altered, payload, manifest)
        with self.assertRaisesRegex(ValueError, '0.24'):
            worker.validate_checkpoint_payload(dict(config, max_observation_age=.3), payload, manifest)
        manifest['condition_rgb_max_age_seconds'] = .3
        worker.save(config['manifest'], manifest)
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            worker.validate_checkpoint_payload(config, payload, manifest)

    def test_legacy_measured_manifest_uses_protected_online_camera_age_default(self):
        config, payload, manifest = self.final_payload()
        del manifest['condition_rgb_max_age_seconds']
        worker.save(config['manifest'], manifest)
        payload['contract']['manifest_sha256'] = worker.sha(config['manifest'])
        worker.validate_checkpoint_payload(config, payload, manifest)
        with self.assertRaisesRegex(ValueError, '0.24'):
            worker.validate_checkpoint_payload(dict(config, max_observation_age=.3), payload, manifest)

    def test_compatibility_receipt_covers_all_nonexecutor_source(self):
        src = self.root/'src/dynamicwam'
        (src/'dualrate').mkdir(parents=True)
        (src/'dualrate/model.py').write_text('weight = 1\n')
        (src/'dualrate/deployment.py').write_text('callback = None\n')
        (src/'dualrate/online.py').write_text('normalization_precision = "float32"\n')
        receipt_path = self.root/'compatibility.json'
        receipt = dict(training_source_sha256='training', evaluation_source_sha256=worker.source_hash(src),
                       same_model_files_sha256={'dualrate/model.py': worker.sha(src/'dualrate/model.py')},
                       inference_only_files_sha256={'dualrate/online.py': dict(training_sha256='1'*64,
                           evaluation_sha256=worker.sha(src/'dualrate/online.py'),
                           change='fp32_image_normalization_and_camera_numeric_tolerance')})
        worker.save(receipt_path, receipt)
        config = dict(source_compatibility_receipt=str(receipt_path), source_compatibility_sha256=worker.sha(receipt_path))
        with patch.object(worker, 'ROOT', self.root):
            worker.validate_source_compatibility(config, {'source_sha256': 'training'})
            with self.assertRaisesRegex(ValueError, 'different source'):
                worker.validate_source_compatibility(config, {'source_sha256': 'wrong'})
            (src/'dualrate/new_model.py').write_text('weight = 2\n')
            receipt['evaluation_source_sha256'] = worker.source_hash(src)
            worker.save(receipt_path, receipt)
            config['source_compatibility_sha256'] = worker.sha(receipt_path)
            with self.assertRaisesRegex(ValueError, 'every non-executor'):
                worker.validate_source_compatibility(config)

    def test_inference_only_compatibility_rejects_missing_extra_and_wrong_hash(self):
        src = self.root/'src/dynamicwam'
        (src/'dualrate').mkdir(parents=True)
        (src/'dualrate/model.py').write_text('weight = 1\n')
        (src/'dualrate/online.py').write_text('normalization_precision = "float32"\n')
        entry = dict(training_sha256='1'*64, evaluation_sha256=worker.sha(src/'dualrate/online.py'),
                     change='fp32_image_normalization_and_camera_numeric_tolerance')
        receipt_path = self.root/'compatibility.json'
        receipt = dict(training_source_sha256='training', evaluation_source_sha256=worker.source_hash(src),
            same_model_files_sha256={'dualrate/model.py': worker.sha(src/'dualrate/model.py')},
            inference_only_files_sha256={'dualrate/online.py': entry})
        variants = [None, {}, {'dualrate/online.py': entry, 'dualrate/model.py': entry},
                    {'dualrate/online.py': dict(entry, evaluation_sha256='2'*64)},
                    {'dualrate/online.py': dict(entry, training_sha256='not-a-hash')},
                    {'dualrate/online.py': dict(entry, change='arbitrary_model_change')},
                    {'dualrate/online.py': dict(entry, extra='unreviewed')}]
        with patch.object(worker, 'ROOT', self.root):
            for value in variants:
                with self.subTest(binding=value):
                    broken = dict(receipt, inference_only_files_sha256=value)
                    worker.save(receipt_path, broken)
                    config = dict(source_compatibility_receipt=str(receipt_path), source_compatibility_sha256=worker.sha(receipt_path))
                    with self.assertRaisesRegex(ValueError, 'inference-only'):
                        worker.validate_source_compatibility(config)


if __name__ == '__main__':
    unittest.main()
