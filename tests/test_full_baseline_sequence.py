"""No CUDA or subprocess launches: finite sequence gate/ownership regressions."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import full_baseline_sequence as sequence


class SequenceTests(unittest.TestCase):
    def pair(self):
        common = dict(run_kind='full_baseline_suite', gpu_uuids=['GPU-'+str(i) for i in range(8)],
            benchmark_contract={'dt': .18}, checkpoint_sha256='official', action_dt=.18)
        tasks = ['task_'+str(i) for i in range(35)]
        return (dict(common, stage='smoke', tasks=tasks[:8], episodes=2, output='/smoke'),
                dict(common, stage='formal', tasks=tasks, episodes=20, output='/formal'))

    def test_pair_and_only_administrative_differences(self):
        smoke, formal = self.pair()
        sequence.validate_pair(smoke, formal)
        formal['action_dt'] = .16
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            sequence.validate_pair(smoke, formal)

    def test_smoke_truncation_and_unpaired_scenes_rejected(self):
        smoke, formal = self.pair()
        smoke['action_limit'] = 16
        with self.assertRaisesRegex(ValueError, 'complete native'):
            sequence.validate_pair(smoke, formal)
        smoke.pop('action_limit')
        smoke['tasks'] = list(reversed(smoke['tasks']))
        with self.assertRaisesRegex(ValueError, 'first-eight-task'):
            sequence.validate_pair(smoke, formal)

    def test_launch_is_fixed_guard_without_retry(self):
        config = dict(storage_root='/SSD/wuruihan', gpu_uuids=['GPU-0'])
        argv = sequence.guard_command('/cfg', '/run', config, 7200, detach=True)
        self.assertIn('--detach', argv)
        self.assertTrue(argv[2].endswith('guarded_dualrate_run.py'))
        self.assertNotIn('--resume', argv)
        self.assertNotIn('bash', argv)

    def clean_fixture(self, root):
        smoke, _ = self.pair()
        source = {'source_sha256': 'frozen'}
        sequence.save(root/'status.json', dict(state='completed', guard_closed=True, errors=[]))
        sequence.save(root/'owned_exit.json', dict(remaining=[], returncode=0))
        sequence.save(root/'kernel_after.json', dict(unchanged=True, same_boot=True, added=[]))
        sequence.save(root/'source.json', source)
        sequence.save(root/'config.frozen.json', smoke)
        sequence.save(root/'authorization.json', dict(identity={'config_sha256': sequence.canonical_hash(smoke)},
                                                     gpu_uuids=smoke['gpu_uuids']))
        events = [dict(event=event, uuid=uuid, path='/dev/nvidia'+str(i), fd=10+i)
            for event in ('after_open', 'after_close') for i, uuid in enumerate(smoke['gpu_uuids'])]
        sequence.save(root/'guard_events.json', events)
        return smoke, source, events

    def test_clean_guard_requires_all_closures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source, events = self.clean_fixture(root)
            self.assertEqual(len(sequence.require_clean_guard(root, config, source)), 8)
            sequence.save(root/'guard_events.json', events[:-1])
            with self.assertRaisesRegex(ValueError, 'eight guard'):
                sequence.require_clean_guard(root, config, source)

    def test_guard_rejects_kernel_or_worker_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source, _ = self.clean_fixture(root)
            sequence.save(root/'kernel_after.json', dict(unchanged=False, same_boot=True, added=['error']))
            with self.assertRaisesRegex(ValueError, 'kernel'):
                sequence.require_clean_guard(root, config, source)
            sequence.save(root/'kernel_after.json', dict(unchanged=True, same_boot=True, added=[]))
            sequence.save(root/'owned_exit.json', dict(remaining=[], returncode=1))
            with self.assertRaisesRegex(ValueError, 'workers'):
                sequence.require_clean_guard(root, config, source)

    def test_stop_never_signals_reused_pid(self):
        with patch.object(sequence, 'proc_identity', return_value=dict(pid=42, start_ticks=2, state='S', ppid=1)):
            with patch.object(sequence.os, 'kill') as kill:
                with self.assertRaisesRegex(ValueError, 'reused'):
                    sequence.request_owned_stop(dict(pid=42, start_ticks=1))
                kill.assert_not_called()

    def test_stop_signals_only_exact_owned_guard(self):
        expected = dict(pid=42, start_ticks=1, state='S', ppid=1)
        with patch.object(sequence, 'proc_identity', return_value=expected):
            with patch.object(sequence.os, 'kill') as kill:
                self.assertTrue(sequence.request_owned_stop(expected)['requested'])
                kill.assert_called_once_with(42, sequence.signal.SIGTERM)

    def test_initial_observer_fails_once_without_repeat_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            sequence.save(path/'status.json', dict(state='failed'))
            with self.assertRaisesRegex(RuntimeError, 'terminated'):
                sequence.initial_snapshot({}, path)

    def liveness_fixture(self, terminal=6.):
        config = dict(action_dt=.18, max_observation_age=.24, model_query_timeout_s=30.)
        events = [dict(event='full_bootstrap_plan', rows=16, install_row=2),
                  dict(event='activation', physical_time=0.)]
        return config, dict(events=events, physical_duration=terminal, inference_calls=1)

    @staticmethod
    def event(kind, now, **extra):
        return dict(event=kind, clock_pair=dict(active=True, physical_time=now,
                    expected_physical_time=now+.02), **extra)

    def test_liveness_normal_full_query_and_install_pass(self):
        config, report = self.liveness_fixture()
        report['events'] += [self.event('capture', 3.24, exposure=3.06, available=3.24),
            self.event('full_query_started', 3.244, query_id=1,
                observation_physical_age=.184, observation_wall_age=.204),
            self.event('full_plan_installed', 3.50, query_id=1, rows=16, execution_end_time=6.48)]
        report['inference_calls'] = 2
        audit = sequence.audit_task_liveness([report], config)[0]
        self.assertEqual(audit['nonbootstrap_installs'], 1)
        self.assertLess(audit['max_fresh_dispatch_wait_seconds'], .01)

    def test_fresh_frames_with_no_dispatch_cannot_pass_valid_episode(self):
        config, report = self.liveness_fixture(120.)
        report['events'] += [self.event('capture', 3.24+.18*i, exposure=3.06+.18*i,
                                      available=3.24+.18*i) for i in range(8)]
        with self.assertRaisesRegex(ValueError, 'starvation'):
            sequence.audit_task_liveness([report], config)

    def test_model_inference_wait_is_not_dispatch_starvation(self):
        config, report = self.liveness_fixture(8.)
        report['events'] += [self.event('capture', 3.24, exposure=3.06, available=3.24),
            self.event('full_query_started', 3.244, query_id=1,
                observation_physical_age=.184, observation_wall_age=.204),
            self.event('capture', 7., exposure=6.84, available=7.)]
        report['inference_calls'] = 2
        self.assertTrue(sequence.audit_task_liveness([report], config)[0]['passed'])

    def test_wall_stale_capture_not_counted_as_fresh_opportunity(self):
        config, report = self.liveness_fixture(3.8)
        event = self.event('capture', 3.24, exposure=3.06, available=3.24)
        event['clock_pair']['expected_physical_time'] = 3.50
        report['events'].append(event)
        audit = sequence.audit_full_liveness(report, config)
        self.assertEqual(audit['fresh_opportunities'], 0)
        with self.assertRaisesRegex(ValueError, 'never issued'):
            sequence.audit_task_liveness([report], config)

    def test_short_episode_can_end_before_next_query(self):
        config, report = self.liveness_fixture(2.)
        self.assertTrue(sequence.audit_task_liveness([report], config)[0]['short_episode'])
        report['physical_duration'] = 3.25
        self.assertTrue(sequence.audit_task_liveness([report], config)[0]['short_episode'])

    def test_pre_liveness_smoke_receipt_cannot_launch_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt_path = Path(temp)/'acceptance.json'
            source = {key: key for key in ('source_sha256', 'guard_sha256', 'supervisor_sha256',
                'distributed_supervisor_sha256', 'scripts_sha256')}
            sequence.save(receipt_path, dict(passed=True, source_contract=source,
                benchmark_contract={'dt': .18}, checkpoint_sha256='official', world_size=8,
                complete_smoke_episodes=16, evidence=[dict(path='/unused', sha256='unused')]))
            config = dict(self.pair()[1], acceptance_receipt=str(receipt_path),
                acceptance_receipt_sha256=sequence.sha(receipt_path), benchmark_contract={'dt': .18},
                init_checkpoint_sha256='official')
            with self.assertRaisesRegex(ValueError, 'same-source'):
                sequence.guard.validate_acceptance(config, source)


class ResumptionSequenceTests(unittest.TestCase):
    def fixture(self, root):
        import full_baseline_resume as resume
        smoke, formal = SequenceTests().pair()
        old = dict(formal, output=str(root/'old'))
        for key in resume.COMPARISON_FIELDS:
            old.setdefault(key, 'fixture_' + key)
        old['init_checkpoint_sha256'] = old['checkpoint_sha256']
        old_path = root/'old.config.json'
        sequence.save(old_path, old)
        receipt = dict(schema=resume.SCHEMA, complete=True, full_tasks=old['tasks'],
            retained_tasks=old['tasks'][:24], remaining_tasks=old['tasks'][24:],
            required_smoke_tasks=old['tasks'][24:32], retained_episodes=480,
            remaining_episodes=220, expected_episodes=700, native_mp4_full_decode=True,
            query_liveness_passed=True, task_exit_zero_proven=True,
            old_config=dict(path=str(old_path), sha256=sequence.sha(old_path)),
            evidence=[dict(path=str(old_path), sha256=sequence.sha(old_path))])
        path = root/'resumption.json'
        sequence.save(path, receipt)
        common = dict(old, resumption_receipt=str(path), resumption_receipt_sha256=sequence.sha(path),
                      storage_root=str(root))
        smoke = dict(common, stage='smoke', episodes=2, tasks=receipt['required_smoke_tasks'], output=str(root/'smoke'))
        formal = dict(common, tasks=receipt['remaining_tasks'], output=str(root/'formal'),
                      aggregate_output=str(root/'aggregate'), aggregate_result_stamp='combined_700')
        source = {key: key for key in ('source_sha256', 'guard_sha256', 'supervisor_sha256',
            'distributed_supervisor_sha256', 'scripts_sha256')}
        return smoke, formal, source

    def accepted(self, root, smoke, source):
        receipt = dict(passed=True, source_contract=source, benchmark_contract=smoke['benchmark_contract'],
            checkpoint_sha256=smoke['checkpoint_sha256'], full_query_liveness_passed=True,
            liveness_contract=sequence.LIVENESS_CONTRACT, native_mp4_full_decode=True,
            clean_worker_exit=True, guard_closed=True, kernel_unchanged=True,
            world_size=8, complete_smoke_episodes=16, smoke_tasks=smoke['tasks'],
            resumption_receipt_sha256=smoke['resumption_receipt_sha256'],
            evidence=[dict(path=smoke['resumption_receipt'], sha256=smoke['resumption_receipt_sha256'])])
        path = root/'acceptance.json'
        sequence.save(path, receipt)
        return receipt, dict(acceptance_receipt=str(path), acceptance_receipt_sha256=sequence.sha(path))

    def test_exact_remaining_partition_pair_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke, formal, _ = self.fixture(Path(temp))
            sequence.validate_pair(smoke, formal)
            formal['resumption_receipt_sha256'] = 'other'
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                sequence.validate_pair(smoke, formal)

    def test_short_formal_without_retention_audit_rejected(self):
        smoke, formal = SequenceTests().pair()
        formal['tasks'] = formal['tasks'][:11]
        with self.assertRaisesRegex(ValueError, 'audited resumption'):
            sequence.validate_pair(smoke, formal)
        with self.assertRaisesRegex(ValueError, 'audited resumption'):
            sequence.guard.validate_acceptance(formal, {})

    def test_wrong_smoke_cohort_cannot_use_remaining_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke, formal, _ = self.fixture(Path(temp))
            smoke['tasks'] = formal['tasks'][1:9]
            with self.assertRaisesRegex(ValueError, 'first-eight-task'):
                sequence.validate_pair(smoke, formal)

    def test_evidence_tamper_rejected_before_smoke_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke, _, source = self.fixture(Path(temp))
            (Path(temp)/'old.config.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                sequence.guard.validate_acceptance(smoke, source)

    def test_formal_guard_binds_smoke_to_exact_retained_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smoke, formal, source = self.fixture(root)
            receipt, binding = self.accepted(root, smoke, source)
            self.assertEqual(sequence.guard.validate_acceptance(dict(formal, **binding), source), receipt)
            for change, error in ((dict(resumption_receipt_sha256='other'), 'another resumption'),
                                  (dict(smoke_tasks=list(reversed(smoke['tasks']))), 'exact failed cohort'),
                                  (dict(native_mp4_full_decode=False), 'same-source'),
                                  (dict(complete_smoke_episodes=15), 'same-source')):
                sequence.save(binding['acceptance_receipt'], dict(receipt, **change))
                bad = dict(binding, acceptance_receipt_sha256=sequence.sha(binding['acceptance_receipt']))
                with self.assertRaisesRegex(ValueError, error):
                    sequence.guard.validate_acceptance(dict(formal, **bad), source)

    def test_aggregate_must_be_fresh_before_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, formal, _ = self.fixture(root)
            Path(formal['aggregate_output']).mkdir()
            with self.assertRaisesRegex(ValueError, 'fresh artifact-disk'):
                sequence.guard.validate_full_baseline_scope(formal)

    def test_real_acceptance_writer_is_consumable_by_formal_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smoke, formal, source = self.fixture(root)
            output = Path(smoke['output'])
            sequence.save(output/'suite_completed.json', dict(complete=True, valid_completed=16,
                every_video_fully_decoded=True, all_workers_exit_zero=True,
                config_sha256=sequence.canonical_hash(smoke), source=source))
            reports = {}
            for task in smoke['tasks']:
                reports[task] = []
                for episode in range(2):
                    directory = output/task/f'episode_{episode:04d}'
                    directory.mkdir(parents=True)
                    video, times = directory/'fixture.mp4', directory/'timestamps.json'
                    video.write_bytes(b'CPU test fixture; decoder separately mocked')
                    times.write_text('[]', encoding='utf-8')
                    report = dict(episode_index=episode, video=str(video),
                                  video_timestamps=str(times), video_probe={'frames': 1})
                    sequence.save(directory/'report.json', report)
                    reports[task].append(report)
            with patch.object(sequence, 'require_clean_guard', return_value=[]), \
                    patch.object(sequence, 'audit_task_liveness', return_value={'passed': True}), \
                    patch('full_baseline_suite.audit_task', side_effect=lambda _, task: (reports[task], [])), \
                    patch('video_artifacts.probe_head_video', return_value={'frames': 1}):
                path = root/'actual_acceptance.json'
                written = sequence.make_acceptance(smoke, root/'guard', source, path)
            formal.update(acceptance_receipt=str(path), acceptance_receipt_sha256=sequence.sha(path))
            self.assertEqual(sequence.guard.validate_acceptance(formal, source), written)
            self.assertEqual(written['smoke_tasks'], smoke['tasks'])
            self.assertEqual(written['resumption_receipt_sha256'], smoke['resumption_receipt_sha256'])

    def test_completed_guard_finalizes_once_without_launching_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, formal, _ = self.fixture(root)
            guard_dir = root/'guard'; guard_dir.mkdir()
            sequence.save(guard_dir/'status.json', dict(state='completed', guard_closed=True, errors=[]))
            sequence.save(guard_dir/'config.frozen.json', formal)
            def finish(config_path, run_dir, destination, stamp):
                self.assertEqual(sequence.read(run_dir/'status.json')['state'], 'completed')
                self.assertEqual(sequence.read(config_path), formal)
                self.assertEqual(stamp, formal['aggregate_result_stamp'])
                Path(destination).mkdir()
                result = dict(complete=True, valid_completed=700)
                sequence.save(Path(destination)/'suite_completed.json', result)
                return result
            with patch('full_baseline_resume.finalize', side_effect=finish) as finalize:
                with patch.object(sequence.guard.subprocess, 'Popen') as spawn:
                    self.assertTrue(sequence.guard.finalize_completed_resumption(formal, guard_dir))
                    spawn.assert_not_called()
                finalize.assert_called_once()
            self.assertEqual(sequence.read(guard_dir/'aggregation.json')['valid_completed'], 700)

    def test_aggregate_failure_keeps_gpu_cleanup_receipt_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, formal, _ = self.fixture(root)
            before = dict(state='completed', guard_closed=True, errors=[])
            sequence.save(root/'status.json', before)
            with patch('full_baseline_resume.finalize', side_effect=ValueError('changed old MP4')) as finalize:
                self.assertFalse(sequence.guard.finalize_completed_resumption(formal, root))
                finalize.assert_called_once()
            self.assertEqual(sequence.read(root/'status.json'), before)
            self.assertEqual(sequence.read(root/'aggregation.json')['state'], 'failed')

    def test_nonresumption_and_smoke_never_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke, _, _ = self.fixture(Path(temp))
            with patch('full_baseline_resume.finalize') as finalize:
                self.assertTrue(sequence.guard.finalize_completed_resumption(smoke, temp))
                self.assertTrue(sequence.guard.finalize_completed_resumption(SequenceTests().pair()[1], temp))
                finalize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
