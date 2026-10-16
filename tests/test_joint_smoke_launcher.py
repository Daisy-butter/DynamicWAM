"""CPU-only launch-plan and gate tests; every execution hook is mocked."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_joint_motionkv_smokes as launcher


class JointSmokeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ids = ['GPU-' + str(i) for i in range(8)]

    def tearDown(self):
        self.tmp.cleanup()

    def closed(self, **updates):
        value = dict(ended_at='finished', final_exit_code=0, driver_exit_clean=True,
            guard_workers_exited=True, guard_all_handles_closed=True, stop_reason=None)
        value.update(updates)
        return value

    def test_missing_running_or_unclosed_receipt_fails(self):
        path = self.root / 'release.json'
        with self.assertRaises(RuntimeError):
            launcher.require_release(path, require_success=False, label='old evaluation')
        for key in ('ended_at', 'final_exit_code', 'driver_exit_clean',
                    'guard_workers_exited', 'guard_all_handles_closed'):
            value = self.closed()
            value.pop(key)
            path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):
                launcher.require_release(path, require_success=False, label='old evaluation')
        path.write_text(json.dumps(self.closed(final_exit_code=1)))
        launcher.require_release(path, require_success=False, label='old evaluation')
        with self.assertRaises(RuntimeError):
            launcher.require_release(path, require_success=True, label='new smoke')

    def test_nccl_defaults_match_existing_helper_and_preserve_other_overrides(self):
        source = dict(NCCL_P2P_DISABLE='1', NCCL_CUMEM_ENABLE='0', NCCL_CUMEM_HOST_ENABLE='0',
                      NCCL_SOCKET_IFNAME='eth0', ACCELERATE_USE_DEEPSPEED='false')
        env, nccl = launcher.training_environment(source)
        self.assertEqual(nccl, dict(NCCL_SOCKET_IFNAME='eth0'))
        for key in ('NCCL_P2P_DISABLE', 'NCCL_CUMEM_ENABLE', 'NCCL_CUMEM_HOST_ENABLE'):
            self.assertNotIn(key, env)
            self.assertIn(key, source)
        self.assertEqual(env['ACCELERATE_USE_DEEPSPEED'], 'true')

    def test_fresh_authorization_is_bound_to_eight_ids_and_mode(self):
        path = self.root / 'authorization.json'
        path.write_text(json.dumps(dict(gpu_uuids=self.ids, device_lifetime_mode='fds_before_telemetry_v1')))
        self.assertEqual(len(launcher.require_fresh_authorization(path, self.ids)), 64)
        with self.assertRaises(RuntimeError):
            launcher.require_fresh_authorization(path, self.ids[:-1])
        os.utime(path, (time.time() - 1801, time.time() - 1801))
        with self.assertRaises(RuntimeError):
            launcher.require_fresh_authorization(path, self.ids)

    def test_actual_profile_plan_targets_warmup_then_joint_four_steps(self):
        plan = launcher.commands(','.join(self.ids), self.root / 'auth.json',
            launcher.SSD / 'outputs/training/never_executed_test_plan', {})
        self.assertEqual([stage['phase'] for stage in plan], ['warmup', 'joint'])
        self.assertEqual(plan[1]['initialization'], plan[0]['full_export'])
        for stage in plan:
            self.assertIn('--nproc_per_node=8', stage['command'])
            self.assertIn('ACCELERATE_USE_DEEPSPEED=true', stage['command'])
            self.assertNotIn('NCCL_P2P_DISABLE=0', stage['command'])
            self.assertIn('--expected-steps', stage['audit'])
            self.assertTrue(stage['full_export'].endswith('multirate_step_4.pt'))

    def test_original_stage2_subset_recipe_preserves_init_and_two_phase_chain(self):
        plan = launcher.commands(','.join(self.ids), self.root / 'auth.json', self.root / 'run', recipe='v33-stage2-subset')
        self.assertEqual([x['phase'] for x in plan], ['warmup', 'joint'])
        self.assertTrue(plan[0]['config'].endswith('joint_motionkv_v33_stage2_subset_warmup_smoke.yaml'))
        self.assertTrue(plan[0]['initialization'].endswith('/outputs/checkpoint_init/absolute_motion_init.pt'))
        self.assertEqual(plan[1]['initialization'], plan[0]['full_export'])
        self.assertNotIn('step_12000', plan[0]['initialization'])
        for stage in plan:
            self.assertIn('--nproc_per_node=8', stage['command'])
            profile = launcher.yaml.safe_load(Path(stage['config']).read_text())
            self.assertEqual(profile['training']['gradient_accumulation_steps'], 2)
            self.assertFalse(profile['training']['initialize_joint_interface'])
            self.assertTrue(profile['pairing']['smoke_subset'])

    def test_source_change_or_prior_training_outputs_block(self):
        with patch.object(launcher, 'frozen_identity', return_value={'head': 'new'}):
            with self.assertRaises(RuntimeError):
                launcher.require_frozen([], {'head': 'old'})
        init = self.root / 'init.pt'
        init.write_text('fixture')
        run = self.root / 'training'
        plan = [dict(training_run=str(run), initialization=str(init))]
        launcher.require_fresh_outputs(plan, self.root / 'new_guard')
        run.mkdir()
        with self.assertRaises(FileExistsError):
            launcher.require_fresh_outputs(plan, self.root / 'new_guard')

    def test_active_old_evaluation_stops_before_any_launch(self):
        protected = self.root / 'protected.json'
        protected.write_text(json.dumps(dict(state='running')))
        args = ['launcher', '--gpu-uuids', ','.join(self.ids), '--authorization', str(self.root / 'none'),
            '--output-dir', str(launcher.SSD / 'outputs/training/never_executed_gate_test'), '--execute']
        with patch.object(sys, 'argv', args), patch.object(launcher, 'PROTECTED', protected), \
                patch.object(launcher.subprocess, 'run', side_effect=AssertionError('must not launch')) as run:
            with self.assertRaises(RuntimeError):
                launcher.main()
            run.assert_not_called()

    def test_two_stage_execution_requires_audit_before_joint(self):
        self.run_mock_pipeline()

    def test_failed_worker_close_prevents_audit_and_joint(self):
        self.run_mock_pipeline(fail_close=True)

    def run_mock_pipeline(self, fail_close=False):
        protected = self.root / 'protected.json'
        protected.write_text(json.dumps(self.closed()))
        authorization = self.root / 'auth.json'
        authorization.write_text(json.dumps(dict(gpu_uuids=self.ids, device_lifetime_mode='fds_before_telemetry_v1')))
        init = self.root / 'init.pt'
        init.write_text('fixture')
        out = self.root / 'pipeline'
        plan = [dict(phase=phase, command=[phase, 'supervisor'], audit=[phase, 'audit'],
            guard=str(out / (phase + '_guard')), training_run=str(self.root / (phase + '_train')),
            initialization=str(init)) for phase in ('warmup', 'joint')]
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            phase, kind = command
            if kind == 'supervisor':
                guard = out / (phase + '_guard')
                guard.mkdir()
                (guard / 'status.json').write_text(json.dumps(self.closed(guard_all_handles_closed=not fail_close)))
                self.assertNotIn('NCCL_P2P_DISABLE', kwargs['env'])
            else:
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                (out / (phase + '_export_audit.json')).write_text(json.dumps(dict(passed=True)))

        args = ['launcher', '--gpu-uuids', ','.join(self.ids), '--authorization', str(authorization),
                '--output-dir', str(out), '--execute']
        with patch.object(sys, 'argv', args), patch.object(launcher, 'SSD', self.root), \
                patch.object(launcher, 'PROTECTED', protected), patch.object(launcher, 'commands', return_value=plan), \
                patch.object(launcher, 'frozen_identity', return_value={'source': 'fixture'}), \
                patch.object(launcher.subprocess, 'run', side_effect=fake_run):
            if fail_close:
                with self.assertRaises(RuntimeError):
                    launcher.main()
            else:
                launcher.main()
        receipt = json.loads((out / 'status.json').read_text())
        self.assertFalse(receipt['formal_training'])
        if fail_close:
            self.assertEqual(calls, [['warmup', 'supervisor']])
            self.assertEqual(receipt['state'], 'failed_no_retry')
        else:
            self.assertEqual(calls, [['warmup', 'supervisor'], ['warmup', 'audit'], ['joint', 'supervisor'], ['joint', 'audit']])
            self.assertEqual(receipt['completed'], ['warmup', 'joint'])


if __name__ == '__main__':
    unittest.main()
