"""CPU-only fail-closed launch tests: no GPU device, CUDA or NVML calls."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import supervise_motionkv_eval as supervisor


BASELINE = {
    'host': 'nvidia', 'boot_id': 'ecebaa85-6ac6-4e66-b7e5-b1d6e7424d2e',
    'anomaly_count': 285,
    'driver_anomalies_sha256': '04a6752f20dcdacbf631d61dfc31dccf9448cc40d3617daba35756f5e67b87d5',
    'latest_driver_anomaly_sha256': '2427e25ea85c5f0ace1b5b104f4cfdf9528201d984075833a101ccad784b6b18',
    'latest_driver_anomaly': '[337573.209209] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601',
    'dmesg_tail': [],
}
# Identity and new lines from the saved formal35x20_eval_guard startup receipt.
# This replays evidence, not the GPU operation that produced it.
STARTUP_DELTA = dict(BASELINE, anomaly_count=286,
    driver_anomalies_sha256='ae4c194973d1b796c7864cb582f6d4f6b1311c0e8723fead9883067ecbb6b426',
    latest_driver_anomaly_sha256='02b86553eb716935a720bbb042db8a68dc67e2eec5663a12c6f6d5d3feea4f64',
    latest_driver_anomaly='[343296.379410] NVRM: knvlinkCoreGetRxSublinkModeCallback: Error getting current RX sublink state!',
    dmesg_tail=[
        '[343296.379408] NVRM: knvlinkExecGspRmRpc_IMPL: Failed to acquire locks for gpumask 0x0',
        '[343296.379410] NVRM: knvlinkCoreGetRxSublinkModeCallback: Error getting current RX sublink state!',
    ])


class LaunchFenceTest(unittest.TestCase):
    def run_main(self, snapshots, *, preflight_error=None, open_error=None,
                 preflight_only=False, mode=supervisor.DEVICE_LIFETIME_MODE,
                 cleanup_error=None, inventory_error=None, guard_count=8):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / 'run'
        uuids = [f'GPU-mocked-{i}' for i in range(8)]
        authorization = root / 'authorization.json'
        authorization.write_text(json.dumps(dict(driver=BASELINE, gpu_uuids=uuids,
            guard_sha256='mocked', device_lifetime_mode=mode)))
        events = []
        guard = Mock()
        guard.handles = []
        def opened():
            events.append('guard_open')
            if open_error:
                raise open_error
        guard.open.side_effect = opened
        guard.close.side_effect = lambda: events.append('guard_close')
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        def spawned(*args, **kwargs):
            events.append('worker')
            return child
        spawn = Mock(side_effect=spawned)
        load = Mock(return_value=guard)
        def telemetry(*args, **kwargs):
            self.assertIn('guard_open', events)
            self.assertNotIn('guard_close', events)
            events.append('telemetry')
            if preflight_error:
                raise preflight_error
            return {'devices': [{'uuid': value, 'pci.bus_id': str(i)} for i, value in enumerate(uuids)]}
        gpu = Mock(side_effect=telemetry)
        inventory = Mock(return_value=[{'uuid': value} for value in uuids])
        if inventory_error:
            inventory.side_effect = inventory_error
        def finish(guard, run, status, process, owned):
            if guard is not None:
                guard.close()
                status['guard_all_handles_closed'] = True
            if cleanup_error:
                raise cleanup_error
        argv = ['supervisor', '--run-dir', str(run), '--gpus', ','.join(uuids[:guard_count]),
            '--guard-gpus', ','.join(uuids[:guard_count]), '--authorization', str(authorization),
            '--stage', 'smoke']
        argv += ['--preflight-only'] if preflight_only else ['--', 'must-never-be-executed']
        env = Mock(return_value={})
        with ExitStack() as stack:
            patches = {
                'ROOT': root, 'verify_ssd': Mock(return_value=(root, {})),
                'inside_ssd': Mock(side_effect=lambda path, ssd: path),
                'driver_evidence': Mock(side_effect=copy.deepcopy(snapshots)),
                'gpu_preflight': gpu, 'proc_gpu_inventory': inventory,
                'file_digest': Mock(return_value='mocked'), 'environment': env,
                'source_files_and_hash': Mock(return_value=({}, 'mocked')),
                'load_device_guard': load, 'finish_device_guard': Mock(side_effect=finish),
            }
            for name, value in patches.items():
                stack.enter_context(patch.object(supervisor, name, value))
            stack.enter_context(patch.object(supervisor.subprocess, 'Popen', spawn))
            stack.enter_context(patch.object(supervisor.subprocess, 'check_output', return_value='mocked'))
            stack.enter_context(patch.object(supervisor.signal, 'signal'))
            stack.enter_context(patch.object(supervisor.time, 'sleep'))
            stack.enter_context(patch.object(sys, 'argv', argv))
            code = supervisor.main()
        status = json.loads((run / 'status.json').read_text())
        phase_file = run / 'driver_phases.json'
        phases = json.loads(phase_file.read_text()) if phase_file.exists() else []
        return dict(code=code, status=status, phases=phases, guard=guard, spawn=spawn,
            load=load, run=run, events=events, gpu=gpu, environment=env)

    def test_saved_delta_before_guard_blocks_telemetry_and_worker(self):
        r = self.run_main([BASELINE, STARTUP_DELTA, STARTUP_DELTA])
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'before_guard_open')
        self.assertTrue(r['status']['blocked_before_worker'])
        self.assertFalse(r['status']['driver_exit_clean'])
        r['guard'].open.assert_not_called()
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()
        self.assertEqual(r['phases'][-1]['delta']['anomaly_count'], 1)
        self.assertEqual(r['phases'][-1]['delta']['new_dmesg_tail_lines'], STARTUP_DELTA['dmesg_tail'])
        self.assertEqual(r['phases'][-1]['baseline_identity'], list(supervisor.driver_identity(BASELINE)))

    def test_after_guard_delta_closes_without_telemetry(self):
        r = self.run_main([BASELINE, BASELINE, STARTUP_DELTA, STARTUP_DELTA])
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'after_guard_open_before_telemetry')
        self.assertTrue(r['status']['guard_all_handles_closed'])
        self.assertEqual(r['events'], ['guard_open', 'guard_close'])
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()

    def test_saved_telemetry_delta_closes_guard_without_worker(self):
        r = self.run_main([BASELINE] * 3 + [STARTUP_DELTA] * 2)
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'after_gpu_preflight')
        self.assertTrue(r['status']['blocked_before_worker'])
        self.assertTrue(r['status']['guard_all_handles_closed'])
        self.assertEqual(r['events'], ['guard_open', 'telemetry', 'guard_close'])
        r['spawn'].assert_not_called()

    def test_environment_gap_is_fenced_before_worker(self):
        r = self.run_main([BASELINE] * 4 + [STARTUP_DELTA] * 2)
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'before_worker_launch')
        r['spawn'].assert_not_called()
        r['guard'].close.assert_called_once()

    def test_clean_path_guards_telemetry_then_worker_then_close(self):
        r = self.run_main([BASELINE] * 6)
        self.assertEqual(r['code'], 0)
        self.assertTrue(r['status']['worker_started'])
        self.assertTrue(r['status']['driver_exit_clean'])
        self.assertEqual(r['events'], ['guard_open', 'telemetry', 'worker', 'guard_close'])
        r['spawn'].assert_called_once()
        self.assertEqual([item['phase'] for item in r['phases']], ['before_guard_open',
            'after_guard_open_before_telemetry', 'after_gpu_preflight', 'before_worker_launch', 'worker_started'])
        manifest = json.loads((r['run'] / 'manifest.json').read_text())
        self.assertEqual(manifest['driver'], BASELINE)
        self.assertEqual(manifest['device_lifetime_mode'], supervisor.DEVICE_LIFETIME_MODE)

    def test_preflight_only_runs_same_telemetry_and_final_close_without_worker(self):
        r = self.run_main([BASELINE] * 5, preflight_only=True)
        self.assertEqual(r['code'], 0)
        self.assertEqual(r['status']['state'], 'preflight_complete')
        self.assertFalse(r['status']['worker_started'])
        self.assertTrue(r['status']['driver_exit_clean'])
        self.assertTrue(r['status']['guard_all_handles_closed'])
        self.assertEqual(r['events'], ['guard_open', 'telemetry', 'guard_close'])
        r['spawn'].assert_not_called()
        r['environment'].assert_not_called()

    def test_occupied_or_failed_telemetry_closes_guard(self):
        r = self.run_main([BASELINE] * 4, preflight_error=RuntimeError('GPU occupied'))
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'gpu_preflight')
        self.assertEqual(r['status']['stop_reason'], 'GPU occupied')
        self.assertTrue(r['status']['guard_all_handles_closed'])
        self.assertEqual(r['events'], ['guard_open', 'telemetry', 'guard_close'])
        r['spawn'].assert_not_called()

    def test_open_exception_closes_guard_and_prevents_queries(self):
        r = self.run_main([BASELINE] * 3, open_error=RuntimeError('mocked partial open'))
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['failure_phase'], 'guard_open')
        r['guard'].close.assert_called_once()
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()

    def test_old_authorization_mode_cannot_open_or_query(self):
        r = self.run_main([BASELINE] * 2, mode=None)
        self.assertEqual(r['code'], 1)
        r['load'].assert_not_called()
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()

    def test_invalid_proc_inventory_cannot_open_or_query(self):
        r = self.run_main([BASELINE] * 2, inventory_error=RuntimeError('invalid device node'))
        self.assertEqual(r['code'], 1)
        r['load'].assert_not_called()
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()

    def test_incomplete_simulator_guard_coverage_is_rejected(self):
        r = self.run_main([BASELINE] * 2, guard_count=1)
        self.assertEqual(r['code'], 1)
        r['load'].assert_not_called()
        r['gpu'].assert_not_called()
        r['spawn'].assert_not_called()

    def test_final_guard_close_error_overrides_zero_child_exit(self):
        r = self.run_main([BASELINE] * 6, cleanup_error=RuntimeError('mocked close failure'))
        self.assertEqual(r['code'], 1)
        self.assertEqual(r['status']['state'], 'failed')
        self.assertEqual(r['status']['stop_reason'], 'mocked close failure')
        self.assertIn('ended_at', r['status'])

    def test_new_anomaly_after_final_close_fails_preflight_only(self):
        r = self.run_main([BASELINE] * 4 + [STARTUP_DELTA], preflight_only=True)
        self.assertEqual(r['code'], 1)
        self.assertFalse(r['status']['driver_exit_clean'])
        self.assertEqual(r['status']['state'], 'failed')
        r['spawn'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
