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
    def run_main(self, snapshots, *, preflight_error=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / 'run'
        uuids = [f'GPU-mocked-{i}' for i in range(8)]
        authorization = root / 'authorization.json'
        authorization.write_text(json.dumps(dict(driver=BASELINE, gpu_uuids=uuids, guard_sha256='mocked')))
        guard = Mock()
        guard.handles = []
        child = Mock(pid=12345, returncode=0)
        child.poll.return_value = 0
        spawn = Mock(return_value=child)
        load = Mock(return_value=guard)
        gpu = Mock(return_value={'devices': [{'uuid': value, 'pci.bus_id': str(i)} for i, value in enumerate(uuids)]})
        if preflight_error:
            gpu.side_effect = preflight_error

        def finish(guard, run, status, process, owned):
            if guard is not None:
                guard.close()
                status['guard_all_handles_closed'] = True

        with ExitStack() as stack:
            patches = {
                'ROOT': root,
                'verify_ssd': Mock(return_value=(root, {})),
                'inside_ssd': Mock(side_effect=lambda path, ssd: path),
                'driver_evidence': Mock(side_effect=copy.deepcopy(snapshots)),
                'gpu_preflight': gpu,
                'file_digest': Mock(return_value='mocked'),
                'environment': Mock(return_value={}),
                'source_files_and_hash': Mock(return_value=({}, 'mocked')),
                'load_device_guard': load,
                'finish_device_guard': Mock(side_effect=finish),
            }
            for name, value in patches.items():
                stack.enter_context(patch.object(supervisor, name, value))
            stack.enter_context(patch.object(supervisor.subprocess, 'Popen', spawn))
            stack.enter_context(patch.object(supervisor.subprocess, 'check_output', return_value='mocked'))
            stack.enter_context(patch.object(supervisor.signal, 'signal'))
            stack.enter_context(patch.object(supervisor.time, 'sleep'))
            stack.enter_context(patch.object(sys, 'argv', ['supervisor', '--run-dir', str(run),
                '--gpus', ','.join(uuids), '--authorization', str(authorization), '--stage', 'formal',
                '--', 'must-never-be-executed']))
            code = supervisor.main()
        status = json.loads((run / 'status.json').read_text())
        phases = json.loads((run / 'driver_phases.json').read_text())
        return code, status, phases, guard, spawn, load, run

    def test_saved_preflight_delta_blocks_before_guard_or_worker(self):
        code, status, phases, guard, spawn, load, run = self.run_main([BASELINE, STARTUP_DELTA, STARTUP_DELTA])
        self.assertEqual(code, 1)
        self.assertEqual(status['failure_phase'], 'after_gpu_preflight')
        self.assertTrue(status['blocked_before_worker'])
        self.assertFalse(status['worker_started'])
        self.assertFalse(status['driver_exit_clean'])
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['final_exit_code'], 1)
        self.assertIn('ended_at', status)
        load.assert_not_called()
        guard.open.assert_not_called()
        spawn.assert_not_called()
        self.assertEqual(phases[-1]['delta']['anomaly_count'], 1)
        self.assertEqual(phases[-1]['delta']['new_dmesg_tail_lines'], STARTUP_DELTA['dmesg_tail'])
        self.assertEqual(phases[-1]['baseline_identity'], list(supervisor.driver_identity(BASELINE)))

    def test_gap_after_source_collection_blocks_before_guard_open(self):
        code, status, phases, guard, spawn, load, run = self.run_main([BASELINE, BASELINE, STARTUP_DELTA, STARTUP_DELTA])
        self.assertEqual(code, 1)
        self.assertEqual(status['failure_phase'], 'before_guard_open')
        guard.open.assert_not_called()
        guard.close.assert_called_once()
        spawn.assert_not_called()
        self.assertEqual(json.loads((run / 'manifest.json').read_text())['driver'], BASELINE)

    def test_after_guard_open_delta_closes_guard_without_worker(self):
        code, status, phases, guard, spawn, load, run = self.run_main([BASELINE, BASELINE, BASELINE, STARTUP_DELTA, STARTUP_DELTA])
        self.assertEqual(code, 1)
        self.assertEqual(status['failure_phase'], 'after_guard_open_before_worker')
        self.assertTrue(status['blocked_before_worker'])
        self.assertTrue(status['guard_all_handles_closed'])
        guard.open.assert_called_once()
        guard.close.assert_called_once()
        spawn.assert_not_called()

    def test_clean_path_launches_once_and_closes_guard(self):
        code, status, phases, guard, spawn, load, run = self.run_main([BASELINE] * 5)
        self.assertEqual(code, 0)
        self.assertTrue(status['worker_started'])
        self.assertTrue(status['driver_exit_clean'])
        self.assertEqual(status['final_exit_code'], 0)
        guard.open.assert_called_once()
        guard.close.assert_called_once()
        spawn.assert_called_once()
        self.assertEqual([item['phase'] for item in phases], ['before_gpu_preflight',
            'after_gpu_preflight', 'before_guard_open', 'after_guard_open_before_worker', 'worker_started'])
        self.assertEqual(json.loads((run / 'manifest.json').read_text())['driver'], BASELINE)

    def test_preflight_exception_gets_terminal_failure_receipt(self):
        code, status, phases, guard, spawn, load, run = self.run_main([BASELINE, STARTUP_DELTA],
            preflight_error=RuntimeError('mocked telemetry failed'))
        self.assertEqual(code, 1)
        self.assertEqual(status['failure_phase'], 'gpu_preflight')
        self.assertTrue(status['blocked_before_worker'])
        self.assertEqual(status['stop_reason'], 'mocked telemetry failed')
        self.assertEqual(status['final_exit_code'], 1)
        self.assertIn('ended_at', status)
        guard.open.assert_not_called()
        spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
