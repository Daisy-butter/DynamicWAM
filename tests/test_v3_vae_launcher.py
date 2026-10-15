import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_joint_motionkv_vae as launcher


class EncodingLauncherTests(unittest.TestCase):
    def test_eight_independent_shards_and_explicit_smoke_limit(self):
        ids = ['GPU-test-%d' % i for i in range(8)]
        jobs = launcher.worker_commands(ids, Path('/grid'), Path('/cache'), Path('/vae'), 1)
        self.assertEqual([j['gpu_uuid'] for j in jobs], ids)
        for i, job in enumerate(jobs):
            command = job['command']
            self.assertEqual(command[command.index('--rank') + 1], str(i))
            self.assertEqual(command[command.index('--world-size') + 1], '8')
            self.assertIn('--guarded-launch', command)
            self.assertIn('--episode-limit', command)
            self.assertNotIn('torch.distributed.run', command)

    def test_full_coverage_is_explicit(self):
        jobs = launcher.worker_commands(['GPU-%d' % i for i in range(8)], '/g', '/c', '/v', 0)
        self.assertTrue(all('--episode-limit' not in j['command'] for j in jobs))
        for ids, limit in [(['GPU-0'] * 8, 1), (['GPU-0'], 1), (['GPU-%d' % i for i in range(8)], -1)]:
            with self.assertRaises(ValueError):
                launcher.worker_commands(ids, '/g', '/c', '/v', limit)

    def test_workers_reject_unguarded_invocation_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'plan.json'
            plan.write_text('{"guard":"/expected/guard"}')
            with patch.dict(launcher.os.environ, {}, clear=True), patch.object(launcher.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'device lifetime guard'):
                    launcher.run_workers(plan)
                popen.assert_not_called()

    def test_smoke_must_encode_new_data_but_full_can_resume_verified_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            out, latent = Path(directory) / 'receipt', Path(directory) / 'latents'
            launcher.require_output_freshness(out, latent, 1)
            latent.mkdir()
            with self.assertRaisesRegex(FileExistsError, 'skip all encoding'):
                launcher.require_output_freshness(out, latent, 1)
            launcher.require_output_freshness(out, latent, 0)
            out.mkdir()
            with self.assertRaises(FileExistsError):
                launcher.require_output_freshness(out, latent, 0)


if __name__ == '__main__':
    unittest.main()
