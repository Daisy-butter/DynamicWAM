from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_joint_motionkv_smokes as launcher


class V3SmokePlans(unittest.TestCase):
    def plan(self, recipe):
        return launcher.commands(','.join('GPU-%d' % i for i in range(8)),
                                 Path('/authorization'), Path('/receipt'), {}, recipe)

    def test_offline_phases_chain_actual_exports(self):
        plan = self.plan('v3')
        self.assertEqual([p['phase'] for p in plan], ['warmup', 'joint'])
        self.assertEqual(plan[1]['initialization'], plan[0]['full_export'])
        for phase in plan:
            self.assertIn('--nproc_per_node=8', phase['command'])
            self.assertIn('--guard-gpus', phase['command'])

    def test_replay_smoke_does_not_require_formal_training_first(self):
        plan = self.plan('v3-replay')
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['phase'], 'replay')
        self.assertEqual(plan[0]['initialization'], self.plan('v3')[1]['full_export'])

    def test_missing_real_replay_rejected_even_when_offline_data_ready(self):
        plan = self.plan('v3-replay')
        with patch('dynamicwam.training.data.physical_grid_dataset.validate_physical_grid_readiness',
                   return_value={'complete': True}), patch(
                'dynamicwam.training.data.real_rollout_dataset.validate_replay_manifest',
                side_effect=FileNotFoundError('real trajectory missing')) as validate:
            with self.assertRaisesRegex(FileNotFoundError, 'real trajectory missing'):
                launcher.require_recipe_data(plan, 'v3-replay')
            self.assertTrue(validate.call_args.kwargs['strict_provenance'])


if __name__ == '__main__':
    unittest.main()
