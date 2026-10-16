"""No CUDA: exact real-label subset and full-scope gates."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from dynamicwam.training.data.physical_grid import json_hash, sha256_file, VERSION
from dynamicwam.training.data.smoke_grid import prepare_smoke_subset, validate_smoke_subset
from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
from dynamicwam.config.multirate import load_multirate_profile
import run_joint_motionkv_smokes as launcher


class SmokeScopeTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'episodes').mkdir(parents=True)
        rows = []
        for index in range(16):
            path = root / 'episodes' / f'{index:06d}.npz'
            sample_ids = np.arange(96) + index * 100
            times = np.arange(112) * .16
            np.savez(path, times=times, commands=np.zeros((112, 14), dtype=np.float32),
                frames=np.arange(112), eligible=np.stack((np.arange(96), sample_ids), axis=1),
                action_times=times[:96, None] + (np.arange(16) + 1) * .16,
                video_frames=np.tile(np.arange(8), (96, 1)),
                video_actual_times=np.tile(np.arange(8) * .32, (96, 1)),
                video_lags=np.zeros((96, 8)))
            rows.append(dict(episode_index=index, subset='train' if index < 8 else 'validation',
                split='clean' if index % 8 < 4 else 'randomized', task_name=f'task{index}',
                episode_name=str(index), lang_group_id=index, clock_sha256='clock', command_sha256='command',
                eligible_samples=96, source_samples=96, index_path=str(path.relative_to(root)), sha256=sha256_file(path)))
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        m = dict(full_scope=True, schema_version=VERSION, counts={'eligible_samples': 1536},
            episode_index_sha256=sha256_file(root / 'episodes.jsonl'), contract={'dt': .16},
            dataset_identity={'native': 'retained'}, packed_root='/packed', source_root='/source',
            partition_seed=3, validation_fraction=.1, episode_partition_identity='fixed-partition',
            legacy_pairing_identity='legacy')
        m['physical_grid_identity'] = json_hash(m)
        (root / 'manifest.json').write_text(json.dumps(m))

    def test_exact_subset_and_holdout_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            parent, subset = Path(d) / 'parent', Path(d) / 'subset'
            self.fixture(parent)
            result = prepare_smoke_subset(parent, subset)
            self.assertEqual(result['samples'], 256)
            self.assertFalse(result['formal_training_ready'])
            manifest = json.loads((subset / 'manifest.json').read_text())
            self.assertFalse(manifest['full_scope'])
            with self.assertRaisesRegex(ValueError, 'cannot authorize full training'):
                validate_physical_grid_readiness(subset, subset / 'absent', require_full=True)
            with self.assertRaises(FileExistsError):
                prepare_smoke_subset(parent, subset)

    def test_rehashed_labels_cannot_replace_original_values(self):
        with tempfile.TemporaryDirectory() as d:
            parent, subset = Path(d) / 'parent', Path(d) / 'subset'
            self.fixture(parent); prepare_smoke_subset(parent, subset)
            rows = [json.loads(l) for l in (subset / 'episodes.jsonl').read_text().splitlines()]
            path = subset / rows[0]['index_path']
            with np.load(path) as z: arrays = {k: z[k] for k in z.files}
            arrays['commands'][0, 0] = 9
            np.savez(path, **arrays)
            rows[0]['sha256'] = sha256_file(path)
            (subset / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest = json.loads((subset / 'manifest.json').read_text())
            with self.assertRaisesRegex(ValueError, 'actual physical labels'):
                validate_smoke_subset(subset, manifest)

    def test_subset_launcher_requires_explicit_scope_and_four_steps(self):
        plan = launcher.commands(','.join(f'GPU-{i}' for i in range(8)), Path('/authorization'), Path('/receipt'), {}, 'v3-subset')
        self.assertEqual(plan[1]['initialization'], plan[0]['full_export'])
        for stage in plan:
            profile = load_multirate_profile(stage['config'])
            self.assertTrue(profile.training['dataset']['smoke_subset'])
            self.assertEqual(profile.training['max_steps'], 4)
        with patch('dynamicwam.training.data.physical_grid_dataset.validate_physical_grid_readiness', return_value={'full_scope': False}) as fn:
            launcher.require_recipe_data(plan, 'v3-subset')
            self.assertFalse(fn.call_args.kwargs['require_full'])
        with self.assertRaisesRegex(ValueError, 'explicit v3-subset'):
            launcher.require_recipe_data(plan, 'v3')

if __name__ == '__main__':
    unittest.main()
