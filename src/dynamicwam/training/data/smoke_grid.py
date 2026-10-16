"""Audited bounded real-data smoke scope; never a formal-training dataset."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dynamicwam.training.data.physical_grid import json_hash, sha256_file

ROW_ARRAYS = ('eligible', 'action_times', 'video_frames', 'video_actual_times', 'video_lags')


def validate_smoke_subset(root, manifest):
    root = Path(root)
    spec = manifest.get('smoke_subset', {})
    if (manifest.get('full_scope') is not False or spec.get('kind') != 'bounded_real_grid_v1'
            or spec.get('max_training_steps') != 4 or spec.get('world_size') != 8
            or spec.get('formal_training_ready') is not False):
        raise ValueError('subset is not an explicit bounded real-data smoke')
    parent = Path(spec['parent_root'])
    pm = json.loads((parent / 'manifest.json').read_text())
    pi = pm.pop('physical_grid_identity')
    if (json_hash(pm) != pi or not pm['full_scope'] or pi != spec['parent_identity']
            or sha256_file(parent / 'manifest.json') != spec['parent_manifest_sha256']
            or sha256_file(parent / 'episodes.jsonl') != pm['episode_index_sha256']):
        raise ValueError('smoke parent grid identity changed')
    for key in ('contract', 'dataset_identity', 'packed_root', 'source_root',
                'partition_seed', 'validation_fraction', 'episode_partition_identity', 'legacy_pairing_identity'):
        if manifest[key] != pm[key]:
            raise ValueError('smoke subset changed original physical/holdout contract: ' + key)
    originals = {e['episode_index']: e for e in map(json.loads, (parent / 'episodes.jsonl').read_text().splitlines())}
    rows = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
    if len(rows) != 16 or manifest['counts']['episodes'] != 16:
        raise ValueError('bounded smoke requires exactly eight train and eight validation episodes')
    groups = {(part, rank): 0 for part in ('train', 'validation') for rank in range(8)}
    ids = []
    for row in rows:
        original = originals[row['episode_index']]
        key = (row['subset'], int(row['episode_index']) % 8)
        if key not in groups:
            raise ValueError('invalid original holdout partition')
        groups[key] += 1
        for name in ('subset', 'split', 'task_name', 'episode_name', 'lang_group_id', 'clock_sha256', 'command_sha256'):
            if row[name] != original[name]:
                raise ValueError('smoke episode source identity changed: ' + name)
        if row['eligible_samples'] != 16 or row['parent_index_sha256'] != original['sha256']:
            raise ValueError('bounded smoke sample count or source hash changed')
        if sha256_file(parent / original['index_path']) != original['sha256'] or sha256_file(root / row['index_path']) != row['sha256']:
            raise ValueError('smoke source/subset index hash changed')
        positions = np.asarray(row['parent_positions'], dtype=np.int64)
        if len(positions) != 16 or len(np.unique(positions)) != 16 or np.any(np.diff(positions) <= 0):
            raise ValueError('smoke sample positions must be unique ordered source rows')
        with np.load(parent / original['index_path']) as source, np.load(root / row['index_path']) as selected:
            if set(source.files) != set(selected.files):
                raise ValueError('smoke index arrays changed')
            for name in source.files:
                expected = source[name][positions] if name in ROW_ARRAYS else source[name]
                if not np.array_equal(selected[name], expected):
                    raise ValueError('smoke must preserve actual physical labels and clocks: ' + name)
            ids.extend(selected['eligible'][:, 1].tolist())
    if any(value != 1 for value in groups.values()):
        raise ValueError('smoke requires one original train/validation episode per shard')
    if manifest['counts']['eligible_samples'] != 256 or not np.array_equal(np.asarray(sorted(ids)), np.load(root / 'eligible_sample_ids.npy')):
        raise ValueError('smoke sample coverage mismatch')
    return dict(scope='bounded_real_grid_smoke_only', episodes=16, samples=256,
                parent_identity=pi, full_scope=False, formal_training_ready=False)


def prepare_smoke_subset(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve previous subsets; use a fresh directory')
    pm = json.loads((parent / 'manifest.json').read_text())
    identity = pm.pop('physical_grid_identity')
    if json_hash(pm) != identity or not pm['full_scope'] or sha256_file(parent / 'episodes.jsonl') != pm['episode_index_sha256']:
        raise ValueError('parent must be the immutable complete physical grid')
    all_rows = [json.loads(line) for line in (parent / 'episodes.jsonl').read_text().splitlines()]
    selected, used = [], set()
    for part in ('train', 'validation'):
        for rank in range(8):
            split = 'clean' if rank < 4 else 'randomized'
            candidates = [e for e in all_rows if e['subset'] == part and e['split'] == split
                          and e['episode_index'] % 8 == rank and e['eligible_samples'] >= 96]
            candidates.sort(key=lambda e: (e['task_name'] in used, e['episode_index']))
            if not candidates:
                raise ValueError('no eligible real episode for smoke shard')
            row = candidates[0]; selected.append(row); used.add(row['task_name'])
    output.mkdir(parents=True)
    (output / 'episodes').mkdir()
    rows, ids = [], []
    for source in sorted(selected, key=lambda e: e['episode_index']):
        if sha256_file(parent / source['index_path']) != source['sha256']:
            raise ValueError('original index changed')
        with np.load(parent / source['index_path']) as data:
            positions = np.linspace(0, len(data['eligible']) - 1, 16, dtype=np.int64)
            arrays = {name: data[name][positions] if name in ROW_ARRAYS else data[name] for name in data.files}
        path = output / source['index_path']
        np.savez_compressed(path, **arrays)
        row = dict(source, eligible_samples=16, parent_index_sha256=source['sha256'],
                   parent_positions=positions.tolist(), sha256=sha256_file(path))
        rows.append(row); ids.extend(arrays['eligible'][:, 1].tolist())
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    np.save(output / 'eligible_sample_ids.npy', np.asarray(sorted(ids), dtype=np.int64))
    manifest = dict(pm)
    manifest.update(full_scope=False, status='bounded_smoke_index_complete_latents_missing',
        episode_index_sha256=sha256_file(output / 'episodes.jsonl'),
        eligible_ids_sha256=sha256_file(output / 'eligible_sample_ids.npy'),
        counts=dict(episodes=16, eligible_samples=256, source_samples=sum(r['source_samples'] for r in rows),
                    train_episodes=8, validation_episodes=8),
        tasks={r['task_name']: {'eligible': sum(x['eligible_samples'] for x in rows if x['task_name'] == r['task_name'])} for r in rows},
        drops={'bounded_smoke_selection': pm['counts']['eligible_samples'] - 256},
        smoke_subset=dict(kind='bounded_real_grid_v1', parent_root=str(parent), parent_identity=identity,
            parent_manifest_sha256=sha256_file(parent / 'manifest.json'), max_training_steps=4,
            world_size=8, formal_training_ready=False))
    manifest['physical_grid_identity'] = json_hash(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return validate_smoke_subset(output, manifest)
