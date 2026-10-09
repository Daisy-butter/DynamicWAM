#!/usr/bin/env python3
"""Pre-register disjoint train/held-out scenes and paired continuous-RT scenes.

Seed selection is performed before preparation, without success-based fallback.
Actual motion pairing is subsequently bound to an accepted-record SHA256.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

FORMAT = 'dynamicwam_world_context_rt_scene_manifest'
VERSION = 1
DEV_TASKS = ('place_shoe', 'place_fan', 'adjust_bottle')
TASKS = tuple(sorted(('adjust_bottle dump_bin_bigbin grab_roller handover_block handover_mic '
    'hanging_mug move_can_pot move_playingcard_away place_a2b_left place_a2b_right '
    'place_bread_skillet place_can_basket place_object_basket put_bottles_dustbin '
    'put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally '
    'beat_block_hammer click_alarmclock click_bell move_pillbottle_pad move_stapler_pad '
    'place_bread_basket place_container_plate place_empty_cup place_fan place_mouse_pad '
    'place_object_scale place_object_stand place_phone_stand place_shoe press_stapler '
    'stamp_seal').split()))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def identity(path):
    p = Path(path).resolve(strict=True)
    return dict(path=str(p), sha256=sha256(p))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, *, exclusive=False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with p.open('x', encoding='utf-8', newline='\n') as f:
            f.write(content)
    else:
        tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(content, encoding='utf-8'); tmp.replace(p)


def partition_rows(rows, seed, validation_fraction):
    """Exact episode-hash ordering in training/data/multirate_sampling.py."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation fraction must be in (0,1)')
    groups, identities = defaultdict(dict), {}
    for r in rows:
        e = int(r['episode_id'])
        key = (str(r['split']), str(r['task_name']), str(r['episode_name']))
        if e in identities and identities[e] != key:
            raise ValueError('conflicting packed episode identity')
        identities[e] = key; groups[key[:2]][e] = key
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda e: hashlib.sha256(
            json.dumps([seed, group[e]], sort_keys=True).encode()).hexdigest())
        n = min(len(ordered)-1, max(1, round(len(ordered)*validation_fraction)))
        result.update({e: 'validation' if i < n else 'train' for i, e in enumerate(ordered)})
    return result


def official_seeds(path):
    v = read_json(path); tasks = v.get('tasks', {})
    if v.get('schema_version') != 1 or v.get('source_arm') != 'official_sync_h16' or set(tasks) != set(TASKS):
        raise ValueError('complete official_sync_h16 exact 35-task exclusion manifest required')
    for task, seeds in tasks.items():
        if (not isinstance(seeds, list) or len(seeds) != 20 or
            any(type(s) is not int or s < 0 for s in seeds) or seeds != sorted(set(seeds))):
            raise ValueError(f'invalid official fixed seed list: {task}')
    return tasks


def build_manifest(*, episodes_path, raw_root, official_path, controller_protocol,
                   partition_seed=20260928, validation_fraction=.1):
    official = official_seeds(official_path)
    rows = [json.loads(x) for x in Path(episodes_path).read_text().splitlines() if x.strip()]
    assignments = partition_rows(rows, partition_seed, validation_fraction)
    by_task, seen = defaultdict(list), set()
    for r in rows:
        key = (str(r['split']), str(r['task_name']), str(r['episode_name']))
        if key in seen:
            raise ValueError('duplicate packed episode identity')
        seen.add(key)
        if r['split'] == 'clean' and r['task_name'] in TASKS:
            if not str(r['episode_name']).isdigit() or int(r.get('sample_count', 0)) <= 0:
                raise ValueError('clean episode requires numeric index and samples')
            by_task[r['task_name']].append(r)
    forbidden = {s for values in official.values() for s in values}
    episodes, seed_files = [], {}
    for task in TASKS:
        path = Path(raw_root)/'clean'/task/'absolute_motion_clean'/'seed.txt'
        seeds = [int(x) for x in path.read_text().split()]
        if not seeds or seeds != sorted(set(seeds)) or seeds[0] < 0:
            raise ValueError(f'invalid original accepted seeds: {path}')
        seed_files[task] = identity(path)
        for partition, required in (('train', 6), ('validation', 1)):
            candidates = sorted((r for r in by_task[task] if assignments[int(r['episode_id'])] == partition),
                                key=lambda r: int(r['episode_name']))
            if len(candidates) < required:
                raise ValueError(f'insufficient original {partition} episodes: {task}')
            for i, r in enumerate(candidates[:required]):
                index = int(r['episode_name'])
                if index >= len(seeds):
                    raise ValueError(f'original episode lacks seed mapping: {task}/{index}')
                seed = seeds[index]
                if seed in forbidden:
                    raise ValueError(f'fixed training scene overlaps official700: {task}/{seed}; replacement prohibited')
                role = 'validation' if partition == 'validation' else ('stage2' if i < 4 else 'stage3')
                episodes.append(dict(scene_id=f'{task}/{seed}', role=role, partition=partition,
                    task=task, seed=seed, source_split='clean', source_episode_name=str(index),
                    source_episode_id=int(r['episode_id']), development=task in DEV_TASKS and
                    (partition == 'validation' or i < 2), accepted_record=f'accepted/{task}/{seed}.pkl'))
    for task in TASKS:
        for seed in official[task]:
            episodes.append(dict(scene_id=f'{task}/{seed}', role='formal', partition='benchmark',
                task=task, seed=seed, development=False, accepted_record=f'accepted/{task}/{seed}.pkl'))
    keys = [(r['task'], r['seed']) for r in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError('selected train/heldout/evaluation scene identities overlap')
    return dict(format=FORMAT, version=VERSION, tasks=list(TASKS), episodes=episodes,
        selection='ascending original episode index: train[0:4] stage2, train[4:6] stage3, validation[0]; no replacement',
        counts=dict(stage2=140, stage3=70, validation=35, formal=700),
        partition=dict(seed=partition_seed, validation_fraction=validation_fraction,
            assignments_sha256=object_hash(sorted(assignments.items()))),
        controller_protocol=identity(controller_protocol),
        inputs=dict(episodes=identity(episodes_path), raw_seed_files=seed_files, official700=identity(official_path)),
        arms=['official_eager', 'graph_zero', 'world_context'], development_ablation='context_off',
        restrictions=['Preparation failure is incomplete evidence, never a replacement seed.',
            'Every arm replays the identical accepted-record bytes, not just the seed.',
            'This continuous protocol is separate from native synchronous DOMINO scores.',
            'Heldout scenes are never aggregated into stage3 training.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', required=True, type=Path); p.add_argument('--raw-root', required=True, type=Path)
    p.add_argument('--official-seed-manifest', required=True, type=Path)
    p.add_argument('--controller-protocol', required=True, type=Path)
    p.add_argument('--partition-seed', type=int, default=20260928)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    v = build_manifest(episodes_path=a.episodes, raw_root=a.raw_root, official_path=a.official_seed_manifest,
        controller_protocol=a.controller_protocol, partition_seed=a.partition_seed, validation_fraction=a.validation_fraction)
    write_json(a.output, v, exclusive=True); print(json.dumps(identity(a.output)))


if __name__ == '__main__':
    main()
