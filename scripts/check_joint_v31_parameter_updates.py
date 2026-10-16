#!/usr/bin/env python3
"""CPU-only comparison of real smoke exports; weight changes are not gradients."""
import argparse
import json
from pathlib import Path
import torch


def group(name):
    for prefix, label in (
        ('backbone.compact_wan.', 'video'),
        ('backbone.action_expert.', 'action'),
        ('backbone.absolute_motion_tokens.', 'native_motion'),
        ('fast_conditioner.proposal_', 'proposal_feedback'),
        ('fast_conditioner.motion_forecast', 'motion_forecast'),
        ('fast_conditioner.', 'interface'),
    ):
        if name.startswith(prefix):
            return label
    return 'other'


def compare(before, after):
    result = {}
    for name, tensor in after.items():
        label = group(name)
        row = result.setdefault(label, dict(tensors=0, compared=0, changed=0,
            added=[], examples=[], max_abs_delta=0.0, all_finite=True))
        row['tensors'] += 1
        if name not in before:
            row['added'].append(name)
            row['all_finite'] &= bool(torch.isfinite(tensor).all())
            continue
        old = before[name]
        if old.shape != tensor.shape:
            raise ValueError('shape changed: ' + name)
        changed, maximum = False, 0.0
        # Bound working memory even for full Video DiT matrices.
        for start in range(0, tensor.numel(), 1048576):
            new = tensor.reshape(-1)[start:start + 1048576]
            prior = old.reshape(-1)[start:start + 1048576]
            row['all_finite'] &= bool(torch.isfinite(new).all())
            changed |= not torch.equal(new, prior)
            if new.numel():
                maximum = max(maximum, float((new.float() - prior.float()).abs().max()))
        row['compared'] += 1
        row['changed'] += int(changed)
        row['max_abs_delta'] = max(row['max_abs_delta'], maximum)
        if changed and len(row['examples']) < 3:
            row['examples'].append(name)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--phase', choices=('warmup', 'joint'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    before = torch.load(a.before, map_location='cpu', weights_only=True, mmap=True)
    after = torch.load(a.after, map_location='cpu', weights_only=True, mmap=True)
    rows = compare(before['model'], after['model'])
    passed = all(row['all_finite'] for row in rows.values())
    passed &= rows['video']['changed'] == 0 if a.phase == 'warmup' else rows['video']['changed'] > 0
    passed &= all(rows[key]['changed'] > 0 for key in ('action', 'native_motion', 'interface'))
    if a.phase == 'joint':
        passed &= rows.get('proposal_feedback', {}).get('changed', 0) > 0
    result = dict(passed=bool(passed), phase=a.phase, before=str(a.before.resolve()),
        after=str(a.after.resolve()), groups=rows,
        limitation='Actual parameter changes verify optimizer/export plumbing; weight decay can also change weights. This is not a per-parameter gradient proof or a quality claim.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
