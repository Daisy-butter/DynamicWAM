"""Read-only CPU audit of real frame coverage on exposure-anchored grids."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def nearest_errors(times, targets):
    hi = np.searchsorted(times, targets)
    lo = np.clip(hi - 1, 0, len(times) - 1)
    hi = np.clip(hi, 0, len(times) - 1)
    return np.minimum(abs(times[lo] - targets), abs(times[hi] - targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--per-task', type=int, default=1)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    path = Path(manifest['episodes_file'])
    if not path.is_absolute():
        path = Path(args.manifest).parent / path
    episodes = json.loads(path.read_text())
    dt = 45 * float(np.float32(.004))
    selected, counts = [], {}
    for ep in episodes:
        task = ep['task']
        if counts.get(task, 0) >= args.per_task:
            continue
        counts[task] = counts.get(task, 0) + 1
        selected.append(ep)
    rows = []
    for ep in selected:
        with h5py.File(ep['raw_path'], 'r') as h:
            raw = h['interception/sim_time_seconds'][:].astype(np.float64)
        if not np.isfinite(raw).all() or (np.diff(raw) < 0).any():
            raise ValueError('invalid recorded timestamps: ' + ep['raw_path'])
        times = raw[np.r_[np.diff(raw) != 0, True]]
        anchors = times[(times + 16 * dt <= times[-1])]
        err = nearest_errors(times, anchors[:, None] + np.arange(1, 9) * 2 * dt)
        old = times[(times - 20 * dt >= times[0])]
        flow = nearest_errors(times, old[:, None] - np.arange(1, 6) * 4 * dt)
        limits = (1e-6, .004001, .030001)
        rows.append(dict(task=ep['task'], episode_uid=ep['episode_uid'], raw_rows=len(raw),
            duplicate_rows=len(raw)-len(times), anchors=len(anchors),
            video_all_eight_counts={str(x): int((err.max(1) <= x).sum()) for x in limits},
            video_max_error_quantiles=np.quantile(err.max(1), [0, .5, .9, 1]).tolist() if len(err) else [],
            flow_anchors=len(old), flow_all_five_counts={str(x): int((flow.max(1) <= x).sum()) for x in limits},
            frame_dt_quantiles=np.quantile(np.diff(times), [0, .5, .9, 1]).tolist()))
    result = dict(action_dt=dt, tasks=len(counts), episodes=len(rows), records=rows)
    result['totals'] = dict(anchors=sum(r['anchors'] for r in rows),
        video_all_eight_counts={str(x): sum(r['video_all_eight_counts'][str(x)] for r in rows) for x in limits},
        flow_anchors=sum(r['flow_anchors'] for r in rows),
        flow_all_five_counts={str(x): sum(r['flow_all_five_counts'][str(x)] for r in rows) for x in limits})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
