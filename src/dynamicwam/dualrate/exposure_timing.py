"""Deterministic physical grids and coupled deployment-event training templates.

No learned alignment and no GPU imports. A template is a complete recorded
event relationship, not independently sampled marginal ages or latencies.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

TIME_CONTRACT = 'exposure_anchor_wall_release_v4'
EVENT_SCHEMA = 'dualrate-exposure-event-templates-v4'
EPS = 1e-6


def grid_start(when, origin, dt):
    if not all(math.isfinite(float(x)) for x in (when, origin, dt)) or dt <= 0:
        raise ValueError('finite physical time and positive interval required')
    position = (float(when) - float(origin)) / dt
    near = round(position)
    # Only floating-point clock noise may identify an exact grid boundary.
    if abs((position - near) * dt) <= EPS:
        return float(origin) + near * dt
    return float(origin) + math.floor(position) * dt


def prefix_count(start, end, dt, *, chunk=16):
    count = round((float(end) - float(start)) / dt)
    if count < 0 or count >= chunk or abs(float(end)-float(start)-count*dt) > EPS:
        raise ValueError('prefix must occupy exact grid rows and leave a predicted suffix')
    return count


def validate_template(row, dt):
    reference_available=row.get('reference_available',True)
    if not isinstance(reference_available,bool):
        raise ValueError('reference availability must be explicit boolean runtime state')
    if not reference_available and (row.get('reference_version')!=-1 or
            not isinstance(row.get('slow_supervision_version'),int) or row['slow_supervision_version']<0):
        raise ValueError('missing-reference row needs a real historical slow supervision anchor, not a valid reference')
    required = ('source_available', 'request_time', 'publish_time', 'query_time',
                'query_rgb_time', 'query_rgb_available', 'source_prefix_end', 'fast_prefix_end')
    if any(k not in row or not math.isfinite(float(row[k])) for k in required):
        raise ValueError('complete finite exposure-relative event template required')
    a, r, c, q = (float(row[k]) for k in required[:4])
    if not -EPS <= a <= r + EPS or not r <= c + EPS or not c <= q + EPS or q <= 0:
        raise ValueError('expected exposure 0 <= available <= request <= publish <= query')
    rgb, available = float(row['query_rgb_time']), float(row['query_rgb_available'])
    if rgb < -EPS or rgb > available + EPS or available > q + EPS:
        raise ValueError('query image is unavailable or precedes source exposure')
    if abs(grid_start(rgb, 0., dt) - rgb) > EPS:
        raise ValueError('query capture must lie on the common execution grid')
    source_n = prefix_count(0., row['source_prefix_end'], dt)
    fast_start = grid_start(q, 0., dt)
    fast_n = prefix_count(fast_start, row['fast_prefix_end'], dt)
    if float(row['source_prefix_end']) < r - EPS:
        raise ValueError('source prefix omits already executed source-to-request history')
    if fast_n < 1 or row['fast_prefix_end'] <= q:
        raise ValueError('fast query has no immutable execution prefix')
    if row['source_prefix_end'] > q + EPS and row['fast_prefix_end'] < row['source_prefix_end'] - EPS:
        raise ValueError('fast prefix drops a still-future source commitment')
    if 'fast_submit_time' in row and row['fast_submit_time'] >= row['fast_prefix_end'] - float(row.get('submit_margin', .01)):
        raise ValueError('measured query missed its protected-prefix deadline')
    return dict(row, reference_available=reference_available,source_prefix_rows=source_n, fast_prefix_rows=fast_n,
                fast_action_start=fast_start)


def validate_templates(payload, *, verify_evidence=True):
    if payload.get('schema') != EVENT_SCHEMA or payload.get('time_contract') != TIME_CONTRACT:
        raise ValueError('exposure event template contract differs')
    if payload.get('kind') not in ('synthetic_engineering', 'measured_parallel'):
        raise ValueError('explicit synthetic or measured event provenance required')
    dt = float(payload['action_interval_seconds'])
    if not math.isfinite(dt) or dt <= 0 or payload.get('clock_domain') != 'scene_step_physics_seconds':
        raise ValueError('physical clock domain and action interval required')
    rows = payload.get('templates', [])
    if not rows:
        raise ValueError('no complete event templates')
    for row in rows:
        validate_template(row, dt)
    if payload['kind'] == 'measured_parallel':
        evidence = payload.get('trace_evidence', [])
        if not evidence or payload.get('fast_queries_while_slow_running', 0) < 1 or payload.get('consumed_reference_versions', 0) < 2:
            raise ValueError('measured parallel data needs overlapping fast queries and refreshed references')
        for item in evidence:
            p = Path(item['path'])
            if not p.is_absolute() or len(item.get('sha256', '')) != 64:
                raise ValueError('absolute trace path and content hash required')
            if verify_evidence and hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('timing evidence changed')
    return payload


def synthetic_templates(dt):
    """Only a declared engineering fixture; forbidden as formal timing evidence."""
    rows = []
    for query_steps, request_steps, ready_steps in ((4, 1, 3), (6, 1, 3), (8, 2, 5)):
        q = query_steps * dt
        rows.append(dict(source_available=.12, request_time=request_steps*dt,
            publish_time=ready_steps*dt, query_time=q, query_rgb_time=(query_steps-1)*dt,
            query_rgb_available=(query_steps-1)*dt+.12,
            source_prefix_end=(request_steps+1)*dt, fast_prefix_end=(query_steps+2)*dt))
    return validate_templates(dict(schema=EVENT_SCHEMA, time_contract=TIME_CONTRACT,
        kind='synthetic_engineering', clock_domain='scene_step_physics_seconds',
        action_interval_seconds=dt, templates=rows, provenance='explicit unmeasured CPU/GPU engineering fixture'))


def load_templates(path, digest=None):
    p = Path(path)
    if digest is not None and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        raise ValueError('event template file changed')
    return validate_templates(json.loads(p.read_text()))
