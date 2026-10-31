"""Read-only, recorded-schedule counterfactual for fast-prefix admission.

Input reports must come from the same deployment and contain runtime wall/physics
events. This estimates candidate prefix counts under the recorded query phase and
lag; it cannot reconstruct pre-existing commitments or changes caused by a new
schedule. Never use it alone as a deployment acceptance or latency guarantee.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics


def summarize(values):
    if not values:
        return None
    values = sorted(values)
    return {"count": len(values), "median": statistics.median(values),
            "p95": values[min(len(values) - 1, math.floor(.95 * len(values)))],
            "p99": values[min(len(values) - 1, math.floor(.99 * len(values)))],
            "max": values[-1]}


def audit(reports, reserve=.30):
    counts = collections.Counter()
    partitions = collections.defaultdict(list)
    late, eligible, late_examples = 0, 0, []
    episodes = 0
    for path, report in reports:
        events, dt = report['events'], float(report['action_interval_s'])
        activation = next(e['wall_time'] for e in events if e['event'] == 'activation')
        starts = {e['query_id']: e for e in events if e['event'] == 'fast_started'
                  and e['request_wall_time'] >= activation}
        episodes += 1
        for e in events:
            if e['event'] != 'fast_committed' or e.get('query_id') not in starts:
                continue
            q = starts[e['query_id']]
            phase = float(q['query_phase_seconds'])
            lag = float(q['wall_physics_lag_seconds'])
            margin = float(q['submit_margin'])
            required = max(2, math.ceil((phase + lag + reserve + margin) / dt))
            counts[required] += 1
            request = float(q['request_wall_time'])
            # This is the exact admission bound assuming existing commitments
            # fit; use observed commit time, not just GPU result-ready time.
            if required == 2:
                eligible += 1
                available = 2 * dt - phase - lag - margin
                from dualrate_continuous_acceptance import submission_wall
                arrived=submission_wall(e,q,events)
                if arrived - request >= available:
                    late += 1
                    if len(late_examples) < 10:
                        late_examples.append({'path': str(path), 'query_id': e['query_id'],
                                              'commit_after_s': e['commit_wall_time'] - request,
                                              'budget_s': available})
            points = [request, float(e['worker_started_wall_time']),
                      float(e['model_started_wall_time']), float(e['gpu_done_wall_time']),
                      float(e['ready_wall_time']), float(e['commit_wall_time'])]
            if any(b < a for a, b in zip(points, points[1:])):
                raise ValueError(f'wall timings not monotonic in {path} query {e["query_id"]}')
            for name, a, b in zip(('queue_wait', 'batch_and_transfer', 'sampling_and_output',
                                   'host_plan', 'owner_publish'), points, points[1:]):
                partitions[name].append(b-a)
            partitions['total_request_to_commit'].append(points[-1]-points[0])
    return {'episodes': episodes, 'reserve_s': reserve,
            'lower_bound_prefix_counts': dict(counts),
            'two_row_assuming_no_extra_existing_commitment': eligible,
            'recorded_schedule_late_for_two': late, 'late_examples': late_examples,
            'wall_partitions_s': {name: summarize(values) for name, values in partitions.items()},
            'limitations': 'Recorded times are not replay; existing commitments and changed execution/scheduling are unknown.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=pathlib.Path)
    parser.add_argument('--reserve', type=float, default=.30)
    args = parser.parse_args()
    if not math.isfinite(args.reserve) or args.reserve < 0:
        parser.error('reserve must be finite and nonnegative')
    reports = ((p, json.loads(p.read_text(encoding='utf-8'))) for p in args.reports)
    print(json.dumps(audit(reports, args.reserve), indent=2))


if __name__ == '__main__':
    main()
