"""CPU-only, independent verification against immutable raw event timestamps."""
import math


def submission_wall(event,start,events):
    if event.get('submission_contract','owner_v4')=='owner_v4':return event['commit_wall_time']
    if event['submission_contract']!='worker_receipt_v8':raise ValueError('unknown RTC receipt contract')
    matches=[e for e in events if e.get('event')=='fast_submitted' and e.get('query_id')==event['query_id']]
    if len(matches)!=1:raise ValueError('exactly one actual mailbox receipt required')
    receipt=matches[0];when=float(receipt['submitted_wall_time'])
    if (when!=event['submitted_wall_time'] or receipt['prefix_hash']!=start['prefix_hash']
            or not event['ready_wall_time']<=when<=event['commit_wall_time']):
        raise ValueError('mailbox receipt identity or monotonic order differs')
    return when


def validate_reference_leases(events, config, activation):
    """Every active query consumes the latest causally published reference.

    A terminal episode may cancel its last query; only committed queries must
    have a matching submission. Bootstrap publications remain valid inputs.
    """
    published, started, active, latest = {}, {}, False, None
    count = commits = 0
    ttl = float(config.get('runtime', {}).get('reference_max_age', 2.))
    aw, ap = activation['wall_time'], activation['physics_time']
    for event in events:
        if event.get('post_terminal'):
            continue
        kind = event.get('event')
        if kind == 'activation':
            active = True
        elif kind == 'slow_completed' and event.get('accepted') is True:
            version = event['version']
            if version in published or not event['source_time'] <= event['completed_time']:
                raise ValueError('invalid slow publication identity/time')
            published[version], latest = event, version
        elif active and kind == 'fast_started':
            version, query = event['reference_version'], event['query_time']
            reference = published.get(version)
            if version==-1 and config.get('continuous_v8'):
                current=published.get(latest)
                if (event['query_id'] in started or
                        not event['request_wall_time']<aw+event['protected_until']-ap-event['submit_margin']):
                    raise ValueError('duplicate or late missing-reference query')
                if current is not None and query<current['source_time']+ttl:
                    raise ValueError('missing-reference query skipped a valid latest reference')
                started[event['query_id']]=event;count+=1
                continue
            if (version < 1 or version != latest or reference is None
                    or not reference['source_time'] <= reference['completed_time'] <= query
                    or not query < reference['source_time'] + ttl
                    or event['source_time'] != reference['source_time']
                    or not math.isclose(event['reference_age'], query-reference['source_time'], abs_tol=1e-7)
                    or event['query_id'] in started):
                raise ValueError('active fast query lacks latest causal unexpired reference')
            if not event['request_wall_time'] < aw + event['protected_until']-ap-event['submit_margin']:
                raise ValueError('active query started after its real wall deadline')
            started[event['query_id']] = event
            count += 1
        elif active and kind == 'fast_committed':
            start = started.get(event['query_id'])
            keys = ('query_time', 'reference_version', 'observation_time', 'protected_until', 'submit_margin')
            if start is None or any(event[k] != start[k] for k in keys):
                raise ValueError('commit differs from its captured query lease')
            deadline = event['protected_until']-event['submit_margin']
            submitted=submission_wall(event,start,events)
            if (not start['request_wall_time'] <= event['ready_wall_time'] <= event['commit_wall_time']
                    or not submitted < aw+deadline-ap
                    or not event['query_time'] <= event['commit_time'] < deadline
                    or event['commit_time']+1e-7 < ap+(submitted if event.get('submission_contract')=='worker_receipt_v8' else event['ready_wall_time'])-aw):
                raise ValueError('commit violated strict deadline or wall release')
            commits += 1
    return dict(active_reference_queries=count, active_commits=commits)


def validate_continuous_report(report, config):
    evidence = report.get('raw_clock_acceptance', {})
    events = report.get('events', [])
    activations = [e for e in events if e.get('event') == 'activation']
    terminals = [e for e in events if e.get('event') == 'terminal_clock']
    if len(activations) != 1 or len(terminals) != 1:
        raise ValueError('one immutable activation and terminal clock evidence required')
    activation, terminal = activations[0], terminals[0]
    aw, ap = activation['wall_time'], activation['physics_time']
    fw, fp = terminal['final_wall_time'], terminal['final_physics_time']
    quantum = report['physics_step_seconds']
    ceiling = float(config['max_backlog']) + float(config['backlog_scheduling_tolerance'])
    values = [aw, ap, fw, fp, quantum, ceiling]
    if not all(math.isfinite(v) for v in values) or fw <= aw or quantum <= 0:
        raise ValueError('invalid raw clock endpoints')
    elapsed, physical = fw-aw, fp-ap
    close = lambda x, y: math.isclose(float(x), float(y), rel_tol=0., abs_tol=1e-7)
    if (not close(elapsed, report['active_wall_seconds'])
            or not close(physical, report['physical_duration'])
            or not close(physical, report['physics_steps']*quantum)
            or not close(aw, evidence.get('activation_wall_time', math.nan))
            or not close(ap, evidence.get('activation_physics_time', math.nan))
            or not close(fw, evidence.get('final_wall_time', math.nan))
            or not close(fp, evidence.get('final_physics_time', math.nan))):
        raise ValueError('raw elapsed/physics evidence mismatch; clock credit is forbidden')
    lags = [elapsed-physical]
    for event in events:
        pair = event.get('clock_pair')
        if not pair or event.get('post_terminal'):
            continue
        raw_expected = ap + pair['wall_time'] - aw
        lag = raw_expected - pair['physical_time']
        if (not close(raw_expected, pair['expected_physical_time'])
                or not close(lag, pair['backlog_seconds'])):
            raise ValueError('event clock rebased from immutable activation')
        lags.append(lag)
    if not all(math.isfinite(v) and -quantum-1e-7 <= v <= ceiling+1e-7 for v in lags):
        raise ValueError('original wall/physics backlog outside shared benchmark')
    if (evidence.get('passed') is not True
            or not evidence.get('checks') or not all(evidence['checks'].values())):
        raise ValueError('raw clock acceptance missing')
    # Short native terminal episodes need not reach either query/camera period.
    # The smoke-suite gate, rather than every episode, requires real overlap.
    reference_checks = validate_reference_leases(events, config, activation)
    return dict(passed=True, raw_wall_seconds=elapsed, physical_seconds=physical,
                independently_observed_max_backlog_s=max(lags), clock_pairs=len(lags)-1,
                **reference_checks)
