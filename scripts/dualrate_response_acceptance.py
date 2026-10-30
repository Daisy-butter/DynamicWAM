"""Actual smoke response gate; never treats recorded counterfactuals as acceptance.

CLI reruns the existing four-episode engineering/video/clock/exit gate first.
Only the final response receipt may be supplied to the formal launcher.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from dualrate_continuous_acceptance import validate_empty_reference

DEFAULT_CRITERIA = dict(min_two_prefix_fraction=.90,
                        max_episode_observation_response_median_s=.60,
                        max_episode_observation_response_p95_s=.65)
PAIR_DIFFERENCES = {'stage', 'tasks', 'episodes', 'output', 'result_stamp',
                    'acceptance_receipt', 'acceptance_receipt_sha256',
                    'task_timeout_seconds', 'cohort_timeout_seconds'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def require(test, message):
    if not test: raise ValueError(message)

def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b) <= 1e-7

def stats(values):
    require(bool(values) and all(math.isfinite(x) for x in values), 'missing/nonfinite measurements')
    a = sorted(values)
    return dict(count=len(a), median=statistics.median(a),
                p95=a[min(len(a)-1, math.floor(.95*len(a)))],
                p99=a[min(len(a)-1, math.floor(.99*len(a)))], min=a[0], max=a[-1])

def validate_criteria(criteria):
    require(set(criteria) == set(DEFAULT_CRITERIA), 'unexpected response criteria')
    require(all(math.isfinite(v) and v > 0 for v in criteria.values()), 'invalid response criteria')
    require(criteria['min_two_prefix_fraction'] <= 1, 'prefix fraction exceeds one')

def episode_response(report):
    events = [e for e in report['events'] if not e.get('post_terminal')]
    activations = [e for e in events if e['event'] == 'activation']
    require(len(activations) == 1, 'one activation required')
    act = activations[0]; aw = act['wall_time']; ap = act['physics_time']
    dt = report['action_interval_s']
    require(dt > 0 and math.isfinite(dt), 'invalid action grid')
    require(report['complete'] is True and report.get('error') is None, 'incomplete/error episode')
    require(report['executable_action_source'] == 'fast_only', 'action source changed')
    require(report['no_skipped_physics_steps'] is True, 'skipped physics steps')
    require(report['raw_clock_acceptance']['passed'] is True, 'raw clock acceptance failed')
    require(act['immutable_wall_origin'] is True and act['inflight_catchup_hold'] is False
            and act['soft_horizon_reserve_hold'] is False, 'pause/rebase/hold enabled')
    held_rows = round(act['held_prefix_seconds']/dt)
    require(held_rows == 2 and close(act['held_prefix_seconds'], 2*dt), 'bootstrap holds changed')
    starts = {}; commits = []; seen_response = set(); response = []; margins_wall=[]; margins_physical=[]
    prefix_counts = {2:0, 3:0}; rows=[]; active=False; active_commits=0
    published = {}; empty_reference_queries = 0
    for event in events:
        kind = event['event']
        if kind == 'slow_completed' and event.get('accepted') is True:
            published[event['version']] = event
        if kind == 'activation': active=True
        if kind == 'actuator_projection':
            require(event['hard_prefix_preserved'] is True, 'actuator projection changed hard prefix')
        if kind in ('hold', 'fallback_hold', 'queue_underrun', 'fast_rejected'):
            raise ValueError('hold/underrun/rejected plan present')
        if active and kind == 'fast_started':
            qid = event['query_id']; q = event['query_time']
            require(qid not in starts, 'duplicate query')
            if event['reference_version'] == -1:
                # The full engineering gate separately checks publication/TTL
                # and every lease. Keep the response gate consistent with it.
                validate_empty_reference(event, published, 2.)
                empty_reference_queries += 1
            else:
                require(event['reference_version'] >= 1, 'invalid reference identity')
            n = event['effective_protect_steps']
            require(type(n) is int and n in (2,3), 'prefix outside trained two/three rows')
            start = ap + math.floor((q-ap)/dt + 1e-9)*dt
            require(close(event['protected_until'], start+n*dt), 'prefix timing differs from global grid')
            starts[qid]=event; prefix_counts[n]+=1
        elif kind == 'fast_committed':
            require(not any(e['query_id']==event['query_id'] for e in commits), 'duplicate commit')
            commits.append(event)
            if active:
                q = starts.get(event['query_id']); require(q is not None, 'commit without active query')
                for k in ('query_time','observation_time','protected_until','submit_margin','reference_version'):
                    require(event[k] == q[k], 'commit changed captured lease: '+k)
                deadline = event['protected_until']-event['submit_margin']
                wm = aw+deadline-ap-event['commit_wall_time']
                pm = deadline-event['commit_time']
                require(math.isfinite(wm) and math.isfinite(pm) and wm > 0 and pm > 0,
                        'actual two-clock deadline missed')
                margins_wall.append(wm);margins_physical.append(pm);active_commits+=1
        elif kind == 'execute':
            row = event['row']; require(row == len(rows), 'skipped/repeated action row')
            require(close(event['start_time'], ap+row*dt) and close(event['end_time'], ap+(row+1)*dt)
                    and close(event['actual_start'], event['start_time']), 'execution moved off real action grid')
            rows.append(event)
            owner = next((e for e in reversed(commits) if e['protected_until'] <= event['start_time']+1e-9), None)
            if row >= held_rows:
                require(owner is not None, 'executed suffix lacks committed fast plan')
            if owner is None or owner['query_id'] in seen_response: continue
            seen_response.add(owner['query_id'])
            if owner['query_id'] not in starts: continue  # Explicit bootstrap exclusion.
            q = starts[owner['query_id']]
            require(owner['commit_wall_time'] <= event['execution_wall_time'], 'execution before commit')
            exposure = q['observation_exposure_wall_time']
            elapsed = event['execution_wall_time']-exposure
            require(math.isfinite(elapsed) and elapsed >= 0 and exposure <= q['request_wall_time'],
                    'invalid real exposure-to-execution clocks')
            response.append(elapsed)
    require(len(rows) == report['started_targets'], 'execute count differs from started count')
    require(report['completed_targets'] in (len(rows), len(rows)-1), 'completed count invalid')
    if report['stop_reason'] == 'action_budget_exhausted':
        require(len(rows)==report['completed_targets']==report['action_limit'], 'budget shortened')
    require(len(starts) > 0 and active_commits > 0, 'smoke did not exercise active queries')
    # Target equality is NOT forbidden: the policy may legitimately choose hold.
    # We forbid executor-inserted holds and bind the executor source separately.
    return dict(task=report['task'],episode=report['episode_index'],prefix_counts=prefix_counts,
                active_queries=len(starts),active_commits=active_commits,executed_rows=len(rows),
                empty_reference_queries=empty_reference_queries,
                observation_to_execution_wall_s=stats(response),
                wall_commit_deadline_margin_s=stats(margins_wall),
                physical_commit_deadline_margin_s=stats(margins_physical),
                response_measurement='recomputed_first_executed_mutable_row_from_actual_events')

def evaluate_reports(reports, criteria=None):
    criteria=dict(DEFAULT_CRITERIA if criteria is None else criteria); validate_criteria(criteria)
    require(len(reports)==4, 'exactly four actual smoke reports required')
    require(len({(r['task'],r['episode_index']) for r in reports})==4, 'duplicate report identity')
    results=[episode_response(r) for r in reports]
    count=sum(x['active_queries'] for x in results)
    two=sum(x['prefix_counts'][2] for x in results)
    fraction=two/count
    failures=[]
    if fraction < criteria['min_two_prefix_fraction']: failures.append('two_prefix_fraction')
    for e in results:
        s=e['observation_to_execution_wall_s']; identity=f"{e['task']}/{e['episode']}"
        if s['median'] > criteria['max_episode_observation_response_median_s']:
            failures.append(identity+':observation_response_median')
        if s['p95'] > criteria['max_episode_observation_response_p95_s']:
            failures.append(identity+':observation_response_p95')
    return dict(passed=not failures,kind='actual_gpu_smoke_response_v1',criteria=criteria,
                two_prefix_fraction=fraction,active_queries=count,episodes=results,failures=failures,
                not_recorded_counterfactual=True,policy_quality_accepted=False)

def run_gate(repo, guard_dir, destination, criteria=None):
    repo=Path(repo).resolve(); guard_dir=Path(guard_dir).resolve(); destination=Path(destination).resolve()
    require(not destination.exists(), 'refuse to overwrite response acceptance')
    sys.path.insert(0,str(repo/'scripts'))
    import dualrate_posttrain_eval as post
    smoke=post.read(guard_dir/'config.frozen.json');source=post.read(guard_dir/'source.json')
    engineering=destination.with_name(destination.stem+'.engineering.json')
    require(not engineering.exists(), 'refuse to overwrite engineering acceptance')
    base=post.make_acceptance(smoke,guard_dir,source,engineering)
    require(base['passed'] is True, 'engineering acceptance failed')
    child_evidence=[]
    for task in smoke['tasks']:
        child_path=Path(smoke['output'])/task/'slow_process.json'
        child=json.loads(child_path.read_text())
        require(child['bootstrap_exact_parity_passed'] is True and child['rng_roundtrip'] is True
                and child['closed'] is True and child['exitcode']==0 and not child['failed']
                and not child['forced_owned_child_cleanup'] and child['calls']>1,
                'slow child exact parity/clean exit failed')
        child_evidence.append(dict(path=str(child_path.resolve()),sha256=sha(child_path)))
    base['slow_process_exact_parity_and_clean_exit']=True
    base['evidence'].extend(child_evidence)
    report_paths=sorted(Path(smoke['output']).glob('*/episode_*/report.json'))
    evidence=[dict(path=str(p.resolve()),sha256=sha(p)) for p in report_paths]
    reports=[json.loads(p.read_text()) for p in report_paths]
    for r in reports:
        require(r['checkpoint_sha256']==smoke['checkpoint_sha256'], 'report checkpoint mismatch')
    result=evaluate_reports(reports,criteria)
    result.update(report_evidence=evidence,engineering_receipt=str(engineering),
                  engineering_receipt_sha256=sha(engineering))
    if not result['passed']:
        failed=destination.with_name(destination.stem+'.failed.json')
        post.atomic(failed,result)
        raise ValueError('response smoke failed: '+', '.join(result['failures']))
    final=dict(base,response_acceptance=result,
               response_contract={k:v for k,v in smoke.items() if k not in PAIR_DIFFERENCES})
    post.atomic(destination,final)
    return final

def validate_for_formal(config, source, receipt_path):
    """Call before the normal guarded validate_acceptance and formal launch."""
    path=Path(receipt_path);r=json.loads(path.read_text())
    require(config['acceptance_receipt']==str(path), 'formal points to wrong response receipt')
    require(config['acceptance_receipt_sha256']==sha(path), 'response receipt hash changed')
    require(r['passed'] is True and r['response_acceptance']['passed'] is True,
            'actual response gate did not pass')
    require(r['slow_process_exact_parity_and_clean_exit'] is True, 'slow process gate absent')
    require(r['response_acceptance']['kind']=='actual_gpu_smoke_response_v1'
            and r['response_acceptance']['not_recorded_counterfactual'] is True, 'wrong response evidence kind')
    require({k:v for k,v in config.items() if k not in PAIR_DIFFERENCES}==r['response_contract'],
            'formal differs from accepted response configuration')
    require(r['source_contract']==source, 'formal source differs from response smoke')
    response=r['response_acceptance']
    require(sha(response['engineering_receipt'])==response['engineering_receipt_sha256'],
            'engineering receipt changed')
    for item in r['evidence']:
        require(sha(item['path'])==item['sha256'], 'engineering evidence changed')
    reports=[]
    require(len(response['report_evidence'])==4, 'four report hashes required')
    for item in response['report_evidence']:
        require(sha(item['path'])==item['sha256'], 'actual report changed')
        reports.append(json.loads(Path(item['path']).read_text()))
    require(evaluate_reports(reports,response['criteria'])['passed'], 'response recomputation failed')
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--guard-dir',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--min-two-prefix-fraction',type=float,default=.90)
    p.add_argument('--max-median',type=float,default=.60);p.add_argument('--max-p95',type=float,default=.65)
    a=p.parse_args();r=run_gate(a.repo,a.guard_dir,a.destination,
        dict(min_two_prefix_fraction=a.min_two_prefix_fraction,
             max_episode_observation_response_median_s=a.max_median,
             max_episode_observation_response_p95_s=a.max_p95))
    print(json.dumps(r['response_acceptance'],sort_keys=True))

if __name__=='__main__': main()
