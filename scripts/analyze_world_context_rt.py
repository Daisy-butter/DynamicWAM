#!/usr/bin/env python3
"""Report only complete, artifact-paired continuous-protocol comparisons.

This table does not inherit scores from native synchronous DOMINO. Missing or
infrastructure-invalid episodes remain missing, never converted to failures.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random

from make_world_context_rt_manifest import identity, read_json, sha256, write_json
from run_world_context_rt_pipeline import (ARMS, Incomplete, IdentityViolation, distribution, quantile,
    validate_episode,validate_episode_identity,validate_camera_events)


def paired_statistics(baseline, treatment, *, seed=20261008, samples=4000):
    if set(baseline)!=set(treatment) or not baseline:
        raise Incomplete('paired statistics require identical nonempty task/seed keys')
    keys=sorted(baseline); differences=[int(treatment[k])-int(baseline[k]) for k in keys]
    wins=sum(d==1 for d in differences); losses=sum(d==-1 for d in differences)
    discordant=wins+losses
    p=1. if not discordant else min(1.,2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant)
    groups=defaultdict(list)
    for key,d in zip(keys,differences):groups[key[0]].append(d)
    tasks=sorted(groups); rng=random.Random(seed)
    bootstrap=[]
    for _ in range(samples):
        sampled=[groups[rng.choice(tasks)] for _ in tasks]
        bootstrap.append(sum(sum(x) for x in sampled)/sum(map(len,sampled)))
    return dict(episodes=len(keys),tasks=len(tasks),baseline_success=sum(baseline.values()),
        treatment_success=sum(treatment.values()),paired_difference=sum(differences)/len(keys),
        treatment_only_success=wins,baseline_only_success=losses,mcnemar_exact_two_sided_p=p,
        task_cluster_bootstrap_95ci=[quantile(bootstrap,.025),quantile(bootstrap,.975)],bootstrap_seed=seed,
        bootstrap_replicates=samples,ci_scope='descriptive across these fixed task clusters, not a preregistered noninferiority test')


def timing_evidence(reports):
    """Use actually executed physical duration, including a partial final slot."""
    timings=defaultdict(list);query_ms=slot_equivalents=nonhold_equivalents=0.
    slots=nonhold_slots=terminal_count=episodes=calls=completed_requests=missing_acquisition=0
    nominal_feedback_slots=effective_feedback_slots=0.
    effective_feedback_intervals=missing_correction_telemetry=0
    for report in reports:
        r=report['report'];episodes+=1;calls+=r['inference_calls'];terminal_count+=r.get('terminal_inflight') is not None
        requests_by_id={q['request_id']:q for q in r['requests'] if 'request_id' in q}
        submitted_sim={e['request_id']:e['observation_age_at_submit_s']+e['observation_sim_time']
            for e in report.get('_events',[]) if e.get('event')=='submit' and 'observation_age_at_submit_s' in e}
        for e in report.get('_events',[]):
            if e.get('event')=='camera_ready':
                timings['camera_observation_age_at_ready_sim_s'].append(e['receive_sim_time']-e['observation_sim_time'])
                timings['camera_acquisition_to_ready_wall_s'].append(e['camera_ready_wall']-e['acquisition_started_wall'])
                timings['renderer_service_wall_s'].append(e['capture_finished_wall']-e['renderer_started_wall'])
            elif e.get('event')=='submit':
                if 'observation_age_at_submit_s' in e:timings['camera_observation_age_at_policy_submit_sim_s'].append(e['observation_age_at_submit_s'])
        for field,target in (('observation_sim_time','completed_observation_sampling_interval_sim_s'),
                             ('submitted_wall','completed_policy_submission_interval_wall_s'),
                             ('actual_install_sim_time','plan_installation_interval_sim_s'),
                             ('actual_install_wall','plan_installation_interval_wall_s')):
            values=[q[field] for q in r['requests'] if field in q]
            timings[target].extend(b-a for a,b in zip(values,values[1:]))
        for interval in r['control_intervals']:
            duration=max(0.,min(interval['end_sim_time'],r['physics_time'])-interval['start_sim_time'])
            fraction=duration/r['control_period_s'];slot_equivalents+=fraction;slots+=1
            if not interval['hold']:nonhold_equivalents+=fraction;nonhold_slots+=1
            request=requests_by_id.get(interval.get('request_id'))
            if request and not interval['hold'] and duration>0:
                row=round(interval['start_sim_time']/r['control_period_s'])-round(request['action_origin_sim_time']/r['control_period_s'])
                meta=request.get('metadata',{})
                if 0<=row<meta.get('feedback_rows',0):
                    nominal_feedback_slots+=fraction
                    norms=meta.get('feedback_row_l2')
                    if (meta.get('feedback_row_l2_units')=='normalized_action' and isinstance(norms,list) and
                            row<len(norms) and math.isfinite(norms[row])):
                        timings['executed_feedback_row_l2_normalized_action'].append(norms[row])
                        if norms[row]>0:
                            effective_feedback_slots+=fraction;effective_feedback_intervals+=1
                    else:missing_correction_telemetry+=1
        for key in ('tick_lag_s','tick_gap_s','action_observation_age_s'):timings[key].extend(r[key])
        timings['episode_wall_s'].append(r['wall_time']);timings['episode_sim_s'].append(r['physics_time'])
        timings['controller_limited_fraction'].append(r['controller_limited_ticks']/max(r['physics_ticks'],1))
        for field in ('camera_inflight_ticks','stale_camera_frames','overwritten_ready_camera_frames',
                      'executed_feedback_covered_rows','executed_nonzero_feedback_rows','expired_feedback_rows'):
            if field in r:timings[field+'_per_episode'].append(r[field])
        timings['warmup_wall_s'].extend(report.get('warmup_query_wall_s',[]))
        captures=[]
        for index,q in enumerate(r['requests']):
            completed_requests+=1;compute=q['worker_compute_s']*1000;query_ms+=compute
            category='first_online_query' if index==0 else ('slow_refresh' if q.get('metadata',{}).get('mode')=='refresh' else 'fast_reuse')
            timings[category+'_compute_ms'].append(compute)
            timings['request_wall_ms'].append(q['request_wall_s']*1000)
            if 'observation_to_return_wall_s' in q:
                timings['observation_to_return_wall_ms'].append(q['observation_to_return_wall_s']*1000)
            else:missing_acquisition+=1
            # This legacy field starts at snapshot time and includes camera age;
            # it is never used as evidence of model/physics overlap.
            timings['observation_to_return_sim_s'].append(q['inference_physics_advanced'])
            if 'physics_advanced_during_worker_s' in q:
                timings['physics_advanced_during_worker_s'].append(q['physics_advanced_during_worker_s'])
            if not q['expired'] and 'actual_install_sim_time' in q:
                timings['observation_to_install_sim_s'].append(q['actual_install_sim_time']-q['observation_sim_time'])
                if q.get('request_id') in submitted_sim:
                    timings['policy_submission_to_install_sim_s'].append(q['actual_install_sim_time']-submitted_sim[q['request_id']])
                if 'actual_install_wall' in q:
                    timings['policy_submission_to_install_wall_s'].append(q['actual_install_wall']-q['submitted_wall'])
                    if 'camera_ready_wall' in q:timings['camera_ready_to_install_wall_s'].append(q['actual_install_wall']-q['camera_ready_wall'])
            timings['clipped_prefix_rows'].append(q['clipped_prefix'])
            n=q.get('metadata',{}).get('planner_graph',{}).get('captures')
            if type(n)is int:captures.append(n)
        if captures:
            timings['captures_observed_at_first_online_return'].append(captures[0])
            timings['captures_after_first_online_return'].append(captures[-1]-captures[0])
    return dict(episodes=episodes,weighted_worker_compute_ms_per_control_slot=query_ms/slot_equivalents if slot_equivalents else None,
        weighted_worker_compute_ms_per_nonhold_slot=query_ms/nonhold_equivalents if nonhold_equivalents else None,
        worker_compute_ms_sum=query_ms,actual_control_slot_equivalents=slot_equivalents,
        actual_nonhold_slot_equivalents=nonhold_equivalents,control_interval_records=slots,nonhold_interval_records=nonhold_slots,
        hold_physical_time_fraction=1-nonhold_equivalents/slot_equivalents if slot_equivalents else None,
        completed_request_count=completed_requests,submitted_request_count=calls,
        terminal_unfinished_requests=terminal_count,terminal_unfinished_fraction=terminal_count/calls if calls else None,
        worker_cost_is_lower_bound_due_to_terminal_unfinished=bool(terminal_count),
        feedback_window_executed_slot_equivalents=nominal_feedback_slots,
        nonzero_feedback_executed_slot_equivalents=effective_feedback_slots,
        nonzero_feedback_executed_intervals=effective_feedback_intervals,
        correction_telemetry_missing_intervals=missing_correction_telemetry,
        nonzero_feedback_threshold_normalized_action_l2=0.,
        nonzero_feedback_scope='nonzero corrected target rows consumed by the controller; velocity/acceleration clipping may leave actuator targets unchanged, so this is not proof of a physical feedback effect',
        acquisition_timestamp_missing_requests=missing_acquisition,timing={k:distribution(v) for k,v in timings.items()})


def timing_report(path):
    result=read_json(path);events=Path(path).with_name('events.jsonl')
    if events.is_file():
        import json
        result['_events']=[json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    return result


def analyze(run):
    run=Path(run).resolve(strict=True); scenes=read_json(run/'scene_manifest.json')
    formal=[x for x in scenes['episodes'] if x['role']=='formal']
    expected={(r['task'],r['seed']):r for r in formal}
    if len(expected)!=700:raise Incomplete('formal scene manifest must contain exactly 700 unique episodes')
    protocol_sha=sha256(run/'controller_protocol.json'); paired=read_json(run/'paired_scene_manifest.json')
    arms={}; outcomes={}; missing=[];paths_by_arm={}
    frozen=read_json(run/'source_manifest.json')
    for arm in ARMS:
        results={}; artifacts=[]; per_task=defaultdict(lambda:[0,0]);paths={}
        for key,scene in sorted(expected.items()):
            path=run/'eval'/arm/scene['task']/str(scene['seed'])/'summary.json'
            if not path.is_file():
                missing.append(dict(arm=arm,task=key[0],seed=key[1],reason='missing')); continue
            report=read_json(path)
            if scene['scene_id'] not in paired['scenes']:
                raise IdentityViolation('evaluation ran a scene without accepted pairing evidence')
            accepted=paired['scenes'][scene['scene_id']]['accepted_record']
            for name,digest in report.get('source_sha256',{}).items():
                if frozen['files'].get(name)!=digest:raise IdentityViolation('evaluation source changed')
            validate_episode_identity(report,scene,accepted['sha256'],protocol_sha,require_accepted=False)
            if report.get('status')!='complete':
                missing.append(dict(arm=arm,task=key[0],seed=key[1],reason=report.get('status')));continue
            try:
                if sha256(accepted['path'])!=accepted['sha256']:raise IdentityViolation('accepted scene changed')
                validate_episode(report,scene,accepted['sha256'],protocol_sha)
                import json
                with path.with_name('events.jsonl').open() as events:
                    validate_camera_events(json.loads(line) for line in events if line.strip())
            except IdentityViolation:
                raise
            except (Incomplete,RuntimeError,ValueError,OSError) as exc:
                missing.append(dict(arm=arm,task=key[0],seed=key[1],reason=str(exc))); continue
            if report['policy_config'].get('mode')!=arm:
                raise IdentityViolation('evaluation directory contains the wrong policy arm')
            r=report['report'];results[key]=r['success'];per_task[key[0]][0]+=r['success'];per_task[key[0]][1]+=1
            artifacts.append(identity(path))
            paths[key]=path
        outcomes[arm]=results
        paths_by_arm[arm]=paths
        arms[arm]=dict(valid_episodes=len(results),successes=sum(results.values()),
            success_rate=sum(results.values())/len(results) if results else None,
            per_task={k:dict(successes=v[0],episodes=v[1]) for k,v in per_task.items()},
            all_valid_timing=timing_evidence(timing_report(p) for p in paths.values()),artifacts=artifacts)
    common=set.intersection(*(set(v) for v in outcomes.values()))
    pairs={}
    if common:
        for arm in ARMS:
            arms[arm]['common_paired_timing']=(arms[arm]['all_valid_timing'] if common==set(paths_by_arm[arm]) else
                timing_evidence(timing_report(paths_by_arm[arm][k]) for k in sorted(common)))
            arms[arm]['common_paired_successes']=sum(outcomes[arm][k] for k in common)
        for arm in ARMS[1:]:
            pairs[arm+'_vs_official_eager']=paired_statistics(
                {k:outcomes['official_eager'][k] for k in common},{k:outcomes[arm][k] for k in common})
        pairs['world_context_vs_graph_zero']=paired_statistics(
            {k:outcomes['graph_zero'][k] for k in common},{k:outcomes['world_context'][k] for k in common})
    return dict(format='dynamicwam_world_context_rt_analysis',version=1,
        status='complete' if not missing and len(common)==700 else 'incomplete',paired_episodes=len(common),
        requested_episodes=700,paired_tasks=len({k[0] for k in common}),
        paired_scene_keys=[dict(task=k[0],seed=k[1]) for k in sorted(common)],
        controller_protocol=identity(run/'controller_protocol.json'),scene_manifest=identity(run/'scene_manifest.json'),
        paired_scene_manifest=identity(run/'paired_scene_manifest.json'),arms=arms,paired=pairs,missing=missing,
        limitations=['All arms use the new continuous physical controller; no published native-DOMINO comparison is implied.',
            'First online query is measured after explicit graph/model warmup; it is not a cold process or cold graph capture.',
            'The weighted worker metric excludes warmup and unfinished terminal requests; wall/simulation distributions report actual execution.',
            'Consumed nonzero feedback targets do not establish a physical actuator change relative to zero feedback; the controller can clip both targets to the same command.',
            'Missing or realtime-invalid episodes are excluded and block a complete 700-pair conclusion.',
            'Teacher agreement and small development canaries are separate from formal closed-loop task success.'])


def markdown(report):
    rows=['# Continuous-physics evaluation', '',
        f"Status: **{report['status']}**. Common valid coverage: **{report['paired_episodes']}/700 episodes**, "
        f"{report['paired_tasks']} tasks. Missing/invalid arm-episodes: {len(report['missing'])}.", '',
        'All comparisons below use the same accepted scenes under the new continuous controller.', '',
        '| Arm | Common successes | Worker ms / actual slot | Fast p95 ms | Observation-to-return p95 ms |',
        '|---|---:|---:|---:|---:|']
    def f(value):return 'unavailable' if value is None else f'{value:.3f}'
    for arm,value in report['arms'].items():
        timing=value.get('common_paired_timing',{});dist=timing.get('timing',{})
        rows.append(f"| {arm} | {value.get('common_paired_successes',0)}/{report['paired_episodes']} | "
            f"{f(timing.get('weighted_worker_compute_ms_per_control_slot'))} | "
            f"{f(dist.get('fast_reuse_compute_ms',{}).get('p95'))} | "
            f"{f(dist.get('observation_to_return_wall_ms',{}).get('p95'))} |")
    rows+=['','Worker cost excludes unfinished terminal requests and is explicitly a lower bound when those exist. '
        'The denominator uses actual physical duration, including a partial final interval. '
        'Consumed nonzero feedback rows establish target delivery to the controller, not a counterfactual actuator or task effect.','']
    for name,pair in report['paired'].items():
        low,high=pair['task_cluster_bootstrap_95ci']
        rows.append(f"- {name}: {pair['paired_difference']*100:+.2f} percentage points; "
            f"task-cluster 95% interval [{low*100:+.2f}, {high*100:+.2f}], "
            f"discordance {pair['treatment_only_success']} wins / {pair['baseline_only_success']} losses.")
    rows+=['','These are descriptive intervals. Neither a small canary nor teacher-error improvement proves noninferiority.',
        'Native synchronous DOMINO scores are not included in this table. See analysis.json for every source identity, '
        'per-task count, missing scene, timing distribution, and capture counter.','']
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=analyze(a.run_root);write_json(a.output,result,exclusive=True)
    with a.output.with_suffix('.md').open('x',encoding='utf-8') as stream:stream.write(markdown(result))
    if result['status']!='complete':raise SystemExit(2)


if __name__=='__main__':main()
