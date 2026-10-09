#!/usr/bin/env python3
"""Paired valid-intersection analysis for the predictive continuous controller."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from make_world_context_rt_manifest import identity,read_json,sha256,write_json
from run_predictive_feedback_pipeline import ARMS,MECHANISM_ARM,mechanism_scenes
from run_world_context_rt_pipeline import validate_episode,validate_episode_identity,validate_camera_events,IdentityViolation,Incomplete
from analyze_world_context_rt import timing_evidence,timing_report,paired_statistics,markdown as main_markdown


def paired_keys(outcomes):
    """The smaller mechanism group must never shrink the three-arm main cohort."""
    main=set.intersection(*(set(outcomes[arm]) for arm in ARMS))
    mechanism=set(outcomes['predictive'])&set(outcomes[MECHANISM_ARM])
    return main,mechanism


def analyze(run):
    scenes=read_json(run/'scene_manifest.json');pairing=read_json(run/'paired_scene_manifest.json')
    frozen=read_json(run/'source_manifest.json')
    protocol_sha=sha256(run/'controller_protocol.json')
    requested={(s['task'],s['seed']):s for s in scenes['episodes'] if s['role']=='formal'}
    if len(requested)!=700:raise IdentityViolation('formal manifest must retain all fixed 700 scenes')
    subset=mechanism_scenes(list(requested.values()))
    registration=scenes.get('mechanism_arm',{})
    if (registration.get('name')!=MECHANISM_ARM or registration.get('ablation')!='current_only' or
        registration.get('scene_ids')!=[s['scene_id'] for s in subset] or registration.get('requested_episodes')!=175):
        raise IdentityViolation('mechanism scenes differ from the preregistered first five seeds per task')
    checkpoint_sha=sha256(run/'stage3/checkpoint.pt')
    arms={};outcomes={};paths={};missing=[]
    for arm in (*ARMS,MECHANISM_ARM):
        results={};valid_paths={};per_task=defaultdict(lambda:[0,0])
        arm_requested=requested if arm in ARMS else {(s['task'],s['seed']):s for s in subset}
        for key,scene in sorted(arm_requested.items()):
            accepted=pairing['scenes'].get(scene['scene_id']);path=run/'eval'/arm/key[0]/str(key[1])/'summary.json'
            if accepted is None or not path.exists():
                missing.append(dict(arm=arm,task=key[0],seed=key[1],reason='preparation or result missing'));continue
            report=read_json(path);digest=accepted['accepted_record']['sha256']
            for name,value in report.get('source_sha256',{}).items():
                if frozen['files'].get(name)!=value:raise IdentityViolation('episode source differs from frozen experiment')
            if sha256(accepted['accepted_record']['path'])!=digest:
                raise IdentityViolation('paired accepted bytes changed after evaluation')
            validate_episode_identity(report,scene,digest,protocol_sha,require_accepted=False)
            try:
                validate_episode(report,scene,digest,protocol_sha)
                with path.with_name('events.jsonl').open() as stream:validate_camera_events(json.loads(x) for x in stream if x.strip())
            except IdentityViolation:raise
            except (Incomplete,RuntimeError,ValueError,OSError) as exc:
                missing.append(dict(arm=arm,task=key[0],seed=key[1],reason=str(exc)));continue
            cfg=report['policy_config'];expected_mode='predictive' if arm==MECHANISM_ARM else arm
            expected_ablation='current_only' if arm==MECHANISM_ARM else 'none'
            if cfg.get('mode')!=expected_mode or cfg.get('ablation')!=expected_ablation:
                raise IdentityViolation('wrong evaluation mode or mechanism ablation')
            if expected_mode=='predictive' and cfg.get('feedback_checkpoint_sha256')!=checkpoint_sha:
                raise IdentityViolation('full and mechanism arms must use the exact same final checkpoint')
            results[key]=report['report']['success'];valid_paths[key]=path
            per_task[key[0]][0]+=int(results[key]);per_task[key[0]][1]+=1
        paths[arm]=valid_paths;outcomes[arm]=results
        arms[arm]=dict(requested_episodes=len(arm_requested),valid_episodes=len(results),successes=sum(results.values()),
            success_rate=sum(results.values())/len(results) if results else None,
            per_task={k:dict(successes=v[0],episodes=v[1]) for k,v in per_task.items()},
            all_valid_timing=timing_evidence(timing_report(p) for p in valid_paths.values()),
            artifacts=[identity(p) for p in valid_paths.values()])
    common,mechanism_common=paired_keys(outcomes);pairs={}
    if common:
        for arm in ARMS:
            arms[arm]['common_paired_successes']=sum(outcomes[arm][k] for k in common)
            arms[arm]['common_paired_timing']=timing_evidence(timing_report(paths[arm][k]) for k in sorted(common))
        for baseline,treatment in [('official_eager','graph_zero'),('official_eager','predictive'),('graph_zero','predictive')]:
            pairs[f'{treatment}_vs_{baseline}']=paired_statistics({k:outcomes[baseline][k] for k in common},
                {k:outcomes[treatment][k] for k in common},seed=20261009)
    mechanism=dict(status='complete' if len(mechanism_common)==175 else 'incomplete',requested_episodes=175,
        paired_episodes=len(mechanism_common),paired_tasks=len({k[0] for k in mechanism_common}),
        selection=registration,checkpoint_sha256=checkpoint_sha,arms={},paired=None,
        paired_scene_keys=[dict(task=t,seed=s) for t,s in sorted(mechanism_common)])
    for arm in ('predictive',MECHANISM_ARM):
        mechanism['arms'][arm]=dict(successes=sum(outcomes[arm][k] for k in mechanism_common),
            episodes=len(mechanism_common),timing=timing_evidence(timing_report(paths[arm][k]) for k in sorted(mechanism_common)))
    if mechanism_common:
        mechanism['paired']=paired_statistics({k:outcomes[MECHANISM_ARM][k] for k in mechanism_common},
            {k:outcomes['predictive'][k] for k in mechanism_common},seed=20261009)
    return dict(format='dynamicwam_predictive_feedback_analysis',version=1,
        status='complete' if len(common)==700 and len(mechanism_common)==175 and not missing else 'incomplete',paired_episodes=len(common),
        main_status='complete' if len(common)==700 else 'incomplete',mechanism=mechanism,
        paired_tasks=len({k[0] for k in common}),requested_episodes=700,arms={arm:arms[arm] for arm in ARMS},
        mechanism_all_valid=arms[MECHANISM_ARM],paired=pairs,missing=missing,
        paired_scene_keys=[dict(task=t,seed=s) for t,s in sorted(common)],
        controller_protocol=identity(run/'controller_protocol.json'),scene_manifest=identity(run/'scene_manifest.json'),
        paired_scene_manifest=identity(run/'paired_scene_manifest.json'),
        limitations=['All three arms share the new continuous controller; historical native DOMINO scores are not comparators.',
            'Infrastructure-invalid and missing results are excluded, never counted as policy failures.',
            'Timing comparisons use the same paired task/seed intersection; unfinished terminal query cost is a lower bound.',
            'The 175-scene current-only mechanism arm is paired separately and never reduces the 700-scene main cohort.',
            'Small development probes/canaries do not establish noninferiority or action-causal identification.'])


def markdown(report):
    text=main_markdown(report);m=report['mechanism']
    lines=['','## Same-checkpoint mechanism comparison','',
        f"Coverage: **{m['paired_episodes']}/175** paired scenes; first five registered numeric seeds of each task. "
        'This subset is separate from the main 700-scene cohort.','',
        '| Arm | Successes | Worker ms / actual slot | Fast p95 ms |','|---|---:|---:|---:|']
    def f(v):return 'unavailable' if v is None else f'{v:.3f}'
    for arm,v in m['arms'].items():
        t=v['timing'];p95=t.get('timing',{}).get('fast_reuse_compute_ms',{}).get('p95')
        lines.append(f"| {arm} | {v['successes']}/{v['episodes']} | {f(t.get('weighted_worker_compute_ms_per_control_slot'))} | {f(p95)} |")
    if m['paired']:
        p=m['paired'];lo,hi=p['task_cluster_bootstrap_95ci']
        lines+=['',f"Full prediction minus current-only: {100*p['paired_difference']:+.2f} percentage points; "
            f"descriptive task-cluster 95% interval [{100*lo:+.2f}, {100*hi:+.2f}]."]
    lines+=['','Both use the identical final checkpoint; the current-only arm bypasses prediction inputs. '
        'This ablation does not by itself identify the effect of actions or prove a causal world model.','']
    return text+'\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=analyze(a.run_root)
    write_json(a.output,report,exclusive=True)
    with a.output.with_suffix('.md').open('x') as stream:stream.write(markdown(report))
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
