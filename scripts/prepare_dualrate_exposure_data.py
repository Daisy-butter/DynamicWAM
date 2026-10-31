"""Build V4 event-coupled indices from read-only demonstrations and timing traces.

The engineering fixture is explicitly synthetic. Formal training requires
measured parallel traces; neither mode manufactures observation timestamps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from dynamicwam.dualrate.exposure_timing import (
    EVENT_SCHEMA, TIME_CONTRACT, synthetic_templates, validate_templates, validate_template)
from dynamicwam.dualrate.producer import (
    NUMERIC_TIME_EPSILON, RGB_TOLERANCE, CONDITION_FRAME_TOLERANCE_SECONDS, CONDITION_RGB_MAX_AGE_SECONDS,
    bound_metadata_path, unique_last_indices, select_frame, _action_targets_supported, verify_file_stat)
from dynamicwam.dualrate.data import sha256_file


def load_events(path):
    text=Path(path).read_text()
    if Path(path).suffix=='.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value=json.loads(text)
    return value if isinstance(value,list) else value['events']


def audit_wall_release(events,dt):
    """Verify release evidence on the original clock, before translating rows."""
    activation=[(i,e) for i,e in enumerate(events) if e.get('event')=='activation']
    if len(activation)!=1:raise ValueError('exactly one measured clock activation required per trace')
    active_index,anchor=activation[0]
    physical=float(anchor['physics_time']);wall=float(anchor['wall_time'])
    quantum=float(anchor['physics_step_seconds']);ticks=int(anchor['action_ticks'])
    if quantum<=0 or abs(quantum*ticks-dt)>1e-8:raise ValueError('trace action/physics grid differs')
    starts={int(e['query_id']):e for e in events if e.get('event')=='fast_started'}
    checked=0
    for event in events[active_index+1:]:
        kind=event.get('event')
        if kind=='capture':
            ready=float(event['owner_ready_wall']);published=float(event['available'])
            owner=float(event['timing']['wall_timestamps'][-1])
        elif kind=='slow_completed' and event.get('accepted') is True:
            ready=float(event['transfer_ready_wall_time']);published=float(event['published_time'])
            owner=float(event['publication_wall_time'])
        elif kind=='fast_committed':
            ready=float(event['ready_wall_time']);published=float(event['commit_time'])
            owner=float(event['commit_wall_time']);start=starts[int(event['query_id'])]
            deadline=wall+float(start['protected_until'])-physical-float(start['submit_margin'])
            if event.get('submission_contract','owner_v4')=='worker_receipt_v8':
                matches=[e for e in events if e.get('event')=='fast_submitted' and e.get('query_id')==event['query_id']]
                if len(matches)!=1:raise ValueError('exactly one worker receipt required')
                receipt=matches[0];submitted=float(receipt['submitted_wall_time'])
                if (submitted!=float(event['submitted_wall_time']) or not ready<=submitted<=owner
                        or receipt['prefix_hash']!=start['prefix_hash'] or submitted>=deadline
                        or published>=float(start['protected_until'])-float(start['submit_margin'])):
                    raise ValueError('invalid actual receipt or physical activation deadline')
                ready=submitted
            elif max(ready,owner)>=deadline:raise ValueError('measured fast query missed wall deadline')
        else:continue
        if not all(math.isfinite(v) for v in (ready,published,owner)) or owner<ready or ready<wall:
            raise ValueError('invalid measured monotonic release timestamps')
        lower=physical+max(0,math.ceil((ready-wall-1e-7)/quantum))*quantum
        if published<lower-1e-6 or published>physical+owner-wall+quantum+1e-6:
            raise ValueError('physical publication predates wall readiness or advances simulated time')
        if abs((published-physical)/quantum-round((published-physical)/quantum))*quantum>1e-6:
            raise ValueError('publication is not a completed real physics step')
        checked+=1
    if not checked:raise ValueError('trace has no active measured wall releases')
    return checked


def measured_templates(paths,dt):
    templates=[]; evidence=[]; overlaps=0; all_versions=set()
    for trace_number,path in enumerate(paths):
        path=Path(path).resolve(strict=True)
        events=load_events(path);audit_wall_release(events,dt)
        starts={};completed={};captures={};commits={};published_order={};started_order={}
        for event_index,event in enumerate(events):
            kind=event.get('event')
            if kind=='capture':captures[float(event['exposure'])]=float(event['available'])
            elif kind=='slow_started':
                starts[int(event['version'])]=event;started_order[int(event['version'])]=event_index
            elif kind in ('slow_completed','slow_published') and event.get('accepted') is True:
                if 'published_time' not in event:
                    raise ValueError('new traces must explicitly record actual M publication time')
                completed[int(event['version'])]=event
                published_order[int(event['version'])]=event_index
            elif kind=='fast_committed':commits[int(event['query_id'])]=event
        for event_index,event in enumerate(events):
            if event.get('event')!='fast_started' or int(event['query_id']) not in commits:
                continue
            version=int(event['reference_version']);q=float(event['query_time'])
            if event_index<=next(i for i,e in enumerate(events) if e.get('event')=='activation'):
                continue  # bootstrap inference is explicitly outside the active clock
            # Prove latest-complete selection, not merely that one M was available.
            available=[v for v,d in completed.items() if published_order[v]<event_index and
                       float(d['published_time'])<=q+NUMERIC_TIME_EPSILON]
            if not available:
                raise ValueError('active query lacks a real causal slow supervision anchor')
            reference_available=version>=0
            anchor=max(available)
            if reference_available and version!=anchor:
                raise ValueError('trace fast query did not select latest published reference')
            if anchor not in starts:raise ValueError('historical slow request missing from trace')
            # A missing/expired reference is NEVER marked available to the fast
            # network. Its last actual completed source is used only to pair an
            # independent native slow supervision example with this fast query.
            source=starts[anchor];done=completed[anchor]
            e=float(source['source_time']);r=float(source['request_time']);c=float(done['published_time'])
            if q<=e+NUMERIC_TIME_EPSILON:continue
            rgb=float(event['observation_time'])
            matching=[(t,a) for t,a in captures.items() if abs(t-rgb)<=NUMERIC_TIME_EPSILON]
            if not matching:continue  # bootstrap image has no ordinary capture event
            a=float(source['source_available_time'])
            row=dict(source_available=a-e,request_time=r-e,publish_time=c-e,query_time=q-e,
                query_rgb_time=rgb-e,query_rgb_available=matching[-1][1]-e,
                source_prefix_end=float(source['prefix_end'])-e,
                fast_prefix_end=float(event['protected_until'])-e,
                fast_submit_time=float(commits[int(event['query_id'])]['commit_time'])-e,
                submit_margin=float(event['submit_margin']),trace_number=trace_number,
                reference_version=version,reference_available=reference_available,
                slow_supervision_version=anchor,query_id=int(event['query_id']))
            validate_template(row,dt);templates.append(row)
            if reference_available:all_versions.add((trace_number,version))
            for other_version,other in starts.items():
                if other_version==version:continue
                # Event order disambiguates equal physical ticks and includes a
                # refresh still unfinished when the finite diagnostic ends.
                if started_order[other_version]<event_index<published_order.get(other_version,math.inf):
                    overlaps+=1;break
        evidence.append(dict(path=str(path),sha256=sha256_file(path)))
    return validate_templates(dict(schema=EVENT_SCHEMA,time_contract=TIME_CONTRACT,kind='measured_parallel',
        clock_domain='scene_step_physics_seconds',action_interval_seconds=dt,templates=templates,
        trace_evidence=evidence,fast_queries_while_slow_running=overlaps,
        consumed_reference_versions=len(all_versions),
        no_reference_queries=sum(not t['reference_available'] for t in templates),
        provenance='coupled actual reference/camera/query/prefix events; offline expert actions are not on-policy'))


def eligible_rows(times,templates,dt,stride,episode_uid,templates_per_source,*,partial_fast_tail=False):
    """Keep source REAL exposures; all other conditions use the same translated axis."""
    times=np.asarray(times,dtype=np.float64)
    next_source=-float('inf')
    valid=unique_last_indices(times)
    for frame in valid:
        e=float(times[frame])
        if e<next_source-NUMERIC_TIME_EPSILON:continue
        next_source=e+stride
        if e+16*dt>times[-1]+NUMERIC_TIME_EPSILON:continue
        try:
            for target in e+np.arange(1,9)*2*dt:
                select_frame(times,target,past_only=False,tolerance=RGB_TOLERANCE)
        except ValueError:continue
        if not _action_targets_supported(times,e+np.arange(1,17)*dt).all():continue
        # Rotation balances templates without multiplying the entire dataset by
        # every query in a long trace. Eligibility scans the remaining templates.
        key=f'{episode_uid}:{int(frame)}'.encode()
        first=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%len(templates)
        added=0
        for shift in range(len(templates)):
            template_id=(first+shift)%len(templates)
            row=validate_template(templates[template_id],dt)
            q=e+row['query_time'];start=e+row['fast_action_start']
            needed=row['fast_prefix_rows']+2 if partial_fast_tail else 16
            if start+needed*dt>times[-1]+NUMERIC_TIME_EPSILON:continue
            try:
                frame_q=select_frame(times,e+row['query_rgb_time'],past_only=True,
                                    tolerance=CONDITION_FRAME_TOLERANCE_SECONDS)
                if q-float(times[frame_q])>CONDITION_RGB_MAX_AGE_SECONDS+NUMERIC_TIME_EPSILON:continue
            except ValueError:continue
            if not _action_targets_supported(times,start+np.arange(1,needed+1)*dt).all():continue
            yield int(frame),template_id
            added+=1
            if added>=templates_per_source:break


def build(source,output,events,*,stride,templates_per_source=2,max_episodes=None,partial_fast_tail=False,train_fraction=1.):
    import h5py
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    if output.exists():raise FileExistsError('new dataset directory required')
    validate_templates(events);dt=float(events['action_interval_seconds'])
    if stride<=0 or templates_per_source<1:raise ValueError('positive index sampling settings required')
    parents={s:json.loads((source/f'{s}.json').read_text()) for s in ('train','validation')}
    m=parents['train'];episodes=json.loads(bound_metadata_path(m,source,'episodes_file').read_text())
    for parent in parents.values():
        for key in ('episodes_file','text_index','prompts_file'):
            bound_metadata_path(parent,source,key)
        for key in ('episodes_file_sha256','source_file_stats_sha256','native_checkpoint_sha256',
                    'action_normalization_sha256','motion_normalization_sha256'):
            if parent[key]!=m[key]:raise ValueError('parent split metadata identity mismatch: '+key)
    stat_path=(source/m['source_file_stats']).resolve(strict=True)
    if sha256_file(stat_path)!=m['source_file_stats_sha256']:raise ValueError('source stat identity changed')
    stats=json.loads(stat_path.read_text())['episodes'];rows={s:[] for s in parents}
    selected=episodes if max_episodes is None else episodes[:max_episodes]
    if not 0<train_fraction<=1:raise ValueError('training episode fraction must be in (0,1]')
    groups={}
    for i,ep in enumerate(selected):
        if ep['split']=='train':groups.setdefault('/'.join(ep['episode_uid'].split('/')[:-1]),[]).append(i)
    chosen=set()
    for values in groups.values():
        values.sort(key=lambda i:hashlib.sha256(('v8-half:'+selected[i]['episode_uid']).encode()).hexdigest())
        chosen.update(values[:max(1,math.ceil(len(values)*train_fraction))])
    # A diagnostic subset retains original episode ids and split identities.
    for episode_id,ep in enumerate(selected):
        if ep['split']=='train' and episode_id not in chosen:continue
        verify_file_stat(ep['raw_path'],stats[ep['episode_uid']]['raw'])
        with h5py.File(ep['raw_path'],'r') as h:times=h['interception/sim_time_seconds'][:]
        rows[ep['split']].extend((episode_id,frame,t) for frame,t in eligible_rows(times,events['templates'],dt,stride,
            ep['episode_uid'],templates_per_source,partial_fast_tail=partial_fast_tail))
        if (episode_id+1)%1000==0:
            print(json.dumps(dict(episodes=episode_id+1,rows={s:len(v) for s,v in rows.items()})),flush=True)
    if not all(rows.values()):raise ValueError('timing index must retain both original data splits')
    missing_template_ids={i for i,t in enumerate(events['templates']) if not t.get('reference_available',True)}
    template_counts={split:np.bincount(np.asarray(values,dtype=np.int64)[:,2],
        minlength=len(events['templates'])).tolist() for split,values in rows.items()}
    no_reference_rows={split:sum(row[2] in missing_template_ids for row in values) for split,values in rows.items()}
    if missing_template_ids and not all(no_reference_rows.values()):
        raise ValueError('index lost measured missing-reference examples from a split')
    output.mkdir(parents=True)
    event_path=output/'event_templates.json';event_path.write_text(json.dumps(events,indent=2))
    audit=dict(schema='dualrate-exposure-index-v4',time_contract=TIME_CONTRACT,episodes=len(selected),
        rows={s:len(v) for s,v in rows.items()},source_frame_semantics='real exposure; never nominal relabel',
        row_semantics=['episode_index','real_source_frame_index','coupled_event_template_id'],
        source_stride_seconds=stride,templates_per_source=templates_per_source,
        future_rgb_error_bound_seconds=RGB_TOLERANCE,query_rgb_old_side_error_bound_seconds=CONDITION_FRAME_TOLERANCE_SECONDS,
        label_time_claim='bounded nearest RGB and bracketed geometric/action interpolation; not exact measurements',
        provenance=events['kind'],simulator_started=False)
    audit['no_reference_rows']=no_reference_rows
    audit['template_event_counts']=[1]*len(events['templates'])
    audit['template_row_counts']=template_counts
    audit['measured_no_reference_fraction']=len(missing_template_ids)/len(events['templates'])
    audit['indexed_no_reference_fraction']={split:no_reference_rows[split]/len(values) for split,values in rows.items()}
    audit['sampling_semantics']='uniform eligible indexed rows; event templates extended to expert trajectories; not exact measured-frequency equivalence'
    audit.update(train_fraction=train_fraction,selected_train_episodes=len(chosen),validation_episodes_unchanged=True,
                 training_statistics='original training split only; includes unselected training episodes',
                 partial_fast_tail=partial_fast_tail)
    audit_path=output/'timing_audit.json';audit_path.write_text(json.dumps(audit,indent=2))
    result={}
    for split,parent in parents.items():
        new=dict(parent)
        for key in ('episodes_file','text_index','prompts_file','source_file_stats'):
            new[key]=str((source/parent[key]).resolve(strict=True))
        for key in ('completion_distribution','timing_derivation'):
            new.pop(key,None)
        row_path=output/f'{split}_rows.npy';np.save(row_path,np.asarray(rows[split],dtype=np.int64))
        new.update(time_contract=TIME_CONTRACT,action_interval_seconds=dt,
            rows_file=str(row_path),rows_sha256=sha256_file(row_path),
            event_templates_file=str(event_path),event_templates_file_sha256=sha256_file(event_path),
            query_ages_seconds=sorted(set(float(t['query_time']) for t in events['templates'])),
            source_stride_seconds=stride,latency_provenance=events['kind']+':coupled_deployment_events',
            condition_rgb_rule='source=real_exposure; current=real_past_frame_under_replayed_camera_availability',
            parent_manifest=str(source/f'{split}.json'),parent_manifest_sha256=sha256_file(source/f'{split}.json'),
            audit_file=str(audit_path),formal_training_ready=False,
            no_reference_rows=no_reference_rows[split],
            reference_availability_semantics='measured runtime availability; expired source remains slow-label-only',
            timing_quality='deterministic event/queue alignment; bounded demonstration resampling error')
        if partial_fast_tail:new['continuous_tail_policy']='valid_prefix_and_two_free_rows'
        path=output/f'{split}.json';path.write_text(json.dumps(new,indent=2));result[split]=str(path)
    return dict(manifests=result,audit=audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--trace',action='append',default=[])
    p.add_argument('--synthetic-engineering',action='store_true')
    p.add_argument('--templates-per-source',type=int,default=2)
    p.add_argument('--max-episodes',type=int)
    p.add_argument('--partial-fast-tail',action='store_true')
    p.add_argument('--train-fraction',type=float,default=1.)
    args=p.parse_args()
    if bool(args.trace)==bool(args.synthetic_engineering):p.error('choose measured traces OR explicit engineering fixture')
    dt=45*float(np.float32(.004))
    events=synthetic_templates(dt) if args.synthetic_engineering else measured_templates(args.trace,dt)
    print(json.dumps(build(args.source,args.output,events,stride=2*dt,
        templates_per_source=args.templates_per_source,max_episodes=args.max_episodes,
        partial_fast_tail=args.partial_fast_tail,train_fraction=args.train_fraction)))


if __name__=='__main__':main()
