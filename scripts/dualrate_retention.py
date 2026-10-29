"""Retain four audited whole tasks; formal configuration remains 35 x 20.

CPU-only artifact operations. Originals stay untouched and keep their original
source/config hashes. This module never launches, retries or controls a process.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='dualrate_four_complete_tasks_retention_v1'
RETAINED={'place_object_basket','hanging_mug','place_bread_basket','place_container_plate'}
OPERATIONAL={'stage','tasks','episodes','output','result_stamp','acceptance_receipt',
    'acceptance_receipt_sha256','task_timeout_seconds','cohort_timeout_seconds',
    'source_compatibility_receipt','source_compatibility_sha256','deployment_artifact_sha256',
    'retained_task_receipt','retained_task_receipt_sha256'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)
def binding(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=sha(path))
def check_binding(row):require(sha(row['path'])==row['sha256'],'retained evidence changed: '+row['path'])

def artifacts(config):
    result={}
    for path,digest in config['deployment_artifact_sha256'].items():
        key=('native_bridge/'+Path(path).name if '/src/dynamicwam/runtime/domino_rt/_sapien_color_reader' in path else path)
        require(key not in result,'ambiguous deployment artifact identity')
        result[key]=digest
    return result

def compare_contracts(old,new):
    require(old['run_kind']==new['run_kind']=='dualrate_eval_suite','wrong suite mode')
    require(old['stage']=='formal' and old['episodes']==20 and len(old['tasks'])==35,'original must be full35x20')
    for key in set(old)|set(new):
        if key not in OPERATIONAL:
            require(key in old and key in new and old[key]==new[key],'retention changes policy/bench contract: '+key)
    require(artifacts(old)==artifacts(new),'retention changes native scene/camera/scoring artifact bytes')
    for config in (old,new):
        require(sha(config['source_compatibility_receipt'])==config['source_compatibility_sha256'],
            'source compatibility receipt bytes changed')
    before=read(old['source_compatibility_receipt']);after=read(new['source_compatibility_receipt'])
    for key in ('training_source_sha256','same_model_files_sha256','inference_only_files_sha256','camera_executor_change'):
        require(key in before and before[key]==after.get(key),'protected model/time/camera source changed: '+key)

def ordered_tasks(old_tasks,smoke_tasks):
    require(len(old_tasks)==35 and len(set(old_tasks))==35,'exact35 distinct original tasks required')
    require(len(smoke_tasks)==2 and len(set(smoke_tasks))==2 and set(smoke_tasks)<=set(old_tasks),'two actual smoke tasks required')
    result=list(old_tasks)
    # Preserve each peer pairing and GPU index; only whole cohorts may move.
    old_pairs=[tuple(old_tasks[i:i+2]) for i in range(0,35,2)]
    new_pairs=[tuple(result[i:i+2]) for i in range(0,35,2)]
    require(collections.Counter(old_pairs)==collections.Counter(new_pairs),'task reordering changes cohort/GPU roles')
    require(all(not (set(pair)&RETAINED) or set(pair)<=RETAINED for pair in new_pairs),'retention splits a cohort')
    require([t for t in result if t not in RETAINED][:2]==smoke_tasks,'smoke must cover next unretained cohort')
    return result

def validate_proof(path,old_source,new_source):
    proof=read(path)
    require(proof.get('schema')=='dualrate_retention_semantics_v1'
        and proof.get('old_source')==old_source and proof.get('new_source')==new_source,
        'semantic review proof belongs to different sources')
    require(proof.get('active_policy_clock_camera_queue_unchanged') is True
        and proof.get('changes_limited_to_diagnostics_retention_supervision') is True
        and proof.get('gc_control_changed') is False,'active semantics not certified unchanged')
    require(bool(proof.get('evidence')),'semantic proof requires hashed reviewed diff/evidence')
    for item in proof['evidence']:check_binding(item)
    return proof

def failed_release(directory,config):
    directory=Path(directory);status=read(directory/'status.json');owned=read(directory/'owned_exit.json')
    kernel=read(directory/'kernel_after.json');auth=read(directory/'authorization.json')
    require(status.get('state')=='failed' and status.get('guard_closed') is True
        and owned.get('remaining')==[] and type(owned.get('returncode')) is int,'old failed run is not cleanly released')
    require(kernel.get('unchanged') is True and kernel.get('same_boot') is True and kernel.get('added')==[],
        'old failed kernel/boot changed')
    require(read(directory/'config.frozen.json')==config and auth.get('identity',{}).get('config_sha256')==canonical(config)
        and auth.get('gpu_uuids')==config['gpu_uuids'],'old guard/config/authorization differs')
    events=read(directory/'guard_events.json')
    opened=[(x['uuid'],x['path'],x['fd']) for x in events if x['event']=='after_open']
    closed=[(x['uuid'],x['path'],x['fd']) for x in events if x['event']=='after_close']
    require(len(opened)==len(closed)==8 and set(opened)==set(closed)
        and {x[0] for x in opened}==set(config['gpu_uuids']),'old eight GPU descriptors not closed exactly')
    return [directory/name for name in ('status.json','owned_exit.json','kernel_before.json','kernel_after.json',
        'source.json','config.frozen.json','authorization.json','guard_events.json')]

def audit_original_task(config,task,source,*,decode=True):
    from dualrate_eval_suite import audit_task
    from dualrate_eval_worker import load_frozen_task
    from video_artifacts import probe_head_video
    require(not config.get('retained_task_receipt'),'nested retention is unsupported')
    reports,metrics=audit_task(config,task);references=load_frozen_task(config,task)
    base=Path(config['output'])/task;completed=read(base/'completed.json')
    evidence=[base/'completed.json',base/'deployment_identity.json',base/'slow_process.json']
    child=read(base/'slow_process.json')
    require(child.get('bootstrap_exact_parity_passed') is True and child.get('rng_roundtrip') is True
        and child.get('closed') is True and child.get('exitcode')==0 and child.get('failed') is False
        and child.get('forced_owned_child_cleanup') is False,'retained slow child parity/closure failed')
    require(len(reports)==len(references)==20,'retained task is not exactly20 episodes')
    for index,(report,(descriptor,_,motion)) in enumerate(zip(reports,references)):
        folder=base/f'episode_{index:04d}';actual=read(folder/'manifest.json')
        require(all(actual.get(k)==v for k,v in descriptor.items()) and (folder/'expert_motion.pkl').read_bytes()==motion,
            'retained scene descriptor/motion differs')
        require(report.get('source_sha256')==source['source_sha256'],'retained report source differs')
        if decode:require(probe_head_video(Path(report['video']))==report['video_probe'],'retained MP4 full decode differs')
        evidence.extend(folder/name for name in ('report.json','metrics.json','manifest.json','expert_motion.pkl'))
        evidence.extend((Path(report['video']),Path(report['video_timestamps'])))
    native=Path(completed['native_result_dir'])
    evidence.extend(native/name for name in ('_result.txt','_metrics.json','_metrics_report.txt','_episodes_detail.json'))
    return reports,metrics,evidence

def prepare(old_config,old_guard,candidate_config,proof_path,destination):
    import guarded_dualrate_run as guard
    old_config=Path(old_config).resolve(strict=True);old=read(old_config);candidate=read(candidate_config)
    require(not Path(destination).exists(),'retention receipt must be new')
    compare_contracts(old,candidate)
    from dualrate_eval_worker import validate_source_compatibility
    validate_source_compatibility(candidate)
    progress=read(Path(old['output'])/'suite_status.json');started=read(Path(old['output'])/'suite_started.json')
    finished=progress.get('finished_tasks',[])
    require(progress.get('state')=='failed' and len(finished)==4 and set(finished)==RETAINED,'retain only exact4 fully finished tasks')
    require(set(old['tasks'][:4])==RETAINED and started['config_sha256']==canonical(old),'original task ordering/config differs')
    old_source=read(Path(old_guard)/'source.json');new_source=guard.frozen_source(ROOT)
    require(started['source']==old_source,'old suite/guard source differs')
    validate_proof(proof_path,old_source,new_source)
    formal_tasks=ordered_tasks(old['tasks'],candidate['tasks'][:2])
    files=failed_release(old_guard,old)+[old_config,Path(old['output'])/'suite_status.json',Path(old['output'])/'suite_started.json',Path(proof_path)]
    for key in ('checkpoint','init_checkpoint','profile'):
        require(sha(old[key])==old[key+'_sha256'],'retained checkpoint/profile changed')
    require(sha(Path(old['prepared'])/'frozen_manifest.json')==old['prepared_sha256'],'prepared scene set changed')
    files.extend((Path(old['prepared'])/'frozen_manifest.json',Path(candidate['source_compatibility_receipt'])))
    require(sha(candidate['source_compatibility_receipt'])==candidate['source_compatibility_sha256'],'candidate source compatibility changed')
    task_audits=[]
    for task in old['tasks'][:4]:
        reports,_,evidence=audit_original_task(old,task,old_source,decode=True);files.extend(evidence)
        task_audits.append(dict(task=task,episodes=20,successes=sum(bool(r['success']) for r in reports)))
    result=dict(schema=SCHEMA,complete=True,old_config=binding(old_config),old_guard=str(Path(old_guard).resolve()),
        old_source=old_source,new_source=new_source,semantic_proof=binding(proof_path),formal_tasks=formal_tasks,
        retained_tasks=old['tasks'][:4],remaining_tasks=[t for t in formal_tasks if t not in RETAINED],
        retained_episodes=80,new_episodes=620,expected_episodes=700,original_task_exit_zero_proven=True,
        native_mp4_full_decode=True,task_audits=task_audits,evidence=[binding(p) for p in sorted(set(files))])
    from guarded_dualrate_smoke import atomic
    atomic(Path(destination),result)
    return result

def validate(config,source=None,*,verify_evidence=True):
    path=config.get('retained_task_receipt');digest=config.get('retained_task_receipt_sha256')
    if path is None and digest is None:return None
    require(bool(path) and bool(digest) and sha(path)==digest,'retention receipt binding differs')
    r=read(path)
    require(r.get('schema')==SCHEMA and r.get('complete') is True
        and r.get('original_task_exit_zero_proven') is True and r.get('native_mp4_full_decode') is True,
        'incomplete retention receipt')
    check_binding(r['old_config']);old=read(r['old_config']['path']);compare_contracts(old,config)
    require(len(r['retained_tasks'])==4 and set(r['retained_tasks'])==RETAINED and r['retained_tasks']==old['tasks'][:4]
        and r['formal_tasks']==ordered_tasks(old['tasks'],[t for t in old['tasks'] if t not in RETAINED][:2])
        and r['remaining_tasks']==[t for t in r['formal_tasks'] if t not in RETAINED]
        and (r['retained_episodes'],r['new_episodes'],r['expected_episodes'])==(80,620,700),'invalid retention partition')
    if config['stage']=='formal':require(config['tasks']==r['formal_tasks'] and config['episodes']==20,'formal must remain exact35x20')
    elif config['stage']=='smoke':require(config['tasks']==r['remaining_tasks'][:2] and config['episodes']==2,'smoke must remain exact2x2')
    else:raise ValueError('unsupported retention stage')
    require(Path(config['output']).resolve()!=Path(old['output']).resolve(),'cannot overwrite original output')
    if source is not None:require(source==r['new_source'],'retention belongs to a different frozen source')
    check_binding(r['semantic_proof']);validate_proof(r['semantic_proof']['path'],r['old_source'],r['new_source'])
    if verify_evidence:
        require(bool(r['evidence']),'retention evidence absent')
        for item in r['evidence']:check_binding(item)
    return r

def expected_smoke_tasks(formal):
    r=validate(formal,verify_evidence=False)
    return r['remaining_tasks'][:2] if r is not None else formal['tasks'][:2]

def retained_owner(config,task):
    if config.get('stage')!='formal' or not config.get('retained_task_receipt'):return None
    r=validate(config,verify_evidence=False)
    return read(r['old_config']['path']) if task in r['retained_tasks'] else None

def initialize_view(config,source):
    """Called inside the suite, before any new task worker starts."""
    r=validate(config,source)
    if r is None or config['stage']!='formal':return [],[],[]
    from dualrate_eval_suite import audit_task
    old=read(r['old_config']['path']);output=Path(config['output']);reports=[];metrics=[]
    for task in r['retained_tasks']:
        origin=(Path(old['output'])/task).resolve(strict=True)
        (output/task).symlink_to(origin,target_is_directory=True)
        native=Path(read(origin/'completed.json')['native_result_dir']).resolve(strict=True)
        alias=output/'eval_result'/task/'dualrate_dynamicwam'/config['task_config']/Path(config['checkpoint']).stem/config['result_stamp']
        alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(native,target_is_directory=True)
        task_reports,task_metrics=audit_task(old,task);reports.extend(task_reports);metrics.extend(task_metrics)
    return list(r['retained_tasks']),reports,metrics

def provenance(config,source):
    r=validate(config,source)
    if r is None or config['stage']!='formal':return None
    old=read(r['old_config']['path']);origins=[]
    for task in config['tasks']:
        retained=task in r['retained_tasks'];owner=old if retained else config
        for episode in range(20):
            path=Path(owner['output'])/task/f'episode_{episode:04d}/report.json'
            origins.append(dict(task=task,episode_index=episode,retained=retained,report=binding(path),
                config_sha256=canonical(owner),source_sha256=(r['old_source'] if retained else source)['source_sha256']))
    return dict(schema=SCHEMA,retained_episodes=80,newly_evaluated_episodes=620,
        original_failed_suite_preserved=True,scored_task_workers_all_exit_zero=True,
        retained_tasks=r['retained_tasks'],old_source=r['old_source'],new_source=source,
        receipt=binding(config['retained_task_receipt']),semantic_proof=r['semantic_proof'],origins=origins)

def validate_completion(config,source,summary):
    expected=provenance(config,source)
    require(summary.get('retention')==expected,'completed suite retention provenance differs')
    if expected is None:return []
    require(summary['valid_completed']==700 and len(expected['origins'])==700,'retention aggregate not complete700')
    r=validate(config,source)
    for task in r['retained_tasks']:
        owner=read(r['old_config']['path']);alias=Path(config['output'])/task
        require(alias.is_symlink() and alias.resolve()==(Path(owner['output'])/task).resolve(),'retained report alias differs')
        native=Path(read(Path(owner['output'])/task/'completed.json')['native_result_dir']).resolve(strict=True)
        view=Path(config['output'])/'eval_result'/task/'dualrate_dynamicwam'/config['task_config']/Path(config['checkpoint']).stem/config['result_stamp']
        require(view.is_symlink() and view.resolve()==native,'retained native result/MP4 alias differs')
    return [Path(config['retained_task_receipt']),Path(r['semantic_proof']['path']),Path(r['old_config']['path'])]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-config','old-guard','candidate-config','proof','destination'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();prepare(a.old_config,a.old_guard,a.candidate_config,a.proof,a.destination)
