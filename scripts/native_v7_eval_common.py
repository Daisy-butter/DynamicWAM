"""CPU-only contracts for final V7 native synchronous DOMINO evaluation."""
import json
import math
import os
from pathlib import Path
import time

import guarded_dualrate_smoke as base

ROOT = Path(__file__).resolve().parents[1]
MODE = 'native_v7_eval_suite'
PROTOCOL = 'native_domino_sync_topp_compact_v7_physical_history_v1'
TRAIN_COMMIT = '4f3b91e66d61575b675d0d12dd9be1f2caaa46e2'
TRAIN_SHA = '54a6908d646c26cd15b0fabf4a8f118cc5cdfa6c7540b324f5b4077223c5c1d1'
read = lambda path: json.loads(Path(path).read_text())
sha = base.file_sha256
save = base.atomic


def validate_config(c, *, bound=True):
    if c['run_kind'] != MODE or c['protocol'] != PROTOCOL or c['time_contract'] != 'native_event_single_v6':
        raise ValueError('native V7 event protocol required; continuous configuration forbidden')
    if c['stage'] not in ('smoke','formal') or c['episodes'] != (2 if c['stage']=='smoke' else 20):
        raise ValueError('full-budget 8x2 smoke then 35x20 formal required')
    if len(c['tasks']) != (8 if c['stage']=='smoke' else 35) or len(set(c['tasks'])) != len(c['tasks']):
        raise ValueError('task count or uniqueness differs')
    if len(c['gpu_uuids'])!=8 or len(set(c['gpu_uuids']))!=8 or c['parallel_environments']!=8:
        raise ValueError('eight independent single-GPU native environments required')
    if c['native_execution'] != dict(replan_actions=2,refresh_actions=4,max_reference_age=1.44):
        raise ValueError('untrained query/refresh/TTL settings')
    if c.get('action_limit') is not None or c.get('runtime') or any(k in c for k in ('backlog_limit','realtime_rate','arm_smoothing')):
        raise ValueError('no continuous clock, budget override or new output smoother in native evaluation')
    if c['action_interval_seconds']!=.18 or not c.get('native_v7') or not c.get('native_v6'):
        raise ValueError('native training time and architecture contract changed')
    if c['seed']!=100000 or c['task_config']!='demo_clean_dynamic' or c['instruction_type']!='seen':
        raise ValueError('frozen native benchmark differs')
    if c['instruction_mode']!='native_instruction' or c.get('full_driven_integration',False):
        raise ValueError('production executes only autonomous fast actions')
    if bound and (not c.get('checkpoint_sha256') or c['checkpoint_sha256']=='PENDING_TRAINING_COMPLETION'):
        raise ValueError('final checkpoint is not bound')
    storage=Path(c['storage_root']).resolve(strict=True)
    output=Path(c['output']).resolve()
    if storage != Path('/SSD_DISK/users/wuruihan') or not output.is_relative_to(storage):
        raise ValueError('SSD output required')
    return output


def paired_contract(c):
    skip={'stage','tasks','episodes','output','result_stamp','acceptance_receipt','acceptance_receipt_sha256'}
    return base.canonical_hash({k:v for k,v in c.items() if k not in skip})


def verify_artifacts(c):
    for path,digest in c['deployment_artifact_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('external source changed: '+path)
    if sha(Path(c['prepared'])/'frozen_manifest.json')!=c['prepared_sha256']:
        raise ValueError('prepared scenes changed')
    if sha(Path(c['manifest']))!=c['training_manifest_sha256']:
        raise ValueError('training data manifest changed')


def validate_checkpoint(c,p):
    expected=read(c['training_config'])
    from dynamicwam.dualrate.train_distributed import phase_contract
    if (p.get('run_kind')!='formal_distributed_training' or p.get('phase')!='cross_time_joint'
            or p.get('phase_step')!=20000 or p.get('global_step')!=31000 or not p.get('phase_complete')):
        raise ValueError('only completed formal Stage3 checkpoint is allowed')
    identity=p['identity']
    if identity['source_sha256']!=TRAIN_SHA or identity['manifest_sha256']!=c['training_manifest_sha256']:
        raise ValueError('checkpoint training identity mismatch')
    if p['contract']!=phase_contract(expected,identity):raise ValueError('checkpoint full contract differs')
    for key in ('architecture','native_v6','native_v7','time_contract','codec_id',
                'action_interval_seconds','action_normalization_sha256','motion_normalization_sha256','init_checkpoint_sha256'):
        if c[key]!=expected[key] or p['contract'].get(key)!=expected[key]:
            raise ValueError('evaluation differs from trained '+key)
    from dynamicwam.dualrate.native_v7_source_compat import validate_deployment_source
    return validate_deployment_source(identity['source_sha256'])


def authorization(c,path, *, task=None,index=None):
    path=Path(path);a=read(path)
    if a.get('mode')!=MODE or not a.get('authorized'):raise ValueError('native suite authorization missing')
    expected=dict(source_sha256=base.source_identity(ROOT/'src/dynamicwam'),
                  config_sha256=base.canonical_hash(c),init_checkpoint_sha256=c['init_checkpoint_sha256'],
                  manifest_sha256=c['training_manifest_sha256'])
    if any(a['identity'].get(k)!=v for k,v in expected.items()):raise ValueError('authorization identity changed')
    if a['source']['scripts_sha256']!=base.source_identity(ROOT/'scripts') or a['gpu_uuids']!=c['gpu_uuids']:
        raise ValueError('scripts or allocation changed')
    if task is None:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(c['gpu_uuids']) or time.time()>=a['expires_at_unix']:
            raise ValueError('suite visibility or expiry mismatch')
    else:
        if task not in c['tasks'] or index not in range(8) or os.environ.get('CUDA_VISIBLE_DEVICES')!=c['gpu_uuids'][index]:
            raise ValueError('worker visibility/task differs')
        claim=read(path.with_suffix('.json.started'));actual=base.proc_identity(claim['pid'])
        if (claim['receipt_sha256']!=sha(path) or claim['started_unix']>=a['expires_at_unix']
                or actual['start_ticks']!=claim['start_ticks'] or actual['state'] in ('Z','X')
                or os.getppid()!=claim['pid']):raise ValueError('worker is not owned by timely suite')
    owner=a['external_guard'];actual=base.proc_identity(owner['pid'])
    if actual['start_ticks']!=owner['start_ticks'] or actual['state'] in ('Z','X'):
        raise ValueError('guard owner changed')
    held={os.readlink(f) for f in Path(f"/proc/{owner['pid']}/fd").iterdir()}
    if len(owner['device_paths'])!=8 or not set(owner['device_paths'])<=held:
        raise ValueError('eight device guards missing')
    ancestor=os.getppid()
    while ancestor>1 and ancestor!=owner['pid']:ancestor=base.proc_identity(ancestor)['ppid']
    if ancestor!=owner['pid']:raise ValueError('guard is not ancestor')
    return a


def audit_events(report):
    """Independent event/physical-clock join, never infer TOPP duration from .18."""
    count=0;physical=0.;queries=[];sources=[];source_times={}
    for e in report['events']:
        if 'query_physical_time' in e:
            q=e['command_index'];s=e['source_command_index'];age=e['reference_age']
            if q!=count or q%2 or e['query_physical_time']!=physical or e['prefix_rows']!=0:
                raise ValueError('query ledger, zero prefix or physical anchor differs')
            if not 0<=q-s<4 or not 0<=age<=1.44 or age!=physical-e['source_physical_time']:
                raise ValueError('stale or incorrectly aligned A10 reference')
            if s==q:source_times[s]=physical
            if source_times.get(s)!=e['source_physical_time'] or e['prior_source_event']!=s:
                raise ValueError('reference provenance changed')
            if e['future_physical_end_times_known'] or e['previous_plan_aligned']!=(q>0):
                raise ValueError('native event/previous plan contract differs')
            if e['prior_row_offset']!=q-s or e['prior_overlap_rows']!=16-(q-s) or not e['graph_ready']:
                raise ValueError('A10 prior or Graph mismatch')
            if e['future_command_events']!=[q+i for i in (1,2,4,8)]*3:
                raise ValueError('Z command-event alignment differs')
            if e['history_sampling']!='physical_time' or abs(e['history_interval_seconds']-.72)>1e-12:
                raise ValueError('history is not in physical seconds')
            for target,exposure,valid in zip(e['history_target_times'],e['history_exposure_times'],e['history_endpoint_valid']):
                if valid and abs(exposure-target)>.060001:raise ValueError('history exposure tolerance exceeded')
            if e['query_seconds']<0 or e['fast_seconds']<=0:raise ValueError('missing measured latency')
            queries.append(e);sources.append(s)
        elif e.get('kind')=='native_topp_execute':
            if e['command_index']!=count or e['physical_start']!=physical or e['physical_end']<physical:
                raise ValueError('TOPP physical ledger gap')
            if abs(e['actual_duration']-(e['physical_end']-physical))>1e-9:raise ValueError('TOPP duration mismatch')
            physical=e['physical_end'];count+=1
        elif e.get('kind')=='native_history_capture':
            if not math.isfinite(e['exposure']) or e['exposure']<physical:raise ValueError('history exposure precedes action')
            if not -1e-6<=e['exposure']-e['requested_exposure']<=e['physics_step_seconds']+1e-6:
                raise ValueError('history capture missed its physical tick')
        else:raise ValueError('unknown native event')
    if count!=report['action_count'] or physical!=report['physical_end'] or not queries:
        raise ValueError('terminal ledger or query liveness mismatch')
    return dict(queries=len(queries),reference_reuses=sum(e['source_command_index']<e['command_index'] for e in queries),
                reference_refreshes=len(set(sources)),passed=True)


def validate_formal_acceptance(c,source):
    validate_config(c)
    if c['stage']=='smoke':return None
    p=Path(c['acceptance_receipt'])
    if sha(p)!=c['acceptance_receipt_sha256']:raise ValueError('smoke acceptance changed')
    a=read(p)
    if (not a['passed'] or a['source']!=source or a['checkpoint_sha256']!=c['checkpoint_sha256']
            or a['paired_contract']!=paired_contract(c) or a['episodes']!=16
            or not a['native_mp4_full_decode'] or not a['all_workers_exit_zero']):
        raise ValueError('same-final-weight native 8x2 smoke acceptance required')
    if a['reference_reuses']<=0 or a['reference_refreshes']<=16:
        raise ValueError('autonomous slow reference reuse/refresh not demonstrated')
    for e in a['evidence']:
        if sha(Path(e['path']))!=e['sha256']:raise ValueError('smoke evidence changed')
    return a
