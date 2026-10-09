#!/usr/bin/env python3
"""Autonomous, fail-stop world-context training under continuous physical time.

Launch once from a fresh immutable snapshot. Each subprocess is an argv list;
exit zero alone never constitutes a gate. No seed replacement, checkpoint
selection, automatic retries, or mutation of older experiment directories.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from make_world_context_rt_manifest import (
    TASKS, DEV_TASKS, build_manifest, identity, object_hash, read_json, sha256, write_json)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'dynamicwam_world_context_rt_pipeline'
VERSION = 1
SOURCE_FOLDERS = ('src', 'scripts', 'configs', 'manifests', 'tests')
SOURCE_DOCUMENTS = ('world_context_rt_pipeline.md','world_context_continuous_design_20261008.md','DOMINO_CONTINUOUS.md')
ARMS = ('official_eager', 'graph_zero', 'world_context')
GLOBAL_BATCH = 24
STAGES = {'stage2': dict(steps=4200, samples=100800, lr=2e-5),
          'stage3': dict(steps=1200, samples=28800, lr=5e-6)}


class GateFailed(RuntimeError):
    """A scientific/runtime gate failed; no automatic design adaptation."""


class Incomplete(RuntimeError):
    """Infrastructure or missing evidence; not a policy failure score."""


class IdentityViolation(Incomplete):
    """A source/scene/protocol identity changed; never continue formal evaluation."""


def finite_number(value, *, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def quantile(values, q):
    if not values or not 0 <= q <= 1 or any(not finite_number(v) for v in values):
        raise ValueError('finite nonempty quantile samples required')
    a = sorted(values); p = q*(len(a)-1); i = math.floor(p); j = math.ceil(p)
    return a[i]+(a[j]-a[i])*(p-i)


def distribution(values):
    if not values:
        return {'count': 0}
    return dict(count=len(values), mean=sum(values)/len(values), p50=quantile(values,.5),
                p95=quantile(values,.95), p99=quantile(values,.99), maximum=max(values))


def source_identity(root):
    files = {}
    for folder in SOURCE_FOLDERS:
        for p in sorted((Path(root)/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                if p.is_symlink():
                    raise ValueError(f'source file may not be a symlink: {p}')
                files[p.relative_to(root).as_posix()] = sha256(p)
    for name in SOURCE_DOCUMENTS:
        p=Path(root)/name
        if p.is_file():files[name]=sha256(p)
    if not files:
        raise ValueError('empty source snapshot')
    return dict(sha256=object_hash(files), files=files)


def resource_budget(inventory, requested='auto'):
    """Rank count is fixed at launch, never selected from transient idle cards."""
    by_index = {r['index']: r for r in inventory}
    if len(by_index) != len(inventory) or len({r['uuid'] for r in inventory}) != len(inventory):
        raise ValueError('ambiguous GPU inventory')
    if requested == 'auto':
        candidates = sorted(inventory, key=lambda r: int(r['index']))[:6]
        ranks = max((n for n in range(1, len(candidates)+1) if GLOBAL_BATCH%(2*n) == 0), default=0)
        chosen = candidates[:ranks]
    else:
        keys = requested.split(',')
        if not keys or len(keys) != len(set(keys)) or any(k not in by_index for k in keys):
            raise ValueError('requested GPU index is absent or repeated')
        chosen = [by_index[k] for k in keys]; ranks = len(chosen)
    if not ranks or ranks > 6 or GLOBAL_BATCH % (2*ranks):
        raise ValueError('1, 2, 3, 4, or 6 visible cards required for exact global batch 24')
    return dict(devices=chosen, world_size=ranks, per_gpu_batch=2,
                gradient_accumulation=GLOBAL_BATCH//(2*ranks), global_batch=GLOBAL_BATCH)


def gpu_inventory():
    out = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(('index','uuid'), map(str.strip,x.split(',')))) for x in out.splitlines() if x.strip()]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        requested = visible.split(',')
        rows = [r for r in rows if r['index'] in requested or r['uuid'] in requested]
    return rows


def gpu_conflicts(devices):
    uuids = {r['uuid'] for r in devices}
    # CUDA-only process queries miss graphics-only SAPIEN/Vulkan owners.
    out = subprocess.check_output(['nvidia-smi','-q','-x'],text=True)
    result = []
    for gpu in ET.fromstring(out).findall('gpu'):
        uuid=gpu.findtext('uuid')
        if uuid not in uuids:continue
        for process in gpu.findall('./processes/process_info'):
            result.append(dict(uuid=uuid,pid=process.findtext('pid'),type=process.findtext('type'),
                               process=process.findtext('process_name')))
    return result


def controller_protocol():
    value=read_json(ROOT/'configs/world_context_rt/controller_protocol.json')
    required=dict(format='dynamicwam_continuous_controller_protocol',level=1,
        physics_period_s=.004,control_period_s=.1,query_slots=2,plan_horizon=16,
        slow_refresh_age=12,max_backlog_s=.1,max_tick_gap_s=.1,max_joint_velocity=2.,
        max_joint_acceleration=20.,physics_timeout_s=30.,native_domino_comparability=False,
        camera_pipeline='independent_scene_snapshot_renderer',effective_correction_rows=4,
        simulation_concurrency=1,image_timestamp='snapshot_physical_time',render_quality='rt_32spp_depth8_oidn')
    if any(value.get(k)!=v for k,v in required.items()) or type(value.get('version'))is not int:
        raise IdentityViolation('canonical controller protocol changed a locked pipeline assumption')
    return value


def external_source_identity(root):
    root=Path(root).resolve(strict=True);files={}
    for folder in ('envs','script','task_config','description/utils'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.yml','.yaml','.json'):
                files[p.relative_to(root).as_posix()]=sha256(p)
    if not files:raise Incomplete('external DOMINO source/config fingerprint is empty')
    return dict(root=str(root),sha256=object_hash(files),files=files)


def compare_external_source_identities(expected,actual):
    """Permit only a storage-root relocation, never different source contents.

    Callers obtain ``actual`` with external_source_identity(), which hashes the
    current bytes. The legacy manifest has no explicit format/version fields;
    if either side adds them (or any other metadata), those fields must match
    exactly. Historical manifests are evidence and are never rewritten here.
    """
    for value in (expected,actual):
        if (not isinstance(value,dict) or not isinstance(value.get('root'),str) or not value['root'] or
            not isinstance(value.get('files'),dict) or not value['files']):
            raise IdentityViolation('external source identity schema is incomplete')
        for name,digest in value['files'].items():
            if (not isinstance(name,str) or not name or name.startswith('/') or '\\' in name or
                any(part in ('','..','.') for part in name.split('/')) or
                not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
                raise IdentityViolation('external source file identity is malformed')
        if value.get('sha256')!=object_hash(value['files']):
            raise IdentityViolation('external source aggregate hash does not match its file manifest')
    if {k:v for k,v in expected.items() if k!='root'}!={k:v for k,v in actual.items() if k!='root'}:
        raise IdentityViolation('external source format, file map or content hash changed')
    return dict(format='dynamicwam_external_source_path_comparison',version=1,
        old_root=expected['root'],new_root=actual['root'],relocated=expected['root']!=actual['root'],
        content_sha256=actual['sha256'],file_count=len(actual['files']),
        scope='root path only; current file bytes rehashed; historical manifest unchanged')


def validate_profile(report):
    xs = report.get('fast_samples_ms')
    if (not isinstance(xs,list) or len(xs)<20 or
            any(not finite_number(x,nonnegative=True) or x == 0 for x in xs)):
        raise Incomplete('loaded profile needs at least 20 finite positive raw fast samples')
    p95=quantile(xs,.95)
    if p95>=100 or report.get('passed') is not True:
        raise GateFailed(f'fast p95={p95:.4f} ms fails 100 ms control budget')
    for name in ('graph_samples_ms','eager_samples_ms'):
        values=report.get(name)
        if (not isinstance(values,list) or len(values)<20 or
                any(not finite_number(v,nonnegative=True) or v==0 for v in values)):
            raise Incomplete('profile lacks raw refresh samples')
    graph=sum(report['graph_samples_ms'])/len(report['graph_samples_ms'])
    eager=sum(report['eager_samples_ms'])/len(report['eager_samples_ms'])
    ratio=(graph+5*sum(xs)/len(xs))/eager
    if ratio>=.85 or report.get('native_action_max_abs_error',math.inf)>1e-5:
        raise GateFailed('loaded service-time or native graph parity gate failed')
    return dict(fast_ms=distribution(xs),amortized_ratio=ratio,
                scope='loaded model service time, separate from continuous controller gate')


def validate_probe(report):
    m=report.get('heldout_teacher_mse',{})
    names=('fresh','initialization','no_context','context_shuffle','zero')
    if any(not finite_number(m.get(k),nonnegative=True) for k in names):
        raise Incomplete('probe lacks finite heldout fresh/initialization/no_context/context_shuffle errors')
    if not (m['fresh'] < m['initialization'] and m['fresh'] <= .99*m['no_context']
            and m['fresh'] <= m['context_shuffle'] and m['fresh'] < m['zero']
            and report.get('minimum_context_gain_fraction')==.01 and report.get('passed') is True):
        raise GateFailed('fixed heldout world-context mechanism probe failed')
    return dict(heldout_teacher_mse=m, limitation='development probe; teacher agreement is not task success')


def validate_episode_identity(report,scene,accepted_sha,protocol_sha,*,require_accepted=True):
    if report.get('format')!='domino_continuous_episode' or report.get('version')!=1:
        raise IdentityViolation('unrecognized continuous episode schema')
    if report.get('task')!=scene['task'] or report.get('seed')!=scene['seed']:
        raise IdentityViolation('continuous replay task/seed identity mismatch')
    record=report.get('accepted_record',{})
    if (require_accepted or record) and record.get('sha256')!=accepted_sha:
        raise IdentityViolation('arms did not replay identical accepted record bytes')
    if report.get('controller_protocol_sha256')!=protocol_sha:
        raise IdentityViolation('episode used another controller protocol')
    control=report.get('controller',{})
    expected=dict(control_period_s=.1,physics_dt=.004,max_velocity=2.,max_acceleration=20.,
        query_slots=2,physics_timeout_s=30.,max_backlog_s=.1,max_tick_gap_s=.1,warmup_queries=1,
        camera_pipeline='independent_scene_snapshot_renderer')
    if any(control.get(k)!=v for k,v in expected.items()) or report.get('mode')!='async':
        raise IdentityViolation('actual controller arguments differ from registered continuous protocol')


def validate_episode(report, scene, accepted_sha, protocol_sha):
    validate_episode_identity(report,scene,accepted_sha,protocol_sha)
    if report.get('status')!='complete':
        raise Incomplete(f'continuous episode is {report.get("status")}, never score it as a policy failure')
    r=report.get('report',report)
    if type(r.get('success')) is not bool:
        raise Incomplete('episode success must be a boolean')
    if r.get('realtime_valid') is not True or r.get('virtual_delay') is not False:
        raise Incomplete('episode did not establish actual continuous wall-clock validity')
    camera_ticks=r.get('camera_inflight_ticks')
    if r.get('asynchronous_camera') is not True or type(camera_ticks)is not int or camera_ticks<=0:
        raise Incomplete('episode lacks independent camera/physics overlap evidence')
    ticks=r.get('physics_ticks'); inflight=r.get('inference_inflight_ticks')
    if type(ticks) is not int or ticks<=0 or type(inflight) is not int or inflight<=0:
        raise GateFailed('no actual positive physics advance during inference')
    advances=[q.get('physics_advanced_during_worker_s') for q in r.get('requests',[])]
    if not advances or any(not finite_number(v,nonnegative=True) for v in advances):
        raise Incomplete('missing worker-interval physics advance evidence; camera age is insufficient')
    if sum(advances)<=0:
        raise GateFailed('no actual physics ticks during completed model worker execution')
    for key in ('tick_lag_s','tick_gap_s'):
        values=r.get(key)
        if (not isinstance(values,list) or not values or
                any(not finite_number(v,nonnegative=True) for v in values)):
            raise Incomplete(f'missing raw continuous timing samples: {key}')
        if max(values)>.1+1e-9:
            raise GateFailed(f'{key} exceeds strict 100 ms infrastructure budget')
    return dict(task=scene['task'],seed=scene['seed'],success=r['success'],accepted_sha256=accepted_sha,
        physics_ticks=ticks,inference_inflight_ticks=inflight,
        physics_advanced_during_worker_s=distribution(advances),tick_lag_s=distribution(r['tick_lag_s']),
        tick_gap_s=distribution(r['tick_gap_s']))


def validate_canary(zero, learned):
    if len(zero)!=len(learned) or not zero or [(x['task'],x['seed']) for x in zero]!=[(x['task'],x['seed']) for x in learned]:
        raise Incomplete('canary arms are not paired')
    a=sum(x['success'] for x in zero); b=sum(x['success'] for x in learned)
    if a==b==0:
        raise GateFailed('no informative successful baseline/treatment episode in fixed canary; design revision required')
    if b < a-1:
        raise GateFailed(f'closed-loop regression tripwire: {b}/{len(learned)} vs zero {a}/{len(zero)}')
    return dict(zero_success=a,learned_success=b,episodes=len(zero),maximum_allowed_loss=1,
                no_success_in_either_arm=a==b==0,
                limitation='small development tripwire, not statistical noninferiority')


def validate_replica(report,scene,accepted_sha,images):
    """Recompute the engineering image gate from retained RGB arrays."""
    from check_scene_replica import VIEWS,compare_repetitions
    if (report.get('format')!='dynamicwam_scene_replica_parity' or report.get('version')!=1 or
            report.get('task')!=scene['task'] or report.get('seed')!=scene['seed'] or
            report.get('accepted_record',{}).get('sha256')!=accepted_sha or
            report.get('instruction_type')!='unseen' or report.get('task_config')!='demo_clean_dynamic'):
        raise IdentityViolation('RGB replica gate scene/accepted bytes/instruction protocol mismatch')
    if report.get('status')!='complete' or report.get('passed') is not True:
        raise GateFailed('actual scene replica RGB/FK parity did not pass')
    if report.get('policy_success_evaluation') is not False or report.get('repetitions')!=4:
        raise Incomplete('replica oracle requires four fixed repetitions, separate from policy success')
    scene_identity=report.get('scene_identity',{}); scene_sha=scene_identity.get('sha256')
    if scene_sha!=object_hash({k:v for k,v in scene_identity.items() if k!='sha256'}):
        raise IdentityViolation('replica scene identity digest mismatch')
    phases=report.get('phases',[])
    if [x.get('phase') for x in phases]!=['initial','after_physical_motion']:
        raise Incomplete('replica oracle lacks both initial and physically moved scenes')
    motion=report.get('motion',{})
    for field in ('physics_time_before','physics_time_after','max_actual_robot_qpos_change','target_translation'):
        if not finite_number(motion.get(field),nonnegative=True):raise Incomplete('invalid physical motion evidence')
    if not (motion['physics_time_after']>motion['physics_time_before'] and
            motion['max_actual_robot_qpos_change']>0 and motion['target_translation']>0 and
            motion.get('robot_motion_observed') is True and motion.get('object_motion_observed') is True):
        raise GateFailed('replica oracle did not exercise both robot and target physical motion')
    recomputed=[]
    for phase,stamp in zip(phases,(motion['physics_time_before'],motion['physics_time_after'])):
        if not finite_number(phase.get('source_time_seconds'),nonnegative=True) or abs(phase['source_time_seconds']-stamp)>1e-7:
            raise IdentityViolation('replica phase snapshot time differs from physical motion record')
        for owner in ('source','replica'):
            oracles=phase.get(owner+'_oracles',[])
            if len(oracles)!=4:raise Incomplete('replica oracle repetition count changed')
            for oracle in oracles:
                tolerance=oracle.get('tolerance')
                if (oracle.get('passed') is not True or oracle.get('physics_steps')!=0 or
                        oracle.get('identity_sha256')!=scene_sha or
                        oracle.get('source_sim_time_seconds')!=phase['source_time_seconds'] or
                        not finite_number(tolerance) or not 0<tolerance<=1e-3):
                    raise Incomplete('replica restore lacks identity/time/zero-step FK evidence')
                for field in ('max_translation_error','max_sign_invariant_quaternion_error'):
                    error=oracle.get(field)
                    if not finite_number(error,nonnegative=True) or error>tolerance:
                        raise GateFailed('replica restore FK/camera pose error exceeds its registered tolerance')
        arrays=images[phase['phase']]
        expected={f'{owner}_{view}' for owner in ('source','replica') for view in VIEWS}
        if set(arrays)!=expected or any(x.ndim!=4 or x.shape[0]!=4 for x in arrays.values()):
            raise Incomplete('retained RGB repetitions do not match the replica gate')
        groups=[[{view:arrays[f'{owner}_{view}'][i] for view in VIEWS} for i in range(4)]
                for owner in ('source','replica')]
        value=compare_repetitions(*groups)
        if phase.get('views')!=value['views'] or phase.get('passed')!=value['passed']:
            raise Incomplete('RGB report differs from metrics recomputed from retained images')
        if value['passed'] is not True:raise GateFailed('retained replica RGB exceeds measured source repeatability')
        recomputed.append(dict(phase=phase['phase'],**value))
    return dict(passed=True,task=scene['task'],seed=scene['seed'],accepted_sha256=accepted_sha,
        instruction_type='unseen',phases=recomputed,
        limitation='finite-sample image/FK engineering gate; not a policy result or distribution-equivalence proof')


def validate_feedback_consumption(reports):
    """Require a corrected target to reach the controller before its slot expires."""
    generated=consumed=intervals=0;consumed_slots=0.;norms=[]
    for report in reports:
        r=report['report'];period=r['control_period_s'];end=r['physics_time']
        if report.get('policy_config',{}).get('mode')!='world_context':
            raise IdentityViolation('feedback consumption gate received another policy arm')
        requests={}
        for q in r['requests']:
            key=q['request_id']
            if key in requests:raise Incomplete('duplicate request in feedback consumption evidence')
            requests[key]=q;meta=q.get('metadata',{});count=meta.get('feedback_rows',0)
            if type(count)is not int or not 0<=count<=4:raise Incomplete('invalid generated correction window')
            if not count:continue
            values=meta.get('feedback_row_l2')
            if (meta.get('feedback_row_l2_units')!='normalized_action' or not isinstance(values,list) or
                    len(values)<count or any(not finite_number(v,nonnegative=True) for v in values[:count])):
                raise Incomplete('missing finite normalized-action correction magnitudes')
            generated+=sum(v>0 for v in values[:count])
        for row in r['control_intervals']:
            duration=max(0.,min(row['end_sim_time'],end)-row['start_sim_time'])
            if row['hold'] or duration<=0:continue
            q=requests.get(row.get('request_id'))
            if q is None:raise Incomplete('consumed control interval lacks its generating request')
            index=round(row['start_sim_time']/period)-round(q['action_origin_sim_time']/period)
            if 'returned_row_index' in row and row['returned_row_index']!=index:
                raise Incomplete('control interval correction row index is inconsistent')
            meta=q.get('metadata',{})
            if not 0<=index<meta.get('feedback_rows',0):continue
            if q.get('expired') is not False or q.get('actual_install_sim_time',math.inf)>row['start_sim_time']+1e-7 or index<q['clipped_prefix']:
                raise Incomplete('correction was attributed to an expired or uninstalled action')
            intervals+=1;norm=meta['feedback_row_l2'][index]
            if norm>0:
                consumed+=1;consumed_slots+=duration/period;norms.append(norm)
    if not generated:raise GateFailed('no nonzero feedback target was generated by the fixed learned-policy canary')
    if not consumed:raise GateFailed('all nonzero feedback targets expired or were unconsumed; refusing long training/evaluation')
    return dict(generated_nonzero_rows=generated,consumed_nonzero_intervals=consumed,
        consumed_feedback_intervals=intervals,consumed_nonzero_slot_equivalents=consumed_slots,
        consumed_target_l2_normalized_action=distribution(norms),nonzero_threshold=0.,
        scope='target delivery to controller only; velocity/acceleration clipping can leave actual actuator commands unchanged')


def validate_camera_events(events):
    count=0;roundoff=[]
    for event in events:
        if event.get('event')!='camera_ready':continue
        count+=1;oracle=event.get('snapshot_oracle',{})
        if oracle.get('passed') is not True or oracle.get('physics_steps')!=0:
            raise Incomplete('camera-ready frame lacks its successful zero-physics-step restore oracle')
        source=oracle.get('source_sim_time_seconds');stamp=event.get('observation_sim_time')
        if not finite_number(source,nonnegative=True) or not finite_number(stamp,nonnegative=True):
            raise Incomplete('camera-ready frame lacks finite snapshot physical time')
        error=abs(source-stamp);roundoff.append(error)
        if error>max(1e-7,abs(stamp)*1e-7):
            raise IdentityViolation('camera image was retimestamped away from its source physical snapshot')
    if not count:raise Incomplete('no restored asynchronous camera-ready frame')
    return dict(camera_ready_frames=count,engine_grid_roundoff_s=distribution(roundoff))


class Pipeline:
    def __init__(self, root):
        self.run=Path(root).resolve(strict=True); self.source=self.run/'source'
        self.config=read_json(self.run/'run_config.json'); self.frozen=read_json(self.run/'source_manifest.json')
        self.protocol=read_json(self.run/'controller_protocol.json'); self.protocol_sha=sha256(self.run/'controller_protocol.json')
        self.budget=self.config['resources']; self.devices=self.budget['devices']
        self.sim_devices=[d for d in self.devices if d['index']=='0']
        if self.config.get('simulation_concurrency')!=1 or len(self.sim_devices)!=1:
            raise Incomplete('simulation requires one qualified GPU-0 worker/render pair')
        (self.run/'tmp').mkdir(exist_ok=True)
        self.command_lock=threading.Lock(); self.stop=threading.Event()
        self.status=dict(format=FORMAT,version=VERSION,status='starting',phases={},started_at=time.time())
        self.scenes=read_json(self.run/'scene_manifest.json'); self.accepted={}
        self.preparation_missing={};self.domino_source=None
        self.external=read_json(self.run/'external_source_manifest.json')
        self.external_path_comparison=compare_external_source_identities(self.external,
            external_source_identity(self.external['root']))
        if self.external_path_comparison['relocated']:
            write_json(self.run/'external_source_relocation.json',self.external_path_comparison,exclusive=True)
        self.external_stats={name:(Path(self.external['root'])/name).stat() for name in self.external['files']}
        self.base=identity(self.config['base_checkpoint'])
        self.input_stats={}
        for name,expected in self.config['inputs'].items():
            if identity(self.config[name])!=expected:
                raise Incomplete(f'input changed before pipeline execution: {name}')
            st=Path(self.config[name]).stat(); self.input_stats[name]=(st.st_size,st.st_mtime_ns)
        self.ids=dict(source_sha256=self.frozen['sha256'],config_sha256=sha256(self.run/'run_config.json'),
            controller_protocol_sha256=self.protocol_sha,base_checkpoint_sha256=self.base['sha256'],
            scene_manifest_sha256=sha256(self.run/'scene_manifest.json'))

    def update(self,**values):
        with self.command_lock:
            self.status.update(values); self.status['updated_at']=time.time()
            write_json(self.run/'status.json',self.status)

    def verify_source(self):
        if source_identity(self.source)!=self.frozen:
            raise Incomplete('frozen executable source changed; refusing to continue')
        # Full input hashes are checked once above. Re-hashing multi-GB models
        # during another card's strict-wall episode would itself disturb physics.
        # Any metadata change fails closed; it is not silently re-accepted.
        for name,expected in self.input_stats.items():
            st=Path(self.config[name]).stat()
            if (st.st_size,st.st_mtime_ns)!=expected:
                raise Incomplete(f'immutable input changed: {name}')
        if sha256(self.run/'controller_protocol.json')!=self.ids['controller_protocol_sha256']:
            raise Incomplete('controller protocol changed')
        for name,expected in self.external_stats.items():
            st=(Path(self.external['root'])/name).stat()
            if (st.st_size,st.st_mtime_ns)!=(expected.st_size,expected.st_mtime_ns):
                raise IdentityViolation(f'external DOMINO source/config metadata changed: {name}')
        for name,key in (('run_config.json','config_sha256'),('scene_manifest.json','scene_manifest_sha256')):
            if sha256(self.run/name)!=self.ids[key]:raise IdentityViolation(f'immutable registration changed: {name}')

    def wait_resources(self,devices):
        while True:
            if self.stop.is_set():
                raise Incomplete('pipeline stopped')
            conflicts=gpu_conflicts(devices)
            if not conflicts:
                return
            self.update(status='waiting_resources',busy_devices=conflicts)
            time.sleep(30)

    def command(self,phase_id,argv,devices=(),timeout=None,allow_failure=False):
        if self.stop.is_set():
            raise Incomplete('pipeline stopped before subprocess')
        self.verify_source()
        if devices:
            self.wait_resources(devices)
        directory=self.run/'phases'/phase_id; directory.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy(); env.update(PYTHONPATH=str(self.source/'src')+os.pathsep+str(self.source/'scripts'),
            PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=','.join(x['uuid'] for x in devices),
            DWAM_PHASE_ID=phase_id,DWAM_COMPLETION_PATH=str(directory/'completion.json'),
            TMPDIR=str(self.run/'tmp'),TMP=str(self.run/'tmp'),TEMP=str(self.run/'tmp'),
            CUDA_CACHE_PATH=str(self.run/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(self.run/'cache/inductor'),
            TRITON_CACHE_DIR=str(self.run/'cache/triton'),XDG_CACHE_HOME=str(self.run/'cache/xdg'),
            __GL_SHADER_DISK_CACHE_PATH=str(self.run/'cache/gl_shader'),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for key in ('XDG_CACHE_HOME','__GL_SHADER_DISK_CACHE_PATH'):
            Path(env[key]).mkdir(parents=True,exist_ok=True)
        for key,value in self.ids.items():
            env['DWAM_PIPELINE_'+key.upper()]=value
        row=dict(phase_id=phase_id,argv=[str(x) for x in argv],cwd=str(self.source),
                 gpu_uuids=[x['uuid'] for x in devices],started_at=time.time(),identities=self.ids)
        timeout=timeout or 86400
        row['timeout_seconds']=timeout
        with self.command_lock:
            with (self.run/'commands.jsonl').open('a') as f:
                f.write(json.dumps(row,sort_keys=True)+'\n')
        self.update(status='running',current_phase=phase_id)
        with (directory/'stdout.log').open('x') as log:
            child=subprocess.Popen(row['argv'],cwd=self.source,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
            try:
                code=child.wait(timeout=timeout)
            except BaseException as exc:
                # This process group was created here. Never touch unrelated GPU users.
                if os.name!='nt': os.killpg(child.pid,signal.SIGTERM)
                else: child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name!='nt':os.killpg(child.pid,signal.SIGKILL)
                    else:child.kill()
                    child.wait(timeout=30)
                if allow_failure and isinstance(exc,subprocess.TimeoutExpired):
                    code=124;row['timeout']=True
                else:raise
        row.update(exit_code=code,finished_at=time.time()); write_json(directory/'command.json',row)
        if allow_failure:
            return code
        if code:
            self.stop.set()
            gate_report=self.run/phase_id/'report.json'
            if gate_report.is_file() and read_json(gate_report).get('passed') is False:
                raise GateFailed(f'{phase_id} reported a failed gate; see {gate_report}')
            raise Incomplete(f'{phase_id} exited {code}; see {directory}/stdout.log')
        return 0

    def complete(self,phase_id,artifacts,evidence):
        self.verify_source()
        values=[]
        for role,path in artifacts.items():
            p=Path(path).resolve(strict=True)
            if not p.is_relative_to(self.run):
                raise Incomplete(f'output artifact escaped experiment directory: {p}')
            values.append(dict(role=role,**identity(p)))
        value=dict(format='dynamicwam_world_context_rt_phase_completion',version=1,phase_id=phase_id,
                   status='complete',identities=self.ids,artifacts=values,evidence=evidence,finished_at=time.time())
        path=self.run/'phases'/phase_id/'completion.json'; write_json(path,value,exclusive=True)
        with self.command_lock:
            self.status['phases'][phase_id]=identity(path)
        self.update()
        return value

    def parallel(self,items,fn,*,devices=None):
        # Fixed device queues prevent a busy card from silently changing the run budget.
        devices=self.devices if devices is None else devices
        queues=[items[i::len(devices)] for i in range(len(devices))]
        def consume(pair):
            device,queue=pair; out=[]
            for item in queue:
                if self.stop.is_set(): break
                try: out.append(fn(item,device))
                except BaseException:
                    self.stop.set(); raise
            return out
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            parts=list(pool.map(consume,zip(devices,queues)))
        return [x for part in parts for x in part]

    def select(self,role,development=False):
        return [s for s in self.scenes['episodes'] if s['role']==role and (not development or s['development'])]

    def episode_command(self,scene,output,*,prepare=False,policy=None,mode='async'):
        p=self.protocol
        a=[self.config['sim_python'],str(self.source/'scripts/eval_domino_continuous.py'),
           '--config',str(self.source/self.config['domino_config']), '--task',scene['task'],'--seed',str(scene['seed']),
           '--accepted-record',str(self.run/scene['accepted_record']),'--output',str(output),
           '--control-period',str(p['control_period_s']),'--query-slots',str(p['query_slots']),
           '--physics-timeout',str(p['physics_timeout_s']),'--max-backlog',str(p['max_backlog_s']),
           '--max-tick-gap',str(p['max_tick_gap_s']),'--max-velocity',str(p['max_joint_velocity']),
           '--max-acceleration',str(p['max_joint_acceleration'])]
        if prepare: a+=['--prepare-only']
        else: a+=['--mode',mode,'--policy-config',str(policy),
                  '--policy-factory','dynamicwam.runtime.world_context_worker:create_worker_policy']
        return a

    def verify_episode_source(self,report):
        for name,digest in report.get('source_sha256',{}).items():
            if self.frozen['files'].get(name)!=digest:
                raise IdentityViolation(f'episode used a different frozen source: {name}')
        value=report.get('domino_source_sha256')
        if value:
            with self.command_lock:
                if self.domino_source is None:self.domino_source={}
                for name,digest in value.items():
                    if self.external['files'].get(name)!=digest:
                        raise IdentityViolation(f'external DOMINO source differs from registration: {name}')
                    if name in self.domino_source and self.domino_source[name]!=digest:
                        raise IdentityViolation(f'external DOMINO source changed across scenes: {name}')
                    self.domino_source[name]=digest

    def prepare_scenes(self,scenes,*,allow_missing=False):
        pending=[s for s in scenes if s['scene_id'] not in self.accepted and s['scene_id'] not in self.preparation_missing]
        def one(scene,device):
            phase=f'prepare/{scene["task"]}/{scene["seed"]}'; out=self.run/phase
            code=self.command(phase,self.episode_command(scene,out,prepare=True),[device],timeout=3600,allow_failure=allow_missing)
            report=read_json(out/'summary.json') if (out/'summary.json').exists() else {}
            self.verify_episode_source(report)
            if code or report.get('status')!='prepared':
                if allow_missing:
                    entry=dict(scene=scene,status='preparation_missing',exit_code=code,
                               summary=identity(out/'summary.json') if (out/'summary.json').exists() else None)
                    write_json(self.run/'phases'/phase/'attempt.json',entry,exclusive=True)
                    return scene['scene_id'],entry
                raise Incomplete('fixed scene preparation failed')
            record=self.run/scene['accepted_record']
            if (report.get('status')!='prepared' or report.get('task')!=scene['task'] or
                report.get('seed')!=scene['seed'] or report.get('accepted_record',{}).get('sha256')!=sha256(record)):
                raise IdentityViolation('fixed accepted scene preparation lacks exact identity')
            entry=dict(scene=scene,accepted_record=identity(record),summary=identity(out/'summary.json'))
            self.complete(phase,dict(summary=out/'summary.json',accepted_record=record),entry)
            return scene['scene_id'],entry
        for key,entry in self.parallel(pending,one,devices=self.sim_devices):
            if entry.get('status')=='preparation_missing':self.preparation_missing[key]=entry
            else:self.accepted[key]=entry
        write_json(self.run/'paired_scene_manifest.json',dict(format='dynamicwam_continuous_paired_manifest',
            version=1,scene_manifest_sha256=self.ids['scene_manifest_sha256'],
            controller_protocol_sha256=self.protocol_sha,domino_source_sha256=self.domino_source,
            scenes=self.accepted,preparation_missing=self.preparation_missing))

    def policy(self,arm,scene=None,checkpoint=None,collection=None):
        return dict(project_root=str(self.source),deploy_config=str(self.source/self.config['domino_config']),
            mode=arm,base_checkpoint_sha256=self.base['sha256'],feedback_checkpoint=str(checkpoint) if checkpoint else None,
            controller_protocol_sha256=self.protocol_sha,control_period_s=.1,refresh_age=12,
            collection_dir=str(collection) if collection else None,samples_per_episode=64,
            task=scene['task'] if scene else None,seed=scene['seed'] if scene else None)

    def episodes(self,name,scenes,arm,checkpoint=None,collect=False,formal=False):
        self.prepare_scenes(scenes,allow_missing=formal)
        eligible=[s for s in scenes if s['scene_id'] in self.accepted]
        def one(scene,device):
            phase=f'{name}/{scene["task"]}/{scene["seed"]}'; out=self.run/phase
            raw=out/'raw' if collect else None; cfg=self.run/'policy_configs'/phase/'worker.json'
            write_json(cfg,self.policy(arm,scene,checkpoint,raw),exclusive=True)
            expected=self.accepted[scene['scene_id']]['accepted_record']['sha256']
            if sha256(self.run/scene['accepted_record'])!=expected:
                raise IdentityViolation('accepted scene record changed before replay')
            code=self.command(phase,self.episode_command(scene,out,policy=cfg),[device],timeout=1200,allow_failure=formal)
            report=read_json(out/'summary.json') if (out/'summary.json').exists() else {}
            self.verify_episode_source(report)
            if formal and (code or report.get('status')!='complete'):
                # Verify all available identities before preserving an invalid attempt.
                if report:validate_episode_identity(report,scene,expected,self.protocol_sha,require_accepted=False)
                entry=dict(scene=scene,task=scene['task'],seed=scene['seed'],output=str(out),
                           status=report.get('status','missing'),exit_code=code,success=None)
                write_json(self.run/'phases'/phase/'attempt.json',entry,exclusive=True)
                return entry
            try:
                evidence=validate_episode(report,scene,expected,self.protocol_sha)
                with (out/'events.jsonl').open() as events:
                    evidence['camera_snapshot_timing']=validate_camera_events(json.loads(line) for line in events if line.strip())
            except IdentityViolation:
                raise
            except (Incomplete,GateFailed,OSError,json.JSONDecodeError) as exc:
                if not formal:raise
                entry=dict(scene=scene,task=scene['task'],seed=scene['seed'],output=str(out),
                    status='infrastructure_invalid',exit_code=code,success=None,validation_error=str(exc))
                write_json(self.run/'phases'/phase/'attempt.json',entry,exclusive=True)
                return entry
            evidence.update(arm=arm,checkpoint=identity(checkpoint) if checkpoint else None)
            self.complete(phase,dict(summary=out/'summary.json',events=out/'events.jsonl',policy=cfg),evidence)
            return dict(scene=scene,output=str(out),policy=str(cfg),raw=str(raw) if raw else None,**evidence)
        attempts=sorted(self.parallel(eligible,one,devices=self.sim_devices),key=lambda r:(r['task'],r['seed']))
        result=[r for r in attempts if type(r.get('success')) is bool]
        path=self.run/name/'aggregate.json'; write_json(path,dict(format='dynamicwam_continuous_arm_aggregate',
            version=1,status='complete' if len(result)==len(scenes) else 'incomplete',arm=arm,controller_protocol_sha256=self.protocol_sha,
            checkpoint=identity(checkpoint) if checkpoint else None,episodes=result,
            invalid_attempts=[r for r in attempts if type(r.get('success')) is not bool],
            preparation_missing=[s for s in scenes if s['scene_id'] not in self.accepted],
            requested_episode_count=len(scenes),success_count=sum(x['success'] for x in result),episode_count=len(result)))
        return result

    def profile(self,name,checkpoint):
        out=self.run/name; out.mkdir(parents=True,exist_ok=False); cfg=out/'worker.json'
        write_json(cfg,self.policy('world_context',checkpoint=checkpoint),exclusive=True)
        self.command(name,[self.config['sim_python'],str(self.source/'scripts/profile_world_context.py'),
            '--policy-config',str(cfg),'--packet',self.config['packet'],'--output',str(out/'report.json')],[self.devices[0]],timeout=1800)
        report=read_json(out/'report.json'); evidence=validate_profile(report)
        if (report.get('head_checkpoint_sha256')!=sha256(checkpoint) or
            report.get('base_checkpoint_sha256')!=self.base['sha256'] or
            report.get('controller_protocol_sha256')!=self.protocol_sha or
            report.get('packet_sha256')!=self.config['inputs']['packet']['sha256'] or
            report.get('policy_config_sha256')!=sha256(cfg)):
            raise Incomplete('profile did not load the requested checkpoint/base')
        self.complete(name,dict(report=out/'report.json',policy=cfg),evidence)

    def replica_check(self,scene):
        import numpy as np
        self.prepare_scenes([scene])
        name='development/scene_replica';out=self.run/name;record=self.run/scene['accepted_record']
        expected=self.accepted[scene['scene_id']]['accepted_record']['sha256']
        if sha256(record)!=expected:raise IdentityViolation('accepted record changed before RGB oracle')
        code=self.command(name,[self.config['sim_python'],str(self.source/'scripts/check_scene_replica.py'),
            '--config',str(self.source/self.config['domino_config']),'--task',scene['task'],'--seed',str(scene['seed']),
            '--accepted-record',str(record),'--task-config','demo_clean_dynamic','--instruction-type','unseen',
            '--output',str(out/'report.json'),'--repeats','4','--motion-ticks','25','--joint-delta','.03'],
            self.sim_devices,timeout=1200,allow_failure=True)
        report=read_json(out/'report.json');self.verify_episode_source(report)
        if code and report.get('passed') is True:
            raise Incomplete('RGB oracle claimed success despite a nonzero subprocess exit')
        required={'scripts/check_scene_replica.py','src/dynamicwam/runtime/domino_rt/scene_snapshot.py',
                  'src/dynamicwam/runtime/domino_rt/render_worker.py','src/dynamicwam/runtime/domino_rt/preparation.py'}
        if not required<=set(report.get('source_sha256',{})) or report.get('config',{}).get('sha256')!=sha256(self.source/self.config['domino_config']):
            raise IdentityViolation('RGB oracle source or configuration identity is incomplete')
        artifacts=dict(report=out/'report.json');images={}
        for phase in report.get('phases',[]):
            item=phase.get('repeated_images',{});path=Path(item['path']).resolve(strict=True)
            if not path.is_relative_to(out.resolve()) or sha256(path)!=item.get('sha256'):
                raise IdentityViolation('RGB oracle retained image path/digest mismatch')
            with np.load(path,allow_pickle=False) as arrays:images[phase['phase']]={key:arrays[key] for key in arrays.files}
            artifacts['images_'+phase['phase']]=path
        evidence=validate_replica(report,scene,expected,images)
        self.complete(name,artifacts,evidence)

    def closed_loop_evidence(self,zero,learned):
        evidence=validate_canary(zero,learned)
        evidence['feedback_consumption']=validate_feedback_consumption(
            [read_json(Path(row['output'])/'summary.json') for row in learned])
        return evidence

    def relabel(self,name,rows):
        def one(row,device):
            phase=f'{name}/{row["task"]}/{row["seed"]}'; output=self.run/phase
            self.command(phase,[self.config['sim_python'],str(self.source/'scripts/relabel_world_context.py'),
                '--episode-dir',row['raw'],'--policy-config',row['policy'],
                '--report',str(Path(row['output'])/'summary.json'),'--output-dir',str(output)],[device],timeout=1800)
            report=read_json(output/'report.json'); completion=read_json(output/'completion.json')
            if report.get('status')!='complete' or completion.get('status')!='complete':
                raise Incomplete('relabel output is incomplete')
            index=output/'index.jsonl'; values=[json.loads(x) for x in index.read_text().splitlines() if x.strip()]
            if not values: raise Incomplete('episode has no teacher labels')
            for value in values:
                p=(output/value['path']).resolve()
                if not p.is_relative_to(output.resolve()) or sha256(p)!=value['sha256']:
                    raise Incomplete('teacher label record has an invalid path or digest')
            # Root relabel owns output/completion.json; our independent phase completion lives under phases/.
            self.complete(phase,dict(index=index,report=output/'report.json',relabel_completion=output/'completion.json'),
                dict(task=row['task'],seed=row['seed'],samples=len(values),source_episode=identity(Path(row['output'])/'summary.json')))
            return dict(**row,index=identity(index),relabel_report=identity(output/'report.json'),sample_count=len(values))
        return self.parallel(rows,one)

    def dataset_manifest(self,name,rows,stage2=None,behavior=None):
        # Dataset reads only these explicitly indexed labeled records; no directory discovery.
        import torch
        sys.path.insert(0,str(self.source/'src'))
        from dynamicwam.models.world_context_feedback import WORLD_CONTEXT_SPEC
        reference=torch.load(self.config['init_plan_feedback'],map_location='cpu',weights_only=True,mmap=True)
        path=self.run/name/'dataset.json'
        value=dict(format='dynamicwam_world_context_feedback_data',version=1,
            controller_protocol=identity(self.run/'controller_protocol.json'),base_checkpoint=self.base,
            action_normalization=reference['action_normalization'],world_context_spec=WORLD_CONTEXT_SPEC,
            source_artifacts=[identity(self.source/name) for name in
                ('scripts/relabel_world_context.py','src/dynamicwam/runtime/world_context_worker.py',
                 'scripts/eval_domino_continuous.py','src/dynamicwam/runtime/domino_rt/continuous_loop.py')],
            scene_manifest=identity(self.run/'scene_manifest.json'),official_seed_manifest=self.config['inputs']['official_seed_manifest'],
            episodes=[dict(task=x['task'],seed=x['seed'],partition=x['scene']['partition'],role=x['scene']['role'],
                behavior='world_context' if x['scene']['role']=='stage3' else 'graph_zero',complete=True,
                behavior_checkpoint_sha256=sha256(behavior) if x['scene']['role']=='stage3' and behavior else None,
                index=x['index'],report=x['relabel_report'],source_summary=identity(Path(x['output'])/'summary.json'),
                accepted_record_sha256=x['accepted_sha256']) for x in rows])
        if stage2: value['stage2_manifest']=identity(stage2)
        write_json(path,value,exclusive=True); return path

    def train(self,name,manifest,*,initial=None,smoke=False):
        phase='stage3' if initial else 'stage2'; recipe=STAGES[phase]; steps=4 if smoke else recipe['steps']
        out=self.run/name; b=self.budget
        argv=[self.config['python'],'-m','torch.distributed.run','--standalone',f'--nproc_per_node={b["world_size"]}',
            str(self.source/'scripts/train_world_context_feedback.py'),'--manifest',str(manifest),'--run-dir',str(out),
            '--steps',str(steps),'--lr',str(recipe['lr']),'--phase',phase,
            '--init-plan-feedback',self.config['init_plan_feedback'],'--batch-size',str(b['per_gpu_batch']),
            '--gradient-accumulation',str(b['gradient_accumulation']),'--seed','20261008',
            '--history-mode','temporal','--world-mode','context','--validate-every',str(steps if smoke else 200)]
        if initial: argv+=['--init-feedback',str(initial)]
        self.command(name,argv,self.devices)
        checkpoint=out/'exports'/f'feedback_step_{steps}.pt'
        evidence=self.checkpoint_evidence(checkpoint,steps,manifest,initial)
        self.complete(name,dict(checkpoint=checkpoint,metrics=out/'metrics.jsonl'),evidence)
        return checkpoint

    def checkpoint_evidence(self,path,steps,manifest,initial):
        import torch
        sys.path.insert(0,str(self.source/'src'))
        from dynamicwam.models.world_context_feedback import WorldContextFeedbackHead
        from dynamicwam.models.world_context_feedback_contract import runtime_contract
        from dynamicwam.models.flow_schedule import BridgeSchedule
        torch.set_num_threads(1); payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        for key,wanted in dict(format='dynamicwam_world_context_feedback_checkpoint',version=1,global_step=steps,
            controller_protocol_sha256=self.protocol_sha,data_manifest_sha256=sha256(manifest)).items():
            if payload.get(key)!=wanted: raise Incomplete(f'export contract mismatch: {key}')
        if payload.get('base_checkpoint',{}).get('sha256')!=self.base['sha256']:
            raise Incomplete('export changed official backbone')
        expected={k:self.budget[k] for k in ('world_size','per_gpu_batch','gradient_accumulation','global_batch')}
        expected.update(optimizer_steps=steps,examples_seen=steps*GLOBAL_BATCH)
        budget=payload.get('metadata',{}).get('training_budget',{})
        if any(budget.get(k)!=v for k,v in expected.items()):
            raise Incomplete('actual training sample/rank budget differs from registration')
        expected_init=sha256(initial or self.config['init_plan_feedback'])
        if payload.get('initialization_sha256')!=expected_init:
            raise Incomplete('checkpoint did not use the registered initialization')
        head=WorldContextFeedbackHead(**payload['spec']); head.load_state_dict(payload['model'],strict=True)
        reference=torch.load(self.config['init_plan_feedback'],map_location='cpu',weights_only=True,mmap=True)
        schedule=BridgeSchedule.from_flow_match(**reference['runtime_contract']['schedule'])
        if (payload.get('action_normalization')!=reference['action_normalization'] or
            payload.get('runtime_contract')!=runtime_contract(schedule,head.residual_limit,'temporal','context') or
            payload.get('phase')!=('stage3' if initial else 'stage2') or
            payload.get('config',{}).get('lr')!=STAGES['stage3' if initial else 'stage2']['lr']):
            raise Incomplete('checkpoint normalization/runtime/phase/learning-rate contract differs')
        if any(not torch.is_tensor(x) or not bool(torch.isfinite(x).all()) or
               (x.is_floating_point() and x.dtype!=torch.float32) for x in payload['model'].values()):
            raise Incomplete('nonfinite or non-FP32 exported head')
        behaviors=payload.get('metadata',{}).get('behavior_examples_seen',{})
        if initial and (behaviors.get('graph_zero')!=steps*12 or behaviors.get('learned')!=steps*12 or
                payload.get('metadata',{}).get('stage3_sampling')!='half_zero_half_exact_stage2_behavior_per_microbatch'):
            raise Incomplete('stage3 must use exactly half zero and half learned examples in every global batch')
        if payload.get('teacher_trainable_parameters')!=0 or payload.get('teacher_loaded') is not False:
            raise Incomplete('the offline teacher must remain absent/frozen during feedback training')
        hashes=payload.get('source_hashes',{})
        if not {'scripts/train_world_context_feedback.py',
            'src/dynamicwam/training/data/world_context_feedback_dataset.py',
            'src/dynamicwam/models/world_context_feedback.py',
            'src/dynamicwam/models/world_context_feedback_contract.py'}<=set(hashes):
            raise Incomplete('checkpoint source provenance is incomplete')
        for name,digest in hashes.items():
            if self.frozen['files'].get(name)!=digest:
                raise Incomplete(f'checkpoint source mismatch: {name}')
        metrics=[json.loads(x) for x in (path.parent.parent/'metrics.jsonl').read_text().splitlines() if x.strip()]
        if not metrics or metrics[-1].get('kind')!='complete' or metrics[-1].get('step')!=steps:
            raise Incomplete('trainer has no completed final optimizer step')
        gradients=[x for x in metrics if x.get('kind')=='train']
        if (not gradients or any(not finite_number(x.get('grad_norm')) or x.get('grad_tensors',0)<=0 for x in gradients)
                or not any(x['grad_norm']>0 for x in gradients)):
            raise Incomplete('missing finite training gradient evidence')
        return dict(checkpoint=identity(path),training_budget=budget,initialization_sha256=expected_init,
                    teacher_parameters_trainable=0)

    def run_all(self):
        cpu_names=['scripts/check_world_context_rt_pipeline.py','scripts/check_world_context_feedback.py',
                   'scripts/check_world_context_worker.py','scripts/check_scene_snapshot_cpu.py',
                   'tests/test_continuous_domino.py']
        for script in cpu_names:
            self.command('cpu/'+script,[self.config['python'],str(self.source/script)])
            self.complete('cpu/'+script,dict(log=self.run/'phases'/'cpu'/script/'stdout.log'),dict(exit_code=0))
        self.profile('initial_profile',Path(self.config['init_plan_feedback']))
        # The first real continuous executions gate clock advancement and infrastructure before long collection/training.
        train_dev=self.select('stage2',True); val_dev=self.select('validation',True)
        first=train_dev[0]
        self.replica_check(first)
        clock_smoke=self.episodes('development/clock_smoke',[first],'graph_zero',collect=True)
        remaining_development=[s for s in train_dev+val_dev if s['scene_id']!=first['scene_id']]
        development=clock_smoke+self.episodes('development/zero',remaining_development,'graph_zero',collect=True)
        development=sorted(development,key=lambda r:(r['task'],r['seed']))
        write_json(self.run/'development/zero_combined.json',dict(status='complete',episodes=development),exclusive=True)
        labels=self.relabel('development/labels',development)
        dev_manifest=self.dataset_manifest('development',labels)
        smoke=self.train('smoke',dev_manifest,smoke=True); self.profile('smoke_profile',smoke)
        probe_dir=self.run/'probe'
        self.command('probe',[self.config['python'],str(self.source/'scripts/probe_world_context.py'),
            '--manifest',str(dev_manifest),'--init-checkpoint',self.config['init_plan_feedback'],
            '--output-dir',str(probe_dir),'--steps','200','--seed','20261008'],[self.devices[0]])
        probe_report=read_json(probe_dir/'report.json'); evidence=validate_probe(probe_report); probe=probe_dir/'checkpoint.pt'
        wanted=dict(manifest_sha256=sha256(dev_manifest),init_checkpoint_sha256=sha256(self.config['init_plan_feedback']),
            controller_protocol_sha256=self.protocol_sha,steps=200,seed=20261008,checkpoint_sha256=sha256(probe))
        if any(probe_report.get(k)!=v for k,v in wanted.items()) or probe_report.get('world_output_gradient_nonzero') is not True:
            raise IdentityViolation('probe data/initialization/budget/checkpoint identity differs')
        self.complete('probe',dict(report=probe_dir/'report.json',checkpoint=probe),evidence)
        self.profile('probe_profile',probe)
        zero_canary=[r for r in development if r['scene']['role']=='stage2']
        learned_canary=self.episodes('development/probe',train_dev,'world_context',probe)
        canary=self.closed_loop_evidence(zero_canary,learned_canary)
        self.episodes('development/context_off',train_dev,'context_off',probe)
        self.complete('probe_closed_loop',dict(zero=self.run/'development/zero_combined.json',
            learned=self.run/'development/probe/aggregate.json',context_off=self.run/'development/context_off/aggregate.json'),canary)
        # Do not reuse heldout labels as training, and do not regenerate completed development scenes.
        prior={(r['task'],r['seed']) for r in development}
        remaining=[s for s in self.select('stage2')+self.select('validation') if (s['task'],s['seed']) not in prior]
        raw2=self.episodes('stage2_collection',remaining,'graph_zero',collect=True)
        labels2=labels+self.relabel('stage2_labels',raw2)
        manifest2=self.dataset_manifest('stage2_data',labels2)
        stage2=self.train('stage2',manifest2); self.profile('stage2_profile',stage2)
        learned=self.episodes('development/stage2',train_dev,'world_context',stage2)
        self.complete('stage2_closed_loop',dict(learned=self.run/'development/stage2/aggregate.json'),self.closed_loop_evidence(zero_canary,learned))
        raw3=self.episodes('stage3_collection',self.select('stage3'),'world_context',stage2,collect=True)
        labels3=self.relabel('stage3_labels',raw3)
        manifest3=self.dataset_manifest('stage3_data',labels2+labels3,stage2=manifest2,behavior=stage2)
        stage3=self.train('stage3',manifest3,initial=stage2); self.profile('stage3_profile',stage3)
        learned=self.episodes('development/stage3',train_dev,'world_context',stage3)
        self.complete('stage3_closed_loop',dict(learned=self.run/'development/stage3/aggregate.json'),self.closed_loop_evidence(zero_canary,learned))
        formal=self.select('formal')
        for arm in ARMS:
            self.episodes('eval/'+arm,formal,arm,stage3 if arm=='world_context' else None,formal=True)
        self.command('analysis',[self.config['python'],str(self.source/'scripts/analyze_world_context_rt.py'),
            '--run-root',str(self.run),'--output',str(self.run/'analysis.json')],allow_failure=True)
        report=read_json(self.run/'analysis.json')
        if report.get('status') not in ('complete','incomplete'):
            raise Incomplete('formal analysis lacks an explicit coverage status')
        self.complete('analysis',dict(report=self.run/'analysis.json',markdown=self.run/'analysis.md'),dict(paired_episodes=report['paired_episodes'],
            evaluation_complete=report['status']=='complete'))
        self.update(status='complete' if report['status']=='complete' else 'evaluated_incomplete',
            finished_at=time.time(),analysis=identity(self.run/'analysis.json'))


def freeze(args):
    run=args.run_root.resolve(); project=args.project_root.resolve(strict=True)
    config_relative=Path(args.domino_config)
    if config_relative.is_absolute() or '..' in config_relative.parts:
        raise ValueError('--domino-config must be a repository-relative immutable config')
    run.mkdir(parents=True,exist_ok=False); source=run/'source'; source.mkdir()
    for folder in SOURCE_FOLDERS:
        if (project/folder).exists():
            shutil.copytree(project/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in SOURCE_DOCUMENTS:
        if not (project/name).is_file():raise Incomplete(f'required design/protocol document missing: {name}')
        shutil.copy2(project/name,source/name)
    for name in ('external','data','.venv'):
        if (project/name).exists(): (source/name).symlink_to((project/name).resolve(),target_is_directory=True)
    protocol=run/'controller_protocol.json'; write_json(protocol,controller_protocol(),exclusive=True)
    manifest=build_manifest(episodes_path=args.episodes,raw_root=args.raw_root,official_path=args.official_seed_manifest,
        controller_protocol=protocol,partition_seed=args.partition_seed,validation_fraction=args.validation_fraction)
    write_json(run/'scene_manifest.json',manifest,exclusive=True)
    write_json(run/'external_source_manifest.json',external_source_identity(project/'external/DOMINO'),exclusive=True)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('execute','detach','prepare_only')}
    config['resources']=resource_budget(gpu_inventory(),args.gpus)
    if not any(d['index']=='0' for d in config['resources']['devices']):
        raise ValueError('GPU0 must be included: renderer/worker isolation beyond the default device is not qualified')
    config['inputs']={k:identity(config[k]) for k in ('base_checkpoint','init_plan_feedback','packet','episodes','official_seed_manifest','python','sim_python')}
    config['runtimes']={k:json.loads(subprocess.check_output([config[k],'-c',
        'import sys,platform,json;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,version=platform.python_version())))'],text=True))
        for k in ('python','sim_python')}
    config['sample_budgets']=STAGES; config['seed']=20261008
    config['domino_config']=str(Path(args.domino_config).as_posix())
    write_json(run/'run_config.json',config,exclusive=True)
    write_json(run/'source_manifest.json',source_identity(source),exclusive=True)
    write_json(run/'status.json',dict(format=FORMAT,version=VERSION,status='prepared',resources=config['resources']),exclusive=True)
    return run


def launch_once(run,python,detach):
    marker=Path(run)/'launch.json'
    write_json(marker,dict(created_at=time.time(),detached=detach),exclusive=True)
    argv=[python,str(Path(run)/'source/scripts/run_world_context_rt_pipeline.py'),'--run-root',str(run),'--execute']
    temporary=Path(run)/'tmp';temporary.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary),PYTHONDONTWRITEBYTECODE='1')
    if not detach:
        return subprocess.call(argv,env=env)
    if os.name=='nt':
        raise ValueError('detached production pipeline is Linux-only')
    with (Path(run)/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,cwd=Path(run)/'source',stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write_json(Path(run)/'driver_pid.json',dict(pid=child.pid,argv=argv,launched_at=time.time()),exclusive=True)
    print(json.dumps(dict(run_root=str(run),pid=child.pid))); return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True); p.add_argument('--project-root',type=Path,default=ROOT)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true'); mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--launch-prepared',action='store_true')
    p.add_argument('--detach',action='store_true'); p.add_argument('--gpus',default='auto')
    p.add_argument('--simulation-concurrency',type=int,choices=[1],default=1,
                   help='one qualified renderer/worker pair; training still uses all registered ranks')
    p.add_argument('--python',default=sys.executable); p.add_argument('--sim-python',default=sys.executable)
    p.add_argument('--domino-config',default='configs/absolute_motion_v2.yaml')
    for name in ('episodes','raw-root','official-seed-manifest','base-checkpoint','init-plan-feedback','packet'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--partition-seed',type=int,default=20260928)
    p.add_argument('--validation-fraction',type=float,default=.1)
    a=p.parse_args()
    if a.launch_prepared:
        run=a.run_root.resolve(strict=True)
        if read_json(run/'status.json').get('status')!='prepared':raise RuntimeError('only a never-started prepared run may launch')
        raise SystemExit(launch_once(run,read_json(run/'run_config.json')['python'],a.detach))
    if not a.execute:
        for key in ('episodes','raw_root','official_seed_manifest','base_checkpoint','init_plan_feedback','packet'):
            if getattr(a,key) is None: p.error('--'+key.replace('_','-')+' is required when creating a run')
        run=freeze(a)
        if a.prepare_only:
            print(json.dumps(dict(run_root=str(run),status='prepared'))); return
        raise SystemExit(launch_once(run,a.python,a.detach))
    run=a.run_root.resolve(strict=True)
    if ROOT.resolve()!=run/'source': raise RuntimeError('execute only from RUN/source snapshot')
    if read_json(run/'status.json').get('status')!='prepared':
        raise RuntimeError('automatic resume/retry prohibited; preserve the existing run')
    import fcntl
    lock=(run/'pipeline.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pipeline=None
    try:
        pipeline=Pipeline(run)
        pipeline.run_all()
    except BaseException as exc:
        failure=dict(status='pending_design_revision' if isinstance(exc,GateFailed) else 'incomplete',
            error=dict(type=type(exc).__name__,message=str(exc)),finished_at=time.time())
        if pipeline is None:
            # Input/source validation in the constructor is a terminal failure
            # too; never leave a dead detached process looking merely prepared.
            previous=read_json(run/'status.json')
            write_json(run/'status.json',dict(previous,**failure,updated_at=time.time()))
        else:
            pipeline.stop.set(); pipeline.update(**failure)
        raise


if __name__=='__main__':
    main()
