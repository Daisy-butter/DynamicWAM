"""Exact four-view sync/async submission parity; bounded, unscored diagnosis."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from full_baseline_worker import canonical_hash, proc_identity, read, save, sha

SCHEMA='full_camera_overlap_parity_v1'
MODE='camera_overlap_v1'
CAMERAS=('left_camera','right_camera','head_camera','front_camera')
CAPTURE_TICKS=(0,2250,2475)
OVERLAP_STEPS=45


def validate_camera_config(config,trace):
    if (config.get('diagnosis_mode')!=MODE or config.get('camera_capture_ticks')!=list(CAPTURE_TICKS)
            or config.get('camera_overlap_steps')!=OVERLAP_STEPS
            or config.get('camera_warmup_captures')!=3 or config.get('tail_hold_rows',0)!=0
            or trace['steps']!=2836 or CAPTURE_TICKS[-1]+OVERLAP_STEPS>=trace['steps']):
        raise ValueError('camera parity requires exact cup12 trace, fixed three exposures, 45 overlap ticks and three warmups')


def typed_metadata(value):
    """Canonical lossless representation, including float bits and array dtype."""
    import numpy as np
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject: raise ValueError('object metadata arrays are unsupported')
        return dict(kind='array',dtype=value.dtype.str,shape=list(value.shape),bytes=value.tobytes(order='C').hex())
    if isinstance(value,np.generic):
        return dict(kind='numpy_scalar',dtype=value.dtype.str,bytes=value.tobytes().hex())
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value): raise ValueError('metadata keys must be strings')
        return {key:typed_metadata(value[key]) for key in sorted(value)}
    if isinstance(value,(list,tuple)):
        return dict(kind=type(value).__name__,items=[typed_metadata(v) for v in value])
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('nonfinite observation metadata')
        return dict(kind='float',hex=value.hex())
    if value is None or type(value) in (str,int,bool): return value
    raise TypeError('unsupported frozen metadata: '+type(value).__name__)


def capture_metadata(captured):
    observation=dict(captured.observation)
    observation['observation']={name:{key:value for key,value in observation['observation'][name].items() if key!='rgb'}
                                for name in captured.camera_names}
    return dict(schema='exact_camera_observation_metadata_v1',camera_names=list(captured.camera_names),
                exposure_hex=float(captured.physics_time_s).hex(),observation=typed_metadata(observation))


def source_evidence(sapien,bridge,config):
    """Read actual loaded libraries, not an inferred dependency installation."""
    loaded={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
            if line.split() and line.split()[-1].startswith('/')}
    result=[]
    for role,name in (('libsapien','libsapien.so'),('libsvulkan2','libsvulkan2.so')):
        matches={p for p in loaded if p.name==name}
        if len(matches)!=1: raise ValueError('unique loaded '+name+' required')
        path=matches.pop();result.append(dict(role=role,path=str(path),sha256=sha(path)))
    shader=Path(sapien.__file__).resolve().parent/'vulkan_shader/rt/camera.rgen'
    native=Path(bridge.__file__).resolve(strict=True)
    if config['deployment_artifact_sha256'].get(str(native))!=sha(native):
        raise ValueError('actual native camera bridge is not frozen in deployment artifacts')
    for role,path in (('rt_camera_shader',shader),('native_bridge',native)):
        path=path.resolve(strict=True);result.append(dict(role=role,path=str(path),sha256=sha(path)))
    return result


class CameraExperiment:
    """Only begin() updates renderer; physics may run until matching finish()."""
    def __init__(self,session,camera,case,*,async_submit,deadline):
        self.session=session;self.camera=camera;self.case=Path(case);self.async_submit=async_submit
        self.deadline=deadline;self.pending=None;self.active_tick=None;self.step_ends=[]
        self.captures=[];self.warmups=[]

    def finish(self,pending):
        return self.camera.finish(pending,timeout=max(.001,min(30.,self.deadline-time.monotonic())))

    def warmup(self):
        source=self.session.physics_time()
        if source!=0.: raise ValueError('camera warmup must precede physical replay')
        for index in range(3):
            captured=self.finish(self.camera.begin(self.session.env,source))
            if self.session.physics_time()!=source or captured.physics_time_s!=source:
                raise ValueError('camera warmup advanced the physical clock')
            if tuple(captured.camera_names)!=CAMERAS: raise ValueError('four original camera names differ')
            self.warmups.append(dict(index=index,exposure=source,
                metadata_sha256=canonical_hash(capture_metadata(captured)),
                rgb_sha256={name:hashlib.sha256(captured.observation['observation'][name]['rgb'].tobytes()).hexdigest()
                            for name in CAMERAS}))

    def before_tick(self,tick):
        if tick not in CAPTURE_TICKS: return
        if self.pending is not None: raise ValueError('renderer update while another capture is pending')
        self.active_tick=tick;self.step_ends=[]
        self.pending=self.camera.begin(self.session.env,self.session.physics_time())
        if not self.async_submit:
            captured=self.finish(self.pending)
            self._save(captured,tick)
            self.pending=None

    def after_tick(self,tick):
        if self.active_tick is None: return
        self.step_ends.append((tick,time.perf_counter()))
        if tick!=self.active_tick+OVERLAP_STEPS: return
        if self.async_submit:
            if self.pending is None: raise ValueError('lost asynchronous capture')
            captured=self.finish(self.pending)
            self._save(captured,tick)
            self.pending=None
        else:
            self.captures[-1]['completion_tick']=tick
        self.active_tick=None;self.step_ends=[]

    def _save(self,captured,completion_tick):
        import numpy as np
        if tuple(captured.camera_names)!=CAMERAS: raise ValueError('four original camera names differ')
        index=len(self.captures);rgb=self.case/f'capture_{index:04d}.npz';metadata=self.case/f'capture_{index:04d}.json'
        images={name:captured.observation['observation'][name]['rgb'] for name in CAMERAS}
        if any(a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3 for a in images.values()):
            raise ValueError('original uint8 RGB arrays required')
        np.savez_compressed(rgb,**images);save(metadata,capture_metadata(captured))
        submitted=float(captured.render_submitted_wall_perf_s);finished=float(captured.readback_finished_wall_perf_s)
        submission_ticks=[tick for tick,end in self.step_ends if end<submitted]
        native_ticks=[tick for tick,end in self.step_ends if end<finished]
        self.captures.append(dict(tick=self.active_tick,exposure=captured.physics_time_s,completion_tick=completion_tick,
            physics_ticks_before_submission_finish=len(submission_ticks),physics_ticks_before_native_finish=len(native_ticks),
            overlapped_submission_ticks=submission_ticks,overlapped_native_ticks=native_ticks,
            actual_step_wall_ends=[dict(tick=tick,wall=end) for tick,end in self.step_ends],
            capture_started_wall=captured.capture_wall_perf_s,render_submitted_wall=submitted,native_finished_wall=finished,
            metadata_path=str(metadata),metadata_sha256=sha(metadata),rgb_path=str(rgb),rgb_sha256=sha(rgb)))

    def report(self):
        if self.pending is not None or self.active_tick is not None or [c['tick'] for c in self.captures]!=list(CAPTURE_TICKS):
            raise ValueError('camera experiment did not consume all fixed exposures')
        return dict(schema=SCHEMA,camera_case='async' if self.async_submit else 'sync',async_submit=self.async_submit,
                    camera_names=list(CAMERAS),capture_ticks=list(CAPTURE_TICKS),overlap_steps=OVERLAP_STEPS,
                    warmup_captures=3,warmups=self.warmups,captures=self.captures)


def compare_cases(sync,asynchronous):
    import numpy as np
    from full_physics_replay import compare_state_files
    for case,expected in ((sync,False),(asynchronous,True)):
        if (case.get('complete') is not True or case.get('environment_closed') is not True
                or case.get('camera_closed') is not True or case.get('async_submit') is not expected
                or case.get('camera_names')!=list(CAMERAS) or case.get('capture_ticks')!=list(CAPTURE_TICKS)
                or case.get('warmup_captures')!=3 or case.get('overlap_steps')!=OVERLAP_STEPS
                or case.get('physics_steps')!=2836 or len(case.get('captures',[]))!=3):
            raise ValueError('camera child contract is incomplete')
        with np.load(case['state_trace_path'],allow_pickle=False) as states:
            if states['states'].shape[0]!=2837 or states['physics_times'].shape!=(2837,):
                raise ValueError('camera state trace must cover every recorded real tick')
            for capture in case['captures']:
                if (float(capture['exposure']).hex()!=float(states['physics_times'][capture['tick']]).hex()
                        or capture['completion_tick']!=capture['tick']+OVERLAP_STEPS
                        or read(capture['metadata_path'])['exposure_hex']!=float(capture['exposure']).hex()):
                    raise ValueError('camera exposure/completion differs from the actual physics grid')
    rgb_equal=metadata_equal=exposure_equal=True;details=[]
    for first,second in zip(sync['captures'],asynchronous['captures']):
        metadata_equal &= read(first['metadata_path'])==read(second['metadata_path'])
        exposure_equal &= float(first['exposure']).hex()==float(second['exposure']).hex()
        if first['tick']!=second['tick'] or first['completion_tick']!=second['completion_tick']:
            raise ValueError('camera captures used different physical intervals')
        with np.load(first['rgb_path'],allow_pickle=False) as a,np.load(second['rgb_path'],allow_pickle=False) as b:
            if set(a.files)!=set(CAMERAS) or set(b.files)!=set(CAMERAS): raise ValueError('four RGB views required')
            matches={name:a[name].dtype==b[name].dtype==np.uint8 and a[name].shape==b[name].shape
                          and a[name].tobytes()==b[name].tobytes() for name in CAMERAS}
            rgb_equal &= all(matches.values());details.append(dict(tick=first['tick'],views=matches))
    parity=compare_state_files(sync['state_trace_path'],asynchronous['state_trace_path'],atol=0.)
    overlaps=any(c['physics_ticks_before_submission_finish']>0 for c in asynchronous['captures'])
    warmups_equal=sync['warmups']==asynchronous['warmups']
    return dict(passed=bool(rgb_equal and metadata_equal and exposure_equal and parity['bitwise_equal'] and overlaps and warmups_equal),
        rgb_bitwise_equal=bool(rgb_equal),metadata_bitwise_equal=bool(metadata_equal),exposure_times_equal=bool(exposure_equal),
        state_bitwise_equal=parity['bitwise_equal'],state_comparison=parity,warmup_bitwise_equal=warmups_equal,
        physics_advanced_while_pending=overlaps,physics_advanced_while_submitting=overlaps,view_comparisons=details)


def main(config_path,authorization):
    import full_physics_replay as replay
    import guarded_dualrate_run as guard
    config_path=Path(config_path).resolve(strict=True);config=read(config_path)
    output,trace=replay.validate_config(config);receipt,claim=replay.validate_authorization(config,authorization)
    output.mkdir(parents=True,exist_ok=True)
    with claim.open('x',encoding='utf-8') as handle:
        json.dump(dict(**proc_identity(os.getpid()),receipt_sha256=sha(authorization),started_unix=time.time()),handle)
    evidence=[];children=[];cases=[];deadline=time.monotonic()+float(config['timeout_seconds'])
    frozen=Path(config['prepared'])/config['task']/f"episode_{config['episode_index']:04d}"
    for path in (Path(config['trace_report']),Path(config['trace_report']).with_name('manifest.json'),
                 Path(config['prepared'])/'frozen_manifest.json',frozen/'manifest.json',frozen/'expert_motion.pkl'):
        evidence.append(dict(path=str(path),sha256=sha(path)))
    common=dict(schema=SCHEMA,diagnostic_only=True,benchmark_acceptance=False,no_score=True,realtime_valid=False,
        source_contract=guard.source_contract(receipt['source']),source_sha256=receipt['identity']['source_sha256'],
        scripts_sha256=receipt['source']['scripts_sha256'],config_sha256=canonical_hash(config),
        task=config['task'],episode_index=12,seed=100014,physics_steps=2836,state_count=2837,
        trace_report_sha256=config['trace_report_sha256'],trace_manifest_sha256=config['trace_manifest_sha256'],
        prepared_sha256=config['prepared_sha256'],manifest_sha256=sha(frozen/'manifest.json'),motion_sha256=sha(frozen/'expert_motion.pkl'),
        benchmark_contract=config['benchmark_contract'],gpu_uuids=config['gpu_uuids'],
        passive_force_dedup=config.get('passive_force_dedup',False),cpu_workers=[0],
        camera_names=list(CAMERAS),capture_ticks=list(CAPTURE_TICKS),overlap_steps=OVERLAP_STEPS,warmup_captures=3)
    try:
        for name in ('sync','async'):
            save(output/'status.json',dict(state='running',camera_case=name,diagnostic_only=True))
            with (output/f'camera_{name}.log').open('xb') as log:
                process=subprocess.Popen([sys.executable,str(replay.ROOT/'scripts/full_physics_replay.py'),
                    '--config',str(config_path),'--authorization',str(Path(authorization).resolve()),'--camera-case',name],
                    stdout=log,stderr=subprocess.STDOUT)
                try: code=process.wait(timeout=max(.01,deadline-time.monotonic()))
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=30.)
                    except subprocess.TimeoutExpired: pass
                    raise
            if code: raise RuntimeError('camera '+name+' child failed; no retry')
            path=output/f'camera_{name}'/'worker_report.json';case=read(path);cases.append(case)
            children.append(dict(case=name,report_path=str(path),sha256=sha(path)))
            for path in (path,Path(case['state_trace_path']),*(Path(c[k]) for c in case['captures'] for k in ('rgb_path','metadata_path'))):
                evidence.append(dict(path=str(path),sha256=sha(path)))
        if cases[0]['camera_source_evidence']!=cases[1]['camera_source_evidence']:
            raise ValueError('loaded camera library/shader/bridge identities changed between children')
        evidence.extend(dict(path=r['path'],sha256=r['sha256']) for r in cases[0]['camera_source_evidence'])
        comparison=compare_cases(*cases)
        save(output/'camera_overlap.json',dict(common,**comparison,complete=True,environment_closed=True,
            all_children_exit_zero=True,children=children,evidence=evidence,camera_source_evidence=cases[0]['camera_source_evidence']))
        if not comparison['passed']: raise ValueError('exact camera overlap parity failed; no tolerance relaxation')
        save(output/'status.json',dict(state='completed',diagnostic_only=True,no_score=True))
        return 0
    except BaseException as exc:
        if not (output/'camera_overlap.json').exists():
            save(output/'camera_overlap.json',dict(common,passed=False,complete=False,children=children,evidence=evidence,error=repr(exc)))
        save(output/'status.json',dict(state='failed',diagnostic_only=True,no_score=True,error=repr(exc),traceback=traceback.format_exc()))
        raise
