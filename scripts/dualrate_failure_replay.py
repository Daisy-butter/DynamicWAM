"""Bounded unscored replay of the recorded cup1 action endpoints.

No model, capture, wall pacing, score, retry or GPU recovery. The exact failure
tick was not saved: row28 is completed diagnostically, then at most 10 holds.
Run only through the existing all-eight-adapter lifetime guard.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dualrate_eval_worker import (canonical_hash,load_frozen_task,proc_identity,
                                 read,save,sha,source_hash,verify_artifacts)
MODE='dualrate_failure_replay'
PROTOCOL='dualrate_v4_exposure_anchor_wall_release_fast_only'
SCENE_FIELDS=('gravity','bounce_threshold','enable_pcm','enable_tgs',
              'enable_ccd','enable_enhanced_determinism','enable_friction_every_iteration')


@contextmanager
def diagnostic_owner_affinity(cpu):
    receipt=dict(enabled=cpu is not None,requested_cpu=cpu,other_threads_changed=False)
    if cpu is None:
        yield receipt;return
    if (type(cpu) is not int or cpu<0 or threading.current_thread() is not threading.main_thread()):
        raise ValueError('diagnostic owner CPU requires nonnegative integer on main thread')
    before=os.sched_getaffinity(0)
    if cpu not in before:raise ValueError('diagnostic owner CPU outside current allowed affinity')
    receipt.update(native_thread_id=threading.get_native_id(),before=sorted(before))
    try:
        os.sched_setaffinity(0,{cpu});receipt['after']=sorted(os.sched_getaffinity(0))
        if receipt['after']!=[cpu]:raise RuntimeError('owner affinity did not bind requested CPU')
        yield receipt
    finally:
        os.sched_setaffinity(0,before);receipt['restored']=sorted(os.sched_getaffinity(0))
        if receipt['restored']!=sorted(before):raise RuntimeError('owner affinity restoration failed')


def scene_fields(config):
    fields={name:getattr(config,name) for name in SCENE_FIELDS}
    fields['gravity']=[float(x) for x in fields['gravity']]
    fields['bounce_threshold']=float(fields['bounce_threshold'])
    return fields


@contextmanager
def diagnostic_dispatcher(physx,workers):
    """Replay-local override; never edit protected production GPU binding.

    Preserve all seven native scene fields on each call. The produced scene's
    original binding metadata remains workers0; this receipt identifies the
    explicit diagnostic override instead of mislabelling it as the baseline.
    """
    if type(workers) is not int or workers not in (0,2,4):
        raise ValueError('diagnostic dispatcher workers must be 0, 2 or 4')
    original=physx.set_scene_config
    receipt=dict(baseline_physx_cpu_workers=0,diagnostic_physx_cpu_workers=workers,
                 counterfactual=bool(workers),calls=[],restored=False)
    def setter(*args,**kwargs):
        result=original(*args,**kwargs)
        expected=scene_fields(physx.get_scene_config())
        if workers:
            original(**expected,cpu_workers=workers)
        if scene_fields(physx.get_scene_config())!=expected:
            raise ValueError('diagnostic dispatcher changed native scene fields')
        receipt['calls'].append(dict(native_scene_fields=expected,native_scene_fields_verified=True))
        return result
    physx.set_scene_config=setter
    try:
        yield receipt
        if not receipt['calls']:
            raise ValueError('diagnostic dispatcher did not observe native scene creation')
    finally:
        changed=physx.set_scene_config is not setter
        physx.set_scene_config=original
        receipt['restored']=True
        if changed:raise RuntimeError('diagnostic scene setter ownership changed')


def compare_reference_states(path,names,states,times):
    """Require byte identity for actual states and physical timestamps."""
    import numpy as np
    with np.load(path,allow_pickle=False) as reference:
        if set(reference.files)!={'names','states','physics_times'}:
            raise ValueError('reference state file layout differs')
        if (not np.array_equal(reference['names'],np.asarray(names))
                or reference['states'].dtype!=states.dtype or reference['states'].shape!=states.shape
                or reference['physics_times'].dtype!=times.dtype or reference['physics_times'].shape!=times.shape
                or not np.isfinite(reference['states']).all() or not np.isfinite(states).all()):
            raise ValueError('reference actual-state names, shape, dtype or finite values differ')
        equal=(reference['states'].tobytes()==states.tobytes()
               and reference['physics_times'].tobytes()==times.tobytes())
        delta=np.abs(reference['states']-states)
        worst=np.unravel_index(int(delta.argmax()),delta.shape)
        return dict(passed=equal,bitwise_equal=equal,atol=0.,state_count=len(states),
            max_absolute_difference=float(delta[worst]),worst_tick=int(worst[0]),worst_field=names[worst[1]])


def validate_trace(report):
    if (report.get('task')!='place_empty_cup' or report.get('episode_index')!=1
            or report.get('complete') is not False or report.get('protocol')!=PROTOCOL
            or 'wall/physics backlog' not in report.get('error','')):
        raise ValueError('exact failed DualRate cup1 backlog trace required')
    rows=[e for e in report['events'] if e.get('event')=='execute']
    if len(rows)!=29:raise ValueError('exact 29 recorded endpoint rows required')
    dt=float(rows[0]['clock_pair']['physics_step_seconds']);action_dt=45*dt
    if not math.isclose(dt,.004000000189989805,abs_tol=1e-12,rel_tol=0):
        raise ValueError('native physics timestep changed')
    for index,event in enumerate(rows):
        target=event['target']
        if (event['row']!=index or event['physical_tick']!=index*45
                or abs(event['actual_start']-index*action_dt)>1e-8
                or abs(event['start_time']-index*action_dt)>1e-8
                or abs(event['end_time']-(index+1)*action_dt)>1e-8
                or event['controller']!='endpoint_linear_arm_native_gripper'
                or len(target)!=14 or not all(math.isfinite(float(x)) for x in target)):
            raise ValueError('recorded endpoint/grid/controller invalid')
    if rows[0]['target']!=rows[1]['target']:
        raise ValueError('two original startup holds differ')
    return dict(dt=dt,action_dt=action_dt,targets=[e['target'] for e in rows],
                known_completed_ticks=28*45,last_recorded_row_ticks_unknown=True)


def validate_config(config):
    if (config.get('run_kind')!=MODE or config.get('task')!='place_empty_cup'
            or config.get('episode_index')!=1 or config.get('expected_episode_seed')!=100001
            or config.get('seed')!=100000 or config.get('episodes')!=20
            or config.get('physx_cpu_workers')!=0 or config.get('passive_force_dedup') is not True):
        raise ValueError('exact failed cup1/seed100001/native workers0/passive-force contract required')
    workers=config.get('diagnostic_physx_cpu_workers',0)
    if type(workers) is not int or workers not in (0,2,4):
        raise ValueError('diagnostic_physx_cpu_workers must be 0, 2 or 4')
    realtime=config.get('diagnostic_realtime_camera',False)
    cpu=config.get('diagnostic_owner_cpu')
    if cpu is not None and (type(cpu) is not int or cpu<0):raise ValueError('invalid diagnostic_owner_cpu')
    if type(realtime) is not bool or realtime and workers!=0:
        raise ValueError('explicit bool realtime camera diagnosis requires original workers0')
    if realtime and (config.get('camera_async_submit') is not True
            or type(config.get('max_backlog')) not in (int,float)
            or config['max_backlog'] not in (.25,.5,1.) or config.get('backlog_scheduling_tolerance')!=.005
            or not config.get('reference_state_trace')):
        raise ValueError('realtime camera requires async, explicit .25/.5/1.+.005 and workers0 reference states')
    if workers or config.get('reference_state_trace') is not None:
        if (not config.get('reference_state_trace') or
                sha(config['reference_state_trace'])!=config.get('reference_state_trace_sha256')):
            raise ValueError('hash-bound workers0 reference state trace required')
    tail=config.get('tail_hold_rows',0);limit=config.get('timeout_seconds',0)
    if type(tail) is not int or not 0<=tail<=10 or not math.isfinite(limit) or not 0<limit<=1800:
        raise ValueError('bounded tail0..10/timeout<=1800 required')
    uuids=config['gpu_uuids']
    if len(uuids)!=8 or len(set(uuids))!=8:raise ValueError('all-eight guarded UUIDs required')
    storage=Path(config['storage_root']).resolve(strict=True);output=Path(config['output']).resolve()
    if storage.name!='wuruihan' or output==storage or not output.is_relative_to(storage):
        raise ValueError('fresh output must be within the artifact disk wuruihan directory')
    for key in ('trace_report','trace_manifest','failed_config'):
        if sha(config[key])!=config[key+'_sha256']:raise ValueError('source evidence changed: '+key)
    original=read(config['failed_config'])
    if original.get('cpu_affinity_plan') is not None or config.get('cpu_affinity_plan') is not None:
        raise ValueError('this replay does not implement CPU-affinity overrides')
    for key in ('protocol','task_config','seed','prepared','prepared_sha256','action_dt',
                'passive_force_dedup','physx_cpu_workers','renderer_runtime','gpu_uuids'):
        if key not in original or config.get(key)!=original[key]:
            raise ValueError('diagnosis changed original native contract: '+key)
    trace=validate_trace(read(config['trace_report']))
    if abs(config['action_dt']-trace['action_dt'])>1e-10:raise ValueError('action grid differs')
    references=load_frozen_task(config,config['task']);descriptor,motion,_=references[1]
    recorded=read(config['trace_manifest'])
    if descriptor['seed']!=100001 or any(recorded.get(k)!=v for k,v in descriptor.items()):
        raise ValueError('failed manifest differs from exact frozen accepted scene')
    verify_artifacts(config)
    return output,trace,descriptor,motion


def authorize(config,path):
    path=Path(path).resolve(strict=True);r=read(path)
    expected=dict(source_sha256=source_hash(ROOT/'src/dynamicwam'),config_sha256=canonical_hash(config),
                  init_checkpoint_sha256=config['init_checkpoint_sha256'])
    visibility=','.join(config['gpu_uuids'])
    if (r.get('mode')!=MODE or r.get('authorized') is not True or r.get('world_size')!=1
            or any(r.get('identity',{}).get(k)!=v for k,v in expected.items())
            or r.get('source',{}).get('scripts_sha256')!=source_hash(ROOT/'scripts')
            or r.get('gpu_uuids')!=config['gpu_uuids'] or r.get('cuda_visible_devices')!=visibility
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=visibility or r['expires_at_unix']<=time.time()):
        raise ValueError('guard source/config/visibility authorization differs')
    guard=r['external_guard'];owner=proc_identity(guard['pid'])
    if (guard.get('mode')!='fds_before_telemetry_v1' or owner['state'] in ('Z','X')
            or owner['start_ticks']!=guard['start_ticks'] or owner['pid']!=os.getppid()):
        raise ValueError('live exact parent lifetime guard required')
    held={str(fd.resolve()) for fd in (Path('/proc')/str(owner['pid'])/'fd').iterdir()}
    adapters={str(p) for p in Path('/dev').glob('nvidia*') if p.name[6:].isdigit()}
    if len(adapters)!=8 or not adapters<=set(guard['device_paths'])<=held:
        raise ValueError('lifetime guard does not hold all eight adapter descriptors')
    storage=Path(config['storage_root']).resolve(strict=True)
    if not path.is_relative_to(storage) or r['storage_root']!=str(storage):
        raise ValueError('authorization outside declared artifact storage')
    with path.with_suffix('.json.consumed').open('x',encoding='utf-8') as handle:
        json.dump(dict(**proc_identity(os.getpid()),receipt_sha256=sha(path),started_unix=time.time()),handle)
    return r


@contextmanager
def timed(phases,name):
    wall=time.perf_counter();thread=time.thread_time_ns();process=time.process_time_ns()
    try:yield
    finally:phases[name]=dict(wall_seconds=time.perf_counter()-wall,
        thread_cpu_seconds=(time.thread_time_ns()-thread)*1e-9,
        process_cpu_seconds=(time.process_time_ns()-process)*1e-9)


def replay_controls(session,trace,state_reader,*,tail,deadline,before_tick=None,after_tick=None):
    """Same 45-tick native endpoint interpolation as the actual DualRate loop."""
    import numpy as np
    from dualrate_physics_optimization import passive_force_scope
    env=session.env;robot=env.robot;dt=trace['dt'];targets=trace['targets']+[trace['targets'][-1]]*tail
    names,initial=state_reader();states=[initial];times=[session.physics_time()];timings=[]
    def invoke_hook(hook,tick):
        if hook is None:return
        try:hook(tick)
        except BaseException as exc:
            exc.replay_timing_prefix=timings
            raise
    if abs(times[0])>1e-12:raise ValueError('scene clock must start at zero')
    previous=np.asarray(targets[0],np.float64);target=previous.copy();terminal=None
    for tick in range(len(targets)*45):
        if time.monotonic()>=deadline:raise TimeoutError('bounded failed-control replay expired')
        invoke_hook(before_tick,tick)
        row=tick//45;phases={}
        if tick%45==0:
            previous=target.copy();target=np.asarray(targets[row],np.float64)
            env.take_action_cnt=int(getattr(env,'take_action_cnt',0))+1
        with timed(phases,'control'):
            position=previous+(target-previous)*((tick%45+1)/45);velocity=(target-previous)/trace['action_dt']
            position[[6,13]]=target[[6,13]];velocity[[6,13]]=0
            with passive_force_scope(robot,enabled=True):
                robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
                robot.set_arm_joints(position[:6],velocity[:6],'left')
                robot.set_arm_joints(position[7:13],velocity[7:13],'right')
                robot.set_gripper(position[6],'left');robot.set_gripper(position[13],'right')
        with timed(phases,'kinematics'):env._update_kinematic_tasks()
        with timed(phases,'physx'):env.scene.step()
        now=session.physics_time()
        if abs(now-(tick+1)*dt)>1e-8:raise ValueError('native step skipped or rewrote physical time')
        with timed(phases,'transient'):
            env._update_transient_checks();session._consume_events();terminal=session.stop_reason()
        if terminal is not None and tick+1<trace['known_completed_ticks']:
            raise ValueError('terminal before original known completed prefix: '+terminal)
        with timed(phases,'yield'):time.sleep(0)
        with timed(phases,'diagnostic_state_snapshot'):
            labels,values=state_reader()
        if labels!=names:raise ValueError('actual state layout changed')
        states.append(values);times.append(now)
        timings.append(dict(tick=tick+1,row=row,physics_time=now,phases=phases,terminal=terminal,
            region='recorded_complete' if row<28 else 'incomplete_row_completion' if row==28 else 'diagnostic_tail'))
        invoke_hook(after_tick,tick+1)
    return names,np.stack(states),np.asarray(times,np.float64),timings


def snapshot_state(env,dynamic_type):
    # Same actual articulation/rigid-body fields as verified full_physics_replay.
    import numpy as np
    fields={}
    for side in ('left','right'):
        entity=getattr(env.robot,side+'_entity')
        fields[side+'_qpos']=entity.get_qpos();fields[side+'_qvel']=entity.get_qvel()
    for name in ('cup','coaster'):
        actor=getattr(env,name).actor;pose=actor.get_pose()
        body=[c for c in actor.get_components() if isinstance(c,dynamic_type)]
        if len(body)!=1:raise ValueError('expected one native dynamic body for '+name)
        fields[name+'_position']=pose.p;fields[name+'_quaternion']=pose.q
        fields[name+'_linear_velocity']=body[0].linear_velocity;fields[name+'_angular_velocity']=body[0].angular_velocity
    labels=[];values=[]
    for name,value in fields.items():
        vector=np.asarray(value,np.float64).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():raise ValueError('nonfinite actual state')
        labels.extend(f'{name}[{i}]' for i in range(len(vector)));values.extend(vector)
    return labels,np.asarray(values,np.float64)


def run(config,authorization):
    output,trace,descriptor,motion=validate_config(config);receipt=authorize(config,authorization)
    output.mkdir(parents=True,exist_ok=False)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay
    binding=validate_dependency_overlay(config)
    if binding!=receipt['python_dependencies']:raise ValueError('guard dependency binding changed')
    activate_dependency_overlay(binding)
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(dict(config,run_kind='domino_deployment_smoke'),os.environ)
    import numpy as np
    import torch
    import cv2
    import yaml
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    gpu=config['gpu_uuids'][0];pci=[]
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID','').strip()==gpu:pci.append(path.parent.name)
    if len(pci)!=1:raise ValueError('GPU UUID/PCI mapping missing')
    os.environ['DYNAMICWAM_GPU_PCI']=pci[0];torch.cuda.set_device(0)
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    renderer=install_gpu_binding()
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env,DominoLevel1Session
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    from dynamicwam.dualrate.deployment import bind_gil_releasing_physx_step
    from dynamicwam.dualrate.contact_cache import StepContactCache
    from dynamicwam.dualrate.producer import episode_prompt
    import sapien
    domino=Path(config['domino_root']);limits=yaml.safe_load((domino/'task_config/_eval_step_limit.yml').read_text())
    env=cache=camera=realtime=owner_affinity=None;deadline=time.monotonic()+config['timeout_seconds'];report=None
    try:
        with SapienInitLock():
            with diagnostic_dispatcher(sapien.physx,config.get('diagnostic_physx_cpu_workers',0)) as dispatcher:
                env,prepared=prepare_level1_env(domino,task_name=config['task'],task_config=config['task_config'],
                    start_seed=descriptor['seed'],instruction_type=config.get('instruction_type','seen'),
                    replay=dict(seed=descriptor['seed'],instruction=descriptor['instruction'],saved_motion=motion['saved_motion']))
        save(output/'diagnostic_dispatcher.json',dispatcher)
        env._rt_episode_info=motion['episode_info']
        if descriptor['arm_tag']:
            from envs.utils.action import ArmTag
            env.arm_tag=ArmTag(descriptor['arm_tag'])
        actual=dict(seed=int(prepared['seed']),instruction=prepared['instruction'],velocity_m_per_s=prepared['velocity'],
            initial_target_position=np.asarray(env._eval_target_actor.actor.get_pose().p),
            initial_state=env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState(),
            stop_on_contact=prepared['stop_on_contact'],check_z_threshold=prepared['check_z_threshold'],
            arm_tag=str(getattr(env,'arm_tag','')),physics_dt_s=float(env.scene.get_timestep()),
            native_action_limit=int(limits[config['task']]))
        for key in ('seed','instruction','arm_tag','stop_on_contact','check_z_threshold','native_action_limit'):
            if actual[key]!=descriptor[key]:raise ValueError('frozen scene differs: '+key)
        for key in ('initial_target_position','initial_state','velocity_m_per_s','physics_dt_s'):
            if not np.allclose(actual[key],descriptor[key],atol=1e-8,rtol=0):raise ValueError('frozen numeric scene differs: '+key)
        if int(env.step_lim)!=int(limits[config['task']]) or not np.allclose(actual['initial_state'],trace['targets'][0],atol=1e-8,rtol=0):
            raise ValueError('initial held command/native action limit differs')
        prompt,_=episode_prompt(config['task'],motion['episode_info']['info'],domino/'description');env.set_instruction(instruction=prompt)
        session=DominoLevel1Session(env,flow=None,observation_config={},stop_on_contact=prepared['stop_on_contact'],
            check_z_threshold=prepared['check_z_threshold'],lift_actor=prepared['lift_actor'],start_z=prepared['start_z'],
            instruction=prompt,control_interval_s=trace['action_dt'])
        cache=StepContactCache(env.scene,session.physics_time).install()
        bind_gil_releasing_physx_step(env,_sapien_color_reader.step_cpu_physx)
        if config.get('diagnostic_realtime_camera',False):
            from dynamicwam.runtime.domino_rt.camera_capture import AsyncCameraCapture
            from replay_camera_clock import RealtimeCameraReplay
            camera=AsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch,
                async_submit=True,worker_initializer=lambda:torch.cuda.set_device(0))
            realtime=RealtimeCameraReplay(session,camera,action_dt=config['action_dt'],
                frame_period=config['frame_period'],max_backlog=config['max_backlog'],
                backlog_scheduling_tolerance=config['backlog_scheduling_tolerance'],
                warmups=config.get('camera_warmup_captures',3))
        with diagnostic_owner_affinity(config.get('diagnostic_owner_cpu')) as owner_affinity:
            names,states,times,timings=replay_controls(session,trace,
                lambda:snapshot_state(env,sapien.physx.PhysxRigidDynamicComponent),tail=config['tail_hold_rows'],deadline=deadline,
                before_tick=None if realtime is None else realtime.before_tick,
                after_tick=None if realtime is None else realtime.after_tick)
            if realtime is not None:realtime.finish()
        if cache.error is not None:raise cache.error
        np.savez_compressed(output/'state_trace.npz',names=np.asarray(names),states=states,physics_times=times)
        save(output/'step_timings.json',timings)
        parity=None
        if config.get('reference_state_trace'):
            if sha(config['reference_state_trace'])!=config['reference_state_trace_sha256']:
                raise ValueError('reference state trace changed during replay')
            parity=compare_reference_states(config['reference_state_trace'],names,states,times)
            parity.update(reference_path=config['reference_state_trace'],reference_sha256=config['reference_state_trace_sha256'])
            save(output/'state_parity.json',parity)
            if not parity['passed']:
                raise ValueError('diagnostic dispatcher state trajectory differs; candidate rejected')
        report=dict(schema='dualrate_failed_endpoint_replay_v1',complete=True,diagnostic_only=True,no_score=True,
            benchmark_acceptance=False,realtime_valid=False,recorded_state_parity_tested=False,
            task=config['task'],episode_index=1,seed=100001,model_calls=0,
            camera_submissions=0 if realtime is None else realtime.submissions,no_wall_pacing=realtime is None,
            diagnostic_realtime_camera=realtime is not None,
            known_completed_ticks=1260,last_recorded_row_ticks_unknown=True,recorded_endpoint_rows=29,
            incomplete_last_row_completed_diagnostically=True,tail_hold_rows=config['tail_hold_rows'],
            tail_ignores_task_terminal_for_diagnostic=True,physics_steps=len(timings),state_count=len(states),
            renderer=renderer,physx_scene_binding=getattr(env.scene,'_dynamicwam_physx_binding',None),
            physics_configuration_overridden=bool(config.get('diagnostic_physx_cpu_workers',0)),
            diagnostic_dispatcher=dispatcher,workers0_reference_state_parity=parity,diagnostic_owner_affinity=owner_affinity,
            candidate_accepted=False,source=receipt['source'],config_sha256=canonical_hash(config),
            states_sha256=hashlib.sha256(states.tobytes()).hexdigest(),state_trace_sha256=sha(output/'state_trace.npz'),
            step_timings_sha256=sha(output/'step_timings.json'),evidence=[dict(path=config[k],sha256=config[k+'_sha256'])
                for k in ('trace_report','trace_manifest','failed_config')])
    except BaseException as exc:
        if realtime is not None and realtime.end_wall is None:
            realtime.end_wall=time.perf_counter();realtime.end_physical=float(session.physics_time())
        if hasattr(exc,'replay_timing_prefix'):
            save(output/'partial_step_timings.json',exc.replay_timing_prefix)
        save(output/'failure.json',dict(complete=False,error=repr(exc),traceback=traceback.format_exc(),
            realtime_camera=None if realtime is None else realtime.report(),
            diagnostic_owner_affinity=owner_affinity));raise
    finally:
        try:
            try:
                if realtime is not None:realtime.close()
                elif camera is not None:camera.close()
            finally:
                if cache is not None:cache.close()
        finally:
            if env is not None:
                try:env.close_env()
                finally:env.release_episode_resources()
    report['environment_closed']=True
    if realtime is not None:report['realtime_camera']=realtime.report()
    save(output/'report.json',report)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(read(args.config),args.authorization))
