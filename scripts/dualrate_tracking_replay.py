"""Unscored, bounded contact/actuator diagnosis using immutable recorded controls.

No model, camera, scoring, retiming, or wall pacing. Counterfactual robot collision
filters exist only in fresh diagnostic scenes; production settings never change.
"""
from pathlib import Path
import argparse
import os
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dualrate_eval_worker import read,save,sha,load_frozen_task,verify_artifacts
from dualrate_failure_replay import authorize
MODE='dualrate_tracking_replay'


def validate_config(c):
    if (c.get('run_kind')!=MODE or c.get('diagnostic_only') is not True
            or c.get('no_score') is not True or c.get('physx_cpu_workers')!=0
            or c.get('passive_force_dedup') is not True):
        raise ValueError('explicit diagnostic native-physics contract required')
    if not 1<=c.get('max_commands',0)<=64 or not 1<=c.get('timeout_seconds',0)<=1800:
        raise ValueError('bounded command count and deadline required')
    if len(c['gpu_uuids'])!=8 or len(set(c['gpu_uuids']))!=8:
        raise ValueError('all eight lifetime guards required')
    root=Path(c['storage_root']).resolve(strict=True);out=Path(c['output']).resolve()
    if root.name!='wuruihan' or out==root or not out.is_relative_to(root):
        raise ValueError('diagnostic output outside user artifact disk')
    if not 1<=len(c.get('cases',[]))<=8:raise ValueError('bounded cases required')
    labels=set()
    for case in c['cases']:
        if case['label'] in labels or '/' in case['label']:raise ValueError('unique simple label required')
        labels.add(case['label'])
        if case['controller'] not in ('c2','linear') or type(case['disable_robot_contacts']) is not bool:
            raise ValueError('explicit replay controller/contact mode required')
        if sha(case['report'])!=case['report_sha256']:raise ValueError('recorded controls changed')
        j=read(case['report']);manifest=read(Path(case['report']).parent/'manifest.json')
        desc,_,_=load_frozen_task(c,j['task'])[j['episode_index']]
        if any(manifest.get(k)!=v for k,v in desc.items()) or j['seed']!=desc['seed'] or not j.get('complete'):
            raise ValueError('replay is not the exact frozen completed scene')
        rows=[e for e in j['events'] if e.get('event')=='execute'][:c['max_commands']]
        if len(rows)<2:raise ValueError('recorded prefix too short')
        for i,e in enumerate(rows):
            if e['row']!=i or abs(e['start_time']-i*c['action_dt'])>1e-8:
                raise ValueError('recorded control grid changed')
        expected='fixed_c2_arm_native_gripper' if case['controller']=='c2' else 'endpoint_linear_arm_native_gripper'
        if any(e.get('controller',j.get('controller'))!=expected for e in rows):
            raise ValueError('controller does not match original recorded controls')
    verify_artifacts(c)
    return out


def contact_rows(scene,robot_links):
    import numpy as np
    result=[]
    for ct in scene.get_contacts():
        bodies=ct.bodies
        if not any(b.entity in robot_links for b in bodies):continue
        impulse=sum(float(np.linalg.norm(p.impulse)) for p in ct.points)
        result.append(dict(actors=[b.entity.name for b in bodies],impulse=impulse,
            points=[dict(impulse=np.asarray(p.impulse).tolist(),position=np.asarray(p.position).tolist(),
                         normal=np.asarray(p.normal).tolist(),separation=float(p.separation)) for p in ct.points]))
    return result


def run_case(c,case,out,deadline):
    import numpy as np
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env,DominoLevel1Session
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    from dynamicwam.dualrate.deployment import bind_gil_releasing_physx_step
    from dynamicwam.dualrate.contact_cache import StepContactCache
    from dynamicwam.dualrate.continuous_owner import ContinuousStepOwner
    from dynamicwam.dualrate.tracking_audit import TrackingAudit
    from dynamicwam.dualrate.c2_executor import evaluate
    from dualrate_physics_optimization import passive_force_scope
    j=read(case['report']);task=j['task'];desc,motion,_=load_frozen_task(c,task)[j['episode_index']]
    rows=[e for e in j['events'] if e.get('event')=='execute'][:c['max_commands']]
    env=cache=None;disabled=[]
    try:
        with SapienInitLock():
            env,prepared=prepare_level1_env(Path(c['domino_root']),task_name=task,task_config=c['task_config'],
                start_seed=desc['seed'],instruction_type=c.get('instruction_type','seen'),
                replay=dict(seed=desc['seed'],instruction=desc['instruction'],saved_motion=motion['saved_motion']))
        env._rt_episode_info=motion['episode_info']
        if desc['arm_tag']:
            from envs.utils.action import ArmTag
            env.arm_tag=ArmTag(desc['arm_tag'])
        robot=env.robot;dt=float(env.scene.get_timestep());adt=45*dt
        actual=robot.get_left_arm_jointState()+robot.get_right_arm_jointState()
        if (prepared['seed']!=desc['seed'] or abs(adt-c['action_dt'])>1e-10
                or not np.allclose(actual,desc['initial_state'],atol=1e-8,rtol=0)
                or not np.allclose(actual,rows[0]['target'],atol=1e-8,rtol=0)
                or not np.allclose(env._eval_target_actor.actor.get_pose().p,desc['initial_target_position'],atol=1e-8,rtol=0)
                or abs(prepared['velocity']-desc['velocity_m_per_s'])>1e-8):
            raise ValueError('replay initial state/motion/grid mismatch')
        session=DominoLevel1Session(env,flow=None,observation_config={},stop_on_contact=prepared['stop_on_contact'],
            check_z_threshold=prepared['check_z_threshold'],lift_actor=prepared['lift_actor'],start_z=prepared['start_z'],
            instruction=desc['instruction'],control_interval_s=adt)
        cache=StepContactCache(env.scene,session.physics_time).install()
        bind_gil_releasing_physx_step(env,_sapien_color_reader.step_cpu_physx)
        links=set()
        for side in ('left','right'):
            for link in getattr(robot,side+'_entity').get_links():
                links.add(link.entity)
                if case['disable_robot_contacts']:
                    for shape in link.get_collision_shapes():
                        groups=list(shape.get_collision_groups())
                        disabled.append(dict(link=link.entity.name,old=groups))
                        shape.set_collision_groups([0,0,0,0])
        if case['disable_robot_contacts'] and not disabled:raise ValueError('no robot shapes disabled')
        audit=TrackingAudit(robot);contacts=[];previous=target=np.asarray(rows[0]['target'],np.float64)
        terminal=[];v0=np.zeros(14)
        with ContinuousStepOwner(env.scene,session.physics_time) as owner:
            for tick in range(len(rows)*45):
                if time.monotonic()>=deadline:raise TimeoutError('bounded tracking diagnosis expired')
                if tick%45==0:
                    v0=(target-previous)/adt;previous=target.copy();target=np.asarray(rows[tick//45]['target'],np.float64)
                    env.take_action_cnt=int(getattr(env,'take_action_cnt',0))+1
                frac=(tick%45+1)/45
                if case['controller']=='c2':q,v,_=evaluate(previous,target,v0,adt,frac*adt)
                else:q=previous+(target-previous)*frac;v=(target-previous)/adt
                q[[6,13]]=target[[6,13]];v[[6,13]]=0
                with passive_force_scope(robot,enabled=True):
                    robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
                    robot.set_arm_joints(q[:6],v[:6],'left');robot.set_arm_joints(q[7:13],v[7:13],'right')
                    robot.set_gripper(q[6],'left');robot.set_gripper(q[13],'right')
                env._update_kinematic_tasks();owner.advance();now=session.physics_time()
                if abs(now-(tick+1)*dt)>1e-8:raise ValueError('real physics tick mismatch')
                cc=contact_rows(env.scene,links)
                if cc:contacts.append(dict(physical_time=now,contacts=cc))
                if case['disable_robot_contacts'] and cc:raise ValueError('counterfactual still has robot contacts')
                if (tick+1)%5==0:audit.sample(now,q,v)
                env._update_transient_checks();session._consume_events()
                reason=session.stop_reason()
                if reason and (not terminal or terminal[-1]['reason']!=reason):terminal.append(dict(time=now,reason=reason))
        r=dict(complete=True,diagnostic_only=True,no_score=True,task=task,seed=desc['seed'],case=case,
            command_count=len(rows),physics_dt=dt,actuator_tracking=audit.receipt(),contacts=contacts,
            disabled_shapes=disabled,terminal_events=terminal,wall_pacing=False,model_calls=0,
            controller_limits_do_not_guarantee_tracking=True,
            gains={side:{key:getattr(robot,side+'_'+key) for key in ('joint_stiffness','joint_damping')}
                   for side in ('left','right')})
        if case['controller']=='c2' and not case['disable_robot_contacts']:
            old=j['actuator_tracking']['rows'][:len(audit.rows)]
            delta=max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
                      for a,b in zip(old,audit.rows) for key in ('actual_q','actual_v','commanded_q','commanded_v'))
            r['recorded_tracking_parity']=dict(max_absolute_difference=delta,passed=delta<1e-6,samples=len(old))
            if len(old)!=len(audit.rows) or delta>=1e-6:raise ValueError('recorded production tracking was not reproduced: '+str(delta))
        save(out/(case['label']+'.json'),r)
        return dict(label=case['label'],max_error=r['actuator_tracking']['max_joint_position_error'],
                    rms_error=r['actuator_tracking']['rms_joint_position_error'],contact_steps=len(contacts))
    finally:
        if cache is not None:cache.close()
        if env is not None:
            try:env.close_env()
            finally:env.release_episode_resources()


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--authorization',required=True);a=a.parse_args()
    c=read(a.config);out=validate_config(c);receipt=authorize(c,a.authorization,mode=MODE)
    out.mkdir(parents=True,exist_ok=False)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay
    binding=validate_dependency_overlay(c)
    if binding!=receipt['python_dependencies']:raise ValueError('dependency binding changed')
    activate_dependency_overlay(binding)
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(dict(c,run_kind='domino_deployment_smoke'),os.environ)
    import torch,cv2
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    pci=[]
    for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID','').strip()==c['gpu_uuids'][0]:pci.append(path.parent.name)
    if len(pci)!=1:raise ValueError('GPU UUID/PCI mapping missing')
    os.environ['DYNAMICWAM_GPU_PCI']=pci[0];torch.cuda.set_device(0)
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    install_gpu_binding();deadline=time.monotonic()+c['timeout_seconds']
    try:
        results=[]
        for case in c['cases']:
            result=run_case(c,case,out,deadline);results.append(result);print(result,flush=True)
        save(out/'report.json',dict(complete=True,no_score=True,diagnostic_only=True,results=results))
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
