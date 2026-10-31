"""Autonomous final-checkpoint native TOPP worker; no Full-driven fallback."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time
import traceback

from native_v7_eval_common import *


def prepare_graph(system,c):
    import torch
    from dynamicwam.dualrate.producer import DominoPhysicalDataset,materialize_native_latents
    from dynamicwam.dualrate.train import _device_batch
    b=materialize_native_latents(_device_batch(torch.utils.data.default_collate(
        [DominoPhysicalDataset(c['manifest'])[0]]),torch.device('cuda:0')),
        system.slow.native.compact_wan.video_model.vae)
    with torch.no_grad():
        world=system.world_readout(system.predict_reference(b))
        world['native_available']=torch.ones_like(b['native_query_index'],dtype=torch.bool)
        g=torch.Generator(device='cuda:0').manual_seed(71)
        eager=system.sample_fast(b,world,generator=g)
        engine=system.enable_fast_graph();g.manual_seed(71)
        actual=system.sample_fast(b,world,generator=g)
        for k in eager:torch.testing.assert_close(actual[k],eager[k],atol=2e-5,rtol=2e-5)
        receipt=engine.seal(expected_shapes=1)
    if not receipt['passed']:raise ValueError('final-weight eager/Graph parity failed')
    return dict(passed=True,graph=receipt,eager_shifted_K10_parity=True)


def language_for(c,m,prompt):
    import torch
    from dynamicwam.dualrate.online import load_cached_language
    try:return load_cached_language(Path(c['manifest']).parent,prompt,index_path=m['text_index'])
    except KeyError:
        from dynamicwam.inference.model_loader import _build_t5_encoder,load_deploy_config
        from prepare_dualrate_text import pack_contexts,native_text_inputs,tokenizer_identity
        profile=load_deploy_config(c['native_profile']);_,ep,tp=native_text_inputs(profile,{'query':prompt})
        ti=read(m['text_index'])['identity']
        if sha(ep)!=ti['source_encoder_sha256'] or tokenizer_identity(tp)!=ti['tokenizer_sha256']:
            raise ValueError('text encoder identity changed')
        encoder=_build_t5_encoder(profile,'cuda:0')
        value,_=pack_contexts(encoder,[prompt],torch.device('cuda:0'))[0]
        del encoder
        return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--authorization',required=True)
    p.add_argument('--task',required=True);p.add_argument('--gpu-index',type=int,required=True);a=p.parse_args()
    c=read(a.config);out=validate_config(c);authorization(c,a.authorization,task=a.task,index=a.gpu_index)
    verify_artifacts(c)
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(dict(c,run_kind='domino_deployment_smoke'),os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay,probe_dependency_imports
    activate_dependency_overlay(validate_dependency_overlay(c));probe_dependency_imports(c,require_flash_attention=True)
    import torch
    import numpy as np
    import yaml
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];pci=None
    for f in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        if uuid in f.read_text():pci=f.parent.name
    if pci is None:raise ValueError('renderer UUID/PCI binding missing')
    os.environ['DYNAMICWAM_GPU_PCI']=pci
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    binding=install_gpu_binding()
    from dynamicwam.dualrate.factory import build_system
    from dynamicwam.dualrate.train import _validate_normalization
    m=read(c['manifest']);_validate_normalization(c,m)
    if sha(Path(c['checkpoint']))!=c['checkpoint_sha256']:raise ValueError('checkpoint changed')
    payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    compatibility=validate_checkpoint(c,payload)
    system=build_system(c,'cuda:0').eval().requires_grad_(False)
    system.load_state_dict(payload['model'],strict=True);del payload
    graph=prepare_graph(system,c)
    (out/a.task).mkdir(parents=True, exist_ok=False)
    save(out/a.task/'model_acceptance.json',dict(graph=graph,source_compatibility=compatibility,renderer=binding,
        checkpoint_sha256=c['checkpoint_sha256'],single_action_decoder=True,full_fallback=False))
    from dynamicwam.dualrate.online import OnlineBatchBuilder
    from dynamicwam.dualrate.native_v7_deployment import NativeDualRatePolicy
    from dynamicwam.dualrate.producer import SCENE_PREFIX
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from dualrate_eval_worker import load_frozen_task
    from video_artifacts import BufferedHeadVideo,validate_video_artifacts,aggregate_native_metrics,write_domino_results
    domino=Path(c['domino_root']);sys.path.insert(0,str(domino/'script'))
    from eval_metrics import EvalMetricsTracker
    settings=yaml.safe_load((domino/'task_config'/f"{c['task_config']}.yml").read_text())
    limits=yaml.safe_load((domino/'task_config/_eval_step_limit.yml').read_text())
    if not settings['use_dynamic'] or settings['dynamic_level']!=1:raise ValueError('native Level1 required')
    references=load_frozen_task(c,a.task);metrics=[];reports=[]
    native=out/'eval_result'/a.task/'dualrate_compact_v7'/c['task_config']/'final_stage3'/c['result_stamp']
    for episode,(descriptor,motion,motion_bytes) in enumerate(references):
        env=None;video=BufferedHeadVideo();d=out/a.task/f'episode_{episode:04d}';d.mkdir(parents=True,exist_ok=False)
        try:
            with SapienInitLock():
                env,prepared=prepare_level1_env(domino,task_name=a.task,task_config=c['task_config'],
                    start_seed=descriptor['seed'],instruction_type=c['instruction_type'],native_protocol=True,
                    replay=dict(seed=descriptor['seed'],instruction=descriptor['instruction'],saved_motion=motion['saved_motion']))
            if prepared['velocity'] is not None:raise ValueError('native object motion was converted to continuous velocity')
            env._rt_episode_info=motion['episode_info']
            if descriptor['arm_tag']:
                from envs.utils.action import ArmTag
                env.arm_tag=ArmTag(descriptor['arm_tag'])
            for key in ('seed','instruction','stop_on_contact','check_z_threshold'):
                if prepared[key]!=descriptor[key]:raise ValueError('frozen scene identity differs: '+key)
            for key,actual in dict(initial_target_position=np.asarray(env._eval_target_actor.actor.get_pose().p),
                initial_state=env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState(),
                physics_dt_s=float(env.scene.get_timestep())).items():
                if not np.allclose(actual,descriptor[key],atol=1e-8,rtol=0):raise ValueError('frozen scene differs: '+key)
            if env.step_lim!=limits[a.task] or env.step_lim!=descriptor['native_action_limit']:
                raise ValueError('native action budget changed')
            prompt=SCENE_PREFIX+prepared['instruction'];language=language_for(c,m,prompt)
            env.set_instruction(instruction=prompt)
            tracker=EvalMetricsTracker(env,settings);tracker.on_episode_start();env._metrics_tracker=tracker
            builder=OnlineBatchBuilder(m,action_mean=m['action_statistics']['mean'],action_scale=m['action_statistics']['std'],language=language,device='cuda:0')
            policy=NativeDualRatePolicy(system,builder,seed=prepared['seed'],**c['native_execution'])
            success=lifted=False;reason='action_budget_exhausted';began=time.monotonic()
            while env.take_action_cnt<env.step_lim:
                obs=env.get_obs();video.append(obs['observation']['head_camera']['rgb'],physics_time_s=policy._time(env),capture_wall_perf_s=time.monotonic())
                policy.step(env,obs)
                if prepared['check_z_threshold'] is not None and not lifted:
                    lifted=float(prepared['lift_actor'].get_pose().p[2])-prepared['start_z']>prepared['check_z_threshold']
                if env.eval_success:
                    success=prepared['check_z_threshold'] is None or lifted
                    reason='success' if success else 'object_not_lifted';break
                if env.check_gripper_contact_dynamic_object() and prepared['stop_on_contact']:env.stop_dynamic_object_motion()
                if not getattr(env,'_dynamic_object_stopped',False) and not env.check_dynamic_object_boundary():
                    env.eval_fail=True;tracker.record_out_of_bounds();reason='out_of_bounds';break
                if env.eval_fail:reason='eval_fail';break
            report=dict(protocol=PROTOCOL,task=a.task,episode_index=episode,seed=prepared['seed'],
                requested_seed=descriptor['requested_seed'],success=success,stop_reason=reason,
                action_count=env.take_action_cnt,action_limit=env.step_lim,physical_end=policy._time(env),
                wall_seconds=time.monotonic()-began,events=policy.records,checkpoint_sha256=c['checkpoint_sha256'],
                config_sha256=base.canonical_hash(c),native_motion_unmodified=True,inference_physics_frozen=True,
                autonomous_fast_only=True,full_driven_integration=False,graph_acceptance=graph)
            report['clock_audit']=audit_events(report)
            report.update(video.finalize(native/f'episode{episode}.mp4',episode_end_physics_s=policy._time(env)))
            if not validate_video_artifacts(report):raise ValueError('native MP4 full decode failed')
            save(d/'report.json',report);save(d/'manifest.json',descriptor)
            metric=asdict(tracker.get_episode_metrics(success,None if reason in ('success','action_budget_exhausted') else reason,seed=prepared['seed']))
            save(d/'metrics.json',metric);metrics.append(metric);reports.append(report)
            write_domino_results(native,c['result_stamp'],c['instruction_type'],aggregate_native_metrics(metrics))
        except BaseException as exc:
            save(d/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if env is not None:env.close_env();env.release_episode_resources()
    verify_artifacts(c)
    save(out/a.task/'completed.json',dict(complete=True,task=a.task,episodes=len(reports),protocol=PROTOCOL,
        config_sha256=base.canonical_hash(c),native_result_dir=str(native)))


if __name__=='__main__':main()
