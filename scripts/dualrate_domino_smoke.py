#!/usr/bin/env python3
"""Guarded finite real DOMINO deployment. No retry or hardware recovery."""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
from dataclasses import asdict,is_dataclass
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from contextlib import ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')
    temporary.replace(path)


def verify_deployment_artifacts(config, file_hash):
    """Bind native binaries and external simulator bytes absent from Python hash."""
    files=config.get('deployment_artifact_sha256',{})
    camera=list((ROOT/'src/dynamicwam/runtime/domino_rt').glob('_sapien_color_reader*.so'))
    domino=Path(config['domino_root']).resolve(strict=True)
    required=[domino/'envs/_base_task.py',domino/'script/eval_metrics.py',
              domino/'task_config'/f"{config['deployment']['task_config']}.yml"]
    if len(camera)!=1 or not files or not all(str(p.resolve()) in files for p in [*camera,*required]):
        raise ValueError('native camera/simulator/task configuration hash manifest required')
    for name,digest in files.items():
        path=Path(name)
        if not path.is_absolute() or file_hash(path)!=digest:
            raise ValueError('deployment artifact changed: '+name)
    return files


def run(stack):
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--authorization',required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    sys.setswitchinterval(float(config.get('python_thread_switch_interval_seconds',.005)))
    if config.get('run_kind')!='domino_deployment_smoke':raise ValueError('explicit deployment mode required')
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    renderer_environment=verify_renderer_environment(config,os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay,probe_dependency_imports
    dependencies=validate_dependency_overlay(config);activate_dependency_overlay(dependencies)
    import torch
    from dynamicwam.dualrate.train import (_validate_storage,_validate_normalization,_gpu_authorization,
        source_identity,canonical_hash,file_sha256)
    output=_validate_storage(config);manifest_path=Path(config['manifest']).resolve(strict=True)
    manifest=json.loads(manifest_path.read_text());_validate_normalization(config,manifest)
    if manifest.get('time_contract')!='exposure_anchor_wall_release_v4':
        raise ValueError('deployment requires the measured hard-time data contract')
    verify_deployment_artifacts(config,file_sha256)
    identity=dict(source_sha256=source_identity(),config_sha256=canonical_hash(config),
        manifest_sha256=file_sha256(manifest_path),init_checkpoint_sha256=config['init_checkpoint_sha256'],
        device_type='cuda',torch_version=str(torch.__version__))
    _gpu_authorization(Path(args.authorization),identity,output,expected_mode='domino_deployment_smoke',
                       expected_dependencies=dependencies)
    def stage(name,**details):
        receipt=dict(stage=name,time_unix=time.time(),monotonic=time.monotonic(),**details)
        save(output/'initialization_stage.json',receipt)
        print(json.dumps(receipt),flush=True)
    stage('authorized',renderer_environment=renderer_environment)
    probe_dependency_imports(config,require_flash_attention=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(1)
    uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(uuids)!=3 or len(set(uuids))!=3:
        raise ValueError('deployment requires three distinct GPUs: renderer0, fast1, slow2')
    fast_device=torch.device('cuda',1);device=torch.device('cuda',2)
    stage('cuda_device_initializing',fast_device=str(fast_device),slow_device=str(device))
    torch.cuda.set_device(device)
    # Verify renderer selection against proc inventory; no independent telemetry.
    pci=None
    for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        fields=dict(line.split(':',1) for line in info.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID','').strip()==uuids[0]:pci=info.parent.name
    if pci is None:raise RuntimeError('renderer UUID/PCI mapping missing')
    os.environ['DYNAMICWAM_GPU_PCI']=pci
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    stage('sapien_binding_initializing',renderer_pci=pci)
    binding=install_gpu_binding()
    stage('sapien_binding_ready',renderer=binding)
    from dynamicwam.dualrate.factory import build_system,place_inference_devices
    stage('model_initializing')
    system=build_system(config,device=device).eval().requires_grad_(False)
    stage('model_initialized')
    checkpoint=Path(config['deployment_checkpoint']).resolve(strict=True)
    if file_sha256(checkpoint)!=config['deployment_checkpoint_sha256']:raise ValueError('deployment weights hash mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    contract=payload.get('contract',{})
    if config.get('continuous_v8'):
        from dynamicwam.dualrate.continuous_contract import validate_initialization_fixture
        validate_initialization_fixture(config,payload)
    from dynamicwam.dualrate.continuous_contract import validate_policy_contract
    validate_policy_contract(config,contract)
    for key in ('architecture','codec_id','init_checkpoint_sha256','action_normalization_sha256','motion_normalization_sha256'):
        if contract.get(key)!=config[key]:raise ValueError('deployment/training contract differs: '+key)
    if contract.get('source_sha256')!=identity['source_sha256']:
        raise ValueError('deployment checkpoint was not trained by current frozen source')
    system.load_state_dict(payload['model'],strict=True)
    placement=place_inference_devices(system,fast_device=fast_device,slow_device=device)
    stage('checkpoint_loaded',inference_device_map=placement)
    # svulkan2 DenoiserOidn::init uses cudaGetDevice, independently of the
    # Vulkan RenderSystem device. Keep the physics/render owner on renderer0;
    # Fast and slow inference workers explicitly bind their distinct devices.
    previous_owner_device=torch.cuda.current_device()
    torch.cuda.set_device(0)
    stage('renderer_owner_bound',previous_cuda_device=previous_owner_device,
          renderer_cuda_device=torch.cuda.current_device(),inference_device_map=placement)
    checkpoint_meta={k:payload.get(k) for k in ('schema','phase','phase_step','global_step','run_kind')};del payload
    if config.get('continuous_v8'):
        from dualrate_slow_process import SlowProcessSystem
        system=SlowProcessSystem(system,config,slow_device=device,receipt_path=output/'slow_process.json')
        stack.callback(system.close)
    import yaml
    from dynamicwam.dualrate.online import OnlineBatchBuilder,load_cached_language
    from dynamicwam.dualrate.producer import episode_prompt,SCENE_PREFIX
    from dynamicwam.dualrate.runtime import RuntimeConfig
    from dynamicwam.dualrate.deployment import run_episode,protocol_for_backlog
    from dualrate_physics_optimization import passive_force_scope
    from dynamicwam.runtime.domino_rt.domino_session import DominoLevel1Session,prepare_level1_env
    from dynamicwam.runtime.domino_rt.camera_capture import AsyncCameraCapture,validate_native_rgb_conversion
    from dynamicwam.runtime.domino_rt import _sapien_color_reader
    save(output/'native_rgb_conversion.json',validate_native_rgb_conversion(_sapien_color_reader))
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from video_artifacts import BufferedHeadVideo,validate_video_artifacts,aggregate_native_metrics,write_domino_results
    domino=Path(config['domino_root']).resolve(strict=True)
    sys.path.insert(0,str(domino/'script'))
    from eval_metrics import EvalMetricsTracker
    limits=yaml.safe_load((domino/'task_config/_eval_step_limit.yml').read_text())
    dep=config['deployment'];runtime_config=RuntimeConfig(**dep['runtime'])
    PROTOCOL=protocol_for_backlog(float(dep.get('max_backlog',.25)))
    if config.get('continuous_v8') and dep.get('camera_async_submit') is not True:
        raise ValueError('continuous V8 smoke requires the production asynchronous camera pipeline')
    task_config=yaml.safe_load((domino/'task_config'/f"{dep['task_config']}.yml").read_text())
    if not task_config.get('use_dynamic') or task_config.get('dynamic_level')!=1:
        raise ValueError('deployment must use real DOMINO Level-1 configuration')
    if dep.get('instruction_mode') not in ('episode_complete_template','native_instruction'):
        raise ValueError('instruction protocol must be explicit')
    results=[]
    save(output/'deployment_identity.json',dict(identity=identity,checkpoint=checkpoint_meta,checkpoint_sha256=config['deployment_checkpoint_sha256'],
        renderer=binding,renderer_uuid=uuids[0],fast_uuid=uuids[1],slow_uuid=uuids[2],
        inference_device_map=placement,gpu_count=len(uuids),protocol=PROTOCOL,time_contract=manifest['time_contract']))
    for task in dep['tasks']:
        task_metrics=[]
        next_seed=int(dep['seed'])
        for episode in range(int(dep.get('episodes',1))):
            env=camera=session=None;video=BufferedHeadVideo();report=None
            episode_dir=output/task/f'episode_{episode:04d}';episode_dir.mkdir(parents=True,exist_ok=False)
            native=output/'eval_result'/task/'dualrate_dynamicwam'/dep['task_config']/checkpoint.stem/dep['result_stamp']
            native.mkdir(parents=True,exist_ok=True)
            try:
                stage('scene_preparing',task=task,episode=episode,requested_seed=next_seed)
                with SapienInitLock():
                    requested_seed=next_seed
                    env,prepared=prepare_level1_env(domino,task_name=task,task_config=dep['task_config'],
                        start_seed=requested_seed,instruction_type=dep.get('instruction_type','seen'))
                stage('scene_ready',task=task,episode=episode,accepted_seed=int(prepared['seed']))
                next_seed=int(prepared['seed'])+1
                info=getattr(env,'_rt_episode_info',None)
                if not info or 'info' not in info:raise ValueError('actual episode language bindings missing')
                if dep['instruction_mode']=='episode_complete_template':
                    prompt,_=episode_prompt(task,info['info'],domino/'description')
                else:
                    prompt=SCENE_PREFIX+prepared['instruction']
                try:
                    language=load_cached_language(manifest_path.parent,prompt,index_path=manifest.get('text_index','text/index.json'))
                except KeyError:
                    # An unseen instruction is encoded exactly, before activation.
                    # Never substitute a different episode's cached context.
                    from dynamicwam.inference.model_loader import _build_t5_encoder,load_deploy_config
                    from prepare_dualrate_text import pack_contexts,native_text_inputs,tokenizer_identity
                    native_config=load_deploy_config(config['native_profile'])
                    _,encoder_path,tokenizer_path=native_text_inputs(native_config,{'query':prompt})
                    ti=Path(manifest.get('text_index','text/index.json'))
                    if not ti.is_absolute():ti=manifest_path.parent/ti
                    text_identity=json.loads(ti.read_text())['identity']
                    if (file_sha256(encoder_path)!=text_identity['source_encoder_sha256'] or
                            tokenizer_identity(tokenizer_path)!=text_identity['tokenizer_sha256']):
                        raise ValueError('online native T5 weights/tokenizer differ from training text bank')
                    encoder=_build_t5_encoder(native_config,str(device))
                    language,token_hash=pack_contexts(encoder,[prompt],device)[0]
                    del encoder
                    text_dir=output/'episode_language'/task/f'episode_{episode:04d}'
                    text_dir.mkdir(parents=True,exist_ok=False)
                    torch.save(language,text_dir/'context.pt')
                    save(text_dir/'identity.json',dict(text=prompt,text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        token_sha256=token_hash,context_sha256=file_sha256(text_dir/'context.pt'),
                        encoder='native_frozen_T5',encoder_sha256=text_identity['source_encoder_sha256'],
                        tokenizer_sha256=text_identity['tokenizer_sha256']))
                env.set_instruction(instruction=prompt)
                builder=OnlineBatchBuilder(manifest,action_mean=manifest['action_statistics']['mean'],
                    action_scale=manifest['action_statistics']['std'],language=language,
                    device=fast_device,slow_device=device)
                if len(builder.entity_roles)!=system.fast.config.future_entities:
                    raise ValueError('online entity roles differ from trained fast head')
                session=DominoLevel1Session(env,flow=None,observation_config={},
                    stop_on_contact=prepared['stop_on_contact'],check_z_threshold=prepared['check_z_threshold'],
                    lift_actor=prepared['lift_actor'],start_z=prepared['start_z'],instruction=prompt,
                    control_interval_s=runtime_config.action_dt)
                session.metrics_tracker=EvalMetricsTracker(env,{'use_dynamic':True});session.metrics_tracker.on_episode_start()
                if _sapien_color_reader.rgb_batch_contract!='float32_rgba_to_u8_rgb_exact_v1':
                    raise ValueError('native batch camera bridge contract differs')
                camera=AsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch,
                    async_submit=bool(dep.get('camera_async_submit',False)),
                    worker_initializer=lambda:torch.cuda.set_device(0))
                save(episode_dir/'manifest.json',dict(task=task,episode_index=episode,requested_seed=requested_seed,seed=prepared['seed'],instruction=prompt,
                    velocity=prepared['velocity'],initial_target_position=session.object_position(),
                    action_dt=runtime_config.action_dt,checkpoint_sha256=config['deployment_checkpoint_sha256']))
                with (episode_dir/'expert_motion.pkl').open('xb') as f:
                    pickle.dump(dict(saved_motion=env._saved_dynamic_motion_info,episode_info=info),f)
                report=run_episode(session,system,builder,camera,runtime_config,
                    action_limit=min(int(limits[task]),int(dep.get('action_limit',limits[task]))),
                    frame_period=float(dep.get('frame_period',runtime_config.action_dt)),max_backlog=float(dep.get('max_backlog',.25)),
                    seed=int(prepared['seed']),video=video,
                    control_scope=lambda:passive_force_scope(env.robot,enabled=bool(dep.get('passive_force_dedup',False))),
                    camera_warmup_captures=int(dep.get('camera_warmup_captures',3)))
                report.update(task=task,episode_index=episode,seed=prepared['seed'],gpu_count=len(uuids),
                              inference_device_map=placement,video_required=True,
                              instruction_mode=dep['instruction_mode'],instruction=prompt)
            except BaseException as exc:
                failed_events=getattr(exc,'dualrate_events',[])
                graph_events=[e for e in failed_events if e['event'] in
                              ('fast_graph_prepared','fast_graph_preparation_failed')]
                graph_status=graph_events[-1]['acceptance'] if graph_events else dict(
                    enabled=runtime_config.fast_cuda_graph,passed=False,sealed=False,shape_count=0)
                report=dict(protocol=PROTOCOL,task=task,episode_index=episode,realtime_valid=False,continuous_valid=False,
                    error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),events=failed_events,
                    fast_graph_acceptance=graph_status)
                raise
            finally:
                if camera is not None:camera.close()
                if report is not None:
                    if session is not None and video.frames:
                        report.update(video.finalize(native/f'episode{episode}.mp4',episode_end_physics_s=session.physics_time()))
                    save(episode_dir/'report.json',report)
                if session is not None and report and report.get('realtime_valid'):
                    if report['stop_reason']=='out_of_bounds':session.metrics_tracker.record_out_of_bounds()
                    metric=session.metrics_tracker.get_episode_metrics(report['success'],None if report['stop_reason'] in ('success','action_budget_exhausted') else report['stop_reason'],seed=report['seed'])
                    metric_dict=asdict(metric) if is_dataclass(metric) else vars(metric)
                    save(episode_dir/'metrics.json',metric_dict);task_metrics.append(metric_dict)
                if env is not None:
                    env.close_env();env.release_episode_resources()
            if not validate_video_artifacts(report):raise RuntimeError('native MP4/sidecar verification failed')
            results.append(report)
            write_domino_results(native,dep['result_stamp'],dep.get('instruction_type','seen'),aggregate_native_metrics(task_metrics))
    verify_deployment_artifacts(config,file_sha256)
    parallel_queries=sum(r['fast_queries_during_slow'] for r in results)
    model_overlap_queries=sum(r['model_overlap_fast_queries'] for r in results)
    consumed_refresh=any(len(r['reference_versions_consumed'])>=2 for r in results)
    graph_parity=bool(results) and all(r.get('fast_graph_acceptance',{}).get('enabled') and
        r['fast_graph_acceptance'].get('passed') and r['fast_graph_acceptance'].get('sealed') and
        r['fast_graph_acceptance'].get('shape_count')==2 for r in results)
    save(output/'deployment_acceptance.json',dict(completed=True,episodes=len(results),real_domino=True,simulation_rate=1.,
        full_model=True,strict_checkpoint_load=True,checkpoint=checkpoint_meta,gpu_count=len(uuids),
        time_contract=manifest['time_contract'],inference_device_map=placement,
        fast_queries_during_slow=parallel_queries,refreshed_reference_consumed=consumed_refresh,
        model_overlap_fast_queries=model_overlap_queries,
        fast_graph_parity_passed=graph_parity,
        hardtime_acceptance_passed=bool(parallel_queries and model_overlap_queries and consumed_refresh and graph_parity),
        reference_versions=[r['reference_versions_consumed'] for r in results],reports=[str(output/r['task']/f"episode_{r['episode_index']:04d}/report.json") for r in results]))
    return 0


def main():
    with ExitStack() as stack:
        return run(stack)


if __name__=='__main__':
    raise SystemExit(main())
