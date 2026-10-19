#!/usr/bin/env python3
"""CPU-only exact checkpoint audit/backbone export for an isolated evaluation.

This verifies weights and timing metadata, not CUDA inference or GPU health.
A copied receipt must be rebound to verified destination paths before use.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from check_motionkv_smoke_export import build_full_cpu_schema,finite_state,file_identity,require
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.inference.multirate_model_loader import (MULTIRATE_CHECKPOINT_FORMAT,
    MULTIRATE_CHECKPOINT_VERSION,FAST_ONLY_FORMAT,extract_multirate_backbone,
    fast_only_matches,load_fast_only_modules,rtc_prefix_contract)
from dynamicwam.inference.fast_runtime import validate_runtime_contract,RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.inference.model_loader import _load_checkpoint_payload,_validate_model_config
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.absolute_motion import ABSOLUTE_MOTION_CHECKPOINT_VERSION


def save_json(path,value):
    require(not path.exists(),f'refusing to overwrite {path}')
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only environment required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    require(not (output/'checkpoint_audit.json').exists(),'audit output already exists')
    full=torch.load(args.full,map_location='cpu',weights_only=True,mmap=True)
    fast=torch.load(args.fast,map_location='cpu',weights_only=True,mmap=True)
    require(full.get('format')==MULTIRATE_CHECKPOINT_FORMAT and fast.get('format')==FAST_ONLY_FORMAT,'wrong export types')
    require(full.get('version')==fast.get('version')==MULTIRATE_CHECKPOINT_VERSION,'wrong export versions')
    require(full.get('global_step')==fast.get('global_step')==args.step,'unexpected checkpoint step')
    require(full['modules']==fast['modules'],'full/fast specs differ')
    spec=full['modules'];contract=full['config']['runtime_contract'];cfg=full['config']
    require(spec['runtime_contract']==contract,'full/module runtime contracts differ')
    require(cfg['phase']=='cross_time_joint','expected Stage3 joint checkpoint')
    require(rtc_prefix_contract(full)==rtc_prefix_contract(fast),'RTC full/fast mismatch')
    require(contract['rtc_prefix_sampling']['mode']=='fixed_h4_v1','fixed H4 required')
    require(spec['fast_steps']==10 and spec['schedule']['video_steps']==2,'expected source 2 joint and fast K10')
    require(contract['source_plan_completion']=='same_noise_native_K10_tail_stopgrad_v1','missing source 2+8 plan')
    require(contract['native_multirate']['source_bridge']=='student_joint','wrong source backbone')
    require(contract['max_age_actions']==31 and contract['max_age_seconds']==4.96,'wrong source cache bounds')
    require(spec['action_expert']['source_physical_time_conditioning'] is True,'missing source time encoder')
    require(contract['training_scope']=='full','subset checkpoint is forbidden')
    require(fast_only_matches(full['model'],fast),'full/fast tensors differ')
    result=dict(schema_version=1,passed=False,device='cpu',origin_host=socket.gethostname(),step=args.step,
        scope='weight schema, tensor equality, normalization and physical-time metadata; no GPU/closed-loop acceptance',
        strict_full_reload=False,full_fast_equal=True,contract=contract,
        full=file_identity(args.full),fast=file_identity(args.fast),
        full_weights=finite_state(full['model'],'full'),fast_weights=finite_state(fast['model'],'fast'))
    profile=load_multirate_profile(args.profile)
    require(profile.training['phase']=='cross_time_joint','profile is not Stage3')
    require(profile.training['model']['action_expert']==cfg['model']['action_expert'],'profile Action differs')
    model=build_full_cpu_schema(full,profile)
    loaded=model.load_state_dict(full['model'],strict=True,assign=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys,'full strict reload failed')
    require(all(p.device.type=='cpu' for p in model.parameters()),'non-CPU model parameters')
    require(all(torch.equal(v,full['model'][k]) for k,v in model.state_dict().items()),'full reload changed values')
    result['strict_full_reload']=True
    result['full_strict_reload']=dict(passed=True,state_keys=len(full['model']),schema='real full model, meta Video construction, strict CPU assign')
    backbone=extract_multirate_backbone(full)
    require(set(backbone)==set(model.backbone.state_dict()),'backbone schema differs')
    source_keys=[k for k in backbone if k.startswith('action_expert.source_time_encoder.')]
    require(len(source_keys)==4,'source time encoder lost weights')
    result['source_time_tensor_keys']=source_keys
    del model;gc.collect()
    modules=load_fast_only_modules(fast)
    for prefix,module in [('backbone.action_expert.',modules.action_expert),('backbone.absolute_motion_tokens.',modules.absolute_motion_tokens),('fast_conditioner.',modules.fast_conditioner)]:
        require(all(torch.equal(v,fast['model'][prefix+k].to(v.dtype)) for k,v in module.state_dict().items()),'fast reload changed values')
    for name,buffer in [('mean','motion_mean'),('scale','motion_scale')]:
        actual=getattr(modules.fast_conditioner,buffer)
        saved=fast['model']['fast_conditioner.'+buffer]
        expected=torch.tensor(spec['motion'][name],dtype=saved.dtype).to(actual.dtype)
        require(torch.equal(actual,expected),'motion metadata differs from serialized dtype values')
        require(torch.equal(actual,getattr(modules.absolute_motion_tokens,'feature_'+name)),'native/interface motion statistics disagree')
    result['strict_fast_reload']=True;result['motion_statistics_match_native']=True
    del modules;gc.collect()
    normal=cfg['action_normalization']
    require(normal==spec['action_normalization'],'full/module action normalizers disagree')
    normalizer=RoboTwinQposNormalizer.from_stats(normal['stats'],epsilon=float(normal['epsilon']))
    runtime=SimpleNamespace(action_interval_seconds=.16,chunk_size=16,num_inference_steps=10,
        num_video_inference_steps=2,flow_match_shift=5.,flow_match_sigma_min=0.,flow_match_extra_one_step=True,
        action_normalizer=normalizer)
    validate_runtime_contract(full,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
    result['runtime_contract_validated']=True
    result['action_stats_sha256']=normal['stats_sha256']
    result['motion_statistics_sha256']=cfg['model']['absolute_motion']['statistics_sha256']
    # A real v2 backbone from this exact Stage3 export, including source_time.
    # No official/Stage2 weights are used as a temporary deployment substitute.
    training_model=profile.training['model']
    result['schema_profile']=file_identity(args.profile)
    base_config=dict(model=dict(initial_checkpoint=args.full.name,
        compact_wan=copy.deepcopy(training_model['compact_wan']),
        action_expert=copy.deepcopy(spec['action_expert']),
        absolute_motion=copy.deepcopy(cfg['model']['absolute_motion'])),
        action_normalization=copy.deepcopy(normal),
        evaluation_provenance=dict(parent_full_sha256=result['full']['sha256'],parent_step=args.step,
            role='exact Stage3 backbone, includes learned source physical time encoder'))
    base_config['model']['compact_wan']['precision']=training_model['compact_wan']['precision']
    bootstrap_action={k:v for k,v in spec['action_expert'].items() if k!='source_physical_time_conditioning'}
    runtime_model=dict(model=dict(compact_wan=base_config['model']['compact_wan'],action_expert=bootstrap_action,
        absolute_motion=base_config['model']['absolute_motion']))
    _validate_model_config(runtime_model,base_config)
    base_path=output/f'backbone_step_{args.step}.pt'
    require(not base_path.exists(),'refusing to overwrite backbone')
    temp=base_path.with_name(base_path.name+'.tmp.'+str(os.getpid()))
    torch.save(dict(format='dynamicwam_absolute_motion_checkpoint',version=ABSOLUTE_MOTION_CHECKPOINT_VERSION,
        global_step=args.step,model=backbone,config=base_config),temp)
    temp.replace(base_path)
    restored=_load_checkpoint_payload(base_path)
    require(set(restored['model'])==set(backbone),'backbone export dropped keys')
    require(all(torch.equal(v,restored['model'][k]) for k,v in backbone.items()),'backbone export changed values')
    result['backbone']=file_identity(base_path);result['backbone_exact_parent_subset']=True
    result['backbone_source_time_preserved']=True
    grid_path=args.physical_grid/'manifest.json';latent_path=args.physical_grid/'future_latents/manifest.json'
    grid=json.loads(grid_path.read_text())
    require(grid['physical_grid_identity']==contract['physical_grid'] and grid['full_scope'] is True,'wrong physical grid')
    require(grid['episode_partition_identity']==contract['episode_partition_identity'],'wrong train/validation partition')
    require(file_identity(latent_path)['sha256']==contract['physical_grid_contract']['latent_manifest_sha256'],'wrong future latent manifest')
    require(grid['contract']['action_dt']==.16 and grid['contract']['action_chunk']==16,'wrong physical grid timing')
    result['data_identity']=dict(physical_grid=contract['physical_grid'],latent_manifest_sha256=contract['physical_grid_contract']['latent_manifest_sha256'],
        episode_partition_identity=contract['episode_partition_identity'],dataset_identity=cfg.get('dataset_identity'),
        physical_grid_manifest=file_identity(grid_path),latent_manifest=file_identity(latent_path))
    require(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
    result['passed']=True;result['cuda_initialized']=False
    result['portable_artifacts']={key:dict(filename=value['path'].split('/')[-1],bytes=value['bytes'],sha256=value['sha256']) for key,value in result.items() if key in ('full','fast','backbone')}
    result['destination_rebind_required']='Verify hashes at destination, then replace artifact paths in a new receipt; preserve this origin receipt.'
    save_json(output/'checkpoint_audit.json',result)
    timing=dict(schema_version=1,protocol='continuous_world_actual_latency_joint_motionkv_async_v3',
        checkpoint=result['full'],physical_grid=result['data_identity'],
        recommended_continuous_sync_contract=dict(nominal_action_interval_seconds=.16,nominal_history_frame_interval_seconds=.16),
        source='actual full Stage3 checkpoint physical_grid_contract and matching full-data manifests; native flow policy_stride=4 remains separate',
        simulation_rate=1.,max_backlog_seconds=.25,infer_clock_tolerance_seconds=.02,bootstrap='prime_then_hold',H=4,R=4,K=10,
        slow_refresh_actions=16,timing_is_not_a_latency_measurement=True,gpu_smoke_passed=False)
    save_json(output/'timing_contract.json',timing)
    print(json.dumps(dict(passed=True,step=args.step,full_sha256=result['full']['sha256'],backbone=result['backbone'],
        strict_full_reload=True,strict_fast_reload=True,cuda_initialized=False,output=str(output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('full','fast','profile','physical-grid','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,default=20000)
    run(p.parse_args())

if __name__=='__main__':main()
