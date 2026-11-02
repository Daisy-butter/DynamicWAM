#!/usr/bin/env python3
"""CPU preflight and detached continuous V8 eval with higher admission reserve.

Sibling of continuous_v8_final_eval: same trained weights and runtime fields, but
allows deployment_min_query_budget_seconds to raise effective min_query_budget
within trained protect_steps=2 / max_protect_steps=3 (see effective_runtime_settings).
This is an explicitly separate evaluated protocol, not the V8 equality gate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

import dualrate_posttrain_eval as post
import dualrate_eval_worker as worker

ROOT=Path(__file__).resolve().parents[1]
RECIPE=[dict(phase=name,steps=steps) for name,steps in
        [('warmup',500),('small_age_joint',4000),('cross_time_joint',8000)]]
HB360_BUDGET=0.36


def _runtime_without_budget(runtime):
    return {k:v for k,v in runtime.items() if k!='min_query_budget_seconds'}


def validate_contract(spec,config,training,manifest):
    if config.get('seed')!=100000 or config.get('task_config')!='demo_clean_dynamic':
        raise ValueError('paired benchmark scene seed/config changed')
    if spec.get('training_recipe')!=RECIPE or config.get('training_recipe')!=RECIPE:
        raise ValueError('this launcher is only for formal V8 500/4000/8000')
    if not config.get('continuous_v8') or any(config.get(k) for k in
            ('native_v5','native_v6','native_v7','engineering_initialization_only')):
        raise ValueError('trained continuous V8 required; native/engineering modes forbidden')
    for key in ('architecture','continuous_v8','runtime','manifest','sequence_id','training_recipe',
                'codec_id','init_checkpoint','init_checkpoint_sha256','action_normalization_sha256',
                'motion_normalization_sha256','time_contract','action_interval_seconds',
                'future_latent_shape','future_field_support','contact_field_support','vision_weights_sha256'):
        if key not in config or config[key]!=training.get(key):
            raise ValueError('final training/evaluation differs: '+key)
    if (config.get('deployment_arm_smoothing') is not None
            or config.get('bootstrap_hold_steps')!=2
            or config.get('max_backlog')!=.5
            or config.get('frame_period')!=training['runtime']['action_dt']
            or config['benchmark_contract'].get('controller')!='endpoint_linear_arm_native_gripper'):
        raise ValueError('executor/timing changes outside hb360 admission reserve are forbidden')
    reserve=float(config.get('deployment_min_query_budget_seconds', -1))
    if reserve!=HB360_BUDGET or float(spec.get('deployment_min_query_budget_seconds', -1))!=HB360_BUDGET:
        raise ValueError('hb360 protocol requires deployment_min_query_budget_seconds=0.36')
    effective=worker.effective_runtime_settings(config)
    trained=training['runtime']
    if _runtime_without_budget(effective)!=_runtime_without_budget(trained):
        raise ValueError('hb360 may only raise min_query_budget_seconds within trained prefix support')
    if (effective['min_query_budget_seconds']!=HB360_BUDGET
            or trained['min_query_budget_seconds']!=.2
            or trained['protect_steps']!=2
            or trained['max_protect_steps']!=3):
        raise ValueError('hb360 effective budget/protect support mismatch')
    if effective==trained:
        raise ValueError('hb360 must differ from equality-gated V8 via higher admission reserve')
    if manifest.get('measured_runtime')!=config['runtime']:
        raise ValueError('measured training runtime differs from deployment')
    from dynamicwam.dualrate.continuous_contract import validate_training_contract
    validate_training_contract(training,manifest)


def check(spec):
    source=post.guard.frozen_source(ROOT)
    if source['source_sha256']!=spec['expected_training_source_sha256']:
        raise ValueError('policy, data or executor Python differs from protected training')
    sequence=post.read(Path(spec['training_sequence_dir'])/'sequence.frozen.json')
    if worker.sha(Path(spec['training_sequence_dir'])/'sequence.frozen.json')!=spec['training_sequence_sha256']:
        raise ValueError('training sequence changed')
    training=post.read(sequence['phases'][-1]['config'])
    for row in sequence['phases']:
        if worker.sha(row['config'])!=row['config_sha256']:
            raise ValueError('protected training template changed')
    configs=[]
    for kind in ('smoke','formal'):
        if worker.sha(spec[kind+'_template'])!=spec[kind+'_template_sha256']:
            raise ValueError('evaluation template changed')
        config=post.read(spec[kind+'_template']);configs.append(config)
        validate_contract(spec,config,training,post.read(config['manifest']))
        if (config['checkpoint_sha256']!='PENDING_TRAINING_COMPLETION' or
                Path(config['checkpoint']).resolve()!=Path(training['output'],'last.pt').resolve()):
            raise ValueError('template must wait for this final checkpoint only')
        worker.validate_config(config,checkpoint_pending=True)
        if Path(config['output']).exists():
            raise ValueError('refusing reused evaluation output: '+config['output'])
    post.validate_pair(*configs)
    config=configs[1]
    worker.verify_artifacts(config)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay
    from dynamicwam.dualrate.renderer_runtime import validate_renderer_runtime
    validate_dependency_overlay(config);validate_renderer_runtime(config)
    if worker.sha(config['vision_weights'])!=config['vision_weights_sha256']:
        raise ValueError('fixed vision encoder changed')
    count=sum(len(worker.load_frozen_task(config,task)) for task in config['tasks'])
    if count!=700:raise ValueError('frozen scene set must contain all 700 exact episodes')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('native video verification dependencies missing')
    free=shutil.disk_usage(spec['storage_root']).free
    if free<config['min_free_bytes']:raise ValueError('insufficient artifact disk space')
    ready=post.training_ready(spec)
    status=post.read(Path(spec['training_sequence_dir'])/'status.json')
    return dict(passed=True,scope='CPU preflight; hb360 higher-admission-reserve GPU acceptance pending',
        protocol_id=spec.get('protocol_id'),deployment_min_query_budget_seconds=HB360_BUDGET,
        effective_runtime=worker.effective_runtime_settings(configs[1]),
        trained_runtime=configs[1]['runtime'],
        source=source,training_state=status['state'],training_phase=status.get('phase'),
        training_complete_accepted=ready is not None,final_global_step=12500,
        frozen_episodes=count,free_bytes=free,environment_groups=config['environment_groups'],
        benchmark=config['benchmark_contract']['evaluation_setting'],
        runtime=config['runtime'],launched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--detach',action='store_true')
    args=parser.parse_args();spec=post.read(args.spec)
    result=check(spec);print(json.dumps(result,ensure_ascii=False),flush=True)
    if args.check:return 0
    sys.argv=[str(ROOT/'scripts/dualrate_posttrain_eval.py'),'--config',str(args.spec.resolve()),
              '--run-dir',spec['controller_dir'],'--detach']
    return post.main()


if __name__=='__main__':raise SystemExit(main())
