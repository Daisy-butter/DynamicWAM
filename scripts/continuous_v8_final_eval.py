#!/usr/bin/env python3
"""CPU preflight and detached final-checkpoint continuous V8 evaluation."""
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


def validate_contract(spec,config,training,manifest):
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
    if (worker.effective_runtime_settings(config)!=training['runtime']
            or config.get('deployment_arm_smoothing') is not None
            or config.get('bootstrap_hold_steps')!=2
            or config.get('max_backlog')!=.5
            or config.get('frame_period')!=training['runtime']['action_dt']
            or config['benchmark_contract'].get('controller')!='endpoint_linear_arm_native_gripper'):
        raise ValueError('executor/timing changes require a separate evaluated protocol')
    if manifest.get('measured_runtime')!=config['runtime']:
        raise ValueError('measured training runtime differs from deployment')
    from dynamicwam.dualrate.continuous_contract import validate_training_contract
    validate_training_contract(training,manifest)


def check(spec):
    source=post.guard.frozen_source(ROOT)
    if source['source_sha256']!=spec['expected_training_source_sha256']:
        raise ValueError('policy, data or executor Python differs from protected training')
    sequence=post.read(Path(spec['training_sequence_dir'])/'sequence.frozen.json')
    if post.file_sha256(Path(spec['training_sequence_dir'])/'sequence.frozen.json')!=spec['training_sequence_sha256']:
        raise ValueError('training sequence changed')
    training=post.read(sequence['phases'][-1]['config'])
    for row in sequence['phases']:
        if post.file_sha256(row['config'])!=row['config_sha256']:
            raise ValueError('protected training template changed')
    configs=[]
    for kind in ('smoke','formal'):
        if post.file_sha256(spec[kind+'_template'])!=spec[kind+'_template_sha256']:
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
    # Saved receipts and /proc only. No CUDA/NVML, guard acquisition or signal.
    ready=post.training_ready(spec)
    status=post.read(Path(spec['training_sequence_dir'])/'status.json')
    return dict(passed=True,scope='CPU preflight; final-policy GPU acceptance pending',
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
