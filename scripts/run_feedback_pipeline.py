#!/usr/bin/env python3
"""Sequential feedback fine-tuning and paired DOMINO Level-1 evaluations."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.training.train.common import get_export_dir,get_run_dir
from run_official_temporal_pipeline import eval_summary,write_json

def checkpoint_evidence(path,phase,step):
    import torch
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert p['global_step']==step
    assert p['config']['runtime_contract']['feedback']['ready']==(phase=='policy')
    assert p['modules']['compact_fast']['feedback_enabled']
    assert all(torch.isfinite(v).all() for k,v in p['model'].items() if k.startswith('fast_policy.'))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return dict(path=str(path),sha256=h.hexdigest(),global_step=step)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6')
    parser.add_argument('--episodes',type=int,default=20)
    args=parser.parse_args()
    root=args.run_root.absolute();root.mkdir(parents=True,exist_ok=True)
    with (root/'pipeline.started').open('x') as f:f.write(str(os.getpid()))
    state=dict(status='running',pid=os.getpid(),source=str(ROOT),completed=[],started=time.time())
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    def save():
        state['updated']=time.time();write_json(root/'status.json',state)
    def run(name,cmd):
        state.update(phase=name,command=cmd,log=str(root/f'{name}.log'));save()
        with (root/f'{name}.log').open('x') as stream:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=p.pid;save();rc=p.wait()
        state.pop('child_pid',None)
        if rc:raise RuntimeError(f'{name} exited {rc}; see {root/name}.log')
        state['completed'].append(name);save()
    try:
        exports={}
        for phase in ('dynamics','policy'):
            cfg=ROOT/'configs'/f'feedback_{phase}_7gpu.yaml'
            profile=load_multirate_profile(cfg)
            if get_run_dir(profile.training).exists():
                raise RuntimeError(f'run already exists: {get_run_dir(profile.training)}; resume explicitly')
            run(phase,[str(ROOT/'.venv/bin/accelerate'),'launch','--multi_gpu','--num_processes',str(len(args.gpus.split(','))),
                '--num_machines','1','--mixed_precision','bf16','--main_process_port','36461','--dynamo_backend','no',
                str(ROOT/'scripts/train_multirate.py'),'--config',str(cfg)])
            steps=profile.training['max_steps'];exports[phase]=get_export_dir(profile.training)/f'multirate_step_{steps}.pt'
            state[f'{phase}_checkpoint']=checkpoint_evidence(exports[phase],phase,steps);save()
        old='/SSD_DISK/users/wuruihan/DynamicWAM/outputs/training/official_temporal_stage3_7gpu_20260930/exports/multirate_step_3500.pt'
        variants=[('feedback_h4',exports['policy'],4,False),('feedback_disabled_h4',exports['policy'],4,True),
                  ('previous_temporal_h4',old,4,False),('official_sync_h4',None,4,False),('official_sync_h16',None,16,False)]
        arms={}
        for name,ckpt,horizon,disable in variants:
            options=dict(policy_name='ciwam.adapters.domino.deploy_policy_sync_flow',ckpt_setting=name,
                         execute_horizon=horizon,eval_video_log=False)
            if ckpt:
                options.update(policy_name='ciwam.adapters.domino.deploy_policy_multirate_flow',multirate_checkpoint=str(ckpt),
                    multirate_scheduling='controlled_ratio',bridge_refresh_ratio=4 if name.startswith('feedback') else 2,
                    multirate_reuse_action_noise=True,multirate_disable_feedback=disable)
            override=root/'eval_configs'/f'{name}.json';write_json(override,options)
            arm=root/'eval'/name
            run(name,[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/eval_domino.py'),
                '--config',str(ROOT/'configs/absolute_motion_v2.yaml'),'--policy-overrides',str(override),
                '--run-root',str(arm),'--gpus',args.gpus,'--episodes-per-task',str(args.episodes),'--start-seed','100000'])
            arms[name]=arm
            report=eval_summary(arms)
            for entry in report['arms'].values():
                rows=entry['timing_rows'];n=sum(x.get('query_count',0) for x in rows)
                steps=sum(x.get('executed_steps',0) for x in rows);ms=sum(x.get('query_ms_sum',0) for x in rows)
                entry.update(query_ms_mean=ms/n if n else None,ms_per_executed_action=ms/steps if steps else None,
                    actual_slow_builds=sum(x.get('actual_slow_builds',0) for x in rows))
            write_json(root/'evaluation_summary.json',report)
        state.update(status='complete',finished=time.time());save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise
    return 0

if __name__=='__main__':raise SystemExit(main())
