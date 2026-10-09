#!/usr/bin/env python3
"""Audit a fixed final checkpoint; no optimization or validation selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from train_predictive_feedback import (initialize_head,load_initialization,evaluate,
    evaluate_policy,sha256,SOURCE_FILES,ROOT)
from dynamicwam.training.data.predictive_feedback import PredictiveFeedbackDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--init-plan-feedback',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('a probe report cannot overwrite prior evidence')
    if a.batch_size<1:raise ValueError('positive batch size required')
    torch.set_num_threads(1)
    head,reference,init=initialize_head(a.init_plan_feedback)
    world=PredictiveFeedbackDataset(a.manifest,partition='validation',reference_checkpoint=reference)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    phase=checkpoint.get('phase')
    if phase not in ('world','joint'):raise ValueError('not a trained predictive checkpoint')
    identity=load_initialization(head,a.checkpoint,reference,world,phase)
    if checkpoint.get('data_manifest_sha256')!=world.manifest_sha256:
        raise ValueError('this fixed probe cannot silently change the registered heldout set')
    device=torch.device(a.device);head.to(device).eval().requires_grad_(False)
    world_report=evaluate(head,world,device=device,batch_size=a.batch_size)
    policy_report=None
    if phase=='joint':
        policy=PredictiveFeedbackDataset(a.manifest,partition='validation',reference_checkpoint=reference,mode='policy')
        policy_report=evaluate_policy(head,policy,device=device,batch_size=a.batch_size)
    report=dict(status='complete',passed=world_report['passed'] and
        (policy_report is None or policy_report['passed']),phase=phase,
        checkpoint=identity,data_manifest_sha256=world.manifest_sha256,
        world=world_report,policy=policy_report,init_plan_feedback_sha256=init['sha256'],
        source_hashes={name:sha256(ROOT/name) for name in SOURCE_FILES},
        script_sha256=sha256(__file__),optimization_performed=False,closed_loop_evaluated=False,
        interpretation='fixed checkpoint/heldout ablations; not hierarchical planning or task-success evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
