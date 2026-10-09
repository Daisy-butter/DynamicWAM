#!/usr/bin/env python3
"""Fixed-budget development probe of usable world context, not a success claim."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train_world_context_feedback import (initialize_head,export_checkpoint,prepare_batch,
    teacher_only_objective,set_seed,sha256)
from dynamicwam.training.data.world_context_feedback_dataset import WorldContextFeedbackDataset,world_context_collate_fn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--init-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--seed',type=int,default=20261008)
    a=p.parse_args()
    if a.steps<1:p.error('positive fixed step budget required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    set_seed(a.seed);torch.set_num_threads(1)
    head,initialization,reference=initialize_head(a.init_checkpoint)
    train=WorldContextFeedbackDataset(a.manifest,partition='train',reference_checkpoint=reference)
    heldout=WorldContextFeedbackDataset(a.manifest,partition='validation',reference_checkpoint=reference)
    initial=copy.deepcopy(head).cuda().eval().requires_grad_(False)
    head.cuda().train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=2e-5,weight_decay=.001)
    loader=DataLoader(train,batch_size=4,shuffle=True,collate_fn=world_context_collate_fn,num_workers=0)
    iterator=iter(loader);seen=0;gradient=[]
    for step in range(a.steps):
        try: batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        inputs,target=prepare_batch(batch,'cuda',history_mode='temporal')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=head(**inputs);loss,_=teacher_only_objective(output,inputs,target)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('probe loss is nonfinite')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('probe gradient is nonfinite')
        gradient.append(float(head.world_output.weight.grad.abs().sum()))
        optimizer.step();seen+=len(target)
    # Use a deterministic different context from the same task whenever possible.
    # This avoids an easier cross-task semantic mismatch masquerading as timing.
    rows=[heldout[i] for i in range(len(heldout))]
    alternatives=[]
    for i,row in enumerate(rows):
        candidate=next((j for j in range(i+1,len(rows)) if rows[j]['task']==row['task'] and
                        rows[j]['bundle_sha256']!=row['bundle_sha256']),None)
        if candidate is None:
            candidate=next((j for j in range(i) if rows[j]['task']==row['task'] and
                            rows[j]['bundle_sha256']!=row['bundle_sha256']),None)
        alternatives.append(candidate)
    selected=[i for i,j in enumerate(alternatives) if j is not None]
    if not selected:raise ValueError('no within-task different-plan context pairs')
    totals={name:[0.,0.] for name in ('fresh','initialization','no_context','context_shuffle','zero')}
    episode_metrics={};head.eval()
    with torch.inference_mode():
        for start in range(0,len(selected),4):
            indices=selected[start:start+4]
            batch=world_context_collate_fn([rows[i] for i in indices])
            inputs,target=prepare_batch(batch,'cuda',history_mode='temporal')
            alternative=world_context_collate_fn([rows[alternatives[i]] for i in indices])
            other,_=prepare_batch(alternative,'cuda',history_mode='temporal')
            shuffled=dict(inputs)
            for k in ('world_values','world_coordinates','world_mask'):
                shuffled[k]=other[k]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predictions=dict(fresh=head(**inputs),initialization=initial(**inputs),
                    no_context=head(**dict(inputs,context_valid=torch.zeros_like(inputs['context_valid']))),
                    context_shuffle=head(**shuffled))
            predictions['zero']=torch.zeros_like(predictions['fresh'])
            for name,prediction in predictions.items():
                _,metric=teacher_only_objective(prediction,inputs,target,l2_weight=0.)
                totals[name][0]+=float(metric['teacher_sse']);totals[name][1]+=float(metric['count'])
                for k,index in enumerate(indices):
                    sub={key:value[k:k+1] for key,value in inputs.items()}
                    _,m=teacher_only_objective(prediction[k:k+1],sub,target[k:k+1],l2_weight=0.)
                    identity=f"{rows[index]['task']}:{rows[index]['seed']}"
                    entry=episode_metrics.setdefault(identity,{}).setdefault(name,[0.,0.])
                    entry[0]+=float(m['teacher_sse']);entry[1]+=float(m['count'])
    mse={k:s/max(n,1) for k,(s,n) in totals.items()}
    passed=bool(max(gradient)>0 and all(torch.isfinite(torch.tensor(list(mse.values())))) and
                mse['fresh']<min(mse[k] for k in ('initialization','no_context','context_shuffle','zero')) and
                mse['fresh']<=.99*mse['no_context'])
    report=dict(status='complete',passed=passed,heldout_teacher_mse=mse,
        paired_records=len(selected),heldout_records=len(heldout),episode_metrics=episode_metrics,
        world_output_gradient_nonzero=max(gradient)>0,steps=a.steps,examples_seen=seen,seed=a.seed,
        manifest_sha256=sha256(a.manifest),init_checkpoint_sha256=sha256(a.init_checkpoint),
        controller_protocol_sha256=train.controller_protocol_sha256,
        interpretation='development feasibility only; not independent confirmation or closed-loop quality',
        minimum_context_gain_fraction=.01,
        gate='fresh lower clipped teacher MSE than initialization, zero, disabled and same-task shuffled context; >=1% vs disabled')
    payload=export_checkpoint(head,reference=reference,data=train,initialization=initialization,
        history_mode='temporal',step=a.steps,phase='probe',examples_seen=seen,validation=report,
        config=dict(world_size=1,batch_size=4,global_batch=4,gradient_accumulation=1))
    torch.save(payload,a.output_dir/'checkpoint.pt')
    report['checkpoint_sha256']=sha256(a.output_dir/'checkpoint.pt')
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
