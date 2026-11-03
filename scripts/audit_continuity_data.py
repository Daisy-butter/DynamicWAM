"""CPU proof of the real replay schema and exact physical commitments."""
import argparse,json
from pathlib import Path
import torch
from dynamicwam.dualrate.continuity_data import ContinuityDataset,reference_identity
from dynamicwam.dualrate.system import DualRateSystem
from dynamicwam.dualrate.residual_v10 import current_teacher_batch
from dynamicwam.dualrate.train import validate_row

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2);d=ContinuityDataset(a.manifest);seen={};examples=[]
 for i,row in enumerate(d.replay):seen.setdefault(row['task'],i)
 assert len(seen)==35
 for task,i in seen.items():
  r=d.replay_item(i);validate_row(r);b={k:v[None] for k,v in r.items()}
  DualRateSystem._validate_clocks(b);DualRateSystem._validate_prefix_pair(b)
  assert r['recorded_replay'] and r['measured_state_valid']
  assert not r['future_loss_valid'].any() and not r['contact_valid'].any()
  assert torch.isfinite(r['measured_state']).all()
  teacher=current_teacher_batch(b);known=teacher['source_known_mask']
  assert known.any() and not known.all()
  first=r['action_intervals'][int(r['fast_known_mask'].sum()),0]
  assert abs(float(teacher['source_action_intervals'][0,int(known.sum())-1,1]-first))<1e-7
  assert torch.all(r['executed_action_intervals'][r['executed_action_valid'],1]<=r['query_time']+1e-7)
  examples.append(dict(task=task,query=float(r['query_time']),source=float(r['source_time']),exposure=float(r['state_time']),prefix=int(r['fast_known_mask'].sum())))
 paired=int((d.previous>=0).sum());assert paired>0
 for i,j in enumerate(d.previous):
  if j<0:continue
  a=d.event_templates['templates'][int(d.rows[i,2])];b=d.event_templates['templates'][int(d.rows[j,2])]
  assert reference_identity(a)==reference_identity(b)
  assert a['publish_time']<=b['query_time']<=a['query_time']
 Path(a.output).write_text(json.dumps(dict(passed=True,tasks=len(seen),replay_rows=len(d.replay),expert_rows=d.expert_length,replay_sampling_slots=d.replay_slots,paired_rows=paired,examples=examples,scope='CPU real-state provenance and same-time contracts; actual Full parity requires GPU smoke'),indent=2))
if __name__=='__main__':main()
