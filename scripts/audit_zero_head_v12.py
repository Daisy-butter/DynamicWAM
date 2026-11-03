"""Prove retained real zero-head commands are unchanged by residual capacity.

No report, action, timestamp or source data is rewritten. CPU-only. This proof
does not claim equivalence for learned nonzero corrections or final policy SR.
"""
import argparse,copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from dynamicwam.dualrate.handoff_v12 import reconcile
from dynamicwam.dualrate.feedback_replay import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--new-config',required=True);p.add_argument('--index',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 old=json.loads(Path(a.old_config).read_text());new=json.loads(Path(a.new_config).read_text())
 os=old['residual_v10'];ns=new['residual_v10'];check=copy.deepcopy(ns)
 check['arm_radius']=os['arm_radius'];check['continuity']['handoff']['same_step_rad']=os['continuity']['handoff']['same_step_rad']
 assert check==os and old['runtime']==new['runtime']
 assert ns['arm_radius']==.24 and ns['continuity']['handoff']['same_step_rad']==.06
 payload=torch.load(old['checkpoint'],map_location='cpu',weights_only=True,mmap=True);weights=payload['model']
 assert not weights['fast.action_out.weight'].any() and not weights['fast.action_out.bias'].any()
 def system(s):return SimpleNamespace(residual_v10=s,handoff_v12=s['continuity']['handoff'],action_mean_v10=weights['action_mean_v10'],action_scale_v10=weights['action_scale_v10'])
 before,after=system(os),system(ns);reports={};maximum=0.;recorded_error=0.;count=0
 index=json.loads(Path(a.index).read_text());torch.set_num_threads(2)
 for entry in index['rows']:
  path=Path(entry['path']);assert digest(path)==entry['sha256'];rp=path.parents[1]/'report.json'
  if str(rp) not in reports:
   report=json.loads(rp.read_text());assert report['continuous_valid'] and report['realtime_valid']
   reports[str(rp)]=(digest(rp),{e['query_id']:e for e in report['events'] if e.get('event')=='whole_plan_handoff_audit'},
                    {e['query_id']:e for e in report['events'] if e.get('event')=='residual_action_audit'})
  _,events,residual=reports[str(rp)];q=int(path.stem.split('_')[-1]);event=events[q]
  assert torch.tensor(event['learned_correction']).abs().max()<2e-5
  row=torch.load(path,map_location='cpu',weights_only=True);b={k:v[None] for k,v in row.items()}
  base=(torch.tensor(event['raw_full_baseline'])[None]-before.action_mean_v10)/before.action_scale_v10
  valid=torch.tensor(residual[q]['draft_valid'])[None]
  x,_,_=reconcile(before,b,base,base,valid);y,_,_=reconcile(after,b,base,base,valid)
  error=float(((x-y)*before.action_scale_v10).abs().max());assert error<2e-6
  re=float(((x-b['fast_action'])*before.action_scale_v10).abs().max());assert re<2e-5
  maximum=max(maximum,error);recorded_error=max(recorded_error,re);count+=1
 assert len(reports)==35 and count>140
 receipt=dict(passed=True,scope='retained zero-residual initialization rollouts only; not learned-policy parity',
  old_residual=os,new_residual=ns,runtime=new['runtime'],reports={p:v[0] for p,v in reports.items()},
  rows=count,max_command_difference_rad=maximum,max_cpu_gpu_record_difference_rad=recorded_error,
  old_config_sha256=digest(Path(a.old_config)),new_config_sha256=digest(Path(a.new_config)),index_sha256=digest(Path(a.index)))
 Path(a.output).write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('old_residual','new_residual','reports','runtime')}))
if __name__=='__main__':main()
