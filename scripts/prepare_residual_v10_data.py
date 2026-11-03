"""Bind measured physical-time expert windows to the V10 executor/label contract.

Never edit source demonstrations or clamp a recorded commitment to make it fit.
Reject unsupported/infeasible windows with counts; preserve the original split.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import h5py
import numpy as np
import torch
from dynamicwam.dualrate.producer import resample_interval_commands,bound_metadata_path,verify_file_stat
from dynamicwam.dualrate.exposure_timing import load_templates,validate_template
from dynamicwam.dualrate.data import sha256_file
from dynamicwam.dualrate.c2_executor import displacement_interval,ARMS

def verified_trace_equivalence(cfg):
    binding=cfg.get('zero_head_trace_equivalence')
    if binding is None:return None
    path=Path(binding['path']).resolve(strict=True)
    if sha256_file(path)!=binding['sha256']:raise ValueError('trace equivalence receipt changed')
    proof=json.loads(path.read_text())
    if (not proof['passed'] or proof['new_residual']!=cfg['residual_v10'] or
        proof['runtime']!=cfg['runtime'] or len(proof['reports'])!=35 or
        proof['max_command_difference_rad']>=2e-6):
        raise ValueError('invalid zero-head trace equivalence')
    return proof

def validate_trace(report,path,cfg,proof):
    if report.get('runtime_contract')!=cfg['runtime']:
        raise ValueError('timing trace runtime changed')
    if report.get('residual_v10')==cfg['residual_v10']:return
    if (proof is None or report.get('residual_v10')!=proof['old_residual'] or
        proof['reports'].get(str(path))!=sha256_file(path)):
        raise ValueError('timing traces require exact settings or measured zero-head command parity')


def prepare(source,output,config):
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    config=Path(config).resolve(strict=True)
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(config.read_text());s=cfg['residual_v10'];dt=s['action_dt']
    proof=verified_trace_equivalence(cfg)
    audit=dict(kind='offline_expert_not_on_policy',config_sha256=sha256_file(config),splits={},formal_training_ready=False)
    if proof is not None:audit['zero_head_trace_equivalence']=cfg['zero_head_trace_equivalence']
    manifests={}
    for split in ('train','validation'):
        path=source/(split+'.json');m=json.loads(path.read_text())
        events=load_templates(bound_metadata_path(m,source,'event_templates_file'),m['event_templates_file_sha256'])
        if events['kind']!='measured_parallel':raise ValueError('formal data requires actual parallel timing traces')
        for evidence in events['trace_evidence']:
            trace_path=Path(evidence['path']).resolve(strict=True)
            if sha256_file(trace_path)!=evidence['sha256']:raise ValueError('timing report changed')
            report=json.loads(trace_path.read_text())
            validate_trace(report,trace_path,cfg,proof)
        ep=json.loads(bound_metadata_path(m,source,'episodes_file').read_text())
        row_source=(source/m['rows_file']).resolve(strict=True)
        if sha256_file(row_source)!=m['rows_sha256']:raise ValueError('input rows changed')
        rows=np.load(row_source)
        stats=json.loads(bound_metadata_path(m,source,'source_file_stats').read_text())['episodes']
        keep=[];reasons=Counter();tested=0
        for epi in np.unique(rows[:,0]):
            e=ep[int(epi)];verify_file_stat(e['raw_path'],stats[e['episode_uid']]['raw'])
            with h5py.File(e['raw_path'],'r') as h:
                times=h['interception/sim_time_seconds'][:];actions=h['joint_action/vector'][:]
            candidates=[];q=[];v=[]
            for row in rows[rows[:,0]==epi]:
                frame,tid=map(int,row[1:]);event=validate_template(events['templates'][tid],dt)
                n=event['fast_prefix_rows'];source_time=float(times[frame])
                if not event['reference_available'] or n<2:
                    reasons['unavailable_reference_or_short_prefix']+=1;continue
                first=event['fast_action_start']+n*dt
                if first+dt>16*dt+1e-7 or event['query_time']>cfg['continuous_v8']['draft_max_age']+1e-7:
                    reasons['no_first_executable_slow_coverage']+=1;continue
                starts=source_time+event['fast_action_start']+np.array([n-2,n-1])*dt
                values,valid,_=resample_interval_commands(times,actions,starts,dt)
                if not valid.all():reasons['unsupported_prefix']+=1;continue
                candidates.append(row);q.append(values[-1,list(ARMS)]);v.append((values[-1,list(ARMS)]-values[-2,list(ARMS)])/dt)
            if candidates:
                bounds=torch.tensor(s['bounds'],dtype=torch.float64)[list(ARMS)]
                if s.get('executor')=='phase_c2':
                    values=torch.tensor(np.asarray(q))
                    ok=torch.isfinite(values)&(values>=bounds[:,0])&(values<=bounds[:,1])
                else:
                    _,_,ok=displacement_interval(torch.tensor(np.asarray(q)),torch.tensor(np.asarray(v)),dt,bounds[:,0],bounds[:,1],s['limits'])
                valid=ok.all(-1).numpy();keep.extend(np.asarray(candidates)[valid]);tested+=len(candidates)
                reasons['infeasible_recorded_prefix']+=int((~valid).sum())
        if not keep:raise ValueError('no feasible windows remain')
        row_path=output/(split+'_rows.npy');np.save(row_path,np.asarray(keep,dtype=np.int64))
        new=dict(m)
        for key in ('episodes_file','text_index','prompts_file','source_file_stats','event_templates_file'):
            new[key]=str(bound_metadata_path(m,source,key))
        for key in ('continuous_acceptance_file','continuous_acceptance_sha256'):new.pop(key,None)
        new.update(rows_file=str(row_path),rows_sha256=sha256_file(row_path),residual_v10=s,
            continuous_gripper_sampling='interval_start_previous_event',continuous_flow_sampling='raw_frames_v1',
            future_anchor='first_mutable_start',measured_runtime=cfg['runtime'],formal_training_ready=False,
            parent_manifest=str(path),parent_manifest_sha256=sha256_file(path))
        if proof is not None:new['zero_head_trace_equivalence']=cfg['zero_head_trace_equivalence']
        audit['splits'][split]=dict(input_rows=len(rows),retained_rows=len(keep),tested_prefixes=tested,rejections=dict(reasons))
        manifests[split]=new
    receipt=output/'prefix_filter.json';receipt.write_text(json.dumps(audit,indent=2))
    for split,m in manifests.items():
        m.update(prefix_filter_file=str(receipt),prefix_filter_sha256=sha256_file(receipt))
        (output/(split+'.json')).write_text(json.dumps(m,indent=2))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.config)))
