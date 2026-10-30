"""Native synchronous sampling: full and terminal windows, unchanged split and labels."""
import json
from pathlib import Path
import numpy as np
from .producer import (sha256_file,bound_metadata_path,verify_file_stat,
                       _action_targets_supported,select_frame,RGB_TOLERANCE,DT)

CONTRACT="native_sync_ordinal_v5"

def native_rows(times,stride,ages):
    t=np.asarray(times,dtype=np.float64)
    if len(t)<2 or stride<=0 or any(a<0 or abs(a/DT-round(a/DT))>1e-7 for a in ages):
        raise ValueError("invalid native waypoint sampling")
    result=[]
    for g in range(max(0,int(np.floor((t[-1]-t[0])/stride))+1)):
        s=float(t[0]+g*stride)
        try:select_frame(t,s,past_only=True,tolerance=RGB_TOLERANCE)
        except ValueError:continue
        for ai,age in enumerate(ages):
            q=s+age
            if q+DT>t[-1]+1e-8:continue
            try:select_frame(t,q,past_only=True,tolerance=RGB_TOLERANCE)
            except ValueError:continue
            # At least the first target is real; no complete-horizon filter.
            if _action_targets_supported(t,np.array([q+DT]))[0]:result.append((g,ai))
    return result

def derive(source,output,ages=(0.,.18,.36,.54,.72,1.08,1.44,2.16)):
    import h5py
    source=Path(source).resolve(strict=True);out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    parents={s:json.loads((source/(s+'.json')).read_text()) for s in ('train','validation')}
    m=parents['train'];episodes=json.loads(bound_metadata_path(m,source,'episodes_file').read_text())
    stats=json.loads((source/m['source_file_stats']).read_text())['episodes']
    rows={s:[] for s in parents};tail=0
    for i,ep in enumerate(episodes):
        verify_file_stat(ep['raw_path'],stats[ep['episode_uid']]['raw'])
        with h5py.File(ep['raw_path'],'r') as f:t=f['interception/sim_time_seconds'][:]
        additions=native_rows(t,m['source_stride_seconds'],ages)
        rows[ep['split']].extend((i,g,a) for g,a in additions)
        tail+=int(sum(t[0]+g*m['source_stride_seconds']+ages[a]+16*DT>t[-1]+1e-8 for g,a in additions))
        if (i+1)%1000==0:print(json.dumps(dict(episodes=i+1,rows={k:len(v) for k,v in rows.items()},tail=tail)),flush=True)
    out.mkdir(parents=True)
    receipt=dict(schema=CONTRACT,episodes=len(episodes),rows={k:len(v) for k,v in rows.items()},
        terminal_windows=tail,query_ages_seconds=ages,
        action_semantics="expert waypoints sampled at nominal .18s; ordinal targets, not promised TOPP end times",
        deployment="native synchronous take_action; zero inference-time physical advancement and zero committed prefix",
        data_distribution="offline expert, not on-policy TOPP replay",future_semantics="actual physical-time labels",
        missing_supervision="partial A masked; incomplete causal video fully masked; Z fields individually masked")
    for split,parent in parents.items():
        new=dict(parent)
        for key in ('episodes_file','text_index','prompts_file','source_file_stats'):
            new[key]=str((source/parent[key]).resolve(strict=True))
        arr=out/(split+'_rows.npy');np.save(arr,np.asarray(rows[split],dtype=np.int64))
        new.update(native_v5=True,time_contract=CONTRACT,action_interval_seconds=DT,
            rows_file=str(arr),rows_sha256=sha256_file(arr),query_ages_seconds=list(ages),
            camera_availability_seconds=0.,condition_rgb_max_age_seconds=.060001,
            parent_manifest=str(source/(split+'.json')),parent_manifest_sha256=sha256_file(source/(split+'.json')),
            timing_derivation=CONTRACT,latency_provenance="physics frozen during inference; wall cost measured separately")
        (out/(split+'.json')).write_text(json.dumps(new,indent=2))
    (out/'audit.json').write_text(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(derive(args.source,args.output)),flush=True)

