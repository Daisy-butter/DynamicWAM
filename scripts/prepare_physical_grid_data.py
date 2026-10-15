#!/usr/bin/env python3
"""Read every episode clock/drive-target trajectory; write a new CPU-only index.

No image, latent, CUDA, NVML or simulator imports. Existing data remain read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import time
import numpy as np
import torch
from dynamicwam.training.data.physical_grid import (DT, VERSION, validate_clock,
    target_grid, action_brackets, video_endpoints, sha256_file, json_hash)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-index', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--max-video-lag',type=float,default=.16)
    p.add_argument('--max-command-gap',type=float,default=.16)
    p.add_argument('--episode-limit',type=int,default=0)
    p.add_argument('--sleep-ms',type=float,default=1.)
    a=p.parse_args(); torch.set_num_threads(1)
    source=Path(a.source_index); old=json.loads((source.parent/'manifest.json').read_text())
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise RuntimeError('refusing to overwrite completed physical grid')
    (out/'episodes').mkdir(exist_ok=True)
    counts=Counter(); drops=Counter(); tasks={}; entries=[]; ids=[]; dt_counts=Counter()
    raw_root=Path(old['source_root']); packed_root=Path(old['packed_root'])
    lang={int(r['episode_id']):int(r['lang_group_id']) for r in map(json.loads,(packed_root/'episodes.jsonl').open())}
    for line in source.open():
        meta=json.loads(line); e=int(meta['episode_index'])
        if a.episode_limit and len(entries)>=a.episode_limit: break
        clock_path=Path(meta['clock_path'])
        clock=torch.load(clock_path,map_location='cpu',weights_only=True)
        frames=np.asarray(clock['frame_index']).reshape(-1).astype(np.int64)
        times=np.asarray(clock['sim_time_seconds']).reshape(-1).astype(np.float64)
        qpath=raw_root/meta['split']/meta['task_name']/'qpos'/f"{meta['episode_name']}.pt"
        commands=np.asarray(torch.load(qpath,map_location='cpu',weights_only=True))
        validate_clock(frames,times,commands)
        dt_counts.update(np.round(np.diff(times),6).tolist())
        kept=[]; action_times=[]; video_indices=[]; video_times=[]; video_lags=[]
        for frame,sample in meta['source_candidates']:
            grid=target_grid(times[frame],times[0])
            if grid[-1]>times[-1]+1e-9:
                drops['past_episode_end']+=1; continue
            lo,hi,alpha,gaps=action_brackets(times,grid)
            vf,vt,vl=video_endpoints(times,grid[1::2])
            if gaps.max()>a.max_command_gap+1e-9:
                drops['command_capture_gap']+=1; continue
            if vl.max()>a.max_video_lag+1e-9:
                drops['video_capture_lag']+=1; continue
            kept.append((frame,sample)); action_times.append(grid)
            video_indices.append(vf); video_times.append(vt); video_lags.append(vl)
        eligible=np.asarray(kept,dtype=np.int64).reshape(-1,2)
        path=out/'episodes'/f'{e:06d}.npz'
        np.savez(path,frames=frames,times=times,commands=commands.astype(np.float32),
            eligible=eligible, action_times=np.asarray(action_times,dtype=np.float64).reshape(-1,16),
            video_frames=np.asarray(video_indices,dtype=np.int64).reshape(-1,8),
            video_actual_times=np.asarray(video_times,dtype=np.float64).reshape(-1,8),
            video_lags=np.asarray(video_lags,dtype=np.float64).reshape(-1,8))
        row={k:meta[k] for k in ('episode_index','episode_name','split','subset','task_name')}
        row.update(lang_group_id=lang[e],index_path=str(path.relative_to(out)),sha256=sha256_file(path),
            clock_sha256=sha256_file(clock_path),command_sha256=sha256_file(qpath),
            source_samples=len(meta['source_candidates']),eligible_samples=len(kept),
            video_max_lag_seconds=float(np.max(video_lags)) if kept else None,
            first_sample_id=int(eligible[0,1]) if kept else None)
        entries.append(row); ids.extend(eligible[:,1].tolist())
        counts['episodes']+=1; counts['source_samples']+=len(meta['source_candidates']);counts['eligible_samples']+=len(kept)
        counts[meta['subset']+'_episodes']+=1
        task=tasks.setdefault(meta['task_name'],Counter());task['source']+=len(meta['source_candidates']);task['eligible']+=len(kept)
        if len(entries)%250==0:
            (out/'progress.json').write_text(json.dumps(dict(counts=counts,drops=drops,updated_at=time.time())))
            print(json.dumps(dict(counts=counts,drops=drops)),flush=True)
        if a.sleep_ms: time.sleep(a.sleep_ms/1000)
    episode_path=out/'episodes.jsonl';episode_path.write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in entries))
    ids=np.asarray(sorted(ids),dtype=np.int64);np.save(out/'eligible_sample_ids.npy',ids)
    if len(np.unique(ids))!=len(ids): raise ValueError('repeated eligible sample identity')
    contract=dict(version=VERSION,action_dt=DT,action_chunk=16,video_frames=8,video_action_ratio=2,
        grid_anchor='episode_first_actual_sim_time', arm_labels='recorded_drive_targets_piecewise_linear',
        gripper_labels='previous_recorded_command_hold',gripper_indices=[6,13],
        observation_clock='actual_capture_time_preserved',video_selection='last_capture_at_or_before_endpoint',
        max_video_lag_seconds=a.max_video_lag,max_command_gap_seconds=a.max_command_gap,
        condition_latents='original_at_actual_condition_frame_with_native_motion_history',
        future_latents='new_vae_encoding_required_no_legacy_fallback',cache_age_actions=list(range(32)),
        prefix_source='expert_demonstration',on_policy_status='blocked_missing_real_same_trajectory_replay')
    m=dict(schema_version=VERSION,status='cpu_index_complete_latents_missing',full_scope=not bool(a.episode_limit),
        source_root=str(raw_root),packed_root=str(packed_root),source_index_sha256=sha256_file(source),
        source_manifest_sha256=sha256_file(source.parent/'manifest.json'),
        dataset_identity=old['dataset_identity'],episode_partition_identity=old['episode_partition_identity'],
        legacy_pairing_identity=old['legacy_pairing_identity'],partition_seed=old['pairing']['partition_seed'],
        validation_fraction=old['pairing']['validation_fraction'],counts=dict(counts),drops=dict(drops),
        tasks=tasks,clock_dt_histogram={str(k):v for k,v in sorted(dt_counts.items())},contract=contract,
        episode_index_sha256=sha256_file(episode_path),eligible_ids_sha256=sha256_file(out/'eligible_sample_ids.npy'))
    m['physical_grid_identity']=json_hash(m)
    (out/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(root=str(out),counts=counts,drops=drops,identity=m['physical_grid_identity'])),flush=True)
if __name__=='__main__': main()
