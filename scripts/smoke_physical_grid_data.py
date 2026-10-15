#!/usr/bin/env python3
"""Real-data CPU smoke without inventing missing VAE latents or GPU claims."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.physical_grid import DT,resample_commands,sample_async_timing,sha256_file
from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical-grid-root',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    root=Path(a.physical_grid_root);out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'manifest.json').read_text());episodes=[json.loads(l) for l in (root/'episodes.jsonl').read_text().splitlines() if l]
    packed=PackedAbsoluteMotionDataset(m['packed_root'],max_open_shards=2)
    normalizer=RoboTwinQposNormalizer.from_stats_path(Path(m['packed_root'])/'action_stats.json',epsilon=1e-6)
    examples=[];summary=[]
    import cv2
    cv2.setNumThreads(1)
    for meta in [episodes[0],episodes[len(episodes)//2],episodes[-1]]:
        path=root/meta['index_path'];assert sha256_file(path)==meta['sha256']
        with np.load(path) as z:ep={k:z[k] for k in z.files}
        elig=ep['eligible'];times=ep['times'][elig[:,0]]
        video=Path(m['source_root'])/meta['split']/meta['task_name']/'videos'/f"{meta['episode_name']}.mp4"
        cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        try:
            for age in (0,4,15,31):
                fp=int(np.searchsorted(times,times[0]+age*DT+1e-9,side='right')-1)
                if fp<0 or times[-1]<times[0]+age*DT:continue
                frame,sid=map(int,elig[fp]);raw=packed[sid]
                torch.testing.assert_close(raw['initial_state'],normalizer.normalize(torch.from_numpy(ep['commands'][frame])),rtol=1e-5,atol=1e-5)
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();assert ok
                views,masks=fast_views_from_composite(np.ascontiguousarray(img[:,:,::-1]))
                targets=ep['action_times'][fp];commands=resample_commands(ep['times'],ep['commands'],targets)
                labels=normalizer.normalize(torch.from_numpy(commands))
                timing=sample_async_timing(times[0],times[fp],targets,seed=20261015,epoch=0,
                    source_id=int(elig[0,1]),current_id=sid)
                assert np.all(targets>times[fp]) and np.allclose(np.diff(targets),DT)
                assert np.all(ep['video_actual_times'][fp]<=targets[1::2]+1e-9)
                assert np.all(ep['video_actual_times'][fp]>times[fp])
                assert bool(torch.isfinite(labels).all())
                # No future tensor is fabricated: this fixture cannot pass the training data gate.
                examples.append(dict(physical_grid_version=1,episode_index=meta['episode_index'],sample_id=sid,
                    actual_capture_time=float(times[fp]),fast_rgb=torch.from_numpy(views),fast_rgb_valid_mask=torch.from_numpy(masks),
                    initial_state=raw['initial_state'],native_motion=raw['absolute_motion_features'],
                    action_sequence=labels,action_target_times=torch.from_numpy(targets.copy()),
                    async_timing=timing,future_latent_status='missing_requires_real_VAE_encoding'))
                summary.append(dict(episode=meta['episode_index'],sample_id=sid,nominal_cache_age=age,
                    actual_cache_age_seconds=float(times[fp]-times[0]),first_target_offset=float(targets[0]-times[fp]),
                    max_video_lag=float(ep['video_lags'][fp].max()),prefix_length=timing['rtc_prefix_lengths']))
        finally:cap.release()
    try:validate_physical_grid_readiness(root,root/'future_latents',verify_hashes=True)
    except FileNotFoundError as exc:gate=dict(blocked=True,reason=str(exc))
    else:gate=dict(blocked=False)
    torch.save(dict(kind='real_CPU_inputs_labels_only_NOT_training_ready',samples=examples),out/'actual_inputs_labels.pt')
    receipt=dict(physical_grid_identity=m['physical_grid_identity'],samples=summary,
        real_rgb_reads=len(examples),native_motion_preserved=True,gpu_used=False,new_future_latent_gate=gate)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
