#!/usr/bin/env python3
"""Encode new physical-grid future videos. Run ONLY under the approved GPU guard.

Eight independent processes use --rank 0..7 --world-size 8 --device cuda:0
with per-worker CUDA_VISIBLE_DEVICES. No distributed backend or NVML calls.
The --finalize mode is CPU only and verifies every index/shard/ID/hash.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import os
import shutil
import numpy as np
from dynamicwam.training.data.physical_grid import sha256_file, json_hash
ENCODER_KIND='native_wan2_2_vae_physical_grid_v1'
PREPROCESS=dict(rgb='cv2_decode_bgr_to_rgb',resize=[[384,320],[192,160]],interpolation='cv2_aspect_pad_then_torch_bilinear_antialias_align_corners_false',range=[-1,1],frames=9,drop_first_latent=True)

def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--physical-grid-root',required=True);p.add_argument('--future-latent-root',required=True)
    p.add_argument('--vae-checkpoint');p.add_argument('--device',default='cuda:0')
    p.add_argument('--rank','--shard-index',dest='rank',type=int,default=0);p.add_argument('--world-size','--num-shards',dest='world_size',type=int,default=1)
    p.add_argument('--episode-limit','--max-episodes',dest='episode_limit',type=int,default=0,help='Incomplete smoke only; finalizer rejects missing episodes')
    p.add_argument('--finalize',action='store_true');p.add_argument('--guarded-launch',action='store_true')
    a=p.parse_args();root=Path(a.physical_grid_root);out=Path(a.future_latent_root)
    m=json.loads((root/'manifest.json').read_text());episodes=[json.loads(l) for l in (root/'episodes.jsonl').read_text().splitlines() if l]
    if sha256_file(root/'episodes.jsonl')!=m['episode_index_sha256']:raise ValueError('grid manifest changed')
    if not 0<=a.rank<a.world_size:raise ValueError('invalid worker rank')
    if a.finalize:
        from safetensors import safe_open
        receipts={}; all_ids=[];vae_ids=set();provenances=set()
        for e in episodes:
            if not e['eligible_samples']:continue
            i=int(e['episode_index']);path=out/'episodes'/f'{i:06d}.safetensors'
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['physical_grid_identity']!=m['physical_grid_identity'] or receipt['index_sha256']!=e['sha256']:
                raise ValueError('encoder index identity mismatch')
            if sha256_file(root/e['index_path'])!=e['sha256'] or sha256_file(path)!=receipt['sha256']:
                raise ValueError('encoder shard/index hash mismatch')
            with np.load(root/e['index_path']) as z:expected=z['eligible'][:,1]
            with safe_open(str(path),framework='np') as f:
                actual=f.get_tensor('sample_ids');shape=f.get_slice('future_latents').get_shape()
            if not np.array_equal(actual,expected) or shape!=[len(expected),48,2,12,10]:
                raise ValueError('encoded coverage or native latent shape mismatch')
            all_ids.extend(actual.tolist());receipts[str(i)]=receipt;vae_ids.add(receipt['vae_sha256'])
            provenance={k:receipt[k] for k in ('encoder_kind','encoder_sha256','vae_module_sha256','vae_checkpoint','vae_sha256','preprocess')}
            if provenance['encoder_kind']!=ENCODER_KIND or provenance['encoder_sha256']!=sha256_file(__file__) or provenance['preprocess']!=PREPROCESS:
                raise ValueError('encoder implementation/provenance mismatch')
            provenances.add(json.dumps(provenance,sort_keys=True))
        expected_ids=np.load(root/'eligible_sample_ids.npy')
        if not np.array_equal(np.asarray(sorted(all_ids)),expected_ids) or len(vae_ids)!=1 or len(provenances)!=1:
            raise ValueError('encoder full coverage or VAE identity mismatch')
        manifest=dict(schema_version=1,complete=True,physical_grid_identity=m['physical_grid_identity'],
            episode_partition_identity=m['episode_partition_identity'],eligible_ids_sha256=m['eligible_ids_sha256'],
            contract=m['contract'],provenance=json.loads(next(iter(provenances))),latent_shape=[48,2,12,10],latent_dtype='float32',episodes=receipts)
        atomic_json(out/'manifest.json',manifest)
        print(json.dumps(dict(complete=True,samples=len(all_ids),episodes=len(receipts))))
        return
    if a.device.startswith('cuda') and not a.guarded_launch:raise RuntimeError('requires approved all-adapter FD-before-telemetry guarded launch')
    if not a.vae_checkpoint:raise ValueError('--vae-checkpoint required')
    resolved=out.resolve();ssd=Path('/SSD_DISK/users/wuruihan').resolve()
    if ssd not in resolved.parents or resolved==Path(m['packed_root']).resolve():
        raise ValueError('new latent outputs must stay in a separate verified SSD directory')
    out.mkdir(parents=True,exist_ok=True)
    missing=[e for e in episodes if e['eligible_samples'] and not (out/'episodes'/f"{int(e['episode_index']):06d}.safetensors").exists()]
    needed=sum(e['eligible_samples'] for e in missing)*48*2*12*10*4+4*1024**3
    if shutil.disk_usage(out).free<needed:raise RuntimeError('SSD free space below full-cache estimate plus4GiB')
    # GPU imports are deliberately below CLI/manifest validation.
    import torch
    import torch.nn.functional as F
    import cv2
    from safetensors.torch import save_file
    from dynamicwam.image import resize_with_padding
    from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE
    torch.set_num_threads(1);cv2.setNumThreads(1)
    vae_hash=sha256_file(a.vae_checkpoint)
    import dynamicwam.vendor.wan.modules.vae2_2 as vae_module
    provenance=dict(encoder_kind=ENCODER_KIND,encoder_sha256=sha256_file(__file__),vae_module_sha256=sha256_file(vae_module.__file__),vae_checkpoint=str(Path(a.vae_checkpoint).resolve()),vae_sha256=vae_hash,preprocess=PREPROCESS)
    vae=Wan2_2_VAE(vae_pth=a.vae_checkpoint,device=a.device)
    (out/'episodes').mkdir(parents=True,exist_ok=True)
    selected=[e for e in episodes if int(e['episode_index'])%a.world_size==a.rank and e['eligible_samples']]
    if a.episode_limit:selected=selected[:a.episode_limit]
    for e in selected:
        i=int(e['episode_index']);path=out/'episodes'/f'{i:06d}.safetensors'
        if path.exists():
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['physical_grid_identity']!=m['physical_grid_identity'] or receipt['index_sha256']!=e['sha256'] or receipt['vae_sha256']!=vae_hash or sha256_file(path)!=receipt['sha256']:
                raise RuntimeError('existing shard failed immutable identity verification')
            print(json.dumps(dict(episode=i,already_verified=True,rank=a.rank)),flush=True)
            continue
        if sha256_file(root/e['index_path'])!=e['sha256']:raise ValueError('episode grid mutated')
        with np.load(root/e['index_path']) as z:eligible=z['eligible'];future=z['video_frames']
        video=Path(m['source_root'])/e['split']/e['task_name']/'videos'/f"{e['episode_name']}.mp4"
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError('video unavailable: '+str(video))
        latents=[]
        try:
            # Bounded one-episode image cache avoids random decoder seeks per sample.
            wanted=set(eligible[:,0].tolist()+future.reshape(-1).tolist());images={};frame=0
            while wanted:
                ok,img=cap.read()
                if not ok:raise RuntimeError('video ended before indexed frames')
                if frame in wanted:
                    images[frame]=np.ascontiguousarray(img[:,:,::-1]);wanted.remove(frame)
                frame+=1
            for j,(condition,sid) in enumerate(eligible):
                pixels=np.stack([resize_with_padding(images[int(f)],(384,320)) for f in [condition,*future[j]]])
                tensor=torch.from_numpy(pixels).permute(0,3,1,2).float().div_(255)
                # Native load_video_frames uses aspect-preserving padding at384x320,
                # followed by the packer's antialiased192x160 future-video resize.
                tensor=F.interpolate(tensor,size=(192,160),mode='bilinear',align_corners=False,antialias=True)
                tensor=(tensor*2-1).permute(1,0,2,3).unsqueeze(0).to(a.device)
                with torch.inference_mode():encoded=vae.encode(tensor)[:,:,1:].contiguous()
                if list(encoded.shape)!=[1,48,2,12,10] or not bool(torch.isfinite(encoded).all()):
                    raise ValueError('new VAE latent shape/nonfinite mismatch')
                latents.append(encoded[0].cpu().float())
        finally:cap.release()
        temporary=path.with_suffix('.tmp')
        save_file(dict(future_latents=torch.stack(latents),sample_ids=torch.from_numpy(eligible[:,1].copy())),str(temporary))
        temporary.replace(path)
        receipt=dict(path=str(path.relative_to(out)),sha256=sha256_file(path),bytes=path.stat().st_size,
            sample_count=len(eligible),index_sha256=e['sha256'],physical_grid_identity=m['physical_grid_identity'],
            **provenance,video_sha256=sha256_file(video),dtype='float32',shape=[48,2,12,10],
            future_input_frames=8,vae_input_frames=9,worker_rank=a.rank)
        atomic_json(path.with_suffix('.json'),receipt)
        print(json.dumps(dict(episode=i,samples=len(eligible),rank=a.rank)),flush=True)
if __name__=='__main__':main()
