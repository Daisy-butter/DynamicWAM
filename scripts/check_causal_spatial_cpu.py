"""Actual dataset/initialization CPU integration, separate from tiny unit tests."""
from pathlib import Path
import argparse,json,sys,traceback,gc
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from train_causal_spatial import load_recipe
from supervise_motionkv_training import atomic_json,utc,source_files_and_hash,file_digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from dynamicwam.training.train.causal_spatial_trainer import build_dataset,align_proposal
 from dynamicwam.training.data.causal_spatial_dataset import collate_causal_spatial,audit_causal_spatial_coverage
 from dynamicwam.inference.causal_spatial_loader import build_from_initialization,native_payload
 r=load_recipe(a.config);out=dict(started_at=utc(),passed=False,gpu_used=False,
  source_sha256=source_files_and_hash(ROOT)[1],config_sha256=file_digest(a.config))
 try:
  out['coverage']=audit_causal_spatial_coverage(r['data']['physical_grid_root'],r['data']['future_latent_root'])
  ds=build_dataset(r);rows=[ds[i] for i in [0,len(ds)//2,len(ds)-1]];b=collate_causal_spatial(rows)
  out['real_samples']={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in b.items() if torch.is_tensor(v)}
  assert b['clean_future'].shape==(3,48,2,12,10)
  assert b['source_condition_latent'].shape==(3,48,2,24,20)
  assert torch.equal(b['future_target_offsets'],b['action_target_offsets'][:,[7,15]])
  assert torch.all(b['spatial_times']<=b['current_time'][:,None])
  model,cfg=build_from_initialization(r['initial_checkpoint'],r['model'],'cpu')
  out['parameters']={k:sum(x.numel() for x in module.parameters()) for k,module in [('total',model),('video',model.backbone.compact_wan),('action',model.backbone.action_expert),('belief',model.belief),('future',model.future_blocks)]}
  fast=model.make_fast_model();fast.load_state_dict(model.fast_state_dict(),strict=True)
  out['fast_strict_roundtrip']=all(torch.equal(v,fast.state_dict()[k]) for k,v in model.fast_state_dict().items())
  assert out['fast_strict_roundtrip']
  del model,fast;gc.collect()
  teacher=native_payload(r['teacher_checkpoint'])
  out['teacher_normalization_equal']=all(teacher['config']['action_normalization']['stats'][k]==cfg['action_normalization']['stats'][k] for k in ['mean','std'])
  assert out['teacher_normalization_equal']
  out['passed']=True
 except BaseException:
  out['error']=traceback.format_exc();raise
 finally:out['finished_at']=utc();atomic_json(a.output,out)
if __name__=='__main__':main()
