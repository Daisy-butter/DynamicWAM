#!/usr/bin/env python3
"""New causal-spatial recipe; legacy training and deployment are untouched."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from dynamicwam.training.train.common import build_accelerator,setup_logging
from train_multirate import finish_training

def load_recipe(path):
 import yaml
 r=yaml.safe_load(Path(path).read_text())
 from dynamicwam.training.train.causal_spatial_trainer import normalize_recipe_numbers
 r=normalize_recipe_numbers(r)
 if r.get('format')!='causal_spatial_training_v1':raise ValueError('unsupported recipe')
 if r['runtime_contract']['simulation_rate']!=1.:raise ValueError('V4 target is fixed 1x physical clock')
 if r['runtime_contract']['H']!=2 or r['runtime_contract']['R']!=2:raise ValueError('V4 recipe requires explicit H2/R2')
 if r['microbatch']*r['accumulation']*8!=128:raise ValueError('formal/smoke global batch must be 128')
 if not 0<=r['warmup_steps']<r['steps']:raise ValueError('warmup outside budget')
 if r['diagnostic'] and r['steps']!=4:raise ValueError('diagnostic is four optimizer updates')
 if not Path(r['run_dir']).resolve().is_relative_to(Path('/SSD_DISK/users/wuruihan')):raise ValueError('run outside verified SSD user root')
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
 r=load_recipe(args.config)
 cfg=dict(name=Path(r['run_dir']).name,checkpoint_dir=str(Path(r['run_dir']).parent),device='cuda',
  gradient_accumulation_steps=r['accumulation'],checkpoint_total_limit=4)
 ds=dict(zero_optimization={'stage':0},bf16={'enabled':True},gradient_clipping=1.,
  train_micro_batch_size_per_gpu=r['microbatch'],gradient_accumulation_steps=r['accumulation'],steps_per_print=1000000)
 a=build_accelerator(cfg,ds,even_batches=False);setup_logging('INFO',a.process_index)
 from accelerate.utils import set_seed
 set_seed(20261021)
 from dynamicwam.training.train.causal_spatial_trainer import CausalSpatialTrainer
 trainer=None
 try:
  trainer=CausalSpatialTrainer(r,a);set_seed(20261021+a.process_index)
  trainer.run_training()
  import torch
  torch.cuda.synchronize(a.device);a.wait_for_everyone()
 finally:finish_training(a,trainer,preserve_exception=sys.exc_info()[0] is not None)
if __name__=='__main__':main()
