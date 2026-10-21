"""Full CPU-only provenance gate; no CUDA imports with visible devices."""
from pathlib import Path
import argparse, json, sys, hashlib, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from dynamicwam.training.data.physical_grid_dataset import validate_physical_grid_readiness
from supervise_motionkv_training import atomic_json, file_digest, utc
SSD=Path('/SSD_DISK/users/wuruihan/DynamicWAM')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 started=time.time();out={'started_at':utc(),'passed':False,'gpu_used':False}
 try:
  out['data']=validate_physical_grid_readiness(SSD/'outputs/data/joint_motionkv_physical_grid_20261015',SSD/'outputs/data/joint_motionkv_physical_grid_20261015/future_latents',verify_hashes=True,require_full=True)
  init=SSD/'outputs/checkpoint_init/absolute_motion_init.pt'
  out['initialization']={'path':str(init),'sha256':file_digest(init),'bytes':init.stat().st_size}
  if out['initialization']['sha256']!='601936c61503d173c38d7c9ae2ce956dfa22825e4b5c4e1096424f9586df540d':raise ValueError('formal Stage2 initialization identity changed')
  out['passed']=True
 except BaseException as exc:
  out['error']=repr(exc);raise
 finally:
  out['elapsed_seconds']=time.time()-started;out['finished_at']=utc();atomic_json(args.output,out)
if __name__=='__main__':main()
