#!/usr/bin/env python3
"""Validate and optionally export one real same-trajectory policy replay."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from dynamicwam.training.data.rollout_replay import load_real_rollout_replay

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--output');a=p.parse_args()
    sample=load_real_rollout_replay(a.manifest)
    if a.output:
        path=Path(a.output)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(sample,path)
    print(json.dumps(dict(valid=True,trajectory_id=sample['provenance']['trajectory_id'],
        prefix_length=sample['rtc_prefix_length'],prefix_source=sample['prefix_source'])))
if __name__=='__main__':main()
