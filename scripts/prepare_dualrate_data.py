#!/usr/bin/env python3
"""CPU-only real DOMINO data preparation; no implicit training/GPU launch."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from dynamicwam.dualrate.producer import build_manifests


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-episodes',type=int)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    config=json.loads(Path(args.config).read_text())
    root=Path('/SSD_DISK/users/wuruihan').resolve()
    output=Path(args.output).resolve()
    if not output.is_relative_to(root):raise ValueError('new data must be within verified SSD wuruihan directory')
    print(json.dumps(build_manifests(config,output,max_episodes=args.max_episodes),indent=2))


if __name__=='__main__':main()
