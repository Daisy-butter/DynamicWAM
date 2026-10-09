#!/usr/bin/env python3
"""CPU numerical comparison of real train/deploy WAN wrapper math on tiny weights.

Imports method bodies via AST so neither VAE, diffusers nor CUDA is required.
This checks wrapper math; it does not certify full checkpoint/GPU equivalence.
"""
from __future__ import annotations
import ast
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import unittest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
PATHS=('src/dynamicwam/training/models/wan_model.py',
       'src/dynamicwam/inference/models/wan_model.py',
       'src/dynamicwam/vendor/wan/modules/model.py')
if os.environ.get('WAN_MATH_SOURCE_BUNDLE'):
    sources=json.loads(Path(os.environ['WAN_MATH_SOURCE_BUNDLE']).read_text())
else:
    sources={p:(ROOT/p).read_text() for p in PATHS}
ns=dict(torch=torch,nn=nn,lru_cache=lru_cache)
vendor=ast.parse(sources[PATHS[2]])
functions=[n for n in vendor.body if isinstance(n,ast.FunctionDef) and n.name in
           ('sinusoidal_embedding_1d','rope_params','rope_apply')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])),PATHS[2],'exec'),ns)
METHODS={'_validate_wan_latent','_patch_video_latent','prepare_multiscale_video_tokens',
         'multiscale_layout_grid_sizes','apply_multiscale_rope','prepare_text_context',
         'prepare_time_embeddings','apply_multiscale_video_head'}
wrappers=[]
for path in PATHS[:2]:
    tree=ast.parse(sources[path]);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanVideoModel')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in METHODS]
    assert {n.name for n in cls.body}==METHODS
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    scope=dict(ns);exec(compile(ast.fix_missing_locations(module),path,'exec'),scope);wrappers.append(scope['WanVideoModel'])
torch.set_num_threads(1)

class TinyHead(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(24,48*4)
    def forward(self,x,e):return self.head(x+e)

class TinyWAN(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=(1,2,2);self.dim=24;self.freq_dim=8;self.text_len=7
        self.patch_embedding=nn.Conv3d(48,24,kernel_size=self.patch_size,stride=self.patch_size)
        self.text_embedding=nn.Linear(6,24)
        self.time_embedding=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,24))
        self.time_projection=nn.Sequential(nn.SiLU(),nn.Linear(24,144))
        self.head=TinyHead();self.freqs=ns['rope_params'](32,12)
    def unpatchify(self,x,grid):
        outputs=[]
        for row,(f,h,w) in zip(x,grid.tolist()):
            out=row.reshape(f,h,w,1,2,2,48).permute(6,0,3,1,4,2,5).reshape(48,f,h*2,w*2)
            outputs.append(out)
        return outputs

class WrapperMathChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93);wan=TinyWAN()
        self.train,self.deploy=[w() for w in wrappers]
        for w in (self.train,self.deploy):w.wan_model=copy.deepcopy(wan);w.precision=torch.float32
        self.condition=torch.randn(2,48,2,8,8);self.future=torch.randn(2,48,2,4,4)

    def test_multiscale_patch_layout_and_dense_lengths_exact(self):
        a=self.train.prepare_multiscale_video_tokens(self.condition,self.future)
        b=self.deploy.prepare_multiscale_video_tokens(self.condition,self.future)
        for i in (0,1,3):self.assertTrue(torch.equal(a[i],b[i]))
        for key in a[2]:
            if torch.is_tensor(a[2][key]):self.assertTrue(torch.equal(a[2][key],b[2][key]))
            else:self.assertEqual(a[2][key],b[2][key])
        self.assertEqual(a[2]['condition_seq_len'],32);self.assertEqual(a[2]['future_seq_len'],8)
        self.assertEqual(a[2]['condition_grid_shape'],(2,4,4));self.assertEqual(a[2]['future_grid_shape'],(2,2,2))
        self.assertEqual(a[1].tolist(),[40,40])

    def test_rope_both_scales_equal_and_future_phase_resets(self):
        _,_,layout,freqs=self.train.prepare_multiscale_video_tokens(self.condition,self.future)
        heads=torch.randn(2,40,2,12)
        a=self.train.apply_multiscale_rope(heads,layout,freqs);b=self.deploy.apply_multiscale_rope(heads,layout,freqs)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(a[:,32:],ns['rope_apply'](heads[:,32:],layout['future_grid_sizes'],freqs)))
        self.assertTrue(torch.equal(a[:,32],heads[:,32]))

    def test_time_scalar_and_per_token_paths_equal(self):
        for t in (torch.tensor([1000.,832.]),torch.tensor([1000.,832.],dtype=torch.bfloat16),
                  torch.arange(80).reshape(2,40).float()):
            a=self.train.prepare_time_embeddings(t,40);b=self.deploy.prepare_time_embeddings(t,40)
            self.assertEqual(tuple(a[0].shape),(2,40,24));self.assertEqual(tuple(a[1].shape),(2,40,6,24))
            for left,right in zip(a,b):self.assertTrue(torch.equal(left,right))

    def test_text_padding_truncation_projection_equal(self):
        texts=[torch.randn(3,6),torch.randn(12,6)]
        a=self.train.prepare_text_context(texts);b=self.deploy.prepare_text_context(texts)
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(2,7,24))

    def test_video_head_slices_only_future_identically(self):
        tokens,_,layout,_=self.train.prepare_multiscale_video_tokens(self.condition,self.future)
        time,_=self.train.prepare_time_embeddings(torch.tensor([1000.,832.]),40)
        a=self.train.apply_multiscale_video_head(tokens,time,layout)
        b=self.deploy.apply_multiscale_video_head(tokens,time,layout)
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(2,48,2,4,4))
        changed=tokens.clone();changed[:,:32]+=1000
        self.assertTrue(torch.equal(a,self.train.apply_multiscale_video_head(changed,time,layout)))

if __name__=='__main__':unittest.main(verbosity=2)
