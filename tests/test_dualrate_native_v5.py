import numpy as np
import torch
import pytest
from dynamicwam.dualrate.native_data import native_rows
from dynamicwam.dualrate.native_v5 import aligned_center

def test_terminal_windows_remain_and_no_beyond_last_target():
    t=np.arange(0,3.001,.06)
    rows=native_rows(t,.36,[0.,.36])
    assert (7,0) in rows  # 2.52 source, fewer than16 future actions
    assert all(g*.36+[0,.36][a]+.18<=t[-1]+1e-8 for g,a in rows)
    assert (0,0) in rows  # zero-prefix bootstrap must actually be trained

def test_prior_ordinal_shift_expiration_and_unrestricted_correction():
    ref=torch.arange(16).float()[None,:,None].expand(1,16,14).clone()
    b=dict(fast_known_actions=torch.zeros_like(ref),fast_known_mask=torch.zeros(1,16,dtype=torch.bool),
           native_query_index=torch.tensor([2]),query_time=torch.tensor([.7],dtype=torch.float64))
    w=dict(reference_actions=ref,native_source_index=torch.tensor([0]),
           native_available=torch.tensor([True]),source_time=torch.tensor([0.],dtype=torch.float64))
    center=aligned_center(b,w)
    assert center[0,0,0]>0 and center[0,0,0]<.3
    assert center[0,14:].count_nonzero()==0
    assert center[..., [6,13]].count_nonzero()==0
    b['fast_known_mask'][0,0]=True
    assert aligned_center(b,w)[0,0].count_nonzero()==0
    w['native_available'].zero_()
    assert aligned_center(b,w).count_nonzero()==0

def test_flow_prior_correct_endpoint_has_no_hard_output_constraint():
    gt=torch.randn(1,16,14);eps=torch.randn_like(gt);center=torch.randn_like(gt)*.15
    sigma=.7;x=(1-sigma)*gt+sigma*(eps+center)
    velocity=eps+center-gt
    torch.testing.assert_close(x-sigma*velocity,gt)

