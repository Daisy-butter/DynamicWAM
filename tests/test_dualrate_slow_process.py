"""Actual spawned CPU RPC, RNG parity and owned-child failure cleanup."""
import json
from pathlib import Path
import socket
import sys
import time

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_slow_process import SlowProcessSystem,_assert_exact,_receive


class ToySystem:
    training=False
    _fast_graph_sampler='initial'
    def __init__(self,offset=0.,delay=0.):self.offset=offset;self.delay=delay
    def predict_reference(self,batch,*,generator):
        time.sleep(self.delay)
        return batch['x']+torch.randn(batch['x'].shape,generator=generator)+self.offset
    def world_readout(self,prediction):
        return {'world_tokens':prediction,'generation_stage':'same_noise_test',
                'world_valid':torch.ones(prediction.shape[:1],dtype=torch.bool)}


def toy_loader(config,device):return ToySystem(config.get('offset',0.),config.get('delay',0.))


def broken_loader(config,device):raise RuntimeError('deliberate loader failure')


def build(model,config=None,**kwargs):
    return SlowProcessSystem(model,config or {},slow_device='cpu',startup_timeout=30.,
        _loader=toy_loader,_test_fixture=True,**kwargs)


def test_actual_spawn_exact_world_rng_and_facade_assignment(tmp_path):
    model=ToySystem();path=tmp_path/'receipt.json'
    service=build(model,receipt_path=path)
    generator=torch.Generator().manual_seed(12)
    expected_generator=torch.Generator().manual_seed(12)
    try:
        for x in (torch.arange(6.).reshape(2,3),torch.ones(1,4)):
            batch={'x':x}
            expected=model.world_readout(model.predict_reference(batch,generator=expected_generator))
            actual=service.world_readout(service.predict_reference(batch,generator=generator))
            _assert_exact(actual,expected)
            assert torch.equal(generator.get_state(),expected_generator.get_state())
            assert torch.equal(batch['x'],x)
        service._fast_graph_sampler=None
        assert model._fast_graph_sampler is None
        assert service.receipt()['calls']==2
        assert json.loads(path.read_text())['bootstrap_exact_parity_passed']
        assert service.receipt()['parent_pid']!=service.receipt()['pid']
    finally:service.close()
    receipt=json.loads(path.read_text())
    assert receipt['closed'] and receipt['exitcode']==0 and not receipt['forced_owned_child_cleanup']


def test_parity_failure_is_not_reported_as_accepted_and_rng_is_restored():
    service=build(ToySystem(),{'offset':1.})
    generator=torch.Generator().manual_seed(12);before=generator.get_state().clone()
    try:
        with pytest.raises(ValueError,match='exact parity mismatch'):
            service.predict_reference({'x':torch.zeros(1,2)},generator=generator)
        assert torch.equal(before,generator.get_state())
        assert not service.receipt()['bootstrap_exact_parity_passed']
        assert service.receipt()['failed']
    finally:service.close(suppress_errors=True)
    assert not service._process.is_alive()


def test_owned_child_timeout_is_bounded_and_never_retried():
    service=build(ToySystem(),{'delay':.3},call_timeout=.05)
    try:
        began=time.monotonic()
        with pytest.raises((TimeoutError,socket.timeout)):
            service.predict_reference({'x':torch.zeros(1,2)},generator=torch.Generator().manual_seed(1))
        assert time.monotonic()-began<1.
        assert service.receipt()['failed'] and service.receipt()['calls']==0
        with pytest.raises(RuntimeError,match='unavailable'):
            service.predict_reference({'x':torch.zeros(1,2)},generator=torch.Generator())
    finally:service.close(suppress_errors=True)
    assert not service._process.is_alive()


def test_partial_ipc_frame_cannot_wait_indefinitely():
    a,b=socket.socketpair()
    try:
        b.sendall(b'\x00\x00')
        with pytest.raises((TimeoutError,socket.timeout)):_receive(a,time.monotonic()+.02)
    finally:a.close();b.close()


def test_exact_check_distinguishes_signed_zero():
    with pytest.raises(ValueError,match='parity mismatch'):
        _assert_exact(torch.tensor([-0.]),torch.tensor([0.]))


def test_spawn_startup_failure_propagates():
    with pytest.raises(RuntimeError,match='startup failed'):
        SlowProcessSystem(ToySystem(),{},slow_device='cpu',startup_timeout=30.,
            _loader=broken_loader,_test_fixture=True)
