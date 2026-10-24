import threading
import pytest
import torch
from dynamicwam.dualrate.gpu_priority import FastPriorityGate, SlowOperatorSlices


def test_queued_fast_precedes_next_slow_and_releases_after_failure():
    gate=FastPriorityGate(); order=[]; started=threading.Event()
    gate.reserve_fast()
    def slow():
        started.set()
        with gate.slow(): order.append('slow')
    thread=threading.Thread(target=slow,daemon=True);thread.start()
    assert started.wait(1)
    with pytest.raises(RuntimeError):
        with gate.fast(lambda: order.append('drained')):
            order.append('fast');raise RuntimeError('inference error')
    thread.join(1)
    assert not thread.is_alive()
    assert order==['fast','drained','slow']
    assert gate.pending==0 and not gate.busy


def test_slices_preserve_tensor_results_and_drain_each_dispatch():
    gate=FastPriorityGate(); calls=[]
    class Stream:
        def synchronize(self): calls.append('drain')
    with SlowOperatorSlices(gate,Stream()):
        value=torch.arange(4.)*2+1
    torch.testing.assert_close(value,torch.tensor([1.,3.,5.,7.]))
    assert len(calls)>=3 and not gate.busy


def test_failed_drain_and_cancel_cannot_leave_slow_waiting():
    gate=FastPriorityGate();gate.reserve_fast();gate.cancel_fast()
    gate.reserve_fast()
    def broken(): raise RuntimeError('stream error')
    with pytest.raises(RuntimeError):
        with gate.fast(broken): pass
    with gate.slow(): assert gate.busy
    assert gate.pending==0 and not gate.busy
