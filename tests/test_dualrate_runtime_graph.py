from types import SimpleNamespace

import pytest
import torch

from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig
from dynamicwam.dualrate.timing import TimingError


class FixtureGraph:
    def __init__(self):
        self.shapes=set();self.sealed=False
    def seal(self,*,expected_shapes):
        assert len(self.shapes)==expected_shapes
        self.sealed=True
    def receipt(self):
        return dict(passed=self.sealed,sealed=self.sealed,shape_count=len(self.shapes))


def fixture():
    # Exercise bootstrap orchestration without claiming CUDA graph execution.
    runtime=DualRateRuntime.__new__(DualRateRuntime)
    runtime.fast_graph=FixtureGraph();runtime._graph_prepared=False
    runtime.clock_bridge=SimpleNamespace(active=False)
    runtime.fast_rng=torch.Generator().manual_seed(41)
    runtime.config=RuntimeConfig(.18,fast_cuda_graph=True)
    events=[];calls=[];runtime.record=events.append;runtime.wall_clock=lambda:10.
    def sample(batch,world,*,steps,generator):
        calls.append((world['layout'],steps))
        runtime.fast_graph.shapes.add(world['layout'])
        return torch.randn(3,generator=generator)
    runtime.system=SimpleNamespace(sample_fast=sample,empty_world=lambda batch:dict(layout='empty'))
    return runtime,events,calls


def test_bootstrap_prepares_both_layouts_preserves_rng_and_seals_once():
    runtime,events,calls=fixture()
    before=runtime.fast_rng.get_state().clone()
    runtime._prepare_fast_graph({},dict(layout='real'))
    assert calls==[('real',10),('empty',10)]
    assert torch.equal(runtime.fast_rng.get_state(),before)
    assert runtime.fast_graph_receipt()==dict(enabled=True,passed=True,sealed=True,shape_count=2)
    assert events[-1]['event']=='fast_graph_prepared'
    runtime.clock_bridge.active=True
    runtime._prepare_fast_graph({},dict(layout='real'))
    assert len(calls)==2


def test_activation_forbids_first_capture_and_failure_restores_query_rng():
    runtime,events,calls=fixture()
    runtime.clock_bridge.active=True
    with pytest.raises(TimingError,match='forbidden after'):
        runtime._prepare_fast_graph({},dict(layout='real'))
    assert not calls
    runtime.clock_bridge.active=False
    before=runtime.fast_rng.get_state().clone()
    def fail(*args,**kwargs):
        torch.randn(10,generator=runtime.fast_rng)
        raise ValueError('parity failure')
    runtime.system.sample_fast=fail
    with pytest.raises(ValueError,match='parity failure'):
        runtime._prepare_fast_graph({},dict(layout='real'))
    assert torch.equal(before,runtime.fast_rng.get_state())
    assert not runtime._graph_prepared and not runtime.fast_graph.sealed
    assert events[-1]['event']=='fast_graph_preparation_failed'


def test_graph_switch_is_explicit_boolean_and_default_off():
    assert RuntimeConfig(.18).fast_cuda_graph is False
    with pytest.raises(ValueError):RuntimeConfig(.18,fast_cuda_graph='true')
