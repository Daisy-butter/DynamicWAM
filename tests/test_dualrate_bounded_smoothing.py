"""Deployment projection contract and real runtime queue feedback, CPU only."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dualrate_eval_worker import validated_arm_smoothing
from dynamicwam.dualrate.actuators import ActuatorProjection
from dynamicwam.dualrate.deployment import BoundedArmSmoothing
from dynamicwam.dualrate.physical_clock import WallPhysicsBridge
from dynamicwam.dualrate.runtime import DualRateRuntime, RuntimeConfig
from dynamicwam.dualrate.timing import ExecutionQueue, TimedPlan
from test_dualrate_live_clock_contract import ManualPool, DT, H


SPEC = dict(kind='bounded_causal_ema_v1', alpha=.85, max_offset_rad=.03)


def native():
    limits = torch.tensor([[-10., 10.]] * 14)
    limits[[6, 13]] = torch.tensor([[0., 1.], [0., 1.]])
    return ActuatorProjection(limits, provenance='explicit CPU fixture')


def test_prefix_grippers_bounds_delta_and_forecast_receipt():
    project = BoundedArmSmoothing(native(), SPEC)
    raw = torch.arange(16, dtype=torch.float32)[:, None].expand(-1, 14).clone() / 10
    raw[:, 6] = torch.arange(16) % 2
    raw[:, 13] = 1 - raw[:, 6]
    original = raw.clone()
    known = torch.arange(16) < 3
    actual, receipt = project(raw, known)
    assert torch.equal(actual[known], raw[known])
    assert torch.equal(actual[:, [6, 13]], raw[:, [6, 13]])
    assert (actual - raw).abs().max() <= .030001
    assert receipt['hard_prefix_preserved'] and receipt['forecast_invalidated']
    assert receipt['smoothing']['grippers_preserved']
    assert torch.equal(raw, original)
    identity, no_change = BoundedArmSmoothing(native(), dict(SPEC, alpha=1.))(raw, known)
    assert torch.equal(identity, raw) and not no_change['forecast_invalidated']
    whole, unchanged = project(raw, torch.ones(16, dtype=torch.bool))
    assert torch.equal(whole, raw) and not unchanged['forecast_invalidated']
    for mask in (torch.zeros(16, dtype=torch.bool), torch.tensor([True, False] * 8)):
        with pytest.raises(ValueError, match='contiguous'):
            project(raw, mask)


@pytest.mark.parametrize('spec', [dict(SPEC, alpha=.84), dict(SPEC, alpha=float('nan')),
    dict(SPEC, alpha=True), dict(SPEC, max_offset_rad=.031), dict(SPEC, max_offset_rad=-1),
    dict(SPEC, max_offset_rad=True), dict(SPEC, extra=1), dict(SPEC, kind='other'), {}])
def test_worker_and_runtime_reject_same_bad_specs(spec):
    with pytest.raises(ValueError):
        validated_arm_smoothing({'deployment_arm_smoothing': spec})
    with pytest.raises(ValueError):
        BoundedArmSmoothing(native(), spec)


def test_default_config_off_and_valid_spec_copied():
    assert validated_arm_smoothing({}) is None
    got = validated_arm_smoothing({'deployment_arm_smoothing': SPEC})
    assert got == SPEC and got is not SPEC


def test_real_runtime_invalidates_z_and_feeds_projected_queue_into_next_prefix():
    physical, wall, events = [0.], [100.], []
    bridge = WallPhysicsBridge(H, wall_clock=lambda: wall[0])
    bridge.activate(0., 100.)
    queue = ExecutionQueue(TimedPlan(0., DT, ((0.,) * 14,) * 16, 1, 0.))
    queue.action_at(0.)
    class System(nn.Module):
        def __init__(self):
            super().__init__()
            self.fast = nn.Linear(14, 14)
            self.slow = nn.Linear(1, 1)
            self.fast.config = SimpleNamespace(action_dim=14, chunk_size=16)
            self.seen = []
        def empty_world(self, batch):
            return {}
        def sample_fast(self, batch, world, **kwargs):
            self.seen.append({k: v.clone() for k, v in batch.items()})
            actions = torch.ones(1, 16, 14)
            actions[:, :, 6] = .25
            actions[:, :, 13] = .75
            return dict(actions=actions, future=torch.ones(1, 12, 134),
                future_valid=torch.ones(1, 12, dtype=torch.bool),
                future_field_valid=torch.ones(1, 12, 134, dtype=torch.bool),
                contact_valid=torch.ones(1, 12, 2, dtype=torch.bool),
                contact_probability=torch.ones(1, 12, 2))
    system = System().eval()
    runtime = DualRateRuntime(system, queue, lambda: physical[0],
        RuntimeConfig(DT, min_query_budget_seconds=.3, max_protect_steps=3),
        action_mean=torch.zeros(14), action_scale=torch.ones(14), clock_bridge=bridge,
        action_projection=BoundedArmSmoothing(native(), SPEC), record=events.append)
    runtime.pool.shutdown(wait=True)
    runtime.pool = ManualPool()
    def batch(lease, ref):
        return dict(query_time=torch.tensor([lease.query_time], dtype=torch.float64),
            action_times=torch.tensor([[lease.prefix.start_time + i * DT for i in range(16)]], dtype=torch.float64))
    try:
        runtime.start_fast(batch, observation_time=0.)
        runtime.pool.run_one()
        output = runtime.poll()
        projected = output['executable_actions_raw']
        assert torch.equal(projected[:2], torch.zeros(2, 14))
        assert float(projected[2, 0]) == pytest.approx(.97)
        assert projected[2, 6] == .25 and projected[2, 13] == .75
        for key in ('future_valid', 'future_field_valid', 'contact_valid', 'future', 'contact_probability'):
            assert not output[key].any()
        physical[0] = 2 * DT
        wall[0] = 100. + physical[0]
        actual = torch.tensor(queue.action_at(physical[0]))
        assert torch.equal(actual, projected[2])
        runtime.start_fast(batch, observation_time=physical[0])
        runtime.pool.run_one()
        runtime.poll()
        second = system.seen[1]
        assert torch.equal(second['fast_known_actions'][0, :2], projected[2:4])
        assert torch.equal(second['fast_known_mask'][0], torch.tensor([True, True] + [False] * 14))
        assert all(e['hard_prefix_preserved'] for e in events if e['event'] == 'actuator_projection')
    finally:
        runtime.close()
