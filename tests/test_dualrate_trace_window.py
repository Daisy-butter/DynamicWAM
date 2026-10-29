"""Lossless CPU history trimming; no claim of GPU latency improvement."""
from types import SimpleNamespace

import pytest
import torch

from dynamicwam.dualrate.deployment import reference_trace_since
from dynamicwam.dualrate.system import DualRateSystem
from dynamicwam.dualrate.timing import ExecutionQueue, TimedPlan
from test_dualrate_system import make_system, generator


@pytest.mark.parametrize('offset', [0., -.5e-7, .5e-7])
def test_long_queue_retains_exact_deviation_values_validity_and_gradients(offset):
    dt = 45 * float(torch.tensor(.004, dtype=torch.float32))
    commands = torch.arange(916 * 2, dtype=torch.float64).reshape(916, 2) / 1000
    queue = ExecutionQueue(TimedPlan(0., dt, tuple(map(tuple, commands.tolist())), 1, 0.))
    for index in range(900):
        queue.action_at(index * dt)
    query = 900 * dt
    starts = torch.arange(895, 911, dtype=torch.float64) * dt + offset
    intervals = torch.stack((starts, starts + dt), -1)[None]
    reference = SimpleNamespace(payload={'reference_action_intervals': intervals})
    full = queue.completed_execution_trace(0., query)
    trimmed = queue.completed_execution_trace(reference_trace_since(reference, 0., query), query)
    assert len(full) == 900 and 0 < len(trimmed) <= 6
    cutoff = len(full) - len(trimmed)
    assert full[cutoff:] == trimmed
    acts = torch.tensor([[x[2] for x in full]], dtype=torch.float64, requires_grad=True)
    ref_actions = torch.randn(1, 16, 2, dtype=torch.float64, requires_grad=True)
    world = dict(world_tokens=torch.zeros(1, 18, 3, dtype=torch.float64),
                 world_types=torch.tensor([[0, 1] + [2] * 16]),
                 reference_actions=ref_actions, reference_action_intervals=intervals)
    batch = dict(executed_actions=acts,
                 executed_action_intervals=torch.tensor([[[x[0], x[1]] for x in full]], dtype=torch.float64),
                 executed_action_valid=torch.ones(1, len(full), dtype=torch.bool),
                 query_time=torch.tensor([query], dtype=torch.float64),
                 fast_known_actions=commands[900:916][None].clone(),
                 action_intervals=torch.tensor([[[i*dt, (i+1)*dt] for i in range(900, 916)]], dtype=torch.float64),
                 fast_known_mask=torch.tensor([[True, True] + [False] * 14]))
    cropped = dict(batch)
    for key in ('executed_actions', 'executed_action_intervals', 'executed_action_valid'):
        cropped[key] = batch[key][:, cutoff:]
    model = SimpleNamespace(test_fixture=False, fast=SimpleNamespace(config=SimpleNamespace(action_dim=2)))
    a = DualRateSystem._action_deviation(model, batch, world)
    b = DualRateSystem._action_deviation(model, cropped, world)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    weight = torch.arange(a[0].numel(), dtype=torch.float64).reshape_as(a[0])
    ga = torch.autograd.grad((a[0] * weight).sum(), (acts, ref_actions), retain_graph=True)
    gb = torch.autograd.grad((b[0] * weight).sum(), (acts, ref_actions))
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_fixed_noise_complete_fast_output_is_unchanged():
    system, batch = make_system()
    system.eval()
    world = system.world_readout(system.predict_reference(batch, generator=generator()))
    starts = torch.arange(20, dtype=torch.float64) * .1
    batch.update(executed_actions=torch.randn(1, 20, 2),
                 executed_action_intervals=torch.stack((starts, starts + .1), -1)[None],
                 executed_action_valid=torch.ones(1, 20, dtype=torch.bool))
    reference = SimpleNamespace(payload=world)
    since = reference_trace_since(reference, 0., float(batch['query_time'][0]))
    keep = batch['executed_action_intervals'][0, :, 1] > since
    cropped = dict(batch)
    for key in ('executed_actions', 'executed_action_intervals', 'executed_action_valid'):
        cropped[key] = batch[key][:, keep]
    a = system.sample_fast(batch, world, steps=2, generator=generator())
    b = system.sample_fast(cropped, world, steps=2, generator=generator())
    for key in ('actions', 'future', 'contact_probability'):
        assert torch.equal(a[key], b[key])


def test_unknown_payloads_retain_original_history_without_device_sync():
    assert reference_trace_since(None, 1., 2.) == 1.
    assert reference_trace_since(SimpleNamespace(version=1), 1., 2.) == 1.
    assert reference_trace_since(SimpleNamespace(payload=None), 1., 2.) == 1.
    assert reference_trace_since(SimpleNamespace(payload=()), 1., 2.) == 1.
    for value in (None, torch.empty(1, 0, 2, dtype=torch.float64),
                  torch.ones(1, 2, 2, device='meta', dtype=torch.float64),
                  torch.tensor([[[float('nan'), 2.]]], dtype=torch.float64),
                  torch.tensor([[[2., 1.]]], dtype=torch.float64)):
        assert reference_trace_since(SimpleNamespace(payload={'reference_action_intervals': value}), 1., 2.) == 1.
