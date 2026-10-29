"""Use the existing CPU owner-loop fixture; no model/GPU work is required."""
import gc
import pytest
from test_dualrate_eval_executor import fixture


def test_owner_loop_collects_timing_without_changing_steps_or_gc(monkeypatch):
    mod, session, system, builder, camera, config, _, _, _ = fixture(monkeypatch)
    callbacks = list(gc.callbacks)
    result = mod.run_episode(session, system, builder, camera, config, action_limit=4)
    assert result['physics_steps'] == 180
    assert result['owner_loop_diagnostics']['tick_count'] == 180
    assert result['owner_loop_diagnostics']['phase_totals']['physx']['count'] == 180
    assert result['realtime_valid'] and result['continuous_valid']
    assert gc.callbacks == callbacks


def test_real_step_stall_fails_same_bound_and_keeps_partial_tick(monkeypatch):
    mod, session, system, builder, camera, config, _, wall, _ = fixture(monkeypatch)
    original = session.env.scene.step
    def delayed_step():
        original()
        wall.value += .26
    session.env.scene.step = delayed_step
    callbacks = list(gc.callbacks)
    with pytest.raises(mod.TimingError, match='wall/physics backlog') as error:
        mod.run_episode(session, system, builder, camera, config, action_limit=4)
    diagnostics = error.value.dualrate_report['owner_loop_diagnostics']
    tick = diagnostics['recent_ticks'][-1]
    assert not tick['complete'] and tick['lag_after'] > .255
    assert tick['spans']['physx']['wall_seconds'] >= .26 - 1e-10
    assert session.env.scene.ticks == 1
    assert gc.callbacks == callbacks
