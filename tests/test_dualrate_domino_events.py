import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dynamicwam.runtime.domino_rt.domino_session import (
    DominoLevel1Session,
    _strict_dynamic_boundary,
    _strict_gripper_contact,
    _strict_stop_dynamic_motion,
)


def make_env(position=(0.0, 0.0, 0.0), contacts=()):
    actor = SimpleNamespace(get_pose=lambda: SimpleNamespace(p=position))
    target = SimpleNamespace(actor=actor, get_name=lambda: "target")
    return SimpleNamespace(
        use_dynamic=True,
        eval_table_bounds=(-0.35, 0.35, -0.15, 0.25),
        _eval_target_actor=target,
        _dynamic_object_stopped=False,
        get_gripper_actor_contact_position=lambda name: list(contacts),
        active_kinematic_tasks=[],
    )


def fail(error):
    def call(*args, **kwargs):
        raise error
    return call


def test_native_boundary_margin_and_contact_predicate():
    env = make_env((0.35 + 0.02, -0.15 - 0.02, 10.0), contacts=[(1, 2, 3)])
    assert _strict_dynamic_boundary(env)  # inclusive 2 cm margin; z is not bounded
    assert _strict_gripper_contact(env)
    env._eval_target_actor.actor.get_pose = lambda: SimpleNamespace(p=(0.37001, 0, 0))
    assert not _strict_dynamic_boundary(env)
    env.get_gripper_actor_contact_position = lambda name: []
    assert not _strict_gripper_contact(env)


def test_expected_disabled_or_missing_actor_results():
    env = make_env()
    env.use_dynamic = False
    assert _strict_dynamic_boundary(env)
    assert not _strict_gripper_contact(env)
    assert not _strict_stop_dynamic_motion(env)
    env.use_dynamic = True
    env._eval_target_actor = None
    assert _strict_dynamic_boundary(env)
    assert not _strict_gripper_contact(env)
    assert not _strict_stop_dynamic_motion(env)


def test_pose_failure_propagates():
    env = make_env()
    error = RuntimeError("simulator get_pose failed")
    env._eval_target_actor.actor.get_pose = fail(error)
    with pytest.raises(RuntimeError) as caught:
        _strict_dynamic_boundary(env)
    assert caught.value is error


def test_contact_read_failure_propagates():
    env = make_env()
    error = RuntimeError("simulator contact read failed")
    env.get_gripper_actor_contact_position = fail(error)
    with pytest.raises(RuntimeError) as caught:
        _strict_gripper_contact(env)
    assert caught.value is error


class RigidComponent:
    def __init__(self, entity):
        self.entity = entity
        self.calls = []

    def set_kinematic(self, value):
        self.calls.append(("kinematic", value))

    def set_linear_velocity(self, value):
        self.calls.append(("linear_velocity", value.copy()))

    def set_angular_velocity(self, value):
        self.calls.append(("angular_velocity", value.copy()))

    def set_linear_damping(self, value):
        self.calls.append(("linear_damping", value))

    def set_angular_damping(self, value):
        self.calls.append(("angular_damping", value))


def stop_fixture(monkeypatch):
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace(
        physx=SimpleNamespace(PhysxRigidDynamicComponent=RigidComponent)))
    env = make_env()
    actor = env._eval_target_actor.actor
    component = RigidComponent(actor)
    actor.get_components = lambda: [component, object()]
    unrelated = {"component": SimpleNamespace(entity=object())}
    env.active_kinematic_tasks = [{"component": component}, unrelated]
    return env, component, unrelated


def test_native_stop_only_removes_target_and_zeros_velocity(monkeypatch):
    env, component, unrelated = stop_fixture(monkeypatch)
    assert _strict_stop_dynamic_motion(env)
    assert env.active_kinematic_tasks == [unrelated]
    assert env._dynamic_object_stopped
    assert [name for name, _ in component.calls] == [
        "kinematic", "linear_velocity", "angular_velocity", "linear_damping", "angular_damping"]
    assert component.calls[0][1] is False
    np.testing.assert_array_equal(component.calls[1][1], np.zeros(3))
    np.testing.assert_array_equal(component.calls[2][1], np.zeros(3))
    assert component.calls[3][1] == component.calls[4][1] == 100.0
    assert _strict_stop_dynamic_motion(env) and len(component.calls) == 5


@pytest.mark.parametrize("method", ["set_kinematic", "set_linear_velocity"])
def test_stop_mutation_failure_propagates_without_success(monkeypatch, method):
    env, component, _ = stop_fixture(monkeypatch)
    error = RuntimeError("simulator mutation failed")
    setattr(component, method, fail(error))
    with pytest.raises(RuntimeError) as caught:
        _strict_stop_dynamic_motion(env)
    assert caught.value is error and not env._dynamic_object_stopped


def test_session_event_path_uses_strict_helpers():
    env = make_env()
    env.check_success = lambda: False
    error = RuntimeError("actual contact read failed")
    env.get_gripper_actor_contact_position = fail(error)
    # A call to the legacy catch-and-continue helper must never occur.
    env.check_gripper_contact_dynamic_object = fail(AssertionError("legacy helper called"))
    session = DominoLevel1Session(
        env, flow=None, observation_config={}, stop_on_contact=True,
        check_z_threshold=None, lift_actor=None, start_z=None)
    with pytest.raises(RuntimeError) as caught:
        session._consume_events()
    assert caught.value is error
