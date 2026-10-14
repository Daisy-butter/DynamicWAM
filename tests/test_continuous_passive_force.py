"""CPU equivalence using actual DOMINO setters and stateful articulation fakes."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from dynamicwam.runtime.domino_continuous.continuous_session import ContinuousSyncSession


def native_robot_methods():
    source = Path(__file__).resolve().parents[1] / "external/DOMINO/envs/robot/robot.py"
    tree = ast.parse(source.read_text())
    robot = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Robot")
    names = {"_entity_qf", "set_arm_joints", "set_gripper", "get_normal_real_gripper_val"}
    module = ast.Module(body=[node for node in robot.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


class Joint:
    def __init__(self):
        self.drive_target = np.zeros(1)
        self.velocity = np.zeros(1)
    def get_drive_target(self): return self.drive_target.copy()
    def set_drive_target(self, value): self.drive_target = np.atleast_1d(value).copy()
    def set_drive_velocity_target(self, value): self.velocity = np.atleast_1d(value).copy()


class Entity:
    def __init__(self):
        self.qpos, self.qvel = np.arange(6, dtype=float), np.ones(6)
        self.qf, self.computes = None, 0
    def compute_passive_force(self, **kwargs):
        assert kwargs == {"gravity": True, "coriolis_and_centrifugal": True}
        self.computes += 1
        return self.qpos + 2 * self.qvel
    def set_qf(self, value): self.qf = value.copy()


NativeRobot = type("NativeRobot", (), native_robot_methods())


class Robot(NativeRobot):
    def __init__(self):
        self.left_entity, self.right_entity = Entity(), Entity()
        self.left_arm_joints, self.right_arm_joints = [Joint() for _ in range(6)], [Joint() for _ in range(6)]
        self.left_gripper, self.right_gripper = [(Joint(), 1., 0.)], [(Joint(), -1., .2)]
        self.left_gripper_scale = self.right_gripper_scale = (0., .045)
        self.left_gripper_val = self.right_gripper_val = 0.
        self.calls, self.fail_on = [], None
    def set_arm_joints(self, position, velocity, arm):
        self.calls.append(("arm", arm, position.tolist(), velocity.tolist()))
        super().set_arm_joints(position, velocity, arm)
        if self.fail_on == arm: raise RuntimeError("setter failure")
    def set_gripper(self, value, arm):
        self.calls.append(("gripper", arm, float(value)))
        super().set_gripper(value, arm)


def session(robot):
    result = ContinuousSyncSession.__new__(ContinuousSyncSession)
    events = []
    def step():
        events.append("scene")
        for entity in (robot.left_entity, robot.right_entity):
            entity.qpos += .01
            entity.qvel += .02
    result.env = SimpleNamespace(robot=robot, scene=SimpleNamespace(step=step),
        _update_kinematic_tasks=lambda: events.append("kinematics"),
        _update_transient_checks=lambda: events.append("transient"))
    result._consume_events = lambda: events.append("events")
    return result, events


def native_step(result, positions, velocities):
    robot = result.env.robot
    robot.set_arm_joints(positions[:6], velocities[:6], "left")
    robot.set_arm_joints(positions[7:13], velocities[7:13], "right")
    robot.set_gripper(positions[6], "left")
    robot.set_gripper(positions[13], "right")
    result.env._update_kinematic_tasks()
    result.env.scene.step()
    result.env._update_transient_checks()
    result._consume_events()


class PassiveForceTests(unittest.TestCase):
    def test_actual_setter_outputs_order_and_next_tick_forces_match(self):
        reference, candidate = Robot(), Robot()
        original, original_events = session(reference)
        optimized, optimized_events = session(candidate)
        for index in range(3):
            positions = np.linspace(-.2, 1.2, 14) + index * .01
            velocities = np.linspace(-1., 1., 14)
            native_step(original, positions, velocities)
            optimized.step_command(positions, velocities)
            self.assertNotIn("_entity_qf", vars(candidate))
            self.assertEqual(reference.calls, candidate.calls)
            self.assertEqual(original_events, optimized_events)
            for side in ("left", "right"):
                actual_entity, expected_entity = getattr(candidate, side + "_entity"), getattr(reference, side + "_entity")
                np.testing.assert_array_equal(actual_entity.qf, expected_entity.qf)
                self.assertEqual(actual_entity.computes, index + 1)
                self.assertEqual(expected_entity.computes, 4 * (index + 1))
                joints_a = getattr(candidate, side + "_arm_joints") + [getattr(candidate, side + "_gripper")[0][0]]
                joints_b = getattr(reference, side + "_arm_joints") + [getattr(reference, side + "_gripper")[0][0]]
                for a, b in zip(joints_a, joints_b):
                    np.testing.assert_array_equal(a.drive_target, b.drive_target)
                    np.testing.assert_array_equal(a.velocity, b.velocity)
                self.assertEqual(getattr(candidate, side + "_gripper_val"), getattr(reference, side + "_gripper_val"))
        stats = optimized.runtime_timing_summary()
        self.assertEqual((stats["qf_requested"], stats["qf_computed"], stats["qf_reused"]), (24, 6, 18))
        self.assertEqual(stats["steps"], 3)
        self.assertEqual(set(stats["timing_ms"]), {"command", "kinematics", "scene", "transient", "events", "total"})

    def test_existing_override_restored_even_after_setter_exception(self):
        robot = Robot()
        previous = robot._entity_qf
        robot._entity_qf = previous
        robot.fail_on = "right"
        result, events = session(robot)
        with self.assertRaisesRegex(RuntimeError, "setter failure"):
            result.step_command(np.zeros(14), np.zeros(14))
        self.assertIs(robot._entity_qf, previous)
        self.assertEqual(events, [])
        self.assertEqual(result.runtime_timing_summary()["qf_computed"], 2)

    def test_no_instance_override_leaks_after_exception_or_hold(self):
        robot = Robot()
        robot.fail_on = "left"
        result, events = session(robot)
        with self.assertRaises(RuntimeError): result.step_command(np.zeros(14), np.zeros(14))
        self.assertNotIn("_entity_qf", vars(robot))
        before = robot.left_entity.computes
        result.step_command(None, None)
        self.assertNotIn("_entity_qf", vars(robot))
        self.assertEqual(robot.left_entity.computes, before + 1)


if __name__ == "__main__": unittest.main()
