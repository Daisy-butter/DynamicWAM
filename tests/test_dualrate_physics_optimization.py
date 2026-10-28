"""Per-tick native-command equivalence, aliasing and restoration checks."""
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_physics_optimization import passive_force_scope


class Entity:
    def __init__(self): self.state=2.;self.force=None;self.evaluations=0
    def compute_passive_force(self,*,gravity,coriolis_and_centrifugal):
        assert gravity and coriolis_and_centrifugal
        self.evaluations+=1
        return self.state*1.25
    def set_qf(self,value): self.force=value


class Robot:
    def __init__(self,dual=True):
        self.left_entity=Entity();self.right_entity=self.left_entity if dual else Entity()
        self.commands=[]
    def _entity_qf(self,entity):
        entity.set_qf(entity.compute_passive_force(gravity=True,coriolis_and_centrifugal=True))
    def set_arm_joints(self,position,velocity,side):
        self._entity_qf(self.left_entity);self._entity_qf(self.right_entity)
        self.commands.append(('arm',tuple(position),tuple(velocity),side))
    def set_gripper(self,value,side):
        self._entity_qf(self.left_entity);self._entity_qf(self.right_entity)
        self.commands.append(('gripper',value,side))


def apply(robot,enabled):
    with passive_force_scope(robot,enabled=enabled) as counts:
        robot._entity_qf(robot.left_entity);robot._entity_qf(robot.right_entity)
        robot.set_arm_joints([.125,-2.25],[.03125,0.],'left')
        robot.set_arm_joints([1.5,-3.],[0.,-.0625],'right')
        robot.set_gripper(.73,'left');robot.set_gripper(.13,'right')
    return counts


class PassiveForceTests(unittest.TestCase):
    def test_same_articulation_ten_to_one_with_identical_native_commands(self):
        baseline=Robot();candidate=Robot()
        self.assertIsNone(apply(baseline,False))
        counts=apply(candidate,True)
        self.assertEqual(baseline.left_entity.evaluations,10)
        self.assertEqual(candidate.left_entity.evaluations,1)
        self.assertEqual(counts,dict(native_evaluations=1,reused_calls=9))
        self.assertEqual(candidate.commands,baseline.commands)
        self.assertEqual(candidate.left_entity.force,baseline.left_entity.force)
        self.assertNotIn('_entity_qf',vars(candidate))
        self.assertNotIn('_full_passive_force_scope',vars(candidate))

    def test_distinct_articulations_are_not_deduplicated_together(self):
        robot=Robot(dual=False)
        counts=apply(robot,True)
        self.assertEqual(counts,dict(native_evaluations=2,reused_calls=8))
        self.assertEqual(robot.left_entity.evaluations,1)
        self.assertEqual(robot.right_entity.evaluations,1)

    def test_fresh_tick_recomputes_changed_state(self):
        robot=Robot();apply(robot,True);old_force=robot.left_entity.force
        robot.left_entity.state=8.
        apply(robot,True)
        self.assertEqual(robot.left_entity.evaluations,2)
        self.assertNotEqual(robot.left_entity.force,old_force)
        self.assertEqual(robot.left_entity.force,10.)
        other=Robot();apply(other,True)
        self.assertEqual(other.left_entity.evaluations,1)

    def test_original_instance_binding_restored_after_setter_error(self):
        robot=Robot();original=lambda entity:Robot._entity_qf(robot,entity)
        robot._entity_qf=original
        with self.assertRaisesRegex(ValueError,'setter failed'):
            with passive_force_scope(robot,enabled=True):
                robot._entity_qf(robot.left_entity)
                raise ValueError('setter failed')
        self.assertIs(robot._entity_qf,original)
        self.assertNotIn('_full_passive_force_scope',vars(robot))

    def test_native_error_restores_class_descriptor_and_never_integrates(self):
        robot=Robot()
        def fail(**kwargs): raise ArithmeticError('native force failed')
        robot.left_entity.compute_passive_force=fail
        with self.assertRaisesRegex(ArithmeticError,'native force failed'):
            apply(robot,True)
        self.assertNotIn('_entity_qf',vars(robot))
        self.assertEqual(robot.commands,[])

    def test_partial_installation_is_rolled_back(self):
        class RestrictedRobot(Robot):
            def __setattr__(self,name,value):
                if name=='_full_passive_force_scope': raise TypeError('marker refused')
                object.__setattr__(self,name,value)
        robot=RestrictedRobot()
        with self.assertRaisesRegex(TypeError,'marker refused'):
            apply(robot,True)
        self.assertNotIn('_entity_qf',vars(robot))
        self.assertNotIn('_full_passive_force_scope',vars(robot))

    def test_nested_scope_rejected_without_corrupting_outer(self):
        robot=Robot()
        with passive_force_scope(robot,enabled=True) as counts:
            wrapper=robot._entity_qf
            with self.assertRaisesRegex(RuntimeError,'already active'):
                with passive_force_scope(robot,enabled=True): pass
            self.assertIs(robot._entity_qf,wrapper)
            robot._entity_qf(robot.left_entity)
        self.assertEqual(counts['native_evaluations'],1)
        self.assertNotIn('_entity_qf',vars(robot))

    def test_other_thread_cannot_touch_scene_force(self):
        robot=Robot();errors=[]
        with passive_force_scope(robot,enabled=True):
            def call():
                try: robot._entity_qf(robot.left_entity)
                except BaseException as exc: errors.append(exc)
            thread=threading.Thread(target=call);thread.start();thread.join()
        self.assertEqual(len(errors),1)
        self.assertIsInstance(errors[0],RuntimeError)
        self.assertEqual(robot.left_entity.evaluations,0)

    def test_external_binding_change_fails_and_restores_original(self):
        robot=Robot()
        with self.assertRaisesRegex(RuntimeError,'binding changed'):
            with passive_force_scope(robot,enabled=True): robot._entity_qf=lambda entity:None
        self.assertNotIn('_entity_qf',vars(robot))

    def test_boolean_configuration_only(self):
        for enabled in (1,0,'true',None):
            with self.assertRaisesRegex(ValueError,'boolean'):
                with passive_force_scope(Robot(),enabled=enabled): pass


if __name__=='__main__': unittest.main()
