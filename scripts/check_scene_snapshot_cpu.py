#!/usr/bin/env python3
"""CPU fake-scene tests of snapshot restore contracts, with no SAPIEN/GPU load.

These test validation/order/FK-oracle logic, not actual SAPIEN FK or image parity.
The real independent renderer still needs its separate loaded-scene test.
"""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('snapshot_cpu_test',
    ROOT/'src/dynamicwam/runtime/domino_rt/scene_snapshot.py')
snapshot=importlib.util.module_from_spec(spec);spec.loader.exec_module(snapshot)


class Pose:
    def __init__(self,p=(0,0,0),q=(1,0,0,0)):
        self.p=np.array(p,dtype=np.float32);self.q=np.array(q,dtype=np.float32)


class Entity:
    def __init__(self,number):
        self.per_scene_id=number;self.components=[];self.pose=Pose();self.mutations=0
    def set_pose(self,value):self.pose=value;self.mutations+=1


class Shape:
    def __init__(self):self.local_pose=Pose();self.scale=np.ones(3);self.front_face='counterclockwise'
    def set_local_pose(self,p):raise RuntimeError('attached RenderShape setter is unavailable')


class Body:
    def __init__(self,shape):
        self.render_shapes=[shape];self.is_enabled=True;self.visibility=1.;self.shading_mode=0
    def enable(self):self.is_enabled=True
    def disable(self):self.is_enabled=False


class Camera:
    def __init__(self,entity):
        self.entity=entity;self.local_pose=Pose();self.is_enabled=True;self.width=self.height=2
    @property
    def global_pose(self):return Pose(self.entity.pose.p+self.local_pose.p)
    def get_intrinsic_matrix(self):return np.eye(3)
    def get_projection_matrix(self):return np.eye(4)
    def set_local_pose(self,p):self.local_pose=p
    def enable(self):self.is_enabled=True
    def disable(self):self.is_enabled=False


class Articulation:
    def __init__(self,entity):self.entity=entity;self.root=Pose();self.qpos=np.zeros(1);self.mutations=0
    def _fk(self):self.entity.pose=Pose(self.root.p+np.array([self.qpos[0],0,0]))
    def get_root_pose(self):return self.root
    def get_qpos(self):return self.qpos
    def get_links(self):return [SimpleNamespace(entity=self.entity)]
    def set_root_pose(self,value):self.root=value;self.mutations+=1;self._fk()
    def set_qpos(self,value):self.qpos=np.array(value);self.mutations+=1;self._fk()


def bridge():
    # Constructor geometry fingerprints are checked by the real-scene test;
    # establish their exact post-construction invariants for restore unit tests.
    b=object.__new__(snapshot.SceneSnapshotBridge)
    entity=Entity(10);link=Entity(11);shape=Shape();body=Body(shape);camera=Camera(entity)
    entity.components=[body,camera]
    b.scene=SimpleNamespace(entities=[entity,link],render_system=SimpleNamespace(ambient_light=[.3,.3,.3]))
    b.env=SimpleNamespace(scene=b.scene,cameras=SimpleNamespace(collect_wrist_camera=False),
        _scene_step_clock=SimpleNamespace(snapshot=lambda:SimpleNamespace(time_seconds=.2)))
    b.entities=list(b.scene.entities);b._entity_ids=[10,11]
    b._component_refs=[tuple(x.components) for x in b.entities]
    b._component_signature=[tuple(type(c).__name__ for c in x.components) for x in b.entities]
    b._parts=[];b._static_materials=[];b.bodies=[([0,0],body,[shape])]
    b._static_shape_states=[[snapshot._shape_state(shape)]]
    b.cameras=[([0,1],camera)];b.lights=[]
    b._projection_identity=[b._camera_projection(camera)]
    b.articulations=[Articulation(link)];b.link_indices={1};b.nonlink_indices=[0]
    b.identity={'sha256':'identity'};b._sequence=0;b.pose_tolerance=1e-4
    return b


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fake=patch.dict(sys.modules,{'sapien':SimpleNamespace(Pose=Pose)})
        self.fake.start();self.addCleanup(self.fake.stop)

    def test_exact_restore_camera_and_fk_without_physics_steps(self):
        source=bridge();target=bridge()
        source.entities[0].pose=Pose([.2,.3,.4]);source.articulations[0].set_qpos([.7])
        source.cameras[0][1].local_pose=Pose([.1,0,0])
        packet=source.capture();result=target.apply(packet)
        self.assertTrue(result['passed']);self.assertEqual(result['physics_steps'],0)
        self.assertEqual(result['source_sim_time_seconds'],.2)
        self.assertLess(result['max_translation_error'],1e-7)
        np.testing.assert_allclose(target.cameras[0][1].global_pose.p,[.3,.3,.4],atol=1e-7)

    def test_identity_and_missing_members_rejected_before_mutation(self):
        for field in ('identity_sha256','entities','articulations','bodies','cameras'):
            target=bridge();packet=bridge().capture()
            packet[field]='wrong' if field=='identity_sha256' else []
            with self.assertRaises(ValueError):target.apply(packet)
            self.assertEqual(target.entities[0].mutations,0)
            self.assertEqual(target.articulations[0].mutations,0)

    def test_nan_rejected_before_any_restore(self):
        target=bridge();packet=bridge().capture();packet['articulations'][0]['qpos'][0]=float('nan')
        with self.assertRaises(ValueError):target.apply(packet)
        self.assertEqual(target.articulations[0].mutations,0)
        self.assertEqual(target.entities[0].mutations,0)

    def test_fk_or_camera_pose_oracle_cannot_silently_disagree(self):
        for field in ('link','camera'):
            target=bridge();packet=bridge().capture()
            expected=(packet['articulations'][0]['link_poses'][0] if field=='link'
                      else packet['cameras'][0]['global_pose'])
            expected[0]+=.01
            with self.assertRaisesRegex(ValueError,'pose oracle failed'):target.apply(packet)

    def test_same_count_component_replacement_rejected(self):
        target=bridge();target.entities[0].components[1]=Camera(target.entities[0])
        with self.assertRaisesRegex(snapshot.UnsupportedSnapshotScene,'component replaced'):target.capture()

    def test_same_count_shape_replacement_rejected(self):
        target=bridge();target.bodies[0][1].render_shapes[0]=Shape()
        with self.assertRaisesRegex(snapshot.UnsupportedSnapshotScene,'shape layout'):target.capture()

    def test_attached_shape_is_validated_without_calling_unavailable_setter(self):
        target=bridge();packet=bridge().capture()
        with self.assertRaisesRegex(RuntimeError,'setter is unavailable'):
            target.bodies[0][2][0].set_local_pose(Pose())
        self.assertTrue(target.apply(packet)['passed'])

    def test_changed_local_shape_values_rejected(self):
        for name in ('local_pose','scale','front_face'):
            target=bridge();shape=target.bodies[0][2][0]
            setattr(shape,name,{'local_pose':Pose([.1,0,0]),'scale':np.full(3,2.),'front_face':'clockwise'}[name])
            with self.assertRaises(snapshot.UnsupportedSnapshotScene):target.capture()

    def test_changed_packet_shape_values_rejected_before_restore(self):
        for name in ('local_pose','scale','front_face'):
            target=bridge();packet=bridge().capture()
            packet['bodies'][0]['shapes'][0][name]={'local_pose':[.1,0,0,1,0,0,0],
                'scale':[2.,2.,2.],'front_face':'clockwise'}[name]
            with self.assertRaises(snapshot.UnsupportedSnapshotScene):target.apply(packet)
            self.assertEqual(target.articulations[0].mutations,0)
            self.assertEqual(target.entities[0].mutations,0)

    def test_missing_source_entity_rejected(self):
        target=bridge();target.scene.entities.pop()
        with self.assertRaisesRegex(snapshot.UnsupportedSnapshotScene,'entities added, removed'):target.capture()


class ImageParityMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('replica_metric_cpu_test',ROOT/'scripts/check_scene_replica.py')
        cls.replica=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.replica)

    def frames(self,values):
        frames=[]
        for value in values:
            image=np.zeros((8,8,3),dtype=np.uint8);image[0,0,0]=value
            frames.append({view:image.copy() for view in self.replica.VIEWS})
        return frames

    def test_deterministic_source_requires_exact_replica(self):
        source=self.frames([0,0,0]);same=copy.deepcopy(source)
        result=self.replica.compare_repetitions(source,same)
        self.assertTrue(result['passed'])
        self.assertTrue(all(x['replica_bitwise_identical'] for x in result['views'].values()))
        same[1]['head_camera'][2,3,0]=1
        result=self.replica.compare_repetitions(source,same)
        self.assertFalse(result['passed']);self.assertFalse(result['views']['head_camera']['passed'])
        self.assertTrue(result['views']['left_camera']['passed'])

    def test_only_measured_source_repeatability_sets_envelope(self):
        source=self.frames([0,1,2]);within=self.frames([1,0,1])
        result=self.replica.compare_repetitions(source,within)
        self.assertTrue(result['passed'])
        self.assertEqual(result['views']['head_camera']['source_empirical_envelope']['maximum'],2)
        outside=self.frames([3,0,1])
        self.assertFalse(self.replica.compare_repetitions(source,outside)['passed'])

    def test_shapes_types_and_repeat_count_must_match(self):
        image=np.zeros((8,8,3),dtype=np.uint8)
        for bad in (np.zeros((8,7,3),dtype=np.uint8),image.astype(np.float32),image[:,:,0]):
            with self.assertRaises(ValueError):self.replica.image_errors(image,bad)
        with self.assertRaises(ValueError):self.replica.compare_repetitions(self.frames([0,0]),self.frames([0,0]))


if __name__=='__main__':unittest.main(verbosity=2)
