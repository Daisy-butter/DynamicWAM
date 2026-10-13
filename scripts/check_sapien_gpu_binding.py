#!/usr/bin/env python3
"""CPU contract checks; real SAPIEN renderer device creation is a separate gate."""
import importlib.util
from pathlib import Path
import types
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('gpu_binding',ROOT/'src/dynamicwam/runtime/domino_continuous/gpu_binding.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)


def fake_sapien(pci='0000:26:00.0',cuda_id=0,can_render=True):
    device=types.SimpleNamespace(pci_string=pci,cuda_id=cuda_id,is_cuda=True,can_render=can_render,name='test GPU')
    calls=[]
    class NativeRenderer:
        def __init__(self,selected=None):self.selected=selected;calls.append(('native_renderer',selected))
    class Renderer(NativeRenderer):
        def __init__(self,**kwargs):super().__init__()
        def create_material(self):return 'preserved material method'
    class Engine:
        def create_scene(self,config='default_scene_config'):return ('unbound',config)
    class FakeScene:
        def __init__(self,systems=None):self.systems=systems
    class RenderSystem:
        def __init__(self,selected=None):self.device=selected;calls.append(('render_system',selected))
    fake=types.SimpleNamespace(Device=lambda selector:device,
        pysapien=types.SimpleNamespace(render=types.SimpleNamespace(SapienRenderer=NativeRenderer)),
        render=types.SimpleNamespace(RenderSystem=RenderSystem),
        physx=types.SimpleNamespace(PhysxCpuSystem=lambda:'CPU_physics',set_scene_config=lambda config:calls.append(('scene_config',config))),
        Engine=Engine,SapienRenderer=Renderer,Scene=FakeScene)
    fake.core=types.SimpleNamespace(Engine=Engine,SapienRenderer=Renderer,Scene=FakeScene)
    return fake,device,calls


class BindingTests(unittest.TestCase):
    def setUp(self):G._BINDING=None
    def test_unconfigured_path_never_imports_sapien(self):
        with mock.patch.object(G.importlib,'import_module',side_effect=AssertionError('unexpected import')):
            self.assertIsNone(G.install_gpu_binding(environment={}))
    def test_explicit_device_binds_both_compat_renderer_and_scene(self):
        fake,device,calls=fake_sapien()
        identity=G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'00000000:26:00.0'},sapien_module=fake)
        self.assertEqual(identity['pci'],'0000:26:00.0')
        self.assertIs(fake.Engine,fake.core.Engine);self.assertIs(fake.SapienRenderer,fake.core.SapienRenderer)
        renderer=fake.core.SapienRenderer(offscreen_only=True)
        self.assertIs(renderer.selected,device);self.assertEqual(renderer.create_material(),'preserved material method')
        scene=fake.core.Engine().create_scene()
        self.assertEqual(scene.systems[0],'CPU_physics');self.assertIs(scene.systems[1].device,device)
        self.assertIn(('scene_config','default_scene_config'),calls)
        self.assertEqual(sum(item[0]=='render_system' for item in calls),1)
    def test_wrong_pci_fails_before_any_class_is_changed(self):
        fake,_,calls=fake_sapien();old=fake.Engine
        with self.assertRaisesRegex(RuntimeError,'does not match'):
            G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'0000:27:00.0'},sapien_module=fake)
        self.assertIs(fake.Engine,old);self.assertEqual(calls,[])
    def test_nonlocal_cuda_and_nonrenderable_are_rejected(self):
        for kwargs in ({'cuda_id':2},{'can_render':False}):
            fake,_,_=fake_sapien(**kwargs)
            with self.assertRaises(RuntimeError):G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'0000:26:00.0'},sapien_module=fake)
    def test_no_resident_rebinding_or_removal(self):
        fake,_,_=fake_sapien();env={'DYNAMICWAM_GPU_PCI':'0000:26:00.0'}
        first=G.install_gpu_binding(environment=env,sapien_module=fake)
        self.assertEqual(G.install_gpu_binding(environment=env,sapien_module=fake),first)
        with self.assertRaises(RuntimeError):G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'0000:27:00.0'},sapien_module=fake)
        with self.assertRaises(RuntimeError):G.install_gpu_binding(environment={},sapien_module=fake)
    def test_renderer_cannot_override_pinned_device(self):
        fake,_,_=fake_sapien();G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'0000:26:00.0'},sapien_module=fake)
        with self.assertRaises(ValueError):fake.SapienRenderer(device='cuda:1')
    def test_invalid_pci_is_rejected(self):
        for value in ('cuda:0','26:00.0','0000:26:00.9','../../device',''):
            with self.assertRaises(ValueError):G.normalize_pci(value)
    def test_unexpected_scene_default_api_is_rejected(self):
        fake,_,_=fake_sapien()
        fake.core.Engine.create_scene=lambda self:None
        with self.assertRaisesRegex(RuntimeError,'Engine.create_scene'):
            G.install_gpu_binding(environment={'DYNAMICWAM_GPU_PCI':'0000:26:00.0'},sapien_module=fake)


if __name__=='__main__':unittest.main()
