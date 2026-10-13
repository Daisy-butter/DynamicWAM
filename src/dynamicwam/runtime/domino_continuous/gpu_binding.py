"""Explicit process-local SAPIEN 3 device binding for continuous DOMINO.

The installed SAPIEN compatibility renderer silently ignores keyword arguments.
CUDA visibility also does not restrict Vulkan enumeration. When the supervisor
pins DYNAMICWAM_GPU_PCI, replace only compatibility classes in this process and
give both the renderer and scene RenderSystem the verified CUDA-local device.
External DOMINO/SAPIEN files are never edited. Without the variable this module
does not import SAPIEN and leaves the original path untouched.
"""
from __future__ import annotations

import importlib
import os
import re
import sys

_BINDING=None


def normalize_pci(value):
    match=re.fullmatch(r'([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',str(value).strip())
    if match is None:raise ValueError('GPU PCI must be domain:bus:device.function')
    domain,bus,device,function=(int(part,16) for part in match.groups())
    return f'{domain:04x}:{bus:02x}:{device:02x}.{function:x}'


def _device_identity(device, expected):
    actual=normalize_pci(device.pci_string)
    if actual!=expected or int(device.cuda_id)!=0 or not device.is_cuda or not device.can_render:
        raise RuntimeError(f'SAPIEN CUDA-local device does not match selected GPU: expected={expected}, '
                           f'actual={actual}, cuda_id={device.cuda_id}, can_render={device.can_render}')
    return dict(pci=actual,cuda_id=int(device.cuda_id),name=str(device.name),
                can_render=bool(device.can_render),selector='cuda:0')


def install_gpu_binding(*, environment=None, sapien_module=None):
    global _BINDING
    environment=os.environ if environment is None else environment
    expected_value=environment.get('DYNAMICWAM_GPU_PCI')
    if not expected_value:
        if _BINDING is not None:
            raise RuntimeError('cannot remove a GPU binding inside a resident process')
        return None
    expected=normalize_pci(expected_value)
    if _BINDING is not None:
        if _BINDING['pci']!=expected:
            raise RuntimeError('cannot change GPU binding inside a resident process')
        return dict(_BINDING)
    sapien=importlib.import_module('sapien') if sapien_module is None else sapien_module
    core=importlib.import_module('sapien.core') if sapien_module is None else sapien.core
    original_engine=core.Engine;original_renderer=core.SapienRenderer
    native_renderer=sapien.pysapien.render.SapienRenderer
    if native_renderer not in original_renderer.__mro__[1:]:
        raise RuntimeError('unsupported SAPIEN compatibility renderer; explicit binding refused')
    defaults=getattr(original_engine.create_scene,'__defaults__',None)
    if defaults is None or len(defaults)!=1:
        raise RuntimeError('unsupported SAPIEN Engine.create_scene compatibility API')
    from dynamicwam.runtime.domino_continuous.sapien_init_lock import SapienInitLock
    with SapienInitLock(device_context=True):
        device=sapien.Device('cuda:0')
    identity=_device_identity(device,expected)
    scene_type=original_engine.create_scene.__globals__.get('Scene',core.Scene)

    class BoundRenderer(original_renderer):
        def __init__(self,**kwargs):
            if kwargs.get('device') is not None and kwargs['device'] is not device:
                raise ValueError('renderer device override conflicts with the pinned process device')
            # The wrapper's super().__init__() discards its kwargs. Invoke its
            # actual pybind base with the verified device instead.
            with SapienInitLock(device_context=True):
                native_renderer.__init__(self,device)

    class BoundEngine(original_engine):
        def create_scene(self,config=defaults[0]):
            sapien.physx.set_scene_config(config)
            with SapienInitLock(device_context=True):
                render_system=sapien.render.RenderSystem(device)
            _device_identity(render_system.device,expected)
            return scene_type([sapien.physx.PhysxCpuSystem(),render_system])

    BoundEngine.__name__='PinnedGpuEngine';BoundRenderer.__name__='PinnedGpuSapienRenderer'
    for module in (core,sapien,sys.modules.get(original_engine.__module__),
                   sys.modules.get(original_renderer.__module__)):
        if module is None:continue
        if getattr(module,'Engine',None) is original_engine:module.Engine=BoundEngine
        if getattr(module,'SapienRenderer',None) is original_renderer:module.SapienRenderer=BoundRenderer
    _BINDING=dict(identity,enabled=True,mechanism='process_local_explicit_renderer_and_render_system')
    return dict(_BINDING)
