"""CPU-only SAPIEN API parity and immutable worker configuration checks."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from dynamicwam.runtime.domino_rt import gpu_binding as binding
from dynamicwam.runtime.domino_rt import sapien_init_lock
import full_baseline_worker as worker


def native_config():
    # Exercise every native setting with deliberately nondefault values.
    return SimpleNamespace(gravity=[1.25,-2.5,7.5],bounce_threshold=.125,
        enable_pcm=False,enable_tgs=False,enable_ccd=True,
        enable_enhanced_determinism=True,enable_friction_every_iteration=False)


class FakePhysx:
    def __init__(self):
        self.config=native_config();self.workers=0;self.calls=[];self.systems=[]

    def set_scene_config(self,*args,**kwargs):
        self.calls.append((args,copy.deepcopy(kwargs)))
        if args:
            assert len(args)==1 and not kwargs
            self.config=copy.deepcopy(args[0]);self.workers=getattr(args[0],'_cpu_workers',0)
        else:
            assert set(kwargs)==set(binding._PHYSX_SCENE_FIELDS)|{'cpu_workers'}
            self.workers=kwargs.pop('cpu_workers');self.config=SimpleNamespace(**kwargs)

    def get_scene_config(self):
        config=copy.deepcopy(self.config);config._cpu_workers=self.workers
        return config

    def PhysxCpuSystem(self):
        config=self.get_scene_config()
        system=SimpleNamespace(cpu_workers=self.workers,get_config=lambda:config)
        self.systems.append(system)
        return system


class PhysxBindingTests(unittest.TestCase):
    def setUp(self):
        self.previous=binding._BINDING;binding._BINDING=None

    def tearDown(self):
        binding._BINDING=self.previous

    def fake_sapien(self):
        physics=FakePhysx()
        class NativeRenderer:
            def __init__(self,device): self.device=device
        class CompatibilityRenderer(NativeRenderer):
            pass
        class FakeScene:
            def __init__(self,systems): self.systems=systems
        class FakeEngine:
            def create_scene(self,config=native_config()): raise AssertionError('unbound engine called')
        device=SimpleNamespace(pci_string='0000:1b:00.0',cuda_id=0,is_cuda=True,
                               can_render=True,name='CPU fixture')
        return SimpleNamespace(core=SimpleNamespace(Engine=FakeEngine,SapienRenderer=CompatibilityRenderer,Scene=FakeScene),
            pysapien=SimpleNamespace(render=SimpleNamespace(SapienRenderer=NativeRenderer)),
            physx=physics,Device=lambda selector:device,
            render=SimpleNamespace(RenderSystem=lambda device:SimpleNamespace(device=device)))

    def test_keyword_worker_count_preserves_every_native_setting(self):
        for count in (1,2,4):
            physics=FakePhysx();original=native_config();before=copy.deepcopy(vars(original))
            result=binding.configure_physx_scene(physics,original,count)
            self.assertEqual(result['native_scene_fields'],before)
            self.assertEqual(vars(original),before)
            self.assertEqual(physics.workers,count)
            self.assertEqual(binding.physx_scene_fields(physics.get_scene_config()),before)
            self.assertIs(physics.calls[0][0][0],original)
            self.assertEqual(len(physics.calls),2)

    def test_default_zero_retains_only_native_object_setter(self):
        physics=FakePhysx();original=native_config()
        result=binding.configure_physx_scene(physics,original,0)
        self.assertEqual(result['cpu_workers'],0)
        self.assertEqual(len(physics.calls),1)
        self.assertIs(physics.calls[0][0][0],original)

    def test_changed_native_field_is_rejected_before_system_creation(self):
        physics=FakePhysx()
        actual=physics.set_scene_config
        def corrupt(*args,**kwargs):
            actual(*args,**kwargs)
            if kwargs: physics.config.enable_tgs=True
        physics.set_scene_config=corrupt
        with self.assertRaisesRegex(RuntimeError,'native scene fields'):
            binding.configure_physx_scene(physics,native_config(),2)
        self.assertEqual(physics.systems,[])

    def test_invalid_count_and_unpinned_nonzero_fail_closed(self):
        for value in (-1,5,True,2.,'2',None):
            physics=FakePhysx()
            with self.assertRaises(ValueError): binding.configure_physx_scene(physics,native_config(),value)
            self.assertEqual(physics.calls,[])
        with self.assertRaisesRegex(ValueError,'pinned'):
            binding.install_gpu_binding(environment={},physx_cpu_workers=2)
        self.assertIsNone(binding.install_gpu_binding(environment={}))

    def test_bound_engine_applies_count_after_each_native_config_override(self):
        sapien=self.fake_sapien();env={'DYNAMICWAM_GPU_PCI':'0000:1b:00.0'}
        with patch.object(sapien_init_lock,'SapienInitLock',side_effect=lambda **kw:nullcontext()):
            installed=binding.install_gpu_binding(environment=env,sapien_module=sapien,physx_cpu_workers=2)
            first=native_config();second=native_config();second.gravity=[0.,0.,-3.]
            for native in (first,second):
                scene=sapien.core.Engine().create_scene(native)
                self.assertEqual(scene.systems[0].cpu_workers,2)
                self.assertEqual(scene._dynamicwam_physx_binding['native_scene_fields'],vars(native))
                self.assertTrue(scene._dynamicwam_physx_binding['native_scene_fields_verified'])
            self.assertEqual(installed['physx_cpu_workers'],2)
            self.assertEqual(binding.install_gpu_binding(environment=env,sapien_module=sapien,physx_cpu_workers=2),installed)
            with self.assertRaisesRegex(RuntimeError,'cannot change'):
                binding.install_gpu_binding(environment=env,sapien_module=sapien,physx_cpu_workers=4)



class WorkerPhysicsConfigTests(unittest.TestCase):
    def fixture(self,root):
        storage=root/'wuruihan';storage.mkdir()
        checkpoint=storage/'full.pt';checkpoint.write_bytes(b'CPU fixture')
        profile=storage/'native.yaml';profile.write_text('fixture: true')
        return dict(run_kind=worker.MODE,stage='smoke',tasks=['task_'+str(i) for i in range(8)],
            gpu_uuids=['GPU-'+str(i) for i in range(8)],episodes=2,action_dt=.18000000854954124,
            frame_period=.18,max_backlog=.25,max_observation_age=.24,profile=str(profile),
            profile_sha256=worker.sha(profile),checkpoint=str(checkpoint),checkpoint_sha256=worker.sha(checkpoint),
            init_checkpoint_sha256=worker.sha(checkpoint),storage_root=str(storage),output=str(storage/'out'))

    def test_default_and_expanded_smoke_with_explicit_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            config=self.fixture(Path(temp))
            worker.validate_config(config)
            config.update(acceptance_smoke_episodes_per_task=10,episodes=10,physx_cpu_workers=2)
            worker.validate_config(config)
            for value in (-1,5,True,2.):
                with self.assertRaisesRegex(ValueError,'physx_cpu_workers'):
                    worker.validate_config(dict(config,physx_cpu_workers=value))
            with self.assertRaisesRegex(ValueError,'configured complete'):
                worker.validate_config(dict(config,episodes=2))


if __name__=='__main__': unittest.main()
