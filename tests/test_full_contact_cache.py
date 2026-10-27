import threading

import pytest

from dynamicwam.dualrate.contact_cache import StepContactCache


class Scene:
    def __init__(self):
        self.tick=0
        self.reads=0
        self.value=object()
        self.failure=None

    def physics_time(self):
        return self.tick*.004000000189989805

    def get_contacts(self):
        self.reads+=1
        if self.failure is not None:
            raise self.failure
        return [self.value]


def test_one_native_read_per_actual_step_with_independent_return_lists():
    scene=Scene();cache=StepContactCache(scene,scene.physics_time).install()
    first=scene.get_contacts();first.append('caller-private');first.pop(0)
    assert scene.get_contacts()==[scene.value] and scene.reads==1
    scene.tick+=1
    assert scene.get_contacts()==[scene.value] and scene.reads==2
    scene.get_contacts()
    assert cache.statistics()['cache_hits']==2
    cache.close()
    assert 'get_contacts' not in vars(scene) and not hasattr(scene,'_full_step_contact_cache')
    assert scene.get_contacts()==[scene.value] and scene.reads==3


def test_no_cache_reuse_across_episode_or_scene_and_restore_instance_override():
    scene=Scene();original=lambda:[scene.value]
    scene.get_contacts=original
    first=StepContactCache(scene,scene.physics_time).install()
    assert scene.get_contacts()==[scene.value]
    first.close()
    assert scene.get_contacts is original
    scene.value=object()
    second=StepContactCache(scene,scene.physics_time).install()
    assert scene.get_contacts()==[scene.value] and second.native_reads==1
    assert first.contacts is None and first.key is None
    second.close()
    other=Scene();third=StepContactCache(other,other.physics_time).install()
    assert other.get_contacts()==[other.value] and third.native_reads==1
    third.close()


def test_native_failure_never_returns_old_step_contacts_or_retries_same_step():
    scene=Scene();cache=StepContactCache(scene,scene.physics_time).install()
    scene.get_contacts();scene.tick+=1
    error=RuntimeError('native contacts failed');scene.failure=error
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            scene.get_contacts()
        assert caught.value is error
    assert scene.reads==2 and cache.failed_reads==1 and cache.contacts is None
    scene.tick+=1;scene.failure=None
    with pytest.raises(RuntimeError) as caught:
        scene.get_contacts()
    assert caught.value is error and scene.reads==2
    cache.close()
    assert scene.get_contacts.__self__ is scene


def test_owner_only_and_no_nested_cache_or_clock_rewind():
    scene=Scene();cache=StepContactCache(scene,scene.physics_time).install()
    with pytest.raises(RuntimeError,match='already installed'):
        StepContactCache(scene,scene.physics_time).install()
    errors=[]
    def read_other_thread():
        try:scene.get_contacts()
        except RuntimeError as exc:errors.append(str(exc))
    thread=threading.Thread(target=read_other_thread);thread.start();thread.join()
    assert errors==['contact reads require the active scene owner'] and scene.reads==0
    scene.tick=2;scene.get_contacts();scene.tick=1
    with pytest.raises(RuntimeError,match='monotonic real physics clock'):
        scene.get_contacts()
    cache.close()


def test_read_that_advances_scene_clock_fails_closed():
    scene=Scene()
    def bad_reader():
        scene.tick+=1
        return [scene.value]
    scene.get_contacts=bad_reader
    cache=StepContactCache(scene,scene.physics_time).install()
    with pytest.raises(RuntimeError,match='scene advanced'):
        scene.get_contacts()
    assert cache.contacts is None and cache.failed_reads==1
    cache.close()


def test_install_failure_rolls_back_reader_before_returning():
    class RestrictedScene(Scene):
        def __setattr__(self,name,value):
            if name=='_full_step_contact_cache':
                raise RuntimeError('marker rejected')
            super().__setattr__(name,value)
    scene=RestrictedScene();cache=StepContactCache(scene,scene.physics_time)
    with pytest.raises(RuntimeError,match='marker rejected'):
        cache.install()
    assert not cache.installed and 'get_contacts' not in vars(scene)
    assert not hasattr(scene,'_full_step_contact_cache')
    assert scene.get_contacts()==[scene.value]

