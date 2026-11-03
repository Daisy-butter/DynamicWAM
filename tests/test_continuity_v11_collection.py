from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_domino_smoke import prepare_train_scene


def test_native_filter_cannot_leak_validation_seed():
    calls=[];closed=[]
    def prepare(seed):
        calls.append(seed)
        return SimpleNamespace(close_env=lambda:closed.append('close'),
            release_episode_resources=lambda:closed.append('release')),{'seed':seed+1}
    _,r=prepare_train_scene(prepare,[{'seed':2},{'seed':4},{'seed':5}])
    assert r['seed']==5 and calls==[2,4] and closed==['close','release']


def test_train_scene_propagates_runtime_and_cleanup_failures():
    def broken(seed):raise RuntimeError('device failure')
    with pytest.raises(RuntimeError,match='device failure'):
        prepare_train_scene(broken,[{'seed':2}])
    def fail():raise RuntimeError('cleanup failure')
    with pytest.raises(RuntimeError,match='cleanup failure'):
        prepare_train_scene(lambda s:(SimpleNamespace(close_env=fail),{'seed':3}),[{'seed':2}])


def test_train_scene_exhaustion_is_explicit():
    env=SimpleNamespace(close_env=lambda:None,release_episode_resources=lambda:None)
    with pytest.raises(ValueError,match='no accepted'):
        prepare_train_scene(lambda s:(env,{'seed':3}),[{'seed':2}])
    with pytest.raises(ValueError,match='training-only'):
        prepare_train_scene(None,[{'seed':100000}])
