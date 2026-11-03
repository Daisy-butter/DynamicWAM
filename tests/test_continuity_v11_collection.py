from types import SimpleNamespace
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dualrate_domino_smoke import prepare_train_scene


def test_pair_identity_includes_commitments_and_causal_publication():
    from dynamicwam.dualrate.continuity_data import reference_identity
    a=dict(trace_number=0,slow_supervision_version=2,source_available=.1,
           request_time=.1,publish_time=.5,source_prefix_end=.36,query_time=.7)
    assert reference_identity(a)==reference_identity(dict(a,query_time=.9))
    for key in ('trace_number','slow_supervision_version','request_time','publish_time','source_prefix_end'):
        assert reference_identity(a)!=reference_identity(dict(a,**{key:a[key]+1}))
    with pytest.raises(ValueError,match='identity missing'):reference_identity({})


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


def test_only_native_missing_grasp_pose_is_a_scene_rejection():
    ns={}
    exec(compile("def __init__():\n raise AssertionError('target_pose cannot be None for move action.')",'/fixture/envs/utils/action.py','exec'),ns)
    exec(compile('def grasp_actor():\n __init__()', '/fixture/envs/_base_task.py','exec'),ns)
    calls=[]
    def prepare(seed):
        calls.append(seed)
        if seed==2:ns['grasp_actor']()
        return object(),{'seed':seed}
    _,r=prepare_train_scene(prepare,[{'seed':2},{'seed':4}])
    assert calls==[2,4] and r['seed']==4
    def unrelated(seed):raise AssertionError('target_pose cannot be None for move action.')
    with pytest.raises(AssertionError):prepare_train_scene(unrelated,[{'seed':2}])
