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


def test_frozen_source_tail_keeps_real_prefix_without_full_future_labels():
    import numpy as np
    from prepare_dualrate_exposure_data import eligible_rows
    from dynamicwam.dualrate.exposure_timing import synthetic_templates,validate_template
    dt=45*float(np.float32(.004));times=np.arange(101)*dt/3
    templates=synthetic_templates(dt)['templates']
    args=(times,templates,dt,2*dt,'task/episode',4)
    old=list(eligible_rows(*args,partial_fast_tail=True))
    new=list(eligible_rows(*args,partial_fast_tail=True,partial_source_tail=True))
    assert set(old)<=set(new) and max(i for i,_ in new)>max(i for i,_ in old)
    tail=[(i,t) for i,t in new if times[i]+16*dt>times[-1]+1e-6]
    assert tail
    for i,t in tail:
        e=validate_template(templates[t],dt)
        assert times[i]+e['source_prefix_end']<=times[-1]+1e-6
        assert times[i]+e['fast_prefix_end']+2*dt<=times[-1]+1e-6


def test_frozen_future_latent_placeholder_has_native_shape_and_no_labels():
    import torch
    from dynamicwam.dualrate.producer import materialize_native_latents
    class Codec:
        dtype=torch.float32
        def encode(self,x):
            assert x.shape[2]==5  # Frozen future video must never be encoded.
            return torch.zeros(x.shape[0],48,2,x.shape[-2]//16,x.shape[-1]//16)
    b=dict(source_condition_frames=torch.zeros(1,3,5,384,320),
           source_future_frames=torch.zeros(1,3,9,192,160),
           feedback_noise_seed=torch.tensor([1]),native_video_valid=torch.tensor([True]))
    r=materialize_native_latents(b,Codec())
    assert r['source_future_latent'].shape==(1,48,2,12,10)
    assert not r['source_video_loss_valid'].any()
