from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch

from dynamicwam.dualrate.online import (
    ObservationHistory, ObservationPairCache, OnlineFrame, history_features)


DT=45*float(np.float32(.004))


def flow_contract():
    return dict(compute_size=[64,64],normalization_percentile=99.,
        farneback=dict(flags=0,iterations=3,levels=3,poly_n=5,poly_sigma=1.2,pyr_scale=.5,winsize=15),
        quality=dict(method='forward_backward_consistency_v1',minimum_reliable_fraction=.1,
            relative_error=.01,absolute_error_squared=.5))


def history(seed=42,count=29):
    image=np.random.default_rng(seed).integers(0,256,(64,64,3),dtype=np.uint8)
    result=ObservationHistory()
    for i in range(count):
        rgb=np.roll(image,i,axis=1)
        result.append(dict(observation={c:dict(rgb=rgb) for c in
            ('head_camera','left_camera','right_camera')},joint_action=dict(vector=np.zeros(14,np.float32))),
            exposure=i*DT,available=i*DT+.1)
    return result


def assert_identical(a,b):
    assert a.keys()==b.keys()
    for key in a:
        if isinstance(a[key],torch.Tensor):assert torch.equal(a[key],b[key]),key
        else:assert np.array_equal(a[key],b[key]),key


def test_real_farneback_cache_is_exact_and_reuses_overlapping_windows(monkeypatch):
    import dynamicwam.absolute_motion as motion
    h=history();contract=flow_contract();cache=ObservationPairCache()
    frames=h.snapshot(24*DT+.1)
    truth=history_features(frames,contract,action_dt=DT)
    assert truth['motion_valid'].all()
    count=[0];original=motion.compute_flow_observation
    def counted(*args,**kwargs):
        count[0]+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(motion,'compute_flow_observation',counted)
    first=history_features(frames,contract,action_dt=DT,pair_cache=cache)
    assert_identical(first,truth);assert count[0]==5
    second=history_features(frames,contract,action_dt=DT,pair_cache=cache)
    assert_identical(second,truth);assert count[0]==5
    # Returned values cannot corrupt the memoized RGB, motion, or spatial pair.
    first['flow_rgb'].fill(0);first['spatial_flow'].fill_(123)
    assert_identical(history_features(frames,contract,action_dt=DT,pair_cache=cache),truth)
    later=h.snapshot(28*DT+.1)
    cached=history_features(later,contract,action_dt=DT,pair_cache=cache)
    assert count[0]==6  # four preceding native pairs are identical frame objects
    assert_identical(cached,history_features(later,contract,action_dt=DT))


def test_contract_and_frame_identity_invalidate_not_just_frame_number(monkeypatch):
    import dynamicwam.absolute_motion as motion
    frames=history(count=25).snapshot(24*DT+.1);contract=flow_contract();cache=ObservationPairCache()
    count=[0];original=motion.compute_flow_observation
    def counted(*args,**kwargs):
        count[0]+=1
        return original(*args,**kwargs)
    monkeypatch.setattr(motion,'compute_flow_observation',counted)
    history_features(frames,contract,action_dt=DT,pair_cache=cache)
    contract['farneback']['winsize']=17
    changed=history_features(frames,contract,action_dt=DT,pair_cache=cache)
    assert count[0]==10
    assert_identical(changed,history_features(frames,contract,action_dt=DT))
    other=history(seed=43,count=25).snapshot(24*DT+.1)
    before=count[0]
    new=history_features(other,contract,action_dt=DT,pair_cache=cache)
    assert count[0]==before+5
    assert_identical(new,history_features(other,contract,action_dt=DT))


def test_history_pixels_and_drive_target_are_strongly_immutable():
    frame=history(count=1).snapshot(.1)[0]
    for value in (*frame.images,frame.drive_target):
        assert not value.flags.writeable
        with pytest.raises(ValueError):value.setflags(write=True)
    assert history(count=1).snapshot(.1)[0] is not frame


def test_mutable_frames_bypass_cache_and_capacity_is_bounded():
    cache=ObservationPairCache(max_entries=2);calls=[]
    image=np.zeros((2,2,3),np.uint8)
    a=OnlineFrame((image,)*3,np.zeros(14),0.,0.,0)
    b=OnlineFrame((image,)*3,np.zeros(14),.18,.18,1)
    def compute():
        calls.append(1);return image.copy()
    cache.compute('native',a,b,'contract',compute)
    image[:]=7
    assert (cache.compute('native',a,b,'contract',compute)==7).all()
    assert len(calls)==2 and len(cache)==0
    frames=tuple(history(count=4).frames)
    for a,b in zip(frames,frames[1:]):
        cache.compute('native',a,b,'contract',lambda:np.ones(1))
    assert len(cache)==2
    cache.compute('native',frames[0],frames[1],'contract',compute)
    assert len(calls)==3 and len(cache)==2


def test_fast_does_not_wait_for_same_pair_being_computed_by_slow():
    frames=tuple(history(count=2).frames);cache=ObservationPairCache()
    entered=Event();release=Event();fast_done=Event()
    expected=np.arange(4,dtype=np.float32)
    def slow():
        entered.set()
        assert release.wait(3)
        return expected.copy()
    def fast():
        fast_done.set();return expected.copy()
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_job=pool.submit(cache.compute,'native',*frames,'contract',slow)
        try:
            assert entered.wait(1)
            fast_job=pool.submit(cache.compute,'native',*frames,'contract',fast)
            assert np.array_equal(fast_job.result(timeout=1),expected)
            assert fast_done.is_set() and not slow_job.done()
        finally:release.set()
        assert np.array_equal(slow_job.result(timeout=1),expected)
    assert len(cache)==1
    assert np.array_equal(cache.compute('native',*frames,'contract',lambda:pytest.fail('cache miss')),expected)


def test_slow_skips_unused_spatial_work_without_changing_its_inputs(monkeypatch):
    import dynamicwam.absolute_motion as motion
    frames=history(count=25).snapshot(24*DT+.1);contract=flow_contract()
    complete=history_features(frames,contract,action_dt=DT)
    calls=[0];original=motion.compute_dense_flow
    def counted(*args,**kwargs):
        calls[0]+=1;return original(*args,**kwargs)
    monkeypatch.setattr(motion,'compute_dense_flow',counted)
    source=history_features(frames,contract,action_dt=DT,include_spatial=False)
    assert calls[0]==10  # five native forward/backward pairs; no unused extra pair
    for key in ('flow_rgb','motion','motion_valid','motion_acceleration_valid','motion_times'):
        assert_identical({key:source[key]},{key:complete[key]})
