import numpy as np
import pytest
import torch

from dynamicwam.dualrate.producer import (
    unique_last_indices,select_frame,resample_actions,view_overlap,materialize_native_latents,
    _letterbox,
)


def test_duplicates_last_and_past_only():
    times=np.array([0.,.06,.06,.12])
    assert unique_last_indices(times).tolist()==[0,2,3]
    assert select_frame(times,.10,past_only=True,tolerance=.060001)==2
    assert select_frame(times,.10,past_only=False,tolerance=.030001)==3
    with pytest.raises(ValueError):select_frame(times,.10,past_only=True,tolerance=.03)


def test_action_arm_interpolation_and_gripper_event_hold():
    times=np.array([0.,.06,.12]);values=np.zeros((3,14),np.float32)
    values[1:]=1;values[2]=2
    result,valid,error=resample_actions(times,values,[.03,.06,.09,.2])
    assert valid.tolist()==[True,True,True,False]
    assert result[0,0]==.5 and result[0,6]==0
    assert result[1,6]==1 and result[2,6]==1
    assert np.allclose(error[:3],[.03,0,.03])


def test_no_large_gap_or_decreasing_interpolation():
    with pytest.raises(ValueError):unique_last_indices([0.,.2,.1])
    _,valid,_=resample_actions([0.,.2],np.zeros((2,14)),[.1])
    assert not valid.any()


def test_mixed_camera_cells_and_source_flow():
    overlap=view_overlap(6,5)
    assert overlap.shape==(30,4)
    assert torch.allclose(overlap.sum(-1),torch.ones(30))
    assert torch.equal(overlap[22],torch.tensor([0.,.5,.5,0.]))
    assert torch.equal(view_overlap(12,10,flow_weight=1.)[:,0],torch.ones(120))


def test_vae_codec_keeps_all_eight_future_frames():
    class VAE:
        def encode(self,items):
            return [torch.zeros(48,1+(x.shape[1]-1)//4,12,10) for x in items]
    batch={'source_condition_frames':torch.zeros(1,3,5,384,320),'source_future_frames':torch.zeros(1,3,9,192,160)}
    out=materialize_native_latents(batch,VAE())
    assert out['source_future_latent'].shape==(1,48,2,12,10)
    assert out['source_condition_latent'].shape[2]==2


def test_native_camera_resolution_has_explicit_padding_mask():
    image=np.ones((240,320,3),np.uint8)*255
    head,head_valid=_letterbox(image,(240,320),(240,320))
    wrist,wrist_valid=_letterbox(image,(240,320),(120,160))
    assert head.shape==wrist.shape==(3,240,320)
    assert int(head_valid.sum())==240*320
    assert int(wrist_valid.sum())==120*160
    assert not wrist[:,~wrist_valid].any()
