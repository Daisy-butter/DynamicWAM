import numpy as np
import pytest
import torch

from dynamicwam.dualrate.producer import (
    unique_last_indices,select_frame,resample_actions,view_overlap,materialize_native_latents,
    _letterbox,
    file_stat_identity,verify_file_stat,
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


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_real_native_vae_wrapper_tensor_abi_keeps_future_groups(dtype):
    # Exercise the exact installed class's encode method and patchify, while
    # replacing only the large learned encoder. No weight load or CUDA work.
    from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE,patchify
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=[]
        def encode(self,videos,scale):
            assert isinstance(videos,torch.Tensor)
            assert videos.dim()==5 and videos.shape[1]==3
            assert videos.dtype==dtype and not torch.is_grad_enabled()
            assert self.training is False and scale==[0.,1.]
            patched=patchify(videos,patch_size=2)
            assert patched.shape[1]==12
            self.calls.append(tuple(videos.shape))
            b,_,t,h,w=videos.shape;n=1+(t-1)//4
            return torch.arange(n,dtype=dtype).view(1,1,n,1,1).expand(b,48,n,h//16,w//16).clone()
    vae=object.__new__(Wan2_2_VAE);vae.dtype=dtype;vae.device='cpu';vae.scale=[0.,1.];vae.model=Encoder()
    batch={'source_condition_frames':torch.zeros(2,3,5,384,320),'source_future_frames':torch.zeros(2,3,9,192,160)}
    out=materialize_native_latents(batch,vae,device='cpu')
    assert vae.model.calls==[(2,3,5,384,320),(2,3,9,192,160)]
    assert out['source_condition_latent'].shape==(2,48,2,24,20)
    assert out['source_future_latent'].shape==(2,48,2,12,10)
    assert (out['source_future_latent'][:,:,0]==1).all()
    assert (out['source_future_latent'][:,:,1]==2).all()
    assert out['source_future_latent'].is_contiguous()


def test_native_camera_resolution_has_explicit_padding_mask():
    image=np.ones((240,320,3),np.uint8)*255
    head,head_valid=_letterbox(image,(240,320),(240,320))
    wrist,wrist_valid=_letterbox(image,(240,320),(120,160))
    assert head.shape==wrist.shape==(3,240,320)
    assert int(head_valid.sum())==240*320
    assert int(wrist_valid.sum())==120*160
    assert not wrist[:,~wrist_valid].any()


def test_source_file_stat_rejects_replacement_and_is_not_claimed_hash(tmp_path):
    p=tmp_path/'raw.bin';p.write_bytes(b'original')
    identity=file_stat_identity(p)
    assert set(identity)=={'resolved_path','size','mtime_ns'}
    verify_file_stat(p,identity)
    p.write_bytes(b'new content')
    with pytest.raises(ValueError,match='stat identity changed'):verify_file_stat(p,identity)
    q=tmp_path/'different.bin';q.write_bytes(b'original')
    with pytest.raises(ValueError,match='stat identity changed'):verify_file_stat(q,identity)
