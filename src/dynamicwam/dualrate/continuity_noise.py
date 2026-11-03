"""Episode video noise and physical-interval action noise shared across replans.

Randomness stays Gaussian, but a new query does not randomly replace an already
predicted interval. This is identical in slow deployment and teacher training.
"""
import hashlib
import torch


def prediction_noise(system,batch,intervals):
    seeds=batch['feedback_noise_seed'];origins=batch['action_grid_origin']
    dt=system.residual_v10['action_dt'];like=batch['source_initial_state']
    indices=((intervals[...,0]-origins[:,None])/dt).round().to(torch.long)
    expected=origins[:,None]+indices.to(torch.float64)*dt
    if not torch.allclose(intervals[...,0],expected,atol=1e-7,rtol=0):
        raise ValueError('noise key must use the actual global physical action grid')
    videos=[];actions=[]
    for n,seed in enumerate(seeds.tolist()):
        vg=torch.Generator(device=like.device).manual_seed(int(seed))
        videos.append(system._noise((1,*system.future_latent_shape),batch['source_condition_latent'],vg))
        rows=[]
        for index in indices[n].tolist():
            key=int.from_bytes(hashlib.sha256(f'{seed}:action:{index}'.encode()).digest()[:8],'little')%(2**63-1)
            ag=torch.Generator(device='cpu').manual_seed(key)
            rows.append(torch.randn(14,generator=ag,dtype=torch.float32))
        actions.append(torch.stack(rows).to(like.device,dtype=like.dtype)[None])
    return torch.cat(videos),torch.cat(actions)
