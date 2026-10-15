"""Strict real-trajectory replay adapter; no synthetic expert/policy mixing.

A replay producer must export timestamped observations from the policy's own
executed trajectory and independently supplied suffix labels on that same
trajectory. An expert demonstration with only its prefix replaced is rejected.
This module does not claim that such rollouts currently exist.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from dynamicwam.training.data.physical_grid import sha256_file

def _read_bound_tensor(root, reference):
    path=(root/reference['path']).resolve()
    if root.resolve() not in path.parents: raise ValueError('replay artifact escapes replay root')
    if sha256_file(path)!=reference['sha256']: raise ValueError('replay artifact hash mismatch')
    return torch.load(path,map_location='cpu',weights_only=True)

def load_real_rollout_replay(manifest_path):
    """Return real source/current inputs, committed prefix and independent labels.

    The returned label namespace must remain supervision-only. It is never a
    replacement for observations under counterfactual model actions.
    """
    path=Path(manifest_path);root=path.parent;m=json.loads(path.read_text())
    required=('trajectory_id','episode_id','policy_checkpoint_sha256','policy_source_sha256','source_observation',
        'current_observation','executed_timeline','suffix_labels','physical_grid_identity','observation_source')
    if any(k not in m for k in required):raise ValueError('incomplete real rollout provenance')
    if m['observation_source']!='policy_executed_same_trajectory':
        raise ValueError('expert/counterfactual observations cannot supervise a policy-prefix replay')
    for key in ('policy_checkpoint_sha256','policy_source_sha256','physical_grid_identity'):
        if len(m[key])!=64 or any(c not in '0123456789abcdef' for c in m[key]):raise ValueError('invalid provenance hash')
    source=_read_bound_tensor(root,m['source_observation']);current=_read_bound_tensor(root,m['current_observation'])
    executed=_read_bound_tensor(root,m['executed_timeline']);labels=_read_bound_tensor(root,m['suffix_labels'])
    for artifact in (source,current,executed,labels):
        if artifact.get('trajectory_id')!=m['trajectory_id'] or artifact.get('episode_id')!=m['episode_id']:
            raise ValueError('replay inputs/actions/labels are not the same actual trajectory')
    for obs in (source,current):
        if obs.get('origin')!='policy_execution_capture' or obs.get('policy_checkpoint_sha256')!=m['policy_checkpoint_sha256']:
            raise ValueError('observation is not from the recorded policy execution')
        for key in ('capture_time','rgb','state','native_motion','motion_interval_valid_mask','motion_acceleration_valid_mask'):
            if key not in obs:raise ValueError('actual replay observation is incomplete')
        if not torch.isfinite(torch.as_tensor(obs['state'])).all():raise ValueError('replay state nonfinite')
    st,ct=float(source['capture_time']),float(current['capture_time'])
    if not np.isfinite([st,ct]).all() or ct<st:raise ValueError('noncausal replay observations')
    if executed.get('policy_checkpoint_sha256')!=m['policy_checkpoint_sha256'] or executed.get('semantics')!='executed_drive_target_queue':
        raise ValueError('prefix is not the actual committed policy queue')
    times=np.asarray(executed['endpoint_times'],dtype=np.float64);actions=torch.as_tensor(executed['actions'])
    committed=np.asarray(executed['committed_before_query'],dtype=bool)
    if actions.shape!=(len(times),14) or len(times)==0 or committed.shape!=times.shape or not torch.isfinite(actions).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid executed action timeline')
    chosen=(times>ct+1e-9)&committed;positions=np.flatnonzero(chosen)
    if len(positions)>4 or (len(positions) and not np.array_equal(positions,np.arange(positions[0],positions[-1]+1))):
        raise ValueError('committed prefix is not contiguous or exceeds trained support')
    if labels.get('origin') not in ('expert_relabel_of_policy_trajectory','verified_same_trajectory_success_rollout'):
        raise ValueError('independent same-trajectory suffix supervision is required')
    if labels.get('observation_sha256')!=m['current_observation']['sha256']:
        raise ValueError('suffix labels correspond to different observations')
    target_times=np.asarray(labels['action_target_times'],dtype=np.float64)
    if target_times.shape!=(16,) or np.any(target_times<=ct) or not np.allclose(np.diff(target_times),.16,atol=1e-8,rtol=0):
        raise ValueError('suffix labels do not use the deployed physical endpoint grid')
    if len(positions) and not np.allclose(times[positions],target_times[:len(positions)],atol=1e-8,rtol=0):
        raise ValueError('committed prefix and independent suffix time grids differ')
    prefix=torch.zeros((16,14),dtype=actions.dtype)
    prefix[:len(positions)]=actions[positions]
    return dict(source_observation=source,current_observation=current,rtc_prefix_actions=prefix,
        rtc_prefix_length=len(positions),joint_targets=labels,prefix_source='real_same_trajectory_policy',
        provenance=m)

# Training consumer; raw trace validation above does not authorize a training dataset.
from dynamicwam.training.data.real_rollout_dataset import RealRolloutReplayDataset
