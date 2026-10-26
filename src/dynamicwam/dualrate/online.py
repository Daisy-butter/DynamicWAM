"""Causal online observations using the same pixels/flow/units as the dataset.

No target actor pose, contacts, segmentation or future frames are read here.
History endpoints sit on the raw-frame durations used by the native cache
(12 frames = 0.72 s). Each endpoint keeps the exposure of the matched capture.
Missing captures are never fabricated.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from .producer import (CAMERAS, CAMERA_AVAILABILITY_SECONDS, CONDITION_RGB_MAX_AGE_SECONDS,
    DT, FLOW_ENDPOINT_TOLERANCE_SECONDS, FLOW_HISTORY_OFFSETS_FRAMES, NUMERIC_TIME_EPSILON,
    RAW_FRAME_SECONDS,
    _letterbox, view_overlap)
from .timing import TimingError


@dataclass(frozen=True)
class OnlineFrame:
    images: tuple
    drive_target: np.ndarray
    exposure: float
    available: float
    frame_id: int


class ObservationHistory:
    def __init__(self, max_frames=128):
        if max_frames < 61:
            raise ValueError('history needs five preceding twelve-frame flow pairs')
        self.frames = deque(maxlen=max_frames)
        self.next_id = 0

    def append(self, observation, *, exposure, available):
        if not math.isfinite(exposure) or not math.isfinite(available) or available < exposure:
            raise TimingError('capture exposure/availability are not causal')
        if self.frames and exposure <= self.frames[-1].exposure:
            raise TimingError('duplicate/decreasing camera exposure')
        images = tuple(np.array(observation['observation'][c]['rgb'], copy=True) for c in CAMERAS)
        for x in images:
            if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError('online camera must supply actual RGB uint8')
            x.flags.writeable = False
        target = np.array(observation['joint_action']['vector'], dtype=np.float32, copy=True)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError('actual frozen drive-target vector required')
        target.flags.writeable = False
        value = OnlineFrame(images, target, float(exposure), float(available), self.next_id)
        self.next_id += 1
        self.frames.append(value)
        return value

    def snapshot(self, now):
        values = tuple(x for x in self.frames if x.available <= now + 1e-9 and x.exposure <= now + 1e-9)
        if not values:
            raise TimingError('no observation was available at query time')
        return values


def composite(images, shape):
    import cv2
    head, left, right = images
    h, w = head.shape[:2]
    bottom = np.concatenate((cv2.resize(left, (w//2,h//2)), cv2.resize(right,(w-w//2,h//2))),axis=1)
    return cv2.resize(np.concatenate((head,bottom),axis=0),(shape[1],shape[0]))


def history_endpoints(frames, *, strict=False, action_dt=DT):
    """Match raw-frame offsets to captures already available at the anchor.

    Offset 0 is the anchor itself and is always matched. Every earlier offset
    is the capture nearest its 0.72 s target, within one raw frame. A miss
    keeps a placeholder so shapes stay fixed, and its mask entry is false.
    The placeholder must not be treated as that endpoint.
    """
    frames = tuple(frames)
    if not frames:
        raise ValueError('empty captured history')
    anchor = frames[-1]
    selected, matched = [], []
    for offset in FLOW_HISTORY_OFFSETS_FRAMES:
        if offset == 0:
            selected.append(anchor)
            matched.append(True)
            continue
        target = anchor.exposure - (offset / 12) * 4 * float(action_dt)
        chosen = None
        best_error = None
        for frame in frames:
            if frame.exposure > anchor.exposure + NUMERIC_TIME_EPSILON:
                break
            error = abs(float(frame.exposure) - target)
            tolerance = NUMERIC_TIME_EPSILON if strict else FLOW_ENDPOINT_TOLERANCE_SECONDS + NUMERIC_TIME_EPSILON
            if error > tolerance:
                continue
            if chosen is None or error < best_error - NUMERIC_TIME_EPSILON or (
                    abs(error - best_error) <= NUMERIC_TIME_EPSILON and frame.exposure >= chosen.exposure):
                chosen, best_error = frame, error
        selected.append(anchor if chosen is None else chosen)
        matched.append(chosen is not None)
    return tuple(selected), tuple(matched)


def history_features(frames, contract, *, strict=False, action_dt=DT):
    """Reproduce the 0.72 s flow windows, including the preceding acceleration pair."""
    from dynamicwam.absolute_motion import (compute_flow_observation, build_motion_features,
        compute_dense_flow, filter_flow_by_forward_backward_consistency)
    frames = tuple(frames)
    if not frames:
        raise ValueError('empty captured history')
    selected, matched = history_endpoints(frames, strict=strict, action_dt=action_dt)
    selected = list(selected)
    flows, displacement, starts, ends, valid = [], [], [], [], []
    for index, (a, b) in enumerate(zip(selected, selected[1:])):
        ok = matched[index] and matched[index + 1] and a.frame_id != b.frame_id and b.exposure > a.exposure
        if ok:
            rgb, stats, _, quality = compute_flow_observation(a.images[0],b.images[0],
                compute_size=tuple(contract['compute_size']),
                normalization_percentile=contract['normalization_percentile'],
                farneback=contract['farneback'],quality=contract['quality'])
        else:
            rgb=np.zeros((*contract['compute_size'],3),np.uint8)
            stats=np.zeros(4,np.float32);quality=False
        flows.append(rgb);displacement.append(stats);starts.append(a.exposure);ends.append(b.exposure)
        valid.append(bool(ok and quality))
    features,mask,accel=build_motion_features(np.stack(displacement),np.array(starts,np.float64),
        np.array(ends,np.float64),np.array(valid,bool))
    intervals=np.stack((starts,ends),axis=-1)[1:]
    intervals[~mask[1:]]=frames[-1].exposure
    spatial=torch.zeros(16,5);spatial_valid=torch.zeros(16,dtype=torch.bool)
    a,b=selected[-2:];pair=torch.tensor([a.exposure,b.exposure],dtype=torch.float64)
    if valid[-1]:
        forward=compute_dense_flow(a.images[0],b.images[0],compute_size=(64,64),farneback=contract['farneback'])
        backward=compute_dense_flow(b.images[0],a.images[0],compute_size=(64,64),farneback=contract['farneback'])
        flow,reliable,_,quality=filter_flow_by_forward_backward_consistency(forward,backward,quality=contract['quality'])
        if quality:
            for y in range(4):
                for x in range(4):
                    k=y*4+x;m=reliable[y*16:(y+1)*16,x*16:(x+1)*16]
                    part=flow[y*16:(y+1)*16,x*16:(x+1)*16];fraction=float(m.mean())
                    if m.any() and fraction>=.1:
                        d=part[m].mean(0);dt=b.exposure-a.exposure
                        spatial[k]=torch.tensor([d[0],d[1],d[0]/dt,d[1]/dt,fraction]);spatial_valid[k]=True
    return dict(flow_rgb=np.stack(flows[1:]),motion=torch.from_numpy(features[1:].copy()),
        motion_valid=torch.from_numpy(mask[1:].copy()),motion_acceleration_valid=torch.from_numpy(accel[1:].copy()),
        motion_times=torch.from_numpy(intervals),spatial_flow=spatial,spatial_flow_valid=spatial_valid,
        spatial_flow_times=pair)


class OnlineBatchBuilder:
    def __init__(self, manifest, *, action_mean, action_scale, language, device='cpu', slow_device=None):
        self.manifest=manifest;self.device=torch.device(device)
        self.slow_device=torch.device(device if slow_device is None else slow_device)
        self.action_dt=float(manifest.get('action_interval_seconds',DT))
        self.hardtime=manifest.get('time_contract')=='exposure_anchor_wall_release_v4'
        if not math.isfinite(self.action_dt) or self.action_dt<=0:
            raise ValueError('finite positive physical action interval required')
        self.mean=torch.as_tensor(action_mean,dtype=torch.float32)
        self.scale=torch.as_tensor(action_scale,dtype=torch.float32)
        if self.mean.shape!=(14,) or self.scale.shape!=(14,) or (self.scale<=0).any():
            raise ValueError('checkpoint normalization required')
        if language['context'].shape!=(512,4096) or language['mask'].shape!=(512,) or not language['mask'].any():
            raise ValueError('exact episode-bound T5 context required')
        self.language=language
        self.camera_availability_seconds=float(manifest.get('camera_availability_seconds',CAMERA_AVAILABILITY_SECONDS))
        self.max_observation_age=float(manifest.get('condition_rgb_max_age_seconds',CONDITION_RGB_MAX_AGE_SECONDS))
        if (not math.isfinite(self.camera_availability_seconds) or self.camera_availability_seconds < 0 or
                not math.isfinite(self.max_observation_age) or
                self.max_observation_age + 1e-9 < self.camera_availability_seconds):
            raise ValueError('camera availability must fit inside the condition age gate')
        self.entity_roles=tuple(manifest.get('active_entity_roles',('object','support','left_gripper','right_gripper')))
        if not self.entity_roles or len(set(self.entity_roles))!=len(self.entity_roles):
            raise ValueError('distinct ordered entity roles required')

    def _batch(self, values, *, device=None):
        return {k:v.unsqueeze(0).to(self.device if device is None else device) for k,v in values.items()}

    def current(self, frames, *, query_time, action_start, executed=()):
        if not frames or frames[-1].available>query_time+1e-9:
            raise TimingError('unavailable frame in fast condition')
        if not 0<=query_time-frames[-1].exposure<=self.max_observation_age+1e-8:
            raise TimingError('latest camera observation is too old')
        rgb,pixels=zip(*[_letterbox(image,tuple(self.manifest['rgb_size']),tuple(self.manifest['rgb_view_sizes'][v]))
                        for v,image in enumerate(frames[-1].images)])
        h=history_features(frames,self.manifest['flow_contract'],action_dt=self.action_dt);h.pop('flow_rgb')
        starts=torch.arange(16,dtype=torch.float64)*self.action_dt+action_start
        trace=[row for row in executed if row[1]<=query_time+1e-8]
        values=dict(h,rgb=torch.stack(rgb),rgb_pixel_valid=torch.stack(pixels),
            rgb_times=torch.full((3,),frames[-1].exposure,dtype=torch.float64),
            state=(torch.from_numpy(frames[-1].drive_target.copy())-self.mean)/self.scale,
            state_time=torch.tensor(frames[-1].exposure,dtype=torch.float64),
            text=self.language['context'],text_valid=self.language['mask'],
            query_time=torch.tensor(query_time,dtype=torch.float64),
            action_times=starts,action_intervals=torch.stack((starts,starts+self.action_dt),-1),
            future_times=torch.tensor([query_time+x for x in (.1,.2,.4,.8)]*len(self.entity_roles),dtype=torch.float64),
            query_entity_valid=torch.ones(len(self.entity_roles),dtype=torch.bool),
            executed_actions=(torch.tensor([x[2] for x in trace],dtype=torch.float32).reshape(-1,14)-self.mean)/self.scale,
            executed_action_intervals=torch.tensor([[x[0],x[1]] for x in trace],dtype=torch.float64).reshape(-1,2),
            executed_action_valid=torch.ones(len(trace),dtype=torch.bool))
        return self._batch(values)

    def source(self, frames, *, source_time, action_start, vae, request_time=None):
        import cv2
        request_time=float(source_time if request_time is None else request_time)
        if not frames or frames[-1].available>request_time+1e-9:
            raise TimingError('unavailable source observation')
        if not 0<=request_time-frames[-1].exposure<=self.max_observation_age+1e-8:
            raise TimingError('latest source camera observation is too old')
        if self.hardtime and (abs(source_time-frames[-1].exposure)>1e-9 or abs(action_start-source_time)>1e-9):
            raise TimingError('slow prediction must start at the actual grid-aligned exposure')
        # Each numeric history row and its RGB flow use the same real pair.
        # Native motion retains each pair's actual dt; nearest endpoints are
        # bounded data sampling, never relabelled as exact nominal endpoints.
        h=history_features(frames,self.manifest['flow_contract'],action_dt=self.action_dt);last=frames[-1]
        raw=np.stack([cv2.resize(x,(320,384)) for x in h['flow_rgb']]+[composite(last.images,(384,320))])
        x=torch.from_numpy(raw).permute(3,0,1,2)[None].to(self.slow_device,dtype=getattr(vae,'dtype',torch.bfloat16))/127.5-1
        with torch.no_grad():
            latent=vae.encode(x)
        if latent.shape[1:]!=(48,2,24,20):
            raise ValueError('native online source codec shape differs from training')
        s=source_time;starts=torch.arange(16,dtype=torch.float64)*self.action_dt+action_start
        hist_start=float(h['motion_times'][h['motion_valid'],0].min()) if h['motion_valid'].any() else last.exposure
        values=dict(source_initial_state=(torch.from_numpy(last.drive_target.copy())-self.mean)/self.scale,
            source_motion=h['motion'],source_motion_valid=h['motion_valid'],
            source_motion_times=h['motion_times'],
            source_motion_acceleration_valid=h['motion_acceleration_valid'],source_text=self.language['context'],
            source_action_times=starts,source_action_intervals=torch.stack((starts,starts+self.action_dt),-1),
            source_future_intervals=torch.tensor([[s,s+8*self.action_dt],[s+8*self.action_dt,s+16*self.action_dt]],dtype=torch.float64),
            source_temporal_support=torch.tensor([h['motion_times'][0].tolist(),[hist_start,last.exposure]],dtype=torch.float64),
            source_view_overlap=torch.cat((view_overlap(12,10,flow_weight=1.),view_overlap(12,10,flow_weight=.75))),
            source_token_valid=torch.cat((torch.full((120,),bool(h['motion_valid'][0])),torch.ones(120,dtype=torch.bool))),
            source_input_type=torch.tensor([1,2]),future_view_overlap=view_overlap(6,5).repeat(2,1),
            future_token_valid=torch.ones(60,dtype=torch.bool),future_input_type=torch.zeros(2,dtype=torch.long),
            source_time=torch.tensor(s,dtype=torch.float64),completed_time=torch.tensor(request_time,dtype=torch.float64),
            query_time=torch.tensor(request_time,dtype=torch.float64),
            source_request_time=torch.tensor(request_time,dtype=torch.float64),
            source_rgb_actual_times=torch.tensor(last.exposure,dtype=torch.float64),
            source_available_time=torch.tensor(last.available,dtype=torch.float64))
        result=self._batch(values,device=self.slow_device);result['source_condition_latent']=latent
        return result


def load_cached_language(root, prompt, *, index_path=None):
    """Only exact text-hash match; never substitute the first task episode."""
    from .data import sha256_file
    root=Path(root).resolve()
    selected=Path(index_path) if index_path is not None else root/'text/index.json'
    if not selected.is_absolute():selected=root/selected
    selected=selected.resolve(strict=True);index=json.loads(selected.read_text())
    key=hashlib.sha256(prompt.encode()).hexdigest()
    entries=index.get('entries',index.get('items',index))
    if key not in entries:
        raise KeyError('episode prompt not precomputed; encode this exact prompt before timed deployment')
    row=entries[key];p=(selected.parent/row['path']).resolve(strict=True)
    if row['text_sha256']!=key or sha256_file(p)!=row['sha256']:
        raise ValueError('episode language cache identity mismatch')
    return torch.load(p,map_location='cpu',weights_only=True)
