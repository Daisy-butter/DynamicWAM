"""Causal online observations using the same pixels/flow/units as the dataset.

No target actor pose, contacts, segmentation or future frames are read here.
History endpoints sit on the raw-frame durations used by the native cache
(12 frames = 0.72 s). Each endpoint keeps the exposure of the matched capture.
Missing captures are never fabricated.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, OrderedDict
import hashlib
import json
import math
import threading
from pathlib import Path
import numpy as np
import torch

from .producer import (CAMERAS, CAMERA_AVAILABILITY_SECONDS, CONDITION_RGB_MAX_AGE_SECONDS,
    DT, FLOW_ENDPOINT_TOLERANCE_SECONDS, FLOW_HISTORY_OFFSETS_FRAMES, NUMERIC_TIME_EPSILON,
    RAW_FRAME_SECONDS,
    _letterbox, view_overlap, flow_history_selection)
from .timing import TimingError


@dataclass(frozen=True)
class OnlineFrame:
    images: tuple
    drive_target: np.ndarray
    exposure: float
    available: float
    frame_id: int
    measured_q: np.ndarray | None = None
    measured_v: np.ndarray | None = None


def _immutable_array(value):
    """A bytes-backed snapshot cannot be made writable again by a consumer."""
    value=np.asarray(value)
    return np.frombuffer(value.tobytes(order='C'),dtype=value.dtype).reshape(value.shape)


def _immutable_head(frame):
    value=frame.images[0]
    if not isinstance(value,np.ndarray) or value.flags.writeable:
        return False
    while isinstance(value,np.ndarray):
        value=value.base
    return isinstance(value,bytes)


def _copy_pair_value(value):
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,tuple):return tuple(_copy_pair_value(x) for x in value)
    return value


class ObservationPairCache:
    """Bounded per-episode memoization; a concurrent miss computes independently.

    Keys bind the actual immutable frame objects and the complete flow contract.
    Entries retain their frames, preventing Python id reuse. Neither Farneback
    nor copying results runs under the metadata lock; fast never waits for a
    slow worker's unfinished computation. Returned arrays/tensors are private.
    """
    def __init__(self,max_entries=32):
        if type(max_entries) is not int or max_entries<1:
            raise ValueError('positive bounded pair-cache capacity required')
        self.max_entries=max_entries
        self._entries=OrderedDict();self._lock=threading.Lock()

    def compute(self,kind,a,b,contract_key,function):
        # Public helpers/tests may supply mutable frames: compute normally;
        # never trust frame_id alone or silently cache mutable image contents.
        if not (_immutable_head(a) and _immutable_head(b)):
            return function()
        key=(kind,id(a),id(b),contract_key)
        with self._lock:
            entry=self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is not None:
            return _copy_pair_value(entry[2])
        value=function()
        saved=_copy_pair_value(value)
        with self._lock:
            # A racing worker may already have published the identical pair.
            # Keep one result, without waiting for any in-flight calculation.
            entry=self._entries.get(key)
            if entry is None:
                self._entries[key]=(a,b,saved)
                while len(self._entries)>self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
        return value

    def __len__(self):
        with self._lock:return len(self._entries)


class ObservationHistory:
    def __init__(self, max_frames=128, *, allow_equal_exposure=False):
        if max_frames < 61:
            raise ValueError('history needs five preceding twelve-frame flow pairs')
        self.frames = deque(maxlen=max_frames)
        self.next_id = 0
        self.allow_equal_exposure=bool(allow_equal_exposure)

    def append(self, observation, *, exposure, available):
        if not math.isfinite(exposure) or not math.isfinite(available) or available < exposure:
            raise TimingError('capture exposure/availability are not causal')
        if self.frames and (exposure < self.frames[-1].exposure or (exposure == self.frames[-1].exposure and not self.allow_equal_exposure)):
            raise TimingError('duplicate/decreasing camera exposure')
        images = tuple(np.asarray(observation['observation'][c]['rgb']) for c in CAMERAS)
        for x in images:
            if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError('online camera must supply actual RGB uint8')
        images=tuple(_immutable_array(x) for x in images)
        target = np.array(observation['joint_action']['vector'], dtype=np.float32, copy=True)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError('actual frozen drive-target vector required')
        target=_immutable_array(target)
        sensor=observation.get('measured_joint_state')
        q=v=None
        if sensor is not None:
            q=np.asarray(sensor['q'],np.float32);v=np.asarray(sensor['v'],np.float32)
            if q.shape!=(12,) or v.shape!=(12,) or not np.isfinite(q).all() or not np.isfinite(v).all() or abs(sensor['physical_time']-exposure)>1e-8:
                raise ValueError('measured state must belong to this exact frozen exposure')
            q,v=_immutable_array(q),_immutable_array(v)
        value = OnlineFrame(images, target, float(exposure), float(available), self.next_id,q,v)
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


def history_endpoints(frames, *, strict=False, action_dt=DT, native_ordinal=False):
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
    if native_ordinal:
        for offset in FLOW_HISTORY_OFFSETS_FRAMES:
            index=len(frames)-1-(offset//12)*4
            selected.append(frames[max(0,index)]);matched.append(index>=0)
        return tuple(selected),tuple(matched)
    if len(frames)==1:
        # Training selects inside a multi-frame episode; online bootstrap has
        # only its first real exposure. No earlier endpoint can be observed.
        return (anchor,)*len(FLOW_HISTORY_OFFSETS_FRAMES),tuple(x==0 for x in FLOW_HISTORY_OFFSETS_FRAMES)
    # Use the training selector, including duplicate, tie and missing-frame rules.
    indices, matched = flow_history_selection(
        [x.exposure for x in frames], len(frames)-1,
        interval_seconds=4*float(action_dt),
        tolerance=0. if strict else FLOW_ENDPOINT_TOLERANCE_SECONDS)
    return tuple(frames[i] for i in indices), tuple(matched)


def history_features(frames, contract, *, strict=False, action_dt=DT, pair_cache=None, include_spatial=True, native_ordinal=False):
    """Reproduce the 0.72 s flow windows, including the preceding acceleration pair."""
    from dynamicwam.absolute_motion import (compute_flow_observation, build_motion_features,
        compute_dense_flow, filter_flow_by_forward_backward_consistency)
    frames = tuple(frames)
    if not frames:
        raise ValueError('empty captured history')
    # Snapshot all settings once; mutation of a caller's dict cannot give a
    # cached value a key different from the parameters actually computed.
    contract_key=json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False)
    contract=json.loads(contract_key)
    def pair_value(kind,a,b,function):
        return function() if pair_cache is None else pair_cache.compute(kind,a,b,contract_key,function)
    selected, matched = history_endpoints(frames, strict=strict, action_dt=action_dt,native_ordinal=native_ordinal)
    selected = list(selected)
    flows, displacement, starts, ends, valid = [], [], [], [], []
    for index, (a, b) in enumerate(zip(selected, selected[1:])):
        ok = matched[index] and matched[index + 1] and a.frame_id != b.frame_id and b.exposure > a.exposure
        if ok:
            rgb, stats, _, quality = pair_value('native',a,b,lambda a=a,b=b:compute_flow_observation(a.images[0],b.images[0],
                compute_size=tuple(contract['compute_size']),
                normalization_percentile=contract['normalization_percentile'],
                farneback=contract['farneback'],quality=contract['quality']))
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
    if include_spatial and valid[-1]:
        def compute_spatial_pair():
            forward=compute_dense_flow(a.images[0],b.images[0],compute_size=(64,64),farneback=contract['farneback'])
            backward=compute_dense_flow(b.images[0],a.images[0],compute_size=(64,64),farneback=contract['farneback'])
            return filter_flow_by_forward_backward_consistency(forward,backward,quality=contract['quality'])
        flow,reliable,_,quality=pair_value('spatial64',a,b,compute_spatial_pair)
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
        self.collect_feedback=False
        self.pair_cache=ObservationPairCache()
        self.slow_device=torch.device(device if slow_device is None else slow_device)
        self.action_dt=float(manifest.get('action_interval_seconds',DT))
        self.hardtime=manifest.get('time_contract')=='exposure_anchor_wall_release_v4'
        self.native_ordinal=manifest.get('time_contract') in ('native_sync_ordinal_v5','native_event_single_v6')
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
        if not 0<=query_time-frames[-1].exposure<=self.max_observation_age+NUMERIC_TIME_EPSILON:
            raise TimingError('latest camera observation is too old')
        rgb,pixels=zip(*[_letterbox(image,tuple(self.manifest['rgb_size']),tuple(self.manifest['rgb_view_sizes'][v]))
                        for v,image in enumerate(frames[-1].images)])
        h=history_features(frames,self.manifest['flow_contract'],action_dt=self.action_dt,
                           pair_cache=self.pair_cache,native_ordinal=self.native_ordinal);h.pop('flow_rgb')
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
        if self.manifest.get('residual_v10',{}).get('continuity'):
            from .feedback_observation import measured_features
            values['measured_state'],values['measured_state_valid']=measured_features(
                frames[-1].measured_q,frames[-1].measured_v,frames[-1].drive_target,
                self.mean,self.scale,query_time-frames[-1].exposure)
        if self.collect_feedback:
            teacher=self.source(frames,source_time=frames[-1].exposure,
                action_start=frames[-1].exposure,request_time=query_time,vae=None)
            values.update({'teacher_'+k[7:]:v[0].cpu() for k,v in teacher.items() if k.startswith('source_')})
        return self._batch(values)

    def source(self, frames, *, source_time, action_start, vae, request_time=None):
        import cv2
        request_time=float(source_time if request_time is None else request_time)
        if not frames or frames[-1].available>request_time+1e-9:
            raise TimingError('unavailable source observation')
        if not 0<=request_time-frames[-1].exposure<=self.max_observation_age+NUMERIC_TIME_EPSILON:
            raise TimingError('latest source camera observation is too old')
        if self.hardtime and (abs(source_time-frames[-1].exposure)>1e-9 or abs(action_start-source_time)>1e-9):
            raise TimingError('slow prediction must start at the actual grid-aligned exposure')
        # Each numeric history row and its RGB flow use the same real pair.
        # Native motion retains each pair's actual dt; nearest endpoints are
        # bounded data sampling, never relabelled as exact nominal endpoints.
        h=history_features(frames,self.manifest['flow_contract'],action_dt=self.action_dt,
                           pair_cache=self.pair_cache,include_spatial=False,native_ordinal=self.native_ordinal);last=frames[-1]
        raw=np.stack([cv2.resize(x,(320,384)) for x in h['flow_rgb']]+[composite(last.images,(384,320))])
        # Match producer FP32 pixel normalization before the native VAE cast;
        # BF16 division/subtraction first introduces a different rounding path.
        x=(torch.from_numpy(raw).permute(3,0,1,2)[None].float()/127.5-1).to(
            'cpu' if vae is None else self.slow_device,dtype=getattr(vae,'dtype',torch.bfloat16))
        latent=None
        if vae is not None:
            with torch.no_grad():latent=vae.encode(x)
        if latent is not None and latent.shape[1:]!=(48,2,24,20):
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
        destination='cpu' if vae is None else self.slow_device
        result=(OnlineBatchBuilder._batch(self,values,device=destination) if vae is None else self._batch(values,device=destination))
        if latent is not None:result['source_condition_latent']=latent
        if self.collect_feedback:
            result['source_condition_frames']=(torch.from_numpy(raw).permute(3,0,1,2)[None].float()/127.5-1)
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
