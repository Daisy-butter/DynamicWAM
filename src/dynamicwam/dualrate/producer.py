"""Real DOMINO schema-v2 -> fixed physical-time DualRate training records.

No simulator is started here. HDF5 is read-only. Native VAE and T5 are frozen
preprocessors supplied by the caller; absent caches fail rather than inventing
latents/text. Images used as *conditions* are always past-only. Future RGB labels
use real nearest frames, retaining exposure/error audit tensors. Projection crops
supervise local image context, including possible occluders, not visible-object
identity. All privileged geometry/contact fields are confined to label creation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (EpisodeRecords, produce_future_labels, normalize_future,
                   sha256_file, _require)

SCHEMA = "dualrate-domino-lazy-v2"
CODEC_ID = "wan22_first_plus_four_causal_native_v2"
CAMERAS = ("head_camera", "left_camera", "right_camera")
DT = .18
RGB_TOLERANCE = .030001
INTERPOLATION_GAP = .060001
QUERY_AGES = (.36, .72)
SCENE_PREFIX = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
# The templates were checked against each task's full_description; several
# upstream first 'seen' prompts omit task steps or reference missing objects.
PROMPTS = {
    "adjust_bottle": "Using the {a} arm, pick up {A} and hold it upright.",
    "beat_block_hammer": "Using the {a} arm, pick up {A} and strike the block.",
    "click_alarmclock": "Using the {a} arm, press the center top button of {A}.",
    "click_bell": "Using the {a} arm, press the top center of {A}.",
    "dump_bin_bigbin": "Pick up {A} and pour its balls into the large bin.",
    "grab_roller": "Use both arms to grasp {A}.",
    "handover_block": "Use the left arm to pick up the red block, hand it to the right arm, and place it on the blue pad.",
    "handover_mic": "Use the {a} arm to pick up {A} and hand it to the {b} arm.",
    "hanging_mug": "Use the left arm to pick up {A}, rotate it and set it down; then use the right arm to hang it on {B}.",
    "move_can_pot": "Using the {a} arm, pick up {B} and place it beside {A}.",
    "move_pillbottle_pad": "Using the {a} arm, pick up {A} and place it on the pad.",
    "move_playingcard_away": "Using the {a} arm, pick up {A} and move it farther outward away from the table.",
    "move_stapler_pad": "Using the {a} arm, place {A} on the {B} mat.",
    "place_a2b_left": "Using the {a} arm, place {A} to the left of {B}.",
    "place_a2b_right": "Using the {a} arm, place {A} to the right of {B}.",
    "place_bread_basket": "Using the {a} arm, pick up {B} and place it inside {A}.",
    "place_bread_skillet": "Using the {a} arm, pick up {B} and place it inside {A}.",
    "place_can_basket": "Using the {a} arm, place {A} inside {B}, then use the other arm to lift {B}.",
    "place_container_plate": "Using the {a} arm, place {B} on {A}.",
    "place_empty_cup": "Pick up {A} and place it on {B}.",
    "place_fan": "Using the {a} arm, place {A} on the {B} mat with the fan facing the robot.",
    "place_mouse_pad": "Using the {a} arm, place {A} on the {B} mat.",
    "place_object_basket": "Use the {a} arm to place {A} in {B}, then use the {b} arm to lift and slightly move {B}.",
    "place_object_scale": "Using the {a} arm, place {B} on {A}.",
    "place_object_stand": "Using the {a} arm, place {A} on {B}.",
    "place_phone_stand": "Using the {a} arm, place {A} on {B}.",
    "place_shoe": "Using the {a} arm, place {A} on the mat.",
    "press_stapler": "Using the {a} arm, press {A}.",
    "put_object_cabinet": "Use the {a} arm to open the drawer of {B}, then use the {b} arm to place {A} inside the drawer.",
    "rotate_qrcode": "Using the {a} arm, pick up {A} and rotate it so the QR code faces the robot.",
    "scan_object": "Use the {b} arm to pick up {B} and the {a} arm to pick up {A}; scan {A} with {B}.",
    "shake_bottle": "Using the {a} arm, pick up {A} and shake it.",
    "shake_bottle_horizontally": "Using the {a} arm, pick up {A} and shake it horizontally.",
    "stamp_seal": "Using the {a} arm, pick up {A} and stamp the {B} mat.",
}


def episode_prompt(task, info, description_root):
    """Bind a complete task instruction to this episode's actual objects/arms."""
    root = Path(description_root)
    provenance = {}
    replacements = {}
    for placeholder, value in info.items():
        if isinstance(value, str) and "/" in value:
            p = root / "objects_description" / (value + ".json")
            obj = json.loads(p.read_text())
            replacements[placeholder] = obj["seen"][0]
            provenance[str(p)] = sha256_file(p)
        else:
            replacements[placeholder] = str(value)
    if task == "put_bottles_dustbin":
        bottles = [replacements[k] for k in ("{A}", "{B}", "{C}") if k in replacements]
        _require(bool(bottles) and "{D}" in replacements, "bottle episode object bindings missing")
        text = "Pick up " + ", ".join(bottles) + " and place them in " + replacements["{D}"] + " to the left of the table."
    else:
        text = PROMPTS[task]
        for key, value in replacements.items():
            text = text.replace(key, value)
    _require(re.search(r"\{[^}]+\}", text) is None, f"unresolved episode language: {task}: {text}")
    task_path = root / "task_instruction" / f"{task}.json"
    provenance[str(task_path)] = sha256_file(task_path)
    return SCENE_PREFIX + text, provenance


def unique_last_indices(times):
    times = np.asarray(times, dtype=np.float64)
    _require(times.ndim == 1 and len(times) > 1 and np.isfinite(times).all(), "invalid episode timestamps")
    _require((np.diff(times) >= 0).all(), "decreasing episode timestamps")
    return np.flatnonzero(np.r_[np.diff(times) != 0, True])


def file_stat_identity(path):
    """Accidental-rewrite detector; explicitly NOT a content/byte hash."""
    resolved=Path(path).resolve(strict=True);stat=resolved.stat()
    return dict(resolved_path=str(resolved),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def verify_file_stat(path, expected):
    _require(file_stat_identity(path)==expected,f'raw/flow file stat identity changed: {path}')


def attach_stat_identity(output):
    """Bind existing prepared episode raw/flow files without rehashing image TBs."""
    output=Path(output)
    episodes=json.loads((output/'episodes.json').read_text())
    sidecar=dict(schema_version=1,identity_kind='resolved_path_size_mtime_ns_not_content_hash',episodes={})
    for ep in episodes:
        sidecar['episodes'][ep['episode_uid']]={key:file_stat_identity(ep[key+'_path']) for key in ('raw','flow')}
    path=output/'source_file_stats.json'
    path.write_text(json.dumps(sidecar,indent=2))
    digest=sha256_file(path)
    for split in ('train','validation'):
        manifest=output/f'{split}.json'
        if manifest.exists():
            m=json.loads(manifest.read_text());m.update(source_file_stats='source_file_stats.json',source_file_stats_sha256=digest,
                                                       source_file_identity_kind=sidecar['identity_kind'])
            manifest.write_text(json.dumps(m,indent=2))
    return dict(episodes=len(episodes),path=str(path),sha256=digest,identity_kind=sidecar['identity_kind'])


def select_frame(times, target, *, past_only, tolerance):
    """Return a real frame index; duplicate timestamps select the last whole row."""
    indices = unique_last_indices(times)
    t = np.asarray(times)[indices]
    j = int(np.searchsorted(t, target, side="right")) - 1
    if past_only:
        _require(j >= 0, "no past frame")
    else:
        options = [i for i in (j, j + 1) if 0 <= i < len(t)]
        j = min(options, key=lambda i: abs(float(t[i]) - target))
    _require(abs(float(t[j]) - target) <= tolerance, "no real frame within requested tolerance")
    return int(indices[j])


def resample_actions(times, values, targets, *, max_gap=INTERPOLATION_GAP):
    """Arm drive targets interpolate; gripper commands use previous-event hold.

    No extrapolation. These are demonstration targets, not measured arm qpos.
    Returned errors describe the distance to the nearer real observation.
    """
    indices = unique_last_indices(times)
    t = np.asarray(times)[indices]
    x = np.asarray(values, dtype=np.float32)[indices]
    out, valid, errors = [], [], []
    for target in np.asarray(targets):
        hi = int(np.searchsorted(t, target, side="left"))
        candidates = [i for i in (hi, hi - 1) if 0 <= i < len(t)]
        exact = next((i for i in candidates if abs(t[i] - target) <= 1e-8), None)
        if exact is not None:
            out.append(x[exact].copy()); valid.append(True); errors.append(abs(t[exact]-target)); continue
        if hi == 0 or hi == len(t) or t[hi]-t[hi-1] > max_gap:
            out.append(np.zeros(14,np.float32)); valid.append(False); errors.append(float("inf")); continue
        lo = hi-1
        a = float((target-t[lo])/(t[hi]-t[lo]))
        v = x[lo]*(1-a)+x[hi]*a
        v[[6,13]] = x[lo,[6,13]]
        out.append(v); valid.append(True); errors.append(min(target-t[lo],t[hi]-target))
    return np.asarray(out), np.asarray(valid), np.asarray(errors)


def view_overlap(grid_h, grid_w, *, flow_weight=0.):
    """Area overlap of original head / half-size wrist camera composite per cell."""
    result = []
    for y in range(grid_h):
        for x in range(grid_w):
            x0,x1,y0,y1 = x/grid_w,(x+1)/grid_w,y/grid_h,(y+1)/grid_h
            area=(x1-x0)*(y1-y0)
            head=max(0.,min(y1,2/3)-y0)*(x1-x0)/area
            wrist=max(0.,y1-max(y0,2/3))
            left=wrist*max(0.,min(x1,.5)-x0)/area
            right=wrist*max(0.,x1-max(x0,.5))/area
            result.append([flow_weight+(1-flow_weight)*head,(1-flow_weight)*left,(1-flow_weight)*right,0.])
    return torch.tensor(result,dtype=torch.float32)


def _rgb(handle, frame, camera):
    import cv2
    encoded = handle[f"observation/{camera}/rgb"][frame]
    array = np.frombuffer(encoded.tobytes() if hasattr(encoded,"tobytes") else encoded, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    _require(image is not None, "corrupt real RGB JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _composite(handle, frame, shape):
    import cv2
    head,left,right = [_rgb(handle,frame,c) for c in CAMERAS]
    h,w=head.shape[:2]
    bottom=np.concatenate([cv2.resize(left,(w//2,h//2)),cv2.resize(right,(w-w//2,h//2))],1)
    return cv2.resize(np.concatenate([head,bottom],0),(shape[1],shape[0]))


def _letterbox(image, shape=(240,320), view_shape=None):
    import cv2
    h,w=image.shape[:2]; view_shape=shape if view_shape is None else view_shape
    _require(view_shape[0]<=shape[0] and view_shape[1]<=shape[1],"view resolution exceeds canvas")
    scale=min(view_shape[0]/h,view_shape[1]/w)
    nh,nw=round(h*scale),round(w*scale); y=(shape[0]-nh)//2; x=(shape[1]-nw)//2
    result=np.zeros((*shape,3),np.uint8);valid=np.zeros(shape,bool)
    result[y:y+nh,x:x+nw]=cv2.resize(image,(nw,nh));valid[y:y+nh,x:x+nw]=True
    return torch.from_numpy(result).permute(2,0,1).float()/255,torch.from_numpy(valid)


class ProjectedContextTeacher:
    """Frozen true-image local context; occlusion is unknown, never assumed clear."""
    def __init__(self, path, digest):
        from .vision import ResNet18Stage3
        self.model=ResNet18Stage3(path,digest).eval().requires_grad_(False)
        matrix=torch.zeros(256,128)
        for i in range(128):matrix[2*i:2*i+2,i]=.5
        self.projection=matrix
        self.projection_sha256=hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
        self.identity=f"resnet18-stage3:{digest}:pairmean128:{self.projection_sha256}:projected_local_context"

    @torch.no_grad()
    def encode(self, handle, frames, positions):
        """Returns [T,4,128] and crop-support mask, keeping support role unknown."""
        import cv2
        crops=[];locations=[]
        result=torch.zeros(len(frames),4,128);valid=torch.zeros(len(frames),4,dtype=torch.bool)
        for i,frame in enumerate(frames):
            for entity in (0,2,3):
                candidates=[]
                point=np.r_[positions[i,entity],1.]
                for camera in CAMERAS:
                    ext=handle[f"observation/{camera}/extrinsic_cv"][frame]
                    intr=handle[f"observation/{camera}/intrinsic_cv"][frame]
                    p=ext@point
                    if not np.isfinite(p).all() or p[2]<=1e-6:continue
                    uv=intr@p;uv=uv[:2]/uv[2]
                    image=_rgb(handle,frame,camera);h,w=image.shape[:2]
                    # A named 32px context window, not an inferred instance box.
                    x,y=np.rint(uv).astype(int);x0,x1=max(0,x-16),min(w,x+16);y0,y1=max(0,y-16),min(h,y+16)
                    if x0<x1 and y0<y1 and 0<=x<w and 0<=y<h and (x1-x0)*(y1-y0)>=512:
                        candidates.append(((x1-x0)*(y1-y0),image[y0:y1,x0:x1]))
                if candidates:
                    crop=max(candidates,key=lambda z:z[0])[1]
                    crops.append(cv2.resize(crop,(64,64)));locations.append((i,entity))
        if crops:
            x=torch.from_numpy(np.stack(crops)).permute(0,3,1,2).float()/255
            x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            feature=self.model(x).mean((-2,-1))@self.projection
            for loc,value in zip(locations,feature):result[loc]=value;valid[loc]=True
        return result,valid


def read_episode_records(path, row, visual_encoder_id):
    import h5py
    with h5py.File(path,"r") as h:
        times=torch.from_numpy(h['interception/sim_time_seconds'][:]).double()
        hands=torch.from_numpy(h['interception/ee_poses_wxyz'][:,:,:3]).float()
        xyz=torch.zeros(len(times),4,3);xyz[:,0]=torch.from_numpy(h['interception/target_pose_wxyz'][:,:3]);xyz[:,2:]=hands
        contacts=torch.from_numpy(h['interception/target_gripper_contact_mask'][:]).float()
        actions=torch.from_numpy(h['joint_action/vector'][:]).float()
    entity_valid=torch.ones(len(times),4,dtype=torch.bool);entity_valid[:,1]=False
    r=EpisodeRecords(row['episode_uid'],row['scene_id'],row['split'],visual_encoder_id,
                     ('tracked_dynamic_target',None,'left_gripper','right_gripper'),times,xyz,hands,
                     torch.zeros(len(times),4,128),entity_valid,torch.ones(len(times),2,dtype=torch.bool),
                     torch.zeros(len(times),4,dtype=torch.bool),torch.zeros(len(times),4,dtype=torch.bool),
                     torch.tensor([0,-1,2,3]).repeat(len(times),1),contacts,torch.ones_like(contacts,dtype=torch.bool),
                     (row.get('relevant_gripper',0),0))
    return r,actions


def _flow_features(handle, times, frame, flow_cache, flow_contract):
    """Original raw numeric history plus freshly computed spatial flow, CPU only."""
    from dynamicwam.absolute_motion import compute_dense_flow, filter_flow_by_forward_backward_consistency
    endpoints=[max(0,frame-k) for k in (48,36,24,12,0)]
    selected=endpoints[1:]
    numeric=torch.from_numpy(flow_cache['motion_features'][selected].copy())
    valid=torch.from_numpy(flow_cache['interval_valid'][selected].copy())
    acceleration=torch.from_numpy(flow_cache['acceleration_valid'][selected].copy())
    intervals=np.stack([flow_cache['interval_start_time_seconds'][selected],flow_cache['interval_end_time_seconds'][selected]],-1)
    _require(np.isfinite(intervals[valid.numpy()]).all() and (intervals[valid.numpy(),1]<=times[frame]+1e-8).all(),"future/nonfinite valid flow in observation")
    # Invalid historical intervals do not acquire invented support from a cache
    # sentinel such as time zero. A finite masked point is only a placeholder.
    intervals[~valid.numpy()]=times[frame]
    spatial=torch.zeros(16,5);spatial_valid=torch.zeros(16,dtype=torch.bool)
    pair=np.array([times[endpoints[-2]],times[frame]])
    if bool(valid[-1]) and pair[1]>pair[0]:
        a=_rgb(handle,endpoints[-2],CAMERAS[0]);b=_rgb(handle,frame,CAMERAS[0])
        forward=compute_dense_flow(a,b,compute_size=(64,64),farneback=flow_contract['farneback'])
        backward=compute_dense_flow(b,a,compute_size=(64,64),farneback=flow_contract['farneback'])
        flow,reliable,_,quality=filter_flow_by_forward_backward_consistency(forward,backward,quality=flow_contract['quality'])
        if quality:
            for y in range(4):
                for x in range(4):
                    k=y*4+x;mask=reliable[y*16:(y+1)*16,x*16:(x+1)*16];part=flow[y*16:(y+1)*16,x*16:(x+1)*16]
                    fraction=float(mask.mean())
                    if mask.any() and fraction>=.1:
                        d=part[mask].mean(0);spatial[k]=torch.tensor([d[0],d[1],d[0]/(pair[1]-pair[0]),d[1]/(pair[1]-pair[0]),fraction]);spatial_valid[k]=True
    return numeric,valid,acceleration,torch.from_numpy(intervals),spatial,spatial_valid,torch.from_numpy(pair)


class DominoPhysicalDataset(Dataset):
    """Lazy all-episode real RGB/labels; VAE encoding occurs in the training process.

    Text cache is mandatory. Teacher runs frozen on CPU. No absent cache silently
    becomes random/zero model conditioning. Metadata JSON and tensor-only caches
    are auditable independently of the original packed data.
    """
    def __init__(self, manifest_path, *, verify_hash=True, max_cached_episodes=2, require_complete_text=True):
        self.path=Path(manifest_path).resolve();self.manifest=json.loads(self.path.read_text())
        m=self.manifest;_require(m['schema_version']==SCHEMA,"wrong DOMINO schema")
        _require(m['action_interval_seconds']==DT and m['codec_id']==CODEC_ID,"unverified timing/codec contract")
        for k in ('native_checkpoint_sha256','action_normalization_sha256','motion_normalization_sha256'):
            _require(bool(re.fullmatch('[0-9a-f]{64}',m.get(k,''))),f'missing {k}')
        _require(m['statistics']['fit_split']=='train',"statistics are not train-only")
        self.rows=np.load(self.path.parent/m['rows_file'],mmap_mode='r')
        _require(self.rows.ndim==2 and self.rows.shape[1]==3,"rows must be episode/start/age triples")
        if verify_hash:_require(sha256_file(self.path.parent/m['rows_file'])==m['rows_sha256'],"row index changed")
        self.episodes=json.loads((self.path.parent/m['episodes_file']).read_text())
        stat_path=self.path.parent/m['source_file_stats']
        _require(sha256_file(stat_path)==m['source_file_stats_sha256'],'source stat sidecar changed')
        self.source_stats=json.loads(stat_path.read_text())
        _require(self.source_stats['identity_kind']=='resolved_path_size_mtime_ns_not_content_hash','unknown source stat semantics')
        self.text_root=self.path.parent/m.get('text_index','text/index.json')
        _require(self.text_root.is_file(),f'episode-specific frozen text cache missing: {self.text_root}')
        text_index=json.loads(self.text_root.read_text())
        if require_complete_text:_require(text_index.get('complete') is True,"complete episode text bank required for training")
        _require(text_index['identity']['prompt_manifest_sha256']==sha256_file(self.path.parent/'prompts.json'),"text bank binds a different prompt manifest")
        self.text_index=text_index['items'];self.text_episodes=text_index['episodes']
        for ep in self.episodes:
            _require(self.text_episodes[ep['episode_uid']]==ep['prompt_id'],"text bank episode binding mismatch")
        self.cache=OrderedDict();self.max_cached_episodes=max_cached_episodes
        self.teacher=None
        self.action_mean=torch.tensor(m['action_statistics']['mean']).float()
        self.action_std=torch.tensor(m['action_statistics']['std']).float().clamp_min(1e-6)

    def __len__(self):return len(self.rows)

    def _episode(self,index):
        ep=self.episodes[index]
        for key in ('raw','flow'):
            verify_file_stat(ep[key+'_path'],self.source_stats['episodes'][ep['episode_uid']][key])
        if index not in self.cache:
            _require(ep['split']==self.manifest['split'],"episode crosses split")
            records,actions=read_episode_records(ep['raw_path'],ep,self.manifest['visual_encoder_id'])
            with np.load(ep['flow_path']) as data:flow={k:data[k] for k in ('flow_rgb','motion_features','interval_valid','acceleration_valid','interval_start_time_seconds','interval_end_time_seconds')}
            self.cache[index]=(records,actions,flow,{})
            while len(self.cache)>self.max_cached_episodes:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self,index):
        import cv2,h5py
        cv2.setNumThreads(1)
        epi,grid,age_id=map(int,self.rows[index]);ep=self.episodes[epi]
        records,raw_actions,flow,visual_cache=self._episode(epi)
        t=records.timestamps.numpy();s=ep['first_time']+grid*self.manifest['source_stride_seconds'];q=s+self.manifest['query_ages_seconds'][age_id]
        source_frame=select_frame(t,s,past_only=True,tolerance=.060001)
        query_frame=select_frame(t,q,past_only=True,tolerance=.060001)
        starts=s+np.arange(16)*DT;fast_starts=q+np.arange(16)*DT
        source_action,sv,_=resample_actions(t,raw_actions.numpy(),starts+DT)
        fast_action,fv,_=resample_actions(t,raw_actions.numpy(),fast_starts+DT)
        _require(sv.all() and fv.all(),"index promised complete action support")
        source_action=(torch.from_numpy(source_action)-self.action_mean)/self.action_std
        fast_action=(torch.from_numpy(fast_action)-self.action_mean)/self.action_std
        future_requests=s+np.arange(1,9)*2*DT
        video_frames=[select_frame(t,x,past_only=False,tolerance=RGB_TOLERANCE) for x in future_requests]
        visual_requests=q+np.array([.1,.2,.4,.8])
        visual_frames=[]
        for target in visual_requests:
            try:visual_frames.append(select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE))
            except ValueError:visual_frames.append(-1)
        with h5py.File(ep['raw_path'],'r') as h:
            sm,smv,sma,smi,_,_,_=_flow_features(h,t,source_frame,flow,self.manifest['flow_contract'])
            motion,mv,ma,mi,spatial,spatial_valid,spatial_times=_flow_features(h,t,query_frame,flow,self.manifest['flow_contract'])
            rgb,pixelvalid=zip(*[_letterbox(_rgb(h,query_frame,c),tuple(self.manifest['rgb_size']),tuple(self.manifest['rgb_view_sizes'][v])) for v,c in enumerate(CAMERAS)])
            endpoints=[max(0,source_frame-k) for k in (36,24,12,0)]
            condition=np.stack([cv2.resize(flow['flow_rgb'][i],(320,384)) for i in endpoints]+[_composite(h,source_frame,(384,320))])
            future=np.stack([_composite(h,source_frame,(192,160))]+[_composite(h,i,(192,160)) for i in video_frames])
            needed=sorted(set(i for i in visual_frames if i>=0 and i not in visual_cache))
            if needed:
                if self.teacher is None:self.teacher=ProjectedContextTeacher(self.manifest['teacher_weights_path'],self.manifest['teacher_weights_sha256'])
                features,valid=self.teacher.encode(h,needed,records.entity_world_positions[needed].numpy())
                visual_cache.update({i:(features[j],valid[j]) for j,i in enumerate(needed)})
        features=torch.zeros_like(records.frozen_visual_features);context_valid=torch.zeros_like(records.visibility)
        for frame in set(i for i in visual_frames if i>=0):features[frame],context_valid[frame]=visual_cache[frame]
        r=replace(records,frozen_visual_features=features,projectability=context_valid,visual_context_valid=context_valid)
        labels=produce_future_labels(r,q,max_interpolation_gap=INTERPOLATION_GAP,max_contact_age=INTERPOLATION_GAP,duplicate_policy='last',max_visual_time_error=RGB_TOLERANCE)
        item=labels.tensors();item['future']=normalize_future(item['future'],item['future_loss_valid'],self.manifest['statistics'])
        textrow=self.text_index[ep['prompt_id']];p=(self.text_root.parent/textrow['path']).resolve()
        _require(textrow['text_sha256']==ep['prompt_id'],"episode prompt identity mismatch")
        _require(sha256_file(p)==textrow['sha256'],"text cache hash mismatch")
        language=torch.load(p,map_location='cpu',weights_only=True)
        _require(language['context'].shape==(512,4096) and language['mask'].shape==(512,),"text cache shape mismatch")
        _require(language['mask'].dtype==torch.bool and language['mask'].any(),"invalid language mask")
        source_known=torch.arange(16)<2;fast_known=torch.arange(16)<2
        source_intervals=torch.from_numpy(np.stack([starts,starts+DT],-1))
        future_intervals=torch.tensor([[s,s+4*2*DT],[s+4*2*DT,s+8*2*DT]],dtype=torch.float64)
        # Source cache is causal: second encoded slice also depends on the first
        # flow slice through codec history. Support reflects the union, not just
        # the four newly consumed inputs. Type distinguishes flow and RGB mixes.
        history_start=float(smi[smv,0].min()) if bool(smv.any()) else float(t[source_frame])
        temporal=torch.tensor([[float(smi[0,0]),float(smi[0,1])],[history_start,float(t[source_frame])]],dtype=torch.float64)
        # Flow-backed slice is head-view. Second slice's new inputs are 3 flow
        # images and one RGB composite; these are input-support overlap fractions.
        source_overlap=torch.cat([view_overlap(12,10,flow_weight=1.),view_overlap(12,10,flow_weight=.75)])
        source_valid=torch.cat([torch.full((120,),bool(smv[0])),torch.ones(120,dtype=torch.bool)])
        item.update(source_condition_frames=torch.from_numpy(condition).permute(3,0,1,2).float()/127.5-1,
                    source_cache_id=torch.tensor([epi,grid],dtype=torch.long),
                    source_future_frames=torch.from_numpy(future).permute(3,0,1,2).float()/127.5-1,
                    source_initial_state=(raw_actions[source_frame]-self.action_mean)/self.action_std,
                    source_motion=sm,source_motion_valid=smv,source_motion_acceleration_valid=sma,
                    source_text=language['context'],source_action=source_action,fast_action=fast_action,
                    source_future_intervals=future_intervals,source_action_times=torch.from_numpy(starts),
                    source_action_intervals=source_intervals,source_temporal_support=temporal,
                    source_view_overlap=source_overlap,source_token_valid=source_valid,
                    source_input_type=torch.tensor([1,2]),future_view_overlap=view_overlap(6,5).repeat(2,1),
                    future_token_valid=torch.ones(60,dtype=torch.bool),future_input_type=torch.zeros(2,dtype=torch.long),
                    source_known_actions=torch.where(source_known[:,None],source_action,0),source_known_mask=source_known,
                    fast_known_actions=torch.where(fast_known[:,None],fast_action,0),fast_known_mask=fast_known,
                    rgb=torch.stack(rgb),rgb_pixel_valid=torch.stack(pixelvalid),rgb_times=torch.full((3,),t[query_frame],dtype=torch.float64),
                    motion=motion,motion_times=mi,motion_valid=mv,motion_acceleration_valid=ma,
                    spatial_flow=spatial,spatial_flow_valid=spatial_valid,spatial_flow_times=spatial_times,
                    state=(raw_actions[query_frame]-self.action_mean)/self.action_std,state_time=torch.tensor(t[query_frame],dtype=torch.float64),
                    text=language['context'],text_valid=language['mask'],source_time=torch.tensor(s,dtype=torch.float64),
                    completed_time=torch.tensor(s+min(.18,q-s),dtype=torch.float64),query_time=torch.tensor(q,dtype=torch.float64),
                    action_times=torch.from_numpy(fast_starts),action_intervals=torch.from_numpy(np.stack([fast_starts,fast_starts+DT],-1)),
                    query_entity_valid=torch.ones(4,dtype=torch.bool),
                    executed_actions=source_action.clone(),executed_action_intervals=source_intervals.clone(),
                    executed_action_valid=source_intervals[:,1]<=q+1e-8,
                    source_action_valid=torch.from_numpy(sv),fast_action_valid=torch.from_numpy(fv),
                    source_rgb_actual_times=torch.tensor(t[source_frame],dtype=torch.float64),
                    future_rgb_actual_times=torch.from_numpy(t[video_frames].copy()),future_rgb_time_errors=torch.from_numpy(t[video_frames]-future_requests),
                    future_visibility_known=torch.zeros(16,dtype=torch.bool),
                    future_visual_actual_times=torch.tensor([t[i] if i>=0 else q for i in visual_frames],dtype=torch.float64).repeat(4))
        item['executed_actions'][~item['executed_action_valid']]=0
        return item


@torch.no_grad()
def materialize_native_latents(batch, vae, *, device=None):
    """Frozen native VAE preprocessing in caller-owned main process; never random.

    ``vae`` is the official WanVAE wrapper exposing encode(list[CTHW]). The
    wrapper handles its native normalization. Full future input is 1+8 frames;
    retaining latent slices 1: preserves both groups of four future frames.
    """
    if 'source_condition_latent' in batch and 'source_future_latent' in batch:return batch
    _require('source_condition_frames' in batch and 'source_future_frames' in batch,'real native RGB frames missing')
    conditions=batch['source_condition_frames'];future=batch['source_future_frames']
    _require(conditions.ndim==future.ndim==5 and conditions.shape[2]==5 and future.shape[2]==9,"incorrect native frame/codec layout")
    model=getattr(vae,'model',vae)
    if isinstance(model,torch.nn.Module):model.eval()
    def encode(x):
        if device is not None:x=x.to(device)
        z=vae.encode(list(x))
        return torch.stack(z) if isinstance(z,(list,tuple)) else z
    result=dict(batch)
    result['source_condition_latent']=encode(conditions)
    all_future=encode(future)
    _require(all_future.ndim==5 and all_future.shape[2]==3,'native 1+8 frame codec must produce three slices')
    result['source_future_latent']=all_future[:,:,1:]
    return result


def write_projection_overlays(raw_root, output, tasks=('adjust_bottle','place_object_basket','scan_object')):
    """CPU visual audit of real world->CV projection, never a visibility test."""
    import cv2,h5py
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for task in tasks:
        path=Path(raw_root)/'clean'/task/'absolute_motion_clean/data/episode0.hdf5'
        with h5py.File(path,'r') as h:
            frame=len(h['interception/sim_time_seconds'])//2
            points=np.concatenate([h['interception/target_pose_wxyz'][frame,:3][None],h['interception/ee_poses_wxyz'][frame,:,:3]])
            for camera in CAMERAS:
                image=_rgb(h,frame,camera)
                ext=h[f'observation/{camera}/extrinsic_cv'][frame];intr=h[f'observation/{camera}/intrinsic_cv'][frame]
                for name,p,color in zip(('target','left-ee','right-ee'),points,((255,60,60),(60,255,60),(60,100,255))):
                    xyz=ext@np.r_[p,1.]
                    if xyz[2]<=1e-6:continue
                    uv=intr@xyz;uv=np.rint(uv[:2]/uv[2]).astype(int);x,y=map(int,uv)
                    if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
                        cv2.rectangle(image,(x-16,y-16),(x+16,y+16),color,1)
                        cv2.drawMarker(image,(x,y),color,cv2.MARKER_CROSS,8,1)
                        cv2.putText(image,name,(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
                cv2.putText(image,f'{task} frame={frame} {camera} ROI=context, occlusion unknown',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                target=output/f'{task}_{camera}.jpg';cv2.imwrite(str(target),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));paths.append(str(target))
    (output/'index.json').write_text(json.dumps(paths,indent=2))
    return paths


def build_manifests(config, output, *, max_episodes=None):
    """Read every real episode; save compact indices, prompt bindings and train stats.

    This performs CPU metadata/teacher work only. Existing files are never
    overwritten. VAE/text preprocessing is a separate caller-owned operation.
    """
    import h5py
    output=Path(output).resolve();_require(not output.exists(),"output directory already exists")
    output.mkdir(parents=True)
    packed=Path(config['packed_root']);raw=Path(config['raw_root']);flowroot=Path(config['flow_root'])
    native=json.loads((packed/'dataset.json').read_text())
    episodes=[];prompts=[];rows={'train':[],'validation':[]};audit={'episodes':0,'source_windows':0,'rows':0,'rejected_action_support':0,'rejected_rgb_tolerance':0,'old_exact_grid_windows':0,'duplicates':0,'by_task':{}}
    teacher=ProjectedContextTeacher(config['teacher_weights_path'],config['teacher_weights_sha256'])
    count=torch.zeros(134,dtype=torch.float64);total=count.clone();total2=count.clone();fitted=set()
    input_episodes=[json.loads(line) for line in (packed/'episodes.jsonl').read_text().splitlines()]
    if max_episodes is not None:input_episodes=input_episodes[:max_episodes]
    scene_cache={};seeds_cache={};visual_fit_tasks=set()
    for native_ep in input_episodes:
        style,task,name=native_ep['split'],native_ep['task_name'],native_ep['episode_name'];number=int(name)
        parent=raw/style/task/('absolute_motion_'+style);p=parent/'data'/f'episode{number}.hdf5'
        key=(style,task)
        if key not in scene_cache:
            scene_cache[key]=json.loads((parent/'scene_info.json').read_text());seeds_cache[key]=list(map(int,(parent/'seed.txt').read_text().split()))
        seed=seeds_cache[key][number]
        # Same task+seed always shares split across styles, even if accepted
        # episode numbering differs after collection success/failure filtering.
        scene_id=f'{task}:seed:{seed}'
        split='validation' if int(hashlib.sha256(scene_id.encode()).hexdigest()[:8],16)%10==0 else 'train'
        uid=f'{style}/{task}/{number}';info=scene_cache[key][f'episode_{number}']['info']
        text,provenance=episode_prompt(task,info,config['description_root']);prompt_id=hashlib.sha256(text.encode()).hexdigest()
        ep=dict(episode_uid=uid,scene_id=scene_id,split=split,task=task,style=style,seed=seed,raw_path=str(p),prompt_id=prompt_id,
                flow_path=str(flowroot/style/task/'videos'/f'{number}.flow.npz'),relevant_gripper=1 if info.get('{a}')=='right' else 0)
        records,actions=read_episode_records(p,ep,teacher.identity);t=records.timestamps.numpy();ep['first_time']=float(t[0]);ep['last_time']=float(t[-1]);ep['frame_count']=len(t)
        idx=len(episodes);episodes.append(ep)
        prompts.append(dict(prompt_id=prompt_id,text=text,episode_uid=uid,task=task,scene_id=scene_id,source_sha256=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest(),sources=provenance,scene_info_path=str(parent/'scene_info.json'),scene_info_sha256=sha256_file(parent/'scene_info.json'),bindings=info))
        audit['duplicates']+=int((np.diff(t)==0).sum());audit['episodes']+=1
        old=sum(np.max(np.abs(t[s:s+49]-t[s]-np.arange(49)*.06))<=1e-6 for s in range(max(0,len(t)-48)));audit['old_exact_grid_windows']+=int(old)
        grid_count=max(0,int(math.floor((t[-1]-t[0]-16*DT)/.36))+1)
        taskcounts=audit['by_task'].setdefault(task,dict(episodes=0,rows=0));taskcounts['episodes']+=1
        for grid in range(grid_count):
            s=t[0]+grid*.36;audit['source_windows']+=1
            try:
                select_frame(t,s,past_only=True,tolerance=INTERPOLATION_GAP)
                for target in s+np.arange(1,9)*2*DT:select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE)
            except ValueError:audit['rejected_rgb_tolerance']+=1;continue
            _,sv,_=resample_actions(t,actions.numpy(),s+np.arange(1,17)*DT)
            if not sv.all():audit['rejected_action_support']+=1;continue
            for ai,age in enumerate(QUERY_AGES):
                q=s+age
                if q+16*DT>t[-1]+1e-8:continue
                _,fv,_=resample_actions(t,actions.numpy(),q+np.arange(1,17)*DT)
                if not fv.all():audit['rejected_action_support']+=1;continue
                rows[split].append([idx,grid,ai]);audit['rows']+=1;taskcounts['rows']+=1
        if split=='train':
            # Geometry normalization sees all training episodes; visual fit uses
            # a named actual-frame subset (one episode per task), never held-out.
            feat=records.frozen_visual_features.clone();vm=torch.zeros_like(records.visibility)
            if task not in visual_fit_tasks:
                frames=np.unique(np.linspace(0,len(t)-1,min(8,len(t)),dtype=int))
                with h5py.File(p,'r') as h:f,v=teacher.encode(h,frames,records.entity_world_positions[frames].numpy())
                feat[frames]=f;vm[frames]=v;visual_fit_tasks.add(task)
            x=torch.zeros(len(t),4,134);mask=torch.zeros_like(x,dtype=torch.bool)
            x[:,:,:128]=feat;mask[:,:,:128]=vm[:,:,None]
            x[:,:,128:131]=records.entity_world_positions;mask[:,:,128:131]=records.entity_valid[:,:,None]
            x[:,0,131:134]=records.entity_world_positions[:,0]-records.gripper_positions[:,ep['relevant_gripper']];mask[:,0,131:134]=True
            x=x.reshape(-1,134).double();mask=mask.reshape(-1,134);x=torch.where(mask,x,0)
            count+=mask.sum(0);total+=x.sum(0);total2+=(x*x).sum(0);fitted.add(scene_id)
        if audit['episodes']%100==0:print(json.dumps({'episodes':audit['episodes'],'rows':audit['rows']}),flush=True)
    split_identity=hashlib.sha256(json.dumps([(e['episode_uid'],e['scene_id'],e['split']) for e in episodes],sort_keys=True).encode()).hexdigest()
    mean=total/count.clamp_min(1);variance=(total2/count.clamp_min(1)-mean.square()).clamp_min(0);std=torch.where(variance>1e-12,variance.sqrt(),torch.ones_like(variance))
    statistics=dict(fit_split='train',visual_encoder_id=teacher.identity,split_manifest_sha256=split_identity,mean=mean.tolist(),std=std.tolist(),count=count.tolist(),
                    train_episode_ids=[e['episode_uid'] for e in episodes if e['split']=='train'],fitted_scene_ids=sorted(fitted),
                    fit_method='all_train_raw_geometry_and_eight_actual_frames_first_train_episode_per_task_visual')
    (output/'episodes.json').write_text(json.dumps(episodes,indent=2));(output/'prompts.json').write_text(json.dumps({'schema_version':1,'text_semantics':'exact_encoder_input_with_scene_prefix','scene_prefix':SCENE_PREFIX,'prompts':prompts},indent=2))
    (output/'audit.json').write_text(json.dumps(audit,indent=2));(output/'statistics.json').write_text(json.dumps(statistics,indent=2))
    torch.save({'projection':teacher.projection},output/'visual_projection.pt')
    field_support=torch.zeros(4,4,134,dtype=torch.bool)
    field_support[0]=True;field_support[2:,:,:131]=True
    contact_support=torch.zeros(4,4,2,dtype=torch.bool);contact_support[0]=True
    for split,items in rows.items():
        if not items:continue
        np.save(output/f'{split}_rows.npy',np.asarray(items,dtype=np.int64))
        manifest={**config,'schema_version':SCHEMA,'split':split,'episodes_file':'episodes.json','rows_file':f'{split}_rows.npy','rows_sha256':sha256_file(output/f'{split}_rows.npy'),
                  'split_manifest_sha256':split_identity,'statistics':statistics,'visual_encoder_id':teacher.identity,'visual_projection_sha256':teacher.projection_sha256,
                  'visual_target_semantics':'projected_local_context_not_visible_object','occlusion_status':'unknown_no_segmentation_or_depth','support_pose_status':'unavailable_masked',
                  'future_field_support':field_support.reshape(16,134).tolist(),'contact_field_support':contact_support.reshape(16,2).tolist(),
                  'source_stride_seconds':.36,'query_ages_seconds':list(QUERY_AGES),'action_interval_seconds':DT,'codec_id':CODEC_ID,
                  'rgb_size':config.get('rgb_size',[240,320]),'rgb_view_sizes':config.get('rgb_view_sizes',[[240,320],[120,160],[120,160]]),
                  'flow_contract':native['absolute_motion']['flow_contract'],'motion_representation':'raw_absolute_motion_v2','action_representation':'checkpoint_mean_std',
                  'action_resampling':'bounded_linear_arm_drive_targets_previous_event_gripper_last_duplicate',
                  'future_rgb_rule':'nearest_real_frame_abs_error_le_0.030001','condition_rgb_rule':'latest_at_or_before_source_or_query_max_age_0.060001',
                  'latency_provenance':'synthetic_completion_0.18s_not_measured','time_grid_cache_rebuilt':True,'formal_training_ready':False,
                  'text_index':'text/index.json','native_latents':'main_process_frozen_native_vae_or_identity_bound_cache','audit_file':'audit.json'}
        (output/f'{split}.json').write_text(json.dumps(manifest,indent=2))
    attach_stat_identity(output)
    return audit
