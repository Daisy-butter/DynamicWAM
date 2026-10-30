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
from .exposure_timing import TIME_CONTRACT, load_templates, validate_template

SCHEMA = "dualrate-domino-lazy-v2"
CODEC_ID = "wan22_first_plus_four_causal_native_v2"
CODEC_TEMPORAL_CONTRACT = {
    "future_interval_semantics": "nominal_requested_forecast_target_bins_not_causal_dependency",
    "future_input_frames": "source_RGB_index_0_then_eight_requested_future_RGB_frames_1_through_8",
    "retained_future_vae_slices": [1, 2],
    "retained_future_causal_input_indices": [list(range(5)), list(range(9))],
    "causal_dependency_note": "second_retained_slice_can_depend_on_source_and_all_eight_future_inputs_via_encoder_cache",
    "hidden_dependency_note": "native_DiT_joint_global_attention_mixes_video_and_action_tokens_not_local_causal_features",
    "readout_mask_note": "target_bins_limit_direct_reads_not_all_indirect_information_paths",
    "source_view_fraction_semantics": "new_input_image_area_fractions_not_exact_neural_receptive_field",
}
CAMERAS = ("head_camera", "left_camera", "right_camera")
DT = .18
RGB_TOLERANCE = .030001
INTERPOLATION_GAP = .060001
RAW_FRAME_SECONDS = .06
# Float noise only. It is not part of the physical delay and must stay far
# below one raw frame, so a 60 ms grid cannot step back an extra sample.
NUMERIC_TIME_EPSILON = 1e-6
# Readback measured 2026-10-24 is about two raw frames (smoke04 median 106 ms,
# 1x probe median 112 ms). The delay is exactly that span. Every stored episode
# then steps 4 ms once and captures every 60 ms, while the query grid is
# anchored at the episode origin. The ideal exposure therefore falls up to one
# capture after the latest readable frame (measured gap 56 ms, never two
# captures). One raw frame is the match tolerance. A skipped capture stays out.
CAMERA_AVAILABILITY_SECONDS = 2 * RAW_FRAME_SECONDS
CONDITION_FRAME_TOLERANCE_SECONDS = RAW_FRAME_SECONDS + 1e-6
# After it becomes readable, the same image remains the latest one for about
# one more capture. Four raw frames is that age ceiling, not a closed-loop
# action deadline.
CONDITION_RGB_MAX_AGE_SECONDS = 4 * RAW_FRAME_SECONDS
FLOW_HISTORY_OFFSETS_FRAMES = (60, 48, 36, 24, 12, 0)
FLOW_ENDPOINT_TOLERANCE_SECONDS = RAW_FRAME_SECONDS
QUERY_AGES = (.36, .72)
LEGACY_ENTITY_ROLES = ('tracked_dynamic_target','support','left_gripper','right_gripper')
ACTIVE_ENTITY_ROLES = ('tracked_dynamic_target','left_gripper','right_gripper')
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


def bound_metadata_path(manifest, root, key, default=None):
    """Resolve and verify immutable metadata even when row hashing is disabled."""
    value=manifest.get(key,default)
    _require(isinstance(value,str) and bool(value),f'missing metadata path {key}')
    path=(Path(root)/value).resolve(strict=True)
    digest=manifest.get(key+'_sha256')
    if manifest.get('metadata_binding_schema')=='dualrate-static-metadata-v1':
        _require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) is not None,
                 f'missing metadata hash {key}')
    if digest is not None:
        _require(sha256_file(path)==digest,f'{key} metadata hash changed')
    return path


def role_token_indices(manifest):
    """Explicit model role selection, independent of future GT availability."""
    roles=tuple(manifest.get('active_entity_roles',LEGACY_ENTITY_ROLES))
    _require(roles in (LEGACY_ENTITY_ROLES,ACTIVE_ENTITY_ROLES),'unverified future entity role layout')
    indices=[4*LEGACY_ENTITY_ROLES.index(role)+h for role in roles for h in range(4)]
    if 'future_token_indices' in manifest:
        _require(manifest['future_token_indices']==indices,'role/token identity mismatch')
    return roles,torch.tensor(indices,dtype=torch.long)


def _tracked_actor_provenance(episodes):
    """Read the recorded target identity; never infer its pose from a scene seed."""
    import pickle
    result=[]
    for index,episode in enumerate(episodes):
        episode=dict(episode)
        raw=Path(episode['raw_path']).resolve(strict=True)
        trajectory=raw.parents[1]/'_traj_data'/(raw.stem+'.pkl')
        payload=trajectory.read_bytes()
        recorded=pickle.loads(payload)
        actor=recorded.get('dynamic_motion_info',{}).get('target_actor_name')
        _require(isinstance(actor,str) and bool(actor),'recorded tracked target actor identity missing')
        episode.update(tracked_actor_name=actor,tracked_actor_identity_source=str(trajectory),
                       tracked_actor_identity_source_sha256=hashlib.sha256(payload).hexdigest(),
                       tracked_pose_field='interception/target_pose_wxyz')
        result.append(episode)
        if (index+1)%1000==0:
            print(json.dumps({'tracked_actor_identities':index+1}),flush=True)
    return result


def bind_metadata_manifests(source, output, *, active_entity_roles=None,
                            capture_tracked_actor_identity=False):
    """Create small V3 manifests referring to unchanged V2 data/cache files.

    No raw/flow/text payload is copied or modified. The new manifests bind the
    episode clocks/roles and the complete text-index mapping as immutable bytes.
    """
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    _require(not output.exists(),'metadata binding output already exists')
    manifests={};bound_episodes=None
    if active_entity_roles is not None:
        roles,indices=role_token_indices({'active_entity_roles':list(active_entity_roles)})
        _require(roles==ACTIVE_ENTITY_ROLES,'V3 production binding requires three supervised roles')
        _require(capture_tracked_actor_identity,'V3 roles require recorded target identity provenance')
        bound_episodes=_tracked_actor_provenance(json.loads((source/'episodes.json').read_text()))
    for split in ('train','validation'):
        parent=source/f'{split}.json';m=json.loads(parent.read_text())
        _require(m['schema_version']==SCHEMA and m['split']==split,'wrong parent manifest')
        for key in ('rows_file','episodes_file','source_file_stats','text_index'):
            path=(source/m[key]).resolve(strict=True)
            m[key]=str(path)
        prompts=(source/m.get('prompts_file','prompts.json')).resolve(strict=True)
        m['prompts_file']=str(prompts)
        _require(sha256_file(Path(m['rows_file']))==m['rows_sha256'],'parent rows changed')
        _require(sha256_file(Path(m['source_file_stats']))==m['source_file_stats_sha256'],'parent stat sidecar changed')
        for key in ('episodes_file','text_index','prompts_file'):
            m[key+'_sha256']=sha256_file(Path(m[key]))
        text=json.loads(Path(m['text_index']).read_text())
        _require(text.get('complete') is True,'complete parent text bank required')
        _require(text['identity']['prompt_manifest_sha256']==m['prompts_file_sha256'],
                 'parent text bank prompt manifest mismatch')
        if m.get('audit_file'):
            m['audit_file']=str((source/m['audit_file']).resolve(strict=True))
        m.update(metadata_binding_schema='dualrate-static-metadata-v1',
                 parent_manifest=str(parent),parent_manifest_sha256=sha256_file(parent))
        if active_entity_roles is not None:
            m.update(active_entity_roles=list(roles),future_token_indices=indices.tolist(),
                label_entity_roles=list(LEGACY_ENTITY_ROLES),future_role_schema='tracked-target-hands-v1',
                future_field_support=[m['future_field_support'][i] for i in indices.tolist()],
                contact_field_support=[m['contact_field_support'][i] for i in indices.tolist()],
                omitted_role_reason='second_related_or_support_entity_has_no_recorded_trajectory',
                visual_target_semantics='projected_local_image_context_including_possible_occluders',
                visual_validity_semantics='actual_frame_and_projection_valid_not_true_object_visibility',
                occlusion_status='not_a_predicted_head_or_visibility_label',
                flow_raw_stride_frames=12,flow_raw_offsets_frames=[48,36,24,12,0],
                flow_acceleration_previous_interval='raw_frame_index_minus_12_actual_center_time_difference',
                statistics_reuse='same_134_field_train_statistics_missing_legacy_support_never_contributed')
        manifests[split]=m
    output.mkdir(parents=True)
    if bound_episodes is not None:
        episode_path=output/'episodes.json'
        episode_path.write_text(json.dumps(bound_episodes,indent=2))
        for m in manifests.values():
            m.update(episodes_file=str(episode_path),episodes_file_sha256=sha256_file(episode_path))
    result={}
    for split,m in manifests.items():
        path=output/f'{split}.json';path.write_text(json.dumps(m,indent=2))
        result[split]=dict(path=str(path),sha256=sha256_file(path),
                           episodes_file_sha256=m['episodes_file_sha256'],
                           text_index_sha256=m['text_index_sha256'],
                           prompts_file_sha256=m['prompts_file_sha256'])
    return result


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


def _nearest_index(times, target, *, anchor, tolerance):
    """Nearest real row to target at or before anchor. None when none is close enough."""
    best = None
    best_error = None
    values = np.asarray(times, dtype=np.float64)
    limit = float(values[int(anchor)]) + NUMERIC_TIME_EPSILON
    for index in unique_last_indices(values):
        index = int(index)
        if index > int(anchor) or float(values[index]) > limit:
            break
        error = abs(float(values[index]) - float(target))
        if error > float(tolerance) + NUMERIC_TIME_EPSILON:
            continue
        if best is None or error < best_error - NUMERIC_TIME_EPSILON or (
                abs(error - best_error) <= NUMERIC_TIME_EPSILON and index > best):
            best, best_error = index, error
    return best


def available_condition_frame(times, event_time, *, delay=CAMERA_AVAILABILITY_SECONDS,
                              tolerance=CONDITION_FRAME_TOLERANCE_SECONDS):
    """Real frame exposed one camera delay before the event.

    Require exposure + delay <= event (apart from numerical roundoff).
    The match tolerance bounds how much older the selected real frame may be;
    it never permits an unavailable newer frame. Before that delay has elapsed,
    the explicit pre-activation bootstrap keeps the earliest real frame.
    """
    event, delay = float(event_time), float(delay)
    _require(math.isfinite(event) and math.isfinite(delay) and delay >= 0 and tolerance > 0,
             "camera availability delay must be finite and non-negative")
    values = np.asarray(times, dtype=np.float64)
    indices = unique_last_indices(values)
    _require(len(indices) > 0, "no camera timestamps")
    earliest = int(indices[0])
    target = event - delay
    if target < float(values[earliest]) - NUMERIC_TIME_EPSILON:
        _require(float(values[earliest]) <= event + NUMERIC_TIME_EPSILON, "no real frame before the event")
        return earliest
    candidates = indices[values[indices] <= target + NUMERIC_TIME_EPSILON]
    _require(len(candidates) > 0, "no camera frame available before the event")
    found = int(candidates[-1])
    _require(target - float(values[found]) <= tolerance + NUMERIC_TIME_EPSILON,
             "no real frame within requested tolerance")
    return found


def flow_history_selection(times, frame, *, tolerance=FLOW_ENDPOINT_TOLERANCE_SECONDS,
                           interval_seconds=12*RAW_FRAME_SECONDS):
    """Physical-time flow endpoints, with an explicit match mask.

    Offset 0 is the anchor frame. Every earlier offset is the raw sample nearest
    its 60 ms target. A miss stays masked; the anchor is not reused as that sample.
    """
    values = np.asarray(times, dtype=np.float64)
    anchor = int(frame)
    _require(0 <= anchor < len(values), "flow anchor is outside the episode")
    anchor_time = float(values[anchor])
    indices, matched = [], []
    for offset in FLOW_HISTORY_OFFSETS_FRAMES:
        if offset == 0:
            indices.append(anchor)
            matched.append(True)
            continue
        found = _nearest_index(values, anchor_time - (offset/12)*interval_seconds, anchor=anchor, tolerance=tolerance)
        indices.append(anchor if found is None else int(found))
        matched.append(found is not None)
    return indices, matched


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


def validate_completion_distribution(value, *, verify_evidence=True):
    """A declared latency distribution, never a disguised clipped timestamp."""
    _require(isinstance(value,dict),'explicit completion distribution required')
    result=dict(value)
    _require(result.get('schema')=='dualrate-completion-discrete-v1','unknown completion distribution')
    _require(result.get('kind') in ('synthetic','measured_quantiles'),'unknown completion provenance')
    _require(result.get('clock_domain')=='simulation_seconds','latencies must use the physical model clock')
    _require(result.get('selection')=='row_hash_uniform_conditioned_available','unknown latency selection')
    values=result.get('latencies_seconds',[])
    _require(bool(values) and all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in values),
             'completion latencies must be positive finite seconds')
    _require(values==sorted(values),'completion quantiles must be ordered')
    _require(isinstance(result.get('seed'),int),'completion selection seed required')
    _require(isinstance(result.get('provenance'),str) and bool(result['provenance']),'completion provenance explanation required')
    if result['kind']=='measured_quantiles':
        levels=result.get('quantile_levels',[])
        _require(len(levels)==len(values) and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in levels)
                 and all(b>a for a,b in zip(levels,levels[1:])),'ordered measured quantile levels required')
        path=Path(result.get('trace_path',''))
        digest=result.get('trace_sha256','')
        _require(path.is_absolute() and re.fullmatch('[0-9a-f]{64}',digest) is not None,
                 'measured latencies require an absolute trace and SHA256')
        if verify_evidence:
            _require(sha256_file(path)==digest,'measured completion trace changed')
    return result


def manifest_completion_distribution(manifest):
    value=manifest.get('completion_distribution')
    if value is None:
        _require('timing_derivation' not in manifest,'derived timing manifest lost completion distribution')
        value=dict(schema='dualrate-completion-discrete-v1',kind='synthetic',clock_domain='simulation_seconds',
                   selection='row_hash_uniform_conditioned_available',latencies_seconds=[.18],seed=0,
                   provenance='legacy_v2_v3_full01_fixed_0.18s_not_measured')
    result=validate_completion_distribution(value)
    for age in manifest.get('query_ages_seconds',QUERY_AGES):
        _require(math.isfinite(age) and age>0,'positive finite query ages required')
        completion_latency_seconds(result,'availability_validation',0,age)
    return result


def completion_latency_seconds(distribution, episode_uid, source_grid, query_age):
    """Deterministic conditional selection; impossible references are rejected."""
    available=[float(x) for x in distribution['latencies_seconds'] if float(x)<=float(query_age)]
    _require(bool(available),'no completed reference is available at this query age')
    identity=f"{distribution['seed']}:{episode_uid}:{int(source_grid)}:{float(query_age).hex()}"
    index=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big')%len(available)
    return available[index]


def _action_targets_supported(times, targets, max_gap=INTERPOLATION_GAP):
    """The same exact/bracket guard as resample_actions, without encoding labels."""
    t=np.asarray(times,dtype=np.float64)[unique_last_indices(times)]
    requested=np.asarray(targets,dtype=np.float64)
    hi=np.searchsorted(t,requested,side='left');lo=hi-1
    hsafe=np.minimum(hi,len(t)-1);lsafe=np.maximum(lo,0)
    exact=((hi<len(t)) & (np.abs(t[hsafe]-requested)<=1e-8)) | ((lo>=0) & (np.abs(t[lsafe]-requested)<=1e-8))
    bracket=(lo>=0) & (hi<len(t)) & ((t[hsafe]-t[lsafe])<=max_gap)
    return exact | bracket


def eligible_timing_rows(times, source_stride, query_ages):
    """Return source-grid/age IDs with real condition and complete A/video support."""
    t=np.asarray(times,dtype=np.float64);unique_last_indices(t)
    _require(math.isfinite(source_stride) and source_stride>0,'positive source stride required')
    _require(bool(query_ages) and all(math.isfinite(age) and age>0 for age in query_ages),'positive query ages required')
    count=max(0,int(math.floor((t[-1]-t[0]-16*DT)/source_stride))+1)
    rows=[]
    for grid in range(count):
        source=float(t[0]+grid*source_stride)
        try:
            available_condition_frame(t,source)
            for target in source+np.arange(1,9)*2*DT:
                select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE)
        except ValueError:
            continue
        if not _action_targets_supported(t,source+np.arange(1,17)*DT).all():
            continue
        for age_id,age in enumerate(query_ages):
            query=source+age
            if query+16*DT>t[-1]+1e-8:
                continue
            try:
                available_condition_frame(t,query)
            except ValueError:
                continue
            if _action_targets_supported(t,query+np.arange(1,17)*DT).all():
                rows.append((grid,age_id))
    return rows


def derive_timing_manifests(source, output, *, query_ages_seconds, completion_distribution):
    """CPU-only timing-index derivation; preserve all raw labels and split/stat identities.

    Invoke only after selecting the explicit synthetic or measured distribution.
    Reads timestamps, never generates image/latent labels or runs a simulator.
    """
    import h5py
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    _require(not output.exists(),'timing derivation output already exists')
    ages=[float(x) for x in query_ages_seconds]
    _require(bool(ages) and ages==sorted(set(ages)),'query ages must be ordered and distinct')
    distribution=validate_completion_distribution(completion_distribution)
    for age in ages:
        _require(math.isfinite(age) and age>0 and abs(age/DT-round(age/DT))<=1e-7,
                 'query ages must be positive aligned physical action-grid multiples')
        completion_latency_seconds(distribution,'availability_validation',0,age)
    parents={split:json.loads((source/f'{split}.json').read_text()) for split in ('train','validation')}
    m=parents['train']
    _require(m.get('metadata_binding_schema')=='dualrate-static-metadata-v1','bound parent metadata required')
    episode_path=bound_metadata_path(m,source,'episodes_file')
    episodes=json.loads(episode_path.read_text())
    stat_path=(source/m['source_file_stats']).resolve(strict=True)
    _require(sha256_file(stat_path)==m['source_file_stats_sha256'],'parent source stat sidecar changed')
    for parent in parents.values():
        _require(parent.get('metadata_binding_schema')=='dualrate-static-metadata-v1' and
                 parent['episodes_file_sha256']==m['episodes_file_sha256'] and
                 parent['source_stride_seconds']==m['source_stride_seconds'] and
                 parent['source_file_stats_sha256']==m['source_file_stats_sha256'],
                 'parent split metadata mismatch')
        for key in ('episodes_file','text_index','prompts_file'):
            bound_metadata_path(parent,source,key)
    stats=json.loads(stat_path.read_text())['episodes'];rows={'train':[],'validation':[]}
    for index,episode in enumerate(episodes):
        verify_file_stat(episode['raw_path'],stats[episode['episode_uid']]['raw'])
        with h5py.File(episode['raw_path'],'r') as handle:
            times=handle['interception/sim_time_seconds'][:]
        _require(float(times[0])==episode['first_time'] and float(times[-1])==episode['last_time'],
                 'episode metadata clock endpoints changed')
        split=episode['split'];_require(split in rows,'unknown parent split')
        rows[split].extend((index,grid,age) for grid,age in eligible_timing_rows(times,m['source_stride_seconds'],ages))
        if (index+1)%1000==0:
            print(json.dumps({'timing_episodes':index+1,'rows':{k:len(v) for k,v in rows.items()}}),flush=True)
    _require(all(rows.values()),'timing derivation lost a complete split')
    output.mkdir(parents=True)
    audit=dict(schema='dualrate-timing-index-derivation-v1',episodes=len(episodes),query_ages_seconds=ages,
               completion_distribution=distribution,raw_images_encoded=0,simulator_started=False,
               row_counts={key:len(value) for key,value in rows.items()},
               availability_semantics='latency distribution conditioned on completion at or before query; never clipped')
    (output/'timing_audit.json').write_text(json.dumps(audit,indent=2))
    result={}
    for split,parent in parents.items():
        new=dict(parent)
        for key in ('episodes_file','text_index','prompts_file','source_file_stats'):
            new[key]=str((source/parent[key]).resolve(strict=True))
        array=output/f'{split}_rows.npy';np.save(array,np.asarray(rows[split],dtype=np.int64))
        new.update(rows_file=str(array),rows_sha256=sha256_file(array),query_ages_seconds=ages,
                   completion_distribution=distribution,latency_provenance=distribution['kind']+':'+distribution['provenance'],
                   camera_availability_seconds=CAMERA_AVAILABILITY_SECONDS,
                   condition_rgb_max_age_seconds=CONDITION_RGB_MAX_AGE_SECONDS,
                   condition_rgb_rule='latest_real_frame_at_or_before_event_minus_camera_availability',
                   parent_manifest=str(source/f'{split}.json'),parent_manifest_sha256=sha256_file(source/f'{split}.json'),
                   audit_file=str(output/'timing_audit.json'),
                   timing_derivation='real_timestamp_complete_action_and_native_video_support_v1')
        path=output/f'{split}.json';path.write_text(json.dumps(new,indent=2))
        result[split]=dict(path=str(path),sha256=sha256_file(path),rows=len(rows[split]))
    return dict(manifests=result,audit=audit)


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
                     ('tracked:'+row.get('tracked_actor_name','dynamic_target'),None,'left_gripper','right_gripper'),times,xyz,hands,
                     torch.zeros(len(times),4,128),entity_valid,torch.ones(len(times),2,dtype=torch.bool),
                     torch.zeros(len(times),4,dtype=torch.bool),torch.zeros(len(times),4,dtype=torch.bool),
                     torch.tensor([0,-1,2,3]).repeat(len(times),1),contacts,torch.ones_like(contacts,dtype=torch.bool),
                     (row.get('relevant_gripper',0),0))
    return r,actions


def _cache_pair_matches(flow_cache, end_index, start_time, end_time):
    """True when the precomputed index-stride row is exactly this physical pair.

    Endpoint search may sit one raw frame off a capture. The cache row is a
    different window unless its recorded endpoints are the same timestamps.
    """
    start = float(flow_cache['interval_start_time_seconds'][end_index])
    end = float(flow_cache['interval_end_time_seconds'][end_index])
    return (abs(start - float(start_time)) <= NUMERIC_TIME_EPSILON
            and abs(end - float(end_time)) <= NUMERIC_TIME_EPSILON)


def _flow_features(handle, times, frame, flow_cache, flow_contract, *, return_flow_rgb=False,
                   interval_seconds=12*RAW_FRAME_SECONDS):
    """Time-matched raw numeric history plus freshly computed spatial flow, CPU only."""
    from dynamicwam.absolute_motion import (compute_flow_observation, build_motion_features,
        compute_dense_flow, filter_flow_by_forward_backward_consistency)
    indices, matched = flow_history_selection(times, int(frame), interval_seconds=interval_seconds)
    values = np.asarray(times, dtype=np.float64)
    # The first pair is the predecessor used only for acceleration. The four
    # returned intervals use the same nominal window as online; actual nearest
    # endpoints and their unequal durations remain in the native motion values.
    slots = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    both, agree = [], []
    for start_slot, end_slot in slots:
        ok = matched[start_slot] and matched[end_slot]
        both.append(ok)
        agree.append(ok and _cache_pair_matches(
            flow_cache, indices[end_slot], values[indices[start_slot]], values[indices[end_slot]]))
    if all(agree[i] or not both[i] for i in range(len(slots))):
        returned = agree[1:]
        selected = np.asarray(indices[2:], dtype=np.int64)
        numeric = torch.from_numpy(flow_cache['motion_features'][selected].copy())
        valid = torch.from_numpy(flow_cache['interval_valid'][selected].copy()) & torch.tensor(returned, dtype=torch.bool)
        acceleration = torch.from_numpy(flow_cache['acceleration_valid'][selected].copy())
        acceleration = acceleration & valid & torch.tensor(agree[:-1], dtype=torch.bool)
        numeric = torch.where(valid[:, None], numeric, torch.zeros_like(numeric))
        intervals = np.stack([values[indices[1:-1]], values[selected]], -1)
        if return_flow_rgb:
            flow_images = [flow_cache['flow_rgb'][i].copy() if ok else
                           np.zeros_like(flow_cache['flow_rgb'][i])
                           for i, ok in zip(selected, returned)]
    else:
        # A time-matched pair is not the cached index-stride row. Recompute that
        # physical window instead of training on the other history.
        displacement, starts, ends, flags = [], [], [], []
        flow_images = []
        for start_slot, end_slot in slots:
            start_i, end_i = int(indices[start_slot]), int(indices[end_slot])
            starts.append(float(values[start_i]))
            ends.append(float(values[end_i]))
            if start_i == end_i or values[end_i] <= values[start_i] or not matched[start_slot] or not matched[end_slot]:
                displacement.append(np.zeros(4, np.float32))
                flags.append(False)
                flow_images.append(np.zeros((*flow_contract['compute_size'], 3), np.uint8))
                continue
            rgb, stats, _, quality = compute_flow_observation(
                _rgb(handle, start_i, CAMERAS[0]), _rgb(handle, end_i, CAMERAS[0]),
                compute_size=tuple(flow_contract['compute_size']),
                normalization_percentile=float(flow_contract['normalization_percentile']),
                farneback=flow_contract['farneback'], quality=flow_contract['quality'])
            displacement.append(np.asarray(stats, dtype=np.float32))
            flags.append(bool(quality))
            flow_images.append(rgb)
        features, mask, accel = build_motion_features(
            np.stack(displacement), np.asarray(starts, np.float64), np.asarray(ends, np.float64), np.asarray(flags, bool))
        numeric = torch.from_numpy(features[1:].copy())
        valid = torch.from_numpy(mask[1:].copy())
        acceleration = torch.from_numpy(accel[1:].copy())
        intervals = np.stack([starts[1:], ends[1:]], -1)
        flow_images = flow_images[1:]
    _require(np.isfinite(intervals[valid.numpy()]).all() and (intervals[valid.numpy(), 1] <= values[int(frame)] + NUMERIC_TIME_EPSILON).all(),
             "future/nonfinite valid flow in observation")
    # Invalid historical intervals do not acquire invented support from a cache
    # sentinel such as time zero. A finite masked point is only a placeholder.
    intervals[~valid.numpy()] = values[int(frame)]
    spatial = torch.zeros(16, 5)
    spatial_valid = torch.zeros(16, dtype=torch.bool)
    previous = int(indices[-2]) if matched[-2] else int(frame)
    pair = np.array([values[previous], values[int(frame)]])
    if bool(valid[-1]) and pair[1] > pair[0]:
        a = _rgb(handle, previous, CAMERAS[0])
        b = _rgb(handle, int(frame), CAMERAS[0])
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
    result = (numeric,valid,acceleration,torch.from_numpy(intervals),spatial,spatial_valid,torch.from_numpy(pair))
    return result + (np.stack(flow_images),) if return_flow_rgb else result


class DominoPhysicalDataset(Dataset):
    """Lazy all-episode real RGB/labels; VAE encoding occurs in the training process.

    Text cache is mandatory. Teacher runs frozen on CPU. No absent cache silently
    becomes random/zero model conditioning. Metadata JSON and tensor-only caches
    are auditable independently of the original packed data.
    """
    def __init__(self, manifest_path, *, verify_hash=True, max_cached_episodes=2, require_complete_text=True):
        self.path=Path(manifest_path).resolve();self.manifest=json.loads(self.path.read_text())
        m=self.manifest;_require(m['schema_version']==SCHEMA,"wrong DOMINO schema")
        self.exposure_anchored=m.get('time_contract')==TIME_CONTRACT
        self.dt=float(m['action_interval_seconds'])
        _require(math.isfinite(self.dt) and self.dt>0 and m['codec_id']==CODEC_ID,"unverified timing/codec contract")
        if m.get('native_v5'):
            _require(m.get('time_contract')==('native_event_single_v6' if m.get('native_v6') else 'native_sync_ordinal_v5'),'native clock contract required')
        elif self.exposure_anchored:
            self.event_templates=load_templates(bound_metadata_path(m,self.path.parent,'event_templates_file'),
                                                m['event_templates_file_sha256'])
            _require(self.event_templates['action_interval_seconds']==self.dt,'event/data action grids differ')
        else:
            _require(self.dt==DT,'legacy physical grid differs')
            self.completion_distribution=manifest_completion_distribution(m)
        for k in ('native_checkpoint_sha256','action_normalization_sha256','motion_normalization_sha256'):
            _require(bool(re.fullmatch('[0-9a-f]{64}',m.get(k,''))),f'missing {k}')
        _require(m['statistics']['fit_split']=='train',"statistics are not train-only")
        self.rows=np.load(self.path.parent/m['rows_file'],mmap_mode='r')
        _require(self.rows.ndim==2 and self.rows.shape[1]==3,"rows must be episode/start/age triples")
        if verify_hash:_require(sha256_file(self.path.parent/m['rows_file'])==m['rows_sha256'],"row index changed")
        self.episodes=json.loads(bound_metadata_path(m,self.path.parent,'episodes_file').read_text())
        stat_path=self.path.parent/m['source_file_stats']
        _require(sha256_file(stat_path)==m['source_file_stats_sha256'],'source stat sidecar changed')
        self.source_stats=json.loads(stat_path.read_text())
        _require(self.source_stats['identity_kind']=='resolved_path_size_mtime_ns_not_content_hash','unknown source stat semantics')
        self.text_root=bound_metadata_path(m,self.path.parent,'text_index','text/index.json')
        _require(self.text_root.is_file(),f'episode-specific frozen text cache missing: {self.text_root}')
        text_index=json.loads(self.text_root.read_text())
        if require_complete_text:_require(text_index.get('complete') is True,"complete episode text bank required for training")
        prompt_path=bound_metadata_path(m,self.path.parent,'prompts_file','prompts.json')
        _require(text_index['identity']['prompt_manifest_sha256']==sha256_file(prompt_path),"text bank binds a different prompt manifest")
        self.text_index=text_index['items'];self.text_episodes=text_index['episodes']
        for ep in self.episodes:
            _require(self.text_episodes[ep['episode_uid']]==ep['prompt_id'],"text bank episode binding mismatch")
        self.cache=OrderedDict();self.max_cached_episodes=max_cached_episodes
        self.teacher=None
        self.action_mean=torch.tensor(m['action_statistics']['mean']).float()
        self.action_std=torch.tensor(m['action_statistics']['std']).float().clamp_min(1e-6)
        self.active_entity_roles,self.future_token_indices=role_token_indices(m)

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
        t=records.timestamps.numpy();dt=self.dt;event=None
        if self.manifest.get('native_v5'):
            s=ep['first_time']+grid*self.manifest['source_stride_seconds']
            q=s+self.manifest['query_ages_seconds'][age_id];completed=s
            source_frame=select_frame(t,s,past_only=True,tolerance=RGB_TOLERANCE)
            query_frame=select_frame(t,q,past_only=True,tolerance=RGB_TOLERANCE)
            starts=s+np.arange(16)*dt;fast_starts=q+np.arange(16)*dt
            event_offset=round((q-s)/dt)
            if self.manifest.get('native_v6'):
                s=float(t[source_frame]);q=float(t[query_frame]);completed=s
            source_known=torch.zeros(16,dtype=torch.bool);fast_known=source_known.clone()
        elif self.exposure_anchored:
            # V4 rows bind a REAL source image index and a complete event template.
            source_frame=grid;s=float(t[source_frame])
            event=validate_template(self.event_templates['templates'][age_id],dt)
            q=s+event['query_time'];completed=s+event['publish_time']
            target=s+event['query_rgb_time']
            query_frame=select_frame(t,target,past_only=True,tolerance=CONDITION_FRAME_TOLERANCE_SECONDS)
            camera_latency=event['query_rgb_available']-event['query_rgb_time']
            query_available=float(t[query_frame])+camera_latency
            _require(query_available<=q+NUMERIC_TIME_EPSILON,'replayed condition was not available at query')
            _require(q-float(t[query_frame])<=float(self.manifest.get('condition_rgb_max_age_seconds',CONDITION_RGB_MAX_AGE_SECONDS))+NUMERIC_TIME_EPSILON,
                     'replayed condition exceeds the deployment image-age gate')
            starts=s+np.arange(16)*dt
            fast_starts=s+event['fast_action_start']+np.arange(16)*dt
            source_known=torch.arange(16)<event['source_prefix_rows']
            fast_known=torch.arange(16)<event['fast_prefix_rows']
        else:
            s=ep['first_time']+grid*self.manifest['source_stride_seconds']
            age=self.manifest['query_ages_seconds'][age_id];q=s+age
            completed=s+completion_latency_seconds(self.completion_distribution,ep['episode_uid'],grid,age)
            delay=float(self.manifest.get('camera_availability_seconds',CAMERA_AVAILABILITY_SECONDS))
            source_frame=available_condition_frame(t,s,delay=delay)
            query_frame=available_condition_frame(t,q,delay=delay)
            starts=s+np.arange(16)*dt;fast_starts=q+np.arange(16)*dt
            event_offset=round((q-s)/dt)
            if self.manifest.get('native_v6'):
                s=float(t[source_frame]);q=float(t[query_frame]);completed=s
            source_known=torch.arange(16)<2;fast_known=torch.arange(16)<2
        source_action,sv,_=resample_actions(t,raw_actions.numpy(),starts+dt)
        fast_action,fv,_=resample_actions(t,raw_actions.numpy(),fast_starts+dt)
        if not self.manifest.get('native_v5'):
            _require(sv.all() and fv.all(),"index promised complete action support")
        source_action=np.where(sv[:,None],source_action,0.)
        fast_action=np.where(fv[:,None],fast_action,0.)
        source_action=(torch.from_numpy(source_action)-self.action_mean)/self.action_std
        fast_action=(torch.from_numpy(fast_action)-self.action_mean)/self.action_std
        future_requests=s+np.arange(1,9)*2*dt
        video_frames=[]; video_valid=[]
        for target in future_requests:
            try: video_frames.append(select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE));video_valid.append(True)
            except ValueError:
                if not self.manifest.get('native_v5'):raise
                video_frames.append(source_frame);video_valid.append(False)
        visual_requests=(fast_starts[0]+np.array([1,2,4,8])*dt if self.manifest.get('native_v6') else q+np.array([.1,.2,.4,.8]))
        visual_frames=[]
        for target in visual_requests:
            try:visual_frames.append(select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE))
            except ValueError:visual_frames.append(-1)
        with h5py.File(ep['raw_path'],'r') as h:
            sm,smv,sma,smi,_,_,_,source_flow_rgb=_flow_features(
                h,t,source_frame,flow,self.manifest['flow_contract'],return_flow_rgb=True,interval_seconds=4*dt)
            motion,mv,ma,mi,spatial,spatial_valid,spatial_times,query_flow_rgb=_flow_features(h,t,query_frame,flow,self.manifest['flow_contract'],interval_seconds=4*dt,return_flow_rgb=True)
            query_condition=np.stack([cv2.resize(image,(320,384)) for image in query_flow_rgb]+[_composite(h,query_frame,(384,320))])
            rgb,pixelvalid=zip(*[_letterbox(_rgb(h,query_frame,c),tuple(self.manifest['rgb_size']),tuple(self.manifest['rgb_view_sizes'][v])) for v,c in enumerate(CAMERAS)])
            condition=np.stack([cv2.resize(image,(320,384)) for image in source_flow_rgb]+[_composite(h,source_frame,(384,320))])
            future=np.stack([_composite(h,source_frame,(192,160))]+[_composite(h,i,(192,160)) for i in video_frames])
            needed=sorted(set(i for i in visual_frames if i>=0 and i not in visual_cache))
            if needed:
                if self.teacher is None:self.teacher=ProjectedContextTeacher(self.manifest['teacher_weights_path'],self.manifest['teacher_weights_sha256'])
                features,valid=self.teacher.encode(h,needed,records.entity_world_positions[needed].numpy())
                visual_cache.update({i:(features[j],valid[j]) for j,i in enumerate(needed)})
        features=torch.zeros_like(records.frozen_visual_features);context_valid=torch.zeros_like(records.visibility)
        for frame in set(i for i in visual_frames if i>=0):features[frame],context_valid[frame]=visual_cache[frame]
        r=replace(records,frozen_visual_features=features,projectability=context_valid,visual_context_valid=context_valid)
        labels=produce_future_labels(r,q,max_interpolation_gap=INTERPOLATION_GAP,max_contact_age=INTERPOLATION_GAP,duplicate_policy='last',max_visual_time_error=RGB_TOLERANCE,
            target_offsets=(visual_requests-q).tolist() if self.manifest.get('native_v6') else None)
        item={key:value.index_select(0,self.future_token_indices) for key,value in labels.tensors().items()}
        item['future']=normalize_future(item['future'],item['future_loss_valid'],self.manifest['statistics'])
        textrow=self.text_index[ep['prompt_id']];p=(self.text_root.parent/textrow['path']).resolve()
        _require(textrow['text_sha256']==ep['prompt_id'],"episode prompt identity mismatch")
        _require(sha256_file(p)==textrow['sha256'],"text cache hash mismatch")
        language=torch.load(p,map_location='cpu',weights_only=True)
        _require(language['context'].shape==(512,4096) and language['mask'].shape==(512,),"text cache shape mismatch")
        _require(language['mask'].dtype==torch.bool and language['mask'].any(),"invalid language mask")
        source_intervals=torch.from_numpy(np.stack([starts,starts+dt],-1))
        future_intervals=torch.tensor([[s,s+4*2*dt],[s+4*2*dt,s+8*2*dt]],dtype=torch.float64)
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
                    source_motion_times=smi,
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
                    completed_time=torch.tensor(completed,dtype=torch.float64),query_time=torch.tensor(q,dtype=torch.float64),
                    action_times=torch.from_numpy(fast_starts),action_intervals=torch.from_numpy(np.stack([fast_starts,fast_starts+dt],-1)),
                    query_entity_valid=torch.ones(len(self.active_entity_roles),dtype=torch.bool),
                    executed_actions=source_action.clone(),executed_action_intervals=source_intervals.clone(),
                    executed_action_valid=source_intervals[:,1]<=q+1e-8,
                    source_action_valid=torch.from_numpy(sv),fast_action_valid=torch.from_numpy(fv),
                    source_rgb_actual_times=torch.tensor(t[source_frame],dtype=torch.float64),
                    future_rgb_actual_times=torch.from_numpy(t[video_frames].copy()),future_rgb_time_errors=torch.from_numpy(t[video_frames]-future_requests),
                    future_visibility_known=torch.zeros(len(self.future_token_indices),dtype=torch.bool),
                    future_visual_actual_times=torch.tensor([t[i] if i>=0 else q for i in visual_frames],dtype=torch.float64).repeat(len(self.active_entity_roles)))
        item['executed_actions'][~item['executed_action_valid']]=0
        if event is not None:
            item.update(reference_available=torch.tensor(event['reference_available'],dtype=torch.bool),
                source_request_time=torch.tensor(s+event['request_time'],dtype=torch.float64),
                source_available_time=torch.tensor(s+event['source_available'],dtype=torch.float64),
                query_available_time=torch.tensor(query_available,dtype=torch.float64),
                source_prefix_end=torch.tensor(s+event['source_prefix_end'],dtype=torch.float64),
                fast_prefix_end=torch.tensor(s+event['fast_prefix_end'],dtype=torch.float64),
                query_rgb_template_time=torch.tensor(s+event['query_rgb_time'],dtype=torch.float64),
                query_rgb_sampling_error=torch.tensor(t[query_frame]-(s+event['query_rgb_time']),dtype=torch.float64))
        if self.manifest.get('native_v5'):
            item['native_source_index']=torch.tensor(0,dtype=torch.long)
            item['native_query_index']=torch.tensor(event_offset,dtype=torch.long)
            item['native_video_valid']=torch.tensor(all(video_valid))
        if self.manifest.get('native_v6'):
            hist_start=float(mi[mv,0].min()) if bool(mv.any()) else q
            item.update(future_events=torch.tensor([event_offset+n for n in (1,2,4,8)]*len(self.active_entity_roles)),
                teacher_condition_frames=torch.from_numpy(query_condition).permute(3,0,1,2).float()/127.5-1,
                teacher_initial_state=item['state'],teacher_motion=motion,teacher_motion_valid=mv,teacher_motion_acceleration_valid=ma,
                teacher_motion_times=mi,teacher_text=language['context'],teacher_known_actions=torch.zeros_like(fast_action),teacher_known_mask=fast_known,
                teacher_action_times=item['action_times'],teacher_action_intervals=item['action_intervals'],
                teacher_future_intervals=torch.tensor([[q,q+8*dt],[q+8*dt,q+16*dt]],dtype=torch.float64),
                teacher_temporal_support=torch.tensor([mi[0].tolist(),[hist_start,q]],dtype=torch.float64),
                teacher_view_overlap=source_overlap,teacher_token_valid=torch.cat([torch.full((120,),bool(mv[0])),torch.ones(120,dtype=torch.bool)]),
                teacher_input_type=torch.tensor([1,2]))
        return item


@torch.no_grad()
def materialize_native_latents(batch, vae, *, device=None):
    """Frozen native VAE preprocessing in caller-owned main process; never random.

    ``vae`` is this checkout's Wan2_2_VAE wrapper: encode accepts one BCTHW
    Tensor and returns one BCTHW Tensor. Its encode owns native CUDA autocast
    and normalization; this adapter preserves its declared input dtype exactly
    as the native dataset packer does. Full future input is 1+8 frames;
    retaining latent slices 1: preserves both groups of four future frames.
    """
    if 'source_condition_latent' in batch and 'source_future_latent' in batch:return batch
    _require('source_condition_frames' in batch and 'source_future_frames' in batch,'real native RGB frames missing')
    conditions=batch['source_condition_frames'];future=batch['source_future_frames']
    _require(conditions.ndim==future.ndim==5 and conditions.shape[0]==future.shape[0] and
             conditions.shape[1]==future.shape[1]==3 and conditions.shape[2]==5 and future.shape[2]==9,
             "incorrect native BCTHW frame/codec layout")
    model=getattr(vae,'model',vae)
    if isinstance(model,torch.nn.Module):model.eval()
    dtype=getattr(vae,'dtype',conditions.dtype)
    _require(dtype in (torch.float32,torch.float16,torch.bfloat16),'unsupported native VAE dtype')
    def encode(x):
        x=x.to(device=x.device if device is None else device,dtype=dtype)
        # The production wrapper owns CUDA autocast. CPU callers need their own
        # matching autocast because that wrapper's CUDA context is disabled there.
        with torch.autocast(device_type=x.device.type,dtype=dtype,
                            enabled=x.device.type=='cpu' and dtype in (torch.bfloat16,torch.float16)):
            z=vae.encode(x)
        _require(isinstance(z,torch.Tensor) and z.ndim==5 and z.shape[0]==x.shape[0],
                 'native Wan2_2_VAE must return a batched Tensor')
        return z
    result=dict(batch)
    result['source_condition_latent']=encode(conditions)
    _require(result['source_condition_latent'].shape[2]==2,'native 4-flow-plus-RGB codec must produce two slices')
    all_future=encode(future)
    _require(all_future.ndim==5 and all_future.shape[2]==3,'native 1+8 frame codec must produce three slices')
    result['source_future_latent']=all_future[:,:,1:].contiguous()
    if 'teacher_condition_frames' in batch:
        result['teacher_condition_latent']=encode(batch['teacher_condition_frames'])
    previous={k[9:]:v for k,v in batch.items() if k.startswith('previous_')}
    if 'source_condition_frames' in previous:
        result.update({'previous_'+k:v for k,v in materialize_native_latents(previous,vae,device=device).items()})
    if 'native_video_valid' in batch:
        result['source_video_loss_valid']=batch['native_video_valid'][:,None,None,None,None].expand_as(result['source_future_latent'])
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
                available_condition_frame(t,s)
                for target in s+np.arange(1,9)*2*DT:select_frame(t,target,past_only=False,tolerance=RGB_TOLERANCE)
            except ValueError:audit['rejected_rgb_tolerance']+=1;continue
            _,sv,_=resample_actions(t,actions.numpy(),s+np.arange(1,17)*DT)
            if not sv.all():audit['rejected_action_support']+=1;continue
            for ai,age in enumerate(QUERY_AGES):
                q=s+age
                if q+16*DT>t[-1]+1e-8:continue
                try:available_condition_frame(t,q)
                except ValueError:audit['rejected_rgb_tolerance']+=1;continue
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
                  'codec_temporal_contract':CODEC_TEMPORAL_CONTRACT,
                  'rgb_size':config.get('rgb_size',[240,320]),'rgb_view_sizes':config.get('rgb_view_sizes',[[240,320],[120,160],[120,160]]),
                  'flow_contract':native['absolute_motion']['flow_contract'],'motion_representation':'raw_absolute_motion_v2','action_representation':'checkpoint_mean_std',
                  'action_resampling':'bounded_linear_arm_drive_targets_previous_event_gripper_last_duplicate',
                  'future_rgb_rule':'nearest_real_frame_abs_error_le_0.030001',
                  'camera_availability_seconds':CAMERA_AVAILABILITY_SECONDS,
                  'condition_rgb_max_age_seconds':CONDITION_RGB_MAX_AGE_SECONDS,
                  'condition_rgb_rule':'latest_real_frame_at_or_before_event_minus_camera_availability',
                  'latency_provenance':'synthetic_completion_0.18s_not_measured','time_grid_cache_rebuilt':True,'formal_training_ready':False,
                  'text_index':'text/index.json','native_latents':'main_process_frozen_native_vae_or_identity_bound_cache','audit_file':'audit.json'}
        (output/f'{split}.json').write_text(json.dumps(manifest,indent=2))
    attach_stat_identity(output)
    return audit
