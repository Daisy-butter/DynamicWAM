#!/usr/bin/env python3
"""Raw development recorder -> strict real-replay training artifacts.

Default is CPU indexing, retaining a blocking missing-latent status. --encode
requires the approved temporary all-adapter GPU guard and actual native VAE.
No policy proposal or failed execution is silently converted into BC labels.
"""
from __future__ import annotations
import argparse,json,sys,shutil
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from dynamicwam.image import resize_with_padding
from dynamicwam.observations.multiview import fast_views_from_composite
from dynamicwam.runtime.domino_continuous.development_replay import load_development_replay,CLOCK_SCHEMA
from dynamicwam.training.data.physical_grid import DT,sha256_file,json_hash,resample_commands,video_endpoints
from dynamicwam.training.data.multirate_sampling import partition_episodes
from dynamicwam.training.data.replay_language import build_replay_language_catalog,match_replay_instruction

def build_replay_samples(directory,*,episode_index,language_catalog,encoder=None):
    """Pure CPU mapping unless an explicit encoder is supplied by guarded CLI.

    Tests can inject a labelled synthetic encoder; CLI never accepts one.
    Source/current grids use their own RECORDED exact offsets, including the
    controller's post-prime clock origin. No episode-zero phase substitution.
    """
    directory=Path(directory);manifest,queries=load_development_replay(directory,require_success=True)
    with np.load(directory/'frames.npz') as z:images=z['rgb'];frame_times=z['capture_times']
    with np.load(directory/'executed.npz') as z:executed={k:z[k] for k in z.files}
    with np.load(directory/'targets.npz') as z:completed={k:z[k] for k in z.files}
    packets={q['query_index']:dict(np.load(directory/q['path'])) for q in queries}
    instructions={q.get('instruction') for q in queries}
    if len(instructions)!=1:raise ValueError('source/current instructions differ within the actual episode')
    language=match_replay_instruction(language_catalog,manifest['scene']['task'],next(iter(instructions)))
    identity=dict(trajectory_id=manifest['trajectory_id'],episode_id=manifest['episode_id'])
    tick_times=np.asarray(executed['end_times'],dtype=np.float64);tick_actions=np.asarray(executed['commands'],dtype=np.float32)
    if len(tick_times)<2 or np.any(np.diff(tick_times)<=0) or not np.isfinite(tick_actions).all():raise ValueError('invalid actual execution trace')
    observations={};targets={};drops=Counter();samples=[]
    for q in queries:
        i=q['query_index'];packet=packets[i];capture=float(q['capture_time'])
        if 'motion_history_end_times' not in packet:raise ValueError('actual motion history capture times missing')
        views,masks=fast_views_from_composite(packet['fast_frame'])
        obs=dict(**identity,origin='policy_execution_capture',policy_checkpoint_sha256=manifest['policy_checkpoint_sha256'],
            observation_id=f"{manifest['trajectory_id']}:{i}",capture_time=capture,instruction=q['instruction'],
            completed_action_count=int(q['completed_action_count']),rgb=torch.from_numpy(views),rgb_valid_mask=torch.from_numpy(masks),
            state=torch.from_numpy(packet['state'].astype(np.float32)),state_semantics='raw_drive_targets',
            native_motion=torch.from_numpy(packet['motion_features'].astype(np.float32)),
            motion_interval_valid_mask=torch.from_numpy(packet['motion_interval_valid_mask']),
            motion_acceleration_valid_mask=torch.from_numpy(packet['motion_acceleration_valid_mask']),
            motion_history_times=torch.from_numpy(packet['motion_history_end_times'].astype(np.float64)))
        if encoder is not None:obs['condition_latent']=encoder('condition',packet['frame'],packet['flow_frames'])
        observations[i]=obs
        endpoint_times=capture+np.asarray(packet['_rtc_action_target_offsets'],dtype=np.float64)
        if endpoint_times.shape!=(16,) or np.any(endpoint_times<=capture) or not np.allclose(np.diff(endpoint_times),DT,rtol=0,atol=1e-8):
            raise ValueError('recorded source/current target grid is invalid')
        if endpoint_times[-1]>tick_times[-1]+1e-9 or endpoint_times[0]<tick_times[0]:
            drops['future_execution_incomplete']+=1;continue
        try:indices,actual,lag=video_endpoints(frame_times,endpoint_times[1::2])
        except ValueError:drops['future_video_incomplete']+=1;continue
        if np.any(actual<=capture) or np.any(lag>.16+1e-9):drops['future_video_gap']+=1;continue
        label=dict(**identity,target_source='successful_self_imitation',target_sampling='actual_executed_controller_command_grid',
            observation_id=obs['observation_id'],raw_actions=torch.from_numpy(resample_commands(tick_times,tick_actions,endpoint_times)),
            action_target_times=torch.from_numpy(endpoint_times),video_actual_times=torch.from_numpy(actual))
        if encoder is not None:label['future_latent']=encoder('future',packet['frame'],images[indices])
        targets[i]=label
    for q in queries:
        if not q.get('result_received') or q.get('bootstrap_only'):continue
        i=q['query_index'];source_i=q['source_query_index'];packet=packets[i]
        if i not in targets or source_i not in targets:drops['source_or_current_future_incomplete']+=1;continue
        wall=q.get('policy_diagnostics',{}).get('asynchronous_world',{})
        if not all(k in wall for k in ('source_published_monotonic','source_consumed_monotonic')):
            drops['missing_cache_publication_wall_evidence']+=1;continue
        st=float(observations[source_i]['capture_time']);ct=float(q['capture_time'])
        age=int(q['completed_action_count'])-int(observations[source_i]['completed_action_count'])
        if age>31 or ct-st>31*DT+1e-8:drops['cache_outside_support']+=1;continue
        d=int(q['prefix_length']);prefix=np.asarray(packet['_rtc_prefix_actions'],dtype=np.float32)[:d]
        bootstrap=age==0 and int(q['completed_action_count'])==0 and ct>st
        timeline=dict(**identity,physics_tick_times=torch.from_numpy(tick_times),physics_tick_raw_commands=torch.from_numpy(tick_actions),
            committed_endpoint_times=torch.tensor(q['prefix_endpoint_times'],dtype=torch.float64),
            committed_raw_actions=torch.from_numpy(prefix),completed_endpoint_times=torch.from_numpy(completed['endpoint_times']),
            completed_raw_actions=torch.from_numpy(completed['commands'].astype(np.float32)))
        if bootstrap:timeline['bootstrap_hold_raw_command']=observations[source_i]['state'].clone()
        previous=None;valid=np.asarray(packet['_rtc_previous_valid'],dtype=bool).reshape(-1)
        if np.any(valid):
            if q['previous_publication_physics_s'] is None:raise ValueError('previous queue publication evidence missing')
            if valid.shape!=(16,) or not np.all(valid[np.flatnonzero(valid)[0]:np.flatnonzero(valid)[-1]+1]):raise ValueError('prior published plan validity must be contiguous')
            previous=dict(trajectory_id=identity['trajectory_id'],publication_time=float(q['previous_publication_physics_s']),
                endpoint_times=torch.tensor(q['previous_endpoint_times'],dtype=torch.float64)[valid],
                raw_actions=torch.from_numpy(np.asarray(packet['_rtc_previous_actions'],dtype=np.float32))[valid])
        blob=dict(**identity,source_observation=observations[source_i],current_observation=observations[i],
            source_targets=targets[source_i],current_targets=targets[i],executed_timeline=timeline,
            previous_published_plan=previous,source_id=episode_index*1000000+source_i,
            query_time=float(q['query_physics_s']),ready_time=float(q['ready_physics_s']),
            cache_publication_lower_bound_time=st,cache_publication_upper_bound_time=float(q['ready_physics_s']),
            cache_publication_time_is_exact=False,source_rtc_prefix_length=int(queries[source_i]['prefix_length']),
            source_rtc_prefix_raw_actions=torch.from_numpy(np.asarray(packets[source_i]['_rtc_prefix_actions'],dtype=np.float32)[:int(queries[source_i]['prefix_length'])].copy()),
            source_rtc_prefix_endpoint_times=torch.tensor(queries[source_i]['prefix_endpoint_times'],dtype=torch.float64),
            source_published_monotonic=float(wall['source_published_monotonic']),
            source_consumed_monotonic=float(wall['source_consumed_monotonic']),receive_monotonic=float(q['receive_monotonic']),
            clock_schema=manifest['clock_schema'],
            bootstrap_mode='prime_then_hold' if bootstrap else 'none')
        scene=manifest['scene'];scene=dict(task=scene['task'],seed=int(scene['seed']),scene_id=f"{scene['task']}:{int(scene['seed'])}",
            scene_sha256=scene.get('scene_sha256',scene['manifest_sha256']),motion_sha256=scene['motion_sha256'])
        record=dict(sample_id=episode_index*1000000+i,episode_index=episode_index,episode_name=identity['episode_id'],
            trajectory_id=identity['trajectory_id'],task_name=scene['task'],split='development',lang_group_id=language['lang_group_id'],
            language_index=language['language_index'],lang_id=language['lang_id'],instruction=language['instruction'],prompt=language['prompt'],
            scene=scene,target_source='successful_self_imitation',trajectory_success=True,cache_age_actions=age,
            bootstrap_mode=blob['bootstrap_mode'],policy_checkpoint_sha256=manifest['policy_checkpoint_sha256'],
            policy_source_sha256=manifest['policy_source_sha256'])
        samples.append((record,blob))
    return manifest,samples,dict(drops)

def native_encoder(vae,device):
    import cv2
    import torch.nn.functional as F
    def encode(kind,first,others):
        first=resize_with_padding(np.asarray(first),(384,320))
        if kind=='condition':
            flows=np.stack([cv2.resize(f,(320,384),interpolation=cv2.INTER_LINEAR) for f in others])
            pixels=np.concatenate([flows,first[None]],axis=0)
            tensor=torch.from_numpy(pixels).permute(3,0,1,2).unsqueeze(0).float().div(127.5).sub(1).to(device)
            with torch.inference_mode():value=vae.encode(tensor)[0].float().cpu()
            shape=(48,2,24,20)
        else:
            pixels=np.stack([first]+[resize_with_padding(f,(384,320)) for f in others])
            tensor=torch.from_numpy(pixels).permute(0,3,1,2).float().div(255)
            tensor=F.interpolate(tensor,size=(192,160),mode='bilinear',align_corners=False,antialias=True)
            tensor=(tensor*2-1).permute(1,0,2,3).unsqueeze(0).to(device)
            with torch.inference_mode():value=vae.encode(tensor)[0,:,1:].float().cpu()
            shape=(48,2,12,10)
        if tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):raise ValueError('native replay VAE shape/nonfinite mismatch')
        return value
    return encode

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',action='append',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--packed-root',required=True);p.add_argument('--physical-grid-root',required=True)
    p.add_argument('--encode',action='store_true');p.add_argument('--vae-checkpoint');p.add_argument('--device',default='cuda:0');p.add_argument('--guarded-launch',action='store_true')
    p.add_argument('--partition-seed',type=int,default=20260928);p.add_argument('--validation-fraction',type=float,default=.1)
    a=p.parse_args();torch.set_num_threads(1)
    import cv2;cv2.setNumThreads(1)
    output=Path(a.output_root).resolve();ssd=Path('/SSD_DISK/users/wuruihan').resolve()
    if ssd not in output.parents:raise ValueError('replay artifacts must stay on verified SSD')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError('use a new replay generation directory')
    grid=json.loads((Path(a.physical_grid_root)/'manifest.json').read_text());packed=Path(a.packed_root)
    language_catalog=build_replay_language_catalog(packed)
    encoder=None;provenance=None
    if a.encode:
        if a.device.startswith('cuda') and not a.guarded_launch:raise RuntimeError('actual replay VAE encoding requires approved guard')
        if not a.vae_checkpoint:raise ValueError('--vae-checkpoint required')
        from dynamicwam.vendor.wan.modules.vae2_2 import Wan2_2_VAE
        import dynamicwam.vendor.wan.modules.vae2_2 as vm
        encoder=native_encoder(Wan2_2_VAE(vae_pth=a.vae_checkpoint,device=a.device),a.device)
        provenance=dict(encoder_kind='native_wan2_2_real_replay_v1',encoder_sha256=sha256_file(__file__),
            vae_module_sha256=sha256_file(vm.__file__),vae_checkpoint=str(Path(a.vae_checkpoint).resolve()),vae_sha256=sha256_file(a.vae_checkpoint))
    records=[];drops=Counter();scenes={}
    for episode,raw in enumerate(a.raw_root):
        raw_path=Path(raw);raw_meta=json.loads((raw_path/'manifest.json').read_text());task=raw_meta['scene']['task']
        manifest,samples,counts=build_replay_samples(raw_path,episode_index=episode,language_catalog=language_catalog,encoder=encoder);drops.update(counts)
        # Copy immutable raw evidence once; the validator verifies its file table.
        evidence=output/'raw'/f'{episode:06d}';shutil.copytree(raw_path,evidence,dirs_exist_ok=False)
        for row,blob in samples:
            path=output/'samples'/f"{row['sample_id']:012d}.pt";path.parent.mkdir(exist_ok=True);torch.save(blob,path)
            row['sample']=dict(path=str(path.relative_to(output)),sha256=sha256_file(path))
            row['raw_trace_manifest']=dict(path=str((evidence/'manifest.json').relative_to(output)),sha256=sha256_file(evidence/'manifest.json'))
            records.append(row);scenes[row['scene']['scene_id']]=row['scene']
    assignments=partition_episodes(records,seed=a.partition_seed,validation_fraction=a.validation_fraction) if records else {}
    for row in records:row['subset']=assignments[row['episode_index']]
    bootstrap_counts=Counter(r['subset'] for r in records if r['bootstrap_mode']=='prime_then_hold')
    complete=bool(a.encode and records and bootstrap_counts['train']>0)
    result=dict(kind='dynamicwam_prepared_real_replay_v1',schema_version=1,clock_schema=CLOCK_SCHEMA,complete=complete,synthetic=False,
        physical_grid_identity=grid['physical_grid_identity'],parent_episode_partition_identity=grid['episode_partition_identity'],
        action_stats_sha256=sha256_file(packed/'action_stats.json'),motion_stats_sha256=sha256_file(packed/'motion_stats.json'),
        formal_inventory_sha256=sha256_file(Path(a.physical_grid_root)/'formal700_exclusion.json'),
        latent_provenance=provenance,language_catalog_identity=language_catalog['identity'],development_scene_allowlist=list(scenes.values()),records=records,
        partition_seed=a.partition_seed,validation_fraction=a.validation_fraction,bootstrap_split_counts=dict(bootstrap_counts),drops=dict(drops),
        status='encoded_pending_training_validation' if complete else 'blocked_missing_real_VAE_or_bootstrap_or_usable_records')
    (output/'manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(records=len(records),complete=complete,drops=dict(drops),bootstrap_split_counts=dict(bootstrap_counts))))
if __name__=='__main__':main()
