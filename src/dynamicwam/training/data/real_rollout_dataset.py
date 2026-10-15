"""Training adapter for independently collected real development trajectories.

No replay is manufactured from demonstrations. The prepared replay manifest
must bind actual policy captures, execution queues, successful self-imitation
or same-trajectory corrective labels, and native VAE encodings. Missing real
replay data fail closed. Bootstrap hold samples retain elapsed physical time
while their executed-action count remains zero.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.training.data.physical_grid import DT,json_hash,sha256_file,resample_commands
from dynamicwam.training.data.multirate_sampling import partition_episodes,pairing_identity
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.paired_world_action_dataset import PairRecord,episode_lang_id
from dynamicwam.training.data.replay_language import build_replay_language_catalog,match_replay_instruction

TARGET_SOURCES={'successful_self_imitation','expert_corrective_same_trajectory'}

def _bound_path(root,ref):
    path=(root/ref['path']).resolve()
    if root.resolve() not in path.parents or not path.is_file() or sha256_file(path)!=ref['sha256']:
        raise ValueError('replay artifact path/hash mismatch')
    return path

def _tensor(value,shape,name,dtype=torch.float32):
    tensor=torch.as_tensor(value,dtype=dtype)
    if tuple(tensor.shape)!=tuple(shape) or not bool(torch.isfinite(tensor).all()):
        raise ValueError('invalid replay '+name+' shape or nonfinite values')
    return tensor

def validate_replay_manifest(replay_root,physical_grid_root,packed_root,*,strict_provenance=True):
    if strict_provenance is not True:raise ValueError('replay provenance validation cannot be disabled')
    root=Path(replay_root);path=root/'manifest.json'
    if not path.is_file():raise FileNotFoundError('real same-trajectory replay not collected/processed: '+str(path))
    m=json.loads(path.read_text());grid=json.loads((Path(physical_grid_root)/'manifest.json').read_text())
    if m.get('kind')!='dynamicwam_prepared_real_replay_v1' or m.get('complete') is not True or m.get('synthetic') is not False:
        raise ValueError('real completed replay required; synthetic fixtures cannot authorize training')
    if m['physical_grid_identity']!=grid['physical_grid_identity'] or m['parent_episode_partition_identity']!=grid['episode_partition_identity']:
        raise ValueError('replay parent physical-grid identity mismatch')
    packed=Path(packed_root)
    for name,key in [('action_stats.json','action_stats_sha256'),('motion_stats.json','motion_stats_sha256')]:
        if sha256_file(packed/name)!=m[key] or m[key]!=grid['dataset_identity'][key]:
            raise ValueError('replay normalization differs from base dataset')
    inventory_path=Path(physical_grid_root)/'formal700_exclusion.json'
    if sha256_file(inventory_path)!=m['formal_inventory_sha256']:raise ValueError('formal scene exclusion identity changed')
    inventory=json.loads(inventory_path.read_text());identity=inventory.pop('identity')
    if identity!=json_hash(inventory) or len(inventory['scenes'])!=700 or len(inventory['accepted_manifest_sha256'])!=35:
        raise ValueError('formal exclusion inventory is incomplete')
    for rel,sha in inventory['accepted_manifest_sha256'].items():
        if sha256_file(Path(inventory['source_root'])/rel)!=sha:raise ValueError('original formal scene manifest changed')
    formal={key:{s[key] for s in inventory['scenes']} for key in ('scene_id','scene_sha256','motion_sha256')}
    allowed={s['scene_id']:s for s in m['development_scene_allowlist']}
    if not allowed:raise ValueError('development scene allowlist is empty')
    for scene in allowed.values():
        if any(scene[key] in formal[key] for key in formal):raise ValueError('development replay leaks a formal test scene')
    provenance=m['latent_provenance'];code_root=Path(__file__).resolve().parents[4]
    official=packed.resolve().parents[2]/'external/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'
    if provenance.get('encoder_kind')!='native_wan2_2_real_replay_v1' or Path(provenance['vae_checkpoint']).resolve()!=official.resolve():
        raise ValueError('real replay requires native official VAE encodings')
    if provenance['vae_sha256']!=sha256_file(official) or provenance['vae_module_sha256']!=sha256_file(code_root/'src/dynamicwam/vendor/wan/modules/vae2_2.py'):
        raise ValueError('replay VAE implementation/weight identity mismatch')
    if provenance['encoder_sha256']!=sha256_file(code_root/'scripts/prepare_real_rollout_replay.py'):
        raise ValueError('replay encoder implementation changed')
    records=m['records']
    if not any(r.get('bootstrap_mode')=='prime_then_hold' and r.get('subset')=='train' for r in records):
        raise ValueError('replay refinement requires real bootstrap prime_then_hold coverage in training split')
    if not records:raise ValueError('prepared replay has no actual training records')
    language_catalog=build_replay_language_catalog(packed)
    if m.get('language_catalog_identity')!=language_catalog['identity']:raise ValueError('replay exact language catalog identity changed')
    checked_raw=set()
    for row in records:
        exact=match_replay_instruction(language_catalog,row['task_name'],row['instruction'])
        if any(row.get(k)!=exact[k] for k in ('lang_group_id','language_index','lang_id','instruction','prompt')):raise ValueError('replay instruction/index differs from exact packed text')
        scene=row['scene']
        if allowed.get(scene['scene_id'])!=scene or any(scene[k] in formal[k] for k in formal):
            raise ValueError('replay record scene is not an authorized development scene')
        if row['target_source'] not in TARGET_SOURCES or (row['target_source']=='successful_self_imitation' and row.get('trajectory_success') is not True):
            raise ValueError('failed deployment actions are not BC/FM targets without real corrective labels')
        if row['target_source']=='expert_corrective_same_trajectory':
            if 'corrective_labels' not in row:raise ValueError('failed trajectory requires a separate actual corrective label artifact')
            _bound_path(root,row['corrective_labels'])
        _bound_path(root,row['sample'])
        raw_manifest=_bound_path(root,row['raw_trace_manifest'])
        if str(raw_manifest) not in checked_raw:
            from dynamicwam.runtime.domino_continuous.development_replay import load_development_replay
            raw,_=load_development_replay(raw_manifest.parent,require_success=row['target_source']=='successful_self_imitation')
            if not raw.get('complete'):raise ValueError('incomplete raw trajectory cannot authorize replay training')
            checked_raw.add(str(raw_manifest))
    normalizer=RoboTwinQposNormalizer.from_stats_path(packed/'action_stats.json',epsilon=1e-6)
    verified_bootstrap=0
    for row in records:
        if row.get('bootstrap_mode')=='prime_then_hold' and row['subset']=='train':
            blob=torch.load(_bound_path(root,row['sample']),map_location='cpu',weights_only=True)
            sample=adapt_replay_sample(blob,row,normalizer=normalizer,language=torch.empty(0))
            if sample['bootstrap_mode']!='prime_then_hold':raise ValueError('bootstrap marker does not match actual hold execution')
            verified_bootstrap+=1
    if verified_bootstrap<1:raise ValueError('no usable real bootstrap sample survived full trace validation')
    return m,grid,dict(latent_manifest_sha256=sha256_file(path),replay_identity=json_hash(m),
        records=len(records),on_policy_ready=True,prefix_source='on_policy_executed',bootstrap_split_counts=dict(Counter(r['subset'] for r in records if r.get('bootstrap_mode')=='prime_then_hold')))

def adapt_replay_sample(blob,record,*,normalizer,language,max_cache_age_actions=31):
    """Validate one real trace sample and normalize all raw14 drive commands."""
    identity=(record['trajectory_id'],record['episode_name'])
    for obj in (blob,blob['source_observation'],blob['current_observation'],blob['source_targets'],blob['current_targets'],blob['executed_timeline']):
        if (obj.get('trajectory_id'),obj.get('episode_id'))!=identity:
            raise ValueError('source/current/targets/actions are not the same real trajectory')
    source,current=blob['source_observation'],blob['current_observation'];timeline=blob['executed_timeline']
    st,ct=float(source['capture_time']),float(current['capture_time']);query=float(blob['query_time']);ready=float(blob['ready_time'])
    if not np.isfinite([st,ct,query,ready]).all() or not st<=ct<=query<=ready:
        raise ValueError('replay clocks are not causal')
    age=int(current['completed_action_count'])-int(source['completed_action_count'])
    if ct-st>max_cache_age_actions*DT+1e-8:raise ValueError('replay physical cache age exceeds trained seconds support')
    if not 0<=age<=max_cache_age_actions:raise ValueError('replay executed-action cache age outside trained support')
    for obs in (source,current):
        if obs.get('instruction')!=record.get('instruction'):raise ValueError('source/current actual instruction differs from replay text binding')
        if obs.get('origin')!='policy_execution_capture' or obs.get('policy_checkpoint_sha256')!=record['policy_checkpoint_sha256']:
            raise ValueError('counterfactual/expert observations cannot be used with policy prefixes')
        history=np.asarray(obs['motion_history_times'],dtype=np.float64)
        valid_history=np.asarray(obs['motion_interval_valid_mask'],dtype=bool)
        if history.shape!=(4,) or valid_history.shape!=(4,) or not np.isfinite(history[valid_history]).all() or np.any(history[valid_history]>float(obs['capture_time'])+1e-9) or np.any(np.diff(history[valid_history])<0):
            raise ValueError('replay motion history is not actual causal capture history')
        _tensor(obs['condition_latent'],(48,2,24,20),'condition_latent')
        _tensor(obs['rgb'],(3,3,160,160),'rgb')
        _tensor(obs['state'],(14,),'raw drive state')
        if obs.get('state_semantics')!='raw_drive_targets':raise ValueError('measured qpos cannot substitute recorded drive targets')
    def targets(obj,obs):
        if obj.get('target_source')!=record['target_source'] or obj.get('observation_id')!=obs['observation_id']:
            raise ValueError('future labels do not correspond to the recorded observation')
        times=_tensor(obj['action_target_times'],(16,),'action endpoint times',torch.float64)
        if not bool((times>float(obs['capture_time'])).all()) or not torch.allclose(times[1:]-times[:-1],torch.full((15,),DT,dtype=torch.float64),atol=1e-8,rtol=0):
            raise ValueError('replay future action labels are not on the physical grid')
        actual=_tensor(obj['video_actual_times'],(8,),'actual future capture times',torch.float64)
        if not bool((actual>float(obs['capture_time'])).all()) or not bool((actual<=times[1::2]+1e-9).all()) or not bool((times[1::2]-actual<=.16+1e-9).all()):
            raise ValueError('replay future video capture mapping is invalid')
        actions=_tensor(obj['raw_actions'],(16,14),'raw future actions')
        if not bool(((actions[:,[6,13]]>=0)&(actions[:,[6,13]]<=1)).all()):raise ValueError('gripper labels out of bounds')
        latent=_tensor(obj['future_latent'],(48,2,12,10),'future_latent')
        # Successful self-imitation must match ACTUALLY EXECUTED endpoints.
        # A predicted proposal that was later replaced cannot become a label.
        if record['target_source']=='successful_self_imitation':
            executed_times=np.asarray(timeline['physics_tick_times'],dtype=np.float64)
            executed_actions=np.asarray(timeline['physics_tick_raw_commands'],dtype=np.float32)
            executed_targets=resample_commands(executed_times,executed_actions,times.numpy())
            if obj.get('target_sampling')!='actual_executed_controller_command_grid' or not np.allclose(executed_targets,actions.numpy(),rtol=0,atol=1e-6):
                raise ValueError('self-imitation target is not the actual executed command trajectory')
        else:
            if obj.get('corrective_label_artifact_sha256')!=record.get('corrective_labels',{}).get('sha256') or not record.get('corrective_labels') or obj.get('corrective_observation_id')!=obs['observation_id']:
                raise ValueError('expert corrective targets require their own same-observation provenance')
        return normalizer.normalize(actions),times,latent,actual
    sa,starget,sl,sv=targets(blob['source_targets'],source)
    fa,ftarget,fl,fv=targets(blob['current_targets'],current)
    prefix_times=np.asarray(timeline['committed_endpoint_times'],dtype=np.float64)
    prefix_raw=np.asarray(timeline['committed_raw_actions'],dtype=np.float32)
    d=len(prefix_times)
    if not 0<=d<=4 or prefix_raw.shape!=(d,14) or not np.isfinite(prefix_raw).all():raise ValueError('invalid real committed prefix')
    if not np.allclose(prefix_times,ftarget[:d].numpy(),atol=1e-8,rtol=0):raise ValueError('committed prefix is not aligned to current target endpoints')
    if d and ready>=prefix_times[-1]+1e-8:raise ValueError('recorded RTC ready time misses its committed prefix suffix deadline')
    prefix=torch.zeros((16,14));prefix[:d]=normalizer.normalize(torch.from_numpy(prefix_raw.copy()))
    source_d=int(blob['source_rtc_prefix_length'])
    source_raw=_tensor(blob['source_rtc_prefix_raw_actions'],(source_d,14),'actual source committed prefix')
    source_prefix_times=_tensor(blob['source_rtc_prefix_endpoint_times'],(source_d,),'source prefix endpoint times',torch.float64)
    if not 0<=source_d<=4 or not torch.allclose(source_prefix_times,starget[:source_d],atol=1e-8,rtol=0):
        raise ValueError('source prefix does not match the actual source query endpoint grid')
    source_prefix=torch.zeros((16,14));source_prefix[:source_d]=normalizer.normalize(source_raw)

    bootstrap=blob.get('bootstrap_mode')=='prime_then_hold'
    if bootstrap:
        if age!=0 or int(source['completed_action_count'])!=0 or not ct>st or d!=4:
            raise ValueError('bootstrap must be real elapsed time with zero executed actions and H4 hold')
        hold=np.asarray(timeline['bootstrap_hold_raw_command'],dtype=np.float32)
        tick_times=np.asarray(timeline['physics_tick_times'],dtype=np.float64)
        tick_commands=np.asarray(timeline['physics_tick_raw_commands'],dtype=np.float32)
        selected=(tick_times>=st-1e-8)&(tick_times<=query+1e-8)
        actual_times=tick_times[selected]
        if hold.shape!=(14,) or len(actual_times)<2 or actual_times[0]>st+.008 or actual_times[-1]<query-.008 or np.any(np.diff(actual_times)>.008):
            raise ValueError('bootstrap requires continuous actual hold execution evidence')
        if not np.allclose(tick_commands[selected],hold,atol=1e-6,rtol=0) or not np.allclose(prefix_raw,hold,atol=1e-6,rtol=0):
            raise ValueError('bootstrap hold command differs from actual executed/committed actions')
    previous=torch.zeros((16,14));valid=torch.zeros(16,dtype=torch.bool)
    plan=blob.get('previous_published_plan')
    if plan is not None:
        if plan.get('trajectory_id')!=identity[0] or float(plan['publication_time'])>query+1e-9:
            raise ValueError('previous plan was not published on this trajectory before the query')
        times=np.asarray(plan['endpoint_times'],dtype=np.float64);actions=np.asarray(plan['raw_actions'],dtype=np.float32)
        if len(times)==0 or not np.isfinite(times).all() or actions.shape!=(len(times),14) or not np.isfinite(actions).all() or np.any(np.diff(times)<=0):raise ValueError('invalid prior published queue')
        covered=(ftarget.numpy()>=times[0])&(ftarget.numpy()<=times[-1]);valid=torch.from_numpy(covered)
        if np.any(covered):previous[valid]=normalizer.normalize(torch.from_numpy(resample_commands(times,actions,ftarget.numpy()[covered])))
    ss=normalizer.normalize(_tensor(source['state'],(14,),'source state'));fs=normalizer.normalize(_tensor(current['state'],(14,),'current state'))
    def motion(obs):return _tensor(obs['native_motion'],(4,12),'native motion')
    def mask(obs,key):return _tensor(obs[key],(4,),key)
    lower=float(blob['cache_publication_lower_bound_time']);upper=float(blob['cache_publication_upper_bound_time'])
    published=float(blob['source_published_monotonic']);consumed=float(blob['source_consumed_monotonic']);received=float(blob['receive_monotonic'])
    if blob.get('cache_publication_time_is_exact') is not False or not np.isfinite([lower,upper,published,consumed,received]).all() or not st<=lower<=upper<=ready or not published<=consumed<=received:
        raise ValueError('cache publication bounds or real monotonic ordering are invalid')
    result=dict(physical_grid_version=1,source_condition_latent=torch.as_tensor(source['condition_latent']).float(),
        source_state=ss,source_motion=motion(source),source_motion_interval_valid_mask=mask(source,'motion_interval_valid_mask'),
        source_motion_acceleration_valid_mask=mask(source,'motion_acceleration_valid_mask'),
        bridge_noise_shape_latent=torch.zeros_like(sl),source_rgb=torch.as_tensor(source['rgb']).float(),
        source_rgb_valid_mask=_tensor(source['rgb_valid_mask'],(3,160,160),'source rgb mask'),
        fast_rgb=torch.as_tensor(current['rgb']).float(),fast_rgb_valid_mask=_tensor(current['rgb_valid_mask'],(3,160,160),'current rgb mask'),
        fast_state=fs,fast_motion=motion(current),fast_motion_interval_valid_mask=mask(current,'motion_interval_valid_mask'),
        fast_motion_acceleration_valid_mask=mask(current,'motion_acceleration_valid_mask'),
        fast_action_sequence=fa,fast_future_latent=fl,fast_condition_latent=torch.as_tensor(current['condition_latent']).float(),
        text_embedding=language,delta_sim_seconds=ct-st,delta_actions=age,source_id=int(blob['source_id']),source_sim_time=st,
        state_delta=fs-ss,offset_label='steady_state',fast_sim_time=ct,source_sample_id=int(blob['source_id']),
        fast_sample_id=int(record['sample_id']),cache_age_actions=age,rtc_prefix_actions=prefix,
        source_rtc_prefix_actions=source_prefix,source_rtc_prefix_lengths=source_d,prefix_source='on_policy_executed',target_source=record['target_source'],previous_plan=previous,
        previous_plan_valid_mask=valid,bootstrap_mode='prime_then_hold' if bootstrap else 'none',
        joint_targets=dict(source_future_latent=sl,source_action_sequence=sa,source_action_target_times=starget,
            fast_action_target_times=ftarget,source_video_actual_times=sv,fast_video_actual_times=fv),
        async_timing=dict(source_capture_time=st,current_capture_time=ct,cache_publication_lower_bound_time=lower,
            cache_publication_upper_bound_time=upper,cache_publication_time_is_exact=False,
            source_published_monotonic=published,source_consumed_monotonic=consumed,receive_monotonic=received,query_phase_seconds=float(query-(ftarget[0].item()-DT)),
            query_time=query,fast_latency_seconds=ready-query,ready_time=ready,rtc_prefix_lengths=d))
    return result

class RealRolloutReplayDataset(Dataset):
    def __init__(self,replay_root,*,packed_root,physical_grid_root,subset='train',partition_seed=20260928,
                 validation_fraction=.1,max_cache_age_actions=31,strict_provenance=True,allow_empty=False,**kwargs):
        self.root=Path(replay_root);self.subset=subset;self.epoch=0;self.max_supported_cache_age_actions=int(max_cache_age_actions)
        self.manifest,grid,self.readiness=validate_replay_manifest(replay_root,physical_grid_root,packed_root,strict_provenance=strict_provenance)
        self.packed=PackedAbsoluteMotionDataset(packed_root,max_open_shards=kwargs.get('max_open_shards',16))
        self.normalizer=RoboTwinQposNormalizer.from_stats_path(Path(packed_root)/'action_stats.json',epsilon=1e-6)
        self.physical_grid_identity=grid['physical_grid_identity'];self.parent_episode_partition_identity=grid['episode_partition_identity']
        self.legacy_pairing_identity=grid['legacy_pairing_identity'];self.replay_identity=self.readiness['replay_identity']
        rows=self.manifest['records'];assignments=partition_episodes(rows,seed=partition_seed,validation_fraction=validation_fraction)
        for row in rows:
            if row['subset']!=assignments[int(row['episode_index'])]:raise ValueError('replay train/holdout assignment changed')
        self.episode_partition=assignments
        self.episode_partition_identity=pairing_identity(rows,assignments,seed=partition_seed,validation_fraction=validation_fraction,clock_mode='required',offsets=(),stress_offsets=())
        self.pairing_identity=json_hash(dict(replay_identity=self.replay_identity,episode_partition_identity=self.episode_partition_identity,max_cache_age_actions=max_cache_age_actions))
        self._rows=[r for r in rows if assignments[int(r['episode_index'])]==subset]
        self.records=[PairRecord(int(r['episode_index']),int(r['sample_id']),int(r['sample_id']),0,0,int(r['lang_group_id']),
            int(r['lang_id']),int(r['cache_age_actions']),'steady_state',r['split'],r['task_name'],r['episode_name']) for r in self._rows]
        self.drop_report=dict(records=len(self._rows),subset=subset,target_sources=dict(Counter(r['target_source'] for r in self._rows)))
        if not self._rows and not allow_empty:raise ValueError('no real replay records in requested subset')
    def __len__(self):return len(self._rows)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __getitem__(self,index):
        row=self._rows[int(index)];path=_bound_path(self.root,row['sample'])
        blob=torch.load(path,map_location='cpu',weights_only=True)
        language=self.packed._load_language_embedding(int(row['lang_group_id']),int(row['language_index']))
        sample=adapt_replay_sample(blob,row,normalizer=self.normalizer,language=language,max_cache_age_actions=self.max_supported_cache_age_actions)
        if sample['bootstrap_mode']!=row.get('bootstrap_mode','none') or sample['cache_age_actions']!=int(row['cache_age_actions']):raise ValueError('replay row timing/hold identity differs from actual sample')
        return sample
    def close(self):pass
