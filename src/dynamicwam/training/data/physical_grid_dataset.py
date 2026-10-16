"""Fixed physical-time labels with causal cache-publication/RTC sampling."""
from __future__ import annotations
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.training.data.physical_grid import (DT, VERSION, json_hash, sha256_file,
    stable_rng, resample_commands, sample_async_timing)
from dynamicwam.training.data.packed_dataset import PackedAbsoluteMotionDataset
from dynamicwam.training.data.paired_world_action_dataset import (PairedWorldActionDataset,
    PairRecord, paired_collate_fn, episode_lang_id)

def validate_physical_grid_readiness(physical_grid_root, future_latent_root, *,
                                     verify_hashes=True, require_full=True):
    """Fail closed on index/holdout/cache identity and complete latent coverage.

    Full hashing belongs in the once-per-launch CPU preflight. Dataset workers
    repeat structural checks and stat every shard, then check each read's IDs.
    """
    root, latent = Path(physical_grid_root), Path(future_latent_root)
    m=json.loads((root/'manifest.json').read_text())
    identity=m.pop('physical_grid_identity'); computed=json_hash(m); m['physical_grid_identity']=identity
    if identity!=computed or m['schema_version']!=VERSION:
        raise ValueError('physical-grid manifest identity/version mismatch')
    if require_full and not m['full_scope']: raise ValueError('subset physical-grid index cannot authorize full training')
    if not m['full_scope']:
        from dynamicwam.training.data.smoke_grid import validate_smoke_subset
        validate_smoke_subset(root,m)
    for path,key in [('episodes.jsonl','episode_index_sha256'),('eligible_sample_ids.npy','eligible_ids_sha256')]:
        if sha256_file(root/path)!=m[key]: raise ValueError('physical-grid index hash mismatch: '+path)
    old=Path(m['packed_root'])
    for name,key in [('dataset.json','metadata_sha256'),('action_stats.json','action_stats_sha256'),('motion_stats.json','motion_stats_sha256')]:
        if sha256_file(old/name)!=m['dataset_identity'][key]: raise ValueError('packed source identity changed: '+name)
    episodes=[json.loads(s) for s in (root/'episodes.jsonl').read_text().splitlines() if s]
    if len(episodes)!=m['counts']['episodes']: raise ValueError('episode coverage mismatch')
    for subset in ('train','validation'):
        if sum(e['subset']==subset for e in episodes)!=m['counts'].get(subset+'_episodes',0):
            raise ValueError('holdout count changed')
    if not (latent/'manifest.json').is_file():
        raise FileNotFoundError('new fixed-grid VAE future latents are required: '+str(latent/'manifest.json'))
    lm=json.loads((latent/'manifest.json').read_text())
    if lm.get('physical_grid_identity')!=identity or lm.get('eligible_ids_sha256')!=m['eligible_ids_sha256']:
        raise ValueError('latent cache belongs to a different physical grid')
    if lm.get('contract')!=m['contract']: raise ValueError('future-video physical time contract mismatch')
    if lm.get('episode_partition_identity')!=m['episode_partition_identity'] or lm.get('complete') is not True:
        raise ValueError('latent cache incomplete or holdout identity mismatch')
    provenance=lm.get('provenance',{})
    code_root=Path(__file__).resolve().parents[4]
    official_vae=old.resolve().parents[2]/'external/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'
    if provenance.get('encoder_kind')!='native_wan2_2_vae_physical_grid_v1' or Path(provenance.get('vae_checkpoint','')).resolve()!=official_vae.resolve():
        raise ValueError('only the actual native official VAE cache can authorize training')
    if provenance.get('encoder_sha256')!=sha256_file(code_root/'scripts/encode_physical_grid_latents.py') or provenance.get('vae_module_sha256')!=sha256_file(code_root/'src/dynamicwam/vendor/wan/modules/vae2_2.py'):
        raise ValueError('encoder/VAE implementation changed; explicit cache migration required')
    expected_preprocess=dict(rgb='cv2_decode_bgr_to_rgb',resize=[[384,320],[192,160]],interpolation='cv2_aspect_pad_then_torch_bilinear_antialias_align_corners_false',range=[-1,1],frames=9,drop_first_latent=True)
    if provenance.get('preprocess')!=expected_preprocess or lm.get('latent_shape')!=[48,2,12,10] or lm.get('latent_dtype')!='float32':
        raise ValueError('latent preprocessing/geometry contract mismatch')
    if verify_hashes and sha256_file(official_vae)!=provenance.get('vae_sha256'):
        raise ValueError('official VAE weights changed')
    expected={str(e['episode_index']):e for e in episodes if e['eligible_samples']}
    if set(lm['episodes'])!=set(expected): raise ValueError('latent cache episode coverage mismatch')
    total=0
    for key,e in expected.items():
        cache=lm['episodes'][key]; path=latent/cache['path']
        if cache['path'] != f'episodes/{int(key):06d}.safetensors': raise ValueError('noncanonical latent shard path')
        if any(cache.get(k)!=v for k,v in provenance.items()) or cache.get('shape')!=[48,2,12,10] or cache.get('dtype')!='float32': raise ValueError('shard encoder/shape/dtype provenance mismatch')
        if cache['sample_count']!=e['eligible_samples'] or cache['index_sha256']!=e['sha256']:
            raise ValueError('latent sample/index coverage mismatch')
        if not path.is_file() or path.stat().st_size!=cache['bytes']: raise ValueError('latent shard missing/truncated')
        if verify_hashes:
            if sha256_file(root/e['index_path'])!=e['sha256']: raise ValueError('episode index changed')
            if sha256_file(path)!=cache['sha256']: raise ValueError('latent shard hash mismatch')
        total+=cache['sample_count']
    if total!=m['counts']['eligible_samples']: raise ValueError('full latent sample coverage mismatch')
    return dict(physical_grid_identity=identity,episode_partition_identity=m['episode_partition_identity'],
        samples=total,episodes=len(expected),full_scope=m['full_scope'],hashes_verified=verify_hashes,
        latent_manifest_sha256=sha256_file(latent/'manifest.json'),
        prefix_source='expert_demonstration',on_policy_ready=False)

class PhysicalGridRecords:
    def __init__(self,dataset,episodes):
        self.dataset=dataset; self.blocks=[];self.starts=[];self.total=0;self.sampling_groups={}
        for meta in episodes:
            ep=dataset._episode(meta['episode_index']); eligible=ep['eligible']
            times=ep['times'][eligible[:,0]] if len(eligible) else np.empty(0)
            for base in range(0,dataset.max_supported_cache_age_actions+1,8):
                positions=np.flatnonzero(times+base*DT<=times[-1]+1e-9).astype(np.int32) if len(times) else np.empty(0,dtype=np.int32)
                if not len(positions): continue
                self.starts.append(self.total); self.blocks.append((meta,base,positions))
                stop=self.total+len(positions)
                self.sampling_groups.setdefault(meta['task_name'],{}).setdefault(meta['episode_index'],{})[
                    ('steady_state',base)]=range(self.total,stop)
                self.total=stop
    def __len__(self): return self.total
    def __getitem__(self,index): return self.dataset._make_record(int(index))

class PhysicalGridPairedWorldActionDataset(PairedWorldActionDataset):
    def __init__(self,packed_root,source_root,*,physical_grid_root,future_latent_root,
                 max_cache_age_actions=31,async_seed=20261015,slow_budget_seconds=2.56,
                 rtc_prefix_max=4,strict_future_latents=True,smoke_subset=False,**kwargs):
        if strict_future_latents is not True: raise ValueError('legacy future latent fallback is forbidden')
        self.physical_grid_root=Path(physical_grid_root);self.future_latent_root=Path(future_latent_root)
        self.readiness=validate_physical_grid_readiness(physical_grid_root,future_latent_root,verify_hashes=False,require_full=not smoke_subset)
        if smoke_subset and self.readiness["full_scope"]: raise ValueError("bounded smoke profile requires the explicitly audited subset")
        self.grid_manifest=json.loads((self.physical_grid_root/'manifest.json').read_text())
        if Path(packed_root).resolve()!=Path(self.grid_manifest['packed_root']).resolve() or Path(source_root).resolve()!=Path(self.grid_manifest['source_root']).resolve():
            raise ValueError('physical-grid source paths disagree')
        if kwargs.get('clock_mode','required')!='required' or int(kwargs.get('chunk_size',16))!=16:
            raise ValueError('physical grid requires exact clocks and 16 endpoints')
        if kwargs.get('partition_seed')!=self.grid_manifest['partition_seed'] or kwargs.get('validation_fraction')!=self.grid_manifest['validation_fraction']:
            raise ValueError('physical-grid holdout configuration changed')
        if not 0<=int(max_cache_age_actions)<=31 or not 1<=int(rtc_prefix_max)<=4:
            raise ValueError('unsupported causal cache-age or RTC prefix support')
        self.packed=PackedAbsoluteMotionDataset(packed_root,max_open_shards=kwargs.get('max_open_shards',16))
        self.source_root=Path(source_root); self.subset=kwargs.get('subset','train');self.clock_mode='required'
        self.chunk_size=16;self.global_downsample_rate=int(kwargs.get('global_downsample_rate',3));self.seconds_per_frame=DT
        self.frame_reader=kwargs.get('frame_reader');self.clock_reader=None;self.allow_empty=kwargs.get('allow_empty',False)
        self.max_open_clocks=int(kwargs.get('max_open_clocks',16));self._clocks=OrderedDict();self._captures=OrderedDict()
        self._grid_episodes=OrderedDict();self._latent_handles=OrderedDict(); self.epoch=0;self.joint_supervision=True
        self.cache_age_jitter=True;self.cache_age_seed=int(async_seed);self.async_seed=int(async_seed)
        self.max_supported_cache_age_actions=int(max_cache_age_actions);self.slow_budget_seconds=float(slow_budget_seconds);self.rtc_prefix_max=int(rtc_prefix_max)
        self.physical_grid_identity=self.grid_manifest['physical_grid_identity']
        self.episode_partition_identity=self.grid_manifest['episode_partition_identity'];self.legacy_pairing_identity=self.grid_manifest['legacy_pairing_identity']
        self.pairing_identity=json_hash(dict(grid=self.physical_grid_identity,async_schema='joint_source_current_prefix_v2',max_age=max_cache_age_actions,
            async_seed=async_seed,slow_budget_seconds=slow_budget_seconds,rtc_prefix_max=rtc_prefix_max))
        all_meta=[json.loads(s) for s in (self.physical_grid_root/'episodes.jsonl').read_text().splitlines() if s]
        self._episode_meta={int(e['episode_index']):e for e in all_meta}
        self.episode_partition={int(e['episode_index']):e['subset'] for e in all_meta}
        selected=[e for e in all_meta if e['subset']==self.subset and e['split'] in set(kwargs['splits'])]
        self.records=PhysicalGridRecords(self,selected);self.drop_report=self.grid_manifest['drops']
        self.normalizer=RoboTwinQposNormalizer.from_stats_path(Path(packed_root)/'action_stats.json',epsilon=1e-6)
        if not len(self.records) and not self.allow_empty: raise ValueError('physical grid kept no pairs')
    def _episode(self,e):
        if e not in self._grid_episodes:
            with np.load(self.physical_grid_root/self._episode_meta[e]['index_path']) as z:
                self._grid_episodes[e]={k:z[k] for k in z.files}
        self._grid_episodes.move_to_end(e)
        while len(self._grid_episodes)>16:self._grid_episodes.popitem(last=False)
        return self._grid_episodes[e]
    def _make_record(self,index):
        if index<0 or index>=len(self.records): raise IndexError(index)
        b=bisect_right(self.records.starts,index)-1;meta,base,positions=self.records.blocks[b]
        pos=int(positions[index-self.records.starts[b]]);e=int(meta['episode_index']);ep=self._episode(e)
        eligible=ep['eligible'];times=ep['times'][eligible[:,0]];frame,sid=map(int,eligible[pos])
        maxage=min(self.max_supported_cache_age_actions,base+7,int(np.floor((times[-1]-times[pos])/DT+1e-8)))
        age=int(stable_rng(self.async_seed,self.epoch,sid,base).integers(base,maxage+1))
        fastpos=max(pos,int(np.searchsorted(times,times[pos]+age*DT+1e-9,side='right')-1))
        ff,fid=map(int,eligible[fastpos]);group=int(meta['lang_group_id'])
        return PairRecord(e,sid,fid,frame,ff,group,episode_lang_id(self.packed.lang_groups[group]),
            age*self.global_downsample_rate,'steady_state',meta['split'],meta['task_name'],meta['episode_name'])
    def _record_for_epoch(self,index):return self._make_record(index)
    def _future(self,e,position,sample_id):
        from safetensors import safe_open
        if e not in self._latent_handles:
            self._latent_handles[e]=safe_open(str(self.future_latent_root/'episodes'/f'{e:06d}.safetensors'),framework='pt',device='cpu')
        handle=self._latent_handles[e];self._latent_handles.move_to_end(e)
        while len(self._latent_handles)>8:self._latent_handles.popitem(last=False)
        if int(handle.get_slice('sample_ids')[position])!=sample_id: raise ValueError('future latent sample mismatch')
        value=handle.get_slice('future_latents')[position]
        if value.dtype!=torch.float32 or tuple(value.shape)!=(48,2,12,10) or not bool(torch.isfinite(value).all()):
            raise ValueError('actual future latent has invalid dtype/shape/nonfinite values')
        return value
    def __getitem__(self,index):
        record=self._make_record(int(index));sample=super().__getitem__(index);ep=self._episode(record.episode_index)
        ids=ep['eligible'][:,1];sp=int(np.searchsorted(ids,record.source_sample_id));fp=int(np.searchsorted(ids,record.fast_sample_id))
        source_time=float(ep['times'][record.source_frame]);fast_time=float(ep['times'][record.fast_frame])
        st,ft=ep['action_times'][sp],ep['action_times'][fp]
        sa=self.normalizer.normalize(torch.from_numpy(resample_commands(ep['times'],ep['commands'],st)))
        fa=self.normalizer.normalize(torch.from_numpy(resample_commands(ep['times'],ep['commands'],ft)))
        sl=self._future(record.episode_index,sp,record.source_sample_id);fl=self._future(record.episode_index,fp,record.fast_sample_id)
        sample.update(physical_grid_version=VERSION,fast_action_sequence=fa,fast_future_latent=fl,
            bridge_noise_shape_latent=torch.zeros_like(sl),rtc_prefix_actions=fa.clone(),
            async_timing=sample_async_timing(source_time,fast_time,ft,seed=self.async_seed,epoch=self.epoch,
                source_id=record.source_sample_id,current_id=record.fast_sample_id,slow_budget_seconds=self.slow_budget_seconds,
                rtc_prefix_max=self.rtc_prefix_max,source_action_times=st),prefix_source='expert_demonstration')
        source_length=sample['async_timing']['source_rtc_prefix_lengths']
        sample['source_rtc_prefix_lengths']=source_length
        sample['source_rtc_prefix_actions']=torch.zeros_like(sa)
        sample['source_rtc_prefix_actions'][:source_length]=sa[:source_length]
        sample['rtc_prefix_actions'][sample['async_timing']['rtc_prefix_lengths']:]=0
        sample['joint_targets'].update(source_future_latent=sl,source_action_sequence=sa,
            source_action_target_times=torch.from_numpy(st.copy()),fast_action_target_times=torch.from_numpy(ft.copy()),
            source_video_actual_times=torch.from_numpy(ep['video_actual_times'][sp].copy()),
            fast_video_actual_times=torch.from_numpy(ep['video_actual_times'][fp].copy()))
        return sample
    def __getstate__(self):
        state=super().__getstate__();state['_grid_episodes']=OrderedDict();state['_latent_handles']=OrderedDict();return state

def physical_grid_collate_fn(batch):
    if any(s.get('physical_grid_version')!=VERSION for s in batch): raise ValueError('mixed grid versions')
    result=paired_collate_fn(batch);result['physical_grid_version']=VERSION
    result['rtc_prefix_actions']=torch.stack([s['rtc_prefix_actions'] for s in batch])
    if any('source_rtc_prefix_actions' in s for s in batch):
        if not all('source_rtc_prefix_actions' in s for s in batch):raise ValueError('source prefix conditioning cannot be missing in part of a batch')
        result['source_rtc_prefix_actions']=torch.stack([s['source_rtc_prefix_actions'] for s in batch])
        result['source_rtc_prefix_lengths']=torch.tensor([s['source_rtc_prefix_lengths'] for s in batch],dtype=torch.long)
    result['async_timing']={k:torch.tensor([s['async_timing'][k] for s in batch],
        dtype=torch.long if k in ('rtc_prefix_lengths','source_rtc_prefix_lengths') else (torch.bool if k=='cache_publication_time_is_exact' else torch.float64)) for k in batch[0]['async_timing']}
    result['source_sim_time']=torch.tensor([s['source_sim_time'] for s in batch],dtype=torch.float64)
    result['delta_sim_seconds']=torch.tensor([s['delta_sim_seconds'] for s in batch],dtype=torch.float64)
    result['prefix_source']=[s['prefix_source'] for s in batch]
    if any('previous_plan' in s for s in batch) and not all('previous_plan' in s for s in batch):
        raise ValueError('offline/replay mixed batches need an explicit context presence router')
    if any('previous_plan' in s for s in batch):
        result['previous_plan']=torch.stack([s.get('previous_plan',torch.zeros_like(s['fast_action_sequence'])) for s in batch])
        result['previous_plan_valid_mask']=torch.stack([s.get('previous_plan_valid_mask',torch.zeros(16,dtype=torch.bool)) for s in batch])
    result['target_source']=[s.get('target_source','expert_demonstration') for s in batch]
    result['bootstrap_mode']=[s.get('bootstrap_mode','none') for s in batch]
    return result
