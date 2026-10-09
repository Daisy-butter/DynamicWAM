"""Hashed, actual-controller observations with offline official-teacher labels.

Neither expert trajectories nor ground-truth future video enter this dataset.
All tensors are recorded from the actual request that ran the behavior policy.
World tensors and the reference plan are deduplicated in an immutable bundle.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dynamicwam.models.world_context_feedback import WORLD_CONTEXT_SPEC

FORMAT = 'dynamicwam_world_context_feedback_data'
RECORD_FORMAT = 'dynamicwam_world_context_record'
BUNDLE_FORMAT = 'dynamicwam_world_context_bundle'
INPUT_KEYS = ('rgb','rgb_valid_mask','state','source_state','plan','age_actions',
    'text','text_mask','previous_head','previous_head_mask','history_dt','history_valid','plan_age_seconds')
WORLD_KEYS = ('world_values','world_coordinates','world_mask')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity_path(entry, root):
    if not isinstance(entry,dict) or set(('path','sha256'))-set(entry):
        raise ValueError('a path/SHA256 identity is required')
    path=Path(entry['path'])
    if not path.is_absolute():path=Path(root)/path
    expected=entry['sha256']
    if not isinstance(expected,str) or len(expected)!=64 or any(x not in '0123456789abcdef' for x in expected):
        raise ValueError('invalid lowercase SHA256')
    return path.resolve(),expected


def checked_bytes(entry,root):
    path,expected=identity_path(entry,root);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f'changed immutable data artifact: {path}')
    return raw,path


def checked_json(entry,root):
    raw,path=checked_bytes(entry,root)
    return json.loads(raw),path


def checked_tensor(entry,root):
    raw,path=checked_bytes(entry,root)
    # New collector canonicalizes numpy values before serializing. No pickle
    # allowlist or unsafe fallback is needed for this independent data format.
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),path


def scalar(value):
    if torch.is_tensor(value):
        if value.numel()!=1:raise ValueError('expected a scalar record field')
        return value.item()
    return value


def _normalized_inputs(record,bundle):
    inputs=dict(record['inputs'])
    for name in ('plan','source_state','text','text_mask'):
        if name not in inputs:
            if name not in bundle:raise ValueError(f'missing bundle-shared input {name}')
            inputs[name]=bundle[name]
        elif name in bundle and not torch.equal(inputs[name],bundle[name]):
            raise ValueError(f'recorded input {name} differs from its exact bundle')
    if set(inputs)!=set(INPUT_KEYS):raise ValueError('head inputs differ from the recorded inference interface')
    result={}
    for name in INPUT_KEYS:
        value=inputs[name]
        if not torch.is_tensor(value) or value.shape[0]!=1:
            raise ValueError(f'recorded input {name} must retain batch size one')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite recorded input {name}')
        result[name]=value[0].clone()
    for name in ('plan','source_state'):
        if not torch.equal(inputs[name],bundle[name]):raise ValueError(f'plan bundle and recorded {name} differ')
        if bundle[name].dtype!=torch.float32:raise ValueError('native action/solver inputs must retain FP32')
        result[name]=bundle[name][0].float().clone()
    if tuple(result['plan'].shape)!=(16,14) or tuple(result['state'].shape)!=(14,) or tuple(result['source_state'].shape)!=(14,):
        raise ValueError('plan/drive-target shape mismatch')
    if tuple(result['rgb'].shape)!=(3,3,160,160) or tuple(result['rgb_valid_mask'].shape)!=(3,160,160):
        raise ValueError('RGB preprocessing is not the shared 160px three-view input')
    if tuple(result['previous_head'].shape)!=(3,160,160) or tuple(result['previous_head_mask'].shape)!=(160,160):
            raise ValueError('previous-snapshot head image shape mismatch')
    for name in ('rgb_valid_mask','previous_head_mask','history_valid','text_mask'):
        if result[name].dtype!=torch.bool:raise ValueError(f'{name} must be a boolean tensor')
    if result['text'].ndim!=2 or result['text'].shape[1]!=4096 or result['text_mask'].shape!=result['text'].shape[:1] or not bool(result['text_mask'].any()):
        raise ValueError('invalid frozen language inputs')
    return result


class WorldContextFeedbackDataset(Dataset):
    def __init__(self,manifest_path,*,partition,reference_checkpoint,max_cached=16):
        if partition not in ('train','validation'):raise ValueError('unknown data partition')
        self.path=Path(manifest_path).resolve();self.manifest_sha256=sha256(self.path)
        self.manifest=json.loads(self.path.read_text());m=self.manifest;root=self.path.parent
        if m.get('format')!=FORMAT or m.get('version')!=1:raise ValueError('unknown world-context data manifest')
        if m.get('base_checkpoint',{}).get('sha256')!=reference_checkpoint['base_checkpoint']['sha256']:
            raise ValueError('data and initialization official WAM differ')
        if m.get('action_normalization')!=reference_checkpoint['action_normalization'] or m.get('world_context_spec')!=WORLD_CONTEXT_SPEC:
            raise ValueError('normalization/world context extraction changed')
        protocol,_=checked_json(m['controller_protocol'],root)
        if not isinstance(protocol,dict):raise ValueError('controller protocol must be an explicit object')
        self.controller_protocol_sha256=m['controller_protocol']['sha256']
        official,_=checked_json(m['official_seed_manifest'],root)
        tasks=official.get('tasks',{})
        if official.get('source_arm')!='official_sync_h16' or len(tasks)!=35 or any(len(v)!=20 or len(set(v))!=20 for v in tasks.values()):
            raise ValueError('the authoritative fixed official 700 episodes must be excluded')
        artifacts=m.get('source_artifacts',[])
        if not artifacts:raise ValueError('collection/relabel source provenance is required')
        for artifact in artifacts:checked_bytes(artifact,root)
        seen={};trials=set();record_digests=set();rows=[];all_rows=[];episodes=[]
        for episode in m['episodes']:
            task,seed=episode['task'],episode['seed'];key=(task,seed)
            trial=key+(episode['behavior'],episode.get('behavior_checkpoint_sha256'))
            if (type(seed)is not int or task not in tasks or seed in tasks[task] or trial in trials or
                    (key in seen and seen[key]!=episode['partition'])):
                raise ValueError('duplicate, malformed, or official evaluation episode entered training')
            seen[key]=episode['partition'];trials.add(trial)
            if episode['partition'] not in ('train','validation') or episode.get('complete') is not True:
                raise ValueError('incomplete episodes or unknown partitions cannot be trained')
            report,_=checked_json(episode['report'],root)
            if (report.get('status')!='complete' or report.get('task')!=task or report.get('seed')!=seed or
                    report.get('controller_protocol_sha256')!=self.controller_protocol_sha256):
                raise ValueError('episode completion/protocol evidence differs')
            raw,index_path=checked_bytes(episode['index'],root)
            records=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            if not records:raise ValueError('a complete training episode has no labeled records')
            previous=-1
            for index in records:
                age=index['age'];command=index['command_index']
                if type(age)is not int or not 0<=age<16 or type(command)is not int or command<=previous:
                    raise ValueError('noncausal/duplicate record order or invalid plan age')
                previous=command
                path,expected=identity_path(index,index_path.parent)
                if sha256(path)!=expected:raise ValueError('labeled record SHA changed')
                if expected in record_digests:raise ValueError('one immutable sample was duplicated in the manifest')
                record_digests.add(expected)
                row=dict(path=str(path),sha256=expected,age=age,command_index=command,
                    plan_id=index['plan_id'],task=task,seed=seed,partition=episode['partition'],
                    behavior=episode['behavior'],behavior_checkpoint_sha256=episode.get('behavior_checkpoint_sha256'))
                all_rows.append(row)
                if episode['partition']==partition:rows.append(row)
            episodes.append(dict(task=task,seed=seed,partition=episode['partition'],
                behavior=episode['behavior'],behavior_checkpoint_sha256=episode.get('behavior_checkpoint_sha256'),
                index_sha256=episode['index']['sha256'],report_sha256=episode['report']['sha256'],records=len(records)))
        if not rows or not any(e['partition']=='train' for e in episodes) or not any(e['partition']=='validation' for e in episodes):
            raise ValueError('nonempty, episode-disjoint train and heldout partitions are required')
        self.records=rows;self.partition=partition;self.max_cached=max(0,int(max_cached));self._cache=OrderedDict();self._bundles=OrderedDict()
        self.active_indices=[i for i,row in enumerate(rows) if row['age']>0]
        if not self.active_indices:raise ValueError('partition has no nonfresh feedback coordinates')
        self.provenance=dict(manifest_path=str(self.path),manifest_sha256=self.manifest_sha256,
            controller_protocol_sha256=self.controller_protocol_sha256,episodes=episodes,
            source_artifacts=artifacts,world_context_spec=dict(WORLD_CONTEXT_SPEC),
            training_behaviors=sorted(set(r['behavior'] for r in all_rows if r['partition']=='train')),
            query_ages={p:dict(sorted(Counter(r['age'] for r in all_rows if r['partition']==p).items())) for p in ('train','validation')},
            state_semantics='joint_drive_targets',labels='offline_official_teacher_only_after_actual_rollout',
            physical_time='recorded_source_current_and_previous_snapshot_clock_not_nominal_action_count')
        self.identity=digest(dict(manifest_sha256=self.manifest_sha256,records=[
            {k:r[k] for k in ('sha256','task','seed','partition','plan_id','age')} for r in all_rows]))
        self.all_record_sha256={r['sha256'] for r in all_rows}
        self.heldout_episodes={(e['task'],e['seed']) for e in episodes if e['partition']=='validation'}
        self.stage2_manifest_sha256=None
        if 'stage2_manifest' in m:
            older,older_path=checked_json(m['stage2_manifest'],root)
            self.stage2_manifest_sha256=m['stage2_manifest']['sha256']
            if older.get('format')!=FORMAT or older.get('version')!=1 or older['controller_protocol']['sha256']!=self.controller_protocol_sha256:
                raise ValueError('stage3 changed the stage2 data/controller provenance')
            previous_heldout={(e['task'],e['seed']) for e in older['episodes'] if e['partition']=='validation'}
            if previous_heldout!=self.heldout_episodes:raise ValueError('stage3 must preserve heldout episode identities')
            for e in older['episodes']:
                data,old_index=checked_bytes(e['index'],older_path.parent)
                if not {json.loads(line)['sha256'] for line in data.decode().splitlines() if line.strip()}<=self.all_record_sha256:
                    raise ValueError('stage3 aggregation removed or replaced a stage2 record')

    def __len__(self):return len(self.records)

    def _remember(self,cache,key,value):
        if self.max_cached:
            cache[key]=value;cache.move_to_end(key)
            while len(cache)>self.max_cached:cache.popitem(last=False)

    def __getitem__(self,index):
        meta=self.records[index];key=meta['sha256']
        if key in self._cache:return self._cache[key]
        record,path=checked_tensor(meta,self.path.parent)
        expected=dict(format=RECORD_FORMAT,version=1,task=meta['task'],seed=meta['seed'],
            plan_id=meta['plan_id'],age=meta['age'],command_index=meta['command_index'],
            behavior=meta['behavior'],behavior_checkpoint_sha256=meta['behavior_checkpoint_sha256'],
            base_checkpoint_sha256=self.manifest['base_checkpoint']['sha256'],
            controller_protocol_sha256=self.controller_protocol_sha256)
        if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('indexed record identity differs from its payload')
        label=record.get('label_provenance',{})
        if (label.get('teacher_only') is not True or label.get('current_observation') is not True or
                label.get('teacher_base_checkpoint_sha256')!=expected['base_checkpoint_sha256']):
            raise ValueError('labels must be the official teacher at the actually observed current state')
        raw_record,_=checked_tensor(label['raw_record'],path.parent)
        if (raw_record.get('world_context_ref')!=record['world_context_ref'] or
                any(raw_record.get(k)!=record.get(k) for k in ('plan_id','age','command_index','behavior','behavior_checkpoint_sha256',
                    'source_sim_time','base_checkpoint_sha256','controller_protocol_sha256')) or
                set(raw_record['inputs'])!=set(record['inputs']) or
                any(not torch.equal(raw_record['inputs'][k],record['inputs'][k]) for k in record['inputs'])):
            raise ValueError('offline relabel changed the actual behavior observation or its plan bundle')
        ref=record['world_context_ref'];bundlekey=ref['sha256']
        if bundlekey in self._bundles:bundle=self._bundles[bundlekey]
        else:
            bundle,_=checked_tensor(ref,path.parent);self._remember(self._bundles,bundlekey,bundle)
        bundle_expected=dict(format=BUNDLE_FORMAT,version=1,plan_id=meta['plan_id'],
            base_checkpoint_sha256=expected['base_checkpoint_sha256'],
            controller_protocol_sha256=self.controller_protocol_sha256)
        if any(bundle.get(k)!=v for k,v in bundle_expected.items()) or bundle.get('world_context_spec')!=WORLD_CONTEXT_SPEC:
            raise ValueError('the record refers to a different generated world/plan bundle')
        if (type(bundle.get('source_action_index')) is not int or
                bundle['source_action_index']>meta['command_index'] or
                record.get('source_action_index')!=bundle['source_action_index']):
            raise ValueError('source-action identity does not match the immutable plan')
        inputs=_normalized_inputs(record,bundle)
        source=float(bundle['source_sim_time']);now=float(record.get('current_sim_time',record.get('sim_time')))
        if (not math.isfinite(source) or not math.isfinite(now) or source<0 or now<source or
                float(record['source_sim_time'])!=source or meta['age']!=int(math.floor((now-source)/.1+1e-6)) or
                int(scalar(inputs['age_actions']))!=meta['age']):
            raise ValueError('plan age does not agree with the actual continuous-clock observation')
        if abs(float(scalar(inputs['plan_age_seconds']))-(now-source))>1e-5:
            raise ValueError('recorded plan-age telemetry changed')
        publication=bundle.get('published_sim_time')
        if publication is not None and (not math.isfinite(float(publication)) or float(publication)<source):
            raise ValueError('publication timestamp predates source observation')
        dt=float(scalar(inputs['history_dt']));valid=bool(scalar(inputs['history_valid']))
        if not math.isfinite(dt) or dt<0 or (valid and dt<=0) or (not valid and dt!=0):
            raise ValueError('invalid actual previous-snapshot history interval')
        for name in WORLD_KEYS:
            value=bundle[name]
            if not torch.is_tensor(value) or value.shape[0]!=1 or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError('nonfinite or invalid packed world tensor')
            inputs[name]=value[0].clone()
        n=inputs['world_values'].shape[0]
        if (n<32 or n%16 or tuple(inputs['world_values'].shape)!=(n,2048) or
                inputs['world_values'].dtype!=torch.bfloat16 or
                tuple(inputs['world_coordinates'].shape)!=(n,4) or inputs['world_coordinates'].dtype!=torch.float32 or
                tuple(inputs['world_mask'].shape)!=(n,) or inputs['world_mask'].dtype!=torch.bool or
                not bool(inputs['world_mask'].all())):
            raise ValueError('packed world-reference layout mismatch')
        coordinates=inputs['world_coordinates']
        if not bool(((coordinates>=0)&(coordinates<=1)).all() &
                ((coordinates[:,0]==0)|(coordinates[:,0]==1)).all() &
                (coordinates[:,0]==0).any() & (coordinates[:,0]==1).any()):
            raise ValueError('world coordinates lost condition/future native bin semantics')
        inputs['context_valid']=torch.tensor(True)
        teacher=record['teacher']
        if not torch.is_tensor(teacher) or tuple(teacher.shape)!=(1,16,14) or teacher.dtype!=torch.float32 or not bool(torch.isfinite(teacher).all()):
            raise ValueError('expected a complete normalized native current-teacher plan')
        result=dict(inputs=inputs,teacher=teacher[0,:4].float().clone(),record_id=key,
            task=meta['task'],seed=meta['seed'],age=meta['age'],behavior=meta['behavior'],
            behavior_checkpoint_sha256=meta['behavior_checkpoint_sha256'],bundle_sha256=bundlekey,
            plan_age_seconds=now-source,history_dt=dt)
        self._remember(self._cache,key,result)
        return result


def world_context_collate_fn(rows):
    if not rows:raise ValueError('empty feedback batch')
    max_text=max(r['inputs']['text'].shape[0] for r in rows)
    max_world=max(r['inputs']['world_values'].shape[0] for r in rows)
    padded=[]
    for row in rows:
        inputs=dict(row['inputs'])
        for name,width in (('text',4096),('world_values',2048),('world_coordinates',4)):
            length=max_text if name=='text' else max_world;value=inputs[name]
            output=torch.zeros(length,width,dtype=value.dtype);output[:len(value)]=value;inputs[name]=output
        for name,length in (('text_mask',max_text),('world_mask',max_world)):
            value=inputs[name];output=torch.zeros(length,dtype=torch.bool);output[:len(value)]=value;inputs[name]=output
        padded.append(inputs)
    return dict(inputs={key:torch.stack([row[key] for row in padded]) for key in padded[0]},
        teacher=torch.stack([row['teacher'] for row in rows]),
        **{key:[row[key] for row in rows] for key in ('record_id','task','seed','age','behavior','behavior_checkpoint_sha256','bundle_sha256')})
