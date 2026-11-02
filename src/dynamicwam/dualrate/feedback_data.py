"""Episode-stratified expert subset plus explicitly identified native replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .native_v9_data import NativeReplayDataset
from .native_v6_data import EventDataset
from .data import sha256_file


def subset_manifest(source,output,fraction=.5,seed=20261102):
    if fraction not in (.5,1.):raise ValueError('explicit half/full episode selection only')
    source=Path(source).resolve(strict=True);output=Path(output)
    if output.exists():raise FileExistsError(output)
    m=json.loads(source.read_text());root=source.parent
    if m['split']!='train':raise ValueError('never select validation/test episodes for training')
    def path(k):return (root/m[k]).resolve(strict=True)
    episodes=json.loads(path('episodes_file').read_text());rows=np.load(path('rows_file'))
    if sha256_file(path('rows_file'))!=m['rows_sha256']:raise ValueError('source index changed')
    groups={}
    for i in sorted(set(rows[:,0].tolist())):
        ep=episodes[i]
        if ep['split']!='train':raise ValueError('split leak')
        # scene_id contains the seed and is almost unique: stratifying on it
        # accidentally retains nearly every episode instead of half the data.
        key=(ep['task'],ep['style'])
        groups.setdefault(key,[]).append(i)
    selected=[];counts={}
    for key,ids in sorted(groups.items()):
        order=sorted(ids,key=lambda i:hashlib.sha256(f'{seed}:{episodes[i]["episode_uid"]}'.encode()).digest())
        keep=order[:max(1,int(np.ceil(len(ids)*fraction)))]
        selected.extend(keep);counts['/'.join(key)]=dict(total=len(ids),selected=len(keep))
    chosen=rows[np.isin(rows[:,0],selected)]
    output.parent.mkdir(parents=True,exist_ok=True);index=output.with_suffix('.rows.npy');np.save(index,chosen)
    for k in ('episodes_file','text_index','prompts_file','source_file_stats'):
        m[k]=str(path(k))
    # Preserve all other paths that were relative to the parent manifest.
    for k in ('visual_projection_path','visual_projection_file','audit_file'):
        if k in m and isinstance(m[k],str) and (root/m[k]).exists():m[k]=str((root/m[k]).resolve())
    m.update(rows_file=str(index.resolve()),rows_sha256=sha256_file(index),
        feedback_subset=dict(fraction=fraction,seed=seed,unit='whole train episode stratified by task/style',
            parent=str(source),parent_sha256=sha256_file(source),selected_episode_ids=[episodes[i]['episode_uid'] for i in sorted(selected)],
            strata=counts,rows=len(chosen),normalization='unchanged original train statistics'))
    output.write_text(json.dumps(m,indent=2));return m['feedback_subset']


def defaults(row):
    row=dict(row)
    row.setdefault('interaction_history',torch.zeros(8,64))
    row.setdefault('interaction_valid',torch.zeros(8,dtype=torch.bool))
    row.setdefault('interaction_intervals',row['query_time'].expand(8,2).clone().to(torch.float64))
    row.setdefault('interaction_command_indices',torch.full((8,),-1,dtype=torch.long))
    row.setdefault('previous_actions',torch.zeros(16,14))
    row.setdefault('previous_valid',torch.zeros(16,dtype=torch.bool))
    row.setdefault('feedback_noise_seed',row.get('frozen_source_seed',torch.tensor(20261102)))
    row.setdefault('native_replay',torch.tensor(False))
    row.setdefault('source_has_committed_waypoint',torch.tensor(False))
    row.setdefault('executed_deviation',torch.tensor(0.))
    return row


def training_seed_manifest(expert_manifest):
    """Predeclare clean TRAIN candidates; native expert feasibility may reject one."""
    p=Path(expert_manifest).resolve(strict=True);m=json.loads(p.read_text())
    if m['split']!='train':raise ValueError('training collection requires train metadata')
    episodes=json.loads((p.parent/m['episodes_file']).read_text())
    selected=set(m['feedback_subset']['selected_episode_ids']);groups={}
    for ep in episodes:
        if ep['episode_uid'] in selected and ep['style']=='clean':
            if ep['split']!='train' or int(ep['seed'])>=100000:raise ValueError('training scene overlaps scored seed domain')
            groups.setdefault(ep['task'],[]).append(ep)
    result={}
    for task,rows in groups.items():
        ordered=sorted(rows,key=lambda row:hashlib.sha256(('collection:'+row['episode_uid']).encode()).digest())
        candidates=[dict(seed=int(e['seed']),episode_uid=e['episode_uid']) for e in ordered]
        if len({e['seed'] for e in candidates})!=len(candidates):raise ValueError('duplicate training seed')
        result[task]=dict(**candidates[0],split='train',style='clean',candidates=candidates)
    if len(result)!=35:raise ValueError('all 35 tasks require declared train scenes')
    return dict(expert_manifest=str(p),expert_manifest_sha256=sha256_file(p),tasks=result,
                selection='fixed train UID order; native expert feasibility only, no policy outcome filtering',strict_train_pool=True)


class FeedbackDataset(NativeReplayDataset):
    def __init__(self,manifest_path,verify_hash=True):
        super().__init__(manifest_path,verify_hash=verify_hash)
        self.expert=EventDataset(self.manifest['expert_manifest'],verify_hash=verify_hash)
        # All available command ages, including terminal windows; age zero alone
        # cannot train a feedback policy whose fresh-observation correction is zero.
        self.expert_indices=np.arange(len(self.expert))
        self.rows=np.zeros((len(self.expert_indices)+len(self.replay),3),dtype=np.int64)
        self.rows[len(self.expert_indices):,2]=1

    def __getitem__(self,index):
        if index<len(self.expert_indices):
            raw=self.expert[int(index)]
            previous={k[9:]:v for k,v in raw.items() if k.startswith('previous_')}
            row=defaults({k:v for k,v in raw.items() if not k.startswith('previous_')})
            previous=defaults(previous)
            uid=self.expert.episodes[int(self.expert.rows[index,0])]['episode_uid']
            seed=int.from_bytes(hashlib.sha256(('feedback:'+uid).encode()).digest()[:8],'big')%((1<<63)-1)
            row['feedback_noise_seed']=torch.tensor(seed);previous['feedback_noise_seed']=torch.tensor(seed)
        else:
            row=super().__getitem__(index)
            required={'interaction_history','interaction_valid','interaction_intervals','interaction_command_indices','feedback_noise_seed'}
            if not required<=row.keys():raise ValueError('native feedback replay lacks measured interaction provenance')
            if int(row['native_query_index'])>0 and not row['interaction_valid'].any():raise ValueError('nonbootstrap replay lost achieved history')
            row=defaults(row)
            from .native_interaction import validate_interaction
            validate_interaction({k:v[None] for k,v in row.items() if isinstance(v,torch.Tensor)})
            # Actual prior proposals are present in the recorded row. Do not
            # pretend the current observation is an earlier observation.
            previous=dict(row);row['pair_valid']=torch.tensor(False)
        row.update({'previous_'+k:v for k,v in previous.items() if not k.startswith('previous_') and k!='pair_valid'})
        row.setdefault('pair_valid',torch.tensor(False))
        return row
