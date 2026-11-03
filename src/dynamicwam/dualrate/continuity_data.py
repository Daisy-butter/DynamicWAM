"""Paired real exposures under the SAME source, without changing timestamps."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .producer import DominoPhysicalDataset


def row_defaults(row):
    row=dict(row)
    row.setdefault('measured_state',torch.zeros(37))
    row.setdefault('measured_state_valid',torch.tensor(False))
    row.setdefault('previous_actions',torch.zeros_like(row['fast_known_actions']))
    row.setdefault('previous_valid',torch.zeros_like(row['fast_known_mask']))
    row.setdefault('previous_reference_same',torch.tensor(False))
    return row


class ContinuityDataset(DominoPhysicalDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.exposure_anchored:raise ValueError('measured parallel physical events required')
        self.previous=np.full(len(self.rows),-1,np.int64)
        groups=defaultdict(list)
        for i,(ep,frame,tid) in enumerate(self.rows):
            event=self.event_templates['templates'][int(tid)]
            groups[(int(ep),int(frame))].append((float(event['query_time']),i,event))
        for values in groups.values():
            values.sort(key=lambda x:(x[0],x[1]))
            last=None
            for time,index,event in values:
                if last is not None and 1e-8<time-last[0]<=2*self.dt+1e-6:
                    self.previous[index]=last[1]
                if last is None or time>last[0]+1e-8:last=(time,index)
        self.expert_length=len(self.rows)
        self.replay=[]
        binding=self.manifest.get('continuity_replay')
        if binding:
            from .feedback_replay import digest
            if digest(binding['path'])!=binding['sha256']:raise ValueError('feedback index changed')
            metadata=json.loads(Path(binding['path']).read_text())
            if metadata['schema']!='continuous_feedback_replay_v11':raise ValueError('wrong replay time contract')
            self.replay=metadata['rows']
            if not self.replay:raise ValueError('empty measured feedback pool')
            for entry in self.replay:
                if entry['seed']>=100000 or digest(entry['path'])!=entry['sha256']:
                    raise ValueError('replay identity or train/test split differs')
            self.replay_slots=max(len(self.replay),self.expert_length//3)
            # 25% achieved-state feedback, 75% ALL selected expert windows.
            # Repetition weights the small real pool; it is not extra data.
            self.rows=np.concatenate((self.rows,np.repeat(self.rows[:1],self.replay_slots,axis=0)))
            self.template=None
            example=torch.load(self.replay[0]['path'],map_location='cpu',weights_only=True)
            self.rng_placeholder=torch.zeros_like(example['source_rng_state'])

    def __getitem__(self,index):
        if index>=self.expert_length:
            return self.replay_item((index-self.expert_length)%len(self.replay))
        row=row_defaults(super().__getitem__(index));p=int(self.previous[index])
        prior=row_defaults(super().__getitem__(p)) if p>=0 else dict(row)
        epi=int(self.rows[index,0]);episode=self.episodes[epi]
        seed=int.from_bytes(hashlib.sha256(episode['episode_uid'].encode()).digest()[:8],'little')%(2**63-1)
        for value in (row,prior):
            value['feedback_noise_seed']=torch.tensor(seed,dtype=torch.long)
            value['action_grid_origin']=value['source_time'].clone()
        row['pair_valid']=torch.tensor(p>=0)
        row.update({'previous_'+k:v for k,v in prior.items() if k not in ('previous_actions','previous_valid','pair_valid')})
        if self.replay:
            row['recorded_replay']=torch.tensor(False)
            row['source_rng_state']=self.rng_placeholder.clone()
            row['recorded_reference_actions']=torch.zeros(16,14)
        return row

    def replay_item(self,index):
        sample=torch.load(self.replay[index]['path'],map_location='cpu',weights_only=True)
        if self.template is None:self.template=self.__getitem__(0)
        # Fixed key/shape interface makes mixed distributed batches deterministic.
        # Placeholders below are always paired with false supervision masks.
        row={k:torch.zeros_like(v) for k,v in self.template.items()}
        for k in row:
            if k in sample:
                if row[k].shape!=sample[k].shape:raise ValueError('replay shape differs: '+k)
                row[k]=sample[k]
        row['pair_valid']=torch.tensor(False)
        row['source_action_valid']=torch.ones(16,dtype=torch.bool)
        row['source_action']=sample['source_known_actions'].clone()
        row['reference_available']=torch.tensor(True)
        row['source_prefix_end']=sample['source_action_intervals'][int(sample['source_known_mask'].sum())-1,1]
        row['fast_prefix_end']=sample['action_intervals'][int(sample['fast_known_mask'].sum())-1,1]
        row['query_available_time']=sample['query_time'].clone()
        row['recorded_replay']=torch.tensor(True)
        for k in ('future_loss_valid','future_visual_encode_valid','contact_valid','source_video_loss_valid'):
            if k in row:row[k]=torch.zeros_like(row[k])
        # Current measurements and previous accepted proposal are actual inputs.
        # No expert future, ROI, contact or success label is assigned to them.
        if not bool(row['measured_state_valid']):raise ValueError('replay lost real joint sensors')
        for k,v in list(row.items()):
            if k.startswith('previous_'):continue
            if 'previous_'+k in row:row['previous_'+k]=v.clone()
        return row
