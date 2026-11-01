"""Expert bootstrap plus actual native execution snapshots, never retimed labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .producer import DominoPhysicalDataset
from .data import sha256_file


class NativeReplayDataset:
    def __init__(self,manifest_path,verify_hash=True):
        self.path=Path(manifest_path);self.manifest=json.loads(self.path.read_text())
        m=self.manifest
        if m.get('motion_sampling')!='raw_frames_v1':raise ValueError('unverified motion cache')
        p=Path(m['expert_manifest'])
        if sha256_file(p)!=m['expert_manifest_sha256']:raise ValueError('expert manifest changed')
        self.expert=DominoPhysicalDataset(p,verify_hash=verify_hash)
        self.expert_indices=np.flatnonzero(self.expert.rows[:,2]==0)
        self.replay=m['native_replay_samples']
        self.rows=np.zeros((len(self.expert_indices)+len(self.replay),3),dtype=np.int64)
        self.rows[len(self.expert_indices):,2]=1
        if verify_hash:
            for r in self.replay:
                if sha256_file(Path(r['path']))!=r['sha256']:raise ValueError('native replay shard changed')

    def __len__(self):return len(self.rows)

    def select_domains(self,domains):
        if not domains or not set(domains)<={'expert_bootstrap','native_replay'}:raise ValueError('explicit training domains required')
        if 'native_replay' in domains and not self.replay:raise ValueError('actual native replay is missing')
        wanted=[0 if d=='expert_bootstrap' else 1 for d in domains]
        if set(wanted)=={0,1}:
            expert=np.flatnonzero(self.rows[:,2]==0);replay=np.flatnonzero(self.rows[:,2]==1)
            chosen=np.random.RandomState(20261101).choice(expert,min(len(expert),len(replay)),replace=False)
            return np.concatenate((chosen,replay))
        return np.flatnonzero(np.isin(self.rows[:,2],wanted))

    def __getitem__(self,index):
        if index<len(self.expert_indices):
            row=self.expert[int(self.expert_indices[index])]
            row['native_replay']=torch.tensor(False);row['source_has_committed_waypoint']=torch.tensor(False);row['executed_deviation']=torch.tensor(0.)
            return row
        r=self.replay[index-len(self.expert_indices)]
        payload=torch.load(r['path'],map_location='cpu',weights_only=True)
        if payload['schema']!='native-v9-actual-replay-v1':raise ValueError('unknown replay semantics')
        row=payload['row']
        if sha256_file(Path(payload['language']))!=payload['language_sha256']:raise ValueError('replay language changed')
        language=torch.load(payload['language'],map_location='cpu',weights_only=True)
        for key in ('source_text','teacher_text','text'):row[key]=language['context']
        if not bool(row['native_replay']) or bool(row['future_loss_valid'].any()) or bool(row['contact_valid'].any()):
            raise ValueError('replay must not borrow expert future/contact labels')
        if float(row['source_time'])>float(row['completed_time']) or float(row['completed_time'])>float(row['query_time']):raise ValueError('noncausal replay')
        if int(row['native_query_index'])<int(row['native_source_index']):raise ValueError('replay command reversal')
        return row


def build_manifest(expert_manifest,replay_roots,output):
    expert_manifest=Path(expert_manifest).resolve();m=json.loads(expert_manifest.read_text())
    entries=[];identities=set()
    for root in replay_roots:
        receipt=Path(root)/'replay_acceptance.json'
        proof=json.loads(receipt.read_text())
        if not proof.get('passed'):raise ValueError('native collection did not pass acceptance')
        for item in proof['samples']:
            identity=(item['task'],item['seed'],item['command_index'])
            if identity in identities:raise ValueError('duplicate native sample')
            identities.add(identity)
            if sha256_file(Path(item['path']))!=item['sha256']:raise ValueError('replay shard mismatch')
            entries.append(item)
    m.update(native_v9_dataset=True,expert_manifest=str(expert_manifest),expert_manifest_sha256=sha256_file(expert_manifest),
        motion_sampling='raw_frames_v1',native_replay_samples=entries,
        native_replay_provenance='actual native TOPP: exposure, publication, command identity; fresh Full endpoint teacher',
        native_replay_future_labels='absent and masked, never reused from another trajectory',
        native_timing_distribution='recorded jointly; not independently resampled or relabelled')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(m,indent=2))
    return dict(path=str(output),sha256=sha256_file(output),samples=len(entries),tasks=sorted({x['task'] for x in entries}))
