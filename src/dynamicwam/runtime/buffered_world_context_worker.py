"""Development collector: preserve old label schema, defer every file write.

The observer/controller remains identical. This collector's CPU copies are
measured acquisition costs, not a latency estimate for the deployed policy.
"""
from __future__ import annotations
from pathlib import Path
import json
import torch
from dynamicwam.runtime.world_context_worker import WorldContextWorker, cpu_copy, exclusive_tensor
from dynamicwam.models.world_context_feedback import WORLD_CONTEXT_SPEC


def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(tensor_bytes(v) for v in value)
    return 0


class BufferedWorldContextWorker(WorldContextWorker):
    def __init__(self,config):
        self.pending_records=[]; self.pending_bundles={}; self.buffer_bytes=0
        self.finalized=False
        super().__init__(config)
        if self.mode!='graph_zero' or not config.get('collection_dir'):
            raise ValueError('buffered development collection requires graph_zero and a fresh destination')
        if Path(config['collection_dir']).exists():
            raise FileExistsError('collection destination must not exist')

    def _record(self,packet,inputs,age):
        if packet.get('_warmup_only') or self.record_count>=int(self.config.get('samples_per_episode',64)):
            return
        if self.finalized: raise RuntimeError('recording after finalization')
        if self.plan_id not in self.pending_bundles:
            bundle=cpu_copy(dict(format='dynamicwam_world_context_bundle',version=1,
                plan=self.plan,source_state=self.source_state,**self.context,
                plan_id=self.plan_id,source_sim_time=self.source_time,
                source_action_index=self.source_action_index,published_sim_time=None,
                source_noise=self.source_noise,sampling_seed=self.source_sampling_seed,
                text=self.text,text_mask=self.text_mask,base_checkpoint_sha256=self.base_sha,
                controller_protocol_sha256=self.protocol_sha,world_context_spec=WORLD_CONTEXT_SPEC))
            self.pending_bundles[self.plan_id]=bundle; self.buffer_bytes+=tensor_bytes(bundle)
        row=cpu_copy(dict(format='dynamicwam_world_context_raw_record',version=1,
            task=str(self.config['task']),seed=int(self.config['seed']),
            inputs={k:v for k,v in inputs.items() if k not in ('plan','source_state','text','text_mask')},
            packet=packet,plan_id=self.plan_id,age=age,command_index=int(packet['command_index']),
            source_action_index=self.source_action_index,sim_time=float(packet['sim_time']),
            history_dt=float(packet.get('history_dt',0.)),current_sim_time=float(packet['sim_time']),
            source_sim_time=self.source_time,behavior=self.mode,behavior_checkpoint_sha256=self.checkpoint_sha,
            base_checkpoint_sha256=self.base_sha,controller_protocol_sha256=self.protocol_sha))
        self.buffer_bytes+=tensor_bytes(row)
        if self.buffer_bytes>512*1024**2:
            raise MemoryError('collector exceeded registered 512MiB bound; no silent dropping')
        self.pending_records.append(row);self.record_count+=1

    def finalize(self):
        """Called explicitly after physics stops; all records written once."""
        if self.finalized: raise RuntimeError('collector already finalized')
        root=Path(self.config['collection_dir']);root.mkdir(parents=True,exist_ok=False)
        bundles={key:exclusive_tensor(root/'bundles'/f'{key}.pt',value)
                 for key,value in self.pending_bundles.items()}
        index=[]
        for i,row in enumerate(self.pending_records):
            saved=exclusive_tensor(root/'raw'/f'{i:06d}.pt',dict(row,world_context_ref=bundles[row['plan_id']]))
            index.append(dict(**saved,plan_id=row['plan_id'],age=row['age'],command_index=row['command_index']))
        (root/'raw_index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index))
        self.finalized=True
        return dict(records=len(index),buffer_tensor_bytes=self.buffer_bytes,writing='after_physics_stop')


def create_worker_policy(config): return BufferedWorldContextWorker(config)
