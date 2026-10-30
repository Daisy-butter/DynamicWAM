"""Event-indexed native training windows with real observation timestamps."""
import json
from pathlib import Path
import torch
from .producer import DominoPhysicalDataset
from .native_data import derive as derive_v5
from .native_v6 import CONTRACT,Z_EVENTS

class EventDataset(DominoPhysicalDataset):
    def __getitem__(self,index):
        row=super().__getitem__(index);current=self.rows[index]
        pair=False;old=index
        if index>0:
            prev=self.rows[index-1]
            if tuple(current[:2])==tuple(prev[:2]):
                offsets=self.manifest['query_event_offsets']
                pair=offsets[int(current[2])]-offsets[int(prev[2])]==2
                if pair:old=index-1
        previous=super().__getitem__(old) if pair else dict(row)
        row.update({'previous_'+k:v for k,v in previous.items()})
        row['pair_valid']=torch.tensor(pair)
        return row

def derive(source,output):
    offsets=[0,2,4,6,8,10,12];ages=[n*.18 for n in offsets]
    report=derive_v5(source,output,ages)
    out=Path(output)
    for split in ('train','validation'):
        p=out/(split+'.json');m=json.loads(p.read_text())
        m.update(native_v6=True,time_contract=CONTRACT,query_event_offsets=offsets,
            future_event_offsets=list(Z_EVENTS),future_target_domain='command_event',
            action_alignment='ordinal targets; .18s is expert sampling only, never a TOPP end time',
            teacher_alignment='fresh query observation and identical command-origin; never stale A10 prior',
            source_clock='actual recorded camera and state exposure',
            paired_queries='two expert command events; previous model plan is proposal only',
            normalization_provenance='fixed original train-only normalization; no validation refit')
        p.write_text(json.dumps(m,indent=2))
    report.update(schema=CONTRACT,future_target_domain='command_event',future_event_offsets=list(Z_EVENTS),query_event_offsets=offsets)
    (out/'audit.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--output');a=p.parse_args()
    print(json.dumps(derive(a.source,a.output)))
