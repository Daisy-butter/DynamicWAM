"""Actual continuous TRAIN-scene feedback, with immutable physical provenance.

Collection is opt-in and never enabled by the scored evaluator. Missing future
labels are masked. Measured q/qdot cannot be synthesized from drive targets.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import torch


def cpu_batch(batch):
    return {k:v.detach().cpu().clone() for k,v in batch.items()}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


class FeedbackRecorder:
    def __init__(self,root,identity):
        if identity['split']!='train' or identity['seed']>=100000:
            raise ValueError('evaluation scenes cannot enter feedback training')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.identity=identity;self.pool=ThreadPoolExecutor(max_workers=1)
        self.pending=[];self.closed=False

    def capture(self,query_id,source,batch,output,reference):
        from .continuity_v11 import align
        old,rng=source
        row=dict(old,**cpu_batch(batch))
        if not bool(row['measured_state_valid'].all()):
            raise ValueError('actual feedback measurements missing in training collection')
        row['source_rng_state']=rng[None]
        row['recorded_replay']=torch.ones(1,dtype=torch.bool)
        row['completed_time']=row['source_time'].new_tensor([reference.completed_time])
        row['recorded_reference_actions']=reference.payload['reference_actions'].detach().cpu()
        row['fast_action']=output['actions'].detach().cpu()
        row['fast_action_valid']=output['draft_valid'].detach().cpu()|row['fast_known_mask']
        # The teacher starts at the real image exposure. All intervals until
        # the student's protected boundary are the actual executed/queued ones.
        ti=row['teacher_action_intervals']
        values=torch.cat((row['executed_actions'],row['fast_known_actions']),1)
        spans=torch.cat((row['executed_action_intervals'],row['action_intervals']),1)
        valid=torch.cat((row['executed_action_valid'],row['fast_known_mask']),1)
        match=(ti[:,:,None]-spans[:,None]).abs().amax(-1)<1e-7
        match=match & valid[:,None]
        if bool((match.sum(-1)>1).any()):raise ValueError('duplicate teacher commitment')
        index=match.long().argmax(-1)
        known=values.gather(1,index[...,None].expand(-1,-1,14))
        protected=row['action_intervals'][0,int(row['fast_known_mask'].sum()),0]
        mask=ti[...,0]<protected-1e-7
        if bool((mask&~match.any(-1)).any()):raise ValueError('teacher commitment gap')
        row['teacher_known_actions']=torch.where(mask[...,None],known,0.)
        row['teacher_known_mask']=mask
        # Keep only source-horizon completed commands; pad to the fixed training
        # shape. No command is moved in time and no unexecuted row becomes past.
        aligned,av=align(row['executed_actions'],row['executed_action_intervals'],
                         row['source_action_intervals'],row['executed_action_valid'])
        row['executed_actions']=aligned
        row['executed_action_intervals']=row['source_action_intervals'].clone()
        row['executed_action_valid']=av
        row={k:v[0] for k,v in row.items() if k not in ('source_condition_latent','teacher_condition_latent')}
        self.pending=[p for p in self.pending if not p.done() or p.result() is None]
        if len(self.pending)>=8:raise RuntimeError('bounded feedback writer is overloaded')
        def write():
            path=self.root/f'query_{query_id:05d}.pt';tmp=path.with_suffix('.tmp')
            torch.save(row,tmp);tmp.replace(path)
            return True
        self.pending.append(self.pool.submit(write))

    def close(self):
        if self.closed:return
        self.closed=True;self.pool.shutdown(wait=True)
        for p in self.pending:p.result()
        entries=[dict(path=str(p),sha256=digest(p)) for p in sorted(self.root.glob('query_*.pt'))]
        (self.root/'manifest.json').write_text(json.dumps(dict(complete=True,identity=self.identity,rows=entries),indent=2))


def build_replay_index(collection,output,expected_tasks):
    roots=[Path(p) for p in collection];rows=[];tasks=set();excluded=[]
    for root in roots:
        for p in sorted(root.glob('*/episode_*/feedback/manifest.json')):
            report_path=p.parents[1]/'report.json'
            report=json.loads(report_path.read_text())
            if report.get('continuous_valid') is not True or report.get('realtime_valid') is not True:
                # A writer can close cleanly after an episode fails. Its partial
                # trace is diagnostic, not an accepted TRAIN episode.
                excluded.append(str(report_path));continue
            m=json.loads(p.read_text());ident=m['identity']
            if not m['complete'] or ident['split']!='train' or ident['seed']>=100000:
                raise ValueError('incomplete or non-training feedback')
            tasks.add(ident['task'])
            for row in m['rows']:
                if digest(row['path'])!=row['sha256']:raise ValueError('replay bytes changed')
                rows.append(dict(row,task=ident['task'],seed=ident['seed']))
    if tasks!=set(expected_tasks) or len(rows)<4*len(tasks):
        raise ValueError('all-task real feedback coverage missing')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(schema='continuous_feedback_replay_v11',tasks=sorted(tasks),rows=rows,excluded_failed_reports=excluded),indent=2))
    return dict(path=str(path),sha256=digest(path),rows=len(rows),tasks=len(tasks))
