#!/usr/bin/env python3
"""Generate frozen-WAM labels after, never during, a continuous rollout."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'external/flash_attn_runtime'))
from dynamicwam.runtime.world_context_worker import WorldContextWorker, exclusive_tensor
from dynamicwam.runtime.ciwam.wam.plan_feedback_policy import file_sha256
from dynamicwam.models.multirate_types import SourceNoise


def restore_packet(value):
    if torch.is_tensor(value):
        return value.numpy()
    if isinstance(value,dict):
        return {k:restore_packet(v) for k,v in value.items()}
    if isinstance(value,list):
        return [restore_packet(v) for v in value]
    return value


def verified_load(entry):
    if file_sha256(entry['path']) != entry['sha256']:
        raise ValueError('record identity changed: '+entry['path'])
    return torch.load(entry['path'],map_location='cpu',weights_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-dir',type=Path,required=True)
    p.add_argument('--policy-config',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    report=json.loads(a.report.read_text())
    # The collector must explicitly attest a finished physical rollout. A
    # success=False episode remains useful behavior data, an infra error does not.
    if report.get('status') != 'complete':
        raise ValueError('cannot label an incomplete/invalid physical rollout')
    config=json.loads(a.policy_config.read_text())
    config.update(mode='graph_zero')
    config.pop('collection_dir',None)
    config.pop('feedback_checkpoint',None)
    policy=WorldContextWorker(config)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    entries=[json.loads(x) for x in (a.episode_dir/'raw_index.jsonl').read_text().splitlines() if x]
    if not entries: raise ValueError('no real behavior observations')
    results=[]
    for n,entry in enumerate(entries):
        raw=verified_load(entry)
        bundle=verified_load(raw['world_context_ref'])
        if (raw['plan_id']!=bundle['plan_id'] or raw['base_checkpoint_sha256']!=policy.base_sha
                or raw['controller_protocol_sha256']!=policy.protocol_sha):
            raise ValueError('raw context/plan/source identity mismatch')
        packet=restore_packet(raw['packet'])
        policy.base.set_instruction(str(packet['instruction']))
        seed=int(entry['sha256'][:15],16)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            source, independent_noise, _ = policy._source(packet, sampling_seed=seed)
            old=bundle['source_noise']
            age=int(raw['age'])
            # Couple the common action suffix, append independently seeded noise
            # for uncovered rows. Video noise is shared without claiming an exact
            # temporal shift of native video latent bins.
            action_noise=independent_noise.actions.clone()
            action_noise[:,:16-age]=old['actions'][:,age:].to(action_noise)
            noise=SourceNoise(future_latent=old['future_latent'].to(independent_noise.future_latent),
                              actions=action_noise)
            teacher,_=policy.sampler.sample(source,noise)
        row={k:v for k,v in raw.items() if k!='packet'}
        row.update(format='dynamicwam_world_context_record',version=1,teacher=teacher.float(),
            label_provenance=dict(teacher_only=True,current_observation=True,
                raw_record_sha256=entry['sha256'],teacher_base_checkpoint_sha256=policy.base_sha,
                raw_record=dict(path=entry['path'],sha256=entry['sha256']),
                generation='offline_after_rollout',noise_coupling='shared_video_shifted_common_action_suffix',
                labeling_seed=seed))
        saved=exclusive_tensor(a.output_dir/'records'/f'{n:06d}.pt',row)
        results.append(dict(**saved,plan_id=raw['plan_id'],command_index=raw['command_index'],age=age))
    index=a.output_dir/'index.jsonl'
    index.write_text(''.join(json.dumps(x)+'\n' for x in results))
    summary=dict(report,status='complete',task=entries and raw['task'],seed=raw['seed'],
        controller_protocol_sha256=policy.protocol_sha,labeled_records=len(results),
        raw_index=dict(path=str((a.episode_dir/'raw_index.jsonl').absolute()),
                       sha256=file_sha256(a.episode_dir/'raw_index.jsonl')),
        collection_report=dict(path=str(a.report.absolute()),sha256=file_sha256(a.report)))
    (a.output_dir/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    completion=dict(status='complete',phase='offline_relabel',controller_protocol_sha256=policy.protocol_sha,
        base_checkpoint_sha256=policy.base_sha,records=len(results),
        outputs=[dict(path=str(q.absolute()),sha256=file_sha256(q)) for q in (index,a.output_dir/'report.json')])
    (a.output_dir/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')


if __name__=='__main__': main()
