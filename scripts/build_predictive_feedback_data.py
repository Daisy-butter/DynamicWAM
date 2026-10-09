#!/usr/bin/env python3
"""Commit episode-isolated dense world pairs and causally joined teacher labels.

All expensive relabeling runs after physical collection. Future observations
and executed command paths are world-model targets, never policy inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from make_world_context_rt_manifest import identity,read_json,sha256,write_json,official_seeds


def resolve_verified(entry,root):
    p=Path(entry['path']);p=p if p.is_absolute() else Path(root)/p
    if sha256(p)!=entry['sha256']:raise ValueError('changed artifact: '+str(p))
    return p


def validate_rows(rows,official):
    seen=set();partitions={'train':0,'validation':0}
    for row in rows:
        task,seed=row['task'],row['seed'];partition=row['scene']['partition']
        if task not in official or type(seed)is not int or seed in official[task] or (task,seed) in seen:
            raise ValueError('duplicate, unknown, or official-evaluation episode')
        if partition not in partitions:raise ValueError('unknown data partition')
        seen.add((task,seed));partitions[partition]+=1
    if not all(partitions.values()):raise ValueError('both train and heldout episodes are required')
    return partitions


def load_reference(path):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if value.get('format')!='dynamicwam_plan_feedback_checkpoint' or value.get('version')!=1:
        raise ValueError('identified trained v1 reference checkpoint required')
    if not value.get('base_checkpoint',{}).get('sha256') or not value.get('action_normalization'):
        raise ValueError('reference lacks official base/normalization identities')
    return value


def verified_query_controller(packet,obs,commands,i):
    """Keep dispatch and executed-prefix information from the SHA-bound raw query."""
    import numpy as np
    from dynamicwam.runtime.domino_rt.trajectory_recorder import CONTROLLER_VECTORS
    controller=packet['controller_state'];source=float(obs['observation_sim_time'][i])
    for key in CONTROLLER_VECTORS:
        if not np.array_equal(np.asarray(controller[key]),obs['controller_'+key][i]):
            raise ValueError('raw source controller differs from dense source snapshot: '+key)
    for key in ('elapsed','period','ticks','limited_ticks','initialized','source_physics_time'):
        if controller[key]!=obs['controller_'+key][i].item():raise ValueError('source controller scalar differs: '+key)
    knots=controller['future_knots'];count=int(obs['controller_future_knot_count'][i])
    if len(knots)!=count:raise ValueError('raw source future queue differs from dense snapshot')
    for j,knot in enumerate(knots):
        if (knot['end_slot']!=int(obs['controller_future_knot_end_slot'][i,j]) or
            not np.array_equal(np.asarray(knot['q']),obs['controller_future_knot_q'][i,j])):
            raise ValueError('raw source future knot differs from dense snapshot')
    dispatch=float(packet['dispatch_sim_time']);state=controller['dispatch_controller']
    if (not bool(obs['policy_submitted'][i]) or dispatch!=float(obs['dispatch_sim_time'][i]) or
        not np.isfinite(dispatch) or dispatch<source or dispatch-source>2.+1e-8 or
        state['source_physics_time']!=dispatch):
        raise ValueError('raw query must match the actual recorded dispatch')
    start,end=round(source/.004),round(dispatch/.004)
    if abs(source-start*.004)>1e-8 or abs(dispatch-end*.004)>1e-8:
        raise ValueError('query prefix does not lie on the actual physical grid')
    prefix=controller['executed_prefix']
    for key in ('start_sim_time','end_sim_time','q_command','velocity_command'):
        if not np.array_equal(np.asarray(prefix[key]),commands[key][start:end]):
            raise ValueError('raw query prefix differs from the commands actually executed: '+key)
    if end>start:
        for key,command in (('q','q_command'),('v','velocity_command')):
            if not np.array_equal(np.asarray(state[key]),commands[command][end-1]):
                raise ValueError('dispatch controller state differs from the last applied command')
    # Do not reconstruct a source-only object: the dispatch state and real past
    # prefix are part of the causal proposal and must survive the dataset join.
    return controller


def build(args):
    rows=read_json(args.episodes_list)
    if isinstance(rows,dict):rows=rows['episodes']
    partitions=validate_rows(rows,official_seeds(args.official_seed_manifest))
    reference=load_reference(args.reference_checkpoint);protocol_sha=sha256(args.controller_protocol)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    episodes=[]
    for row in rows:
        directory=Path(row['output']);completion_path=directory/'trajectory/completion.json'
        completion=read_json(completion_path)
        manifest_path=resolve_verified(completion['manifest'],completion_path.parent)
        trajectory=read_json(manifest_path);metadata=trajectory['metadata'];provenance=trajectory['provenance']
        if (completion.get('status')!='complete' or trajectory.get('status')!='complete' or
            metadata['task']!=row['task'] or metadata['seed']!=row['seed'] or
            metadata.get('realtime_valid') is not True or provenance.get('controller_protocol_sha256')!=protocol_sha):
            raise ValueError('dense episode has no complete matching physical rollout')
        summary=read_json(directory/'summary.json')
        if (summary.get('status')!='complete' or summary['accepted_record']['sha256']!=provenance['accepted_record_sha256'] or
            sha256(row['policy'])!=provenance['policy_config_sha256']):
            raise ValueError('dense trajectory and collection summary/config identities differ')
        entry=dict(task=row['task'],seed=row['seed'],partition=row['scene']['partition'],
            trajectory_completion=identity(completion_path),source_summary=identity(directory/'summary.json'),
            behavior=summary['policy_config']['mode'],
            behavior_checkpoint_sha256=summary['policy_config'].get('feedback_checkpoint_sha256'),
            accepted_record_sha256=provenance['accepted_record_sha256'])
        if row.get('policy_index'):
            index=resolve_verified(row['policy_index'],Path(args.episodes_list).parent)
            entry['policy_index']=identity(index)
        elif row.get('index'):
            entry['policy_index']=build_policy_records(row,trajectory,manifest_path,out,reference)
        elif args.require_policy:
            raise ValueError('policy training requires actual current-observation teacher labels')
        episodes.append(entry)
    manifest=dict(format='dynamicwam_predictive_feedback_dataset',version=1,
        base_checkpoint=reference['base_checkpoint'],action_normalization=reference['action_normalization'],
        controller_protocol_sha256=protocol_sha,official_seed_manifest=identity(args.official_seed_manifest),
        episodes=episodes,initialization=identity(args.reference_checkpoint),
        input_episodes=identity(args.episodes_list),controller_protocol=identity(args.controller_protocol),
        source_hashes={name:sha256(ROOT/name) for name in ('scripts/build_predictive_feedback_data.py',
            'src/dynamicwam/training/data/predictive_feedback.py',
            'src/dynamicwam/runtime/domino_rt/trajectory_recorder.py')})
    path=out/'dataset.json';write_json(path,manifest,exclusive=True)
    from dynamicwam.training.data.predictive_feedback import PredictiveFeedbackDataset
    evidence={}
    for partition in ('train','validation'):
        data=PredictiveFeedbackDataset(path,partition=partition,reference_checkpoint=reference)
        evidence[partition]=dict(samples=len(data),provenance=data.provenance)
        if args.require_policy:
            policy=PredictiveFeedbackDataset(path,partition=partition,reference_checkpoint=reference,mode='policy')
            evidence[partition]['policy_samples']=len(policy)
    report=dict(format='dynamicwam_predictive_data_build',version=1,status='complete',
        partitions=partitions,evidence=evidence,dataset=identity(path),policy_required=args.require_policy,
        future_target_scope='observed later frames and actual covered command path; not policy inputs')
    write_json(out/'report.json',report,exclusive=True)
    write_json(out/'completion.json',dict(status='complete',dataset=identity(path),report=identity(out/'report.json')),exclusive=True)
    return path


def build_policy_records(row,trajectory,manifest_path,out,reference):
    """Bind offline teacher observations to capture IDs, not nearest timestamps."""
    import numpy as np
    import torch
    from dynamicwam.action_normalization import RoboTwinQposNormalizer
    from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path
    from dynamicwam.runtime.domino_rt.trajectory_recorder import CONTROLLER_VECTORS
    normal=reference['action_normalization']
    normalizer=RoboTwinQposNormalizer.from_stats(normal['stats'],epsilon=normal['epsilon'])
    mean,scale=normalizer._mean.numpy(),normalizer._std.numpy()
    obs_path=resolve_verified(trajectory['artifacts']['observations'],manifest_path.parent)
    with np.load(obs_path,allow_pickle=False) as storage:
        keys=('capture_id','observation_sim_time','policy_state','dispatch_sim_time','policy_submitted')+tuple('controller_'+k for k in
            (*CONTROLLER_VECTORS,'elapsed','period','ticks','limited_ticks','initialized','source_physics_time',
             'future_knot_count','future_knot_q','future_knot_end_slot'))
        obs={k:storage[k].copy() for k in keys}
    commands_path=resolve_verified(trajectory['artifacts']['commands'],manifest_path.parent)
    with np.load(commands_path,allow_pickle=False) as storage:
        commands={k:storage[k].copy() for k in ('start_sim_time','end_sim_time','q_command','velocity_command')}
    indices={int(k):i for i,k in enumerate(obs['capture_id'])}
    index_path=resolve_verified(row['index'],Path(row['output']))
    folder=out/'policy'/row['task']/str(row['seed']);folder.mkdir(parents=True,exist_ok=False)
    entries=[];seen=set()
    def load(entry,root):return torch.load(resolve_verified(entry,root),map_location='cpu',weights_only=True)
    def unbatch(value):
        if torch.is_tensor(value) and value.shape[:1]==(1,):return value[0].detach().cpu().clone()
        return value
    for n,line in enumerate(index_path.read_text().splitlines()):
        if not line.strip():continue
        label_identity=json.loads(line);label=load(label_identity,index_path.parent)
        provenance=label['label_provenance'];raw=load(provenance['raw_record'],index_path.parent)
        bundle=load(raw['world_context_ref'],index_path.parent);packet=raw['packet'];capture=int(packet['capture_id'])
        if capture not in indices or capture in seen:raise ValueError('label capture absent/repeated in exact dense episode')
        i=indices[capture];stamp=float(obs['observation_sim_time'][i]);seen.add(capture)
        if (packet['observation_sim_time']!=stamp or raw['task']!=row['task'] or raw['seed']!=row['seed'] or
            label['base_checkpoint_sha256']!=reference['base_checkpoint']['sha256'] or
            label['controller_protocol_sha256']!=trajectory['provenance']['controller_protocol_sha256'] or
            provenance.get('teacher_only') is not True or provenance.get('current_observation') is not True or
            provenance.get('generation')!='offline_after_rollout'):
            raise ValueError('teacher label did not originate at this exact causal observation')
        np.testing.assert_array_equal(np.asarray(packet['state']),obs['policy_state'][i])
        controller=verified_query_controller(packet,obs,commands,i)
        inputs={k:unbatch(v) for k,v in label['inputs'].items()}
        inputs.update({k:unbatch(bundle[k]) for k in ('plan','source_state','text','text_mask')})
        path,dt,proposal_provenance=proposal_action_path(controller=controller,plan=inputs['plan'].numpy(),
            age=int(raw['age']),observation_time=stamp,mean=mean,scale=scale)
        inputs.update(proposal_path=torch.from_numpy(path),proposal_dt=torch.from_numpy(dt))
        value=dict(format='dynamicwam_predictive_feedback_record',version=1,task=row['task'],seed=row['seed'],
            capture_id=capture,observation_sim_time=stamp,dispatch_sim_time=float(packet['dispatch_sim_time']),
            base_checkpoint_sha256=reference['base_checkpoint']['sha256'],
            controller_protocol_sha256=trajectory['provenance']['controller_protocol_sha256'],
            trajectory_manifest_sha256=sha256(manifest_path),inputs=inputs,teacher=unbatch(label['teacher']),
            proposal_provenance=proposal_provenance,label_provenance=dict(provenance,
                source_sha256=trajectory['provenance']['source_sha256'],original_label=label_identity))
        destination=folder/f'{n:06d}.pt'
        with destination.open('xb') as stream:torch.save(value,stream)
        entries.append(dict(**identity(destination),capture_id=capture))
    if not entries:raise ValueError('no current-observation policy labels')
    target=folder/'index.jsonl'
    with target.open('x') as stream:stream.write(''.join(json.dumps(x)+'\n' for x in entries))
    return identity(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('episodes-list','reference-checkpoint','official-seed-manifest','controller-protocol','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--require-policy',action='store_true')
    build(p.parse_args())


if __name__=='__main__':main()
