#!/usr/bin/env python3
"""World warmup followed by joint predictive bounded-feedback training.

The trained v1 visual encoder is frozen. Only latent_predictor learns from
actual, fully covered control paths and later real camera captures. Evaluation
uses the final optimizer step and episode-level paired sufficient statistics.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dynamicwam.models.predictive_feedback import PredictiveFeedbackHead
from dynamicwam.models.temporal_plan_feedback import remaining_window
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.predictive_feedback_contract import FORMAT, VERSION
from dynamicwam.training.data.predictive_feedback import (
    PredictiveFeedbackDataset, predictive_collate_fn, sha256,
    MINIMUM_NOMINAL_HORIZONS, WINDOW_OFFSETS, HORIZON_RULE,
)

SOURCE_FILES = ('scripts/train_predictive_feedback.py',
    'src/dynamicwam/training/data/predictive_feedback.py',
    'src/dynamicwam/models/predictive_feedback.py',
    'src/dynamicwam/models/predictive_feedback_contract.py',
    'src/dynamicwam/runtime/domino_rt/proposal_path.py',
    'src/dynamicwam/runtime/domino_rt/timed_control.py',
    'src/dynamicwam/models/temporal_plan_feedback.py',
    'src/dynamicwam/models/compact_fast_policy.py',
    'src/dynamicwam/models/fast_observation_encoder.py',
    'src/dynamicwam/models/plan_feedback.py',
    'src/dynamicwam/models/flow_schedule.py',
    'src/dynamicwam/action_normalization.py',
    'src/dynamicwam/observations/multiview.py')


def initialize_head(path):
    reference = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (reference.get('format') != 'dynamicwam_plan_feedback_checkpoint' or reference.get('version') != 1 or
            type(reference.get('global_step')) is not int or reference['global_step'] < 1 or
            not reference.get('base_checkpoint', {}).get('sha256') or not reference.get('action_normalization')):
        raise ValueError('a trained, identified v1 plan-feedback checkpoint is required')
    head = PredictiveFeedbackHead(**reference['spec'])
    schedule=BridgeSchedule.from_flow_match(**reference['runtime_contract']['schedule'])
    expected=dict(source_plan='immutable_official_full_16',execute_horizon=4,refresh_actions=16,
        ages=[0,4,8,12],age0_correction=False,gripper_correction=False,residual_units='normalized_qpos',
        residual_limit=head.residual_limit,preprocess='three_view_160_imagenet_mask_v1',schedule=schedule.to_config(),
        precision=dict(video='bfloat16',action='float32',solver='float32',feedback_parameters='float32',feedback_autocast='bfloat16'))
    if reference['runtime_contract']!=expected or (len(schedule.video_sigmas),len(schedule.action_sigmas))!=(2,10):
        raise ValueError('v1 official sampler/precision/runtime contract differs')
    mapping = head.initialize_from_plan_feedback(reference['model'])
    # This probe must not silently update the pretrained policy or its targets.
    head.requires_grad_(False)
    head.latent_predictor.requires_grad_(True)
    if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError('finite FP32 master parameters are required')
    return head, reference, dict(path=str(Path(path).resolve()), sha256=sha256(path),
        source_global_step=reference['global_step'], mapping=mapping)


def configure_phase(head,phase):
    if phase not in ('world','joint'):raise ValueError('unknown phase')
    head.requires_grad_(phase=='joint')
    head.vision.requires_grad_(False)
    head.latent_predictor.requires_grad_(True)
    head.train();head.vision.eval()


def load_initialization(head,path,reference,data,phase):
    initial=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (initial.get('format')!=FORMAT or initial.get('version')!=VERSION or
            initial.get('spec')!=head.constructor_spec or initial.get('architecture_version')!=head.ARCHITECTURE_VERSION or
            initial.get('base_checkpoint',{}).get('sha256')!=reference['base_checkpoint']['sha256'] or
            initial.get('action_normalization')!=reference['action_normalization'] or
            initial.get('controller_protocol_sha256')!=data.protocol_sha256 or
            initial.get('minimum_nominal_horizons')!=list(MINIMUM_NOMINAL_HORIZONS) or
            initial.get('horizon_rule')!=HORIZON_RULE or
            initial.get('window_offsets')!=[list(x) for x in WINDOW_OFFSETS] or
            initial.get('global_step',0)<1 or initial.get('target_encoder')!='fixed_v1_vision'):
        raise ValueError('predictive initialization identities or architecture differ')
    if phase=='joint' and initial.get('training_scope') not in ('world_only','joint'):
        raise ValueError('joint training requires trained latent initialization')
    if any(not bool(torch.isfinite(x).all()) for x in initial['model'].values()):
        raise ValueError('nonfinite predictive initialization')
    old_splits=initial.get('dataset_provenance',{}).get('episodes',{})
    if old_splits:
        old_val={tuple(x) for x in old_splits['validation']}
        old_train={tuple(x) for x in old_splits['train']}
        new_val={tuple(x) for x in data.provenance['episodes']['validation']}
        new_train={tuple(x) for x in data.provenance['episodes']['train']}
        if old_val & new_train or old_train & new_val:
            raise ValueError('predictive initialization crosses episode train/validation identities')
    head.load_state_dict(initial['model'],strict=True)
    if phase=='joint' and not bool(head.latent_predictor.output.weight.detach().abs().sum()>0):
        raise ValueError('joint training cannot start behind two zero initialized branches')
    return dict(path=str(Path(path).resolve()),sha256=sha256(path),global_step=initial['global_step'],
        phase=initial.get('phase'),data_manifest_sha256=initial.get('data_manifest_sha256'))


def teacher_objective(residual,inputs,teacher,*,limit=.1,l2_weight=.01):
    base,tail=remaining_window(inputs['plan'],inputs['age_actions'])
    arm=torch.ones(14,device=teacher.device,dtype=torch.bool);arm[[6,13]]=False
    # First proposal horizon is one interval after installation. Rows whose
    # command endpoint is at/before installation can never be consumed.
    first_live=torch.round((inputs['proposal_dt'][:,0].float()-.1)/.1).long()
    live=torch.arange(4,device=teacher.device)[None,:]>=first_live[:,None]
    mask=(tail&live)[:,:,None]&arm[None,None]&(inputs['age_actions']>0)[:,None,None]
    target=(teacher.float()-base.float()).clamp(-limit,limit)
    weight=mask.expand_as(target).float();count=weight.sum()
    sse=((residual.float()-target).square()*weight).sum()
    penalty=(residual.float().square()*weight).sum()
    return (sse+l2_weight*penalty)/count.clamp_min(1),dict(numerator=sse+l2_weight*penalty,
        sse=sse,count=count,zero_sse=(target.square()*weight).sum(),
        expired_prefix_rows=(tail&~live).sum(),
        clip_count=(((teacher.float()-base.float()).abs()>limit)*weight).sum())


@torch.no_grad()
def encode_batch(head, batch, device):
    tensors = {k:v.to(device, non_blocking=True) for k,v in batch.items() if torch.is_tensor(v)}
    # Frozen targets and context use the identical FP32 encoder, outside AMP.
    with torch.autocast(device.type, enabled=False):
        encoded = head.encode_observations(tensors['rgb'], tensors['rgb_valid_mask'],
            tensors['previous_head'], tensors['previous_head_mask'], tensors['history_valid'])
        b, h = tensors['future_rgb'].shape[:2]
        target, valid = head.encode_targets(tensors['future_rgb'].flatten(0,1),
            tensors['future_rgb_valid_mask'].flatten(0,1))
        target, valid = target.reshape(b,h,48,-1), valid.reshape(b,h,48)
    encoded.update(history_dt=tensors['history_dt'], history_valid=tensors['history_valid'],
        actual_action_path=tensors['actual_action_path'], future_dt=tensors['future_dt'],
        future_valid=tensors['future_valid'], target=target.float(), target_valid=valid)
    return encoded


def predict(head, encoded, *, ablation='none', action_path=None):
    return head.predict_latents(current_tokens=encoded['current_tokens'],
        previous_head_tokens=encoded['previous_head_tokens'], previous_valid=encoded['previous_valid'],
        history_dt=encoded['history_dt'], history_valid=encoded['history_valid'],
        token_valid=encoded['token_valid'],
        action_path=encoded['actual_action_path'] if action_path is None else action_path,
        future_dt=encoded['future_dt'], future_valid=encoded['future_valid'], ablation=ablation).float()


def valid_tokens(encoded):
    return (encoded['target_valid'] & encoded['token_valid'][:,None] & encoded['future_valid'][:,:,None])


def latent_objective(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError('latent objective shape/mask differs')
    squared = (prediction.float()-target.float()).square()
    count = mask.sum()*target.shape[-1]
    numerator = (squared*mask[...,None]).sum()
    return numerator/count.clamp_min(1), dict(sse=numerator, count=count)


def dynamic_quartile_mask(encoded, valid):
    """Target-defined diagnostic mask only; never fed to predictor/policy.

    Select the largest ceil(25%) changes separately within each 16-patch view,
    preventing wrist motion from displacing every head-camera patch.
    """
    change = (encoded['target']-encoded['current_tokens'][:,None]).square().mean(-1)
    change, mask = change.reshape(*change.shape[:2],3,16), valid.reshape(*valid.shape[:2],3,16)
    rank = change.masked_fill(~mask, -torch.inf).argsort(dim=-1,descending=True,stable=True).argsort(dim=-1)
    count = ((mask.sum(-1)+3)//4).unsqueeze(-1)
    return (mask & (rank<count) & (change>0)).flatten(-2)


def constant_velocity(encoded):
    """Head-only extrapolation; wrists use persistence (no wrist history input)."""
    current = encoded['current_tokens']
    output = current[:,None].expand(-1,2,-1,-1).clone()
    history = encoded['history_valid'][:,None] & encoded['previous_valid'] & encoded['token_valid'][:,:16]
    dt = encoded['history_dt'].clamp_min(1e-6)
    speed = (current[:,:16]-encoded['previous_head_tokens'])/dt[:,None,None]
    output[:,:,:16] += (speed[:,None] * encoded['future_dt'][:,:,None,None] * history[:,None,:,None])
    return output


def shuffled_action_donors(data):
    """Fixed same-task, duration-matched action controls, independently per horizon.

    No future images or target features participate in selecting donors. This
    is an input dependence check, not randomized causal identification.
    """
    result = np.full((len(data),2),-1,dtype=np.int64)
    rows = data.records
    for i,row in enumerate(rows):
        for h in range(2):
            if not row['future_valid'][h]:
                continue
            candidates = [j for j,other in enumerate(rows) if j != i and other['task']==row['task'] and
                other['future_valid'][h] and abs(float(other['future_dt'][h]-row['future_dt'][h])) <= .004+1e-7 and
                not np.array_equal(other['actual_action_path'][h],row['actual_action_path'][h])]
            if candidates:
                # Prefer another episode; otherwise a well-separated observed
                # interval. Original elapsed time remains fixed for both arms.
                result[i,h] = min(candidates,key=lambda j:(rows[j]['seed']==row['seed'],
                    -abs(rows[j]['observation_sim_time']-row['observation_sim_time']),j))
    return result


def add_statistics(accumulator, name, prediction, target, masks):
    slots = accumulator.setdefault(name,{})
    for group,mask in masks.items():
        _,metric = latent_objective(prediction,target,mask)
        pair = slots.setdefault(group,[0.,0.])
        pair[0] += float(metric['sse']);pair[1] += float(metric['count'])


def summarized(statistics):
    return {name:{group:dict(sse=v[0],coordinates=v[1],mse=v[0]/v[1] if v[1] else None)
        for group,v in groups.items()} for name,groups in statistics.items()}


def episode_bootstrap(episodes, *, seed=20261009, draws=5000):
    keys=sorted(episodes)
    result={}
    rng=np.random.default_rng(seed)
    for comparator in ('copy_current','constant_velocity','nohistory','noaction','shuffled_actions'):
        group='shuffle_matched' if comparator=='shuffled_actions' else 'all'
        pairs=np.asarray([[episodes[k].get(name,{}).get(group,[0.,0.]) for name in ('fresh',comparator)] for k in keys])
        differences=[]
        for _ in range(draws):
            total=pairs[rng.integers(0,len(keys),size=len(keys))].sum(0)
            if total[0,1] and total[1,1]:differences.append(total[0,0]/total[0,1]-total[1,0]/total[1,1])
        result[comparator]=dict(fresh_minus_comparator_mse_ci95=np.quantile(differences,[.025,.975]).tolist()
            if differences else None,episodes=len(keys),draws=draws,seed=seed,
            limitation='episodes are the resampling unit; a few episodes provide weak uncertainty estimates')
    return result


@torch.no_grad()
def evaluate(head, data, *, device, batch_size):
    head.eval()
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=predictive_collate_fn)
    donors=shuffled_action_donors(data)
    totals,episodes={},{}
    count=torch.zeros(48,1,device=device,dtype=torch.float64)
    feature_sum=feature_square=None
    position=0;matched_horizons=0;available_horizons=0
    for batch in loader:
        encoded=encode_batch(head,batch,device)
        b=len(batch['record_id']);valid=valid_tokens(encoded)
        dynamic=dynamic_quartile_mask(encoded,valid)
        path=encoded['actual_action_path'].clone()
        shuffle_valid=torch.zeros(b,2,device=device,dtype=torch.bool)
        for i in range(b):
            for h,j in enumerate(donors[position+i]):
                if j>=0:
                    path[i,h]=torch.as_tensor(data.records[j]['actual_action_path'][h],device=device)
                    shuffle_valid[i,h]=True
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            predictions={name:predict(head,encoded,ablation=ablation) for name,ablation in
                (('fresh','none'),('nohistory','nohistory'),('noaction','noaction'))}
            predictions['shuffled_actions']=predict(head,encoded,action_path=path)
        predictions['copy_current']=encoded['current_tokens'][:,None].expand_as(encoded['target'])
        predictions['constant_velocity']=constant_velocity(encoded)
        matched=valid & shuffle_valid[:,:,None]
        masks=dict(all=valid,dynamic_top_quartile=dynamic,shuffle_matched=matched)
        for name,lo,hi in [('head',0,16),('left_wrist',16,32),('right_wrist',32,48)]:
            region=torch.zeros_like(valid);region[:,:,lo:hi]=True
            masks[name]=valid & region;masks[name+'_dynamic']=dynamic & region
        for h in range(2):
            region=torch.zeros_like(valid);region[:,h]=True;masks[f'horizon_{h}']=valid & region
        for name,prediction in predictions.items():
            if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite latent evaluation')
            add_statistics(totals,name,prediction,encoded['target'],masks)
            for i,(task,seed) in enumerate(zip(batch['task'],batch['seed'])):
                ep=episodes.setdefault(f'{task}:{seed}',{})
                add_statistics(ep,name,prediction[i:i+1],encoded['target'][i:i+1],
                    {k:v[i:i+1] for k,v in masks.items()})
        values=encoded['target'].double();weight=valid[...,None].double()
        part=(values*weight).sum((0,1));part_square=(values.square()*weight).sum((0,1))
        feature_sum=part if feature_sum is None else feature_sum+part
        feature_square=part_square if feature_square is None else feature_square+part_square
        count+=weight.sum((0,1))
        matched_horizons+=int((shuffle_valid & encoded['future_valid']).sum())
        available_horizons+=int(encoded['future_valid'].sum());position+=b
    metrics=summarized(totals)
    fresh=metrics['fresh']['all']['mse']
    gates=dict(copy_current= fresh <= .99*metrics['copy_current']['all']['mse'],
        constant_velocity=fresh <= .99*metrics['constant_velocity']['all']['mse'],
        nohistory=fresh <= metrics['nohistory']['all']['mse'],
        noaction=fresh <= metrics['noaction']['all']['mse'],
        shuffled_actions=(matched_horizons>0 and metrics['fresh']['shuffle_matched']['mse'] <=
                          metrics['shuffled_actions']['shuffle_matched']['mse']))
    moving=metrics['fresh']['dynamic_top_quartile']['mse']
    copy_moving=metrics['copy_current']['dynamic_top_quartile']['mse']
    gates['moving_patches']=bool(moving is not None and copy_moving is not None and moving<=.99*copy_moving)
    variance=(feature_square/count.clamp_min(1)-(feature_sum/count.clamp_min(1)).square()).clamp_min(0)
    views={name:float(variance[lo:hi].mean()) for name,lo,hi in
        [('head',0,16),('left_wrist',16,32),('right_wrist',32,48)]}
    return dict(passed=all(gates.values()),gates=gates,metrics=metrics,episode_statistics=episodes,
        episode_clustered_ci=episode_bootstrap(episodes),target_feature_variance_across_records=views,
        shuffle=dict(matched_horizons=matched_horizons,valid_horizons=available_horizons,
            coverage=matched_horizons/max(available_horizons,1),duration_tolerance_seconds=.004,
            rule='same task, distinct path, duration matched; prefer different episode else farthest observed source'),
        constant_velocity_scope='head extrapolation only; wrists persistence; invalid history persistence',
        dynamic_mask='target-defined per-view top quartile of strictly positive latent temporal change; evaluation only',
        interpretation='fixed-target latent prediction feasibility; not policy quality or causal action identification')


@torch.no_grad()
def evaluate_policy(head,data,*,device,batch_size,initial=None):
    """Same trained weights with the predictive path ablated, not retraining."""
    head.eval();totals={};episodes={};position=0;shuffle_count=0
    paths,durations=[],[]
    for i in range(len(data)):
        proposal=data[i]['policy_inputs']
        paths.append(proposal['proposal_path']);durations.append(proposal['proposal_dt'])
    donors=[]
    for i,row in enumerate(data.records):
        choices=[j for j,other in enumerate(data.records) if j!=i and other['task']==row['task'] and
            not torch.equal(paths[i],paths[j]) and
            bool(((durations[i]-durations[j]).abs()<=.004+1e-8).all())]
        donors.append(min(choices,key=lambda j:(data.records[j]['seed']==row['seed'],j)) if choices else None)
    for batch in DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=predictive_collate_fn):
        inputs={k:v.to(device) for k,v in batch['policy_inputs'].items()};teacher=batch['teacher'].to(device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            predictions={name:head(**inputs,ablation=ablation).float() for name,ablation in
                (('fresh','none'),('current_only','current_only'),('noaction','noaction'),('nohistory','nohistory'))}
            if initial is not None:predictions['initialization']=initial(**inputs).float()
            changed=dict(inputs);changed['proposal_path']=inputs['proposal_path'].clone()
            available=torch.zeros(len(teacher),device=device,dtype=torch.bool)
            for i,j in enumerate(donors[position:position+len(teacher)]):
                if j is not None:changed['proposal_path'][i]=paths[j].to(device);available[i]=True
            shuffled=head(**changed).float()
        predictions['zero']=torch.zeros_like(predictions['fresh'])
        for name,pred in predictions.items():
            if not bool(torch.isfinite(pred).all()):raise FloatingPointError('nonfinite policy prediction')
            _,metric=teacher_objective(pred,inputs,teacher,l2_weight=0.)
            sums=totals.setdefault(name,[0.,0.]);sums[0]+=float(metric['sse']);sums[1]+=float(metric['count'])
            for i,(task,seed) in enumerate(zip(batch['task'],batch['seed'])):
                _,m=teacher_objective(pred[i:i+1],{k:v[i:i+1] for k,v in inputs.items()},teacher[i:i+1],l2_weight=0.)
                entry=episodes.setdefault(f'{task}:{seed}',{}).setdefault(name,[0.,0.])
                entry[0]+=float(m['sse']);entry[1]+=float(m['count'])
        if bool(available.any()):
            sub={k:v[available] for k,v in inputs.items()}
            for name,pred in (('fresh_shuffle_matched',predictions['fresh']),('shuffled_proposal',shuffled)):
                _,metric=teacher_objective(pred[available],sub,teacher[available],l2_weight=0.)
                sums=totals.setdefault(name,[0.,0.]);sums[0]+=float(metric['sse']);sums[1]+=float(metric['count'])
            shuffle_count+=int(available.sum())
        position+=len(teacher)
    metrics={k:dict(sse=s,count=n,mse=s/n if n else None) for k,(s,n) in totals.items()}
    fresh=metrics['fresh']['mse']
    gates={k:fresh<=metrics[k]['mse'] for k in ('zero','current_only','noaction','nohistory')}
    if initial is not None:gates['initialization']=fresh<metrics['initialization']['mse']
    gates['shuffled_proposal']=bool(shuffle_count and
        metrics['fresh_shuffle_matched']['mse']<=metrics['shuffled_proposal']['mse'])
    # This tripwire is subordinate to actual-policy canary evaluation. It is
    # not a claim that single-teacher squared error identifies the best action.
    return dict(passed=all(gates.values()),gates=gates,metrics=metrics,episode_statistics=episodes,
        shuffled_proposal_records=shuffle_count,total_records=len(data),
        shuffle_duration_tolerance_seconds=.004,
        interpretation='teacher-only bounded action imitation; ablations use same trained weights; closed-loop quality unmeasured')


class TrainingForward(torch.nn.Module):
    def __init__(self,head,phase,l2_weight):
        super().__init__();self.head=head;self.phase=phase;self.l2_weight=l2_weight
    def forward(self,encoded,policy_inputs=None,teacher=None):
        predicted=predict(self.head,encoded)
        _,world=latent_objective(predicted,encoded['target'],valid_tokens(encoded))
        policy={'numerator':world['sse']*0.,'sse':world['sse']*0.,'count':world['count']*0}
        if self.phase=='joint':
            residual=self.head(**policy_inputs)
            _,policy=teacher_objective(residual,policy_inputs,teacher,l2_weight=self.l2_weight)
        return world['sse'],world['count'],policy['numerator'],policy['count'],policy['sse']


class ExactEpochSampler(torch.utils.data.Sampler):
    def __init__(self,size,rank,world_size,batch_size,seed):
        self.size,self.rank,self.world_size,self.batch_size,self.seed=size,rank,world_size,batch_size,seed
        self.epoch=0;self.effective=size//(world_size*batch_size)*(world_size*batch_size)
        if not self.effective:raise ValueError('training records smaller than one full global microbatch')
    def set_epoch(self,epoch):self.epoch=epoch
    def __iter__(self):
        values=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter(values[:self.effective][self.rank::self.world_size])
    def __len__(self):return self.effective//self.world_size


class BalancedBehaviorSampler(torch.utils.data.Sampler):
    """Explicit repeated permutations; each rank microbatch is half each behavior."""
    def __init__(self,records,rank,world_size,batch_size,seed):
        if batch_size<2 or batch_size%2:raise ValueError('refinement requires even per-rank batch size')
        self.groups=[[i for i,r in enumerate(records) if r['behavior']==kind] for kind in ('graph_zero','predictive')]
        if any(not x for x in self.groups):raise ValueError('refinement requires retained zero and learned behavior')
        self.rank,self.world_size,self.half,self.seed=rank,world_size,batch_size//2,seed
        self.epoch=0;self.batches=math.ceil(max(map(len,self.groups))/(self.half*world_size))
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.batches*self.half*2
    def __iter__(self):
        streams=[];total=self.batches*self.half*self.world_size
        for g,group in enumerate(self.groups):
            rng=random.Random(self.seed+self.epoch*1009+g*7919);stream=[]
            while len(stream)<total:
                permutation=list(group);rng.shuffle(permutation);stream.extend(permutation)
            streams.append(stream[:total])
        result=[]
        for batch in range(self.batches):
            start=(batch*self.world_size+self.rank)*self.half
            for stream in streams:result.extend(stream[start:start+self.half])
        return iter(result)


def export_checkpoint(head,reference,data,initialization,*,phase,step,source_hashes,config,
                      validation,examples_seen,active_coordinates_seen):
    if any(not bool(torch.isfinite(v).all()) for v in head.state_dict().values()):
        raise FloatingPointError('nonfinite predictive export')
    for name,value in reference['model'].items():
        if name.startswith('vision.') and not torch.equal(head.state_dict()[name].cpu(),value.cpu()):
            raise RuntimeError('frozen visual target changed')
    payload=dict(format=FORMAT,version=VERSION,phase=phase,
        training_scope='world_only' if phase=='world' else 'joint',deployable=phase=='joint',
        model={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},spec=head.constructor_spec,
        architecture_version=head.ARCHITECTURE_VERSION,global_step=step,
        base_checkpoint=reference['base_checkpoint'],action_normalization=reference['action_normalization'],
        initialization=initialization,data_manifest_sha256=data.manifest_sha256,
        dataset_provenance=data.provenance,
        controller_protocol_sha256=data.protocol_sha256,source_hashes=source_hashes,
        minimum_nominal_horizons=list(MINIMUM_NOMINAL_HORIZONS),horizon_rule=HORIZON_RULE,
        window_offsets=[list(x) for x in WINDOW_OFFSETS],path_bins=4,
        target_encoder='fixed_v1_vision',policy_labels_used=phase=='joint',validation=validation,
        metadata=dict(examples_seen=examples_seen,active_coordinates_seen=active_coordinates_seen,
            training_budget=dict(world_size=config['world_size'],per_gpu_batch=config['batch_size'],
                gradient_accumulation=config['gradient_accumulation'],global_batch=config['global_batch'],
                optimizer_steps=step,examples_seen=examples_seen),config=config))
    # The shared runtime contract is owned by deployment. A joint checkpoint
    # cannot be advertised as deployable while that contract is absent.
    if phase=='joint':
        from dynamicwam.models.predictive_feedback_contract import runtime_contract
        schedule=BridgeSchedule.from_flow_match(**reference['runtime_contract']['schedule'])
        payload['runtime_contract']=runtime_contract(schedule,head.residual_limit)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--init-plan-feedback',type=Path,required=True)
    parser.add_argument('--init-feedback',type=Path)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=('world','joint'),default='world')
    parser.add_argument('--smoke-only',action='store_true',help='execution-only gate; never scientific evidence')
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--gradient-accumulation','--accum',type=int,default=1)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--world-weight',type=float,default=1.)
    parser.add_argument('--l2-weight',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=20261009)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if (min(args.steps,args.batch_size,args.gradient_accumulation)<1 or
            not np.isfinite([args.lr,args.world_weight,args.l2_weight]).all() or
            args.lr<=0 or args.world_weight<=0 or args.l2_weight<0):
        parser.error('positive finite fixed training budget required')
    if args.phase=='joint' and not args.init_feedback:
        parser.error('joint phase requires a completed predictor warmup checkpoint')
    rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'))
    device=torch.device(f"cuda:{os.environ.get('LOCAL_RANK','0')}" if world_size>1 else args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('unsupported device')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world_size>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
    head,reference,initialization=initialize_head(args.init_plan_feedback)
    mode='world' if args.phase=='world' else 'policy'
    train=PredictiveFeedbackDataset(args.manifest,partition='train',reference_checkpoint=reference,mode=mode)
    heldout=PredictiveFeedbackDataset(args.manifest,partition='validation',reference_checkpoint=reference,mode=mode)
    world_heldout=(heldout if args.phase=='world' else PredictiveFeedbackDataset(args.manifest,
        partition='validation',reference_checkpoint=reference,mode='world'))
    if args.init_feedback:
        initialization['predictive']=load_initialization(head,args.init_feedback,reference,train,args.phase)
    refinement=(args.phase=='joint' and initialization.get('predictive',{}).get('phase')=='joint')
    if refinement:
        learned={r['behavior_checkpoint_sha256'] for r in train.records if r['behavior']=='predictive'}
        if learned!={initialization['predictive']['sha256']}:
            raise ValueError('learned behavior must come from the exact checkpoint being refined')
    elif any(r['behavior']!='graph_zero' for r in train.records):
        raise ValueError('initial predictor/policy stages require registered zero-feedback behavior')
    configure_phase(head,args.phase)
    run=args.run_dir.resolve()
    if rank==0:run.mkdir(parents=True,exist_ok=False)
    if world_size>1:dist.barrier()
    source_hashes={name:sha256(ROOT/name) for name in SOURCE_FILES}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(world_size=world_size,global_batch=world_size*args.batch_size*args.gradient_accumulation)
    config.update(stage='stage3' if refinement else 'stage2' if args.phase=='joint' else 'world',
        behavior_sampling='half_zero_half_exact_initialization_per_microbatch' if refinement else 'zero_behavior')
    manifest=dict(format=FORMAT+'_run',version=1,phase=args.phase,
        data_provenance=train.provenance,initialization=initialization,source_hashes=source_hashes,config=config,
        target_encoder='fixed_v1_vision; no target optimization',validation_selection='final optimizer step only',
        actual_future_action_usage='world loss only; deployment policy receives independent candidate path',
        trainable='latent_predictor only' if args.phase=='world' else 'feedback and predictor; fixed vision and full WAM')
    if rank==0:(run/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    head.to(device)
    initial_head=copy.deepcopy(head).eval().requires_grad_(False) if rank==0 and args.phase=='joint' else None
    train_forward=TrainingForward(head,args.phase,args.l2_weight)
    wrapped=(DistributedDataParallel(train_forward,device_ids=[device.index] if device.type=='cuda' else None)
        if world_size>1 else train_forward)
    parameters=[p for p in head.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.001)
    sampler=(BalancedBehaviorSampler(train.records,rank,world_size,args.batch_size,args.seed) if refinement else
        ExactEpochSampler(len(train),rank,world_size,args.batch_size,args.seed))
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=0,drop_last=True,
        collate_fn=predictive_collate_fn)
    initial=evaluate(head,world_heldout,device=device,batch_size=args.batch_size) if rank==0 else None
    if world_size>1:dist.barrier()
    iterator=iter(loader);epoch=0;examples=coordinates=0;gradient_groups={}
    started=time.perf_counter()
    log=(run/'metrics.jsonl').open('x') if rank==0 else contextlib.nullcontext()
    with log as stream:
        for step in range(1,args.steps+1):
            prepared=[];counts=torch.zeros(2,device=device,dtype=torch.float64)
            for _ in range(args.gradient_accumulation):
                try:batch=next(iterator)
                except StopIteration:
                    epoch+=1;sampler.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
                encoded=encode_batch(head,batch,device)
                policy_inputs=({k:v.to(device) for k,v in batch['policy_inputs'].items()} if args.phase=='joint' else None)
                teacher=batch['teacher'].to(device) if args.phase=='joint' else None
                counts[0]+=valid_tokens(encoded).sum()*encoded['target'].shape[-1]
                if args.phase=='joint':
                    _,m=teacher_objective(torch.zeros_like(teacher),policy_inputs,teacher)
                    counts[1]+=m['count']
                prepared.append((encoded,policy_inputs,teacher))
            if world_size>1:dist.all_reduce(counts)
            if counts[0]<=0 and args.phase=='world':raise ValueError('no valid world coordinates in global batch')
            if args.phase=='joint' and counts[1]<=0:raise ValueError('no active teacher coordinates in global batch')
            head.train();optimizer.zero_grad(set_to_none=True)
            sums=torch.zeros(3,device=device,dtype=torch.float64)
            for micro,(encoded,policy_inputs,teacher) in enumerate(prepared):
                sync=(wrapped.no_sync() if world_size>1 and micro+1<args.gradient_accumulation else contextlib.nullcontext())
                # DDP accumulation must keep both forward and backward inside
                # no_sync. AMP covers forward/loss only; global coordinate
                # counts already include every rank and accumulated microbatch.
                with sync:
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        wnum,wcount,pnum,pcount,psse=wrapped(encoded,policy_inputs,teacher)
                        loss=world_size*args.world_weight*wnum/counts[0].clamp_min(1)
                        if args.phase=='joint':loss=loss+world_size*pnum/counts[1].clamp_min(1)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective')
                    loss.backward()
                sums+=torch.stack((wnum.detach().double(),pnum.detach().double(),psse.detach().double()))
            if any(p.grad is not None for p in head.vision.parameters()):
                raise RuntimeError('gradients entered frozen target encoder')
            if any(p.grad is None for p in parameters):
                missing=[n for n,p in head.named_parameters() if p.requires_grad and p.grad is None]
                raise RuntimeError('unused trainable parameters: '+str(missing))
            for name,p in head.named_parameters():
                if p.grad is not None:
                    group=('latent_output' if name.startswith('latent_predictor.output.') else
                        'latent_body' if name.startswith('latent_predictor.') else
                        'future_readout' if name.startswith('future_readout.') else 'policy')
                    gradient_groups[group]=max(gradient_groups.get(group,0.),float(p.grad.detach().abs().sum()))
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();examples+=config['global_batch'];coordinates+=int(counts.sum())
            if world_size>1:dist.all_reduce(sums)
            if rank==0 and (step==1 or step%20==0 or step==args.steps):
                row=dict(step=step,world_loss=float(sums[0]/counts[0].clamp_min(1)),
                    policy_loss=float(sums[1]/counts[1].clamp_min(1)),teacher_mse=float(sums[2]/counts[1].clamp_min(1)),
                    grad_norm=float(norm),gradient_max_abs_sum=gradient_groups,examples_seen=examples,
                    active_coordinates_seen=coordinates,elapsed_seconds=time.perf_counter()-started)
                text=json.dumps(row,allow_nan=False);print(text,flush=True);stream.write(text+'\n');stream.flush()
    if world_size>1:dist.barrier()
    passed=True
    if rank==0:
        final=evaluate(head,world_heldout,device=device,batch_size=args.batch_size)
        policy=(evaluate_policy(head,heldout,device=device,batch_size=args.batch_size,initial=initial_head)
            if args.phase=='joint' else None)
        required=('latent_output','latent_body')+(('future_readout','policy') if args.phase=='joint' else ())
        gradients_ok=all(gradient_groups.get(k,0)>0 for k in required)
        scientific_passed=bool(final['passed'] and gradients_ok and (policy is None or policy['passed']))
        passed=gradients_ok if args.smoke_only else scientific_passed
        validation=dict(world=final,policy=policy,passed=passed,gradient_groups_nonzero=gradients_ok)
        payload=export_checkpoint(head,reference,train,initialization,phase=args.phase,step=args.steps,
            source_hashes=source_hashes,config=config,validation=validation,examples_seen=examples,
            active_coordinates_seen=coordinates)
        torch.save(payload,run/'checkpoint.pt')
        report=dict(status='complete',phase=args.phase,training_scope=payload['training_scope'],passed=passed,
            global_step=args.steps,examples_seen=examples,active_coordinates_seen=coordinates,
            gradient_max_abs_sum=gradient_groups,initialization=initial,final=validation,
            data_manifest_sha256=train.manifest_sha256,checkpoint_sha256=sha256(run/'checkpoint.pt'),
            source_hashes=source_hashes,policy_training_performed=args.phase=='joint',closed_loop_evaluated=False)
        report.update(smoke_only=args.smoke_only,scientific_passed=scientific_passed)
        (run/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if world_size>1:
        value=torch.tensor(int(passed),device=device);dist.broadcast(value,0);passed=bool(value)
        dist.destroy_process_group()
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
