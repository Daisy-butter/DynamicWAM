#!/usr/bin/env python3
"""Train an observation-routed world readout from actual-controller labels.

The official video/action model is not loaded. State inputs are DOMINO drive
targets, not measured physical joint positions. No expert trajectory targets
or future ground-truth video enter this independent recipe. Stage3 aggregates
newly learned behavior with the retained stage2 zero-feedback observations.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from dynamicwam.models.temporal_plan_feedback import remaining_window
from dynamicwam.models.world_context_feedback import WorldContextFeedbackHead, WORLD_CONTEXT_SPEC
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.world_context_feedback_contract import FORMAT, VERSION, AGES, runtime_contract
from dynamicwam.training.data.world_context_feedback_dataset import WorldContextFeedbackDataset, world_context_collate_fn

SOURCE_FILES=('scripts/train_world_context_feedback.py',
    'src/dynamicwam/training/data/world_context_feedback_dataset.py',
    'src/dynamicwam/models/world_context_feedback_contract.py',
    'src/dynamicwam/models/world_context_feedback.py',
    'src/dynamicwam/models/temporal_plan_feedback.py','src/dynamicwam/models/compact_fast_policy.py',
    'src/dynamicwam/models/fast_observation_encoder.py','src/dynamicwam/observations/multiview.py',
    'src/dynamicwam/action_normalization.py')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def teacher_only_objective(residual, inputs, teacher, *, limit=.1, l2_weight=.01):
    """Return a mean objective plus additive sufficient statistics.

    DDP/accumulation must normalize summed loss_numerator by global count,
    rather than averaging means with different valid tail lengths.
    """
    age=inputs['age_actions']
    if not bool(((age[:,None]==torch.tensor(AGES,device=age.device)[None]).any(1)).all()):
        raise ValueError('world feedback supports only integer ages within the native sixteen-row plan')
    base,tail=remaining_window(inputs['plan'],age)
    arm=torch.ones(14,dtype=torch.bool,device=age.device);arm[[6,13]]=False
    mask=tail[:,:,None]&arm[None,None]&(age>0)[:,None,None]
    delta=teacher.float()-base.float();target=delta.clamp(-limit,limit)
    prediction=residual.float();weight=mask.float();count=weight.expand_as(prediction).sum()
    sse=((prediction-target).square()*weight).sum()
    penalty=(prediction.square()*weight).sum()
    numerator=sse+l2_weight*penalty
    metrics=dict(loss_numerator=numerator,teacher_sse=sse,count=count,
        zero_teacher_sse=(target.square()*weight).sum(),
        raw_teacher_sse=((prediction-delta).square()*weight).sum(),
        zero_raw_teacher_sse=(delta.square()*weight).sum(),residual_square_sum=penalty,
        residual_abs_sum=(prediction.abs()*weight).sum(),
        teacher_clip_count=((delta.abs()>limit).float()*weight).sum())
    return numerator/count.clamp_min(1.),metrics


def prepare_batch(batch, device, *, history_mode, world_mode='context', history_dropout=0., generator=None):
    if history_mode not in ('static','temporal') or not 0<=history_dropout<1:
        raise ValueError('invalid history mode/dropout')
    inputs={k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()}
    # Same RNG consumption for static and temporal training arms.
    keep=(torch.rand(inputs['history_valid'].shape,device=device,generator=generator)>=history_dropout
          if history_dropout else torch.ones_like(inputs['history_valid']))
    inputs['history_valid']=inputs['history_valid']&keep
    if history_mode=='static':inputs['history_valid']=torch.zeros_like(inputs['history_valid'])
    if world_mode not in ('context','disabled'):raise ValueError('unknown world readout ablation')
    if world_mode=='disabled':inputs['context_valid']=torch.zeros_like(inputs['context_valid'])
    return inputs,batch['teacher'].to(device=device,dtype=torch.float32,non_blocking=True)


def initialize_head(v1_path, *, init_feedback=None, data=None, history_mode='temporal', world_mode='context'):
    reference=torch.load(v1_path,map_location='cpu',weights_only=True,mmap=True)
    if (reference.get('format')!='dynamicwam_plan_feedback_checkpoint' or reference.get('version')!=1
            or type(reference.get('global_step')) is not int or reference['global_step']<1):
        raise ValueError('warmstart must be a trained v1 plan-feedback export')
    head=WorldContextFeedbackHead(**reference['spec'])
    schedule=BridgeSchedule.from_flow_match(**reference['runtime_contract']['schedule'])
    if (schedule.to_config()['video_steps'],schedule.to_config()['action_steps'])!=(2,10):
        raise ValueError('on-policy data must retain the official two-video/ten-action sampler')
    legacy=dict(source_plan='immutable_official_full_16',execute_horizon=4,refresh_actions=16,
        ages=[0,4,8,12],age0_correction=False,gripper_correction=False,residual_units='normalized_qpos',
        residual_limit=head.residual_limit,preprocess='three_view_160_imagenet_mask_v1',schedule=schedule.to_config(),
        precision=dict(video='bfloat16',action='float32',solver='float32',feedback_parameters='float32',feedback_autocast='bfloat16'))
    if reference['runtime_contract']!=legacy or not reference.get('action_normalization') or not reference.get('base_checkpoint',{}).get('sha256'):
        raise ValueError('v1 architecture/base/normalization contract is incomplete or incompatible')
    mapping=head.initialize_from_plan_feedback(reference['model'])
    identity=dict(kind='function_preserving_plan_feedback_v1',path=str(Path(v1_path).resolve()),sha256=sha256(v1_path),
        source_global_step=reference['global_step'],mapping=mapping)
    if init_feedback:
        initial=torch.load(init_feedback,map_location='cpu',weights_only=True,mmap=True)
        expected=dict(format=FORMAT,version=VERSION,spec=head.constructor_spec,history_mode=history_mode,
            world_mode=world_mode,action_normalization=reference['action_normalization'],
            runtime_contract=runtime_contract(schedule,head.residual_limit,history_mode=history_mode,world_mode=world_mode))
        if (data is None or any(initial.get(k)!=v for k,v in expected.items())
                or initial.get('base_checkpoint',{}).get('sha256')!=reference['base_checkpoint']['sha256']
                or initial.get('global_step',0)<=0 or initial.get('phase')!='stage2'
                or initial.get('controller_protocol_sha256')!=data.controller_protocol_sha256
                or initial.get('data_manifest_sha256')!=data.stage2_manifest_sha256):
            raise ValueError('stage3 initialization changed data ancestry/base/model/controller contract')
        learned={row['behavior_checkpoint_sha256'] for row in data.records if row['behavior']!='graph_zero'}
        if learned!={sha256(init_feedback)}:
            raise ValueError('stage3 learned-behavior records must come from the exact stage2 checkpoint')
        head.load_state_dict(initial['model'],strict=True)
        identity=dict(kind='world_feedback_refinement',path=str(Path(init_feedback).resolve()),sha256=sha256(init_feedback),
            source_global_step=initial['global_step'],v1_ancestry=identity)
    if any(p.dtype!=torch.float32 or not bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError('all trainable parameters must be finite FP32')
    return head,identity,reference


def export_checkpoint(head, *, reference, data, initialization, history_mode, step,
                      world_mode='context',validation=None, config=None, phase='stage2',
                      examples_seen=0,active_coords_seen=0,behavior_examples_seen=None):
    if any(not bool(torch.isfinite(value).all()) for value in head.state_dict().values()):
        raise FloatingPointError('nonfinite feedback export')
    schedule=BridgeSchedule.from_flow_match(**reference['runtime_contract']['schedule'])
    return dict(format=FORMAT,version=VERSION,model={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        spec=head.constructor_spec,global_step=int(step),phase=phase,history_mode=history_mode,world_mode=world_mode,
        base_checkpoint=reference['base_checkpoint'],action_normalization=reference['action_normalization'],
        runtime_contract=runtime_contract(schedule,head.residual_limit,history_mode=history_mode,world_mode=world_mode),
        world_context_spec=dict(WORLD_CONTEXT_SPEC),controller_protocol_sha256=data.controller_protocol_sha256,
        data_manifest_sha256=data.manifest_sha256,initialization_sha256=initialization['sha256'],
        dataset_identity=data.identity,data_provenance=data.provenance,initialization=initialization,
        training_behaviors=list(data.provenance['training_behaviors']),
        source_hashes={name:sha256(ROOT/name) for name in SOURCE_FILES},
        teacher_loaded=False,teacher_trainable_parameters=0,validation=validation,config=config or {},
        metadata=dict(training_budget=dict(world_size=(config or {}).get('world_size',1),
            per_gpu_batch=(config or {}).get('batch_size',0),gradient_accumulation=(config or {}).get('gradient_accumulation',1),
            global_batch=(config or {}).get('global_batch',0),optimizer_steps=int(step),
            examples_seen=int(examples_seen),active_coords_seen=int(active_coords_seen)),
            behavior_examples_seen=behavior_examples_seen or {},
            stage3_sampling='half_zero_half_exact_stage2_behavior_per_microbatch' if phase=='stage3' else None))


class ExactDistributedSampler(Sampler):
    """Equal training rank lengths without duplicated padded tail samples."""
    def __init__(self,size,*,rank=0,world_size=1,batch_size=1,shuffle=True,seed=0):
        self.size,self.rank,self.world_size,self.batch_size=size,rank,world_size,batch_size
        self.shuffle,self.seed,self.epoch=shuffle,seed,0
        self.effective=(size//(world_size*batch_size))*(world_size*batch_size) if shuffle else size
        if shuffle and not self.effective:raise ValueError('training partition is smaller than a global batch')
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __iter__(self):
        values=list(range(self.size))
        if self.shuffle:random.Random(self.seed+self.epoch).shuffle(values)
        return iter(values[:self.effective][self.rank::self.world_size])
    def __len__(self):return len(range(self.rank,self.effective,self.world_size))


class BalancedBehaviorSampler(Sampler):
    """Every local microbatch contains exactly half old zero and new behavior.

    Independently seeded permutations are cycled to the larger bucket. Reuse
    is explicit rather than an accidental DistributedSampler padding repeat.
    """
    def __init__(self,records,*,rank,world_size,batch_size,seed=0):
        if batch_size<2 or batch_size%2:raise ValueError('stage3 requires an even per-rank microbatch')
        self.groups=[[i for i,r in enumerate(records) if r['age']>0 and (r['behavior']=='graph_zero')==zero]
                     for zero in (True,False)]
        if any(not group for group in self.groups):raise ValueError('stage3 requires old zero and new learned behavior')
        self.rank,self.world_size,self.batch_size,self.seed=rank,world_size,batch_size,seed
        self.epoch=0;self.half=batch_size//2
        self.batches=math.ceil(max(map(len,self.groups))/(world_size*self.half))
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return self.batches*self.batch_size
    def __iter__(self):
        streams=[];total=self.batches*self.world_size*self.half
        for group_id,group in enumerate(self.groups):
            rng=random.Random(self.seed+self.epoch*1009+group_id*9176);stream=[]
            while len(stream)<total:
                shuffled=list(group);rng.shuffle(shuffled);stream.extend(shuffled)
            streams.append(stream[:total])
        values=[]
        for batch in range(self.batches):
            start=(batch*self.world_size+self.rank)*self.half
            old,new=(stream[start:start+self.half] for stream in streams)
            values.extend(old+new)
        return iter(values)


class MappedSampler(Sampler):
    def __init__(self,indices,sampler):self.indices,self.sampler=indices,sampler
    def __len__(self):return len(self.sampler)
    def __iter__(self):return (self.indices[index] for index in self.sampler)
    def set_epoch(self,epoch):self.sampler.set_epoch(epoch)


def mean_metrics(sums):
    n=sums.get('count',0.)
    result={k:v/max(n,1.) for k,v in sums.items() if k!='count'}
    result['active_arm_coordinates']=n
    return result


@torch.no_grad()
def validate(head, loader, *, device, history_mode, world_mode, world_size):
    head.eval();keys=('loss_numerator','teacher_sse','count','zero_teacher_sse','raw_teacher_sse',
        'zero_raw_teacher_sse','residual_square_sum','residual_abs_sum','teacher_clip_count')
    totals=torch.zeros(1+len(AGES),len(keys),device=device,dtype=torch.float64)
    samples=torch.zeros(1+len(AGES),device=device,dtype=torch.float64)
    context_off=torch.zeros(len(keys),device=device,dtype=torch.float64)
    shuffled=torch.zeros(len(keys),device=device,dtype=torch.float64)
    fresh_matched=torch.zeros(len(keys),device=device,dtype=torch.float64)
    for batch in loader:
        inputs,target=prepare_batch(batch,device,history_mode=history_mode,world_mode=world_mode)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            prediction=head(**inputs)
            off=dict(inputs,context_valid=torch.zeros_like(inputs['context_valid']))
            off_prediction=head(**off)
            order=torch.roll(torch.arange(len(target),device=device),1)
            different=torch.tensor([batch['bundle_sha256'][i]!=batch['bundle_sha256'][(i-1)%len(target)]
                for i in range(len(target))],device=device,dtype=torch.bool)
            mismatch=dict(inputs)
            for name in ('world_values','world_coordinates','world_mask'):mismatch[name]=inputs[name][order]
            shuffled_prediction=head(**mismatch)
        _,off_metrics=teacher_only_objective(off_prediction,inputs,target,limit=head.residual_limit,l2_weight=0.)
        context_off+=torch.stack([off_metrics[k].double() for k in keys])
        if bool(different.any()):
            selected_inputs={k:v[different] for k,v in inputs.items()}
            for pred,acc in ((shuffled_prediction,shuffled),(prediction,fresh_matched)):
                _,part=teacher_only_objective(pred[different],selected_inputs,target[different],limit=head.residual_limit,l2_weight=0.)
                acc+=torch.stack([part[k].double() for k in keys])
        for j,age in enumerate((None,)+tuple(AGES)):
            selected=torch.ones(len(target),device=device,dtype=torch.bool) if age is None else inputs['age_actions']==age
            if not bool(selected.any()):continue
            _,m=teacher_only_objective(prediction[selected],{k:v[selected] for k,v in inputs.items()},target[selected],
                limit=head.residual_limit,l2_weight=0.)
            totals[j]+=torch.stack([m[k].double() for k in keys]);samples[j]+=selected.sum()
    if world_size>1:
        for item in (totals,samples,context_off,shuffled,fresh_matched):dist.all_reduce(item)
    if not bool(torch.isfinite(totals).all()):raise FloatingPointError('nonfinite validation')
    result={}
    for j,age in enumerate((None,)+tuple(AGES)):
        result['all' if age is None else f'age_{age}']=dict(mean_metrics(dict(zip(keys,totals[j].cpu().tolist()))),samples=int(samples[j]))
    for name,value in (('world_disabled',context_off),('world_shuffled_different_bundle',shuffled),('world_fresh_matched',fresh_matched)):
        result[name]=mean_metrics(dict(zip(keys,value.cpu().tolist())))
    head.train()
    return result


def atomic_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');torch.save(payload,temporary);temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True);parser.add_argument('--lr',type=float,required=True)
    parser.add_argument('--phase',choices=('stage2','stage3'),required=True)
    parser.add_argument('--init-plan-feedback',type=Path,required=True,help='v1 baseline reference and default warmstart')
    parser.add_argument('--init-feedback',type=Path,help='stage3 same-format initialization; v1 supplies immutable identity')
    parser.add_argument('--history-mode',choices=('static','temporal'),default='static')
    parser.add_argument('--world-mode',choices=('context','disabled'),default='context')
    parser.add_argument('--history-dropout',type=float,default=0.);parser.add_argument('--l2-weight',type=float,default=.01)
    parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--gradient-accumulation','--accum',type=int,default=2)
    parser.add_argument('--workers',type=int,default=0);parser.add_argument('--max-cached',type=int,default=32)
    parser.add_argument('--validate-every',type=int,default=100);parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261008)
    args=parser.parse_args()
    if (min(args.steps,args.batch_size,args.gradient_accumulation,args.validate_every,args.log_every)<1
            or args.lr<=0 or not math.isfinite(args.lr) or args.history_dropout!=0. or args.l2_weight<0
            or not math.isfinite(args.l2_weight) or (args.phase=='stage3')!=bool(args.init_feedback)):
        parser.error('invalid training budget/loss/history settings')
    if not torch.cuda.is_available():raise RuntimeError('use the CPU check script for unit tests; training requires CUDA')
    rank,world_size,local_rank=(int(os.environ.get(k,d)) for k,d in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    if world_size>1:dist.init_process_group('nccl')
    set_seed(args.seed)
    reference=torch.load(args.init_plan_feedback,map_location='cpu',weights_only=True,mmap=True)
    train=WorldContextFeedbackDataset(args.manifest,partition='train',reference_checkpoint=reference,max_cached=args.max_cached)
    validation=WorldContextFeedbackDataset(args.manifest,partition='validation',reference_checkpoint=reference,max_cached=args.max_cached)
    if args.phase=='stage2' and set(train.provenance['training_behaviors'])!={'graph_zero'}:
        raise ValueError('stage2 is initialized on actual zero-feedback plan behavior')
    if args.phase=='stage3' and set(train.provenance['training_behaviors'])!={'graph_zero','world_context'}:
        raise ValueError('stage3 aggregates old zero and newly deployed world-context feedback behavior')
    head,initialization,reference=initialize_head(args.init_plan_feedback,init_feedback=args.init_feedback,
        data=train,history_mode=args.history_mode,world_mode=args.world_mode)
    head.to(device=device,dtype=torch.float32).train()
    wrapped=DistributedDataParallel(head,device_ids=[local_rank],broadcast_buffers=False) if world_size>1 else head
    train_sampler=(BalancedBehaviorSampler(train.records,rank=rank,world_size=world_size,batch_size=args.batch_size,seed=args.seed)
        if args.phase=='stage3' else MappedSampler(train.active_indices,ExactDistributedSampler(len(train.active_indices),
            rank=rank,world_size=world_size,batch_size=args.batch_size,shuffle=True,seed=args.seed)))
    samplers=dict(train=train_sampler,validation=ExactDistributedSampler(len(validation),rank=rank,
        world_size=world_size,batch_size=args.batch_size,shuffle=False,seed=args.seed))
    loaders={name:DataLoader(data,batch_size=args.batch_size,sampler=samplers[name],num_workers=args.workers,
        collate_fn=world_context_collate_fn,pin_memory=True,drop_last=False,
        **({'persistent_workers':True} if args.workers else {})) for name,data in [('train',train),('validation',validation)]}
    run=args.run_dir.resolve()
    if rank==0:
        run.mkdir(parents=True,exist_ok=False)
        (run/'data_manifest.json').write_bytes(args.manifest.read_bytes())
        (run/'source_train_world_context_feedback.py').write_bytes(Path(__file__).read_bytes())
        (run/'data_provenance.json').write_text(json.dumps(train.provenance,indent=2))
    if world_size>1:dist.barrier()
    handle=(run/'metrics.jsonl').open('x') if rank==0 else None
    def log(kind,step=0,**values):
        if rank==0:
            line=json.dumps(dict(kind=kind,step=step,wall_time=time.time(),**values),allow_nan=False)
            print(line,flush=True);handle.write(line+'\n');handle.flush()
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(world_size=world_size,global_batch=args.batch_size*world_size*args.gradient_accumulation)
    initial=export_checkpoint(head,reference=reference,data=train,initialization=initialization,
        history_mode=args.history_mode,world_mode=args.world_mode,step=0,config=configuration,phase=args.phase)
    if rank==0:
        (run/'training_manifest.json').write_text(json.dumps({k:v for k,v in initial.items() if k!='model'},indent=2))
    log('initialized',train_samples=len(train),validation_samples=len(validation),
        trainable_parameters=sum(p.numel() for p in head.parameters()),teacher_loaded=False,teacher_trainable_parameters=0,
        state_semantics='joint_drive_targets',global_batch=configuration['global_batch'],
        query_age_counts=train.provenance['query_ages'],controller_protocol_sha256=train.controller_protocol_sha256)
    initial_validation=validate(head,loaders['validation'],device=device,history_mode=args.history_mode,
        world_mode=args.world_mode,world_size=world_size)
    log('initial_validation',metrics=initial_validation)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.001)
    # Fixed LR is deliberate for the small fixed-label experiment, with no
    # validation-based checkpoint/hyperparameter selection.
    iterator=iter(loaders['train']);epoch=0;started=time.perf_counter()
    examples_seen=active_coords_seen=0;behavior_examples_seen={'graph_zero':0,'learned':0}
    generator=torch.Generator(device=device).manual_seed(args.seed+1000+rank)
    for step in range(1,args.steps+1):
        micros=[];micro_behaviors=[]
        for _ in range(args.gradient_accumulation):
            try:batch=next(iterator)
            except StopIteration:
                epoch+=1;samplers['train'].set_epoch(epoch);iterator=iter(loaders['train']);batch=next(iterator)
            micros.append(prepare_batch(batch,device,history_mode=args.history_mode,world_mode=args.world_mode,
                history_dropout=args.history_dropout,generator=generator))
            micro_behaviors.extend(batch['behavior'])
            if args.phase=='stage3' and batch['behavior'].count('graph_zero')*2!=len(batch['behavior']):
                raise RuntimeError('stage3 local microbatch lost its exact half-old/half-new mixture')
        denominator=sum((torch.minimum(torch.full_like(inputs['age_actions'],4),16-inputs['age_actions'])
            *(inputs['age_actions']>0)).sum().double()*12 for inputs,_ in micros)
        if world_size>1:dist.all_reduce(denominator)
        if not bool(denominator>0):raise RuntimeError('optimizer step contains no supervised feedback coordinates')
        optimizer.zero_grad(set_to_none=True);sums=None
        for j,(inputs,target) in enumerate(micros):
            sync=wrapped.no_sync() if world_size>1 and j+1<len(micros) else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=wrapped(**inputs)
                _,metrics=teacher_only_objective(prediction,inputs,target,limit=head.residual_limit,l2_weight=args.l2_weight)
                loss=metrics['loss_numerator']*world_size/denominator
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite on-policy loss')
                loss.backward()
            if sums is None:sums={k:v.detach().double() for k,v in metrics.items()}
            else:
                for k,v in metrics.items():sums[k]+=v.detach().double()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        gradients=sum(p.grad is not None for p in head.parameters())
        if gradients!=sum(1 for _ in head.parameters()):raise RuntimeError('feedback graph has unused parameters')
        optimizer.step()
        seen=torch.tensor([len(micro_behaviors),micro_behaviors.count('graph_zero')],device=device,dtype=torch.long)
        if world_size>1:dist.all_reduce(seen)
        examples_seen+=int(seen[0]);active_coords_seen+=int(denominator)
        behavior_examples_seen['graph_zero']+=int(seen[1]);behavior_examples_seen['learned']+=int(seen[0]-seen[1])
        if step%args.log_every==0 or step in (1,args.steps):
            keys=sorted(sums);vector=torch.stack([sums[k] for k in keys])
            if world_size>1:dist.all_reduce(vector)
            log('train',step,**mean_metrics(dict(zip(keys,vector.cpu().tolist()))),grad_norm=float(norm),grad_tensors=gradients,
                lr=args.lr,data_epoch=epoch,steps_per_second=step/(time.perf_counter()-started),
                examples_seen=examples_seen,active_coords_seen=active_coords_seen,behavior_examples_seen=behavior_examples_seen)
        if step%args.validate_every==0 or step==args.steps:
            measured=validate(head,loaders['validation'],device=device,history_mode=args.history_mode,
                world_mode=args.world_mode,world_size=world_size)
            measured['initial_validation']=initial_validation
            log('validation',step,metrics=measured)
            if rank==0:
                payload=export_checkpoint(head,reference=reference,data=train,initialization=initialization,
                    history_mode=args.history_mode,world_mode=args.world_mode,step=step,validation=measured,config=configuration,
                    phase=args.phase,examples_seen=examples_seen,active_coords_seen=active_coords_seen,
                    behavior_examples_seen=behavior_examples_seen)
                atomic_save(payload,run/'exports'/f'feedback_step_{step}.pt')
                atomic_save(dict(global_step=step,optimizer=optimizer.state_dict(),data_epoch=epoch),run/'optimizer_latest.pt')
            if world_size>1:dist.barrier()
    log('complete',args.steps,export=str(run/'exports'/f'feedback_step_{args.steps}.pt'))
    if handle:handle.close()
    if world_size>1:dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
