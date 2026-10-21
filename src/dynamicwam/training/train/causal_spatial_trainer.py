"""V4 joint spatial-world/action training, with explicit data and causal contracts."""
from __future__ import annotations
import copy, gc, hashlib, json, os, re, shutil, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dynamicwam.models.rtc_prefix import PrefixCondition,apply_prefix,masked_suffix_mse,prefix_mask
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.native_wam_teacher import sample_native_wam
from dynamicwam.models.multirate_types import SourceObservation,SourceNoise
from dynamicwam.inference.causal_spatial_loader import (build_from_initialization,build_native,native_payload,
 load_full,make_payload,make_fast_payload,write_payload,sha256)
from dynamicwam.training.train.common import build_training_flow_scheduler
from dynamicwam.training.data.physical_grid_dataset import PhysicalGridPairedWorldActionDataset
from dynamicwam.training.data.multirate_sampling import BalancedPairSampler,ProcessGroupAlignedSampler
from dynamicwam.training.utils.scheduler import create_scheduler


def tensor_to(value,device):
 if torch.is_tensor(value):return value.to(device,non_blocking=True)
 if isinstance(value,dict):return {k:tensor_to(v,device) for k,v in value.items()}
 if isinstance(value,list):return [tensor_to(v,device) for v in value]
 return value

def build_dataset(recipe,subset='train'):
 from dynamicwam.training.data.causal_spatial_dataset import CausalSpatialDataset
 d=recipe['data']
 base=PhysicalGridPairedWorldActionDataset(d['packed_root'],d['source_root'],
  physical_grid_root=d['physical_grid_root'],future_latent_root=d['future_latent_root'],
  max_cache_age_actions=31,async_seed=20261021,slow_budget_seconds=2.56,rtc_prefix_max=2,
  rtc_prefix_sampling='delay_minimum_v1',strict_future_latents=True,smoke_subset=False,
  splits=['clean','randomized'],execute_horizon=2,every_fast_queries=8,global_downsample_rate=3,
  chunk_size=16,seconds_per_frame=1/30,stress_offsets=[],max_open_shards=4,
  subset=subset,partition_seed=20260928,validation_fraction=.1,clock_mode='required',
  cache_age_actions=list(range(32)),joint_supervision=True,cache_age_jitter=True,cache_age_seed=20261021)
 return CausalSpatialDataset(base,window_size=recipe['model']['history_length'],
  window_stride_seconds=.32,prefix_length=2,force_fresh_memory=recipe['phase']=='interface_adapt')

def align_proposal(actions,source_times,target_times):
 """No extrapolation or expired-tail padding; gripper is zero-order hold."""
 times=source_times.double().contiguous();targets=target_times.double().contiguous()
 right=torch.searchsorted(times,targets,right=True).clamp(1,times.shape[1]-1);left=right-1
 tl=times.gather(1,left);tr=times.gather(1,right)
 al=actions.gather(1,left[...,None].expand(-1,-1,actions.shape[-1]));ar=actions.gather(1,right[...,None].expand(-1,-1,actions.shape[-1]))
 w=((targets-tl)/(tr-tl).clamp_min(1e-9)).clamp(0,1).float()
 aligned=al+(ar-al)*w[...,None]
 for i in (6,13):aligned[...,i]=torch.where(targets>=tr-1e-9,ar[...,i],al[...,i])
 valid=(targets>=times[:,:1]-1e-9)&(targets<=times[:,-1:]+1e-9)
 return torch.where(valid[...,None],aligned,0.),valid

class Objective(nn.Module):
 def __init__(self,model,loss):
  super().__init__();self.student=model;self.weights=loss
 def forward(self,batch,targets,teacher_action=None):
  out=self.student(batch)
  action=masked_suffix_mse(out['action_pred'],targets['action_velocity'],batch['rtc_prefix'])
  future=F.mse_loss(out['video_pred'].float(),targets['future_velocity'].float())
  valid=out['belief_valid'].bool()
  error=(out['belief_prediction'].float()-out['belief_target'].detach().float()).square()
  lv=out['belief_logvar'].float().clamp(-4,4)
  while valid.ndim<error.ndim:valid=valid.unsqueeze(-1)
  valid=valid.expand_as(error)
  belief=torch.where(valid,.5*(error*torch.exp(-lv)+lv),0.).sum()/valid.sum().clamp_min(1)
  # Keep a fixed target scale. Reporting MSE prevents an NLL reduction from
  # being misinterpreted as better spatial prediction merely by inflating variance.
  belief_mse=torch.where(valid,error,0.).sum()/valid.sum().clamp_min(1)
  total=action+self.weights['future']*future+self.weights['belief']*belief
  metrics=dict(action=action.detach(),future=future.detach(),belief=belief.detach(),belief_mse=belief_mse.detach())
  if self.weights.get('continuity',0)>0:
   sigma=batch['action_t'].float().reshape(-1,1,1)/1000.
   endpoint=batch['noisy_actions'].float()-sigma*out['action_pred'].float()
   previous=batch['previous_plan'].detach().float()
   agreement=(previous-targets['clean_actions']).square().mean(-1)<.04
   valid=batch['previous_plan_valid'].bool()&agreement&(~prefix_mask(batch['rtc_prefix']))
   indices=torch.arange(endpoint.shape[1],device=endpoint.device)[None,:]
   decay=torch.exp(-(indices-2).clamp_min(0).float())
   weight=valid.float()*decay
   continuity=((endpoint-previous).square().mean(-1)*weight).sum()/weight.sum().clamp_min(1)
   total=total+self.weights['continuity']*continuity;metrics['continuity']=continuity.detach()
  if self.weights.get('native_video',0)>0:
   from torch.utils.checkpoint import checkpoint
   native=checkpoint(self.student.forward_native,targets['native_batch'],use_reentrant=False) if self.training and torch.is_grad_enabled() else self.student.forward_native(targets['native_batch'])
   anchor=F.mse_loss(native['video_pred'].float(),targets['future_velocity'].float())
   total=total+self.weights['native_video']*anchor
   metrics['native_video']=anchor.detach()
  if teacher_action is not None:
   distill=masked_suffix_mse(out['action_pred'],teacher_action.detach(),batch['rtc_prefix'])
   total=total+self.weights['teacher_action']*distill;metrics['teacher_action']=distill.detach()
  return total,metrics


class CausalSpatialTrainer:
 def __init__(self,recipe,accelerator):
  from dynamicwam.training.data.causal_spatial_dataset import collate_causal_spatial
  self.recipe=recipe;self.acc=accelerator;self.device=accelerator.device
  self.run=Path(recipe['run_dir']);self.run.mkdir(parents=True,exist_ok=True)
  self.train=build_dataset(recipe);self.val=build_dataset(recipe,'validation')
  self.data_identity=dict(self.train.packed.dataset_identity)
  self.data_identity.update(physical_grid_identity=self.train.physical_grid_identity,
   episode_partition_identity=self.train.episode_partition_identity,
   spatial_contract=self.train.contract,spatial_pairing_identity=self.train.pairing_identity,
   observation_contract='causal_capture_window_v1',command_history='recorded_capture_commands_not_tick_trace',
   on_policy_ready=False,soft_plan_source='frozen_native_teacher_on_previous_fast_query_observation',
   normalization='legacy_fixed_pretrained_action_and_motion_scales_fp32_before_cast')
  if recipe.get('continue_from'):
   model,p=load_full(recipe['continue_from'],self.device)
   if p['data_identity']!=self.data_identity:raise ValueError('data identity changed between phases')
   self.native_config=p['native_config'];self.initialization=p['initialization']
  else:
   model,self.native_config=build_from_initialization(recipe['initial_checkpoint'],recipe['model'],self.device)
   self.initialization=dict(path=recipe['initial_checkpoint'],sha256=sha256(recipe['initial_checkpoint']),kind='stage1_video_wo_motion_action_fresh_motion')
  self.validate_statistics(model)
  model.requires_grad_(True)
  if recipe['phase']=='interface_adapt':
   model.backbone.action_expert.requires_grad_(False)
   model.backbone.absolute_motion_tokens.requires_grad_(False)
  video=model.backbone.compact_wan
  groups=[];seen=set()
  for name,module,lr in [('video',video,recipe['lr']['video']),('action',model.backbone.action_expert,recipe['lr']['action']),('motion',model.backbone.absolute_motion_tokens,recipe['lr']['action'])]:
   params=[p for p in module.parameters() if p.requires_grad]
   seen.update(id(p) for p in module.parameters())
   if params:groups.append(dict(params=params,lr=lr,name=name))
  groups.append(dict(params=[p for p in model.parameters() if id(p) not in seen and p.requires_grad],lr=recipe['lr']['new'],name='new_spatial_future'))
  self.objective=Objective(model,recipe['loss'])
  self.optimizer=torch.optim.AdamW(groups,weight_decay=.001)
  self.teacher=None
  if recipe.get('teacher_checkpoint'):
   self.initialization['teacher_sha256']=sha256(recipe['teacher_checkpoint'])
   p=native_payload(recipe['teacher_checkpoint']);self.teacher=build_native(p['config'],self.device)
   self.teacher.load_state_dict(p['model'],strict=True);self.teacher.eval().requires_grad_(False)
   # Teacher actions share the exact action coordinate system; motion tokens
   # retain their own published normalization inside the teacher module.
   if any(p['config']['action_normalization']['stats'][k]!=self.native_config['action_normalization']['stats'][k] for k in ('mean','std')):
    raise ValueError('teacher action normalization differs; explicit conversion required')
   del p;gc.collect()
  self.samplers={}
  def loader(dataset,subset):
   base=BalancedPairSampler(dataset.records,samples_per_episode=10,seed=20261021 if subset=='train' else 20261022)
   sampler=ProcessGroupAlignedSampler(base,batch_size=recipe['microbatch'],num_processes=accelerator.num_processes)
   self.samplers[subset]=sampler
   return DataLoader(dataset,batch_size=recipe['microbatch'],sampler=sampler,collate_fn=collate_causal_spatial,
    num_workers=recipe.get('workers',0),pin_memory=True,drop_last=True)
  self.loader=loader(self.train,'train');self.val_loader=loader(self.val,'validation')
  self.objective,self.optimizer,self.loader=accelerator.prepare(self.objective,self.optimizer,self.loader)
  # DeepSpeed casts module buffers as well as parameters to BF16. Keep the
  # fixed normalization statistics exact, so normalization precedes precision
  # reduction in training and in the strict exported inference module.
  module=accelerator.unwrap_model(self.objective).student.backbone.absolute_motion_tokens
  cfg=self.native_config['model']['absolute_motion']
  module.feature_mean=torch.tensor(cfg['feature_mean'],device=self.device,dtype=torch.float32).reshape_as(module.feature_mean)
  module.feature_scale=torch.tensor(cfg['feature_scale'],device=self.device,dtype=torch.float32).reshape_as(module.feature_scale)
  self.val_loader=accelerator.prepare_data_loader(self.val_loader)
  self.scheduler=create_scheduler(self.optimizer,total_steps=recipe['steps'],warmup_steps=recipe['warmup_steps'],min_lr_ratio=.01)
  accelerator.register_for_checkpointing(self.scheduler)
  scheduler=build_training_flow_scheduler({'flow_matching':dict(shift=5.,sigma_min=0.,extra_one_step=True,training_timesteps=1000)})
  self.timesteps=scheduler.timesteps.to(self.device);self.sigmas=scheduler.sigmas.to(self.device)
  self.native_schedule=BridgeSchedule.from_flow_match()
  self.step=0;self.epoch=0;self.batch_offset=0;self.last_metrics={}
  self.finite_loss_checks=0;self.finite_grad_checks=0;self.last_grad_norm=None
  self.gradient_evidence={};self.gradient_hooks=[]
  for name,param in model.named_parameters():
   if not param.requires_grad:continue
   group=('video' if name.startswith('backbone.compact_wan.') else 'action' if name.startswith('backbone.action_expert.') else
    'motion' if name.startswith('backbone.absolute_motion_tokens.') else 'observer' if name.startswith('belief.') else
    'plan' if 'plan' in name else 'future_and_interfaces')
   if group not in self.gradient_evidence:self.gradient_evidence[group]=torch.zeros((),device=self.device)
   def observe_grad(g,group=group):
    if self.step<4:self.gradient_evidence[group].add_(g.detach().float().reshape(-1)[:128].abs().sum())
   self.gradient_hooks.append(param.register_hook(observe_grad))

 def validate_statistics(self,model):
  packed=self.train.packed
  if packed.dataset_identity['action_stats_sha256']!=self.native_config['action_normalization']['stats_sha256']:
   raise ValueError('action normalization identity mismatch')
  motion=packed.motion_statistics;config=self.native_config['model']['absolute_motion']
  for key,native in [('mean','feature_mean'),('scale','feature_scale')]:
   if list(motion[key])!=list(config[native]):raise ValueError('native motion statistics mismatch')

 def prepare(self,sample,*,validation=False):
  s=tensor_to(sample,self.device);a=s['clean_actions'].float();v=s['clean_future'].float()
  indices=torch.randint(len(self.sigmas),(a.shape[0],),device=self.device)
  sigma=self.sigmas[indices].float();t=self.timesteps[indices].float()
  an=torch.randn_like(a);vn=torch.randn_like(v)
  noisy_a=(1-sigma[:,None,None])*a+sigma[:,None,None]*an
  noisy_v=(1-sigma.reshape(-1,1,1,1,1))*v+sigma.reshape(-1,1,1,1,1)*vn
  prefix=PrefixCondition(s['rtc_prefix'].float(),s['rtc_prefix_lengths'].long())
  # Explicit model-input allowlist: future observations and their measured
  # capture times are targets, never inference-time conditioning.
  fields=('source_condition_latent','source_text_embeddings','source_time','source_id',
   'initial_state','absolute_motion_features','absolute_motion_interval_valid_mask',
   'absolute_motion_acceleration_valid_mask','action_target_offsets','future_target_offsets',
   'spatial_rgb','spatial_flow','spatial_valid','spatial_flow_valid','spatial_dt','executed_actions',
   'executed_valid','previous_plan','previous_plan_valid','observation_age')
  batch={k:s[k] for k in fields}
  batch.update(noisy_actions=apply_prefix(noisy_a,prefix),future_latent=noisy_v,action_t=t,video_t=t,rtc_prefix=prefix)
  native=dict(condition_latent=s['clean_current_condition'],future_latent=noisy_v,noisy_actions=batch['noisy_actions'],
   initial_state=batch['initial_state'],text_embeddings=batch['source_text_embeddings'],action_t=t,video_t=t,rtc_prefix=prefix,
   absolute_motion_features=batch['absolute_motion_features'],absolute_motion_interval_valid_mask=batch['absolute_motion_interval_valid_mask'],
   absolute_motion_acceleration_valid_mask=batch['absolute_motion_acceleration_valid_mask'],action_target_offsets=batch['action_target_offsets'])
  teacher_action=None
  if self.teacher is not None:
   with torch.no_grad(),self.acc.autocast():
    if self.recipe['loss'].get('teacher_action',0)>0:teacher_action=self.teacher(native)['action_pred'].detach()
    # A generated proposal, never a future ground-truth action target. This is
    # offline causal context augmentation, not a claim of on-policy replay.
    if (validation or self.step%self.recipe.get('teacher_plan_interval',2)==0):
     source=SourceObservation(condition_latent=s['plan_condition_latent'],state=s['plan_state'],motion=s['plan_motion'],
      motion_interval_valid_mask=s['plan_motion_interval_valid_mask'],motion_acceleration_valid_mask=s['plan_motion_acceleration_valid_mask'],
      text_embeddings=s['plan_text_embeddings'],source_id=s['plan_id'],sim_time=s['plan_time'],
      action_target_offsets=s['plan_action_target_offsets'])
     noise=SourceNoise(torch.randn_like(v),torch.randn_like(a))
     plan=sample_native_wam(self.teacher,source,noise,self.native_schedule)
     st=s['plan_action_target_times'].double()
     ft=s['current_time'].double()[:,None]+s['action_target_offsets'].double()
     previous,valid=align_proposal(plan,st,ft)
     valid=valid&s['plan_context_valid'].bool()[:,None]
     overlap=torch.arange(a.shape[1],device=self.device)[None,:]
     valid=valid&(overlap>=2)&(overlap<4)
     if not validation:valid=valid&(torch.rand((len(a),1),device=self.device)>.2)
     batch['previous_plan']=previous.detach();batch['previous_plan_valid']=valid
  return batch,dict(action_velocity=an-a,future_velocity=vn-v,native_batch=native,clean_actions=a),teacher_action

 def checkpoint(self,*,final=False):
  self.acc.wait_for_everyone()
  if self.acc.is_main_process:
   model=self.acc.unwrap_model(self.objective).student
   payload=make_payload(model,native_config=self.native_config,recipe=self.recipe,data_identity=self.data_identity,
    step=self.step,initialization=self.initialization,diagnostic=self.recipe['diagnostic'])
   path=self.run/'exports'/f'causal_spatial_step_{self.step}.pt';write_payload(path,payload)
   fast=self.run/'exports'/f'causal_spatial_fast_step_{self.step}.pt';write_payload(fast,make_fast_payload(payload))
   metadata=dict(step=self.step,path=str(path),sha256=sha256(path),bytes=path.stat().st_size,fast_path=str(fast),fast_sha256=sha256(fast),final=final)
   (self.run/'latest_export.json').write_text(json.dumps(metadata,indent=2))
   del payload;gc.collect()
  # Full optimizer/scheduler/RNG state for an exact phase resume. Phase-to-phase
  # initialization intentionally uses weights only with a new optimizer.
  self.acc.save_state(str(self.run/'state'/f'step_{self.step}'))
  self.acc.wait_for_everyone()
  if self.acc.is_main_process:
   state_dir=self.run/'state'/f'step_{self.step}'
   progress=dict(step=self.step,epoch=self.epoch,batch_offset=self.batch_offset,
    recipe_sha256=hashlib.sha256(json.dumps(self.recipe,sort_keys=True).encode()).hexdigest(),
    data_identity_sha256=hashlib.sha256(json.dumps(self.data_identity,sort_keys=True).encode()).hexdigest())
   (state_dir/'trainer_state.json').write_text(json.dumps(progress,indent=2))
   files=[dict(path=str(p.relative_to(state_dir)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(state_dir.rglob('*')) if p.is_file() and p.name!='state_manifest.json']
   (state_dir/'state_manifest.json').write_text(json.dumps(dict(step=self.step,files=files),indent=2))
   # Retain all small evaluation exports, but only the two latest complete
   # optimizer-state directories of THIS new run (ZeRO0 duplicates rank state).
   state_root=(self.run/'state').resolve()
   old=sorted([p for p in state_root.iterdir() if p.is_dir() and re.fullmatch(r'step_[0-9]+',p.name)
    and (p/'state_manifest.json').is_file()],key=lambda p:int(p.name.split('_')[1]))
   for directory in old[:-2]:
    resolved=directory.resolve()
    if resolved.parent!=state_root or resolved==state_dir.resolve():raise RuntimeError('unsafe state retention target')
    shutil.rmtree(resolved)
  self.acc.wait_for_everyone()

 def validate(self):
  self.objective.eval();metrics=[]
  # Fixed RNG for paired loss comparisons, restoring train RNG afterwards.
  with torch.random.fork_rng(devices=[self.device]),torch.no_grad():
   torch.manual_seed(20261023+self.acc.process_index)
   for i,s in enumerate(self.val_loader):
    if i>=self.recipe['validation_batches']:break
    batch,target,teacher=self.prepare(s,validation=True)
    with self.acc.autocast():loss,m=self.objective(batch,target,teacher)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite validation loss')
    metrics.append(torch.stack([loss.detach()]+[m[k] for k in sorted(m)]))
   if metrics:
    values=self.acc.reduce(torch.stack(metrics).mean(0),reduction='mean').cpu().tolist()
    if self.acc.is_main_process:
     with (self.run/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.step,metrics=dict(zip(['total']+sorted(m),values))))+'\n')
  self.objective.train()

 def run_training(self):
  started=time.time();self.objective.train();self.optimizer.zero_grad();completed=False;error_type=None;nonfinite_loss=0;nonfinite_grad=0
  if self.acc.is_main_process:
   (self.run/'recipe.json').write_text(json.dumps(self.recipe,indent=2));(self.run/'data_identity.json').write_text(json.dumps(self.data_identity,indent=2))
  try:
   while self.step<self.recipe['steps']:
    self.train.set_epoch(self.epoch);self.samplers['train'].set_epoch(self.epoch)
    for sample in self.loader:
     with self.acc.accumulate(self.objective):
      # DeepSpeed may perform the optimizer update inside accelerator.backward.
      # Set this update's LR before any microbatch, including the first one.
      for group,base_lr in zip(self.optimizer.param_groups,self.scheduler.base_lrs,strict=True):
       group['lr']=self.scheduler._lr_at(min(self.step+1,self.recipe['steps']),base_lr)
      batch,targets,teacher=self.prepare(sample)
      with self.acc.autocast():loss,metrics=self.objective(batch,targets,teacher)
      if not torch.isfinite(loss):
       nonfinite_loss+=1;raise RuntimeError('nonfinite causal-spatial loss')
      self.finite_loss_checks+=1
      self.acc.backward(loss)
      if self.acc.sync_gradients:
       norm=self.acc.clip_grad_norm_(self.objective.parameters(),1.)
       if norm is None and hasattr(self.objective,'get_global_grad_norm'):norm=self.objective.get_global_grad_norm()
       if norm is None:raise RuntimeError('optimizer boundary lacks measured global gradient norm')
       if norm is not None:
        if not torch.isfinite(torch.as_tensor(norm)):
         nonfinite_grad+=1;raise RuntimeError('nonfinite gradient')
        self.finite_grad_checks+=1;self.last_grad_norm=float(norm)
      self.optimizer.step();self.optimizer.zero_grad()
      if self.acc.sync_gradients:
       self.scheduler.step();self.step+=1
       values=self.acc.reduce(torch.stack([loss.detach()]+[metrics[k] for k in sorted(metrics)]),reduction='mean').float().cpu().tolist()
       self.last_metrics=dict(zip(['total']+sorted(metrics),values))
       if self.acc.is_main_process and (self.step%self.recipe['log_interval']==0 or self.step==1):
        row=dict(step=self.step,phase=self.recipe['phase'],elapsed_seconds=time.time()-started,**self.last_metrics)
        print(json.dumps(row),flush=True)
        with (self.run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
       if self.step%self.recipe['validation_interval']==0 and self.step<self.recipe['steps']:self.validate()
       if self.step%self.recipe['checkpoint_interval']==0 and self.step<self.recipe['steps']:self.checkpoint()
     self.batch_offset+=1
     if self.step>=self.recipe['steps']:break
    self.epoch+=1;self.batch_offset=0
   self.validate();self.checkpoint(final=True)
   completed=True
  except BaseException as exc:
   error_type=type(exc).__name__;raise
  finally:
   receipt=dict(rank=self.acc.process_index,step=self.step,peak_allocated=torch.cuda.max_memory_allocated(self.device),
    peak_reserved=torch.cuda.max_memory_reserved(self.device),last_metrics=self.last_metrics,world_size=self.acc.num_processes,
    completed=completed,error_type=error_type,finite_loss_checks=self.finite_loss_checks,finite_grad_checks=self.finite_grad_checks,
    nonfinite_loss_count=nonfinite_loss,nonfinite_grad_count=nonfinite_grad,oom_count=torch.cuda.memory_stats(self.device).get('num_ooms',0),
    optimizer_updates=self.step,last_grad_norm=self.last_grad_norm,
    gradient_evidence={k:float(v) for k,v in self.gradient_evidence.items()})
   (self.run/f'rank_{self.acc.process_index}_resources.json').write_text(json.dumps(receipt,indent=2))

 def release_training_resources(self):
  for handle in self.gradient_hooks:handle.remove()
  self.gradient_hooks=[];self.gradient_evidence.clear()
  self.scheduler=None;self.timesteps=None;self.sigmas=None
  self.objective=None;self.optimizer=None;self.teacher=None;self.loader=None;self.val_loader=None
  gc.collect()
