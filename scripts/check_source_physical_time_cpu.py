"""Source-only physical target phase: tiny actual native joint/Action CPU tests."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from check_joint_motionkv_v3_training_cpu import v3_fixture,cpu_attention
from check_joint_source_prefix_cpu import TinyCompact
from dynamicwam.models.small_wam import SmallWAMActionModel,SmallWAMActionConfig
from dynamicwam.models.causal_world_bridge import CausalWorldBridge
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.training.train.multirate_trainer import configure_multirate_trainable,multirate_optimizer_groups
from dynamicwam.inference.multirate_model_loader import export_fast_only_payload,load_fast_only_modules

def run():
 torch.set_num_threads(2);torch.manual_seed(20261016);checks=[]
 with ExitStack() as stack:
  for name in ('dynamicwam.models.small_wam.flash_attention','dynamicwam.vendor.wan.modules.model.flash_attention','dynamicwam.models.action_core_ops.flash_attention','dynamicwam.models.action_core_ops.dense_action_attention'):
   stack.enter_context(patch(name,cpu_attention))
  model,batch=v3_fixture();old=model.backbone
  cfg=SmallWAMActionConfig(SimpleNamespace(dim=12,num_heads=2,num_layers=2,head_dim=6),14,14,16,24,48,2,False,4,(0.,)*12,(1.,)*12)
  core=SmallWAMActionModel(cfg,TinyCompact());core.action_expert.load_state_dict(old.action_expert.state_dict());core.absolute_motion_tokens.load_state_dict(old.absolute_motion_tokens.state_dict());model.backbone=core
  model.fast_conditioner=JointMotionKVConditioner(**dict(model.fast_conditioner.constructor_spec,proposal_feedback=True,complete_source_plan=True));model.bridge=CausalWorldBridge(model.schedule,complete_source_plan=True)
  batch['source_aligned'].update(source_query=True,source_action_target_offsets=batch['source'].action_target_offsets)
  source=batch['source'];noise=batch['source_noise'];before=deepcopy(model)
  core.action_expert.configure_source_physical_time_conditioning(True);encoder=core.action_expert.source_time_encoder
  configure_multirate_trainable(model,'joint');configure_multirate_trainable(before,'joint')
  original=before.bridge(before.backbone,source,noise,detach_cache=False);zero=model.bridge(core,source,noise,detach_cache=False)
  assert torch.equal(zero.source_completed_plan,original.source_completed_plan)
  assert all(torch.equal(a,b) for a,b in zip(zero.video_k+zero.video_v,original.video_k+original.video_v))
  assert torch.equal(model(batch)['native_loss'],before(batch)['native_loss'])
  checks.append('legacy_disabled_and_enabled_zero_projection_preserve_nativeKV_fullK10_and_joint_loss_bitexact')
  model.zero_grad(set_to_none=True);model(batch)['native_loss'].backward();assert encoder[-1].weight.grad.abs().sum()>0
  with torch.no_grad():encoder[-1].weight.normal_(std=.03)
  inputs=source.as_model_batch();oldtokens,n=before.backbone._build_action_tokens(inputs,noise.actions,source.state);tokens,_=core._build_action_tokens(inputs,noise.actions,source.state)
  assert torch.equal(tokens[:,:1],oldtokens[:,:1]) and torch.equal(tokens[:,n:],oldtokens[:,n:]) and not torch.equal(tokens[:,1:n],oldtokens[:,1:n])
  fast_inputs=dict(inputs);fast_inputs.pop('source_query');fast_inputs.pop('source_action_target_offsets');fast_tokens,_=core._build_action_tokens(fast_inputs,noise.actions,source.state);assert torch.equal(fast_tokens,oldtokens)
  checks.append('only_source_action_inputtokens_change_state_motion_registers_and_fast_inputtokens_bitexact')
  shifted=replace(source,action_target_offsets=source.action_target_offsets-.07)
  with patch.object(core,'_build_action_tokens',wraps=core._build_action_tokens) as calls:
   a=model.bridge(core,source,noise,detach_cache=False)
  assert len(calls.call_args_list)==10
  assert all(torch.equal(c.args[0]['source_action_target_offsets'],source.action_target_offsets) for c in calls.call_args_list)
  b=model.bridge(core,shifted,noise,detach_cache=False)
  assert not torch.equal(a.source_completed_plan,b.source_completed_plan)
  assert any(not torch.equal(x,y) for x,y in zip(a.video_k+a.video_v,b.video_k+b.video_v))
  checks.append('same_noise_phase_shift_changes_nativeVideoKV_and_finalplan_all2joint8tail_receive_exact_offsets')
  for phase in ('warmup','joint'):
   configure_multirate_trainable(model,phase);model.joint_video_weight=0. if phase=='warmup' else .1;model.joint_action_weight=.1;model.zero_grad(set_to_none=True)
   model(batch)['native_loss'].backward()
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
   assert all(p.grad.abs().sum()>0 for p in encoder.parameters())
   if phase=='warmup':assert all(p.grad is None for p in core.compact_wan.parameters())
   groups=multirate_optimizer_groups(model,dict(action_learning_rate=1e-5,fast_learning_rate=1e-4,video_learning_rate=2e-6,weight_decay=.001),phase)
   ids=[id(p) for g in groups for p in g['params']];assert len(ids)==len(set(ids));assert all(ids.count(id(p))==1 for p in encoder.parameters())
  checks.append('warmup_sourceFM_and_joint_objective_train_timeMLP_with_finite_gradients_exactly_once_optimizer_ownership')
  configure_multirate_trainable(model,'joint');model.zero_grad(set_to_none=True);model.gradient_checkpointing=False;plain=model(batch)['native_loss'];plain.backward();grads={k:v.grad.clone() for k,v in encoder.named_parameters()};model.zero_grad(set_to_none=True);model.gradient_checkpointing=True;recomputed=model(batch)['native_loss'];recomputed.backward();assert torch.equal(plain,recomputed);assert all(torch.allclose(p.grad,grads[k],atol=1e-6,rtol=1e-5) for k,p in encoder.named_parameters())
  checks.append('activation_checkpoint_output_and_source_time_grad_parity')
  spec={'action_expert':dict(dim=24,ffn_dim=48,num_layers=2,state_dim=14,action_dim=14,chunk_size=16,source_physical_time_conditioning=True),'wan_attention':dict(dim=12,num_heads=2,head_dim=6),'motion':dict(history_count=4,mean=[0.]*12,scale=[1.]*12),'fast_conditioner':dict(model.fast_conditioner.constructor_spec)}
  exported=export_fast_only_payload(model.state_dict(),module_spec=spec,global_step=0);loaded=load_fast_only_modules(exported)
  assert all(torch.equal(v,loaded.action_expert.source_time_encoder.state_dict()[k]) for k,v in encoder.state_dict().items())
  broken=deepcopy(exported);broken['modules']['action_expert'].pop('source_physical_time_conditioning')
  try:load_fast_only_modules(broken)
  except RuntimeError:pass
  else:raise AssertionError('lost source-time architecture metadata accepted')
  try:core._build_action_tokens(dict(inputs,source_action_target_offsets=None),noise.actions,source.state)
  except ValueError:pass
  else:raise AssertionError('missing source time offsets accepted')
  checks.append('full_actor_export_strict_reload_and_missingflag_or_missingphysical_offsets_rejected')
 print(json.dumps({'passed':True,'checks':checks,'added_parameters':sum(p.numel() for p in encoder.parameters()),'production_dim768_parameters':2*64+64+64*768+768,'limitation':'Actual tiny native Wan blocks with CPU attention and compact projection fixture; not full Video/GPU or quality proof'},indent=2))
if __name__=='__main__':run()
