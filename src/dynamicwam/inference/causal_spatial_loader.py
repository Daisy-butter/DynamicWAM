"""Strict, separately versioned V4 checkpoints and audited native initialization."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy, hashlib
import torch
from dynamicwam.models.small_wam import SmallWAMActionConfig, SmallWAMActionModel
from dynamicwam.training.models.compact_wan import CompactWANConfig, CompactWANModel

FORMAT='dynamicwam_causal_spatial_checkpoint'
VERSION=1
INIT_SHA='601936c61503d173c38d7c9ae2ce956dfa22825e4b5c4e1096424f9586df540d'

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def build_native(config,device='cpu'):
 """Reconstruct the exact Stage1/wo-motion geometry, without loading Wan5B."""
 m=config['model'];w=dict(m['compact_wan']);a=m['action_expert'];motion=m['absolute_motion']
 w['future_video_size']=tuple(w['future_video_size'])
 compact=CompactWANModel.from_config(CompactWANConfig(**w),device=device)
 return SmallWAMActionModel(SmallWAMActionConfig(compact_wan=compact.config,
  action_dim=a['action_dim'],state_dim=a['state_dim'],chunk_size=a['chunk_size'],
  ae_dim=a['dim'],ae_ffn_dim=a['ffn_dim'],ae_num_layers=a['num_layers'],wan_frozen=False,
  motion_history_count=motion['history_count'],motion_feature_mean=tuple(motion['feature_mean']),
  motion_feature_scale=tuple(motion['feature_scale']),
  source_physical_time_conditioning=bool(a.get('source_physical_time_conditioning',False))),compact).to(device)

def native_payload(path,*,formal_init=False):
 if formal_init and sha256(path)!=INIT_SHA:raise ValueError('formal initialization must be audited Stage1 + wo_motion + fresh motion')
 p=torch.load(path,map_location='cpu',weights_only=True)
 if p.get('format')!='dynamicwam_absolute_motion_checkpoint' or p.get('version')!=2:
  raise ValueError('expected native absolute motion checkpoint v2')
 return p

def build_from_initialization(path,model_config,device='cpu',formal_init=True):
 from dynamicwam.models.causal_spatial import CausalSpatialModel, CausalSpatialConfig
 p=native_payload(path,formal_init=formal_init)
 backbone=build_native(p['config'],device)
 backbone.load_state_dict(p['model'],strict=True)
 model=CausalSpatialModel(backbone,CausalSpatialConfig(**model_config)).to(device)
 return model,p['config']

def make_payload(model,*,native_config,recipe,data_identity,step,initialization,diagnostic):
 return dict(format=FORMAT,version=VERSION,global_step=int(step),
  model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
  model_config=asdict(model.config),native_config=copy.deepcopy(native_config),
  recipe=copy.deepcopy(recipe),data_identity=copy.deepcopy(data_identity),
  initialization=copy.deepcopy(initialization),diagnostic_only=bool(diagnostic),
  action_normalization=copy.deepcopy(native_config['action_normalization']),
  runtime_contract=copy.deepcopy(recipe['runtime_contract']))

def load_full(path,device='cpu',*,allow_diagnostic=False):
 from dynamicwam.models.causal_spatial import CausalSpatialModel, CausalSpatialConfig
 p=torch.load(path,map_location='cpu',weights_only=True)
 if p.get('format')!=FORMAT or p.get('version')!=VERSION:raise ValueError('not a causal-spatial V4 checkpoint')
 if p.get('diagnostic_only') and not allow_diagnostic:raise ValueError('diagnostic weights cannot initialize formal training')
 if p['runtime_contract']!=p['recipe']['runtime_contract']:raise ValueError('runtime contract disagreement')
 model=CausalSpatialModel(build_native(p['native_config'],device),CausalSpatialConfig(**p['model_config'])).to(device)
 model.load_state_dict(p['model'],strict=True,assign=True)
 model.to(device)
 return model,p

def write_payload(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)

def make_fast_payload(payload):
 p={k:v for k,v in payload.items() if k!='model'}
 p['format']=FORMAT+'_fast';p['model']={k:v for k,v in payload['model'].items() if not k.startswith('backbone.compact_wan.')}
 p['source_full_state_format']=FORMAT
 return p

def load_fast(path,device='cpu',*,allow_diagnostic=False):
 from dynamicwam.models.causal_spatial import CausalSpatialModel,CausalSpatialActionCore,CausalSpatialConfig
 from dynamicwam.models.action_expert import ActionExpert,ActionExpertConfig
 from dynamicwam.models.absolute_motion_tokens import AbsoluteMotionTokenModule
 p=torch.load(path,map_location='cpu',weights_only=True)
 if p.get('format')!=FORMAT+'_fast' or p.get('version')!=VERSION:raise ValueError('not a V4 fast checkpoint')
 if p.get('diagnostic_only') and not allow_diagnostic:raise ValueError('diagnostic checkpoint deployment forbidden')
 if p['runtime_contract']!=p['recipe']['runtime_contract']:raise ValueError('runtime contract disagreement')
 m=p['native_config']['model'];a=m['action_expert'];w=m['compact_wan'];motion=m['absolute_motion']
 config=SmallWAMActionConfig(compact_wan=CompactWANConfig(**w),action_dim=a['action_dim'],state_dim=a['state_dim'],chunk_size=a['chunk_size'],
  ae_dim=a['dim'],ae_ffn_dim=a['ffn_dim'],ae_num_layers=a['num_layers'],wan_frozen=False,
  motion_history_count=motion['history_count'],motion_feature_mean=tuple(motion['feature_mean']),motion_feature_scale=tuple(motion['feature_scale']),
  source_physical_time_conditioning=bool(a.get('source_physical_time_conditioning',False)))
 action=ActionExpert(ActionExpertConfig(dim=a['dim'],ffn_dim=a['ffn_dim'],num_layers=a['num_layers'],state_dim=a['state_dim'],action_dim=a['action_dim'],chunk_size=a['chunk_size'],source_physical_time_conditioning=config.source_physical_time_conditioning),
  dict(dim=w['dim'],num_heads=w['num_heads'],head_dim=w['head_dim']))
 motion_module=AbsoluteMotionTokenModule(dim=a['dim'],history_count=motion['history_count'],feature_mean=motion['feature_mean'],feature_scale=motion['feature_scale'])
 core=CausalSpatialActionCore(action,motion_module,config)
 model=CausalSpatialModel(core,CausalSpatialConfig(**p['model_config']),patch_size=(1,2,2)).to(device)
 model.load_state_dict(p['model'],strict=True,assign=True)
 model.to(device)
 return model,p
