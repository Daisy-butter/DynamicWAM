"""CPU-only strict audit of the original Stage2 initialization and fresh v3.3 interface."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from check_motionkv_smoke_export import build_full_cpu_schema
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.absolute_motion import validate_checkpoint_motion_metadata
from dynamicwam.inference.multirate_model_loader import install_full_checkpoint
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.training.train.multirate_trainer import (MultirateTrainer,fast_module_spec,
    configure_multirate_trainable,multirate_optimizer_groups)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def audit():
 torch.set_num_threads(2)
 profiles={k:load_multirate_profile(ROOT/'configs/multirate'/v) for k,v in {
 'stage2':'joint_motionkv_v33_stage2.yaml','stage3':'joint_motionkv_v33_stage3.yaml',
 'smoke2':'joint_motionkv_v33_stage2_subset_warmup_smoke.yaml','smoke3':'joint_motionkv_v33_stage2_subset_joint_smoke.yaml'}.items()}
 cfg=profiles['stage2'].training;path=Path(cfg['model']['initial_checkpoint'])
 payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 assert payload['format']=='dynamicwam_absolute_motion_checkpoint' and payload['version']==2
 assert payload['global_step']==0 and payload['config']['training_stage']=='absolute_motion_initialization'
 assert cfg['initialize_joint_interface'] is False and cfg['allow_pretrained_dataset_transfer'] is False
 lineage=payload['initialization'];inputs={'motion_init':{'path':str(path),'sha256':sha(path)}}
 for label,key in [('wo_motion','source_checkpoint'),('stage1','source_stage1_checkpoint')]:
  p=Path(lineage[key]);digest=sha(p);assert digest==lineage[key+'_sha256'];inputs[label]={'path':str(p),'sha256':digest}
 packed=Path(cfg['dataset']['root']);metadata=json.loads((packed/'dataset.json').read_text())
 motion=dict(metadata['absolute_motion']);motion.pop('statistics_file');motion=validate_checkpoint_motion_metadata(motion)
 assert motion==payload['config']['model']['absolute_motion']
 expected_identity=dict(format=metadata['format'],version=metadata['version'],dataset_fingerprint=metadata['dataset_fingerprint'],action_stats_sha256=sha(packed/'action_stats.json'),motion_statistics_sha256=motion['statistics_sha256'])
 assert payload['config']['dataset_identity']==expected_identity
 assert payload['config']['action_normalization']['stats']==json.loads((packed/'action_stats.json').read_text())['robotwin_qpos']
 assert inputs['wo_motion']['path'].endswith('/DynamicWAM_wo_motion.pt')
 stub=SimpleNamespace(config=cfg,dataset_identity=expected_identity,motion_checkpoint_metadata=motion)
 MultirateTrainer._check_checkpoint_identity(stub,payload,str(path))
 fake=copy.deepcopy({k:v for k,v in payload.items() if k!='model'});fake['config']['runtime_contract']={'training_scope':'smoke_subset'}
 try:MultirateTrainer._check_checkpoint_identity(stub,fake,'forbidden_smoke_fixture')
 except RuntimeError as e:assert 'cannot initialize or resume formal training' in str(e)
 else:raise AssertionError('smoke checkpoint admitted to formal training')
 spec=fast_module_spec(cfg,history_count=motion['history_count'],motion_mean=motion['feature_mean'],motion_scale=motion['feature_scale'])
 spec.update(schedule=BridgeSchedule.from_training_config(cfg).to_config(),fast_steps=10)
 torch.manual_seed(cfg['initialization_seed']);model=build_full_cpu_schema({'modules':spec},profiles['stage2'])
 before=model.state_dict();incoming={'backbone.'+k:v for k,v in payload['model'].items()}
 fresh=set(before)-set(incoming)
 assert all(k.startswith(('fast_conditioner.','backbone.action_expert.source_time_encoder.')) for k in fresh)
 assert len([k for k in fresh if 'source_time_encoder.' in k])==4
 fingerprints={k:tensor_sha(before[k]) for k in fresh}
 # Materialize only the meta Video schema from its exact native initialization.
 model.load_state_dict({**before,**incoming},strict=True,assign=True)
 assert install_full_checkpoint(model,payload,allow_v2_init=True)=='v2'
 installed=model.state_dict()
 assert all(torch.equal(installed[k],v) for k,v in incoming.items())
 assert all(tensor_sha(installed[k])==v for k,v in fingerprints.items())
 assert all(bool(torch.isfinite(v).all()) for v in installed.values() if v.is_floating_point())
 assert all(torch.count_nonzero(installed[k])==0 for k in fresh if 'source_time_encoder.2.' in k)
 model.native_source_bridge='student_joint';model.native_teacher_weight=0
 trainability={}
 for stage,phase in [('stage2','warmup'),('stage3','joint')]:
  configure_multirate_trainable(model,phase);groups=multirate_optimizer_groups(model,profiles[stage].training,phase)
  video=[p for p in model.backbone.compact_wan.parameters()];action=list(model.backbone.action_expert.parameters());motionp=list(model.backbone.absolute_motion_tokens.parameters());interface=list(model.fast_conditioner.parameters())
  assert all(p.requires_grad for p in action+motionp+interface)
  assert all(p.requires_grad==(stage=='stage3') for p in video)
  trainability[stage]={'video_trainable':stage=='stage3','action_native_motion_interface_source_time_trainable':True,'optimizer_groups':[{'name':g['name'],'lr':g['lr'],'elements':sum(p.numel() for p in g['params'])} for g in groups]}
 for k,p in profiles.items():
  c=p.training;assert c['initialize_joint_interface'] is False
  if k.startswith('smoke'):
    assert c['gradient_accumulation_steps']==2 and c['per_device_batch_size']==1
  else:
    assert c['gradient_accumulation_steps']==1 and c['per_device_batch_size']==16
  assert c['loss']['action_weight']==.1 and c['motion_prediction_weight']==.1
  assert bool(c['dataset']['smoke_subset'])==k.startswith('smoke')
 assert profiles['stage2'].training['max_steps']==80000 and profiles['stage3'].training['max_steps']==40000
 assert profiles['stage3'].training['model']['initial_checkpoint'].endswith('/joint_motionkv_v33_stage2_b128_20261016/exports/multirate_step_80000.pt')
 assert profiles['smoke3'].training['model']['initial_checkpoint'].endswith('/joint_motionkv_v33_stage2_subset_smoke_b128_20261016/exports/multirate_step_4.pt')
 return dict(passed=True,cuda_used=False,forward_run=False,initialization=inputs,global_step=0,stage2_lineage='Stage1 Video + wo_motion Action + originally fresh native absolute motion; newly fresh v3.3 interface and source-time module',native_tensors_preserved=len(incoming),new_fresh_tensors=len(fresh),new_tensor_names=sorted(fresh),strict_loader=True,stats_and_dataset_identity_match=True,smoke_formal_init_rejected=True,trainability=trainability,effective_global_batch=128,original_global_batch=128,same_exposure_as_original=True,full_data_readiness_checked=False)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('run with CUDA_VISIBLE_DEVICES= for CPU-only audit')
 if a.output.exists():raise FileExistsError('preserve prior audit receipt')
 result=audit();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='new_tensor_names'},indent=2))
