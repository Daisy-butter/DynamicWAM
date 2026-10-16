from copy import deepcopy
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'scripts'))
from test_v3_deployment_contract import contract_fixture
from dynamicwam.inference.fast_runtime import validate_runtime_contract,RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.inference.multirate_model_loader import rtc_prefix_contract
from dynamicwam.training.data.physical_grid import rtc_prefix_sampling_contract
from dynamicwam.config.multirate import load_multirate_profile
import run_joint_motionkv_smokes as launcher
class SourceTimeContractTests(unittest.TestCase):
 def test_source_time_metadata_must_match_architecture(self):
  _,p,r=contract_fixture(3);p['modules']['action_expert']['source_physical_time_conditioning']=True
  with self.assertRaisesRegex(RuntimeError,'source physical'):validate_runtime_contract(p,r,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
  p['config'].setdefault('model',{}).setdefault('action_expert',{})['source_physical_time_conditioning']=True
  p['config']['runtime_contract']['source_physical_time_conditioning']='native_action_tokens_target_offset_phase_v1'
  validate_runtime_contract(p,r,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
  p['modules']['action_expert']['source_physical_time_conditioning']=False
  with self.assertRaisesRegex(RuntimeError,'source physical'):validate_runtime_contract(p,r,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
 def test_fixed_h4_contract_and_negative_cases(self):
  _,p,r=contract_fixture(3);c=p['config']['runtime_contract'];c['rtc_prefix_sampling']=rtc_prefix_sampling_contract('fixed_h4_v1',.2);c['rtc']['includes_zero_prefix']=False
  self.assertFalse(rtc_prefix_contract(p)['includes_zero_prefix']);validate_runtime_contract(p,r,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
  for key,value in [('current_prefix_actions',3),('source_prefix_actions',[0,1,4]),('source_zero_role','real_policy'),('on_policy_ready',True),('source_prefix_condition_dropout_probability',float('nan'))]:
   bad=deepcopy(p);bad['config']['runtime_contract']['rtc_prefix_sampling'][key]=value
   with self.assertRaisesRegex(RuntimeError,'fixed-H4'):rtc_prefix_contract(bad)
  bad=deepcopy(p);bad['config']['runtime_contract']['rtc']['includes_zero_prefix']=True
  with self.assertRaisesRegex(RuntimeError,'RTC prefix'):rtc_prefix_contract(bad)
 def test_actual_replay_export_cannot_claim_offline_sampler_or_zero_fast_coverage(self):
  from types import SimpleNamespace
  from dynamicwam.training.train.multirate_trainer import rtc_sampling_export_metadata
  _,p,r=contract_fixture(3);c=p['config']['runtime_contract'];cfg={'dataset':{'kind':'real_rollout'}}
  dataset=SimpleNamespace(replay_identity='a'*64,rtc_prefix_sampling_contract=rtc_prefix_sampling_contract('fixed_h4_v1'))
  metadata=rtc_sampling_export_metadata(cfg,dataset);self.assertEqual(metadata['mode'],'recorded_real_prefix_v1')
  c.update(rtc_prefix_sampling=metadata,replay_identity='a'*64,training_data_kind='real_rollout');c['rtc']['includes_zero_prefix']=False
  self.assertFalse(rtc_prefix_contract(p)['includes_zero_prefix']);validate_runtime_contract(p,r,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
  bad=deepcopy(p);bad['config']['runtime_contract']['training_data_kind']='physical_grid'
  with self.assertRaisesRegex(RuntimeError,'recorded RTC'):rtc_prefix_contract(bad)
  bad=deepcopy(p);bad['config']['runtime_contract']['rtc_prefix_sampling']['replay_identity']='b'*64
  with self.assertRaisesRegex(RuntimeError,'recorded RTC'):rtc_prefix_contract(bad)
  self.assertEqual(rtc_sampling_export_metadata({'dataset':{'kind':'physical_grid'}},dataset)['mode'],'fixed_h4_v1')
 def test_legacy_zero_prefix_contract_is_preserved(self):
  _,p,r=contract_fixture(3);self.assertTrue(rtc_prefix_contract(p)['includes_zero_prefix']);p['config']['runtime_contract']['rtc']['includes_zero_prefix']=False
  with self.assertRaises(RuntimeError):rtc_prefix_contract(p)
 def test_v33_smoke_launcher_uses_distinct_two_stage_outputs(self):
  ids=','.join('GPU-'+str(i) for i in range(8));a=launcher.commands(ids,Path('/authorization'),Path('/receipt'),{},'v3-subset');b=launcher.commands(ids,Path('/authorization'),Path('/receipt'),{},'v33-subset')
  self.assertEqual(len(b),2);self.assertTrue(all('v33_subset' in s['config'] for s in b));self.assertNotEqual(a[0]['training_run'],b[0]['training_run']);self.assertEqual(b[0]['full_export'],b[1]['initialization'])
  for stage in b:
   cfg=load_multirate_profile(stage['config']).training
   self.assertTrue(cfg['model']['action_expert']['source_physical_time_conditioning']);self.assertEqual(cfg['dataset']['rtc_prefix_sampling'],'fixed_h4_v1');self.assertEqual(cfg['max_steps'],4)
if __name__=='__main__':unittest.main()
