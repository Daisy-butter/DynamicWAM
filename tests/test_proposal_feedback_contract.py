"""Proposal feature flag and bounded smoke configuration fail closed."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_v3_deployment_contract import contract_fixture
from dynamicwam.inference.fast_runtime import validate_runtime_contract,RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.config.multirate import load_multirate_profile

class ProposalContractTests(unittest.TestCase):
    def test_enabled_feedback_requires_matching_metadata(self):
        _,payload,runtime=contract_fixture(3)
        payload['modules']['fast_conditioner']['proposal_feedback']=True
        with self.assertRaisesRegex(RuntimeError,'proposal feedback'):
            validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
        payload['config']['runtime_contract']['proposal_feedback']='sigma_conditioned_physical_plan_v1'
        validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
        payload['modules']['fast_conditioner']['proposal_feedback']=False
        with self.assertRaisesRegex(RuntimeError,'proposal feedback'):
            validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)

    def test_completed_plan_requires_distinct_runtime_provenance(self):
        _,payload,runtime=contract_fixture(3)
        payload['modules']['fast_conditioner'].update(proposal_feedback=True,complete_source_plan=True)
        payload['config']['runtime_contract']['proposal_feedback']='sigma_conditioned_physical_plan_v1'
        with self.assertRaisesRegex(RuntimeError,'completed source plan'):
            validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
        payload['config']['runtime_contract']['source_plan_completion']='same_noise_native_K10_tail_stopgrad_v1'
        validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
        payload['modules']['fast_conditioner']['complete_source_plan']=False
        with self.assertRaisesRegex(RuntimeError,'completed source plan'):
            validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)

    def test_original_v3_without_feedback_field_keeps_original_semantics(self):
        _,payload,runtime=contract_fixture(3)
        payload['modules']['fast_conditioner'].pop('proposal_feedback',None)
        payload['config']['runtime_contract'].pop('proposal_feedback',None)
        validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)

    def test_subset_rejects_formal_resume_replay_and_non_v3(self):
        path=ROOT/'configs/multirate/joint_motionkv_v3_warmup_smoke.yaml'
        raw=yaml.safe_load(path.read_text());raw['pairing']['smoke_subset']=True
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            target=Path(folder)/'profile.yaml';target.write_text(yaml.safe_dump(raw))
            cfg=load_multirate_profile(target).training
            self.assertTrue(cfg['dataset']['smoke_subset'])
            for fields in ({'max_steps':5},{'resume_from':'/tmp/resume'}):
                bad=deepcopy(raw);bad['training'].update(fields);target.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError,'smoke_subset'):
                    load_multirate_profile(target)
            bad=deepcopy(raw);bad['pairing']['kind']='real_rollout';target.write_text(yaml.safe_dump(bad))
            with self.assertRaisesRegex(ValueError,'smoke_subset'):load_multirate_profile(target)
            bad=deepcopy(raw);bad['model']['fast_conditioner']['interface_version']=2;target.write_text(yaml.safe_dump(bad))
            with self.assertRaisesRegex(ValueError,'smoke_subset'):load_multirate_profile(target)

    def test_all_v3_profiles_enable_new_feedback(self):
        for stage in ('warmup','joint','replay'):
            for tail in ('','_smoke'):
                cfg=load_multirate_profile(ROOT/f'configs/multirate/joint_motionkv_v3_{stage}{tail}.yaml').training
                self.assertTrue(cfg['model']['fast_conditioner']['proposal_feedback'])

if __name__=='__main__':unittest.main()
