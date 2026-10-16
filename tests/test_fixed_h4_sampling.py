import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.training.data.physical_grid import (
    sample_async_timing, target_grid, rtc_prefix_sampling_contract, json_hash,
)


class FixedH4SamplingTest(unittest.TestCase):
    def sample(self, epoch, source=.037, current=1.071, **extra):
        return sample_async_timing(source,current,target_grid(current,0.),
            source_action_times=target_grid(source,0.),seed=9,epoch=epoch,
            source_id=5,current_id=17,**extra)

    def test_fixed_h4_does_not_change_actual_latency_or_capture_clocks(self):
        # Boundary samples also retain the legacy zero-latency draw, but the
        # steady controller still has four committed actions.
        for current in (1.071,1.12):
            for epoch in range(80):
                old=self.sample(epoch,current=current)
                new=self.sample(epoch,current=current,rtc_prefix_sampling='fixed_h4_v1')
                for key in ('source_capture_time','current_capture_time','query_time','ready_time',
                            'fast_latency_seconds','cache_publication_time','slow_latency_seconds','query_phase_seconds'):
                    self.assertEqual(new[key],old[key],key)
                self.assertEqual(new['rtc_prefix_lengths'],4)
                self.assertLess(new['ready_time'],new['commit_horizon_time'])
                self.assertLessEqual(new['cache_publication_time'],new['query_time'])
                self.assertAlmostEqual(new['commit_horizon_seconds'],new['commit_horizon_time']-current)

    def test_source_condition_dropout_is_only_zero_or_four(self):
        samples=[self.sample(i,rtc_prefix_sampling='fixed_h4_v1') for i in range(1000)]
        lengths=[s['source_rtc_prefix_lengths'] for s in samples]
        self.assertEqual(set(lengths),{0,4})
        self.assertTrue(.15 < lengths.count(0)/len(lengths) < .25)
        for p,expected in [(0.,4),(1.,0)]:
            self.assertEqual(self.sample(1,rtc_prefix_sampling='fixed_h4_v1',
                source_prefix_condition_dropout_probability=p)['source_rtc_prefix_lengths'],expected)
        contract=rtc_prefix_sampling_contract('fixed_h4_v1',.2)
        self.assertEqual(contract['source_zero_role'],'bootstrap_condition_dropout_only')
        self.assertFalse(contract['on_policy_ready']);self.assertFalse(contract['hold_coverage'])
        self.assertEqual(contract['prefix_source'],'expert_demonstration')

    def test_legacy_default_retains_variable_prefixes(self):
        samples=[self.sample(i,source=0.) for i in range(200)]
        self.assertEqual({s['rtc_prefix_lengths'] for s in samples},{1,2,3,4})
        self.assertEqual({s['source_rtc_prefix_lengths'] for s in samples},{0,1,2,3,4})
        self.assertNotIn('commit_horizon_time',samples[0])
        self.assertEqual(self.sample(3),self.sample(3,rtc_prefix_sampling='delay_minimum_v1'))

    def test_fixed_mode_rejects_incompatible_or_noncausal_inputs(self):
        for extra in ({'rtc_prefix_max':3},{'source_prefix_condition_dropout_probability':-1},
                      {'source_prefix_condition_dropout_probability':float('nan')}):
            with self.assertRaises(ValueError):self.sample(1,rtc_prefix_sampling='fixed_h4_v1',**extra)
        with self.assertRaises(ValueError):self.sample(1,source=2.,current=1.,rtc_prefix_sampling='fixed_h4_v1')
        with self.assertRaises(ValueError):self.sample(1,rtc_prefix_sampling='typo')

    def test_identity_contract_changes_with_mode_and_dropout(self):
        legacy=dict(grid='grid-id',async_schema='joint_source_current_prefix_v2',max_age=31,
            async_seed=20261015,slow_budget_seconds=2.56,rtc_prefix_max=4)
        fixed={**legacy,'rtc_prefix_sampling':rtc_prefix_sampling_contract('fixed_h4_v1',.2)}
        changed={**legacy,'rtc_prefix_sampling':rtc_prefix_sampling_contract('fixed_h4_v1',.3)}
        self.assertEqual(len({json_hash(x) for x in (legacy,fixed,changed)}),3)

    def test_profile_parser_forwards_mode_and_rejects_replay_override(self):
        root=Path(__file__).resolve().parents[1]
        raw=yaml.safe_load((root/'configs/multirate/joint_motionkv_v3_subset_warmup_smoke.yaml').read_text())
        raw['pairing']['rtc_prefix_sampling']='fixed_h4_v1'
        raw['pairing']['source_prefix_condition_dropout_probability']=.2
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.yaml'
            p.write_text(yaml.safe_dump(raw))
            parsed=load_multirate_profile(p).training['dataset']
            self.assertEqual(parsed['rtc_prefix_sampling'],'fixed_h4_v1')
            self.assertEqual(parsed['source_prefix_condition_dropout_probability'],.2)
            raw['training']['rtc_prefix_max']=3;p.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError,'fixed_h4'):load_multirate_profile(p)
            raw['training']['rtc_prefix_max']=4;raw['pairing']['kind']='real_rollout';p.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError,'real replay'):load_multirate_profile(p)


if __name__=='__main__':unittest.main()
