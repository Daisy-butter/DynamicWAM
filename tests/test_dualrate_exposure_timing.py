import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dynamicwam.dualrate.exposure_timing import (TIME_CONTRACT, grid_start, prefix_count,
    synthetic_templates, validate_template, validate_templates)
from dynamicwam.dualrate.system import DualRateSystem
from dynamicwam.dualrate.train_distributed import selected_rows, phase_contract


DT=45*float(np.float32(.004))


class ExposureTimingTests(unittest.TestCase):
    def test_midrow_query_preserves_grid_and_deadline(self):
        q=6*DT+.044
        row=dict(source_available=.12,request_time=DT+.016,publish_time=4*DT,
            query_time=q,query_rgb_time=5*DT,query_rgb_available=5*DT+.12,
            source_prefix_end=3*DT,fast_prefix_end=8*DT)
        checked=validate_template(row,DT)
        self.assertAlmostEqual(checked['fast_action_start'],6*DT)
        self.assertEqual(checked['source_prefix_rows'],3)
        self.assertEqual(checked['fast_prefix_rows'],2)
        self.assertAlmostEqual(row['fast_prefix_end']-q,2*DT-.044)

    def test_physical_boundary_noise_is_not_a_frame_tolerance(self):
        self.assertAlmostEqual(grid_start(2*DT+1e-8,0,DT),2*DT)
        self.assertAlmostEqual(grid_start(2*DT-.004,0,DT),DT)
        with self.assertRaises(ValueError):prefix_count(0,2*DT+.004,DT)

    def test_incomplete_or_future_reference_rejected(self):
        fixture=synthetic_templates(DT)
        row=copy.deepcopy(fixture['templates'][0]);row['publish_time']=row['query_time']+.001
        with self.assertRaises(ValueError):validate_template(row,DT)
        row=copy.deepcopy(fixture['templates'][0]);row['source_prefix_end']=0
        with self.assertRaises(ValueError):validate_template(row,DT)

    def test_fake_measured_fixture_rejected(self):
        payload=synthetic_templates(DT);payload['kind']='measured_parallel'
        with self.assertRaises(ValueError):validate_templates(payload,verify_evidence=False)

    def test_deadline_and_fresh_image_are_checked(self):
        row=synthetic_templates(DT)['templates'][0]
        bad=dict(row,query_rgb_available=row['query_time']+.01)
        with self.assertRaises(ValueError):validate_template(bad,DT)
        bad=dict(row,fast_submit_time=row['fast_prefix_end']-.005,submit_margin=.01)
        with self.assertRaises(ValueError):validate_template(bad,DT)

    def test_v4_age_filter_uses_template_ids(self):
        # Sorted unique age metadata is deliberately not a template-id map.
        dataset=SimpleNamespace(exposure_anchored=True,
            manifest={'time_contract':TIME_CONTRACT,'query_ages_seconds':[.72,1.08]},
            event_templates={'templates':[{'query_time':1.08},{'query_time':.72},{'query_time':1.08}]},
            rows=np.array([[0,10,0],[0,12,1],[1,15,2]],dtype=np.int64))
        self.assertEqual(selected_rows(dataset,[1.08]).tolist(),[0,2])
        self.assertEqual(selected_rows(dataset,[.72]).tolist(),[1])

    def test_source_availability_is_not_relabelled_as_exposure(self):
        batch={key:torch.tensor([value],dtype=torch.float64) for key,value in
            dict(source_time=1.,source_available_time=1.12,source_request_time=1.18,
                 completed_time=1.72,query_time=2.,query_available_time=1.94).items()}
        DualRateSystem._validate_clocks(batch)
        bad=dict(batch,completed_time=torch.tensor([1.],dtype=torch.float64))
        with self.assertRaises(ValueError):DualRateSystem._validate_clocks(bad)

    def test_phase_transition_binds_v4_clock_without_changing_legacy(self):
        config={k:'fixture' for k in ('sequence_id','init_checkpoint_sha256',
            'action_normalization_sha256','motion_normalization_sha256','architecture','codec_id','optimizer_kind')}
        config['world_size']=8
        identity={k:'fixture' for k in ('source_sha256','manifest_sha256','device_type','torch_version')}
        legacy=phase_contract(config,identity)
        self.assertNotIn('time_contract',legacy)
        v4=phase_contract(dict(config,time_contract=TIME_CONTRACT,action_interval_seconds=DT),identity)
        self.assertEqual(v4['time_contract'],TIME_CONTRACT)
        self.assertEqual(v4['action_interval_seconds'],DT)
        self.assertNotEqual(v4,phase_contract(dict(config,time_contract=TIME_CONTRACT,action_interval_seconds=.18),identity))


if __name__=='__main__':unittest.main()
