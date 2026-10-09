#!/usr/bin/env python3
"""CPU gate contracts only; these tests are not real physics evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).with_name('supervise_domino_continuous_preflight.py')
spec=importlib.util.spec_from_file_location('continuous_preflight',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def summary():
    return dict(status='complete',accepted_record=dict(sha256='accepted'),
        controller=dict(camera_pipeline='independent_scene_snapshot_renderer',control_period_s=.1,
            physics_dt=.004,max_backlog_s=.1,max_tick_gap_s=.1),report=dict(
            realtime_valid=True,asynchronous_camera=True,virtual_delay=False,
            startup_first_query_included=True,camera_inflight_ticks=50,inference_inflight_ticks=80,
            tick_lag_s=[.004,.009],tick_gap_s=[.004,.013],physics_time=2.,wall_time=2.002,
            stop_reason='physics_timeout',success=False,
            requests=[dict(worker_compute_s=.25,physics_advanced_during_worker_s=.248,
                metadata=dict(age_actions=4))]))


class Gates(unittest.TestCase):
    def test_complete_continuous_evidence_and_reuse(self):
        self.assertEqual(module.check_real_clock(summary(),'accepted',model=True)['nonfresh_queries'],1)

    def test_paused_worker_or_camera_and_relaxed_limit_rejected(self):
        for field,value in [('asynchronous_camera',False),('camera_inflight_ticks',0),
                ('virtual_delay',True),('realtime_valid',False)]:
            item=summary();item['report'][field]=value
            with self.assertRaises(RuntimeError):module.check_real_clock(item,'accepted')
        item=summary();item['controller']['max_tick_gap_s']=.2
        with self.assertRaises(RuntimeError):module.check_real_clock(item,'accepted')
        item=summary();item['report']['requests'][0]['physics_advanced_during_worker_s']=0
        with self.assertRaises(RuntimeError):module.check_real_clock(item,'accepted')

    def test_clock_nan_and_fresh_only_not_passing(self):
        for value in [float('nan'),.10001]:
            item=summary();item['report']['tick_gap_s']=[value]
            with self.assertRaises(RuntimeError):module.check_real_clock(item,'accepted')
        item=summary();item['report']['requests'][0]['metadata']['age_actions']=0
        with self.assertRaises(RuntimeError):module.check_real_clock(item,'accepted',model=True)

    def test_snapshot_clock_roundoff_and_retimestamp_rejected(self):
        event=dict(event='camera_ready',observation_sim_time=.1,
            snapshot_oracle=dict(passed=True,physics_steps=0,source_sim_time_seconds=.1000000047))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'events.jsonl';path.write_text(json.dumps(event)+'\n')
            self.assertEqual(module.check_snapshot_events(path)['verified_captures'],1)
            event['observation_sim_time']=.2;path.write_text(json.dumps(event)+'\n')
            with self.assertRaises(RuntimeError):module.check_snapshot_events(path)


if __name__=='__main__':unittest.main()
