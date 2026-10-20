"""CPU-only explicit scaled protocol and finite wall-budget regression checks."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_motionkv_joint_eval_pipeline import (
    PROTOCOL, SCALED_PROTOCOL, evaluation_clock, wall_budgets, suite_command,
    verify_report_clock, verify_stage, save,
)
from test_joint_eval_pipeline import StageAcceptanceTests


class ScaledClockTests(unittest.TestCase):
    def test_only_explicit_supported_rates(self):
        self.assertEqual(evaluation_clock(), PROTOCOL)
        self.assertEqual(evaluation_clock(.25), SCALED_PROTOCOL)
        for rate in (0., .5, 2., float('nan'), float('inf'), True, '0.25'):
            with self.subTest(rate=rate), self.assertRaises(RuntimeError): evaluation_clock(rate)

    def test_wall_budgets_preserve_default_1x_and_fit_scaled_48h_broker(self):
        self.assertEqual(wall_budgets(SimpleNamespace()), dict(task_timeout_s=14400.,cohort_timeout_s=3600.,
            smoke_max_runtime_s=21600.,formal_max_runtime_s=172800.))
        scaled = wall_budgets(SimpleNamespace(simulation_rate=.25))
        self.assertEqual(scaled, dict(task_timeout_s=57600.,cohort_timeout_s=14400.,
            smoke_max_runtime_s=21600.,formal_max_runtime_s=144000.))
        self.assertEqual(scaled['smoke_max_runtime_s'] + scaled['formal_max_runtime_s'], 46 * 3600)
        self.assertEqual(wall_budgets(SimpleNamespace(simulation_rate=.25,task_timeout_s=999))['task_timeout_s'], 999)
        for value in (-1, 0, float('inf'), float('nan')):
            with self.assertRaises(RuntimeError): wall_budgets(SimpleNamespace(formal_max_runtime_s=value))

    def test_scaled_command_keeps_full_protocol_controls(self):
        args=SimpleNamespace(repo=Path('/tmp/repo'),run_root=Path('/tmp/run'),config=Path('/tmp/eval.yaml'),
            checkpoint=Path('/tmp/20k.pt'),checkpoint_audit=Path('/tmp/audit.json'),timing_contract=Path('/tmp/time.json'),
            frozen_scenes=Path('/tmp/frozen'),gpus=','.join(f'GPU-{i}' for i in range(8)),seed=100000,
            simulation_rate=.25,scene_compatibility=None)
        for stage in ('smoke','formal'):
            _,command=suite_command(args,stage=stage,tasks=['POB','PCP'])
            values={key:command[i+1] for i,key in enumerate(command[:-1]) if key.startswith('--') and not command[i+1].startswith('--')}
            for key,value in {'--simulation-rate':'0.25','--action-interval-s':'0.16',
                '--max-backlog-s':'0.25','--max-cohort-size':'8','--history-frame-period-s':'0.16',
                '--task-timeout-s':'57600.0','--cohort-timeout-s':'14400.0'}.items(): self.assertEqual(values[key],value)
            self.assertIn('--joint-v3',command)
            for flag in ('--resume','--probe-action-limit','--retry','--solo-tasks'):self.assertNotIn(flag,command)
            self.assertEqual('--tasks' in command, stage == 'smoke')
            self.assertEqual(values['--episodes'], '2' if stage == 'smoke' else '20')

    def test_scaled_requires_false_realtime_and_valid_continuous_clock(self):
        report=dict(protocol=SCALED_PROTOCOL,simulation_rate=.25,clock='scaled_wall_clock',
                    clock_valid=True,continuous_valid=True,realtime_valid=False,virtual_delay=False)
        verify_report_clock(report,simulation_rate=.25)
        for key,value in [('protocol',PROTOCOL),('simulation_rate',1.),('clock','actual_wall_clock'),
            ('clock_valid',False),('continuous_valid',False),('realtime_valid',True),('virtual_delay',True)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):verify_report_clock(dict(report,**{key:value}),simulation_rate=.25)


class ScaledStageAcceptanceTests(StageAcceptanceTests):
    def test_scaled_stage_is_accepted_only_when_explicit_and_consistent(self):
        save(self.root/'suite_started.json',dict(self.spec,protocol=SCALED_PROTOCOL,simulation_rate=.25))
        for path in self.root.glob('*/episode_*/report.json'):
            report=json.loads(path.read_text())
            report.update(protocol=SCALED_PROTOCOL,simulation_rate=.25,clock='scaled_wall_clock',realtime_valid=False)
            save(path,report)
        module=SimpleNamespace(summarize=lambda output,spec:self.summary)
        with patch.dict(sys.modules,{'eval_domino_motionkv_rtc_suite':module}):
            kwargs=dict(expected_tasks=self.tasks,episodes=2,uuids=self.uuids,hashes=self.hashes,
                        frozen_sha='frozen',previous_source='source')
            self.assertEqual(verify_stage(self.root,simulation_rate=.25,**kwargs)[1]['valid_completed'],16)
            with self.assertRaises(RuntimeError):verify_stage(self.root,**kwargs)
            path=self.root/self.tasks[0]/'episode_0000'/'report.json'
            report=json.loads(path.read_text());report['realtime_valid']=True;save(path,report)
            with self.assertRaises(RuntimeError):verify_stage(self.root,simulation_rate=.25,**kwargs)


if __name__ == '__main__':unittest.main()
