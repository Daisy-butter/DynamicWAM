#!/usr/bin/env python3
"""Pure-stdlib evidence and failure-isolation regression checks, no GPU access."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name('run_temporal_feedback_pipeline.py')
tree = ast.parse(SOURCE.read_text())
names = {'EvaluationIncomplete', 'literal_constant', 'percentile', 'measured', 'latency_evidence',
    'probe_evidence', 'task_evidence', 'canary_evidence', 'formal_arms', 'common_evidence', 'Pipeline',
    'same_path', 'runtime_identity_evidence', 'runtime_evidence'}
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
nodes += [node for node in tree.body if isinstance(node, ast.Assign) and any(
    isinstance(target, ast.Name) and target.id == 'TASKS' for target in node.targets)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data))


ns = dict(Path=Path, ast=ast, json=json, math=math, sha256=sha256, write_json=write_json)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)


class TemporalPipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        ns['ROOT'] = self.root
        (self.root/'scripts').mkdir()
        (self.root/'scripts/profile_temporal_feedback.py').write_text("FILES = ('model.py',)\n")
        (self.root/'model.py').write_text('source')
        self.file = self.root/'latency.json'
        interpreter=self.root/'benchmark/bin/python3.10';interpreter.parent.mkdir(parents=True);interpreter.write_bytes(b'python')
        self.runtime=dict(executable=str(interpreter),python='3.10.20',torch='2.4.1+cu121',cuda='12.1',cudnn=90100,
            prefix=str(interpreter.parent.parent),gpu_name='NVIDIA RTX A6000',gpu_capability=[8,6])
        self.report = dict(base_checkpoint_sha256='official', passed=True, selected_horizon=2,
            runtime=copy.deepcopy(self.runtime),
            zero_feedback_native_full_plan_exact=True, cold_feedback_query_ms=1200.,
            source_hashes={'model.py': sha256(self.root/'model.py')},
            native_zero_anchors=[dict(horizon=h, variation=v, exact=True) for h in (1, 2, 4) for v in range(3)],
            official_eager=self.series(320.), native_graph_planner=self.series(225.),
            feedback_query=self.series(4.5), by_horizon={})
        for h in (1, 2, 4):
            cycle = 225.+(16//h-1)*4.5
            self.report['by_horizon'][str(h)] = dict(fast=self.series(4.5), ms_per_action=cycle/16,
                ratio_to_official=cycle/320., ratio_to_graph_zero=cycle/225., passed=cycle/320. <= .85)
        self.rows = {f'{arm}/{task}': dict(seeds=[100000, 100001], success_count=1)
            for arm in ('zero', 'temporal') for task in ns['TASKS']}

    @staticmethod
    def series(value):
        return dict(samples_ms=[value]*20, mean_ms=value, p95_ms=value)

    def latency(self, report=None):
        write_json(self.file, report or self.report)
        return ns['latency_evidence'](self.file, 'official', expected_runtime=self.runtime)

    def test_budget_selects_h2_without_relaxing_h1_gate(self):
        evidence = self.latency()
        self.assertEqual(evidence['selected_horizon'], 2)
        self.assertFalse(evidence['by_horizon']['1']['passed'])
        self.assertAlmostEqual(evidence['by_horizon']['2']['ms_per_action'], 256.5/16)
        self.assertGreater(evidence['by_horizon']['2']['ratio_to_graph_zero'], 1.)

    def test_horizon_or_latency_or_source_forgery_is_rejected(self):
        bad = []
        value = copy.deepcopy(self.report); value['selected_horizon'] = 1; bad.append(value)
        value = copy.deepcopy(self.report); value['by_horizon']['1']['passed'] = True; bad.append(value)
        value = copy.deepcopy(self.report); value['by_horizon']['2']['fast']['p95_ms'] = 1; bad.append(value)
        value = copy.deepcopy(self.report); value['by_horizon']['2']['ratio_to_official'] = .1; bad.append(value)
        value = copy.deepcopy(self.report); value['source_hashes']['model.py'] = 'old'; bad.append(value)
        value = copy.deepcopy(self.report); value['native_zero_anchors'].pop(); bad.append(value)
        value = copy.deepcopy(self.report); value['native_zero_anchors'][0]['exact'] = False; bad.append(value)
        value = copy.deepcopy(self.report); value['official_eager']['samples_ms'][0] = float('nan'); bad.append(value)
        value = copy.deepcopy(self.report); value['official_eager']['samples_ms'].pop(); bad.append(value)
        for i, value in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(RuntimeError):
                self.latency(value)

    def test_loaded_checkpoint_and_packet_identity_are_required(self):
        checkpoint, packet = self.root/'head.pt', self.root/'packet.npz'
        checkpoint.write_bytes(b'head'); packet.write_bytes(b'packet')
        self.report.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), packet_sha256=sha256(packet))
        write_json(self.file, self.report)
        ns['latency_evidence'](self.file, 'official', checkpoint, packet, self.runtime)
        packet.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            ns['latency_evidence'](self.file, 'official', checkpoint, packet, self.runtime)

    def test_latency_rejects_training_interpreter_or_missing_deployment_provenance(self):
        for field,value in [('python','3.12.10'),('torch','2.8.0+cu128'),('cuda','12.8'),('cudnn',90200),
                ('prefix',str(self.root)),('executable',str(self.root/'python')),('gpu_capability',None)]:
            report=copy.deepcopy(self.report);report['runtime'][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.latency(report)
        report=copy.deepcopy(self.report);report.pop('runtime')
        with self.assertRaises(RuntimeError):self.latency(report)
        write_json(self.file,self.report)
        with self.assertRaises(RuntimeError):ns['latency_evidence'](self.file,'official')

    def test_independent_runtime_probe_does_not_initialize_cuda_and_uses_configured_launcher(self):
        path=self.root/'runtime.log'
        value=dict(self.runtime,cuda_initialized=False)
        path.write_text('optional import warning\n'+json.dumps(value)+'\n')
        expected=ns['runtime_identity_evidence'](path,self.runtime['executable'])
        self.assertEqual(expected['python'],'3.10.20')
        value['cuda_initialized']=True;path.write_text(json.dumps(value))
        with self.assertRaises(RuntimeError):ns['runtime_identity_evidence'](path,self.runtime['executable'])

    def test_loaded_latency_command_uses_domino_interpreter_verbatim(self):
        pipeline=ns['Pipeline'].__new__(ns['Pipeline'])
        pipeline.root,pipeline.python,pipeline.deployment_python=self.root,'training-python','snapshot/external/robotwin/bin/python3.10'
        pipeline.gpus,pipeline.env=['0'],{}
        pipeline.args=type('Args',(),dict(packet=self.root/'packet'))()
        captured=[]
        pipeline.run=lambda name,command,verify,**kwargs:captured.append(command)
        pipeline.profile_checkpoint('loaded',self.root/'head.pt')
        self.assertEqual(captured[0][0],pipeline.deployment_python)

    def test_canary_is_a_declared_tripwire_not_a_quality_estimate(self):
        self.assertTrue(ns['canary_evidence'](self.rows)['passed'])
        for task in ns['TASKS']:
            self.rows[f'temporal/{task}']['success_count'] = 0
        self.assertFalse(ns['canary_evidence'](self.rows)['passed'])
        for task in ns['TASKS']:
            self.rows[f'zero/{task}']['success_count'] = 0
        result = ns['canary_evidence'](self.rows)
        self.assertTrue(result['passed'])
        self.assertTrue(result['no_success_in_either_arm'])
        self.rows['temporal/place_shoe']['seeds'] = [100000, 100002]
        with self.assertRaises(ns['EvaluationIncomplete']):
            ns['canary_evidence'](self.rows)

    def test_formal_arms_deduplicate_h4_but_keep_history_and_zero_controls(self):
        self.assertEqual(set(ns['formal_arms'](2)), {'zero_H2', 'temporal_H2', 'temporal_H4', 'no_history_H2'})
        self.assertEqual(set(ns['formal_arms'](4)), {'zero_H4', 'temporal_H4', 'no_history_H4'})

    def task(self, success=True):
        return dict(seeds=[100000], episodes=[dict(seed=100000, success=success)], success_count=int(success),
            reported_episodes=1, executed_steps=4, query_ms_sum=12.)

    def test_valid_common_intersection_never_imputes_missing_tasks_as_zero(self):
        arms = dict(zero=dict(tasks={'a': self.task(), 'b': self.task()}),
            temporal=dict(tasks={'a': self.task(False)}))
        result = ns['common_evidence'](arms, ['a', 'b'])
        self.assertEqual(result['valid_common_tasks'], ['a'])
        self.assertEqual(result['arms']['zero']['episodes'], 1)
        self.assertEqual(result['pairs']['temporal_minus_zero']['regressed'], 1)
        arms['temporal']['tasks'] = {}
        result = ns['common_evidence'](arms, ['a', 'b'])
        self.assertIsNone(result['arms']['temporal']['success_rate'])
        self.assertEqual(result['arms']['temporal']['episodes'], 0)

    def test_all_formal_arms_run_after_one_incomplete_arm(self):
        pipeline = ns['Pipeline'].__new__(ns['Pipeline'])
        pipeline.root, pipeline.python, pipeline.gpus = self.root, 'python', ['0']
        pipeline.deploy_config, pipeline.fixed_policy, pipeline.manifest = self.root/'config', self.root/'eval.py', self.root/'seeds'
        pipeline.seeds = {'a': [100000]}
        pipeline.record = lambda *args: None
        called = []
        def run(name, command, verify, **kwargs):
            called.append(name)
            self.assertTrue(kwargs['allow_nonzero'])
            return dict(status='incomplete' if len(called) == 1 else 'complete', command_exit_code=2 if len(called) == 1 else 0,
                tasks={} if len(called) == 1 else {'a': self.task()})
        pipeline.run = run
        self.assertFalse(pipeline.evaluate(self.root/'head.pt', 2))
        self.assertEqual(len(called), 4)
        self.assertEqual(called[-1], 'formal_no_history_H2')

    def test_task_evidence_checks_horizon_history_and_adapter_scope(self):
        directory = self.root/'task'; (directory/'logs').mkdir(parents=True)
        checkpoint = self.root/'head.pt'; checkpoint.write_bytes(b'head')
        aggregate = dict(success_count=1, episodes=[dict(seed=100000, success=True), dict(seed=100001, success=False)])
        write_json(directory/'aggregate.json', aggregate)
        ns['valid_task_aggregate'] = lambda *args, **kwargs: aggregate
        timing = dict(checkpoint=str(checkpoint), base_checkpoint_sha256='official', feedback_enabled=True,
            history_enabled=True, execute_horizon=2, query_count=1, executed_steps=2, query_ms_sum=10.)
        query = dict(observation_to_action_ms=12., adapter_preprocess_ms=2.,
            decision_interval_sim_ms=None, decision_interval_wall_ms=None)
        log = directory/'logs/slot0.log'
        def write():
            log.write_text('\n'.join(['TEMPORAL-FEEDBACK-QUERY '+repr(query), 'TEMPORAL-FEEDBACK-TIMING '+repr(timing)]*2))
        write()
        result = ns['task_evidence'](directory, [100000, 100001], checkpoint, 'official', feedback=True, history=True, horizon=2)
        self.assertEqual(result['ms_per_executed_action'], 5.)
        self.assertEqual(result['response']['observation_to_action_ms']['mean'], 12.)
        self.assertIsNone(result['response']['decision_interval_sim_ms']['p95'])
        timing['history_enabled'] = False; write()
        with self.assertRaises(RuntimeError):
            ns['task_evidence'](directory, [100000, 100001], checkpoint, 'official', feedback=True, history=True, horizon=2)
        ns['valid_task_aggregate'] = lambda *args, **kwargs: None
        with self.assertRaises(ns['EvaluationIncomplete']):
            ns['task_evidence'](directory, [100000, 100001], checkpoint, 'official', feedback=True, history=True, horizon=2)

    def test_probe_reuse_requires_v2_fixed_budget_source_and_cache_identity(self):
        (self.root/'scripts/probe_temporal_feedback.py').write_text("SOURCE_FILES = ('model.py',)\n")
        config, cache, checkpoint = self.root/'config.yaml', self.root/'cache.pt', self.root/'probe.pt'
        config.write_text('config'); cache.write_bytes(b'cache'); checkpoint.write_bytes(b'head')
        base = dict(path='official.pt', sha256='official')
        metadata = dict(format='dynamicwam_temporal_feedback_probe_cache', version=2,
            base_checkpoint=base, config_sha256=sha256(config), dataset_identity='dataset', pairing_identity='pairs',
            train_samples=512, validation_samples=256, source_hashes={'model.py':sha256(self.root/'model.py')},
            split=dict(overlap=0, train_episodes=32, heldout_episodes=16),
            deployment_contract=dict(format='dynamicwam_temporal_plan_feedback_checkpoint', version=2,
                history_training_raw_lags=[1,2,3,6,9,12,18,24,36], history_sampling='pair_identity_sha256_v1',
                plan_age_seconds='telemetry_only'))
        report = dict(format='dynamicwam_temporal_feedback_probe_report', version=1, cache_metadata=metadata,
            cache_path=str(cache), cache_sha256=sha256(cache),
            deployment_checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
            metrics={name:dict(all=dict(target_mse=mse, zero_target_mse=1.05, rows=256, active_coordinates=10000))
                for name,mse in [('static',1.),('temporal',.98),('shuffled',.99)]},
            gate=dict(passed=True, relative_arm_mse_gain=.02),
            args=dict(train_samples=512,validation_samples=256,teacher_batch_size=1,batch_size=32,
                epochs=20,lr=2e-4,seed=20261006,history_dropout=.15))
        ns['probe_checkpoint_evidence'] = lambda *args: dict(phase='analysis_probe')
        path=self.root/'probe.json'
        def check(value):
            write_json(path,value)
            return ns['probe_evidence'](path,config=config,base=base,smoke=dict(dataset_identity='dataset',pairing_identity='pairs'))
        self.assertTrue(check(report)['passed'])
        bad=[]
        value=copy.deepcopy(report);value['cache_metadata']['version']=1;bad.append(value)
        value=copy.deepcopy(report);value['cache_metadata']['split']['overlap']=1;bad.append(value)
        value=copy.deepcopy(report);value['cache_metadata']['deployment_contract']['history_training_raw_lags']=[3];bad.append(value)
        value=copy.deepcopy(report);value['args']['seed']=7;bad.append(value)
        value=copy.deepcopy(report);value['args']['epochs']=200;bad.append(value)
        value=copy.deepcopy(report);value['metrics']['temporal']['all']['rows']=0;bad.append(value)
        value=copy.deepcopy(report);value['metrics']['shuffled']['all']['target_mse']=.9;bad.append(value)
        value=copy.deepcopy(report);value['cache_sha256']='old';bad.append(value)
        value=copy.deepcopy(report);value['cache_metadata']['source_hashes']['model.py']='old';bad.append(value)
        value=copy.deepcopy(report)
        for arm in ('static','temporal','shuffled'):value['metrics'][arm]['all']['zero_target_mse']=.97
        bad.append(value)
        value=copy.deepcopy(report);value['metrics']['temporal']['all'].pop('zero_target_mse');bad.append(value)
        value=copy.deepcopy(report);value['metrics']['static']['all']['zero_target_mse']=.9;bad.append(value)
        for i,value in enumerate(bad):
            with self.subTest(i=i),self.assertRaises(RuntimeError):check(value)
        worse=copy.deepcopy(report)
        for arm in ('static','temporal','shuffled'):worse['metrics'][arm]['all']['zero_target_mse']=.97
        with self.assertRaisesRegex(RuntimeError,'beating a worse static head is insufficient'):check(worse)
        equal=copy.deepcopy(report)
        for arm in ('static','temporal','shuffled'):equal['metrics'][arm]['all']['zero_target_mse']=.98
        self.assertTrue(check(equal)['temporal_not_worse_than_zero'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
