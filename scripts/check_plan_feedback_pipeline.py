#!/usr/bin/env python3
"""Pure-stdlib evidence-gate checks; no trainer, CUDA or subprocess launch."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).with_name('run_plan_feedback_pipeline.py')
tree=ast.parse(path.read_text())
names={'EvaluationIncomplete','latency_evidence','canary_evidence','task_evidence'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('TASKS','PROFILE_FILES') for t in n.targets)]
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ns=dict(Path=Path,ast=ast,json=json,math=math,sha256=sha256)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);ns['ROOT']=self.root
        for p in ns['PROFILE_FILES']:
            file=self.root/p;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(p)
        self.report=dict(base_checkpoint_sha256='official',zero_feedback_native_full_plan_exact=True,passed=True,
            source_hashes={p:sha256(self.root/p) for p in ns['PROFILE_FILES']})
        for name,value in [('official_eager',200.),('native_graph_planner',100.),('feedback_query',10.)]:
            self.report[name]=dict(samples_ms=[value]*20,mean_ms=value,p95_ms=value)
        self.file=self.root/'latency.json'
        self.rows={f'{arm}/{task}':dict(seeds=[100000,100001],success_count=1)
            for arm in ('zero_plan_control','feedback') for task in ns['TASKS']}

    def latency(self,report=None):
        self.file.write_text(json.dumps(self.report if report is None else report))
        return ns['latency_evidence'](self.file,'official')

    def test_valid_latency_computes_budget_from_raw_means(self):
        evidence=self.latency()
        self.assertAlmostEqual(evidence['ratio_to_original_h16'],.65)
        self.assertAlmostEqual(evidence['ratio_to_graph_h16'],1.3)
        self.assertAlmostEqual(evidence['ms_per_action'],8.125)

    def test_latency_rejects_stale_source_false_zero_anchor_or_forged_tail(self):
        cases=[]
        stale=copy.deepcopy(self.report);stale['source_hashes'][ns['PROFILE_FILES'][0]]='wrong';cases.append(stale)
        anchor=copy.deepcopy(self.report);anchor['zero_feedback_native_full_plan_exact']=False;cases.append(anchor)
        forged=copy.deepcopy(self.report);forged['feedback_query']['p95_ms']=1.;cases.append(forged)
        fast=copy.deepcopy(self.report);fast['feedback_query']=dict(samples_ms=[30.]*20,mean_ms=30.,p95_ms=30.);cases.append(fast)
        few=copy.deepcopy(self.report);few['feedback_query']['samples_ms']=[10.]*19;cases.append(few)
        nonfinite=copy.deepcopy(self.report);nonfinite['feedback_query']['samples_ms'][0]=float('nan');cases.append(nonfinite)
        for i,case in enumerate(cases):
            with self.subTest(case=i),self.assertRaises(RuntimeError):self.latency(case)

    def test_canary_counts_separate_arms_and_enforces_declared_tripwire(self):
        self.assertTrue(ns['canary_evidence'](self.rows)['passed'])
        rows=copy.deepcopy(self.rows)
        for task in ns['TASKS']:rows[f'feedback/{task}']['success_count']=0
        self.assertFalse(ns['canary_evidence'](rows)['passed'])
        for task in ns['TASKS']:rows[f'zero_plan_control/{task}']['success_count']=0
        result=ns['canary_evidence'](rows)
        self.assertTrue(result['passed']);self.assertTrue(result['no_success_in_either_arm'])

    def test_canary_rejects_missing_arm_or_replaced_seed(self):
        rows=copy.deepcopy(self.rows);rows.pop(next(iter(rows)))
        with self.assertRaises(RuntimeError):ns['canary_evidence'](rows)
        rows=copy.deepcopy(self.rows);rows['feedback/place_shoe']['seeds']=[100000,100002]
        with self.assertRaises(ns['EvaluationIncomplete']):ns['canary_evidence'](rows)

    def test_task_timing_checks_loader_identity_and_weighted_aggregation(self):
        directory=self.root/'task';(directory/'logs').mkdir(parents=True)
        checkpoint=self.root/'feedback.pt';checkpoint.write_bytes(b'example')
        aggregate=dict(success_count=1,episodes=[dict(seed=100000,success=True),dict(seed=100001,success=False)])
        (directory/'aggregate.json').write_text(json.dumps(aggregate))
        ns['valid_task_aggregate']=lambda *args,**kwargs:aggregate
        timings=[dict(checkpoint=str(checkpoint),base_checkpoint_sha256='official',feedback_enabled=True,
            execute_horizon=4,query_count=q,executed_steps=s,query_ms_sum=m) for q,s,m in [(1,4,10.),(3,12,90.)]]
        log=directory/'logs/slot0.log'
        def write():log.write_text('\n'.join('PLAN-FEEDBACK-TIMING '+repr(r) for r in timings))
        write();result=ns['task_evidence'](directory,[100000,100001],checkpoint,'official',feedback=True)
        self.assertEqual(result['ms_per_executed_action'],6.25);self.assertEqual(result['query_ms_mean'],25.)
        timings[0]['feedback_enabled']=False;write()
        with self.assertRaises(RuntimeError):ns['task_evidence'](directory,[100000,100001],checkpoint,'official',feedback=True)
        ns['valid_task_aggregate']=lambda *args,**kwargs:None
        with self.assertRaises(ns['EvaluationIncomplete']):ns['task_evidence'](directory,[100000,100001],checkpoint,'official',feedback=True)

if __name__=='__main__':unittest.main(verbosity=2)
