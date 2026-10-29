import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_dualrate_response_sequence as sequence

# Execute the production stdlib helpers, not a permissive string-coercing mock.
# This retains Path.open/with_suffix semantics without importing GPU dependencies.
helper_path=Path(__file__).resolve().parents[1]/'scripts/guarded_dualrate_smoke.py'
helper_ast=ast.parse(helper_path.read_text(encoding='utf-8'))
helper_nodes=[n for n in helper_ast.body if isinstance(n,ast.FunctionDef) and n.name in ('file_sha256','atomic')]
assert len(helper_nodes)==2
helper_namespace=dict(Path=Path,hashlib=hashlib,json=json,os=os)
exec(compile(ast.Module(body=helper_nodes,type_ignores=[]),str(helper_path),'exec'),helper_namespace)
file_sha256=helper_namespace['file_sha256']
atomic=helper_namespace['atomic']

class SequenceTest(unittest.TestCase):
    def test_actual_production_hash_requires_path(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture';path.write_bytes(b'actual Path contract')
            self.assertEqual(file_sha256(path),hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(AttributeError):file_sha256(str(path))

    def test_formal_construction_preserves_policy(self):
        tasks=[f'task_{i}' for i in range(35)]
        smoke=dict(tasks=tasks[:2],episodes=2,stage='smoke',runtime={'reserve':.3},checkpoint_sha256='ckpt')
        formal=sequence.make_formal(smoke,tasks,Path('/out'))
        self.assertEqual(formal['runtime'],smoke['runtime'])
        self.assertEqual(formal['checkpoint_sha256'],'ckpt')
        self.assertEqual(formal['episodes'],20)
        self.assertEqual(formal['tasks'],tasks)
        with self.assertRaises(ValueError):sequence.make_formal(smoke,tasks[:2],Path('/out'))

    def exercise(self, smoke_rc):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'sequence';out.mkdir()
            sp=root/'smoke.config.json';sp.write_text('{}')
            tp=root/'tasks.json';tp.write_text('[]')
            smoke=dict(storage_root=str(root),gpu_uuids=['gpu'],output=str(root/'smoke'),checkpoint_sha256='ckpt')
            formal=dict(smoke,output=str(root/'formal'))
            post=SimpleNamespace(file_sha256=file_sha256,proc_identity=lambda p:dict(pid=p,start_ticks=1),atomic=atomic,
                require_clean_guard=lambda *a:None,validate_pair=lambda *a:None,
                audit_suite=lambda *a:({},[],[]))
            guard=SimpleNamespace(frozen_source=lambda r:{'source':'same'},source_contract=lambda s:s,
                                  validate_acceptance=lambda *a:None)
            worker=SimpleNamespace(validate_config=lambda c:None)
            response=SimpleNamespace(run_gate=lambda r,g,p:atomic(p,{'passed':True}),validate_for_formal=lambda *a:True)
            calls=[]
            class Child:
                pid=123
                def __init__(self,rc):self.rc=rc
                def wait(self,timeout=None):return self.rc
                def poll(self):return self.rc
            def popen(cmd,**kw):
                calls.append((cmd,kw));return Child(smoke_rc if len(calls)==1 else 0)
            args=argparse.Namespace(repo=root,root=root,run_dir=out,smoke_config=sp,tasks_file=tp,
                                    smoke_max_runtime=28800,formal_max_runtime=172800)
            prepared=(root,root,(guard,post,worker,response),smoke,formal,{'source':'same'},{'cpu':True})
            lock_fixture=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
            with patch.dict(sys.modules,{'fcntl':lock_fixture}),patch.object(sequence,'preflight',return_value=prepared),patch.object(sequence.subprocess,'Popen',side_effect=popen):
                rc=sequence.run(args)
            return rc,calls,json.loads((out/'status.json').read_text())

    def test_failed_smoke_never_launches_formal(self):
        rc,calls,status=self.exercise(1)
        self.assertEqual(rc,1);self.assertEqual(len(calls),1);self.assertEqual(status['state'],'failed')

    def test_successful_sequence_owns_bounded_guard_processes(self):
        rc,calls,status=self.exercise(0)
        self.assertEqual(rc,0);self.assertEqual(len(calls),2);self.assertEqual(status['valid_completed'],700)
        for (cmd,kw),limit in zip(calls,('28800','172800')):
            self.assertIn('--max-runtime',cmd);self.assertIn(limit,cmd)
            self.assertNotIn('--detach',cmd);self.assertTrue(kw['start_new_session'])

if __name__=='__main__':unittest.main()
