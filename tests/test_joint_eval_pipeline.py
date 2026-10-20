"""CPU-only fail-closed pipeline receipts, scope and command contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_motionkv_joint_eval_pipeline import (
    GRAPH_KIND,LIFETIME_MODE,DRIVER_KEYS,check_graph_receipt,verify_support_guard,
    contained,smoke_tasks,suite_command,verify_stage,PROTOCOL,sha,save,revalidate_graph_receipt,close_kernel_broker,
)


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.uuids=[f'GPU-{i}' for i in range(8)]
        self.driver=dict(zip(DRIVER_KEYS,('node100','boot100',0,'all','last')))
        self.auth=dict(driver=self.driver,guard_sha256='guard',device_lifetime_mode=LIFETIME_MODE,
                       created_at='2026-10-19T00:00:00+00:00')
        artifact=self.root/'actual_graph.json';artifact.write_text('{"actual_test":true}')
        self.hashes=dict(checkpoint_sha256='ckpt20k',config_sha256='config',source_sha256='source')
        self.receipt=dict(schema_version=1,kind=GRAPH_KIND,passed=True,host='node100',boot_id='boot100',
            **self.hashes,gpu_uuids_tested=self.uuids[:1],guard_gpu_uuids=self.uuids,
            graph_parity=dict(passed=True,cases=20,max_abs_error=0.),
            background_concurrency=dict(passed=True,fast_consumed_while_slow_running=True),
            evidence_files=[dict(path=str(artifact),sha256=sha(artifact))])

    def check_graph(self,receipt):
        return check_graph_receipt(receipt,auth=self.auth,uuids=self.uuids,
                                   hashes=self.hashes,storage=self.root)

    def test_single_gpu_graph_with_all_adapter_guards_is_valid(self):
        self.assertEqual(self.check_graph(self.receipt),self.uuids[:1])

    def test_wrong_graph_provenance_or_scope_rejected(self):
        for key,value in [('schema_version',0),('kind','oldgraph'),('passed',False),('host','node50'),
                ('boot_id','oldboot'),('checkpoint_sha256','step12000'),('config_sha256','old'),
                ('source_sha256','old'),('gpu_uuids_tested',[]),('gpu_uuids_tested',['GPU-other']),
                ('guard_gpu_uuids',self.uuids[:1]),('evidence_files',[])]:
            with self.subTest(key=key,value=value):
                bad=copy.deepcopy(self.receipt);bad[key]=value
                with self.assertRaises(RuntimeError):self.check_graph(bad)

    def test_graph_parity_and_actual_overlap_are_both_required(self):
        for key,patches in [('graph_parity',dict(passed=False)),('graph_parity',dict(cases=0)),
                           ('graph_parity',dict(max_abs_error=float('nan'))),
                           ('background_concurrency',dict(fast_consumed_while_slow_running=False))]:
            bad=copy.deepcopy(self.receipt);bad[key].update(patches)
            with self.assertRaises(RuntimeError):self.check_graph(bad)

    def test_evidence_hash_tamper_rejected(self):
        Path(self.receipt['evidence_files'][0]['path']).write_text('changed')
        with self.assertRaises(RuntimeError):self.check_graph(self.receipt)

    def support_fixture(self,preflight):
        directory=self.root/('preflight' if preflight else 'graph');directory.mkdir()
        events=[dict(event=kind,uuid=uuid,fd=i,path=f'/dev/nvidia{i}',monotonic=float(stamp))
            for kind,stamp in [('after_open',1),('after_close',2)] for i,uuid in enumerate(self.uuids)]
        save(directory/'device_guard_events.json',events)
        status=dict(final_exit_code=0,exit_code=0,ended_at='done',started_at='2026-10-19T00:01:00+00:00',
                    worker_started=not preflight,driver_exit_clean=True,guard_all_handles_closed=True,
                    guard_workers_exited=True,device_guard_events_sha256=sha(directory/'device_guard_events.json'))
        manifest=dict(stage='graph',source_sha256='source',device_lifetime_mode=LIFETIME_MODE,
                      authorization_sha256='auth',device_guard_sha256='guard',preflight_only=preflight,
                      worker_gpu_uuids=self.uuids if preflight else self.uuids[:1],
                      guard_gpu_uuids=self.uuids,driver=self.driver)
        save(directory/'status.json',status);save(directory/'manifest.json',manifest)
        save(directory/'driver_postexit.json',self.driver)
        return directory,status,manifest

    def verify_guard(self,directory,preflight):
        verify_support_guard(directory,preflight=preflight,tested=self.uuids if preflight else self.uuids[:1],
                             uuids=self.uuids,auth=self.auth,auth_sha='auth',source_sha='source')

    def test_preflight_and_real_graph_are_distinct_acceptances(self):
        for preflight in (True,False):
            directory,status,manifest=self.support_fixture(preflight)
            self.verify_guard(directory,preflight)
            with self.assertRaises(RuntimeError):self.verify_guard(directory,not preflight)

    def test_any_unclean_support_lifecycle_rejects_expansion(self):
        directory,status,manifest=self.support_fixture(False)
        for key,value in [('guard_workers_exited',False),('guard_all_handles_closed',False),
                          ('driver_exit_clean',False),('exit_code',1),('final_exit_code',1),
                          ('stop_reason','driver changed'),('worker_started',False)]:
            save(directory/'status.json',dict(status,**{key:value}))
            with self.assertRaises(RuntimeError):self.verify_guard(directory,False)
        save(directory/'status.json',status)
        save(directory/'driver_postexit.json',dict(self.driver,anomaly_count=1))
        with self.assertRaises(RuntimeError):self.verify_guard(directory,False)

    def test_smoke_has_eight_tasks_and_both_previous_failure_tasks(self):
        tasks=['place_object_basket','place_container_plate']+[f'task{i}' for i in range(33)]
        result=smoke_tasks(tasks)
        self.assertEqual(len(result),8)
        self.assertTrue(set(tasks[:2])<=set(result))
        with self.assertRaises(RuntimeError):smoke_tasks(tasks,','.join(tasks[1:9]))
        with self.assertRaises(RuntimeError):smoke_tasks(tasks,','.join([tasks[0]]*8))

    def test_commands_cannot_smuggle_scaled_clock_or_probe_into_formal(self):
        args=SimpleNamespace(repo=self.root,run_root=self.root/'runs',config=self.root/'eval.yaml',
             checkpoint=self.root/'multirate_step_20000.pt',checkpoint_audit=self.root/'audit.json',
             timing_contract=self.root/'timing.json',frozen_scenes=self.root/'scenes',
             gpus=','.join(self.uuids),seed=100000,task_timeout_s=14400,cohort_timeout_s=3600,
             scene_compatibility=self.root/'compatibility.json')
        for stage,count in [('smoke','2'),('formal','20')]:
            output,command=suite_command(args,stage=stage,tasks=['POB','PCP'])
            self.assertIn('--joint-v3',command)
            self.assertEqual(command[command.index('--simulation-rate')+1],'1')
            self.assertEqual(command[command.index('--action-interval-s')+1],'0.16')
            self.assertEqual(command[command.index('--max-cohort-size')+1],'8')
            self.assertEqual(command[command.index('--episodes')+1],count)
            self.assertNotIn('--probe-action-limit',command)
            self.assertNotIn('--resume',command)
            self.assertEqual('--tasks' in command,stage=='smoke')
            self.assertTrue(output.is_relative_to(args.run_root))
            self.assertIn('--scene-compatibility',command)

    def test_path_escape_rejected(self):
        with self.assertRaises(RuntimeError):contained(self.root/'..'/'outside',self.root)


class StageAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.tasks=[f'task{i}' for i in range(8)]
        self.uuids=[f'GPU-{i}' for i in range(8)]
        self.hashes=dict(checkpoint_sha256='checkpoint',config_sha256='config',timing_contract_sha256='timing',
                         scene_compatibility_sha256='scenes')
        self.spec=dict(protocol=PROTOCOL,simulation_rate=1.,virtual_delay=False,prepare_only=False,tasks=self.tasks,
            episodes_per_task=2,probe_action_limit=None,max_cohort_size=8,solo_tasks=None,
            scheduling_mode='phased_episode_cohorts_v1',opencv_threads=1,gpus=self.uuids,
            action_interval_s=.16,history_frame_period_s=.16,max_backlog_s=.25,infer_clock_tolerance_s=.02,
            frozen_manifest_sha256='frozen',source_digest='source',**self.hashes)
        self.summary=dict(complete=True,valid_completed=16,planned_total=16)
        save(self.root/'suite_started.json',self.spec);save(self.root/'summary.json',self.summary)
        save(self.root/'suite_completed.json',dict(self.summary,all_processes_exited=True,
            finished=[dict(task=task,gpu=gpu,exit_code=0) for task,gpu in zip(self.tasks,self.uuids)]))
        for task in self.tasks:
            for episode in range(2):
                save(self.root/task/f'episode_{episode:04d}'/'report.json',dict(
                    protocol=PROTOCOL,simulation_rate=1.,clock='actual_wall_clock',
                    clock_valid=True,continuous_valid=True,realtime_valid=True,virtual_delay=False,
                    camera_equivalence=dict(passed=True),post_terminal_policy_drain=dict(slow_quiescent=True),
                    probe_action_limit=None,joint_v3=True,scene_compatibility_sha256='scenes'))

    def verify(self):
        module=SimpleNamespace(summarize=lambda output,spec:self.summary)
        with patch.dict(sys.modules,{'eval_domino_motionkv_rtc_suite':module}):
            return verify_stage(self.root,expected_tasks=self.tasks,episodes=2,uuids=self.uuids,
                                hashes=self.hashes,frozen_sha='frozen',previous_source='source')

    def test_complete_native_episode_receipts_are_required(self):
        self.assertEqual(self.verify()[1]['valid_completed'],16)
        for key,value in [('protocol','scaled'),('simulation_rate',.25),('probe_action_limit',64),
                          ('max_cohort_size',2),('action_interval_s',.2),('gpus',self.uuids[:1]),
                          ('source_digest','other'),('scene_compatibility_sha256','other')]:
            save(self.root/'suite_started.json',dict(self.spec,**{key:value}))
            with self.assertRaises(RuntimeError):self.verify()
        save(self.root/'suite_started.json',self.spec)

    def test_unfinished_background_slow_cannot_pass_stage(self):
        path=self.root/self.tasks[0]/'episode_0000'/'report.json'
        report=json.loads(path.read_text());report['post_terminal_policy_drain']['slow_quiescent']=False
        save(path,report)
        with self.assertRaisesRegex(RuntimeError,'quiescent'):self.verify()

    def test_saved_aggregate_must_equal_independent_recomputation(self):
        save(self.root/'summary.json',dict(self.summary,valid_completed=15))
        with self.assertRaisesRegex(RuntimeError,'aggregate'):self.verify()


class FinalLifecycleTests(unittest.TestCase):
    def test_owned_terminal_stage_stops_only_its_broker(self):
        calls=[];status={}
        close_kernel_broker(status,enabled=True,launched=True,socket_path='/run/owned.sock',
                            stopper=lambda path:calls.append(path) or {'ok':True,'stopping':True})
        self.assertEqual(calls,['/run/owned.sock']);self.assertTrue(status['kernel_broker_stop']['stopped'])

    def test_no_launch_or_pending_cleanup_preserves_broker(self):
        for enabled,launched,status in [(False,True,{}),(True,False,{}),
                (True,True,{'supervisor_cleanup_pending_pid':123})]:
            with self.subTest(enabled=enabled,launched=launched,status=status):
                close_kernel_broker(status,enabled=enabled,launched=launched,socket_path='/run/owned.sock',
                                    stopper=lambda path:self.fail('must not stop'))

    def test_graph_raw_failure_cannot_be_hidden_by_passed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'raw.json';save(path,dict(kind='motionkv_rtc_native_graph',passed=False))
            receipt=dict(passed=True,evidence_files=[dict(path=str(path),sha256=sha(path))])
            fake=SimpleNamespace(accept=lambda **kw:(_ for _ in ()).throw(RuntimeError('actual graph failed')))
            with patch.dict(sys.modules,{'accept_joint_deployment_graph':fake}):
                with self.assertRaisesRegex(RuntimeError,'actual graph failed'):
                    revalidate_graph_receipt(receipt,guard_dir=temporary,checkpoint=path,config=path)

    def test_graph_summary_must_match_reconstructed_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'raw.json';save(path,dict(kind='motionkv_rtc_native_graph',passed=True))
            receipt=dict(passed=True,evidence_files=[dict(path=str(path),sha256=sha(path))],created_at='old')
            fake=SimpleNamespace(accept=lambda **kw:dict(receipt,created_at='now'))
            with patch.dict(sys.modules,{'accept_joint_deployment_graph':fake}):
                revalidate_graph_receipt(receipt,guard_dir=temporary,checkpoint=path,config=path)
                bad=dict(receipt,passed=False)
                with self.assertRaisesRegex(RuntimeError,'summary differs'):
                    revalidate_graph_receipt(bad,guard_dir=temporary,checkpoint=path,config=path)


if __name__=='__main__':unittest.main()

