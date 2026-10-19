"""CPU-only negative/positive evidence validation; never fabricates GPU results."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import accept_joint_deployment_graph as converter


class JointGraphAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='graph-receipt-test-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.guard=self.root/'guard';self.guard.mkdir()
        self.checkpoint=self.root/'weights.pt';self.checkpoint.write_bytes(b'unit-test-placeholder-not-model')
        self.config=self.root/'deploy.yaml';self.config.write_text('unit_test: true\n')
        self.raw_path=self.root/'raw.json';self.auth_path=self.root/'authorization.json'
        self.source='b'*64;self.uuids=['GPU-test-'+str(i) for i in range(8)]
        self.driver=dict(host='test-host',boot_id='test-boot',anomaly_count=0,
            driver_anomalies_sha256='c'*64,latest_driver_anomaly_sha256=None)
        self.auth=dict(driver=self.driver,guard_sha256='d'*64,created_at='2026-10-19T00:00:00+00:00')
        self.write(self.auth_path,self.auth)
        events=[]
        for i,uuid in enumerate(self.uuids):
            for name,when in [('after_open',1.),('after_close',3.)]:
                events.append(dict(event=name,uuid=uuid,fd=10+i,path='/dev/nvidia'+str(i),monotonic=when))
        self.write(self.guard/'device_guard_events.json',events)
        self.manifest=dict(stage='graph',preflight_only=False,source_sha256=self.source,
            device_lifetime_mode='fds_before_telemetry_v1',authorization=str(self.auth_path),
            authorization_sha256=converter.sha(self.auth_path),device_guard_sha256=self.auth['guard_sha256'],
            worker_gpu_uuids=self.uuids[:1],guard_gpu_uuids=self.uuids,driver=self.driver,
            command=['python',str(SCRIPTS/'check_motionkv_graph.py'),'--checkpoint',str(self.checkpoint),
                '--deploy-config',str(self.config),'--output',str(self.raw_path),'--concurrent-slow'])
        self.status=dict(exit_code=0,final_exit_code=0,ended_at='2026-10-19T00:02:00+00:00',stop_reason=None,
            driver_exit_clean=True,guard_all_handles_closed=True,guard_workers_exited=True,worker_started=True,
            started_at='2026-10-19T00:01:00+00:00',device_guard_events_sha256=converter.sha(self.guard/'device_guard_events.json'))
        binding=lambda path:dict(path=str(path),bytes=path.stat().st_size,sha256=converter.sha(path))
        self.raw=dict(kind='motionkv_rtc_native_graph',passed=True,inputs_unchanged=True,
            conditioner_interface_version=3,trained_fast_steps=10,tested_fast_steps=10,
            engine_only_override=False,trained_prefix_max=4,tested_prefix=4,
            checkpoint=str(self.checkpoint),config=str(self.config),checkpoint_sha256=converter.sha(self.checkpoint),
            config_sha256=converter.sha(self.config),source_sha256=self.source,
            input_bindings=dict(checkpoint=binding(self.checkpoint),config=binding(self.config),source_sha256=self.source),
            cases=[dict(name=name,max_abs_error=0.) for name in sorted(converter.REQUIRED_CASES)],
            initial_eager_finite=True,initial_graph_finite=True,caller_inputs_unchanged=True,
            rejects_prefix_beyond_training_range=True,graph_stats=dict(enabled=True,captures=1,replays=20),
            concurrent_slow=dict(passed=True,fast_calls_before_slow_publication=1,worker_closed=True,
                producer_completion_event_ready=True,completed_cache_consumed=True,producer_stream_distinct=True,
                fast_captures_before=1,fast_captures_after=1,slow_started_monotonic=1.,slow_completed_monotonic=2.,
                calls=[dict(slow_unpublished_at_start=True,started_monotonic=1.5)]))
        self.write(self.guard/'driver_postexit.json',self.driver)

    def write(self,path,value):path.write_text(json.dumps(value))

    def invoke(self):
        self.write(self.raw_path,self.raw);self.write(self.guard/'manifest.json',self.manifest)
        self.write(self.guard/'status.json',self.status)
        with patch.object(converter,'current_source_sha',return_value=self.source):
            return converter.accept(raw_path=self.raw_path,guard_dir=self.guard,checkpoint=self.checkpoint,config=self.config)

    def test_complete_evidence_maps_to_pipeline_schema(self):
        receipt=self.invoke()
        self.assertTrue(receipt['passed']);self.assertEqual(receipt['graph_parity']['cases'],20)
        self.assertEqual(receipt['host'],self.driver['host']);self.assertEqual(receipt['gpu_uuids_tested'],self.uuids[:1])
        self.assertEqual(receipt['guard_gpu_uuids'],self.uuids)
        self.assertTrue(any(x['path']==str(self.raw_path) for x in receipt['evidence_files']))

    def test_raw_failure_override_missing_concurrency_and_nonfinite_are_rejected(self):
        original=copy.deepcopy(self.raw)
        for field,value in [('passed',False),('tested_fast_steps',4),('engine_only_override',True),
                ('concurrent_slow',{}),('inputs_unchanged',False),('error','probe failed')]:
            with self.subTest(field=field):
                self.raw=copy.deepcopy(original);self.raw[field]=value
                with self.assertRaises(RuntimeError):self.invoke()
        self.raw=copy.deepcopy(original);self.raw['cases'][0]['max_abs_error']=float('nan')
        with self.assertRaises(RuntimeError):self.invoke()

    def test_hash_binding_and_command_mismatches_are_rejected(self):
        original=copy.deepcopy(self.raw)
        self.raw['checkpoint_sha256']='e'*64
        with self.assertRaises(RuntimeError):self.invoke()
        self.raw=copy.deepcopy(original);self.raw['source_sha256']='e'*64
        with self.assertRaises(RuntimeError):self.invoke()
        self.raw=copy.deepcopy(original);self.manifest['command']+=['--fast-steps-override','10']
        with self.assertRaises(RuntimeError):self.invoke()
        self.manifest['command']=self.manifest['command'][:-2];self.config.write_text('changed: true')
        with self.assertRaises(RuntimeError):self.invoke()

    def test_guard_cleanup_and_postexit_identity_fail_closed(self):
        original=copy.deepcopy(self.status)
        for field,value in [('final_exit_code',1),('guard_workers_exited',False),
                ('guard_all_handles_closed',False),('driver_exit_clean',False)]:
            self.status=copy.deepcopy(original);self.status[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.invoke()
        self.status=original
        self.write(self.guard/'driver_postexit.json',dict(self.driver,anomaly_count=1))
        with self.assertRaises(RuntimeError):self.invoke()

    def test_missing_dynamic_input_coverage_and_false_overlap_are_rejected(self):
        case=self.raw['cases'].pop()
        with self.assertRaises(RuntimeError):self.invoke()
        self.raw['cases'].append(case)
        self.raw['concurrent_slow']['calls'][0]['started_monotonic']=2.5
        with self.assertRaises(RuntimeError):self.invoke()

if __name__=='__main__':unittest.main()
