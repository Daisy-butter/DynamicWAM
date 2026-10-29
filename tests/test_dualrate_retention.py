"""Real suite/worker/post APIs over CPU fixtures; video codec alone is injected.

No fixture is a GPU or benchmark acceptance result.
"""
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
WORK=HERE.parent
sys.path[:0]=[str(HERE/'scripts'),str(HERE/'test_support'),str(WORK/'dualrate_continuous_repair_20261028/scripts')]
import dualrate_retention as keep
import dualrate_eval_suite as suite
import dualrate_eval_worker as worker
import dualrate_posttrain_eval as post
import guarded_dualrate_run as guard
import video_artifacts as video

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return path

PROBE=dict(codec='h264',pixel_format='yuv420p',width=2,height=2,fps=10.,frame_count=1)
class FixtureWriter:
    def __init__(self,path):self.path=path
    def write(self,frame):self.path.write_bytes(b'CPU codec fixture, not actual MP4')
    def close(self):pass

class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'wuruihan';self.root.mkdir()
        source_config=Path(os.environ.get('DUALRATE_RETENTION_TEST_FORMAL_CONFIG',
            str(WORK.parent/'outputs/dualrate_response_20261029/formal50_01.config.json')))
        self.old=json.loads(source_config.read_text());self.tasks=self.old['tasks']
        self.old_source=dict(source_sha256='old-source',scripts_sha256='old-scripts')
        self.new_source=dict(source_sha256='diagnostic-source',scripts_sha256='adapter-scripts')
        compat=dict(training_source_sha256='trained',same_model_files_sha256={'dualrate/runtime.py':'same'},
                    inference_only_files_sha256={'dualrate/online.py':'same'},camera_executor_change={'camera':'same'})
        cp=save(self.root/'compat.json',compat)
        self.old.update(output=str(self.root/'old'),prepared=str(self.root/'prepared'),
            source_compatibility_receipt=str(cp),source_compatibility_sha256=keep.sha(cp))
        for key in ('checkpoint','init_checkpoint','profile'):
            p=self.root/key;p.write_bytes(b'CPU identity fixture')
            self.old[key]=str(p);self.old[key+'_sha256']=keep.sha(p)
        self.entries=[]
        for task in self.tasks[:4]:self.prepared_task(task)
        frozen=save(Path(self.old['prepared'])/'frozen_manifest.json',dict(complete=True,episodes=self.entries))
        self.old['prepared_sha256']=keep.sha(frozen)
        save(Path(self.old['prepared'])/'suite_started.json',dict(tasks=self.tasks,seed=self.old['seed'],episodes_per_task=20))
        save(Path(self.old['prepared'])/'suite_completed.json',dict(complete=True))
        self.candidate=dict(self.old,stage='smoke',episodes=2,tasks=self.tasks[4:6],output=str(self.root/'smoke'),result_stamp='new')
        self.old_path=save(self.root/'old_config.json',self.old)
        self.candidate_path=save(self.root/'candidate.json',self.candidate)
        for task in self.tasks[:4]:self.task_output(self.old,task,self.old_source)
        save(Path(self.old['output'])/'suite_status.json',dict(state='failed',finished_tasks=self.tasks[:4]))
        save(Path(self.old['output'])/'suite_started.json',dict(source=self.old_source,config_sha256=keep.canonical(self.old)))
        self.guard=self.root/'old_guard';self.guard.mkdir()
        rows=[dict(event=e,uuid=u,path='/dev/nvidia'+str(i),fd=i+30) for e in ('after_open','after_close') for i,u in enumerate(self.old['gpu_uuids'])]
        for name,value in {'status.json':dict(state='failed',guard_closed=True),
            'owned_exit.json':dict(remaining=[],returncode=1),'kernel_before.json':{},
            'kernel_after.json':dict(unchanged=True,same_boot=True,added=[]),'source.json':self.old_source,
            'config.frozen.json':self.old,'authorization.json':dict(identity=dict(config_sha256=keep.canonical(self.old)),gpu_uuids=self.old['gpu_uuids']),
            'guard_events.json':rows}.items():save(self.guard/name,value)
        evidence=save(self.root/'reviewed_diff.json',dict(fixture=True,active_unchanged=True))
        self.proof=save(self.root/'proof.json',dict(schema='dualrate_retention_semantics_v1',old_source=self.old_source,
            new_source=self.new_source,active_policy_clock_camera_queue_unchanged=True,
            changes_limited_to_diagnostics_retention_supervision=True,gc_control_changed=False,evidence=[keep.binding(evidence)]))
        self.receipt_path=self.root/'retention.json'
        with patch.object(guard,'frozen_source',return_value=self.new_source),patch.object(worker,'validate_source_compatibility',return_value={}),patch.object(video,'probe_head_video',return_value=PROBE) as probe:
            self.receipt=keep.prepare(self.old_path,self.guard,self.candidate_path,self.proof,self.receipt_path)
            self.assertEqual(probe.call_count,80)
        self.formal=dict(self.old,output=str(self.root/'new'),result_stamp='new',retained_task_receipt=str(self.receipt_path),
            retained_task_receipt_sha256=keep.sha(self.receipt_path))
        self.smoke=dict(self.formal,stage='smoke',tasks=self.tasks[4:6],episodes=2,output=str(self.root/'smoke'))

    def prepared_task(self,task):
        for i in range(20):
            folder=Path(self.old['prepared'])/task/f'episode_{i:04d}'
            desc=dict(task=task,episode_index=i,requested_seed=self.old['seed']+i,seed=self.old['seed']+i)
            manifest=save(folder/'manifest.json',desc)
            motion=folder/'expert_motion.pkl';motion.write_bytes(pickle.dumps(dict(seed=desc['seed'],episode_info={'info':{}})))
            self.entries.append(dict(task=task,episode_index=i,seed=desc['seed'],manifest_sha256=keep.sha(manifest),motion_sha256=keep.sha(motion)))

    def task_output(self,config,task,source):
        base=Path(config['output'])/task
        native=Path(config['output'])/'eval_result'/task/'dualrate_dynamicwam'/config['task_config']/Path(config['checkpoint']).stem/config['result_stamp']
        native.mkdir(parents=True)
        evidence=dict(passed=True,checks={'immutable':True},activation_wall_time=100.,activation_physics_time=0.,final_wall_time=101.01,final_physics_time=1.)
        for i in range(20):
            folder=base/f'episode_{i:04d}';folder.mkdir(parents=True)
            desc=dict(task=task,episode_index=i,requested_seed=config['seed']+i,seed=config['seed']+i)
            save(folder/'manifest.json',desc)
            (folder/'expert_motion.pkl').write_bytes(pickle.dumps(dict(seed=desc['seed'],episode_info={'info':{}})))
            clip=video.BufferedHeadVideo();clip.append(np.zeros((2,2,3),dtype=np.uint8),physics_time_s=0.,capture_wall_perf_s=100.)
            vr=clip.finalize(native/f'episode{i}.mp4',episode_end_physics_s=1.,writer_factory=FixtureWriter,probe_fn=lambda p:dict(PROBE))
            report=dict(task=task,episode_index=i,seed=desc['seed'],success=False,complete=True,protocol=config['protocol'],
                config_sha256=keep.canonical(config),checkpoint_sha256=config['checkpoint_sha256'],prepared_sha256=config['prepared_sha256'],
                source_sha256=source['source_sha256'],realtime_valid=True,continuous_valid=True,video_required=True,
                executable_action_source='fast_only',fast_graph_acceptance=dict(enabled=True,passed=True,sealed=True,shape_count=2),
                raw_clock_acceptance=evidence,physics_step_seconds=.004,physics_steps=250,active_wall_seconds=1.01,physical_duration=1.,
                events=[dict(event='activation',wall_time=100.,physics_time=0.),dict(event='terminal_clock',**evidence)],**vr)
            save(folder/'report.json',report);save(folder/'metrics.json',dict(seed=desc['seed'],success=False))
        for name in ('_result.txt','_metrics.json','_metrics_report.txt','_episodes_detail.json'):(native/name).write_text('{}')
        save(base/'completed.json',dict(complete=True,task=task,episodes=20,protocol=config['protocol'],
            config_sha256=keep.canonical(config),native_result_dir=str(native)))
        save(base/'deployment_identity.json',dict(fixture=True))
        save(base/'slow_process.json',dict(bootstrap_exact_parity_passed=True,rng_roundtrip=True,closed=True,exitcode=0,failed=False,forced_owned_child_cleanup=False))

    def test_real_imported_pair_and_owner_apis(self):
        self.assertEqual(keep.validate(self.formal,self.new_source)['new_episodes'],620)
        post.validate_pair(self.smoke,self.formal)
        reports,metrics=suite.audit_task(self.formal,self.tasks[0])
        self.assertEqual(len(reports),20);self.assertEqual(reports[0]['config_sha256'],keep.canonical(self.old))
        self.assertIsNone(keep.retained_owner(self.formal,self.tasks[4]))
        self.assertEqual(keep.expected_smoke_tasks(self.formal),self.tasks[4:6])

    def test_reject_changed_policy_or_native_bytes(self):
        for key,value in [('max_backlog',.5),('checkpoint_sha256','other'),('prepared_sha256','other')]:
            with self.assertRaises(ValueError):keep.validate(dict(self.formal,**{key:value}),self.new_source)
        config=copy.deepcopy(self.formal);name=next(iter(config['deployment_artifact_sha256']));config['deployment_artifact_sha256'][name]='changed'
        with self.assertRaisesRegex(ValueError,'artifact bytes'):keep.validate(config,self.new_source)
        with self.assertRaisesRegex(ValueError,'frozen source'):keep.validate(self.formal,dict(source_sha256='other'))

    def test_tamper_or_partial_task_rejected(self):
        path=Path(self.old['output'])/self.tasks[0]/'episode_0000/metrics.json';path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'evidence changed'):keep.validate(self.formal,self.new_source)

    def test_full700_view_and_final_post_gate(self):
        Path(self.formal['output']).mkdir()
        try:retained,reports,metrics=keep.initialize_view(self.formal,self.new_source)
        except OSError as exc:
            if getattr(exc,'winerror',None)==1314:self.skipTest('Windows lacks symlink privilege; run this same test on Linux')
            raise
        self.assertEqual((len(retained),len(reports),len(metrics)),(4,80,80))
        for task in self.tasks[4:]:self.task_output(self.formal,task,self.new_source)
        summary=dict(complete=True,valid_completed=700,all_workers_exit_zero=True,every_video_fully_decoded=True,
            source=self.new_source,checkpoint_sha256=self.formal['checkpoint_sha256'],config_sha256=keep.canonical(self.formal),
            retention=keep.provenance(self.formal,self.new_source))
        save(Path(self.formal['output'])/'suite_completed.json',summary)
        with patch.object(video,'probe_head_video',return_value=PROBE) as probe:
            _,evidence,decoded=post.audit_suite(self.formal,self.new_source,700)
        self.assertEqual(len(decoded),700);self.assertEqual(probe.call_count,700)
        self.assertIn(self.receipt_path,evidence)
        self.assertEqual(summary['retention']['retained_episodes'],80)
        self.assertEqual(len(summary['retention']['origins']),700)

if __name__=='__main__':unittest.main()
