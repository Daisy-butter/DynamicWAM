from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import full_baseline_resume as gate
from full_baseline_worker import save, sha


class CameraParityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        root = self.root; output = root/'parity'; prepared = root/'prepared'; guard = root/'guard'
        manifest = prepared/'place_empty_cup/episode_0012/manifest.json'
        save(manifest, dict(task='place_empty_cup', episode_index=12, seed=100014))
        motion = manifest.with_name('expert_motion.pkl'); motion.write_bytes(b'actual-motion')
        save(prepared/'frozen_manifest.json', {'complete': True})
        trace_manifest = root/'formal06/place_empty_cup/episode_0012/manifest.json'
        save(trace_manifest, dict(gate.read(manifest),eval_instruction='Pick the cup',
                                 prompt_source='episode_complete_template',passive_force_dedup=True))
        trace = trace_manifest.with_name('report.json')
        save(trace, dict(task='place_empty_cup', episode_index=12, physics_steps=2836))
        cup = dict(task='place_empty_cup', episode_index=12, seed=100014,
                   manifest_sha256=sha(manifest), motion_sha256=sha(motion))
        required = [dict(task='place_object_stand', episode_index=7, seed=100007,
                         manifest_sha256='stand', motion_sha256='motion'), cup]
        source = {key: 'current-'+key for key in ('source_sha256','scripts_sha256','guard_sha256',
            'supervisor_sha256','distributed_supervisor_sha256')}
        camera_sources = []
        for role in ('libsapien','libsvulkan2','rt_camera_shader','native_bridge'):
            file = root/role; file.write_bytes(role.encode())
            camera_sources.append(dict(role=role,path=str(file.resolve()),sha256=sha(file)))
        artifacts = {row['path']: row['sha256'] for row in camera_sources if row['role']=='native_bridge'}
        common = dict(prepared=str(prepared), prepared_sha256=sha(prepared/'frozen_manifest.json'),
            benchmark_contract={'grid':.18}, passive_force_dedup=True, deployment_artifact_sha256=artifacts)
        diagnostic = dict(common,run_kind='full_physics_replay',diagnosis_mode='camera_overlap_v1',
            output=str(output),cpu_workers=[0],gpu_uuids=['GPU-0'],camera_capture_ticks=[0,2250,2475],
            camera_overlap_steps=45,camera_warmup_captures=3,trace_report=str(trace),trace_report_sha256=sha(trace),
            trace_manifest_sha256=sha(trace_manifest))
        save(guard/'status.json',dict(state='completed',guard_closed=True,errors=[]))
        save(guard/'owned_exit.json',dict(remaining=[],returncode=0))
        save(guard/'kernel_after.json',dict(unchanged=True,same_boot=True,added=[]))
        save(guard/'source.json',source); save(guard/'config.frozen.json',diagnostic)
        save(guard/'authorization.json',dict(mode='full_physics_replay',source=source,
            identity={'config_sha256':gate.canonical_hash(diagnostic)},gpu_uuids=['GPU-0']))
        save(guard/'authorization.json.consumed',{'receipt_sha256':sha(guard/'authorization.json')})
        save(guard/'guard_events.json',[dict(event=event,uuid=f'GPU-{i}',path=f'/dev/nvidia{i}',fd=i+10)
            for event in ('after_open','after_close') for i in range(8)])
        names = ['left_camera','right_camera','head_camera','front_camera']; ticks=[0,2250,2475]
        files = [trace,trace_manifest,manifest,motion,prepared/'frozen_manifest.json',*[Path(row['path']) for row in camera_sources]]
        children = []
        for name in ('sync','async'):
            directory = output/('camera_'+name); directory.mkdir(parents=True)
            state = directory/'state_trace.npz'; state.write_bytes(b'full-state-trace'); files.append(state)
            captures=[]
            for index,tick in enumerate(ticks):
                metadata=directory/f'capture_{index:04d}.json'; save(metadata,dict(tick=tick))
                rgb=directory/f'capture_{index:04d}.npz'; rgb.write_bytes(b'four-full-RGB-arrays')
                captures.append(dict(tick=tick,physics_ticks_before_submission_finish=3 if name=='async' else 0,
                    physics_ticks_before_native_finish=5,metadata_path=str(metadata),metadata_sha256=sha(metadata),
                    rgb_path=str(rgb),rgb_sha256=sha(rgb)))
                files.extend([metadata,rgb])
            report=directory/'worker_report.json'
            save(report,dict(camera_case=name,async_submit=name=='async',complete=True,environment_closed=True,
                camera_names=names,warmup_captures=3,capture_ticks=ticks,overlap_steps=45,captures=captures,
                camera_source_evidence=camera_sources))
            children.append(dict(case=name,report_path=str(report),sha256=sha(report))); files.append(report)
        flags = ('passed','complete','rgb_bitwise_equal','metadata_bitwise_equal','exposure_times_equal',
                 'state_bitwise_equal','physics_advanced_while_pending','physics_advanced_while_submitting',
                 'environment_closed','all_children_exit_zero')
        proof = dict({key:True for key in flags},schema='full_camera_overlap_parity_v1',**cup,**source,
            source_contract=source,config_sha256=gate.canonical_hash(diagnostic),benchmark_contract=common['benchmark_contract'],
            prepared_sha256=common['prepared_sha256'],camera_names=names,capture_ticks=ticks,warmup_captures=3,
            overlap_steps=45,trace_report_sha256=sha(trace),trace_manifest_sha256=sha(trace_manifest),
            physics_steps=2836,state_count=2837,children=children,camera_source_evidence=camera_sources,
            evidence=[dict(path=str(file),sha256=sha(file)) for file in files])
        path=output/'camera_overlap.json'; save(path,proof)
        config=dict(common,camera_async_submit=True,physx_cpu_workers=0,gpu_uuids=[f'GPU-{i}' for i in range(8)],
            acceptance_smoke_episodes_per_task=13,camera_async_parity_receipt=str(path),
            camera_async_parity_receipt_sha256=sha(path),camera_async_parity_guard=str(guard))
        return config,proof,source,required

    def validate(self,config,source,required):
        with patch.object(gate,'expanded_acceptance_contract',return_value={'required_acceptance_cases':required}):
            return gate.validate_camera_async_submit(config,source)

    def replace_proof(self,config,proof):
        save(config['camera_async_parity_receipt'],proof)
        config['camera_async_parity_receipt_sha256']=sha(config['camera_async_parity_receipt'])

    def test_exact_gpu_proof_with_both_children_and_closed_guard(self):
        config,proof,source,required=self.fixture()
        self.assertEqual(self.validate(config,source,required)['proof']['state_count'],2837)
        self.assertIsNone(gate.validate_camera_async_submit({}))
        with self.assertRaises(ValueError): gate.validate_camera_async_submit({'camera_async_submit':1})

    def test_nonexact_missing_overlap_short_trace_and_wrong_seed_rejected(self):
        config,proof,source,required=self.fixture()
        for change in ({'rgb_bitwise_equal':False},{'metadata_bitwise_equal':False},{'state_bitwise_equal':False},
                       {'physics_advanced_while_submitting':False},{'state_count':2836},{'seed':100012},
                       {'scripts_sha256':'old'},{'source_contract':{}},{'camera_source_evidence':[]}):
            with self.subTest(change=change):
                self.replace_proof(config,dict(proof,**change))
                with self.assertRaises(ValueError): self.validate(config,source,required)

    def test_empty_or_mutated_artifact_evidence_rejected(self):
        config,proof,source,required=self.fixture()
        self.replace_proof(config,dict(proof,evidence=[]))
        with self.assertRaises(ValueError): self.validate(config,source,required)
        self.replace_proof(config,proof)
        Path(proof['children'][1]['report_path']).with_name('capture_0002.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'): self.validate(config,source,required)

    def test_readback_only_overlap_does_not_pass(self):
        config,proof,source,required=self.fixture()
        child=proof['children'][1]; report=Path(child['report_path']); data=gate.read(report)
        for capture in data['captures']: capture['physics_ticks_before_submission_finish']=0
        save(report,data); child['sha256']=sha(report)
        for evidence in proof['evidence']:
            if Path(evidence['path'])==report: evidence['sha256']=sha(report)
        self.replace_proof(config,proof)
        with self.assertRaisesRegex(ValueError,'readback-only'): self.validate(config,source,required)

    def test_augmented_eval_manifest_must_preserve_every_frozen_field(self):
        config,proof,source,required=self.fixture()
        directory=Path(config['camera_async_parity_guard']); diagnostic=gate.read(directory/'config.frozen.json')
        manifest=Path(diagnostic['trace_report']).with_name('manifest.json')
        save(manifest,dict(gate.read(manifest),seed=100012))
        diagnostic['trace_manifest_sha256']=sha(manifest)
        save(directory/'config.frozen.json',diagnostic)
        auth=gate.read(directory/'authorization.json'); auth['identity']['config_sha256']=gate.canonical_hash(diagnostic)
        save(directory/'authorization.json',auth)
        save(directory/'authorization.json.consumed',{'receipt_sha256':sha(directory/'authorization.json')})
        proof.update(config_sha256=gate.canonical_hash(diagnostic),trace_manifest_sha256=sha(manifest))
        for evidence in proof['evidence']:
            if Path(evidence['path'])==manifest: evidence['sha256']=sha(manifest)
        self.replace_proof(config,proof)
        with self.assertRaisesRegex(ValueError,'frozen scene descriptor'):
            self.validate(config,source,required)

    def test_failed_guard_missing_case_and_short_smoke_rejected(self):
        config,proof,source,required=self.fixture()
        with self.assertRaisesRegex(ValueError,'episode12'):
            self.validate(dict(config,acceptance_smoke_episodes_per_task=10),source,required)
        with self.assertRaisesRegex(ValueError,'both exact'):
            self.validate(config,source,required[:1])
        save(Path(config['camera_async_parity_guard'])/'kernel_after.json',dict(unchanged=False,same_boot=True,added=['x']))
        with self.assertRaisesRegex(ValueError,'cleanly closed'): self.validate(config,source,required)


if __name__ == '__main__': unittest.main()
