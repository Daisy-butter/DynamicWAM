"""Synthetic schema fixtures only; never accepted as real training readiness."""
import importlib.util,json,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.runtime.domino_continuous.development_replay import DevelopmentReplayRecorder
from dynamicwam.training.data.real_rollout_dataset import adapt_replay_sample,RealRolloutReplayDataset,validate_replay_manifest
from dynamicwam.training.data.physical_grid_dataset import physical_grid_collate_fn
from dynamicwam.training.data.physical_grid import sha256_file
from dynamicwam.training.data.replay_language import match_replay_instruction
CATALOG=dict(identity='synthetic_only',groups={'dev_task':[dict(instruction='different instruction',prompt='prefix different instruction',lang_group_id=0,language_index=0,lang_id=0),dict(instruction='development task',prompt='prefix development task',lang_group_id=0,language_index=1,lang_id=1)]})

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('replay_prepare_test',ROOT/'scripts/prepare_real_rollout_replay.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)

class ReplayRoundtripTest(unittest.TestCase):
    def fixture(self,parent):
        raw=parent/'raw';scene=dict(task='dev_task',seed=42,development_scene_id='dev42',manifest_sha256='c'*64,motion_sha256='d'*64)
        recorder=DevelopmentReplayRecorder(raw,scope=dict(formal_exclusion_verified=True),scene=scene,
            trajectory_id='synthetic-schema-only',episode_id='episode0',policy_checkpoint_sha256='a'*64,policy_source_sha256='b'*64)
        hold=np.zeros(14,dtype=np.float32);recorder.begin(0.,hold)
        image=np.zeros((360,320,3),dtype=np.uint8)
        for t in sorted(set(np.arange(0,5.01,.16).tolist()+[1.])):recorder.record_frame(t,image,for_inference=True)
        for i in range(1250):recorder.record_step(i*.004,(i+1)*.004,None,None,hold)
        offsets=np.arange(1,17,dtype=np.float64)*.16
        def packet(capture,bootstrap):
            return dict(frame=image,fast_frame=image,state=hold,flow_frames=np.zeros((4,64,64,3),np.uint8),
                motion_features=np.zeros((4,12),np.float32),motion_interval_valid_mask=np.zeros(4,bool),
                motion_acceleration_valid_mask=np.zeros(4,bool),_rtc_prefix_actions=np.zeros((16,14),np.float32),
                _rtc_previous_actions=np.zeros((16,14),np.float32),_rtc_previous_valid=np.ones(16,bool),
                _rtc_action_target_offsets=offsets,sim_time=capture,instruction='development task',
                _rtc_prefix_length=0 if bootstrap else 4,_rtc_observation_action_index=0,_rtc_bootstrap_only=bootstrap)
        # Recorder's time source is the actual history diagnostic, including invalid missing slots.
        for i,capture in enumerate((0.,1.)):
            p=packet(capture,i==0)
            recorder.record_query(i,p,dispatch_wall_s=10+capture,dispatch_physics_s=capture,previous_publication_physics_s=0.,motion_history=dict(endpoint_capture_times_s=[None]*6))
            recorder.query_arrays[i]['motion_history_end_times']=np.full(4,np.nan)
            recorder.query_arrays[i]['motion_history_start_times']=np.full(4,np.nan)
            recorder.record_result(dict(query_index=i,receive_wall_s=10.8 if i==0 else 11.1,
                receive_monotonic=1000010.8 if i==0 else 1000011.1,
                receive_physics_s=.8 if i==0 else 1.1,installed=True,policy_diagnostics=dict(asynchronous_world=dict(
                    source_sim_time=0.,source_action_index=0,source_published_monotonic=1000010.8,
                    source_consumed_monotonic=1000010.8 if i==0 else 1000011.02))),np.zeros((16,14),np.float32))
        recorder.finalize(success=True,complete=True)
        def synthetic_encoder(kind,first,others):return torch.zeros((48,2,24,20) if kind=='condition' else (48,2,12,10))
        manifest,samples,drops=prep.build_replay_samples(raw,episode_index=0,language_catalog=CATALOG,encoder=synthetic_encoder)
        self.assertEqual(len(samples),1);self.assertEqual(drops,{})
        row,blob=samples[0];row['subset']='train'
        normalizer=RoboTwinQposNormalizer(mean=torch.arange(14).float(),std=torch.full((14,),2.),epsilon=1e-6)
        return row,blob,normalizer,raw

    def test_raw_recorder_adapter_collate_actual_training_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));sample=adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
            batch=physical_grid_collate_fn([sample])
            self.assertEqual(batch['delta_actions'].item(),0);self.assertEqual(batch['delta_sim_seconds'].item(),1.)
            self.assertEqual(batch['async_timing']['rtc_prefix_lengths'].item(),4)
            self.assertEqual(batch['source_rtc_prefix_lengths'].item(),0)
            self.assertEqual(blob['receive_monotonic'],1000011.1)
            self.assertGreater(blob['source_consumed_monotonic'],1000000.)
            self.assertEqual(blob['clock_schema'],'host_monotonic_and_episode_elapsed_v1')
            self.assertEqual(batch['source_rtc_prefix_actions'].abs().sum().item(),0.)
            self.assertFalse(batch['async_timing']['cache_publication_time_is_exact'].item())
            self.assertNotIn('cache_publication_time',batch['async_timing'])
            torch.testing.assert_close(batch['rtc_prefix_actions'][0,:4],norm.normalize(torch.zeros(4,14)))
            self.assertEqual(tuple(batch['joint_targets']['source_video_actual_times'].shape),(1,8))
            self.assertEqual(batch['bootstrap_mode'],['prime_then_hold'])
            # Dataset __getitem__ consumes the same fully validated raw->prepared artifact.
            sample_path=Path(d)/'prepared.pt';torch.save(blob,sample_path)
            row['sample']=dict(path=sample_path.name,sha256=sha256_file(sample_path))
            dataset=RealRolloutReplayDataset.__new__(RealRolloutReplayDataset)
            dataset.root=Path(d);dataset._rows=[row];dataset.normalizer=norm;dataset.max_supported_cache_age_actions=31
            class Packed:
                def _load_language_embedding(self,group,index):
                    assert group==0 and index==1
                    return torch.full((1,512),float(index))
            dataset.packed=Packed();loaded=dataset[0]
            torch.testing.assert_close(loaded['rtc_prefix_actions'],sample['rtc_prefix_actions'])
            self.assertEqual(row['language_index'],1);self.assertEqual(loaded['text_embedding'].mean().item(),1.)
            batch=physical_grid_collate_fn([loaded])
            if os.environ.get('REPLAY_FIXTURE_OUTPUT'):
                destination=Path(os.environ['REPLAY_FIXTURE_OUTPUT']);destination.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(kind='synthetic_schema_fixture_NOT_real_training_ready',batch=batch),destination)

    def test_unknown_actual_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'actual-instruction T5 cache'):
            match_replay_instruction(CATALOG,'dev_task','not in the language bank')
    def test_source_current_instruction_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['source_observation']['instruction']='different instruction'
            with self.assertRaisesRegex(ValueError,'actual instruction differs'):
                adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
    def test_failed_trace_cannot_be_self_imitation(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));m=json.loads((raw/'manifest.json').read_text());m['success']=False;m['action_target_source']='unavailable_failed_trajectory';(raw/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'failed/incomplete'):prep.build_replay_samples(raw,episode_index=0,language_catalog=CATALOG)
    def test_counterfactual_observation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['current_observation']['origin']='expert_observation'
            with self.assertRaisesRegex(ValueError,'counterfactual'):adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
    def test_bootstrap_requires_actual_hold(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['executed_timeline']['physics_tick_raw_commands'][50,0]=1
            with self.assertRaisesRegex(ValueError,'hold command'):adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
    def test_wall_publication_order_is_not_fabricated_physics_time(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['source_published_monotonic']=1000012.
            with self.assertRaisesRegex(ValueError,'publication bounds'):adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
    def test_mixed_relative_receipt_and_legacy_blob_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['receive_monotonic']=11.1
            with self.assertRaisesRegex(ValueError,'monotonic ordering'):
                adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
            blob.pop('clock_schema')
            with self.assertRaisesRegex(ValueError,'recollect and reconvert'):
                adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))

    def test_unexecuted_policy_proposal_is_not_a_target(self):
        with tempfile.TemporaryDirectory() as d:
            row,blob,norm,raw=self.fixture(Path(d));blob['current_targets']['raw_actions'][0,0]=2.
            with self.assertRaisesRegex(ValueError,'not the actual executed'):adapt_replay_sample(blob,row,normalizer=norm,language=torch.zeros(1,512))
    def test_synthetic_cache_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text(json.dumps(dict(kind='dynamicwam_prepared_real_replay_v1',complete=True,synthetic=True)))
            # The physical manifest exists only to reach the real-vs-synthetic gate.
            grid=root/'grid';grid.mkdir();(grid/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'synthetic'):validate_replay_manifest(root,grid,root)

if __name__=='__main__':unittest.main()
