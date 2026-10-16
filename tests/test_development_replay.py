"""Raw development replay uses observed trajectories; no GPU or expert labels."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_motionkv_rtc_controller import Clock,Session,Worker
from test_async_motionkv_policy import FakePolicy,value
from dynamicwam.runtime.domino_continuous.continuous_rtc import run_continuous_rtc_episode
from dynamicwam.runtime.domino_continuous.development_replay import (
    DevelopmentReplayRecorder,load_development_replay,validate_development_scope,select_development_scene,sha256)


def digest(value):return hashlib.sha256(str(value).encode()).hexdigest()

def scope_fixture(directory):
    formal=dict(schema_version=1,kind="formal700_scene_exclusion",scenes=[dict(task=f"t{i//20}",seed=i,
        scene_sha256=digest(f"scene{i}"),motion_sha256=digest(f"motion{i}")) for i in range(700)])
    scene=dict(task="development",seed=9000,development_scene_id="development:9000",
        manifest_sha256=digest("devscene"),motion_sha256=digest("devmotion"))
    allow=dict(schema_version=1,purpose="development_replay",scenes=[scene])
    formal_path=directory/"formal.json";formal_path.write_text(json.dumps(formal))
    allow_path=directory/"allow.json";allow_path.write_text(json.dumps(allow))
    return validate_development_scope(allow_path,formal_path),scene,allow_path,formal_path


class ReplayTests(unittest.TestCase):
    def test_formal_identity_or_any_hash_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scope,scene,allow,formal=scope_fixture(root)
            self.assertEqual(select_development_scene(scope,task="development",seed=9000,
                manifest_sha256=scene["manifest_sha256"],motion_sha256=scene["motion_sha256"]),scene)
            for changed in (dict(task="t0",seed=0),dict(motion_sha256=digest("motion1")),dict(manifest_sha256=digest("scene1"))):
                row=dict(scene,**changed);allow.write_text(json.dumps(dict(schema_version=1,purpose="development_replay",scenes=[row])))
                with self.assertRaisesRegex(ValueError,"overlaps"):
                    validate_development_scope(allow,formal)

    def collect(self, root, *, success=True):
        scope,scene,*_=scope_fixture(root)
        recorder=DevelopmentReplayRecorder(root/"trace",scope=scope,scene=scene,trajectory_id="trajectory",
            episode_id="development:9000",policy_checkpoint_sha256=digest("policy"),policy_source_sha256=digest("code"))
        clock=Clock();policy=FakePolicy(value)
        class ObservedSession(Session):
            def __init__(self):
                super().__init__(clock,interval_s=.16,terminal_tick=450 if success else None)
                self.recorded_times=[]
            def current_drive_targets(self):return np.zeros(14)
            def history_diagnostics(self):
                # These are selected actual fixture capture times, not generated
                # evenly-spaced timestamps for observations that never existed.
                observed=self.recorded_times[-6:]
                return {"endpoint_capture_times_s":[None]*(6-len(observed))+observed}
            def finish_capture(self):
                capture_time=self.pending[0]
                result=super().finish_capture()
                image=np.full((6,4,3),len(self.recorded_times)%255,np.uint8)
                self.recorded_times.append(capture_time)
                recorder.record_frame(capture_time,image,for_inference=result is not None)
                if result is not None:
                    result.update(state=self.q.copy(),instruction="test",frame=image,flow_frames=np.zeros((4,2,2,3),np.uint8),
                        motion_features=np.zeros((4,12),np.float32),motion_interval_valid_mask=np.zeros(4,bool),
                        motion_acceleration_valid_mask=np.zeros(4,bool))
                return result
        class PolicyWorker(Worker):
            def poll(self):
                result=super().poll()
                if result is not None:
                    result["actions"]=policy.sample(self.calls[-1])
                    result["policy_timing"]=deepcopy(policy.last_policy_timing)
                return result
        session=ObservedSession();worker=PolicyWorker(clock,durations=(1.1,.05))
        try:
            report=run_continuous_rtc_episode(session,worker,task="development",episode_seed=9000,action_limit=24,
                action_interval_s=.16,frame_period_s=.16,joint_v3=True,clock=clock,sleep=clock.sleep,replay_recorder=recorder)
            recorder.finalize(success=report["success"],complete=True)
        finally:policy.close()
        return root/"trace",report

    def test_actual_controller_records_bootstrap_hold_and_true_source_packet(self):
        with tempfile.TemporaryDirectory() as temp:
            path,report=self.collect(Path(temp))
            manifest,queries=load_development_replay(path,require_success=True)
            self.assertTrue(report["success"])
            self.assertEqual(manifest["action_target_source"],"successful_self_imitation")
            self.assertEqual(manifest['schema'],'dynamicwam_development_replay_v2')
            self.assertIn('plan_switch_diagnostics.json',manifest['files'])
            for q in queries:
                if q.get('result_received'):
                    source=q['policy_diagnostics']['asynchronous_world']
                    self.assertLessEqual(source['source_consumed_monotonic'],q['receive_monotonic'])
                    self.assertGreaterEqual(q['receive_monotonic'],0.)
            first,second=queries[:2]
            self.assertEqual(first["completed_action_count"],second["completed_action_count"])
            self.assertEqual(second["source_query_index"],0)
            self.assertGreater(second["capture_time"]-first["capture_time"],1.)
            with np.load(path/second["path"],allow_pickle=False) as query:
                np.testing.assert_array_equal(query["_rtc_prefix_actions"],np.zeros((4,14)))
                self.assertEqual(query["motion_history_end_times"].shape,(4,))
            with np.load(path/"frames.npz",allow_pickle=False) as frames:
                self.assertGreater(np.count_nonzero(frames["capture_times"] < second["capture_time"]),3)
                self.assertTrue(np.isclose(frames["capture_times"],.32).any())
            with np.load(path/"executed.npz",allow_pickle=False) as executed:
                self.assertTrue(executed["held"].any())
                self.assertTrue(np.all(np.diff(executed["end_times"])>0))
                np.testing.assert_equal(executed["commands"][executed["held"]],0.)
            with np.load(path/"targets.npz",allow_pickle=False) as targets:
                self.assertEqual(len(targets["commands"]),report["completed_targets"])
                self.assertLess(targets["endpoint_times"][-1],report["physics_time_s"]+1e-8)

    def test_failed_rollout_is_saved_but_never_self_imitation(self):
        with tempfile.TemporaryDirectory() as temp:
            path,report=self.collect(Path(temp),success=False)
            self.assertFalse(report["success"])
            manifest,_=load_development_replay(path)
            self.assertEqual(manifest["action_target_source"],"unavailable_failed_trajectory")
            with self.assertRaisesRegex(ValueError,"behavior-cloning"):
                load_development_replay(path,require_success=True)

    def test_missing_source_is_not_replaced_with_current_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            path,_=self.collect(Path(temp))
            queries=json.loads((path/"queries.json").read_text())
            queries[1]["source_capture_time"]=.12345
            (path/"queries.json").write_text(json.dumps(queries))
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["files"]["queries.json"]=sha256(path/"queries.json")
            (path/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"source observation missing"):
                load_development_replay(path)

    def test_legacy_clock_schema_requires_recollection(self):
        with tempfile.TemporaryDirectory() as temp:
            path,_=self.collect(Path(temp))
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['schema']='dynamicwam_development_replay_v1'
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'recollect.*never infer'):
                load_development_replay(path)

    def test_mixed_relative_and_absolute_clock_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path,_=self.collect(Path(temp))
            queries=json.loads((path/'queries.json').read_text())
            queries[1]['receive_monotonic']=queries[1]['ready_wall_s']
            (path/'queries.json').write_text(json.dumps(queries))
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['files']['queries.json']=sha256(path/'queries.json')
            (path/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'do not mix relative'):
                load_development_replay(path)

    def test_trace_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path,_=self.collect(Path(temp))
            with (path/"targets.npz").open("ab") as handle:handle.write(b"changed")
            with self.assertRaisesRegex(ValueError,"hash"):
                load_development_replay(path)


if __name__=="__main__":unittest.main()
