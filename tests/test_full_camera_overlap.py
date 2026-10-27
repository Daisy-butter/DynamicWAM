"""CPU-only strict parity and capture/physics ordering; no native imports."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_camera_overlap as camera


def observation(tick):
    return dict(observation={name:dict(rgb=np.full((2,3,3),tick%255,np.uint8),
        extrinsic=np.eye(4,dtype=np.float32)) for name in camera.CAMERAS},
        joint_action={'vector':np.array([float(tick),-0.])},pointcloud=[],endpose={})


class FakeCamera:
    def __init__(self,session,asynchronous):
        self.session=session;self.async_submit=asynchronous;self.pending=None;self.calls=[]
    def begin(self,env,source):
        if self.pending is not None: raise AssertionError('overlapping renderer updates')
        self.calls.append(('begin',env.tick))
        self.pending=SimpleNamespace(observation=observation(env.tick),camera_names=camera.CAMERAS,
            physics_time_s=source,capture_wall_perf_s=0.,render_submitted_wall_perf_s=2.,
            readback_finished_wall_perf_s=3.)
        return self.pending
    def finish(self,pending,timeout):
        assert pending is self.pending and timeout>0
        self.calls.append(('finish',self.session.env.tick));self.pending=None
        return pending


class CameraOverlapTests(unittest.TestCase):
    def test_camera_mode_requires_all_exact_probe_parameters(self):
        good=dict(diagnosis_mode=camera.MODE,camera_capture_ticks=list(camera.CAPTURE_TICKS),
                  camera_overlap_steps=45,camera_warmup_captures=3,tail_hold_rows=0)
        camera.validate_camera_config(good,dict(steps=2836))
        for change in ({'camera_overlap_steps':20},{'camera_capture_ticks':[0,2250]},
                       {'camera_warmup_captures':2},{'tail_hold_rows':1}):
            with self.assertRaises(ValueError): camera.validate_camera_config(dict(good,**change),dict(steps=2836))
        with self.assertRaises(ValueError): camera.validate_camera_config(good,dict(steps=2521))

    def test_metadata_preserves_dtype_signed_zero_and_all_float_bits(self):
        a=dict(x=np.array([-0.],np.float32),y=-0.)
        b=dict(x=np.array([0.],np.float32),y=0.)
        self.assertNotEqual(camera.typed_metadata(a),camera.typed_metadata(b))
        self.assertNotEqual(camera.typed_metadata(np.array([1.],np.float32)),
                            camera.typed_metadata(np.array([1.],np.float64)))
        with self.assertRaises(ValueError): camera.typed_metadata(np.array([object()]))

    def build_case(self,root,asynchronous):
        case=root/('async' if asynchronous else 'sync');case.mkdir()
        env=SimpleNamespace(tick=0);session=SimpleNamespace(env=env,physics_time=lambda:env.tick*.004000000189989805)
        device=FakeCamera(session,asynchronous)
        probe=camera.CameraExperiment(session,device,case,async_submit=asynchronous,deadline=float('inf'))
        probe.warmup()
        with patch.object(camera.time,'perf_counter',return_value=1.):
            for origin in camera.CAPTURE_TICKS:
                env.tick=origin;probe.before_tick(origin)
                for tick in range(origin+1,origin+46):
                    env.tick=tick;probe.after_tick(tick)
        report=probe.report()
        path=case/'state_trace.npz'
        np.savez_compressed(path,names=np.array(['qpos']),states=np.arange(2837,dtype=np.float64)[:,None],
                            physics_times=np.arange(2837)*.004000000189989805)
        report.update(complete=True,environment_closed=True,camera_closed=True,physics_steps=2836,state_trace_path=str(path))
        return report,device

    def test_sync_finishes_before_physics_async_finishes_after_45_ticks(self):
        with tempfile.TemporaryDirectory() as temp:
            sync,sync_camera=self.build_case(Path(temp),False)
            asynchronous,async_camera=self.build_case(Path(temp),True)
            self.assertEqual(sync_camera.calls[6:8],[('begin',0),('finish',0)])
            self.assertEqual(async_camera.calls[6:8],[('begin',0),('finish',45)])
            self.assertEqual(sync['captures'][0]['physics_ticks_before_submission_finish'],0)
            self.assertEqual(asynchronous['captures'][0]['physics_ticks_before_submission_finish'],45)
            result=camera.compare_cases(sync,asynchronous)
            self.assertTrue(result['passed']);self.assertTrue(result['physics_advanced_while_submitting'])
            self.assertEqual(result['state_comparison']['compared_ticks'],2837)

    def test_single_pixel_mismatch_is_failure_without_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            sync,_=self.build_case(Path(temp),False);asynchronous,_=self.build_case(Path(temp),True)
            path=Path(asynchronous['captures'][1]['rgb_path'])
            with np.load(path,allow_pickle=False) as stored: arrays={key:stored[key].copy() for key in stored.files}
            arrays['head_camera'][0,0,0]+=1;np.savez_compressed(path,**arrays)
            result=camera.compare_cases(sync,asynchronous)
            self.assertFalse(result['passed']);self.assertFalse(result['rgb_bitwise_equal'])

    def test_missing_submission_overlap_does_not_pass_readback_only(self):
        with tempfile.TemporaryDirectory() as temp:
            sync,_=self.build_case(Path(temp),False);asynchronous,_=self.build_case(Path(temp),True)
            for capture in asynchronous['captures']:
                capture['physics_ticks_before_submission_finish']=0
                capture['physics_ticks_before_native_finish']=45
            result=camera.compare_cases(sync,asynchronous)
            self.assertFalse(result['passed']);self.assertFalse(result['physics_advanced_while_submitting'])

    def test_metadata_or_physics_state_mismatch_fails_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            sync,_=self.build_case(Path(temp),False);asynchronous,_=self.build_case(Path(temp),True)
            metadata=Path(asynchronous['captures'][0]['metadata_path'])
            row=camera.read(metadata);row['observation']['joint_action']['vector']['bytes']='00';camera.save(metadata,row)
            self.assertFalse(camera.compare_cases(sync,asynchronous)['metadata_bitwise_equal'])
            path=Path(asynchronous['state_trace_path'])
            with np.load(path,allow_pickle=False) as stored: arrays={key:stored[key].copy() for key in stored.files}
            arrays['states'][2249,0]+=1e-12;np.savez_compressed(path,**arrays)
            self.assertFalse(camera.compare_cases(sync,asynchronous)['state_bitwise_equal'])


if __name__=='__main__': unittest.main()
