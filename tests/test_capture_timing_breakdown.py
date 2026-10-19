"""CPU-only capture service attribution; no renderer or CUDA imports."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/dynamicwam/runtime/domino_continuous'))
from camera_stream import completed_capture_timing


class CaptureTimingTests(unittest.TestCase):
    def capture(self):
        return SimpleNamespace(capture_wall_perf_s=10., render_submitted_wall_perf_s=10.12,
            readback_started_wall_perf_s=10.25, readback_finished_wall_perf_s=10.29,
            finished_wall_perf_s=10.36, camera_readback_seconds={'head_camera':.009},
            observation={'frame':np.arange(6,dtype=np.uint8).reshape(1,2,3)}, physics_time_s=4.2)

    def timing(self,captured=None,**changes):
        values=dict(owner_started=10.35,owner_consumed=10.361,video_done=10.362,
                    history_done=10.363,owner_finished=10.368);values.update(changes)
        return completed_capture_timing(captured or self.capture(),**values)

    def test_pre_reader_and_post_reader_delays_close_total_wall_time(self):
        result=self.timing()
        for name,value in [('render_submit_ms',120.),('render_to_readback_start_ms',130.),
                ('native_readback_ms',40.),('readback_finished_to_owner_start_ms',60.),
                ('readback_finished_to_owner_consumed_ms',71.),('owner_consume_ms',11.),
                ('video_append_ms',1.),('history_push_ms',1.),('packet_and_flow_ms',5.),
                ('completed_capture_wall_ms',368.),('accounted_capture_wall_ms',368.),
                ('owner_finish_ms',18.),('unaccounted_capture_wall_ms',0.)]:
            self.assertAlmostEqual(result[name],value,places=8,msg=name)
        self.assertEqual(result['owner_wait_for_readback_ms'],0.)
        self.assertEqual(result['timestamps_s']['readback_started'],10.25)

    def test_blocking_owner_finish_is_not_negative_service_delay(self):
        result=self.timing(owner_started=10.26)
        self.assertAlmostEqual(result['owner_wait_for_readback_ms'],30.)
        self.assertEqual(result['readback_finished_to_owner_start_ms'],0.)
        self.assertAlmostEqual(result['accounted_capture_wall_ms'],368.)

    def test_diagnostics_preserve_frame_identity_pixels_and_physical_timestamp(self):
        captured=self.capture();observation=captured.observation;pixels=observation['frame'];before=pixels.copy()
        result=self.timing(captured)
        self.assertIs(captured.observation,observation);self.assertIs(captured.observation['frame'],pixels)
        np.testing.assert_array_equal(pixels,before);self.assertEqual(captured.physics_time_s,4.2)
        result['camera_readback_seconds']['head_camera']=99.
        self.assertEqual(captured.camera_readback_seconds['head_camera'],.009)

    def test_nonfinite_or_backwards_timestamps_are_rejected(self):
        for field,value in [('readback_started_wall_perf_s',9.),('readback_finished_wall_perf_s',float('nan')),
                            ('finished_wall_perf_s',10.28)]:
            captured=self.capture();setattr(captured,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):self.timing(captured)
        with self.assertRaises(ValueError):self.timing(owner_started=10.37)


if __name__=='__main__':unittest.main()
