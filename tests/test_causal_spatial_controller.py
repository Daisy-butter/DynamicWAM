"""Behavioral clock tests for the isolated H2/R2 continuous episode driver.

Fake model/camera durations exercise scheduling only, not GPU/model acceptance.
"""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
from test_motionkv_rtc_controller import Clock,Session,Worker
from dynamicwam.runtime.domino_continuous.wall_sync import RealtimeViolation
from dynamicwam.runtime.continuous_causal_spatial import (
    run_continuous_causal_spatial_episode,validate_causal_spatial_report,
)


class CausalSpatialControllerTests(unittest.TestCase):
    def run_case(self,*,actions=12,durations=(.05,),session_args=None,worker_args=None,controller_args=None):
        clock=Clock()
        session=Session(clock,interval_s=.16,**(session_args or {}))
        worker=Worker(clock,durations,**(worker_args or {}))
        report=run_continuous_causal_spatial_episode(session,worker,task="causal-spatial-cpu",
            episode_seed=37,action_limit=actions,clock=clock,sleep=clock.sleep,
            **(controller_args or {}))
        return session,worker,report

    def test_h2_queries_and_suffix_effects_use_exact_320ms_grid(self):
        session,worker,report=self.run_case(actions=20)
        self.assertEqual(report["completed_targets"],20)
        self.assertEqual(session.native_actions,20)
        self.assertEqual(report["execute_horizon"],2)
        self.assertEqual(report["rtc_prefix_max"],2)
        self.assertEqual([p["_rtc_prefix_length"] for p in worker.calls],[0]+[2]*(len(worker.calls)-1))
        origins=[p["_rtc_observation_action_index"] for p in worker.calls[1:]]
        self.assertEqual(origins,list(range(0,origins[-1]+1,2)))
        for request in report["requests"][1:]:
            self.assertAlmostEqual(request["observation_to_suffix_effect_s"],.32)
            self.assertLessEqual(request["install_physics_s"],request["suffix_not_before_physics_s"]+1e-9)
        self.assertTrue(report["realtime_valid"])
        self.assertAlmostEqual(report["simulation_rate"],1.)
        validate_causal_spatial_report(report)

    def test_slow_bootstrap_moves_world_under_native_hold_not_generated_plan(self):
        clock=Clock();session=Session(clock,interval_s=.16)
        held=np.full(14,.3);held[[6,13]]=0.
        session.current_drive_targets=lambda:held.copy()
        session.q[:]=.6
        worker=Worker(clock,durations=(.7,.05))
        report=run_continuous_causal_spatial_episode(session,worker,task="bootstrap",episode_seed=2,
            action_limit=8,clock=clock,sleep=clock.sleep)
        self.assertGreater(report["requests"][0]["inference_physics_advance_s"],.65)
        self.assertFalse(report["requests"][0]["bootstrap_actions_executed"])
        np.testing.assert_array_equal(worker.calls[1]["_rtc_prefix_actions"],np.repeat(held[None],2,axis=0))
        self.assertFalse(report["startup_delay_excluded"])

    def test_late_compute_cannot_extend_h2(self):
        with self.assertRaises(RealtimeViolation):self.run_case(durations=(.05,.4))

    def test_late_poll_cannot_backdate_result_using_paused_physics(self):
        with self.assertRaises(RealtimeViolation):
            self.run_case(durations=(.05,.27),worker_args={"poll_stall":.08})

    def test_committed_prefix_change_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"prefix"):
            self.run_case(worker_args={"mutate_prefix":True})

    def test_nonprimary_rate_and_horizon_are_rejected(self):
        for setting in ({"simulation_rate":.25},{"execute_horizon":4},{"prefix_max":4}):
            with self.assertRaises(ValueError):self.run_case(controller_args=setting)

    def test_blocking_camera_cost_cannot_move_observation_deadline(self):
        clock=Clock()
        class BlockingCamera(Session):
            def begin_capture(self,*,for_inference):
                super().begin_capture(for_inference=for_inference)
                if self.native_actions and for_inference:self.clock.sleep(.34)
        session=BlockingCamera(clock,interval_s=.16)
        worker=Worker(clock,durations=(.05,))
        with self.assertRaises(RealtimeViolation):
            run_continuous_causal_spatial_episode(session,worker,task="camera",episode_seed=2,
                action_limit=12,clock=clock,sleep=clock.sleep)

    def test_invalid_reports_cannot_claim_h2_or_valid_clock(self):
        _,_,report=self.run_case()
        for key,value in (("execute_horizon",4),("realtime_valid",False),("simulation_rate",.25)):
            invalid=copy.deepcopy(report);invalid[key]=value
            with self.assertRaises((ValueError,RuntimeError)):validate_causal_spatial_report(invalid)

    def test_partial_final_block_preserves_exact_native_action_budget(self):
        session,_,report=self.run_case(actions=9)
        self.assertEqual(report["completed_targets"],9)
        self.assertEqual(session.native_actions,9)
        self.assertEqual(report["blocks"][-1]["count"],1)

    def test_wrong_response_identity_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"out-of-order"):
            self.run_case(worker_args={"wrong_id":True})

    def test_only_installed_short_overlap_is_exposed_to_next_request(self):
        _,worker,_=self.run_case()
        for packet in worker.calls:
            expected=[] if packet["_rtc_queue_generation"]==0 else [2,3]
            self.assertEqual(np.flatnonzero(packet["_rtc_previous_valid"]).tolist(),expected)
            self.assertTrue(np.all(packet["_rtc_previous_actions"][~packet["_rtc_previous_valid"]]==0))

    def test_capture_metadata_keeps_exposure_and_actual_wall_availability_separate(self):
        _,worker,report=self.run_case()
        captures=[record for record in report["captures"] if record["for_inference"]]
        for packet,capture in zip(worker.calls,captures):
            self.assertAlmostEqual(packet["sim_time"],capture["physics_s"])
            self.assertAlmostEqual(packet["_rtc_capture_wall_perf_s"],100.+capture["wall_s"])
            self.assertGreater(packet["_rtc_capture_available_wall_perf_s"],packet["_rtc_capture_wall_perf_s"])
            np.testing.assert_array_equal(packet["_rtc_capture_drive_targets"],capture["capture_drive_targets"])


if __name__=="__main__":unittest.main()
