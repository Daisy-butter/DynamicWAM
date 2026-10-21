"""CPU serving packet contract; real model/DOMINO acceptance remains separate."""
from pathlib import Path
import sys
import json
import importlib.util
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.inference.causal_spatial_policy import (
    validate_packet, capture_metadata, CausalSpatialPolicy, EXPECTED_RUNTIME_CONTRACT)
from dynamicwam.inference.causal_spatial_config import read_deployment


class PolicyPacketTests(unittest.TestCase):
    def packet(self, bootstrap=False):
        return dict(_rtc_contract="continuous_world_actual_latency_causal_spatial_h2r2_v1",
            _rtc_bootstrap_only=bootstrap, _rtc_observation_action_index=0,
            _rtc_prefix_length=0 if bootstrap else 2,
            _rtc_prefix_actions=np.ones((0 if bootstrap else 2, 14), dtype=np.float64) * .123456789123,
            _rtc_action_target_offsets=np.arange(1, 17, dtype=np.float64) * .16,
            _rtc_previous_actions=np.ones((16, 14), np.float32),
            _rtc_previous_valid=np.ones(16, bool), _rtc_action_interval_s=.16, sim_time=2.,
            _rtc_physics_timestep_s=.004, _rtc_observation_physics_s=2.,
            _rtc_action_grid_start_physics_s=None if bootstrap else 2.,
            _rtc_action_origin_physics_s=2.,
            _rtc_action_target_times=2. + np.arange(1, 17, dtype=np.float64) * .16,
            _rtc_capture_drive_targets=np.arange(14, dtype=np.float64),
            _rtc_capture_wall_perf_s=10., _rtc_capture_available_wall_perf_s=10.3)

    def test_same_h2_prefix_and_only_editable_two_row_context(self):
        packet = self.packet()
        bootstrap, origin, prefix, offsets, previous, valid = validate_packet(packet)
        self.assertFalse(bootstrap)
        np.testing.assert_array_equal(prefix, packet["_rtc_prefix_actions"])
        self.assertEqual(prefix.dtype, np.float64)
        np.testing.assert_array_equal(np.flatnonzero(valid), [2, 3])
        self.assertTrue(np.all(previous[~valid] == 0))
        np.testing.assert_array_equal(offsets[[7, 15]], [1.28, 2.56])

    def test_bootstrap_has_no_proposed_action_context(self):
        bootstrap, _, prefix, _, previous, valid = validate_packet(self.packet(True))
        self.assertTrue(bootstrap)
        self.assertEqual(prefix.shape, (0, 14))
        self.assertFalse(valid.any())
        self.assertFalse(previous.any())

    def test_h4_and_offgrid_observations_rejected(self):
        for key, value in (("_rtc_prefix_length", 4), ("_rtc_observation_action_index", 3),
                           ("_rtc_action_interval_s", .25), ("_rtc_contract", "legacy")):
            packet = self.packet(); packet[key] = value
            with self.assertRaises(ValueError): validate_packet(packet)
        packet = self.packet(); packet["_rtc_action_target_offsets"] += .01
        with self.assertRaises(ValueError): validate_packet(packet)

    def phase_packet(self, origin):
        """Exact observed domino_joint02 epoch and native float32 timestep."""
        packet = self.packet()
        grid = .25600001215934753
        dt = float(np.float32(.004))
        capture = grid + origin * 40 * dt
        anchor = grid + origin * .16
        targets = anchor + np.arange(1, 17, dtype=np.float64) * .16
        packet.update(_rtc_observation_action_index=origin, sim_time=capture,
            _rtc_physics_timestep_s=dt, _rtc_observation_physics_s=capture,
            _rtc_action_grid_start_physics_s=grid, _rtc_action_origin_physics_s=anchor,
            _rtc_action_target_times=targets, _rtc_action_target_offsets=targets - capture)
        return packet

    def test_real_native_tick_phase_is_preserved_without_moving_endpoints(self):
        for origin in (2, 100, 1000, 10000):
            packet = self.phase_packet(origin)
            original = packet["_rtc_action_target_offsets"].copy()
            _, _, _, offsets, _, _ = validate_packet(packet)
            np.testing.assert_array_equal(offsets, original)
            np.testing.assert_allclose(offsets + packet["sim_time"],
                packet["_rtc_action_target_times"], atol=1e-12, rtol=0)
        failure = self.phase_packet(2)
        self.assertEqual(failure["sim_time"], .576000027358532)
        self.assertAlmostEqual(failure["_rtc_action_origin_physics_s"] - failure["sim_time"],
                               -1.5199184359997457e-8, delta=1e-16)
        self.assertFalse(np.allclose(failure["_rtc_action_target_offsets"],
                                    np.arange(1, 17) * .16, atol=1e-8, rtol=0))

    def test_real_tick_delay_and_changed_absolute_grid_still_rejected(self):
        packet = self.phase_packet(2)
        packet["sim_time"] += packet["_rtc_physics_timestep_s"]
        packet["_rtc_observation_physics_s"] = packet["sim_time"]
        packet["_rtc_action_target_offsets"] = packet["_rtc_action_target_times"] - packet["sim_time"]
        with self.assertRaises(ValueError): validate_packet(packet)
        packet = self.phase_packet(2)
        packet["_rtc_action_target_times"] += .004
        packet["_rtc_action_target_offsets"] += .004
        with self.assertRaises(ValueError): validate_packet(packet)
        packet = self.phase_packet(2)
        packet["_rtc_action_target_offsets"] = np.arange(1, 17) * .16
        with self.assertRaises(ValueError): validate_packet(packet)
        packet = self.phase_packet(2)
        packet["_rtc_action_interval_s"] += 1e-7
        with self.assertRaises(ValueError): validate_packet(packet)

    def test_invalid_forecast_tail_never_becomes_context(self):
        packet = self.packet()
        packet["_rtc_previous_valid"][:] = False
        packet["_rtc_previous_actions"][:] = np.nan
        _, _, _, _, previous, valid = validate_packet(packet)
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(previous).all())
        packet["_rtc_previous_valid"][3] = True
        with self.assertRaises(ValueError): validate_packet(packet)

    def test_original_capture_commands_and_availability_are_preserved(self):
        packet = self.packet()
        packet.update(frame=np.zeros((4, 4, 3), np.uint8), state=np.full(14, -100.))
        policy = object.__new__(CausalSpatialPolicy)
        policy._event = 0
        policy.window = SimpleNamespace(append_composite=lambda frame, **kwargs: kwargs)
        observed = policy._observe(packet)
        np.testing.assert_array_equal(observed["capture_drive_targets"], packet["_rtc_capture_drive_targets"])
        self.assertEqual(observed["available_wall_s"], 10.3)
        self.assertEqual(observed["observation_physics_s"], 2.)
        self.assertEqual(observed["event_id"], 1)

    def test_missing_or_noncausal_capture_metadata_is_rejected(self):
        packet = self.packet(); del packet["_rtc_capture_drive_targets"]
        with self.assertRaises(KeyError): capture_metadata(packet)
        packet = self.packet(); packet["_rtc_capture_available_wall_perf_s"] = 9.
        with self.assertRaises(ValueError): capture_metadata(packet)
        packet = self.packet(); packet["_rtc_capture_drive_targets"][0] = np.nan
        with self.assertRaises(ValueError): capture_metadata(packet)

    def test_training_recipe_runtime_contract_is_exact(self):
        config = Path(__file__).resolve().parents[1] / "configs/causal_spatial/interface_adapt_smoke.yaml"
        self.assertEqual(json.loads(config.read_text())["runtime_contract"], EXPECTED_RUNTIME_CONTRACT)

    def test_deployment_wrapper_separates_native_assets_from_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        deploy, native = read_deployment(root / "configs/causal_spatial/deploy.yaml")
        self.assertEqual(native.name, "motionkv_stage3_20k_eval_20261019.yaml")
        self.assertEqual(deploy["runtime_contract"], EXPECTED_RUNTIME_CONTRACT)
        with self.assertRaises(ValueError): read_deployment(root / "configs/causal_spatial/interface_adapt_smoke.yaml")

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
                         "actual CPU serving geometry requires torch and OpenCV")
    def test_actual_warmup_and_live_window_have_identical_tensor_geometry(self):
        from dynamicwam.runtime.causal_spatial_contract import FiniteObservationWindow
        policy = object.__new__(CausalSpatialPolicy)
        policy._event = 0
        policy.device = "cpu"
        policy.window = FiniteObservationWindow("shape-audit")
        policy.normalizer = SimpleNamespace(normalize=lambda value: value)
        memory = SimpleNamespace(source=SimpleNamespace(observation_physics_s=0.))
        packet = self.packet()
        packet.update(frame=np.zeros((360, 320, 3), np.uint8),
            motion_features=np.zeros((4, 12), np.float32),
            motion_interval_valid_mask=np.zeros(4, bool), motion_acceleration_valid_mask=np.zeros(4, bool))
        expected = {"initial_state": (1, 14), "absolute_motion_features": (1, 4, 12),
            "spatial_rgb": (1, 4, 3, 3, 160, 160), "spatial_flow": (1, 4, 3, 2, 160, 160),
            "spatial_valid": (1, 4, 3, 1, 160, 160), "spatial_dt": (1, 4),
            "recorded_capture_commands": (1, 4, 14), "command_valid": (1, 4),
            "previous_plan": (1, 16, 14), "previous_plan_valid": (1, 16),
            "future_target_offsets": (1, 2), "action_target_offsets": (1, 16)}
        for index in range(4):
            packet["sim_time"] = .32 * index
            packet["_rtc_capture_wall_perf_s"] = 10 + .32 * index
            packet["_rtc_capture_available_wall_perf_s"] = 10.1 + .32 * index
            observation = policy._observe(packet)
            batch = policy._model_batch(packet, observation, memory,
                packet["_rtc_prefix_actions"], packet["_rtc_action_target_offsets"],
                packet["_rtc_previous_actions"], packet["_rtc_previous_valid"])
            for name, shape in expected.items(): self.assertEqual(tuple(batch[name].shape), shape)
            self.assertEqual(tuple(batch["rtc_prefix"].actions.shape), (1, 16, 14))
            self.assertEqual(batch["rtc_prefix"].lengths.tolist(), [2])
            np.testing.assert_array_equal(batch["initial_state"][0].numpy(), packet["_rtc_capture_drive_targets"])


if __name__ == "__main__": unittest.main()
