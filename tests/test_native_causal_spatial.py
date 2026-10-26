"""CPU protocol regressions; these do not claim GPU/model acceptance."""
import sys
import unittest
import importlib.util
import io
import ast
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.inference.native_causal_spatial_policy import (
    MODEL_OFFSETS, NATIVE_PROTOCOL, NativeRTCPrefixQueue, NativeCausalSpatialPolicy, validate_native_packet,
)


class NativeQueueTests(unittest.TestCase):
    def test_bootstrap_and_shifted_two_row_prefix_and_soft_overlap(self):
        initial = np.arange(14, dtype=np.float64)/100
        queue = NativeRTCPrefixQueue(initial)
        prefix, previous, valid = queue.query()
        np.testing.assert_array_equal(prefix, np.stack([initial]*2))
        self.assertFalse(valid.any())
        plan = np.arange(16*14, dtype=np.float64).reshape(16, 14)/1000
        plan[:2] = prefix
        queue.install(plan)
        for origin in (0, 1):
            np.testing.assert_array_equal(queue.current(), initial)
            queue.acknowledge(origin+1)
        prefix, previous, valid = queue.query()
        np.testing.assert_array_equal(prefix, plan[2:4])
        np.testing.assert_array_equal(np.flatnonzero(valid), [2, 3])
        np.testing.assert_array_equal(previous[2:4], plan[4:6])
        self.assertTrue((previous[~valid] == 0).all())
        new_plan = np.full((16,14), 9., dtype=np.float64)
        new_plan[:2] = prefix
        queue.install(new_plan)
        np.testing.assert_array_equal(queue.current(), plan[2])
        self.assertEqual(set(queue.rows), set(range(2, 18)))

    def test_cannot_skip_requery_execute_inflight_or_mutate_prefix(self):
        queue = NativeRTCPrefixQueue(np.zeros(14))
        with self.assertRaises(RuntimeError): queue.current()
        prefix, _, _ = queue.query()
        with self.assertRaises(RuntimeError): queue.current()
        with self.assertRaises(RuntimeError): queue.query()
        with self.assertRaises(ValueError): queue.install(np.ones((16,14)))
        plan = np.zeros((16,14)); queue.install(plan)
        with self.assertRaises(RuntimeError): queue.acknowledge(2)
        queue.acknowledge(1)
        with self.assertRaises(RuntimeError): queue.query()
        queue.acknowledge(2)
        with self.assertRaises(RuntimeError): queue.current()

    def test_prefix_ownership_and_fp32_roundoff(self):
        queue = NativeRTCPrefixQueue(np.full(14, .123456789))
        prefix, _, _ = queue.query()
        plan = np.zeros((16,14)); plan[:2] = prefix.astype(np.float32)
        queue.install(plan)
        np.testing.assert_array_equal(queue.current(), prefix[0])
        plan[0] = -1
        np.testing.assert_array_equal(queue.current(), prefix[0])


class NativePacketTests(unittest.TestCase):
    def packet(self, origin=2, physics=7.812):
        return dict(_native_protocol=NATIVE_PROTOCOL, _native_action_origin=origin,
            _native_prefix_length=2, _native_prefix_actions=np.ones((2,14)),
            _native_model_target_offsets=MODEL_OFFSETS.copy(),
            _native_previous_actions=np.arange(224).reshape(16,14), _native_previous_valid=np.ones(16,bool),
            sim_time=physics, _rtc_capture_drive_targets=np.arange(14),
            _rtc_capture_wall_perf_s=20., _rtc_capture_available_wall_perf_s=20.1)

    def test_actual_physics_not_relabelled_to_nominal_queue_clock(self):
        packet = self.packet(origin=6, physics=7.812)
        origin, prefix, offsets, previous, valid = validate_native_packet(packet)
        self.assertEqual(origin, 6); self.assertEqual(packet["sim_time"], 7.812)
        np.testing.assert_array_equal(offsets, MODEL_OFFSETS)
        np.testing.assert_array_equal(np.flatnonzero(valid), [2,3])
        np.testing.assert_array_equal(previous[2:4], packet["_native_previous_actions"][2:4])
        self.assertTrue((previous[~valid] == 0).all())

    def test_bootstrap_masks_all_previous_and_rejects_d0_or_continuous(self):
        packet = self.packet(origin=0)
        self.assertFalse(validate_native_packet(packet)[-1].any())
        for changes in ({"_native_prefix_length":0}, {"_native_action_origin":1},
                        {"_native_protocol":"continuous_world_actual_latency_causal_spatial_h2r2_v1"},
                        {"_native_model_target_offsets":MODEL_OFFSETS+.004}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_native_packet(dict(packet, **changes))

    def test_history_receives_actual_capture_and_drive_targets(self):
        policy = object.__new__(NativeCausalSpatialPolicy)
        policy._native_last_physics_s = None
        seen = []
        policy._observe = lambda packet: seen.append(packet["sim_time"]) or packet["sim_time"]
        self.assertEqual(policy.observe_native(self.packet(0, 7.812)), 7.812)
        self.assertEqual(policy.observe_native(self.packet(2, 8.392)), 8.392)
        self.assertEqual(seen, [7.812,8.392])
        with self.assertRaises(ValueError): policy.observe_native(self.packet(4,8.392))

    def test_expired_memory_waits_and_refreshes_current_source_without_clamping_age(self):
        policy = object.__new__(NativeCausalSpatialPolicy)
        policy._future = None; policy._slow_status = lambda: None
        policy.contract = SimpleNamespace(max_memory_age_s=4.96)
        policy.memories = SimpleNamespace(current=SimpleNamespace(source=SimpleNamespace(observation_physics_s=0.)))
        calls = []
        def submit(packet, observation, origin):
            calls.append((packet["sim_time"], origin))
            def complete(timeout):
                policy.memories.current = SimpleNamespace(source=SimpleNamespace(observation_physics_s=packet["sim_time"]))
            policy._future = SimpleNamespace(result=complete)
            return True
        policy._submit_memory = submit
        policy.memories.snapshot = lambda **kwargs: (policy.memories.current, kwargs)
        memory, waits = policy._native_memory(self.packet(16, 8.392), object(), 16)
        self.assertEqual(calls, [(8.392,16)])
        self.assertEqual(waits[0]["reason"], "native_topp_memory_age_refresh")
        self.assertEqual(memory[1]["observation_physics_s"], 8.392)

    def test_sampling_reuses_fp32_denormalization_and_keeps_exact_prefix(self):
        policy = object.__new__(NativeCausalSpatialPolicy)
        policy._text = policy.vae = object(); policy.device = "cpu-fixture"
        policy._source_origin = 0; policy._future = None
        policy._slow_status = lambda: None
        policy.observe_native = lambda packet: SimpleNamespace(spatial_times=np.array([0.,1.,2.,7.812]))
        memory = SimpleNamespace(payload=object(), source=SimpleNamespace(event_id=3, observation_physics_s=7.),
                                 publication=2, completed_wall_s=1.)
        policy._native_memory = lambda *args: (memory, [])
        policy._model_batch = lambda *args: {"validated_batch":True}
        policy._sync_stream = lambda: None
        events = []
        class Tensor:
            def float(self): events.append("float32"); return self
            def squeeze(self, dim): return self
            def cpu(self): return self
            def numpy(self): return np.zeros((16,14), np.float32)
        def denormalize(value):
            self.assertEqual(events, ["float32"])
            events.append("denormalize")
            return value
        policy.normalizer = SimpleNamespace(denormalize=denormalize)
        policy._graph = SimpleNamespace(replay=lambda *args: {"actions":Tensor()}, captures=1, replays=1)
        policy._submit_memory = lambda *args: events.append("refresh_after_fast") or True
        fake_torch = SimpleNamespace(Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed:None),
            randn=lambda *args, **kwargs:None, inference_mode=nullcontext, autocast=lambda *args, **kwargs:nullcontext(),
            bfloat16=object())
        packet = self.packet(16); packet["_sampling_seed"] = 4
        packet["_native_prefix_actions"][:] = .123456789123
        with patch.dict(sys.modules, {"torch":fake_torch}):
            actions = policy.sample(packet)
        np.testing.assert_array_equal(actions[:2], packet["_native_prefix_actions"])
        self.assertEqual(events, ["float32","denormalize","refresh_after_fast"])
        self.assertEqual(policy.last_policy_timing["soft_plan_valid_rows"], 2)


class NativeAdapterTests(unittest.TestCase):
    def fixture(self, *, advance_during_inference=False):
        # Load the real adapter with only its image conversion boundary stubbed.
        # No OpenCV, Torch, SAPIEN or GPU is imported by this CPU fixture.
        name = "dynamicwam.runtime.ciwam.adapters.domino._native_adapter_cpu_fixture"
        path = Path(__file__).resolve().parents[1] / "src/dynamicwam/runtime/ciwam/adapters/domino/deploy_policy_native_causal_spatial.py"
        spec = importlib.util.spec_from_file_location(name, path)
        adapter = importlib.util.module_from_spec(spec)
        composite = SimpleNamespace(composite_robotwin_frame=lambda obs, config:obs["frame"],
            extract_state=lambda obs:obs["state"], head_camera_frame=lambda obs:obs["frame"])
        with patch.dict(sys.modules, {"dynamicwam.runtime.ciwam.adapters.domino.composite":composite}):
            spec.loader.exec_module(adapter)
        env = SimpleNamespace(take_action_cnt=0, physics=2.3, test_num=0)
        env._scene_step_clock = SimpleNamespace(snapshot=lambda:SimpleNamespace(time_seconds=env.physics))
        executed, queries, observed = [], [], []
        def take_action(action, action_type):
            self.assertEqual(action_type, "qpos")
            executed.append(action.copy())
            env.physics += [.4,.012,.8,.1,.2,.3][env.take_action_cnt]
            env.take_action_cnt += 1
        env.take_action = take_action
        motion = SimpleNamespace(flow_rgb=np.zeros((4,1,1,3),np.uint8), motion_features=np.zeros((4,12)),
            interval_valid_mask=np.zeros(4,bool), acceleration_valid_mask=np.zeros(4,bool))
        policy = SimpleNamespace(config={"observation":{}}, last_policy_timing={})
        def sample(packet):
            queries.append(packet)
            policy.last_policy_timing = {"memory_waits":[]}
            actions = np.full((16,14), float(packet["_native_action_origin"]+1))
            actions[:2] = packet["_native_prefix_actions"]
            if advance_during_inference: env.physics += .004
            return actions
        policy.sample = sample
        policy.observe_native = lambda packet: observed.append(packet["sim_time"])
        model = SimpleNamespace(bind_env=lambda *args:None, queue=NativeRTCPrefixQueue(np.zeros(14)),
            policy=policy, flow=SimpleNamespace(push=lambda *args, **kwargs:None, observation=lambda:motion, diagnostics={}),
            stepper=adapter.NativeStepper(env, action_interval_seconds=.16), sampling_seed=1,
            query_ms=[], memory_wait_count=0)
        return adapter, model, env, executed, queries, observed

    def test_one_take_action_per_callback_and_queries_every_two_native_actions(self):
        adapter, model, env, executed, queries, observed = self.fixture()
        for i in range(6):
            with redirect_stdout(io.StringIO()):
                adapter.eval(env, model, {"frame":np.zeros((1,1,3),np.uint8), "state":np.zeros(14)})
            self.assertEqual(env.take_action_cnt, i+1)
        self.assertEqual([p["_native_action_origin"] for p in queries], [0,2,4])
        np.testing.assert_allclose([p["sim_time"] for p in queries], [2.3,2.712,3.612])
        self.assertEqual(len(observed), 3)
        self.assertTrue(np.array_equal(executed[0], np.zeros(14)))
        self.assertTrue(np.array_equal(executed[1], np.zeros(14)))
        self.assertTrue(np.array_equal(executed[2], np.ones(14)))
        self.assertTrue(np.array_equal(executed[3], np.ones(14)))
        self.assertTrue(np.array_equal(executed[4], np.full(14,3.)))
        self.assertEqual(len(model.stepper.telemetry.records), 6)

    def test_inference_cannot_advance_physics(self):
        adapter, model, env, executed, _, _ = self.fixture(advance_during_inference=True)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError,"advanced simulator"):
            adapter.eval(env, model, {"frame":np.zeros((1,1,3),np.uint8), "state":np.zeros(14)})
        self.assertEqual(executed, [])
        self.assertEqual(env.take_action_cnt, 0)

    def test_reset_flushes_native_telemetry_and_rebinds_new_initial_commands(self):
        adapter, model, env, _, _, _ = self.fixture()
        model.stepper = None; model.queue = None
        prepared = []; reset = []
        model.policy.prepare_instruction = lambda instruction:prepared.append(instruction)
        model.policy.reset_native_episode = lambda:reset.append("policy") or {"slow_quiescent":True}
        model.flow.reset = lambda:reset.append("flow")
        env.get_instruction = lambda:"first instruction"
        adapter.NativeCausalSpatialDeployModel.bind_env(model, env, {"state":np.zeros(14)})
        model.stepper.execute(np.zeros(14))
        captured = io.StringIO()
        with redirect_stdout(captured):
            adapter.NativeCausalSpatialDeployModel.finish_episode(model)
        line = next(x for x in captured.getvalue().splitlines() if x.startswith("SYNC-EPISODE-SUMMARY "))
        summary = ast.literal_eval(line.removeprefix("SYNC-EPISODE-SUMMARY "))
        self.assertEqual(summary["steps"], 1); self.assertEqual(summary["kinds"], {"action":1})
        self.assertIn("SYNC-QUERY-TIMING ", captured.getvalue())
        self.assertEqual(reset, ["policy","flow"])
        self.assertIsNone(model.queue); self.assertIsNone(model.stepper)
        second = SimpleNamespace(take_action_cnt=0, get_instruction=lambda:"second instruction")
        adapter.NativeCausalSpatialDeployModel.bind_env(model, second, {"state":np.ones(14)})
        np.testing.assert_array_equal(model.queue.query()[0], np.ones((2,14)))
        self.assertEqual(prepared, ["first instruction","second instruction"])


if __name__ == "__main__":
    unittest.main()
