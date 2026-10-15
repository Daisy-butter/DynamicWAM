"""CPU checks for versioned runtime, graph ownership and actual RTC adapter wiring."""
from contextlib import nullcontext
from unittest.mock import patch
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
sys.path.insert(0,str(ROOT / "tests"))
from check_joint_motion_kv_conditioner import joint_fixture
from check_motion_kv_conditioner import fixture
from dynamicwam.models.joint_motion_kv_conditioner import JointMotionKVConditioner
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.models.rtc_prefix import PrefixCondition
from dynamicwam.action_normalization import RoboTwinQposNormalizer
from dynamicwam.inference.multirate_model_loader import build_fast_conditioner, install_immutable_source_backbone
from dynamicwam.inference.fast_runtime import validate_runtime_contract, RTC_TIMESTAMPED_CONTROLLER
from dynamicwam.inference.native_fast_graph import NativeFastGraph, _BRIDGE_CONTEXT_TENSORS
from dynamicwam.runtime.domino_continuous.continuous_rtc import run_continuous_rtc_episode, JOINT_V3_PROTOCOL, PROTOCOL
from eval_domino_motionkv_rtc import validate_rtc_report, checked_episode
from test_async_motionkv_policy import FakePolicy, value
from test_motionkv_rtc_controller import Clock, Session, Worker


def contract_fixture(version):
    model, *_ = fixture() if version == 1 else joint_fixture()
    if version == 3:
        model = JointMotionKVConditioner(**dict(model.constructor_spec,interface_version=3))
    native = dict(version=version,source_bridge="immutable_official" if version == 1 else "student_joint",
        world_refresh_actions=16,fast_steps=10,teacher={"sha256":"test-immutable"})
    if version == 3:native["slow_schedule"]="asynchronous_latest_completed_v1"
    schedule=BridgeSchedule.from_flow_match()
    norm=dict(stats=dict(mean=[0.]*14,std=[1.]*14),epsilon=1e-6)
    rtc=dict(version=1,mode="training_time_action_prefix",max_prefix_actions=4,clean_timestep=0.,
        loss="per_sample_suffix_mean",includes_zero_prefix=True,queue_clock="absolute_action_effect_times",
        overflow="reject_and_refresh",execute_suffix_actions=4)
    contract=dict(version=1,native_multirate=native,max_age_actions={1:12,2:15,3:31}[version],
        preprocess="three_view_160_imagenet_mask_v1",clock_mode="required",schedule=schedule.to_config(),
        execute_horizon=4,rtc=rtc)
    if version == 3:contract.update(max_age_seconds=4.96,previous_plan="soft_context_only; hard_prefix_is_irrevocable",
        source_retention="fixed_grid_source_video_action_fm",slow_source_prefix="clean_committed_v1",physical_grid="a"*64,
        physical_grid_contract=dict(schema_version=1,action_interval_seconds=.16,action_horizon=16,
            source_plan_alignment="physical_time_linear_arm_previous_hold_gripper",latent_manifest_sha256="b"*64))
    modules=dict(native_multirate=True,architecture="native",fast_conditioner=model.constructor_spec,
        fast_steps=10,fast_schedule=schedule.with_action_steps(10).to_config(),schedule=schedule.to_config(),
        runtime_contract=contract,action_normalization=norm,action_expert=dict(chunk_size=16))
    payload=dict(modules=modules,config=dict(runtime_contract=contract,action_normalization=norm))
    sc=schedule.to_config()
    runtime=SimpleNamespace(flow_match_shift=sc["shift"],flow_match_sigma_min=sc["sigma_min"],
        flow_match_extra_one_step=sc["extra_one_step"],num_inference_steps=sc["action_steps"],
        num_video_inference_steps=sc["video_steps"],chunk_size=16,action_interval_seconds=.16,
        action_normalizer=RoboTwinQposNormalizer.from_stats(norm["stats"],epsilon=norm["epsilon"]))
    return model,payload,runtime


class RuntimeContractTests(unittest.TestCase):
    def test_legacy_v1_v2_and_v3_are_explicit_and_load_strictly(self):
        for version in (1,2,3):
            with self.subTest(version=version):
                model,payload,runtime=contract_fixture(version)
                loaded=build_fast_conditioner(model.constructor_spec)
                loaded.load_state_dict(model.state_dict(),strict=True)
                validate_runtime_contract(payload,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)
                if version >= 2:
                    backbone=torch.nn.Linear(3,4)
                    before={k:v.clone() for k,v in backbone.state_dict().items()}
                    install_immutable_source_backbone(backbone,payload["config"]["runtime_contract"],{})
                    self.assertFalse(backbone.training)
                    self.assertTrue(all(torch.equal(v,before[k]) for k,v in backbone.state_dict().items()))

    def test_v3_rejects_untrained_age_and_synchronous_schedule(self):
        _,payload,runtime=contract_fixture(3)
        for field,value in (("max_age_actions",15),("max_age_seconds",5.0),("slow_schedule","synchronous"),("version",2),
                            ("physical_grid",None),("physical_grid_contract",{}),("previous_plan","hard_commit_all"),("source_retention","stale_target"),("slow_source_prefix",None)):
            bad=deepcopy(payload)
            contract=bad["config"]["runtime_contract"]
            target=contract["native_multirate"] if field in ("slow_schedule","version") else contract
            target[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                validate_runtime_contract(bad,runtime,rtc_controller=RTC_TIMESTAMPED_CONTROLLER)

    def test_source_prefix_uses_snapshot_normalization_and_explicit_d0(self):
        from dynamicwam.runtime.ciwam.wam.multirate_policy import normalize_source_rtc_prefix
        norm=RoboTwinQposNormalizer.from_stats(dict(mean=[1.]*14,std=[2.]*14),epsilon=1e-6)
        reference=torch.zeros(1,16,14)
        packet=dict(_rtc_prefix_length=2,_rtc_prefix_actions=np.full((2,14),3.,np.float32))
        actions,lengths=normalize_source_rtc_prefix(packet,norm,reference)
        torch.testing.assert_close(actions[:,:2],norm.normalize(torch.full((1,2,14),3.)))
        self.assertFalse(actions[:,2:].any())
        self.assertEqual(lengths.tolist(),[2])
        packet['_rtc_prefix_actions'][:]=100
        self.assertFalse(torch.equal(actions[:,:2],norm.normalize(torch.full((1,2,14),100.))))
        zero,lengths=normalize_source_rtc_prefix(dict(_rtc_prefix_length=0,_rtc_prefix_actions=np.empty((0,14))),norm,reference)
        self.assertFalse(zero.any());self.assertEqual(lengths.tolist(),[0])
        with self.assertRaises(ValueError):
            normalize_source_rtc_prefix(dict(_rtc_prefix_length=5,_rtc_prefix_actions=np.zeros((5,14))),norm,reference)

    def test_failed_producer_drains_owned_stream_before_error_escapes(self):
        from dynamicwam.runtime.ciwam.wam.multirate_policy import MultiratePolicy
        order=[]
        class Event:
            def record(self,stream):order.append("record")
            def synchronize(self):order.append("drain")
        def fail(packet):
            order.append("launch")
            raise ValueError("producer failed after enqueue")
        owner=SimpleNamespace(base=SimpleNamespace(runner=SimpleNamespace(device="cuda:0")),
            worker=None,_async_world=True,_producer_stream=object(),
            _build_bridge=fail,_complete_bridge_value=lambda x:x)
        # CUDA interfaces are replaced completely; this proves ordering without
        # querying, creating a context on, or allocating any actual GPU.
        with patch("torch.cuda.stream",return_value=nullcontext()),patch("torch.cuda.Event",Event):
            with self.assertRaisesRegex(ValueError,"after enqueue"):
                MultiratePolicy._build_bridge_completed(owner,{})
        self.assertEqual(order,["launch","record","drain"])

    def test_v3_graph_owns_and_refreshes_all_time_and_plan_inputs(self):
        _,current,bridge,text,_=joint_fixture()
        offsets=torch.arange(1,17,dtype=torch.float64)[None].expand(2,-1)*.16
        current=replace(current,previous_plan=torch.randn(2,16,14),previous_plan_valid=torch.ones(2,16,dtype=torch.bool),
            action_target_offsets=offsets,age=replace(current.age,delta_sim_seconds=torch.tensor([.16,.32],dtype=torch.float64)))
        bridge=replace(bridge,interface_version=3,source_action_target_offsets=offsets)
        core=SimpleNamespace(native_multirate=True,fast_schedule=BridgeSchedule.from_flow_match(),
            backbone=SimpleNamespace(action_attention_spec=SimpleNamespace(precision=torch.float32)))
        graph=NativeFastGraph(core,enabled=False)
        noise=torch.randn(2,16,14);prefix=PrefixCondition(current.committed_prefix,current.committed_prefix_lengths)
        obs,world=graph._canonical_context(current,bridge,noise,prefix)
        graph._allocate(obs,world,text,None,noise,prefix)
        for field in _BRIDGE_CONTEXT_TENSORS:
            source,dest=getattr(world,field),getattr(graph.bridge,field)
            if source is not None:
                self.assertTrue(torch.equal(source,dest),field)
                self.assertNotEqual(source.data_ptr(),dest.data_ptr(),field)
                self.assertEqual(dest.data_ptr(),getattr(graph.conditioning_bridge,field).data_ptr(),field)
        for field in ("previous_plan","previous_plan_valid","action_target_offsets"):
            self.assertNotEqual(getattr(graph.current,field).data_ptr(),getattr(current,field).data_ptr())
        changed=replace(current,previous_plan=current.previous_plan+.7,previous_plan_valid=~current.previous_plan_valid,
            action_target_offsets=offsets+.03,observation_delay_seconds=.6,
            age=replace(current.age,delta_sim_seconds=torch.tensor(2.8,dtype=torch.float64),delta_actions=20))
        newworld=replace(bridge,source_action_target_offsets=offsets+.02)
        graph._copy_inputs(changed,newworld,text,None,noise,prefix)
        for field in ("previous_plan","previous_plan_valid","action_target_offsets"):
            self.assertTrue(torch.equal(getattr(graph.current,field),getattr(changed,field)))
        self.assertTrue(torch.equal(graph.bridge.source_action_target_offsets,newworld.source_action_target_offsets))
        self.assertEqual(graph.conditioning_bridge.source_action_target_offsets.data_ptr(),graph.bridge.source_action_target_offsets.data_ptr())
        self.assertTrue(torch.equal(graph.current.age.delta_actions,torch.full((2,),20.)))
        for t in (graph.current.age.delta_sim_seconds,graph.current.action_target_offsets,graph.bridge.source_action_target_offsets):
            self.assertEqual(t.dtype,torch.float64)
        signature=graph._signature(obs,world,text,None,noise,prefix)
        self.assertNotEqual(signature,graph._signature(replace(obs,previous_plan=None),world,text,None,noise,prefix))
        self.assertNotEqual(signature,graph._signature(obs,replace(world,source_action_target_offsets=None),text,None,noise,prefix))
        graph.allow_recapture=False;graph.invalidate()
        self.assertFalse(graph.allow_recapture)


class RealPolicyReportTests(unittest.TestCase):
    def real_report(self):
        clock=Clock();policy=FakePolicy(value)
        class ObservedSession(Session):
            def finish_capture(self):
                result=super().finish_capture()
                if result is not None:result.update(state=self.q.copy(),instruction="test")
                return result
        class PolicyWorker(Worker):
            def poll(self):
                result=super().poll()
                if result is not None:
                    result["actions"]=policy.sample(self.calls[-1])
                    result["policy_timing"]=deepcopy(policy.last_policy_timing)
                return result
        session=ObservedSession(clock,interval_s=.16)
        worker=PolicyWorker(clock,durations=(1.1,.05))
        try:
            report=run_continuous_rtc_episode(session,worker,task="v3",episode_seed=2,action_limit=12,
                action_interval_s=.16,frame_period_s=.16,joint_v3=True,clock=clock,sleep=clock.sleep)
            report["physics_dt_s"]=session.timestep()
            self.assertEqual(len(policy.fast_calls),len(worker.calls)-1)
            self.assertEqual(report["requests"][0]["policy_diagnostics"]["cache_age_seconds"],0.)
            return report
        finally:policy.close()

    def test_actual_controller_policy_bootstrap_and_publication_receipt_pass(self):
        report=self.real_report()
        validate_rtc_report(report)
        self.assertEqual(report["protocol"],JOINT_V3_PROTOCOL)
        self.assertGreater(report["requests"][1]["policy_diagnostics"]["cache_age_seconds"],1.)

    def test_report_rejects_future_unpublished_cross_episode_and_missing_source(self):
        report=self.real_report()
        for field,value in (("source_sim_time",100.),("source_published_monotonic",float("inf")),
                            ("generation",1000),("slow_schedule","synchronous")):
            bad=deepcopy(report)
            bad["requests"][1]["policy_diagnostics"]["asynchronous_world"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_rtc_report(bad)
        bad=deepcopy(report);bad["requests"][1]["install_wall_s"]+=1.
        with self.assertRaisesRegex(ValueError,"deadline"):validate_rtc_report(bad)

    def test_checked_episode_binds_protocol_in_both_directions(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            for stored,expected in ((PROTOCOL,JOINT_V3_PROTOCOL),(JOINT_V3_PROTOCOL,PROTOCOL)):
                (directory/"report.json").write_text(json.dumps({"protocol":stored}))
                for filename in ("metrics.json","manifest.json"):(directory/filename).write_text("{}")
                with self.assertRaisesRegex(ValueError,"protocol differs"):
                    checked_episode(directory,"v3",0,1.,expected_protocol=expected)


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
