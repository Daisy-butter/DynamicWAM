"""Pure CPU fail-closed sequence gates, using synthetic receipt files only."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
spec=importlib.util.spec_from_file_location("formal_sequence",ROOT/"scripts/run_causal_spatial_formal_sequence.py")
sequence=importlib.util.module_from_spec(spec);spec.loader.exec_module(sequence)


class FormalSequenceGateTests(unittest.TestCase):
    def setUp(self):
        from dynamicwam.inference.causal_spatial_policy import EXPECTED_RUNTIME_CONTRACT
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.ssd=self.root/"ssd";self.ssd.mkdir()
        (self.root/"scripts").mkdir();(self.root/"scripts/device_lifetime_guard.py").write_text("fixture guard")
        self.configdir=self.root/"configs/causal_spatial";self.configdir.mkdir(parents=True)
        for name,value in (("ROOT",self.root),("SSD",self.ssd)):
            p=patch.object(sequence,name,value);p.start();self.addCleanup(p.stop)
        self.identity=dict(git_head="a"*40,source_sha256="b"*64)
        self.acceptance=dict(format=sequence.FORMAT,passed=True,**self.identity,configs={},
            evidence=dict(smokes={},data=self.emit("data.json",dict(passed=True,gpu_used=False)),
                          cpu=self.emit("cpu.json",dict(passed=True,gpu_used=False,source_sha256=self.identity['source_sha256'],fast_strict_roundtrip=True,teacher_normalization_equal=True))))
        previous=None
        for phase in sequence.PHASES:
            recipe=dict(phase=phase,steps=sequence.STEPS[phase],diagnostic=False,microbatch=4,accumulation=4,
                run_dir=str(self.ssd/phase),initial_checkpoint="audited-native-init",teacher_checkpoint="native-teacher",
                model={"contract":"model"},runtime_contract=copy.deepcopy(EXPECTED_RUNTIME_CONTRACT),data={"dataset":"actual"},
                lr={"new":1e-4},loss={"future":.1},continue_from="" if previous is None else
                str(Path(previous["run_dir"])/"exports"/f"causal_spatial_step_{previous['steps']}.pt"))
            config=self.emit(self.configdir/(phase+".yaml"),recipe);self.acceptance["configs"][phase]=config
            smoke=copy.deepcopy(recipe);smoke.update(steps=4,diagnostic=True,run_dir=str(self.ssd/(phase+"_smoke")),continue_from="")
            smoke_config=self.emit(self.configdir/(phase+"_smoke.yaml"),smoke)
            full=self.emit(Path(smoke["run_dir"])/"exports/causal_spatial_step_4.pt",{"fixture":"full"})
            fast=self.emit(Path(smoke["run_dir"])/"exports/causal_spatial_fast_step_4.pt",{"fixture":"fast"})
            audit=dict(format="causal_spatial_export_audit_v1",passed=True,phase=phase,diagnostic=True,
                config=smoke_config,full_checkpoint=full,fast_checkpoint=fast,
                full_strict_reload=dict(passed=True),fast_strict_reload=dict(passed=True),full_fast_equal=dict(passed=True),
                saved_state={"fixture":"complete"},ranks=[dict(rank=i,world_size=8,completed=True,error_type=None,
                    step=4,optimizer_updates=4,oom_count=0,nonfinite_loss_count=0,nonfinite_grad_count=0,
                    finite_grad_checks=4,finite_loss_checks=16) for i in range(8)])
            bundle=self.guard(phase,"smoke",8);bundle["audit"]=self.emit(phase+"_audit.json",audit)
            self.acceptance["evidence"]["smokes"][phase]=bundle;previous=recipe
        graph=dict(passed=True,source_sha256=self.identity["source_sha256"],checkpoint=full,
            runtime=dict(torch="fixture-torch",python="fixture-python",cuda="fixture-cuda"),
            cpu_strict_reload=dict(artifact_roundtrip=True),full_fast_gpu_parity=dict(passed=True),
            background_memory=dict(passed=True),graph=dict(captures=1,replays=22,production_shared_graph=True,
                shared_graph_module="dynamicwam.inference.causal_spatial_graph"),
            plan_causality=dict(belief_invariant=True,invalid_or_outside_overlap_exact_noop=True),peak_memory=dict(num_ooms=0))
        bundle=self.guard("graph","graph",1);bundle["result"]=self.emit("graph.json",graph)
        self.acceptance["evidence"]["graph"]=bundle
        self.acceptance["evidence"]["domino"]=self.guard("domino","smoke",1)
        self.path=self.root/"acceptance.json";self.flush()
        self.real_domino=sequence.check_domino
        self.domino=patch.object(sequence,"check_domino",return_value={});self.mock_domino=self.domino.start();self.addCleanup(self.domino.stop)
        launch=patch.object(sequence.subprocess,"Popen",side_effect=AssertionError("CPU gates must not launch any process"))
        launch.start();self.addCleanup(launch.stop)

    def emit(self,path,value):
        path=Path(path)
        if not path.is_absolute():path=self.root/path
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=sequence.digest(path))

    def guard(self,name,stage,workers):
        directory=self.ssd/(name+"_guard");ids=["GPU-"+str(i) for i in range(8)]
        status=dict(ended_at="fixture-end",final_exit_code=0,driver_exit_clean=True,guard_workers_exited=True,
                    guard_all_handles_closed=True,stop_reason=None)
        manifest=dict(**self.identity,stage=stage,preflight_only=False,device_lifetime_mode="fds_before_telemetry_v1",
            guard_gpu_uuids=ids,worker_gpu_uuids=ids[:workers],proc_gpu_inventory=[dict(uuid=x) for x in ids],
            device_guard_sha256=sequence.digest(self.root/"scripts/device_lifetime_guard.py"))
        return dict(guard_status=self.emit(directory/"status.json",status),guard_manifest=self.emit(directory/"manifest.json",manifest))

    def mutate(self,ref,**values):
        data=sequence.read(ref["path"]);data.update(values);ref.update(self.emit(ref["path"],data))

    def flush(self):self.path.write_text(json.dumps(self.acceptance))

    def validate(self):
        self.flush();return sequence.validate_acceptance(self.path,identity=self.identity)

    def test_complete_bound_receipts_pass_without_launch(self):
        result=self.validate();self.assertEqual(result["identity"],self.identity)
        self.assertEqual(set(result["recipes"]),set(sequence.PHASES));self.mock_domino.assert_called_once()

    def test_missing_phase_smoke_is_rejected(self):
        del self.acceptance["evidence"]["smokes"]["joint"]
        with self.assertRaisesRegex(RuntimeError,"three actual"):self.validate()

    def test_stale_source_or_incomplete_acceptance_is_rejected(self):
        for values in (dict(source_sha256="other"),dict(passed=False)):
            original=copy.deepcopy(self.acceptance);self.acceptance.update(values)
            with self.assertRaises(RuntimeError):self.validate()
            self.acceptance=original

    def test_changed_artifact_cannot_reuse_old_hash(self):
        Path(self.acceptance["evidence"]["cpu"]["path"]).write_text("{}")
        with self.assertRaisesRegex(RuntimeError,"artifact changed"):self.validate()

    def test_diagnostic_formal_or_wrong_continuation_is_rejected(self):
        self.mutate(self.acceptance["configs"]["policy_pretrain"],continue_from="smoke.pt")
        with self.assertRaisesRegex(RuntimeError,"previous formal final"):self.validate()

    def test_formal_diagnostic_recipe_is_rejected(self):
        self.mutate(self.acceptance["configs"]["interface_adapt"],diagnostic=True)
        with self.assertRaisesRegex(RuntimeError,"formal phase"):self.validate()

    def test_smoke_trainable_recipe_must_match_formal(self):
        ref=self.acceptance["configs"]["joint"];self.mutate(ref,lr={"new":.2})
        with self.assertRaisesRegex(RuntimeError,"smoke differs.*lr"):self.validate()

    def test_smoke_requires_eight_real_rank_updates_and_strict_reload(self):
        ref=self.acceptance["evidence"]["smokes"]["joint"]["audit"]
        audit=sequence.read(ref["path"]);audit["ranks"][7]["optimizer_updates"]=3
        ref.update(self.emit(ref["path"],audit))
        with self.assertRaisesRegex(RuntimeError,"update budget"):self.validate()

    def test_unclean_guard_blocks_later_phase(self):
        ref=self.acceptance["evidence"]["graph"]["guard_status"];self.mutate(ref,guard_all_handles_closed=False)
        with self.assertRaisesRegex(RuntimeError,"cleanly closed"):self.validate()

    def test_graph_requires_actual_checkpoint_and_refresh_roundtrip(self):
        ref=self.acceptance["evidence"]["graph"]["result"]
        self.mutate(ref,cpu_strict_reload=dict(artifact_roundtrip=False))
        with self.assertRaisesRegex(RuntimeError,"acceptance incomplete"):self.validate()

    def test_graph_must_test_joint_smoke_checkpoint(self):
        ref=self.acceptance["evidence"]["graph"]["result"];self.mutate(ref,checkpoint={"sha256":"old-weights"})
        with self.assertRaisesRegex(RuntimeError,"joint-smoke"):self.validate()

    def test_existing_formal_output_is_never_overwritten_or_resumed(self):
        (self.ssd/"joint").mkdir()
        with self.assertRaisesRegex(RuntimeError,"preserve existing"):self.validate()

    def test_domino_and_video_failure_blocks_formal(self):
        self.mock_domino.side_effect=RuntimeError("native MP4/full-decode sidecar validation failed")
        with self.assertRaisesRegex(RuntimeError,"MP4"):self.validate()

    def test_failed_strict_reload_flag_is_not_truthy_acceptance(self):
        ref=self.acceptance["evidence"]["smokes"]["joint"]["audit"]
        self.mutate(ref,full_strict_reload=dict(passed=False))
        with self.assertRaisesRegex(RuntimeError,"missing/failed full_strict"):self.validate()

    def test_actual_domino_gate_rejects_short_episode_and_changed_video_bytes(self):
        from test_causal_spatial_controller import CausalSpatialControllerTests
        _,_,report=CausalSpatialControllerTests().run_case()
        checkpoint=sequence.read(self.acceptance["evidence"]["graph"]["result"]["path"])["checkpoint"]
        video=self.root/"episode0.mp4";video.write_bytes(b"synthetic receipt fixture, not a decoded video")
        sidecar=self.root/"episode0.timestamps.json"
        probe=dict(codec="h264",pixel_format="yuv420p",width=2,height=2,fps=10.,frame_count=2)
        metadata=dict(schema="domino_native_head_video_timestamps_v1",video=video.name,frame_count=2,fps=10.,
            video_sha256=sequence.digest(video),video_bytes=video.stat().st_size,continuous_physics_video=False,
            sampling="existing_head_captures",playback_timing="native_frame_index",width=2,height=2,probe=probe,
            frames=[dict(frame_index=i,physics_time_s=i*.1,capture_wall_perf_s=100.+i*.1) for i in range(2)])
        sidecar.write_text(json.dumps(metadata))
        report.update(full_native_episode=True,checkpoint=checkpoint["path"],checkpoint_sha256=checkpoint["sha256"],
            worker_startup_metadata=dict(cuda_graph=True,graph_captures=1,graph_eager_parity=True,
                shared_graph_module="dynamicwam.inference.causal_spatial_graph",
                runtime=dict(torch="fixture-torch",python="fixture-python",cuda="fixture-cuda")),
            video=str(video),video_timestamps=str(sidecar),video_sha256=sequence.digest(video),video_bytes=video.stat().st_size,
            video_timestamps_sha256=sequence.digest(sidecar),video_timestamps_bytes=sidecar.stat().st_size,
            video_frames=2,video_fps=10.,video_continuous_physics=False,video_probe=probe)
        for index,row in enumerate(report["requests"]):
            row["policy_diagnostics"]=dict(cuda_graph=True,graph_captures=1,slow_launched=index==1,
                source_publication=1 if index<=1 else 2,source_sim_time=row["observation_physics_s"])
        bundle=dict(report=self.emit("domino-report.json",report))
        bundle["completed"]=self.emit("domino-complete.json",dict(passed=True,full_native_episode=True,
            report=bundle["report"]["path"],video=str(video)))
        runtime=report["worker_startup_metadata"]["runtime"]
        self.real_domino(bundle,checkpoint,runtime)
        with self.assertRaisesRegex(RuntimeError,"runtime differ"):
            self.real_domino(bundle,checkpoint,dict(torch="other",python="fixture-python",cuda="fixture-cuda"))
        original=copy.deepcopy(report)
        for row in report["requests"]:row["policy_diagnostics"]["source_publication"]=1
        bundle["report"].update(self.emit(bundle["report"]["path"],report))
        with self.assertRaisesRegex(RuntimeError,"live slow-memory"):self.real_domino(bundle,checkpoint,runtime)
        bundle["report"].update(self.emit(bundle["report"]["path"],original))
        video.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError,"MP4"):self.real_domino(bundle,checkpoint,runtime)
        self.mutate(bundle["completed"],full_native_episode=False)
        with self.assertRaisesRegex(RuntimeError,"full native"):self.real_domino(bundle,checkpoint,runtime)


if __name__=="__main__":unittest.main()
