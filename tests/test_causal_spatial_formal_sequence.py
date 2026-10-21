"""Pure CPU fail-closed sequence gates, using synthetic receipt files only."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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
        self.acceptance["evidence"]["domino"]=self.guard("domino","smoke",2)
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
            guard_gpu_uuids=ids,worker_gpu_uuids=ids[:workers],proc_gpu_inventory=[dict(uuid=x,pci_bus_id=f"0000:{i+16:02x}:00.0") for i,x in enumerate(ids)],
            device_guard_sha256=sequence.digest(self.root/"scripts/device_lifetime_guard.py"))
        if name=="domino":
            manifest["command"]=["/usr/bin/env","CUDA_VISIBLE_DEVICES="+",".join(ids[:2]),
                "DYNAMICWAM_GPU_PCI=0000:10:00.0","DYNAMICWAM_SIM_CUDA=0","DYNAMICWAM_INFER_CUDA=1",
                "python",str(self.root/"scripts/smoke_causal_spatial_domino.py")]
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
        roles=dict(format="causal_spatial_gpu_roles_v1",allocation="two_gpu_render_policy_split",compute_gpu_count=2,
            visible_uuids=["GPU-0","GPU-1"],verified=True,
            renderer=dict(local_index=0,uuid="GPU-0",pci="0000:10:00.0",identity_source="actual_sapien_device_pci_to_guarded_proc_uuid"),
            policy=dict(local_index=1,current_device=1,model_device="cuda:1",uuid="GPU-1",pci="0000:11:00.0",identity_source="worker_startup_gpu_identity"))
        report["gpu_roles"]=roles
        report["worker_startup_metadata"]["gpu_identity"]={key:value for key,value in roles["policy"].items() if key!="identity_source"}
        report["worker_startup_metadata"]["gpu_identity"]["visible_uuids"]=roles["visible_uuids"]
        manifest=sequence.read(self.acceptance["evidence"]["domino"]["guard_manifest"]["path"])
        for index,row in enumerate(report["requests"]):
            row["policy_diagnostics"]=dict(cuda_graph=True,graph_captures=1,slow_launched=index==1,
                source_publication=1 if index<=1 else 2,source_sim_time=row["observation_physics_s"])
        bundle=dict(report=self.emit("domino-report.json",report))
        bundle["completed"]=self.emit("domino-complete.json",dict(passed=True,full_native_episode=True,
            report=bundle["report"]["path"],video=str(video),gpu_roles=roles))
        runtime=report["worker_startup_metadata"]["runtime"]
        self.real_domino(bundle,checkpoint,runtime,manifest)
        with self.assertRaisesRegex(RuntimeError,"runtime differ"):
            self.real_domino(bundle,checkpoint,dict(torch="other",python="fixture-python",cuda="fixture-cuda"),manifest)
        original=copy.deepcopy(report)
        for row in report["requests"]:row["policy_diagnostics"]["source_publication"]=1
        bundle["report"].update(self.emit(bundle["report"]["path"],report))
        with self.assertRaisesRegex(RuntimeError,"live slow-memory"):self.real_domino(bundle,checkpoint,runtime,manifest)
        bundle["report"].update(self.emit(bundle["report"]["path"],original))
        video.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError,"MP4"):self.real_domino(bundle,checkpoint,runtime,manifest)
        self.mutate(bundle["completed"],full_native_episode=False)
        with self.assertRaisesRegex(RuntimeError,"full native"):self.real_domino(bundle,checkpoint,runtime,manifest)


class TrainingSourceCompatibilityTests(unittest.TestCase):
    """Real local Git objects; no Torch, GPU, receipt rewriting or launches."""
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        override=patch.object(sequence,"ROOT",self.root);override.start();self.addCleanup(override.stop)
        self.git("init","-q")
        self.write("src/dynamicwam/runtime/causal_spatial_contract.py",
            "from collections import deque\ndef unchanged(x): return x\nclass FiniteObservationWindow:\n    capacity=4\n")
        self.write("src/dynamicwam/inference/causal_spatial_policy.py","def infer(): return 1\n")
        self.write("src/dynamicwam/training/data/causal_spatial_dataset.py",
            "from dynamicwam.observations.causal_spatial_window import select\n")
        self.write("src/dynamicwam/observations/causal_spatial_window.py","def select(): return 4\n")
        self.write("src/dynamicwam/models/causal_spatial.py","MODEL_DIM=768\n")
        self.wrapper="def main():\n    if a.mode=='training':\n        cmd='train8'\n    elif a.mode=='graph':\n        cmd='graph1'\n    else:\n        cmd='domino1'\n"
        self.write("scripts/run_causal_spatial_guarded.py",self.wrapper)
        self.write("configs/causal_spatial/joint.yaml",'{"lr": 0.00003}\n')
        self.base=self.commit()
        override=patch.object(sequence,"TRAINING_SMOKE_BASE_HEAD",self.base["git_head"])
        override.start();self.addCleanup(override.stop)

    def git(self,*args):
        return subprocess.check_output(["git","-c","core.autocrlf=false","-c","user.name=CPU Regression",
            "-c","user.email=cpu@example.invalid",*args],cwd=self.root,text=True).strip()

    def write(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value,encoding="utf-8",newline="\n")

    def commit(self):
        self.git("add",".");self.git("commit","-qm","CPU compatibility fixture")
        head=self.git("rev-parse","HEAD");tree=sequence._git_tree(head)
        paths={p for p in tree if p.startswith(("src/","scripts/")) and p.endswith(".py")}
        return dict(git_head=head,source_sha256=sequence._source_hash(sequence._git_blobs(head,paths)))

    def test_actual_git_deployment_only_changes_have_exact_hashes(self):
        self.write("src/dynamicwam/inference/causal_spatial_policy.py","def infer(): return 2\n")
        self.write("src/dynamicwam/runtime/causal_spatial_contract.py",
            "from collections import deque, OrderedDict\nfrom concurrent.futures import ThreadPoolExecutor\n"
            "def unchanged(x): return x\nclass FiniteObservationWindow:\n    capacity=8\n")
        target=self.commit();receipt=sequence.build_training_source_compatibility(self.base,target)
        self.assertEqual(len(receipt["changed_files"]),2)
        self.assertTrue(receipt["window_noncache_ast_unchanged"])
        self.assertTrue(receipt["current_cpu_graph_domino_required"])
        for row in receipt["changed_files"]:
            self.assertEqual(row["after_sha256"],sequence.digest(self.root/row["path"]))
            self.assertNotEqual(row["before_sha256"],row["after_sha256"])
        path=self.root/"compatibility.json";path.write_text(json.dumps(receipt))
        acceptance=dict(training_source_compatibility=dict(path=str(path),sha256=sequence.digest(path)))
        self.assertEqual(sequence.training_smoke_identity(acceptance,target),self.base)
        receipt["protected_tracked_files"]+=1;path.write_text(json.dumps(receipt))
        acceptance["training_source_compatibility"]["sha256"]=sequence.digest(path)
        with self.assertRaisesRegex(RuntimeError,"actual Git evidence"):
            sequence.training_smoke_identity(acceptance,target)

    def test_model_data_or_config_change_cannot_reuse_smokes(self):
        self.write("src/dynamicwam/models/causal_spatial.py","MODEL_DIM=512\n")
        with self.assertRaisesRegex(RuntimeError,"non-allowlisted"):
            sequence.build_training_source_compatibility(self.base,self.commit())

    def test_contract_functions_outside_cache_class_cannot_change(self):
        self.write("src/dynamicwam/runtime/causal_spatial_contract.py",
            "from collections import deque\ndef unchanged(x): return x+1\nclass FiniteObservationWindow:\n    capacity=4\n")
        with self.assertRaisesRegex(RuntimeError,"outside deployment"):
            sequence.build_training_source_compatibility(self.base,self.commit())

    def test_source_hash_cannot_be_borrowed_from_different_guard(self):
        self.write("src/dynamicwam/inference/causal_spatial_policy.py","def infer(): return 2\n")
        target=self.commit();bad=dict(self.base,source_sha256="0"*64)
        with self.assertRaisesRegex(RuntimeError,"old guard source hash"):
            sequence.build_training_source_compatibility(bad,target)

    def test_unknown_base_and_tracked_addition_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"exact audited"):
            sequence.build_training_source_compatibility(dict(self.base,git_head="0"*40),dict(self.base))
        self.write("src/dynamicwam/inference/new_deployment.py","x=1\n")
        with self.assertRaisesRegex(RuntimeError,"additions/deletions"):
            sequence.build_training_source_compatibility(self.base,self.commit())

    def test_wrapper_domino_only_change_allowed_but_training_change_rejected(self):
        path="scripts/run_causal_spatial_guarded.py"
        self.write(path,self.wrapper.replace("domino1","domino2"))
        receipt=sequence.build_training_source_compatibility(self.base,self.commit())
        self.assertTrue(receipt["guard_training_and_graph_ast_unchanged"])
        self.write(path,self.wrapper.replace("train8","train7"))
        with self.assertRaisesRegex(RuntimeError,"outside the DOMINO-only"):
            sequence.build_training_source_compatibility(self.base,self.commit())

    def test_only_explicit_new_role_regression_is_admitted(self):
        self.write("tests/test_causal_spatial_smoke_gpu_roles.py","def test_roles(): pass\n")
        receipt=sequence.build_training_source_compatibility(self.base,self.commit())
        row=receipt["changed_files"][0]
        self.assertEqual(row["status"],"A");self.assertIsNone(row["before_sha256"])


class DualGpuAdmissionTests(unittest.TestCase):
    def test_actual_wrapper_domino_branch_uses_two_uuids_and_exec(self):
        tree=ast.parse((ROOT/"scripts/run_causal_spatial_guarded.py").read_text())
        graph=ast.dump(ast.parse("a.mode=='graph'",mode="eval").body,include_attributes=False)
        branches=[node.orelse for node in ast.walk(tree) if isinstance(node,ast.If)
            and ast.dump(node.test,include_attributes=False)==graph]
        self.assertEqual(len(branches),1)
        ids=["GPU-"+str(i) for i in range(8)]
        inventory=[dict(uuid=uuid,pci_bus_id=f"0000:{i+16:02x}:00.0") for i,uuid in enumerate(ids)]
        environment=dict(ROOT=ROOT,Path=Path,ids=ids,run=Path("/diagnostics/domino"),
            proc_gpu_inventory=lambda:inventory,a=SimpleNamespace(deploy_config="deploy.yaml",checkpoint="actual.pt",formal_scene_manifest=None))
        exec(compile(ast.Module(body=branches[0],type_ignores=[]),"actual_domino_branch","exec"),environment)
        command=environment["cmd"]
        self.assertEqual(environment["compute"],ids[:2]);self.assertEqual(command[0],"/usr/bin/env")
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-0,GPU-1",command)
        self.assertIn("DYNAMICWAM_SIM_CUDA=0",command);self.assertIn("DYNAMICWAM_INFER_CUDA=1",command)
        self.assertIn("DYNAMICWAM_GPU_PCI=0000:10:00.0",command)

    def test_actual_roles_and_command_must_match_guarded_inventory(self):
        roles=dict(format="causal_spatial_gpu_roles_v1",allocation="two_gpu_render_policy_split",compute_gpu_count=2,
            visible_uuids=["GPU-A","GPU-B"],verified=True,
            renderer=dict(local_index=0,uuid="GPU-A",pci="00000000:10:00.0",identity_source="actual_sapien_device_pci_to_guarded_proc_uuid"),
            policy=dict(local_index=1,current_device=1,model_device="cuda:1",uuid="GPU-B",pci="0000:11:00.0",identity_source="worker_startup_gpu_identity"))
        manifest=dict(worker_gpu_uuids=["GPU-A","GPU-B"],proc_gpu_inventory=[dict(uuid="GPU-A",pci_bus_id="0000:10:00.0"),
            dict(uuid="GPU-B",pci_bus_id="0000:11:00.0")],command=["/usr/bin/env","CUDA_VISIBLE_DEVICES=GPU-A,GPU-B",
            "DYNAMICWAM_GPU_PCI=0000:10:00.0","DYNAMICWAM_SIM_CUDA=0","DYNAMICWAM_INFER_CUDA=1",
            "python",str(sequence.ROOT/"scripts/smoke_causal_spatial_domino.py")])
        sequence.check_domino_gpu_roles(roles,manifest)
        for field,value in (("current_device",0),("uuid","GPU-A"),("pci","0000:12:00.0"),("model_device","cuda:0")):
            invalid=copy.deepcopy(roles);invalid["policy"][field]=value
            with self.assertRaises(RuntimeError):sequence.check_domino_gpu_roles(invalid,manifest)
        invalid=copy.deepcopy(manifest);invalid["command"][1]="CUDA_VISIBLE_DEVICES=GPU-B,GPU-A"
        with self.assertRaisesRegex(RuntimeError,"command does not bind"):
            sequence.check_domino_gpu_roles(roles,invalid)
        invalid=copy.deepcopy(manifest);invalid["worker_gpu_uuids"]=["GPU-A"]
        with self.assertRaisesRegex(RuntimeError,"exactly two"):
            sequence.check_domino_gpu_roles(roles,invalid)


if __name__=="__main__":unittest.main()
