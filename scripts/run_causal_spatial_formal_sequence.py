#!/usr/bin/env python3
"""Receipt-gated, one-shot formal phase sequence; CPU waiting, no retries/polls.

Acceptance schema: causal_spatial_formal_acceptance_v1; passed=true; git_head,
source_sha256; configs[phase]={path,sha256}; evidence.data/cpu={path,sha256};
evidence.smokes[phase]={audit,guard_status,guard_manifest}; evidence.graph=
{result,guard_status,guard_manifest}; evidence.domino={completed,report,
guard_status,guard_manifest}. Every artifact reference is {path,sha256}.
The root writes acceptance only after actual acceptance, never this controller.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
SSD=Path("/SSD_DISK/users/wuruihan/DynamicWAM")
OUT=SSD/"outputs/diagnostics/causal_spatial_v4_20261021"
PHASES=("interface_adapt","policy_pretrain","joint")
STEPS=dict(zip(PHASES,(2000,30000,20000)))
FORMAT="causal_spatial_formal_acceptance_v1"
TRAINING_SMOKE_BASE_HEAD="4af56fa925a8b67aa880ccf69ba85c1b3b270724"
TRAINING_REUSE_ALLOWED=frozenset({
    "src/dynamicwam/inference/causal_spatial_policy.py",
    "src/dynamicwam/runtime/causal_spatial_contract.py",
    "scripts/run_causal_spatial_formal_sequence.py",
    "scripts/run_causal_spatial_guarded.py",
    "scripts/smoke_causal_spatial_domino.py",
    "tests/test_causal_spatial_policy.py",
    "tests/test_causal_spatial_contract.py",
    "tests/test_causal_spatial_formal_sequence.py",
    "tests/test_causal_spatial_smoke_gpu_roles.py",
})
TRAINING_REUSE_ADDED_TESTS=frozenset({"tests/test_causal_spatial_smoke_gpu_roles.py"})


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def artifact(ref,*,json_value=True):
    require(isinstance(ref,dict) and set(ref)>={"path","sha256"},"artifact requires path and SHA256")
    path=Path(ref["path"]).resolve(strict=True)
    require(path.is_file() and digest(path)==ref["sha256"],"artifact changed: "+str(path))
    return read(path) if json_value else path


def source_identity():
    from supervise_motionkv_training import source_files_and_hash
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),
            "freeze and commit the source before formal sequence")
    return dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                source_sha256=source_files_and_hash(ROOT)[1])


def _git_tree(revision):
    raw=subprocess.check_output(["git","ls-tree","-r","-z",revision],cwd=ROOT)
    result={}
    for item in raw.split(b"\0"):
        if not item:continue
        metadata,path=item.split(b"\t",1);mode,kind,oid=metadata.decode().split()
        result[path.decode("utf-8")]=dict(mode=mode,kind=kind,oid=oid)
    return result


def _git_blobs(revision,paths):
    paths=sorted(paths)
    require(all("\n" not in p and "\r" not in p for p in paths),"unsupported source filename")
    request="".join(revision+":"+p+"\n" for p in paths).encode()
    data=subprocess.check_output(["git","cat-file","--batch"],cwd=ROOT,input=request)
    result={};offset=0
    for path in paths:
        newline=data.index(b"\n",offset);header=data[offset:newline].split()
        require(len(header)==3 and header[1]==b"blob","source is not a Git blob: "+path)
        size=int(header[2]);offset=newline+1
        result[path]=data[offset:offset+size];offset+=size
        require(data[offset:offset+1]==b"\n","invalid git batch boundary");offset+=1
    require(offset==len(data),"extra Git batch output")
    return result


def _source_hash(blobs):
    value=hashlib.sha256()
    for path,content in sorted(blobs.items()):
        if path.startswith(("src/","scripts/")) and path.endswith(".py"):
            value.update(path.encode()+b"\0"+content+b"\0")
    return value.hexdigest()


def _window_external_ast(content):
    """Only the deployment cache class and OrderedDict import may change."""
    tree=ast.parse(content)
    tree.body=[node for node in tree.body if not(isinstance(node,ast.ClassDef) and node.name=="FiniteObservationWindow")]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=="collections":
            require(all(alias.name!="OrderedDict" or alias.asname is None for alias in node.names),"OrderedDict alias changed")
            node.names=[alias for alias in node.names if alias.name!="OrderedDict"]
        if isinstance(node,ast.ImportFrom) and node.module=="concurrent.futures":
            require(all(alias.name=="ThreadPoolExecutor" and alias.asname is None for alias in node.names),
                    "only the bounded window ThreadPoolExecutor import is permitted")
            node.names=[]
    tree.body=[node for node in tree.body if not(isinstance(node,ast.ImportFrom) and not node.names)]
    return ast.dump(tree,include_attributes=False)


def _guard_training_ast(content):
    """Erase only DOMINO's worker-command branch; compare everything else."""
    tree=ast.parse(content);matches=0
    expected=ast.dump(ast.parse("a.mode=='training'",mode="eval").body,include_attributes=False)
    graph=ast.dump(ast.parse("a.mode=='graph'",mode="eval").body,include_attributes=False)
    for node in ast.walk(tree):
        if not isinstance(node,ast.If) or ast.dump(node.test,include_attributes=False)!=expected:continue
        if len(node.orelse)!=1 or not isinstance(node.orelse[0],ast.If):continue
        branch=node.orelse[0]
        if ast.dump(branch.test,include_attributes=False)!=graph:continue
        require(bool(branch.orelse),"guard DOMINO branch missing")
        branch.orelse=[ast.Pass()];matches+=1
    require(matches==1,"guard command branch is ambiguous")
    return ast.dump(tree,include_attributes=False)


def build_training_source_compatibility(base_identity,target_identity):
    """Recompute narrow compatibility from real Git objects; never trust a flag.

    This admits old *training smokes only*. CPU, Graph, DOMINO, guards, model,
    optimizer, loaders, datasets and every recipe retain their original gates.
    """
    base=base_identity["git_head"];target=target_identity["git_head"]
    require(base==TRAINING_SMOKE_BASE_HEAD and base!=target,"training reuse requires the exact audited base commit")
    require(bool(re.fullmatch(r"[0-9a-f]{40}",target)),"invalid compatibility target HEAD")
    require(subprocess.run(["git","merge-base","--is-ancestor",base,target],cwd=ROOT,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,"training smoke source is not a Git ancestor")
    before=_git_tree(base);after=_git_tree(target)
    added=set(after)-set(before);removed=set(before)-set(after)
    require(not removed and added<=TRAINING_REUSE_ADDED_TESTS,
            "training reuse forbids tracked file additions/deletions/renames except the exact GPU-role regression")
    changed=sorted(path for path in set(before)|set(after) if before.get(path)!=after.get(path))
    require(bool(changed) and set(changed)<=TRAINING_REUSE_ALLOWED,
            "training reuse contains a non-allowlisted changed file: "+str(sorted(set(changed)-TRAINING_REUSE_ALLOWED)))
    for path in changed:
        require(after[path]["mode"]=="100644" and after[path]["kind"]=="blob" and
                (path in added or before[path]["mode"]==after[path]["mode"] and before[path]["kind"]==after[path]["kind"]),
                "changed source mode/type is not a regular Python file")
    python_paths={path for path in before if path.startswith(("src/","scripts/")) and path.endswith(".py")}
    loaded=python_paths|set(changed)
    old=_git_blobs(base,loaded-set(added));new=_git_blobs(target,loaded)
    require(_source_hash(old)==base_identity["source_sha256"],"old guard source hash does not match its Git commit")
    require(_source_hash(new)==target_identity["source_sha256"],"current source hash does not match its Git commit")
    contract="src/dynamicwam/runtime/causal_spatial_contract.py"
    require(_window_external_ast(old[contract])==_window_external_ast(new[contract]),
            "runtime contract changed outside deployment FiniteObservationWindow cache")
    wrapper="scripts/run_causal_spatial_guarded.py"
    if wrapper in changed:
        require(_guard_training_ast(old[wrapper])==_guard_training_ast(new[wrapper]),
                "guard wrapper changed outside the DOMINO-only command branch")
    protected_roots=("src/dynamicwam/training/","src/dynamicwam/models/","src/dynamicwam/observations/")
    protected_exact={"src/dynamicwam/inference/causal_spatial_loader.py","scripts/train_causal_spatial.py",
                     "scripts/check_causal_spatial_export.py"}
    deployment_modules=("dynamicwam.runtime.causal_spatial_contract","dynamicwam.inference.causal_spatial_policy")
    for path in python_paths:
        if not(path.startswith(protected_roots) or path in protected_exact):continue
        require(old[path]==new[path],"training/data/model/loader bytes changed: "+path)
        for node in ast.walk(ast.parse(new[path])):
            imports=[alias.name for alias in node.names] if isinstance(node,ast.Import) else (
                [node.module or ""] if isinstance(node,ast.ImportFrom) else [])
            require(not any(name==m or name.startswith(m+".") for name in imports for m in deployment_modules),
                    "protected training code imports a modified deployment module: "+path)
    protected=[dict(path=path,**before[path]) for path in sorted(before) if path not in TRAINING_REUSE_ALLOWED]
    return dict(format="causal_spatial_training_source_compatibility_v1",base_identity=base_identity,
        target_identity=target_identity,git_ancestor=True,scope="three_existing_training_smokes_only",
        allowed_changed_files=sorted(TRAINING_REUSE_ALLOWED),changed_files=[dict(path=path,status="A" if path in added else "M",
            before_sha256=None if path in added else hashlib.sha256(old[path]).hexdigest(),after_sha256=hashlib.sha256(new[path]).hexdigest(),
            before_git_blob=None if path in added else before[path]["oid"],after_git_blob=after[path]["oid"]) for path in changed],
        protected_tracked_files=len(protected),
        protected_inventory_sha256=hashlib.sha256(json.dumps(protected,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        training_data_models_configs_loaders_unchanged=True,window_noncache_ast_unchanged=True,
        shared_observation_helpers_unchanged=True,guard_training_and_graph_ast_unchanged=True,
        current_cpu_graph_domino_required=True)


def training_smoke_identity(acceptance,identity):
    reference=acceptance.get("training_source_compatibility")
    if reference is None:return identity
    receipt=artifact(reference)
    expected=build_training_source_compatibility(receipt["base_identity"],identity)
    require(receipt==expected,"training source compatibility receipt differs from actual Git evidence")
    return receipt["base_identity"]


def check_guard(bundle,identity,*,expected_stage,workers):
    status=artifact(bundle["guard_status"]);manifest=artifact(bundle["guard_manifest"])
    path=Path(bundle["guard_status"]["path"]).resolve()
    require(path.name=="status.json" and Path(bundle["guard_manifest"]["path"]).resolve()==path.parent/"manifest.json",
            "guard status/manifest must belong to one lifecycle")
    require(bool(status.get("ended_at")) and status.get("final_exit_code")==0 and
            all(status.get(key) is True for key in ("driver_exit_clean","guard_workers_exited","guard_all_handles_closed")),
            "acceptance lifecycle workers and guards have not cleanly closed")
    for key in ("git_head","source_sha256"):require(manifest[key]==identity[key],"guard source identity mismatch")
    require(manifest.get("stage")==expected_stage and manifest.get("preflight_only") is False,"wrong guard stage")
    require(manifest.get("device_lifetime_mode")=="fds_before_telemetry_v1","wrong device lifetime mode")
    ids=manifest.get("guard_gpu_uuids",[]);selected=manifest.get("worker_gpu_uuids",[])
    require(len(ids)==len(set(ids))==8 and len(selected)==len(set(selected))==workers and set(selected)<=set(ids),
            "guard or compute GPU coverage mismatch")
    require(set(ids)=={row["uuid"] for row in manifest["proc_gpu_inventory"]},"guard differs from actual inventory")
    require(manifest["device_guard_sha256"]==digest(ROOT/"scripts/device_lifetime_guard.py"),"guard helper changed")
    require(not status.get("stop_reason"),"lifecycle stopped with a failure")
    return path.parent,manifest


def check_export(value,recipe,*,diagnostic):
    require(value.get("format")=="causal_spatial_export_audit_v1" and value.get("passed") is True,"strict export audit failed")
    require(value.get("phase")==recipe["phase"] and value.get("diagnostic") is diagnostic,"audit phase/diagnostic mismatch")
    for key in ("full_strict_reload","fast_strict_reload","full_fast_equal"):
        require(value.get(key,{}).get("passed") is True,"audit missing/failed "+key)
    require(bool(value.get("saved_state")),"audit missing saved optimizer/RNG state")
    ranks=value.get("ranks",[])
    require(len(ranks)==8 and {r["rank"] for r in ranks}==set(range(8)),"audit needs all eight ranks")
    for row in ranks:
        require(row.get("world_size")==8 and row.get("completed") is True and row.get("error_type") is None,
                "rank incomplete")
        require(row.get("step")==row.get("optimizer_updates")==recipe["steps"],"rank update budget mismatch")
        require(all(row.get(k)==0 for k in ("oom_count","nonfinite_loss_count","nonfinite_grad_count")),"rank numerical/OOM failure")
        require(row.get("finite_grad_checks")==recipe["steps"] and
                row.get("finite_loss_checks",0)>=recipe["steps"]*recipe["accumulation"],"rank finite-check coverage missing")
    canonical=Path(recipe["run_dir"]).resolve()/"exports"/f"causal_spatial_step_{recipe['steps']}.pt"
    require(Path(value["full_checkpoint"]["path"]).resolve()==canonical,"export is not the canonical final checkpoint")
    artifact(value["full_checkpoint"],json_value=False)
    artifact(value["fast_checkpoint"],json_value=False)
    return value["full_checkpoint"]


def _pci(value):
    match=re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",str(value))
    require(match is not None,"invalid GPU PCI identity")
    return tuple(int(part,16) for part in match.groups())


def check_domino_gpu_roles(roles,manifest):
    selected=manifest.get("worker_gpu_uuids",[])
    require(len(selected)==len(set(selected))==2,"DOMINO requires exactly two compute GPUs")
    inventory={row["uuid"]:row["pci_bus_id"] for row in manifest["proc_gpu_inventory"]}
    require(roles.get("format")=="causal_spatial_gpu_roles_v1" and roles.get("verified") is True and
            roles.get("allocation")=="two_gpu_render_policy_split" and roles.get("compute_gpu_count")==2 and
            roles.get("visible_uuids")==selected,"DOMINO actual dual-GPU allocation is absent or mismatched")
    render=roles.get("renderer",{});policy=roles.get("policy",{})
    require(render.get("local_index")==0 and render.get("uuid")==selected[0] and
            render.get("identity_source")=="actual_sapien_device_pci_to_guarded_proc_uuid",
            "renderer identity is not actual local GPU0")
    require(policy.get("local_index")==1 and policy.get("current_device")==1 and
            policy.get("model_device")=="cuda:1" and policy.get("uuid")==selected[1] and
            policy.get("identity_source")=="worker_startup_gpu_identity","policy identity is not actual local GPU1")
    require(_pci(render.get("pci"))==_pci(inventory[selected[0]]) and
            _pci(policy.get("pci"))==_pci(inventory[selected[1]]),"DOMINO actual PCI roles differ from guarded UUID inventory")
    command=manifest.get("command",[])
    require(command and command[0]=="/usr/bin/env","DOMINO must preserve guarded worker PID through env exec")
    assignments={};index=1
    while index<len(command) and "=" in command[index]:
        key,value=command[index].split("=",1)
        require(key not in assignments,"duplicate DOMINO environment binding")
        assignments[key]=value;index+=1
    require(assignments.get("CUDA_VISIBLE_DEVICES")==",".join(selected) and
            assignments.get("DYNAMICWAM_SIM_CUDA")=="0" and assignments.get("DYNAMICWAM_INFER_CUDA")=="1" and
            _pci(assignments.get("DYNAMICWAM_GPU_PCI"))==_pci(render["pci"]),
            "DOMINO command does not bind render0/policy1 to its actual identities")
    require(str(ROOT/"scripts/smoke_causal_spatial_domino.py") in command[index:],"wrong dual-GPU DOMINO worker command")


def check_domino(bundle,checkpoint,runtime,guard_manifest):
    from dynamicwam.runtime.continuous_causal_spatial import validate_causal_spatial_report
    from video_artifacts import validate_video_artifacts
    complete=artifact(bundle["completed"]);report=artifact(bundle["report"])
    require(complete.get("passed") is True and complete.get("full_native_episode") is True,
            "complete full native DOMINO episode is required")
    require(Path(complete["report"]).resolve()==Path(bundle["report"]["path"]).resolve(),"DOMINO report reference mismatch")
    require(report.get("full_native_episode") is True and report.get("realtime_valid") is True and
            report.get("continuous_valid") is True,"DOMINO is not full 1x continuous acceptance")
    require(report.get("completed_targets",0)>0 and report.get("inference_calls",0)>=2,"no actual policy handoff")
    require(report["checkpoint_sha256"]==checkpoint["sha256"] and
            Path(report["checkpoint"]).resolve()==Path(checkpoint["path"]).resolve(),"DOMINO tested a different checkpoint")
    roles=report.get("gpu_roles",{})
    require(complete.get("gpu_roles")==roles,"DOMINO completed/report actual GPU role disagreement")
    check_domino_gpu_roles(roles,guard_manifest)
    startup=report.get("worker_startup_metadata",{})
    actual=startup.get("gpu_identity",{})
    require(all(actual.get(key)==roles["policy"].get(key) for key in
                ("local_index","current_device","uuid","model_device")) and
            actual.get("visible_uuids")==roles["visible_uuids"] and
            _pci(actual.get("pci"))==_pci(roles["policy"]["pci"]),
            "reported policy roles differ from actual child CUDA startup identity")
    measured_runtime=startup.get("runtime",{})
    require(all(isinstance(runtime.get(key),str) and bool(runtime[key]) and measured_runtime.get(key)==runtime[key]
                for key in ("torch","python","cuda")),"Graph and DOMINO Python/Torch runtime differ or are missing")
    require(startup.get("cuda_graph") is True and startup.get("graph_captures")==1 and
            startup.get("graph_eager_parity") is True and
            startup.get("shared_graph_module")=="dynamicwam.inference.causal_spatial_graph",
            "DOMINO did not run the parity-checked production Graph")
    fast=[row for row in report["requests"] if row.get("installed") and not row.get("bootstrap_only")]
    require(bool(fast) and all(row.get("policy_diagnostics",{}).get("cuda_graph") is True and
            row["policy_diagnostics"].get("graph_captures")==1 for row in fast),"DOMINO fast path used a fallback or recapture")
    refreshed=False
    for index,row in enumerate(fast):
        source=row["policy_diagnostics"]
        if source.get("slow_launched") is not True:continue
        for later in fast[index+1:]:
            consumed=later["policy_diagnostics"]
            if (consumed.get("source_publication",-1)>source.get("source_publication",-1)>=0 and
                    consumed.get("source_sim_time",float("-inf"))>=row["observation_physics_s"]-1e-8):
                refreshed=True;break
    require(refreshed,"DOMINO lacks a live slow-memory launch followed by a newly consumed publication")
    validate_causal_spatial_report(report)
    # The committed sidecar binds the complete ffprobe frame decode to MP4 SHA.
    # Revalidate those hashes and timestamps; no GPU access or repeated decode.
    require(validate_video_artifacts(report),"native MP4/full-decode sidecar validation failed")
    require(complete["video"]==report["video"],"DOMINO completion video mismatch")
    return report


def validate_acceptance(path,*,identity=None):
    from dynamicwam.inference.causal_spatial_policy import EXPECTED_RUNTIME_CONTRACT
    a=read(path);identity=source_identity() if identity is None else identity
    require(a.get("format")==FORMAT and a.get("passed") is True,"root complete acceptance is required")
    for key in ("git_head","source_sha256"):require(a.get(key)==identity[key],"acceptance source identity mismatch")
    require(set(a["configs"])==set(PHASES),"all three formal recipes must be bound")
    recipes={};prior=None
    for phase in PHASES:
        ref=a["configs"][phase];config=ROOT/"configs/causal_spatial"/(phase+".yaml")
        require(Path(ref["path"]).resolve()==config.resolve(),"unexpected formal recipe path")
        recipe=artifact(ref)
        require(recipe["phase"]==phase and recipe["steps"]==STEPS[phase] and recipe["diagnostic"] is False,
                "formal phase/budget/diagnostic contract mismatch")
        require(recipe["microbatch"]*recipe["accumulation"]*8==128,"formal global batch must be 128")
        require(recipe["runtime_contract"]==EXPECTED_RUNTIME_CONTRACT,"formal recipe differs from the production runtime contract")
        require(Path(recipe["run_dir"]).resolve().is_relative_to(SSD.resolve()),"formal training output outside SSD")
        require(not Path(recipe["run_dir"]).exists(),"preserve existing formal training output")
        if prior is None:require(not recipe.get("continue_from"),"first formal phase must use independent audited initialization")
        else:
            expected=Path(prior["run_dir"])/"exports"/f"causal_spatial_step_{prior['steps']}.pt"
            require(Path(recipe["continue_from"]).resolve()==expected.resolve(),"phase continuation must use exact previous formal final export")
            for key in ("model","runtime_contract","data","initial_checkpoint"):
                require(recipe[key]==prior[key],"phase contract/data/initialization changed")
        recipes[phase]=recipe;prior=recipe
    evidence=a["evidence"]
    for key in ("data","cpu"):
        value=artifact(evidence[key]);require(value.get("passed") is True and value.get("gpu_used") is False,key+" gate failed")
    cpu=artifact(evidence["cpu"])
    require(cpu.get('source_sha256')==identity['source_sha256'],'actual CPU model/data acceptance is stale')
    require(cpu.get("fast_strict_roundtrip") is True and cpu.get("teacher_normalization_equal") is True,"actual CPU model/data gate incomplete")
    require(set(evidence["smokes"])==set(PHASES),"all three actual training smokes required")
    smoke_identity=training_smoke_identity(a,identity)
    checkpoint=None
    for phase in PHASES:
        bundle=evidence["smokes"][phase];value=artifact(bundle["audit"])
        smoke_config=ROOT/"configs/causal_spatial"/(phase+"_smoke.yaml")
        smoke=artifact(value["config"])
        require(Path(value["config"]["path"]).resolve()==smoke_config.resolve(),"wrong smoke recipe")
        require(smoke["steps"]==4 and smoke["diagnostic"] is True,"smoke must have four actual updates")
        for key in ("model","runtime_contract","data","microbatch","accumulation","lr","loss","teacher_checkpoint"):
            require(smoke[key]==recipes[phase][key],"smoke differs from formal recipe: "+key)
        checkpoint=check_export(value,smoke,diagnostic=True)
        check_guard(bundle,smoke_identity,expected_stage="smoke",workers=8)
    graph=evidence["graph"];result=artifact(graph["result"])
    check_guard(graph,identity,expected_stage="graph",workers=1)
    require(result.get("passed") is True and result.get("source_sha256")==identity["source_sha256"],"actual Graph gate failed or stale")
    require(result["checkpoint"]["sha256"]==checkpoint["sha256"],"Graph must test actual final joint-smoke export")
    require(result["cpu_strict_reload"].get("artifact_roundtrip") is True and
            result["full_fast_gpu_parity"].get("passed") is True and result["background_memory"].get("passed") is True,
            "Graph full/fast/actual-memory acceptance incomplete")
    require(result["graph"].get("captures")==1 and result["graph"].get("replays",0)>=20,"actual Graph mutation coverage incomplete")
    require(result["graph"].get("production_shared_graph") is True and
            result["graph"].get("shared_graph_module")=="dynamicwam.inference.causal_spatial_graph",
            "Graph acceptance must use the production shared implementation")
    require(result["plan_causality"].get("belief_invariant") is True and
            result["plan_causality"].get("invalid_or_outside_overlap_exact_noop") is True,"plan causality acceptance missing")
    require(result["peak_memory"].get("num_ooms")==0,"Graph OOM receipt failed")
    domino=evidence["domino"];last_guard,domino_manifest=check_guard(domino,identity,expected_stage="smoke",workers=2)
    check_domino(domino,checkpoint,result.get("runtime",{}),domino_manifest)
    return dict(acceptance_sha256=digest(path),identity=identity,recipes=recipes,previous_guard=str(last_guard))


def check_unchanged(acceptance,context):
    require(digest(acceptance)==context["acceptance_sha256"],"acceptance changed during sequence")
    require(source_identity()==context["identity"],"source changed during sequence")
    value=read(acceptance)
    for phase in PHASES:artifact(value["configs"][phase])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance",type=Path,required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--detach",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    require(bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",args.label)),"invalid unique sequence label")
    directory=OUT/(args.label+"_sequence")
    if args.detach:
        require(not args.worker,"detach and worker conflict")
        # Validation is CPU-only; no training controller exists before this gate.
        validate_acceptance(args.acceptance)
        directory.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),"--acceptance",str(args.acceptance.resolve()),
                 "--label",args.label,"--worker"]
        with (directory/"controller.log").open("x") as log:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(directory/"launch.json",dict(pid=child.pid,command=command));print(json.dumps(dict(pid=child.pid,status=str(directory/"status.json"))))
        return
    require(args.worker and directory.is_dir(),"use --detach for a fresh durable sequence")
    status=dict(format="causal_spatial_formal_sequence_v1",state="validating",passed=False,pid=os.getpid(),
                retry_policy="none",recurring_inspection=False,phases=[])
    active=None
    def update(**values):
        status.update(values,updated_at=datetime.now(timezone.utc).isoformat());save(directory/"status.json",status)
    def stop(number,_frame):raise RuntimeError("sequence interrupted: "+str(number))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        update();context=validate_acceptance(args.acceptance)
        update(acceptance=str(args.acceptance.resolve()),acceptance_sha256=context["acceptance_sha256"],**context["identity"])
        previous=context["previous_guard"];last_export=None
        for phase in PHASES:
            check_unchanged(args.acceptance,context);recipe=context["recipes"][phase]
            require(not Path(recipe["run_dir"]).exists(),"formal training output appeared before launch")
            if last_export is not None:
                require(Path(recipe["continue_from"]).resolve()==Path(last_export["path"]).resolve(),"previous formal export path changed")
                artifact(last_export,json_value=False)
            label=args.label+"_"+phase;run=OUT/label
            run.mkdir(exist_ok=False)
            command=[sys.executable,str(ROOT/"scripts/run_causal_spatial_guarded.py"),"--mode","training",
                     "--label",label,"--config",str(ROOT/"configs/causal_spatial"/(phase+".yaml")),
                     "--previous-guard",previous,"--worker"]
            update(state="phase_running",active_phase=phase)
            with (run/"controller.log").open("x") as log:
                active=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
                                        CAUSAL_SPATIAL_FORMAL_ACCEPTANCE=str(args.acceptance.resolve())),
                                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(active_controller_pid=active.pid)
                code=active.wait();active=None
            require(code==0,"formal controller failed; no retry or later phase")
            controller=read(run/"status.json")
            require(controller.get("state")=="completed" and controller.get("passed") is True,"formal controller is not complete")
            bundle={key:dict(path=str(run/"guard"/filename),sha256=digest(run/"guard"/filename))
                    for key,filename in (("guard_status","status.json"),("guard_manifest","manifest.json"))}
            check_guard(bundle,context["identity"],expected_stage="formal",workers=8)
            audit=read(run/"export_audit.json")
            require(audit["config"]["sha256"]==read(args.acceptance)["configs"][phase]["sha256"],"formal export audit recipe mismatch")
            last_export=check_export(audit,recipe,diagnostic=False)
            if recipe.get("continue_from"):
                require(audit["continue_from"]["sha256"]==status["phases"][-1]["export"]["sha256"],"formal export lineage mismatch")
            check_unchanged(args.acceptance,context)
            status["phases"].append(dict(phase=phase,guard=str(run/"guard"),audit=str(run/"export_audit.json"),export=last_export))
            previous=str(run/"guard");update(state="phase_accepted")
        update(state="completed",passed=True,active_phase=None,active_controller_pid=None)
    except BaseException:
        update(state="failed_no_retry",error=traceback.format_exc())
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=180.)
            except subprocess.TimeoutExpired:update(cleanup_pending=True)
        raise


if __name__=="__main__":main()
