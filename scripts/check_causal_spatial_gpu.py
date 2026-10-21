#!/usr/bin/env python3
"""Real-data/full-size M+b+Z/Action GPU acceptance, launched by a fresh guard.

No DOMINO rendering, camera, IPC, physics, task performance or 1x capacity claim.
This script never launches another job, retries, or changes a GPU/driver setting.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from dynamicwam.inference.causal_spatial_graph import JointGraph, map_tensors, copy_tensors


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def guarded_context(output, config=None):
    run=Path(os.environ.get("WAM_TRAINING_RUN_DIR","/missing"))
    require((run/"manifest.json").is_file(),"must run inside the owned guarded supervisor")
    manifest=json.loads((run/"manifest.json").read_text())
    require(manifest.get("device_lifetime_mode")=="fds_before_telemetry_v1",
            "requires current FD-before-telemetry lifecycle")
    guard_ids=manifest.get("guard_gpu_uuids",[])
    require(len(guard_ids)==len(set(guard_ids))==8,"temporary guard must cover all eight adapters")
    require(set(guard_ids)=={row["uuid"] for row in manifest.get("proc_gpu_inventory",[])},
            "guard coverage must match the supervisor's actual eight-adapter inventory")
    require(manifest.get("stage")=="graph" and manifest.get("preflight_only") is False,
            "actual graph worker requires an execution-stage guard manifest")
    require(manifest.get("device_guard_sha256")==digest(ROOT/"scripts/device_lifetime_guard.py"),
            "current guard helper differs from supervisor")
    storage=Path(os.environ.get("WAM_SSD_ROOT","/missing")).resolve()
    require(output.resolve().is_relative_to(storage),"output must stay on the supervisor's verified SSD")
    compute=os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")
    require(len(compute)==1 and compute==manifest.get("worker_gpu_uuids") and set(compute)<=set(guard_ids),
            "graph worker must use exactly the supervisor's one guarded compute UUID")
    authorization=Path(manifest["authorization"])
    require(digest(authorization)==manifest["authorization_sha256"],"guard authorization changed")
    auth=json.loads(authorization.read_text())
    require(auth.get("guard_sha256")==manifest["device_guard_sha256"] and
            auth.get("source_sha256")==manifest["source_sha256"] and
            auth.get("device_lifetime_mode")==manifest["device_lifetime_mode"] and
            set(auth.get("gpu_uuids",[]))==set(guard_ids),"guard authorization identity mismatch")
    if config is not None:
        require(auth.get("config_sha256")==digest(config),"config differs from guarded authorization")
    events=json.loads((run/"device_guard_events.json").read_text())
    opened=[row["uuid"] for row in events if row.get("event")=="after_open"]
    require(len(opened)==8 and set(opened)==set(guard_ids) and
            not any(row.get("event") in ("before_close","after_close") for row in events),
            "all eight temporary guard descriptors must be open before the GPU worker")
    return manifest,run


def fingerprint(value):
    import torch
    h=hashlib.sha256()
    def visit(tensor):
        h.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
        h.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        return tensor
    map_tensors(value,visit)
    return h.hexdigest()


def profile(call, count):
    import torch
    samples=[]; result=None
    for _ in range(count):
        torch.cuda.current_stream().synchronize(); start=time.perf_counter()
        result=call(); torch.cuda.current_stream().synchronize()
        samples.append((time.perf_counter()-start)*1000)
    ordered=sorted(samples)
    def percentile(q):
        position=(len(ordered)-1)*q; lo=int(position); hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo])
    return result,dict(samples_ms=samples,count=len(samples),p50_ms=percentile(.5),
                       p95_ms=percentile(.95),max_ms=max(samples))


def output_parity(actual, expected, *, atol, rtol):
    import torch
    errors={}
    for name in ("actions","future_latent","belief_state"):
        a,e=actual[name],expected[name]
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(e).all()),f"nonfinite {name}")
        errors[name]=float((a.float()-e.float()).abs().max().item())
        torch.testing.assert_close(a,e,atol=atol,rtol=rtol)
    return errors


INPUT_KEYS=("source_condition_latent","source_text_embeddings","source_time","source_id",
    "initial_state","text_embeddings","absolute_motion_features","absolute_motion_interval_valid_mask",
    "absolute_motion_acceleration_valid_mask","action_target_offsets","future_target_offsets","observation_age",
    "spatial_rgb","spatial_flow","spatial_valid","spatial_flow_valid","spatial_dt",
    "previous_plan","previous_plan_valid")


def inference_packet(sample, device):
    import torch
    from dynamicwam.models.rtc_prefix import PrefixCondition,validate_prefix_condition
    from dynamicwam.training.data.causal_spatial_dataset import COMMAND_HISTORY_SEMANTICS
    require(sample.get("command_history_semantics")==COMMAND_HISTORY_SEMANTICS==
            "recorded_capture_drive_targets_not_dense_executed_control",
            "command aliases require explicit capture drive-target provenance")
    packet=map_tensors({key:sample[key] for key in INPUT_KEYS},lambda t:t.to(device))
    # Dataset compatibility names carry sparse interval-left capture commands,
    # never a measured state or a dense realized actuator trajectory.
    packet["recorded_capture_commands"]=sample["executed_actions"].to(device)
    packet["command_valid"]=sample["executed_valid"].to(device)
    packet["rtc_prefix"]=PrefixCondition(sample["rtc_prefix"].float().to(device),
                                         sample["rtc_prefix_lengths"].long().to(device))
    validate_prefix_condition(packet["rtc_prefix"],max_prefix=2)
    # Actual future capture times/clean suffix/future latents are NEVER inputs.
    require(torch.allclose(packet["future_target_offsets"].double(),
                           packet["action_target_offsets"][:,[7,15]].double(),atol=1e-8,rtol=0),
            "future input timing must be nominal requested action endpoints 8/16")
    return packet


def altered_cases(packet,noise):
    """Interface perturbations; these are not claimed physical rollout samples."""
    import torch
    mutations=(
        ("rgb",lambda p,n:p["spatial_rgb"].add_(.03125)),
        ("flow",lambda p,n:p["spatial_flow"].add_(.125)),
        ("all_images_invalid",lambda p,n:p["spatial_valid"].zero_()),
        ("flow_validity",lambda p,n:p["spatial_flow_valid"].logical_not_()),
        ("elapsed_time",lambda p,n:p["spatial_dt"].mul_(.75)),
        ("capture_commands",lambda p,n:p["recorded_capture_commands"].add_(.125)),
        ("command_validity",lambda p,n:p["command_valid"].logical_not_()),
        ("native_state",lambda p,n:p["initial_state"].add_(.125)),
        ("native_motion",lambda p,n:p["absolute_motion_features"].add_(.125)),
        ("native_motion_validity",lambda p,n:p["absolute_motion_interval_valid_mask"].zero_()),
        ("memory_age",lambda p,n:p["observation_age"].add_(.16)),
        ("action_target_time",lambda p,n:p["action_target_offsets"].add_(.04)),
        ("future_target_time",lambda p,n:p["future_target_offsets"].add_(.04)),
        ("prefix_values",lambda p,n:p["rtc_prefix"].actions.add_(.125)),
        ("empty_prefix",lambda p,n:p["rtc_prefix"].lengths.zero_()),
        ("action_noise",lambda p,n:n[0].add_(.125)),
        ("future_noise",lambda p,n:n[1].add_(.125)),
    )
    for name,mutate in mutations:
        p=map_tensors(packet,lambda t:t.clone());n=map_tensors(noise,lambda t:t.clone())
        mutate(p,n);yield name,p,n
    p=map_tensors(packet,lambda t:t.clone())
    p["previous_plan"][:,2:4]=noise[0][:,2:4]
    p["previous_plan_valid"].zero_();p["previous_plan_valid"][:,2:4]=True
    yield "explicit_short_queue_overlap",p,noise
    p=map_tensors(p,lambda t:t.clone());p["previous_plan_valid"].zero_()
    yield "overlap_removed_values_retained",p,noise
    p=map_tensors(packet,lambda t:t.clone())
    p["previous_plan"].copy_(noise[0]);p["previous_plan_valid"].fill_(True)
    p["previous_plan_valid"][:,2:4]=False
    yield "only_outside_short_overlap_valid",p,noise
    yield "return_to_original_h2",packet,noise


def concurrent_probe(model,graph,packet,memory,next_packet,noise,encode,eager,args):
    import torch
    trace={};started=threading.Event();producer=torch.cuda.Stream()
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    anchor=torch.cuda.Event(enable_timing=True);anchor.record();anchor.synchronize()
    original=fingerprint(memory)
    def produce():
        try:
            with torch.inference_mode(),torch.cuda.stream(producer),torch.autocast("cuda",dtype=torch.bfloat16):
                producer.wait_event(anchor);begin.record(producer)
                trace["start_wall_s"]=time.perf_counter();started.set()
                value=encode(next_packet);end.record(producer);end.synchronize()
                trace.update(memory=value,end_wall_s=time.perf_counter(),event_ready=end.query())
        except BaseException:
            trace["error"]=traceback.format_exc();started.set()
    thread=threading.Thread(target=produce,name="bounded-observation-memory-probe",daemon=True)
    thread.start();require(started.wait(30.),"memory producer did not begin")
    calls=[]
    for index in range(5):
        unpublished="memory" not in trace
        before=time.perf_counter()
        actual=graph.replay(packet,memory,noise);torch.cuda.current_stream().synchronize()
        calls.append(dict(index=index,start_wall_s=before,end_wall_s=time.perf_counter(),
                          slow_unpublished_at_start=unpublished))
        require(all(bool(torch.isfinite(actual[k]).all()) for k in actual),"nonfinite concurrent sampler")
    thread.join(120.);require(not thread.is_alive(),"memory producer exceeded bounded probe timeout")
    require("error" not in trace,trace.get("error","memory producer failed"))
    require(trace.get("event_ready") is True,"memory was not complete at publication")
    require(any(c["slow_unpublished_at_start"] and trace["start_wall_s"]<=c["start_wall_s"]<trace["end_wall_s"]
                for c in calls),"no actual fast request began before slow completion")
    require(fingerprint(memory)==original,"producer or fast sampler mutated published old memory")
    fresh=trace.pop("memory")
    expected=eager(packet,fresh,noise);actual=graph.replay(packet,fresh,noise)
    errors=output_parity(actual,expected,atol=args.atol,rtol=args.rtol)
    trace.update(passed=True,calls=calls,old_memory_unchanged=True,new_memory_consumed=True,
        refreshed_parity=errors,captures=graph.captures,slow_gpu_ms=begin.elapsed_time(end),
        slow_wall_ms=(trace["end_wall_s"]-trace["start_wall_s"])*1000,
        overlap_claim="fast request interval overlaps actual producer lifetime; kernel simultaneity not proved",
        source_semantics="second genuine dataset observation, no fabricated sim advance or valid rollout claim")
    return trace


def run(args, report):
    import torch
    import yaml
    from dynamicwam.inference.causal_spatial_loader import build_from_initialization,load_full,load_fast
    from dynamicwam.training.train.causal_spatial_trainer import build_dataset
    from dynamicwam.training.data.causal_spatial_dataset import collate_causal_spatial
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from supervise_motionkv_training import source_files_and_hash
    require(source_files_and_hash(ROOT)[1]==report["source_sha256"],"source changed after guarded launch")
    recipe=yaml.safe_load(args.config.read_text())
    contract=recipe["runtime_contract"]
    require(all(contract.get(k)==v for k,v in {"dt":.16,"H":2,"R":2,"chunk_size":16,"K":10,"simulation_rate":1.}.items()),
            "checker accepts only the frozen primary .16/H2/R2/chunk16/K10/1x candidate")
    torch.manual_seed(args.seed);torch.set_num_threads(1)
    report["runtime_contract"]=contract
    checkpoint=args.checkpoint
    if checkpoint:
        model,payload=load_full(checkpoint,"cpu",allow_diagnostic=True)
        require(payload["runtime_contract"]==contract,"checkpoint and requested runtime contract differ")
        fast_path=checkpoint.with_name(checkpoint.name.replace("causal_spatial_step_","causal_spatial_fast_step_"))
        require(fast_path!=checkpoint and fast_path.is_file(),"actual corresponding fast export required")
        fast,fast_payload=load_fast(fast_path,"cpu",allow_diagnostic=True)
        require(fast_payload["runtime_contract"]==contract,"fast runtime contract differs")
        report["checkpoint"]={"path":str(checkpoint),"sha256":digest(checkpoint),"global_step":payload["global_step"],
                              "diagnostic_only":payload.get("diagnostic_only"),"fast_sha256":digest(fast_path)}
        del payload,fast_payload
    else:
        model,_=build_from_initialization(recipe["initial_checkpoint"],recipe["model"],"cpu",formal_init=True)
        fast=model.make_fast_model()
        report["checkpoint"]={"path":recipe["initial_checkpoint"],"sha256":digest(recipe["initial_checkpoint"]),
                              "kind":"audited native initialization plus deterministically initialized new modules"}
    full_state=model.fast_state_dict();fast_state=fast.state_dict()
    require(full_state.keys()==fast_state.keys() and all(torch.equal(v,fast_state[k]) for k,v in full_state.items()),
            "strict full/fast CPU parameters disagree")
    report["cpu_strict_reload"]={"passed":True,"full_strict":True,"fast_strict":True,"full_fast_equal":True,
        "artifact_roundtrip":bool(checkpoint),"initialization_only_limit":None if checkpoint else "new V4 modules checked in memory; no V4 export exists yet"}
    del full_state,fast_state;gc.collect()
    dataset=build_dataset(recipe,"validation")
    require(len(dataset)>=2,"at least two actual dataset samples required")
    indices=(0,max(1,len(dataset)//2))
    samples=[collate_causal_spatial([dataset[index]]) for index in indices]
    report["actual_data"]={"indices":list(indices),"dataset_length":len(dataset),"fixture":"real physical-grid samples",
        "prefix_source":"expert H2 for diagnostic fixture only", "source_times":[s["source_time"].tolist() for s in samples],
        "current_times":[s["current_time"].tolist() for s in samples],
        "manifests":{key:digest(Path(recipe["data"][key])/"manifest.json") for key in ("physical_grid_root","future_latent_root")}}
    require(torch.cuda.is_available(),"actual CUDA device required; no CPU surrogate")
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    model=model.eval().requires_grad_(False).to("cuda:0")
    packets=[inference_packet(s,"cuda:0") for s in samples]
    generator=torch.Generator(device="cuda:0").manual_seed(args.seed)
    noise=(torch.randn(samples[0]["clean_actions"].shape,device="cuda:0",generator=generator),
           torch.randn(samples[0]["clean_future"].shape,device="cuda:0",generator=generator))
    schedule=BridgeSchedule.from_flow_match(action_steps=10)
    sigmas=torch.cat((schedule.action_sigmas,torch.zeros(1))).to("cuda:0")
    require(len(sigmas)==11,"actual K10 Euler trajectory requires eleven sigma endpoints")
    report["schedule"]={"sigmas":sigmas.cpu().tolist(),"joint_Z_Action_steps":10,"old_source_plan_generation":False}
    packet=packets[0]
    def encode(p):return model.encode_memory(p["source_condition_latent"],p["source_text_embeddings"],p["source_time"],p["source_id"])
    def eager(p,m,n):return model.sample(p,action_noise=n[0],future_noise=n[1],sigmas=sigmas,memory=m)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        memory=encode(packet);eager(packet,memory,noise);torch.cuda.synchronize()
        memory,report["memory_encode"]=profile(lambda:encode(packet),args.samples)
        inputs_before=fingerprint((packet,memory,noise))
        _,report["belief_only"]=profile(lambda:model.prepare_belief(dict(packet,noisy_actions=noise[0])),args.samples)
        baseline,report["eager_belief_joint_K10"]=profile(lambda:eager(packet,memory,noise),args.samples)
        baseline=map_tensors(baseline,lambda t:t.clone())
        _,report["sequential_M_b_Z_K10"]=profile(lambda:eager(packet,encode(packet),noise),args.samples)
        graph=JointGraph(model,packet,memory,noise,sigmas)
        actual,report["graph_belief_joint_K10"]=profile(lambda:graph.replay(packet,memory,noise),args.samples)
        report["cases"]=[dict(name="baseline_h2",errors=output_parity(actual,baseline,atol=args.atol,rtol=args.rtol))]
        for name,p,n in altered_cases(packet,noise):
            expected=eager(p,memory,n);actual=graph.replay(p,memory,n)
            errors=output_parity(actual,expected,atol=args.atol,rtol=args.rtol)
            if name in {"explicit_short_queue_overlap","overlap_removed_values_retained",
                        "only_outside_short_overlap_valid"}:
                # b describes the observed current state. An editable plan
                # conditions A/Z only, even when a trained plan gate is open.
                torch.testing.assert_close(expected["belief_state"],baseline["belief_state"],atol=0,rtol=0)
                torch.testing.assert_close(actual["belief_state"],baseline["belief_state"],atol=args.atol,rtol=args.rtol)
                if name!="explicit_short_queue_overlap":
                    output_parity(expected,baseline,atol=0,rtol=0)
            length=int(p["rtc_prefix"].lengths[0].item())
            torch.testing.assert_close(actual["actions"][:,:length],p["rtc_prefix"].actions[:,:length],atol=0,rtol=0)
            report["cases"].append(dict(name=name,errors=errors,hard_prefix_exact=True))
        report["plan_causality"]={"belief_invariant":True,"permitted_editable_rows":[2,3],
                                  "invalid_or_outside_overlap_exact_noop":True}
        report["background_memory"]=concurrent_probe(model,graph,packet,memory,packets[1],noise,encode,eager,args)
        require(fingerprint((packet,memory,noise))==inputs_before,"caller-owned inputs changed")
        report["inputs_unchanged"]=True
        report["graph"]={"captures":graph.captures,"replays":graph.replays,"new_memory_recaptured":False}
        require(graph.captures==1,"refresh must reuse the actual captured graph")
        # Release the full model's GPU allocation before testing the actual fast
        # export. The externally published M remains valid and owned separately.
        graph.graph.reset();del graph,actual,expected
        model.to("cpu");gc.collect();torch.cuda.synchronize()
        fast=fast.eval().requires_grad_(False).to("cuda:0")
        fast_output=fast.sample(packet,action_noise=noise[0],future_noise=noise[1],sigmas=sigmas,memory=memory)
        report["full_fast_gpu_parity"]={"passed":True,"errors":output_parity(fast_output,baseline,atol=args.atol,rtol=args.rtol)}
    report["peak_memory"]={"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved(),
                           "num_ooms":torch.cuda.memory_stats().get("num_ooms",0)}
    require(source_files_and_hash(ROOT)[1]==report["source_sha256"],"source changed during actual acceptance")
    report["passed"]=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--samples",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20261021)
    parser.add_argument("--atol",type=float,default=.002)
    parser.add_argument("--rtol",type=float,default=.002)
    args=parser.parse_args()
    require(args.samples>=2 and args.atol>=0 and args.rtol>=0,"invalid diagnostic samples/tolerance")
    require(not args.output.exists(),"preserve previous acceptance/failure receipts")
    manifest,guard=guarded_context(args.output,args.config)
    report=dict(kind="causal_spatial_actual_gpu_v1",passed=False,config=str(args.config.resolve()),
        config_sha256=digest(args.config),source_sha256=manifest["source_sha256"],guard=str(guard),
        tolerance={"atol":args.atol,"rtol":args.rtol},
        boundaries={"native_full_size":True,"real_dataset":True,"camera_render_IPC_physics":False,
                    "full_DOMINO":False,"realtime_acceptance":False,"performance_claim":False},
        timing="synchronized current-stream wall time; preprocessing and physical execution excluded",
        final_guard_release_required=True)
    try:
        run(args,report)
    except BaseException:
        report["error"]=traceback.format_exc();raise
    finally:
        save(args.output,report)
        print(json.dumps(dict(output=str(args.output),passed=report["passed"],full_DOMINO=False)))


if __name__=="__main__":main()
