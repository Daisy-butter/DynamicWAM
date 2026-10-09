#!/usr/bin/env python3
"""Guarded native-action multirate training, early gates, and paired evaluation.

Run from an immutable source snapshot. Failure stops the chain. Existing
training directories are never overwritten or implicitly resumed. Small
probes establish functionality and timing, not statistical non-inferiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.models.flow_schedule import BridgeSchedule
from dynamicwam.training.train.common import get_export_dir, get_run_dir
from run_official_temporal_pipeline import eval_summary, write_json

PACKET = Path("/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/feedback_preflight_20261003/synthetic_packet.npz")
TASKS = ("place_shoe", "place_fan", "adjust_bottle")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_state(pid):
    path = Path("/proc") / str(int(pid))
    try:
        # comm may contain spaces/parentheses; state follows the final ')'.
        state = (path / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
        command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return state, command
    except FileNotFoundError:
        return None, ""


def source_identity(run_root):
    files = {}
    for folder in ("src", "scripts", "configs", "manifests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path.relative_to(ROOT).as_posix()] = sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    launcher_manifest = run_root / "source_manifest.json"
    return {"source": str(ROOT), "sha256": digest, "files": files,
            "launcher_manifest_sha256": sha256(launcher_manifest) if launcher_manifest.is_file() else None}


def assert_gpus_idle(gpus):
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    by_uuid = {line.split(",")[1].strip(): line.split(",")[0].strip()
               for line in inventory.splitlines() if "," in line}
    present = set(by_uuid.values())
    if not set(gpus).issubset(present):
        raise RuntimeError(f"requested GPUs {gpus} are not present: {sorted(present)}")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                      "--format=csv,noheader,nounits"], text=True)
    conflicts = []
    for line in active.splitlines():
        parts = line.split(",", 2)
        if len(parts) == 3 and by_uuid.get(parts[0].strip()) in gpus:
            conflicts.append(line)
    if conflicts:
        raise RuntimeError("selected GPUs have active compute processes; nothing was stopped: " + "; ".join(conflicts))


def checkpoint_evidence(path, training):
    """Check schedule, teacher identity, finite tensors, and frozen video equality."""
    import torch
    torch.set_num_threads(1)
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = int(training["max_steps"])
    if payload.get("global_step") != step:
        raise RuntimeError(f"wrong checkpoint step: {path}")
    modules = payload["modules"]
    contract = payload["config"]["runtime_contract"]
    native = contract.get("native_multirate", {})
    phase, fast_steps = training["native_phase"], int(training["native_fast_steps"])
    if modules.get("architecture") != "native" or not modules.get("native_multirate"):
        raise RuntimeError("checkpoint is not the native-capacity architecture")
    if native.get("phase") != phase or native.get("fast_steps") != fast_steps or not native.get("world_aware"):
        raise RuntimeError("checkpoint native phase/sampler differs from training recipe")
    if native.get("source_bridge") != "immutable_official" or native.get("world_refresh_actions") != 16:
        raise RuntimeError("checkpoint lacks the immutable official source/budget contract")
    slow = modules["schedule"]
    if slow != contract["schedule"] or slow["action_steps"] != 10 or slow["video_steps"] != 2:
        raise RuntimeError("slow schedule is not the official 10/2 schedule")
    expected_fast = BridgeSchedule.from_flow_match(**slow).with_action_steps(fast_steps).to_config()
    if modules.get("fast_schedule") != expected_fast or modules.get("fast_steps") != fast_steps:
        raise RuntimeError("independent fast schedule metadata mismatch")
    if contract["max_age_actions"] != 12 or contract["execute_horizon"] != 4 or contract.get("max_age_seconds") is not None:
        raise RuntimeError("deployment age/cadence differs from the trained native regime")
    teacher_path = Path(training["model"]["native_teacher_checkpoint"])
    teacher_identity = native["teacher"]
    teacher_hash = sha256(teacher_path)
    if Path(teacher_identity["path"]).resolve() != teacher_path.resolve() or teacher_identity["sha256"] != teacher_hash:
        raise RuntimeError("official teacher identity changed")
    official = torch.load(teacher_path, map_location="cpu", weights_only=True, mmap=True)
    if official["config"]["action_normalization"] != payload["config"]["action_normalization"]:
        raise RuntimeError("action normalization changed")
    state = payload["model"]
    for key, value in state.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite exported tensor: {key}")
    frozen_count = 0
    for key, value in official["model"].items():
        if key.startswith(("action_expert.", "absolute_motion_tokens.")):
            continue
        incoming = state.get("backbone." + key)
        if incoming is None or incoming.shape != value.shape or not torch.equal(incoming, value.to(incoming.dtype)):
            raise RuntimeError(f"frozen official video/backbone tensor changed: {key}")
        frozen_count += 1
    if not frozen_count or not any(k.startswith("fast_conditioner.world_q.") for k in state):
        raise RuntimeError("missing frozen video or learned world router")
    return {"path": str(path), "sha256": sha256(path), "global_step": step,
            "phase": phase, "fast_steps": fast_steps, "teacher": teacher_identity,
            "frozen_video_tensors_equal": frozen_count, "runtime_contract": contract}


def validation_evidence(path, *, require_ages):
    marker = "Native multirate validation: "
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if marker in line:
            rows.append(json.loads(line.split(marker, 1)[1]))
    if not rows:
        raise RuntimeError(f"no native validation record in {path}")
    final = rows[-1]
    if not all(math.isfinite(float(v)) for v in final.values()):
        raise RuntimeError("native validation contains nonfinite metrics")
    if require_ages:
        for age in (0, 4, 8, 12):
            if final.get(f"val/age_{age}/samples", 0) <= 0:
                raise RuntimeError(f"pilot validation did not cover deployment age {age}")
    return final


def latency_evidence(path, checkpoint):
    report = json.loads(Path(path).read_text())
    if Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise RuntimeError("latency report used another checkpoint")
    solver = report.get("solver", {})
    if (solver.get("overridden") is not False or solver.get("quality_evaluation_allowed") is not True
            or solver.get("trained_fast_steps") != solver.get("measured_fast_steps")):
        raise RuntimeError("pipeline gate refuses untrained or unspecified solver overrides")
    bridge = report["slow_bridge"]["mean_ms"]
    fast = report["fast_query"]["mean_ms"]
    sync = report["sync_sampler"]["mean_ms"]
    p95 = report["fast_query"]["p95_ms"]
    matched = report.get("sync_sampler_matched_precision")
    if not isinstance(matched, dict):
        raise RuntimeError("pipeline gate requires the BF16 matched-precision official baseline")
    sync_matched = matched["mean_ms"]
    values = (bridge, fast, sync, sync_matched, p95)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise RuntimeError("invalid latency measurement")
    if report["ratio"] != 4 or report["execute_horizon"] != 4 or not report.get("native_multirate"):
        raise RuntimeError("latency report used another architecture/cadence")
    if any(len(report[key].get("samples_ms", [])) < 20 for key in (
            "slow_bridge", "fast_query", "sync_sampler", "sync_sampler_matched_precision")):
        raise RuntimeError("latency gate requires at least twenty samples per path")
    measured, baseline = (bridge / 4 + fast) / 4, sync / 16
    matched_baseline = sync_matched / 16
    result = {"multirate_ms_per_action": measured, "official_h16_ms_per_action": baseline,
              "ratio_to_official_h16": measured / baseline, "fast_p95_ms": p95,
              "matched_precision_official_h16_ms_per_action": matched_baseline,
              "ratio_to_matched_precision_official_h16": measured / matched_baseline,
              "required_ratio_at_most": .8, "required_fast_p95_below_ms": 100.,
              "required_matched_precision_ratio_at_most": .9,
              "published_precision_gate_passed": measured <= .8 * baseline,
              "matched_precision_gate_passed": measured <= .9 * matched_baseline,
              "passed": measured <= .8 * baseline and measured <= .9 * matched_baseline and p95 < 100.,
              "solver": solver,
              "scope": "fixed synthetic input, serial model service time; precision baselines reported separately; not a success-rate or end-to-end gate"}
    if not result["passed"]:
        raise RuntimeError("latency gate failed; retrain a changed solver explicitly: " + json.dumps(result))
    return result


def probe_evidence(path):
    rows = json.loads(Path(path).read_text())
    if len(rows) != 6 or {(r["arm"], r["task"]) for r in rows} != {(a, t) for a in ("sync", "multirate") for t in TASKS}:
        raise RuntimeError("pilot probe did not produce all six task/arm records")
    for row in rows:
        if row["return_code"] != 0 or not row.get("paired_episode_seeds_match") or len(row["episodes"]) != 2:
            raise RuntimeError("pilot probe failed or has unmatched/incomplete episodes")
        for field in ("query_ms_mean", "ms_per_executed_action"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) or row[field] <= 0:
                raise RuntimeError("pilot probe lacks finite runtime timing")
    return {"rows": rows, "gate": "functional, finite timing and paired seeds only; six episodes cannot establish quality"}


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.gpus = [s.strip() for s in args.gpus.split(",")]
        if len(self.gpus) != 7 or len(set(self.gpus)) != 7:
            raise ValueError("this recipe requires seven distinct GPU indices")
        # OS lock is released on crash. The manifest and completed phase records
        # persist; a resumed process must still validate every skipped output.
        import fcntl
        self.lock = (self.root / "pipeline.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = source_identity(self.root)
        status_path = self.root / "status.json"
        if status_path.exists():
            if not args.resume_completed:
                raise FileExistsError("status exists; use --resume-completed after inspecting it")
            self.state = json.loads(status_path.read_text())
            old_identity = json.loads((self.root / "pipeline_source_manifest.json").read_text())
            if old_identity != fingerprint:
                raise RuntimeError("source snapshot hash changed; refusing mixed-code resume")
            old_child = self.state.get("child_pid")
            if old_child and process_state(old_child)[0] not in (None, "Z"):
                raise RuntimeError(f"recorded child {old_child} is still active")
        else:
            self.state = {"completed": [], "evidence": {}, "started": time.time()}
            write_json(self.root / "pipeline_source_manifest.json", fingerprint)
        self.state.update(status="running", pid=os.getpid(), source=str(ROOT),
                          source_sha256=fingerprint["sha256"], resumed=bool(args.resume_completed))
        self.state.pop("error", None)
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.save()

    def save(self):
        self.state["updated"] = time.time()
        write_json(self.root / "status.json", self.state)

    def run(self, name, command, verify, *, env=None):
        if name in self.state["completed"]:
            evidence = verify()
            if evidence != self.state["evidence"][name]:
                raise RuntimeError(f"completed output identity changed: {name}")
            print(f"verified completed phase {name}", flush=True)
            return evidence
        assert_gpus_idle(self.gpus)
        log = self.root / f"{name}.log"
        if log.exists():
            raise FileExistsError(f"uncompleted phase log exists; no implicit retry: {log}")
        self.state.update(phase=name, command=command, log=str(log))
        self.save()
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env or self.env, stdout=stream, stderr=subprocess.STDOUT)
            self.state["child_pid"] = child.pid
            self.save()
            code = child.wait()
        self.state.pop("child_pid", None)
        if code:
            raise RuntimeError(f"{name} exited {code}: {log}")
        evidence = verify()
        self.state["evidence"][name] = evidence
        self.state["completed"].append(name)
        self.save()
        print(f"completed {name}: {json.dumps(evidence, default=str)[:1500]}", flush=True)
        return evidence

    def train(self, name, config_name, *, reuse=False, require_ages=False):
        config = ROOT / "configs/multirate" / config_name
        train = load_multirate_profile(config).training
        if int(train["per_device_batch_size"]) * int(train["gradient_accumulation_steps"]) * 7 != 14:
            raise RuntimeError("unexpected global batch; recipe budget must be reconsidered")
        export = get_export_dir(train) / f"multirate_step_{train['max_steps']}.pt"
        reused_before = self.state["evidence"].get(name, {}).get("explicitly_reused_smoke", False)
        def verify():
            evidence = checkpoint_evidence(export, train)
            if not reuse and not reused_before:
                evidence["validation"] = validation_evidence(self.root / f"{name}.log", require_ages=require_ages)
            return evidence
        if reuse and name not in self.state["completed"]:
            evidence = verify()
            evidence["explicitly_reused_smoke"] = True
            self.state["evidence"][name] = evidence
            self.state["completed"].append(name)
            self.save()
            return export
        if name in self.state["completed"]:
            evidence = verify()
            if self.state["evidence"][name].get("explicitly_reused_smoke"):
                evidence["explicitly_reused_smoke"] = True
            if evidence != self.state["evidence"][name]:
                raise RuntimeError(f"completed checkpoint evidence changed: {name}")
            return export
        if get_run_dir(train).exists():
            raise FileExistsError(f"training run exists; refusing to overwrite or restart: {get_run_dir(train)}")
        command = [str(ROOT / ".venv/bin/accelerate"), "launch", "--multi_gpu", "--num_processes", "7",
                   "--num_machines", "1", "--mixed_precision", "bf16", "--main_process_port", str(self.args.port),
                   "--dynamo_backend", "no", str(ROOT / "scripts/train_multirate.py"), "--config", str(config)]
        self.run(name, command, verify)
        return export

    def execute(self):
        if self.args.wait_pid:
            self.state.update(phase="wait_previous_training", wait_pid=self.args.wait_pid)
            self.save()
            while True:
                state, command = process_state(self.args.wait_pid)
                if state in (None, "Z"):
                    break
                if "feedback" not in command:
                    raise RuntimeError("wait PID is not the expected feedback job; possible PID reuse")
                time.sleep(20)
                self.save()
        assert_gpus_idle(self.gpus)
        smoke2 = self.train("smoke_stage2", "native_smoke_stage2.yaml", reuse=self.args.skip_smoke)
        smoke3 = self.train("smoke_stage3", "native_smoke_stage3.yaml", reuse=self.args.skip_smoke)
        profile_path = self.root / "latency_preflight.json"
        profile_env = dict(self.env, CUDA_VISIBLE_DEVICES=self.gpus[0])
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/profile_multirate.py"),
                   "--deploy-config", str(ROOT / "configs/absolute_motion_v2.yaml"), "--checkpoint", str(smoke3),
                   "--packet", str(self.args.packet), "--instruction", "place the shoe", "--output", str(profile_path),
                   "--samples", "20", "--ratio", "4", "--baseline-fast-precision"]
        self.run("latency_preflight", command, lambda: latency_evidence(profile_path, smoke3), env=profile_env)
        self.train("pilot_stage2", "native_pilot_stage2.yaml", require_ages=True)
        pilot3 = self.train("pilot_stage3", "native_pilot_stage3.yaml", require_ages=True)
        probe_root = self.root / "pilot_probe"
        same_gpu = ",".join([self.gpus[0]] * 3)
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/eval_multirate_probe.py"),
                   "--checkpoint", str(pilot3), "--run-root", str(probe_root), "--tasks", ",".join(TASKS),
                   "--episodes", "2", "--sync-horizon", "16", "--official-sync", "--scheduling", "controlled_ratio",
                   "--bridge-refresh-ratio", "4", "--serial", "--sync-gpus", same_gpu, "--multirate-gpus", same_gpu]
        self.run("pilot_probe", command, lambda: probe_evidence(probe_root / "probe_report.json"))
        exports = {2: self.train("stage2", "native_stage2_7gpu.yaml", require_ages=True),
                   3: self.train("stage3", "native_stage3_7gpu.yaml", require_ages=True)}
        arms = {}
        for name, stage, horizon in (("official_sync_h16", None, 16), ("official_sync_h4", None, 4),
                                     ("native_stage2_h4", 2, 4), ("native_stage3_h4", 3, 4)):
            options = dict(policy_name="ciwam.adapters.domino.deploy_policy_sync_flow", ckpt_setting=name,
                           execute_horizon=horizon, eval_video_log=False)
            if stage:
                options.update(policy_name="ciwam.adapters.domino.deploy_policy_multirate_flow",
                    multirate_checkpoint=str(exports[stage]), bridge_refresh_ratio=4,
                    multirate_scheduling="controlled_ratio", multirate_reuse_action_noise=False)
            override = self.root / "eval_configs" / f"{name}.json"
            if override.exists() and json.loads(override.read_text()) != options:
                raise RuntimeError("evaluation override changed on resume")
            write_json(override, options)
            arm = self.root / "eval" / name
            arms[name] = arm
            def verify_eval(arm=arm, name=name):
                aggregate = json.loads((arm / "aggregate.json").read_text())
                if (aggregate["requested_episodes"] != 35 * self.args.episodes or len(aggregate["tasks"]) != 35
                        or aggregate.get("completed_tasks") != 35
                        or aggregate.get("reported_episodes") != 35 * self.args.episodes):
                    raise RuntimeError("official evaluation is incomplete")
                for task in aggregate["tasks"]:
                    task_data = json.loads(Path(task["aggregate"]).read_text())
                    if len(task_data.get("episodes", [])) != self.args.episodes:
                        raise RuntimeError("task episode manifest is incomplete")
                report = eval_summary(arms)
                if not report["episode_seeds_match"]:
                    raise RuntimeError("evaluation episode manifests differ across arms")
                for entry in report["arms"].values():
                    rows = entry["timing_rows"]
                    queries = sum(r.get("query_count", 0) for r in rows)
                    steps = sum(r.get("executed_steps", 0) for r in rows)
                    ms = sum(r.get("query_ms_sum", 0) for r in rows)
                    if not queries or not steps or not math.isfinite(ms) or ms <= 0:
                        raise RuntimeError("evaluation lacks finite query/action timing")
                    entry.update(query_ms_mean=ms / queries, ms_per_executed_action=ms / steps,
                                 actual_slow_builds=sum(r.get("actual_slow_builds", 0) for r in rows))
                write_json(self.root / "evaluation_summary.json", report)
                return {"aggregate": str(arm / "aggregate.json"), "sha256": sha256(arm / "aggregate.json"),
                        "requested_episodes": aggregate["requested_episodes"], "success_count": aggregate["success_count"],
                        "episode_seeds_match": True}
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/eval_domino.py"),
                       "--config", str(ROOT / "configs/absolute_motion_v2.yaml"), "--policy-overrides", str(override),
                       "--run-root", str(arm), "--gpus", self.args.gpus, "--episodes-per-task", str(self.args.episodes),
                       "--start-seed", "100000"]
            self.run(name, command, verify_eval)
        self.state.update(status="complete", finished=time.time())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--port", type=int, default=36561)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true", help="explicitly verify and reuse both existing smoke exports")
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    pipeline = Pipeline(args)
    try:
        pipeline.execute()
    except BaseException as error:
        pipeline.state.update(status="failed", error=repr(error), failed=time.time())
        pipeline.save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
