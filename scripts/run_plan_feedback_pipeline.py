#!/usr/bin/env python3
"""Fail-stop, immutable-source plan-feedback training and fixed-seed evaluation.

Run this file from RUN_ROOT/source. No implicit resume, seed replacement, or
automatic retraining is permitted. A six-episode canary is a regression tripwire,
not evidence of statistical non-inferiority. The launcher creates the snapshot.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dynamicwam.config import load_profile
from dynamicwam.config.multirate import load_multirate_profile
from dynamicwam.evaluation.run_domino_level1_suite import (
    load_episode_seed_manifest, valid_task_aggregate, write_json,
)
from make_fixed_seed_domino_eval import generate
from run_native_pipeline import assert_gpus_idle, sha256, source_identity

TASKS = ("place_shoe", "place_fan", "adjust_bottle")
PROFILE_FILES = (
    "src/dynamicwam/models/plan_feedback.py", "src/dynamicwam/models/flow_schedule.py",
    "src/dynamicwam/inference/native_plan_graph.py",
    "src/dynamicwam/runtime/ciwam/wam/plan_feedback_policy.py", "scripts/profile_plan_feedback.py",
    "src/dynamicwam/models/native_wam_teacher.py", "src/dynamicwam/models/causal_world_bridge.py",
)
PACKET = Path("/SSD_DISK/users/wuruihan/DynamicWAM/outputs/evaluation/feedback_preflight_20261003/synthetic_packet.npz")


class EvaluationIncomplete(RuntimeError):
    """Preparation/incomplete output is not a policy failure or a zero score."""


def latency_evidence(path, base_sha, checkpoint=None):
    report = json.loads(Path(path).read_text())
    if report.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("latency profile used a different official checkpoint")
    for name in PROFILE_FILES:
        if report.get("source_hashes", {}).get(name) != sha256(ROOT / name):
            raise RuntimeError(f"latency profile predates current source: {name}")
    if report.get("zero_feedback_native_full_plan_exact") is not True:
        raise RuntimeError("missing exact zero-feedback/native-plan anchor")
    if checkpoint is not None:
        if (not report.get("checkpoint") or Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve()
                or report.get("checkpoint_sha256") != sha256(checkpoint)):
            raise RuntimeError("loaded latency profile used another feedback checkpoint")
    for name in ("official_eager", "native_graph_planner", "feedback_query"):
        values = report[name]["samples_ms"]
        if len(values) < 20 or any(not math.isfinite(float(v)) or v <= 0 for v in values):
            raise RuntimeError(f"latency gate needs at least twenty finite positive samples: {name}")
        if not math.isclose(sum(values) / len(values), report[name]["mean_ms"], rel_tol=1e-6):
            raise RuntimeError(f"latency mean differs from raw samples: {name}")
    native = report["native_graph_planner"]["mean_ms"]
    fast = report["feedback_query"]["mean_ms"]
    official = report["official_eager"]["mean_ms"]
    ratio = (native + 3 * fast) / official
    ordered = sorted(report["feedback_query"]["samples_ms"])
    position = .95 * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    if not math.isclose(p95, report["feedback_query"]["p95_ms"], rel_tol=1e-6):
        raise RuntimeError("latency p95 differs from raw samples")
    if not math.isfinite(p95) or ratio > .9 or p95 >= 30 or report.get("passed") is not True:
        raise RuntimeError(f"latency gate failed: original-H16 ratio={ratio}, fast p95={p95}")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "ratio_to_original_h16": ratio, "ratio_to_graph_h16": (native + 3 * fast) / native,
            "fast_p95_ms": p95, "ms_per_action": (native + 3 * fast) / 16,
            "scope": "model-query service time; excludes rendering, environment execution and RTC"}


def checkpoint_evidence(path, *, steps, base, training, initialization=None):
    import torch
    from dynamicwam.models.flow_schedule import BridgeSchedule
    from dynamicwam.models.plan_feedback import PlanFeedbackHead
    from train_plan_feedback import runtime_contract
    torch.set_num_threads(1)
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {"format": "dynamicwam_plan_feedback_checkpoint", "version": 1,
                "global_step": steps, "base_checkpoint": base, "world_size": 7, "global_batch": 14,
                "phase": "feedback_refine" if initialization else "feedback_adapt"}
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"feedback export contract mismatch: {name}")
    head = PlanFeedbackHead()
    if payload.get("spec") != head.constructor_spec:
        raise RuntimeError("feedback head specification differs from the deployed default")
    contract = runtime_contract(BridgeSchedule.from_training_config(training), head.residual_limit)
    if payload.get("runtime_contract") != contract:
        raise RuntimeError("exported runtime contract differs from training/deployment")
    head.load_state_dict(payload["model"], strict=True)
    for key, value in payload["model"].items():
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"invalid/nonfinite feedback tensor: {key}")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise RuntimeError(f"feedback master/export is not FP32: {key}")
    for name in ("scripts/train_plan_feedback.py", "src/dynamicwam/models/plan_feedback.py",
                 "src/dynamicwam/models/flow_schedule.py", "src/dynamicwam/models/native_wam_teacher.py",
                 "src/dynamicwam/models/causal_world_bridge.py"):
        if payload.get("source_hashes", {}).get(name) != sha256(ROOT / name):
            raise RuntimeError(f"export trained with another source version: {name}")
    wanted_init = {"path": str(Path(initialization).resolve()), "sha256": sha256(initialization)} if initialization else None
    if payload.get("initialization") != wanted_init:
        raise RuntimeError("stage3 is not initialized from the just-verified stage2")
    recipe = payload.get("config", {})
    expected_recipe = dict(steps=steps, lr=5e-5 if initialization else 2e-4,
                          batch_size=1, gradient_accumulation=2, validate_every=500,
                          expert_weight=.25, residual_l2=.01)
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        raise RuntimeError("export training recipe differs from the fixed pipeline budget")
    official = torch.load(base["path"], map_location="cpu", weights_only=True, mmap=True)
    if payload.get("action_normalization") != official["config"]["action_normalization"]:
        raise RuntimeError("feedback normalization differs from the official planner")
    if not payload.get("dataset_identity") or not payload.get("pairing_identity"):
        raise RuntimeError("missing training data/pairing identity")
    for partition in ("train_ages", "validation_ages"):
        ages = {int(k): int(v) for k, v in payload["data"][partition].items()}
        if set(ages) != {0, 4, 8, 12} or min(ages.values()) <= 0:
            raise RuntimeError(f"dataset does not cover deployed ages: {partition}")
    validation = payload["validation"]
    if not validation or any(not math.isfinite(float(v)) for v in validation.values()):
        raise RuntimeError("export has no finite held-out validation")
    if any(validation.get(f"age_{age}/samples", 0) <= 0 for age in (0, 4, 8, 12)):
        raise RuntimeError("held-out validation missed a deployed age")
    rows = [json.loads(line) for line in (path.parent.parent / "metrics.jsonl").read_text().splitlines() if line.strip()]
    initialized = [r for r in rows if r.get("kind") == "initialized"]
    completed = [r for r in rows if r.get("kind") == "complete"]
    if (len(initialized) != 1 or initialized[0].get("teacher_trainable_parameters") != 0
            or initialized[0].get("global_batch") != 14 or not completed or completed[-1].get("step") != steps):
        raise RuntimeError("missing complete training/frozen-teacher evidence")
    gradients = [r for r in rows if r.get("kind") == "train"]
    if (not gradients or any(not math.isfinite(float(r["grad_norm"])) or r["grad_tensors"] <= 0 for r in gradients)
            or not any(r["grad_norm"] > 0 for r in gradients)):
        raise RuntimeError("missing finite nonzero feedback gradient evidence")
    return {"path": str(path), "sha256": sha256(path), "global_step": steps,
            "base_checkpoint": base, "spec": payload["spec"], "global_batch": 14,
            "teacher_trainable_parameters": 0, "validation": validation,
            "dataset_identity": payload["dataset_identity"], "pairing_identity": payload["pairing_identity"]}


def task_evidence(directory, seeds, checkpoint, base_sha, *, feedback):
    directory = Path(directory)
    aggregate_path = directory / "aggregate.json"
    aggregate = valid_task_aggregate(aggregate_path, len(seeds), expected_start_seed=100000,
                                     expected_episode_seeds=seeds)
    if aggregate is None:
        raise EvaluationIncomplete(f"incomplete/preparation-failed or mismatched fixed-seed task: {directory}")
    if (any(type(ep.get("success")) is not bool for ep in aggregate["episodes"])
            or aggregate["success_count"] != sum(ep["success"] for ep in aggregate["episodes"])):
        raise EvaluationIncomplete(f"inconsistent episode success accounting: {directory}")
    rows = []
    for log in sorted((directory / "logs").glob("slot*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("PLAN-FEEDBACK-TIMING "):
                rows.append(ast.literal_eval(line.split(" ", 1)[1]))
    if len(rows) != len(seeds):
        raise EvaluationIncomplete(f"missing episode timing/loader evidence: {directory}")
    for row in rows:
        if (Path(row["checkpoint"]).resolve() != Path(checkpoint).resolve()
                or row.get("base_checkpoint_sha256") != base_sha
                or row.get("feedback_enabled") is not feedback or row.get("execute_horizon") != 4):
            raise RuntimeError(f"evaluation loaded the wrong feedback arm: {directory}")
        if any(not math.isfinite(float(row.get(key, 0))) or row.get(key, 0) <= 0
               for key in ("query_count", "executed_steps", "query_ms_sum")):
            raise RuntimeError(f"evaluation has nonfinite/empty action timing: {directory}")
    queries = sum(row["query_count"] for row in rows)
    steps = sum(row["executed_steps"] for row in rows)
    ms = sum(row["query_ms_sum"] for row in rows)
    return {"aggregate": str(aggregate_path), "sha256": sha256(aggregate_path),
            "seeds": seeds, "reported_episodes": len(seeds), "success_count": aggregate["success_count"],
            "feedback_enabled": feedback, "query_count": queries, "executed_steps": steps,
            "query_ms_sum": ms, "query_ms_mean": ms / queries, "ms_per_executed_action": ms / steps}


def canary_evidence(rows):
    if set(rows) != {f"{arm}/{task}" for arm in ("zero_plan_control", "feedback") for task in TASKS}:
        raise RuntimeError("canary requires all six task/arm records")
    for task in TASKS:
        left, right = rows[f"zero_plan_control/{task}"], rows[f"feedback/{task}"]
        if left["seeds"] != right["seeds"] or len(left["seeds"]) != 2:
            raise EvaluationIncomplete("canary episode seeds do not match exactly")
    zero = sum(rows[f"zero_plan_control/{task}"]["success_count"] for task in TASKS)
    trained = sum(rows[f"feedback/{task}"]["success_count"] for task in TASKS)
    return {"zero_successes": zero, "feedback_successes": trained, "episodes_per_arm": 6,
            "passed": trained >= zero - 1, "rows": rows,
            "no_success_in_either_arm": zero == 0 and trained == 0,
            "scope": "small-sample regression tripwire; not statistical equivalence/non-inferiority"}


class Pipeline:
    def __init__(self, args):
        import fcntl
        self.args, self.root = args, args.run_root.resolve()
        if ROOT.resolve() != self.root / "source":
            raise RuntimeError("execute the immutable snapshot at RUN_ROOT/source/scripts/run_plan_feedback_pipeline.py")
        self.gpus = [s.strip() for s in args.gpus.split(",")]
        if len(self.gpus) != 7 or len(set(self.gpus)) != 7 or not all(s.isdigit() for s in self.gpus):
            raise ValueError("this recipe requires seven distinct GPU indices")
        self.lock = (self.root / "pipeline.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("status.json", "commands.jsonl", "pipeline_source_manifest.json", "smoke", "stage2", "stage3", "canary", "eval"):
            if (self.root / name).exists():
                raise FileExistsError(f"fresh run required; refusing to overwrite {self.root / name}")
        self.fingerprint = source_identity(self.root)
        write_json(self.root / "pipeline_source_manifest.json", self.fingerprint)
        self.state = dict(status="running", phase="preflight", started=time.time(), pid=os.getpid(),
                          source=str(ROOT), source_sha256=self.fingerprint["sha256"], completed=[], evidence={})
        self.python = str(ROOT / ".venv/bin/python")
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(self.gpus), OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        self.env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + self.env.get("PYTHONPATH", "")
        self.save()

    def save(self):
        self.state["updated"] = time.time()
        write_json(self.root / "status.json", self.state)

    def record(self, name, evidence):
        self.state["evidence"][name] = evidence
        self.state["completed"].append(name)
        self.save()

    def run(self, name, command, verify=lambda: {"passed": True}, *, env=None, gpu=True, evaluation=False):
        if source_identity(self.root) != self.fingerprint:
            raise RuntimeError("source snapshot changed during the run")
        if gpu:
            assert_gpus_idle(self.gpus)
        log = self.root / f"{name}.log"
        self.state.update(phase=name, command=command, log=str(log))
        self.save()
        with (self.root / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.time(), phase=name, argv=command, cwd=str(ROOT),
                                        cuda_visible_devices=(env or self.env).get("CUDA_VISIBLE_DEVICES"))) + "\n")
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env or self.env, stdout=stream, stderr=subprocess.STDOUT)
            self.state["child_pid"] = child.pid
            self.save()
            code = child.wait()
        self.state.pop("child_pid", None)
        if code:
            error = EvaluationIncomplete if evaluation else RuntimeError
            raise error(f"{name} exited {code}; inspect {log}; no automatic retry or seed substitution")
        evidence = verify()
        self.record(name, evidence)
        print(f"completed {name}", flush=True)
        return evidence

    def train(self, name, steps, lr, initialization=None):
        directory = self.root / name
        if directory.exists():
            raise FileExistsError(directory)
        checkpoint = directory / "exports" / f"feedback_step_{steps}.pt"
        command = [self.python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=7",
            str(ROOT / "scripts/train_plan_feedback.py"), "--config", str(self.args.config),
            "--run-dir", str(directory), "--steps", str(steps), "--lr", str(lr),
            "--batch-size", "1", "--gradient-accumulation", "2", "--validate-every", "500",
            "--validation-batches", "32", "--workers", "2"]
        if initialization:
            command.extend(["--init-feedback", str(initialization)])
        self.run(name, command, lambda: checkpoint_evidence(checkpoint, steps=steps, base=self.base,
                                                           training=self.training, initialization=initialization))
        return checkpoint

    def options(self, arm, checkpoint):
        return dict(policy_name="ciwam.adapters.domino.deploy_policy_plan_feedback", ckpt_setting=arm,
                    execute_horizon=4, eval_video_log=False, plan_feedback_checkpoint=str(checkpoint),
                    plan_feedback_disabled=arm == "zero_plan_control", plan_graph_enabled=True)

    def evaluate(self, checkpoint):
        import yaml
        rows = {}
        for arm in ("zero_plan_control", "feedback"):
            for task in TASKS:
                directory = self.root / "canary" / arm / task
                worker = self.profile.domino_base_config()
                worker.update(self.options(arm, checkpoint), task_name=task, start_seed=100000,
                              fixed_episode_seeds=self.seeds[task][:2], eval_output_root=str(directory))
                config = self.root / "eval_configs" / f"canary_{arm}_{task}.yaml"
                config.parent.mkdir(parents=True, exist_ok=True)
                with config.open("x") as stream:
                    yaml.safe_dump(worker, stream, sort_keys=False)
                command = [self.python, str(ROOT / "src/dynamicwam/evaluation/run_domino_parallel_eval.py"),
                    "--python", self.benchmark["python"], "--domino-root", self.benchmark["domino_root"],
                    "--eval-policy", str(self.fixed_policy), "--config", str(config),
                    "--runtime-root", self.benchmark["runtime_root"], "--curobo-root", self.benchmark["curobo_root"],
                    "--run-root", str(directory), "--gpus", self.gpus[0], "--total-episodes", "2",
                    "--start-seed", "100000", "--slot-seed-stride", "100000", "--timeout-seconds",
                    str(self.benchmark["task_timeout_seconds"]), "--stagger-seconds", "0"]
                rows[f"{arm}/{task}"] = self.run(f"canary_{arm}_{task}", command,
                    lambda d=directory, t=task, a=arm: task_evidence(d, self.seeds[t][:2], checkpoint,
                        self.base["sha256"], feedback=a == "feedback"), evaluation=True)
        gate = canary_evidence(rows)
        write_json(self.root / "canary_summary.json", gate)
        self.record("canary_gate", gate)
        if not gate["passed"]:
            raise RuntimeError("canary quality gate failed: feedback loses more than one of six paired episodes; no retraining")
        arms = {}
        for arm in ("zero_plan_control", "feedback"):
            directory = self.root / "eval" / arm
            override = self.root / "eval_configs" / f"{arm}.json"
            write_json(override, self.options(arm, checkpoint))
            command = [self.python, str(ROOT / "scripts/eval_domino.py"), "--config", str(self.deploy_config),
                "--eval-policy", str(self.fixed_policy), "--policy-overrides", str(override),
                "--run-root", str(directory), "--gpus", ",".join(self.gpus), "--episodes-per-task", "20",
                "--start-seed", "100000", "--episode-seed-manifest", str(self.manifest)]
            def verify(directory=directory, arm=arm):
                try:
                    aggregate = json.loads((directory / "aggregate.json").read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise EvaluationIncomplete(f"missing/invalid formal aggregate: {directory}") from error
                if (aggregate.get("completed_tasks") != 35 or aggregate.get("reported_episodes") != 700
                        or aggregate.get("requested_episodes") != 700):
                    raise EvaluationIncomplete(f"formal evaluation incomplete: {directory}")
                tasks = {task: task_evidence(directory / "tasks" / task, self.seeds[task], checkpoint,
                    self.base["sha256"], feedback=arm == "feedback") for task in self.seeds}
                return dict(tasks=tasks, success_count=sum(t["success_count"] for t in tasks.values()),
                            reported_episodes=700, episode_manifest_sha256=sha256(self.manifest),
                            aggregate_sha256=sha256(directory / "aggregate.json"))
            arms[arm] = self.run(f"formal_{arm}", command, verify, evaluation=True)
            write_json(self.root / "evaluation_summary.json", dict(arms=arms,
                scope="fixed accepted seeds; synchronous DOMINO execution, no RTC; expert preparation is rerun and motion paths are not bitwise paired"))

    def execute(self):
        self.deploy_config = ROOT / "configs/absolute_motion_v2.yaml"
        self.profile = load_profile(self.deploy_config)
        self.benchmark = self.profile.benchmark_config()
        if (Path(self.benchmark["project_root"]).resolve() != ROOT
                or Path(self.profile.domino_base_config()["dynamicwam_root"]).resolve() != ROOT
                or Path(self.benchmark["runtime_root"]).resolve() != ROOT / "src/dynamicwam/runtime"):
            raise RuntimeError("evaluation profile escaped the immutable source snapshot")
        self.training = load_multirate_profile(self.args.config).training
        base_path = Path(self.training["model"]["native_teacher_checkpoint"]).resolve()
        self.base = {"path": str(base_path), "sha256": sha256(base_path)}
        self.seeds, manifest_evidence, raw = load_episode_seed_manifest(self.args.manifest,
            tasks=list(self.benchmark["tasks"]), episodes_per_task=20, start_seed=100000)
        self.manifest = self.root / "episode_seed_manifest.json"
        if self.manifest.exists() and self.manifest.read_bytes() != raw:
            raise RuntimeError("run-root manifest differs from requested official-H16 seeds")
        if not self.manifest.exists():
            self.manifest.write_bytes(raw)
        self.record("manifest", manifest_evidence)
        self.record("latency_preflight", latency_evidence(self.args.latency_report, self.base["sha256"]))
        cpu_env = dict(self.env, CUDA_VISIBLE_DEVICES="")
        for script in ("check_plan_feedback.py", "check_plan_feedback_training.py",
                       "check_plan_feedback_pipeline.py", "check_wan_wrapper_equivalence.py"):
            self.run(script.removesuffix(".py"), [self.python, str(ROOT / "scripts" / script)], env=cpu_env, gpu=False)
        self.fixed_policy = self.root / "fixed_eval_policy.py"
        self.record("fixed_evaluator", generate(Path(self.benchmark["eval_policy"]), self.fixed_policy))
        if self.args.skip_verified_smoke:
            evidence = json.loads(self.args.skip_verified_smoke.read_text())
            checkpoint, profile = Path(evidence["checkpoint"]), Path(evidence["latency_report"])
            if sha256(checkpoint) != evidence["checkpoint_sha256"] or sha256(profile) != evidence["latency_report_sha256"]:
                raise RuntimeError("explicit smoke reuse evidence hash mismatch")
            self.record("smoke", checkpoint_evidence(checkpoint, steps=4, base=self.base, training=self.training))
            self.record("smoke_latency", latency_evidence(profile, self.base["sha256"], checkpoint))
            self.record("smoke_reuse", {"explicit_evidence": str(self.args.skip_verified_smoke),
                                        "sha256": sha256(self.args.skip_verified_smoke)})
        else:
            checkpoint = self.train("smoke", 4, 2e-4)
            profile = self.root / "smoke_latency.json"
            self.run("smoke_latency", [self.python, str(ROOT / "scripts/profile_plan_feedback.py"),
                "--checkpoint", str(checkpoint), "--output", str(profile), "--samples", "30", "--packet", str(self.args.packet)],
                lambda: latency_evidence(profile, self.base["sha256"], checkpoint),
                env=dict(self.env, CUDA_VISIBLE_DEVICES=self.gpus[0]))
        stage2 = self.train("stage2", 7000, 2e-4)
        stage3 = self.train("stage3", 2000, 5e-5, initialization=stage2)
        self.evaluate(stage3)
        self.state.update(status="complete", finished=time.time())
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latency-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multirate/native_stage2_7gpu.yaml")
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--skip-verified-smoke", type=Path,
        help="Explicit JSON evidence: checkpoint, checkpoint_sha256, latency_report, latency_report_sha256; requires 7-rank step4/globalbatch14")
    args = parser.parse_args()
    args.config = args.config.resolve()
    if not args.config.is_relative_to(ROOT):
        parser.error("training config must be inside the immutable snapshot")
    pipeline = Pipeline(args)
    try:
        pipeline.execute()
    except BaseException as error:
        pipeline.state.update(status="incomplete" if isinstance(error, EvaluationIncomplete) else "failed",
                              error=repr(error), stopped=time.time())
        pipeline.save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
