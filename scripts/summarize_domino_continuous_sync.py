#!/usr/bin/env python3
"""Summarize one real-delay setting without hiding invalid or missing episodes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
from statistics import mean

PROTOCOL = "continuous_world_actual_latency_sync16_continuous_track_v2"
PROTOCOL_NOMINAL_TIMING = "continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1"
CONTROLLER = "continuous_shape_preserving_bounded_joint_track"
CONTROLLER_NOMINAL = "continuous_shape_preserving_nominal_timing_track"
KNOWN_PROTOCOLS = {
    PROTOCOL: {"timing_mode": "native_mplib_limits", "limit_retiming": True, "controller": CONTROLLER},
    PROTOCOL_NOMINAL_TIMING: {
        "timing_mode": "nominal_training_timing", "limit_retiming": False, "controller": CONTROLLER_NOMINAL},
}
TIMING_KEYS = ("action_interval_s", "history_frame_period_s", "history_endpoint_tolerance_s",
               "timing_contract_sha256")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def protocol_errors(report, spec):
    """Independent checks for the new full-chunk, non-RTC execution contract."""
    issues = []
    meta = KNOWN_PROTOCOLS[spec["protocol"]]
    required = {"controller": meta["controller"], "clock": "actual_wall_clock", "virtual_delay": False,
                "startup_delay_excluded": False, "rows_dropped": 0, "guided_requests": 0,
                "execution_inference_overlap": False, "asynchronous_camera_readback": True,
                "timing_mode": meta["timing_mode"], "limit_retiming": meta["limit_retiming"]}
    for key, expected in required.items():
        if report.get(key) != expected or (isinstance(expected, bool) and report.get(key) is not expected):
            issues.append(key)
    for key in TIMING_KEYS:
        if report.get(key) != spec.get(key):
            issues.append(key)
    limits = report.get("clock_limits", {})
    for observed, threshold, limit_key in (("max_backlog_s", "max_backlog_s", "max_backlog_s"),
             ("max_inference_clock_error_s", "infer_clock_tolerance_s", "inference_clock_error_s")):
        value = report.get(observed)
        if (not finite_number(value) or value < 0 or value > spec[threshold] + 1e-8
                or limits.get(limit_key) != spec[threshold]):
            issues.append(observed)
    previous_end = None
    for index, request in enumerate(report.get("requests", [])):
        prefix = f"request[{index}]"
        if request.get("guidance") is not False or request.get("prefetch") is not False or request.get("rows_dropped") != 0:
            issues.append(prefix + ".rtc_or_prefix")
        started, completed = request.get("started_rows", []), request.get("completed_rows", [])
        if (not isinstance(started, list) or started != list(range(len(started))) or len(started) > 16
                or not isinstance(completed, list) or completed != list(range(len(completed)))
                or len(completed) > len(started)):
            issues.append(prefix + ".row_order")
        if request.get("installed") is True and not started:
            issues.append(prefix + ".installed_without_row0")
        if previous_end is not None and (not finite_number(request.get("submit_physics_s"))
                                         or request["submit_physics_s"] < previous_end - 1e-8):
            issues.append(prefix + ".query_execution_overlap")
        if index < len(report.get("requests", [])) - 1 and completed != list(range(16)):
            issues.append(prefix + ".next_query_before_full16")
        previous_end = request.get("end_physics_s")
        # Terminal events truncate the clock interval. Check complete intervals
        # only; terminal mismatch is retained in diagnostics without backdating.
        if request.get("inference_clock_check") == "complete_interval":
            error = request.get("inference_clock_error_s")
            if not finite_number(error) or abs(error) > spec["infer_clock_tolerance_s"] + 1e-8:
                issues.append(prefix + ".inference_clock_error")
        history = request.get("history", {})
        if history:
            if (history.get("frame_period_s") != spec["history_frame_period_s"]
                    or history.get("endpoint_tolerance_s") != spec["history_endpoint_tolerance_s"]):
                issues.append(prefix + ".history_contract")
        trajectory = request.get("trajectory", {})
        if trajectory:
            if trajectory.get("nominal_action_interval_s") != spec["action_interval_s"]:
                issues.append(prefix + ".action_interval")
            stretch = trajectory.get("stretch")
            if not finite_number(stretch) or stretch < 1 - 1e-8:
                issues.append(prefix + ".trajectory_stretch")
            if not meta["limit_retiming"] and stretch > 1 + 1e-6:
                issues.append(prefix + ".nominal_timing_must_not_stretch")
            if trajectory.get("limit_retiming") is not meta["limit_retiming"]:
                issues.append(prefix + ".limit_retiming_flag")
    return issues


def distribution(values):
    values = sorted(float(value) for value in values if finite_number(value))
    if not values:
        return {"n": 0}
    def percentile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"n": len(values), "mean": mean(values), "min": values[0],
            "p50": percentile(.5), "p95": percentile(.95), "max": values[-1]}


def summarize(run):
    spec = json.loads((run / "suite_started.json").read_text())
    if spec.get("protocol") not in KNOWN_PROTOCOLS or spec.get("virtual_delay") is not False:
        raise ValueError("not a single-setting actual-wall-clock continuous_sync run")
    protocol = spec["protocol"]
    meta = KNOWN_PROTOCOLS[protocol]
    expected_controller = meta["controller"]
    # Older native suites may omit timing_mode; infer from protocol.
    if spec.get("timing_mode", meta["timing_mode"]) != meta["timing_mode"]:
        raise ValueError("suite timing_mode does not match protocol")
    if spec.get("controller", expected_controller) != expected_controller or spec.get("policy_label") != "dynamicwam_continuous_sync":
        raise ValueError("suite manifest has the wrong continuous controller or policy label")
    for key in TIMING_KEYS[:-1]:
        if not finite_number(spec.get(key)) or spec[key] <= 0:
            raise ValueError("suite manifest lacks the explicit physical timing contract")
    for key, maximum in (("max_backlog_s", .25), ("infer_clock_tolerance_s", .02)):
        if not finite_number(spec.get(key)) or not 0 < spec[key] <= maximum:
            raise ValueError("suite manifest relaxes the required realtime quality gates")
    contract_path = run / "timing_contract.json"
    if (not contract_path.is_file() or hashlib.sha256(contract_path.read_bytes()).hexdigest() != spec.get("timing_contract_sha256")
            or spec.get("source_sha256", {}).get("timing_contract.json") != spec.get("timing_contract_sha256")):
        raise ValueError("missing or mismatched frozen timing contract")
    recommendation = json.loads(contract_path.read_bytes())["recommended_continuous_sync_contract"]
    expected = (recommendation["nominal_action_interval_seconds"], recommendation["nominal_history_frame_interval_seconds"],
                recommendation["nominal_history_frame_interval_seconds"] / 2)
    if any(abs(spec[key] - value) > 1e-12 for key, value in zip(TIMING_KEYS, expected)):
        raise ValueError("suite settings differ from the training-calibrated timing contract")
    tasks = spec["tasks"]
    count = int(spec["episodes_per_task"])
    if len(set(tasks)) != len(tasks) or count < 1:
        raise ValueError("invalid suite manifest")
    identity_keys = ("semantics_id", "source_generation", "source_digest")
    if any(not isinstance(spec.get(key), str) or not spec[key] for key in identity_keys):
        raise ValueError("suite manifest lacks an explicit semantics/generation/digest identity")
    task_results, errors = [], []
    reasons = Counter()
    generations = {}
    generation_digests = {}
    semantics = set()
    diagnostic_values = {key: [] for key in ("history_interval_dt_s", "history_endpoint_abs_offset_s",
        "trajectory_stretch", "trajectory_duration_s", "nominal_trajectory_duration_s", "model_compute_wall_s",
        "query_capture_wall_s", "all_capture_duration_s", "camera_trajectory_overlap_s", "realtime_factor",
        "terminal_unresolved_wall_lag_s", "max_backlog_s", "max_inference_clock_error_s")}
    phase_totals = Counter()
    interval_mask_counts = Counter()
    for task in tasks:
        task_dir = run / task
        error_file = task_dir / "error.json"
        if error_file.exists():
            errors.append({"task": task, **json.loads(error_file.read_text())})
        accepted = []
        missing = []
        invalid = []
        invalid_protocol_details = {}
        seeds = set()
        acceptance_path = task_dir / "accepted_manifest.json"
        accepted_descriptors = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else []
        acceptance = {}
        expected_seed = int(spec["seed"])
        broken_chain = False
        for index, descriptor in enumerate(accepted_descriptors):
            if (descriptor.get("task") != task or descriptor.get("episode_index") != index
                    or descriptor.get("requested_seed") != expected_seed
                    or not isinstance(descriptor.get("seed"), int) or descriptor["seed"] < expected_seed):
                broken_chain = True
            if not broken_chain:
                acceptance[index] = descriptor
                expected_seed = descriptor["seed"] + 1
        for episode in range(count):
            directory = task_dir / f"episode_{episode:04d}"
            report_file, metric_file = directory / "report.json", directory / "metrics.json"
            if not report_file.exists():
                missing.append(episode)
                continue
            report = json.loads(report_file.read_text())
            descriptor_path = directory / "manifest.json"
            descriptor = json.loads(descriptor_path.read_text()) if descriptor_path.exists() else {}
            if (report.get("protocol") != protocol or report.get("realtime_valid") is not True
                    or report.get("episode_index") != episode or report.get("task") != task
                    or report.get("probe_action_limit") is not None or not metric_file.exists()
                    or any(report.get(key) != spec[key] for key in identity_keys)
                    or episode not in acceptance
                    or any(report.get(key) != descriptor.get(key) or report.get(key) != acceptance[episode].get(key)
                           for key in ("task", "episode_index", "seed", "requested_seed"))):
                invalid.append(episode)
                continue
            issues = protocol_errors(report, spec)
            if issues:
                invalid.append(episode)
                invalid_protocol_details[str(episode)] = issues
                continue
            metrics = json.loads(metric_file.read_text())
            if spec.get("video"):
                from video_artifacts import validate_video_artifacts
                if not report.get("video_required") or not validate_video_artifacts(report):
                    invalid.append(episode)
                    continue
            seed = int(report["seed"])
            score = float(metrics["manipulation_score"])
            if (seed in seeds or not math.isfinite(score) or not 0 <= score <= 100
                    or bool(metrics["success"]) != bool(report["success"]) or metrics.get("seed") != seed):
                invalid.append(episode)
                continue
            seeds.add(seed)
            accepted.append((bool(report["success"]), score))
            reasons[str(report["stop_reason"])] += 1
            semantics.add(report.get("semantics_id", PROTOCOL))
            generation = str(report.get("source_generation", "initial"))
            generations.setdefault(generation, []).append((bool(report["success"]), score))
            generation_digests.setdefault(generation, set()).add(report.get("source_digest", "legacy_unrecorded"))
            for key in ("camera_trajectory_overlap_s", "realtime_factor", "terminal_unresolved_wall_lag_s", "max_backlog_s", "max_inference_clock_error_s"):
                if finite_number(report.get(key)):
                    diagnostic_values[key].append(report[key])
            for key, value in report.get("phase_physics_seconds", {}).items():
                if finite_number(value): phase_totals[key] += value
            for capture in report.get("captures", []):
                if finite_number(capture.get("duration_s")):
                    diagnostic_values["all_capture_duration_s"].append(capture["duration_s"])
            for request in report.get("requests", []):
                for key in ("model_compute_wall_s",):
                    if finite_number(request.get(key)): diagnostic_values[key].append(request[key])
                if finite_number(request.get("capture_wall_s")):
                    diagnostic_values["query_capture_wall_s"].append(request["capture_wall_s"])
                history = request.get("history", {})
                diagnostic_values["history_interval_dt_s"].extend(value for value in history.get("interval_dt_s", []) if finite_number(value))
                diagnostic_values["history_endpoint_abs_offset_s"].extend(abs(value) for value in history.get("endpoint_offsets_s", []) if finite_number(value))
                for mask_name in ("temporal_valid_mask", "quality_valid_mask", "interval_valid_mask", "acceleration_valid_mask"):
                    mask = history.get(mask_name, [])
                    interval_mask_counts[mask_name + "_total"] += len(mask)
                    interval_mask_counts[mask_name + "_valid"] += sum(value is True for value in mask)
                trajectory = request.get("trajectory", {})
                for source, dest in (("stretch", "trajectory_stretch"), ("duration_s", "trajectory_duration_s"),
                                     ("nominal_duration_s", "nominal_trajectory_duration_s")):
                    if finite_number(trajectory.get(source)): diagnostic_values[dest].append(trajectory[source])
        completion_path = task_dir / "completed.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
        completion_valid = (completion.get("episodes") == count and completion.get("protocol") == protocol
                            and completion.get("probe_action_limit") is None)
        if spec.get("video") and completion_valid:
            task_spec_file = task_dir / "task_started.json"
            task_spec = json.loads(task_spec_file.read_text()) if task_spec_file.exists() else {}
            native = Path(task_spec["native_result_dir"]) if task_spec.get("native_result_dir") else None
            if native is None or not all((native / name).is_file() for name in
                    ("_result.txt", "_metrics_report.txt", "_metrics.json", "_episodes_detail.json", "_evaluation_status.json")):
                completion_valid = False
            else:
                status = json.loads((native / "_evaluation_status.json").read_text())
                details = json.loads((native / "_episodes_detail.json").read_text())
                completion_valid = (status.get("complete") is True and status.get("completed_episodes") == count
                                    and len(details) == count)
        task_results.append({"task": task, "planned": count, "valid_completed": len(accepted),
                             "completion_marker_valid": completion_valid,
                             "archived_attempts": len(list((task_dir / "attempts").glob("episode_*/attempt_*"))),
                             "missing_episode_indices": missing, "invalid_episode_indices": invalid,
                             "invalid_protocol_details": invalid_protocol_details,
                             "successes": sum(item[0] for item in accepted),
                             "provisional_sr_percent": 100 * mean(item[0] for item in accepted) if accepted else None,
                             "provisional_ms": mean(item[1] for item in accepted) if accepted else None})
    suite_error = run / "suite_error.json"
    if suite_error.exists():
        errors.append({"supervisor": True, **json.loads(suite_error.read_text())})
    for task in tasks:
        exit_file = run / "logs" / f"{task}.exit.json"
        if exit_file.exists():
            exit_status = json.loads(exit_file.read_text())
            if exit_status.get("status") != 0:
                errors.append({"task": task, "supervisor_exit": exit_status})
        else:
            errors.append({"task": task, "error": "task supervisor exit record is missing"})
    if len(semantics) > 1:
        errors.append({"error": "multiple semantics_id values; refuse mixed formal aggregation",
                       "semantics_ids": sorted(semantics)})
    if len(generations) > 1:
        errors.append({"error": "multiple source generations; formal aggregation requires a uniform runtime",
                       "source_generations": sorted(generations)})
    if any(len(values) > 1 for values in generation_digests.values()):
        errors.append({"error": "source generation label has multiple code digests; refuse mixed formal aggregation"})
    complete = not errors and all(row["valid_completed"] == count and row["completion_marker_valid"]
                                 for row in task_results)
    valid = sum(row["valid_completed"] for row in task_results)
    result = {"protocol": protocol, "complete": complete,
              "timing_mode": meta["timing_mode"], "limit_retiming": meta["limit_retiming"],
              "controller": expected_controller,
              "formal_35x20": bool(spec.get("formal_35x20")) and len(tasks) == 35 and count == 20,
              "formal_35x100": bool(spec.get("formal_35x100")) and len(tasks) == 35 and count == 100,
              "planned_tasks": len(tasks), "planned_episodes_per_task": count,
              "planned_total": len(tasks) * count, "valid_completed": valid,
              "successes_valid_completed": sum(row["successes"] for row in task_results),
              "invalid_count": sum(len(row["invalid_episode_indices"]) for row in task_results),
              "missing_count": sum(len(row["missing_episode_indices"]) for row in task_results),
              "full_suite_macro_sr_percent": mean(row["provisional_sr_percent"] for row in task_results) if complete else None,
              "full_suite_macro_ms": mean(row["provisional_ms"] for row in task_results) if complete else None,
              "stop_reasons_valid_completed": dict(reasons), "task_errors": errors,
              "task_results": task_results, "semantics_ids": sorted(semantics),
              "replay_from": spec.get("replay_from"),
              "source_generation_results": {generation: {
                  "valid_completed": len(rows), "successes": sum(row[0] for row in rows),
                  "source_digests": sorted(generation_digests[generation]),
                  "provisional_micro_sr_percent": 100 * mean(row[0] for row in rows),
                  "provisional_micro_ms": mean(row[1] for row in rows)}
                  for generation, rows in generations.items()}}
    result["timing_contract_sha256"] = spec["timing_contract_sha256"]
    result["physical_timing"] = {key: spec[key] for key in TIMING_KEYS[:-1]}
    result["diagnostics_valid_completed"] = {key: distribution(values) for key, values in diagnostic_values.items()}
    result["diagnostics_valid_completed"]["phase_physics_seconds_total"] = dict(phase_totals)
    result["diagnostics_valid_completed"]["history_mask_counts"] = dict(interval_mask_counts)
    total_physics = sum(phase_totals.values())
    result["diagnostics_valid_completed"]["phase_physics_fraction"] = {
        key: value / total_physics for key, value in phase_totals.items()} if total_physics else {}
    result["diagnostics_note"] = "Camera-motion overlap is concurrent time, not an additional exclusive phase. Terminal inference clock mismatches remain diagnostic and are not treated as complete inference intervals."
    result["formal_full_suite"] = bool(spec.get("formal_full_suite", result["formal_35x100"])) and len(tasks) == 35
    result["video_required"] = bool(spec.get("video"))
    result["formal_result_available"] = complete and result["formal_full_suite"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    result = summarize(run)
    temporary = run / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(run / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
