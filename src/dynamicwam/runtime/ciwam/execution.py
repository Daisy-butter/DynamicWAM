from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NativeStepRecord:
    index: int
    wall_ms: float
    overrun_ms: float


@dataclass
class NativeExecutionTelemetry:
    records: List[NativeStepRecord] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": len(self.records),
            "kinds": {"action": len(self.records)} if self.records else {},
            "overrun_steps": sum(record.overrun_ms > 1.0 for record in self.records),
        }


def query_timing_summary(samples_ms: list[float], executed_steps: int) -> Dict[str, Any]:
    """Wall time of policy queries. This is not the simulator step timer."""

    if not samples_ms:
        return {
            "query_count": 0,
            "query_ms_mean": None,
            "query_ms_p50": None,
            "query_ms_p95": None,
            "executed_steps": int(executed_steps),
            "query_ms_sum": 0.0,
            "ms_per_executed_action": None,
        }
    ordered = sorted(float(value) for value in samples_ms)
    total = sum(ordered)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "query_count": len(ordered),
        "executed_steps": int(executed_steps),
        "query_ms_p95": ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)],
        "query_ms_mean": total / len(ordered),
        "query_ms_p50": median,
        "query_ms_sum": total,
        "ms_per_executed_action": (total / executed_steps) if executed_steps else None,
    }


class NativeStepper:
    """Execute actions directly in DOMINO without pacing or synthetic holds."""

    def __init__(self, task_env, *, action_interval_seconds: float) -> None:
        self.env = task_env
        self.action_interval_s = float(action_interval_seconds)
        if self.action_interval_s <= 0.0:
            raise ValueError("action_interval_seconds must be positive")
        self.telemetry = NativeExecutionTelemetry()

    def execute(self, action) -> None:
        start = time.perf_counter()
        self.env.take_action(action, action_type="qpos")
        end = time.perf_counter()
        self.telemetry.records.append(
            NativeStepRecord(
                index=int(self.env.take_action_cnt),
                wall_ms=(end - start) * 1000.0,
                overrun_ms=max(
                    0.0,
                    (end - start) - self.action_interval_s,
                )
                * 1000.0,
            )
        )
