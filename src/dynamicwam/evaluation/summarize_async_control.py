"""Aggregate ASYNC-EPISODE-SUMMARY lines into control-oriented metrics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def read_async_summaries(log_path: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not log_path.is_file():
        return summaries
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "ASYNC-EPISODE-SUMMARY " not in line:
            continue
        try:
            value = ast.literal_eval(line.split("ASYNC-EPISODE-SUMMARY ", 1)[1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            summaries.append(value)
    return summaries


def collect_async_summaries(run_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    logs_root = run_root / "logs"
    if logs_root.is_dir():
        for path in sorted(logs_root.rglob("*.log")):
            summaries.extend(read_async_summaries(path))
    for path in sorted(run_root.rglob("*.log")):
        if logs_root in path.parents:
            continue
        summaries.extend(read_async_summaries(path))
    return summaries


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_control(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    slow = [int(item.get("slow_forwards", 0)) for item in summaries]
    fast = [int(item.get("fast_forwards", 0)) for item in summaries]
    fast_ms = [
        float(item["mean_fast_ms"])
        for item in summaries
        if item.get("mean_fast_ms") is not None
    ]
    slow_ms = [
        float(item["mean_slow_ms"])
        for item in summaries
        if item.get("mean_slow_ms") is not None
    ]
    deltas = [
        float(item["max_boundary_delta"])
        for item in summaries
        if item.get("max_boundary_delta") is not None
    ]
    total_slow = sum(slow)
    total_fast = sum(fast)
    return {
        "episodes": len(summaries),
        "open_loop_horizon": summaries[0].get("open_loop_horizon") if summaries else None,
        "slow_forwards": total_slow,
        "fast_forwards": total_fast,
        "fast_share": (
            total_fast / (total_slow + total_fast)
            if (total_slow + total_fast)
            else None
        ),
        "mean_fast_ms": _mean(fast_ms),
        "mean_slow_ms": _mean(slow_ms),
        "max_boundary_delta": max(deltas) if deltas else None,
    }
