"""Fixed-ratio replanning. A new query starts after H_exec completed actions.

The unused tail of a 16-step sample is discarded. Pending actions do not delay
that replan, and the bridge rebuild count follows ``query_id % R``.

``bridge_lead_actions > 0`` is the same-cadence schedule: one fast sample per
execute horizon, and the next world bridge is snapshotted that many completed
actions before the decision. The decision itself does not rebuild the bridge.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ControlledMultirateController:
    execute_horizon: int = 4
    every_fast_queries: int = 1
    chunk_size: int = 16
    bridge_lead_actions: int = 0
    pending: deque[np.ndarray] = field(default_factory=deque)
    executed_since_query: int = 0
    executed_total: int = 0
    query_id: int = 0
    slow_builds: int = 0
    fast_queries: int = 0
    bridge_snapshot_submitted: bool = False

    def __post_init__(self) -> None:
        if self.execute_horizon < 1 or self.execute_horizon > self.chunk_size:
            raise ValueError("H_exec must be in [1, chunk_size]")
        if self.every_fast_queries < 1:
            raise ValueError("R must be >= 1")
        if self.bridge_lead_actions < 0:
            raise ValueError("bridge lead must be non-negative")
        if self.bridge_lead_actions >= self.execute_horizon:
            raise ValueError("bridge lead must be shorter than the execute horizon")

    def needs_query(self) -> bool:
        return (not self.pending) or self.executed_since_query >= self.execute_horizon

    def needs_bridge_snapshot(self) -> bool:
        """True once per chunk, ``bridge_lead_actions`` before the next decision."""

        if self.bridge_lead_actions < 1 or self.bridge_snapshot_submitted:
            return False
        return len(self.pending) == self.bridge_lead_actions

    def note_bridge_snapshot(self) -> None:
        if not self.needs_bridge_snapshot():
            raise RuntimeError("bridge snapshot is not due")
        self.bridge_snapshot_submitted = True
        self.slow_builds += 1

    def rebuild_bridge(self) -> bool:
        if self.bridge_lead_actions > 0:
            return False
        return self.query_id % self.every_fast_queries == 0

    def bridge_age_actions(self) -> int:
        """Actions since the bridge this query will read. Call before ``commit``.

        Same-cadence startup builds the bridge at the decision (age 0). Later
        decisions read the snapshot taken ``bridge_lead_actions`` earlier.
        """

        if self.bridge_lead_actions > 0:
            return 0 if self.query_id == 0 else self.bridge_lead_actions
        slot = self.query_id % self.every_fast_queries
        if slot == 0:
            return 0
        return slot * self.execute_horizon

    def commit(self, actions: np.ndarray) -> None:
        if actions.ndim != 2 or int(actions.shape[0]) != self.chunk_size:
            raise ValueError(
                f"expected a [{self.chunk_size}, A] chunk, got {actions.shape}"
            )
        if self.rebuild_bridge():
            self.slow_builds += 1
        self.pending = deque(
            np.asarray(row) for row in actions[: self.execute_horizon]
        )
        self.executed_since_query = 0
        self.bridge_snapshot_submitted = False
        self.query_id += 1
        self.fast_queries += 1

    def take_action(self) -> np.ndarray:
        if not self.pending:
            raise RuntimeError("no pending action")
        return self.pending.popleft()

    def mark_executed(self) -> None:
        self.executed_since_query += 1
        self.executed_total += 1

    def reset(self) -> None:
        self.pending.clear()
        self.executed_since_query = 0
        self.query_id = 0
        self.executed_total = 0
        self.slow_builds = 0
        self.fast_queries = 0
        self.bridge_snapshot_submitted = False


def forward_budget(
    *,
    execute_horizon: int = 4,
    refresh_queries: int = 4,
    action_steps: int = 10,
    joint_steps: int = 2,
) -> dict[str, int]:
    """Count forwards over one bridge lifetime and the two synchronous baselines.

    The synchronous sampler spends ``joint_steps`` on the video-action prefix and
    the remaining action steps on action-only denoising. The fast path repeats a
    full action schedule on every query. World work stays at ``joint_steps`` for
    the whole lifetime, so it is cheaper than replanning the full model every
    ``execute_horizon`` actions and more action-heavy than one long chunk.
    """

    if execute_horizon < 1 or refresh_queries < 1 or action_steps < joint_steps:
        raise ValueError("budget counts require a positive horizon and a joint prefix")
    if joint_steps < 1:
        raise ValueError("the published bridge comes from the joint prefix")
    action_only = action_steps - joint_steps
    return {
        "executed_actions": execute_horizon * refresh_queries,
        "multirate_world_joint_forwards": joint_steps,
        "multirate_fast_action_forwards": refresh_queries * action_steps,
        "sync_long_chunk_joint_forwards": joint_steps,
        "sync_long_chunk_action_forwards": action_only,
        "sync_matched_horizon_joint_forwards": refresh_queries * joint_steps,
        "sync_matched_horizon_action_forwards": refresh_queries * action_only,
    }
