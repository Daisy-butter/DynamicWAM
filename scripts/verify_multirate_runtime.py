"""Check H_exec replanning, fixed R, and bridge-slot lifetime."""

from __future__ import annotations

import sys

import numpy as np

from dynamicwam.observations.multiview import fast_views_from_composite, split_composite
from dynamicwam.runtime.ciwam.wam.bridge_store import BridgeStore, SlotState
from dynamicwam.runtime.ciwam.wam.multirate_control import ControlledMultirateController
from dynamicwam.training.data.paired_index import (
    pair_fits_episode,
    steady_state_offsets,
)


def _check_controller() -> None:
    control = ControlledMultirateController(
        execute_horizon=4,
        every_fast_queries=4,
        chunk_size=16,
    )
    for _query in range(4):
        if not control.needs_query():
            raise SystemExit("a query is required when the horizon is exhausted")
        actions = np.arange(16 * 14, dtype=np.float32).reshape(16, 14)
        control.commit(actions)
        if len(control.pending) != 4:
            raise SystemExit("only H_exec actions may stay in the execution queue")
        for step in range(4):
            control.take_action()
            control.mark_executed()
            query_due = control.needs_query()
            if step < 3 and query_due:
                raise SystemExit("replan started before H_exec actions were executed")
            if step == 3 and not query_due:
                raise SystemExit("replan did not start after H_exec executed actions")
    if control.fast_queries != 4 or control.slow_builds != 1:
        raise SystemExit(
            f"R=4 over 4 queries built {control.slow_builds} bridges, expected 1"
        )
    _check_same_cadence()


def _check_same_cadence() -> None:
    control = ControlledMultirateController(
        execute_horizon=16,
        every_fast_queries=4,
        chunk_size=16,
        bridge_lead_actions=4,
    )
    if control.rebuild_bridge():
        raise SystemExit("same-cadence decision must not rebuild the bridge")
    snapshots: list[int] = []
    queries: list[int] = []
    actions = np.zeros((16, 14), dtype=np.float32)
    for _step in range(32):
        if control.needs_bridge_snapshot():
            snapshots.append(control.executed_total)
            control.note_bridge_snapshot()
        if control.needs_query():
            if control.rebuild_bridge():
                raise SystemExit("same-cadence query rebuilt the bridge")
            queries.append((control.executed_total, control.bridge_age_actions()))
            control.commit(actions)
        control.take_action()
        control.mark_executed()
    if queries != [(0, 0), (16, 4)] or snapshots != [12, 28]:
        raise SystemExit(f"same-cadence clock queries={queries} snapshots={snapshots}")
    if control.fast_queries != 2 or control.slow_builds != 2:
        raise SystemExit(
            f"same-cadence counts fast={control.fast_queries} slow={control.slow_builds}"
        )


def _check_store() -> None:
    store = BridgeStore[str](capacity=2)
    first = store.claim_free()
    store.mark_pending(first, "bridge-0", source_id=0, generation=1)
    store.mark_producer_complete(first)
    if not store.publish(first):
        raise SystemExit("the first completed bridge should become ready")
    held = store.acquire(generation=1)
    second = store.claim_free()
    store.mark_pending(second, "bridge-1", source_id=1, generation=1)
    store.mark_producer_complete(second)
    if not store.publish(second):
        raise SystemExit("a newer bridge should replace the ready pointer")
    if held.state is not SlotState.RETIRED or held.payload != "bridge-0":
        raise SystemExit("an in-use bridge must stay readable after retirement")
    store.release(held)
    if held.state is not SlotState.FREE:
        if held.consumers_done:
            raise SystemExit("CPU release alone must not recycle a slot before GPU completion")
    store.mark_consumer_complete(held)
    if held.state is not SlotState.FREE:
        raise SystemExit("a retired slot should recycle after the consumer event")
    try:
        store.claim_free()
        store.mark_pending(store.slots[store.ready_id], "overwrite", source_id=2, generation=1)
    except RuntimeError:
        return
    raise SystemExit("the published slot must not be rewritten in place")


def _check_views_and_pairs() -> None:
    frame = np.zeros((360, 320, 3), dtype=np.uint8)
    frame[:240, :, 0] = 10
    frame[240:, :160, 1] = 20
    frame[240:, 160:, 2] = 30
    head, left, right = split_composite(frame)
    if int(head[0, 0, 0]) != 10 or int(left[0, 0, 1]) != 20 or int(right[0, 0, 2]) != 30:
        raise SystemExit("composite crops did not recover head, left, and right")
    views, masks = fast_views_from_composite(frame)
    if views.shape != (3, 3, 160, 160) or masks.shape != (3, 160, 160):
        raise SystemExit(f"unexpected fast view tensors {views.shape} {masks.shape}")
    if not pair_fits_episode(0, 12, total_frames=80, chunk_size=16, global_downsample_rate=3):
        raise SystemExit("a legal fast frame was rejected")
    if pair_fits_episode(0, 40, total_frames=80, chunk_size=16, global_downsample_rate=3):
        raise SystemExit("a fast frame whose action labels pass the episode end was accepted")
    offsets = steady_state_offsets(
        execute_horizon=4,
        every_fast_queries=4,
        global_downsample_rate=3,
    )
    if offsets != [0, 12, 24, 36]:
        raise SystemExit(f"R=4 steady-state offsets are {offsets}, expected 0/12/24/36")


def main() -> None:
    _check_controller()
    _check_store()
    _check_views_and_pairs()
    print("verified multirate runtime contracts")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"verify_multirate_runtime failed: {exc}", file=sys.stderr)
        raise
