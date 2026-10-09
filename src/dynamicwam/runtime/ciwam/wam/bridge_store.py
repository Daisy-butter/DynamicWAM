"""Bounded slots for a published world bridge.

The slot that fast queries can still select is never overwritten. A retired
slot is recycled only after its CPU holders are gone and the consumer GPU
work that read it has finished. Tests set those flags without a CUDA event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar
from threading import RLock
from functools import wraps

def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call

T = TypeVar("T")


class SlotState(Enum):
    FREE = "free"
    WRITING = "writing"
    PENDING_GPU = "pending_gpu"
    READY = "ready"
    RETIRED = "retired"


@dataclass
class BridgeSlot(Generic[T]):
    slot_id: int
    state: SlotState = SlotState.FREE
    payload: T | None = None
    source_id: int = -1
    generation: int = -1
    cpu_refs: int = 0
    producer_done: bool = True
    consumers_done: bool = True


@dataclass
class BridgeStore(Generic[T]):
    capacity: int
    slots: list[BridgeSlot[T]] = field(init=False)
    ready_id: int | None = None
    generation: int | None = None
    _lock: object = field(default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 2:
            raise ValueError("bridge store needs at least a ready slot and a write slot")
        self.slots = [BridgeSlot(slot_id=index) for index in range(self.capacity)]

    @synchronized
    def claim_free(self) -> BridgeSlot[T]:
        for slot in self.slots:
            if slot.state is SlotState.FREE:
                slot.state = SlotState.WRITING
                slot.producer_done = False
                slot.consumers_done = True
                slot.cpu_refs = 0
                return slot
        raise RuntimeError("no free bridge slot; the slow worker must wait or drop")

    @synchronized
    def mark_pending(
        self,
        slot: BridgeSlot[T],
        payload: T,
        *,
        source_id: int,
        generation: int,
    ) -> None:
        if slot.state is not SlotState.WRITING:
            raise RuntimeError(f"slot {slot.slot_id} is {slot.state}, not WRITING")
        if self.ready_id == slot.slot_id:
            raise RuntimeError("the published slot cannot be rewritten in place")
        slot.payload = payload
        slot.source_id = int(source_id)
        slot.generation = int(generation)
        slot.state = SlotState.PENDING_GPU

    @synchronized
    def mark_producer_complete(self, slot: BridgeSlot[T]) -> None:
        slot.producer_done = True

    @synchronized
    def publish(self, slot: BridgeSlot[T]) -> bool:
        if slot.state is not SlotState.PENDING_GPU:
            raise RuntimeError(f"slot {slot.slot_id} is not waiting for the producer event")
        if not slot.producer_done:
            return False
        if self.generation is None:
            self.generation = slot.generation
        if slot.generation != self.generation:
            slot.state = SlotState.RETIRED
            self._reclaim(slot)
            return False
        if self.ready_id is not None:
            current = self.slots[self.ready_id]
            if current.generation == slot.generation and current.source_id > slot.source_id:
                slot.state = SlotState.RETIRED
                self._reclaim(slot)
                return False
            current.state = SlotState.RETIRED
            self.ready_id = None
            self._reclaim(current)
        slot.state = SlotState.READY
        self.ready_id = slot.slot_id
        return True

    @synchronized
    def acquire(self, *, generation: int) -> BridgeSlot[T]:
        if self.ready_id is None:
            raise RuntimeError("no ready bridge")
        slot = self.slots[self.ready_id]
        if slot.generation != generation or slot.state is not SlotState.READY:
            raise RuntimeError("ready bridge does not match the current episode")
        slot.cpu_refs += 1
        slot.consumers_done = False
        return slot

    @synchronized
    def release(self, slot: BridgeSlot[T]) -> None:
        if slot.cpu_refs <= 0:
            raise RuntimeError(f"slot {slot.slot_id} has no CPU holders")
        slot.cpu_refs -= 1
        self._reclaim(slot)

    @synchronized
    def mark_consumer_complete(self, slot: BridgeSlot[T]) -> None:
        if slot.cpu_refs > 0:
            raise RuntimeError("consumer GPU work is still referenced on the CPU")
        slot.consumers_done = True
        self._reclaim(slot)

    @synchronized
    def reset(self, generation: int) -> None:
        if self.generation is not None and generation <= self.generation:
            raise ValueError("episode generation must advance")
        self.generation = int(generation)
        self.ready_id = None
        for slot in self.slots:
            if slot.state is SlotState.READY:
                slot.state = SlotState.RETIRED
            self._reclaim(slot)

    def _reclaim(self, slot: BridgeSlot[T]) -> None:
        if slot.state is not SlotState.RETIRED:
            return
        if slot.cpu_refs > 0 or not slot.consumers_done:
            return
        if self.ready_id == slot.slot_id:
            return
        slot.payload = None
        slot.state = SlotState.FREE
        slot.producer_done = True
        slot.consumers_done = True
