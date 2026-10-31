"""Deadline-bounded receipt; execution remains exclusively with the owner.

The mailbox stamps arrival itself after all GPU and host work. No producer may
backdate submission or touch the advancing physics clock. A timely receipt is
not immediate permission to execute: wall release and physical lease still apply.
"""
from dataclasses import dataclass
import threading
from .timing import TimingError, StalePlan


@dataclass(frozen=True)
class FastReceipt:
    lease: object
    result: tuple
    submitted_wall: float
    earliest_physical_time: float | None


class FastMailbox:
    def __init__(self, wall_clock, bridge):
        self.wall_clock,self.bridge=wall_clock,bridge
        self.lock=threading.Lock()
        self.lease=None
        self.receipt=None
        self.closed=False

    def register(self,lease):
        with self.lock:
            if self.closed or self.lease is not None:
                raise TimingError('fast submission slot unavailable')
            self.lease=lease

    def submit(self,lease,result):
        with self.lock:
            if self.closed or self.lease!=lease or self.receipt is not None:
                raise StalePlan('duplicate, cancelled or old-episode submission')
            if result[0].lease != lease:
                raise StalePlan('submitted plan and lease differ')
            # No producer timestamp is used as proof of arrival.
            now=float(self.wall_clock())
            earliest=None
            if self.bridge is not None and self.bridge.active:
                self.bridge.check_deadline(lease.protected_until,lease.submit_margin,wall_now=now)
                earliest=self.bridge.availability_time(now)
            self.receipt=FastReceipt(lease,result,now,earliest)
            return self.receipt

    def peek(self,lease):
        with self.lock:
            if self.closed or self.lease!=lease:
                raise StalePlan('receipt belongs to another live slot')
            return self.receipt

    def check_pending(self,lease):
        # Checking for an on-time arrival and declaring a missed wall deadline
        # must be one transition, otherwise a receipt can race an owner poll.
        with self.lock:
            if self.closed or self.lease!=lease:
                raise StalePlan('receipt belongs to another live slot')
            if self.receipt is None and self.bridge is not None and self.bridge.active:
                self.bridge.check_deadline(lease.protected_until,lease.submit_margin)
            return self.receipt

    def finish(self,lease):
        with self.lock:
            if self.lease!=lease:
                raise StalePlan('submission no longer owns slot')
            self.lease=self.receipt=None

    def close(self):
        with self.lock:
            self.closed=True
            self.lease=self.receipt=None
