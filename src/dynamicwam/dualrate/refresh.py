"""Bounded refresh decisions from calibrated observable-world mismatch.

This controller never moves a clock or blocks a fast query. A caller must supply
a residual comparing the SAME observable, coordinate and physical time. An action
disagreement, an arbitrary latent distance or simulator GT is not such a residual.
Uncalibrated deployments keep fixed-period refresh; this controller is opt-in.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RefreshConfig:
    high: float
    low: float
    persistence: int = 3
    min_interval: float = .72
    max_interval: float = 2.16
    budget_per_second: float = 1. / .72

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (self.high,self.low,self.min_interval,
                self.max_interval,self.budget_per_second)) or not 0 <= self.low < self.high
                or type(self.persistence) is not int or self.persistence < 1
                or not 0 < self.min_interval <= self.max_interval
                or self.budget_per_second <= 0
                or self.max_interval * self.budget_per_second < 1):
            raise ValueError('invalid hysteresis/refresh budget; max wait must be feasible')


class RefreshController:
    def __init__(self, config, start_time=0.):
        self.config = config
        self.last = self.now = float(start_time)
        if not math.isfinite(self.now):
            raise ValueError('finite physical start required')
        self.tokens = 0.
        self.count = 0
        self.pending = False
        self.latest_frame = None
        self.last_observed_frame = -1
        self.inflight = False
        self.launched = 0

    def advance(self, now):
        if not math.isfinite(now) or now < self.now:
            raise ValueError('refresh clock must be monotonic physical seconds')
        self.tokens = min(1., self.tokens+(now-self.now)*self.config.budget_per_second)
        self.now = float(now)

    def observe(self, *, now, frame_id, residual, comparable):
        self.advance(now)
        if type(frame_id) is not int or frame_id < self.last_observed_frame:
            raise ValueError('out-of-order observation identity')
        if frame_id == self.last_observed_frame:
            return  # Querying the same image again is not persistent evidence.
        self.last_observed_frame = frame_id
        self.latest_frame = frame_id
        if not comparable:
            self.count = 0
            return
        if residual is None or not math.isfinite(residual) or residual < 0:
            raise ValueError('finite comparable world residual required')
        if residual >= self.config.high:
            self.count += 1
        elif residual <= self.config.low:
            self.count = 0
            self.pending = False
        if self.count >= self.config.persistence:
            self.pending = True  # Coalesce while the one slow job is busy.

    def due(self, now):
        self.advance(now)
        if (self.inflight or self.latest_frame is None or self.tokens < 1.-1e-9
                or now-self.last < self.config.min_interval-1e-9):
            return None
        if self.pending:
            return 'persistent_world_mismatch'
        if now-self.last >= self.config.max_interval-1e-9:
            return 'maximum_reference_wait'
        return None

    def started(self, now):
        reason = self.due(now)
        if reason is None:
            raise ValueError('refresh was not authorized by budget/controller')
        self.tokens = max(0., self.tokens-1.)
        self.last = float(now)
        self.inflight = True
        self.pending = False
        self.count = 0
        self.launched += 1
        return dict(reason=reason,physical_time=now,frame_id=self.latest_frame)

    def completed(self):
        if not self.inflight:
            raise ValueError('duplicate or unknown refresh completion')
        self.inflight = False
