"""CPU-only rendezvous separating replay setup from live RTC episodes.

Workers are fixed for a cohort. They may set up concurrently, but none starts a
live episode until every peer is ready. After terminal drain, every peer must
finish before encoding, teardown, or next-episode setup. No live clock pauses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

MODE = "phased_episode_cohorts_v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def write_abort(directory, reason):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(directory / "abort.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as handle:
        json.dump(dict(reason=str(reason), pid=os.getpid(), wall_time_s=time.time()), handle)


class EpisodeCohort:
    def __init__(self, directory, task):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.task = task
        self.members = self.manifest["tasks"]
        if (self.manifest.get("mode") != MODE or not self.members
                or len(self.members) != len(set(self.members)) or task not in self.members
                or any(not name or Path(name).name != name for name in self.members)):
            raise ValueError("invalid fixed RTC cohort")
        self.timeout_s = float(self.manifest["barrier_timeout_s"])
        if not 0 < self.timeout_s < float("inf"):
            raise ValueError("invalid cohort timeout")

    def abort(self, reason):
        write_abort(self.directory, reason)

    def wait(self, episode, phase):
        if phase not in {"ready", "finished"} or not 0 <= episode < self.manifest["episodes"]:
            raise ValueError("invalid episode rendezvous")
        arrival = self.directory / f"episode_{episode:04d}" / phase
        arrival.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        atomic_json(arrival / f"{self.task}.json", dict(task=self.task, pid=os.getpid(),
            phase=phase, episode=episode, monotonic_s=start, wall_time_s=time.time()))
        while True:
            if (self.directory / "abort.json").exists():
                raise RuntimeError("RTC cohort aborted; see abort.json")
            if all((arrival / f"{member}.json").is_file() for member in self.members):
                result = dict(phase=phase, episode=episode, wait_wall_s=time.monotonic() - start,
                    monotonic_s=time.monotonic(), wall_time_s=time.time())
                atomic_json(arrival / f"{self.task}.released.json", result)
                return result
            if time.monotonic() - start > self.timeout_s:
                self.abort(f"{self.task}: timeout at episode {episode} phase {phase}")
                raise TimeoutError("RTC cohort rendezvous timeout")
            time.sleep(.02)


def live_start_stagger(index, count, action_interval_s, simulation_rate=1.):
    """Unique H4 phases, distributed across refresh16 before episode time zero.

    Refresh16/N alone aliases four peers onto each H4 phase at N=8.
    Preserve each independent episode clock/seed; only host scheduling changes.
    """
    import math
    if (type(index) is not int or type(count) is not int
            or not 1 <= count <= 8 or not 0 <= index < count
            or not math.isfinite(action_interval_s) or action_interval_s <= 0
            or not math.isfinite(simulation_rate) or not 0 < simulation_rate <= 1):
        raise ValueError("invalid cohort phase parameters")
    fast = 4 * float(action_interval_s) / simulation_rate
    slow_bucket = (4 * index) // count
    return slow_bucket * fast + index * fast / count
