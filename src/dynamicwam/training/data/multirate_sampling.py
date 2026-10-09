"""Episode holdout and deterministic task/episode/age balanced sampling."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import random
from torch.utils.data import Sampler


def partition_episodes(rows, *, seed=0, validation_fraction=.1):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    groups = defaultdict(dict)
    identities = {}
    for row in rows:
        episode = int(row['episode_index'])
        identity = (str(row['split']), str(row['task_name']), str(row['episode_name']))
        if episode in identities and identities[episode] != identity:
            raise ValueError("episode index has inconsistent identity")
        identities[episode] = identity
        groups[identity[:2]][episode] = identity
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda e: hashlib.sha256(
            json.dumps([seed, group[e]], sort_keys=True).encode()).hexdigest())
        # Single-episode strata remain train-only and are visible in the manifest.
        nval = min(len(ordered)-1, max(1, round(len(ordered)*validation_fraction)))
        for i, episode in enumerate(ordered):
            result[episode] = 'validation' if i < nval else 'train'
    return result


def pairing_identity(rows, assignments, *, seed, validation_fraction, clock_mode, offsets, stress_offsets):
    episodes = sorted({(int(r['episode_index']), str(r['split']), str(r['task_name']),
                        str(r['episode_name']), assignments[int(r['episode_index'])]) for r in rows})
    payload = dict(version=1, seed=seed, validation_fraction=validation_fraction,
                   clock_mode=clock_mode, offsets=list(offsets), stress_offsets=list(stress_offsets), episodes=episodes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class BalancedPairSampler(Sampler):
    """Draw a fixed number of pairs from every episode, then shard across ranks.

    Each epoch yields ``samples_per_episode`` indices per episode. Task size does
    not change that quota. Ranks split one shared shuffle, so they do not repeat
    the same draw.
    """

    def __init__(
        self,
        records,
        *,
        samples_per_episode=10,
        stress_probability=0.,
        seed=0,
        rank=0,
        world_size=1,
        age_weights=None,
    ):
        if samples_per_episode < 1 or not 0 <= stress_probability < 1:
            raise ValueError("invalid paired sampling budget")
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError("sampler rank is outside the process group")
        self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, r in enumerate(records):
            self.groups[r.task_name][r.episode_index][(r.offset_label, r.offset)].append(i)
        # Ordinary dicts are picklable by spawn-based DataLoader processes.
        self.groups = {t:{e:dict(ages) for e,ages in eps.items()} for t,eps in self.groups.items()}
        self.episodes = [
            (task, episode)
            for task in sorted(self.groups)
            for episode in sorted(self.groups[task])
        ]
        self.samples_per_episode = int(samples_per_episode)
        self.stress_probability = float(stress_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.age_weights = None if age_weights is None else {int(k): float(v) for k, v in age_weights.items()}
        if self.age_weights is not None and any(weight <= 0.0 for weight in self.age_weights.values()):
            raise ValueError("age weights must be positive")
        self.count = len(self.episodes) * self.samples_per_episode

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.count == 0:
            return 0
        return (self.count - self.rank + self.world_size - 1) // self.world_size

    def _choose(self, rng, ages):
        stress = rng.random() < self.stress_probability
        keys = sorted(key for key in ages if (key[0] == "stress") == stress)
        if not keys:
            keys = sorted(key for key in ages if key[0] == "steady_state")
        if not keys:
            raise RuntimeError("episode has no steady-state pairs")
        if self.age_weights is not None and not stress:
            weights = [self.age_weights.get(int(key[1]), 1.0) for key in keys]
            key = rng.choices(keys, weights=weights, k=1)[0]
        else:
            key = rng.choice(keys)
        return rng.choice(ages[key])

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for task, episode in self.episodes:
            ages = self.groups[task][episode]
            for _ in range(self.samples_per_episode):
                indices.append(self._choose(rng, ages))
        rng.shuffle(indices)
        return iter(indices[self.rank :: self.world_size])


class ProcessGroupAlignedSampler(Sampler):
    """Truncate a global sampler so Accelerate never cycles early batches.

    With ``even_batches=True``, Accelerate pads the incomplete final process
    round by repeating the first batches of the epoch. Multirate training keeps
    one global sequence and lets Accelerate shard it; this wrapper drops the
    leftover round so every rank sees the same step count without that reuse.
    """

    def __init__(self, base: Sampler, *, batch_size: int, num_processes: int) -> None:
        if batch_size < 1 or num_processes < 1:
            raise ValueError("batch size and process count must be positive")
        self.base = base
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def _usable(self) -> int:
        total = len(self.base)
        full_batches = total // self.batch_size
        aligned_batches = (full_batches // self.num_processes) * self.num_processes
        return aligned_batches * self.batch_size

    def __len__(self) -> int:
        return self._usable()

    def __iter__(self):
        usable = self._usable()
        if usable == 0 and len(self.base) > 0:
            raise RuntimeError(
                "paired epoch is shorter than one full process round; "
                f"need at least {self.batch_size * self.num_processes} samples"
            )
        for index, value in enumerate(self.base):
            if index >= usable:
                return
            yield value
