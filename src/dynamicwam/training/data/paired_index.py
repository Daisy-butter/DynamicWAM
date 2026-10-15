"""Index rules for pairing an earlier source frame with a later fast frame.

``global_downsample_rate`` maps a demonstration condition frame to action
labels. It is not a claim that one online action always advances three camera
frames. Online age is counted in completed actions and simulator time.
"""

from __future__ import annotations


def action_label_indices(
    condition_frame: int,
    *,
    chunk_size: int,
    global_downsample_rate: int,
) -> list[int]:
    if condition_frame < 0 or chunk_size < 1 or global_downsample_rate < 1:
        raise ValueError("action label indices require a non-negative frame and positive steps")
    return [
        condition_frame + (step + 1) * global_downsample_rate
        for step in range(chunk_size)
    ]


def pair_fits_episode(
    source_frame: int,
    fast_frame: int,
    *,
    total_frames: int,
    chunk_size: int,
    global_downsample_rate: int,
) -> bool:
    if fast_frame < source_frame:
        return False
    last_action = action_label_indices(
        fast_frame,
        chunk_size=chunk_size,
        global_downsample_rate=global_downsample_rate,
    )[-1]
    return last_action < total_frames


def steady_state_offsets(
    *,
    execute_horizon: int,
    every_fast_queries: int,
    global_downsample_rate: int,
) -> list[int]:
    """Demonstration-frame offsets consumed between slow refreshes.

    R=4 and H_exec=4 consume ages 0, 4, 8, 12 actions, which is 0, 12, 24, 36
    raw frames under the demonstration mapping. The next refresh is not itself
    a consumed age.
    """
    if execute_horizon < 1 or every_fast_queries < 1:
        raise ValueError("execute horizon and R must be positive")
    return [
        step * execute_horizon * global_downsample_rate
        for step in range(every_fast_queries)
    ]


def deployment_age_weights(
    *,
    execute_horizon: int,
    every_fast_queries: int,
    global_downsample_rate: int,
) -> dict[int, float]:
    """Weight each steady frame offset by how often deployment reads it.

    Controlled refresh visits every slot once per cycle. After startup, a
    successful independent worker revisits the later slots and does not return
    to age 0. Counting both schedules draws each reused age twice as often as a
    fresh bridge. Ages beyond the trained TTL are not added here.
    """

    offsets = steady_state_offsets(
        execute_horizon=execute_horizon,
        every_fast_queries=every_fast_queries,
        global_downsample_rate=global_downsample_rate,
    )
    return {
        int(offset): 1.0 if step == 0 else 2.0
        for step, offset in enumerate(offsets)
    }


def causal_cache_offsets(*, cache_age_actions, global_downsample_rate: int) -> list[int]:
    """Map an explicitly trained cache-age support to demonstration frames.

    Cache age describes old information at the current observation, not a future
    observation or a claim about fixed demonstration action duration.
    """
    ages = list(cache_age_actions)
    if (global_downsample_rate < 1 or not ages or any(
        isinstance(age, bool) or not isinstance(age, int) or age < 0 for age in ages
    ) or len(set(ages)) != len(ages) or 0 not in ages):
        raise ValueError("cache ages must be unique nonnegative integers including zero")
    return [age * global_downsample_rate for age in sorted(ages)]


def action_target_times(frame_times: dict[int, float], condition_frame: int, *,
                        chunk_size: int, global_downsample_rate: int) -> list[float]:
    """Exact simulator endpoint times for packed target joint positions.

    These are TARGET endpoint times, not action command START times. A runtime
    queue must convert explicitly using its own anchor/interval semantics.
    """
    import math
    indices = [condition_frame] + action_label_indices(condition_frame,
        chunk_size=chunk_size, global_downsample_rate=global_downsample_rate)
    try:
        times = [float(frame_times[i]) for i in indices]
    except KeyError as exc:
        raise ValueError(f"missing simulator timestamp for action label frame {exc.args[0]}") from exc
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("action target simulator timestamps must be finite and strictly increasing")
    return times[1:]


def jittered_cache_age(*, source_sample_id: int, base_age: int, slot_width: int,
                       valid_ages, seed: int, epoch: int) -> int:
    """A reproducible draw within one causal cache slot; independent of rank.

    Source identity and epoch vary the draw without expanding pair records.
    The caller supplies only ages with a complete packed action target window.
    """
    import hashlib
    import json
    valid = sorted(int(age) for age in valid_ages)
    if slot_width < 1 or epoch < 0 or not valid or any(
        age < base_age or age >= base_age + slot_width for age in valid
    ):
        raise ValueError("jitter ages must fit one nonempty causal cache slot")
    digest = hashlib.sha256(json.dumps([int(seed), int(epoch), int(source_sample_id),
        int(base_age)], separators=(",", ":")).encode()).digest()
    return valid[int.from_bytes(digest[:8], "big") % len(valid)]
