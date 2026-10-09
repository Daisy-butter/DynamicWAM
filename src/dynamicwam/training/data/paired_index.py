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
