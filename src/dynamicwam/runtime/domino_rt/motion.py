"""Level-1 object motion tied to simulator time."""

from __future__ import annotations

from typing import Any

import numpy as np


def realtime_target_position(
    start_position: np.ndarray,
    velocity: np.ndarray,
    elapsed_seconds: float,
    constrain_z: float | None,
) -> np.ndarray:
    """Position of a constant-velocity target after ``elapsed_seconds``."""
    target = np.asarray(start_position, dtype=np.float64) + np.asarray(
        velocity,
        dtype=np.float64,
    ) * float(elapsed_seconds)
    if constrain_z is not None:
        target = np.array(target, dtype=np.float64, copy=True)
        target[2] = float(constrain_z)
    return target


def step_realtime_velocity_task(task: dict[str, Any], dt: float) -> None:
    """Advance one Level-1 realtime task by a physics tick."""
    import sapien

    task["elapsed_time"] = float(task["elapsed_time"]) + float(dt)
    target_position = realtime_target_position(
        task["start_position"],
        task["velocity"],
        task["elapsed_time"],
        task.get("constrain_z"),
    )
    task["component"].set_kinematic_target(
        sapien.Pose(p=target_position, q=task["initial_orientation"])
    )


def install_realtime_kinematic_dispatch(env: Any) -> None:
    """Teach an environment to step ``realtime_velocity`` tasks.

    The native extended-velocity update is left in place for every other task.
    """
    if getattr(env, "_realtime_motion_installed", False):
        return
    original = env._update_kinematic_tasks

    def update_with_realtime() -> None:
        tasks = list(env.active_kinematic_tasks)
        realtime = [task for task in tasks if task.get("type") == "realtime_velocity"]
        env.active_kinematic_tasks = [
            task for task in tasks if task.get("type") != "realtime_velocity"
        ]
        original()
        timestep = float(env.scene.get_timestep())
        for task in realtime:
            step_realtime_velocity_task(task, timestep)
        env.active_kinematic_tasks = list(env.active_kinematic_tasks) + realtime

    env._update_kinematic_tasks = update_with_realtime
    env._realtime_motion_installed = True


def adopt_level1_realtime_motion(env: Any) -> np.ndarray:
    """Switch the eval's one Level-1 task onto physical time.

    Returns the expert velocity in meters per second. Native
    ``extended_velocity`` behavior is unchanged until this is called.
    """
    saved = getattr(env, "_saved_dynamic_motion_info", None)
    if not isinstance(saved, dict):
        raise RuntimeError("level-1 realtime motion requires saved expert motion")
    params = saved.get("trajectory_params")
    if not isinstance(params, dict) or params.get("type") != "velocity":
        raise RuntimeError(
            "the realtime interface currently supports level-1 velocity motion"
        )
    velocity = np.asarray(params["velocity"], dtype=np.float64).copy()
    converted = 0
    for task in env.active_kinematic_tasks:
        if task.get("type") != "extended_velocity":
            continue
        task["type"] = "realtime_velocity"
        task["velocity"] = velocity.copy()
        task["elapsed_time"] = 0.0
        converted += 1
    if converted != 1:
        raise RuntimeError(
            f"expected one level-1 extended-velocity task, found {converted}"
        )
    install_realtime_kinematic_dispatch(env)
    return velocity
