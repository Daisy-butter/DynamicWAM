"""One DOMINO joint target, played for one fixed control interval."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def control_tick_count(interval_s: float, dt: float) -> int:
    """Physics ticks in one DynamicWAM control step.

    The published control interval is 100 ms. At the DOMINO scene step of
    1/250 s that is 25 ticks, whatever duration TOPP would have chosen.
    """
    interval = float(interval_s)
    step = float(dt)
    if interval <= 0.0 or step <= 0.0:
        raise ValueError(
            f"control interval and timestep must be positive, got {interval} and {step}"
        )
    ticks = int(round(interval / step))
    if ticks < 1:
        raise ValueError(
            f"control interval {interval} is shorter than one physics tick {step}"
        )
    return ticks


class QposSegment:
    """TOPP playback clipped or padded to one control interval."""

    def __init__(
        self,
        env: Any,
        *,
        left_position: np.ndarray | None,
        left_velocity: np.ndarray | None,
        left_gripper: np.ndarray,
        right_position: np.ndarray | None,
        right_velocity: np.ndarray | None,
        right_gripper: np.ndarray,
        left_steps: int,
        right_steps: int,
        use_left_arm: bool,
        use_right_arm: bool,
        control_ticks: int,
    ) -> None:
        self.env = env
        self.left_position = left_position
        self.left_velocity = left_velocity
        self.left_gripper = left_gripper
        self.right_position = right_position
        self.right_velocity = right_velocity
        self.right_gripper = right_gripper
        self.left_steps = int(left_steps)
        self.right_steps = int(right_steps)
        self.use_left_arm = bool(use_left_arm)
        self.use_right_arm = bool(use_right_arm)
        self.control_ticks = int(control_ticks)
        self.left_index = 0
        self.right_index = 0
        self.applied = 0
        # When paced, playback may not run ahead of the policy-time horizon.
        self.paced = False
        self.progress_limit = 1.0
        if self.left_steps <= 0 or self.right_steps <= 0 or self.control_ticks <= 0:
            raise ValueError("a qpos segment needs a positive tick count on each arm")

    @property
    def finished(self) -> bool:
        return self.applied >= self.control_ticks

    def apply(self) -> None:
        """Command one physics tick inside this control interval. Does not step physics."""
        if self.finished:
            raise RuntimeError("qpos segment is already finished")
        if self.paced and self.control_ticks > 0:
            self.progress_limit = min(1.0, (self.applied + 1) / float(self.control_ticks))
        topp_done = (
            self.left_index >= self.left_steps and self.right_index >= self.right_steps
        )
        if topp_done or self._both_gated():
            self._hold()
        else:
            self._apply_topp_tick()
        self.applied += 1

    def _hold(self) -> None:
        robot = self.env.robot
        robot._entity_qf(robot.left_entity)
        robot._entity_qf(robot.right_entity)

    def _allowed(self, steps: int) -> int:
        if not self.paced:
            return int(steps)
        return int(math.ceil(self.progress_limit * int(steps) - 1.0e-9))

    def _both_gated(self) -> bool:
        if not self.paced:
            return False
        return (
            self.left_index >= self._allowed(self.left_steps)
            and self.right_index >= self._allowed(self.right_steps)
        )

    def _apply_topp_tick(self) -> None:
        robot = self.env.robot
        if (
            self.left_index < self.left_steps
            and self.left_index < self._allowed(self.left_steps)
            and self.left_index / self.left_steps
            <= self.right_index / self.right_steps
        ):
            if self.use_left_arm:
                assert self.left_position is not None
                assert self.left_velocity is not None
                robot.set_arm_joints(
                    self.left_position[self.left_index],
                    self.left_velocity[self.left_index],
                    "left",
                )
            robot.set_gripper(self.left_gripper[self.left_index], "left")
            self.left_index += 1
        if (
            self.right_index < self.right_steps
            and self.right_index < self._allowed(self.right_steps)
            and self.right_index / self.right_steps
            <= self.left_index / self.left_steps
        ):
            if self.use_right_arm:
                assert self.right_position is not None
                assert self.right_velocity is not None
                robot.set_arm_joints(
                    self.right_position[self.right_index],
                    self.right_velocity[self.right_index],
                    "right",
                )
            robot.set_gripper(self.right_gripper[self.right_index], "right")
            self.right_index += 1


def plan_qpos_segment(
    env: Any,
    action: np.ndarray,
    control_ticks: int | None = None,
) -> QposSegment:
    """Plan native TOPP from the current joints through one or more absolute waypoints.

    ``control_ticks`` clips or pads playback. The realtime episode leaves it
    unset so the arm follows the planned path until both arms finish, instead
    of stopping after one 100 ms slice of a multi-second reach.
    """
    target = np.asarray(action, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    left_jointstate = env.robot.get_left_arm_jointState()
    right_jointstate = env.robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1
    right_arm_dim = len(right_jointstate) - 1
    current = np.asarray(left_jointstate + right_jointstate, dtype=np.float64)
    left_arm = target[:, :left_arm_dim]
    left_gripper_action = target[:, left_arm_dim]
    right_arm = target[
        :,
        left_arm_dim + 1 : left_arm_dim + right_arm_dim + 1,
    ]
    right_gripper_action = target[:, left_arm_dim + right_arm_dim + 1]
    left_current = current[:left_arm_dim]
    right_current = current[left_arm_dim + 1 : left_arm_dim + right_arm_dim + 1]
    left_path = np.vstack((left_current, left_arm))
    right_path = np.vstack((right_current, right_arm))

    use_left, left_position, left_velocity, left_steps = _plan_arm(
        env.robot.left_mplib_planner,
        left_path,
    )
    use_right, right_position, right_velocity, right_steps = _plan_arm(
        env.robot.right_mplib_planner,
        right_path,
    )
    left_gripper = _gripper_track(
        env.robot.get_left_gripper_val(),
        left_gripper_action,
        left_steps,
    )
    right_gripper = _gripper_track(
        env.robot.get_right_gripper_val(),
        right_gripper_action,
        right_steps,
    )
    playback_ticks = (
        max(left_steps, right_steps)
        if control_ticks is None
        else int(control_ticks)
    )
    return QposSegment(
        env,
        left_position=left_position,
        left_velocity=left_velocity,
        left_gripper=left_gripper,
        right_position=right_position,
        right_velocity=right_velocity,
        right_gripper=right_gripper,
        left_steps=left_steps,
        right_steps=right_steps,
        use_left_arm=use_left,
        use_right_arm=use_right,
        control_ticks=playback_ticks,
    )


def _plan_arm(
    planner: Any,
    path: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, int]:
    try:
        _times, position, velocity, _acceleration, _duration = planner.TOPP(
            path,
            1 / 250,
            verbose=False,
        )
        steps = int(position.shape[0])
        if steps == 0:
            return False, None, None, 50
        return True, np.asarray(position), np.asarray(velocity), steps
    except Exception:
        return False, None, None, 50


def _gripper_track(current: Any, targets: np.ndarray, steps: int) -> np.ndarray:
    path = np.hstack((np.asarray(current, dtype=np.float64).reshape(-1), targets))
    counts = _gripper_region_lengths(steps, len(targets))
    pieces: list[float] = []
    for index in range(1, path.shape[0]):
        region = np.linspace(path[index - 1], path[index], counts[index] + 1)[1:]
        pieces.extend(float(value) for value in region)
    track = np.asarray(pieces, dtype=np.float64)
    if track.shape[0] != steps:
        raise RuntimeError(
            f"gripper track length {track.shape[0]} does not match {steps} ticks"
        )
    return track


def _gripper_region_lengths(steps: int, regions: int) -> list[int]:
    remainder = steps % regions
    lengths = [
        steps // regions + (1 if index < remainder else 0) for index in range(regions)
    ]
    return [0, *lengths]


def node_velocity(points: np.ndarray, index: int, interval_s: float) -> np.ndarray:
    """Velocity of one 100 ms knot, using the neighboring knots.

    The first and last knots take the one-sided difference. Interior knots take
    the central difference, so the spline passes through every joint target
    with a nonzero speed whenever the targets themselves are moving.
    """
    count = int(points.shape[0])
    if count <= 1:
        return np.zeros(points.shape[1], dtype=np.float64)
    interval = float(interval_s)
    if interval <= 0.0:
        raise ValueError(f"waypoint interval must be positive, got {interval}")
    if index <= 0:
        return (points[1] - points[0]) / interval
    if index >= count - 1:
        return (points[-1] - points[-2]) / interval
    return (points[index + 1] - points[index - 1]) / (2.0 * interval)


def sample_linear_waypoints(
    points: np.ndarray,
    tau: float,
    interval_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity on the straight segment between two RTC actions.

    Knot ``i`` is the pose returned for control step ``i``. During that step
    the command moves toward knot ``i + 1`` at the velocity of that one step.
    """
    schedule = np.asarray(points, dtype=np.float64)
    if schedule.ndim != 2 or schedule.shape[0] < 1:
        raise ValueError(f"waypoints must have shape [count, dim], got {schedule.shape}")
    if schedule.shape[0] == 1 or float(tau) <= 0.0:
        if schedule.shape[0] == 1:
            return schedule[0].copy(), np.zeros(schedule.shape[1], dtype=np.float64)
        return schedule[0].copy(), (schedule[1] - schedule[0]) / float(interval_s)
    interval = float(interval_s)
    if interval <= 0.0:
        raise ValueError(f"waypoint interval must be positive, got {interval}")
    span = (schedule.shape[0] - 1) * interval
    if float(tau) >= span:
        return schedule[-1].copy(), np.zeros(schedule.shape[1], dtype=np.float64)
    index = min(schedule.shape[0] - 2, int(math.floor(float(tau) / interval + 1.0e-12)))
    local = (float(tau) - index * interval) / interval
    position = (1.0 - local) * schedule[index] + local * schedule[index + 1]
    velocity = (schedule[index + 1] - schedule[index]) / interval
    return position, velocity


def sample_timed_waypoints(
    points: np.ndarray,
    tau: float,
    interval_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity on the cubic schedule through ``points``.

    Knot ``i`` is the pose at ``i * interval_s``. The cubic matches those
    poses and :func:`node_velocity`, so the command does not stop at a knot.
    Past the last knot the pose is held and the velocity is zero.
    """
    schedule = np.asarray(points, dtype=np.float64)
    if schedule.ndim != 2 or schedule.shape[0] < 1:
        raise ValueError(f"waypoints must have shape [count, dim], got {schedule.shape}")
    if schedule.shape[0] == 1 or float(tau) <= 0.0:
        if schedule.shape[0] == 1:
            return schedule[0].copy(), np.zeros(schedule.shape[1], dtype=np.float64)
        return schedule[0].copy(), node_velocity(schedule, 0, interval_s)
    span = (schedule.shape[0] - 1) * float(interval_s)
    if float(tau) >= span:
        return schedule[-1].copy(), np.zeros(schedule.shape[1], dtype=np.float64)
    index = min(
        schedule.shape[0] - 2,
        int(math.floor(float(tau) / float(interval_s) + 1.0e-12)),
    )
    local = (float(tau) - index * float(interval_s)) / float(interval_s)
    previous = schedule[index]
    following = schedule[index + 1]
    previous_velocity = node_velocity(schedule, index, interval_s)
    following_velocity = node_velocity(schedule, index + 1, interval_s)
    local_sq = local * local
    local_cu = local_sq * local
    position_basis = (
        (2.0 * local_cu - 3.0 * local_sq + 1.0) * previous
        + (local_cu - 2.0 * local_sq + local) * float(interval_s) * previous_velocity
        + (-2.0 * local_cu + 3.0 * local_sq) * following
        + (local_cu - local_sq) * float(interval_s) * following_velocity
    )
    velocity_basis = (
        (6.0 * local_sq - 6.0 * local) * previous
        + (3.0 * local_sq - 4.0 * local + 1.0) * float(interval_s) * previous_velocity
        + (-6.0 * local_sq + 6.0 * local) * following
        + (3.0 * local_sq - 2.0 * local) * float(interval_s) * following_velocity
    )
    return position_basis, velocity_basis / float(interval_s)


class TimedJointTrack:
    """Play absolute joint targets on the 100 ms clock, without a new TOPP.

    RTC consumes one action per control step. Knot 0 is the action for now.
    Until the next knot, the command moves in a straight line at that step's
    velocity. There is no pull from the joints commanded before the swap.
    """

    time_indexed = True

    def __init__(
        self,
        env: Any,
        actions: np.ndarray,
        *,
        interval_s: float,
        dt: float,
    ) -> None:
        target = np.asarray(actions, dtype=np.float64)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        if target.shape[0] < 1:
            raise ValueError("a timed joint track needs at least one waypoint")
        self.env = env
        self.interval_s = float(interval_s)
        self.dt = float(dt)
        if self.interval_s <= 0.0 or self.dt <= 0.0:
            raise ValueError("interval and timestep must be positive")
        left_jointstate = env.robot.get_left_arm_jointState()
        right_jointstate = env.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1
        right_arm_dim = len(right_jointstate) - 1
        expected = left_arm_dim + right_arm_dim + 2
        if target.shape[1] != expected:
            raise ValueError(
                f"action dim {target.shape[1]} does not match arms+grippers {expected}"
            )
        self.left = target[:, :left_arm_dim]
        self.left_gripper = target[:, left_arm_dim]
        self.right = target[
            :,
            left_arm_dim + 1 : left_arm_dim + right_arm_dim + 1,
        ]
        self.right_gripper = target[:, left_arm_dim + right_arm_dim + 1]
        self.applied = 0
        self._tau0 = 0.0
        self.control_ticks = max(1, int(round(target.shape[0] * self.interval_s / self.dt)))
        self.paced = False

    def prime(self, elapsed_s: float) -> None:
        """Start part-way through this suffix.

        ``elapsed_s`` is how far the world already is past the suffix origin.
        The next ``apply`` samples that phase instead of knot 0.
        """
        elapsed = max(0.0, float(elapsed_s))
        self._tau0 = elapsed
        consumed = int(math.floor(elapsed / self.dt + 1.0e-9))
        if consumed >= self.control_ticks:
            self.applied = self.control_ticks
            return
        self.control_ticks -= consumed

    @property
    def finished(self) -> bool:
        return self.applied >= self.control_ticks

    def apply(self) -> None:
        if self.finished:
            raise RuntimeError("timed joint track is already finished")
        tau = self._tau0 + self.applied * self.dt
        robot = self.env.robot
        left_position, left_velocity = sample_linear_waypoints(self.left, tau, self.interval_s)
        right_position, right_velocity = sample_linear_waypoints(
            self.right, tau, self.interval_s
        )
        robot.set_arm_joints(left_position, left_velocity, "left")
        robot.set_arm_joints(right_position, right_velocity, "right")
        left_grip, _left_grip_velocity = sample_linear_waypoints(
            self.left_gripper.reshape(-1, 1),
            tau,
            self.interval_s,
        )
        right_grip, _right_grip_velocity = sample_linear_waypoints(
            self.right_gripper.reshape(-1, 1),
            tau,
            self.interval_s,
        )
        robot.set_gripper(float(left_grip[0]), "left")
        robot.set_gripper(float(right_grip[0]), "right")
        self.applied += 1


def plan_timed_track(
    env: Any,
    action: np.ndarray,
    *,
    interval_s: float,
    dt: float,
    elapsed_s: float = 0.0,
) -> TimedJointTrack:
    """Schedule one RTC suffix on the policy clock."""
    track = TimedJointTrack(env, action, interval_s=interval_s, dt=dt)
    if float(elapsed_s) > 0.0:
        track.prime(elapsed_s)
    return track
