"""DOMINO Level-1 session: physics ticks, object time, and one joint target."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from dynamicwam.runtime.ciwam.adapters.domino.composite import (
    composite_robotwin_frame,
    extract_state,
    head_camera_frame,
)
from dynamicwam.runtime.ciwam.flow import HeadFlowBuffer

from .motion import adopt_level1_realtime_motion
from .segment import QposSegment, TimedJointTrack, plan_qpos_segment, plan_timed_track


class HeadVideoWriter:
    """Write head-camera frames at the observation rate into one mp4."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._proc: subprocess.Popen[bytes] | None = None
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HWC uint8 RGB, got {frame.shape}")
        if self._proc is None:
            height, width = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required to write the eval video")
            self._proc = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        status = self._proc.wait()
        self._proc = None
        if status != 0:
            raise RuntimeError(f"ffmpeg exited with status {status} for {self.path}")


class DominoLevel1Session:
    """Tick a prepared Level-1 environment without blocking inside one target."""

    def __init__(
        self,
        env: Any,
        *,
        flow: HeadFlowBuffer,
        observation_config: dict[str, Any],
        stop_on_contact: bool,
        check_z_threshold: float | None,
        lift_actor: Any,
        start_z: float | None,
        video: HeadVideoWriter | None = None,
        instruction: str = "",
        control_interval_s: float = 0.1,
    ) -> None:
        self.env = env
        self.flow = flow
        self.observation_config = observation_config
        self.stop_on_contact = stop_on_contact
        self.check_z_threshold = check_z_threshold
        self.lift_actor = lift_actor
        self.start_z = start_z
        self.video = video
        self.instruction = instruction
        self.control_interval_s = float(control_interval_s)
        if self.control_interval_s <= 0.0:
            raise ValueError("control_interval_s must be positive")
        self.has_lifted = False
        self._stop_reason: str | None = None
        self._glide: dict[str, Any] | None = None
        self.metrics_tracker: Any | None = None
        self._metrics_next = 0.1

    def timestep(self) -> float:
        return float(self.env.scene.get_timestep())

    def physics_time(self) -> float:
        return float(self.env._scene_step_clock.snapshot().time_seconds)

    def object_position(self) -> np.ndarray:
        actor = getattr(self.env, "_eval_target_actor", None)
        if actor is None:
            raise RuntimeError("level-1 target actor is not initialized")
        return np.asarray(actor.actor.get_pose().p, dtype=np.float64).copy()

    def capture_packet(self) -> dict[str, Any]:
        """One control-grid sample: video, flow history, image, and proprioception."""
        observation = self.env.get_obs()
        if self.video is not None:
            self.video.write(head_camera_frame(observation))
        self.flow.push(
            head_camera_frame(observation),
            simulator_time_seconds=self.physics_time(),
        )
        return self._packet_from_observation(observation)

    def capture_policy_view(self) -> dict[str, Any]:
        """Image and proprioception between control ticks.

        Flow and video stay on the 100 ms grid. A request that cannot start on
        that grid still sees the current view, without inserting a second
        history frame at the same or an in-between time.
        """
        observation = self.env.get_obs()
        frame = composite_robotwin_frame(observation, self.observation_config)
        return {
            "frame": np.array(frame, copy=True),
            "state": np.array(extract_state(observation), copy=True),
        }

    def _packet_from_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        frame = composite_robotwin_frame(observation, self.observation_config)
        motion = self.flow.observation()
        return {
            "instruction": self.instruction,
            "frame": np.array(frame, copy=True),
            "state": np.array(extract_state(observation), copy=True),
            "flow_frames": np.array(motion.flow_rgb, copy=True),
            "motion_features": np.array(motion.motion_features, copy=True),
            "motion_interval_valid_mask": np.array(
                motion.interval_valid_mask,
                copy=True,
            ),
            "motion_acceleration_valid_mask": np.array(
                motion.acceleration_valid_mask,
                copy=True,
            ),
        }

    def native_action_count(self) -> int:
        return int(getattr(self.env, "take_action_cnt", 0))

    def begin_native_action(self) -> None:
        """Count one policy step the way native ``take_action`` does.

        ``extended_velocity`` reads this counter and moves the object by one
        scaled step for the whole TOPP, not by the expert speed times physics time.
        """
        self.env.take_action_cnt = self.native_action_count() + 1

    def begin_native_path(self, waypoints: int) -> None:
        """Glide the object across one native step per waypoint of this path.

        The distance is the native per-action displacement times the number of
        waypoints. It is spread over the whole TOPP instead of snapping at
        each waypoint and then waiting.
        """
        task = self._extended_task()
        count = int(waypoints)
        if task is None or count < 1:
            self._glide = None
            return
        origin = self.object_position()
        step = np.asarray(task["step_velocity"], dtype=np.float64).copy()
        self.env.take_action_cnt = self.native_action_count() + count
        task["last_policy_step"] = self.native_action_count()
        task["policy_target"] = origin.copy()
        self._glide = {
            "task": task,
            "origin": origin,
            "step": step,
            "waypoints": count,
        }

    def finish_native_path(self) -> None:
        if self._glide is None:
            return
        self._apply_glide(1.0)
        final = self._glide["origin"] + self._glide["step"] * self._glide["waypoints"]
        constrained_z = self._glide["task"].get("constrain_z")
        if constrained_z is not None:
            final = np.array(final, dtype=np.float64, copy=True)
            final[2] = float(constrained_z)
        task = self._glide["task"]
        task["policy_target"] = final.copy()
        task["start_position"] = final.copy()
        task["last_policy_step"] = self.native_action_count()
        self._glide = None

    def plan(self, action: np.ndarray) -> QposSegment:
        return plan_qpos_segment(self.env, action)

    def plan_suffix(self, actions: np.ndarray, *, elapsed_s: float = 0.0) -> TimedJointTrack:
        """Follow the remaining chunk on the 100 ms clock.

        ``elapsed_s`` is the time already lived past this suffix's origin, so
        a handoff inside a 100 ms knot does not jump back to the knot start.
        """
        return plan_timed_track(
            self.env,
            actions,
            interval_s=self.control_interval_s,
            dt=self.timestep(),
            elapsed_s=elapsed_s,
        )

    def step(self, segment: QposSegment | None) -> None:
        if segment is None:
            self._hold()
        else:
            segment.apply()
        self.env._update_kinematic_tasks()
        if (
            segment is not None
            and self._glide is not None
            and segment.control_ticks > 0
        ):
            self._apply_glide(segment.applied / segment.control_ticks)
        self.env.scene.step()
        self._consume_events()
        self._record_policy_metrics()

    def _record_policy_metrics(self) -> None:
        """Sample DOMINO's per-action metrics once per 100 ms, as take_action does."""
        tracker = self.metrics_tracker
        if tracker is None:
            return
        while self.physics_time() + 1.0e-9 >= self._metrics_next:
            try:
                tracker.on_step()
            except Exception:
                pass
            self._metrics_next += 0.1

    def _extended_task(self) -> dict[str, Any] | None:
        for task in getattr(self.env, "active_kinematic_tasks", []):
            if task.get("type") == "extended_velocity":
                return task
        return None

    def _apply_glide(self, fraction: float) -> None:
        glide = self._glide
        if glide is None:
            return
        import sapien

        progress = min(1.0, max(0.0, float(fraction)))
        target = glide["origin"] + glide["step"] * glide["waypoints"] * progress
        constrained_z = glide["task"].get("constrain_z")
        if constrained_z is not None:
            target = np.array(target, dtype=np.float64, copy=True)
            target[2] = float(constrained_z)
        glide["task"]["component"].set_kinematic_target(
            sapien.Pose(p=target, q=glide["task"]["initial_orientation"])
        )

    def stop_reason(self) -> str | None:
        return self._stop_reason

    def _hold(self) -> None:
        robot = self.env.robot
        robot._entity_qf(robot.left_entity)
        robot._entity_qf(robot.right_entity)

    def _consume_events(self) -> None:
        if self._stop_reason is not None:
            return
        if self.check_z_threshold is not None and not self.has_lifted:
            current_z = float(self.lift_actor.get_pose().p[2])
            assert self.start_z is not None
            if current_z - self.start_z > self.check_z_threshold:
                self.has_lifted = True
        if self.env.check_success():
            self.env.eval_success = True
            if self.check_z_threshold is not None and not self.has_lifted:
                self._stop_reason = "object_not_lifted"
            else:
                self._stop_reason = "success"
            return
        if not getattr(self.env, "use_dynamic", False):
            return
        if (
            self.env.check_gripper_contact_dynamic_object()
            and self.stop_on_contact
        ):
            self.env.stop_dynamic_object_motion()
        if not getattr(self.env, "_dynamic_object_stopped", False):
            if not self.env.check_dynamic_object_boundary():
                self.env.eval_fail = True
                self._stop_reason = "out_of_bounds"


def _expert_velocity(env: Any) -> np.ndarray:
    saved = getattr(env, "_saved_dynamic_motion_info", None)
    params = saved.get("trajectory_params") if isinstance(saved, dict) else None
    if not isinstance(params, dict) or params.get("type") != "velocity":
        raise RuntimeError("native-clock RTC requires a level-1 velocity task")
    return np.asarray(params["velocity"], dtype=np.float64).copy()


def prepare_level1_env(
    domino_root: Path,
    *,
    task_name: str,
    task_config: str,
    start_seed: int,
    instruction_type: str,
    replay: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run DOMINO's expert filter and move the object at the training speed.

    Training advances the object by the expert velocity. One 100 ms action is
    ``velocity * 0.1`` seconds of that motion. The episode uses that velocity
    on physics time, instead of spreading those steps across a TOPP.

    A seed whose extended trajectory hits a prohibited area makes
    ``init_dynamic_motion_for_eval`` return false. Native eval skips that seed
    and does not count it. This function does the same before returning.
    """
    root = domino_root.expanduser().resolve()
    os.chdir(root)
    for entry in (
        root,
        root / "script",
        root / "policy",
        root / "description" / "utils",
    ):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    import eval_policy  # noqa: WPS433  DOMINO evaluates with its script directory first.

    from envs import CONFIGS_PATH

    with open(root / "task_config" / f"{task_config}.yml", encoding="utf-8") as handle:
        args = eval_policy.yaml.load(handle.read(), Loader=eval_policy.yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "domino-rt-interface"
    args["policy_name"] = "domino_rt"
    args["eval_mode"] = True
    args["eval_video_log"] = False
    _install_embodiment(args, CONFIGS_PATH, eval_policy.yaml)
    env = eval_policy.class_decorator(task_name)
    if not args.get("use_dynamic", False):
        raise RuntimeError(f"{task_config} is not a dynamic DOMINO config")
    seed = int(start_seed)
    init_failures = 0
    while True:
        if replay is None:
            seed = _accept_expert_seed(env, args, seed, eval_policy.UnStableError)
            saved = getattr(env, "_saved_dynamic_motion_info", None)
        else:
            # A2 paired delays must reuse one accepted expert motion. Running
            # the planner again can change duration/start position at the same seed.
            import copy
            if int(replay["seed"]) != seed:
                raise ValueError("replay seed differs from requested seed")
            saved = copy.deepcopy(replay["saved_motion"])
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        args["render_freq"] = render_freq
        if saved is not None:
            env._saved_dynamic_motion_info = saved
        if replay is None:
            episode_info = getattr(env, "_rt_episode_info", None)
            if episode_info is None:
                raise RuntimeError("expert check did not keep episode info for the instruction")
            descriptions = eval_policy.generate_episode_descriptions(
                task_name,
                [episode_info["info"]],
                1,
            )
            instruction = str(np.random.choice(descriptions[0][instruction_type]))
        else:
            instruction = str(replay["instruction"])
        env.set_instruction(instruction=instruction)
        if env.init_dynamic_motion_for_eval():
            break
        if replay is not None:
            env.close_env()
            env.release_episode_resources()
            raise RuntimeError(f"accepted A2 motion replay failed at seed {seed}; no substitution")
        init_failures += 1
        print(f"dynamic motion init failed for seed {seed}, retrying", flush=True)
        try:
            env.close_env()
        except Exception:
            traceback.print_exc()
        env.release_episode_resources()
        if init_failures >= 2000:
            raise RuntimeError(
                f"dynamic motion init failed from {start_seed} through {seed}"
            )
        seed += 1
    # Native init stores one scaled step per action. Switch that task onto
    # the expert velocity so each physics second matches training.
    velocity = adopt_level1_realtime_motion(env)
    env.take_action_cnt = 0
    clock = env._scene_step_clock
    # Homing and the expert rehearsal advance the same step counter. The
    # episode clock starts when the policy loop starts.
    print(
        f"reset episode clock from {clock.time_seconds:.3f}s",
        flush=True,
    )
    clock.step_index = 0
    clock.time_seconds = 0.0
    dynamic_config = env.get_dynamic_motion_config() or {}
    check_z_threshold = dynamic_config.get("check_z_threshold")
    lift_actor = dynamic_config.get("check_z_actor", dynamic_config.get("target_actor"))
    start_z = None
    if check_z_threshold is not None and lift_actor is not None:
        start_z = float(lift_actor.get_pose().p[2])
    return env, {
        "instruction": instruction,
        "seed": seed,
        "velocity": velocity,
        "stop_on_contact": bool(dynamic_config.get("stop_on_contact", True)),
        "check_z_threshold": check_z_threshold,
        "lift_actor": lift_actor,
        "start_z": start_z,
    }


def _install_embodiment(args: dict[str, Any], configs_path: str, yaml_module: Any) -> None:
    embodiment_type = args.get("embodiment")
    with open(os.path.join(configs_path, "_embodiment_config.yml"), encoding="utf-8") as handle:
        embodiments = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    with open(os.path.join(configs_path, "_camera_config.yml"), encoding="utf-8") as handle:
        cameras = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head_type]["h"]
    args["head_camera_w"] = cameras[head_type]["w"]

    def embodiment_file(name: str) -> str:
        robot_file = embodiments[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"embodiment {name} has no file")
        return str(robot_file)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment must contain 1 or 3 items")
    args["left_embodiment_config"] = _read_yaml(
        os.path.join(args["left_robot_file"], "config.yml"),
        yaml_module,
    )
    args["right_embodiment_config"] = _read_yaml(
        os.path.join(args["right_robot_file"], "config.yml"),
        yaml_module,
    )


def _read_yaml(path: str, yaml_module: Any) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml_module.load(handle.read(), Loader=yaml_module.FullLoader)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"expected a mapping in {path}")
    return loaded


def _accept_expert_seed(env: Any, args: dict[str, Any], start_seed: int, unstable: type) -> int:
    seed = int(start_seed)
    render_freq = args["render_freq"]
    # Native DOMINO keeps skipping unstable scenes until one is feasible.
    # A short cap makes the caller retry an almost identical window.
    for _attempt in range(2000):
        args["render_freq"] = 0
        print(f"expert check seed {seed}", flush=True)
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
            episode_info = env.play_once()
            saved_motion = getattr(env, "_saved_dynamic_motion_info", None)
            env.close_env()
        except unstable:
            env.close_env()
            seed += 1
            args["render_freq"] = render_freq
            continue
        except Exception:
            traceback.print_exc()
            try:
                env.close_env()
            except Exception:
                traceback.print_exc()
            seed += 1
            args["render_freq"] = render_freq
            continue
        args["render_freq"] = render_freq
        if env.plan_success and env.check_success():
            env._rt_episode_info = episode_info
            env._saved_dynamic_motion_info = saved_motion
            return seed
        seed += 1
    raise RuntimeError(
        f"no feasible level-1 seed found from {start_seed} through {seed - 1}"
    )
