"""DOMINO integration for full-chunk continuous synchronous playback."""
from copy import deepcopy
import time
import numpy as np

from .domino_session import DominoLevel1Session
from .camera_stream import SelectiveAsyncCameraCapture, PhysicalTimeHeadHistory
from .continuous_track_v2 import ContinuousJointTrack
from ..ciwam.adapters.domino.composite import head_camera_frame, composite_robotwin_frame

# Timing modes keep the continuous world / no-RTC loop identical; only whether
# MPLIB joint v/a limits may stretch chunk duration changes.
TIMING_NATIVE_MPLIB = "native_mplib_limits"
TIMING_NOMINAL_TRAINING = "nominal_training_timing"
PROTOCOL_BY_TIMING = {
    TIMING_NATIVE_MPLIB: "continuous_world_actual_latency_sync16_continuous_track_v2",
    TIMING_NOMINAL_TRAINING: "continuous_world_actual_latency_sync16_continuous_track_nominal_timing_v1",
}
CONTROLLER_BY_TIMING = {
    TIMING_NATIVE_MPLIB: "continuous_shape_preserving_bounded_joint_track",
    TIMING_NOMINAL_TRAINING: "continuous_shape_preserving_nominal_timing_track",
}


class ContinuousSyncSession(DominoLevel1Session):
    def __init__(self, *args, action_interval_s=.16, save_video=True,
                 timing_mode="native_mplib_limits", replay_recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        from . import _sapien_color_reader
        self.camera_reader = SelectiveAsyncCameraCapture(batch_reader=_sapien_color_reader.read_rgb_batch)
        self.pending_capture = None
        self.replay_recorder = replay_recorder
        self._step_profile = {"steps": 0, "commanded_steps": 0,
            "qf_requested": 0, "qf_computed": 0, "qf_reused": 0, "timing_ms": {}}
        self.action_interval_s = float(action_interval_s)
        if timing_mode not in PROTOCOL_BY_TIMING:
            raise ValueError(
                f"timing_mode must be one of {sorted(PROTOCOL_BY_TIMING)}; got {timing_mode!r}")
        self.timing_mode = timing_mode
        self.protocol = PROTOCOL_BY_TIMING[timing_mode]
        self.controller = CONTROLLER_BY_TIMING[timing_mode]
        self.limit_retiming = timing_mode == TIMING_NATIVE_MPLIB
        if save_video:
            from video_artifacts import BufferedHeadVideo
            self.head_video = BufferedHeadVideo()
        else:
            self.head_video = None
        self._v, self._a, self._q_limits, self._constraint_source = self._robot_limits()

    def _robot_limits(self):
        robot = self.env.robot
        velocities, accelerations, positions, sources = [], [], [], []
        for side in ("left", "right"):
            planner = getattr(robot, f"{side}_mplib_planner").planner
            ids = list(planner.move_group_joint_indices)
            names = [planner.user_joint_names[i] for i in ids]
            arm_names = [joint.get_name() for joint in getattr(robot, f"{side}_arm_joints")]
            if len(names) != 6 or len(arm_names) != 6 or set(names) != set(arm_names):
                raise ValueError(f"unexpected {side} arm joint order: {names} vs {arm_names}")
            order = [names.index(name) for name in arm_names]
            v = np.asarray(planner.joint_vel_limits, dtype=float)[order]
            a = np.asarray(planner.joint_acc_limits, dtype=float)[order]
            bounds = np.asarray(planner.joint_limits, dtype=float)[np.asarray(ids)[order]]
            if v.shape != (6,) or a.shape != (6,) or bounds.shape != (6,2):
                raise ValueError("MPLIB arm constraints have unexpected shape")
            velocities.extend([*v, 1.])
            accelerations.extend([*a, 1.])
            positions.extend([*bounds, [0.,1.]])
            sources.append({"arm": side, "joint_names": arm_names, "velocity_limits": v.tolist(),
                            "acceleration_limits": a.tolist(), "source": "existing_native_MPLIB_planner"})
        return np.asarray(velocities), np.asarray(accelerations), np.asarray(positions), sources

    def current_drive_targets(self):
        """Exact native commands retained during a hold, for opt-in replay."""
        robot = self.env.robot
        return np.asarray(robot.get_left_arm_jointState()+robot.get_right_arm_jointState(), dtype=np.float64)

    def current_qpos(self):
        # The ordinary jointState API returns drive targets, not articulation
        # feedback. Use measured arm positions for interpolation and tracking.
        before = time.perf_counter()
        result = np.asarray(self.env.robot.get_left_arm_real_jointState()+self.env.robot.get_right_arm_real_jointState(), dtype=float)
        self._record_step_timing("feedback", time.perf_counter() - before)
        return result

    def plan_chunk(self, actions):
        current = self.current_qpos()
        # Physical position limits are an explicit execution guard. It never
        # removes or reorders actions; every altered coordinate is reported.
        targets = np.clip(np.asarray(actions, dtype=float), self._q_limits[:,0], self._q_limits[:,1])
        limits = self._q_limits.copy()
        initial_error = np.maximum(limits[:,0]-current, 0)+np.maximum(current-limits[:,1], 0)
        # A measured articulation can slightly overshoot its stop. Start from
        # that actual pose and return within bounds, without teleporting qpos.
        limits[:,0] = np.minimum(limits[:,0], current)
        limits[:,1] = np.maximum(limits[:,1], current)
        axes = np.ones(14, dtype=bool); axes[[6,13]] = False
        # Native mode: MPLIB default v/a (=1.0 when unset) may stretch timing.
        # Nominal ablation: same Hermite path at training 160 ms; limits are only
        # reported, never used to lengthen the chunk.
        track = ContinuousJointTrack(current, targets, self.action_interval_s, self._v, self._a,
                                     position_limits=limits, constrained_axes=axes,
                                     limit_retiming=self.limit_retiming)
        track.guard_diagnostics = {
            "constraint_source": self._constraint_source,
            "position_guard_clipped_coordinates": int(np.count_nonzero(targets != actions)),
            "position_guard_max_change": float(np.max(np.abs(targets-actions))),
            "initial_position_limit_error": initial_error.tolist(),
            "feedback_source": "measured_arm_qpos; gripper_coordinates_are_native_normalized_commands",
            "gripper_control": "native_set_gripper; normalized coordinates excluded from arm v/a constraints",
            "timing_mode": self.timing_mode,
            "limit_retiming": self.limit_retiming,
            "protocol": self.protocol,
            "mplib_velocity_limits": self._v.tolist(),
            "mplib_acceleration_limits": self._a.tolist(),
        }
        return track

    def validate_camera_reader(self, reference_observation):
        """Compare all four already-rendered buffers before the episode clock.

        Caller obtains reference with env.get_obs(); this method never renders,
        advances physics, or mutates the scene. No fallback is permitted.
        """
        import hashlib
        from . import _sapien_color_reader
        if self.camera_reader.busy:
            raise RuntimeError("cannot validate while a capture is pending")
        cameras = self.camera_reader._camera_handles(self.env)
        started = time.perf_counter()
        images, durations = _sapien_color_reader.read_rgb_batch(tuple(handle for _, handle in cameras))
        if len(images) != len(cameras) or len(durations) != len(cameras):
            raise RuntimeError("native batch validation camera count mismatch")
        receipts = []
        for (name, _), image, duration in zip(cameras, images, durations):
            reference = reference_observation["observation"][name]["rgb"]
            if (reference.dtype != np.uint8 or image.dtype != np.uint8
                    or not image.flags.owndata or not image.flags.c_contiguous
                    or not np.array_equal(reference, image)):
                raise RuntimeError(f"native RGB batch differs from original camera pixels: {name}")
            receipts.append({"camera": name, "shape": list(image.shape),
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "equal": True, "native_readback_conversion_s": float(duration)})
        return {"contract": _sapien_color_reader.rgb_batch_contract,
            "passed": True, "camera_count": len(cameras), "all_equal": True, "cameras": receipts,
            "wall_ms": 1000 * (time.perf_counter() - started),
            "same_submitted_buffers": True, "new_render": False, "physics_stepped": False}

    def begin_capture(self, *, for_inference):
        self.capture_for_inference = bool(for_inference)
        self.pending_capture = self.camera_reader.begin(self.env, physics_time_s=self.physics_time(),
                                                        head_only=not for_inference and self.replay_recorder is None)

    def capture_ready(self):
        return self.camera_reader.ready(self.pending_capture)

    def latest_capture_time(self):
        return self.flow.latest_timestamp

    def history_diagnostics(self):
        return self.flow.diagnostics

    def record_video_capture(self, captured):
        if self.head_video is not None:
            self.head_video.append(head_camera_frame(captured.observation), physics_time_s=captured.physics_time_s,
                                   capture_wall_perf_s=captured.capture_wall_perf_s)

    def finish_capture(self):
        before = time.perf_counter()
        captured = self.camera_reader.finish(self.pending_capture)
        consumed = time.perf_counter()
        self.pending_capture = None
        self.record_video_capture(captured)
        video_done = time.perf_counter()
        self.flow.push(head_camera_frame(captured.observation), simulator_time_seconds=captured.physics_time_s)
        history_done = time.perf_counter()
        if self.replay_recorder is not None:
            self.replay_recorder.record_frame(captured.physics_time_s,
                composite_robotwin_frame(captured.observation,self.observation_config),
                for_inference=self.capture_for_inference)
        packet = None
        if self.capture_for_inference:
            self.env.now_obs = deepcopy(captured.observation)
            packet = self._packet_from_observation(captured.observation)
        done = time.perf_counter()
        self.last_capture_diagnostics = {
            "render_submit_ms": 1000 * (captured.render_submitted_wall_perf_s - captured.capture_wall_perf_s),
            "native_readback_ms": 1000 * (captured.readback_finished_wall_perf_s - captured.readback_started_wall_perf_s),
            "camera_readback_seconds": captured.camera_readback_seconds,
            "owner_consume_ms": 1000 * (consumed - before),
            "video_append_ms": 1000 * (video_done - consumed),
            "history_push_ms": 1000 * (history_done - video_done),
            "packet_and_flow_ms": 1000 * (done - history_done),
            "owner_finish_ms": 1000 * (done - before),
        }
        return packet

    def _ensure_step_profile(self):
        if not hasattr(self, "_step_profile"):
            self._step_profile = {"steps": 0, "commanded_steps": 0,
                "qf_requested": 0, "qf_computed": 0, "qf_reused": 0, "timing_ms": {}}
        return self._step_profile

    def _record_step_timing(self, name, elapsed_s):
        self._ensure_step_profile()
        bucket = self._step_profile["timing_ms"].setdefault(name, {"count": 0, "total": 0., "max": 0.})
        value = 1000 * elapsed_s
        bucket["count"] += 1
        bucket["total"] += value
        bucket["max"] = max(bucket["max"], value)

    def runtime_timing_summary(self):
        result = deepcopy(self._ensure_step_profile())
        for bucket in result["timing_ms"].values():
            bucket["mean"] = bucket["total"] / bucket["count"] if bucket["count"] else 0.
        return result

    def step_command(self, position, velocity):
        self._ensure_step_profile()
        before = time.perf_counter()
        if position is None:
            self._hold()
        else:
            robot = self.env.robot
            original_qf = robot._entity_qf
            had_override = "_entity_qf" in vars(robot)
            previous_override = vars(robot).get("_entity_qf")
            seen = set()
            requested = computed = 0

            def once_per_entity(entity):
                nonlocal requested, computed
                requested += 1
                key = id(entity)
                if key not in seen:
                    # No physics step/qpos/qvel mutation occurs between the
                    # four native setters. Passive gravity/Coriolis forces are
                    # therefore identical across their repeated calls. Keep
                    # every original arm/gripper setter and ordering intact.
                    original_qf(entity)
                    seen.add(key)
                    computed += 1

            robot._entity_qf = once_per_entity
            try:
                robot.set_arm_joints(position[:6], velocity[:6], "left")
                robot.set_arm_joints(position[7:13], velocity[7:13], "right")
                robot.set_gripper(position[6], "left")
                robot.set_gripper(position[13], "right")
            finally:
                if had_override:
                    robot._entity_qf = previous_override
                else:
                    del robot._entity_qf
                self._step_profile["qf_requested"] += requested
                self._step_profile["qf_computed"] += computed
                self._step_profile["qf_reused"] += requested - computed
            self._step_profile["commanded_steps"] += 1
        command_done = time.perf_counter()
        self.env._update_kinematic_tasks()
        kinematics_done = time.perf_counter()
        self.env.scene.step()
        scene_done = time.perf_counter()
        self.env._update_transient_checks()
        transient_done = time.perf_counter()
        self._consume_events()
        events_done = time.perf_counter()
        self._step_profile["steps"] += 1
        for name, elapsed in (("command", command_done - before),
                              ("kinematics", kinematics_done - command_done),
                              ("scene", scene_done - kinematics_done),
                              ("transient", transient_done - scene_done),
                              ("events", events_done - transient_done),
                              ("total", events_done - before)):
            self._record_step_timing(name, elapsed)

    def record_target_metrics(self):
        if self.metrics_tracker is not None:
            self.metrics_tracker.on_step()
