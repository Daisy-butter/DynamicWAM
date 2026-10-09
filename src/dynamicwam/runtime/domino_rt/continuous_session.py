"""Single physics owner; per-tick task semantics, clocked camera packets."""
from __future__ import annotations
from typing import Any
import time
import numpy as np
from dynamicwam.runtime.ciwam.adapters.domino.composite import (
    composite_robotwin_frame, extract_state, head_camera_frame)
from dynamicwam.runtime.ciwam.flow import HeadFlowBuffer


class ContinuousSession:
    def __init__(self, env: Any, prepared: dict, *, flow_config: dict,
                 observation_config: dict, control_period: float = .1) -> None:
        self.env = env; self.prepared = prepared
        self.flow = HeadFlowBuffer(flow_config)
        self.observation_config = observation_config
        self.period = control_period
        self._previous_head = None; self._previous_stamp = None
        self._stop_reason = None; self._lifted = False
        self.metrics_tracker = None
        self._next_metric = control_period
        self.command = self.command_state()
        self.renderer = None
        self.snapshot_bridge = None
        self.trajectory_recording = False

    def attach_renderer(self, renderer) -> None:
        import json
        from .scene_snapshot import SceneSnapshotBridge
        self.snapshot_bridge = SceneSnapshotBridge(self.env)
        if json.dumps(self.snapshot_bridge.identity,sort_keys=True) != json.dumps(renderer.identity,sort_keys=True):
            raise RuntimeError('renderer replica scene layout/assets differ from physics owner')
        self.renderer = renderer

    @property
    def asynchronous_camera(self) -> bool:
        return self.renderer is not None

    @property
    def camera_busy(self) -> bool:
        return self.renderer is not None and self.renderer.busy

    def begin_capture(self, metadata: dict) -> int:
        if self.renderer is None:
            raise RuntimeError('asynchronous capture requires replica renderer')
        started = time.perf_counter()
        snapshot = self.snapshot_bridge.capture()
        source_stamp=float(snapshot['source_sim_time_seconds'])
        grid_stamp=float(metadata['observation_sim_time'])
        if abs(source_stamp-grid_stamp)>max(1e-7,abs(grid_stamp)*1e-7):
            raise RuntimeError('pose snapshot and camera metadata refer to different physical instants')
        drive_state=self.command_state()
        packet = dict(metadata, instruction=self.prepared['instruction'],
            state=drive_state.astype(np.float32),
            state_semantics='commanded_joint_drive_targets',
            acquisition_started_wall=started,
            snapshot_finished_wall=time.perf_counter(),
            source_physics_time_seconds=source_stamp,
            capture_clock_roundoff_s=source_stamp-grid_stamp,
            snapshot_sequence=snapshot['sequence'])
        if self.trajectory_recording:
            packet['_trajectory_measured_joint_state']=self.measured_joint_state()
            packet['_trajectory_drive_targets']=drive_state
        return self.renderer.submit(snapshot, packet)

    def poll_capture(self) -> dict | None:
        output = self.renderer.poll()
        if output is None:
            return None
        packet = dict(output['packet'])
        stamp = float(packet['observation_sim_time'])
        images = output['images']
        head = np.ascontiguousarray(images['head_camera']).copy()
        previous = head.copy() if self._previous_head is None else self._previous_head.copy()
        dt = 0. if self._previous_stamp is None else stamp-self._previous_stamp
        if dt < 0:
            raise RuntimeError('renderer delivered nonmonotonic capture timestamps')
        self.flow.push(head,simulator_time_seconds=stamp)
        flow_started=time.perf_counter()
        motion=self.flow.observation()
        observation={'observation':{key:{'rgb':value} for key,value in images.items()}}
        packet.update(frame=np.array(composite_robotwin_frame(observation,self.observation_config),copy=True),
            cameras=images,head_rgb=head,previous_head_rgb=previous,history_dt=dt,
            history_valid=self._previous_stamp is not None and dt>0,
            flow_frames=np.array(motion.flow_rgb,copy=True),
            motion_features=np.array(motion.motion_features,copy=True),
            motion_interval_valid_mask=np.array(motion.interval_valid_mask,copy=True),
            motion_acceleration_valid_mask=np.array(motion.acceleration_valid_mask,copy=True),
            capture_id=output['capture_id'],capture_finished_wall=output['renderer_finished_wall'],
            camera_ready_wall=time.perf_counter(),renderer_started_wall=output['renderer_started_wall'],
            snapshot_oracle=output['snapshot_oracle'])
        packet['_capture_timing']=dict(snapshot_s=packet['snapshot_finished_wall']-packet['acquisition_started_wall'],
            renderer_s=output['renderer_finished_wall']-output['renderer_started_wall'],
            flow_and_packaging_s=time.perf_counter()-flow_started,
            camera_pipeline='independent_scene_snapshot_renderer')
        self._previous_head=head;self._previous_stamp=stamp
        return packet

    def timestep(self) -> float:
        return float(self.env.scene.get_timestep())

    def physics_time(self) -> float:
        return float(self.env._scene_step_clock.snapshot().time_seconds)

    def command_state(self) -> np.ndarray:
        # DOMINO's jointState accessors return drive targets, NOT measured q.
        return np.asarray([*self.env.robot.get_left_arm_jointState(),
                           *self.env.robot.get_right_arm_jointState()], dtype=np.float64)

    def measured_joint_state(self) -> dict:
        """Read actual articulation state at the source snapshot, never drive targets.

        Unsupported fields remain absent with a reason. They must not be filled
        with commanded targets and later presented as measured proprioception.
        """
        result={}
        for side in ('left','right'):
            entity=getattr(self.env.robot,side+'_entity',None);item={};errors=[]
            for kind in ('qpos','qvel'):
                getter=getattr(entity,'get_'+kind,None)
                if not callable(getter):errors.append(kind+': unavailable');continue
                try:value=np.asarray(getter(),dtype=np.float64).copy()
                except (AttributeError,NotImplementedError,RuntimeError) as exc:
                    errors.append(kind+': '+type(exc).__name__);continue
                if value.ndim!=1 or not np.isfinite(value).all():raise RuntimeError('invalid measured articulation state')
                item[kind]=value
            names=getattr(entity,'get_active_joints',None)
            if callable(names):
                try:item['joint_names']=[str(getattr(joint,'name','')) for joint in names()]
                except (AttributeError,NotImplementedError,RuntimeError) as exc:
                    errors.append('joint_names: '+type(exc).__name__)
            if errors:item['error']='; '.join(errors)
            result[side]=item
        return result

    def object_position(self) -> np.ndarray:
        actor = self.env._eval_target_actor
        return np.asarray(actor.actor.get_pose().p, dtype=np.float64).copy()

    def capture_packet(self) -> dict:
        if self.renderer is not None:
            # Startup-only blocking wrapper. The timed loop uses begin/poll.
            stamp=self.physics_time()
            self.begin_capture(dict(sim_time=stamp,observation_sim_time=stamp))
            deadline=time.monotonic()+60.
            while True:
                packet=self.poll_capture()
                if packet is not None:return packet
                if time.monotonic()>deadline:raise TimeoutError('renderer warmup capture timed out')
                time.sleep(.001)
        started = time.perf_counter()
        stamp = self.physics_time()
        # Instrument existing synchronous calls without changing render quality,
        # adding another picture, or introducing concurrent scene ownership.
        substeps = {}
        originals = []
        def measured(function, name):
            def run(*args, **kwargs):
                before = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    substeps[name] = substeps.get(name, 0.) + time.perf_counter()-before
            return run
        try:
            for owner, name, label in ((self.env, '_update_render', 'update_render_s'),
                (self.env.cameras, 'update_picture', 'submit_picture_s'),
                (self.env.cameras, 'get_rgb', 'rgb_readback_s')):
                function = getattr(owner, name)
                originals.append((owner, name, function))
                setattr(owner, name, measured(function, label))
            observation = self.env.get_obs()
        finally:
            for owner, name, function in originals:
                setattr(owner, name, function)
        observed = time.perf_counter()
        head = np.ascontiguousarray(head_camera_frame(observation)).copy()
        previous = head.copy() if self._previous_head is None else self._previous_head.copy()
        dt = 0. if self._previous_stamp is None else stamp-self._previous_stamp
        self.flow.push(head, simulator_time_seconds=stamp)
        motion = self.flow.observation()
        flowed = time.perf_counter()
        cameras = {key: np.array(observation['observation'][key]['rgb'], copy=True)
                   for key in ('head_camera', 'left_camera', 'right_camera')}
        packet = dict(instruction=self.prepared['instruction'],
            frame=np.array(composite_robotwin_frame(observation, self.observation_config), copy=True),
            state=np.array(extract_state(observation), copy=True), cameras=cameras,
            head_rgb=head, previous_head_rgb=previous, history_dt=dt,
            history_valid=self._previous_stamp is not None and dt > 0,
            state_semantics='commanded_joint_drive_targets',
            flow_frames=np.array(motion.flow_rgb, copy=True),
            motion_features=np.array(motion.motion_features, copy=True),
            motion_interval_valid_mask=np.array(motion.interval_valid_mask, copy=True),
            motion_acceleration_valid_mask=np.array(motion.acceleration_valid_mask, copy=True))
        self._previous_head=head; self._previous_stamp=stamp
        packet['_capture_timing'] = dict(observation_s=observed-started,
            flow_s=flowed-observed, packaging_s=time.perf_counter()-flowed, **substeps)
        return packet

    def warm_camera(self) -> None:
        """No policy query, physics, flow history, or episode-clock advance."""
        if self.renderer is None:
            self.env.get_obs()

    def step_command(self, q: np.ndarray, velocity: np.ndarray) -> None:
        robot = self.env.robot
        robot._entity_qf(robot.left_entity); robot._entity_qf(robot.right_entity)
        robot.set_arm_joints(q[:6], velocity[:6], 'left')
        robot.set_arm_joints(q[7:13], velocity[7:13], 'right')
        robot.set_gripper(float(q[6]), 'left'); robot.set_gripper(float(q[13]), 'right')
        self.command = np.array(q, copy=True)
        self.env._update_kinematic_tasks()
        self.env.scene.step()
        # Native task transient latches are required even during inference holds.
        self.env._update_transient_checks()
        self.env.take_action_cnt = int(self.physics_time()/self.period + 1e-8)
        self._events()
        if self.metrics_tracker is not None and self.physics_time()+1e-9 >= self._next_metric:
            self.metrics_tracker.on_step()
            self._next_metric += self.period

    def _events(self) -> None:
        if self._stop_reason is not None:
            return
        p = self.prepared
        if p['check_z_threshold'] is not None and not self._lifted:
            self._lifted = float(p['lift_actor'].get_pose().p[2])-p['start_z'] > p['check_z_threshold']
        if self.env.check_success():
            self.env.eval_success = True
            self._stop_reason = 'object_not_lifted' if p['check_z_threshold'] is not None and not self._lifted else 'success'
            return
        if self.env.check_gripper_contact_dynamic_object() and p['stop_on_contact']:
            self.env.stop_dynamic_object_motion()
        if not getattr(self.env, '_dynamic_object_stopped', False) and not self.env.check_dynamic_object_boundary():
            self.env.eval_fail = True; self._stop_reason = 'out_of_bounds'

    def stop_reason(self) -> str | None:
        return self._stop_reason
