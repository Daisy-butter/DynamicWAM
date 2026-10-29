"""Diagnostic-only production clock/camera replay; never edits policy or score."""
import time


class RealtimeCameraReplay:
    def __init__(self,session,camera,*,action_dt,frame_period,max_backlog=.25,
                 backlog_scheduling_tolerance=.005,warmups=3,dependencies=None,
                 wall_clock=None,sleep=None):
        if dependencies is None:
            from dynamicwam.dualrate.deployment import (ContinuousWallPhysicsBridge,
                warm_camera,next_grid_capture_tick,exact_grid_index)
            from dynamicwam.dualrate.online import ObservationHistory
            from video_artifacts import BufferedHeadVideo
            dependencies=(ContinuousWallPhysicsBridge,warm_camera,next_grid_capture_tick,
                          exact_grid_index,ObservationHistory,BufferedHeadVideo)
        Bridge,warm,self.next_capture,self.index,History,Video=dependencies
        self.wall=wall_clock or time.perf_counter;self.sleep=sleep or time.sleep
        self.session=session;self.camera=camera
        self.dt=float(session.timestep());self.initial=float(session.physics_time())
        self.action_dt=float(action_dt);self.capture_ticks=round(frame_period/self.dt)
        if (self.initial!=0. or abs(self.dt-.004)>1e-8 or self.capture_ticks<1
                or abs(frame_period-self.capture_ticks*self.dt)>1e-10
                or self.capture_ticks%45 or abs(action_dt-45*self.dt)>1e-10
                or type(max_backlog) not in (int,float) or max_backlog not in (.25,.5,1.)
                or backlog_scheduling_tolerance!=.005):
            raise ValueError('diagnostic camera requires exact grid and explicit .25/.5/1. backlog with .005 tolerance')
        self.bridge=Bridge(self.dt,wall_clock=self.wall,
            backlog_scheduling_tolerance=backlog_scheduling_tolerance)
        self.max_backlog=float(max_backlog);self.max_lag=0.;self.min_lag=0.
        self.history=History();self.video=Video();self.pending=self.completed=None
        self.captures=[];self.skipped=0;self.next_capture_tick=self.capture_ticks
        self.steps=0;self.finished=False;self.closed=False;self.end_wall=self.end_physical=None;self.submissions=0
        captured=warm(camera,session,count=warmups,emit=lambda event:None)
        self.history.append(captured.observation,exposure=captured.physics_time_s,available=self.initial)
        self.video.append(captured.observation['observation']['head_camera']['rgb'],
            physics_time_s=self.initial,capture_wall_perf_s=captured.capture_wall_perf_s)

    def checked(self):
        now=float(self.session.physics_time())
        wall=self.wall();lag=self.bridge.expected_physics(wall)-now
        self.max_lag=max(self.max_lag,lag);self.min_lag=min(self.min_lag,lag)
        self.bridge.check_backlog(now,self.max_backlog,wall_time=wall)
        return now

    def before_tick(self,tick):
        if not self.bridge.active:self.bridge.activate(self.initial,self.wall())
        while True:
            now=self.checked()
            if self.index(now,self.initial,self.dt,tolerance=1e-8)!=tick:
                raise ValueError('camera replay and actual physical tick differ')
            if self.pending is not None and self.camera.ready(self.pending):
                self.completed=self.camera.finish(self.pending);self.pending=None
                render=self.completed.render_submitted_wall_perf_s-self.completed.capture_wall_perf_s
                source_tick=self.index(self.completed.physics_time_s,self.initial,self.dt,tolerance=1e-8)
                self.next_capture_tick=max(self.next_capture_tick,
                    self.next_capture(source_tick,self.capture_ticks,self.dt,render))
                now=self.checked()
            if self.completed is not None and self.bridge.ready(self.completed.finished_wall_perf_s,now):
                release=self.wall();capture=self.completed;self.completed=None
                self.index(capture.physics_time_s,self.initial,self.action_dt)
                self.history.append(capture.observation,exposure=capture.physics_time_s,available=now)
                self.video.append(capture.observation['observation']['head_camera']['rgb'],
                    physics_time_s=capture.physics_time_s,capture_wall_perf_s=capture.capture_wall_perf_s)
                self.captures.append(dict(exposure=capture.physics_time_s,available=now,
                    finished_wall=capture.finished_wall_perf_s,release_wall=release,
                    processed_wall=self.wall(),availability_lower_bound=self.bridge.availability_time(capture.finished_wall_perf_s)))
            now=self.checked()
            if tick>=self.next_capture_tick and tick%self.capture_ticks==0:
                self.skipped+=max(0,(tick-self.next_capture_tick)//self.capture_ticks)
                if self.pending is None and self.completed is None:
                    self.pending=self.camera.begin(self.session.env,now)
                    self.submissions+=1
                    after=self.checked()
                    if after!=now or self.pending.physics_time_s!=now:
                        raise ValueError('camera changed physical time/exposure')
                else:self.skipped+=1
                self.next_capture_tick=tick+self.capture_ticks
            now=self.checked()
            wait=(now-self.initial)+self.dt-(self.wall()-self.bridge.wall_origin)
            if wait>0:
                self.sleep(min(wait,.001));continue
            self.checked()
            return

    def after_tick(self,tick):
        now=self.checked()
        if abs(now-(self.initial+tick*self.dt))>1e-8:
            raise ValueError('camera replay skipped a real physics step')
        self.steps=tick

    def finish(self):
        self.end_physical=self.checked();self.end_wall=self.wall();self.finished=True

    def report(self):
        wall=self.end_wall if self.end_wall is not None else self.wall()
        physical=float(self.session.physics_time()) if self.end_physical is None else self.end_physical
        return dict(mode='diagnostic_realtime_camera',diagnostic_only=True,no_policy=True,no_score=True,
            completed=self.finished,simulation_rate=1.,physics_steps=self.steps,
            active_wall_seconds=None if not self.bridge.active else wall-self.bridge.wall_origin,
            physical_seconds=physical-self.initial,max_backlog_s=self.max_lag,min_backlog_s=self.min_lag,
            raw_end_lag_s=None if not self.bridge.active else self.bridge.expected_physics(wall)-physical,
            backlog_limit_s=self.max_backlog+self.bridge.backlog_scheduling_tolerance,
            configured_max_backlog_s=self.max_backlog,
            backlog_scheduling_tolerance_s=self.bridge.backlog_scheduling_tolerance,
            backlog_protocol='diagnostic_explicit_backlog_v2',original_max_backlog_s=.25,
            backlog_limit_changed_from_original=self.max_backlog!=.25,
            captures=self.captures,camera_submissions=self.submissions,
            skipped_capture_requests=self.skipped,
            video_frames=self.video.frames,all_native_steps_and_state_snapshot_costs_included=True,
            time_deducted_seconds=0.,camera_closed=self.closed,
            limitation='Extra per-tick parity state sampling consumes real budget; failure is not proof production must fail.')

    def close(self):
        if self.closed:return
        if self.end_wall is None:
            self.end_wall=self.wall();self.end_physical=float(self.session.physics_time())
        try:
            if self.pending is not None:
                self.camera.finish(self.pending);self.pending=None
        finally:
            self.camera.close();self.closed=True
