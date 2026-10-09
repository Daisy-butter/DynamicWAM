"""CPU clock/queue/controller tests, no DOMINO or model dependencies."""
from __future__ import annotations
from pathlib import Path
import sys
import os
import time
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dynamicwam.runtime.domino_rt.timed_control import TimedPlanQueue, LimitedHermiteController
from dynamicwam.runtime.domino_rt.continuous_loop import run_continuous_episode
from dynamicwam.runtime.domino_rt.process_worker import ProcessPolicyWorker
from dynamicwam.runtime.domino_rt.preparation import _task_state, _restore_task_state


class SleepPolicy:
    def __init__(self, config): self.latency=config['latency']
    def sample(self, packet):
        time.sleep(self.latency)
        return dict(actions=np.ones((16,14))*.03,metadata={'pid':os.getpid()})


def create_sleep_policy(config): return SleepPolicy(config)


class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now
    def sleep(self, dt): self.now+=max(0., dt)


class Session:
    def __init__(self, clock, capture_cost=0., success_at=None):
        self.clock=clock; self.time=0.; self.capture_cost=capture_cost; self.success_at=success_at
        self.q=np.zeros(14); self.commands=[]; self.captures=[]
    def timestep(self): return .004
    def physics_time(self): return self.time
    def command_state(self): return self.q.copy()
    def object_position(self): return np.array([self.time, 0., 0.])
    def capture_packet(self):
        self.captures.append(self.time); self.clock.sleep(self.capture_cost)
        return {'state':self.q.copy()}
    def step_command(self, q, v):
        self.time+=.004; self.q=q.copy(); self.commands.append((self.time, q.copy(), v.copy()))
    def stop_reason(self):
        return 'success' if self.success_at is not None and self.time>=self.success_at else None


class Worker:
    def __init__(self, clock, latency=.24, rows=16):
        self.clock=clock; self.latency=latency; self.rows=rows; self.busy=False; self.counter=0
        self.submissions=[]
    def submit(self, packet):
        assert not self.busy
        self.counter+=1; self.busy=True; self.packet=packet.copy(); self.start=self.clock()
        self.submissions.append(self.packet)
        return self.counter
    def poll(self):
        if not self.busy or self.clock()-self.start<self.latency-1e-9: return None
        self.busy=False
        return dict(request_id=self.counter, observation_sim_time=self.packet['observation_sim_time'],
            actions=np.full((self.rows,14), .05), worker_started_wall=self.start,
            worker_finished_wall=self.start+self.latency, metadata={})


class AsyncCameraSession(Session):
    asynchronous_camera=True
    def __init__(self,clock,render_delay=.12):
        super().__init__(clock);self.render_delay=render_delay
        self.render_job=None;self.latest=None;self.capture_id=0
    @property
    def camera_busy(self):return self.render_job is not None
    def capture_packet(self):raise AssertionError('timed loop must not block on camera')
    def begin_capture(self,metadata):
        self.capture_id+=1
        packet=dict(metadata,state=self.q.copy(),capture_id=self.capture_id,
                    acquisition_started_wall=self.clock(),snapshot_finished_wall=self.clock())
        if self.render_job is None:self.render_job=(self.clock(),packet)
        else:self.latest=packet
        return self.capture_id
    def poll_capture(self):
        if self.render_job is None:return None
        started,packet=self.render_job
        if self.clock()-started<self.render_delay-1e-9:return None
        output=dict(packet,renderer_started_wall=started,capture_finished_wall=started+self.render_delay,
                    camera_ready_wall=self.clock(),snapshot_oracle={'passed':True})
        self.render_job=None
        if self.latest is not None:
            self.render_job=(self.clock(),self.latest);self.latest=None
        return output


def result(rows=16, origin=0., value=.1, **metadata):
    return dict(actions=np.full((rows,14), value), observation_sim_time=origin,
                request_id=1, metadata=metadata)


class QueueTests(unittest.TestCase):
    def test_real_late_result_clipped_only_future_interval(self):
        q=TimedPlanQueue(.1)
        record=q.stage(result(), .324)
        self.assertEqual(record['clipped_prefix'],4)
        self.assertIsNone(q.publish(3))
        self.assertEqual(q.publish(4)['installed_rows'],12)
        self.assertIsNone(q.pop(4)); self.assertIsNotNone(q.pop(5))

    def test_expired_result_cannot_remove_old_tail(self):
        q=TimedPlanQueue(.1); q.stage(result(),0); q.publish(0)
        self.assertTrue(q.stage(result(rows=2,value=.2),.304)['expired'])
        self.assertAlmostEqual(q.peek(5)[0],.1)

    def test_fast_partial_updates_do_not_delete_rest(self):
        q=TimedPlanQueue(.1); q.stage(result(),0); q.publish(0)
        q.stage(result(rows=2, origin=.2, value=.3, replace_tail=False),.2); q.publish(2)
        self.assertAlmostEqual(q.peek(3)[0],.3)
        self.assertAlmostEqual(q.peek(5)[0],.1)

    def test_action_origin_differs_from_observation(self):
        q=TimedPlanQueue(.1)
        out=q.stage(result(origin=.3,action_origin_sim_time=.2),.32)
        self.assertEqual(out['clipped_prefix'],2)
        self.assertEqual(out['observation_sim_time'],.3)
        self.assertEqual(out['action_origin_sim_time'],.2)

    def test_no_non_grid_origin_or_nonfinite(self):
        q=TimedPlanQueue(.1)
        with self.assertRaises(ValueError): q.stage(result(origin=.125),.2)
        bad=result(); bad['actions'][0,0]=np.nan
        with self.assertRaises(ValueError): q.stage(bad,0)

    def test_waited_pending_can_expire_without_deleting_tail(self):
        q=TimedPlanQueue(.1); q.stage(result(),0); q.publish(0)
        q.stage(result(rows=2,origin=.1),.15)
        self.assertTrue(q.publish(4)['expired'])
        self.assertIsNotNone(q.peek(6))


class ControllerTests(unittest.TestCase):
    def test_command_continuity_and_derivative_limits(self):
        c=LimitedHermiteController(np.zeros(14),max_velocity=.4,max_acceleration=2.)
        c.begin(np.ones(14)*2)
        lastq=c.q.copy(); lastv=c.v.copy()
        for index in range(100):
            if index == 35: c.begin(-np.ones(14)*2)
            q,v=c.step(.004)
            np.testing.assert_allclose((q-lastq)/.004,v,atol=1e-10)
            self.assertLessEqual(np.max(np.abs(v[np.r_[0:6,7:13]])),.4+1e-12)
            self.assertLessEqual(np.max(np.abs((v-lastv)[np.r_[0:6,7:13]])),.008+1e-12)
            lastq=q; lastv=v
        self.assertGreater(c.limited_ticks,0)


class LoopTests(unittest.TestCase):
    def run_loop(self, *, latency=.24, mode='async', capture_cost=0., rows=16, timeout=1., success_at=None):
        clock=Clock(); session=Session(clock,capture_cost,success_at)
        worker=Worker(clock,latency,rows)
        report=run_continuous_episode(session,worker,task='mock',seed=3,mode=mode,
            physics_timeout=timeout,clock=clock,sleep=clock.sleep)
        return report, session, worker

    def test_initial_query_world_really_moves(self):
        report,session,worker=self.run_loop()
        self.assertTrue(report['realtime_valid'])
        self.assertGreaterEqual(report['requests'][0]['inference_physics_advanced'],.24-1e-9)
        self.assertGreater(report['inference_inflight_ticks'],50)
        self.assertGreater(report['object_displacement'][0],.9)
        self.assertTrue(worker.submissions[0]['_reset_policy'])
        self.assertFalse(worker.submissions[1]['_reset_policy'])

    def test_sync_holds_instead_of_freezing_and_executes_all_remaining(self):
        report,session,worker=self.run_loop(mode='sync',timeout=2.)
        self.assertEqual(len(worker.submissions),2)
        self.assertAlmostEqual(worker.submissions[1]['sim_time'],1.6)
        self.assertGreater(report['hold_slots'],0)
        self.assertAlmostEqual(report['physics_time'],2.)

    def test_async_more_frequent_queries_with_same_physics(self):
        async_r,_,async_w=self.run_loop(mode='async')
        sync_r,_,sync_w=self.run_loop(mode='sync')
        self.assertGreater(len(async_w.submissions),len(sync_w.submissions))
        self.assertEqual(async_r['physics_ticks'],sync_r['physics_ticks'])

    def test_short_result_expired_no_retroactive_execution(self):
        report,session,_=self.run_loop(latency=.24,rows=2)
        self.assertGreater(report['expired_results'],0)
        np.testing.assert_equal(session.q,np.zeros(14))

    def test_camera_cost_is_clocked_not_excluded(self):
        report,_,_=self.run_loop(capture_cost=.02)
        self.assertTrue(report['realtime_valid'])
        self.assertGreater(max(report['tick_gap_s']),.019)
        self.assertGreater(max(report['tick_lag_s']),.015)

    def test_camera_stall_is_infra_invalid_not_policy_failure(self):
        report,_,_=self.run_loop(capture_cost=.15)
        self.assertFalse(report['realtime_valid'])
        self.assertEqual(report['stop_reason'],'infrastructure_invalid')
        self.assertFalse(report['success'])

    def test_success_during_inference_hold_is_detected(self):
        report,_,_=self.run_loop(latency=.5,success_at=.052)
        self.assertEqual(report['stop_reason'],'success')
        self.assertEqual(report['installed_results'],0)
        self.assertTrue(report['terminal_inflight'])

    def test_catchup_crossing_boundary_consumes_already_published_tail(self):
        clock=Clock()
        class BlockedSession(Session):
            def step_command(self,q,v):
                super().step_command(q,v)
                if abs(self.time-.292)<1e-8:
                    self.clock.sleep(.03)  # Owner-side work crosses the .3 boundary.
        session=BlockedSession(clock); worker=Worker(clock,.04)
        report=run_continuous_episode(session,worker,task='mock',seed=3,
            physics_timeout=.6,clock=clock,sleep=clock.sleep)
        interval=next(row for row in report['control_intervals'] if row['slot']==3)
        self.assertTrue(interval['catch_up'])
        self.assertFalse(interval['hold'])
        self.assertIsNotNone(interval['request_id'])
        self.assertTrue(report['realtime_valid'])

    def test_float32_sapien_timestep_does_not_accumulate_offgrid_origins(self):
        clock=Clock()
        class Float32Scene(Session):
            def timestep(self): return float(np.float32(.004))
            def step_command(self,q,v):
                super().step_command(q,v)
                self.time+=float(np.float32(.004))-.004
        session=Float32Scene(clock); worker=Worker(clock,.03)
        report=run_continuous_episode(session,worker,task='mock',seed=3,
            physics_timeout=4.,clock=clock,sleep=clock.sleep)
        self.assertTrue(report['realtime_valid'])
        self.assertEqual(report['physics_time'],4.)
        self.assertGreater(report['inference_calls'],10)

    def test_independent_camera_latency_does_not_block_physics(self):
        clock=Clock();session=AsyncCameraSession(clock,.12);worker=Worker(clock,.03)
        report=run_continuous_episode(session,worker,task='mock',seed=3,
            physics_timeout=1.,clock=clock,sleep=clock.sleep)
        self.assertTrue(report['realtime_valid'])
        self.assertGreater(report['camera_inflight_ticks'],100)
        self.assertLess(max(report['tick_gap_s']),.005)
        self.assertEqual(worker.submissions[0]['observation_sim_time'],0.)
        self.assertGreaterEqual(report['requests'][0]['observation_to_return_wall_s'],.15-1e-8)
        self.assertLess(report['requests'][0]['request_wall_s'],.04)

    def test_camera_old_branch_dropped_not_retimestamped(self):
        clock=Clock();session=AsyncCameraSession(clock,.12)
        class PlanWorker(Worker):
            def poll(self):
                value=super().poll()
                if value is not None:value['metadata']['plan_id']=str(value['request_id'])
                return value
        worker=PlanWorker(clock,.12)
        report=run_continuous_episode(session,worker,task='mock',seed=3,
            physics_timeout=1.2,clock=clock,sleep=clock.sleep)
        self.assertTrue(report['realtime_valid'])
        self.assertGreater(report['stale_camera_frames'],0)
        for packet in worker.submissions:
            self.assertAlmostEqual(packet['sim_time'],packet['command_index']*.1)


class ProcessTests(unittest.TestCase):
    def test_parent_keeps_ticking_during_real_process_inference(self):
        worker=ProcessPolicyWorker(factory='test_continuous_domino:create_sleep_policy',config={'latency':.12})
        try:
            worker.submit({'observation_sim_time':0.,'state':np.zeros(14)})
            ticks=0; result=None; deadline=time.monotonic()+5
            while result is None:
                ticks+=1; time.sleep(.004); result=worker.poll()
                self.assertLess(time.monotonic(),deadline)
            self.assertGreaterEqual(ticks,10)
            self.assertNotEqual(result['metadata']['pid'],os.getpid())
            self.assertGreaterEqual(result['worker_finished_wall']-result['worker_started_wall'],.115)
        finally: worker.close()


class PreparationTests(unittest.TestCase):
    def test_missing_expert_arm_tag_fails_closed(self):
        for task in ('put_object_cabinet','place_object_scale'):
            with self.assertRaises(RuntimeError): _restore_task_state(object(),{},task)

    def test_task_state_copies_only_portable_task_fields(self):
        class Fake:
            origin_z=.78
            position=np.array([1.,2.,3.])
            scene_actor=object()
            base_internal=900
        with tempfile.NamedTemporaryFile(mode='w',suffix='.py',dir=Path(__file__).parent,delete=False) as handle:
            handle.write('class Task:\n def run(self):\n  self.origin_z=0\n  self.position=None\n  self.scene_actor=None\n')
            path=Path(handle.name)
        try:
            state=_task_state(Fake(),path)
            self.assertEqual(set(state),{'origin_z','position'})
            state['position']['value'][0]=9
            self.assertEqual(Fake.position[0],1)
        finally: path.unlink()


if __name__=='__main__': unittest.main()
