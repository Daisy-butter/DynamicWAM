import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest


root=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('camera_clock',root/'scripts/replay_camera_clock.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=root/'src/dynamicwam/dualrate'
if not source.is_dir():source=root.parent/'cohort_repair_20261029/src/dynamicwam/dualrate'
physical=source/'physical_clock.py'
if not physical.exists():physical=root.parent/'dualrate_continuous_repair_20261028/src/dynamicwam/dualrate/physical_clock.py'
namespace=dict(math=math,time=time,TimingError=ValueError,DeadlineMiss=ValueError)
for file,names in ((physical,{'WallPhysicsBridge'}),(source/'deployment.py',{
        'ContinuousWallPhysicsBridge','exact_grid_index','next_grid_capture_tick','capture_timing','warm_camera'})):
    nodes=[node for node in ast.parse(file.read_text()).body
           if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),namespace)


class Fixture:
    def __init__(self,delay=.1):
        self.wall=100.;self.ticks=0;self.dt=.004000000189989805;self.delay=delay
        self.session=SimpleNamespace(physics_time=lambda:self.ticks*self.dt,
            timestep=lambda:self.dt,env=object())
        self.closed=False;self.active=None;self.begun=[]
        self.camera=SimpleNamespace(begin=self.begin,finish=self.finish,ready=self.ready,
                                    close=lambda:setattr(self,'closed',True))
        class History:
            def __init__(self):self.frames=[]
            def append(self,observation,**kwargs):self.frames.append(kwargs)
        class Video:
            def __init__(self):self.frames=0
            def append(self,*args,**kwargs):self.frames+=1
        deps=(namespace['ContinuousWallPhysicsBridge'],namespace['warm_camera'],
              namespace['next_grid_capture_tick'],namespace['exact_grid_index'],History,Video)
        self.adapter=module.RealtimeCameraReplay(self.session,self.camera,action_dt=45*self.dt,
            frame_period=45*self.dt,dependencies=deps,wall_clock=lambda:self.wall,sleep=self.sleep)
    def sleep(self,seconds):self.wall+=seconds
    def begin(self,env,now):
        assert self.active is None
        captured=SimpleNamespace(physics_time_s=now,capture_wall_perf_s=self.wall,
            render_submitted_wall_perf_s=self.wall+self.delay-.01,
            readback_started_wall_perf_s=self.wall+self.delay-.01,
            readback_finished_wall_perf_s=self.wall+self.delay,finished_wall_perf_s=None,
            camera_readback_seconds={},observation={'observation':{'head_camera':{'rgb':object()}}})
        self.active=captured;self.begun.append(now);return captured
    def ready(self,pending):return self.wall>=pending.readback_finished_wall_perf_s
    def finish(self,pending=None):
        pending=self.active if pending is None else pending
        self.wall=max(self.wall,pending.readback_finished_wall_perf_s)
        pending.finished_wall_perf_s=self.wall;self.active=None;return pending
    def run(self,ticks=180):
        for tick in range(ticks):
            self.adapter.before_tick(tick)
            self.wall+=.001;self.ticks+=1
            self.wall+=.0002  # Actual diagnostic state sampling is not deducted.
            self.adapter.after_tick(tick+1)
        self.adapter.finish();self.adapter.close()


class CameraClockTests(unittest.TestCase):
    def test_real_tick_pacing_release_and_cleanup(self):
        fixture=Fixture();fixture.run();report=fixture.adapter.report()
        self.assertEqual(fixture.ticks,180)
        self.assertTrue(report['completed'] and report['camera_closed'])
        self.assertGreaterEqual(report['active_wall_seconds'],report['physical_seconds'])
        self.assertTrue(report['captures'])
        self.assertTrue(all(e['available']+1e-9>=e['availability_lower_bound'] for e in report['captures']))
        self.assertEqual(report['time_deducted_seconds'],0.)
        self.assertLess(report['max_backlog_s'],.255)

    def test_native_stall_rejected_without_clock_rebase_or_extra_steps(self):
        fixture=Fixture();fixture.adapter.before_tick(0);origin=fixture.adapter.bridge.wall_origin
        fixture.wall+=.27;fixture.ticks=1
        with self.assertRaisesRegex(ValueError,'backlog'):fixture.adapter.after_tick(1)
        self.assertEqual(fixture.ticks,1)
        self.assertEqual(origin,fixture.adapter.bridge.wall_origin)
        self.assertGreater(fixture.adapter.report()['max_backlog_s'],.255)
        fixture.adapter.close();self.assertTrue(fixture.closed)

    def test_slow_render_defers_captures_but_never_stops_physics(self):
        fixture=Fixture(delay=.29);fixture.run(ticks=270)
        active=fixture.begun[3:]
        self.assertTrue(active)
        self.assertTrue(all(abs(t/(45*fixture.dt)-round(t/(45*fixture.dt)))<1e-8 for t in active))
        self.assertTrue(all(b-a>=2*45*fixture.dt-1e-8 for a,b in zip(active,active[1:])))
        self.assertEqual(fixture.ticks,270)


if __name__=='__main__':unittest.main()
