"""CPU-only exact-grid/control/state-parity tests; no simulator or child launch."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import full_physics_replay as replay

HAS_NUMPY = importlib.util.find_spec('numpy') is not None


def failure_trace():
    dt=.004000000189989805
    return dict(task='place_object_stand',episode_index=7,complete=False,
        stop_reason='infrastructure_error',error='TimingError: wall/physics backlog .251 outside',
        protocol='continuous_full_sync16_v4_grid180_linear_bootstrap_v2',
        physics_step_seconds=dt,action_interval_s=45*dt,physics_steps=96,
        physical_duration=96*dt,started_targets=3,
        owner_stage_timing={'totals':{'transient':{'calls':95}}},
        events=[dict(event='execute',row=r,physical_tick=45*r,actual_start=45*r*dt,
                     target=[0.]*14,row_kind='bootstrap_hold' if r<2 else 'generated',
                     controller='endpoint_linear_arm_native_gripper') for r in range(3)])


class TraceTests(unittest.TestCase):
    def test_exact_partial_failure_tick_is_preserved(self):
        result=replay.validate_trace(failure_trace(),task='place_object_stand',episode_index=7)
        self.assertEqual(result['steps'],96)
        self.assertEqual(len(result['targets']),3)
        self.assertEqual(result['action_dt'],45*result['dt'])

    def test_corrupt_or_different_failure_never_replays(self):
        for mutate in (lambda r:r.update(complete=True),lambda r:r.update(physics_steps=135),
                       lambda r:r['events'][1].update(row=2),
                       lambda r:r['events'][2].update(actual_start=.36),
                       lambda r:r['events'][2]['target'].__setitem__(0,float('nan')),
                       lambda r:r['owner_stage_timing']['totals']['transient'].update(calls=96)):
            r=failure_trace();mutate(r)
            with self.subTest(r=r['physics_steps']):
                with self.assertRaises(ValueError):
                    replay.validate_trace(r,task='place_object_stand',episode_index=7)

    def test_noncanonical_finite_limits_rejected_before_artifact_reads(self):
        base=dict(run_kind=replay.MODE,cpu_workers=[0,2,4],task='place_object_stand',episode_index=7,
                  seed=100000,episodes=20,timeout_seconds=100,parity_atol=1e-6)
        for change in ({'cpu_workers':[2]},{'timeout_seconds':float('inf')},{'parity_atol':1e-4},
                       {'episode_index':0},{'timeout_seconds':float('nan')}):
            with self.assertRaises(ValueError):replay.validate_config(dict(base,**change))


@unittest.skipUnless(HAS_NUMPY,'numpy unavailable on local control host')
class NumericalTests(unittest.TestCase):
    def test_executor_matches_rows_and_preserves_partial_last_tick(self):
        import numpy as np
        dt=.004000000189989805;scene=SimpleNamespace(ticks=0);arms=[];grippers=[];events=[]
        scene.step=lambda:setattr(scene,'ticks',scene.ticks+1)
        robot=SimpleNamespace(left_entity=None,right_entity=None,_entity_qf=lambda x:None,
            set_arm_joints=lambda p,v,side:arms.append((scene.ticks,side,p.copy(),v.copy())),
            set_gripper=lambda value,side:grippers.append((scene.ticks,side,value)))
        env=SimpleNamespace(scene=scene,robot=robot,take_action_cnt=0,
                            _update_kinematic_tasks=lambda:None,_update_transient_checks=lambda:None)
        session=SimpleNamespace(env=env,physics_time=lambda:scene.ticks*dt,
            _consume_events=lambda:events.append(scene.ticks),stop_reason=lambda:None)
        trace=dict(dt=dt,action_dt=45*dt,steps=96,targets=[[0.]*14,[0.]*14,[1.]*14])
        names,states,times,rows=replay.replay_controls(session,trace,
            lambda:(['tick'],np.array([scene.ticks],np.float64)),deadline=float('inf'))
        self.assertEqual(env.take_action_cnt,3);self.assertEqual([r['steps'] for r in rows],[45,45,6])
        self.assertEqual(events,list(range(1,96)));self.assertEqual(states.shape,(97,1))
        self.assertEqual(times[-1],96*dt)
        np.testing.assert_allclose(arms[180][2],np.full(6,1/45),atol=0,rtol=0)
        np.testing.assert_allclose(arms[-2][2],np.full(6,6/45),atol=0,rtol=0)
        self.assertEqual(grippers[180][2],1.)

    def test_actual_state_excludes_drive_target_proxy(self):
        import numpy as np
        class Body:
            linear_velocity=np.array([.1,.2,.3]);angular_velocity=np.array([.4,.5,.6])
        entity=SimpleNamespace(get_qpos=lambda:np.array([1.,2.]),get_qvel=lambda:np.array([3.,4.]))
        actor=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=[5.,6.,7.],q=[1.,0.,0.,0.]),
                              get_components=lambda:[Body()])
        env=SimpleNamespace(robot=SimpleNamespace(left_entity=entity,right_entity=entity),
                            object=SimpleNamespace(actor=actor),displaystand=SimpleNamespace(actor=actor))
        names,values=replay.snapshot_state(env,Body)
        self.assertEqual(values[names.index('left_qpos[0]')],1.)
        self.assertEqual(values[names.index('displaystand_linear_velocity[1]')],.2)
        self.assertTrue(all('target' not in n for n in names))

    def test_parity_checks_every_tick_not_only_final_or_rows(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.npz';b=Path(tmp)/'b.npz'
            data=dict(names=np.array(['qpos[0]','actor_p[0]']),states=np.zeros((97,2)),
                      physics_times=np.arange(97)*.004000000189989805)
            np.savez_compressed(a,**data);np.savez_compressed(b,**data)
            self.assertTrue(replay.compare_state_files(a,b,atol=1e-6)['passed'])
            changed=copy.deepcopy(data);changed['states'][17,1]=1e-4
            np.savez_compressed(b,**changed)
            result=replay.compare_state_files(a,b,atol=1e-6)
            self.assertFalse(result['passed']);self.assertEqual(result['worst_tick'],17)
            changed=copy.deepcopy(data);changed['physics_times'][17]+=.004
            np.savez_compressed(b,**changed)
            with self.assertRaises(ValueError):replay.compare_state_files(a,b,atol=1e-6)

    def test_timeout_does_not_advance_scene(self):
        import numpy as np
        scene=SimpleNamespace(step=lambda:self.fail('expired replay stepped physics'))
        env=SimpleNamespace(scene=scene,robot=object())
        session=SimpleNamespace(env=env,physics_time=lambda:0.)
        with self.assertRaises(TimeoutError):
            replay.replay_controls(session,dict(dt=.004,action_dt=.18,steps=1,targets=[[0.]*14]),
                                   lambda:(['x'],np.array([0.])),deadline=0.)


if __name__=='__main__':unittest.main()
