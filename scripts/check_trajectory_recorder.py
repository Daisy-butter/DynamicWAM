#!/usr/bin/env python3
"""CPU-only tests of exact applied-action recording and bounded persistence."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
tempfile.tempdir=str(Path.cwd())
from dynamicwam.runtime.domino_rt.trajectory_recorder import TrajectoryRecorder,TrajectoryRecordingError,ExecutedCommandRing
from dynamicwam.runtime.domino_rt.continuous_loop import run_continuous_episode
from test_continuous_domino import AsyncCameraSession,Clock,Worker


def fake_encoder(frame):
    if frame.dtype!=np.uint8:raise AssertionError('raw camera bytes expected')
    return np.full((3,160,160,3),int(frame[0,0,0]),dtype=np.uint8),np.ones((3,160,160),dtype=np.bool_)


class RecordedSession(AsyncCameraSession):
    def begin_capture(self,metadata):
        capture_id=super().begin_capture(metadata)
        packet=self.latest if self.latest is not None else self.render_job[1]
        packet.update(frame=np.full((360,320,3),capture_id,dtype=np.uint8),
            source_physics_time_seconds=metadata['observation_sim_time'],snapshot_sequence=capture_id,
            instruction='move the test object',flow_frames=np.full((4,64,64,3),capture_id,dtype=np.uint8),
            motion_features=np.full((4,8),capture_id*.25,dtype=np.float32),
            motion_interval_valid_mask=np.array([False,False,True,True]),
            motion_acceleration_valid_mask=np.array([False,False,False,True]),
            previous_head_rgb=np.full((240,320,3),max(0,capture_id-1),dtype=np.uint8),
            history_dt=.2 if capture_id>0 else 0.,history_valid=capture_id>0)
        if getattr(self,'trajectory_recording',False):
            packet['_trajectory_drive_targets']=self.q.copy()
            packet['_trajectory_measured_joint_state']={'left':dict(qpos=self.q[:6]+1.,qvel=np.full(6,.25),
                joint_names=[f'left_{i}' for i in range(6)]),'right':dict(error='unavailable')}
        return capture_id
    def poll_capture(self):
        result=super().poll_capture()
        if result is not None:result['snapshot_oracle'].update(physics_steps=0,
            source_sim_time_seconds=result['source_physics_time_seconds'])
        return result


class RecorderTests(unittest.TestCase):
    def run_episode(self,*,record=True,latency=.24,rows=16):
        clock=Clock();session=RecordedSession(clock);worker=Worker(clock,latency,rows)
        recorder=TrajectoryRecorder(max_seconds=.8) if record else None
        events=[]
        def sink(event):
            if record:self.assertGreaterEqual(session.physics_time(),.8-1e-8)
            events.append(event)
        with patch('dynamicwam.runtime.domino_rt.trajectory_recorder._load_view_encoder',return_value=fake_encoder), \
                patch('builtins.open',side_effect=AssertionError('disk IO during physics')), \
                patch.object(Path,'open',side_effect=AssertionError('disk IO during physics')):
            report=run_continuous_episode(session,worker,task='mock',seed=3,physics_timeout=.8,
                clock=clock,sleep=clock.sleep,sink=sink,trajectory_recorder=recorder)
        return report,session,worker,recorder,events

    def test_every_logged_q_and_velocity_equals_actual_step_arguments(self):
        report,session,worker,recorder,events=self.run_episode()
        actual=np.array([q for _,q,_ in session.commands]);velocity=np.array([v for _,_,v in session.commands])
        np.testing.assert_array_equal(recorder.commands['q_command'][:recorder.tick_count],actual)
        np.testing.assert_array_equal(recorder.commands['velocity_command'][:recorder.tick_count],velocity)
        self.assertEqual(recorder.tick_count,200);self.assertEqual(report['physics_ticks'],200)
        np.testing.assert_allclose(recorder.commands['end_sim_time'][:200],np.arange(1,201)*.004,atol=0,rtol=0)
        self.assertTrue(recorder.commands['hold'][:recorder.tick_count].any())
        self.assertTrue((~recorder.commands['hold'][:recorder.tick_count]).any())
        self.assertEqual(len(events),len(recorder.events));self.assertEqual(recorder.state,'complete')
        self.assertFalse(session.trajectory_recording)
        self.assertTrue(all(not any(k.startswith('_trajectory') for k in p) for p in worker.submissions))

    def test_source_controller_snapshot_and_actual_query_identity(self):
        _,session,worker,recorder,_=self.run_episode(latency=.7)
        n=recorder.frame_count;frames=recorder.frames
        submitted={int(p['capture_id']):p for p in worker.submissions}
        self.assertLess(len(submitted),n)
        for i in range(n):
            capture=int(frames['capture_id'][i]);tick=int(frames['controller_ticks'][i])
            self.assertEqual(tick,round(frames['observation_sim_time'][i]/.004))
            expected=np.zeros(14) if tick==0 else session.commands[tick-1][1]
            np.testing.assert_array_equal(frames['controller_q'][i],expected)
            np.testing.assert_array_equal(frames['controller_start_q'][i],expected)
            self.assertEqual(frames['controller_elapsed'][i],0.)
            self.assertTrue(frames['controller_initialized'][i])
            self.assertEqual(frames['policy_submitted'][i],capture in submitted)
            if capture in submitted:
                p=submitted[capture]
                self.assertEqual(frames['policy_sampling_seed'][i],p['_sampling_seed'])
                self.assertEqual(frames['dispatch_sim_time'][i],p['dispatch_sim_time'])
                self.assertGreaterEqual(frames['dispatch_sim_time'][i],frames['received_sim_time'][i])
                np.testing.assert_array_equal(recorder.policy_arrays['policy_frame'][i],p['frame'])
                for key in ('flow_frames','motion_features','motion_interval_valid_mask',
                            'motion_acceleration_valid_mask','previous_head_rgb'):
                    np.testing.assert_array_equal(recorder.policy_arrays[key][i],p[key])
            else:
                self.assertEqual(frames['policy_sampling_seed'][i],-1)
                self.assertEqual(frames['policy_request_id'][i],-1)
                self.assertEqual(frames['dispatch_sim_time'][i],frames['observation_sim_time'][i])
        self.assertEqual(recorder.metadata['instruction'],'move the test object')

    def test_dispatch_snapshot_and_prefix_are_exact_past_applied_commands(self):
        _,session,worker,recorder,_=self.run_episode(latency=.01)
        actual=np.asarray([q for _,q,_ in session.commands]);velocity=np.asarray([v for _,_,v in session.commands])
        partial_segment=False
        for packet in worker.submissions:
            source=packet['observation_sim_time'];dispatch=packet['dispatch_sim_time']
            self.assertGreater(dispatch,source)
            state=packet['controller_state'];prefix=state['executed_prefix'];d=state['dispatch_controller']
            start,end=round(source/.004),round(dispatch/.004)
            np.testing.assert_array_equal(prefix['q_command'],actual[start:end])
            np.testing.assert_array_equal(prefix['velocity_command'],velocity[start:end])
            np.testing.assert_array_equal(prefix['start_sim_time'],np.arange(start,end)*.004)
            np.testing.assert_array_equal(prefix['end_sim_time'],np.arange(start+1,end+1)*.004)
            np.testing.assert_array_equal(d['q'],actual[end-1]);np.testing.assert_array_equal(d['v'],velocity[end-1])
            self.assertEqual(d['source_physics_time'],dispatch);self.assertEqual(state['source_physics_time'],source)
            self.assertEqual(d['ticks'],end);partial_segment|=0<d['elapsed']<.1
        self.assertTrue(partial_segment)

    def test_executed_prefix_ring_is_bounded_owned_and_rejects_gaps_or_future(self):
        ring=ExecutedCommandRing();self.assertEqual(ring.q.nbytes+ring.velocity.nbytes,500*14*8*2)
        for tick in range(750):ring.append(np.full(14,tick),np.full(14,-tick),tick_index=tick)
        prefix=ring.prefix(1.,3.)
        self.assertEqual(len(prefix['q_command']),500)
        np.testing.assert_array_equal(prefix['q_command'][:,0],np.arange(250,750))
        ring.append(np.full(14,750),np.full(14,-750),tick_index=750)
        self.assertEqual(prefix['q_command'][0,0],250)  # No alias to overwritten ring storage.
        for source,dispatch in ((0.,3.004),(3.004,3.008),(1.005,3.004),(float('nan'),3.004)):
            with self.assertRaises(TrajectoryRecordingError):ring.prefix(source,dispatch)
        with self.assertRaises(TrajectoryRecordingError):ring.append(np.zeros(14),np.zeros(14),tick_index=752)
        self.assertEqual(ring.prefix(3.004,3.004)['q_command'].shape,(0,14))

    def test_builder_preserves_dispatch_and_checks_it_against_dense_truth(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from build_predictive_feedback_data import verified_query_controller
        _,_,worker,recorder,_=self.run_episode(latency=.01)
        for packet in worker.submissions:
            i=recorder.capture_indices[packet['capture_id']]
            controller=verified_query_controller(packet,recorder.frames,recorder.commands,i)
            self.assertIs(controller,packet['controller_state'])
        packet=worker.submissions[0];i=recorder.capture_indices[packet['capture_id']]
        bad=copy.deepcopy(packet);bad['controller_state']['q'][0]+=.1
        with self.assertRaises(ValueError):verified_query_controller(bad,recorder.frames,recorder.commands,i)
        bad=copy.deepcopy(packet);bad['controller_state']['executed_prefix']['q_command'][0,0]+=.1
        with self.assertRaises(ValueError):verified_query_controller(bad,recorder.frames,recorder.commands,i)
        bad=copy.deepcopy(packet);bad['dispatch_sim_time']+=.004
        with self.assertRaises(ValueError):verified_query_controller(bad,recorder.frames,recorder.commands,i)
        bad=copy.deepcopy(packet);del bad['controller_state']['dispatch_controller']
        with self.assertRaises(KeyError):verified_query_controller(bad,recorder.frames,recorder.commands,i)

    def test_real_proposal_helper_consumes_delayed_loop_packets(self):
        from dynamicwam.runtime.domino_rt.proposal_path import proposal_action_path,forecast_horizons
        _,_,worker,_,_=self.run_episode(latency=.01)
        for packet in worker.submissions:
            source=packet['observation_sim_time'];dispatch=packet['dispatch_sim_time']
            path,dt,provenance=proposal_action_path(controller=packet['controller_state'],plan=np.full((16,14),.05),
                age=packet['command_index'],observation_time=source,mean=np.zeros(14),scale=np.ones(14))
            self.assertEqual(path.shape,(2,4,28));self.assertTrue(np.isfinite(path).all())
            np.testing.assert_array_equal(dt,forecast_horizons(source,dispatch).astype(np.float32))
            self.assertGreater(provenance['assumed_install_sim_time'],dispatch)
            self.assertEqual(provenance['known_executed_prefix_ticks'],round((dispatch-source)/.004))
            self.assertFalse(provenance['actual_future_commands_used'])

    def test_controller_snapshot_resumes_identical_limited_hermite_steps(self):
        from dynamicwam.runtime.domino_rt.trajectory_recorder import controller_snapshot,CONTROLLER_VECTORS
        from dynamicwam.runtime.domino_rt.timed_control import LimitedHermiteController
        original=LimitedHermiteController(np.zeros(14));original.begin(np.ones(14),np.ones(14)*2)
        for _ in range(7):original.step(.004)
        state=controller_snapshot(original);json.dumps(state,allow_nan=False)
        restored=LimitedHermiteController(np.zeros(14))
        for key in CONTROLLER_VECTORS:setattr(restored,key,np.asarray(state[key],dtype=np.float64))
        for key in ('elapsed','period','ticks','limited_ticks'):setattr(restored,key,state[key])
        for _ in range(25):
            oq,ov=original.step(.004);rq,rv=restored.step(.004)
            np.testing.assert_array_equal(oq,rq);np.testing.assert_array_equal(ov,rv)

    def test_recording_does_not_change_controller_or_policy_observations(self):
        _,a,wa,_,_=self.run_episode(record=False);_,b,wb,_,_=self.run_episode()
        np.testing.assert_array_equal(np.array([x[1:] for x in a.commands]),np.array([x[1:] for x in b.commands]))
        self.assertEqual([p['observation_sim_time'] for p in wa.submissions],[p['observation_sim_time'] for p in wb.submissions])

    def test_all_camera_ready_frames_retained_even_when_policy_is_busy(self):
        _,_,worker,recorder,_=self.run_episode(latency=.7,rows=2)
        self.assertGreater(recorder.frame_count,len(worker.submissions))
        np.testing.assert_array_equal(recorder.commands['q_command'][:recorder.tick_count],np.zeros((200,14)))
        self.assertTrue(recorder.commands['hold'][:recorder.tick_count].all())
        frames=recorder.frame_count
        np.testing.assert_allclose(recorder.measured['left_qpos'][:frames],recorder.frames['drive_targets'][:frames,:6]+1.)
        self.assertTrue(recorder.measured['left_qpos_valid'][:frames].all())
        self.assertNotIn('right_qpos',recorder.measured)
        self.assertEqual(len(recorder.measurement_errors),frames)
        self.assertTrue(np.all(recorder.frames['received_sim_time'][:frames]>recorder.frames['observation_sim_time'][:frames]))

    def test_no_persistence_before_finish_or_overwrite_after_finish(self):
        with tempfile.TemporaryDirectory() as d:
            fresh=TrajectoryRecorder(max_seconds=.8)
            with self.assertRaises(TrajectoryRecordingError):fresh.write(Path(d)/'early',provenance={})
            self.assertFalse((Path(d)/'early').exists())
            *_,recorder,_=self.run_episode()
            provenance=dict(source_sha256={'fake.py':'a'*64},controller_protocol_sha256='b'*64,
                accepted_record_sha256='c'*64,policy_config_sha256='d'*64)
            output=Path(d)/'trajectory';completion=recorder.write(output,provenance=provenance)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(completion['status'],'complete')
            for value in manifest['artifacts'].values():
                self.assertEqual(hashlib.sha256((output/value['path']).read_bytes()).hexdigest(),value['sha256'])
            with np.load(output/'commands.npz',allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['q_command'],recorder.commands['q_command'][:200])
            with self.assertRaises(FileExistsError):recorder.write(output,provenance=provenance)

    def test_memory_overrun_fails_without_silent_dropping(self):
        with self.assertRaises(ValueError):TrajectoryRecorder(max_seconds=31)
        with self.assertRaises(TrajectoryRecordingError):TrajectoryRecorder(max_seconds=30,memory_limit_bytes=20*1024**2)
        recorder=TrajectoryRecorder(max_seconds=.8,event_limit_bytes=8)
        with patch('dynamicwam.runtime.domino_rt.trajectory_recorder._load_view_encoder',return_value=fake_encoder):
            recorder.begin(task='t',seed=0,origin_sim_time=0.,initial_drive_targets=np.zeros(14),physics_timeout=.8)
        with self.assertRaises(TrajectoryRecordingError):recorder.event({'event':'larger_than_budget'})
        self.assertEqual(recorder.events,[])

    def test_true_measured_reader_never_substitutes_drive_targets(self):
        source=ast.parse((ROOT/'src/dynamicwam/runtime/domino_rt/continuous_session.py').read_text())
        cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='ContinuousSession')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='measured_joint_state')
        module=ast.Module(body=[method],type_ignores=[]);scope={'np':np}
        exec(compile(ast.fix_missing_locations(module),'actual_measured_reader','exec'),scope)
        robot=SimpleNamespace(left_entity=SimpleNamespace(get_qpos=lambda:np.arange(6),get_qvel=lambda:np.ones(6),
            get_active_joints=lambda:[SimpleNamespace(name=str(i)) for i in range(6)]),right_entity=None)
        result=scope['measured_joint_state'](SimpleNamespace(env=SimpleNamespace(robot=robot)))
        np.testing.assert_array_equal(result['left']['qpos'],np.arange(6))
        self.assertNotIn('qpos',result['right']);self.assertIn('unavailable',result['right']['error'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('cv2'),'real shared encoder requires deployment CPU dependencies')
    def test_real_shared_encoder_has_exact_letterboxed_rgb_and_masks(self):
        from dynamicwam.runtime.domino_rt.trajectory_recorder import _load_view_encoder
        from dynamicwam.observations.multiview import split_composite,letterbox_view
        frame=np.random.default_rng(42).integers(0,256,size=(360,320,3),dtype=np.uint8)
        actual,mask=_load_view_encoder()(frame)
        for i,part in enumerate(split_composite(frame)):
            rgb,expected_mask=letterbox_view(part)
            np.testing.assert_array_equal(actual[i],np.rint(rgb*255).astype(np.uint8))
            np.testing.assert_array_equal(mask[i],expected_mask)


if __name__=='__main__':unittest.main(verbosity=2)
