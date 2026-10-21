"""Actual native queue adapter tests; CPU only, no simulator or model doubles."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import importlib.util
import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dynamicwam.runtime.causal_spatial_contract import (
    CausalSpatialContract, CausalPlanQueue, FiniteObservationWindow,
    LatestObservationMemory, RealtimeViolation,
)


def deterministic_flow(previous, current):
    """Geometry fixture only; actual OpenCV/CUDA performance is not tested."""
    height,width,_=previous.shape
    flow=np.zeros((height,width,2),dtype=np.float32)
    flow[...,0]=current[...,0].astype(float)-previous[...,0]
    return flow,np.ones((height,width),dtype=bool)


def window(episode="ep", contract=CausalSpatialContract()):
    return FiniteObservationWindow(episode,contract,flow_backend=deterministic_flow)


def observe(window, event=0, t=0., wall=10., executed=None):
    return window.append(event_id=event, observation_physics_s=t, available_wall_s=wall,
        spatial_rgb=np.full((3,3,2,2), event, dtype=np.float32),
        raw_rgb=np.full((3,2,2,3),event,dtype=np.uint8),valid=np.ones((3,1,2,2)),
        capture_drive_targets=np.full(14,event,dtype=np.float32),executed=executed)


def prepared(contract=CausalSpatialContract()):
    history = window("ep", contract)
    observation = observe(history)
    store = LatestObservationMemory("ep", contract)
    source = store.begin(observation)
    store.publish(source, ("observation-only-memory",), completed_wall_s=10., completion_ready=True)
    queue = CausalPlanQueue(episode="ep", initial_actions=np.zeros((16,14)),
        grid_start_physics_s=0., physics_epoch_s=0., wall_epoch_s=10., contract=contract)
    return history, store, queue, observation


def candidate(request, suffix_value=1.):
    actions = np.full((16,14), suffix_value)
    actions[:len(request.prefix_actions)] = request.prefix_actions
    return actions


def install(queue, request, actions=None, t=.1, wall=10.1, **kwargs):
    return queue.install(version=request.version, actions=candidate(request) if actions is None else actions,
        action_target_times=request.action_target_times, ready_physics_s=t, ready_wall_s=wall, **kwargs)


class ContractTests(unittest.TestCase):
    def test_primary_contract_is_h2r2_k10_1x(self):
        c = CausalSpatialContract()
        self.assertEqual((c.committed_actions,c.replan_actions,c.denoising_steps), (2,2,10))
        self.assertAlmostEqual(c.commitment_s,.32)
        for changes in ({"simulation_rate":.25}, {"denoising_steps":2},
                        {"committed_actions":1,"replan_actions":2}, {"action_interval_s":.161}):
            with self.assertRaises(ValueError): replace(c,**changes)

    def test_finite_history_is_owned_padded_and_causal(self):
        c = CausalSpatialContract(window_size=2)
        h = window("ep",c)
        first = observe(h)
        self.assertEqual(first.observation_valid.tolist(),[True,True])
        self.assertTrue(first.valid.all())
        self.assertFalse(first.flow_valid.any())
        with self.assertRaises(ValueError): first.spatial_rgb.setflags(write=True)
        second = observe(h,1,.16,10.2)
        self.assertAlmostEqual(second.dt[-1],.16)
        third = observe(h,2,.32,10.4)
        self.assertEqual(third.spatial_rgb[:,0,0,0,0].tolist(),[0,2])
        self.assertEqual(third.spatial_event_ids.tolist(),[0,2])
        self.assertFalse(first.spatial_rgb.any())
        self.assertNotIn("previous_plan", third.model_inputs())
        with self.assertRaises(ValueError): observe(h,2,.32,10.5)
        with self.assertRaises(ValueError): observe(h,3,.3,10.6)

    def test_history_only_accepts_past_execution_receipts(self):
        h,store,q,obs = prepared()
        request=q.request(obs,store.current,request_wall_s=10.)
        install(q,request)
        receipt=q.acknowledge_executed(action_index=0,completed_physics_s=.16,
                                      measured_qpos=np.full(14,.75))
        next_obs=observe(h,1,.16,10.2,receipt)
        self.assertTrue(next_obs.executed_valid[-1])
        # The interval uses its left capture's raw drive target, not the next
        # planned action or measured position at the interval's right endpoint.
        np.testing.assert_array_equal(next_obs.recorded_capture_commands[-1],np.zeros(14))
        self.assertFalse(np.array_equal(receipt.action,receipt.measured_qpos))
        with self.assertRaises(ValueError): observe(h,2,.17,10.3,replace(receipt,completed_physics_s=.4))
        with self.assertRaises(ValueError): observe(h,2,.17,10.3,replace(receipt,episode="other"))

    def test_latest_memory_does_not_regress_or_overwrite_belief(self):
        h=window(); store=LatestObservationMemory("ep")
        old_obs=observe(h); old=store.begin(old_obs)
        new_obs=observe(h,1,.16,10.2); new=store.begin(new_obs)
        self.assertTrue(store.publish(new,"new",completed_wall_s=10.3,completion_ready=True))
        self.assertFalse(store.publish(old,"old",completed_wall_s=10.4,completion_ready=True))
        self.assertEqual(store.current.payload,"new")
        self.assertEqual(h.snapshot().revision,new_obs.revision)
        with self.assertRaises(ValueError): store.publish(old,"replay",completed_wall_s=10.5,completion_ready=True)

    def test_memory_completion_availability_and_age_are_required(self):
        contract=CausalSpatialContract(max_memory_age_s=.5)
        h=window(contract=contract); obs=observe(h)
        store=LatestObservationMemory("ep",contract)
        token=store.begin(obs)
        with self.assertRaises(RuntimeError): store.publish(token,"x",completed_wall_s=10.2,completion_ready=False)
        with self.assertRaises(ValueError): store.publish(token,"x",completed_wall_s=9.,completion_ready=True)
        store.publish(token,"x",completed_wall_s=10.2,completion_ready=True)
        with self.assertRaises(ValueError): store.snapshot(observation_physics_s=.1,request_wall_s=10.1)
        with self.assertRaises(ValueError): store.snapshot(observation_physics_s=-.1,request_wall_s=10.3)
        with self.assertRaises(ValueError): store.snapshot(observation_physics_s=.51,request_wall_s=10.6)

    def test_physical_window_matches_offline_helper_after_eviction(self):
        from dynamicwam.observations.causal_spatial_window import causal_window_indices
        h=window(); times=np.arange(21)*.1
        for i,t in enumerate(times):
            snap=observe(h,i,float(t),10+float(t))
            selected,actual,dt,valid=causal_window_indices(times[:i+1],i)
            np.testing.assert_array_equal(snap.spatial_event_ids,selected)
            np.testing.assert_array_equal(snap.spatial_times,actual)
            np.testing.assert_array_equal(snap.dt,dt)
            np.testing.assert_array_equal(snap.command_valid,valid)
            expected=np.concatenate((selected[:1],selected[:-1])).astype(float)
            expected[~valid]=0.
            np.testing.assert_array_equal(snap.recorded_capture_commands[:,0],expected)
        self.assertLess(len(h._frames),len(times))
        self.assertEqual(set(snap.model_inputs()),{"spatial_rgb","spatial_flow","spatial_valid",
            "spatial_flow_valid","spatial_dt","recorded_capture_commands","command_valid"})

    def test_parallel_memory_publication_selects_latest_source(self):
        h=window(); store=LatestObservationMemory("ep")
        sources=[store.begin(observe(h,i,i*.16,10+i*.16)) for i in range(12)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda s: store.publish(s,s.event_id,completed_wall_s=20.,
                                                completion_ready=True),reversed(sources)))
        self.assertEqual(store.current.source.event_id,11)
        self.assertEqual(store.current.payload,11)

    def test_complete_pair_cache_reuses_flow_and_warped_validity_only_for_same_exposures(self):
        from dynamicwam.observations import causal_spatial_window as shared
        with patch.object(shared,"flow_pair_validity",wraps=shared.flow_pair_validity) as validity:
            backend=unittest.mock.Mock(side_effect=deterministic_flow)
            h=FiniteObservationWindow("ep",flow_backend=backend)
            for event in range(4):
                observe(h,event,event*.32,10+event*.32)
                self.assertEqual(backend.call_count,event*3)
                self.assertEqual(validity.call_count,event*3)
            previous=h.snapshot()
            h.snapshot()
            self.assertEqual((backend.call_count,validity.call_count),(9,9))
            observe(h,4,1.28,11.28)
            self.assertEqual((backend.call_count,validity.call_count),(12,12))
            # Only the new 3-view pair was evaluated; the previous 2 pairs
            # retained their FULL masked tensors and previous-padding validity.
            np.testing.assert_array_equal(h.snapshot().flow[1:3],previous.flow[2:4])
            np.testing.assert_array_equal(h.snapshot().flow_valid[1:3],previous.flow_valid[2:4])
            self.assertEqual(len(h._flows),9)
            for event in range(5,30):observe(h,event,event*.32,10+event*.32)
            live={frame[0] for frame in h._frames}
            self.assertTrue(all(a in live and b in live for a,b,*_ in h._flows))
            self.assertLessEqual(len(h._flows),9)
            self.assertEqual((backend.call_count,validity.call_count),(87,87))

    def test_cached_full_tensors_equal_uncached_shared_flow_semantics_with_padding(self):
        from dynamicwam.observations.causal_spatial_window import causal_window_indices,flow_pair_validity
        def subpixel_flow(previous,current):
            h,w,_=previous.shape
            delta=np.empty((h,w,2),np.float32)
            delta[...,0]=.5 if current[0,0,0]%2 else -.5
            delta[...,1]=.25
            confidence=np.ones((h,w),bool);confidence[1,2]=False
            return delta,confidence
        h=FiniteObservationWindow("ep",flow_backend=subpixel_flow)
        exposures=[]
        for event,t in enumerate((0.,.1,.32,.49,.64,.96,1.28,1.61,1.92)):
            raw=np.full((3,6,8,3),event,np.uint8)
            rgb=raw.transpose(0,3,1,2).astype(np.float32)/255
            valid=np.ones((3,1,6,8),bool)
            valid[0,0,:,0]=False;valid[1,0,-1,:]=False
            valid[2,0,2,event%8]=False
            exposures.append((t,raw.copy(),valid.copy()))
            snap=h.append(event_id=event,observation_physics_s=t,available_wall_s=10+t,
                spatial_rgb=rgb,raw_rgb=raw,valid=valid,capture_drive_targets=np.full(14,event))
            raw[:]=255;valid[:]=False # the cache owns immutable exposure inputs
            indices,_,_,transitions=causal_window_indices([e[0] for e in exposures],event)
            expected=np.zeros_like(snap.flow);expected_valid=np.zeros_like(snap.flow_valid)
            for row in range(1,4):
                if not transitions[row]:continue
                previous,current=exposures[indices[row-1]],exposures[indices[row]]
                for view in range(3):
                    delta,confidence=subpixel_flow(previous[1][view],current[1][view])
                    mask=flow_pair_validity(delta,confidence,previous[2][view,0],current[2][view,0])
                    expected[row,view]=np.where(mask[...,None],delta,0).transpose(2,0,1)
                    expected_valid[row,view,0]=mask
            np.testing.assert_array_equal(snap.flow,expected)
            np.testing.assert_array_equal(snap.flow_valid,expected_valid)
            np.testing.assert_array_equal(h.snapshot().flow,expected)
        self.assertTrue(snap.flow_valid.any())
        self.assertTrue((~snap.flow_valid[1:]).any())
        for pair_flow,pair_valid in h._flows.values():
            with self.assertRaises(ValueError):pair_flow.setflags(write=True)
            with self.assertRaises(ValueError):pair_valid.setflags(write=True)

    def test_three_view_parallel_pairs_match_sequential_and_join_before_return(self):
        barrier=threading.Barrier(3)
        thread_ids=set();lock=threading.Lock()
        def parallel_backend(previous,current):
            with lock:thread_ids.add(threading.get_ident())
            barrier.wait(timeout=5.)
            return deterministic_flow(previous,current)
        parallel=FiniteObservationWindow("ep",flow_backend=parallel_backend)
        sequential=FiniteObservationWindow("ep",flow_backend=deterministic_flow,flow_workers=1)
        for event in range(5):
            a=observe(parallel,event,event*.32,10+event*.32)
            b=observe(sequential,event,event*.32,10+event*.32)
            for name in ("flow","flow_valid","spatial_rgb","valid","dt","recorded_capture_commands",
                         "command_valid","spatial_times","spatial_event_ids"):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            self.assertEqual(list(parallel._flows),list(sequential._flows))
            self.assertFalse(any(t.name.startswith("causal-spatial-flow") for t in threading.enumerate()))
        self.assertGreaterEqual(len(thread_ids),3)
        self.assertEqual(parallel._last_flow_workers,3)
        parallel.snapshot()
        self.assertEqual(parallel._last_flow_workers,0)

    def test_parallel_failure_propagates_without_cache_publication_or_orphan_threads(self):
        barrier=threading.Barrier(3)
        def failing(previous,current):
            barrier.wait(timeout=5.)
            raise RuntimeError("flow backend failure")
        h=FiniteObservationWindow("ep",flow_backend=failing)
        observe(h)
        with self.assertRaisesRegex(RuntimeError,"flow backend failure"):
            observe(h,1,.32,10.32)
        self.assertEqual(len(h._flows),0)
        self.assertFalse(any(t.name.startswith("causal-spatial-flow") for t in threading.enumerate()))
        for workers in (0,4,True,1.5):
            with self.assertRaises(ValueError):FiniteObservationWindow("ep",flow_workers=workers)

    @unittest.skipUnless(importlib.util.find_spec("cv2"),"OpenCV unavailable in local CPU runtime")
    def test_real_opencv_parallel_matches_sequential_and_respects_global_threads(self):
        import cv2
        saved=cv2.getNumThreads()
        rng=np.random.default_rng(31)
        def append(h,event,raw):
            valid=np.ones((3,1,*raw.shape[1:3]),bool);valid[:,:,:,0]=False
            return h.append(event_id=event,observation_physics_s=event*.32,available_wall_s=10+event*.32,
                spatial_rgb=raw.transpose(0,3,1,2).astype(np.float32)/255,raw_rgb=raw,valid=valid,
                capture_drive_targets=np.zeros(14))
        try:
            cv2.setNumThreads(1)
            parallel=FiniteObservationWindow("ep");sequential=FiniteObservationWindow("ep",flow_workers=1)
            base=rng.integers(0,256,(3,64,80,3),dtype=np.uint8)
            for event in range(5):
                raw=np.roll(base,event,axis=2)
                a,b=append(parallel,event,raw),append(sequential,event,raw)
                np.testing.assert_array_equal(a.flow,b.flow)
                np.testing.assert_array_equal(a.flow_valid,b.flow_valid)
            self.assertEqual(parallel._last_flow_workers,3)
            cv2.setNumThreads(2)
            fallback=FiniteObservationWindow("ep")
            append(fallback,0,base);append(fallback,1,np.roll(base,1,axis=2))
            self.assertEqual(fallback._last_flow_workers,1)
            self.assertEqual(cv2.getNumThreads(),2)
        finally:cv2.setNumThreads(saved)


class NativeQueueAdapterTests(unittest.TestCase):
    def test_native_queue_reused_with_explicit_h_and_no_bootstrap_overlap(self):
        h,store,q,obs=prepared()
        request=q.request(obs,store.current,request_wall_s=10.)
        self.assertEqual(request.prefix_actions.shape,(2,14))
        self.assertEqual(len(request.reservation.actions),2)
        self.assertAlmostEqual(request.reservation.deadline_physics_s,.32)
        self.assertFalse(request.overlap_valid.any())
        self.assertFalse(request.overlap_actions.any())
        self.assertEqual(q.next_block().shape,(2,14))
        with self.assertRaises(ValueError): q.next_block(4)
        np.testing.assert_allclose(request.action_target_times,np.arange(1,17)*.16)
        np.testing.assert_allclose(request.future_target_offsets,[1.28,2.56])

    def test_explicit_h_does_not_silently_reserve_four(self):
        for h in (1,2,3,4):
            history,store,q,obs=prepared(CausalSpatialContract(committed_actions=h,replan_actions=1))
            request=q.request(obs,store.current,request_wall_s=10.)
            self.assertEqual(len(request.reservation.actions),h)
            self.assertAlmostEqual(request.reservation.deadline_physics_s,h*.16)

    def test_absolute_grid_and_queue_overlap_advance_together(self):
        h,store,q,obs=prepared()
        first=q.request(obs,store.current,request_wall_s=10.)
        actions=np.arange(224,dtype=float).reshape(16,14)/1000
        actions[:2]=0
        install(q,first,actions)
        q.acknowledge_executed(action_index=0,completed_physics_s=.16)
        receipt=q.acknowledge_executed(action_index=1,completed_physics_s=.32)
        with self.assertRaises(RuntimeError): q.next_action()
        obs2=observe(h,1,.32,10.32,receipt)
        second=q.request(obs2,store.current,request_wall_s=10.32)
        np.testing.assert_allclose(second.action_target_times,(2+np.arange(1,17))*.16)
        np.testing.assert_array_equal(second.prefix_actions,actions[2:4].astype(np.float32))
        np.testing.assert_array_equal(np.flatnonzero(second.overlap_valid),[2,3])
        np.testing.assert_array_equal(second.overlap_actions[2:4],actions[4:6].astype(np.float32))
        self.assertFalse(second.overlap_actions[~second.overlap_valid].any())

    def test_actual_wall_deadline_wins_when_physics_is_behind(self):
        h,store,q,obs=prepared(); request=q.request(obs,store.current,request_wall_s=10.)
        before=q.native.take(0,16)
        with self.assertRaisesRegex(RealtimeViolation,"late suffix"):
            install(q,request,t=.1,wall=10.321)
        np.testing.assert_array_equal(q.native.take(0,16),before)

    def test_no_result_cannot_execute_past_committed_prefix(self):
        h,store,q,obs=prepared(); q.request(obs,store.current,request_wall_s=10.)
        q.acknowledge_executed(action_index=0,completed_physics_s=.16)
        q.acknowledge_executed(action_index=1,completed_physics_s=.32)
        with self.assertRaises(RealtimeViolation): q.next_action()

    def test_result_version_grid_prefix_checks_are_atomic(self):
        h,store,q,obs=prepared(); request=q.request(obs,store.current,request_wall_s=10.)
        with self.assertRaises(RuntimeError):
            q.install(version=replace(request.version,episode="other"),actions=candidate(request),
                action_target_times=request.action_target_times,ready_physics_s=.1,ready_wall_s=10.1)
        with self.assertRaises(ValueError):
            q.install(version=request.version,actions=candidate(request),
                action_target_times=request.action_target_times+.001,ready_physics_s=.1,ready_wall_s=10.1)
        bad=candidate(request);bad[0,0]=.5
        with self.assertRaises(RuntimeError): install(q,request,bad)
        np.testing.assert_array_equal(q.native.take(0,16),np.zeros((16,14)))
        install(q,request)
        with self.assertRaises(RuntimeError): install(q,request)

    def test_belief_advancing_does_not_invalidate_owned_request(self):
        h,store,q,obs=prepared(); request=q.request(obs,store.current,request_wall_s=10.)
        newer=observe(h,1,.16,10.16)
        self.assertGreater(newer.revision,request.version.belief_revision)
        install(q,request,t=.2,wall=10.2)
        self.assertEqual(q.native.generation,1)

    def test_execution_acknowledgement_cannot_precede_execution(self):
        h,store,q,obs=prepared(); q.request(obs,store.current,request_wall_s=10.)
        with self.assertRaises(ValueError): q.acknowledge_executed(action_index=0,completed_physics_s=.01)
        q.acknowledge_executed(action_index=0,completed_physics_s=.16)
        with self.assertRaises(ValueError): q.acknowledge_executed(action_index=0,completed_physics_s=.16)

    def test_camera_and_memory_cannot_move_query_boundary(self):
        h,store,q,obs=prepared()
        with self.assertRaises(ValueError):
            q.request(replace(obs,observation_physics_s=.02),store.current,request_wall_s=10.)
        with self.assertRaises(ValueError):
            q.request(obs,replace(store.current,completed_wall_s=10.1),request_wall_s=10.)
        wrong=replace(store.current,source=replace(store.current.source,contract_version="old-h4"))
        with self.assertRaisesRegex(ValueError,"contract version"):
            q.request(obs,wrong,request_wall_s=10.)


if __name__ == "__main__":
    unittest.main()
