#!/usr/bin/env python3
"""CPU-only adversarial checks for registration, pairing, and fail-stop gates."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

# Windows sandbox temp redirection may point outside the writable workspace.
tempfile.tempdir=str(Path.cwd())

from make_world_context_rt_manifest import TASKS, build_manifest, partition_rows, write_json,object_hash
from run_world_context_rt_pipeline import (
    GateFailed, Incomplete, IdentityViolation, Pipeline, controller_protocol, resource_budget, source_identity, gpu_conflicts,
    validate_canary, validate_episode, validate_probe, validate_profile,validate_replica,validate_feedback_consumption,
    validate_camera_events)
from analyze_world_context_rt import paired_statistics, timing_evidence


def episode():
    return dict(format='domino_continuous_episode',version=1,status='complete',task='place_shoe',seed=3,
        controller_protocol_sha256='p'*64,accepted_record={'sha256':'a'*64},mode='async',
        controller=dict(control_period_s=.1,physics_dt=.004,max_velocity=2.,max_acceleration=20.,query_slots=2,
            physics_timeout_s=30.,max_backlog_s=.1,max_tick_gap_s=.1,warmup_queries=1,
            camera_pipeline='independent_scene_snapshot_renderer'),
        report=dict(success=False,realtime_valid=True,virtual_delay=False,physics_ticks=100,
            inference_inflight_ticks=30,tick_lag_s=[.004,.005],tick_gap_s=[.004,.004],
            asynchronous_camera=True,camera_inflight_ticks=10,
            requests=[dict(physics_advanced_during_worker_s=.02)]))


class ResourceTests(unittest.TestCase):
    def inventory(self,n):return [dict(index=str(i),uuid=f'GPU-{i}') for i in range(n)]

    def test_six_card_exact_budget(self):
        b=resource_budget(self.inventory(6));self.assertEqual((b['world_size'],b['per_gpu_batch'],b['gradient_accumulation'],b['global_batch']),(6,2,2,24))

    def test_rank_adaptation_preserves_budget(self):
        self.assertEqual(resource_budget(self.inventory(4))['gradient_accumulation'],3)
        self.assertEqual(resource_budget(self.inventory(5))['world_size'],4)
        self.assertEqual(resource_budget(self.inventory(7))['world_size'],6)
        with self.assertRaises(ValueError):resource_budget(self.inventory(5),'0,1,2,3,4')
        with self.assertRaises(ValueError):resource_budget(self.inventory(6),'0,0')

    def test_graphics_renderer_is_a_resource_conflict(self):
        xml='<nvidia_smi_log><gpu><uuid>GPU-0</uuid><processes><process_info><pid>42</pid><type>G</type><process_name>sapien</process_name></process_info></processes></gpu></nvidia_smi_log>'
        with patch('run_world_context_rt_pipeline.subprocess.check_output',return_value=xml):
            self.assertEqual(gpu_conflicts(self.inventory(1))[0]['type'],'G')
            self.assertEqual(gpu_conflicts([dict(index='1',uuid='GPU-1')]),[])


class GateTests(unittest.TestCase):
    def test_false_success_is_valid_policy_evidence(self):
        value=validate_episode(episode(),dict(task='place_shoe',seed=3),'a'*64,'p'*64)
        self.assertIs(value['success'],False)

    def test_invalid_not_counted_as_policy_failure(self):
        for key,value in [('status','infrastructure_invalid'),('seed',4),('controller_protocol_sha256','other')]:
            r=episode();r[key]=value
            with self.assertRaises(Incomplete):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)

    def test_same_seed_is_insufficient(self):
        with self.assertRaises(Incomplete):validate_episode(episode(),dict(task='place_shoe',seed=3),'different','p'*64)

    def test_label_cannot_hide_actual_controller_change(self):
        r=episode();r['controller']['physics_timeout_s']=60
        with self.assertRaises(Incomplete):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)

    def test_no_physics_advance_and_deadline_violation(self):
        for key,value in [('inference_inflight_ticks',0),('tick_gap_s',[.1001]),('tick_lag_s',[.11])]:
            r=episode();r['report'][key]=value
            with self.assertRaises(GateFailed):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)

    def test_camera_age_cannot_fake_inference_overlap(self):
        r=episode();r['report']['requests']=[dict(inference_physics_advanced=.5)]
        with self.assertRaises(Incomplete):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)
        r['report']['requests'][0]['physics_advanced_during_worker_s']=0.
        with self.assertRaises(GateFailed):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)

    def test_fast_synchronous_camera_is_not_the_registered_pipeline(self):
        r=episode();r['report']['asynchronous_camera']=False
        with self.assertRaises(Incomplete):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)
        r=episode();r['report']['camera_inflight_ticks']=0
        with self.assertRaises(Incomplete):validate_episode(r,dict(task='place_shoe',seed=3),'a'*64,'p'*64)

    def test_profile_recomputes_service_budget(self):
        r=dict(passed=True,fast_samples_ms=[5.]*20,graph_samples_ms=[200.]*20,
            eager_samples_ms=[400.]*20,native_action_max_abs_error=0.)
        self.assertAlmostEqual(validate_profile(r)['amortized_ratio'],.5625)
        r['fast_samples_ms']=[40.]*20
        with self.assertRaises(GateFailed):validate_profile(r)
        r['fast_samples_ms']=[float('nan')]*20
        with self.assertRaises(Incomplete):validate_profile(r)

    def test_probe_requires_predictive_context_not_boolean_attestation(self):
        r=dict(passed=True,minimum_context_gain_fraction=.01,
            heldout_teacher_mse=dict(fresh=.8,initialization=1.,no_context=.9,context_shuffle=.95,zero=1.1))
        validate_probe(r)
        for key in ('initialization','no_context','context_shuffle','zero'):
            changed=copy.deepcopy(r);changed['heldout_teacher_mse'][key]=.7
            with self.assertRaises(GateFailed):validate_probe(changed)

    def test_canary_no_seed_replacement(self):
        zero=[dict(task='x',seed=i,success=True) for i in range(3)]
        learned=copy.deepcopy(zero);learned[0]['success']=False
        validate_canary(zero,learned)
        learned[1]['success']=False
        with self.assertRaises(GateFailed):validate_canary(zero,learned)
        learned[1]['seed']=99
        with self.assertRaises(Incomplete):validate_canary(zero,learned)

    def test_all_failed_canary_cannot_authorize_long_training(self):
        zero=[dict(task='x',seed=i,success=False) for i in range(6)]
        with self.assertRaisesRegex(GateFailed,'no informative successful'):
            validate_canary(zero,copy.deepcopy(zero))
        learned=copy.deepcopy(zero);learned[0]['success']=True
        self.assertEqual(validate_canary(zero,learned)['learned_success'],1)

    def test_async_camera_timestamp_must_remain_physical_snapshot_time(self):
        event=dict(event='camera_ready',observation_sim_time=.2,
            snapshot_oracle=dict(passed=True,physics_steps=0,source_sim_time_seconds=.200000002))
        self.assertEqual(validate_camera_events([event])['camera_ready_frames'],1)
        event['observation_sim_time']=.4
        with self.assertRaises(IdentityViolation):validate_camera_events([event])
        event['observation_sim_time']=.2;event['snapshot_oracle']['physics_steps']=1
        with self.assertRaises(Incomplete):validate_camera_events([event])


class ReplicaAndConsumptionTests(unittest.TestCase):
    def replica(self):
        import numpy as np
        from check_scene_replica import VIEWS,compare_repetitions
        scene_identity=dict(entity_count=3);scene_identity['sha256']=object_hash(scene_identity)
        oracle=dict(passed=True,physics_steps=0,identity_sha256=scene_identity['sha256'],
            tolerance=1e-4,max_translation_error=0.,max_sign_invariant_quaternion_error=0.)
        phases=[];images={}
        for name,stamp in [('initial',1.),('after_physical_motion',1.1)]:
            groups=[{view:np.zeros((8,8,3),dtype=np.uint8) for view in VIEWS} for _ in range(4)]
            value=compare_repetitions(groups,groups);oracles=[dict(**oracle,source_sim_time_seconds=stamp) for _ in range(4)]
            phases.append(dict(**value,phase=name,source_time_seconds=stamp,
                source_oracles=copy.deepcopy(oracles),replica_oracles=copy.deepcopy(oracles)))
            images[name]={f'{owner}_{view}':np.stack([g[view] for g in groups]) for owner in ('source','replica') for view in VIEWS}
        report=dict(format='dynamicwam_scene_replica_parity',version=1,status='complete',passed=True,
            task='place_shoe',seed=3,accepted_record={'sha256':'a'*64},instruction_type='unseen',
            task_config='demo_clean_dynamic',policy_success_evaluation=False,repetitions=4,
            scene_identity=scene_identity,phases=phases,motion=dict(physics_time_before=1.,physics_time_after=1.1,
                max_actual_robot_qpos_change=.01,target_translation=.01,robot_motion_observed=True,object_motion_observed=True))
        return report,images

    def test_replica_recomputes_from_images_and_requires_unseen_pairing(self):
        report,images=self.replica();scene=dict(task='place_shoe',seed=3)
        self.assertTrue(validate_replica(report,scene,'a'*64,images)['passed'])
        images['initial']['replica_head_camera'][0,0,0,0]=1
        with self.assertRaises(Incomplete):validate_replica(report,scene,'a'*64,images)
        report,images=self.replica();report['instruction_type']='seen'
        with self.assertRaises(IdentityViolation):validate_replica(report,scene,'a'*64,images)
        report,images=self.replica()
        with self.assertRaises(IdentityViolation):validate_replica(report,scene,'b'*64,images)

    def test_replica_boolean_does_not_replace_physical_pose_evidence(self):
        scene=dict(task='place_shoe',seed=3)
        report,images=self.replica();report['phases'][1]['replica_oracles'][0]['physics_steps']=1
        with self.assertRaises(Incomplete):validate_replica(report,scene,'a'*64,images)
        report,images=self.replica();report['motion']['target_translation']=0.
        with self.assertRaises(GateFailed):validate_replica(report,scene,'a'*64,images)

    def feedback(self,*,start=.2,end=.25,norms=None):
        q=dict(request_id=1,action_origin_sim_time=0.,expired=False,actual_install_sim_time=.2,
            clipped_prefix=2,metadata=dict(feedback_rows=4,feedback_row_l2=[.1,.2,.3,.4] if norms is None else norms,
                feedback_row_l2_units='normalized_action'))
        return dict(policy_config=dict(mode='world_context'),report=dict(physics_time=end,control_period_s=.1,
            requests=[q],control_intervals=[dict(start_sim_time=start,end_sim_time=start+.1,hold=False,
                request_id=1,returned_row_index=round(start/.1))]))

    def test_consumption_requires_nonexpired_positive_duration_targets(self):
        value=validate_feedback_consumption([self.feedback()])
        self.assertAlmostEqual(value['consumed_nonzero_slot_equivalents'],.5)
        self.assertEqual(value['generated_nonzero_rows'],4)
        # The immutable tail executes successfully, but all four corrected rows
        # have elapsed: generated residuals and a returned plan cannot pass.
        with self.assertRaisesRegex(GateFailed,'expired or were unconsumed'):
            validate_feedback_consumption([self.feedback(start=.4,end=.5)])
        with self.assertRaises(GateFailed):validate_feedback_consumption([self.feedback(start=.2,end=.2)])
        with self.assertRaisesRegex(GateFailed,'no nonzero'):
            validate_feedback_consumption([self.feedback(norms=[0.]*4)])

    def test_consumption_rejects_missing_units_and_false_installation(self):
        report=self.feedback();report['report']['requests'][0]['metadata']['feedback_row_l2_units']='radians'
        with self.assertRaises(Incomplete):validate_feedback_consumption([report])
        report=self.feedback();report['report']['requests'][0]['actual_install_sim_time']=.3
        with self.assertRaises(Incomplete):validate_feedback_consumption([report])
        report=self.feedback();report['report']['control_intervals'][0]['returned_row_index']=0
        with self.assertRaises(Incomplete):validate_feedback_consumption([report])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        # Windows sandbox denies GetFinalPathName on newly created temp files.
        # Only the test identity PATH normalization is shimmed; real file bytes
        # and every manifest content hash are still exercised. Linux is unmodified.
        if os.name=='nt':
            from make_world_context_rt_manifest import sha256
            self.patch=patch('make_world_context_rt_manifest.identity',
                lambda p:dict(path=str(Path(p).absolute()),sha256=sha256(p)))
            self.patch.start();self.addCleanup(self.patch.stop)

    def fixture(self,root):
        rows=[];raw=root/'raw';official=root/'official.json';protocol=root/'protocol.json'
        for t,task in enumerate(TASKS):
            for e in range(20):rows.append(dict(episode_id=t*20+e,split='clean',task_name=task,episode_name=str(e),sample_count=5))
            path=raw/'clean'/task/'absolute_motion_clean'/'seed.txt';path.parent.mkdir(parents=True)
            path.write_text('\n'.join(str(t*100+e) for e in range(20)))
        episodes=root/'episodes.jsonl';episodes.write_text('\n'.join(json.dumps(r) for r in rows))
        write_json(official,dict(schema_version=1,source_arm='official_sync_h16',
            tasks={task:list(range(100000+t*20,100020+t*20)) for t,task in enumerate(TASKS)}))
        write_json(protocol,controller_protocol())
        return dict(episodes_path=episodes,raw_root=raw,official_path=official,controller_protocol=protocol),rows

    def test_fixed_counts_disjoint_partitions_and_original_seed_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            args,rows=self.fixture(Path(directory));m=build_manifest(**args)
            self.assertEqual(m,build_manifest(**args))
            for role,count in m['counts'].items():self.assertEqual(sum(x['role']==role for x in m['episodes']),count)
            self.assertEqual(len({(x['task'],x['seed']) for x in m['episodes']}),945)
            assignments=partition_rows(rows,20260928,.1)
            for row in m['episodes']:
                if row['role']!='formal':self.assertEqual(assignments[row['source_episode_id']],row['partition'])
            self.assertEqual(sum(x['development'] and x['role']=='stage2' for x in m['episodes']),6)
            self.assertEqual(sum(x['development'] and x['role']=='validation' for x in m['episodes']),3)

    def test_official_overlap_rejected_without_selecting_later_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_=self.fixture(Path(directory));m=build_manifest(**args)
            chosen=next(x for x in m['episodes'] if x['role']=='stage2')
            v=json.loads(args['official_path'].read_text());task=chosen['task']
            v['tasks'][task]=sorted([chosen['seed']]+v['tasks'][task][1:]);write_json(args['official_path'],v)
            with self.assertRaisesRegex(ValueError,'replacement prohibited'):build_manifest(**args)

    def test_exact_task_set(self):
        with tempfile.TemporaryDirectory() as directory:
            args,_=self.fixture(Path(directory));v=json.loads(args['official_path'].read_text())
            v['tasks']['invented']=v['tasks'].pop(TASKS[0]);write_json(args['official_path'],v)
            with self.assertRaises(ValueError):build_manifest(**args)

    def test_source_hash_detects_mutation_and_excludes_pyc(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir();p=root/'scripts/a.py';p.write_text('a=1')
            before=source_identity(root);(root/'scripts/a.pyc').write_bytes(b'noise')
            self.assertEqual(before,source_identity(root));p.write_text('a=2')
            self.assertNotEqual(before,source_identity(root))


class AnalysisTests(unittest.TestCase):
    def test_paired_discordance(self):
        a={('task',i):i<3 for i in range(4)};b={('task',i):i!=0 for i in range(4)}
        r=paired_statistics(a,b,samples=40)
        self.assertEqual((r['treatment_only_success'],r['baseline_only_success']),(1,1))
        self.assertEqual(r['paired_difference'],0.)
        b.pop(('task',0))
        with self.assertRaises(Incomplete):paired_statistics(a,b)

    def test_partial_terminal_slot_and_unfinished_request_cost(self):
        r=episode();r['report'].update(physics_time=.15,wall_time=.155,control_period_s=.1,
            inference_calls=2,terminal_inflight={'request_id':2},controller_limited_ticks=0,
            control_intervals=[dict(start_sim_time=0.,end_sim_time=.1,hold=True),
                               dict(start_sim_time=.1,end_sim_time=.2,hold=False)],action_observation_age_s=[.1],
            requests=[dict(worker_compute_s=.03,request_wall_s=.035,observation_to_return_wall_s=.06,
                inference_physics_advanced=.04,expired=False,actual_install_sim_time=.1,
                observation_sim_time=0.,clipped_prefix=1,metadata={})])
        value=timing_evidence([r])
        self.assertAlmostEqual(value['actual_control_slot_equivalents'],1.5)
        self.assertAlmostEqual(value['actual_nonhold_slot_equivalents'],.5)
        self.assertAlmostEqual(value['weighted_worker_compute_ms_per_control_slot'],20.)
        self.assertEqual(value['terminal_unfinished_fraction'],.5)
        self.assertEqual(value['timing']['observation_to_return_wall_ms']['p95'],60.)

    def test_correction_only_counts_rows_consumed_after_prefix_clipping(self):
        r=episode();r['report'].update(physics_time=.35,wall_time=.36,control_period_s=.1,
            inference_calls=1,terminal_inflight=None,controller_limited_ticks=0,
            control_intervals=[dict(start_sim_time=.2,end_sim_time=.3,hold=False,request_id=1),
                               dict(start_sim_time=.3,end_sim_time=.4,hold=False,request_id=1)],action_observation_age_s=[.2,.3],
            requests=[dict(request_id=1,action_origin_sim_time=0.,worker_compute_s=.03,request_wall_s=.035,
                inference_physics_advanced=.04,expired=False,actual_install_sim_time=.2,
                observation_sim_time=0.,clipped_prefix=2,metadata=dict(feedback_rows=4,
                    feedback_row_l2=[.1,.2,0.,.3],feedback_row_l2_units='normalized_action'))])
        value=timing_evidence([r])
        self.assertAlmostEqual(value['feedback_window_executed_slot_equivalents'],1.5)
        self.assertAlmostEqual(value['nonzero_feedback_executed_slot_equivalents'],.5)
        self.assertEqual(value['nonzero_feedback_executed_intervals'],1)

    def test_sampling_submission_install_and_worker_overlap_are_distinct(self):
        r=episode();r['report'].update(physics_time=.6,wall_time=.61,control_period_s=.1,
            inference_calls=2,terminal_inflight=None,controller_limited_ticks=0,
            control_intervals=[dict(start_sim_time=0.,end_sim_time=.1,hold=True),
                dict(start_sim_time=.1,end_sim_time=.6,hold=False)],action_observation_age_s=[.1],
            requests=[dict(request_id=i,worker_compute_s=.03,request_wall_s=.05,
                inference_physics_advanced=.12,physics_advanced_during_worker_s=.024,
                expired=False,actual_install_sim_time=.2+i*.3,actual_install_wall=100.21+i*.31,
                observation_sim_time=i*.2,submitted_wall=100.1+i*.22,
                camera_ready_wall=100.08+i*.22,clipped_prefix=2,metadata={}) for i in range(2)])
        r['_events']=[dict(event='submit',request_id=i,observation_sim_time=i*.2,
            observation_age_at_submit_s=.08) for i in range(2)]
        value=timing_evidence([r]);d=value['timing']
        self.assertAlmostEqual(value['hold_physical_time_fraction'],1/6)
        for name,expected in [('completed_observation_sampling_interval_sim_s',.2),
                ('completed_policy_submission_interval_wall_s',.22),
                ('plan_installation_interval_sim_s',.3),('plan_installation_interval_wall_s',.31),
                ('physics_advanced_during_worker_s',.024),('observation_to_return_sim_s',.12)]:
            self.assertAlmostEqual(d[name]['p50'],expected)
        self.assertNotIn('inference_physics_advanced_s',d)

    def test_task_specific_domino_hashes_accumulate_without_false_conflict(self):
        p=Pipeline.__new__(Pipeline);p.command_lock=threading.Lock();p.domino_source=None
        p.frozen={'files':{}};p.external={'files':{'envs/a.py':'one','envs/b.py':'two'}}
        p.verify_episode_source({'domino_source_sha256':{'envs/a.py':'one'}})
        p.verify_episode_source({'domino_source_sha256':{'envs/b.py':'two'}})
        self.assertEqual(len(p.domino_source),2)
        with self.assertRaises(IdentityViolation):p.verify_episode_source({'domino_source_sha256':{'envs/a.py':'changed'}})


if __name__=='__main__':unittest.main(verbosity=2)
