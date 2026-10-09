#!/usr/bin/env python3
"""CPU checks of immutable data joins, fixed budgets and failure semantics."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
tempfile.tempdir=str(Path.cwd())
import run_predictive_feedback_pipeline as p
from build_predictive_feedback_data import validate_rows
from make_world_context_rt_manifest import sha256,write_json
from analyze_predictive_feedback import paired_keys


class PipelineTests(unittest.TestCase):
    def inventory(self,n):return [dict(index=str(i),uuid='GPU-'+str(i)) for i in range(n)]
    def test_eight_gpus_keep_fixed_global32(self):
        b=p.resource_budget(self.inventory(8));self.assertEqual(b['world_size'],8)
        self.assertEqual(b['per_gpu_batch'],2);self.assertEqual(b['gradient_accumulation'],2)
        self.assertEqual(b['world_size']*b['per_gpu_batch']*b['gradient_accumulation'],32)
    def test_five_or_six_visible_uses_four_without_budget_drift(self):
        for n in (5,6,7):
            b=p.resource_budget(self.inventory(n));self.assertEqual(b['world_size'],4)
            self.assertEqual(b['gradient_accumulation'],4)
        for bad in ('0,1,2','1,2','0,0','0,9'):
            with self.assertRaises(ValueError):p.resource_budget(self.inventory(8),bad)
    def test_gpu_uuid_or_index_aliases_are_rejected(self):
        rows=self.inventory(2);rows[1]['uuid']=rows[0]['uuid']
        with self.assertRaises(ValueError):p.resource_budget(rows)
    def test_registered_formal_sample_budgets_exact(self):
        for key in ('world','stage2','stage3'):
            self.assertEqual(p.RECIPES[key]['samples'],p.RECIPES[key]['steps']*32)
        for key in ('development_world','development_joint'):
            self.assertEqual(p.RECIPES[key]['samples'],p.RECIPES[key]['steps']*8)
    def test_smoke_cannot_be_promoted_to_scientific_probe(self):
        r=dict(status='complete',phase='world',global_step=4,examples_seen=128,
            data_manifest_sha256='a',checkpoint_sha256='b',smoke_only=True,passed=True,source_hashes={'x':'y'},
            training_scope='world_only',policy_training_performed=False,scientific_passed=False,
            final=dict(passed=True,gradient_groups_nonzero=True,world=dict(passed=False),policy=None))
        p.validate_training_report(r,phase='world',steps=4,global_batch=32,manifest_sha='a',checkpoint_sha='b',smoke=True)
        with self.assertRaises(p.core.Incomplete):
            p.validate_training_report(r,phase='world',steps=4,global_batch=32,manifest_sha='a',checkpoint_sha='b')
        r['passed']=False
        with self.assertRaises(p.core.GateFailed):
            p.validate_training_report(r,phase='world',steps=4,global_batch=32,manifest_sha='a',checkpoint_sha='b',smoke=True)
    def test_wrong_data_or_partial_sample_budget_fails(self):
        r=dict(status='complete',phase='joint',global_step=200,examples_seen=1600,
            data_manifest_sha256='a',checkpoint_sha256='b',smoke_only=False,passed=True,source_hashes={'x':'y'},
            training_scope='joint',policy_training_performed=True,scientific_passed=True,
            final=dict(passed=True,gradient_groups_nonzero=True,world=dict(passed=True),policy=dict(passed=True)))
        for key,value in [('examples_seen',1599),('data_manifest_sha256','wrong'),('phase','world'),('source_hashes',{})]:
            bad=dict(r,**{key:value})
            with self.assertRaises(p.core.Incomplete):
                p.validate_training_report(bad,phase='joint',steps=200,global_batch=8,manifest_sha='a',checkpoint_sha='b')
        for key in ('world','policy'):
            bad=copy.deepcopy(r);bad['final'][key]['passed']=False
            with self.assertRaises(p.core.GateFailed):
                p.validate_training_report(bad,phase='joint',steps=200,global_batch=8,manifest_sha='a',checkpoint_sha='b')
    def test_episode_split_and_official_exclusion(self):
        rows=[dict(task='t',seed=1,scene={'partition':'train'}),dict(task='t',seed=2,scene={'partition':'validation'})]
        self.assertEqual(validate_rows(rows,{'t':[99]}),dict(train=1,validation=1))
        for bad in (rows+[rows[0]],rows[:1],[rows[0],dict(rows[1],seed=99)],
                    [rows[0],dict(rows[1],seed=1)]):
            with self.assertRaises(ValueError):validate_rows(bad,{'t':[99]})
    def test_dense_provenance_and_bytes_not_just_status(self):
        # Isolate canonical path resolution from this hash/provenance test;
        # Windows sandbox handles may deny realpath on just-created files.
        with tempfile.TemporaryDirectory() as d,patch.object(p,'identity',side_effect=lambda q:
                dict(path=str(Path(q).absolute()),sha256=sha256(q))):
            root=Path(d);blob=root/'observations.npz';commands=root/'commands.npz'
            times=np.array([0.,.2,.4]);ticks=150;start=np.arange(ticks)*.004;end=(np.arange(ticks)+1)*.004
            arrays=dict(rgb=np.zeros((3,3,160,160,3),np.uint8),rgb_valid_mask=np.ones((3,3,160,160),np.bool_),
                observation_sim_time=times,source_physics_time=times,received_sim_time=times+.012,
                dispatch_sim_time=np.array([.012,.2,.4]),policy_submitted=np.array([True,False,False]))
            np.savez(blob,**arrays)
            np.savez(commands,q_command=np.zeros((ticks,14)),velocity_command=np.zeros((ticks,14)),
                start_sim_time=start,end_sim_time=end,engine_start_time=start,engine_end_time=end,tick_index=np.arange(ticks))
            scene=dict(task='t',seed=1)
            m=dict(format='dynamicwam_continuous_trajectory',status='complete',frames=3,ticks=ticks,
                metadata=dict(task='t',seed=1,realtime_valid=True,origin_sim_time=0.,physical_end_time=.6),physics_dt=.004,control_period=.1,
                provenance=dict(accepted_record_sha256='a',controller_protocol_sha256='b',policy_config_sha256='c'),
                artifacts={name:dict(path=path.name,sha256=sha256(path),bytes=path.stat().st_size)
                    for name,path in [('observations',blob),('commands',commands)]})
            write_json(root/'manifest.json',m)
            write_json(root/'completion.json',dict(format='dynamicwam_continuous_trajectory_completion',status='complete',
                manifest=dict(path='manifest.json',sha256=sha256(root/'manifest.json'))))
            self.assertEqual(p.validate_dense(root,scene,'a','b','c')['ticks'],ticks)
            self.assertEqual(p.validate_dense(root,scene,'a','b','c')['supported_future_pairs'],[2,1])
            with self.assertRaises(p.core.IdentityViolation):p.validate_dense(root,scene,'wrong','b','c')
            def rewrite_observation(changes):
                np.savez(blob,**dict(arrays,**changes))
                m['artifacts']['observations']=dict(path=blob.name,sha256=sha256(blob),bytes=blob.stat().st_size)
                write_json(root/'manifest.json',m)
                write_json(root/'completion.json',dict(format='dynamicwam_continuous_trajectory_completion',status='complete',
                    manifest=dict(path='manifest.json',sha256=sha256(root/'manifest.json'))))
            rewrite_observation(dict(dispatch_sim_time=np.array([.32,.32,.4]),policy_submitted=np.array([True,True,False])))
            with self.assertRaises(p.core.Incomplete):p.validate_dense(root,scene,'a','b','c')
            rewrite_observation(dict(dispatch_sim_time=np.array([.012,.24,.4])))
            with self.assertRaises(p.core.Incomplete):p.validate_dense(root,scene,'a','b','c')
            blob.write_bytes(b'tampered')
            with self.assertRaises(p.core.IdentityViolation):p.validate_dense(root,scene,'a','b','c')
    def test_profile_requires_both_proposals_with_real_elapsed_prefix(self):
        report=dict(status='complete',passed=True,ablation='none',fast_samples_ms=[5.]*30,
            graph_samples_ms=[150.]*30,eager_samples_ms=[300.]*30,native_action_max_abs_error=0.,
            gate=dict(includes_candidate_controller=True,includes_action_device_to_host=True,
                includes_rgb_preprocessing=True,includes_final_corrected_candidate_forecast=True,
                candidate_controller_calls_per_fast_query=2),
            proposal_provenance=dict(known_executed_prefix_ticks=41,actual_future_commands_used=False),
            observation_fixture=dict(camera_and_dispatch_lag_s=.164,includes_elapsed_prefix_helper_cost=True))
        self.assertLess(p.validate_predictive_profile(report)['amortized_ratio'],.85)
        for key,value in [('candidate_controller_calls_per_fast_query',1),('includes_final_corrected_candidate_forecast',False)]:
            bad=copy.deepcopy(report);bad['gate'][key]=value
            with self.assertRaises(p.core.Incomplete):p.validate_predictive_profile(bad)
        bad=copy.deepcopy(report);bad['proposal_provenance']['known_executed_prefix_ticks']=0
        with self.assertRaises(p.core.Incomplete):p.validate_predictive_profile(bad)
    def test_profile_receipt_preserves_phase_name_after_source_verification(self):
        with tempfile.TemporaryDirectory() as d:
            obj=object.__new__(p.Pipeline);obj.run=Path(d);obj.source=ROOT
            obj.config=dict(sim_python='python',packet='packet.npz',domino_config='configs/absolute_motion_v2.yaml',
                inputs={'packet':{'sha256':'h'}})
            obj.base={'sha256':'h'};obj.protocol_sha='h';obj.devices=[{'uuid':'fake'}]
            source_name='src/dynamicwam/models/predictive_feedback.py'
            obj.frozen=dict(files={'scripts/profile_predictive_feedback.py':'h',source_name:'h'})
            report=dict(head_checkpoint_sha256='h',base_checkpoint_sha256='h',controller_protocol_sha256='h',
                policy_config_sha256='h',packet_sha256='h',profile_source_sha256='h',
                source_hashes={source_name:'h'},loaded_dynamicwam_modules={'dynamicwam.models.predictive_feedback':
                    dict(path=str(ROOT/source_name),sha256='h')})
            commands=[];receipts=[]
            def command(phase,argv,*args,**kwargs):
                commands.append(phase);write_json(Path(argv[argv.index('--output')+1]),report)
            obj.command=command;obj.complete=lambda phase,artifacts,evidence:receipts.append((phase,artifacts))
            # This test isolates phase routing; Windows sandbox can deny realpath
            # even for readable source files. Production source checks stay strict.
            with patch.object(p,'sha256',return_value='h'),patch.object(p,'validate_predictive_profile',return_value={'passed':True}),\
                    patch.object(Path,'resolve',lambda path,*args,**kwargs:path.absolute()):
                obj.profile('development/loaded_profile','checkpoint.pt')
            self.assertEqual(commands,['development/loaded_profile'])
            self.assertEqual([phase for phase,_ in receipts],['development/loaded_profile'])
            self.assertEqual(receipts[0][1]['report'],obj.run/'development/loaded_profile/report.json')
    def test_collection_uses_buffered_worker_and_explicit_dense_flag(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);obj=object.__new__(p.Pipeline);obj.run=run;obj.source=ROOT
            obj.config=dict(sim_python='python',domino_config='configs/absolute_motion_v2.yaml')
            obj.protocol=p.core.controller_protocol();scene=dict(task='t',seed=1,accepted_record='accepted/t/1.pkl')
            cfg=run/'worker.json';write_json(cfg,dict(mode='graph_zero',record_trajectory=True))
            a=obj.episode_command(scene,run/'output',policy=cfg)
            self.assertIn('--record-trajectory',a)
            self.assertEqual(a[a.index('--policy-factory')+1],'dynamicwam.runtime.buffered_world_context_worker:create_worker_policy')
            write_json(cfg,dict(mode='predictive',record_trajectory=False))
            a=obj.episode_command(scene,run/'output',policy=cfg)
            self.assertNotIn('--record-trajectory',a)
            self.assertEqual(a[a.index('--policy-factory')+1],'dynamicwam.runtime.predictive_feedback_worker:create_worker_policy')
            write_json(cfg,dict(mode='predictive',record_trajectory=True))
            a=obj.episode_command(scene,run/'output',policy=cfg)
            self.assertIn('--record-trajectory',a)
            self.assertEqual(a[a.index('--policy-factory')+1],'dynamicwam.runtime.buffered_predictive_feedback_worker:create_worker_policy')
    def test_predictive_correction_must_reach_valid_installed_row(self):
        report=dict(policy_config={'mode':'predictive'},report=dict(physics_time=.3,control_period_s=.1,
            requests=[dict(request_id=1,action_origin_sim_time=0.,expired=False,actual_install_sim_time=.2,
                clipped_prefix=2,metadata=dict(feedback_rows=4,feedback_row_l2=[.1]*4,feedback_row_l2_units='normalized_action'))],
            control_intervals=[dict(start_sim_time=.2,end_sim_time=.3,hold=False,request_id=1,returned_row_index=2)]))
        self.assertEqual(p.validate_predictive_consumption([report])['consumed_nonzero_intervals'],1)
        bad=copy.deepcopy(report);bad['report']['requests'][0]['expired']=True
        with self.assertRaises(p.core.Incomplete):p.validate_predictive_consumption([bad])
        bad=copy.deepcopy(report);bad['report']['control_intervals'][0]['hold']=True
        with self.assertRaises(p.core.GateFailed):p.validate_predictive_consumption([bad])
        bad=copy.deepcopy(report);bad['policy_config']['mode']='world_context'
        with self.assertRaises(p.core.IdentityViolation):p.validate_predictive_consumption([bad])
    def test_zero_vs_zero_canary_is_uninformative(self):
        rows=[dict(task='t',seed=i,success=False) for i in range(6)]
        with self.assertRaises(p.core.GateFailed):p.core.validate_canary(rows,rows)
    def test_mechanism_subset_is_fixed_and_does_not_reduce_main_cohort(self):
        rows=[dict(task=f't{t:02d}',seed=s,scene_id=f'{t}/{s}') for t in range(35) for s in range(20)]
        subset=p.mechanism_scenes(list(reversed(rows)))
        self.assertEqual(len(subset),175)
        self.assertTrue(all(r['seed']<5 for r in subset))
        all_keys={(r['task'],r['seed']) for r in rows};sub_keys={(r['task'],r['seed']) for r in subset}
        outcomes={arm:{k:False for k in all_keys} for arm in p.ARMS}
        outcomes[p.MECHANISM_ARM]={k:False for k in sub_keys}
        main,mechanism=paired_keys(outcomes)
        self.assertEqual(len(main),700);self.assertEqual(len(mechanism),175)
        outcomes[p.MECHANISM_ARM].pop(next(iter(sub_keys)))
        main,mechanism=paired_keys(outcomes)
        self.assertEqual(len(main),700);self.assertEqual(len(mechanism),174)
        with self.assertRaises(p.core.IdentityViolation):p.mechanism_scenes(rows[:-1])
    def test_mechanism_policy_dispatches_actual_current_only_same_checkpoint(self):
        obj=object.__new__(p.Pipeline);obj.source=ROOT
        obj.config=dict(domino_config='configs/absolute_motion_v2.yaml');obj.base={'sha256':'base'};obj.protocol_sha='protocol'
        with patch.object(p,'sha256',return_value='identical_checkpoint'):
            full=obj.policy('predictive',checkpoint='final.pt')
            ablation=obj.policy(p.MECHANISM_ARM,checkpoint='final.pt')
        self.assertEqual(full['feedback_checkpoint_sha256'],ablation['feedback_checkpoint_sha256'])
        self.assertEqual(full['mode'],ablation['mode']);self.assertEqual(ablation['mode'],'predictive')
        self.assertEqual(full['ablation'],'none');self.assertEqual(ablation['ablation'],'current_only')
        self.assertEqual({k:v for k,v in full.items() if k!='ablation'},
            {k:v for k,v in ablation.items() if k!='ablation'})


if __name__=='__main__':unittest.main(verbosity=2)
