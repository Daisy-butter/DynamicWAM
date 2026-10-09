#!/usr/bin/env python3
"""One-launch, immutable, fail-stop action-conditioned latent feedback experiment."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_world_context_rt_pipeline as core
from make_world_context_rt_manifest import identity,read_json,sha256,write_json,object_hash

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
FORMAT='dynamicwam_predictive_feedback_pipeline'
VERSION=1
GLOBAL_BATCH=32
ARMS=('official_eager','graph_zero','predictive')
MECHANISM_ARM='predictive_current_only'
MECHANISM_SEEDS_PER_TASK=5
RECIPES=dict(world=dict(steps=1000,samples=32000,lr=2e-4),
    stage2=dict(steps=3150,samples=100800,lr=2e-5),stage3=dict(steps=900,samples=28800,lr=5e-6),
    development_world=dict(steps=200,samples=1600,lr=2e-4),
    development_joint=dict(steps=200,samples=1600,lr=2e-5))
# This process reuses the audited infrastructure implementation from the SAME
# frozen source; no import points at a mutable repository or an older run.
core.SOURCE_DOCUMENTS=tuple(dict.fromkeys((*core.SOURCE_DOCUMENTS,'predictive_feedback_pipeline.md',
    'jepa_predictive_feedback_design_20261009.md')))


def mechanism_scenes(formal):
    """Register the first five numeric seeds per task before any outcomes exist."""
    by_task={}
    for scene in formal:by_task.setdefault(scene['task'],[]).append(scene)
    if len(by_task)!=35 or any(len(rows)!=20 or len({r['seed'] for r in rows})!=20 for rows in by_task.values()):
        raise core.IdentityViolation('mechanism subset requires the exact 35 by 20 formal manifest')
    return [scene for task in sorted(by_task) for scene in sorted(by_task[task],key=lambda r:r['seed'])[:MECHANISM_SEEDS_PER_TASK]]


def resource_budget(inventory,requested='auto'):
    by_index={r['index']:r for r in inventory}
    if len(by_index)!=len(inventory) or len({r['uuid'] for r in inventory})!=len(inventory):
        raise ValueError('ambiguous GPU inventory')
    if requested=='auto':
        candidates=sorted(inventory,key=lambda x:int(x['index']))[:8]
        ranks=max((n for n in (1,2,4,8) if n<=len(candidates)),default=0)
        chosen=candidates[:ranks]
    else:
        keys=requested.split(',')
        if len(keys)!=len(set(keys)) or any(k not in by_index for k in keys):raise ValueError('unknown/duplicate GPU')
        chosen=[by_index[k] for k in keys];ranks=len(chosen)
    if ranks not in (1,2,4,8) or not any(x['index']=='0' for x in chosen):
        raise ValueError('global32 supports 1/2/4/8 ranks and requires qualified simulation GPU0')
    return dict(devices=chosen,world_size=ranks,per_gpu_batch=2,
        gradient_accumulation=GLOBAL_BATCH//(2*ranks),global_batch=GLOBAL_BATCH)


def validate_training_report(report,*,phase,steps,global_batch,manifest_sha,checkpoint_sha,smoke=False):
    if (report.get('status')!='complete' or report.get('phase')!=phase or report.get('global_step')!=steps or
        report.get('examples_seen')!=steps*global_batch or report.get('data_manifest_sha256')!=manifest_sha or
        report.get('checkpoint_sha256')!=checkpoint_sha or report.get('smoke_only',False)!=smoke):
        raise core.Incomplete('training receipt does not match the fixed source/data/sample budget')
    if report.get('passed') is not True:raise core.GateFailed('registered final-step training/probe gate failed')
    if not report.get('source_hashes'):raise core.Incomplete('training source identity absent')
    final=report.get('final',{})
    if (report.get('training_scope')!=('world_only' if phase=='world' else 'joint') or
        report.get('policy_training_performed')!=(phase=='joint') or
        final.get('passed') is not True or final.get('gradient_groups_nonzero') is not True):
        raise core.Incomplete('training phase, final receipt or gradient evidence differs')
    if not smoke and (report.get('scientific_passed') is not True or
        final.get('world',{}).get('passed') is not True or
        (phase=='joint' and final.get('policy',{}).get('passed') is not True)):
        raise core.GateFailed('world/policy heldout mechanism gates did not all pass')
    return dict(phase=phase,steps=steps,global_batch=global_batch,examples_seen=steps*global_batch,
        scope='execution only' if smoke else 'final-step heldout; not closed-loop success')


def validate_dense(directory,scene,accepted_sha,protocol_sha,policy_sha):
    import numpy as np
    from dynamicwam.runtime.domino_rt.proposal_path import forecast_horizons
    directory=Path(directory);completion=read_json(directory/'completion.json')
    if completion.get('format')!='dynamicwam_continuous_trajectory_completion' or completion.get('status')!='complete':
        raise core.Incomplete('dense trajectory has no successful commit marker')
    manifest=directory/completion['manifest']['path']
    if sha256(manifest)!=completion['manifest']['sha256']:raise core.IdentityViolation('trajectory manifest changed')
    value=read_json(manifest);m=value['metadata'];p=value['provenance']
    if (value.get('status')!='complete' or m['task']!=scene['task'] or m['seed']!=scene['seed'] or
        m.get('realtime_valid') is not True or p.get('accepted_record_sha256')!=accepted_sha or
        p.get('controller_protocol_sha256')!=protocol_sha or p.get('policy_config_sha256')!=policy_sha or
        value.get('physics_dt')!=.004 or value.get('control_period')!=.1):
        raise core.IdentityViolation('dense trajectory and replay contract differ')
    if value.get('frames',0)<3 or value.get('ticks',0)<1:
        raise core.Incomplete('no useful actual future image/command trajectory')
    for artifact in value['artifacts'].values():
        path=directory/artifact['path']
        if sha256(path)!=artifact['sha256'] or path.stat().st_size!=artifact['bytes']:
            raise core.IdentityViolation('dense numerical artifact changed')
    with np.load(directory/value['artifacts']['observations']['path'],allow_pickle=False) as obs,\
         np.load(directory/value['artifacts']['commands']['path'],allow_pickle=False) as cmd:
        f,t=value['frames'],value['ticks'];times=obs['observation_sim_time']
        start,end=cmd['start_sim_time'],cmd['end_sim_time']
        if (times.shape!=(f,) or np.any(np.diff(times)<=0) or not np.isfinite(times).all() or times[0]<0 or
            obs['rgb'].shape!=(f,3,160,160,3) or obs['rgb'].dtype!=np.uint8 or
            obs['rgb_valid_mask'].shape!=(f,3,160,160) or obs['rgb_valid_mask'].dtype!=np.bool_ or
            start.shape!=(t,) or end.shape!=(t,) or not np.array_equal(cmd['tick_index'],np.arange(t)) or
            not np.allclose(start,np.arange(t)*.004,atol=1e-8,rtol=0) or
            not np.allclose(end,(np.arange(t)+1)*.004,atol=1e-8,rtol=0) or times[-1]>end[-1]+1e-8):
            raise core.Incomplete('dense numerical images/timestamps do not cover the complete physical grid')
        origin=m['origin_sim_time']
        for key in ('q_command','velocity_command'):
            if cmd[key].shape!=(t,14) or not np.isfinite(cmd[key]).all():raise core.Incomplete('missing finite applied command path')
        if (not np.allclose(obs['source_physics_time'],times,atol=1e-7,rtol=1e-7) or
            np.any(obs['received_sim_time']<times-1e-7) or
            not np.allclose(cmd['engine_start_time']-origin,start,atol=.0004,rtol=0) or
            not np.allclose(cmd['engine_end_time']-origin,end,atol=.0004,rtol=0) or
            abs(m['physical_end_time']-end[-1])>1e-8):
            raise core.Incomplete('recorded camera/command times differ from actual physics')
        if not {'dispatch_sim_time','policy_submitted'}<=set(obs.files):
            raise core.Incomplete('dense recording lacks actual dispatch-clock query identity')
        dispatch=obs['dispatch_sim_time'];submitted=obs['policy_submitted']
        if (dispatch.shape!=(f,) or submitted.shape!=(f,) or submitted.dtype!=np.bool_ or
            not np.isfinite(dispatch).all() or np.any(dispatch<times-1e-8) or np.any(dispatch-times>2.) or
            np.any(np.abs(dispatch[~submitted]-times[~submitted])>1e-8) or
            np.any(dispatch[submitted]<obs['received_sim_time'][submitted]-1e-8) or
            np.any(dispatch[submitted]>end[-1]+1e-8) or
            np.any(np.abs(dispatch/.004-np.rint(dispatch/.004))>1e-6) or not submitted.any()):
            raise core.Incomplete('dense dispatch timing is missing, unqueried-retimestamped, or outside executed physics')
        supported=[0,0];queried_supported=[0,0]
        for stamp,send,queried in zip(times,dispatch,submitted):
            for h,(desired,(left,right)) in enumerate(zip(forecast_horizons(stamp,send),((-.1,.1),(-.1,.2)))):
                valid=bool(np.any((times-stamp>desired+left+1e-8)&(times-stamp<=desired+right+1e-8)))
                supported[h]+=valid;queried_supported[h]+=bool(queried and valid)
        if not all(supported):raise core.Incomplete('dense episode has no observed future support at both horizons')
    return dict(frames=value['frames'],ticks=value['ticks'],supported_future_pairs=supported,
        queried_future_pairs=queried_supported,actual_queries=int(submitted.sum()),
        future_clock='shared forecast_horizons(source,actual_dispatch); unqueried dispatch defaults to source',
        completion=identity(directory/'completion.json'))


def validate_predictive_profile(report):
    evidence=core.validate_profile(report)
    gate=report.get('gate',{});proposal=report.get('proposal_provenance',{})
    fixture=report.get('observation_fixture',{})
    if (report.get('status')!='complete' or report.get('ablation')!='none' or
        any(gate.get(k) is not True for k in ('includes_candidate_controller','includes_action_device_to_host',
            'includes_rgb_preprocessing','includes_final_corrected_candidate_forecast')) or
        gate.get('candidate_controller_calls_per_fast_query')!=2 or
        not core.finite_number(fixture.get('camera_and_dispatch_lag_s')) or fixture['camera_and_dispatch_lag_s']<=0 or
        fixture.get('includes_elapsed_prefix_helper_cost') is not True or
        proposal.get('known_executed_prefix_ticks',0)<=0 or proposal.get('actual_future_commands_used') is not False):
        raise core.Incomplete('profile does not cover the full delayed-dispatch two-proposal path')
    evidence['dispatch_fixture']=fixture
    return evidence


def validate_predictive_consumption(reports):
    generated=consumed=0;duration_total=0.
    for report in reports:
        if report.get('policy_config',{}).get('mode')!='predictive':
            raise core.IdentityViolation('consumption gate requires the actual predictive arm')
        value=report['report'];requests={}
        for request in value['requests']:
            key=request['request_id']
            if key in requests:raise core.Incomplete('duplicate predictive request identity')
            requests[key]=request;meta=request.get('metadata',{});count=meta.get('feedback_rows',0)
            if type(count)is not int or not 0<=count<=4:raise core.Incomplete('invalid correction window')
            if not count:continue
            norms=meta.get('feedback_row_l2')
            if (meta.get('feedback_row_l2_units')!='normalized_action' or not isinstance(norms,list) or
                len(norms)<count or any(not core.finite_number(v,nonnegative=True) for v in norms[:count])):
                raise core.Incomplete('missing finite correction-target telemetry')
            generated+=sum(v>0 for v in norms[:count])
        for interval in value['control_intervals']:
            duration=max(0.,min(interval['end_sim_time'],value['physics_time'])-interval['start_sim_time'])
            if interval['hold'] or duration<=0:continue
            request=requests.get(interval.get('request_id'))
            if request is None:raise core.Incomplete('consumed target lacks its generating request')
            row=round(interval['start_sim_time']/value['control_period_s'])-round(request['action_origin_sim_time']/value['control_period_s'])
            if 'returned_row_index' in interval and interval['returned_row_index']!=row:
                raise core.Incomplete('consumed target row identity differs')
            meta=request.get('metadata',{})
            if not 0<=row<meta.get('feedback_rows',0):continue
            if (request.get('expired') is not False or
                request.get('actual_install_sim_time',math.inf)>interval['start_sim_time']+1e-7 or row<request['clipped_prefix']):
                raise core.Incomplete('feedback target was attributed to an expired/uninstalled row')
            if meta['feedback_row_l2'][row]>0:consumed+=1;duration_total+=duration
    if not generated or not consumed:
        raise core.GateFailed('no nonzero predictive correction target was generated and consumed before expiry')
    return dict(generated_nonzero_rows=generated,consumed_nonzero_intervals=consumed,
        consumed_nonzero_slot_equivalents=duration_total/.1,
        scope='nonzero target delivery, not proof of counterfactual actuator motion after clipping')


class Pipeline(core.Pipeline):
    def __init__(self,root):
        super().__init__(root)
        if self.config.get('format')!=FORMAT or self.config.get('recipes')!=RECIPES:
            raise core.IdentityViolation('predictive experiment recipe/schema differs')
        self.status.update(format=FORMAT,version=VERSION)
        if self.budget['global_batch']!=GLOBAL_BATCH:raise core.IdentityViolation('global batch changed')

    def complete(self,phase_id,artifacts,evidence):
        value=super().complete(phase_id,artifacts,evidence)
        # The common phase receipt schema is deliberately retained. Its source,
        # run-config and data identities bind it to this new experiment.
        return value

    def policy(self,arm,scene=None,checkpoint=None,collection=None):
        value=super().policy('predictive' if arm==MECHANISM_ARM else arm,scene,checkpoint,collection)
        value.update(feedback_checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
            ablation='current_only' if arm==MECHANISM_ARM else 'none',record_trajectory=collection is not None)
        return value

    def closed_loop_evidence(self,zero,learned):
        evidence=core.validate_canary(zero,learned)
        evidence['feedback_consumption']=validate_predictive_consumption(
            [read_json(Path(row['output'])/'summary.json') for row in learned])
        return evidence

    def episode_command(self,scene,output,*,prepare=False,policy=None,mode='async'):
        argv=super().episode_command(scene,output,prepare=prepare,policy=policy,mode=mode)
        if not prepare:
            cfg=read_json(policy)
            factory=('dynamicwam.runtime.buffered_predictive_feedback_worker:create_worker_policy'
                if cfg['mode']=='predictive' and cfg.get('record_trajectory')
                else 'dynamicwam.runtime.predictive_feedback_worker:create_worker_policy' if cfg['mode']=='predictive'
                else 'dynamicwam.runtime.buffered_world_context_worker:create_worker_policy' if cfg.get('record_trajectory')
                else 'dynamicwam.runtime.world_context_worker:create_worker_policy')
            argv[argv.index('--policy-factory')+1]=factory
            if cfg.get('record_trajectory'):argv.append('--record-trajectory')
        return argv

    def prepare_scenes(self,scenes,*,allow_missing=False):
        # Reuse accepted expert motion bytes, not a previous policy outcome.
        previous=self.config.get('accepted_run')
        if previous:
            registered=self.config['accepted_pairing']
            if identity(Path(previous)/'paired_scene_manifest.json')!=registered:
                raise core.IdentityViolation('previous accepted-scene manifest changed')
            old=read_json(registered['path'])
            for scene in scenes:
                key=scene['scene_id'];entry=old.get('scenes',{}).get(key)
                if key in self.accepted or entry is None:continue
                original=Path(entry['accepted_record']['path'])
                if sha256(original)!=entry['accepted_record']['sha256']:raise core.IdentityViolation('accepted scene bytes changed')
                destination=self.run/scene['accepted_record'];destination.parent.mkdir(parents=True,exist_ok=True)
                with original.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
                value=dict(scene=scene,accepted_record=identity(destination),reused_from=entry['accepted_record'])
                self.accepted[key]=value
                self.complete(f'accepted_reuse/{key}',dict(accepted_record=destination),value)
        return super().prepare_scenes(scenes,allow_missing=allow_missing)

    def episodes(self,name,scenes,arm,checkpoint=None,collect=False,formal=False):
        rows=super().episodes(name,scenes,arm,checkpoint,collect,formal)
        if collect:
            for row in rows:
                evidence=validate_dense(Path(row['output'])/'trajectory',row['scene'],row['accepted_sha256'],
                    self.protocol_sha,sha256(row['policy']))
                self.complete(f'dense/{name}/{row["task"]}/{row["seed"]}',
                    dict(completion=Path(row['output'])/'trajectory/completion.json'),evidence)
        return rows

    def dataset(self,name,rows):
        rowfile=self.run/'data_inputs'/f'{name}.json';write_json(rowfile,rows,exclusive=True)
        out=self.run/name
        self.command(name,[self.config['python'],str(self.source/'scripts/build_predictive_feedback_data.py'),
            '--episodes-list',str(rowfile),'--reference-checkpoint',self.config['init_plan_feedback'],
            '--official-seed-manifest',self.config['official_seed_manifest'],
            '--controller-protocol',str(self.run/'controller_protocol.json'),'--output-dir',str(out),'--require-policy'])
        report=read_json(out/'report.json');completion=read_json(out/'completion.json')
        if report.get('status')!='complete' or completion.get('status')!='complete':raise core.Incomplete('dataset build incomplete')
        if any(report['evidence'][p].get('policy_samples',0)<=0 for p in ('train','validation')):
            raise core.Incomplete('empty causal teacher training or heldout partition')
        self.complete(name,dict(dataset=out/'dataset.json',report=out/'report.json',completion=out/'completion.json'),report['evidence'])
        return out/'dataset.json'

    def train_predictive(self,name,manifest,recipe,*,phase,initial=None,smoke=False,development=False):
        budget=dict(world_size=1,per_gpu_batch=8,gradient_accumulation=1,global_batch=8) if development else self.budget
        devices=self.devices[:1] if development else self.devices
        steps=4 if smoke else recipe['steps'];out=self.run/name
        argv=[self.config['python']]
        if budget['world_size']>1:
            argv+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node',str(budget['world_size'])]
        argv+=[str(self.source/'scripts/train_predictive_feedback.py'),'--manifest',str(manifest),
            '--init-plan-feedback',self.config['init_plan_feedback'],'--run-dir',str(out),'--phase',phase,
            '--steps',str(steps),'--batch-size',str(budget['per_gpu_batch']),
            '--gradient-accumulation',str(budget['gradient_accumulation']),'--lr',str(recipe['lr']),
            '--seed',str(self.config['seed'])]
        if initial:argv+=['--init-feedback',str(initial)]
        if smoke:argv.append('--smoke-only')
        self.command(name,argv,devices,timeout=172800)
        checkpoint=out/'checkpoint.pt';report=read_json(out/'report.json')
        evidence=validate_training_report(report,phase=phase,steps=steps,global_batch=budget['global_batch'],
            manifest_sha=sha256(manifest),checkpoint_sha=sha256(checkpoint),smoke=smoke)
        for path,digest in report['source_hashes'].items():
            if self.frozen['files'].get(path)!=digest:raise core.IdentityViolation('trainer imported a different source')
        self.complete(name,dict(report=out/'report.json',checkpoint=checkpoint,metrics=out/'metrics.jsonl'),evidence)
        return checkpoint

    def profile(self,name,checkpoint):
        out=self.run/name;out.mkdir(parents=True,exist_ok=False);cfg=out/'worker.json'
        write_json(cfg,self.policy('predictive',checkpoint=checkpoint),exclusive=True)
        self.command(name,[self.config['sim_python'],str(self.source/'scripts/profile_predictive_feedback.py'),
            '--policy-config',str(cfg),'--packet',self.config['packet'],'--output',str(out/'report.json')],
            self.devices[:1],timeout=3600)
        report=read_json(out/'report.json');evidence=validate_predictive_profile(report)
        if (report.get('head_checkpoint_sha256')!=sha256(checkpoint) or report.get('base_checkpoint_sha256')!=self.base['sha256'] or
            report.get('controller_protocol_sha256')!=self.protocol_sha or report.get('policy_config_sha256')!=sha256(cfg) or
            report.get('packet_sha256')!=self.config['inputs']['packet']['sha256']):
            raise core.IdentityViolation('loaded predictive latency profile identities differ')
        if (report.get('profile_source_sha256')!=self.frozen['files']['scripts/profile_predictive_feedback.py'] or
            not report.get('source_hashes') or not report.get('loaded_dynamicwam_modules')):
            raise core.IdentityViolation('loaded profile source provenance is absent or changed')
        for source_name,digest in report['source_hashes'].items():
            if self.frozen['files'].get(source_name)!=digest:raise core.IdentityViolation('profile used a different source')
        for value in report['loaded_dynamicwam_modules'].values():
            path=Path(value['path']).resolve(strict=True)
            if not path.is_relative_to(self.source) or self.frozen['files'].get(path.relative_to(self.source).as_posix())!=value['sha256']:
                raise core.IdentityViolation('profile imported dynamicwam outside the frozen source')
        self.complete(name,dict(report=out/'report.json',policy=cfg),evidence)

    def run_all(self):
        self.update(status='running')
        checks=('scripts/check_world_context_worker.py','scripts/check_trajectory_recorder.py',
            'scripts/check_predictive_proposal.py','scripts/check_predictive_profile.py','scripts/check_buffered_predictive_collection.py',
            'tests/test_predictive_feedback_model.py','scripts/check_predictive_feedback_training.py',
            'scripts/check_predictive_feedback_pipeline.py','scripts/check_scene_snapshot_cpu.py',
            'tests/test_continuous_domino.py')
        if self.config['stop_after']=='dense_clock':
            checks=('scripts/check_world_context_worker.py','scripts/check_trajectory_recorder.py',
                'scripts/check_predictive_proposal.py','scripts/check_predictive_profile.py','scripts/check_buffered_predictive_collection.py',
                'scripts/check_predictive_feedback_pipeline.py','scripts/check_scene_snapshot_cpu.py','tests/test_continuous_domino.py')
        for script in checks:
            self.command('cpu/'+script,[self.config['python'],str(self.source/script)])
            self.complete('cpu/'+script,dict(log=self.run/'phases/cpu'/script/'stdout.log'),dict(exit_code=0))
        train_dev=self.select('stage2',True);val_dev=self.select('validation',True);first=train_dev[0]
        self.replica_check(first)
        first_rows=self.episodes('development/dense_clock',[first],'graph_zero',collect=True)
        if self.config['stop_after']=='dense_clock':
            self.update(status='preflight_complete',finished_at=time.time(),
                scope='one accepted scene; no training or quality claim; new formal run still required')
            return
        rest=[s for s in train_dev+val_dev if s['scene_id']!=first['scene_id']]
        development=first_rows+self.episodes('development/zero',rest,'graph_zero',collect=True)
        labels=self.relabel('development/labels',development)
        data=self.dataset('development/data',labels)
        if self.config['stop_after']=='development_data':
            self.update(status='data_complete',finished_at=time.time(),dataset=identity(data))
            return
        self.train_predictive('smoke',data,RECIPES['world'],phase='world',smoke=True)
        world=self.train_predictive('development/world_probe',data,RECIPES['development_world'],phase='world',development=True)
        head=self.train_predictive('development/policy_probe',data,RECIPES['development_joint'],phase='joint',initial=world,development=True)
        self.profile('development/loaded_profile',head)
        zero=[r for r in development if r['scene']['role']=='stage2']
        learned=self.episodes('development/learned',train_dev,'predictive',head)
        self.complete('development/closed_loop',dict(learned=self.run/'development/learned/aggregate.json'),
            self.closed_loop_evidence(zero,learned))
        previous={(r['task'],r['seed']) for r in development}
        remaining=[s for s in self.select('stage2')+self.select('validation') if (s['task'],s['seed']) not in previous]
        extra=self.episodes('stage2_collection',remaining,'graph_zero',collect=True)
        labels2=labels+self.relabel('stage2_labels',extra);data2=self.dataset('stage2_data',labels2)
        # Formal world training starts from the registered v1 initialization,
        # rather than quietly giving the selected development model extra steps.
        world2=self.train_predictive('world_pretrain',data2,RECIPES['world'],phase='world')
        stage2=self.train_predictive('stage2',data2,RECIPES['stage2'],phase='joint',initial=world2)
        self.profile('stage2_profile',stage2)
        learned=self.episodes('development/stage2',train_dev,'predictive',stage2)
        self.complete('stage2_closed_loop',dict(learned=self.run/'development/stage2/aggregate.json'),self.closed_loop_evidence(zero,learned))
        new=self.episodes('stage3_collection',self.select('stage3'),'predictive',stage2,collect=True)
        labels3=self.relabel('stage3_labels',new);data3=self.dataset('stage3_data',labels2+labels3)
        stage3=self.train_predictive('stage3',data3,RECIPES['stage3'],phase='joint',initial=stage2)
        self.profile('stage3_profile',stage3)
        learned=self.episodes('development/stage3',train_dev,'predictive',stage3)
        self.complete('stage3_closed_loop',dict(learned=self.run/'development/stage3/aggregate.json'),self.closed_loop_evidence(zero,learned))
        formal=self.select('formal')
        for arm in ARMS:self.episodes('eval/'+arm,formal,arm,stage3 if arm=='predictive' else None,formal=True)
        self.episodes('eval/'+MECHANISM_ARM,mechanism_scenes(formal),MECHANISM_ARM,stage3,formal=True)
        self.command('analysis',[self.config['python'],str(self.source/'scripts/analyze_predictive_feedback.py'),
            '--run-root',str(self.run),'--output',str(self.run/'analysis.json')],allow_failure=True)
        report=read_json(self.run/'analysis.json')
        if report.get('status') not in ('complete','incomplete'):raise core.Incomplete('formal analysis absent')
        self.complete('analysis',dict(report=self.run/'analysis.json',markdown=self.run/'analysis.md'),
            dict(paired_episodes=report['paired_episodes'],status=report['status']))
        self.update(status='complete' if report['status']=='complete' else 'evaluated_incomplete',finished_at=time.time())


def freeze(args):
    core.resource_budget=resource_budget
    run=core.freeze(args)
    config=read_json(run/'run_config.json');config.update(format=FORMAT,version=VERSION,recipes=RECIPES,
        sample_budgets=RECIPES,seed=20261009)
    if args.accepted_run:
        external=read_json(args.accepted_run/'external_source_manifest.json')
        if external!=read_json(run/'external_source_manifest.json'):raise core.IdentityViolation('old accepted source differs')
        config['accepted_pairing']=identity(args.accepted_run/'paired_scene_manifest.json')
    scenes=read_json(run/'scene_manifest.json')
    subset=mechanism_scenes([s for s in scenes['episodes'] if s['role']=='formal'])
    scenes.update(arms=list(ARMS),mechanism_arm=dict(name=MECHANISM_ARM,mode='predictive',ablation='current_only',
        checkpoint='same final stage3 checkpoint as predictive',seeds_per_task=MECHANISM_SEEDS_PER_TASK,
        selection='first_numeric_seeds_per_task',scene_ids=[s['scene_id'] for s in subset],requested_episodes=len(subset)))
    write_json(run/'scene_manifest.json',scenes);write_json(run/'run_config.json',config)
    write_json(run/'status.json',dict(format=FORMAT,version=VERSION,status='prepared',resources=config['resources']))
    return run


def launch_once(run,python,detach):
    write_json(run/'launch.json',dict(created_at=time.time(),detached=detach),exclusive=True)
    argv=[python,str(run/'source/scripts/run_predictive_feedback_pipeline.py'),'--run-root',str(run),'--execute']
    tmp=run/'tmp';tmp.mkdir(exist_ok=True);env=os.environ.copy()
    env.update(TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),PYTHONDONTWRITEBYTECODE='1')
    if not detach:return subprocess.call(argv,env=env)
    if os.name=='nt':raise ValueError('autonomous deployment requires Linux')
    with (run/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,cwd=run/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    write_json(run/'driver_pid.json',dict(pid=child.pid,argv=argv),exclusive=True)
    print(json.dumps(dict(run_root=str(run),pid=child.pid)));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--project-root',type=Path,default=ROOT)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true')
    modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--launch-prepared',action='store_true')
    p.add_argument('--detach',action='store_true');p.add_argument('--gpus',default='auto')
    p.add_argument('--stop-after',choices=['dense_clock','development_data','all'],default='all')
    p.add_argument('--simulation-concurrency',type=int,choices=[1],default=1)
    p.add_argument('--python',default=sys.executable);p.add_argument('--sim-python',default=sys.executable)
    p.add_argument('--domino-config',default='configs/absolute_motion_v2.yaml')
    for key in ('episodes','raw-root','official-seed-manifest','base-checkpoint','init-plan-feedback','packet','accepted-run'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--partition-seed',type=int,default=20260928);p.add_argument('--validation-fraction',type=float,default=.1)
    a=p.parse_args()
    if a.launch_prepared:
        run=a.run_root.resolve(strict=True)
        if read_json(run/'status.json').get('status')!='prepared':raise RuntimeError('only an unstarted prepared run can launch')
        return launch_once(run,read_json(run/'run_config.json')['python'],a.detach)
    if not a.execute:
        for key in ('episodes','raw_root','official_seed_manifest','base_checkpoint','init_plan_feedback','packet'):
            if getattr(a,key) is None:p.error('--'+key.replace('_','-')+' is required')
        run=freeze(a)
        if a.prepare_only:print(json.dumps(dict(run_root=str(run),status='prepared')));return 0
        return launch_once(run,a.python,a.detach)
    run=a.run_root.resolve(strict=True)
    if ROOT.resolve()!=run/'source' or read_json(run/'status.json').get('status')!='prepared':
        raise RuntimeError('execute exactly once, from the prepared source snapshot')
    import fcntl
    lock=(run/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pipeline=None
    try:
        pipeline=Pipeline(run);pipeline.run_all()
    except BaseException as exc:
        failure=dict(status='pending_design_revision' if isinstance(exc,core.GateFailed) else 'incomplete',
            error=dict(type=type(exc).__name__,message=str(exc)),finished_at=time.time())
        if pipeline is None:write_json(run/'status.json',dict(read_json(run/'status.json'),**failure))
        else:pipeline.stop.set();pipeline.update(**failure)
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
