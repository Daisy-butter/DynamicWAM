"""Finite native TOPP deployment smoke; guarded, original episode rules, no retry."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dualrate_domino_smoke import save

class FullDrivenIntegrationEnv:
    """TEST ONLY: execute aligned Full actions while observing compact proposals.

    This is not the deployed policy, a fallback, or a policy score. It tests
    real native physics/observation/cache plumbing before the student is trained.
    """
    def __init__(self,env,policy):
        self.env=env;self.policy=policy;self.targets=[]
    def __getattr__(self,name):return getattr(self.env,name)
    def take_action(self,proposal,action_type):
        import torch
        p=self.policy;offset=p.command_index-p.source_index
        if not 0<=offset<16:raise RuntimeError('diagnostic Full plan lacks this command')
        raw=p.world['reference_actions'][0,offset].detach().cpu()*p.builder.scale+p.builder.mean
        raw,_=p.bounds(raw[None],torch.zeros(1,dtype=torch.bool))
        self.targets.append(dict(command_index=p.command_index,source_command_index=p.source_index,
            proposed_action=proposal.tolist(),executed_full_action=raw[0].tolist()))
        return self.env.take_action(raw[0].numpy(),action_type=action_type)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--device');p.add_argument('--authorization')
    args=p.parse_args();c=json.loads(Path(args.config).read_text())
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(c,os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay,probe_dependency_imports
    dep=validate_dependency_overlay(c);activate_dependency_overlay(dep)
    import torch
    from dynamicwam.dualrate.train import _validate_storage,_validate_normalization,_gpu_authorization,source_identity,canonical_hash,file_sha256
    out=_validate_storage(c);m=json.loads(Path(c['manifest']).read_text());_validate_normalization(c,m)
    if m['time_contract']!='native_event_single_v6' or c['run_kind']!='domino_deployment_smoke':raise ValueError('native smoke only')
    for f,h in c['deployment_artifact_sha256'].items():
        if file_sha256(f)!=h:raise ValueError('native external source changed: '+f)
    identity=dict(source_sha256=source_identity(),config_sha256=canonical_hash(c),manifest_sha256=file_sha256(c['manifest']),
                  init_checkpoint_sha256=c['init_checkpoint_sha256'],device_type='cuda',torch_version=str(torch.__version__))
    _gpu_authorization(Path(args.authorization),identity,out,expected_mode='domino_deployment_smoke',expected_dependencies=dep)
    probe_dependency_imports(c,require_flash_attention=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(uuids)!=1:raise ValueError('native smoke requires one explicitly bound renderer/policy GPU')
    pci=None
    for f in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        if uuids[0] in f.read_text():pci=f.parent.name
    if pci is None:raise RuntimeError('renderer PCI identity unavailable')
    os.environ['DYNAMICWAM_GPU_PCI']=pci
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    binding=install_gpu_binding()
    from dynamicwam.dualrate.factory import build_system
    system=build_system(c,'cuda:0').eval().requires_grad_(False)
    diagnostic=Path(c['diagnostic_checkpoint']).resolve(strict=True)
    if file_sha256(diagnostic)!=c['diagnostic_checkpoint_sha256']:raise ValueError('diagnostic state hash changed')
    payload=torch.load(diagnostic,map_location='cpu',weights_only=True)
    if payload.get('run_kind')!='distributed_training_smoke' or payload.get('phase')!='cross_time_joint' or not payload.get('phase_complete'):
        raise ValueError('completed diagnostic-only phase state required')
    for key in ('source_sha256','manifest_sha256','init_checkpoint_sha256'):
        if payload['identity'][key]!=identity[key]:raise ValueError('diagnostic state identity differs: '+key)
    for key in ('architecture','native_v6','native_v7','time_contract','codec_id'):
        if payload['contract'][key]!=c[key]:raise ValueError('diagnostic model contract differs: '+key)
    system.load_state_dict(payload['model'],strict=True);del payload
    from dynamicwam.dualrate.native_v7_acceptance import verify_compact as verify_native_motor
    from dynamicwam.dualrate.producer import DominoPhysicalDataset,materialize_native_latents
    from dynamicwam.dualrate.train import _device_batch
    fixture=materialize_native_latents(_device_batch(torch.utils.data.default_collate(
        [DominoPhysicalDataset(c['manifest'])[0]]),torch.device('cuda:0')),system.slow.native.compact_wan.video_model.vae)
    save(out/'compact_acceptance.json',verify_native_motor(system,fixture));del fixture
    # A briefly trained engineering state is NOT a formal initialization/evaluation.
    if c.get('diagnostic_initial_weights') is not False:raise ValueError('explicit diagnostic training provenance required')
    from dynamicwam.dualrate.online import OnlineBatchBuilder,load_cached_language
    from dynamicwam.dualrate.native_v7_deployment import NativeDualRatePolicy,PROTOCOL
    from dynamicwam.dualrate.producer import episode_prompt,SCENE_PREFIX
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from video_artifacts import BufferedHeadVideo,validate_video_artifacts,aggregate_native_metrics,write_domino_results
    domino=Path(c['domino_root']);sys.path.insert(0,str(domino/'script'))
    import yaml
    from eval_metrics import EvalMetricsTracker
    from dataclasses import asdict
    task_settings=yaml.safe_load((domino/'task_config'/f"{c['deployment']['task_config']}.yml").read_text())
    if not task_settings['use_dynamic'] or task_settings['dynamic_level']!=1:raise ValueError('native level1 required')
    reports=[];integration=[]
    for task in c['deployment']['tasks']:
      metrics=[]
      for episode in range(c['deployment']['episodes']):
        env=None;video=BufferedHeadVideo();d=out/task/f'episode_{episode:04d}';d.mkdir(parents=True,exist_ok=False)
        try:
            with SapienInitLock():
                env,prepared=prepare_level1_env(domino,task_name=task,task_config=c['deployment']['task_config'],
                    start_seed=c['deployment']['seed']+episode,instruction_type=c['deployment']['instruction_type'],native_protocol=True)
            if prepared['velocity'] is not None:raise RuntimeError('native motion was converted')
            prompt=SCENE_PREFIX+prepared['instruction']
            try:language=load_cached_language(Path(c['manifest']).parent,prompt,index_path=m['text_index'])
            except KeyError:
                from dynamicwam.inference.model_loader import _build_t5_encoder,load_deploy_config
                from prepare_dualrate_text import pack_contexts,native_text_inputs,tokenizer_identity
                profile=load_deploy_config(c['native_profile'])
                _,ep,tp=native_text_inputs(profile,{'query':prompt})
                ti=json.loads(Path(m['text_index']).read_text())['identity']
                if file_sha256(ep)!=ti['source_encoder_sha256'] or tokenizer_identity(tp)!=ti['tokenizer_sha256']:
                    raise ValueError('native text encoder identity changed')
                encoder=_build_t5_encoder(profile,'cuda:0')
                language,_=pack_contexts(encoder,[prompt],torch.device('cuda:0'))[0];del encoder
            env.set_instruction(instruction=prompt)
            tracker=EvalMetricsTracker(env,task_settings);tracker.on_episode_start()
            env._metrics_tracker=tracker
            builder=OnlineBatchBuilder(m,action_mean=m['action_statistics']['mean'],action_scale=m['action_statistics']['std'],language=language,device='cuda:0')
            policy=NativeDualRatePolicy(system,builder,seed=int(prepared['seed']))
            lifted=False;success=False;reason='action_budget_exhausted';began=time.monotonic()
            while env.take_action_cnt<env.step_lim:
                obs=env.get_obs()
                video.append(obs['observation']['head_camera']['rgb'],physics_time_s=policy._time(env),capture_wall_perf_s=time.monotonic())
                policy.step(env,obs)
                if prepared['check_z_threshold'] is not None and not lifted:
                    lifted=float(prepared['lift_actor'].get_pose().p[2])-prepared['start_z']>prepared['check_z_threshold']
                if env.eval_success:
                    success=prepared['check_z_threshold'] is None or lifted
                    reason='success' if success else 'object_not_lifted';break
                if env.check_gripper_contact_dynamic_object() and prepared['stop_on_contact']:env.stop_dynamic_object_motion()
                if not getattr(env,'_dynamic_object_stopped',False) and not env.check_dynamic_object_boundary():
                    env.eval_fail=True;tracker.record_out_of_bounds();reason='out_of_bounds';break
                if env.eval_fail:reason='eval_fail';break
            native=out/'eval_result'/task/'dualrate_compact_v7'/c['deployment']['task_config']/'initial_diagnostic'/c['deployment']['result_stamp']
            report=dict(protocol=PROTOCOL,task=task,episode_index=episode,seed=prepared['seed'],success=success,
                stop_reason=reason,action_count=env.take_action_cnt,action_limit=env.step_lim,
                physical_end=policy._time(env),wall_seconds=time.monotonic()-began,events=policy.records,
                diagnostic_initial_weights=False,diagnostic_checkpoint_sha256=c['diagnostic_checkpoint_sha256'],
                performance_accepted=False,native_motion_unmodified=True)
            report.update(video.finalize(native/f'episode{episode}.mp4',episode_end_physics_s=policy._time(env)))
            if not validate_video_artifacts(report):raise RuntimeError('native video validation failed')
            save(d/'report.json',report);reports.append(report)
            metric=asdict(tracker.get_episode_metrics(success,None if reason in ('success','action_budget_exhausted') else reason,seed=prepared['seed']))
            save(d/'metrics.json',metric);metrics.append(metric)
            write_domino_results(native,c['deployment']['result_stamp'],c['deployment']['instruction_type'],aggregate_native_metrics(metrics))
            # Preserve the autonomous rollout above, including all failures.
            # A separate fresh scene tests cache reuse on native Full-driven
            # physical trajectories. No clock, TTL or action budget is relaxed.
            env.close_env();env.release_episode_resources();env=None
            with SapienInitLock():
                env,replayed=prepare_level1_env(domino,task_name=task,task_config=c['deployment']['task_config'],
                    start_seed=int(prepared['seed']),instruction_type=c['deployment']['instruction_type'],native_protocol=True)
            if replayed['seed']!=prepared['seed'] or replayed['instruction']!=prepared['instruction']:
                raise RuntimeError('integration scene identity changed')
            env.set_instruction(instruction=prompt)
            shadow=NativeDualRatePolicy(system,builder,seed=int(prepared['seed']))
            driver=FullDrivenIntegrationEnv(env,shadow)
            for _ in range(8):
                if env.eval_success or env.eval_fail or env.take_action_cnt>=env.step_lim:break
                shadow.step(driver,env.get_obs())
                if env.check_gripper_contact_dynamic_object() and prepared['stop_on_contact']:env.stop_dynamic_object_motion()
                if not getattr(env,'_dynamic_object_stopped',False) and not env.check_dynamic_object_boundary():break
            integration.append(dict(task=task,diagnostic_driver='Full_A10_test_only',scored=False,
                action_count=env.take_action_cnt,physical_end=shadow._time(env),events=shadow.records,targets=driver.targets))
            save(d/'full_driven_integration.json',integration[-1])

        except BaseException as exc:
            save(d/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if env is not None:env.close_env();env.release_episode_resources()
    queries=[e for r in reports for e in r['events'] if 'query_physical_time' in e]
    if not queries or not any(e['source_command_index']>0 for e in queries):raise RuntimeError('no refreshed world consumed')
    iq=[e for r in integration for e in r['events'] if 'query_physical_time' in e]
    if not any(e['source_command_index']<e['command_index'] for e in iq):raise RuntimeError('no native Full-driven multirate cache reuse')
    if not any(e['source_command_index']>0 for e in iq):raise RuntimeError('no refreshed native Full-driven reference')
    # Independent ordinal/physical-clock replay before formal can be authorized.
    for report in reports+integration:
        count=0;last=None;sources={}
        for event in report['events']:
            if 'query_physical_time' in event:
                q=event['command_index'];s=event['source_command_index']
                if q!=count or q%2 or s>q:raise RuntimeError('query ledger mismatch')
                if last is not None and event['query_physical_time']!=last:raise RuntimeError('query exposure changed')
                if s==q:sources[s]=event['query_physical_time']
                if sources.get(s)!=event['source_physical_time']:raise RuntimeError('reference origin changed')
                if event['future_command_events']!=[q+n for n in (1,2,4,8)]*3:raise RuntimeError('Z event mismatch')
                if event['prefix_rows']!=0 or event['future_physical_end_times_known']:raise RuntimeError('native domain mismatch')
                if event['previous_plan_aligned']!=(q>0):raise RuntimeError('previous proposal missing')
                if event['prior_row_offset']!=q-s or event['prior_overlap_rows']!=max(0,16-(q-s)):raise RuntimeError('A10 join mismatch')
                if event['prior_source_event']!=s or not event['graph_ready']:raise RuntimeError('prior/graph mismatch')
            else:
                if event['command_index']!=count or event['physical_end']<event['physical_start']:raise RuntimeError('execution clock mismatch')
                if last is not None and event['physical_start']!=last:raise RuntimeError('physical gap')
                last=event['physical_end'];count+=1
        if count!=report['action_count'] or last!=report['physical_end']:raise RuntimeError('terminal ledger mismatch')
    for f,h in c['deployment_artifact_sha256'].items():
        if file_sha256(f)!=h:raise RuntimeError('external source changed while running')
    save(out/'deployment_acceptance.json',dict(passed=True,protocol=PROTOCOL,episodes=len(reports),
        native_motion_unmodified=True,all_video_passed=True,reference_refreshed_and_reused=True,reuse_evidence_scope='Full_driven_integration_not_student_performance',
        autonomous_reference_reuses=sum(e['source_command_index']<e['command_index'] for e in queries),
        integration_reference_reuses=sum(e['source_command_index']<e['command_index'] for e in iq),
        inference_physics_frozen=True,integer_physical_prior_replay_passed=True,query_count=len(queries),binding=binding,source_sha256=identity['source_sha256'],
        initial_weight_fixture=False,diagnostic_training_fixture=True,
        diagnostic_checkpoint_sha256=c['diagnostic_checkpoint_sha256'],performance_accepted=False))

if __name__=='__main__':main()

