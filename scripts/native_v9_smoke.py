"""Guarded finite native V9 rollout/collection. No retry or driver operations."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dualrate_domino_smoke import save


def graph_acceptance(system,c):
    import torch
    from dynamicwam.dualrate.producer import DominoPhysicalDataset,materialize_native_latents
    from dynamicwam.dualrate.train import _device_batch
    from dynamicwam.dualrate.native_v9 import draft
    m=json.loads(Path(c['manifest']).read_text());mp=m.get('expert_manifest',c['manifest'])
    b=materialize_native_latents(_device_batch(torch.utils.data.default_collate([DominoPhysicalDataset(mp)[0]]),torch.device('cuda:0')),system.slow.native.compact_wan.video_model.vae)
    with torch.no_grad():
        world=system.world_readout(system.predict_reference(b));world['native_available']=torch.ones(1,device='cuda:0',dtype=torch.bool)
        gen=torch.Generator(device='cuda:0').manual_seed(42)
        eager=system.sample_fast(b,world,generator=gen)
        prior,valid=draft(world,b,1.44)
        zero=not bool(system.fast.action_out.weight.any() or system.fast.action_out.bias.any())
        if zero:torch.testing.assert_close(eager['actions'][valid],prior[valid],atol=2e-6,rtol=0)
        engine=system.enable_fast_graph();gen.manual_seed(42)
        actual=system.sample_fast(b,world,generator=gen)
        for k in eager:torch.testing.assert_close(actual[k],eager[k],atol=2e-5,rtol=2e-5)
        changed=dict(world,reference_actions=world['reference_actions']+.125)
        altered=system.sample_fast(b,changed,generator=gen)
        if zero:torch.testing.assert_close(altered['actions'][valid],prior[valid]+.125,atol=2e-6,rtol=0)
        receipt=engine.seal(expected_shapes=1)
    return dict(passed=receipt['passed'],graph=receipt,zero_correction_parity=zero,changed_draft_checked=True)


class ReplayRecorder:
    def __init__(self):self.rows=[]
    def __call__(self,batch,source,frames,event):
        from dualrate_slow_process import _cpu_tree
        from dynamicwam.dualrate.online import history_endpoints
        chosen,_=history_endpoints(frames,action_dt=.18)
        unique={f.exposure:f for f in chosen}
        self.rows.append((_cpu_tree({k:v for k,v in batch.items() if k!='text'}),
            {k:v for k,v in source.items() if k!='source_text'},tuple(unique[k] for k in sorted(unique)),dict(event)))

    def write(self,folder,system,builder,task,seed):
        import torch
        from dynamicwam.dualrate.train import validate_row,file_sha256
        from dualrate_slow_process import _cpu_tree
        folder.mkdir(parents=True,exist_ok=False);language=folder/'language.pt'
        torch.save(builder.language,language);entries=[]
        for number,(current,source,frames,event) in enumerate(self.rows):
            now=event['query_physical_time'];q=event['command_index']
            with torch.no_grad():teacher=builder.source(frames,source_time=now,action_start=now,vae=system.slow.native.compact_wan.video_model.vae)
            teacher['source_known_actions']=torch.zeros(1,16,14,device=builder.device)
            teacher['source_known_mask']=torch.zeros(1,16,dtype=torch.bool,device=builder.device)
            row={**source,**current}
            row.update({'teacher_'+k[7:]:v for k,v in _cpu_tree(teacher).items() if k.startswith('source_') and k!='source_text'})
            row.update(source_text=builder.language['context'][None],text=builder.language['context'][None],teacher_text=builder.language['context'][None],
                source_time=source['source_time'],completed_time=source['completed_time'],native_source_index=source['native_source_index'],
                source_action=torch.zeros(1,16,14),fast_action=torch.zeros(1,16,14),fast_action_valid=torch.ones(1,16,dtype=torch.bool),
                source_future_latent=torch.zeros(1,48,2,12,10),future=torch.zeros(1,12,134),future_loss_valid=torch.zeros(1,12,134,dtype=torch.bool),
                contact=torch.zeros(1,12,2),contact_valid=torch.zeros(1,12,2,dtype=torch.bool),
                future_events=torch.tensor([[q+n for n in (1,2,4,8)]*3]),native_replay=torch.tensor([True]),
                teacher_seed=torch.tensor([(seed+7000003*q+17)%((1<<63)-1)]))
            row={k:v[0].cpu().clone() for k,v in row.items()}
            validate_row(row)
            for k in ('source_text','text','teacher_text'):row.pop(k)
            path=folder/f'query_{number:04d}.pt'
            torch.save(dict(schema='native-v9-actual-replay-v1',row=row,language=str(language),language_sha256=file_sha256(language)),path)
            entries.append(dict(path=str(path),sha256=file_sha256(path),task=task,seed=seed,command_index=q,
                source_index=event['source_command_index'],physical_age=event['reference_age'],source_inflight=bool(source['source_inflight'][0])))
        self.rows.clear();return entries


def audit(events):
    count=0;end=0.;sources={};queries=[]
    for e in events:
        if e['kind']=='slow_request':sources[e['version']]=e
        if e['kind']=='slow_publish':
            r=sources[e['version']]
            if r['source_index']!=e['source_index'] or r['source_time']!=e['source_time']:raise ValueError('source identity changed')
            if not e['source_time']<=e['complete_physical']<=e['consumed_at_physical']+1e-8:raise ValueError('publication clocks')
        if e['kind']=='fast_query':
            if e['command_index']!=count or abs(e['query_physical_time']-end)>1e-7:raise ValueError('fast command/clock ledger')
            if e['prefix_rows'] or not 0<=count-e['source_command_index']<16 or e['reference_age']<0:raise ValueError('invalid native reference')
            queries.append(e)
        if e['kind']=='native_topp_execute':
            if e['command_index']!=count or abs(e['physical_start']-end)>1e-7 or e['physical_end']<end:raise ValueError('TOPP clock ledger')
            count+=1;end=e['physical_end']
    if not queries:raise ValueError('no fast queries')
    model_intervals=[(e['model_start_wall'],e['model_end_wall']) for e in events if e['kind']=='slow_publish' and e.get('model_start_wall') is not None]
    executing=[(e['execute_start_wall'],e['execute_end_wall']) for e in events if e['kind']=='native_topp_execute' and 'execute_start_wall' in e]
    actual_overlap=sum(any(max(a,c)<min(b,d) for c,d in executing) for a,b in model_intervals)
    return dict(passed=True,model_execution_overlap=actual_overlap,commands=count,queries=len(queries),overlapped=sum(x['slow_inflight_at_fast_start'] for x in queries),
        reused=sum(x['reference_reused'] for x in queries),exceptional_waits=sum(x['wait_reason']=='no_valid_reference' for x in queries),
        partial_sources=sum(x.get('committed_rows',0)>0 for x in events if x['kind']=='slow_request'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--device');p.add_argument('--authorization');a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    from dynamicwam.dualrate.renderer_runtime import verify_renderer_environment
    verify_renderer_environment(c,os.environ)
    from dynamicwam.dualrate.dependencies import validate_dependency_overlay,activate_dependency_overlay,probe_dependency_imports
    dep=validate_dependency_overlay(c);activate_dependency_overlay(dep)
    import torch,yaml
    from dynamicwam.dualrate.train import _validate_storage,_validate_normalization,_gpu_authorization,source_identity,canonical_hash,file_sha256
    out=_validate_storage(c);m=json.loads(Path(c['manifest']).read_text());_validate_normalization(c,m)
    for f,h in c['deployment_artifact_sha256'].items():
        if file_sha256(f)!=h:raise ValueError('external simulator changed: '+f)
    identity=dict(source_sha256=source_identity(),config_sha256=canonical_hash(c),manifest_sha256=file_sha256(c['manifest']),
        init_checkpoint_sha256=c['init_checkpoint_sha256'],device_type='cuda',torch_version=str(torch.__version__))
    _gpu_authorization(Path(a.authorization),identity,out,expected_mode='domino_deployment_smoke',expected_dependencies=dep)
    probe_dependency_imports(c,require_flash_attention=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(uuids)!=2:raise ValueError('reserve one renderer/fast GPU and one independent slow GPU')
    for f in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        if uuids[0] in f.read_text():os.environ['DYNAMICWAM_GPU_PCI']=f.parent.name
    from dynamicwam.runtime.domino_rt.gpu_binding import install_gpu_binding
    install_gpu_binding()
    from dynamicwam.dualrate.factory import build_system
    system=build_system(c,'cuda:0').eval().requires_grad_(False)
    if c.get('checkpoint'):
        if file_sha256(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        payload=torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        if payload['identity']['source_sha256']!=source_identity() or payload['identity']['manifest_sha256']!=file_sha256(c['manifest']):raise ValueError('trained source/data contract differs')
        for key in ('architecture','native_v6','native_v9','time_contract','codec_id'):
            if payload['contract'][key]!=c[key]:raise ValueError('trained model contract differs: '+key)
        if not payload.get('phase_complete') or payload.get('phase')!='cross_time_joint':raise ValueError('final complete phase required')
        system.load_state_dict(payload['model'],strict=True);del payload
    save(out/'model_acceptance.json',graph_acceptance(system,c))
    from dynamicwam.dualrate.online import OnlineBatchBuilder
    from dynamicwam.dualrate.native_v9_deployment import NativeResidualPolicy,PROTOCOL
    from dynamicwam.dualrate.producer import SCENE_PREFIX
    from dynamicwam.runtime.domino_rt.domino_session import prepare_level1_env
    from dynamicwam.runtime.domino_rt.sapien_init_lock import SapienInitLock
    from native_v7_eval_worker import language_for
    from native_v9_slow_service import NativeSlowService
    from video_artifacts import BufferedHeadVideo,validate_video_artifacts,aggregate_native_metrics,write_domino_results
    from dataclasses import asdict
    domino=Path(c['domino_root']);sys.path.insert(0,str(domino/'script'))
    from eval_metrics import EvalMetricsTracker
    settings=yaml.safe_load((domino/'task_config'/f"{c['deployment']['task_config']}.yml").read_text())
    service=NativeSlowService(c);reports=[];samples=[]
    try:
      for task in c['deployment']['tasks']:
       metrics=[];next_seed=c['deployment']['seed'];references=None
       if c.get('prepared'):
        from dualrate_eval_worker import load_frozen_task
        references=load_frozen_task(dict(c,seed=c['deployment']['seed'],episodes=c['deployment']['episodes']),task)
       for ep in range(c['deployment']['episodes']):
        env=None;d=out/task/f'episode_{ep:04d}';d.mkdir(parents=True,exist_ok=False)
        try:
            descriptor=motion=None;replay=None
            if references:
                descriptor,motion,_=references[ep];next_seed=descriptor['seed']
                replay=dict(seed=descriptor['seed'],instruction=descriptor['instruction'],saved_motion=motion['saved_motion'])
            with SapienInitLock():env,prepared=prepare_level1_env(domino,task_name=task,task_config=c['deployment']['task_config'],
                start_seed=next_seed,instruction_type=c['deployment']['instruction_type'],native_protocol=True,replay=replay)
            next_seed=prepared['seed']+1
            if descriptor:
                import numpy as np
                env._rt_episode_info=motion['episode_info']
                if descriptor['arm_tag']:
                    from envs.utils.action import ArmTag
                    env.arm_tag=ArmTag(descriptor['arm_tag'])
                for key in ('seed','instruction','stop_on_contact','check_z_threshold'):
                    if prepared[key]!=descriptor[key]:raise ValueError('frozen native scene changed: '+key)
                for key,value in dict(initial_target_position=np.asarray(env._eval_target_actor.actor.get_pose().p),initial_state=env.robot.get_left_arm_jointState()+env.robot.get_right_arm_jointState(),physics_dt_s=float(env.scene.get_timestep())).items():
                    if not np.allclose(value,descriptor[key],atol=1e-8,rtol=0):raise ValueError('frozen native initial state changed: '+key)
                if env.step_lim!=descriptor['native_action_limit']:raise ValueError('native task budget changed')
                save(d/'manifest.json',descriptor)
            if prepared['velocity'] is not None:raise ValueError('native motion converted')
            prompt=SCENE_PREFIX+prepared['instruction'];language=language_for(c,m,prompt);env.set_instruction(instruction=prompt)
            builder=OnlineBatchBuilder(m,action_mean=m['action_statistics']['mean'],action_scale=m['action_statistics']['std'],language=language,device='cuda:0')
            recorder=ReplayRecorder() if c.get('collect_native_replay') else None
            policy=NativeResidualPolicy(system,builder,service,seed=prepared['seed'],record_query=recorder)
            tracker=EvalMetricsTracker(env,settings);tracker.on_episode_start();env._metrics_tracker=tracker
            video=BufferedHeadVideo();success=lifted=False;reason='action_budget_exhausted';start=time.monotonic()
            while env.take_action_cnt<env.step_lim:
                obs=env.get_obs();video.append(obs['observation']['head_camera']['rgb'],physics_time_s=policy._time(env),capture_wall_perf_s=time.monotonic())
                policy.step(env,obs)
                if prepared['check_z_threshold'] is not None and not lifted:lifted=float(prepared['lift_actor'].get_pose().p[2])-prepared['start_z']>prepared['check_z_threshold']
                if env.eval_success:
                    success=prepared['check_z_threshold'] is None or lifted;reason='success' if success else 'object_not_lifted';break
                if env.check_gripper_contact_dynamic_object() and prepared['stop_on_contact']:env.stop_dynamic_object_motion()
                if not getattr(env,'_dynamic_object_stopped',False) and not env.check_dynamic_object_boundary():
                    env.eval_fail=True;tracker.record_out_of_bounds();reason='out_of_bounds';break
                if env.eval_fail:reason='eval_fail';break
            policy.finish();checks=audit(policy.records)
            report=dict(protocol=PROTOCOL,task=task,episode_index=ep,seed=prepared['seed'],success=success,stop_reason=reason,
                action_count=env.take_action_cnt,action_limit=env.step_lim,physical_end=policy._time(env),wall_seconds=time.monotonic()-start,
                events=policy.records,clock_audit=checks,native_motion_unmodified=True,performance_accepted=False,
                purpose=c.get('purpose','engineering_smoke'),checkpoint_sha256=c.get('checkpoint_sha256'),config_sha256=canonical_hash(c))
            native=out/'eval_result'/task/'native_v9'/c['deployment']['task_config']/'candidate'/c['deployment']['result_stamp']
            report.update(video.finalize(native/f'episode{ep}.mp4',episode_end_physics_s=policy._time(env)))
            if not validate_video_artifacts(report):raise ValueError('native MP4 decode failed')
            save(d/'report.json',report);reports.append(report)
            metric=asdict(tracker.get_episode_metrics(success,None if reason in ('success','action_budget_exhausted') else reason,seed=prepared['seed']))
            save(d/'metrics.json',metric);metrics.append(metric)
            write_domino_results(native,c['deployment']['result_stamp'],c['deployment']['instruction_type'],aggregate_native_metrics(metrics))
            if recorder:samples.extend(recorder.write(d/'replay',system,builder,task,prepared['seed']))
        except BaseException as exc:
            save(d/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if env is not None:env.close_env();env.release_episode_resources()
    finally:service.close()
    checks=[r['clock_audit'] for r in reports]
    if not sum(x['model_execution_overlap'] for x in checks) or not sum(x['reused'] for x in checks) or not sum(x['partial_sources'] for x in checks):raise ValueError('independent slow/fast path unexercised')
    save(out/'deployment_acceptance.json',dict(passed=True,protocol=PROTOCOL,episodes=len(reports),checks=checks,slow_child_exitcode=service.process.exitcode))
    if c.get('collect_native_replay'):save(out/'replay_acceptance.json',dict(passed=True,samples=samples,labels='fresh-state Full teacher; Z/contact masked'))


if __name__=='__main__':main()
